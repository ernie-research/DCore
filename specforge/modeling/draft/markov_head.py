# coding=utf-8
"""Markov Head variants for DSpark speculative decoding."""

from typing import Optional

import torch
from torch import nn

from .dflash import sample


class VanillaMarkov(nn.Module):
    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(self, logits: torch.Tensor, *, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(self, base_logits: torch.Tensor, *, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]) -> torch.Tensor:
        if base_logits.size(2) == 0:
            return base_logits
        markov_bias = self.compute_step_bias(token_ids, hidden_states)
        return base_logits + markov_bias

    def sample_block_tokens(self, base_logits: torch.Tensor, *, first_prev_token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor], temperature: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            return torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device), base_logits

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()
        for step_idx in range(proposal_len):
            step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]
            step_logits = self.apply_step_logits(base_logits[:, step_idx, :], token_ids=prev_token_ids, hidden_states=step_hidden)
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = sample(step_logits.unsqueeze(1), temperature).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


class GatedMarkovHead(VanillaMarkov):
    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "gated"
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_gate(self, token_ids: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return torch.sigmoid(self.gate_proj(gate_inputs))

    def compute_step_bias(self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]) -> torch.Tensor:
        assert hidden_states is not None
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = self.compute_gate(token_ids, hidden_states).to(dtype=prev_embeddings.dtype)
        return self.project_bias(gate * prev_embeddings)


class RNNHead(VanillaMarkov):
    """GRU-like recurrent head that carries state across block positions."""

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "rnn"
        self.hidden_size = hidden_size
        self.state_size = markov_rank
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def _rnn_step(self, state: torch.Tensor, prev_embeddings: torch.Tensor, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        proj = self.joint_proj(z)
        gate_raw, candidate_raw, output_raw = proj.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.project_bias(torch.tanh(output_raw))
        return new_state, bias

    def compute_step_bias(self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]) -> torch.Tensor:
        assert hidden_states is not None
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = torch.zeros_like(prev_embeddings)
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias

    def apply_block_logits(self, base_logits: torch.Tensor, *, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]) -> torch.Tensor:
        assert hidden_states is not None
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits

        leading_shape = base_logits.shape[:-2]
        state = torch.zeros(*leading_shape, self.markov_rank, device=base_logits.device, dtype=hidden_states.dtype)

        output_logits = []
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])
            h_k = hidden_states[..., k, :]
            state, bias = self._rnn_step(state, prev_emb, h_k)
            output_logits.append(base_logits[..., k, :] + bias)
        return torch.stack(output_logits, dim=-2)

    def sample_block_tokens(self, base_logits: torch.Tensor, *, first_prev_token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor], temperature: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states is not None
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            return torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device), base_logits

        state = torch.zeros(batch_size, self.markov_rank, device=base_logits.device, dtype=hidden_states.dtype)
        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()

        for step_idx in range(proposal_len):
            prev_emb = self.get_prev_embeddings(prev_token_ids)
            h_k = hidden_states[:, step_idx, :]
            state, bias = self._rnn_step(state, prev_emb, h_k)
            step_logits = base_logits[:, step_idx, :] + bias
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = sample(step_logits.unsqueeze(1), temperature).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


def build_markov_head(config) -> Optional[nn.Module]:
    markov_rank = int(config.markov_rank)
    if markov_rank == 0:
        return None
    markov_head_type = str(config.markov_head_type).lower()
    if markov_head_type == "vanilla":
        return VanillaMarkov(vocab_size=config.vocab_size, markov_rank=markov_rank)
    if markov_head_type == "gated":
        return GatedMarkovHead(vocab_size=config.vocab_size, markov_rank=markov_rank, hidden_size=config.hidden_size)
    if markov_head_type == "rnn":
        return RNNHead(vocab_size=config.vocab_size, markov_rank=markov_rank, hidden_size=config.hidden_size)
    raise ValueError(f"Unsupported markov_head_type: {markov_head_type!r}")


__all__ = ["VanillaMarkov", "GatedMarkovHead", "RNNHead", "build_markov_head"]
