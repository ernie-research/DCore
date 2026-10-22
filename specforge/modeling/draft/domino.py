# coding=utf-8
"""Domino draft model entry point."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from specforge.utils import get_device_type

from .dflash import DFlashDraftModel
from .registry import register_draft


@register_draft
class DominoDraftModel(DFlashDraftModel):
    """DFlash backbone with Domino's GRU logits correction."""

    expected_projector_type = "domino"

    def __init__(self, config) -> None:
        dflash_config = getattr(config, "dflash_config", None) or {}
        projector_type = dflash_config.get("projector_type")
        if projector_type is None:
            dflash_config["projector_type"] = self.expected_projector_type
            config.dflash_config = dflash_config
        elif projector_type != self.expected_projector_type:
            raise ValueError(
                "DominoDraftModel requires " "dflash_config.projector_type='domino'."
            )
        super().__init__(config)

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        self.emb_dim = int(dflash_config["emb_dim"])
        self.gru_hidden_dim = int(dflash_config["gru_hidden_dim"])
        self.pure_draft_prefix_len = int(dflash_config.get("pure_draft_prefix_len", 0))
        self.shift_label = bool(dflash_config.get("shift_label", False))

        self.prefix_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        in_dim = config.hidden_size + self.gru_hidden_dim
        self.embed_proj = nn.Sequential(
            nn.Linear(in_dim, self.emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.emb_dim, config.vocab_size, bias=False),
        )

        # Ascend NPU DynamicGRU does not support bfloat16. Keep the fp16 copy
        # outside module registration so FSDP does not see mixed parameter dtypes.
        object.__setattr__(self, "_gru_fp16", None)

    @property
    def suffix_start(self) -> int:
        return (
            self.pure_draft_prefix_len
            if self.shift_label
            else (1 + self.pure_draft_prefix_len)
        )

    def _get_gru_fp16(self, device: torch.device) -> nn.GRU:
        gru_fp16 = self._gru_fp16
        if (
            gru_fp16 is None
            or next(gru_fp16.parameters()).device != device
            or gru_fp16.hidden_size != self.prefix_gru.hidden_size
        ):
            gru_fp16 = nn.GRU(
                input_size=self.prefix_gru.weight_ih_l0.shape[1],
                hidden_size=self.prefix_gru.hidden_size,
                num_layers=1,
                batch_first=True,
                bias=False,
            )
            gru_fp16.to(device=device, dtype=torch.float16)
            gru_fp16.weight_ih_l0.requires_grad = False
            gru_fp16.weight_hh_l0.requires_grad = False
            object.__setattr__(self, "_gru_fp16", gru_fp16)
        return gru_fp16

    def _run_gru(self, gru_inputs: torch.Tensor) -> torch.Tensor:
        if get_device_type() == "npu" and gru_inputs.dtype == torch.bfloat16:
            gru_fp16 = self._get_gru_fp16(gru_inputs.device)
            gru_fp16.weight_ih_l0.data.copy_(self.prefix_gru.weight_ih_l0.data.half())
            gru_fp16.weight_hh_l0.data.copy_(self.prefix_gru.weight_hh_l0.data.half())
            return gru_fp16(gru_inputs.half())[0].to(gru_inputs.dtype)
        return self.prefix_gru(gru_inputs)[0]

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_ids
        if prev_token_embeddings is None:
            raise ValueError("DominoDraftModel requires prev_token_embeddings")

        bsz, n_blocks, block_size = base_logits.shape[:3]
        if self.shift_label:
            gru_inputs = prev_token_embeddings.reshape(bsz * n_blocks, block_size, -1)
            gru_out = self._run_gru(gru_inputs)
            gru_out = gru_out.reshape(bsz, n_blocks, block_size, -1)
            prefix_states = gru_out[:, :, self.suffix_start :, :]
        else:
            gru_inputs = prev_token_embeddings[:, :, : block_size - 1, :].reshape(
                bsz * n_blocks, block_size - 1, -1
            )
            gru_out = self._run_gru(gru_inputs)
            gru_out = gru_out.reshape(bsz, n_blocks, block_size - 1, -1)
            prefix_states = gru_out[:, :, self.suffix_start - 1 :, :]

        z_n = hidden_states[:, :, self.suffix_start :, :]
        concat_features = torch.cat([z_n, prefix_states], dim=-1)
        logits_e = self.embed_proj(concat_features)

        prefix_logits = base_logits[:, :, : self.suffix_start, :]
        suffix_logits = base_logits[:, :, self.suffix_start :, :] + logits_e
        return torch.cat([prefix_logits, suffix_logits], dim=2)


__all__ = ["DominoDraftModel"]


# ---------------------------------------------------------------------------
# Domino inference spec_generate — uses autoregressive GRU correction.
# ---------------------------------------------------------------------------

from transformers import DynamicCache  # noqa: E402

from .dflash import (  # noqa: E402
    _gather_probs,
    _logits_to_probs,
    _sample_from_probs,
    _sample_residual,
    extract_context_feature,
    sample,
)


@torch.inference_mode()
def domino_spec_generate(
    model: DominoDraftModel,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list,
    temperature: float,
    confidence_threshold: float = 0.0,
):
    """Autoregressive speculative generation with Domino GRU correction."""
    model.eval()
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    device = target.device

    block_size = model.block_size
    prefix_len = model.pure_draft_prefix_len
    shift_label = model.shift_label
    k_draft = block_size if shift_label else block_size - 1

    output_ids = torch.full(
        (1, max_length + block_size + 1),
        model.mask_token_id, dtype=torch.long, device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

    past_kv_target = DynamicCache()
    past_kv_draft = DynamicCache()

    # Prefill
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_kv_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    # Decode
    acceptance_lengths = []
    start = num_input_tokens
    while start < max_length:
        block_output_ids = output_ids[:, start:start + block_size].clone()

        # Draft backbone forward
        noise_embedding = target.model.embed_tokens(block_output_ids)
        parallel_hiddens = model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, past_kv_draft.get_seq_length():start + block_size],
            past_key_values=past_kv_draft,
            use_cache=True,
            is_causal=False,
        )
        if not shift_label:
            parallel_hiddens = parallel_hiddens[:, -block_size + 1:, :]
        past_kv_draft.crop(start)

        # Base logits from target LM head
        base_logits = target.lm_head(parallel_hiddens)  # [1, k_draft, V]

        # Build verify_ids with Domino GRU correction
        verify_ids = torch.full(
            (1, k_draft + 1), model.mask_token_id, dtype=torch.long, device=device,
        )
        verify_ids[:, 0] = output_ids[:, start]

        draft_probs_list = []

        # Prefix positions: use base logits directly (no GRU)
        if prefix_len > 0:
            prefix_probs = _logits_to_probs(base_logits[:, :prefix_len, :], temperature)
            prefix_tokens = _sample_from_probs(prefix_probs)
            verify_ids[:, 1:1 + prefix_len] = prefix_tokens
            draft_probs_list.append(prefix_probs)

        # GRU-corrected positions
        realized_prefix_ids = verify_ids[:, :1 + prefix_len]
        realized_prefix_embeds = target.model.embed_tokens(realized_prefix_ids)
        _, gru_hidden = model.prefix_gru(realized_prefix_embeds)

        for i in range(prefix_len, k_draft):
            z_i = parallel_hiddens[:, i:i + 1, :]
            s_i = gru_hidden.transpose(0, 1)
            bias = model.embed_proj(torch.cat([z_i, s_i], dim=-1))
            corrected_logits = base_logits[:, i:i + 1, :] + bias

            pos_probs = _logits_to_probs(corrected_logits, temperature)
            current_token = _sample_from_probs(pos_probs)
            verify_ids[:, i + 1:i + 2] = current_token
            draft_probs_list.append(pos_probs)

            if i + 1 < k_draft:
                new_embed = target.model.embed_tokens(current_token)
                _, gru_hidden = model.prefix_gru(new_embed, gru_hidden)

        draft_probs = torch.cat(draft_probs_list, dim=1)

        # Target verification
        output = target(
            verify_ids,
            position_ids=position_ids[:, start:start + k_draft + 1],
            past_key_values=past_kv_target,
            use_cache=True,
            output_hidden_states=True,
        )
        target_probs = _logits_to_probs(output.logits, temperature)

        # Rejection sampling
        proposed_tokens = verify_ids[:, 1:]
        sel_target = _gather_probs(target_probs[:, :-1, :], proposed_tokens)
        sel_draft = _gather_probs(draft_probs, proposed_tokens).clamp_min(1e-8)
        accept_prob = torch.clamp(sel_target / sel_draft, max=1.0)
        accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)
        accept_prefix = accept_mask.cumprod(dim=1)
        acceptance_length = int(accept_prefix.sum(dim=1)[0].item())

        # Commit accepted tokens
        output_ids[:, start:start + acceptance_length + 1] = verify_ids[:, :acceptance_length + 1]

        # Bonus token
        if acceptance_length < k_draft:
            bonus = _sample_residual(
                target_probs[0, acceptance_length],
                draft_probs[0, acceptance_length],
            )
        else:
            bonus = _sample_from_probs(target_probs[:, -1:, :]).view(-1)
        output_ids[:, start + acceptance_length + 1] = bonus

        start += acceptance_length + 1
        past_kv_target.crop(start)
        target_hidden = extract_context_feature(
            output.hidden_states, model.target_layer_ids
        )[:, :acceptance_length + 1, :]
        acceptance_lengths.append(acceptance_length + 1)

        if stop_token_ids and any(
            sid in output_ids[0, num_input_tokens:start] for sid in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != model.mask_token_id]
    if stop_token_ids:
        stop_t = torch.tensor(stop_token_ids, device=device)
        stop_idx = torch.isin(output_ids[0, num_input_tokens:], stop_t).nonzero(as_tuple=True)[0]
        if stop_idx.numel() > 0:
            output_ids = output_ids[:, :num_input_tokens + stop_idx[0] + 1]

    return output_ids, acceptance_lengths
