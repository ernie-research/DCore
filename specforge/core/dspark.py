# coding=utf-8
"""DSpark Online Training Wrapper + Composite Loss."""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.core.dflash import (
    compute_accept_len,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
    dpace_weight,
)
from specforge.core.dspark_loss import (
    compute_ce_loss,
    compute_confidence_loss,
)


class OnlineDSparkModel(nn.Module):
    """DSpark online training wrapper: CE + L1 distillation + confidence BCE."""

    def __init__(
        self,
        draft_model: DSparkDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        ce_loss_alpha: float = 0.1,
        l1_loss_alpha: float = 0.9,
        kl_loss_alpha: float = 0.0,
        confidence_head_alpha: float = 1.0,
        position_weighting: str = "decay",
        dpace_alpha: float = 0.5,
    ):
        super().__init__()
        if position_weighting not in ("decay", "dpace"):
            raise ValueError(
                "position_weighting must be 'decay' or 'dpace', "
                f"got {position_weighting!r}"
            )
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.ce_loss_alpha = ce_loss_alpha
        self.l1_loss_alpha = l1_loss_alpha
        self.kl_loss_alpha = kl_loss_alpha
        self.confidence_head_alpha = confidence_head_alpha
        self.position_weighting = position_weighting
        self.dpace_alpha = dpace_alpha

    def _sample_anchor_positions(self, seq_len, loss_mask, device):
        bsz = loss_mask.shape[0]
        # DeepSpec: anchor valid if both loss_mask[i] and loss_mask[i+1] are set
        num_candidates = max(seq_len - 1, 0)
        if num_candidates == 0:
            raise ValueError("should preprocess the data.")
        anchor_valid = loss_mask[:, :num_candidates] > 0.5
        first_target_valid = loss_mask[:, 1 : num_candidates + 1] > 0.5
        valid = anchor_valid & first_target_valid

        valid_counts = valid.sum(dim=1)
        max_n = int(self.num_anchors)

        if num_candidates == 0 or valid_counts.max().item() == 0:
            raise ValueError("should preprocess the data.")

        indices = (
            torch.arange(num_candidates, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.full_like(indices, seq_len + 1)
        )

        random_vals = torch.rand(bsz, num_candidates, device=device)
        random_vals = torch.where(valid, random_vals, torch.full_like(random_vals, 2.0))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        if num_candidates < max_n:
            pad = torch.full(
                (bsz, max_n - num_candidates),
                seq_len + 1,
                dtype=gathered.dtype,
                device=device,
            )
            gathered = torch.cat([gathered, pad], dim=1)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
        return anchors, keep_mask

    def _create_position_ids(self, anchor_positions):
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )
        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )
        return self.embed_tokens(noise_ids)

    def _distill_chunk_per_token(
        self,
        flat_draft_logits: torch.Tensor,
        position_idx: torch.Tensor,
        target_hidden_chunk: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-position L1, soft-label CE and -H(p) against the target, one chunk.

        The second return value is the soft-label cross-entropy
        C_i = -sum_x p_i(x) log q_psi(x) (D-PACE App. E), the third is the
        draft-independent offset sum_x p log p that turns it into KL(p||q):
        the caller adds them rather than having this function build a second
        full-vocab tensor. exp(-C_i) is the D-PAKL acceptance proxy.

        Run under torch.utils.checkpoint recomputation: each chunk's full-vocab
        float32 probability tensors (~0.3 GB at Qwen3's vocabulary) would
        otherwise all be retained for backward at once, which does not fit next
        to the sglang server (mirrors OnlineDFlashModel._lk_chunk_terms).
        The L1 matches compute_l1_distillation_loss's per-token values.

        The target branch stays *inside* the checkpoint on purpose. Handing the
        precomputed probs in as an argument would be faster in backward (no
        target GEMM to re-run, ~4% of a step) but checkpoint saves its inputs,
        so every chunk's probs would stay alive until backward: num_valid x
        vocab x 2B, i.e. 2.5-5 GB at bsz2 x 512 anchors x block16 on Qwen3.
        Recomputing the GEMM is the cheaper side of that trade here.

        The gather by position_idx happens inside so the checkpoint retains the
        already-alive full logits tensor rather than a per-chunk gathered copy.
        ``target_hidden_chunk`` arrives already 2-D: with a 4-D input F.linear
        dispatches to a broadcast aten::bmm that re-streams the whole
        [hidden, vocab] matrix per batch element, ~16x slower than the
        equivalent 2-D GEMM.
        """
        draft_logits_chunk = flat_draft_logits.index_select(0, position_idx)
        with torch.no_grad():
            t_probs = F.softmax(self.lm_head(target_hidden_chunk).float(), dim=-1)
        draft_probs = F.softmax(draft_logits_chunk.float(), dim=-1)
        l1_per_token = (draft_probs - t_probs).abs().sum(dim=-1)
        if self.kl_loss_alpha > 0:
            draft_log_probs = F.log_softmax(draft_logits_chunk.float(), dim=-1)
            # p underflows to exactly 0 over most of the vocabulary once the
            # target distribution is peaked, and log q can be -inf on that same
            # tail, so the plain product gives 0 * -inf = nan -- which then
            # spreads to every position through the .sum(-1) and to the whole
            # draft through the backward. Those terms are 0 in
            # C_i = -sum_x p log q, so mask them out instead of multiplying:
            # p > 0 positions keep their value (a genuinely infinite CE stays
            # infinite rather than being silently zeroed). This mirrors the
            # xlogy below, which already guards the target-side product.
            soft_ce_per_token = -torch.where(
                t_probs > 0, t_probs * draft_log_probs, 0.0
            ).sum(-1)
            with torch.no_grad():
                neg_entropy_per_token = torch.xlogy(t_probs, t_probs).sum(-1)
        else:
            soft_ce_per_token = torch.zeros_like(l1_per_token)
            neg_entropy_per_token = torch.zeros_like(l1_per_token)
        return l1_per_token, soft_ce_per_token, neg_entropy_per_token

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            attn_mask = create_dflash_block_mask(
                anchor_positions, block_keep_mask, seq_len, self.block_size, device
            )
        else:
            attn_mask = create_dflash_sdpa_mask(
                anchor_positions, block_keep_mask, seq_len, self.block_size, device
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=attn_mask,
        )

        n_blocks = anchor_positions.shape[1]
        output_hidden_4d = output_hidden.view(bsz, n_blocks, self.block_size, -1)
        draft_logits = self.lm_head(output_hidden)
        draft_logits_4d = draft_logits.view(bsz, n_blocks, self.block_size, -1)

        # --- Labels: anchor+1 .. anchor+block_size (DeepSpec convention) ---
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )

        # --- Prev token ids for markov: anchor, anchor+1, ..., anchor+block_size-1 ---
        prev_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        prev_label_indices = (anchor_positions.unsqueeze(-1) + prev_offsets).clamp(
            max=seq_len - 1
        )
        prev_token_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, prev_label_indices
        )

        # --- Apply markov correction ---
        if self.draft_model.markov_head is not None:
            draft_logits_4d = self.draft_model.markov_head.apply_block_logits(
                draft_logits_4d,
                token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
            )

        # --- Eval mask: contiguous valid prefix (cumprod), matching DeepSpec ---
        target_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        eval_mask_raw = (
            valid_label_mask
            & (target_loss_mask_gathered > 0.5)
            & block_keep_mask.unsqueeze(-1)
        )
        eval_mask = eval_mask_raw.to(torch.int32).cumprod(dim=-1).bool()

        # Token-level CE (no grad): feeds the metrics and, when the acceptance
        # proxy is the hard-label one, the D-PACE weights. Computed from the
        # markov-corrected logits.
        with torch.no_grad():
            flat_logits = draft_logits_4d.reshape(-1, draft_logits_4d.size(-1))
            pred_ids = flat_logits.argmax(dim=-1)
            flat_targets = target_ids.view(-1)
            ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        # --- L1 / KL distillation losses (draft grad flows through the draft
        # softmax; target distribution stays in no_grad). l1_per_token is
        # always computed when the capture is available because the confidence
        # BCE target consumes it even when l1_loss_alpha == 0.
        # This runs before the position weights because D-PAKL derives its
        # acceptance proxy from the soft-label CE computed here. The position
        # set is eval_mask: both weightings are strictly positive inside it, so
        # the full-vocab work still runs only on positions that reach a loss;
        # truncated suffixes and padded blocks are skipped entirely.
        l1_loss = torch.tensor(0.0, device=device)
        kl_loss = torch.tensor(0.0, device=device)
        l1_per_token = None
        l1_valid = None
        kl_valid = None
        soft_ce_per_token = None
        valid_idx = None
        conf_head_active = (
            self.confidence_head_alpha > 0
            and self.draft_model.confidence_head is not None
        )
        # Only L1/KL need the draft softmax to stay differentiable; the
        # confidence BCE target is consumed detached, so a pure-CE run can build
        # the same per-token L1 under no_grad instead of a checkpoint.
        distill_needs_grad = self.l1_loss_alpha > 0 or self.kl_loss_alpha > 0
        if (
            distill_needs_grad or conf_head_active
        ) and target_last_hidden_states is not None:
            valid_idx = eval_mask.view(-1).nonzero(as_tuple=True)[0]
            l1_per_token = torch.zeros(bsz, n_blocks, self.block_size, device=device)
            num_valid = valid_idx.numel()
            if num_valid > 0:
                target_pred_indices = (safe_label_indices - 1).clamp(min=0)
                flat_target_idx = target_pred_indices.view(-1)[valid_idx]
                batch_idx = torch.div(
                    valid_idx, n_blocks * self.block_size, rounding_mode="floor"
                )
                flat_draft_logits = draft_logits_4d.view(-1, draft_logits_4d.size(-1))
                chunk_size = 512
                l1_chunks = []
                soft_ce_chunks = []
                neg_entropy_chunks = []
                for c_start in range(0, num_valid, chunk_size):
                    c_end = min(c_start + chunk_size, num_valid)
                    sel = valid_idx[c_start:c_end]
                    with torch.no_grad():
                        t_hidden = target_last_hidden_states[
                            batch_idx[c_start:c_end], flat_target_idx[c_start:c_end]
                        ]
                    if distill_needs_grad:
                        (
                            l1_chunk_per_token,
                            soft_ce_chunk_per_token,
                            neg_entropy_chunk,
                        ) = checkpoint(
                            self._distill_chunk_per_token,
                            flat_draft_logits,
                            sel,
                            t_hidden,
                            use_reentrant=False,
                        )
                    else:
                        with torch.no_grad():
                            (
                                l1_chunk_per_token,
                                soft_ce_chunk_per_token,
                                neg_entropy_chunk,
                            ) = self._distill_chunk_per_token(
                                flat_draft_logits, sel, t_hidden
                            )
                    del t_hidden
                    l1_chunks.append(l1_chunk_per_token)
                    if self.kl_loss_alpha > 0:
                        soft_ce_chunks.append(soft_ce_chunk_per_token)
                        neg_entropy_chunks.append(neg_entropy_chunk)
                l1_valid = torch.cat(l1_chunks, dim=0)
                del l1_chunks
                if self.kl_loss_alpha > 0:
                    soft_ce_valid = torch.cat(soft_ce_chunks, dim=0)
                    kl_valid = torch.cat(neg_entropy_chunks, dim=0) + soft_ce_valid
                    # D-PAKL acceptance proxy: a_hat_i = exp(-C_i), the Jensen
                    # lower bound of <p_i, q_psi> (D-PACE App. E, Eq. 10-11).
                    with torch.no_grad():
                        soft_ce_per_token = torch.zeros(
                            bsz, n_blocks, self.block_size, device=device
                        )
                        soft_ce_per_token.view(-1)[
                            valid_idx
                        ] = soft_ce_valid.detach().float()
                    del soft_ce_valid
                del soft_ce_chunks, neg_entropy_chunks
                # Scatter back for the confidence BCE target (consumed under
                # no_grad; masked positions carry weight 0 so zeros are inert).
                with torch.no_grad():
                    l1_per_token.view(-1)[valid_idx] = l1_valid.detach()
        elif distill_needs_grad or conf_head_active:
            raise ValueError(
                "DSpark L1/KL distillation and the confidence head both need the "
                "target's final hidden states, but target_last_hidden_states is "
                "None. Call target_model.enable_last_hidden_state() before "
                "training, or set --confidence-head-alpha 0 together with "
                "--l1-loss-alpha 0 --kl-loss-alpha 0."
            )

        # --- Loss weight mask: static decay, or the D-PACE accepted-length
        # weights (dflash.dpace_weight, Eq. 8). The acceptance proxy is the
        # hard-label draft confidence exp(-CE) when the loss on each position is
        # hard CE / L1, and exp(-C_i) with C_i the soft-label CE under KL
        # distillation: with a soft target no single sampled token exists, so
        # D-PAKL (D-PACE App. E) substitutes the Jensen bound exp(-C_i) for q_i.
        if self.position_weighting == "dpace":
            with torch.no_grad():
                if soft_ce_per_token is not None:
                    accept_proxy = torch.exp(-soft_ce_per_token)
                else:
                    accept_proxy = torch.exp(-ce_per_token.view_as(target_ids))
                dpace_w = dpace_weight(
                    accept_proxy,
                    eval_mask.float(),
                    eval_mask,
                    "dpace",
                    self.dpace_alpha,
                )
            weight_mask = eval_mask.float() * dpace_w
        else:
            weight_mask = eval_mask.float()
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                positions = torch.arange(self.block_size, device=device).view(1, 1, -1)
                decay = torch.exp(-positions.float() / self.loss_decay_gamma)
                weight_mask = weight_mask * decay

        flat_weights = weight_mask.view(-1)
        valid_token_count = flat_weights.sum().detach().clone()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(valid_token_count, op=dist.ReduceOp.SUM)
        valid_token_count = valid_token_count + 1e-6

        # --- CE loss ---
        ce_loss = compute_ce_loss(
            draft_logits_4d, target_ids, weight_mask, valid_token_count
        )

        # --- Weighted reductions of the per-token distillation terms ---
        if l1_valid is not None:
            l1_loss = (
                l1_valid.to(weight_mask.dtype) * flat_weights[valid_idx]
            ).sum() / valid_token_count
        if kl_valid is not None:
            kl_loss = (
                kl_valid.to(weight_mask.dtype) * flat_weights[valid_idx]
            ).sum() / valid_token_count
        del draft_logits, draft_logits_4d

        # --- Confidence loss ---
        conf_loss = torch.tensor(0.0, device=device)
        conf_abs_error = torch.tensor(0.0, device=device)
        if conf_head_active:
            conf_preds = self.draft_model.predict_confidence(
                output_hidden_4d, prev_token_ids=prev_token_ids
            )
            conf_loss = compute_confidence_loss(
                conf_preds,
                l1_per_token,
                weight_mask,
                valid_token_count,
            )
            with torch.no_grad():
                accept_target = (1.0 - 0.5 * l1_per_token).clamp(0, 1)
                conf_abs_error = (
                    (conf_preds.float().sigmoid() - accept_target).abs() * weight_mask
                ).sum() / valid_token_count

        loss = (
            self.ce_loss_alpha * ce_loss
            + self.l1_loss_alpha * l1_loss
            + self.kl_loss_alpha * kl_loss
            + self.confidence_head_alpha * conf_loss
        ) * world_size

        # --- Metrics ---
        with torch.no_grad():
            binary_eval_mask = (weight_mask > 0.5).view(-1)
            correct = (pred_ids == flat_targets) & binary_eval_mask
            accuracy_denom = binary_eval_mask.sum().float()
            actual_count = accuracy_denom + 1e-6
            accuracy = correct.sum().float() / actual_count

            pos_valid = binary_eval_mask.view(-1, self.block_size)
            pos_acc = correct.view(-1, self.block_size).sum(0).float() / (
                pos_valid.sum(0).float() + 1e-6
            )
            pos_loss = (ce_per_token.view(-1, self.block_size) * pos_valid.float()).sum(
                0
            ) / (pos_valid.sum(0).float() + 1e-6)

            valid_mask_4d = (weight_mask > 0).bool()
            pred_accept_len = compute_accept_len(
                pred_ids.view_as(target_ids), target_ids, valid_mask_4d
            )
            valid_block_mask = valid_mask_4d.any(dim=2)
            num_valid_blocks = valid_block_mask.sum().float() + 1e-6
            avg_accept_len = (
                (pred_accept_len + 1.0) * valid_block_mask.float()
            ).sum() / num_valid_blocks

        return (
            loss,
            accuracy,
            {
                "pos_acc": pos_acc,
                "pos_loss": pos_loss,
                "accept_len": avg_accept_len.detach(),
                "accuracy_denom": accuracy_denom,
                "ce_loss": ce_loss.detach(),
                "l1_loss": l1_loss.detach(),
                "kl_loss": kl_loss.detach(),
                "confidence_loss": conf_loss.detach(),
                "confidence_abs_error": conf_abs_error,
            },
        )
