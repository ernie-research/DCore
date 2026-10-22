# coding=utf-8
"""DFlash Training Wrapper."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from specforge.modeling.draft.dflash import DFlashDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None

# NPU workaround: flex_attention is not available on Ascend NPU.
if hasattr(torch, "npu") and torch.npu.is_available():
    FLEX_ATTENTION_AVAILABLE = False


# The loss is a (learning objective) x (per-position weighting) product, and the
# two axes are configured independently. Historically they were fused into a
# single ``--loss-type`` string, which left cells such as "CE/L1 mix under D-PACE
# weights" unreachable; the old strings are kept as aliases below so finished
# runs stay reproducible by name.
#
# Objectives: ``ce`` hard-label CE on the sampled token; ``kl`` full-vocab
# KL(p‖q); ``cetv`` the fixed 0.1·CE + 0.9·‖p−q‖₁ mix; ``kltv`` the same mix
# with the soft-label KL; ``lk`` DFly's dynamic λ·KL + (1−λ)·TV interpolation.
_VALID_OBJECTIVES = {"ce", "kl", "cetv", "kltv", "lk"}

# Objectives that need the target's full output distribution, i.e. the target's
# final hidden states must be captured by the target engine.
_DISTILL_OBJECTIVES = {"kl", "cetv", "kltv", "lk"}

# Per-position weightings: ``decay`` is EAGLE-3's static exp(-k/gamma) (0.8^k at
# gamma = 1/ln(1.25)); the ``dpace*`` family is D-PACE's detached continuation
# value, in its full form and its two ablations.
DPACE_LOSS_TYPES = {
    "dpace",
    "dpace-cumulative-confidence-only",
    "dpace-continuation-value-only",
}
_VALID_POSITION_WEIGHTINGS = {"decay"} | DPACE_LOSS_TYPES

# Legacy ``--loss-type`` strings -> (objective, position weighting).
_LOSS_TYPE_ALIASES = {
    "dflash": ("ce", "decay"),
    "dpace": ("ce", "dpace"),
    "dpace-cumulative-confidence-only": (
        "ce",
        "dpace-cumulative-confidence-only",
    ),
    "dpace-continuation-value-only": ("ce", "dpace-continuation-value-only"),
    "dpace-lk": ("lk", "dpace"),
    "kltv": ("kltv", "decay"),
    "cetv": ("cetv", "decay"),
    "dpace-kl": ("kl", "dpace"),
}
_VALID_LOSS_TYPES = set(_LOSS_TYPE_ALIASES) | _VALID_OBJECTIVES


def resolve_loss_spec(
    loss_type: str, position_weighting: Optional[str] = None
) -> Tuple[str, str]:
    """Resolve ``--loss-type`` (+ optional override) to (objective, weighting).

    A canonical objective name defaults to the static ``decay`` weighting; a
    legacy alias defaults to the weighting it used to imply. An explicit
    ``position_weighting`` always wins, which is what makes the off-diagonal
    cells (e.g. ``cetv`` under D-PACE weights) reachable.
    """
    if loss_type in _LOSS_TYPE_ALIASES:
        objective, default_weighting = _LOSS_TYPE_ALIASES[loss_type]
    elif loss_type in _VALID_OBJECTIVES:
        objective, default_weighting = loss_type, "decay"
    else:
        raise ValueError(
            f"loss_type={loss_type!r}; must be one of {sorted(_VALID_LOSS_TYPES)}"
        )

    weighting = default_weighting if position_weighting is None else position_weighting
    if weighting not in _VALID_POSITION_WEIGHTINGS:
        raise ValueError(
            f"position_weighting={weighting!r}; must be one of "
            f"{sorted(_VALID_POSITION_WEIGHTINGS)}"
        )
    return objective, weighting

# Positions per chunk in the full-vocab distillation losses (the LK family and
# the DSpark variant's L1). The fp32 probability tensors are the dominant cost
# (vocab x 4B per position, per distribution), so the per-position terms are
# accumulated chunk by chunk under recomputation rather than materialized for
# all positions at once. 1024 is the block-chunk of 32 that measured 1.27x
# faster than 16 at bsz2 x 512 anchors x block16 on H800 (for +0.15 GiB peak),
# restated in positions now that the loops skip the zero-weight ones instead of
# walking whole blocks.
_DISTILL_POSITION_CHUNK = 1024


def dpace_weight(
    prob: torch.Tensor,
    binary_mask: torch.Tensor,
    binary_mask_b: torch.Tensor,
    loss_type: str,
    dpace_alpha: float,
) -> torch.Tensor:
    """Compute detached D-PACE position weights.

    ``prob`` is the draft probability on the target token at each draft
    position. Invalid positions are treated as multiplicative no-ops inside
    prefix products and excluded from suffix sums; the caller still
    multiplies the returned weights by ``binary_mask`` before reduction.
    """
    smooth = (1.0 - dpace_alpha) * prob + dpace_alpha
    smooth = torch.where(binary_mask_b, smooth, torch.ones_like(smooth))
    prefix = torch.cumprod(smooth, dim=-1)

    if loss_type == "dpace-cumulative-confidence-only":
        return prefix

    suffix = torch.flip(
        torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1),
        dims=[-1],
    )

    if loss_type == "dpace":
        return suffix
    if loss_type == "dpace-continuation-value-only":
        return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
    raise ValueError(f"unknown D-PACE loss_type {loss_type!r}")


def compute_accept_len(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Compute per-block acceptance length."""
    correct = (pred_ids_4d == target_ids_4d) | (~valid_mask_4d)
    accept_prefix = correct.long().cumprod(dim=2) * valid_mask_4d.long()
    return accept_prefix.sum(dim=2).float()


def create_dflash_sdpa_mask(anchor_positions, block_keep_mask, S, block_size, device):
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)  # (1, 1, Q_LEN, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(
        1, 1, 1, -1
    )  # (1, 1, 1, KV_LEN)

    q_block_ids = q_indices // block_size

    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    final_mask = (mask_context | mask_draft) & valid_block
    return final_mask


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
):
    """Construct Flex Attention BlockMask for DFlash training.

    KV: [Context (S tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos).
      2. Intra-block attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    return create_block_mask(
        dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with DFlash and D-PACE losses."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        loss_type: str = "dflash",
        position_weighting: Optional[str] = None,
        dpace_alpha: float = 0.5,
        anchor_prediction: bool = False,
        lk_eta: float = 3.0,
        kltv_kl_alpha: float = 0.1,
        kltv_l1_alpha: float = 0.9,
        cetv_ce_alpha: float = 0.1,
        cetv_l1_alpha: float = 0.9,
    ):
        super().__init__()
        objective, position_weighting = resolve_loss_spec(loss_type, position_weighting)
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dpace_alpha must be in [0, 1], got {dpace_alpha}")
        if lk_eta < 0:
            raise ValueError(f"lk_eta must be >= 0, got {lk_eta}")
        if kltv_kl_alpha < 0:
            raise ValueError(f"kltv_kl_alpha must be >= 0, got {kltv_kl_alpha}")
        if kltv_l1_alpha < 0:
            raise ValueError(f"kltv_l1_alpha must be >= 0, got {kltv_l1_alpha}")
        if cetv_ce_alpha < 0:
            raise ValueError(f"cetv_ce_alpha must be >= 0, got {cetv_ce_alpha}")
        if cetv_l1_alpha < 0:
            raise ValueError(f"cetv_l1_alpha must be >= 0, got {cetv_l1_alpha}")

        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.anchor_prediction = anchor_prediction
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.loss_type = loss_type
        self.objective = objective
        self.position_weighting = position_weighting
        # D-PACE Eq (18) reduces the weighted hard-CE sum by the batch size
        # rather than by the weight sum; that is what the finished hard-label
        # D-PACE runs used, so the pair keeps it. Every other pair (including
        # D-PAKL) takes the weighted mean.
        self.batch_reduction = objective == "ce" and position_weighting in (
            DPACE_LOSS_TYPES
        )
        self.dpace_alpha = dpace_alpha
        self.lk_eta = float(lk_eta)
        self.kltv_kl_alpha = float(kltv_kl_alpha)
        self.kltv_l1_alpha = float(kltv_l1_alpha)
        self.cetv_ce_alpha = float(cetv_ce_alpha)
        self.cetv_l1_alpha = float(cetv_l1_alpha)

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            raise ValueError("should preprocess the data.")

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
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

    def _dpace_weight(
        self,
        prob: torch.Tensor,
        binary_mask: torch.Tensor,
        binary_mask_b: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        return dpace_weight(
            prob, binary_mask, binary_mask_b, loss_type, self.dpace_alpha
        )

    def _decay_weights(self, device: torch.device) -> Optional[torch.Tensor]:
        """Exponential position decay ``exp(-(k - offset)⁺ / gamma)``, or None.

        Shaped ``[1, 1, block_size]``. Note ``0.8^k == exp(-k / gamma)`` for
        ``gamma = 1 / ln(1.25) ≈ 4.4814``, which is how EAGLE-3's position
        weights are expressed here.
        """
        if self.loss_decay_gamma is None or self.loss_decay_gamma <= 0:
            return None
        k = torch.arange(self.block_size, device=device).view(1, 1, -1)
        offset = 0 if self.anchor_prediction else 1
        return torch.exp(-(k - offset).clamp(min=0).float() / self.loss_decay_gamma)

    def _lk_chunk_terms(
        self,
        flat_draft_logits: torch.Tensor,
        position_idx: torch.Tensor,
        target_hidden_chunk: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-position distribution loss, TV, KL and soft-label CE for one chunk.

        ``position_idx`` selects the chunk's positions out of the flattened
        ``[bsz · n_blocks · block_size, vocab]`` draft logits; the gather happens
        inside so the checkpoint retains the already-alive full logits tensor
        rather than a per-chunk gathered copy. ``target_hidden_chunk`` is the
        matching ``[chunk, hidden]`` slice of the target's final hidden states.

        For ``lk``: ``LK = λ·KL(p‖q) + (1−λ)·TV`` with
        ``λ = exp(−η·sg[1−TV])`` — the dynamic interpolation of DFly's Eq (6):
        KL dominates while the single-position acceptance rate ``α = 1−TV`` is
        low (it has a non-vanishing gradient on tokens the draft assigns ~0
        mass), and TV takes over as α rises (it is the quantity acceptance
        actually depends on).

        For ``kltv``: the fixed mix ``0.1·KL + 0.9·‖p−q‖₁`` (DSpark's configured
        ratio with the hard-label CE swapped for the soft-label one).

        ``tv_loss`` / ``kl_loss`` are reported on the TV scale in both cases so
        the two objectives stay directly comparable.

        The position weights are applied by the caller rather than here: D-PAKL
        builds them from the soft-label CE ``C_i = −Σ p·log q`` returned as the
        fourth value (D-PACE App. E).

        Every full-vocab temporary here costs ``positions × vocab × 4B``, so the
        arithmetic is written to materialize as few of them as possible: the
        upcast is folded into the softmax via ``dtype=`` instead of a separate
        ``.float()`` copy, the L1 distance is one fused reduction, and the
        target's entropy is reduced on its own rather than through a full-vocab
        subtraction.
        """
        draft_logits_chunk = flat_draft_logits.index_select(0, position_idx)
        # bf16/fp16 activations accumulate in fp32; float64 (the CPU formula
        # tests) keeps its precision.
        acc_dtype = torch.promote_types(draft_logits_chunk.dtype, torch.float32)

        with torch.no_grad():
            # The hidden states arrive already flattened to 2-D. With a 4-D
            # input, F.linear dispatches to aten::bmm with a broadcast weight,
            # which re-streams the whole [hidden, vocab] matrix per batch element
            # and runs ~16x slower than the equivalent 2-D GEMM (measured 13.3 ms
            # vs 0.82 ms per chunk on H800). Numerically identical.
            target_logits = self.lm_head(target_hidden_chunk)
            target_probs = torch.softmax(target_logits, dim=-1, dtype=acc_dtype)
            del target_logits
            # Σ p·log p has no gradient path (p is detached and λ is detached),
            # so it is reduced to one scalar per position here and added back
            # below, instead of subtracting two full-vocab tensors.
            neg_target_entropy = torch.xlogy(target_probs, target_probs).sum(dim=-1)

        draft_log_probs = torch.log_softmax(draft_logits_chunk, dim=-1, dtype=acc_dtype)

        # ‖p − q‖₁, and TV = ½·‖p − q‖₁ — vector_norm fuses the abs and the sum
        # into a single reduction, so the abs never lands in memory.
        l1 = torch.linalg.vector_norm(
            target_probs - draft_log_probs.exp(), ord=1, dim=-1
        )
        tv = 0.5 * l1
        # C_i = −Σ p·log q. KL(p‖q) = C_i + Σ p·log p, and the entropy offset is
        # draft-independent, so the two differ only by a constant under backward.
        soft_ce = -(target_probs * draft_log_probs).sum(dim=-1)
        kl = neg_target_entropy + soft_ce

        if self.objective == "kltv":
            mixed = self.kltv_kl_alpha * kl + self.kltv_l1_alpha * l1
        elif self.objective == "cetv":
            # The hard-label CE term is added token-wise in forward(), outside
            # this chunked full-vocab loop.
            mixed = self.cetv_l1_alpha * l1
        elif self.objective == "kl":
            mixed = kl
        else:
            lam = torch.exp(-self.lk_eta * (1.0 - tv).detach().clamp(0.0, 1.0))
            mixed = lam * kl + (1.0 - lam) * tv
        return mixed, tv.detach(), kl.detach(), soft_ce.detach()

    def _compute_lk_terms(
        self,
        draft_logits_4d: torch.Tensor,
        target_last_hidden_states: torch.Tensor,
        safe_label_indices: torch.Tensor,
        weight_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-position ``(loss, TV, KL, soft-label CE)`` over all anchors.

        Only positions with ``weight_mask > 0`` get the full-vocab treatment:
        every consumer multiplies these tensors by weights derived from the same
        mask before reducing, and ``dpace_weight`` replaces the masked entries of
        the soft-label CE with a multiplicative 1.0 before its prefix product, so
        padded blocks, out-of-bounds labels, prompt tokens and (without
        ``anchor_prediction``) each block's position 0 can be skipped outright
        rather than computed and multiplied by zero. The skipped entries are
        filled with 0.

        Chunked over the selected positions under recomputation: the fp32
        probability tensors are ~0.6 GB per 1024-position chunk at Qwen3's
        vocabulary and would not fit for all positions at once, let alone be
        retained for backward. Only the per-position scalars (four
        ``[bsz, n_blocks, block_size]`` tensors) cross the checkpoint boundary.

        The weighting and reduction happen in the caller because D-PAKL derives
        its position weights from the soft-label CE returned here.
        """
        bsz, n_blocks, block_size, vocab_size = draft_logits_4d.shape
        device = draft_logits_4d.device
        per_position_shape = (bsz, n_blocks, block_size)
        num_positions = bsz * n_blocks * block_size
        acc_dtype = torch.promote_types(draft_logits_4d.dtype, torch.float32)

        valid_idx = (weight_mask.reshape(-1) > 0).nonzero(as_tuple=True)[0]
        num_valid = valid_idx.numel()
        if num_valid == 0:
            zeros = torch.zeros(per_position_shape, dtype=acc_dtype, device=device)
            return zeros, zeros.clone(), zeros.clone(), zeros.clone()

        # The target hidden state at index i predicts the token at i + 1, so the
        # label at safe_label_indices is predicted from safe_label_indices - 1.
        target_pred_indices = (safe_label_indices - 1).clamp(min=0)
        flat_target_idx = target_pred_indices.reshape(-1)[valid_idx]
        batch_idx = torch.div(valid_idx, n_blocks * block_size, rounding_mode="floor")
        flat_draft_logits = draft_logits_4d.reshape(-1, vocab_size)

        mixed_chunks = []
        tv_chunks = []
        kl_chunks = []
        soft_ce_chunks = []

        for start in range(0, num_valid, _DISTILL_POSITION_CHUNK):
            end = min(start + _DISTILL_POSITION_CHUNK, num_valid)
            with torch.no_grad():
                target_hidden_chunk = target_last_hidden_states[
                    batch_idx[start:end], flat_target_idx[start:end]
                ]
            chunk_mixed, chunk_tv, chunk_kl, chunk_soft_ce = checkpoint(
                self._lk_chunk_terms,
                flat_draft_logits,
                valid_idx[start:end],
                target_hidden_chunk,
                use_reentrant=False,
            )
            mixed_chunks.append(chunk_mixed)
            tv_chunks.append(chunk_tv)
            kl_chunks.append(chunk_kl)
            soft_ce_chunks.append(chunk_soft_ce)

        mixed_valid = torch.cat(mixed_chunks, dim=0)
        del mixed_chunks
        # index_copy keeps the gradient path to the selected positions; the rest
        # stay at 0 and are multiplied by a 0 weight downstream.
        mixed = torch.zeros(
            num_positions, dtype=mixed_valid.dtype, device=device
        ).index_copy(0, valid_idx, mixed_valid)

        def _scatter_detached(chunks: list) -> torch.Tensor:
            flat = torch.zeros(num_positions, dtype=acc_dtype, device=device)
            flat[valid_idx] = torch.cat(chunks, dim=0).to(acc_dtype)
            return flat.view(per_position_shape)

        with torch.no_grad():
            tv = _scatter_detached(tv_chunks)
            kl = _scatter_detached(kl_chunks)
            soft_ce = _scatter_detached(soft_ce_chunks)

        return mixed.view(per_position_shape), tv, kl, soft_ce

    def _forward_draft_blocks(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )
        return anchor_positions, block_keep_mask, output_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Parallel block-wise training forward pass; returns
        (loss, accuracy, metrics) — same shape as Domino's forward.

        ``target_last_hidden_states`` (``[bsz, seq_len, hidden]``, post-norm) is
        only needed by the distribution objectives (``kl``/``cetv``/``kltv``/
        ``lk``), which build the target distribution as
        ``lm_head(target_last_hidden_states)``.
        """
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )

        logits = self.lm_head(output_hidden)

        # --- Labels: position k predicts token at anchor+k (standard) or anchor+k+1 (anchor_prediction) ---
        label_start = 1 if self.anchor_prediction else 0
        label_offsets = torch.arange(
            label_start, label_start + self.block_size, device=device
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # --- Weight mask: block validity * bounds * (optionally exclude pos 0) * loss_mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        if not self.anchor_prediction:
            pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
            weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        binary_eval_mask = weight_mask.view(-1)

        # --- Cross entropy ---
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)

        # The hard-label CE only needs an autograd graph when it is part of the
        # objective (``ce`` / ``cetv``). The distribution objectives read it at
        # most detached, to build D-PACE's hard-label acceptance proxy, and
        # building the graph there would retain a full-vocab log-softmax for a
        # backward that never reaches it.
        ce_needs_grad = self.objective in ("ce", "cetv") and torch.is_grad_enabled()
        with torch.set_grad_enabled(ce_needs_grad):
            loss_per_token = F.cross_entropy(
                flat_logits, flat_targets, reduction="none"
            )

        extra_metrics: Dict[str, torch.Tensor] = {}

        # --- Objective terms over the target's full distribution (chunked) ---
        distill_per_token = None
        soft_ce_per_token = None
        if self.objective in _DISTILL_OBJECTIVES:
            if target_last_hidden_states is None:
                raise ValueError(
                    f"objective={self.objective!r} requires "
                    "target_last_hidden_states. Build the target engine so it "
                    "captures the final hidden states "
                    "(DFlashTargetEngine.enable_last_hidden_state)."
                )
            distill_per_token, tv_per_token, kl_per_token, soft_ce_per_token = (
                self._compute_lk_terms(
                    logits.view(bsz, anchor_positions.size(1), self.block_size, -1),
                    target_last_hidden_states,
                    safe_label_indices,
                    weight_mask,
                )
            )

        # --- Per-position weights ---
        if self.position_weighting == "decay":
            # EAGLE-3's static position weights (0.8^k via gamma).
            loss_weights = weight_mask
            decay_weights = self._decay_weights(device)
            if decay_weights is not None:
                loss_weights = loss_weights * decay_weights
        else:
            # D-PACE's detached continuation value, built from a single-position
            # acceptance proxy. With a hard target that proxy is q(z*) =
            # exp(-CE(z*)); with a soft target there is no single sampled token,
            # so it is the Jensen lower bound â_i = exp(-C_i) of ⟨p_i, q_ψ⟩ with
            # C_i = -Σ p·log q the soft-label CE (D-PAKL, D-PACE App. E).
            with torch.no_grad():
                if self.objective in ("kl", "kltv"):
                    accept_proxy = torch.exp(-soft_ce_per_token)
                else:
                    accept_proxy = torch.exp(-loss_per_token.view_as(target_ids))
                dpace_weights = self._dpace_weight(
                    accept_proxy,
                    weight_mask,
                    weight_mask > 0,
                    self.position_weighting,
                )
            loss_weights = weight_mask * dpace_weights

        denom = float(bsz) if self.batch_reduction else loss_weights.sum() + 1e-6

        # --- Reduction ---
        if self.objective == "ce":
            loss = (loss_per_token * loss_weights.view(-1)).sum() / denom
        else:
            loss = (distill_per_token * loss_weights).sum() / denom
            if self.objective == "cetv":
                # Same weighted reduction as the pure-CE objective, over the
                # same weights as the L1 term.
                loss = loss + self.cetv_ce_alpha * (
                    (loss_per_token * loss_weights.view(-1)).sum() / denom
                )
            extra_metrics = {
                "tv_loss": (tv_per_token * loss_weights).sum() / denom,
                "kl_loss": (kl_per_token * loss_weights).sum() / denom,
            }

        # --- Accuracy ---
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            accuracy_denom = binary_eval_mask.sum()
            accuracy = correct.sum().float() / (accuracy_denom + 1e-6)

        metrics = {"accuracy_denom": accuracy_denom}
        metrics.update(extra_metrics)
        return loss, accuracy, metrics


class OnlineDominoModel(OnlineDFlashModel):
    """Domino online training wrapper over DFlash block-parallel components."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        shift_label: bool = False,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            loss_type="dflash",
        )
        self.shift_label = shift_label

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = max(1, min(self.num_anchors, int(valid_counts.max().item()) - 1))

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def _build_domino_head_inputs(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_ids: torch.Tensor,
        output_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n, bs = target_ids.shape
        hidden4d = output_hidden.reshape(bsz, n, bs, output_hidden.shape[-1])

        prev_ids = target_ids
        if self.shift_label:
            prev_offsets = torch.arange(
                0, self.block_size, device=input_ids.device
            ).view(1, 1, -1)
            prev_indices = (anchor_positions.unsqueeze(-1) + prev_offsets).clamp(
                max=input_ids.size(1) - 1
            )
            prev_ids = torch.gather(
                input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
                2,
                prev_indices,
            )

        return hidden4d, prev_ids

    def _apply_domino_head(
        self,
        base_logits4d: torch.Tensor,
        hidden4d: torch.Tensor,
        prev_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        head_token_ids = prev_ids if self.shift_label else target_ids
        head_token_embeddings = self.embed_tokens(head_token_ids)
        return self.draft_model.apply_logits_head(
            base_logits4d,
            hidden_states=hidden4d,
            prev_token_embeddings=head_token_embeddings,
        )

    def _compute_extra_metrics(
        self,
        pred_ids: torch.Tensor,
        flat_base_logits: torch.Tensor,
        flat_targets: torch.Tensor,
        binary_eval_mask: torch.Tensor,
        actual_token_count: torch.Tensor,
        target_ids: torch.Tensor,
        eval_weight_mask: torch.Tensor,
        final_loss: torch.Tensor,
        base_loss: torch.Tensor,
        lambda_base: float,
    ) -> Dict[str, torch.Tensor]:
        bsz, n, bs = target_ids.shape

        base_pred_ids = torch.argmax(flat_base_logits, dim=-1)
        base_correct = (base_pred_ids == flat_targets) & (binary_eval_mask > 0.5)
        base_accuracy = base_correct.sum().float() / actual_token_count

        valid_mask_4d = (eval_weight_mask > 0).bool()
        pred_accept_len = compute_accept_len(
            pred_ids.view(bsz, n, bs), target_ids, valid_mask_4d
        )
        base_accept_len = compute_accept_len(
            base_pred_ids.view(bsz, n, bs), target_ids, valid_mask_4d
        )

        valid_block_mask = valid_mask_4d.any(dim=2)
        num_valid_blocks = valid_block_mask.sum().float() + 1e-6
        avg_accept_len = (
            (pred_accept_len + 1.0) * valid_block_mask.float()
        ).sum() / num_valid_blocks
        base_avg_accept_len = (
            (base_accept_len + 1.0) * valid_block_mask.float()
        ).sum() / num_valid_blocks

        return {
            "final_loss": final_loss.detach(),
            "base_loss": base_loss.detach(),
            "base_accuracy": base_accuracy.detach(),
            "accept_len": avg_accept_len.detach(),
            "base_accept_len": base_avg_accept_len.detach(),
            "lambda_base": torch.tensor(lambda_base, device=final_loss.device),
        }

    def _compute_weighted_losses(
        self,
        final_logits: torch.Tensor,
        base_logits: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        lambda_base: float,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        flat_logits = final_logits.reshape(-1, final_logits.size(-1))
        flat_base_logits = base_logits.reshape(-1, base_logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)

        valid_token_count = flat_weights.sum() + 1e-6

        final_loss_per_token = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        )
        final_loss = (final_loss_per_token * flat_weights).sum() / valid_token_count

        base_loss_per_token = F.cross_entropy(
            flat_base_logits, flat_targets, reduction="none"
        )
        base_loss = (base_loss_per_token * flat_weights).sum() / valid_token_count

        loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss

        return loss, final_loss, base_loss, flat_logits, flat_base_logits, flat_targets

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        lambda_base: float = 0.0,
    ):
        """Parallel Domino training forward pass."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )

        label_start = 1 if self.shift_label else 0
        label_offsets = torch.arange(
            label_start, label_start + self.block_size, device=device
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_target_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_target_indices,
        )

        bsz, n, bs = target_ids.shape
        base_logits = self.lm_head(output_hidden)
        hidden4d, prev_ids = self._build_domino_head_inputs(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            target_ids=target_ids,
            output_hidden=output_hidden,
        )
        base_logits4d = base_logits.reshape(bsz, n, bs, -1)
        final_logits = self._apply_domino_head(
            base_logits4d=base_logits4d,
            hidden4d=hidden4d,
            prev_ids=prev_ids,
            target_ids=target_ids,
        ).reshape(bsz, n * bs, -1)

        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        if not self.shift_label:
            pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
            weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_target_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        eval_weight_mask = weight_mask.clone()
        binary_eval_mask = weight_mask.view(-1)

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            offset = 0 if self.shift_label else 1
            decay_weights = torch.exp(
                -(k - offset).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        loss, final_loss, base_loss, flat_logits, flat_base_logits, flat_targets = (
            self._compute_weighted_losses(
                final_logits=final_logits,
                base_logits=base_logits,
                target_ids=target_ids,
                weight_mask=weight_mask,
                lambda_base=lambda_base,
            )
        )

        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            accuracy_denom = binary_eval_mask.sum()
            actual_token_count = accuracy_denom + 1e-6
            accuracy = correct.sum().float() / actual_token_count

            metrics = self._compute_extra_metrics(
                pred_ids=pred_ids,
                flat_base_logits=flat_base_logits,
                flat_targets=flat_targets,
                binary_eval_mask=binary_eval_mask,
                actual_token_count=actual_token_count,
                target_ids=target_ids,
                eval_weight_mask=eval_weight_mask,
                final_loss=final_loss,
                base_loss=base_loss,
                lambda_base=lambda_base,
            )
            metrics["accuracy_denom"] = accuracy_denom

        return loss, accuracy, metrics
