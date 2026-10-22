"""
Causal Weight Coupling (CWC) — causal weight coupling for the Shotgun Segmented Head.

## Problem
Shotgun's Segmented Head reshapes a single projection into K segments, each with an
independent argmax. Segments are fully conditionally independent: segment j's
prediction cannot use any information from the preceding j-1 segments.

## Approach
Inject a "causal cumulative low-rank correction" into segment j's logits:
    logits_j_final = logits_j_base + gate_j * Σ_{k=0}^{j-1} (h @ V_k)(U_k^T)

where (V_k, U_k) is the low-rank factor pair for position k; everything is computed
efficiently via batched matmuls + cumsum.
Parameter overhead is ~3.5M (Qwen3-8B, K=16, r=16) and FLOPs overhead is <0.6%.

## Zero-initialization guarantee
U_stack all zeros + gate all zeros → CWC is a no-op at the start of training, so the
model behaves exactly like the original Shotgun.
Supports training from scratch (no second-stage warmup needed).
"""

import math

import torch
import torch.nn as nn


class CausalWeightCoupling(nn.Module):
    """Causal weight coupling: inject causal positional priors from segments 0..j-1
    into segment j of the Segmented Head.

    Formulation:
        α_k = h @ V_k          ∈ R^{B, r}     # causal features at position k
        β_k = α_k @ U_k^T      ∈ R^{B, V_s}   # contribution of position k to logits
        correction_j = Σ_{k=0}^{j-1} β_k       # causal prefix sum (shift-right cumsum)
        logits_j = L_base_j + gate_j * correction_j

    Efficient implementation: all K α_k / β_k are computed with two batched matmuls,
    and the cumsum runs in parallel over dim=1.
    """

    def __init__(
        self,
        hidden_size: int,       # H: backbone hidden size (e.g. 4096)
        segment_vocab_size: int,  # V_s: per-segment vocab (incl. OOV clusters if hierarchical)
        num_segments: int,      # K: number of segments (e.g. 16)
        rank: int = 16,         # r: rank of the low-rank factors (8/16/32 recommended)
        gate_init: float = 0.0, # initial gate value; 0 keeps zero-init safe
    ):
        super().__init__()
        self.num_segments = num_segments
        self.rank = rank
        self.segment_vocab_size = segment_vocab_size

        # V factors: extract causal features from the hidden vector h.
        # Stored as a flat [K*r, H] matrix so h @ V_flat.T yields α for all K positions
        # at once.
        self.V_flat = nn.Parameter(torch.empty(num_segments * rank, hidden_size))

        # U factors: map causal features into logit space.
        # [K, V_s, r]; all β_k are computed in one batch via einsum('bkr, kvr -> bkv').
        self.U_stack = nn.Parameter(torch.empty(num_segments, segment_vocab_size, rank))

        # One learnable gate scalar per segment, controlling the correction strength.
        # Initialized to 0: the correction is 0 at the start of training, so model
        # behavior is unchanged.
        self.gate = nn.Parameter(torch.full((num_segments,), gate_init))

        self._init_parameters()

    def _init_parameters(self):
        """Initialization strategy:
        - V_flat: Kaiming uniform (same as nn.Linear's default), giving causal feature
          extraction a sensible initial range
        - U_stack: **all zeros** → even with non-zero V_flat, β = α @ U^T = 0, so the
          correction is still 0
        - gate: already set to 0 in the constructor → double zero-init safeguard
        """
        nn.init.kaiming_uniform_(self.V_flat, a=math.sqrt(5))
        nn.init.zeros_(self.U_stack)

    def forward(self, h: torch.Tensor, L_base: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:      [B, H]      backbone hidden vector at the anchor
            L_base: [B, K, V_s] logits of the original Segmented Head (after the
                                shotgun_head projection + reshape)

        Returns:
            [B, K, V_s] causally corrected logits (can be fed directly to the loss or
            argmax)

        Computation:
            1. h @ V_flat.T → α [B, K, r]        # extract per-segment causal features
            2. einsum(α, U_stack) → β [B, K, V_s] # project into logit space
            3. cumsum(β, shift_right) → correction # causal prefix sum
            4. L_base + gate * correction → output # gated fusion
        """
        B = h.size(0)
        K = self.num_segments
        r = self.rank

        # Upcast to float32 for the correction to avoid cumsum precision loss in bf16
        h_c = h.float()
        V_c = self.V_flat.float()
        U_c = self.U_stack.float()
        gate_c = self.gate.float()

        # Step 1: batched causal feature extraction
        # h_c: [B, H] @ V_c.T: [H, K*r] → [B, K*r] → reshape → [B, K, r]
        alpha = (h_c @ V_c.T).reshape(B, K, r)

        # Step 2: batched projection into logit space
        # alpha[b,k,:] @ U_stack[k,:,:].T → beta[b,k,:] ∈ R^{V_s}
        beta = torch.einsum('bkr, kvr -> bkv', alpha, U_c)  # [B, K, V_s]

        # Step 3: causal prefix sum (shift right by 1)
        # correction[j] = Σ_{k=0}^{j-1} β_k
        # Segment 0 gets no correction (no predecessors), so correction[:, 0, :] = 0
        correction = torch.zeros_like(beta)
        if K > 1:
            correction[:, 1:, :] = torch.cumsum(beta[:, :-1, :], dim=1)

        # Step 4: gated fusion
        # gate: [K] → unsqueeze to [1, K, 1] for broadcast multiplication
        gated = gate_c.unsqueeze(0).unsqueeze(-1) * correction  # [B, K, V_s]

        # Add back to the original logits and cast back to the original dtype
        return L_base + gated.to(L_base.dtype)
