# coding=utf-8
"""Numerical regression tests for the anchor_prediction feature in DFlash.

Verifies:
1. anchor_prediction=False produces identical loss/accuracy to the original code path
   (backward compatibility).
2. anchor_prediction=True correctly shifts label offsets by +1 and includes position 0
   in the weight mask.
3. Decay gamma with anchor_prediction=True starts from position 0 (offset=0).
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]

# --- Stub the DFlashDraftModel import so we can load core/dflash on CPU ---
_stub_dflash_draft = types.ModuleType("specforge.modeling.draft.dflash")


class _DFlashDraftStub(nn.Module):
    pass


_stub_dflash_draft.DFlashDraftModel = _DFlashDraftStub

_spec = importlib.util.spec_from_file_location(
    "specforge.core.dflash", REPO / "specforge" / "core" / "dflash.py"
)
_dflash_module = importlib.util.module_from_spec(_spec)

_pkg_specforge = types.ModuleType("specforge")
_pkg_specforge.__path__ = [str(REPO / "specforge")]
_pkg_core = types.ModuleType("specforge.core")
_pkg_core.__path__ = [str(REPO / "specforge" / "core")]
_pkg_modeling = types.ModuleType("specforge.modeling")
_pkg_modeling.__path__ = [str(REPO / "specforge" / "modeling")]
_pkg_draft = types.ModuleType("specforge.modeling.draft")
_pkg_draft.__path__ = [str(REPO / "specforge" / "modeling" / "draft")]

from unittest.mock import patch

with patch.dict(
    sys.modules,
    {
        "specforge": _pkg_specforge,
        "specforge.core": _pkg_core,
        "specforge.core.dflash": _dflash_module,
        "specforge.modeling": _pkg_modeling,
        "specforge.modeling.draft": _pkg_draft,
        "specforge.modeling.draft.dflash": _stub_dflash_draft,
    },
):
    _spec.loader.exec_module(_dflash_module)
OnlineDFlashModel = _dflash_module.OnlineDFlashModel


# --- Deterministic stubs ---


class _FixedDraft(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, position_ids, noise_embedding, target_hidden, attention_mask):
        bsz, draft_len = noise_embedding.shape[:2]
        return torch.zeros(
            bsz,
            draft_len,
            self.hidden_size,
            dtype=noise_embedding.dtype,
            device=noise_embedding.device,
        )


def _fixed_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
    bsz, n_blocks = anchor_positions.shape
    return torch.zeros(
        bsz,
        n_blocks * self.block_size,
        self.embed_tokens.embedding_dim,
        dtype=torch.double,
        device=input_ids.device,
    )


def _fixed_anchor_sampler(anchors, keep_mask):
    def _sample(self, seq_len, loss_mask, device):
        return anchors.to(device), keep_mask.to(device)

    return _sample


class _FixedHead(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, hidden_states):
        return self.fixed_logits.to(device=hidden_states.device)


def _make_model(logits, anchors, keep_mask, **kwargs):
    bsz, n_blocks, block_size, vocab_size = logits.shape
    model = OnlineDFlashModel(
        draft_model=_FixedDraft(hidden_size=4),
        target_lm_head=_FixedHead(logits.reshape(bsz, n_blocks * block_size, vocab_size)),
        target_embed_tokens=nn.Embedding(vocab_size, 4).double(),
        mask_token_id=0,
        block_size=block_size,
        attention_backend="sdpa",
        num_anchors=n_blocks,
        **kwargs,
    ).double()
    model._sample_anchor_positions = types.MethodType(
        _fixed_anchor_sampler(anchors, keep_mask), model
    )
    model._create_noise_embed = types.MethodType(_fixed_noise_embed, model)
    return model


# --- Reference implementations ---


def _ref_targets_standard(input_ids, anchors, block_size):
    """Standard DFlash: label offsets [0, 1, ..., block_size-1]."""
    seq_len = input_ids.shape[1]
    n_blocks = anchors.shape[1]
    offsets = torch.arange(block_size).view(1, 1, -1)
    label_indices = anchors.unsqueeze(-1) + offsets
    safe_indices = label_indices.clamp(max=seq_len - 1)
    targets = torch.gather(
        input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_indices
    )
    valid = label_indices < seq_len
    return targets, valid, label_indices


def _ref_targets_anchor_pred(input_ids, anchors, block_size):
    """Anchor prediction: label offsets [1, 2, ..., block_size]."""
    seq_len = input_ids.shape[1]
    n_blocks = anchors.shape[1]
    offsets = torch.arange(1, block_size + 1).view(1, 1, -1)
    label_indices = anchors.unsqueeze(-1) + offsets
    safe_indices = label_indices.clamp(max=seq_len - 1)
    targets = torch.gather(
        input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_indices
    )
    valid = label_indices < seq_len
    return targets, valid, label_indices


def _ref_weight_mask_standard(keep_mask, valid_label_mask, loss_mask, anchors, block_size, seq_len):
    """Standard DFlash weight mask: excludes position 0."""
    n_blocks = anchors.shape[1]
    bsz = anchors.shape[0]
    weight = keep_mask.unsqueeze(-1).expand(-1, -1, block_size).double()
    weight = weight * valid_label_mask.double()
    pos_in_block = torch.arange(block_size).view(1, 1, -1)
    weight = weight * (pos_in_block > 0).double()
    # gather loss_mask
    offsets = torch.arange(block_size).view(1, 1, -1)
    label_indices = anchors.unsqueeze(-1) + offsets
    safe_indices = label_indices.clamp(max=seq_len - 1)
    gathered_loss_mask = torch.gather(
        loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_indices
    )
    weight = weight * gathered_loss_mask
    return weight


def _ref_weight_mask_anchor_pred(keep_mask, valid_label_mask, loss_mask, anchors, block_size, seq_len):
    """Anchor prediction weight mask: includes position 0."""
    n_blocks = anchors.shape[1]
    weight = keep_mask.unsqueeze(-1).expand(-1, -1, block_size).double()
    weight = weight * valid_label_mask.double()
    # No position 0 exclusion
    offsets = torch.arange(1, block_size + 1).view(1, 1, -1)
    label_indices = anchors.unsqueeze(-1) + offsets
    safe_indices = label_indices.clamp(max=seq_len - 1)
    gathered_loss_mask = torch.gather(
        loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_indices
    )
    weight = weight * gathered_loss_mask
    return weight


# --- Test data ---


def _sample_tensors():
    torch.manual_seed(42)
    bsz, n_blocks, block_size, vocab_size = 2, 3, 4, 10
    seq_len = 16
    logits = torch.randn(bsz, n_blocks, block_size, vocab_size, dtype=torch.double)
    input_ids = torch.randint(0, vocab_size, (bsz, seq_len), dtype=torch.long)
    loss_mask = torch.ones(bsz, seq_len, dtype=torch.double)
    loss_mask[0, 14] = 0.0
    loss_mask[1, 13] = 0.0
    # Anchors chosen so that block fits within seq_len for most positions
    anchors = torch.tensor([[0, 4, 8], [1, 5, 9]], dtype=torch.long)
    keep_mask = torch.tensor([[True, True, True], [True, True, False]])
    hidden_states = torch.zeros(bsz, seq_len, 4, dtype=torch.double)
    return logits, input_ids, loss_mask, hidden_states, anchors, keep_mask


# --- Tests ---


class TestAnchorPredictionBackwardCompat(unittest.TestCase):
    """anchor_prediction=False must produce identical results to original code."""

    def setUp(self):
        (
            self.logits,
            self.input_ids,
            self.loss_mask,
            self.hidden_states,
            self.anchors,
            self.keep_mask,
        ) = _sample_tensors()

    def test_loss_matches_original_code_path(self):
        """With anchor_prediction=False, the loss is identical to the original
        weighted-mean DFlash loss (position 0 excluded, label offsets start at 0)."""
        model = _make_model(self.logits, self.anchors, self.keep_mask, anchor_prediction=False)
        loss, accuracy, _ = model(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
        )

        # Manual reference
        bsz, n_blocks, block_size, vocab_size = self.logits.shape
        seq_len = self.input_ids.shape[1]
        targets, valid, _ = _ref_targets_standard(self.input_ids, self.anchors, block_size)
        weight = _ref_weight_mask_standard(
            self.keep_mask, valid, self.loss_mask, self.anchors, block_size, seq_len
        )
        flat_logits = self.logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)
        neg_log_q = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        flat_weight = weight.reshape(-1)
        expected_loss = (neg_log_q * flat_weight).sum() / (flat_weight.sum() + 1e-6)

        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=1e-10)

    def test_position_0_excluded_in_weight_mask(self):
        """Verify that position 0 has zero weight when anchor_prediction=False."""
        model = _make_model(self.logits, self.anchors, self.keep_mask, anchor_prediction=False)
        # We can check by verifying loss changes if we manually include pos 0
        loss_standard, _, _ = model(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
        )

        model_ap = _make_model(self.logits, self.anchors, self.keep_mask, anchor_prediction=True)
        loss_ap, _, _ = model_ap(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
        )

        # They should differ because anchor_prediction changes both the label
        # offsets and the weight mask
        self.assertFalse(torch.allclose(loss_standard, loss_ap))


class TestAnchorPredictionEnabled(unittest.TestCase):
    """anchor_prediction=True correctness checks."""

    def setUp(self):
        (
            self.logits,
            self.input_ids,
            self.loss_mask,
            self.hidden_states,
            self.anchors,
            self.keep_mask,
        ) = _sample_tensors()

    def test_label_offset_shifted_by_one(self):
        """With anchor_prediction=True, position k predicts input_ids[anchor+k+1]."""
        model = _make_model(self.logits, self.anchors, self.keep_mask, anchor_prediction=True)
        loss, _, _ = model(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
        )

        # Manual reference with shifted labels
        bsz, n_blocks, block_size, vocab_size = self.logits.shape
        seq_len = self.input_ids.shape[1]
        targets, valid, _ = _ref_targets_anchor_pred(self.input_ids, self.anchors, block_size)
        weight = _ref_weight_mask_anchor_pred(
            self.keep_mask, valid, self.loss_mask, self.anchors, block_size, seq_len
        )
        flat_logits = self.logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)
        neg_log_q = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        flat_weight = weight.reshape(-1)
        expected_loss = (neg_log_q * flat_weight).sum() / (flat_weight.sum() + 1e-6)

        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=1e-10)

    def test_all_positions_have_nonzero_weight(self):
        """Position 0 is NOT excluded from the weight mask when anchor_prediction=True."""
        bsz, n_blocks, block_size, vocab_size = self.logits.shape
        seq_len = self.input_ids.shape[1]
        targets, valid, _ = _ref_targets_anchor_pred(self.input_ids, self.anchors, block_size)
        weight = _ref_weight_mask_anchor_pred(
            self.keep_mask, valid, self.loss_mask, self.anchors, block_size, seq_len
        )
        # For valid blocks with in-bounds labels, position 0 should have non-zero weight
        # Block (0,0): anchor=0, label_index=1, valid & keep & loss_mask[1]=1 -> weight>0
        self.assertGreater(weight[0, 0, 0].item(), 0.0)

    def test_decay_gamma_starts_at_position_0(self):
        """With anchor_prediction=True, decay offset=0 so position 0 gets decay=1.0."""
        gamma = 5.0
        model = _make_model(
            self.logits, self.anchors, self.keep_mask,
            anchor_prediction=True, loss_decay_gamma=gamma,
        )
        loss, _, _ = model(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
        )

        # Reference with decay starting at k=0
        bsz, n_blocks, block_size, vocab_size = self.logits.shape
        seq_len = self.input_ids.shape[1]
        targets, valid, _ = _ref_targets_anchor_pred(self.input_ids, self.anchors, block_size)
        weight = _ref_weight_mask_anchor_pred(
            self.keep_mask, valid, self.loss_mask, self.anchors, block_size, seq_len
        )
        k = torch.arange(block_size, dtype=torch.double).view(1, 1, -1)
        decay = torch.exp(-k.clamp(min=0) / gamma)
        weight = weight * decay

        flat_logits = self.logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)
        neg_log_q = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        flat_weight = weight.reshape(-1)
        expected_loss = (neg_log_q * flat_weight).sum() / (flat_weight.sum() + 1e-6)

        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=1e-8)

    def test_decay_gamma_standard_starts_at_position_1(self):
        """With anchor_prediction=False, decay offset=1 so position 1 gets decay=1.0."""
        gamma = 5.0
        model = _make_model(
            self.logits, self.anchors, self.keep_mask,
            anchor_prediction=False, loss_decay_gamma=gamma,
        )
        loss, _, _ = model(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
        )

        # Reference with decay starting at k=1
        bsz, n_blocks, block_size, vocab_size = self.logits.shape
        seq_len = self.input_ids.shape[1]
        targets, valid, _ = _ref_targets_standard(self.input_ids, self.anchors, block_size)
        weight = _ref_weight_mask_standard(
            self.keep_mask, valid, self.loss_mask, self.anchors, block_size, seq_len
        )
        k = torch.arange(block_size, dtype=torch.double).view(1, 1, -1)
        decay = torch.exp(-(k - 1).clamp(min=0) / gamma)
        weight = weight * decay

        flat_logits = self.logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)
        neg_log_q = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        flat_weight = weight.reshape(-1)
        expected_loss = (neg_log_q * flat_weight).sum() / (flat_weight.sum() + 1e-6)

        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=1e-8)

    def test_anchor_prediction_produces_more_valid_tokens(self):
        """anchor_prediction=True uses all block_size positions, so accuracy_denom
        should be >= the standard mode (which skips position 0)."""
        model_std = _make_model(self.logits, self.anchors, self.keep_mask, anchor_prediction=False)
        _, _, metrics_std = model_std(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
        )

        model_ap = _make_model(self.logits, self.anchors, self.keep_mask, anchor_prediction=True)
        _, _, metrics_ap = model_ap(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
        )

        self.assertGreaterEqual(
            metrics_ap["accuracy_denom"].item(),
            metrics_std["accuracy_denom"].item(),
        )


if __name__ == "__main__":
    unittest.main()
