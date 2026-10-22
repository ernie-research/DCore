"""Tests for the D-PACE loss variants on OnlineDominoModel.

Only ``_compute_weighted_losses`` is exercised, so the tests run on CPU without
building a draft model or loading target weights.
"""

import unittest
import warnings

import torch
import torch.nn.functional as F

from specforge.core.domino import OnlineDominoModel

VOCAB = 17
BSZ, NBLOCK, BLOCK = 2, 3, 5


def _bare_model(loss_type: str, dpace_alpha: float = 0.5, loss_decay_gamma=None):
    return OnlineDominoModel(
        draft_model=None,
        target_lm_head=None,
        target_embed_tokens=None,
        mask_token_id=0,
        block_size=BLOCK,
        loss_type=loss_type,
        dpace_alpha=dpace_alpha,
        loss_decay_gamma=loss_decay_gamma,
    )


def _naive_shared_dpace(final_logits, base_logits, target_ids, m0, alpha, lam):
    """Reference: one shared weight tensor derived from the final head."""
    flat_targets = target_ids.reshape(-1)
    ce_f = F.cross_entropy(
        final_logits.reshape(-1, VOCAB), flat_targets, reduction="none"
    ).view_as(target_ids)
    ce_b = F.cross_entropy(
        base_logits.reshape(-1, VOCAB), flat_targets, reduction="none"
    ).view_as(target_ids)

    q = torch.exp(-ce_f)
    smooth = torch.where(m0 > 0, (1.0 - alpha) * q + alpha, torch.ones_like(q))
    prefix = torch.cumprod(smooth, dim=-1)
    weight = torch.flip(
        torch.cumsum(torch.flip(prefix * m0, dims=[-1]), dim=-1), dims=[-1]
    )

    m = m0 * weight
    denom = m.sum() + 1e-6
    final_loss = (ce_f * m).sum() / denom
    base_loss = (ce_b * m).sum() / denom
    return (1.0 - lam) * final_loss + lam * base_loss, final_loss, base_loss


class TestDominoDPaceLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        shape = (BSZ, NBLOCK * BLOCK, VOCAB)
        self.final_logits = torch.randn(shape)
        self.base_logits = torch.randn(shape)
        self.target_ids = torch.randint(0, VOCAB, (BSZ, NBLOCK, BLOCK))

        # Mirror forward(): position 0 is excluded and the tail of the last
        # block is out of bounds.
        m0 = torch.ones(BSZ, NBLOCK, BLOCK)
        m0[:, :, 0] = 0.0
        m0[:, -1, -2:] = 0.0
        self.m0 = m0

    def _call(self, model, lam=0.5):
        return model._compute_weighted_losses(
            final_logits=self.final_logits,
            base_logits=self.base_logits,
            target_ids=self.target_ids,
            weight_mask=self.m0.clone(),
            lambda_base=lam,
        )

    def test_dpace_matches_naive_shared_weight_reference(self):
        alpha, lam = 0.5, 0.5
        loss, final_loss, base_loss, *_ = self._call(
            _bare_model("dpace", dpace_alpha=alpha), lam=lam
        )
        exp_loss, exp_final, exp_base = _naive_shared_dpace(
            self.final_logits, self.base_logits, self.target_ids, self.m0, alpha, lam
        )
        torch.testing.assert_close(final_loss, exp_final)
        torch.testing.assert_close(base_loss, exp_base)
        torch.testing.assert_close(loss, exp_loss)

    def test_both_terms_share_the_same_weight(self):
        """base_loss must be reweighted by the final head's D-PACE weight."""
        loss_type = "dpace"
        _, _, base_loss, *_ = self._call(_bare_model(loss_type))

        # Recompute base_loss with a weight derived from base_logits instead;
        # it must differ, proving the shared-final-weight path is taken.
        swapped = _naive_shared_dpace(
            self.base_logits, self.base_logits, self.target_ids, self.m0, 0.5, 0.0
        )[1]
        self.assertFalse(torch.allclose(base_loss, swapped))

    def test_alpha_one_degenerates_to_static_valid_position_count(self):
        """alpha=1 makes the weight data-independent: W[k] = #valid j >= k."""
        _, final_loss, _, *_ = self._call(_bare_model("dpace", dpace_alpha=1.0))

        static = torch.flip(torch.cumsum(torch.flip(self.m0, dims=[-1]), dim=-1), dims=[-1])
        m = self.m0 * static
        ce = F.cross_entropy(
            self.final_logits.reshape(-1, VOCAB),
            self.target_ids.reshape(-1),
            reduction="none",
        ).view_as(self.target_ids)
        expected = (ce * m).sum() / (m.sum() + 1e-6)
        torch.testing.assert_close(final_loss, expected)

    def test_ce_path_is_unchanged(self):
        _, final_loss, base_loss, *_ = self._call(_bare_model("ce"))
        flat_targets = self.target_ids.reshape(-1)
        flat_weights = self.m0.reshape(-1)
        denom = flat_weights.sum() + 1e-6
        for logits, got in (
            (self.final_logits, final_loss),
            (self.base_logits, base_loss),
        ):
            ce = F.cross_entropy(
                logits.reshape(-1, VOCAB), flat_targets, reduction="none"
            )
            torch.testing.assert_close(got, (ce * flat_weights).sum() / denom)

    def test_decay_gamma_is_disabled_and_warned_for_dpace(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = _bare_model("dpace", loss_decay_gamma=7.0)
        self.assertIsNone(model.loss_decay_gamma)
        self.assertTrue(any("loss_decay_gamma" in str(w.message) for w in caught))

    def test_decay_gamma_is_kept_for_ce(self):
        model = _bare_model("ce", loss_decay_gamma=7.0)
        self.assertEqual(model.loss_decay_gamma, 7.0)

    def test_invalid_dpace_alpha_rejected(self):
        with self.assertRaisesRegex(ValueError, "dpace_alpha"):
            _bare_model("dpace", dpace_alpha=1.5)


if __name__ == "__main__":
    unittest.main()
