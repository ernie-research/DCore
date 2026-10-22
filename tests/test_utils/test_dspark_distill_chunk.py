import unittest

import torch
import torch.nn as nn

from specforge.core.dspark import OnlineDSparkModel


class _DistillStub:
    """Minimal stand-in: _distill_chunk_per_token only reads these two attrs."""

    def __init__(self, lm_head: nn.Module, kl_loss_alpha: float = 1.0):
        self.lm_head = lm_head
        self.kl_loss_alpha = kl_loss_alpha


def _run_chunk(draft_logits, target_logits, kl_loss_alpha=1.0):
    vocab = draft_logits.size(-1)
    lm_head = nn.Linear(vocab, vocab, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(vocab))
    stub = _DistillStub(lm_head, kl_loss_alpha)
    position_idx = torch.arange(draft_logits.size(0))
    return OnlineDSparkModel._distill_chunk_per_token(
        stub, draft_logits, position_idx, target_logits
    )


class TestDSparkDistillChunk(unittest.TestCase):

    def test_saturated_draft_logit_does_not_produce_nan(self):
        # A -inf draft logit makes log_softmax -inf on that entry, while the
        # peaked target puts exactly 0 probability there: the unguarded
        # p * log q product used to evaluate to 0 * -inf = nan.
        draft_logits = torch.tensor(
            [[0.0, 1.0, float("-inf")], [2.0, float("-inf"), 0.5]],
            requires_grad=True,
        )
        target_logits = torch.tensor([[300.0, -300.0, -300.0], [-300.0, -300.0, 300.0]])
        target_probs = torch.softmax(target_logits, dim=-1)
        self.assertTrue((target_probs == 0).any(), "target must have exact zeros")

        l1, soft_ce, neg_entropy = _run_chunk(draft_logits, target_logits)

        self.assertTrue(torch.isfinite(l1).all())
        self.assertTrue(torch.isfinite(soft_ce).all())
        self.assertTrue(torch.isfinite(neg_entropy).all())

        # The backward is where a nan would poison every draft parameter.
        soft_ce.sum().backward()
        self.assertTrue(torch.isfinite(draft_logits.grad).all())

    def test_soft_ce_matches_masked_reference(self):
        torch.manual_seed(0)
        draft_logits = torch.randn(4, 7, requires_grad=True)
        target_logits = torch.randn(4, 7) * 8.0

        _, soft_ce, neg_entropy = _run_chunk(draft_logits, target_logits)

        t_probs = torch.softmax(target_logits.float(), dim=-1)
        log_q = torch.log_softmax(draft_logits.detach().float(), dim=-1)
        expected_ce = -(t_probs * log_q).sum(-1)
        torch.testing.assert_close(soft_ce, expected_ce)

        # soft_ce + neg_entropy is the KL the caller reduces.
        kl = soft_ce + neg_entropy
        expected_kl = torch.nn.functional.kl_div(
            log_q, t_probs, reduction="none"
        ).sum(-1)
        torch.testing.assert_close(kl, expected_kl, atol=1e-5, rtol=1e-4)
        self.assertTrue((kl >= -1e-5).all())

    def test_kl_disabled_returns_zero_terms(self):
        draft_logits = torch.randn(3, 5, requires_grad=True)
        target_logits = torch.randn(3, 5)

        l1, soft_ce, neg_entropy = _run_chunk(
            draft_logits, target_logits, kl_loss_alpha=0.0
        )

        self.assertTrue(torch.equal(soft_ce, torch.zeros(3)))
        self.assertTrue(torch.equal(neg_entropy, torch.zeros(3)))
        self.assertTrue(torch.isfinite(l1).all())


if __name__ == "__main__":
    unittest.main()
