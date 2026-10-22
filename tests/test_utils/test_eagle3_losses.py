import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.train_eagle3 import accumulate_metric_sum, build_parser
from specforge.core.eagle3 import (
    OnlineEagle3Model,
    QwenVLOnlineEagle3Model,
    _aggregate_dpace_loss,
    _compute_loss_and_acceptance_rate,
    _distill_step_loss,
    _finalize_step_losses,
    _hard_ce_step_loss,
    resolve_eagle_loss_spec,
    ttt_step_loss_weights,
)
from specforge.core.loss import LogSoftmaxLoss


class TestEagle3LossConfig(unittest.TestCase):
    BASE_ARGS = [
        "--target-model-path",
        "target",
        "--train-data-path",
        "train",
        "--output-dir",
        "output",
    ]

    def test_cli_defaults_and_choices(self):
        parser = build_parser()
        args = parser.parse_args(self.BASE_ARGS)
        self.assertEqual(args.loss_type, "kl")
        self.assertEqual(args.dpace_alpha, 0.5)
        self.assertFalse(hasattr(args, "lk_loss_type"))

        args = parser.parse_args(
            self.BASE_ARGS
            + ["--loss-type", "dpace", "--dpace-alpha", "0.25"]
        )
        self.assertEqual(args.loss_type, "dpace")
        self.assertEqual(args.dpace_alpha, 0.25)

    def test_cli_rejects_invalid_values(self):
        parser = build_parser()
        for extra_args in (
            ["--loss-type", "alpha"],
            ["--dpace-alpha", "-0.1"],
            ["--dpace-alpha", "1.1"],
            ["--position-weighting", "exp"],
        ):
            with self.subTest(extra_args=extra_args), self.assertRaises(SystemExit):
                parser.parse_args(self.BASE_ARGS + extra_args)

    def test_cli_exposes_the_full_objective_x_weighting_matrix(self):
        parser = build_parser()
        args = parser.parse_args(self.BASE_ARGS)
        self.assertIsNone(args.position_weighting)
        self.assertIsNone(args.loss_decay_gamma)
        self.assertEqual(args.ttt_length, 7)
        self.assertEqual(
            (args.kltv_kl_alpha, args.kltv_l1_alpha), (0.1, 0.9)
        )
        self.assertEqual((args.cetv_ce_alpha, args.cetv_l1_alpha), (0.1, 0.9))

        for objective in ("ce", "kl", "cetv", "kltv"):
            for weighting in ("decay", "dpace"):
                with self.subTest(objective=objective, weighting=weighting):
                    args = parser.parse_args(
                        self.BASE_ARGS
                        + [
                            "--loss-type",
                            objective,
                            "--position-weighting",
                            weighting,
                            "--loss-decay-gamma",
                            "7.0",
                        ]
                    )
                    self.assertEqual(
                        resolve_eagle_loss_spec(
                            args.loss_type, args.position_weighting
                        ),
                        (objective, weighting),
                    )
                    self.assertEqual(args.loss_decay_gamma, 7.0)

    def test_legacy_dpace_alias_still_means_hard_ce(self):
        self.assertEqual(resolve_eagle_loss_spec("dpace"), ("ce", "dpace"))
        # an explicit weighting overrides the alias default
        self.assertEqual(resolve_eagle_loss_spec("dpace", "decay"), ("ce", "decay"))
        with self.assertRaisesRegex(ValueError, "position_weighting"):
            resolve_eagle_loss_spec("kl", "exp")
        with self.assertRaisesRegex(ValueError, "loss_type"):
            resolve_eagle_loss_spec("dflash")

    def test_normal_and_vlm_constructors_validate_early(self):
        draft_model = nn.Linear(2, 2)
        model = OnlineEagle3Model(
            draft_model=draft_model, loss_type="ce", dpace_alpha=0.0
        )
        self.assertEqual(model.loss_type, "ce")
        self.assertEqual(model.dpace_alpha, 0.0)
        self.assertEqual((model.objective, model.position_weighting), ("ce", "decay"))

        off_diagonal = OnlineEagle3Model(
            draft_model=draft_model, loss_type="kltv", position_weighting="dpace"
        )
        self.assertEqual(
            (off_diagonal.objective, off_diagonal.position_weighting),
            ("kltv", "dpace"),
        )

        vlm_model = QwenVLOnlineEagle3Model(
            target_model=nn.Identity(),
            draft_model=draft_model,
            processor=object(),
            loss_type="dpace",
            dpace_alpha=1.0,
        )
        self.assertEqual(vlm_model.loss_type, "dpace")
        self.assertEqual(vlm_model.dpace_alpha, 1.0)
        self.assertEqual(
            (vlm_model.objective, vlm_model.position_weighting), ("ce", "dpace")
        )

        with self.assertRaisesRegex(ValueError, "loss_type"):
            OnlineEagle3Model(draft_model=draft_model, loss_type="alpha")
        with self.assertRaisesRegex(ValueError, "dpace_alpha"):
            QwenVLOnlineEagle3Model(
                target_model=nn.Identity(),
                draft_model=draft_model,
                processor=object(),
                dpace_alpha=1.01,
            )

        logits = torch.randn(1, 1, 2)
        target_p = torch.softmax(torch.randn(1, 1, 2), dim=-1)
        position_mask = torch.ones(1, 1, 1)
        with self.assertRaisesRegex(ValueError, "loss_type"):
            _compute_loss_and_acceptance_rate(
                logits=logits,
                target_p=target_p,
                target_p_on_draft=target_p,
                position_mask=position_mask,
                loss_type="lambda",
            )


class TestEagle3HardCE(unittest.TestCase):
    def test_hard_labels_masking_and_gradients(self):
        logits = torch.tensor(
            [
                [
                    [1.2, -0.3, 0.4],
                    [-0.2, 0.7, 1.1],
                    [0.3, -0.8, 1.4],
                ]
            ],
            requires_grad=True,
        )
        target_p = torch.tensor(
            [
                [
                    [0.1, 0.8, 0.1],
                    [0.7, 0.2, 0.1],
                    [0.1, 0.2, 0.7],
                ]
            ]
        )
        position_mask = torch.tensor([[[1], [0], [1]]])

        loss, (per_token_ce, q, mask), valid_count = _hard_ce_step_loss(
            logits, target_p, position_mask
        )
        labels = torch.tensor([[1, 0, 2]])
        expected_per_token = F.cross_entropy(
            logits.detach().view(-1, 3), labels.view(-1), reduction="none"
        ).view(1, 3)
        expected = (expected_per_token * mask).sum() / 2

        torch.testing.assert_close(per_token_ce.detach(), expected_per_token)
        torch.testing.assert_close(loss.detach(), expected)
        torch.testing.assert_close(q, torch.exp(-expected_per_token))
        self.assertFalse(q.requires_grad)
        self.assertEqual(valid_count.item(), 2)

        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[:, 1], torch.zeros_like(logits.grad[:, 1]))
        self.assertGreater(logits.grad[:, 0].abs().sum().item(), 0)
        self.assertGreater(logits.grad[:, 2].abs().sum().item(), 0)

    def test_all_masked_step_is_zero_with_zero_gradient(self):
        logits = torch.randn(2, 3, 5, requires_grad=True)
        target_p = torch.softmax(torch.randn(2, 3, 5), dim=-1)
        position_mask = torch.zeros(2, 3, 1)

        loss, _, valid_count = _hard_ce_step_loss(logits, target_p, position_mask)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(valid_count.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


class TestEagle3Dpace(unittest.TestCase):
    @staticmethod
    def _manual_weights(q, mask, alpha):
        smooth = (1 - alpha) * q + alpha
        smooth = torch.where(mask.bool(), smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)
        return torch.flip(
            torch.cumsum(torch.flip(prefix * mask, dims=[-1]), dim=-1),
            dims=[-1],
        )

    def test_weights_formula_detachment_and_singleton_aggregation(self):
        ce0 = torch.tensor([[0.2, 0.7]], requires_grad=True)
        ce1 = torch.tensor([[0.4, 1.1]], requires_grad=True)
        mask0 = torch.tensor([[1.0, 1.0]])
        mask1 = torch.tensor([[1.0, 0.0]])
        q0 = torch.exp(-ce0.detach())
        q1 = torch.exp(-ce1.detach())
        steps = [(ce0, q0, mask0), (ce1, q1, mask1)]
        alpha = 0.3

        with self.assertRaisesRegex(ValueError, "dpace_alpha"):
            _aggregate_dpace_loss(steps, 1.1)
        loss, weight_sum = _aggregate_dpace_loss(steps, alpha)
        ce = torch.stack([ce0, ce1], dim=-1)
        q = torch.stack([q0, q1], dim=-1)
        mask = torch.stack([mask0, mask1], dim=-1)
        weights = self._manual_weights(q, mask, alpha) * mask
        expected = (ce * weights).sum() / weights.sum()

        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(weight_sum, weights.sum())

        losses, metric_losses, metric_denoms = _finalize_step_losses(
            loss_type="dpace",
            dpace_alpha=alpha,
            losses=[ce0.mean(), ce1.mean()],
            metric_losses=[ce0.mean().detach(), ce1.mean().detach()],
            metric_loss_denoms=[mask0.sum(), mask1.sum()],
            dpace_step_losses=steps,
        )
        self.assertEqual(len(losses), 1)
        self.assertEqual(len(metric_losses), 1)
        self.assertEqual(len(metric_denoms), 1)
        torch.testing.assert_close(losses[0], expected)

        loss.backward()
        expected_grad = weights / weights.sum()
        torch.testing.assert_close(ce0.grad, expected_grad[..., 0])
        torch.testing.assert_close(ce1.grad, expected_grad[..., 1])

    def test_masked_rollout_positions_are_finite(self):
        ce0 = torch.tensor([[0.3, 0.5]], requires_grad=True)
        ce1 = torch.tensor([[0.8, 0.9]], requires_grad=True)
        ce2 = torch.tensor([[0.4, 0.2]], requires_grad=True)
        masks = [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
        ]
        steps = [
            (ce, torch.exp(-ce.detach()), mask)
            for ce, mask in zip((ce0, ce1, ce2), masks)
        ]

        loss, weight_sum = _aggregate_dpace_loss(steps, dpace_alpha=0.5)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(weight_sum.item(), 0)
        loss.backward()
        torch.testing.assert_close(ce1.grad, torch.zeros_like(ce1.grad))

    def test_all_masked_rollouts_return_zero(self):
        ce = torch.tensor([[0.3, 0.5]], requires_grad=True)
        mask = torch.zeros_like(ce)
        loss, weight_sum = _aggregate_dpace_loss(
            [(ce, torch.exp(-ce.detach()), mask)], dpace_alpha=0.5
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(weight_sum.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(ce.grad, torch.zeros_like(ce.grad))


class TestTttStepLossWeights(unittest.TestCase):
    def test_default_reproduces_eagle3_and_gamma_preserves_total_weight(self):
        self.assertEqual(ttt_step_loss_weights(7), [0.8**i for i in range(7)])

        # gamma = 1/ln(1.25) is 0.8^i written in exponential form
        equivalent = ttt_step_loss_weights(7, 1.0 / math.log(1.25))
        for got, want in zip(equivalent, [0.8**i for i in range(7)]):
            self.assertAlmostEqual(got, want, places=12)

        exp7 = ttt_step_loss_weights(7, 7.0)
        # only the decay shape changes: the total weight (and thus the effective
        # learning rate) matches 0.8^i, and the ratio is exp(1/gamma)
        self.assertAlmostEqual(sum(exp7), sum(ttt_step_loss_weights(7)), places=12)
        self.assertAlmostEqual(exp7[0] / exp7[1], math.exp(1 / 7.0), places=12)
        self.assertGreater(exp7[-1], 0.8**6)

        # dpace fuses the rollout into a single loss
        self.assertEqual(ttt_step_loss_weights(1, 7.0), [1.0])
        with self.assertRaisesRegex(ValueError, "loss_decay_gamma"):
            ttt_step_loss_weights(7, 0.0)


class TestEagle3DistillObjectives(unittest.TestCase):
    """``cetv`` / ``kltv`` / per-token ``kl``: the rows that make the
    objective x weighting matrix square for EAGLE-3."""

    def setUp(self):
        torch.manual_seed(11)
        self.bsz, self.seq_len, self.vocab = 2, 5, 9
        self.logits = torch.randn(
            self.bsz, self.seq_len, self.vocab, dtype=torch.float32
        )
        self.target_p = torch.softmax(
            torch.randn(self.bsz, self.seq_len, self.vocab), dim=-1
        )
        self.position_mask = torch.ones(self.bsz, self.seq_len, 1, dtype=torch.int32)
        self.position_mask[0, -2:] = 0

    def _references(self, logits):
        log_q = F.log_softmax(logits.float(), dim=-1)
        p = self.target_p
        l1 = (p - log_q.exp()).abs().sum(-1)
        soft_ce = -(p * log_q).sum(-1)
        kl = torch.xlogy(p, p).sum(-1) + soft_ce
        labels = p.argmax(-1)
        ce = F.cross_entropy(
            logits.float().reshape(-1, self.vocab), labels.reshape(-1), reduction="none"
        ).view_as(labels)
        return l1, soft_ce, kl, ce

    def _step(self, objective, **alphas):
        kwargs = dict(
            kltv_kl_alpha=0.1,
            kltv_l1_alpha=0.9,
            cetv_ce_alpha=0.1,
            cetv_l1_alpha=0.9,
        )
        kwargs.update(alphas)
        return _distill_step_loss(
            self.logits,
            self.target_p,
            self.position_mask,
            objective=objective,
            **kwargs,
        )

    def test_objectives_match_their_closed_forms_under_the_mask(self):
        mask = self.position_mask.squeeze(-1).float()
        l1, _soft_ce, kl, ce = self._references(self.logits)

        for objective, expected_per_token in (
            ("kl", kl),
            ("kltv", 0.1 * kl + 0.9 * l1),
            ("cetv", 0.9 * l1 + 0.1 * ce),
        ):
            with self.subTest(objective=objective):
                loss, (per_token, _q, got_mask), denom = self._step(objective)
                torch.testing.assert_close(got_mask, mask)
                torch.testing.assert_close(denom, mask.sum())
                # masked-out positions are left at 0 rather than computed
                torch.testing.assert_close(
                    per_token * mask, expected_per_token * mask, atol=1e-5, rtol=1e-5
                )
                torch.testing.assert_close(
                    loss,
                    (expected_per_token * mask).sum() / mask.sum(),
                    atol=1e-5,
                    rtol=1e-5,
                )

    def test_acceptance_proxy_follows_the_objective_label_type(self):
        _l1, soft_ce, _kl, ce = self._references(self.logits)
        mask = self.position_mask.squeeze(-1).float()

        # soft-label objectives use the Jensen bound exp(-C_i) on <p, q>
        for objective in ("kl", "kltv"):
            with self.subTest(objective=objective):
                _loss, (_per_token, q, _m), _denom = self._step(objective)
                torch.testing.assert_close(
                    q * mask, torch.exp(-soft_ce) * mask, atol=1e-5, rtol=1e-5
                )
                self.assertFalse(q.requires_grad)

        # cetv is trained against the argmax token, so it uses exp(-CE(z*))
        _loss, (_per_token, q, _m), _denom = self._step("cetv")
        torch.testing.assert_close(q, torch.exp(-ce), atol=1e-5, rtol=1e-5)
        self.assertFalse(q.requires_grad)

    def test_alphas_select_the_mix_and_gradients_reach_all_valid_positions(self):
        mask = self.position_mask.squeeze(-1).float()
        l1, _soft_ce, kl, ce = self._references(self.logits)

        # kltv with l1_alpha=0 is pure KL; cetv with l1_alpha=0 is pure scaled CE
        loss, _terms, _denom = self._step("kltv", kltv_kl_alpha=1.0, kltv_l1_alpha=0.0)
        torch.testing.assert_close(
            loss, (kl * mask).sum() / mask.sum(), atol=1e-5, rtol=1e-5
        )
        loss, _terms, _denom = self._step("cetv", cetv_ce_alpha=1.0, cetv_l1_alpha=0.0)
        torch.testing.assert_close(
            loss, (ce * mask).sum() / mask.sum(), atol=1e-5, rtol=1e-5
        )

        for objective in ("kl", "kltv", "cetv"):
            with self.subTest(objective=objective):
                self.logits = self.logits.detach().requires_grad_(True)
                self._step(objective)[0].backward()
                grad = self.logits.grad
                self.assertTrue(torch.isfinite(grad).all())
                # every valid position contributes, masked-out ones do not
                per_position = grad.abs().sum(-1)
                self.assertTrue((per_position[mask > 0] > 0).all())
                torch.testing.assert_close(
                    per_position[mask == 0], torch.zeros_like(per_position[mask == 0])
                )

    def test_all_masked_step_is_zero_and_backward_safe(self):
        self.logits = self.logits.detach().requires_grad_(True)
        self.position_mask = torch.zeros_like(self.position_mask)
        loss, (per_token, _q, _m), denom = self._step("kltv")
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(denom.item(), 0.0)
        torch.testing.assert_close(per_token, torch.zeros_like(per_token))
        loss.backward()

    def test_dpace_weighting_is_reachable_for_every_objective(self):
        for objective in ("ce", "kl", "cetv", "kltv"):
            with self.subTest(objective=objective):
                logits = self.logits.detach().clone().requires_grad_(True)
                _ar, _loss, step_terms, _denom = _compute_loss_and_acceptance_rate(
                    logits=logits,
                    target_p=self.target_p,
                    target_p_on_draft=self.target_p,
                    position_mask=self.position_mask,
                    loss_type=objective,
                    position_weighting="dpace",
                )
                self.assertIsNotNone(step_terms)
                fused, weight_sum = _aggregate_dpace_loss(
                    [step_terms, step_terms], dpace_alpha=0.5
                )
                self.assertGreater(weight_sum.item(), 0)
                fused.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertGreater(logits.grad.abs().sum().item(), 0)


@unittest.skipUnless(torch.cuda.is_available(), "LogSoftmaxLoss requires CUDA/Triton")
class TestEagle3KLRegression(unittest.TestCase):
    def test_kl_output_and_gradient_match_fused_loss_exactly(self):
        torch.manual_seed(7)
        target_p = torch.softmax(torch.randn(2, 4, 11, device="cuda"), dim=-1)
        target_p_on_draft = target_p * 0.9
        position_mask = torch.tensor(
            [
                [[1], [1], [0], [1]],
                [[1], [0], [1], [1]],
            ],
            device="cuda",
        )
        logits_expected = torch.randn(2, 4, 11, device="cuda", requires_grad=True)
        logits_actual = logits_expected.detach().clone().requires_grad_(True)

        expected = LogSoftmaxLoss.apply(
            logits_expected, target_p, position_mask
        )
        expected.backward()
        _, actual, dpace_step, _ = _compute_loss_and_acceptance_rate(
            logits=logits_actual,
            target_p=target_p,
            target_p_on_draft=target_p_on_draft,
            position_mask=position_mask,
            loss_type="kl",
        )
        actual.backward()

        self.assertIsNone(dpace_step)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            logits_actual.grad, logits_expected.grad, rtol=0, atol=0
        )


class TestAccumulateMetricSum(unittest.TestCase):
    """dpace reduces the rollout to one loss, so the loss lists are shorter than
    the per-position accuracy lists; accumulating them under a shared loop bound
    used to raise IndexError on the second training step."""

    def test_shorter_loss_group_accumulates_independently(self):
        acc = [torch.tensor(float(i)) for i in range(7)]
        loss = [torch.tensor(2.0)]

        acc_sums = accumulate_metric_sum(None, acc)
        loss_sums = accumulate_metric_sum(None, loss)
        acc_sums = accumulate_metric_sum(acc_sums, acc)
        loss_sums = accumulate_metric_sum(loss_sums, loss)

        self.assertEqual(len(acc_sums), 7)
        self.assertEqual(len(loss_sums), 1)
        torch.testing.assert_close(acc_sums[3], torch.tensor(6.0))
        torch.testing.assert_close(loss_sums[0], torch.tensor(4.0))

    def test_seed_does_not_alias_the_inputs(self):
        values = [torch.tensor(1.0)]
        sums = accumulate_metric_sum(None, values)
        accumulate_metric_sum(sums, values)

        torch.testing.assert_close(values[0], torch.tensor(1.0))
        torch.testing.assert_close(sums[0], torch.tensor(2.0))


if __name__ == "__main__":
    unittest.main()
