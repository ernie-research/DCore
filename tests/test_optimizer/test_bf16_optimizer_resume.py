"""Resume guards for BF16Optimizer.

Adam moments and fp32 masters are rank-local FSDP shards. Loading moments that
belong to another rank used to succeed silently and then blow up inside
`optimizer.step()` with a `_foreach_lerp_` size error, so the loader now refuses
mismatched moments instead.
"""

import unittest

import torch
import torch.distributed as dist

from specforge.optimizer import BF16Optimizer


def _make_optimizer(numel, seed=0, total_steps=1000):
    torch.manual_seed(seed)
    model = torch.nn.Linear(numel, 1, bias=False)
    optimizer = BF16Optimizer(model, lr=1e-3, total_steps=total_steps)
    return model, optimizer


def _take_step(model, optimizer):
    for param in model.parameters():
        param.grad = torch.full_like(param, 0.1)
    optimizer.step()


class TestResumeMoments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # BF16Optimizer.load_state_dict logs via rank0, which needs a group.
        cls._owns_group = not dist.is_initialized()
        if cls._owns_group:
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://127.0.0.1:29555",
                rank=0,
                world_size=1,
            )

    @classmethod
    def tearDownClass(cls):
        if cls._owns_group:
            dist.destroy_process_group()

    def test_matching_shards_restore_moments_and_lr(self):
        model, optimizer = _make_optimizer(8)
        _take_step(model, optimizer)
        shared = optimizer.shared_state_dict()
        rank_state = optimizer.rank_state_dict()
        expected = optimizer.optimizer.state[optimizer.fp32_params[0]][
            "exp_avg"
        ].clone()
        expected_lr = optimizer.get_learning_rate()

        _, fresh = _make_optimizer(8, seed=1)
        fresh.load_state_dict(shared, fp32_state=rank_state)

        restored = fresh.optimizer.state[fresh.fp32_params[0]]["exp_avg"]
        torch.testing.assert_close(restored, expected)
        torch.testing.assert_close(
            fresh.fp32_params[0], optimizer.fp32_params[0], rtol=0, atol=0
        )
        self.assertAlmostEqual(fresh.get_learning_rate(), expected_lr)

    def test_mismatched_shards_skip_moments_but_keep_schedule(self):
        """A rank-0-shaped moment must not be loaded into another rank's shard."""
        model, optimizer = _make_optimizer(8)
        _take_step(model, optimizer)
        shared = optimizer.shared_state_dict()
        foreign = optimizer.rank_state_dict()
        expected_lr = optimizer.get_learning_rate()

        other_model, other = _make_optimizer(16, seed=1)
        other.load_state_dict(
            shared, fp32_state={"optimizer_state_dict": foreign["optimizer_state_dict"]}
        )

        self.assertEqual(len(other.optimizer.state), 0)
        self.assertAlmostEqual(other.get_learning_rate(), expected_lr)
        _take_step(other_model, other)  # would raise if moments had been loaded

    def test_legacy_shared_moments_are_ignored_when_shapes_differ(self):
        """Older checkpoints put rank 0's moments in training_state.pt."""
        model, optimizer = _make_optimizer(8)
        _take_step(model, optimizer)
        legacy = {
            **optimizer.shared_state_dict(),
            "optimizer_state_dict": optimizer.optimizer.state_dict(),
        }

        other_model, other = _make_optimizer(16, seed=1)
        other.load_state_dict(legacy)

        self.assertEqual(len(other.optimizer.state), 0)
        _take_step(other_model, other)

    def test_legacy_foreign_masters_recloned_instead_of_raising(self):
        """Older checkpoints saved only rank 0's fp32 masters, whose shapes are
        wrong (often `(0,)`) on every other rank; the resume must degrade to
        re-cloning from bf16 rather than aborting the run."""
        model, optimizer = _make_optimizer(8)
        _take_step(model, optimizer)
        legacy = {**optimizer.shared_state_dict(), **optimizer.state_dict()}
        expected_lr = optimizer.get_learning_rate()

        other_model, other = _make_optimizer(16, seed=1)
        other.load_state_dict(legacy)

        for param, master in zip(other.model_params, other.fp32_params):
            torch.testing.assert_close(
                master, param.detach().to(torch.float32), rtol=0, atol=0
            )
        self.assertAlmostEqual(other.get_learning_rate(), expected_lr)
        _take_step(other_model, other)


if __name__ == "__main__":
    unittest.main()
