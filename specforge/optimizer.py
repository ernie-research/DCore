import logging

import torch
import torch.distributed as dist

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)


class BF16Optimizer:
    """AdamW over fp32 master copies of the bf16 trainable params, with grad
    clipping and cosine warmup scheduling."""

    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
    ):
        # defaults copied from EAGLE traineagle3 ds_config.json
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.optimizer = torch.optim.AdamW(
            self.fp32_params, lr=lr, weight_decay=weight_decay
        )
        self.last_grad_norm = None
        self._grad_norm_process_group = None
        self._reduce_grad_norm_across_ranks = True
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    def configure_grad_norm_reduction(
        self, *, process_group=None, enabled: bool = True
    ) -> None:
        """Configure the group that owns disjoint gradient shards.

        FSDP backends disable the reduction for replicated/NO_SHARD parameters.
        """
        self._grad_norm_process_group = process_group
        self._reduce_grad_norm_across_ranks = enabled

    def _clip_grad_norm(self):
        """Clip all FSDP shards with one global L2-norm coefficient."""
        grads = [mp.grad for mp in self.fp32_params if mp.grad is not None]
        if grads:
            total_norm_sq = torch.stack([grad.square().sum() for grad in grads]).sum()
        else:
            device = self.fp32_params[0].device if self.fp32_params else "cpu"
            total_norm_sq = torch.zeros((), dtype=torch.float32, device=device)

        if (
            self._reduce_grad_norm_across_ranks
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.all_reduce(
                total_norm_sq,
                op=dist.ReduceOp.SUM,
                group=self._grad_norm_process_group,
            )

        total_norm = total_norm_sq.sqrt()
        clip_coef = torch.clamp(self.max_grad_norm / (total_norm + 1e-6), max=1.0)
        for grad in grads:
            grad.mul_(clip_coef)
        return total_norm

    def step(self):
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )
        grad_norm = self._clip_grad_norm()
        self.last_grad_norm = grad_norm.detach()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None
        return self.last_grad_norm

    def _moment_shape_mismatch(self, opt_state):
        """Describe the first shape mismatch between the saved Adam moments and
        this rank's params, or return None when they line up.

        torch's ``Optimizer.load_state_dict`` maps saved param ids positionally
        and never validates shapes, so a mismatch stays invisible until a
        cryptic ``_foreach_*`` size error inside ``step()``. Moments are
        rank-local FSDP shards, so loading another rank's moments is exactly the
        way to hit that.
        """
        saved_ids = [
            pid for group in opt_state["param_groups"] for pid in group["params"]
        ]
        if len(saved_ids) != len(self.fp32_params):
            return (
                f"{len(saved_ids)} saved params vs "
                f"{len(self.fp32_params)} local params"
            )
        id_to_param = dict(zip(saved_ids, self.fp32_params))
        for pid, moments in opt_state["state"].items():
            param = id_to_param.get(pid)
            if param is None:
                return f"saved state references unknown param id {pid}"
            for name, value in moments.items():
                if (
                    torch.is_tensor(value)
                    and value.dim() > 0
                    and value.shape != param.shape
                ):
                    return (
                        f"param {pid} {name} {tuple(value.shape)} vs local shard "
                        f"{tuple(param.shape)}"
                    )
        return None

    def _fp32_master_mismatch(self, saved_fp32):
        """Describe why the saved fp32 masters are unusable on this rank, or
        return None when they line up.

        Masters are rank-local FSDP shards, so a checkpoint that persisted only
        rank 0's list carries the wrong shapes for every other rank.
        """
        if len(saved_fp32) != len(self.fp32_params):
            return (
                f"{len(saved_fp32)} saved masters vs "
                f"{len(self.fp32_params)} local masters"
            )
        for i, (saved, mp) in enumerate(zip(saved_fp32, self.fp32_params)):
            if saved.shape != mp.shape:
                return (
                    f"master {i} {tuple(saved.shape)} vs local shard "
                    f"{tuple(mp.shape)}"
                )
        return None

    def _any_rank(self, flag: bool) -> bool:
        """True when `flag` holds on any rank, so all ranks take the same branch."""
        if not (dist.is_available() and dist.is_initialized()):
            return flag
        device = self.fp32_params[0].device if self.fp32_params else "cpu"
        tensor = torch.tensor([int(flag)], dtype=torch.int32, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(tensor.item())

    def load_state_dict(self, state_dict, fp32_state=None):
        """Restore optimizer/scheduler state and, when present, the rank-local
        fp32 master params; without them the masters are re-cloned from the
        bf16 weights and the resume is not numerically faithful.

        Args:
            state_dict: The training_state.pt dict (scheduler, position, and for
                        older checkpoints the rank-0 optimizer state).
            fp32_state: Optional per-rank state from optimizer_rankN.pt. When it
                        carries fp32_params / optimizer_state_dict, those win
                        over the ones in state_dict.
        """
        # Adam moments are rank-local FSDP shards, so prefer the per-rank file.
        opt_state = None
        if fp32_state is not None:
            opt_state = fp32_state.get("optimizer_state_dict")
        if opt_state is None:
            opt_state = state_dict.get("optimizer_state_dict")

        if opt_state is None:
            reason = "checkpoint carries no optimizer_state_dict"
        else:
            reason = self._moment_shape_mismatch(opt_state)
        if self._any_rank(reason is not None):
            logger.warning(
                "skipping Adam moment restore (%s); moments restart from zero "
                "while weights and lr schedule resume exactly",
                reason or "mismatch on another rank",
            )
        else:
            self.optimizer.load_state_dict(opt_state)
            print_on_rank0("Successfully loaded optimizer state_dict.")
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")

        # Determine fp32 params source: prefer per-rank file, fallback to state_dict
        saved_fp32 = None
        if fp32_state is not None:
            saved_fp32 = fp32_state.get("fp32_params")
        if saved_fp32 is None:
            saved_fp32 = state_dict.get("fp32_params")

        if saved_fp32 is None:
            reason = "checkpoint carries no fp32_params"
        else:
            reason = self._fp32_master_mismatch(saved_fp32)
        if self._any_rank(reason is not None):
            logger.warning(
                "re-cloning fp32 master params from bf16 weights (%s); weights "
                "and lr schedule resume exactly, but the masters lose the "
                "precision they had accumulated beyond bf16",
                reason or "mismatch on another rank",
            )
            with torch.no_grad():
                for p, mp in zip(self.model_params, self.fp32_params):
                    mp.data.copy_(p.detach().to(torch.float32))
        else:
            with torch.no_grad():
                for saved, mp in zip(saved_fp32, self.fp32_params):
                    mp.data.copy_(saved.to(mp.device, mp.dtype))
            print_on_rank0("Successfully loaded fp32 master params.")

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            # rank-local fp32 masters; without them a resume re-quantizes from bf16
            "fp32_params": [t.detach().cpu() for t in self.fp32_params],
        }

    def shared_state_dict(self):
        """Rank-invariant state only; pair it with `rank_state_dict` per rank."""
        return {
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def rank_state_dict(self):
        """Rank-local Adam moments and fp32 masters (both are FSDP shards)."""
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "fp32_params": [t.detach().cpu() for t in self.fp32_params],
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
