"""Minimal torch.distributed helpers for the HF-Direct harness (torchrun).

Ported from ``third_party/Domino/code/distributed.py`` so the HF-Direct
harness has no dependency on ``third_party/Domino``. When launched without
torchrun (no ``RANK`` env), everything degrades to a single-process no-op.
"""

from __future__ import annotations

import datetime
import os
import warnings
from typing import Any, List, Optional

from torch import distributed as dist

__all__ = [
    "init", "is_initialized", "size", "rank", "local_size", "local_rank",
    "is_main", "barrier", "gather", "all_gather", "destroy",
]


def init() -> None:
    if "RANK" not in os.environ:
        warnings.warn("`RANK` not set; skipping distributed init (single process).")
        return
    dist.init_process_group(backend="nccl", init_method="env://",
                            timeout=datetime.timedelta(hours=2))


def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def rank() -> int:
    return int(os.environ.get("RANK", 0))


def local_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main() -> bool:
    return rank() == 0


def barrier() -> None:
    if is_initialized():
        dist.barrier()


def gather(obj: Any, dst: int = 0) -> Optional[List[Any]]:
    if not is_initialized():
        return [obj]
    if is_main():
        objs: list = [None for _ in range(size())]
        dist.gather_object(obj, objs, dst=dst)
        return objs
    dist.gather_object(obj, dst=dst)
    return None


def all_gather(obj: Any) -> List[Any]:
    if not is_initialized():
        return [obj]
    objs: list = [None for _ in range(size())]
    dist.all_gather_object(objs, obj)
    return objs


def destroy() -> None:
    if is_initialized():
        dist.destroy_process_group()
