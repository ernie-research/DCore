"""Harness abstraction + lazy registry.

Harness implementations are imported lazily so that, e.g., running only the
SGLang harness does not import the HF-Direct model-loading stack (and vice
versa).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from specbench.config import BenchConfig
from specbench.methods import MethodSpec
from specbench.metrics import BenchResult

_HARNESS_NAMES = ("sglang", "hf_direct")


class Harness(ABC):
    #: short identifier, also stored in each BenchResult.harness
    name: str

    @abstractmethod
    def run(self, method: MethodSpec, cfg: BenchConfig,
            done: Optional[set[tuple[str, Optional[int]]]] = None) -> list[BenchResult]:
        """Run ``method`` under ``cfg`` and return canonical results.

        ``done`` is a set of ``(dataset, concurrency)`` already present in the
        results file for this method+harness; those are skipped (resume).
        """
        raise NotImplementedError


def list_harness_names() -> tuple[str, ...]:
    return _HARNESS_NAMES


def get_harness(name: str) -> Harness:
    key = name.strip().lower()
    if key == "sglang":
        from specbench.harness.sglang import SGLangHarness
        return SGLangHarness()
    if key in ("hf_direct", "hf-direct", "hf"):
        from specbench.harness.hf_direct import HFDirectHarness
        return HFDirectHarness()
    raise KeyError(f"Unknown harness '{name}'. Available: {', '.join(_HARNESS_NAMES)}")
