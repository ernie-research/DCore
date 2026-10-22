"""Evaluation harnesses (evaluation protocols).

Each harness knows how to run a :class:`~specbench.methods.MethodSpec` under a
:class:`~specbench.config.BenchConfig` and return canonical
:class:`~specbench.metrics.BenchResult` records.
"""

from specbench.harness.base import Harness, get_harness, list_harness_names

__all__ = ["Harness", "get_harness", "list_harness_names"]
