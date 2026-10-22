"""Shared, standardized benchmark configuration.

Centralizing these knobs is what makes the comparison *fair*: every method
within a harness runs with identical sampling, warmup, and server settings.
Only the per-method speculative wiring (algorithm / draft path / spec params)
comes from :mod:`specbench.methods`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Repo root = parent of the ``specbench`` package directory.
REPO_ROOT = Path(__file__).resolve().parents[1]


def seed_for_repeat(repeat: int) -> int:
    """RNG seed for the 1-based ``repeat``: 42, then 0, 1, 2, ...

    The seed is pinned per repeat rather than per server launch so that repeat
    ``r`` replays identically whether or not repeat ``r-1`` ran in the same
    process — which is what makes a resumed sweep comparable to an uninterrupted
    one. The first repeat keeps the conventional 42, so a single-shot run (all of
    temperature 0, which is forced to one repeat) is seeded 42.
    """
    return 42 if repeat <= 1 else repeat - 2


@dataclass
class BenchConfig:
    # --- what to run -----------------------------------------------------
    target_model: str
    datasets: list[str]
    draft_model: Optional[str] = None
    output_dir: str = "benchmark_results/specbench"
    run_baseline: bool = True
    # Structured run directory (RUN_ID). When set, results/responses/report/
    # manifest are laid out under it and responses are split per dataset.
    run_dir: Optional[str] = None
    resume: bool = True             # skip (method,dataset,concurrency) already in results
    # Per-sample response logging (prompt + generated text + per-sample metrics).
    save_responses: bool = True
    responses_path: Optional[str] = None  # flat fallback when run_dir is unset
    results_output: Optional[str] = None   # flat results path when run_dir is unset

    # --- shared fairness knobs (apply to BOTH harnesses) -----------------
    num_samples: int = 128          # timed prompts per dataset
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1

    # --- SGLang harness knobs -------------------------------------------
    attention_backend: str = "flashinfer"
    tp_size: int = 1
    dp_size: int = 1                # independent single-GPU data-parallel replicas
    speculative_use_rejection_sampling: bool = False  # temp>0 correct sampling
    speculative_draft_temperature: Optional[float] = None
    # Optional per-run overrides of a method's registry speculative params.
    # Used e.g. to run EAGLE3 as a chain of a specific depth (block size):
    # block16 -> num_steps/num_draft_tokens=15, block8 -> 7 (topk=1).
    spec_num_steps: Optional[int] = None
    spec_eagle_topk: Optional[int] = None
    spec_num_draft_tokens: Optional[int] = None
    dtype: str = "bfloat16"
    mem_fraction_static: float = 0.8
    concurrencies: list[int] = field(default_factory=lambda: [1])  # per replica
    # Repeated timed measurements of every (dataset, concurrency) unit, all inside
    # the SAME server launch. temperature>0 rejection sampling makes a single
    # measurement noisy; repeating in-process avoids paying server startup and
    # cuda-graph capture per repeat. The radix cache is flushed between repeats so
    # a later repeat cannot serve the identical prompt set from a cached prefix,
    # and the RNG is re-seeded per repeat (see :func:`seed_for_repeat`).
    # Forced to 1 at temperature 0 regardless of what the caller asked for.
    repeats: int = 1
    max_running_requests: int = 128
    cuda_graph_max_bs: int = 64
    page_size: int = 1
    disable_cuda_graph: bool = False
    disable_radix_cache: bool = False
    batch_requests: bool = False
    warmup_requests: int = 3
    warmup_max_new_tokens: int = 1024
    skip_first_requests: int = 1
    timeout_s: int = 3600
    host: str = "127.0.0.1"
    port_base: int = 20000

    # --- HF-Direct harness knobs ----------------------------------------
    hf_attn_implementation: str = "flash_attention_2"
    confidence_threshold: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.datasets, str):
            self.datasets = [d.strip() for d in self.datasets.split(",") if d.strip()]
        self.concurrencies = [int(c) for c in self.concurrencies if int(c) >= 1]
        if not self.concurrencies:
            self.concurrencies = [1]
        self.repeats = max(int(self.repeats), 1)
        if float(self.temperature) == 0.0:
            # Greedy decoding is deterministic: repeats would replay the exact
            # same token sequence, so they can only cost wall clock. Hard rule,
            # not a default — a caller-supplied --repeats is overridden here.
            self.repeats = 1
        if not self.datasets:
            raise ValueError("BenchConfig.datasets must be non-empty")
        if self.speculative_draft_temperature is not None:
            draft_temperature = float(self.speculative_draft_temperature)
            if not math.isfinite(draft_temperature) or draft_temperature < 0:
                raise ValueError(
                    "BenchConfig.speculative_draft_temperature must be finite and >= 0"
                )
            self.speculative_draft_temperature = draft_temperature

    def resolve_output_dir(self) -> Path:
        p = Path(self.output_dir)
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def dataset_specs(self) -> list[tuple[str, int]]:
        """Parse ``datasets`` entries into (name, count) pairs.

        Each entry may be ``"name"`` (uses ``num_samples``) or ``"name:count"``
        (per-dataset override), matching the original benchmark_sglang_tasks
        ``dataset:count`` syntax. So the full 10-benchmark sweep can be given as
        ``gsm8k:1319,math500:500,aime25:30,...``.
        """
        specs: list[tuple[str, int]] = []
        for entry in self.datasets:
            if ":" in entry:
                name, count_s = entry.split(":", 1)
                specs.append((name.strip(), int(count_s.strip())))
            else:
                specs.append((entry.strip(), int(self.num_samples)))
        return specs

    # --- structured run-directory layout ---------------------------------
    def _run_dir(self) -> Optional[Path]:
        return Path(self.run_dir) if self.run_dir else None

    def results_file(self, fallback: Optional[str] = None) -> Path:
        """Aggregate metrics jsonl (small: one row per method x dataset x conc)."""
        rd = self._run_dir()
        if rd:
            return rd / "results.jsonl"
        target = fallback or self.results_output
        if target:
            return Path(target)
        return self.resolve_output_dir() / "results.jsonl"

    def responses_file(self, dataset: str, method: str = "") -> Optional[Path]:
        """Per-dataset, per-method responses jsonl (large; split by dataset/method)."""
        if not self.save_responses:
            return None
        rd = self._run_dir()
        if rd:
            return rd / "responses" / dataset / f"{method}.jsonl"
        if self.responses_path:
            return Path(self.responses_path)
        return None

    def logs_dir(self) -> Path:
        rd = self._run_dir()
        return (rd / "logs") if rd else (self.resolve_output_dir() / "logs")

    def report_file(self) -> Path:
        rd = self._run_dir()
        return (rd / "report.md") if rd else (self.resolve_output_dir() / "report.md")

    def manifest_file(self) -> Optional[Path]:
        rd = self._run_dir()
        return (rd / "manifest.json") if rd else None

    def profiles_file(self, method: str) -> Path:
        """Per-method DSpark profile JSON (covers all datasets in the combo)."""
        rd = self._run_dir()
        base = rd if rd else self.resolve_output_dir()
        return base / "profiles" / f"{method}.json"

    def apply_spec_overrides(self, base_params: dict) -> dict:
        """Merge CLI --spec-* overrides over a method's registry spec params."""
        params = dict(base_params)
        if self.spec_num_steps is not None:
            params["speculative_num_steps"] = self.spec_num_steps
        if self.spec_eagle_topk is not None:
            params["speculative_eagle_topk"] = self.spec_eagle_topk
        if self.spec_num_draft_tokens is not None:
            params["speculative_num_draft_tokens"] = self.spec_num_draft_tokens
        return params
