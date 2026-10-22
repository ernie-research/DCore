"""Canonical result schema shared by every harness.

Field names are deliberately harness-neutral (``accept_length``,
``tokens_per_second``) so results from the SGLang and HF-Direct harnesses land
in the same columns. See :mod:`specbench.report` for how they are aggregated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class BenchResult:
    harness: str                       # "sglang" | "hf_direct"
    method: str                        # "baseline" | "eagle3" | "dflash" | "dpace" | ...
    dataset: str

    tokens_per_second: float
    accept_length: Optional[float] = None
    baseline_tokens_per_second: Optional[float] = None
    speedup: Optional[float] = None

    num_samples: int = 0
    concurrency: Optional[int] = None  # per SGLang replica; None => serial HF-Direct
    # 1-based index of the repeated measurement of this (dataset, concurrency)
    # unit within one server launch. 1 when repeats are not used.
    repeat: int = 1

    target_model: str = ""
    draft_model: Optional[str] = None
    backend: str = ""                  # sglang attention backend / hf attn_implementation
    temperature: float = 0.0
    max_new_tokens: int = 0
    latency_s: Optional[float] = None

    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def write_results(path: str | Path, results: Iterable[BenchResult], *, append: bool = False) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out.open(mode, encoding="utf-8") as f:
        for r in results:
            f.write(r.to_json() + "\n")
    return out


def read_results(path: str | Path) -> list[BenchResult]:
    out: list[BenchResult] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            known = {k: rec.get(k) for k in BenchResult.__dataclass_fields__ if k in rec}
            out.append(BenchResult(**known))
    return out


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    """Append arbitrary dict records as jsonl (used for per-sample responses)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out
