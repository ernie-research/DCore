"""Load and normalize the prompts file used by the web UI and the recorder.

prompts.json is a JSON list of benchmarks, each with a name and its questions:

    [
      {"benchmark": "DIY",
       "questions": [{"title": "A short label", "prompt": "..."}, ...]},
      ...
    ]

The legacy flat shape (a list of {"title", "prompt"} dicts or plain strings) is
also accepted and treated as a single "DIY" benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_BENCHMARK = "DIY"


def load_benchmarks(path: Path) -> list[dict]:
    """Return [{"benchmark": str, "questions": [{"title", "prompt"}, ...]}, ...]."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        return []
    legacy = isinstance(data[0], str) or (isinstance(data[0], dict) and "questions" not in data[0])
    if legacy:
        return [{
            "benchmark": DEFAULT_BENCHMARK,
            "questions": [
                it if isinstance(it, dict) else {"title": f"Prompt {i + 1}", "prompt": str(it)}
                for i, it in enumerate(data)
            ],
        }]
    return [{
        "benchmark": b.get("benchmark", f"Benchmark {i + 1}"),
        "questions": list(b["questions"]),
    } for i, b in enumerate(data)]


def flatten(path: Path) -> list[str]:
    """All prompts in file order, for offline recording."""
    return [q["prompt"] for b in load_benchmarks(path) for q in b["questions"]]