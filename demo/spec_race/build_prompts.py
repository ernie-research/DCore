"""Regenerate demo/spec_race/prompts.json from the cached SpecBench datasets.

Takes the first N questions of each of the 10 SpecBench datasets (loaded via
``specbench.datasets.load_and_process_dataset``, backed by ``cache/dataset/``)
and appends the DIY questions from ``diy_prompts.json`` as the last benchmark.

    python -m demo.spec_race.build_prompts [--per-benchmark N] [--out PATH]

Run from the repo root; needs the ``datasets`` package (and its cache).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from specbench.datasets import load_and_process_dataset

HERE = Path(__file__).resolve().parent
DIY_FILE = HERE / "diy_prompts.json"
OUT_FILE = HERE / "prompts.json"

# Order and display names mirror specbench/report.py::_DATASET_ORDER/_DATASET_COL_NAME.
BENCHMARKS = [
    ("gsm8k", "GSM8K"),
    ("math500", "MATH-500"),
    ("aime25", "AIME25"),
    ("humaneval", "HumanEval"),
    ("mbpp", "MBPP"),
    ("livecodebench", "LCB"),
    ("swe-bench", "SWE-Bench"),
    ("mt-bench", "MT-Bench"),
    ("alpaca", "Alpaca"),
    ("arena-hard", "Arena-Hard"),
]

TITLE_LIMIT = 56


def _text(obj) -> str:
    """Flatten one ``turns`` entry (str / dict / list) into prompt text."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return _text(obj.get("content") or obj.get("text") or str(obj))
    if isinstance(obj, (list, tuple)):
        return "\n".join(t for t in (_text(x) for x in obj) if t)
    return str(obj)


def make_title(prompt: str) -> str:
    """A short label identifying the question: prefer the specific part over
    any shared boilerplate (LiveCodeBench's system prompt, HumanEval's
    'Write a solution...' prefix)."""
    if "### Question:" in prompt:
        one = " ".join(prompt.split("### Question:", 1)[1].split())
    else:
        for line in prompt.splitlines():
            if line.strip().startswith("def "):
                one = " ".join(line.split())
                break
        else:
            one = " ".join(prompt.split())
    return one if len(one) <= TITLE_LIMIT else one[: TITLE_LIMIT - 1] + "…"


def load_diy() -> list[dict]:
    items = json.loads(DIY_FILE.read_text(encoding="utf-8"))
    return [{"title": it["title"], "prompt": it["prompt"]} for it in items]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-benchmark", type=int, default=3, help="questions per benchmark")
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args()

    out = []
    for name, label in BENCHMARKS:
        ds = load_and_process_dataset(name)
        questions = []
        for row in ds:
            if len(questions) >= args.per_benchmark:
                break
            prompt = _text(row["turns"]).strip()
            if prompt:
                questions.append({"title": make_title(prompt), "prompt": prompt})
        out.append({"benchmark": label, "questions": questions})
        print(f"{label:10s} <- {name:14s} {len(questions)} questions")

    out.append({"benchmark": "DIY", "questions": load_diy()})
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n = sum(len(b["questions"]) for b in out)
    print(f"wrote {args.out}: {len(out)} benchmarks, {n} questions")


if __name__ == "__main__":
    main()