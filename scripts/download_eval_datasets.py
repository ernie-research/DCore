#!/usr/bin/env python3
"""Download all 7 evaluation datasets to cache/dataset/ for offline use."""

import json
import os
import sys
from pathlib import Path

# Ensure network access
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

from datasets import Dataset, load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / "cache" / "dataset"

DATASETS = {
    "gsm8k": {
        "args": ("openai/gsm8k", "main"),
        "kwargs": {"split": "test"},
    },
    "math500": {
        "args": ("HuggingFaceH4/MATH-500",),
        "kwargs": {"split": "test"},
    },
    "aime25": {
        "args": ("MathArena/aime_2025",),
        "kwargs": {"split": "train"},
    },
    "humaneval": {
        "args": ("openai/openai_humaneval",),
        "kwargs": {"split": "test"},
    },
    "mbpp": {
        "args": ("google-research-datasets/mbpp", "sanitized"),
        "kwargs": {"split": "test"},
    },
    "mt-bench": {
        "args": ("HuggingFaceH4/mt_bench_prompts",),
        "kwargs": {"split": "train"},
    },
    "alpaca": {
        "args": ("tatsu-lab/alpaca",),
        "kwargs": {"split": "train"},
    },
    "livecodebench": {
        "args": ("json",),
        "kwargs": {
            "data_files": {
                "test": [
                    f"https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/{f}"
                    for f in ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
                ]
            },
            "split": "test",
        },
    },
}


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    failed = []

    for name, cfg in DATASETS.items():
        out_dir = CACHE_DIR / name
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[SKIP] {name} already exists at {out_dir}")
            continue

        print(f"[DOWNLOAD] {name} ...")
        try:
            ds = load_dataset(*cfg["args"], **cfg["kwargs"])
            ds.save_to_disk(str(out_dir))
            print(f"[OK] {name} -> {out_dir} ({len(ds)} rows)")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed.append(name)

    if failed:
        print(f"\nFailed datasets: {failed}")
        sys.exit(1)
    else:
        print(f"\nAll datasets saved to {CACHE_DIR}")


if __name__ == "__main__":
    main()
