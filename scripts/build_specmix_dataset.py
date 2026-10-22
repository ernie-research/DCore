"""Build ``specmix``: one fixed, shuffled mixed-traffic evaluation set.

The 10 SpecBench datasets are truncated to a fixed per-source quota, merged into
a single dataset and shuffled with a fixed seed, so a benchmark run issues
requests from all domains interleaved -- the way a real serving endpoint sees
traffic -- instead of one homogeneous domain at a time.

The result is frozen on disk at ``cache/dataset/specmix`` and loaded through the
normal ``specbench.datasets.load_and_process_dataset("specmix")`` path.

The frozen set used in the paper is published at
https://huggingface.co/datasets/wingchiuloong/specmix and is downloaded
automatically when ``cache/dataset/specmix`` is missing, so this script is only
needed to rebuild it from the source benchmarks. A rebuild can differ from the
published copy if a source dataset changes upstream (e.g. swe-bench falls back to
SWE-bench_Lite when its preferred source is unavailable).

Each row carries:
  * ``turns``        -- single-element list with the already-formatted prompt
                        (identical string to what the per-dataset path produces)
  * ``source``       -- originating dataset name
  * ``source_index`` -- row index inside that dataset (for cross-referencing)
  * ``gold``         -- gold answer for scorable sources, ``""`` otherwise

Usage::

    python scripts/build_specmix_dataset.py            # write cache/dataset/specmix
    python scripts/build_specmix_dataset.py --stats     # only print composition
"""

from __future__ import annotations

import argparse
from collections import Counter

from datasets import Dataset, Features, Sequence, Value

from specbench.accuracy import extract_gold
from specbench.config import REPO_ROOT
from specbench.datasets import load_and_process_dataset

DATASET_NAME = "specmix"
SEED = 42

# Per-source quota. Rows are taken as the first N of each dataset (same slice a
# normal ``--datasets name:N`` run would use), then the merged set is shuffled.
MIX: tuple[tuple[str, int], ...] = (
    ("gsm8k", 500),
    ("math500", 500),
    ("aime25", 30),
    ("humaneval", 164),
    ("mbpp", 257),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard", 500),
    ("swe-bench", 300),
)

FEATURES = Features({
    "turns": Sequence(Value("large_string")),
    "source": Value("string"),
    "source_index": Value("int32"),
    "gold": Value("large_string"),
})


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for name, quota in MIX:
        ds = load_and_process_dataset(name)
        take = min(quota, len(ds))
        if take < quota:
            print(f"[specmix] warning: {name} has only {len(ds)} rows, "
                  f"quota {quota} truncated to {take}")
        for i in range(take):
            row = ds[i]
            rows.append({
                "turns": [row["turns"][0]],
                "source": name,
                "source_index": i,
                "gold": extract_gold(name, row) or "",
            })
        print(f"[specmix] {name}: {take} rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="build the specmix evaluation set")
    parser.add_argument("--stats", action="store_true",
                        help="print composition of the existing set and exit")
    parser.add_argument("--output", default=None,
                        help=f"output dir (default: cache/dataset/{DATASET_NAME})")
    args = parser.parse_args()

    out_dir = args.output or str(REPO_ROOT / "cache" / "dataset" / DATASET_NAME)

    if args.stats:
        ds = load_and_process_dataset(DATASET_NAME)
        counts = Counter(ds["source"])
        print(f"{DATASET_NAME}: {len(ds)} rows")
        for name, _ in MIX:
            print(f"  {name:15s} {counts.get(name, 0)}")
        print(f"  scorable (gold != ''): {sum(1 for g in ds['gold'] if g)}")
        print("  first 10 sources:", ds["source"][:10])
        return

    rows = build_rows()
    ds = Dataset.from_list(rows, features=FEATURES).shuffle(seed=SEED)
    ds.save_to_disk(out_dir)
    print(f"[specmix] wrote {len(ds)} rows (seed={SEED}) to {out_dir}")


if __name__ == "__main__":
    main()
