#!/usr/bin/env python3
"""Aggregate profiling JSON files into LaTeX tables (median & mean).

Usage:
    python scripts/profile_to_latex.py <profiles_dir> [--output-dir <dir>]

Reads all *.json files in <profiles_dir>, computes median and mean of each
timing metric across all ranks and records (skipping warmup), and emits two
LaTeX tables: profile_median.tex and profile_mean.tex.

Records with forward_ct <= 2 are skipped (warmup). All batch sizes are mixed.
Missing metrics for a method are shown as '---'.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# Ordered columns: core shared first, then method-specific in logical order.
COLUMN_ORDER = [
    "step_cpu_ms",
    "step_gpu_ms",
    "draft_gpu_ms",
    "draft_backbone_gpu_ms",
    "draft_kv_materialize_gpu_ms",
    "draft_extend_gpu_ms",
    "draft_prepare_gpu_ms",
    "draft_tree_build_gpu_ms",
    "target_verify_gpu_ms",
    "target_forward_gpu_ms",
    "accept_bonus_gpu_ms",
    "verify_sample_gpu_ms",
]

# Short display names for LaTeX headers.
COLUMN_LABELS = {
    "step_cpu_ms": "step\\textsubscript{cpu}",
    "step_gpu_ms": "step\\textsubscript{gpu}",
    "draft_gpu_ms": "draft",
    "draft_backbone_gpu_ms": "backbone",
    "draft_kv_materialize_gpu_ms": "kv\\_mat",
    "draft_extend_gpu_ms": "extend",
    "draft_prepare_gpu_ms": "prepare",
    "draft_tree_build_gpu_ms": "tree",
    "target_verify_gpu_ms": "verify",
    "target_forward_gpu_ms": "tgt\\_fwd",
    "accept_bonus_gpu_ms": "bonus",
    "verify_sample_gpu_ms": "vsample",
}

WARMUP_SKIP = 2  # skip records with forward_ct <= this value


def load_profile(path: Path) -> dict[str, np.ndarray]:
    """Load a profile JSON, return {metric: array_of_values} across all ranks."""
    data = json.loads(path.read_text())
    ranks = data.get("ranks", [])
    metric_values: dict[str, list[float]] = {}

    for rank_data in ranks:
        for record in rank_data.get("records", []):
            if record.get("forward_ct", 0) <= WARMUP_SKIP:
                continue
            for key, value in record.items():
                if key.endswith("_ms") and isinstance(value, (int, float)):
                    metric_values.setdefault(key, []).append(float(value))

    return {k: np.array(v) for k, v in metric_values.items()}


def determine_columns(all_methods: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """Return ordered list of columns that appear in at least one method."""
    present = set()
    for metrics in all_methods.values():
        present.update(metrics.keys())
    return [c for c in COLUMN_ORDER if c in present]


def format_val(v: Optional[float]) -> str:
    if v is None:
        return "---"
    if v < 0.1:
        return f"{v:.3f}"
    if v < 10:
        return f"{v:.2f}"
    return f"{v:.1f}"


def build_latex_table(
    all_methods: dict[str, dict[str, np.ndarray]],
    columns: list[str],
    agg: str,  # "median" or "mean"
) -> str:
    """Build a LaTeX tabular string."""
    agg_fn = np.median if agg == "median" else np.mean

    col_spec = "l" + "r" * len(columns)
    headers = [COLUMN_LABELS.get(c, c.replace("_", "\\_")) for c in columns]

    lines = []
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("method & " + " & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    for method_name in sorted(all_methods.keys()):
        metrics = all_methods[method_name]
        cells = []
        for col in columns:
            if col in metrics and len(metrics[col]) > 0:
                cells.append(format_val(float(agg_fn(metrics[col]))))
            else:
                cells.append("---")
        escaped_name = method_name.replace("_", "\\_")
        lines.append(f"{escaped_name} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles_dir", type=Path, help="Directory containing profile JSONs")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: same as profiles_dir parent)")
    args = parser.parse_args()

    profiles_dir = args.profiles_dir
    if not profiles_dir.is_dir():
        print(f"Error: {profiles_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or profiles_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all profiles.
    all_methods: dict[str, dict[str, np.ndarray]] = {}
    for json_file in sorted(profiles_dir.glob("*.json")):
        method_name = json_file.stem
        metrics = load_profile(json_file)
        if metrics:
            all_methods[method_name] = metrics

    if not all_methods:
        print("No profile data found.", file=sys.stderr)
        sys.exit(1)

    columns = determine_columns(all_methods)
    print(f"Methods: {len(all_methods)}, Columns: {len(columns)}")
    print(f"  {columns}")

    for agg in ("median", "mean"):
        table = build_latex_table(all_methods, columns, agg)
        out_path = output_dir / f"profile_{agg}.tex"
        out_path.write_text(table)
        print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
