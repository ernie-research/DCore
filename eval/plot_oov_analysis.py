#!/usr/bin/env python3
"""Visualize OOV acceptance analysis results.

Reads oov_analysis.jsonl and produces:
  1. oov_acceptance_bar.pdf  — dual-axis bar chart (acceptance rate + OOV proportion)
  2. oov_acceptance_by_position.pdf — per-position line chart (OOV vs In-Vocab)

Usage:
    python eval/plot_oov_analysis.py <oov_analysis_dir>
    python eval/plot_oov_analysis.py path/to/oov_analysis.jsonl [--out-dir .]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MAPPING_LABELS = {
    "vocab_mapping_top32000": "32000",
    "vocab_mapping": "18992",
    "vocab_mapping_16seg": "9496",
}


def load_data(jsonl_path):
    """Load and aggregate across datasets. Returns (agg, per_dataset)."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    # Aggregate across all datasets
    agg = {}  # mapping_name -> aggregated stats
    per_dataset = {}  # dataset_name -> same structure as agg
    for rec in records:
        ds = rec["dataset"]
        if ds not in per_dataset:
            per_dataset[ds] = {}
        for mname, mdata in rec["mappings"].items():
            if mname not in agg:
                agg[mname] = {"iv_total": 0, "iv_accept": 0, "oov_total": 0, "oov_accept": 0,
                              "vocab_size": mdata["vocab_size"], "oov_count": mdata["oov_count"],
                              "per_position": {}}
            agg[mname]["iv_total"] += mdata["in_vocab_total"]
            agg[mname]["iv_accept"] += mdata["in_vocab_accepted"]
            agg[mname]["oov_total"] += mdata["oov_total"]
            agg[mname]["oov_accept"] += mdata["oov_accepted"]
            # Merge per_position if present
            if "per_position" in mdata:
                for pos, pdata in mdata["per_position"].items():
                    if pos not in agg[mname]["per_position"]:
                        agg[mname]["per_position"][pos] = {"iv_accept": 0, "iv_total": 0, "oov_accept": 0, "oov_total": 0}
                    for k in ("iv_accept", "iv_total", "oov_accept", "oov_total"):
                        agg[mname]["per_position"][pos][k] += pdata[k]
            # Per-dataset
            if mname not in per_dataset[ds]:
                per_dataset[ds][mname] = {"iv_total": 0, "iv_accept": 0, "oov_total": 0, "oov_accept": 0,
                                          "vocab_size": mdata["vocab_size"], "oov_count": mdata["oov_count"],
                                          "per_position": {}}
            per_dataset[ds][mname]["iv_total"] += mdata["in_vocab_total"]
            per_dataset[ds][mname]["iv_accept"] += mdata["in_vocab_accepted"]
            per_dataset[ds][mname]["oov_total"] += mdata["oov_total"]
            per_dataset[ds][mname]["oov_accept"] += mdata["oov_accepted"]
            if "per_position" in mdata:
                for pos, pdata in mdata["per_position"].items():
                    if pos not in per_dataset[ds][mname]["per_position"]:
                        per_dataset[ds][mname]["per_position"][pos] = {"iv_accept": 0, "iv_total": 0, "oov_accept": 0, "oov_total": 0}
                    for k in ("iv_accept", "iv_total", "oov_accept", "oov_total"):
                        per_dataset[ds][mname]["per_position"][pos][k] += pdata[k]
    return agg, per_dataset


def load_per_position_from_log(log_dir, mapping_name="vocab_mapping_16seg", dataset=None):
    """Fallback: parse per-position data from log files when JSONL lacks it."""
    pos_data = {}  # pos -> {iv_accept, iv_total, oov_accept, oov_total} aggregated
    pattern = re.compile(r"pos (\d+): in-vocab [\d.]+% \((\d+)/(\d+)\)\s+OOV [\d.]+% \((\d+)/(\d+)\)")
    glob_pattern = f"{dataset}.log" if dataset else "*.log"
    for log_file in sorted(Path(log_dir).glob(glob_pattern)):
        in_section = False
        with open(log_file) as f:
            for line in f:
                if f"[{mapping_name}]" in line and "vocab_size=" in line:
                    in_section = True
                    continue
                if in_section:
                    m = pattern.search(line)
                    if m:
                        pos = int(m.group(1))
                        if pos not in pos_data:
                            pos_data[pos] = {"iv_accept": 0, "iv_total": 0, "oov_accept": 0, "oov_total": 0}
                        pos_data[pos]["iv_accept"] += int(m.group(2))
                        pos_data[pos]["iv_total"] += int(m.group(3))
                        pos_data[pos]["oov_accept"] += int(m.group(4))
                        pos_data[pos]["oov_total"] += int(m.group(5))
                    elif line.strip() == "" or (line[0] == "["):
                        in_section = False
    return pos_data


def plot_bar_chart(agg, out_path):
    """Fig 1: Dual-axis bar chart — acceptance rate + OOV token proportion."""
    ordered = ["vocab_mapping_top32000", "vocab_mapping", "vocab_mapping_16seg"]
    ordered = [m for m in ordered if m in agg]

    labels = [MAPPING_LABELS.get(m, m) for m in ordered]
    iv_rates = [agg[m]["iv_accept"] / agg[m]["iv_total"] * 100 for m in ordered]
    oov_rates = [agg[m]["oov_accept"] / agg[m]["oov_total"] * 100 for m in ordered]
    oov_pcts = [agg[m]["oov_total"] / (agg[m]["iv_total"] + agg[m]["oov_total"]) * 100 for m in ordered]

    x = np.arange(len(ordered))
    width = 0.3

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    bars_iv = ax1.bar(x - width/2, iv_rates, width, label="In-Vocab Accept Rate", color="#4C72B0", edgecolor="white")
    bars_oov = ax1.bar(x + width/2, oov_rates, width, label="OOV Accept Rate", color="#DD8452", edgecolor="white")

    # Annotate gap above OOV bar
    for i in range(len(ordered)):
        gap = iv_rates[i] - oov_rates[i]
        ax1.annotate(f"{gap:.0f}pp gap", xy=(x[i] + width/2, oov_rates[i] + 2), fontsize=8, ha="center", color="#555")

    # OOV proportion as markers on right axis
    ax2.plot(x, oov_pcts, "D-", color="#55A868", markersize=8, linewidth=2, label="OOV Token %")

    ax1.set_xlabel("Vocabulary Mapping")
    ax1.set_ylabel("Acceptance Rate (%)")
    ax2.set_ylabel("OOV Token Proportion (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax1.set_title("OOV vs In-Vocab: Acceptance Rate & Token Proportion", fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_position_chart(agg, out_dir, log_dir=None, suffix="", title_extra="", dataset=None):
    """Fig 2: Per-position acceptance rate line chart + OOV proportion."""
    mapping = "vocab_mapping_16seg"
    out_path = os.path.join(out_dir, f"oov_acceptance_by_position{suffix}.pdf")

    # Try JSONL per_position data first
    pp = None
    if mapping in agg and agg[mapping].get("per_position"):
        pp = agg[mapping]["per_position"]
    elif log_dir:
        pp = load_per_position_from_log(log_dir, mapping, dataset=dataset)

    if not pp:
        print(f"No per-position data available. Skipping position chart{suffix}.")
        return

    positions = sorted(pp.keys(), key=lambda p: int(p))
    iv_rates = [pp[p]["iv_accept"] / pp[p]["iv_total"] * 100 if pp[p]["iv_total"] else 0 for p in positions]
    oov_rates = [pp[p]["oov_accept"] / pp[p]["oov_total"] * 100 if pp[p]["oov_total"] else 0 for p in positions]
    oov_pct = [pp[p]["oov_total"] / (pp[p]["iv_total"] + pp[p]["oov_total"]) * 100
               if (pp[p]["iv_total"] + pp[p]["oov_total"]) else 0 for p in positions]

    x = [int(p) for p in positions]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, iv_rates, "o-", color="#4C72B0", linewidth=2, markersize=5, label="In-Vocab Accept Rate")
    ax.plot(x, oov_rates, "s-", color="#DD8452", linewidth=2, markersize=5, label="OOV Accept Rate")
    ax.fill_between(x, oov_rates, iv_rates, alpha=0.1, color="#DD8452")

    # Right axis: OOV token proportion at each position
    ax2 = ax.twinx()
    ax2.plot(x, oov_pct, "^--", color="#55A868", linewidth=1.5, markersize=5, label="OOV Token %")
    ax2.set_ylabel("OOV Token Proportion (%)", color="#55A868")
    ax2.tick_params(axis="y", labelcolor="#55A868")
    ax2.set_ylim(0, 100)

    ax.set_xlabel("Draft Position")
    ax.set_ylabel("Acceptance Rate (%)")
    title = "OOV vs In-Vocab: Acceptance Rate & OOV Proportion by Draft Position"
    if title_extra and title_extra != "all datasets":
        title += f" [{title_extra}]"
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot OOV acceptance analysis")
    parser.add_argument("input", help="Path to oov_analysis.jsonl or its parent directory")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        jsonl_path = input_path / "oov_analysis.jsonl"
        log_dir = input_path
    else:
        jsonl_path = input_path
        log_dir = input_path.parent

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or str(log_dir)
    os.makedirs(out_dir, exist_ok=True)

    agg, per_dataset = load_data(str(jsonl_path))
    plot_bar_chart(agg, os.path.join(out_dir, "oov_acceptance_bar.pdf"))
    # Aggregated position chart
    plot_position_chart(agg, out_dir, log_dir=str(log_dir), suffix="", title_extra="all datasets")
    # Per-dataset position charts
    for ds, ds_agg in per_dataset.items():
        plot_position_chart(ds_agg, out_dir, log_dir=str(log_dir), suffix=f"_{ds}", title_extra=ds, dataset=ds)


if __name__ == "__main__":
    main()
