#!/usr/bin/env python3
"""Cluster-bootstrap token-level PQ entropy gaps by request id.

The input directory contains one or more method subdirectories with PQ JSONL
files. Files are streamed; only per-rid, per-position sums and counts are kept
in memory.

For each position ``i`` the script compares rows with ``i < nc`` (accepted
path) against rows with ``i == nc`` (rejection frontier). The entropy gap is a
paired cluster statistic whenever a rid contributes to both groups.

Usage:
    python eval/analyze_pq_cluster_bootstrap.py \
        --pq-dir outputs/eval/qwen3-4b/20260905_profile/b8/t1rs/dt10/c64/pq \
        --budget 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PQ_DIR = ROOT / (
    "outputs/eval/qwen3-4b/20260905_profile/b8/t1rs/dt10/c64/pq"
)
# The order follows the Qwen3-4B sweep script; unknown future methods are
# appended alphabetically.
METHOD_ORDER = [
    "eagle3",
    "dflash",
    "dflare",
    "dpace",
    "domino",
    "dspark_markov",
    "dcore",
    "dspark_markov_origin",
]
# Per rid and position, the twelve cells are:
# accepted entropy sum/count, frontier entropy sum/count,
# online acceptance sum/count, frontier hypothetical acceptance sum/count,
# attempted target entropy sum/count, attempted p/q acceptance sum/count.
N_STATS = 12


class DataError(RuntimeError):
    """Raised when the PQ dump violates the expected schema or invariants."""


def _new_rid_stats(num_positions: int) -> np.ndarray:
    return np.zeros((num_positions, N_STATS), dtype=np.float64)


def _new_path_stats(num_positions: int) -> np.ndarray:
    # Indexed by [nc, position], with four cells: entropy sum/count and
    # p/q acceptance sum/count. nc includes the full-window acceptance case.
    return np.zeros((num_positions + 1, num_positions, 4), dtype=np.float64)


def _finite_float(value: object, name: str, path: Path, line_no: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataError(f"{path}:{line_no}: {name} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise DataError(f"{path}:{line_no}: {name} is not finite: {value!r}")
    return result


def _load_method(
    method_dir: Path, num_positions: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Stream one method's JSONL files and return rid aggregates and checks."""
    rid_stats: dict[str, np.ndarray] = defaultdict(
        lambda: _new_rid_stats(num_positions)
    )
    path_stats: dict[str, np.ndarray] = defaultdict(
        lambda: _new_path_stats(num_positions)
    )
    paths = sorted(method_dir.glob("*.jsonl"))
    checks = {
        "files": len(paths),
        "rows": 0,
        "position_counts": np.zeros(num_positions, dtype=np.int64),
        "nc_counts": np.zeros(num_positions + 1, dtype=np.int64),
        "bad_invariant": 0,
    }
    if not paths:
        raise DataError(f"No JSONL files found in {method_dir}")

    required = {"rid", "i", "nc", "acc", "lp", "lq", "ent"}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(f"{path}:{line_no}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise DataError(f"{path}:{line_no}: JSON row must be an object")
                missing = required.difference(row)
                if missing:
                    raise DataError(f"{path}:{line_no}: missing fields {sorted(missing)}")

                rid_value = row["rid"]
                if rid_value is None or not str(rid_value):
                    raise DataError(f"{path}:{line_no}: rid must be non-empty")
                rid = str(rid_value)
                try:
                    position = int(row["i"])
                    nc = int(row["nc"])
                    acc = int(row["acc"])
                except (TypeError, ValueError) as exc:
                    raise DataError(f"{path}:{line_no}: i/nc/acc must be integers") from exc
                if not 0 <= position < num_positions:
                    raise DataError(
                        f"{path}:{line_no}: expected i in 0..{num_positions - 1}, "
                        f"got {position}"
                    )
                if not 0 <= nc <= num_positions:
                    raise DataError(
                        f"{path}:{line_no}: expected nc in 0..{num_positions}, got {nc}"
                    )
                if acc not in (0, 1):
                    raise DataError(f"{path}:{line_no}: expected binary acc, got {acc}")
                lp = _finite_float(row["lp"], "lp", path, line_no)
                lq = None if row["lq"] is None else _finite_float(row["lq"], "lq", path, line_no)
                entropy = _finite_float(row["ent"], "ent", path, line_no)
                pq_acceptance = (
                    None if lq is None else math.exp(min(lp - lq, 0.0))
                )

                expected_acc = int(position < nc)
                if acc != expected_acc:
                    checks["bad_invariant"] += 1
                checks["rows"] += 1
                checks["position_counts"][position] += 1
                checks["nc_counts"][nc] += 1

                cells = rid_stats[rid][position]
                if position < nc:
                    cells[0] += entropy
                    cells[1] += 1.0
                if position == nc:
                    cells[2] += entropy
                    cells[3] += 1.0
                    if pq_acceptance is not None:
                        cells[6] += pq_acceptance
                        cells[7] += 1.0
                if position <= nc:
                    cells[4] += acc
                    cells[5] += 1.0
                    cells[8] += entropy
                    cells[9] += 1.0
                    if pq_acceptance is not None:
                        cells[10] += pq_acceptance
                        cells[11] += 1.0
                # Rows past the frontier condition on a rejected draft token, so
                # they are counterfactual and never enter the nc-keyed grid.
                if position <= nc:
                    path_cells = path_stats[rid][nc, position]
                    path_cells[0] += entropy
                    path_cells[1] += 1.0
                    if pq_acceptance is not None:
                        path_cells[2] += pq_acceptance
                        path_cells[3] += 1.0

    return rid_stats, path_stats, checks


def _column_means(values: np.ndarray) -> np.ndarray:
    """Mean over sampled rid rows, ignoring unavailable NaN cells."""
    valid = np.isfinite(values)
    counts = valid.sum(axis=1)
    sums = np.nansum(values, axis=1)
    return np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _bootstrap_metric(
    matrix: np.ndarray, reps: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap each position using only its valid rid clusters."""
    boot = np.full((reps, matrix.shape[1]), np.nan)
    chunk_size = 64
    for position in range(matrix.shape[1]):
        values = matrix[:, position]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        for start in range(0, reps, chunk_size):
            stop = min(start + chunk_size, reps)
            indices = rng.integers(
                0, values.size, size=(stop - start, values.size)
            )
            boot[start:stop, position] = np.mean(values[indices], axis=1)
    return boot


def _bootstrap_method(
    gap_matrix: np.ndarray,
    online_matrix: np.ndarray,
    hypothetical_matrix: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Bootstrap metrics using only rids valid for each metric."""
    observed_gap = np.array(
        [
            np.nanmean(gap_matrix[:, i])
            if np.isfinite(gap_matrix[:, i]).any()
            else np.nan
            for i in range(gap_matrix.shape[1])
        ]
    )
    gap_boot = _bootstrap_metric(gap_matrix, reps, rng)
    online_boot = _bootstrap_metric(online_matrix, reps, rng)
    hypothetical_boot = _bootstrap_metric(hypothetical_matrix, reps, rng)
    return {
        "gap": gap_boot,
        "gap_null": gap_boot - observed_gap[np.newaxis, :],
        "online": online_boot,
        "hypothetical": hypothetical_boot,
    }


def _ci(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    return tuple(np.percentile(values, [2.5, 97.5]))


def _mean(values: list[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else math.nan


def _gap_p_value(null_boot: np.ndarray, observed: float) -> float:
    null_boot = null_boot[np.isfinite(null_boot)]
    if null_boot.size == 0 or not math.isfinite(observed):
        return math.nan
    # Centered bootstrap is the null distribution for H_frontier-H_accepted=0.
    return float((np.count_nonzero(np.abs(null_boot) >= abs(observed)) + 1) / (null_boot.size + 1))


def _significance(q_value: float) -> str:
    if not math.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def _bh_fdr(rows: list[dict]) -> None:
    valid = [(idx, row["gap_p_raw"]) for idx, row in enumerate(rows) if math.isfinite(row["gap_p_raw"])]
    if not valid:
        return
    order = sorted(valid, key=lambda item: item[1])
    m = len(order)
    adjusted = [0.0] * m
    running = 1.0
    for rank_from_end, (_, p_value) in enumerate(reversed(order), 1):
        rank = m - rank_from_end + 1
        running = min(running, p_value * m / rank)
        adjusted[rank - 1] = running
    for (row_idx, _), q_value in zip(order, adjusted):
        rows[row_idx]["gap_q_fdr"] = min(max(q_value, 0.0), 1.0)
        rows[row_idx]["significance"] = _significance(rows[row_idx]["gap_q_fdr"])


def _format_value(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    if isinstance(value, np.floating):
        return float(value)
    return value


def _build_method_results(
    method: str,
    rid_stats: dict[str, np.ndarray],
    path_stats: dict[str, np.ndarray],
    checks: dict,
    reps: int,
    rng: np.random.Generator,
    num_positions: int,
) -> tuple[list[dict], dict, dict, list[dict]]:
    rid_ids = sorted(rid_stats)
    n_rids = len(rid_ids)
    accepted_matrix = np.full((n_rids, num_positions), np.nan)
    frontier_matrix = np.full((n_rids, num_positions), np.nan)
    gap_matrix = np.full((n_rids, num_positions), np.nan)
    online_matrix = np.full((n_rids, num_positions), np.nan)
    hypothetical_matrix = np.full((n_rids, num_positions), np.nan)
    target_entropy_matrix = np.full((n_rids, num_positions), np.nan)
    pq_matrix = np.full((n_rids, num_positions), np.nan)
    accepted_obs = np.zeros(num_positions, dtype=np.int64)
    frontier_obs = np.zeros(num_positions, dtype=np.int64)
    online_obs = np.zeros(num_positions, dtype=np.int64)
    hypothetical_obs = np.zeros(num_positions, dtype=np.int64)

    for rid_idx, rid in enumerate(rid_ids):
        values = rid_stats[rid]
        for position in range(num_positions):
            accepted_sum, accepted_count = values[position, 0:2]
            frontier_sum, frontier_count = values[position, 2:4]
            online_sum, online_count = values[position, 4:6]
            hypothetical_sum, hypothetical_count = values[position, 6:8]
            target_entropy_sum, target_entropy_count = values[position, 8:10]
            pq_sum, pq_count = values[position, 10:12]
            if accepted_count:
                accepted_matrix[rid_idx, position] = accepted_sum / accepted_count
                accepted_obs[position] += int(accepted_count)
            if frontier_count:
                frontier_matrix[rid_idx, position] = frontier_sum / frontier_count
                frontier_obs[position] += int(frontier_count)
            if accepted_count and frontier_count:
                gap_matrix[rid_idx, position] = (
                    frontier_sum / frontier_count - accepted_sum / accepted_count
                )
            if online_count:
                online_matrix[rid_idx, position] = online_sum / online_count
                online_obs[position] += int(online_count)
            if hypothetical_count:
                hypothetical_matrix[rid_idx, position] = hypothetical_sum / hypothetical_count
                hypothetical_obs[position] += int(hypothetical_count)
            if target_entropy_count:
                target_entropy_matrix[rid_idx, position] = target_entropy_sum / target_entropy_count
            if pq_count:
                pq_matrix[rid_idx, position] = pq_sum / pq_count

    bootstrap = _bootstrap_method(
        gap_matrix, online_matrix, hypothetical_matrix, reps=reps, rng=rng
    )
    gap_point = np.array(
        [np.nanmean(gap_matrix[:, i]) if np.isfinite(gap_matrix[:, i]).any() else np.nan for i in range(num_positions)]
    )
    online_point = np.array(
        [np.nanmean(online_matrix[:, i]) if np.isfinite(online_matrix[:, i]).any() else np.nan for i in range(num_positions)]
    )
    hypothetical_point = np.array(
        [
            np.nanmean(hypothetical_matrix[:, i])
            if np.isfinite(hypothetical_matrix[:, i]).any()
            else np.nan
            for i in range(num_positions)
        ]
    )
    target_entropy_point = np.array(
        [
            np.nanmean(target_entropy_matrix[:, i])
            if np.isfinite(target_entropy_matrix[:, i]).any()
            else np.nan
            for i in range(num_positions)
        ]
    )
    pq_point = np.array(
        [
            np.nanmean(pq_matrix[:, i])
            if np.isfinite(pq_matrix[:, i]).any()
            else np.nan
            for i in range(num_positions)
        ]
    )

    results = []
    for position in range(num_positions):
        accepted_values = accepted_matrix[:, position]
        frontier_values = frontier_matrix[:, position]
        gap_values = gap_matrix[:, position]
        online_values = online_matrix[:, position]
        hypothetical_values = hypothetical_matrix[:, position]
        gap_ci = _ci(bootstrap["gap"][:, position])
        online_ci = _ci(bootstrap["online"][:, position])
        hypothetical_ci = _ci(bootstrap["hypothetical"][:, position])
        results.append(
            {
                "method": method,
                "position": position,
                "accepted_entropy": _mean(accepted_values),
                "frontier_entropy": _mean(frontier_values),
                "target_entropy": target_entropy_point[position],
                "pq_acceptance": pq_point[position],
                "entropy_gap_frontier_minus_accepted": gap_point[position],
                "gap_ci_low": gap_ci[0],
                "gap_ci_high": gap_ci[1],
                "gap_p_raw": _gap_p_value(bootstrap["gap_null"][:, position], gap_point[position]),
                "gap_q_fdr": math.nan,
                "significance": "",
                "accepted_rids": int(np.isfinite(accepted_values).sum()),
                "frontier_rids": int(np.isfinite(frontier_values).sum()),
                "gap_rids": int(np.isfinite(gap_values).sum()),
                "online_acceptance": online_point[position],
                "online_ci_low": online_ci[0],
                "online_ci_high": online_ci[1],
                "online_rids": int(np.isfinite(online_values).sum()),
                "frontier_hypothetical_acceptance": hypothetical_point[position],
                "frontier_hypothetical_ci_low": hypothetical_ci[0],
                "frontier_hypothetical_ci_high": hypothetical_ci[1],
                "frontier_hypothetical_rids": int(np.isfinite(hypothetical_values).sum()),
                "accepted_observations": int(accepted_obs[position]),
                "frontier_observations": int(frontier_obs[position]),
                "online_observations": int(online_obs[position]),
                "frontier_hypothetical_observations": int(hypothetical_obs[position]),
            }
        )

    # Method-level summaries use the same rid bootstrap. Position-equal means
    # give every position equal weight; rid-equal means give every rid equal
    # weight after averaging its available positions.
    position_means = np.array(
        [np.nanmean(gap_matrix[:, i]) if np.isfinite(gap_matrix[:, i]).any() else np.nan for i in range(num_positions)]
    )
    rid_means = np.array(
        [np.nanmean(gap_matrix[i, :]) if np.isfinite(gap_matrix[i, :]).any() else np.nan for i in range(n_rids)]
    )
    all_gaps = gap_matrix[np.isfinite(gap_matrix)]
    method_summary = {
        "method": method,
        "position_equal_gap": _mean(position_means),
        "position_equal_ci_low": _ci(_column_means(bootstrap["gap"]))[0],
        "position_equal_ci_high": _ci(_column_means(bootstrap["gap"]))[1],
        "rid_equal_gap": _mean(rid_means),
        "rid_equal_ci_low": math.nan,
        "rid_equal_ci_high": math.nan,
        "rid_weighted_gap": _mean(all_gaps),
        "effective_rids": int(np.isfinite(rid_means).sum()),
        "position_gap_rids": [int(np.isfinite(gap_matrix[:, i]).sum()) for i in range(num_positions)],
    }
    # A second summary bootstrap over valid rid-level means gives the
    # equal-rid aggregate a CI with the intended cluster unit.
    valid_rid_means = rid_means[np.isfinite(rid_means)]
    if valid_rid_means.size:
        rid_boot = _bootstrap_metric(valid_rid_means[:, np.newaxis], reps, rng)[:, 0]
        method_summary["rid_equal_ci_low"], method_summary["rid_equal_ci_high"] = _ci(rid_boot)

    # Keep nc explicit for the figure: each rid is averaged within a path
    # first, then paths are averaged across rids. Both metrics stop at the nc
    # frontier, so every path occupies the lower-left triangle i <= nc.
    path_rows = []
    for nc in range(1, num_positions + 1):
        for position in range(num_positions):
            entropy_values = []
            pq_values = []
            entropy_observations = 0
            pq_observations = 0
            for rid in rid_ids:
                entropy_sum, entropy_count, pq_sum, pq_count = path_stats[rid][nc, position]
                if entropy_count:
                    entropy_values.append(entropy_sum / entropy_count)
                    entropy_observations += int(entropy_count)
                if pq_count:
                    pq_values.append(pq_sum / pq_count)
                    pq_observations += int(pq_count)
            path_rows.append(
                {
                    "method": method,
                    "nc": nc,
                    "position": position,
                    "target_entropy": _mean(entropy_values),
                    "pq_acceptance": _mean(pq_values),
                    "path_rids": len(entropy_values),
                    "pq_rids": len(pq_values),
                    "path_observations": entropy_observations,
                    "pq_observations": pq_observations,
                }
            )

    return results, method_summary, {
        "files": checks["files"],
        "rows": checks["rows"],
        "rids": n_rids,
        "position_counts": checks["position_counts"],
        "nc_counts": checks["nc_counts"],
        "bad_invariant": checks["bad_invariant"],
    }, path_rows


def _write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "method", "position", "accepted_entropy", "frontier_entropy",
        "target_entropy", "pq_acceptance",
        "entropy_gap_frontier_minus_accepted", "gap_ci_low", "gap_ci_high",
        "gap_p_raw", "gap_q_fdr", "significance", "accepted_rids",
        "frontier_rids", "gap_rids", "online_acceptance", "online_ci_low",
        "online_ci_high", "online_rids", "frontier_hypothetical_acceptance",
        "frontier_hypothetical_ci_low", "frontier_hypothetical_ci_high",
        "frontier_hypothetical_rids", "accepted_observations",
        "frontier_observations", "online_observations",
        "frontier_hypothetical_observations",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_value(row[field]) for field in fields})


def _write_path_csv(path_rows: list[dict], path: Path) -> None:
    fields = [
        "method", "nc", "position", "target_entropy", "pq_acceptance",
        "path_rids", "pq_rids", "path_observations", "pq_observations",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in path_rows:
            writer.writerow({field: _format_value(row[field]) for field in fields})


def _plot(path_rows: list[dict], methods: list[str], path: Path, budget: int) -> None:
    """Draw one lower-left-triangle entropy heatmap panel per method."""
    num_positions = budget - 1
    if len(methods) > 8:
        raise DataError("the requested 2x4 figure supports at most 8 methods")

    ncs = range(1, num_positions + 1)
    by_method = {
        method: {nc: [None] * num_positions for nc in ncs} for method in methods
    }
    for row in path_rows:
        if row["method"] in by_method and row["nc"] in by_method[row["method"]]:
            by_method[row["method"]][row["nc"]][row["position"]] = row

    entropy_values = np.asarray(
        [row["target_entropy"] for row in path_rows], dtype=float
    )
    entropy_values = entropy_values[np.isfinite(entropy_values)]
    if entropy_values.size:
        entropy_min = float(np.min(entropy_values))
        entropy_max = float(np.max(entropy_values))
        if entropy_min == entropy_max:
            entropy_max = entropy_min + 1.0
    else:
        entropy_min, entropy_max = 0.0, 1.0

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#D9D9D9")
    norm = matplotlib.colors.Normalize(vmin=entropy_min, vmax=entropy_max)
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.0))
    axes = np.asarray(axes).reshape(-1)
    fig.subplots_adjust(
        left=0.06, right=0.98, top=0.88, bottom=0.07, hspace=0.24, wspace=0.24
    )

    row_centers = num_positions - np.arange(num_positions) - 0.5
    position_centers = np.arange(num_positions, dtype=float) + 0.5
    # Long chains would smear one label per cell into an unreadable band.
    tick_stride = max(1, math.ceil(num_positions / 16))
    tick_indices = np.arange(0, num_positions, tick_stride)
    # A "0.42" label is ~2.4 em wide; keep it inside 90% of one cell, where a
    # panel is about 245 pt across.
    label_size = min(8.0, 0.9 * (245.0 / num_positions) / 2.4)

    for panel_index, (method, ax) in enumerate(zip(methods, axes)):
        entropy_grid = np.full((num_positions, num_positions), np.nan)
        for row_index, nc in enumerate(ncs):
            for position, row in enumerate(by_method[method][nc]):
                if row is not None and math.isfinite(row["target_entropy"]):
                    entropy_grid[row_index, position] = row["target_entropy"]

        ax.imshow(
            np.ma.masked_invalid(entropy_grid),
            cmap=cmap,
            norm=norm,
            interpolation="none",
            origin="upper",
            extent=(0, num_positions, 0, num_positions),
            aspect="equal",
        )

        for row_index in range(num_positions):
            for position in range(num_positions):
                value = entropy_grid[row_index, position]
                if not math.isfinite(value):
                    continue
                ax.text(
                    position_centers[position],
                    row_centers[row_index],
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=label_size,
                    # viridis is dark below the midpoint of the scale.
                    color="white" if norm(value) < 0.55 else "#101010",
                )

        ax.set_xlim(0, num_positions)
        ax.set_ylim(0, num_positions)
        ax.set_box_aspect(1)
        ax.set_xticks(position_centers[tick_indices])
        ax.set_xticklabels([str(int(i)) for i in tick_indices], fontsize=7)
        ax.set_yticks(row_centers[tick_indices])
        if panel_index % 4 == 0:
            ax.set_yticklabels(
                [f"nc={int(i) + 1}" for i in tick_indices], fontsize=7
            )
            ax.set_ylabel("Acceptance length nc", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_title(method, fontsize=10, pad=5)
        if panel_index >= 4:
            ax.set_xlabel("Draft position i", fontsize=8)
        ax.tick_params(length=2, colors="#333333")
        ax.set_facecolor("#D9D9D9")
        ax.set_xticks(np.arange(num_positions + 1), minor=True)
        ax.set_yticks(np.arange(num_positions + 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.7)

    for ax in axes[len(methods):]:
        ax.set_visible(False)

    colorbar_ax = fig.add_axes([0.35, 0.935, 0.30, 0.016])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.set_label("Target entropy (nats)", fontsize=8, labelpad=2)
    colorbar.ax.tick_params(labelsize=7, length=2)
    fig.suptitle(
        f"Target entropy on the realized path i<=nc by acceptance length "
        f"(B={budget})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _print_summary(
    methods: list[str],
    rows: list[dict],
    summaries: dict[str, dict],
    checks: dict[str, dict],
    budget: int,
) -> None:
    print("\nData integrity")
    print(f"  methods with JSONL: {len(methods)} ({', '.join(methods)})")
    for method in methods:
        check = checks[method]
        positions = ",".join(str(int(x)) for x in check["position_counts"])
        print(
            f"  {method}: files={check['files']} rows={check['rows']} rids={check['rids']} "
            f"rows_by_position=[{positions}] invariant_bad={check['bad_invariant']}"
        )
    print(
        f"  Required invariants: i=0..{budget - 2} present; "
        "acc == (i < nc) for every row."
    )

    print("\nMethod-level entropy-gap summary")
    print(
        f"{'method':<23} {'position-equal':>13} {'95% CI':>21} "
        f"{'rid-equal':>11} {'95% CI':>21} {'rid-weighted':>13} {'q<.05':>8}"
    )
    print("-" * 122)
    for method in methods:
        summary = summaries[method]
        significant = [
            f"i{row['position']}" for row in rows
            if row["method"] == method and math.isfinite(row["gap_q_fdr"]) and row["gap_q_fdr"] < 0.05
        ]
        print(
            f"{method:<23} {summary['position_equal_gap']:>13.4f} "
            f"[{summary['position_equal_ci_low']:.4f}, {summary['position_equal_ci_high']:.4f}] "
            f"{summary['rid_equal_gap']:>11.4f} "
            f"[{summary['rid_equal_ci_low']:.4f}, {summary['rid_equal_ci_high']:.4f}] "
            f"{summary['rid_weighted_gap']:>13.4f} "
            f"{','.join(significant) if significant else '-':>8}"
        )
        print(f"  {method} effective paired rid by position: {summary['position_gap_rids']}")
    print("  Gap test: paired rid cluster differences; p is a centered two-sided bootstrap null; q is BH-FDR over all method x position tests.")
    print("  Online i<nc acceptance is structurally 1 and frontier i=nc acceptance is structurally 0; the figure shows nc-conditioned target entropy restricted to the realized path i<=nc.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pq-dir", type=Path, default=DEFAULT_PQ_DIR, help=f"PQ root (default: {DEFAULT_PQ_DIR})")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: --pq-dir)")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method names; default: every subdirectory containing JSONL")
    parser.add_argument("--bootstrap-reps", type=int, default=5000, help="Cluster bootstrap repetitions (default: 5000)")
    parser.add_argument("--seed", type=int, default=0, help="NumPy random seed (default: 0)")
    parser.add_argument("--budget", type=int, required=True, help="Verification budget B; drawable positions are 0..B-2")
    args = parser.parse_args()
    if args.budget < 2:
        parser.error("--budget must be at least 2")
    if args.bootstrap_reps < 100:
        parser.error("--bootstrap-reps must be at least 100")

    pq_dir = args.pq_dir.expanduser().resolve()
    out_dir = (args.out_dir or pq_dir).expanduser().resolve()
    if not pq_dir.is_dir():
        parser.error(f"PQ directory does not exist: {pq_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered = sorted(
        path.name for path in pq_dir.iterdir()
        if path.is_dir() and any(path.glob("*.jsonl"))
    )
    if args.methods:
        requested = [method.strip() for method in args.methods.split(",") if method.strip()]
        methods = [method for method in requested if method in discovered]
        missing = sorted(set(requested) - set(methods))
        if missing:
            parser.error(f"Requested methods without JSONL: {', '.join(missing)}")
    else:
        methods = [method for method in METHOD_ORDER if method in discovered]
        methods.extend(method for method in discovered if method not in methods)
    if not methods:
        parser.error(f"No method subdirectories with JSONL under {pq_dir}")
    if len(methods) > 8:
        parser.error("the requested 2x4 figure supports at most 8 methods")

    rng = np.random.default_rng(args.seed)
    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    checks: dict[str, dict] = {}
    path_rows: list[dict] = []
    num_positions = args.budget - 1
    for method in methods:
        print(f"[{method}] streaming {pq_dir / method}", flush=True)
        rid_stats, method_path_stats, method_checks = _load_method(
            pq_dir / method, num_positions
        )
        if method_checks["bad_invariant"]:
            raise DataError(
                f"{method}: {method_checks['bad_invariant']} rows violate acc == (i < nc)"
            )
        if np.any(method_checks["position_counts"] == 0):
            missing_positions = np.flatnonzero(method_checks["position_counts"] == 0).tolist()
            raise DataError(f"{method}: no observations at positions {missing_positions}")
        rows, summary, check, method_path_rows = _build_method_results(
            method,
            rid_stats,
            method_path_stats,
            method_checks,
            args.bootstrap_reps,
            rng,
            num_positions,
        )
        all_rows.extend(rows)
        path_rows.extend(method_path_rows)
        summaries[method] = summary
        checks[method] = check

    _bh_fdr(all_rows)
    csv_path = out_dir / "pq_entropy_cluster_bootstrap.csv"
    path_csv_path = out_dir / "pq_entropy_paths.csv"
    png_path = out_dir / "pq_entropy_cluster_bootstrap.png"
    _write_csv(all_rows, csv_path)
    _write_path_csv(path_rows, path_csv_path)
    _plot(path_rows, methods, png_path, args.budget)
    _print_summary(methods, all_rows, summaries, checks, args.budget)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved path CSV: {path_csv_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    try:
        main()
    except DataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
