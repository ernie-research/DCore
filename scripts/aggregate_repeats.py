#!/usr/bin/env python3
"""Aggregate repeated specbench measurements into best-of-N / mean tables.

Usage:
    python scripts/aggregate_repeats.py <sweep_root> [--output <file.md>]

``eval/run_specbench_mixed_sweep.sh`` passes ``--repeats N`` so every
(dataset, concurrency) unit is measured N times inside one server launch;
each measurement is one row in ``<run_dir>/results.jsonl``, distinguished by the
``repeat`` field. ``specbench.cli report`` renders those rows one by one and
never aggregates them, so this script does the cross-repeat statistics.

Grouping key: (run_dir relative to sweep_root, method, draft_model, dataset,
concurrency).

Aggregation policy — must stay in sync with :func:`specbench.report.best_tps`
and with the protocol section of the paper:

* Speed metrics (``tok/s``, ``wall tok/s``, ``latency s``) report the **best**
  repeat: max for throughput, min for latency. Repeats replay identical work
  with the same seed, so host interference can only ever make a round slower;
  the best round is the least contaminated estimate.
* ``accept_len`` reports the **mean** with sample std and CV%. Acceptance is a
  property of the drafter, not of the machine, so it has no systematic
  direction of contamination and the mean is the better estimator.

Every best-of cell also prints ``spread``, the relative gap between the slowest
and fastest repeat. Best-of-N deliberately hides outliers from the reported
number, so the spread column is what remains to detect them: it is the signal
to re-run a unit, and units above ``HIGH_SPREAD_THRESHOLD`` are also listed on
stdout.

``speedup`` is computed here rather than read from the rows: the SGLang harness
leaves it empty and ``report`` joins a single baseline row per (dataset, conc).
It is best-over-best, so both sides use the same rule.

The ``baseline/`` subtree is skipped because ``merge_baseline`` already copies
those rows into every run_dir.

``--baseline-from <dir>`` borrows the baseline from another sweep, for runs made
with ``SKIP_BASELINE=1`` — only valid when both sweeps share the same serving
config.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Optional

# (column header, extractor, aggregation) — extractors return None when the
# metric is absent. "max"/"min" take the best repeat, "mean" averages.
METRICS: list[tuple[str, Any, str]] = [
    ("tok/s", lambda r: r.get("tokens_per_second"), "max"),
    ("wall tok/s", lambda r: (r.get("extra") or {}).get("wall_toks_per_s"), "max"),
    ("latency s", lambda r: r.get("latency_s"), "min"),
    ("accept_len", lambda r: r.get("accept_length"), "mean"),
]
HIGH_CV_THRESHOLD = 5.0      # percent; warn on accept_len CV above this
HIGH_SPREAD_THRESHOLD = 5.0  # percent; warn on tok/s spread above this


def load_rows(sweep_root: Path) -> dict[str, list[dict]]:
    """run_dir (relative to sweep_root) -> raw result rows."""
    configs: dict[str, list[dict]] = {}
    for path in sorted(sweep_root.rglob("results.jsonl")):
        rel = path.parent.relative_to(sweep_root)
        if "baseline" in rel.parts:
            continue
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            configs[str(rel)] = rows
    return configs


def group(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for rec in rows:
        key = (
            rec.get("method", ""),
            rec.get("draft_model") or "",
            rec.get("dataset", ""),
            rec.get("concurrency") or 0,
        )
        groups.setdefault(key, []).append(rec)
    return groups


def baseline_index(rows: list[dict]) -> dict[tuple, float]:
    """(dataset, concurrency) -> best baseline tok/s over its repeats."""
    collected: dict[tuple, list[float]] = {}
    for rec in rows:
        if rec.get("method") != "baseline":
            continue
        tps = rec.get("tokens_per_second")
        if tps:
            key = (rec.get("dataset", ""), rec.get("concurrency") or 0)
            collected.setdefault(key, []).append(float(tps))
    return {k: max(v) for k, v in collected.items()}


def external_baseline(root: Path) -> dict[tuple, float]:
    """(dataset, concurrency) -> best baseline tok/s, from another sweep's tree."""
    collected: dict[tuple, list[float]] = {}
    for path in sorted(root.rglob("results.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("method") != "baseline":
                    continue
                tps = rec.get("tokens_per_second")
                if tps:
                    key = (rec.get("dataset", ""), rec.get("concurrency") or 0)
                    collected.setdefault(key, []).append(float(tps))
    return {k: max(v) for k, v in collected.items()}


def stats(values: list[float]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """(mean, sample std, CV%) — std/CV are None when fewer than two values."""
    if not values:
        return None, None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None, None
    std = statistics.stdev(values)
    return mean, std, (std / mean * 100.0 if mean else None)


def spread(values: list[float]) -> Optional[float]:
    """Relative gap between the worst and best repeat, in percent."""
    if len(values) < 2:
        return None
    best = max(values)
    return (best - min(values)) / best * 100.0 if best else None


def fmt_best(values: list[float], mode: str) -> str:
    """Best repeat, annotated with n and the spread it hides."""
    if not values:
        return "---"
    best = max(values) if mode == "max" else min(values)
    if len(values) < 2:
        return f"{best:.2f}"
    sp = spread(values)
    sp_s = f", spread {sp:.1f}%" if sp is not None else ""
    return f"{best:.2f} (n={len(values)}{sp_s})"


def fmt_mean(values: list[float]) -> str:
    mean, std, cv = stats(values)
    if mean is None:
        return "---"
    if std is None:
        return f"{mean:.2f}"
    cv_s = f" ({cv:.1f}%)" if cv is not None else ""
    return f"{mean:.2f} ±{std:.2f}{cv_s}"


def fmt(values: list[float], mode: str) -> str:
    return fmt_mean(values) if mode == "mean" else fmt_best(values, mode)


def build_markdown(
    configs: dict[str, list[dict]],
    ext_base: Optional[dict[tuple, float]] = None,
    ext_source: str = "",
) -> tuple[str, list[str]]:
    """Returns the markdown body and a list of stdout warning lines."""
    if ext_base is None:
        speedup_note = (
            "`speedup` is the run's own best baseline repeat as denominator."
        )
    else:
        speedup_note = (
            f"`speedup` uses the **external** baseline from `{ext_source}` "
            "(best repeat per (dataset, concurrency)), because this sweep was "
            "run without its own baseline."
        )
    lines = [
        "# Repeat aggregation (speed = best of n, accept_len = mean ±std, CV%)",
        "",
        "> Repeats are separate timed measurements of the same unit inside one "
        "server launch (`--repeats`), with the radix cache flushed between them. "
        "Speed metrics report the **best** repeat (max tok/s, min latency) "
        "because identical replayed work can only be slowed by host "
        "interference, never sped up; `spread` is the relative gap to the worst "
        "repeat, which is what the best-of rule hides. `accept_len` is the mean "
        "over repeats. " + speedup_note,
        "",
    ]
    warnings: list[str] = []
    headers = [h for h, _, _ in METRICS] + ["speedup"]
    header = "| method | dataset | c | n | " + " | ".join(headers) + " |"
    divider = "|---" * (4 + len(headers)) + "|"

    for config in sorted(configs):
        rows = configs[config]
        base = baseline_index(rows)
        lines += [f"## {config}", "", header, divider]
        for key, recs in sorted(group(rows).items()):
            method, _draft, dataset, conc = key
            cells = []
            for name, extract, mode in METRICS:
                values = [float(extract(r)) for r in recs if extract(r) is not None]
                cells.append(fmt(values, mode))
                if name == "tok/s":
                    sp = spread(values)
                    if sp is not None and sp >= HIGH_SPREAD_THRESHOLD:
                        warnings.append(
                            f"  high spread: {config} {method} c={conc} "
                            f"dataset={dataset} tok/s spread={sp:.1f}% over "
                            f"n={len(values)}; best-of hides it, consider re-running"
                        )
                elif name == "accept_len":
                    _, _, cv = stats(values)
                    if cv is not None and cv >= HIGH_CV_THRESHOLD:
                        warnings.append(
                            f"  unstable acceptance: {config} {method} c={conc} "
                            f"dataset={dataset} accept_len CV={cv:.1f}% over "
                            f"n={len(values)}"
                        )
            tps_values = [float(r["tokens_per_second"]) for r in recs
                          if r.get("tokens_per_second")]
            speedup = None
            if method != "baseline" and tps_values:
                b = (ext_base if ext_base is not None else base).get((dataset, conc))
                if b:
                    speedup = max(tps_values) / b
                else:
                    warnings.append(
                        f"  no baseline for {config} {method} "
                        f"dataset={dataset} c={conc}; speedup left empty"
                    )
            cells.append("---" if speedup is None else f"{speedup:.2f}")
            lines.append(
                f"| {method} | {dataset} | {conc} | {len(recs)} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n", warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_root", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--baseline-from",
        type=Path,
        default=None,
        help="borrow baseline rows from another sweep tree (for SKIP_BASELINE runs)",
    )
    args = ap.parse_args()

    if not args.sweep_root.is_dir():
        print(f"sweep root not found: {args.sweep_root}")
        return 1
    configs = load_rows(args.sweep_root)
    if not configs:
        print(f"no results.jsonl found under {args.sweep_root}")
        return 1

    ext_base = None
    if args.baseline_from is not None:
        if not args.baseline_from.is_dir():
            print(f"baseline source not found: {args.baseline_from}")
            return 1
        ext_base = external_baseline(args.baseline_from)
        if not ext_base:
            print(f"no baseline rows found under {args.baseline_from}")
            return 1
        for key in sorted(ext_base):
            print(f"external baseline {key}: {ext_base[key]:.2f} tok/s")

    body, warnings = build_markdown(configs, ext_base, str(args.baseline_from))
    out = args.output or args.sweep_root / "repeats_summary.md"
    out.write_text(body, encoding="utf-8")
    print(f"repeat summary: {out}")
    for line in warnings:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
