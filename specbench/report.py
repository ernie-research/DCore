"""Aggregate BenchResult jsonl into a comparison report.

Comparison rules:
  * Results are grouped by harness. tok/s is ONLY compared within a harness
    (SGLang vs HF-Direct are different inference stacks).
  * ``accept_length`` is harness-independent and shown for all.
  * SGLang speedup is computed by joining each method against the ``baseline``
    method on (dataset, concurrency). HF-Direct rows already carry their own
    per-run ``speedup``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Optional

from specbench.metrics import BenchResult, read_results

_PROFILE_WARMUP_SKIP = 2  # skip records with forward_ct <= this value (warmup)

# --- Repeat aggregation policy ---------------------------------------------
# A repeat re-runs the same unit with the same seed inside one server launch,
# so the work performed is identical round to round and only the wall clock
# differs. Co-tenant processes, thermal throttling and host jitter can only
# make a round slower, never faster, so the best round is the least
# contaminated estimate of a method's speed. Throughput therefore takes the
# best repeat; acceptance length is a property of the drafter rather than of
# the machine, has no systematic direction of contamination, and keeps the
# lower-variance mean.
#
# scripts/aggregate_repeats.py applies the same rule (and its mirror, min, to
# wall-clock latency); the protocol section of the paper states it. Keep all
# three in sync.


def best_tps(values: list[float]) -> float:
    """Throughput of the fastest repeat."""
    return max(values)


def load_profile_medians(profiles_dir: str | Path) -> dict[str, dict[str, float]]:
    """Median of every ``*_ms`` metric per method from ``<profiles_dir>/*.json``.

    Same convention as scripts/profile_to_latex.py: all ranks and batch sizes
    pooled, warmup records dropped.
    """
    stats: dict[str, dict[str, float]] = {}
    for path in sorted(Path(profiles_dir).glob("*.json")):
        values: dict[str, list[float]] = defaultdict(list)
        data = json.loads(path.read_text())
        for rank in data.get("ranks", []):
            for record in rank.get("records", []):
                if record.get("forward_ct", 0) <= _PROFILE_WARMUP_SKIP:
                    continue
                for key, v in record.items():
                    if key.endswith("_ms") and isinstance(v, (int, float)):
                        values[key].append(float(v))
        if values:
            stats[path.stem] = {k: median(v) for k, v in values.items()}
    return stats


def _fmt(v: Optional[float], spec: str) -> str:
    return "N/A" if v is None else format(v, spec)


def build_report(results: list[BenchResult]) -> str:
    by_harness: dict[str, list[BenchResult]] = defaultdict(list)
    for r in results:
        by_harness[r.harness].append(r)

    lines: list[str] = ["# specbench comparison report", ""]
    lines.append("> tok/s is only comparable **within** a harness. "
                 "`accept_length` is comparable across harnesses.")
    lines.append("")

    for harness in sorted(by_harness):
        rows = by_harness[harness]
        lines.append(f"## Harness: `{harness}`")
        lines.append("")

        # baseline tok/s lookup for SGLang speedup join
        baseline_tps: dict[tuple[str, Optional[int]], float] = {}
        for r in rows:
            if r.method == "baseline":
                baseline_tps[(r.dataset, r.concurrency)] = r.tokens_per_second

        lines.append("| method | draft ckpt | dataset | conc/replica | global conc | accept_len | tok/s | speedup |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in sorted(rows, key=lambda x: (x.method, x.draft_model or "", x.dataset, x.concurrency or 0)):
            if r.speedup is not None:
                speedup = r.speedup
            elif r.method != "baseline":
                b = baseline_tps.get((r.dataset, r.concurrency))
                speedup = (r.tokens_per_second / b) if (b and b > 0) else None
            else:
                speedup = None
            ckpt = os.path.basename(r.draft_model.rstrip("/")) if r.draft_model else "-"
            global_concurrency = r.extra.get("global_concurrency") if r.extra else None
            lines.append(
                f"| {r.method} | {ckpt} | {r.dataset} | {_fmt(r.concurrency, 'd') if r.concurrency is not None else '-'} "
                f"| {_fmt(global_concurrency, 'd')} | {_fmt(r.accept_length, '.3f')} "
                f"| {_fmt(r.tokens_per_second, ',.2f')} | {_fmt(speedup, '.3f')} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def build_mixed_breakdown(run_dir: str | Path) -> str:
    """Per-source markdown tables for mixed-traffic runs (e.g. ``specmix``).

    A mixed run reports a single aggregate row, which is token-weighted and
    therefore dominated by the long-form chat/agent sources. Splitting the
    per-sample responses by their originating dataset shows where each method's
    acceptance actually comes from.
    """
    from specbench.accuracy import MIXED_DATASETS
    from specbench.datasets import load_and_process_dataset

    out: list[str] = []
    for ds_name in sorted(MIXED_DATASETS):
        ds_dir = Path(run_dir) / "responses" / ds_name
        files = sorted(ds_dir.glob("*.jsonl")) if ds_dir.is_dir() else []
        if not files:
            continue

        sources = load_and_process_dataset(ds_name)["source"]
        # method -> source -> [completion_tokens, spec_verify_ct, e2e_latency]
        stats: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
        seen: dict[str, set[int]] = defaultdict(set)
        pooled_tokens: dict[str, float] = defaultdict(float)
        for path in files:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    src = sources[rec["sample_index"]]
                    tokens = rec.get("completion_tokens") or 0
                    cell = stats[rec.get("method", "unknown")][src]
                    cell[0] += tokens
                    cell[1] += rec.get("spec_verify_ct") or 0
                    cell[2] += rec.get("e2e_latency") or 0.0
                    seen[src].add(rec["sample_index"])
                    pooled_tokens[src] += tokens

        methods = sorted(stats)
        order = [s for s in _DATASET_ORDER if s in seen]
        order += [s for s in sorted(seen) if s not in order]
        total_tokens = sum(pooled_tokens.values()) or 1.0

        out.append(f"## Mixed-traffic breakdown: `{ds_name}`")
        out.append("")
        out.append("> `reqs` / `tok share` describe the source's weight in the "
                   "mixture (token share pooled over all methods). The `TOTAL` "
                   f"row is token-weighted and matches the `{ds_name}` row above.")
        out.append("")

        for title, note, num, den in (
            ("accept_len", "sum(completion_tokens) / sum(spec_verify_ct)", 0, 1),
            ("per-request output tok/s", "sum(completion_tokens) / sum(e2e_latency)", 0, 2),
        ):
            def _cell(method: str, src: Optional[str]) -> str:
                if src is None:
                    vals = stats[method].values()
                    n, d = sum(v[num] for v in vals), sum(v[den] for v in vals)
                else:
                    v = stats[method].get(src, [0.0, 0.0, 0.0])
                    n, d = v[num], v[den]
                return format(n / d, ".3f") if d > 0 else "-"

            out.append(f"### {title} by source")
            out.append("")
            out.append(f"> {note}")
            out.append("")
            out.append("| source | reqs | tok share | " + " | ".join(methods) + " |")
            out.append("| --- | ---: | ---: | " + " | ".join("---:" for _ in methods) + " |")
            for src in order:
                cells = " | ".join(_cell(m, src) for m in methods)
                out.append(f"| {src} | {len(seen[src])} | "
                           f"{100 * pooled_tokens[src] / total_tokens:.1f}% | {cells} |")
            cells = " | ".join(_cell(m, None) for m in methods)
            out.append(f"| **TOTAL** | {sum(len(v) for v in seen.values())} | "
                       f"100.0% | {cells} |")
            out.append("")

    return "\n".join(out) + "\n" if out else ""


_DATASET_COL_NAME: dict[str, str] = {
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "aime25": "AIME25",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "livecodebench": "LCB",
    "swe-bench": "SWE-Bench",
    "mt-bench": "MT-Bench",
    "alpaca": "Alpaca",
    "arena-hard": "Arena-Hard",
}

_DATASET_ORDER = list(_DATASET_COL_NAME.keys())

# Benchmarks grouped by how predictable the target model's continuations are.
# Must stay in sync with the paper's main table
# (Table 1).
_LOW_ENTROPY = ["gsm8k", "math500", "humaneval", "mbpp", "livecodebench"]
_HIGH_ENTROPY = ["aime25", "swe-bench", "mt-bench", "alpaca", "arena-hard"]

# Run key -> row label, in the order the paper lists them. Methods missing here
# are appended alphabetically under their raw key.
_METHOD_ROW_NAME: dict[str, str] = {
    "baseline": "Baseline",
    "eagle3": "Eagle-3",
    "eagle3_origin": "Eagle-3(orig)",
    "eagle3_tree": "Eagle-3(tree)",
    "eagle3_origin_tree": "Eagle-3(orig,tree)",
    "eagle3_3layer": "Eagle-3-3L",
    "eagle3_3l_tree": "Eagle-3-3L(tree)",
    "eagle3_kl_tree": "Eagle-3+KL(tree)",
    "eagle3_ce_tree": "Eagle-3+CE(tree)",
    "eagle3_dpace_tree": "Eagle-3+DPace(tree)",
    "dflash": "DFlash",
    "dflash_3l": "DFlash-3L",
    "dflash_7l": "DFlash-7L",
    "dflash_origin": "DFlash(orig)",
    "dflash_anchor": "DFlash-Anchor",
    "dflare": "DFlare",
    "dflare_5hs_5l": "DFlare-5HS",
    "dflare_9hs_7l": "DFlare-7L",
    "dflare_origin": "DFlare-7L(orig)",
    "dflare_dpace": "DFlare+DPace",
    "dflare_9hs": "DFlare-9HS",
    "dflare_dpace_9hs": "DFlare+DPace-9HS",
    "dflare_dpace_9hs_anchor": "DFlare+DPace-9HS-Anchor",
    "dflare_dpace_9hs_b300": "DFlare+DPace-9HS(B300)",
    "dflare_dpace_9hs_anchor_b300": "DFlare+DPace-9HS-Anchor(B300)",
    "dflare_kltv_9hs_b300": "DFlare+KLTV-9HS(B300)",
    "dflare_kltv_9hs_anchor_b300": "DFlare+KLTV-9HS-Anchor(B300)",
    "dcore": "DCore",
    "dpace": "DPace",
    "domino": "Domino",
    "domino_origin": "Domino(orig)",
    "domino_pool2048": "Domino-Pool2048",
    "domino_without_curriculum": "Domino(no-curr)",
    "domino_without_curriculum_dpace": "Domino+DPace(no-curr)",
    "dspark_markov": "DSpark-Markov",
    "dspark_origin": "DSpark(orig)",
    "dspark_markov_origin": "DSpark-Markov(orig)",
    "dspark_rnn": "DSpark-RNN",
}
_METHOD_ORDER = list(_METHOD_ROW_NAME)

# Shaded Avg column. Inlined rather than using the paper's \newcolumntype{A}
# shorthand so the emitted table needs no preamble beyond the packages listed
# in the generated comment header.
_AVG_COL = ">{\\columncolor{avgbg}}c"


def load_baseline_tps(run_dir: str | Path) -> dict[str, float]:
    """``dataset -> baseline tok/s`` from a baseline run dir's results.jsonl.

    Repeated measurements of the same dataset collapse to the best repeat,
    matching the per-cell aggregation in :func:`build_latex_table` so that a
    speedup is best-over-best rather than best-over-mean.

    Returns an empty dict when the file or the ``baseline`` rows are missing,
    which makes :func:`build_latex_table` fall back to absolute throughput.
    """
    path = Path(run_dir) / "results.jsonl"
    if not path.is_file():
        return {}
    per_ds: dict[str, list[float]] = defaultdict(list)
    for r in read_results(path):
        if r.method == "baseline" and r.tokens_per_second:
            per_ds[r.dataset].append(float(r.tokens_per_second))
    return {ds: best_tps(v) for ds, v in per_ds.items()}


def _entropy_groups(present: list[str]) -> list[tuple[str, list[str]]]:
    """Split the present datasets into the paper's two entropy groups."""
    groups = []
    for label, members in (("Low-Entropy", _LOW_ENTROPY),
                           ("High-Entropy", _HIGH_ENTROPY)):
        cols = [d for d in members if d in present]
        if cols:
            groups.append((label, cols))
    other = [d for d in present
             if d not in _LOW_ENTROPY and d not in _HIGH_ENTROPY]
    if other:
        # "Other" only makes sense next to the entropy groups; on its own (e.g.
        # a specmix-only run) it is just the benchmark list.
        groups.append(("Other" if groups else "Benchmarks", other))
    return groups


def _caption(label: str, groups: list[tuple[str, list[str]]],
             ratio_mode: bool, agg_name: str, n_repeats: int = 1) -> str:
    """Caption mirroring the paper's main table."""
    tex_label = label.replace("_", "\\_").replace("%", "\\%")
    n = sum(len(cols) for _, cols in groups)
    parts = [
        "\\caption{\\textbf{%s.} Decoding throughput (TPS) and acceptance "
        "length $\\tau$ over %d benchmark%s" % (tex_label, n, "" if n == 1 else "s"),
    ]
    if len(groups) > 1:
        named = ", ".join(
            "\\textsc{%s} (%s)" % (t, ", ".join(_DATASET_COL_NAME.get(d, d)
                                                for d in cols))
            for t, cols in groups)
        parts.append(", grouped into %s" % named)
    parts.append(". ")
    if ratio_mode:
        parts.append(
            "The \\textsc{Baseline} row reports absolute throughput in "
            "tokens/s; every other row reports its speedup ($\\times$) over "
            "the \\textsc{Baseline} throughput measured in the same column. ")
    else:
        parts.append(
            "Every cell reports absolute throughput in tokens/s: no "
            "\\textsc{Baseline} covering all benchmarks was available for "
            "this run, so speedups are not shown. ")
    if n_repeats > 1:
        parts.append(
            "Every unit was measured %d times; TPS is the \\emph{best} of "
            "those repeats and $\\tau$ their mean. Repeats replay identical "
            "work, so a slower round reflects host interference rather than "
            "the method. " % n_repeats)
    if any(len(cols) > 1 for _, cols in groups):
        parts.append(
            "\\textsc{Avg} (shaded) is the %s over the benchmarks of its own "
            "group." % agg_name)
    return "".join(parts).rstrip() + "}"


def build_latex_table(
    results: list[BenchResult],
    average: str = "weighted",
    profile_stats: Optional[dict[str, dict[str, float]]] = None,
    baseline_tps: Optional[dict[str, float]] = None,
    run_label: Optional[str] = None,
) -> str:
    """Generate the paper's main-table LaTeX float for one run dir.

    Layout matches the paper's main table (Table 1):
    rows = methods, columns = benchmarks (TPS, tau) grouped into Low-Entropy and
    High-Entropy, each group closed by a shaded per-group Avg pair.

    The ``Baseline`` row carries absolute tok/s; every other row carries its
    speedup over the Baseline of the same column. The Baseline is taken from the
    ``baseline`` rows of ``results`` when present, otherwise from
    ``baseline_tps`` (see :func:`load_baseline_tps`). With neither, every cell
    falls back to absolute tok/s and the emitted comment header says so.

    ``average="weighted"`` weights each benchmark of a group by its
    ``num_samples``; ``average="mean"`` gives every benchmark equal weight.
    Repeated measurements of the same (method, benchmark) — the ``repeat``
    field, produced by ``run --repeats N`` — collapse to one cell using the
    policy documented at :func:`best_tps`: TPS takes the best repeat, $\\tau$
    their mean. Cells carry no best/runner-up decoration.

    When ``profile_stats`` is given (see :func:`load_profile_medians`), a
    trailing Latency group with $T_v$ (target verify) and $T_d$ (draft) is
    appended. Those come from per-forward profiling records, not from repeats,
    so they are medians and the best-repeat rule does not apply. Latency cells
    are not decorated.
    """
    if average not in ("weighted", "mean"):
        raise ValueError(f"unknown average mode: {average}")
    # Group by method -> dataset -> metrics. A (method, dataset) unit measured
    # with --repeats N contributes N rows (see BenchResult.repeat).
    samples: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"tps": [], "al": []}))
    # Track num_samples per dataset for weighted average
    dataset_samples: dict[str, int] = {}

    for r in results:
        cell = samples[r.method][r.dataset]
        if r.tokens_per_second is not None:
            cell["tps"].append(float(r.tokens_per_second))
        if r.accept_length is not None:
            cell["al"].append(float(r.accept_length))
        if r.num_samples > 0:
            dataset_samples[r.dataset] = r.num_samples

    n_repeats = 1
    method_data: dict[str, dict[str, dict[str, float | None]]] = defaultdict(dict)
    for m, per_ds in samples.items():
        for ds, cell in per_ds.items():
            n_repeats = max(n_repeats, len(cell["tps"]))
            method_data[m][ds] = {
                "tps": best_tps(cell["tps"]) if cell["tps"] else None,
                "al": fmean(cell["al"]) if cell["al"] else None,
            }

    # Determine which datasets are present (preserving canonical order)
    present_datasets = [d for d in _DATASET_ORDER if d in dataset_samples]
    # Also include datasets not in canonical list
    for d in sorted(dataset_samples.keys()):
        if d not in present_datasets:
            present_datasets.append(d)

    groups = _entropy_groups(present_datasets)

    # Baseline denominators: prefer the ones measured in this very run.
    base = {d: v for d, v in (baseline_tps or {}).items() if d in present_datasets}
    for ds, info in method_data.get("baseline", {}).items():
        if info.get("tps"):
            base[ds] = info["tps"]
    ratio_mode = bool(base) and all(d in base for d in present_datasets)

    # Method rows: Baseline first, then the canonical order, then the rest.
    known = [m for m in _METHOD_ORDER if m in method_data]
    methods = known + sorted(m for m in method_data if m not in _METHOD_ROW_NAME)
    if "baseline" in methods:
        methods.remove("baseline")
    spec_methods = list(methods)
    show_baseline = "baseline" in method_data or bool(base)
    all_rows = (["baseline"] if show_baseline else []) + spec_methods

    def _weight(ds: str) -> float:
        ns = dataset_samples.get(ds, 0)
        if ns <= 0:
            return 0.0
        return float(ns) if average == "weighted" else 1.0

    def _group_mean(per_ds: dict[str, float | None], cols: list[str]) -> float | None:
        num = tot = 0.0
        for ds in cols:
            v, w = per_ds.get(ds), _weight(ds)
            if v is not None and w > 0:
                num += v * w
                tot += w
        return (num / tot) if tot > 0 else None

    # Absolute values throughout; ratios are formed at print time so that a
    # group's speedup is mean(method) / mean(baseline), not mean of ratios.
    value_grid: dict[str, dict[str, float | None]] = defaultdict(dict)
    for m in all_rows:
        src = method_data.get(m, {})
        tps = {ds: (src.get(ds) or {}).get("tps") for ds in present_datasets}
        al = {ds: (src.get(ds) or {}).get("al") for ds in present_datasets}
        if m == "baseline":
            # A borrowed baseline (--baseline-dir) has no rows in this results
            # file; fall back to the denominators we were handed.
            tps = {ds: (tps.get(ds) if tps.get(ds) is not None else base.get(ds))
                   for ds in present_datasets}
        for ds in present_datasets:
            value_grid[m][f"{ds}_tps"] = tps[ds]
            value_grid[m][f"{ds}_al"] = al[ds]
        for gi, (_, cols) in enumerate(groups):
            value_grid[m][f"g{gi}_tps"] = _group_mean(tps, cols)
            value_grid[m][f"g{gi}_al"] = _group_mean(al, cols)

    # Denominator of every TPS column (None -> print the absolute value).
    denom: dict[str, float | None] = {}
    if ratio_mode:
        for ds in present_datasets:
            denom[f"{ds}_tps"] = base[ds]
        for gi, (_, cols) in enumerate(groups):
            denom[f"g{gi}_tps"] = _group_mean(base, cols)

    def _tps_cell(method: str, col_key: str) -> str:
        v = value_grid[method].get(col_key)
        if v is None:
            return "--"
        d = denom.get(col_key)
        if d is None or method == "baseline":
            return format(v, ".1f")
        return f"{v / d:.2f}$\\times$"

    def _al_cell(method: str, col_key: str) -> str:
        if method == "baseline":
            return "1"          # no drafter, so one accepted token per step
        v = value_grid[method].get(col_key)
        if v is None:
            return "--"
        return format(v, ".2f")

    # ---- LaTeX emission ---------------------------------------------------
    # Column layout: Method, then per group (2 per benchmark + a shaded Avg
    # pair), then an optional Latency pair. A group of one benchmark gets no
    # Avg, since it would just duplicate that benchmark's cells. A vertical
    # rule closes every group except the table's last column block.
    show_avg = [len(cols) > 1 for _, cols in groups]
    spans: list[tuple[int, int]] = []   # 1-based benchmark span of each group
    col = 2
    for gi, (_, cols) in enumerate(groups):
        spans.append((col, col + 2 * len(cols) - 1))
        col += 2 * len(cols) + (2 if show_avg[gi] else 0)
    n_cols = col - 1 + (2 if profile_stats else 0)

    def _rule(gi: int) -> str:
        """The `|` that closes group gi, empty for the table's last block."""
        return "" if (gi == len(groups) - 1) and not profile_stats else "|"

    parts = ["l"]
    for gi, (_, cols) in enumerate(groups):
        if show_avg[gi]:
            parts.append("c" * (2 * len(cols)))
            parts.append(_AVG_COL + " " + _AVG_COL + _rule(gi))
        else:
            parts.append("c" * (2 * len(cols)) + _rule(gi))
    if profile_stats:
        parts.append("cc")
    col_spec = " ".join(parts)

    def _avg_head(gi: int, body: str) -> str:
        return "\\multicolumn{2}{%s%s}{%s}" % (_AVG_COL, _rule(gi), body)

    def _latency(method: str) -> tuple[Optional[float], Optional[float]]:
        """(target verify ms, draft ms) for a method, or (None, None)."""
        st = (profile_stats or {}).get(method)
        if not st:
            return None, None
        verify = st.get("target_verify_gpu_ms")
        draft = st.get("draft_gpu_ms")
        if draft is None:
            step = st.get("step_cpu_ms")
            draft = (
                (step - verify) if (verify is not None and step is not None) else None
            )
        return verify, draft

    # Caption/label identify the run by its sweep-relative path; the absolute
    # prefix is noise and makes the \label unreadably long.
    run_path = (run_label or "run").strip("/")
    label = run_path
    for prefix in ("cache/sglang_bench_results/", "benchmark_results/specbench/"):
        _, sep, tail = label.partition(prefix)
        if sep:
            label = tail
            break
    slug = "".join(c if c.isalnum() else "_" for c in label).strip("_")
    agg_name = ("sample-count-weighted mean" if average == "weighted"
                else "unweighted mean")
    lines: list[str] = []
    lines.append("% " + "-" * 73)
    lines.append("% Generated by: python -m specbench.cli report "
                 f"--run-dir {run_path}")
    lines.append("% Do not edit by hand -- rerun the command instead.")
    lines.append("% Requires in the preamble: booktabs, multirow, graphicx,")
    lines.append("% colortbl, and \\definecolor{avgbg}{RGB}{232,240,250}.")
    if ratio_mode:
        lines.append("% TPS = speedup over the Baseline of the same column; the")
        lines.append("% Baseline row itself is absolute tok/s.")
    else:
        lines.append("% WARNING: no Baseline covering every benchmark was found,")
        lines.append("% so ALL TPS cells are absolute tok/s, NOT speedups. Pass")
        lines.append("% --baseline-dir <baseline run dir> to get speedups.")
    if n_repeats > 1:
        lines.append(f"% Repeat aggregation over n={n_repeats}: TPS is the BEST")
        lines.append("% repeat (max tok/s), tau is the MEAN of the repeats.")
    lines.append("% " + "-" * 73)
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(_caption(label, groups, ratio_mode, agg_name, n_repeats))
    lines.append(f"\\label{{tab:{slug}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row 1: group titles + the Avg labels, which span rows 1 and 2.
    hdr1 = [""]
    for gi, (title, cols) in enumerate(groups):
        # Without an Avg pair, the group title cell is what has to carry the
        # vertical rule closing the group.
        spec = "c" if show_avg[gi] else "c" + _rule(gi)
        hdr1.append("\\multicolumn{%d}{%s}{\\textsc{%s}}" % (2 * len(cols), spec, title))
        if show_avg[gi]:
            hdr1.append(_avg_head(gi, "\\multirow{2}{*}{\\textsc{Avg}}"))
    if profile_stats:
        hdr1.append("\\multicolumn{2}{c}{Latency (ms)}")
    lines.append(" & ".join(hdr1) + " \\\\")
    cmi = [f"\\cmidrule(lr){{{a}-{b}}}" for a, b in spans]
    if profile_stats:
        cmi.append(f"\\cmidrule(lr){{{n_cols - 1}-{n_cols}}}")
    lines.append(" ".join(cmi))

    # Header row 2: benchmark names; the Avg cells stay empty (spanned above).
    hdr2 = ["Method"]
    for gi, (_, cols) in enumerate(groups):
        for j, ds in enumerate(cols):
            last_in_group = (not show_avg[gi]) and j == len(cols) - 1
            spec = "c" + (_rule(gi) if last_in_group else "")
            hdr2.append("\\multicolumn{2}{%s}{%s}" % (spec, _DATASET_COL_NAME.get(ds, ds)))
        if show_avg[gi]:
            hdr2.append(_avg_head(gi, ""))
    if profile_stats:
        hdr2.append("\\multicolumn{2}{c}{}")
    lines.append(" & ".join(hdr2) + " \\\\")

    # Header row 3: the metric of each sub-column.
    hdr3 = [""]
    for gi, (_, cols) in enumerate(groups):
        hdr3 += ["TPS", "$\\tau$"] * (len(cols) + (1 if show_avg[gi] else 0))
    if profile_stats:
        hdr3 += ["$T_{v}$", "$T_{d}$"]
    lines.append(" & ".join(hdr3) + " \\\\")
    lines.append("\\midrule")

    def _row(m: str) -> str:
        cells = [_METHOD_ROW_NAME.get(m, m.replace("_", "\\_"))]
        for gi, (_, cols) in enumerate(groups):
            for ds in cols:
                cells.append(_tps_cell(m, f"{ds}_tps"))
                cells.append(_al_cell(m, f"{ds}_al"))
            if show_avg[gi]:
                cells.append(_tps_cell(m, f"g{gi}_tps"))
                cells.append(_al_cell(m, f"g{gi}_al"))
        if profile_stats:
            verify, draft = _latency(m)
            cells.append("--" if verify is None else format(verify, ".2f"))
            cells.append("--" if draft is None else format(draft, ".2f"))
        return " & ".join(cells) + " \\\\"

    if show_baseline:
        lines.append(_row("baseline"))
        if spec_methods:
            lines.append("\\midrule")
    lines += [_row(m) for m in spec_methods]

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table*}")

    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="specbench report generator")
    parser.add_argument("--input", required=True, help="results jsonl (from `specbench run`)")
    parser.add_argument("--output-md", default=None, help="write markdown report here")
    args = parser.parse_args()

    results = read_results(args.input)
    md = build_report(results)
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(md, encoding="utf-8")
        print(f"Wrote report to {args.output_md}")
    else:
        print(md)


if __name__ == "__main__":
    main()
