"""Emit the p/q entropy heatmap figures from the speculative p/q dumps.

Six panels (2 verification temperatures x 3 methods). Each panel draws the
B=16 triangle with the B=8 triangle nested in its blank upper-right corner, a
marginal bar strip giving the share of steps at each acceptance length, and one
shared viridis scale across all twelve grids.

Rows are keyed by nc (accepted DRAFT tokens) but LABELLED nc+1, because every
step also emits one bonus token: nc=0 still delivers one token, and
nc=num_draft delivers the full budget B.

Two variants, selected with --variant:
  qwen3-4b  reads the analyzer's pq_entropy_paths.csv for the nc >= 1 rows and
            rescans the raw dumps only for the nc = 0 row and the share strips.
  qwen3-8b  has no analyzer CSV, so every cell comes from a full scan of the
            raw dumps pooled over all measured concurrencies (336 files, 85 GiB).
"""

import argparse
import csv
import json
import math
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

METHODS = [("eagle3", "Eagle-3"), ("dflash", "DFlash"), ("dspark_markov", "DSpark")]
# (directory, num_draft). num_draft + 1 == the verification budget B.
BUDGETS = [("b16", 15), ("b8", 7)]
NUM_LEVELS = 101

# --- geometry, in units of one B=16 cell -----------------------------------
CELL = 7.6  # pt
STRIP_W = 2.2  # marginal share strip, to the right of each B=16 grid
STRIP_GAP = 0.4
PANEL_GAP_X = 1.5
PANEL_GAP_Y = 1.5
BAR_H = 0.72  # share bar thickness, in rows
# The blank corner of a B=16 panel is exactly rows nc=0..7 x cols i=8..14, the
# 7-wide 8-tall block [8,15]x[8,16] -- which is the smallest box an 8x7 B=8
# grid fits in. Nothing about this placement is free.
INSET_SCALE = 0.68
INSET_X0, INSET_Y0 = 8.35, 8.50
INSET_STRIP_X = 7.3  # in inset cell units
INSET_STRIP_W = 1.6

N16_COL, N16_ROW = 15, 16  # cols = num_draft, rows = num_draft + 1 (nc = 0..15)
N8_COL, N8_ROW = 7, 8
PANEL_W = N16_COL + STRIP_GAP + STRIP_W
PANEL_PITCH_X = PANEL_W + PANEL_GAP_X
PANEL_PITCH_Y = N16_ROW + PANEL_GAP_Y

VARIANTS = {
    "qwen3-4b": {
        "data": ROOT / "outputs/eval/qwen3-4b/20260907_draft_entropy",
        "out": ROOT / "outputs/figures/pq_entropy_heatmap.tex",
        "cache": "nc0_first_slot_scan.json",
        "source": "csv",
        "regen_args": "",
        "temps": [("t0/dt00", r"$T{=}0$"), ("t1rs/dt10", r"$T{=}1$, r.s.")],
        "conc": {"t0/dt00": ["c64"], "t1rs/dt10": ["c64"]},
        # full strip length; the largest observed share is 30.8%
        "share_max": 0.32,
        "hdr_setting": "Qwen3-4B, SpecBench, C = 64",
        "hdr_provenance": [
            "% PROVENANCE: outputs/eval/qwen3-4b/20260907_draft_entropy/b{8,16}/",
            "% {t0/dt00,t1rs/dt10}/c64/pq/pq_entropy_paths.csv for the n_c >= 1 rows,",
            "% produced by eval/analyze_pq_cluster_bootstrap.py; that script aggregates",
            "% only n_c >= 1, so the n_c = 0 row and the share strips are recomputed by",
            "% the generator straight from the raw p/q dumps (the i = 0 records, one per",
            "% step) written by sglang's spec_pq_recorder, using the same two-stage mean.",
            "% Draft temperature is 0 in the T = 0 row and 1 in the T = 1 row.",
        ],
        "hdr_scale_note": [
            f"% levels are baked in as vir0..vir{NUM_LEVELS - 1}. Adding the n_c = 0 row",
            "% pushed the upper end from 0.70 to 0.95, which flattens contrast in the",
            "% low-entropy interior; that is the price of one comparable scale.",
        ],
        "hdr_tau_note": [
            "% micro-average over recorded steps at C = 64 -- NOT the per-benchmark tau",
            "% of the result tables, do not quote one against the other):",
        ],
        "cap_setting": r"Qwen3-4B, SpecBench, $C{=}64$",
        "cap_tail": [
            "    Entropy is always measured on the raw $T{=}1$ distribution; the row",
            "    label is the verification temperature.}",
        ],
        "label": "fig:pq_entropy_heatmap_qwen3_4b",
    },
    "qwen3-8b": {
        "data": ROOT / ("outputs/eval/qwen3-8b/"
                        "20260907_specmix_concurrency_with_profile"),
        "out": ROOT / "outputs/figures/pq_entropy_heatmap_qwen3_8b.tex",
        "cache": "pq_entropy_full_scan.json",
        "source": "scan",
        "regen_args": " --variant qwen3-8b",
        # Both rows draft at T = 0; only the verification temperature moves.
        # b16/t1rs/dt10 has no eagle3 dumps, so the hot-draft config cannot
        # supply a complete row and is not used here.
        "temps": [("t0/dt00", r"$T{=}0$"), ("t1rs/dt00", r"$T{=}1$, r.s.")],
        "conc": {"t0/dt00": ["c1", "c2", "c64"],
                 "t1rs/dt00": ["c1", "c2", "c4", "c64"]},
        "share_max": 0.36,
        "hdr_setting": "Qwen3-8B, SpecBench, all concurrencies pooled",
        "hdr_provenance": [
            "% PROVENANCE: outputs/eval/qwen3-8b/",
            "% 20260907_specmix_concurrency_with_profile/b{8,16}/{t0/dt00,t1rs/dt00}/",
            "% c*/pq/<method>/*.jsonl -- the raw p/q dumps written by sglang's",
            "% spec_pq_recorder, 8 files per config (one per dp rank). No analyzer CSV",
            "% exists for this model, so the generator scans every dump line itself",
            "% (336 files, 85 GiB) and reproduces the aggregation of",
            "% eval/analyze_pq_cluster_bootstrap.py exactly: a cell takes the i <= n_c",
            "% records, averages within a request id, then averages those per-request",
            "% means. Unlike the Qwen3-4B figure every row including n_c = 0 comes out",
            "% of that one scan (the analyzer skips n_c = 0). ALL usable dumps are",
            "% counted: C = 1, 2, 64 at T = 0 and C = 1, 2, 4, 64 at T = 1, pooled --",
            "% concurrency changes batch composition, not the conditional entropy a",
            "% cell measures. Request ids are clustered within a dump file (a request",
            "% is served by one dp rank) and are never pooled across configurations.",
            "% b16/t1rs/dt10 is not used: it has no eagle3 dumps. b60 has no p/q dumps",
            "% at all. Draft temperature is 0 in BOTH rows -- only the verification",
            "% temperature changes, so the two rows isolate rejection sampling at the",
            "% target from any change in the draft itself.",
        ],
        "hdr_scale_note": [
            f"% levels are baked in as vir0..vir{NUM_LEVELS - 1}. The limits are fitted",
            "% to this figure's own data, so cells are comparable within the figure but",
            "% NOT against the Qwen3-4B figure's colourbar.",
        ],
        "hdr_tau_note": [
            "% micro-average over every recorded step pooled across concurrencies --",
            "% NOT the per-benchmark tau of the result tables, do not quote one against",
            "% the other):",
        ],
        "cap_setting": r"Qwen3-8B, SpecBench, all measured concurrencies pooled",
        "cap_tail": [
            "    Entropy is always measured on the raw $T{=}1$ distribution; the row",
            "    label is the \\emph{verification} temperature, and the draft is sampled",
            "    at $T{=}0$ in both rows, so the contrast between the rows isolates",
            "    rejection sampling at the target. Every available dump is pooled --",
            "    $C \\in \\{1,2,64\\}$ at $T{=}0$ and $C \\in \\{1,2,4,64\\}$ at $T{=}1$,",
            "    more than $10^6$ recorded steps per panel.}",
        ],
        "label": "fig:pq_entropy_heatmap_qwen3_8b",
    },
}
V = VARIANTS["qwen3-4b"]  # rebound by main()


def share_gridlines() -> list[float]:
    """Faint 10% ticks inside the strip, all that fit under share_max."""
    return [0.1 * k for k in range(1, int(V["share_max"] / 0.1) + 1)]


def read_path_csv(budget_dir: str, temp_dir: str, method: str,
                  num_draft: int) -> np.ndarray:
    """Grid[nc, i] of mean target entropy, from the nc >= 1 rows of the CSV.

    The analyzer only aggregates nc >= 1, so row 0 comes back all-NaN and is
    filled in later from the raw dumps by scan_first_slot().
    """
    conc = V["conc"][temp_dir][0]
    csv_path = (V["data"] / budget_dir / temp_dir / conc
                / "pq/pq_entropy_paths.csv")
    grid = np.full((num_draft + 1, num_draft), np.nan)
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != method or not row["target_entropy"]:
                continue
            grid[int(row["nc"]), int(row["position"])] = float(row["target_entropy"])
    return grid


def scan_first_slot(budget_dir: str, temp_dir: str, method: str,
                    num_draft: int) -> dict:
    """One pass over the raw dumps, keeping only the i == 0 records.

    Every speculative step emits exactly one i == 0 record, so those records
    are a 1:1 sample of steps: their nc histogram IS the acceptance-length
    distribution, and the nc == 0 subset is the only data for grid row 0
    (at nc = 0 the sole non-counterfactual slot is i = 0). The entropy mean is
    the analyzer's two-stage mean -- within a rid first, then across rids.
    """
    conc = V["conc"][temp_dir][0]
    method_dir = V["data"] / budget_dir / temp_dir / conc / "pq" / method
    hist = [0] * (num_draft + 1)
    per_rid: dict[str, list[float]] = {}
    for path in sorted(method_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if '"i":0,' not in line:
                    continue
                row = json.loads(line)
                if int(row["i"]) != 0:
                    continue
                nc = int(row["nc"])
                hist[nc] += 1
                if nc == 0:
                    cell = per_rid.setdefault(str(row["rid"]), [0.0, 0.0])
                    cell[0] += float(row["ent"])
                    cell[1] += 1.0
    means = [s / n for s, n in per_rid.values() if n > 0]
    return {"hist": hist, "nc0_entropy": sum(means) / len(means), "rids": len(means)}


def scan_dump_file(task: tuple) -> tuple:
    """Full first stage of the two-stage mean for one dump file.

    The dumps are ~83 GB of JSONL with a fixed key order, so the four fields
    that matter are sliced out by hand; json.loads costs about 6x more. Request
    ids are unique within a run and a request is served by exactly one dp rank,
    so a rid never spans files: the per-rid means can be reduced here and the
    parent only has to sum (sum of rid means, number of rids) per cell.
    """
    key, path, num_draft, size = task
    n_row, n_col = num_draft + 1, num_draft
    hist = [0] * n_row
    per_rid: dict[bytes, list] = {}
    with open(path, "rb") as handle:
        for line in handle:
            if len(line) < 32:
                continue
            a = line.index(b'"rid":"') + 7
            b = line.index(b'"', a)
            rid = line[a:b]
            a = line.index(b'"i":', b) + 4
            b = line.index(b",", a)
            i = int(line[a:b])
            a = line.index(b'"nc":', b) + 5
            b = line.index(b",", a)
            nc = int(line[a:b])
            if i == 0:
                hist[nc] += 1  # exactly one i = 0 record per step
            if i > nc:
                continue  # past the frontier: counterfactual, never aggregated
            a = line.index(b'"ent":', b) + 6
            b = line.index(b",", a)
            cell = per_rid.get(rid)
            if cell is None:
                cell = per_rid[rid] = [np.zeros((n_row, n_col)),
                                       np.zeros((n_row, n_col))]
            cell[0][nc, i] += float(line[a:b])
            cell[1][nc, i] += 1.0
    sums = np.zeros((n_row, n_col))
    rids = np.zeros((n_row, n_col))
    for total, count in per_rid.values():
        seen = count > 0
        sums[seen] += total[seen] / count[seen]
        rids[seen] += 1.0
    return key, hist, sums, rids, len(per_rid), size


def scan_dumps(pending: list[tuple], workers: int) -> dict:
    """Scan every dump of every pending (budget, temp, method) group."""
    tasks: list[tuple] = []
    accum: dict[str, dict] = {}
    for key, budget_dir, temp_dir, method, num_draft in pending:
        accum[key] = {"hist": [0] * (num_draft + 1),
                      "sums": np.zeros((num_draft + 1, num_draft)),
                      "rids": np.zeros((num_draft + 1, num_draft)),
                      "n_rid": 0}
        for conc in V["conc"][temp_dir]:
            method_dir = V["data"] / budget_dir / temp_dir / conc / "pq" / method
            paths = sorted(method_dir.glob("*.jsonl"))
            if not paths:
                raise SystemExit(f"no dumps under {method_dir}")
            tasks.extend((key, str(p), num_draft, p.stat().st_size)
                         for p in paths)
    # Biggest first: files differ 8x in size, and that is most of the balance.
    tasks.sort(key=lambda t: -t[3])
    total = sum(t[3] for t in tasks)
    done = 0
    print(f"  scanning {len(tasks)} files, {total / 2**30:.1f} GiB")
    with Pool(min(workers, len(tasks))) as pool:
        for key, hist, sums, rids, n_rid, size in pool.imap_unordered(
                scan_dump_file, tasks):
            acc = accum[key]
            for nc, count in enumerate(hist):
                acc["hist"][nc] += count
            acc["sums"] += sums
            acc["rids"] += rids
            acc["n_rid"] += n_rid
            done += size
            print(f"\r  {done / 2**30:6.1f}/{total / 2**30:.1f} GiB",
                  end="", flush=True)
    print()
    out = {}
    for key, acc in accum.items():
        rids = acc["rids"]
        grid = np.where(rids > 0, acc["sums"] / np.maximum(rids, 1.0), np.nan)
        out[key] = {
            "hist": acc["hist"],
            # None, not NaN: bare NaN is not valid JSON.
            "grid": [[None if math.isnan(v) else v for v in row]
                     for row in grid.tolist()],
            "rid_counts": [[int(c) for c in row] for row in rids.tolist()],
            "rids": acc["n_rid"],
        }
        print(f"  {key}: N={sum(acc['hist'])} rids={acc['n_rid']} "
              f"cells={int((rids > 0).sum())}")
    return out


def load_scans(workers: int) -> dict:
    cache = V["data"] / V["cache"]
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    keys = [(f"{budget_dir}|{temp_dir}|{method}",
             budget_dir, temp_dir, method, num_draft)
            for budget_dir, num_draft in BUDGETS
            for temp_dir, _ in V["temps"]
            for method, _ in METHODS]
    pending = [k for k in keys if k[0] not in cached]
    if not pending:
        return cached
    if V["source"] == "csv":
        for key, budget_dir, temp_dir, method, num_draft in pending:
            cached[key] = scan_first_slot(
                budget_dir, temp_dir, method, num_draft)
            print(f"  scanned {key}: N={sum(cached[key]['hist'])}")
        cache.write_text(json.dumps(cached, indent=1))
    else:
        cached.update(scan_dumps(pending, workers))
        cache.write_text(json.dumps(cached))
    return cached


def emit_header(e, lo, hi, vmin, vmax, share_peak, totals, shares, tau) -> None:
    grid = share_gridlines()
    gap_pt = (V["share_max"] - grid[-1]) / V["share_max"] * STRIP_W * CELL
    e("% " + "-" * 73)
    e("% AUTO-GENERATED. Regenerate with, from the repository root,")
    e(f"%   python eval/make_pq_entropy_heatmap_tikz.py{V['regen_args']}")
    e("% Hand edits to the \\fill list will be lost; change the generator instead.")
    e("% Figure: target entropy on the realized proposal path, conditioned on the")
    e(f"% accepted length ({V['hdr_setting']}).")
    e("%")
    e("% WHAT ONE CELL IS. Row n_c, column i holds the mean entropy (nats) of the")
    e("% target's next-token distribution at chain slot i, averaged over every")
    e("% speculative step whose verification accepted exactly n_c DRAFTED tokens.")
    e("% Only i <= n_c is drawn: slots past the frontier condition on a token that")
    e("% verification rejected, so they are counterfactual and are dropped. That")
    e("% restriction is what makes each panel a lower-left triangle. Averaging is")
    e("% two-stage -- within a request first, then across requests -- so long")
    e("% generations cannot dominate.")
    e("%")
    e("% ROWS ARE KEYED BY n_c BUT LABELLED n_c+1. Every step emits n_c accepted")
    e("% drafts PLUS one bonus token drawn at the frontier, and that bonus is")
    e("% unconditionally kept. So n_c = 0 is not a wasted step -- it still emits")
    e("% one token -- and n_c+1 is the step's token yield. The top row is n_c = 0")
    e("% (label 1, a single cell at i = 0) and the bottom row is n_c = B-1")
    e("% (label B, the whole budget accepted).")
    e("%")
    e("% i = n_c IS THE FRONTIER: the first slot verification did NOT accept, and")
    e("% the distribution the bonus token is drawn from. Those cells carry a black")
    e("% border. The bottom row (n_c = B-1) has none: every draft was accepted, so")
    e("% the chain has no rejected slot and the step yields exactly B tokens.")
    e("%")
    e("% SHARE STRIP. The grey bars to the right of each grid are the marginal")
    e("% distribution of n_c: bar row n_c is the fraction of recorded steps that")
    e("% accepted exactly n_c drafts. Both budgets use the SAME strip scale")
    e(f"% (0 .. {V['share_max']:.0%}, faint gridlines at "
      f"{'/'.join(f'{g:.0%}'[:-1] for g in grid)}%), so B=8 and B=16")
    e(f"% bars are directly comparable; the observed peak is {share_peak:.1%}. Only")
    e("% a left baseline is drawn -- a box edge at the strip maximum would sit")
    e(f"% ~{gap_pt:.0f}pt from the {grid[-1]:.0%} gridline.")
    e("%")
    e("% ENTROPY AT T = 0. Greedy target rows are reported at T = 1 (the recorder")
    e("% maps temperature 0 to 1 before normalizing), i.e. the same raw T = 1")
    e("% next-token entropy H used in the entropy-split appendix. The temperature")
    e("% label on a row is the SAMPLING temperature used to verify, not the")
    e("% temperature the entropy is measured at.")
    e("%")
    e("% NESTED PANELS. B = 16 fills the panel; B = 8 sits in the framed inset. The")
    e("% inset placement is forced, not aesthetic: in a B = 16 panel cell (n_c, i)")
    e("% is blank iff i >= n_c + 1, so the only all-blank block big enough is rows")
    e("% n_c = 0..7 x columns i = 8..14 -- 7 wide, 8 tall, exactly the aspect of the")
    e("% B = 8 grid. Both share the axes convention (n_c downward, i rightward),")
    e("% the share-strip scale, and the single colour scale below, so every cell in")
    e("% the figure is comparable to every other.")
    e("%")
    for line in V["hdr_provenance"]:
        e(line)
    e("%")
    e(f"% SHARED SCALE: [{lo:.2f}, {hi:.2f}] nats, covering the observed")
    e(f"% [{vmin:.4f}, {vmax:.4f}] over all twelve grids. Regenerate the")
    e(f"% \\definecolor block together with these limits -- {NUM_LEVELS} viridis")
    for line in V["hdr_scale_note"]:
        e(line)
    e("%")
    e("% MEASURED, for auditing (N steps; tau = sum over n_c of (n_c+1)*share, a")
    for line in V["hdr_tau_note"]:
        e(line)
    for budget_dir, _ in BUDGETS:
        for temp_dir, _ in V["temps"]:
            for method, _ in METHODS:
                key = (budget_dir, temp_dir, method)
                full = shares[key][-1]
                e(f"%   {budget_dir:4s} {temp_dir:10s} {method:14s} "
                  f"N={totals[key]:7d}  tau={tau(key):.3f}  "
                  f"full-accept share={full:.3%}")
    e("%")
    e("% GEOMETRY. One B=16 cell is \\hcell; a grid is 15 cells wide and 16 tall,")
    e(f"% plus a {STRIP_W:g}-cell share strip, so a panel is {PANEL_W:g} x")
    e(f"% {N16_ROW:g} cells. Columns are pitched {PANEL_PITCH_X:g} apart, rows")
    e(f"% {PANEL_PITCH_Y:g}, so the panel block is")
    e(f"% {2 * PANEL_PITCH_X + PANEL_W:.1f} x {PANEL_PITCH_Y + N16_ROW:.1f} cells;")
    e("% with the ~3 cells of row labels on the left that is about")
    e(f"% {CELL * (2 * PANEL_PITCH_X + PANEL_W + 3):.0f}pt against a 455pt")
    e("% \\textwidth. Shrink \\hcell to fit a narrower text block; nothing else is")
    e("% hand-placed except the inset origin and the colourbar.")
    e("% " + "-" * 73)
    e("")
    e(f"% viridis, sampled at {NUM_LEVELS} levels; used by cells and colourbar.")


def emit_grid(e, grid, n_row, n_col, level_of) -> None:
    """Cells, then the frontier borders on top of them."""
    for nc in range(n_row):
        for i in range(n_col):
            v = grid[nc, i]
            if math.isfinite(v):
                e("    \\fill[vir%d] (%d,%d) rectangle ++(1,1);"
                  % (level_of(v), i, n_row - nc - 1))
    for nc in range(n_row):
        if nc < n_col and math.isfinite(grid[nc, nc]):
            e("    \\draw[frontier] (%d,%d) rectangle ++(1,1);"
              % (nc, n_row - nc - 1))


def emit_strip(e, share, n_row, x0, width, frame_style) -> None:
    """Marginal share of steps per acceptance length, one bar per grid row.

    Only a left baseline is drawn, no enclosing box: a box edge at share_max
    would land ~1pt from the top gridline and read as a doubled rule.
    """
    share_max = V["share_max"]
    for grid_share in share_gridlines():
        gx = x0 + width * grid_share / share_max
        e("    \\draw[sharegrid] (%.4f,0) -- (%.4f,%d);" % (gx, gx, n_row))
    for nc, value in enumerate(share):
        length = width * value / share_max
        y = n_row - nc - 0.5 - BAR_H / 2
        e("    \\fill[sharebar] (%.4f,%.4f) rectangle ++(%.4f,%g);"
          % (x0, y, length, BAR_H))
    e("    \\draw[%s] (%.4f,0) -- (%.4f,%d);" % (frame_style, x0, x0, n_row))


def emit_body(e, levels, grids, shares, level_of, lo, hi) -> None:
    for i, (r, g, b) in enumerate(levels):
        e("\\definecolor{vir%d}{RGB}{%d,%d,%d}" % (i, r, g, b))
    e("")

    e("\\begin{figure*}[t]")
    e("    \\centering")
    e("    \\def\\hcell{%gpt}" % CELL)
    e("    \\begin{tikzpicture}[x=\\hcell, y=\\hcell,")
    e("        % A drafted slot that verification rejected: the chain's frontier.")
    e("        frontier/.style={draw=black, line width=0.45pt},")
    e("        insetframe/.style={draw=black!55, line width=0.4pt},")
    e("        panelframe/.style={draw=black!35, line width=0.4pt},")
    e("        sharebar/.style={fill=black!45},")
    e("        sharegrid/.style={draw=black!20, line width=0.3pt},")
    e("        ticklab/.style={font=\\tiny, inner sep=1pt},")
    e("        axlab/.style={font=\\scriptsize, inner sep=1pt},")
    e("    ]")

    strip_x = N16_COL + STRIP_GAP
    top_grid = share_gridlines()[-1]
    s = INSET_SCALE
    inset_w = INSET_STRIP_X + INSET_STRIP_W
    for row_index, (temp_dir, temp_label) in enumerate(V["temps"]):
        y_off = -row_index * PANEL_PITCH_Y
        for col_index, (method, method_label) in enumerate(METHODS):
            x_off = col_index * PANEL_PITCH_X
            e("")
            e("    %% ===== %s, %s %s" % (method_label, temp_dir, "=" * 24))
            e("    \\begin{scope}[shift={(%g,%g)}]" % (x_off, y_off))

            key16 = ("b16", temp_dir, method)
            emit_grid(e, grids[key16], N16_ROW, N16_COL, level_of)
            e("    %% marginal share of steps by accepted length")
            emit_strip(e, shares[key16], N16_ROW, strip_x, STRIP_W, "panelframe")

            key8 = ("b8", temp_dir, method)
            e("    %% B=8 inset, dropped into the blank upper-right corner")
            e("    \\begin{scope}[shift={(%g,%g)}, xscale=%g, yscale=%g]"
              % (INSET_X0, INSET_Y0, s, s))
            e("    \\fill[white] (-0.2,-0.2) rectangle (%g,%g);"
              % (inset_w + 0.2, N8_ROW + 0.2))
            emit_grid(e, grids[key8], N8_ROW, N8_COL, level_of)
            emit_strip(e, shares[key8], N8_ROW, INSET_STRIP_X, INSET_STRIP_W,
                       "insetframe")
            e("    \\draw[insetframe] (-0.2,-0.2) rectangle (%g,%g);"
              % (inset_w + 0.2, N8_ROW + 0.2))
            e("    \\end{scope}")
            e("    %% budget tags; both sit in blank rows (n_c=0 is blank for")
            e("    %% i>=1, n_c=1 for i>=2).")
            e("    \\node[ticklab, anchor=south west] at (%g,%g) {$B{=}8$};"
              % (INSET_X0 - 0.2 * s, INSET_Y0 + s * (N8_ROW + 0.2) + 0.12))
            e("    \\node[ticklab, anchor=west] at (1.25,%g) {$B{=}16$};"
              % (N16_ROW - 0.5))
            e("    \\draw[panelframe] (0,0) rectangle (%d,%d);" % (N16_COL, N16_ROW))

            if row_index == len(V["temps"]) - 1:
                for i in range(0, N16_COL, 2):
                    e("    \\node[ticklab, anchor=north] at (%g,-0.1) {%d};"
                      % (i + 0.5, i))
                e("    \\node[axlab, anchor=north] at (%g,-1.05) "
                  "{Chain slot $i$};" % (N16_COL / 2))
                e("    \\node[ticklab, anchor=north] at (%g,-0.1) {0};" % strip_x)
                e("    \\node[ticklab, anchor=north] at (%.4f,-0.1) {%d\\%%};"
                  % (strip_x + STRIP_W * top_grid / V["share_max"],
                     round(top_grid * 100)))
                e("    \\node[axlab, anchor=north] at (%.4f,-1.05) {Share};"
                  % (strip_x + STRIP_W / 2))
            if col_index == 0:
                for nc in [0, 1, 3, 5, 7, 9, 11, 13, 15]:
                    e("    \\node[ticklab, anchor=east] at (-0.15,%g) {%d};"
                      % (N16_ROW - nc - 0.5, nc + 1))
                e("    \\node[axlab, anchor=south, rotate=90] at (-1.35,%g) "
                  "{Tokens emitted $n_c{+}1$};" % (N16_ROW / 2))
                e("    \\node[font=\\small, anchor=south, rotate=90] at "
                  "(-2.6,%g) {%s};" % (N16_ROW / 2, temp_label))
            if row_index == 0:
                e("    \\node[font=\\small\\bfseries, anchor=south] at (%g,%g) "
                  "{%s};" % (N16_COL / 2, N16_ROW + 0.15, method_label))
            e("    \\end{scope}")

    emit_colourbar(e, lo, hi)
    emit_caption(e)


def emit_colourbar(e, lo, hi) -> None:
    bar_w, bar_h = 16.0, 0.55
    total_w = 2 * PANEL_PITCH_X + PANEL_W
    bar_x = (total_w - bar_w) / 2
    # Just under the x labels of the bottom row. The scale label sits to the
    # LEFT of the bar, not above it, so it cannot reach the middle panel's
    # "Chain slot i".
    bar_y = -PANEL_PITCH_Y - 2.4
    e("")
    e("    %% ===== shared colour scale " + "=" * 24)
    e("    %% One scale for all twelve grids; strips are the same vir* levels.")
    e("    \\begin{scope}[shift={(%g,%g)}]" % (bar_x, bar_y))
    strip = bar_w / NUM_LEVELS
    for i in range(NUM_LEVELS):
        e("    \\fill[vir%d] (%.4f,0) rectangle ++(%.4f,%g);"
          % (i, i * strip, strip * 1.02, bar_h))
    e("    \\draw[panelframe] (0,0) rectangle (%g,%g);" % (bar_w, bar_h))
    for frac, label in [(0.0, f"{lo:.2f}"), (0.5, f"{(lo + hi) / 2:.2f}"),
                        (1.0, f"{hi:.2f}")]:
        e("    \\draw[panelframe] (%g,0) -- (%g,-0.2);"
          % (frac * bar_w, frac * bar_w))
        e("    \\node[ticklab, anchor=north] at (%g,-0.22) {%s};"
          % (frac * bar_w, label))
    e("    \\node[axlab, anchor=east] at (-0.45,%g) "
      "{Target entropy $H$ (nats)};" % (bar_h / 2))
    e("    \\end{scope}")


def emit_caption(e) -> None:
    e("    \\end{tikzpicture}")
    e("    \\caption{\\textbf{Where a speculative chain runs out of certainty}")
    e("    (%s). Cell $(n_c, i)$ is the mean target" % V["cap_setting"])
    e("    entropy at chain slot $i$ over the steps that accepted exactly $n_c$")
    e("    drafted tokens; only the realized path $i \\le n_c$ is shown, which is")
    e("    why each panel is a lower-left triangle. Rows are labelled by the")
    e("    step's token yield $n_c{+}1$, since the bonus token drawn at the")
    e("    frontier is always kept -- the top row $n_c{=}0$ still emits one")
    e("    token, and the bottom row emits the full budget. Black-bordered cells")
    e("    are the frontier $i{=}n_c$, the one slot verification rejected and the")
    e("    distribution the bonus token comes from; the bottom row accepted every")
    e("    draft and so has none. Grey bars give the share of steps at each")
    e("    acceptance length, on the same $0$--%d\\%% scale in both budgets. Each"
      % round(V["share_max"] * 100))
    e("    panel nests $B{=}8$ (framed inset) inside the blank corner of")
    e("    $B{=}16$, and all twelve grids share the single colour scale below.")
    for line in V["cap_tail"]:
        e(line)
    e("    \\label{%s}" % V["label"])
    e("\\end{figure*}")
    e("")
# PLACEHOLDER_REST


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="qwen3-4b", choices=sorted(VARIANTS))
    parser.add_argument("--workers", type=int, default=32,
                        help="processes for the raw-dump scan")
    args = parser.parse_args()
    global V
    V = VARIANTS[args.variant]

    scans = load_scans(args.workers)
    grids: dict[tuple[str, str, str], np.ndarray] = {}
    shares: dict[tuple[str, str, str], list[float]] = {}
    totals: dict[tuple[str, str, str], int] = {}
    for budget_dir, num_draft in BUDGETS:
        for temp_dir, _ in V["temps"]:
            for method, _ in METHODS:
                key = (budget_dir, temp_dir, method)
                scan = scans[f"{budget_dir}|{temp_dir}|{method}"]
                if V["source"] == "csv":
                    grid = read_path_csv(budget_dir, temp_dir, method, num_draft)
                    grid[0, 0] = scan["nc0_entropy"]
                else:
                    grid = np.array([[np.nan if v is None else v for v in row]
                                     for row in scan["grid"]])
                total = sum(scan["hist"])
                grids[key] = grid
                shares[key] = [c / total for c in scan["hist"]]
                totals[key] = total

    finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids.values()])
    vmin, vmax = float(finite.min()), float(finite.max())
    # Round outward so the colourbar carries quotable end labels.
    lo = math.floor(vmin * 20) / 20
    hi = math.ceil(vmax * 20) / 20
    share_peak = max(max(s) for s in shares.values())
    assert share_peak <= V["share_max"], (share_peak, V["share_max"])

    cmap = matplotlib.colormaps["viridis"]
    levels = [
        tuple(int(round(255 * c)) for c in cmap(i / (NUM_LEVELS - 1))[:3])
        for i in range(NUM_LEVELS)
    ]

    def level_of(value: float) -> int:
        frac = (value - lo) / (hi - lo)
        return int(min(NUM_LEVELS - 1, max(0, round(frac * (NUM_LEVELS - 1)))))

    def tau(key) -> float:
        return sum((nc + 1) * p for nc, p in enumerate(shares[key]))

    lines: list[str] = []
    e = lines.append
    emit_header(e, lo, hi, vmin, vmax, share_peak, totals, shares, tau)
    emit_body(e, levels, grids, shares, level_of, lo, hi)
    V["out"].parent.mkdir(parents=True, exist_ok=True)
    V["out"].write_text("\n".join(lines))

    print(f"wrote {V['out']}")
    print(f"scale [{lo:.2f}, {hi:.2f}] from observed [{vmin:.4f}, {vmax:.4f}]")
    print(f"share peak {share_peak:.4f} against strip max {V['share_max']}")
    for key, g in sorted(grids.items()):
        print(f"  {key}: {int(np.isfinite(g).sum())} cells, "
              f"[{np.nanmin(g):.3f}, {np.nanmax(g):.3f}], "
              f"N={totals[key]}, tau={tau(key):.3f}")


if __name__ == "__main__":
    main()
