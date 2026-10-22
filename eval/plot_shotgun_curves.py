"""Plot acc/tps curves across shotgun1.2_8seg_dpace checkpoints.

For each ckpt dir, use the LATEST eval_* subdir (by name sort) that has a
summary.jsonl. X-axis = global training step parsed from "epoch_E_step_S".
"""

import json
import re
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "outputs/perfectblend/shotgun1.2_8seg_dpace"
DATASETS = ["gsm8k", "math500", "humaneval", "mt-bench"]

# ckpt_dir -> step
def parse_step(name):
    m = re.search(r"step_(\d+)", name)
    return int(m.group(1)) if m else -1

# Collect: step -> {dataset -> (acc, tps)}
rows = {}
for ckpt in sorted(glob.glob(os.path.join(ROOT, "epoch_*_step_*"))):
    if not os.path.isdir(ckpt):
        continue
    step = parse_step(os.path.basename(ckpt))
    eval_dirs = sorted(glob.glob(os.path.join(ckpt, "eval_*")))
    summary = None
    for ed in reversed(eval_dirs):  # latest first
        sf = os.path.join(ed, "summary.jsonl")
        if os.path.isfile(sf) and os.path.getsize(sf) > 0:
            summary = sf
            break
    if summary is None:
        continue
    per_ds = {}
    with open(summary) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            per_ds[r["dataset"]] = (r.get("avg_accept_len", 0), r.get("avg_tok_per_s", 0))
    rows[step] = (os.path.basename(ckpt), per_ds)

steps = sorted(rows)
print(f"Collected {len(steps)} checkpoints")

# Build series per dataset (only points where that dataset exists)
def series(ds, idx):
    xs, ys = [], []
    for s in steps:
        v = rows[s][1].get(ds)
        if v is not None:
            xs.append(s)
            ys.append(v[idx])
    return xs, ys

def mean_series(idx):
    xs, ys = [], []
    for s in steps:
        vals = [rows[s][1][d][idx] for d in DATASETS if d in rows[s][1]]
        if len(vals) == len(DATASETS):  # only full-coverage ckpts
            xs.append(s)
            ys.append(sum(vals) / len(vals))
    return xs, ys

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors = {"gsm8k": "#1f77b4", "math500": "#ff7f0e", "humaneval": "#2ca02c", "mt-bench": "#d62728"}

# --- Left: acceptance length ---
ax = axes[0]
for ds in DATASETS:
    xs, ys = series(ds, 0)
    ax.plot(xs, ys, marker="o", ms=4, label=ds, color=colors[ds])
mxs, mys = mean_series(0)
ax.plot(mxs, mys, marker="s", ms=6, lw=2.5, label="mean (4 ds)", color="black", ls="--")
# annotate best mean
if mys:
    bi = max(range(len(mys)), key=lambda i: mys[i])
    ax.annotate(f"best mean\nstep {mxs[bi]}\n{mys[bi]:.2f}",
                xy=(mxs[bi], mys[bi]), xytext=(mxs[bi], mys[bi] - 0.5),
                ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black"))
ax.set_xlabel("training step")
ax.set_ylabel("avg accept length")
ax.set_title("Acceptance length vs step")
ax.grid(True, alpha=0.3)
ax.legend()

# --- Right: tok/s ---
ax = axes[1]
for ds in DATASETS:
    xs, ys = series(ds, 1)
    ax.plot(xs, ys, marker="o", ms=4, label=ds, color=colors[ds])
mxs, mys = mean_series(1)
ax.plot(mxs, mys, marker="s", ms=6, lw=2.5, label="mean (4 ds)", color="black", ls="--")
if mys:
    bi = max(range(len(mys)), key=lambda i: mys[i])
    ax.annotate(f"best mean\nstep {mxs[bi]}\n{mys[bi]:.0f}",
                xy=(mxs[bi], mys[bi]), xytext=(mxs[bi], mys[bi] - 12),
                ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black"))
ax.set_xlabel("training step")
ax.set_ylabel("avg tok/s")
ax.set_title("Throughput vs step")
ax.grid(True, alpha=0.3)
ax.legend()

fig.suptitle("shotgun1.2_8seg_dpace — accept length & throughput across checkpoints", fontsize=13)
fig.tight_layout()
out = os.path.join(ROOT, "curves_acc_tps.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved {out}")

# Also dump the table to stdout
print("\nstep      " + "".join(f"{d:>22}" for d in DATASETS) + f"{'mean_acc':>10}{'mean_tps':>10}")
for s in steps:
    pd = rows[s][1]
    cells = ""
    accs, tpss = [], []
    for d in DATASETS:
        if d in pd:
            a, t = pd[d]
            cells += f"{a:>10.3f}/{t:<11.1f}"
            accs.append(a); tpss.append(t)
        else:
            cells += f"{'-':>22}"
    ma = sum(accs)/len(accs) if accs else 0
    mt = sum(tpss)/len(tpss) if tpss else 0
    print(f"{s:<10}{cells}{ma:>10.3f}{mt:>10.1f}")
