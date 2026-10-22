#!/usr/bin/env python3
"""Break acceptance length down per source benchmark of a mixed dataset.

Usage:
    python scripts/aggregate_tau_by_source.py <sweep_root> [--output <file.md>]

``cache/dataset/specmix`` (built by ``scripts/build_specmix_dataset.py``) is a
shuffled mixture of ten benchmarks and carries a ``source`` column, but the
harness only ever reports one aggregate ``accept_length`` per (method,
concurrency) row in ``results.jsonl``. The per-request dumps under
``<run_dir>/responses/<dataset>/<method>.jsonl`` do carry ``sample_index``, which
is the row index into the dataset, so joining the two recovers a per-benchmark
acceptance length.

tau is token-weighted, matching the harness definition:

    tau = sum(completion_tokens) / sum(spec_verify_ct)

over the requests of one (run_dir, method, source) group. Averaging the
per-request ``spec_accept_length`` instead would weight a 20-token reply the same
as a 2000-token one.

CAVEAT: the harness overwrites the response dump on every repeat, so only the
LAST repeat survives on disk. Every number here is therefore a single
measurement, unlike the mean-of-repeats in ``repeats_summary.md``. That is
tolerable because tau is the most stable metric in the sweep (cross-repeat CV
<= 0.5%), but it does mean the aggregate column here can differ from the
``accept_len`` mean in ``repeats_summary.md`` by a few thousandths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_sources(dataset_dir: Path) -> list[str]:
    """sample_index -> source name, read from the on-disk HF dataset."""
    from datasets import load_from_disk

    return list(load_from_disk(str(dataset_dir))["source"])


def tally(path: Path, sources: list[str]) -> dict[str, tuple[int, int, int]]:
    """source -> (requests, completion_tokens, spec_verify_ct)."""
    out: dict[str, tuple[int, int, int]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            verify = rec.get("spec_verify_ct")
            if not verify:
                continue
            src = sources[rec["sample_index"]]
            n, tok, ver = out.get(src, (0, 0, 0))
            out[src] = (n + 1, tok + rec["completion_tokens"], ver + verify)
    return out


def collect(sweep_root: Path, dataset_root: Path) -> tuple[dict, dict[str, int]]:
    """(run_dir, dataset) -> {method: {source: tau}}, plus requests per source."""
    tables: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    counts: dict[str, int] = {}
    cache: dict[str, list[str]] = {}
    for path in sorted(sweep_root.rglob("responses/*/*.jsonl")):
        dataset = path.parent.name
        run_dir = str(path.parents[2].relative_to(sweep_root))
        if dataset not in cache:
            cache[dataset] = load_sources(dataset_root / dataset)
        per_source = tally(path, cache[dataset])
        if not per_source:
            continue
        taus = {src: tok / ver for src, (_n, tok, ver) in per_source.items()}
        total_tok = sum(tok for _n, tok, _v in per_source.values())
        total_ver = sum(ver for _n, _t, ver in per_source.values())
        taus["all"] = total_tok / total_ver
        tables.setdefault((run_dir, dataset), {})[path.stem] = taus
        for src, (n, _t, _v) in per_source.items():
            counts[src] = n
    return tables, counts


def build_markdown(tables: dict, counts: dict[str, int]) -> str:
    sources = sorted(counts)
    lines = [
        "# Acceptance length per source benchmark",
        "",
        "> Token-weighted tau = sum(completion_tokens) / sum(spec_verify_ct) over "
        "the requests of one (config, method, source) group, joined to the "
        "dataset's `source` column through `sample_index`. Written by "
        "`python scripts/aggregate_tau_by_source.py <sweep>`.",
        "",
        "> Only the LAST repeat's per-request dump survives on disk, so unlike "
        "`repeats_summary.md` these are single measurements (tau's cross-repeat "
        "CV is <= 0.5%, so the `all` column still lands within a few thousandths "
        "of the `accept_len` mean there). Best per column in **bold**.",
        "",
        "Requests per source: "
        + ", ".join(f"{src} {counts[src]}" for src in sources)
        + f" (total {sum(counts.values())}).",
        "",
    ]
    header = "| method | " + " | ".join(sources) + " | all |"
    divider = "|---" * (len(sources) + 2) + "|"
    for (run_dir, dataset) in sorted(tables):
        methods = tables[(run_dir, dataset)]
        best = {
            col: max(
                (m[col] for m in methods.values() if col in m), default=None
            )
            for col in sources + ["all"]
        }
        lines += [f"## {run_dir} ({dataset})", "", header, divider]
        for method in sorted(methods):
            cells = []
            for col in sources + ["all"]:
                tau = methods[method].get(col)
                if tau is None:
                    cells.append("---")
                    continue
                top = best[col] is not None and round(tau, 2) == round(best[col], 2)
                cells.append(f"**{tau:.2f}**" if top else f"{tau:.2f}")
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_root", type=Path)
    ap.add_argument("--dataset-root", type=Path, default=Path("cache/dataset"))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if not args.sweep_root.is_dir():
        print(f"sweep root not found: {args.sweep_root}")
        return 1
    tables, counts = collect(args.sweep_root, args.dataset_root)
    if not tables:
        print(f"no per-request response dumps found under {args.sweep_root}")
        return 1

    out = args.output or args.sweep_root / "tau_by_source.md"
    out.write_text(build_markdown(tables, counts), encoding="utf-8")
    print(f"per-source tau: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
