"""Token-level predictive entropy of the target model on the SpecBench datasets.

Answers "which benchmarks are low- vs high-entropy" with measurements instead of
the hand-written grouping in ``specbench/report.py`` (``_LOW_ENTROPY`` /
``_HIGH_ENTROPY``).

Protocol (locked):
  * **Inherent T=1 entropy.** The server is launched with
    ``SGLANG_RETURN_ORIGINAL_LOGPROB=1``, so returned logprobs are the raw
    ``log_softmax(logits)`` taken *before* temperature scaling. Under greedy
    sampling sglang takes the ``is_all_greedy`` branch of
    ``sglang/srt/layers/sampler.py``, which already does exactly that, so both
    paths agree and the number is comparable across datasets.
  * **Free generation.** The model writes its own continuation and entropy is
    scored at every generated position. Prompt positions are excluded:
    ``return_logprob=True`` with ``logprob_start_len=-1`` makes the scheduler
    start logprobs at ``len(input_ids)``.
  * **Top-K truncation with an exact bound.** Only the top-K logprobs come back
    over HTTP, so ``H_K = -sum_i p_i log p_i`` is a lower bound. Given residual
    tail mass ``M``, the tail adds at most ``M * log((V-K)/M)``; the summary
    reports that as ``entropy_upper``. If lower and upper agree to ~3 decimals
    the truncation is immaterial.

Usage:
    python eval/measure_entropy.py --model /path/to/Qwen3-8B \
        --out-dir benchmark_results/entropy/qwen3-8b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specbench.datasets import load_and_process_dataset
from specbench.harness.sglang import (  # reuse the launch/teardown plumbing
    _flush_cache,
    _launch_with_retry,
    _stop_processes,
)
from specbench.report import _HIGH_ENTROPY, _LOW_ENTROPY

DEFAULT_DATASETS = _LOW_ENTROPY + _HIGH_ENTROPY

def _sequence_stats(top_logprobs: list, vocab_size: int) -> dict | None:
    """Per-step entropy stats for one response, averaged over generated tokens."""
    h_lower_sum = 0.0
    tail_bound_sum = 0.0
    top1_sum = 0.0
    tail_sum = 0.0
    tail_max = 0.0
    steps = 0
    for step in top_logprobs:
        if not step:
            continue
        # each entry is [logprob, token_id, token_text]; topk order is descending
        logprobs = [float(e[0]) for e in step]
        probs = [math.exp(lp) for lp in logprobs]
        h_lower_sum += -sum(p * lp for p, lp in zip(probs, logprobs))
        mass = sum(probs)
        tail = min(max(1.0 - mass, 0.0), 1.0)
        n_tail = max(vocab_size - len(probs), 1)
        if tail > 1e-12:
            tail_bound_sum += tail * math.log(n_tail / tail)
        top1_sum += max(probs)
        tail_sum += tail
        tail_max = max(tail_max, tail)
        steps += 1
    if steps == 0:
        return None
    return {
        "num_tokens": steps,
        "entropy": h_lower_sum / steps,
        "entropy_upper": (h_lower_sum + tail_bound_sum) / steps,
        "top1_prob": top1_sum / steps,
        "tail_mass": tail_sum / steps,
        "tail_mass_max": tail_max,
    }


def _score_prompt(base_url: str, prompt: str, *, max_new_tokens: int, top_k_logprobs: int,
                  temperature: float, timeout_s: int, vocab_size: int) -> dict | None:
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": float(temperature),
                "max_new_tokens": int(max_new_tokens),
            },
            "return_logprob": True,
            "top_logprobs_num": int(top_k_logprobs),
            # -1 + return_logprob => scheduler starts logprobs at len(input_ids),
            # i.e. response positions only (scheduler.py:2247-2251).
            "logprob_start_len": -1,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    out = resp.json()
    meta = out.get("meta_info", {}) or {}
    stats = _sequence_stats(meta.get("output_top_logprobs") or [], vocab_size)
    if stats is None:
        return None
    stats["finish_reason"] = meta.get("finish_reason")
    return stats


def _build_prompts(dataset_name: str, tokenizer, limit: int, seed: int) -> list[str]:
    ds = load_and_process_dataset(dataset_name)
    if len(ds) == 0:
        raise RuntimeError(f"Dataset {dataset_name!r} is empty.")
    if len(ds) > limit:
        ds = ds.shuffle(seed=seed).select(range(limit))
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["turns"][0]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for row in ds
    ]


def _run_dataset(base_urls: list[str], prompts: list[str], *, concurrency: int,
                 desc: str, **score_kw) -> list[dict]:
    n_rep = len(base_urls)
    with ThreadPoolExecutor(max_workers=max(concurrency * n_rep, 1)) as pool:
        futures = {
            pool.submit(_score_prompt, base_urls[i % n_rep], prompt, **score_kw): i
            for i, prompt in enumerate(prompts)
        }
        rows = []
        with tqdm(total=len(futures), desc=desc, unit="sample",
                  disable=not sys.stderr.isatty()) as bar:
            for future in as_completed(futures):
                stats = future.result()
                if stats is not None:
                    rows.append({"sample_index": futures[future], **stats})
                bar.update(1)
    return sorted(rows, key=lambda r: r["sample_index"])


def _aggregate(dataset_name: str, rows: list[dict]) -> dict:
    """Sequence-averaged mean/std (primary) plus a token-weighted mean."""
    per_seq = [r["entropy"] for r in rows]
    total_tokens = sum(r["num_tokens"] for r in rows)
    weighted = sum(r["entropy"] * r["num_tokens"] for r in rows) / max(total_tokens, 1)
    return {
        "dataset": dataset_name,
        "num_sequences": len(rows),
        "num_tokens": total_tokens,
        "entropy": statistics.fmean(per_seq),
        "entropy_std": statistics.stdev(per_seq) if len(per_seq) > 1 else 0.0,
        "entropy_token_weighted": weighted,
        "entropy_upper": statistics.fmean([r["entropy_upper"] for r in rows]),
        "top1_prob": statistics.fmean([r["top1_prob"] for r in rows]),
        "tail_mass": statistics.fmean([r["tail_mass"] for r in rows]),
        "tail_mass_max": max(r["tail_mass_max"] for r in rows),
        "mean_response_tokens": total_tokens / max(len(rows), 1),
    }


def _print_table(summaries: list[dict]) -> None:
    labels = {d: "Low" for d in _LOW_ENTROPY} | {d: "High" for d in _HIGH_ENTROPY}
    header = (f"{'dataset':<14} {'paper':>5} {'H(T=1)':>8} {'±std':>7} {'H_upper':>8} "
              f"{'top1_p':>7} {'tail':>9} {'#seq':>5} {'#tok':>9}")
    print("\n" + header)
    print("-" * len(header))
    for s in sorted(summaries, key=lambda s: s["entropy"]):
        print(f"{s['dataset']:<14} {labels.get(s['dataset'], '-'):>5} "
              f"{s['entropy']:>8.4f} {s['entropy_std']:>7.4f} {s['entropy_upper']:>8.4f} "
              f"{s['top1_prob']:>7.4f} {s['tail_mass']:>9.2e} "
              f"{s['num_sequences']:>5} {s['num_tokens']:>9}")


def _launch_replicas(model: str, gpus: list[str], server_args: list[str],
                     logs_dir: Path) -> list[tuple[subprocess.Popen, str]]:
    launched = []
    for i, gpu in enumerate(gpus):
        env = {"SGLANG_RETURN_ORIGINAL_LOGPROB": "1"}
        if gpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        proc, url = _launch_with_retry(
            model, "127.0.0.1", 20000 + i * 2000, server_args,
            env_overrides=env, log_path=logs_dir / f"entropy_gpu{gpu or i}.log",
            nccl_port_base=21000 + i * 2000)
        launched.append((proc, url))
        print(f"[entropy] replica {i} ready at {url} (GPU={gpu or 'default'})")
    return launched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="target model path, e.g. Qwen3-8B")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--out-dir", default="benchmark_results/entropy")
    ap.add_argument("--max-samples", type=int, default=500,
                    help="cap per dataset; larger sets are shuffled then truncated")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--top-k-logprobs", type=int, default=64,
                    help="K for the truncated entropy; check tail_mass in the summary")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature for the trajectory only; reported "
                         "entropy is always the raw T=1 distribution")
    ap.add_argument("--concurrency", type=int, default=32, help="in-flight requests per replica")
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.8)
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[1] / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vocab_size = int(AutoConfig.from_pretrained(args.model).vocab_size)
    print(f"[entropy] vocab_size={vocab_size} K={args.top_k_logprobs}")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpus = [g.strip() for g in visible.split(",") if g.strip()] or [""]
    if args.tp_size > 1:
        gpus = [""]  # one tp-sharded server owns all visible GPUs

    server_args = [
        "--trust-remote-code",
        "--tp-size", str(args.tp_size),
        "--dtype", "bfloat16",
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--max-running-requests", str(max(args.concurrency, 1)),
    ]

    procs: list[subprocess.Popen] = []
    summaries: list[dict] = []
    seq_path = out_dir / "per_sequence.jsonl"
    try:
        launched = _launch_replicas(args.model, gpus, server_args, out_dir / "logs")
        procs = [p for p, _ in launched]
        base_urls = [u for _, u in launched]

        with seq_path.open("w", encoding="utf-8") as seq_file:
            for name in datasets:
                prompts = _build_prompts(name, tokenizer, args.max_samples, args.seed)
                for url in base_urls:
                    _flush_cache(url)
                rows = _run_dataset(
                    base_urls, prompts, concurrency=args.concurrency,
                    desc=f"{name} ({len(prompts)})",
                    max_new_tokens=args.max_new_tokens,
                    top_k_logprobs=args.top_k_logprobs,
                    temperature=args.temperature, timeout_s=args.timeout_s,
                    vocab_size=vocab_size)
                for row in rows:
                    seq_file.write(json.dumps({"dataset": name, **row}) + "\n")
                seq_file.flush()
                summary = _aggregate(name, rows)
                summaries.append(summary)
                print(f"[entropy] {name}: H={summary['entropy']:.4f} "
                      f"(±{summary['entropy_std']:.4f}, upper={summary['entropy_upper']:.4f}, "
                      f"tail={summary['tail_mass']:.2e}, {summary['num_tokens']} tokens)")
    finally:
        _stop_processes(procs)

    meta = {
        "model": args.model,
        "vocab_size": vocab_size,
        "top_k_logprobs": args.top_k_logprobs,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "trajectory_temperature": args.temperature,
        "logprob_temperature": 1.0,
        "seed": args.seed,
        "datasets": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _print_table(summaries)
    print(f"\n[entropy] per-sequence: {seq_path}\n[entropy] summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()



