#!/usr/bin/env python3
"""Evaluate DSpark speculative decoding via a running SGLang server.

Protocol matches the official HF eval (eval/benchmark_dflash.py):
  - chat template with enable_thinking=False
  - max_new_tokens 2048, temperature 0
  - full official sample counts per dataset
  - multi-turn (mt-bench runs both turns with accumulated history)
  - accept length = per-step pooled mean = sum(completion_tokens) / sum(spec_verify_ct)

Usage:
  python3 scripts/eval_dspark_sglang.py --port 30000
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "Domino" / "code"))

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Official counts from eval/legacy/run_dspark_eval.sh TASKS
DEFAULT_TASKS = {
    "gsm8k": 1319, "math500": 500, "aime25": 30,
    "humaneval": 164, "mbpp": 257, "livecodebench": 700,
    "mt-bench": 80, "alpaca": 500, "arena-hard": 500, "swe-bench": 300,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--model-path", type=str,
                    default=str(ROOT / "cache" / "models" / "Qwen3-8B"))
    ap.add_argument("--tasks", type=str, default=None,
                    help="Comma list dataset:max_samples; default = official counts")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--output-file", type=str,
                    default=str(ROOT / "outputs" / "dspark_sglang_eval" / "results.json"))
    args = ap.parse_args()

    if args.tasks:
        tasks = {}
        for part in args.tasks.split(","):
            name, _, n = part.partition(":")
            tasks[name] = int(n) if n else None
    else:
        tasks = dict(DEFAULT_TASKS)

    from transformers import AutoTokenizer
    from model import load_and_process_dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_url = f"http://{args.host}:{args.port}"

    def run_instance(turns):
        """Run all turns of one sample; return per-turn meta_info list."""
        messages = []
        metas = []
        for user_content in turns:
            messages.append({"role": "user", "content": user_content})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            r = requests.post(f"{base_url}/generate", json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                },
            }, timeout=3600)
            r.raise_for_status()
            out = r.json()
            metas.append(out.get("meta_info", {}))
            messages.append({"role": "assistant", "content": out.get("text", "")})
        return metas

    summary = []
    per_sample_path = Path(args.output_file).with_suffix(".samples.jsonl")
    per_sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_f = open(per_sample_path, "a", encoding="utf-8")

    for ds_name, max_samples in tasks.items():
        dataset = load_and_process_dataset(ds_name)
        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.shuffle(seed=0).select(range(max_samples))

        t_ds = time.time()
        tok_total, verify_total, lat_total = 0, 0, 0.0
        n_ok = 0
        done = 0

        def worker(idx_row):
            idx, row = idx_row
            return idx, run_instance(row["turns"])

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for idx, metas in ex.map(worker, enumerate(dataset)):
                ct = sum(m.get("completion_tokens", 0) for m in metas)
                vc = sum(m.get("spec_verify_ct", 0) or 0 for m in metas)
                lat = sum(m.get("e2e_latency", 0.0) for m in metas)
                if ct > 0 and vc > 0:
                    tok_total += ct
                    verify_total += vc
                    lat_total += lat
                    n_ok += 1
                sample_f.write(json.dumps({
                    "dataset": ds_name, "idx": idx,
                    "completion_tokens": ct, "spec_verify_ct": vc,
                    "e2e_latency": lat,
                    "accept_length": (ct / vc) if vc else None,
                }) + "\n")
                sample_f.flush()
                done += 1
                if done % 20 == 0 or done == len(dataset):
                    running_al = tok_total / verify_total if verify_total else 0
                    print(f"[{ds_name} {done}/{len(dataset)}] "
                          f"pooled_accept_len={running_al:.3f}", flush=True)

        row = {
            "dataset": ds_name,
            "num_samples": n_ok,
            "avg_accept_len": round(tok_total / verify_total, 3) if verify_total else None,
            "avg_tok_per_s": round(tok_total / lat_total, 1) if lat_total else None,
            "total_completion_tokens": tok_total,
            "wall_time_sec": round(time.time() - t_ds, 1),
        }
        summary.append(row)
        print(json.dumps(row), flush=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    sample_f.close()
    print("\n=== FINAL SUMMARY ===")
    print(f"{'dataset':<16}{'n':>6}{'accept_len':>12}{'tok/s':>10}")
    for row in summary:
        print(f"{row['dataset']:<16}{row['num_samples']:>6}"
              f"{row['avg_accept_len']:>12.3f}{row['avg_tok_per_s']:>10.1f}")
    print(f"\nResults written to {args.output_file}")


if __name__ == "__main__":
    main()
