"""Batch-record races offline (no browser needed), for later replay in the UI.

Every server configuration is launched once and reused for all prompts and
presets that share it, which keeps the number of slow sglang launches minimal.

    python -m demo.spec_race.record --model Qwen3-8B --budget 16 \
        --presets 0:0 1:0 1:1 --prompts-file demo/spec_race/prompts.json
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from . import prompts as prompts_mod
from . import registry, runner
from .engine import EngineOptions, ServerPool, TokenizerCache, plan_gpus


async def _main(args: argparse.Namespace) -> None:
    if args.prompt:
        prompts = list(args.prompt)
    else:
        prompts = prompts_mod.flatten(Path(args.prompts_file))
    methods = runner.ordered_methods(args.methods, not args.no_baseline)

    opts = EngineOptions(gpu=args.gpu, gpus_spec=args.gpus, mem_fraction=args.mem_fraction,
                         attention_backend=args.attention_backend,
                         cuda_graph_max_bs=args.cuda_graph_max_bs)
    pool = ServerPool(opts)
    toks = TokenizerCache()

    recs = {
        (preset, i): runner.new_recording(args.model, args.budget, preset, p,
                                          args.max_new_tokens, args.seed, methods)
        for preset in args.presets for i, p in enumerate(prompts)
    }
    for (preset, i), rec in recs.items():
        rec["id"] += f"_p{i}"
    input_ids = {i: toks.encode_prompt(args.model, p) for i, p in enumerate(prompts)}
    for (preset, i), rec in recs.items():
        rec["prompt_tokens"] = len(input_ids[i])

    # Group work by server key so each server is launched exactly once.
    groups: dict[str, list[tuple[str, str]]] = {}
    for method in methods:
        for preset in args.presets:
            key = registry.server_spec(args.model, method, args.budget, preset).key
            groups.setdefault(key, []).append((method, preset))

    occupied = [s.gpu for s in pool.servers.values() if s.alive()]
    devices, parallel = plan_gpus(args.gpus, len(groups), occupied=occupied)
    pool.configure(devices, parallel)
    print(f"=== {len(devices)} GPU(s) {devices}: "
          f"{'parallel' if parallel else 'serial'} ===", flush=True)

    async def log(ev: dict) -> None:
        if ev["type"] == "phase":
            print(f"[entropy] {ev['message']}", flush=True)
        elif ev["type"] == "entropy":
            st = ev["stats"]
            fmt = lambda x: "-" if x is None else f"{x:.3f}"
            print(f"[{ev['method']}] entropy: all={fmt(st['all'])} accepted={fmt(st['accepted'])} "
                  f"rejected={fmt(st['rejected'])} full_bonus={fmt(st['full_bonus'])}", flush=True)
        if ev["type"] in ("status", "error", "done"):
            detail = ev.get("message") or ev.get("summary", {})
            if isinstance(detail, dict):
                detail = (f"tokens={detail.get('n_tokens')} tau={detail.get('tau') or 0:.2f} "
                          f"tok/s={detail.get('tok_s') or 0:.1f}")
            print(f"[{ev['method']}] {ev['type']}: {detail}", flush=True)

    async def run_group(key: str, jobs: list[tuple[str, str]]) -> None:
        print(f"=== server {key} ===", flush=True)
        for method, preset in jobs:
            for i in range(len(prompts)):
                rec = recs[(preset, i)]
                await runner.run_one(pool, toks, rec, method, log, input_ids=input_ids[i])
                runner.save_recording(rec)

    try:
        if parallel:
            await asyncio.gather(*(run_group(k, j) for k, j in groups.items()))
        else:
            for key, jobs in groups.items():
                await run_group(key, jobs)
        if not args.no_entropy:
            print("=== target entropy ===", flush=True)
            for (preset, i), rec in recs.items():
                await runner.score_recording(pool, toks, rec, log, args.entropy_top_k,
                                             input_ids=input_ids[i])
                runner.save_recording(rec)
    finally:
        await pool.stop_all()
    for rec in recs.values():
        print(f"saved {runner.save_recording(rec)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen3-8B", choices=list(registry.MODELS))
    ap.add_argument("--budget", type=int, default=16, choices=list(registry.BUDGETS))
    ap.add_argument("--presets", nargs="+", default=list(registry.TEMPERATURE_PRESETS),
                    choices=list(registry.TEMPERATURE_PRESETS))
    ap.add_argument("--methods", nargs="+", default=[m["key"] for m in registry.METHODS],
                    choices=[m["key"] for m in registry.METHODS])
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--prompt", action="append", help="prompt text (repeatable)")
    ap.add_argument("--prompts-file", default=str(Path(__file__).with_name("prompts.json")))
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default="0",
                    help="single device for serial mode (DEPRECATED, use --gpus)")
    ap.add_argument("--gpus", default=None,
                    help="devices for the sglang servers: default 0-7 (all devices, one per "
                         "server), or 0, 0,1,2, or auto (check free GPUs, fall back to serial "
                         "when there are fewer than servers)")
    ap.add_argument("--mem-fraction", type=float, default=0.8)
    ap.add_argument("--attention-backend", default="flashinfer")
    ap.add_argument("--cuda-graph-max-bs", type=int, default=4)
    ap.add_argument("--no-entropy", action="store_true", help="skip per-token target entropy")
    ap.add_argument("--entropy-top-k", type=int, default=20)
    args = ap.parse_args()
    if args.gpus is None:
        # Default: one device per server on the full 8-GPU pool. An explicit
        # legacy `--gpu N` still means serial mode on device N.
        args.gpus = args.gpu if args.gpu != "0" else "0-7"
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
