"""Unified specbench CLI.

Runs are organized into a structured directory (RUN_ID) for experiment
management::

    benchmark_results/specbench/<RUN_ID>/
        manifest.json          # full reproducible config + git commit
        results.jsonl          # aggregate metrics (one row per method x dataset x conc)
        responses/<dataset>/<method>.jsonl   # per-sample prompt+response, split per dataset+method
        report.md

Typical usage (see eval/run_eval_single.sh and eval/run_eval_concurrent.sh)::

    RUN_DIR=$(python -m specbench.cli new-run --harness sglang --target-model M \\
        --temperature 0 --dp-size 8 --concurrency 1 --scale full10 --tag main)
    python -m specbench.cli run --run-dir "$RUN_DIR" --method baseline --harness sglang ...
    python -m specbench.cli run --run-dir "$RUN_DIR" --method dflash  --harness sglang --draft-model ...
    python -m specbench.cli report --run-dir "$RUN_DIR"

Runs resume by default: a (method,dataset,concurrency) already present in
results.jsonl is skipped, so re-invoking the same --run-dir continues where it
left off.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from specbench.config import REPO_ROOT, BenchConfig
from specbench.harness import get_harness, list_harness_names
from specbench.methods import get_method, list_methods
from specbench.metrics import read_results


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--method", required=True, help="registered method name (see `list`)")
    p.add_argument("--harness", required=True, choices=list(list_harness_names()))
    p.add_argument("--target-model", required=True)
    p.add_argument("--draft-model", default=None)
    p.add_argument("--datasets", required=True, help="comma-separated 'name' or 'name:count'")
    p.add_argument("--num-samples", type=int, default=128, help="fallback count for entries without :count")

    # output layout: prefer --run-dir (structured); --output is the flat fallback
    p.add_argument("--run-dir", default=None, help="structured run directory (RUN_ID)")
    p.add_argument("--output", default=None, help="flat results jsonl (used only without --run-dir)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="do not skip (method,dataset,conc) already present in results")
    p.add_argument("--no-save-responses", dest="save_responses", action="store_false")
    p.add_argument("--responses-output", default=None,
                   help="flat responses jsonl (used only without --run-dir)")

    # shared fairness knobs
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--no-baseline", dest="run_baseline", action="store_false",
                   help="HF-Direct: skip the per-sample baseline generate")

    # sglang knobs
    p.add_argument("--attention-backend", default="flashinfer")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--dp-size", type=int, default=1,
                   help="Number of single-GPU data-parallel replicas.")
    p.add_argument("--speculative-use-rejection-sampling", action="store_true",
                   help="Correct target-distribution sampling at temperature>0.")
    p.add_argument("--speculative-draft-temperature", type=float, default=None,
                   help="Draft-side temperature override; 0 forces deterministic drafting.")
    p.add_argument("--spec-num-steps", type=int, default=None,
                   help="Override method's speculative num_steps (e.g. EAGLE3 chain depth).")
    p.add_argument("--spec-eagle-topk", type=int, default=None,
                   help="Override method's speculative eagle topk.")
    p.add_argument("--spec-num-draft-tokens", type=int, default=None,
                   help="Override method's speculative num_draft_tokens.")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--mem-fraction-static", type=float, default=0.8)
    p.add_argument("--concurrencies", default="1",
                   help="comma-separated concurrency per SGLang replica")
    p.add_argument("--repeats", type=int, default=1,
                   help="repeated timed measurements per (dataset,concurrency), "
                        "all within one server launch (cache flushed and RNG "
                        "re-seeded between them). Forced to 1 at temperature 0, "
                        "where decoding is deterministic")
    p.add_argument("--max-running-requests", type=int, default=128)
    p.add_argument("--cuda-graph-max-bs", type=int, default=64)
    p.add_argument("--page-size", type=int, default=1)
    p.add_argument("--disable-cuda-graph", action="store_true")
    p.add_argument("--disable-radix-cache", action="store_true")
    p.add_argument("--warmup-requests", type=int, default=3)
    p.add_argument("--warmup-max-new-tokens", type=int, default=1024)
    p.add_argument("--skip-first-requests", type=int, default=1)
    p.add_argument("--timeout-s", type=int, default=3600)
    p.add_argument("--host", default="127.0.0.1")

    # hf-direct knobs
    p.add_argument("--hf-attn-implementation", default="flash_attention_2",
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--confidence-threshold", type=float, default=0.0)


def _cfg_from_args(args: argparse.Namespace) -> BenchConfig:
    responses_path = None
    if not args.run_dir and args.save_responses:
        responses_path = args.responses_output
        if not responses_path and args.output:
            out = Path(args.output)
            responses_path = str(out.with_name(out.stem + ".responses.jsonl"))
    return BenchConfig(
        target_model=args.target_model,
        draft_model=args.draft_model,
        datasets=args.datasets,
        num_samples=args.num_samples,
        run_baseline=args.run_baseline,
        run_dir=args.run_dir,
        resume=args.resume,
        save_responses=args.save_responses,
        responses_path=responses_path,
        results_output=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        attention_backend=args.attention_backend,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        speculative_use_rejection_sampling=args.speculative_use_rejection_sampling,
        speculative_draft_temperature=args.speculative_draft_temperature,
        spec_num_steps=args.spec_num_steps,
        spec_eagle_topk=args.spec_eagle_topk,
        spec_num_draft_tokens=args.spec_num_draft_tokens,
        dtype=args.dtype,
        mem_fraction_static=args.mem_fraction_static,
        concurrencies=[int(c) for c in str(args.concurrencies).split(",") if c.strip()],
        repeats=args.repeats,
        max_running_requests=args.max_running_requests,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        page_size=args.page_size,
        disable_cuda_graph=args.disable_cuda_graph,
        disable_radix_cache=args.disable_radix_cache,
        warmup_requests=args.warmup_requests,
        warmup_max_new_tokens=args.warmup_max_new_tokens,
        skip_first_requests=args.skip_first_requests,
        timeout_s=args.timeout_s,
        host=args.host,
        hf_attn_implementation=args.hf_attn_implementation,
        confidence_threshold=args.confidence_threshold,
    )


def _compute_done(cfg: BenchConfig, harness: str, method: str) -> set:
    """Units already completed for THIS method+harness AND THIS exact draft ckpt.

    Keying on draft_model ensures that re-running the same method with a
    different checkpoint (same run dir) is NOT wrongly skipped. ``repeat`` is
    part of the key so an interrupted multi-repeat run resumes at the repeat it
    died on instead of redoing all of them.
    """
    done: set = set()
    if not cfg.resume:
        return done
    rf = cfg.results_file()
    if rf.exists():
        for r in read_results(rf):
            if r.harness == harness and r.method == method and r.draft_model == cfg.draft_model:
                done.add((r.dataset, r.concurrency, r.repeat))
    return done


def _cmd_run(args: argparse.Namespace) -> None:
    if not args.run_dir and not args.output:
        raise SystemExit("Provide --run-dir (recommended) or --output.")
    method = get_method(args.method)
    harness = get_harness(args.harness)
    cfg = _cfg_from_args(args)

    done = _compute_done(cfg, args.harness, args.method)
    if done:
        print(f"[specbench] resume: {len(done)} (dataset,conc,repeat) already done for "
              f"{args.method}/{args.harness}, will skip them.")

    results = harness.run(method, cfg, done)

    if cfg.run_dir:
        from specbench.runmeta import update_manifest
        mf = cfg.manifest_file()
        if mf is not None:
            update_manifest(mf, cfg=cfg, method=method, harness=args.harness)

    print(f"[specbench] {args.method}/{args.harness}: {len(results)} new record(s); "
          f"results -> {cfg.results_file()}")


def _cmd_new_run(args: argparse.Namespace) -> None:
    from specbench.runmeta import make_run_id
    base = Path(args.base_dir)
    if not base.is_absolute():
        base = REPO_ROOT / base
    run_id = make_run_id(
        harness=args.harness, target_model=args.target_model, temperature=args.temperature,
        rejection_sampling=args.rejection_sampling, dp_size=args.dp_size,
        concurrency=args.concurrency, scale=args.scale, tag=args.tag)
    run_dir = base / run_id
    (run_dir / "responses").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    print(str(run_dir))


def _cmd_report(args: argparse.Namespace) -> None:
    from specbench.report import (build_report, build_latex_table,
                                  build_mixed_breakdown, load_profile_medians,
                                  load_baseline_tps)
    if args.run_dir:
        results_path = Path(args.run_dir) / "results.jsonl"
        output_md = args.output_md or str(Path(args.run_dir) / "report.md")
        tex_outputs = {
            "weighted": str(Path(args.run_dir) / "report.tex"),
            "mean": str(Path(args.run_dir) / "report_mean.tex"),
        }
        profiles_dir = Path(args.profile_dir) if args.profile_dir else Path(args.run_dir) / "profiles"
    else:
        results_path = Path(args.input)
        output_md = args.output_md
        tex_outputs = {}
        profiles_dir = Path(args.profile_dir) if args.profile_dir else None
    baseline_tps: dict[str, float] = {}
    if args.baseline_dir:
        baseline_tps = load_baseline_tps(args.baseline_dir)
        if not baseline_tps:
            print(f"WARNING: no baseline rows under {args.baseline_dir}; "
                  "LaTeX tables will fall back to absolute tok/s")
    run_label = args.run_dir or args.input
    results = read_results(results_path)
    md = build_report(results)
    if args.run_dir:
        md += build_mixed_breakdown(args.run_dir)
    if output_md:
        Path(output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(output_md).write_text(md, encoding="utf-8")
        print(f"Wrote report to {output_md}")
    else:
        print(md)
    # LaTeX tables: weighted average + unweighted mean
    for average, output_tex in tex_outputs.items():
        tex = build_latex_table(results, average=average,
                                baseline_tps=baseline_tps, run_label=run_label)
        Path(output_tex).write_text(tex, encoding="utf-8")
        print(f"Wrote LaTeX table ({average}) to {output_tex}")
    # Same tables plus profiling latency columns, when profiling data exists
    profile_stats = load_profile_medians(profiles_dir) if (profiles_dir and profiles_dir.is_dir()) else {}
    if profile_stats and tex_outputs:
        suffix = {"weighted": "report_weighted_with_profling.tex",
                  "mean": "report_mean_with_profiling.tex"}
        for average, name in suffix.items():
            tex = build_latex_table(results, average=average,
                                    profile_stats=profile_stats,
                                    baseline_tps=baseline_tps,
                                    run_label=run_label)
            out = Path(tex_outputs[average]).parent / name
            out.write_text(tex, encoding="utf-8")
            print(f"Wrote LaTeX table ({average}+profiling) to {out}")


def _cmd_accuracy(args: argparse.Namespace) -> None:
    from specbench.accuracy import score_responses
    ds = args.datasets.split(",") if args.datasets else None
    score_responses(args.run_dir, datasets=ds)


def _cmd_list(_: argparse.Namespace) -> None:
    print("Harnesses:", ", ".join(list_harness_names()))
    print("\nMethods:")
    for m in list_methods():
        sg = "sglang" if m.sglang_supported else "-"
        hf = "hf_direct" if m.hf_direct_supported else "-"
        print(f"  {m.name:10s} family={m.family:9s} harness=[{sg}, {hf}]  {m.note}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="specbench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run one method under one harness")
    _add_run_args(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_new = sub.add_parser("new-run", help="create a RUN_ID directory and print its path")
    p_new.add_argument("--harness", required=True, choices=list(list_harness_names()))
    p_new.add_argument("--target-model", required=True)
    p_new.add_argument("--temperature", type=float, default=0.0)
    p_new.add_argument("--rejection-sampling", action="store_true")
    p_new.add_argument("--dp-size", type=int, default=1)
    p_new.add_argument("--concurrency", type=int, default=1,
                       help="concurrency per SGLang replica")
    p_new.add_argument("--scale", default="")
    p_new.add_argument("--tag", default="")
    p_new.add_argument("--base-dir", default="benchmark_results/specbench")
    p_new.set_defaults(func=_cmd_new_run)

    p_report = sub.add_parser("report", help="aggregate results into a report")
    p_report.add_argument("--run-dir", default=None, help="read <run-dir>/results.jsonl")
    p_report.add_argument("--input", default=None, help="explicit results jsonl")
    p_report.add_argument("--output-md", default=None)
    p_report.add_argument("--profile-dir", default=None,
                          help="profiling JSON dir for latency columns (default: <run-dir>/profiles)")
    p_report.add_argument("--baseline-dir", default=None,
                          help="run dir holding the no-speculation baseline, used as the "
                               "denominator of the LaTeX speedup columns. Needed when this "
                               "run dir has no baseline rows of its own (e.g. SKIP_BASELINE=1 "
                               "sweeps, or a concurrency the baseline was not run at). "
                               "Without it the LaTeX tables show absolute tok/s.")
    p_report.set_defaults(func=_cmd_report)

    p_list = sub.add_parser("list", help="list registered methods and harnesses")
    p_list.set_defaults(func=_cmd_list)

    p_acc = sub.add_parser("accuracy", help="score responses against gold answers")
    p_acc.add_argument("--run-dir", required=True, help="Path to run directory")
    p_acc.add_argument("--datasets", default=None,
                       help="Comma-separated dataset names (default: all scorable)")
    p_acc.set_defaults(func=_cmd_accuracy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
