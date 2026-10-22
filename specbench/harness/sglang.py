"""SGLang serving harness — the production-representative protocol.

Launches one ``sglang serve`` process per method, drives it over HTTP, and
reports end-to-end wall-clock tok/s plus ``spec_accept_length`` from the
server's ``meta_info``. Metric definitions are preserved from the original
``third_party/Domino/code/benchmark_sglang.py`` so historical SGLang numbers
remain comparable.

Baseline is just the ``baseline`` method (no speculative args); speedup is
computed later in :mod:`specbench.report` by joining method vs baseline rows.
"""

from __future__ import annotations

import json
import os
import random
import re
import shlex
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree

from specbench.config import BenchConfig, seed_for_repeat
from specbench.datasets import load_and_process_dataset
from specbench.harness.base import Harness
from specbench.methods import MethodSpec
from specbench.metrics import BenchResult, append_jsonl, write_results

_SERVER_LAUNCH_TIMEOUT = 3600


# --- low-level server plumbing --------------------------------------------

def _is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", int(port))) != 0


def _find_available_port(base_port: int) -> int:
    port = int(base_port) + random.randint(100, 1000)
    while True:
        if _is_port_available(port):
            return port
        port = port + 42 if port < 60000 else port - 43


def _popen_launch_server(
    model: str,
    base_url: str,
    other_args: list[str],
    env_overrides: Optional[dict] = None,
    log_path: Optional[Path] = None,
) -> subprocess.Popen:
    host, port_s = base_url[len("http://"):].rsplit(":", 1)
    command = ["sglang", "serve", "--model-path", model,
               *[str(x) for x in other_args], "--host", host, "--port", port_s]
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})

    if log_path is None:
        process = subprocess.Popen(command, env=env)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n$ {shlex.join(command)}\n")
            log_file.flush()
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

    deadline = time.time() + _SERVER_LAUNCH_TIMEOUT
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited early with code {process.returncode}.")
        try:
            resp = requests.get(base_url + "/health", timeout=5)
            if resp.status_code == 200:
                return process
            last_error = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)
    try:
        kill_process_tree(process.pid)
    except Exception:
        pass
    raise TimeoutError(f"Timed out waiting for SGLang server at {base_url}; last_error={last_error}")


def _flush_cache(base_url: str) -> None:
    for i in range(5):
        try:
            resp = requests.get(base_url + "/flush_cache", timeout=60)
            resp.raise_for_status()
            return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1.0 + i)
    print(f"[specbench.sglang] warning: /flush_cache failed; continuing. error={last_error}")


def _set_random_seed(base_url: str, seed: int) -> None:
    """Re-seed the server RNG so each repeat is independently reproducible.

    ``--random-seed`` only applies at launch, so repeats inside one launch would
    otherwise share a single advancing RNG stream and a resumed repeat could
    never be replayed. Requires the ``random_seed`` key in the scheduler's
    ``set_internal_state`` allow-list.
    """
    try:
        resp = requests.post(
            base_url + "/set_internal_state",
            json={"server_args": {"random_seed": int(seed)}},
            timeout=60,
        )
        resp.raise_for_status()
        # The endpoint reports one flag per scheduler (one per TP/DP rank).
        updated = resp.json()
        if not isinstance(updated, list):
            updated = [updated]
        if not updated or not all(updated):
            print(f"[specbench.sglang] warning: {base_url} rejected random_seed={seed} "
                  f"({updated}); this repeat is not reproducible.")
    except Exception as exc:
        print(f"[specbench.sglang] warning: failed to set random_seed={seed} on "
              f"{base_url}: {exc}")


def _dump_dspark_profile(base_urls: list[str], path: Path) -> None:
    """Fetch DSpark info records from all replicas and write to JSON."""
    _dump_spec_profile(base_urls, path, key="dspark_info_record")


def _dump_spec_profile(base_urls: list[str], path: Path, key: str = "dspark_info_record") -> None:
    """Fetch spec info records (any algorithm) from all replicas and write to JSON."""
    ranks = []
    for url in base_urls:
        try:
            resp = requests.get(url + "/server_info", timeout=30)
            resp.raise_for_status()
            info = resp.json()
            internal_states = info.get("internal_states", [])
            for state in internal_states:
                record = state.get(key)
                if record is not None:
                    ranks.append(record)
        except Exception as exc:
            print(f"[specbench.sglang] warning: failed to fetch spec profile from {url}: {exc}")
    if ranks:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ranks": ranks}, indent=2), encoding="utf-8")


def _send_generate(base_url: str, prompt: str, *, max_new_tokens: int, timeout_s: int,
                   temperature: float, top_p: float, top_k: int,
                   rid: Optional[str] = None) -> dict:
    payload = {"text": prompt, "sampling_params": {
        "temperature": float(temperature), "top_p": float(top_p),
        "top_k": int(top_k), "max_new_tokens": int(max_new_tokens)}}
    if rid is not None:
        # Caller-supplied rid: server-side token-level dumps (p/q records) carry
        # it, which is the only way to attribute records back to a dataset.
        payload["rid"] = rid
    resp = requests.post(base_url + "/generate", json=payload, timeout=int(timeout_s))
    resp.raise_for_status()
    return resp.json()


@dataclass(frozen=True)
class _Metrics:
    # Primary, DP-invariant: token-weighted mean of per-request decode speed.
    # Concurrency is defined per replica, so concurrency=1 keeps one request
    # active on every replica and still reports the single-GPU request speed.
    per_request_toks_per_s: float
    # Aggregate wall-clock throughput of the timed phase (grows with parallelism).
    wall_toks_per_s: float
    wall_latency_s: float
    output_tokens: int
    spec_accept_length: Optional[float]
    spec_verify_ct_sum: int


def _timed_send(base_url: str, prompt: str, **kw) -> tuple[float, dict]:
    t0 = time.perf_counter()
    out = _send_generate(base_url, prompt, **kw)
    return time.perf_counter() - t0, out


def _run_requests(base_urls: list[str], *, prompts: list[str], max_new_tokens: int, concurrency: int,
                  timeout_s: int, expect_spec: bool, temperature: float, top_p: float, top_k: int,
                  warmup_prompts: list[str], warmup_max_new_tokens: int,
                  skip_prompts: list[str], progress_desc: Optional[str] = None,
                  rid_prefix: Optional[str] = None,
                  ) -> tuple[_Metrics, list[dict]]:
    n_rep = max(len(base_urls), 1)

    def _url_for(i: int) -> str:
        return base_urls[i % n_rep]

    def _rid_for(stage: str, i: int) -> Optional[str]:
        return None if rid_prefix is None else f"{rid_prefix}|{stage}|{i}"

    def _issue(
        pool_prompts: list[str], mnt: int, stage: str, rid_stage: str
    ) -> list[tuple[float, dict]]:
        if not pool_prompts:
            return []
        with ExitStack() as stack:
            pools = [
                stack.enter_context(ThreadPoolExecutor(max_workers=max(int(concurrency), 1)))
                for _ in range(n_rep)
            ]
            futures = [
                pools[i % n_rep].submit(
                    _timed_send, _url_for(i), prompt, max_new_tokens=mnt,
                    timeout_s=timeout_s, temperature=temperature,
                    top_p=top_p, top_k=top_k, rid=_rid_for(rid_stage, i),
                )
                for i, prompt in enumerate(pool_prompts)
            ]
            completed = as_completed(futures)
            with tqdm(
                completed,
                total=len(futures),
                desc=f"{progress_desc} {stage}" if progress_desc else None,
                unit="sample",
                disable=not progress_desc or not sys.stderr.isatty(),
            ) as progress:
                return [future.result() for future in progress]

    def _issue_indexed(pool_prompts: list[str], mnt: int) -> list[tuple[int, float, dict]]:
        if not pool_prompts:
            return []
        with ExitStack() as stack:
            pools = [
                stack.enter_context(ThreadPoolExecutor(max_workers=max(int(concurrency), 1)))
                for _ in range(n_rep)
            ]
            fut_to_idx = {
                pools[i % n_rep].submit(
                    _timed_send, _url_for(i), prompt, max_new_tokens=mnt,
                    timeout_s=timeout_s, temperature=temperature,
                    top_p=top_p, top_k=top_k, rid=_rid_for("timed", i),
                ): i
                for i, prompt in enumerate(pool_prompts)
            }
            out = []
            completed = as_completed(fut_to_idx)
            with tqdm(
                completed,
                total=len(fut_to_idx),
                desc=progress_desc,
                unit="sample",
                disable=not progress_desc or not sys.stderr.isatty(),
            ) as progress:
                for future in progress:
                    latency, response = future.result()
                    out.append((fut_to_idx[future], latency, response))
            return out

    # Warmup + skip: issued (so the request stream matches) but excluded from timing.
    _issue(warmup_prompts, warmup_max_new_tokens or max_new_tokens, "warmup", "warmup")
    _issue(skip_prompts, max_new_tokens, "skip", "skip")

    start = time.perf_counter()
    timed = _issue_indexed(prompts, max_new_tokens)
    wall_latency = time.perf_counter() - start

    total_tokens = 0
    sum_req_latency = 0.0
    spec_verify_ct_sum = 0
    samples: list[dict] = []
    for idx, req_latency, out in sorted(timed, key=lambda x: x[0]):
        meta = out.get("meta_info", {}) or {}
        toks = int(meta.get("completion_tokens", 0))
        total_tokens += toks
        # server-side e2e_latency is preferred when present (excludes client
        # queueing); else fall back to client-measured per-request latency.
        rl = float(meta.get("e2e_latency", req_latency)) or req_latency
        sum_req_latency += rl
        spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
        req_accept = meta.get("spec_accept_length")
        if req_accept is not None:
            try:
                req_accept = float(req_accept)
            except (TypeError, ValueError):
                req_accept = None
        samples.append({
            "sample_index": idx,
            "prompt": prompts[idx],
            "response": out.get("text", ""),
            "completion_tokens": toks,
            "spec_accept_length": (float(req_accept) if req_accept is not None else None),
            "spec_verify_ct": int(meta.get("spec_verify_ct", 0)),
            "e2e_latency": (float(meta["e2e_latency"]) if "e2e_latency" in meta else req_latency),
            "finish_reason": meta.get("finish_reason"),
        })

    if expect_spec and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "Speculative sanity check failed: no `spec_verify_ct` observed in "
            "responses (speculative decoding may not have been enabled)."
        )

    metrics = _Metrics(
        per_request_toks_per_s=(total_tokens / sum_req_latency if sum_req_latency > 0 else 0.0),
        wall_toks_per_s=total_tokens / max(wall_latency, 1e-6),
        wall_latency_s=float(wall_latency),
        output_tokens=int(total_tokens),
        spec_accept_length=(
            float(total_tokens / spec_verify_ct_sum) if spec_verify_ct_sum > 0 else None
        ),
        spec_verify_ct_sum=int(spec_verify_ct_sum),
    )
    return metrics, samples


def _cycle_take(pool: list[str], n: int) -> list[str]:
    if n <= 0 or not pool:
        return []
    return [pool[i % len(pool)] for i in range(n)]


# --- harness ---------------------------------------------------------------

def _launch_with_retry(
    model: str,
    host: str,
    port_base: int,
    other_args: list[str],
    env_overrides: Optional[dict] = None,
    log_path: Optional[Path] = None,
    attempts: int = 5,
    nccl_port_base: Optional[int] = None,
):
    """Launch one server, retrying on early exit / port races.

    Each attempt re-picks a fresh http port (and a fresh --nccl-port when
    ``nccl_port_base`` is provided), so an EADDRINUSE from a lingering
    TIME_WAIT socket on the previous method's rendezvous port does not
    persist across retries. Used to launch N single-GPU replicas (each
    pinned to one GPU via env_overrides CUDA_VISIBLE_DEVICES), which
    avoids sglang's DP multi-port rendezvous entirely.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        base_url = f"http://{host}:{_find_available_port(port_base)}"
        if nccl_port_base is not None:
            # Strip any pre-existing --nccl-port <value> pair and pick a fresh one.
            filtered: list[str] = []
            skip_next = False
            for a in other_args:
                if skip_next:
                    skip_next = False
                    continue
                if a == "--nccl-port":
                    skip_next = True
                    continue
                filtered.append(a)
            attempt_args = [*filtered, "--nccl-port", str(_find_available_port(nccl_port_base))]
        else:
            attempt_args = other_args
        try:
            proc = _popen_launch_server(
                model, base_url, attempt_args, env_overrides, log_path
            )
            return proc, base_url
        except (RuntimeError, TimeoutError) as exc:
            last_exc = exc
            print(f"[specbench.sglang] launch attempt {attempt}/{attempts} failed: {exc}")
            time.sleep(5)
    raise RuntimeError(f"SGLang server failed to launch after {attempts} attempts: {last_exc}")


def _replica_env(method: MethodSpec, gpu: Optional[str] = None) -> Optional[dict[str, str]]:
    env = dict(method.sglang_env)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    pq_dir = os.environ.get("SGLANG_SPEC_DUMP_PQ_PATH", "")
    if pq_dir:
        # One p/q dump subdirectory per method so replicas of different methods
        # never share a directory (records are one JSONL per server process).
        env["SGLANG_SPEC_DUMP_PQ_PATH"] = str(Path(pq_dir) / method.name)
    return env or None


def _replica_log_path(
    cfg: BenchConfig, method: MethodSpec, replica_index: int, gpu: Optional[str]
) -> Path:
    if cfg.draft_model:
        draft_path = Path(cfg.draft_model.rstrip("/"))
        variant = f"{draft_path.parent.name}__{draft_path.name}"
    else:
        variant = "target-only"
    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{method.name}__{variant}__gpu{gpu or replica_index}",
    )
    return cfg.logs_dir() / f"{safe_name}.log"


def _stop_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        try:
            kill_process_tree(process.pid)
            process.wait(timeout=30)
        except Exception:
            pass


def _launch_replicas(
    method: MethodSpec,
    cfg: BenchConfig,
    server_args: list[str],
    n_rep: int,
    gpus: list[str],
) -> list[tuple[subprocess.Popen, str]]:
    def launch(replica_index: int) -> tuple[subprocess.Popen, str]:
        gpu = gpus[replica_index] if gpus else None
        replica_port_base = cfg.port_base + replica_index * 2000
        nccl_port_base = replica_port_base + 1000
        return _launch_with_retry(
            cfg.target_model,
            cfg.host,
            replica_port_base,
            server_args,
            env_overrides=_replica_env(method, gpu),
            log_path=_replica_log_path(cfg, method, replica_index, gpu),
            nccl_port_base=nccl_port_base,
        )

    print(f"[specbench.sglang] launching {n_rep} {method.name} replicas in parallel...")
    launched: list[Optional[tuple[subprocess.Popen, str]]] = [None] * n_rep
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=n_rep) as pool:
        future_to_index = {pool.submit(launch, i): i for i in range(n_rep)}
        with tqdm(
            total=n_rep,
            desc=f"launch {method.name}",
            unit="replica",
            disable=not sys.stderr.isatty(),
        ) as progress:
            for future in as_completed(future_to_index):
                replica_index = future_to_index[future]
                try:
                    launched[replica_index] = future.result()
                except Exception as exc:
                    errors.append(exc)
                progress.update(1)

    successful = [item for item in launched if item is not None]
    if errors:
        _stop_processes([process for process, _ in successful])
        raise RuntimeError(
            f"Failed to launch {len(errors)}/{n_rep} SGLang replicas. "
            f"See {cfg.logs_dir()} for server logs."
        ) from errors[0]
    return [item for item in launched if item is not None]


class SGLangHarness(Harness):
    name = "sglang"

    def _server_args(self, method: MethodSpec, cfg: BenchConfig) -> list[str]:
        # Each replica is a single-GPU (tp=1, dp=1) server. Data parallelism is
        # achieved by launching cfg.dp_size such replicas and round-robining
        # requests, which avoids sglang's DP multi-port rendezvous (that
        # collides with the host MPI watcher's low ports).
        args = [
            "--trust-remote-code",
            "--attention-backend", cfg.attention_backend,
            "--tp-size", str(cfg.tp_size),
            "--dtype", cfg.dtype,
            "--mem-fraction-static", str(cfg.mem_fraction_static),
            "--max-running-requests", str(cfg.max_running_requests),
            "--page-size", str(cfg.page_size),
            # Without this sglang picks a fresh random seed per launch. Repeats
            # re-seed over HTTP; this only pins the first one.
            "--random-seed", str(seed_for_repeat(1)),
        ]
        if cfg.disable_cuda_graph:
            args += ["--disable-cuda-graph", "--disable-piecewise-cuda-graph"]
        else:
            args += ["--cuda-graph-bs", *[str(i) for i in range(1, int(cfg.cuda_graph_max_bs) + 1)],
                     "--cuda-graph-max-bs", str(cfg.cuda_graph_max_bs)]
        if cfg.disable_radix_cache:
            args.append("--disable-radix-cache")

        algo = method.sglang_speculative_algorithm
        if algo:  # speculative method
            if not cfg.draft_model:
                raise ValueError(f"Method '{method.name}' requires --draft-model.")
            args += ["--speculative-algorithm", algo,
                     "--speculative-draft-model-path", cfg.draft_model]
            params = cfg.apply_spec_overrides(method.sglang_spec_params)
            for flag, key in (("--speculative-num-steps", "speculative_num_steps"),
                              ("--speculative-eagle-topk", "speculative_eagle_topk"),
                              ("--speculative-num-draft-tokens", "speculative_num_draft_tokens")):
                if key in params:
                    args += [flag, str(params[key])]
            if cfg.speculative_draft_temperature is not None:
                args += [
                    "--speculative-draft-temperature",
                    str(cfg.speculative_draft_temperature),
                ]
            if (
                cfg.speculative_use_rejection_sampling
                and algo in {"EAGLE", "EAGLE3"}
            ):
                args.append("--speculative-use-rejection-sampling")
        return args

    def run(self, method: MethodSpec, cfg: BenchConfig, done=None) -> list[BenchResult]:
        method.check_harness("sglang")
        done = done or set()
        expect_spec = method.sglang_speculative_algorithm is not None

        # Nothing left to do (resume): skip launching a server entirely.
        repeats = max(int(cfg.repeats), 1)
        pending = [(d, count, c, rep) for d, count in cfg.dataset_specs()
                   for c in cfg.concurrencies for rep in range(1, repeats + 1)
                   if (d, c, rep) not in done]
        if not pending:
            print(f"[specbench.sglang] {method.name}: all (dataset,conc,repeat) already done; skip.")
            return []

        tokenizer = AutoTokenizer.from_pretrained(cfg.target_model)

        def build_prompts(dataset_name: str, n: int) -> list[str]:
            ds = load_and_process_dataset(dataset_name)
            if len(ds) == 0:
                raise RuntimeError(f"Dataset {dataset_name!r} is empty.")
            prompts = []
            for i in range(min(len(ds), n)):
                prompts.append(tokenizer.apply_chat_template(
                    [{"role": "user", "content": ds[i]["turns"][0]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False))
            return prompts

        # Launch cfg.dp_size single-GPU replicas, one per visible GPU.
        n_rep = max(int(cfg.dp_size), 1)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpus = [g.strip() for g in visible.split(",") if g.strip()] if visible else []
        if gpus and len(gpus) < n_rep:
            print(f"[specbench.sglang] warning: dp_size={n_rep} but only {len(gpus)} "
                  f"GPU(s) visible ({visible}); launching {len(gpus)} replicas.")
            n_rep = len(gpus)

        server_args = self._server_args(method, cfg)
        procs: list[subprocess.Popen] = []
        base_urls: list[str] = []
        results: list[BenchResult] = []
        try:
            launched = _launch_replicas(method, cfg, server_args, n_rep, gpus)
            procs = [process for process, _ in launched]
            base_urls = [url for _, url in launched]
            print(f"[specbench.sglang] {method.name}: {len(base_urls)} replicas ready, "
                  f"GPU={gpus[:n_rep] or ['default']}, server logs={cfg.logs_dir()}")

            for url in base_urls:
                _send_generate(url, "Hello", max_new_tokens=8, timeout_s=min(cfg.timeout_s, 300),
                               temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)

            for dataset_name, count in cfg.dataset_specs():
                if all((dataset_name, c, rep) in done
                       for c in cfg.concurrencies for rep in range(1, repeats + 1)):
                    print(f"[specbench.sglang] skip dataset {dataset_name} (already done)")
                    continue
                timed = build_prompts(dataset_name, count)
                if not timed:
                    continue
                for conc in cfg.concurrencies:
                    global_concurrency = conc * len(base_urls)
                    for repeat in range(1, repeats + 1):
                        if (dataset_name, conc, repeat) in done:
                            print(f"[specbench.sglang] skip {dataset_name} conc={conc} "
                                  f"repeat={repeat} (already done)")
                            continue
                        # Flush per repeat, not per (dataset,conc): repeats reuse the
                        # exact same prompt set, so a warm radix cache would let a
                        # later repeat serve whole prefixes from cache. Re-seeding
                        # here (rather than only at launch) makes repeat r replay
                        # the same sampling whether or not r-1 ran in this process.
                        seed = seed_for_repeat(repeat)
                        for url in base_urls:
                            _flush_cache(url)
                            _set_random_seed(url, seed)
                        warmup_pool = timed  # single fixed prompt set; cycle for warmup/skip
                        warmup_prompts = _cycle_take(
                            warmup_pool, cfg.warmup_requests * global_concurrency
                        )
                        skip_prompts = _cycle_take(
                            warmup_pool, cfg.skip_first_requests * len(base_urls)
                        )
                        rep_desc = f" rep={repeat}/{repeats}" if repeats > 1 else ""
                        m, samples = _run_requests(
                            base_urls, prompts=timed, max_new_tokens=cfg.max_new_tokens,
                            concurrency=conc, timeout_s=cfg.timeout_s, expect_spec=expect_spec,
                            temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
                            warmup_prompts=warmup_prompts, warmup_max_new_tokens=cfg.warmup_max_new_tokens,
                            skip_prompts=skip_prompts,
                            rid_prefix=(
                                f"{method.name}|{dataset_name}|c{conc}|r{repeat}"
                            ),
                            progress_desc=(
                                f"{method.name} {dataset_name} "
                                f"c={conc}/replica (global={global_concurrency}){rep_desc}"
                            ))
                        br = BenchResult(
                            harness=self.name, method=method.name, dataset=dataset_name,
                            tokens_per_second=m.per_request_toks_per_s,
                            accept_length=m.spec_accept_length,
                            num_samples=len(timed), concurrency=conc, repeat=repeat,
                            target_model=cfg.target_model,
                            draft_model=(cfg.draft_model if expect_spec else None),
                            backend=cfg.attention_backend, temperature=cfg.temperature,
                            max_new_tokens=cfg.max_new_tokens, latency_s=m.wall_latency_s,
                            extra={"output_tokens": m.output_tokens,
                                   "spec_verify_ct_sum": m.spec_verify_ct_sum,
                                   "wall_toks_per_s": m.wall_toks_per_s,
                                   "replicas": len(base_urls),
                                   "dp_size": cfg.dp_size,
                                   "concurrency_per_replica": conc,
                                   "global_concurrency": global_concurrency,
                                   "repeats": repeats,
                                   "random_seed": seed,
                                   "rejection_sampling": cfg.speculative_use_rejection_sampling})
                        # Responses are only kept for the last repeat: they are large
                        # and `accuracy` scores one set per (dataset, method).
                        if cfg.save_responses and repeat == repeats:
                            rpath = cfg.responses_file(dataset_name, method.name)
                            if rpath:
                                rpath.parent.mkdir(parents=True, exist_ok=True)
                                rpath.unlink(missing_ok=True)
                                append_jsonl(rpath, ({
                                    "harness": self.name, "method": method.name,
                                    "dataset": dataset_name,
                                    "draft_model": (cfg.draft_model if expect_spec else None),
                                    "concurrency": conc,
                                    "concurrency_per_replica": conc,
                                    "global_concurrency": global_concurrency,
                                    "repeat": repeat,
                                    "temperature": cfg.temperature, **s} for s in samples))
                        # Commit the result row incrementally so resume works at
                        # (dataset,concurrency,repeat) granularity even if a later
                        # unit crashes.
                        write_results(cfg.results_file(), [br], append=True)
                        results.append(br)
                        acc = f"{m.spec_accept_length:.3f}" if m.spec_accept_length is not None else "n/a"
                        print(f"[specbench.sglang] {method.name} {dataset_name} "
                              f"conc/replica={conc} global_conc={global_concurrency} "
                              f"repeat={repeat}/{repeats} "
                              f"per_req_tok/s={m.per_request_toks_per_s:,.2f} "
                              f"wall_tok/s={m.wall_toks_per_s:,.2f} accept_len={acc}")
            # After all datasets, dump accumulated spec profile (only when profiling enabled).
            _profile_env = os.environ.get("SGLANG_SPEC_DEBUG_DUMP") or os.environ.get("SGLANG_DSPARK_DEBUG_DUMP")
            if _profile_env:
                _key_map = {
                    "DSPARK": "dspark_info_record",
                    "DFLASH": "dflash_info_record",
                    "EAGLE3": "eagle_info_record",
                    "EAGLE": "eagle_info_record",
                }
                _record_key = _key_map.get(method.sglang_speculative_algorithm)
                if _record_key:
                    _dump_spec_profile(
                        base_urls, cfg.profiles_file(method.name), key=_record_key
                    )
        finally:
            _stop_processes(procs)
        return results
