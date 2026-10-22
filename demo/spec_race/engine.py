"""SGLang server lifecycle + per-verify-step streaming for the race demo.

One `sglang serve` process per (model, method, budget, draft temperature):
draft models and draft temperature are fixed at launch. Devices are assigned
round-robin, one server per GPU when the pool has enough; otherwise every
server shares the first device and the pool keeps at most `max_resident`
servers alive (default 1), evicting the least recently used one first.

With `--stream-interval 1 --incremental-streaming-output` every SSE chunk of
`/generate` carries exactly the tokens committed by one verify step:
`[accepted drafts..., bonus]`. The first chunk is the prefill token.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import zlib
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Optional

import aiohttp

from . import registry
from .registry import BASELINE, ServerSpec

StatusCb = Callable[[str], Awaitable[None]]

DEMO_DIR = Path(__file__).resolve().parent
LOG_DIR = DEMO_DIR / "logs"


class EngineOptions:
    def __init__(
        self,
        gpu: str = "0",
        max_resident: int = 1,
        mem_fraction: float = 0.8,
        attention_backend: str = "flashinfer",
        dtype: str = "bfloat16",
        launch_timeout: float = 1800.0,
        cuda_graph_max_bs: int = 4,
        disable_cuda_graph: bool = False,
        extra_server_args: Optional[list[str]] = None,
        gpus: Optional[list[int]] = None,
        gpus_spec: str = "auto",
    ):
        self.gpu = gpu
        self.gpus = gpus  # when set, each server is pinned to one of these devices
        self.gpus_spec = gpus_spec  # "auto" | "0" | "0,1,2" | "0-7", resolved per race
        self.max_resident = max(1, int(max_resident))
        self.mem_fraction = mem_fraction
        self.attention_backend = attention_backend
        self.dtype = dtype
        self.launch_timeout = launch_timeout
        self.cuda_graph_max_bs = cuda_graph_max_bs
        self.disable_cuda_graph = disable_cuda_graph
        self.extra_server_args = list(extra_server_args or [])


def _free_port(exclude: frozenset[int] = frozenset()) -> int:
    """A port that is free on every interface right now.

    sglang's HTTP listener binds 127.0.0.1 (--host) while torch's TCPStore
    (`--nccl-port`) binds the wildcard / the machine IP, so a port held on a
    non-loopback interface (a leftover server, a derived port, another
    service) is invisible to a 127.0.0.1 probe but kills the TCPStore bind
    with EADDRINUSE. Hence probe with an ephemeral wildcard bind.

    `exclude` guards against handing out the same port twice in a row: the
    socket is closed before sglang binds it, so a second call can return the
    very port just freed.
    """
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            p = s.getsockname()[1]
        if p not in exclude:
            return p


def nvidia_device_count() -> int:
    """Number of visible GPUs, or 0 when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return 0
        return len([l for l in out.stdout.splitlines() if l.strip()])
    except (OSError, subprocess.SubprocessError):
        return 0


def free_gpus(min_free_mb: int = 1024) -> list[int]:
    """Indices of GPUs currently not holding model memory (for `--gpus auto`)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    idx = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            try:
                if int(parts[1]) < min_free_mb:
                    idx.append(int(parts[0]))
            except ValueError:
                continue
    return idx


def parse_gpu_spec(spec: str) -> list[int]:
    """Parse '0', '0,1,2' or '0-3' into a device list."""
    devs: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            if not (lo.isdigit() and hi.isdigit()):
                raise ValueError(f"bad --gpus range: {part}")
            devs.extend(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            devs.append(int(part))
        else:
            raise ValueError(f"bad --gpus value: {part}")
    return devs


def plan_gpus(spec: str, n_servers: int,
              occupied: Optional[list[int]] = None) -> tuple[list[int], bool]:
    """Decide the device list and whether to run the race in parallel.

    `auto` picks the currently free GPUs plus the devices that already host a
    resident server of this pool (they can be reused / evicted by the LRU);
    parallel only when at least `n_servers` devices are available, otherwise
    everything runs serially on the first device.
    """
    if n_servers <= 0:
        return [0], False
    if spec and spec != "auto":
        devs = parse_gpu_spec(spec)
        if not devs:
            raise ValueError("--gpus must name at least one device")
        if len(devs) >= n_servers:
            return devs[:n_servers], True
        return devs, False
    occ = set(occupied or [])
    avail = sorted(set(free_gpus()) | occ)
    if len(avail) >= n_servers:
        return avail[:n_servers], True
    return [avail[0]] if avail else [0], False


def _sglang_cmd() -> list[str]:
    exe = shutil.which("sglang")
    if exe:
        return [exe, "serve"]
    return [sys.executable, "-m", "sglang.launch_server"]


def build_server_command(
    spec: ServerSpec, opts: EngineOptions, port: int, nccl_port: int, gpu: int = 0
) -> tuple[list[str], dict[str, str]]:
    """Build the sglang command line. `port`/`nccl_port` are chosen by the
    caller (ServerPool._pick_port) so every handed-out port is tracked and no
    two servers can claim the same one for the HTTP listener and TCPStore.
    """
    target = registry.target_path(spec.model, download=True)
    args = [
        *_sglang_cmd(),
        "--model-path", str(target),
        "--trust-remote-code",
        "--attention-backend", opts.attention_backend,
        "--tp-size", "1",
        "--dtype", opts.dtype,
        "--mem-fraction-static", str(opts.mem_fraction),
        "--max-running-requests", "4",
        "--page-size", "1",
        "--random-seed", "1",
        "--stream-interval", "1",
        "--incremental-streaming-output",
        "--nccl-port", str(nccl_port),
    ]
    if opts.disable_cuda_graph:
        args += ["--disable-cuda-graph", "--disable-piecewise-cuda-graph"]
    else:
        bs = int(opts.cuda_graph_max_bs)
        args += ["--cuda-graph-bs", *[str(i) for i in range(1, bs + 1)],
                 "--cuda-graph-max-bs", str(bs)]

    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    if spec.method != BASELINE:
        meta = registry.METHOD_BY_KEY[spec.method]
        ckpt = registry.checkpoint_path(spec.model, spec.method, spec.budget,
                                        download=True)
        if ckpt is None:
            raise ValueError(f"No checkpoint registered for {spec.key}")
        args += [
            "--speculative-algorithm", meta["algorithm"],
            "--speculative-draft-model-path", str(ckpt),
            "--speculative-draft-temperature", str(spec.draft_temperature),
        ]
        if meta["algorithm"] == "EAGLE3":
            steps, topk, ndt = registry.EAGLE3_TREE[spec.budget]
            args += ["--speculative-num-steps", str(steps),
                     "--speculative-eagle-topk", str(topk),
                     "--speculative-num-draft-tokens", str(ndt)]
            if spec.rejection_sampling:
                args.append("--speculative-use-rejection-sampling")
        if meta["algorithm"] == "DSPARK":
            env["SGLANG_RAGGED_VERIFY_MODE"] = "static"
    args += [*opts.extra_server_args, "--host", "127.0.0.1", "--port", str(port)]
    return args, env


class ManagedServer:
    def __init__(self, spec: ServerSpec, proc: subprocess.Popen, port: int, log_path: Path,
                 gpu: int = 0):
        self.spec = spec
        self.proc = proc
        self.port = port
        self.gpu = gpu
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.last_used = time.time()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def info(self) -> dict:
        return {"key": self.spec.key, "pid": self.proc.pid, "url": self.base_url,
                "alive": self.alive(), "log": str(self.log_path),
                "gpu": self.gpu}


def _tail(path: Path, n: int = 30) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=n))
    except OSError:
        return ""


class ServerPool:
    def __init__(self, opts: EngineOptions):
        self.opts = opts
        self.servers: dict[str, ManagedServer] = {}
        self._lock = asyncio.Lock()
        self.gpus: list[int] = opts.gpus or [0]
        self.parallel = False
        self._cap = max(1, opts.max_resident)  # resident + in-flight launches
        self._launch_seq = 0  # round-robin device assignment
        self._launching: dict[str, asyncio.Task] = {}  # one launch task per spec key
        self._launching_gpus: set[int] = set()  # devices a launch task is loading onto
        self._procs: set = set()  # every live server process (incl. half-launched)
        self._recent_ports: dict[int, float] = {}  # ports handed out recently

    def configure(self, gpus: list[int], parallel: bool) -> None:
        """Set the devices and whether the race runs concurrently (one per GPU)."""
        self.gpus = [int(g) for g in gpus] or [0]
        self.parallel = bool(parallel)
        # In parallel mode every planned device hosts exactly one server, so the
        # LRU cap must not evict them; a user-set max_resident is the serial floor.
        self._cap = max(self.opts.max_resident, len(self.gpus))

    def _gpu_for(self, spec: ServerSpec) -> int:
        """Next device in round-robin order, preferring a device with no live
        or in-flight server; when every device is taken the resident server of
        the chosen device is replaced (see `_launch`)."""
        busy = {s.gpu for s in self.servers.values() if s.alive()} | self._launching_gpus
        free = [g for g in self.gpus if g not in busy]
        pool = free or self.gpus
        gpu = pool[self._launch_seq % len(pool)]
        self._launch_seq += 1
        return gpu

    def _pick_port(self) -> int:
        now = time.time()
        self._recent_ports = {p: t for p, t in self._recent_ports.items() if now - t < 120}
        while True:
            p = _free_port(frozenset(self._recent_ports))
            if p not in self._recent_ports:
                self._recent_ports[p] = now
                return p

    async def acquire(self, spec: ServerSpec, on_status: Optional[StatusCb] = None) -> ManagedServer:
        """Return a running server for `spec`, launching one if necessary.

        Servers live for the whole demo process (keyed by model/method/budget/
        preset, not by prompt), so a re-run with a different prompt reuses the
        already-loaded weights. The launch itself runs outside the lock so
        several servers can load their weights concurrently on different GPUs.
        """
        async with self._lock:
            srv = self.servers.get(spec.key)
            if srv is not None and srv.alive():
                srv.last_used = time.time()
                if on_status:
                    # Reuse, not reload: weights are still in memory.
                    await on_status(f"warm server (pid {srv.proc.pid}), reusing weights")
                return srv
            task = self._launching.get(spec.key)
            if task is None:
                while len(self.servers) + len(self._launching) >= self._cap:
                    lru = min(self.servers.values(), key=lambda s: s.last_used)
                    if on_status:
                        await on_status(f"stopping {lru.spec.key} to free GPU memory")
                    await self._stop(lru)
                task = asyncio.ensure_future(self._launch_registered(spec, on_status))
                self._launching[spec.key] = task
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # The caller is gone: stop the launch (it holds GPU memory) and
            # re-raise; the finally-block below cleans up the task entry.
            task.cancel()
            try:
                await task
            except BaseException:
                pass
            raise
        finally:
            async with self._lock:
                if task.done():
                    self._launching.pop(spec.key, None)

    async def _launch_registered(self, spec: ServerSpec,
                                 on_status: Optional[StatusCb]) -> ManagedServer:
        srv = await self._launch(spec, on_status)
        async with self._lock:
            self.servers[spec.key] = srv
        return srv

    async def _launch(self, spec: ServerSpec, on_status: Optional[StatusCb]) -> ManagedServer:
        # Ports are only free at pick time; a dying previous server or lingering
        # TIME_WAIT sockets can still take one before sglang binds it. Retry
        # with fresh ports when the log shows the classic EADDRINUSE. Both the
        # HTTP port and the TCPStore (--nccl-port) port go through
        # `_pick_port`, so no server can claim the same port twice and every
        # retry draws a port that has not been handed out recently.
        last_exc: Optional[BaseException] = None
        for attempt in range(3):
            port = self._pick_port()
            nccl_port = self._pick_port()
            gpu = self._gpu_for(spec)
            self._launching_gpus.add(gpu)
            try:
                # One server per device: if a resident server already holds
                # this device (e.g. the same method at another draft
                # temperature), evict it first. The LRU eviction in `acquire`
                # targets an arbitrary server, whose device may not be this
                # one, so loading weights into an occupied device OOMs.
                for srv in [s for s in self.servers.values()
                            if s.gpu == gpu and s.alive()]:
                    if on_status:
                        await on_status(f"stopping {srv.spec.key} to free GPU {gpu}")
                    await self._stop(srv)
                cmd, env_over = build_server_command(spec, self.opts, port, nccl_port, gpu)
                env = os.environ.copy()
                env.update(env_over)
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                log_path = LOG_DIR / (spec.key.replace("/", "__") + ".log")
                log_f = log_path.open("a", encoding="utf-8")
                log_f.write(f"\n$ {gpu=} {' '.join(cmd)}\n")
                log_f.flush()
                # New session -> one process group, so the scheduler/detokenizer
                # children die together with the launcher on killpg.
                proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                log_f.close()
                self._procs.add(proc)
                srv = ManagedServer(spec, proc, port, log_path, gpu)
                try:
                    await self._wait_ready(srv, on_status)
                    return srv
                except BaseException as exc:
                    # Includes cancellation: never leave a half-launched server on the GPU.
                    await asyncio.shield(self._stop(srv))
                    last_exc = exc
                    if (isinstance(exc, Exception) and attempt < 2
                            and self._port_conflict(log_path)):
                        if on_status:
                            await on_status(f"port collision, retrying {spec.key} "
                                            f"({attempt + 1}/3)")
                        await asyncio.sleep(5)
                        continue
                    raise
            finally:
                self._launching_gpus.discard(gpu)

    @staticmethod
    def _port_conflict(log_path: Path) -> bool:
        try:
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                size = max(0, f.tell() - 4096)
                f.seek(size)
                tail = f.read().decode("utf-8", "replace")
        except OSError:
            return False
        return "EADDRINUSE" in tail or "already in use" in tail

    async def _wait_ready(self, srv: ManagedServer, on_status: Optional[StatusCb]) -> None:
        spec, proc, log_path = srv.spec, srv.proc, srv.log_path
        if on_status:
            await on_status(f"launching sglang (pid {proc.pid}, port {srv.port})")

        t0 = time.time()
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                if not srv.alive():
                    raise RuntimeError(
                        f"sglang exited with code {proc.returncode} while launching "
                        f"{spec.key}. Log {log_path}:\n{_tail(log_path)}")
                elapsed = time.time() - t0
                if elapsed > self.opts.launch_timeout:
                    raise TimeoutError(f"Timed out launching {spec.key}; see {log_path}")
                try:
                    async with session.get(srv.base_url + "/health") as r:
                        if r.status == 200:
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                if on_status and int(elapsed) % 10 < 2:
                    await on_status(f"loading weights / capturing cuda graphs ({elapsed:.0f}s)")
                await asyncio.sleep(2)

        if on_status:
            await on_status(f"warming up ({time.time() - t0:.0f}s)")
        warm = {"text": "Hello", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            async with session.post(srv.base_url + "/generate", json=warm) as r:
                r.raise_for_status()
                await r.read()

    async def _stop(self, srv: ManagedServer) -> None:
        self.servers.pop(srv.spec.key, None)
        self._procs.discard(srv.proc)
        if srv.alive():
            try:
                os.killpg(srv.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.to_thread(srv.proc.wait, 30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(srv.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.to_thread(srv.proc.wait)
        # Orphaned workers may still be exiting; give the driver time to free memory.
        await asyncio.sleep(3)

    async def stop_all(self) -> None:
        for t in list(self._launching.values()):
            t.cancel()
        for t in list(self._launching.values()):
            try:
                await t
            except BaseException:
                pass
        self._launching.clear()
        async with self._lock:
            for srv in list(self.servers.values()):
                await self._stop(srv)

    def kill_all_sync(self) -> None:
        """Best-effort cleanup for atexit / signal handlers."""
        for proc in list(self._procs):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self._procs.clear()
        self.servers.clear()
        self._launching.clear()

    def info(self) -> list[dict]:
        return [s.info() for s in self.servers.values()]


class IncrementalDetokenizer:
    """Per-token text pieces that stay correct across multi-byte boundaries.

    A token whose bytes do not complete a character yields "" and the text is
    attributed to the token that completes it.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids: list[int] = []
        self.prefix = 0
        self.read = 0

    def push(self, tid: int) -> str:
        self.ids.append(int(tid))
        dec = lambda ids: self.tok.decode(ids, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        prefix_text = dec(self.ids[self.prefix:self.read])
        new_text = dec(self.ids[self.prefix:])
        if len(new_text) > len(prefix_text) and not new_text.endswith("\ufffd"):
            self.prefix, self.read = self.read, len(self.ids)
            return new_text[len(prefix_text):]
        return ""


_SPEC_META_KEYS = (
    "completion_tokens", "prompt_tokens", "finish_reason", "e2e_latency",
    "spec_verify_ct", "spec_accept_length", "spec_accept_rate",
    "spec_num_correct_drafts", "spec_num_proposed_drafts",
)


async def _post_quiet(session: aiohttp.ClientSession, url: str, **kw) -> None:
    try:
        async with session.request(kw.pop("method", "POST"), url, **kw) as r:
            await r.read()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        pass


async def stream_generate(
    srv: ManagedServer,
    input_ids: list[int],
    params: dict,
    detok: IncrementalDetokenizer,
    window: int,
    seed: Optional[int] = None,
) -> AsyncIterator[dict]:
    """Yield one event per committed step, then a final {"final": meta} event.

    Step event: {"t": seconds since request sent, "ids": [...], "pieces": [...],
    "kind": "prefill" | "verify" | "merged"}. "merged" means the tokenizer
    manager coalesced several steps into one chunk (event-loop backlog), so
    the step boundary is unknown.
    """
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout, read_bufsize=1 << 22) as session:
        # Same starting conditions for every method: empty prefix cache, same RNG seed.
        await _post_quiet(session, srv.base_url + "/flush_cache", method="GET")
        if seed is not None:
            await _post_quiet(session, srv.base_url + "/set_internal_state",
                              json={"server_args": {"random_seed": int(seed)}})
        payload = {"input_ids": input_ids, "sampling_params": params, "stream": True}
        n_steps = 0
        final_meta: dict = {}
        t0 = time.perf_counter()
        async with session.post(srv.base_url + "/generate", json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(f"/generate HTTP {resp.status}: {await resp.text()}")
            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(f"sglang error: {chunk['error']}")
                t = time.perf_counter() - t0
                meta = chunk.get("meta_info") or {}
                final_meta = {k: meta[k] for k in _SPEC_META_KEYS if k in meta}
                ids = [int(x) for x in (chunk.get("output_ids") or [])]
                if not ids:
                    continue
                if n_steps == 0:
                    kind = "prefill" if len(ids) == 1 else "merged"
                else:
                    kind = "verify" if len(ids) <= window else "merged"
                n_steps += 1
                yield {"t": round(t, 5), "ids": ids,
                       "pieces": [detok.push(i) for i in ids], "kind": kind}
        yield {"final": final_meta, "e2e": round(time.perf_counter() - t0, 5)}


def topk_entropy(logprobs: list[float]) -> float:
    """Entropy (nats) of a distribution known only through its top-k logprobs.

    The exact top-k term plus a lower bound for the tail: every tail token has
    p <= p_k, so the tail contributes at least r * -log(p_k), where r is the
    leftover mass. With k=20 the tail is usually tiny for an LLM.
    """
    lps = [lp for lp in logprobs if lp is not None and math.isfinite(lp)]
    if not lps:
        return 0.0
    h, mass = 0.0, 0.0
    for lp in lps:
        p = math.exp(lp)
        h -= p * lp
        mass += p
    rest = 1.0 - mass
    if rest > 1e-6:
        h += rest * -min(lps)
    return max(0.0, h)


async def score_entropy(srv: ManagedServer, prompt_ids: list[int], output_ids: list[int],
                        top_k: int = 20) -> list[Optional[float]]:
    """Target entropy H(p(. | prefix)) at T=1 for every committed output token.

    One prefill-only request to a target-only server: input logprobs come from
    the raw logits (no temperature), and the distribution at position P+j-1
    is the one the target used to commit output token j.
    """
    if not output_ids:
        return []
    start = len(prompt_ids) - 1  # entry 0 is always None, entry j+1 -> output token j
    payload = {
        "input_ids": list(prompt_ids) + list(output_ids),
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": start,
        "top_logprobs_num": int(top_k),
        "return_text_in_logprobs": False,
    }
    timeout = aiohttp.ClientTimeout(total=600, sock_connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(srv.base_url + "/generate", json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(f"entropy /generate HTTP {resp.status}: {await resp.text()}")
            out = await resp.json()
    if isinstance(out, list):
        out = out[0]
    meta = out.get("meta_info") or {}
    tops = meta.get("input_top_logprobs") or []
    ids = [e[1] if e else None for e in (meta.get("input_token_logprobs") or [])]
    full = payload["input_ids"]
    # The server may move logprob_start_len (e.g. multimodal); locate it from the ids.
    first = len(full) - len(tops)
    if first < 0 or (ids and ids[-1] is not None and ids[-1] != full[-1]):
        raise RuntimeError("unexpected input_top_logprobs layout from sglang")
    ents: list[Optional[float]] = []
    for j in range(len(output_ids)):
        k = len(prompt_ids) + j - first
        row = tops[k] if 0 <= k < len(tops) else None
        ents.append(round(topk_entropy([e[0] for e in row]), 4) if row else None)
    return ents


def summarize(steps: list[dict], e2e: float, final_meta: dict) -> dict:
    n_tokens = sum(len(s["ids"]) for s in steps)
    verify = [s for s in steps if s["kind"] != "prefill"]
    ttft = steps[0]["t"] if steps else None
    decode_time = (e2e - ttft) if ttft is not None else None
    decode_tokens = sum(len(s["ids"]) for s in verify)
    return {
        "n_tokens": n_tokens,
        "n_steps": len(verify),
        "tau": (decode_tokens / len(verify)) if verify else None,
        "ttft": ttft,
        "e2e": e2e,
        "tok_s": n_tokens / e2e if e2e > 0 else None,
        "decode_tok_s": (decode_tokens / decode_time) if decode_time else None,
        "merged_steps": sum(1 for s in steps if s["kind"] == "merged"),
        "server_meta": final_meta,
    }


class TokenizerCache:
    def __init__(self):
        self._toks: dict[str, object] = {}

    def get(self, model: str):
        if model not in self._toks:
            from transformers import AutoTokenizer

            self._toks[model] = AutoTokenizer.from_pretrained(
                str(registry.target_path(model, download=True)), trust_remote_code=True)
        return self._toks[model]

    def encode_prompt(self, model: str, prompt: str) -> list[int]:
        tok = self.get(model)
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tok.encode(text, add_special_tokens=False)
