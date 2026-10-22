"""Web UI for the speculative-decoding race demo.

    python -m demo.spec_race.app --gpu 0 --port 8099   (from the repo root)

or use demo/spec_race/run_demo.sh, which also sets up the environment.
NOTE: the HTTP API has no authentication. It binds 127.0.0.1 by default;
only pass --host 0.0.0.0 on a trusted network.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import prompts as prompts_mod
from . import registry, runner
from .engine import EngineOptions, ServerPool, TokenizerCache

STATIC_DIR = Path(__file__).resolve().parent / "static"
PROMPTS_FILE = Path(__file__).resolve().parent / "prompts.json"


class RunRequest(BaseModel):
    model: str
    budget: int = 16
    preset: str = "0:0"
    prompt: str = Field(min_length=1, max_length=20000)
    max_new_tokens: int = Field(default=8192, ge=1, le=32768)
    seed: Optional[int] = 0
    methods: list[str] = Field(default_factory=lambda: [m["key"] for m in registry.METHODS])
    baseline: bool = True
    entropy: bool = True  # score per-token target entropy after the race


def create_app(opts: EngineOptions, entropy_top_k: int = 20) -> FastAPI:
    pool = ServerPool(opts)
    toks = TokenizerCache()
    gpu_lock = asyncio.Lock()
    current: dict = {"task": None}
    atexit.register(pool.kill_all_sync)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await pool.stop_all()

    app = FastAPI(title="Speculative decoding race", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/config")
    async def config():
        prompts = []
        if PROMPTS_FILE.exists():
            prompts = prompts_mod.load_benchmarks(PROMPTS_FILE)
        return {
            "methods": registry.METHODS,
            "baseline": registry.BASELINE_META,
            "models": list(registry.MODELS),
            "budgets": list(registry.BUDGETS),
            "presets": list(registry.TEMPERATURE_PRESETS),
            "availability": registry.availability(),
            "prompts": prompts,
            "gpu": opts.gpu,
            "gpus": opts.gpus_spec,
            "max_resident": opts.max_resident,
        }

    @app.get("/api/recordings")
    async def recordings():
        return runner.list_recordings()

    @app.get("/api/recordings/{rec_id}")
    async def recording(rec_id: str):
        rec = runner.load_recording(rec_id)
        if rec is None:
            raise HTTPException(404, "recording not found")
        return rec

    @app.get("/api/servers")
    async def servers():
        return {"busy": gpu_lock.locked(), "servers": pool.info()}

    @app.post("/api/servers/stop")
    async def stop_servers():
        if gpu_lock.locked():
            raise HTTPException(409, "a run is in progress")
        await pool.stop_all()
        return {"ok": True}

    @app.post("/api/run")
    async def run(req: RunRequest, request: Request):
        if req.model not in registry.MODELS:
            raise HTTPException(400, f"unknown model {req.model}")
        if req.budget not in registry.BUDGETS:
            raise HTTPException(400, f"budget must be one of {registry.BUDGETS}")
        if req.preset not in registry.TEMPERATURE_PRESETS:
            raise HTTPException(400, f"unknown preset {req.preset}")
        unknown = set(req.methods) - set(registry.METHOD_BY_KEY)
        if unknown:
            raise HTTPException(400, f"unknown methods {sorted(unknown)}")
        busy = current["task"] is not None and not current["task"].done()
        if busy or gpu_lock.locked():
            return JSONResponse({"detail": "GPU busy: another run is in progress"}, status_code=409)

        queue: asyncio.Queue = asyncio.Queue()

        async def emit(ev: dict) -> None:
            await queue.put(ev)

        async def job():
            async with gpu_lock:
                try:
                    await runner.run_race(pool, toks, req.model, req.budget, req.preset,
                                          req.prompt, req.max_new_tokens, req.seed,
                                          req.methods, req.baseline, emit,
                                          entropy=req.entropy, entropy_top_k=entropy_top_k,
                                      gpus=opts.gpus_spec)
                except Exception as exc:
                    await emit({"type": "fatal", "message": f"{type(exc).__name__}: {exc}"})
                finally:
                    await queue.put(None)

        task = asyncio.create_task(job())
        current["task"] = task

        async def sse():
            # A closed browser tab does not cancel the run: it finishes and is
            # saved under recordings/. Use POST /api/cancel to abort.
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        break
                    yield ": keep-alive\n\n"
                    continue
                if ev is None:
                    break
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/cancel")
    async def cancel():
        task = current.get("task")
        if task is None or task.done():
            return {"cancelled": False}
        task.cancel()
        return {"cancelled": True}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099, help="web UI port")
    ap.add_argument("--gpu", default="0",
                    help="single device for serial mode (DEPRECATED, use --gpus)")
    ap.add_argument("--gpus", default=None,
                    help="devices for the sglang servers: default 0-7 (all devices, one per "
                         "method), or 0, 0,1,2, or auto (check free GPUs, fall back to serial "
                         "when there are fewer than methods)")
    ap.add_argument("--max-resident", type=int, default=1,
                    help="sglang servers kept alive at once (>1 needs a smaller --mem-fraction)")
    ap.add_argument("--mem-fraction", type=float, default=0.8)
    ap.add_argument("--attention-backend", default="flashinfer")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--cuda-graph-max-bs", type=int, default=4)
    ap.add_argument("--disable-cuda-graph", action="store_true")
    ap.add_argument("--launch-timeout", type=float, default=1800)
    ap.add_argument("--entropy-top-k", type=int, default=20,
                    help="top-k logprobs used to estimate the per-token target entropy")
    args = ap.parse_args()
    if args.gpus is None:
        # Default: one device per method on the full 8-GPU pool. An explicit
        # legacy `--gpu N` still means serial mode on device N.
        args.gpus = args.gpu if args.gpu != "0" else "0-7"
    opts = EngineOptions(
        gpu=args.gpu, max_resident=args.max_resident, mem_fraction=args.mem_fraction,
        attention_backend=args.attention_backend, dtype=args.dtype,
        launch_timeout=args.launch_timeout, cuda_graph_max_bs=args.cuda_graph_max_bs,
        disable_cuda_graph=args.disable_cuda_graph, gpus_spec=args.gpus,
    )
    uvicorn.run(create_app(opts, args.entropy_top_k), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
