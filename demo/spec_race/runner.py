"""Run one prompt through several methods sequentially and record the race.

Methods run one at a time on the single GPU, so each gets the whole device and
the recorded per-step timestamps are directly comparable. The browser replays
all recordings from t=0 side by side, which shows the race as if the methods
had run concurrently on separate GPUs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Awaitable, Callable, Optional

from . import registry
from .engine import (
    IncrementalDetokenizer,
    ServerPool,
    TokenizerCache,
    plan_gpus,
    score_entropy,
    stream_generate,
    summarize,
)
from .registry import BASELINE

Emit = Callable[[dict], Awaitable[None]]

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"


def new_recording(model: str, budget: int, preset: str, prompt: str,
                  max_new_tokens: int, seed: Optional[int], methods: list[str]) -> dict:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target_t, draft_t = registry.TEMPERATURE_PRESETS[preset]
    return {
        "id": f"{stamp}_{model}_b{budget}_t{preset.replace(':', '-')}_{digest}",
        "created": time.time(),
        "model": model,
        "budget": int(budget),
        "preset": preset,
        "target_temperature": target_t,
        "draft_temperature": draft_t,
        "prompt": prompt,
        "max_new_tokens": int(max_new_tokens),
        "seed": seed,
        "methods": methods,
        "runs": {},
    }


def save_recording(rec: dict) -> Path:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = RECORDINGS_DIR / f"{rec['id']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def list_recordings() -> list[dict]:
    out = []
    if not RECORDINGS_DIR.exists():
        return out
    for p in sorted(RECORDINGS_DIR.glob("*.json"), reverse=True):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "id": rec["id"], "model": rec["model"], "budget": rec["budget"],
            "preset": rec["preset"], "prompt": rec["prompt"][:160],
            "created": rec.get("created"),
            "methods": [m for m, r in rec.get("runs", {}).items() if r.get("status") == "done"],
        })
    return out


def load_recording(rec_id: str) -> Optional[dict]:
    path = (RECORDINGS_DIR / f"{rec_id}.json").resolve()
    if path.parent != RECORDINGS_DIR.resolve() or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


async def run_one(pool: ServerPool, toks: TokenizerCache, rec: dict, method: str,
                  emit: Emit, input_ids: Optional[list[int]] = None) -> dict:
    """Run `rec`'s prompt with one method; stores and returns the run entry."""
    model, budget, preset = rec["model"], rec["budget"], rec["preset"]

    async def status(msg: str, state: str = "launching") -> None:
        await emit({"type": "status", "method": method, "state": state, "message": msg})

    run = {"method": method, "status": "error", "steps": [], "summary": None, "error": None}
    rec["runs"][method] = run
    try:
        if method != BASELINE:
            ckpt = registry.checkpoint_path(model, method, budget, download=True)
            if ckpt is None or not ckpt.exists():
                raise FileNotFoundError(f"checkpoint missing: {ckpt}")
        spec = registry.server_spec(model, method, budget, preset)
        run["server"] = spec.key
        srv = await pool.acquire(spec, on_status=status)
        if input_ids is None:
            input_ids = toks.encode_prompt(model, rec["prompt"])
        detok = IncrementalDetokenizer(toks.get(model))
        params = registry.sampling_params(preset, rec["max_new_tokens"])
        await status("generating", "running")
        e2e, final_meta = 0.0, {}
        async for ev in stream_generate(srv, input_ids, params, detok,
                                        registry.verify_window(method, budget),
                                        seed=rec.get("seed")):
            if "final" in ev:
                e2e, final_meta = ev["e2e"], ev["final"]
                continue
            run["steps"].append(ev)
            await emit({"type": "step", "method": method, **ev})
        run["summary"] = summarize(run["steps"], e2e, final_meta)
        run["status"] = "done"
        await emit({"type": "done", "method": method, "summary": run["summary"]})
    except Exception as exc:  # keep racing the remaining methods
        run["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        await emit({"type": "error", "method": method, "message": run["error"]})
    return run


def ordered_methods(methods: list[str], baseline: bool) -> list[str]:
    """Race order: baseline (Target AR) first, then EAGLE-3, then the rest.

    Non-baseline methods keep their `registry.METHODS` relative order; EAGLE-3
    is pulled right after the baseline so the race reads
    [Target AR, EAGLE-3, DFlash, DFlare, ...]. Without a baseline EAGLE-3
    leads the field. Mirrored by buildCells() in the frontend.
    """
    known = [m["key"] for m in registry.METHODS]
    picked = [m for m in known if m in methods]
    if "eagle3" in picked:
        picked = ["eagle3"] + [m for m in picked if m != "eagle3"]
    if not baseline:
        return picked
    return [BASELINE] + picked


def entropy_stats(steps: list[dict], ent: list, window: int, is_baseline: bool) -> dict:
    """Mean target entropy of accepted drafts vs. the token at the rejection point.

    In a verify step [d1..dk, b], d* are accepted drafts and b is the target's
    own token. If the step is shorter than the window, b sits exactly where the
    draft proposal was rejected; otherwise everything was accepted ("full").
    The last step is skipped because EOS / max_new_tokens may truncate it.
    """
    acc, rej, full, every = [], [], [], []
    i = 0
    last = len(steps) - 1
    for n_step, s in enumerate(steps):
        n = len(s["ids"])
        h = [x for x in ent[i:i + n]]
        i += n
        every += [x for x in h if x is not None]
        if is_baseline or s["kind"] != "verify" or n_step == last:
            continue
        acc += [x for x in h[:-1] if x is not None]
        if h and h[-1] is not None:
            (full if n >= window else rej).append(h[-1])

    def mean(v):
        return round(sum(v) / len(v), 4) if v else None

    return {"all": mean(every), "accepted": mean(acc), "rejected": mean(rej),
            "full_bonus": mean(full), "n_all": len(every), "n_accepted": len(acc),
            "n_rejected": len(rej), "n_full_bonus": len(full)}


async def score_recording(pool: ServerPool, toks: TokenizerCache, rec: dict, emit: Emit,
                          top_k: int = 20, input_ids: Optional[list[int]] = None) -> None:
    """Attach per-token target entropy to every finished run of `rec`.

    Uses the target-only server (same weights as every method's verifier), so
    all methods are scored by the identical distribution.
    """
    runs = [r for r in rec["runs"].values() if r.get("status") == "done" and r.get("steps")]
    todo = [r for r in runs if "entropy" not in r]
    if not todo:
        return
    model = rec["model"]

    async def status(msg: str, state: str = "launching") -> None:
        await emit({"type": "phase", "message": f"entropy scorer: {msg}"})

    spec = registry.server_spec(model, BASELINE, rec["budget"], rec["preset"])
    srv = await pool.acquire(spec, on_status=status)
    if input_ids is None:
        input_ids = toks.encode_prompt(model, rec["prompt"])
    rec["entropy"] = {"scorer": spec.key, "temperature": 1.0, "top_k": int(top_k),
                      "estimator": "top-k + tail lower bound", "unit": "nats"}
    for run in todo:
        method = run["method"]
        await emit({"type": "phase", "message": f"scoring target entropy: {method}"})
        try:
            out_ids = [t for s in run["steps"] for t in s["ids"]]
            ent = await score_entropy(srv, input_ids, out_ids, top_k)
            run["entropy"] = ent
            run["entropy_stats"] = entropy_stats(
                run["steps"], ent, registry.verify_window(method, rec["budget"]),
                method == BASELINE)
            await emit({"type": "entropy", "method": method, "entropy": ent,
                        "stats": run["entropy_stats"]})
        except Exception as exc:  # a scoring failure must not lose the recording
            traceback.print_exc()
            await emit({"type": "phase", "message": f"entropy failed for {method}: {exc}"})


async def run_race(pool: ServerPool, toks: TokenizerCache, model: str, budget: int,
                   preset: str, prompt: str, max_new_tokens: int, seed: Optional[int],
                   methods: list[str], baseline: bool, emit: Emit,
                   entropy: bool = False, entropy_top_k: int = 20,
                   gpus: str = "auto") -> dict:
    order = ordered_methods(methods, baseline)
    occupied = [s.gpu for s in pool.servers.values() if s.alive()]
    devices, parallel = plan_gpus(gpus, len(order), occupied=occupied)
    pool.configure(devices, parallel)
    rec = new_recording(model, budget, preset, prompt, max_new_tokens, seed, order)
    rec["parallel"] = parallel
    rec["gpus"] = devices
    await emit({"type": "start", "recording": {k: v for k, v in rec.items() if k != "runs"}})
    if parallel:
        msg = f"parallel on GPUs {devices} (one server per GPU)"
    elif len(devices) > 1:
        msg = (f"serial on GPUs {devices} ({len(devices)} GPUs for "
               f"{len(order)} methods)")
    else:
        msg = (f"serial on GPU {devices[0]} (fewer free GPUs than "
               f"{len(order)} methods)")
    await emit({"type": "phase", "message": msg})
    input_ids = toks.encode_prompt(model, prompt)
    rec["prompt_tokens"] = len(input_ids)
    try:
        if parallel:
            await asyncio.gather(*(
                run_one(pool, toks, rec, m, emit, input_ids=input_ids) for m in order))
        else:
            for method in order:
                await run_one(pool, toks, rec, method, emit, input_ids=input_ids)
                save_recording(rec)
        if entropy:
            try:
                await score_recording(pool, toks, rec, emit, entropy_top_k,
                                      input_ids=input_ids)
            except Exception as exc:
                traceback.print_exc()
                await emit({"type": "phase", "message": f"entropy scoring failed: {exc}"})
    finally:
        save_recording(rec)  # keep partial runs even when the race is cancelled
    await emit({"type": "finished", "recording_id": rec["id"]})
    return rec
