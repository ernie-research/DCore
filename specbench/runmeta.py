"""Run identity + manifest helpers for experiment management.

A "run" is one coherent sweep and lives in its own directory named:

    {YYYYMMDD-HHMMSS}__{target}__{harness}__t{temp}[rs]__dp{dp}xc{conc}__{scale}[__{tag}]

The file name only carries the axes used to *distinguish/sort* runs; the full,
reproducible parameter set lives in ``manifest.json`` inside the directory.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


def slug(text: str, maxlen: int = 32) -> str:
    """Filesystem-safe short slug (keep the basename for model paths)."""
    base = os.path.basename(str(text).rstrip("/")) or str(text)
    base = re.sub(r"[^0-9A-Za-z._-]+", "-", base).strip("-.")
    return (base[:maxlen] or "x").lower()


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def make_run_id(*, harness: str, target_model: str, temperature: float,
                rejection_sampling: bool, dp_size: int, concurrency: int,
                scale: str = "", tag: str = "", now: Optional[datetime] = None) -> str:
    ts = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    temp_tag = "t0" if float(temperature) == 0.0 else f"t{temperature:g}".replace(".", "p")
    if rejection_sampling:
        temp_tag += "rs"
    parts = [ts, slug(target_model), harness, temp_tag, f"dp{dp_size}xc{concurrency}"]
    if scale:
        parts.append(slug(scale, 16))
    if tag:
        parts.append(slug(tag, 24))
    return "__".join(parts)


def _ckpt_info(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    info = {"draft_model": str(path), "abs_path": str(p.resolve()) if p.exists() else str(path),
            "basename": p.name}
    m = re.search(r"(epoch[_-]?\d+)|(step[_-]?\d+)", p.name)
    if m:
        info["ckpt_tag"] = p.name
    try:
        if p.exists():
            info["mtime"] = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
            config_path = p / "config.json"
            if config_path.is_file():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                architectures = config.get("architectures")
                if architectures:
                    info["architecture"] = (
                        architectures[0] if isinstance(architectures, list) else architectures
                    )
                if "block_size" in config:
                    info["block_size"] = config["block_size"]
                # DSpark checkpoints carry the head settings in dflash_config
                # (what DSparkDraftModel reads); dspark_config is the sglang-side
                # mirror and the only place some older checkpoints have them.
                head_config = config.get("dflash_config") or config.get("dspark_config") or {}
                if "markov_head_type" in head_config:
                    info["markov_head_type"] = head_config["markov_head_type"]
    except Exception:
        pass
    return info


def collect_env() -> dict:
    """Snapshot the runtime environment so results stay reproducible."""
    env: dict = {"python": platform.python_version(), "specbench_commit": git_commit()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    for mod, key in (("transformers", "transformers"), ("sglang", "sglang"),
                     ("flashinfer", "flashinfer")):
        try:
            m = __import__(mod)
            env[key] = getattr(m, "__version__", "unknown")
        except Exception:
            pass
    # sglang fork submodule commit (which serving code produced these numbers)
    try:
        sg_dir = Path(__file__).resolve().parents[1] / "third_party" / "sglang"
        env["sglang_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(sg_dir),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        pass
    return env


# Instance-specific / path fields excluded from the shared protocol snapshot.
_PROTOCOL_EXCLUDE = {
    "target_model", "draft_model", "datasets", "run_dir", "responses_path",
    "results_output", "output_dir",
}


def _protocol(cfg) -> dict:
    from dataclasses import asdict
    d = asdict(cfg)
    return {k: v for k, v in d.items() if k not in _PROTOCOL_EXCLUDE}


def update_manifest(manifest_path: Path, *, cfg, method, harness: str) -> None:
    """Create or merge the run manifest.

    Captures the FULL evaluation protocol (every knob), the environment
    (torch/sglang/flashinfer versions, GPU, sglang fork commit), and per-method
    speculative parameters + draft checkpoint. Every draft ckpt ever used is
    also appended to ``draft_log`` so nothing is forgotten.
    """
    method_name = method.name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = {
            "run_id": manifest_path.parent.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "harness": harness,
            "target_model": cfg.target_model,
            "target_model_abs": str(Path(cfg.target_model).resolve()) if Path(cfg.target_model).exists() else cfg.target_model,
            "datasets": cfg.dataset_specs(),
            "protocol": _protocol(cfg),
            "env": collect_env(),
            "methods": {},
            "draft_log": [],
        }

    info = _ckpt_info(cfg.draft_model)
    # per-method speculative protocol (algorithm + tree params)
    if info is None:
        info = {}
    info["speculative_algorithm"] = method.sglang_speculative_algorithm
    info["speculative_params"] = cfg.apply_spec_overrides(method.sglang_spec_params)
    info["method_env"] = dict(method.sglang_env)
    info["ragged_verify_mode"] = method.sglang_env.get("SGLANG_RAGGED_VERIFY_MODE")
    info["family"] = method.family

    methods = data.setdefault("methods", {})
    prev = methods.get(method_name)
    if prev and prev.get("draft_model") != info.get("draft_model"):
        print(f"[specbench] WARNING: method '{method_name}' now uses a DIFFERENT ckpt "
              f"than previously recorded in this run:\n  prev: {prev.get('draft_model')}\n  new : {info.get('draft_model')}\n"
              f"  both are kept in draft_log; results rows carry their own draft_model.")
    methods[method_name] = info
    data.setdefault("draft_log", []).append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": method_name,
        "draft_model": info.get("draft_model"),
    })
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
