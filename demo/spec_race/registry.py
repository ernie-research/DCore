"""Method / checkpoint registry for the speculative-decoding race demo.

`MODELS` stores the *trained* checkpoint trees (kept audit-friendly:
``outputs/<model>/recipe-level/budget{N}/<method>-block{N}/epoch_X_step_Y``),
which only exist on the training machine. ``checkpoint_path`` resolves a
(model, method, budget) triplet in three tiers, so the demo also runs on
machines that only have the published weights:

1. local trained artifacts — used when present (default);
2. published Hugging Face repos (``wingchiuloong/DCore-*-Qwen3-4B-*`` and the
   ``ernie-research`` Qwen3-8B series), fetched via ``snapshot_download`` and
   cached, so subsequent launches are offline;
3. ``None`` when neither is available (the UI then marks the row unavailable).

Set ``SPEC_RACE_USE_RELEASED=1`` to skip tier 1 and go straight to the
published repos — handy on machines without ``outputs/``, or to test the
release path while trained artifacts still exist locally. The ``download``
flag of ``checkpoint_path`` controls whether a cache miss may hit the network
(kept off by default so ``availability()`` never triggers a multi-GB fetch).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]

# Skip local trained artifacts and resolve straight to the published repos.
RELEASED_ENV = "SPEC_RACE_USE_RELEASED"

# HF namespace per target model (repo id = f"DCore-{method}-{model}-Budget{n}").
REPO_PREFIX = {"Qwen3-8B": "ernie-research", "Qwen3-4B": "wingchiuloong"}
_REPO_METHOD = {
    "dflash": "DFlash", "dflare": "DFlare", "dpace": "DPace",
    "domino": "Domino", "dspark": "DSpark", "eagle3": "EAGLE3", "dcore": "",
}


def repo_id(model: str, method: str, budget: int) -> str:
    """Published repo id for a triplet (the release-time stable identity)."""
    name = f"{_REPO_METHOD[method]}-" if _REPO_METHOD[method] else ""
    budget_part = "" if method == "eagle3" else f"-Budget{int(budget)}"
    return f"{REPO_PREFIX[model]}/DCore-{name}{model}{budget_part}"

BASELINE = "baseline"

# Display order of the race. "ours" marks the paper method for styling.
METHODS: list[dict] = [
    {"key": "dflash", "label": "DFlash", "algorithm": "DFLASH", "color": "#4e79a7"},
    {"key": "dflare", "label": "DFlare", "algorithm": "DFLASH", "color": "#59a14f"},
    {"key": "dpace", "label": "DPace", "algorithm": "DFLASH", "color": "#b07aa1"},
    {"key": "domino", "label": "Domino", "algorithm": "DFLASH", "color": "#f28e2b"},
    {"key": "dspark", "label": "DSpark", "algorithm": "DSPARK", "color": "#76b7b2"},
    {"key": "eagle3", "label": "EAGLE-3", "algorithm": "EAGLE3", "color": "#edc948"},
    {"key": "dcore", "label": "DCore", "algorithm": "DFLASH", "color": "#e15759", "ours": True},
]
BASELINE_META = {"key": BASELINE, "label": "Target AR", "algorithm": None, "color": "#9aa0a6"}
METHOD_BY_KEY = {m["key"]: m for m in [*METHODS, BASELINE_META]}

BUDGETS = (8, 16)

# target:draft temperature presets
TEMPERATURE_PRESETS: dict[str, tuple[float, float]] = {
    "0:0": (0.0, 0.0),
    "1:0": (1.0, 0.0),
    "1:1": (1.0, 1.0),
}

# EAGLE3 runs as a tree: (num_steps, topk, num_draft_tokens), matching the
# eval sweep's EAGLE3_TREE_CONFIGS. Constraint: num_draft_tokens - 1 <= num_steps * topk.
EAGLE3_TREE = {8: (7, 4, 8), 16: (7, 10, 16)}

_Q8 = "outputs/Qwen3-8B/recipe-level"
_Q4 = "outputs/Qwen3-4B/recipe-level"

MODELS: dict[str, dict] = {
    "Qwen3-8B": {
        "target": "cache/models/Qwen3-8B",
        "checkpoints": {
            ("dflash", 8): f"{_Q8}/budget8/dflash-block8/epoch_3_step_264279",
            ("dflash", 16): f"{_Q8}/budget16/dflash-block16/epoch_3_step_262647",
            ("dflare", 8): f"{_Q8}/budget8/dflare-block8/epoch_2_step_264279",
            ("dflare", 16): f"{_Q8}/budget16/dflare-block16/epoch_3_step_262647",
            ("dpace", 8): f"{_Q8}/budget8/dpace-block8/epoch_2_step_264279",
            ("dpace", 16): f"{_Q8}/budget16/dpace-block16/epoch_2_step_262647",
            ("domino", 8): f"{_Q8}/budget8/domino-block8/epoch_2_step_264279",
            ("domino", 16): f"{_Q8}/budget16/domino-block16/epoch_2_step_262647",
            ("dspark", 8): f"{_Q8}/budget8/dspark-markov-block7/epoch_2_step_262647",
            ("dspark", 16): f"{_Q8}/budget16/dspark-markov-block15/epoch_2_step_262647",
            ("eagle3", 8): f"{_Q8}/budget8/eagle3/epoch_2_step_266421",
            ("eagle3", 16): f"{_Q8}/budget8/eagle3/epoch_2_step_266421",
            ("dcore", 8): f"{_Q8}/budget8/dcore-block8/epoch_2_step_264279",
            ("dcore", 16): f"{_Q8}/budget16/dcore-block16/epoch_3_step_262647",
        },
    },
    "Qwen3-4B": {
        "target": "cache/models/Qwen3-4B",
        "checkpoints": {
            ("dflash", 8): f"{_Q4}/budget8/dflash-block8/epoch_3_step_264135",
            ("dflash", 16): f"{_Q4}/budget16/dflash-block16/epoch_3_step_262419",
            ("dflare", 8): f"{_Q4}/budget8/dflare-block8/epoch_3_step_264135",
            ("dflare", 16): f"{_Q4}/budget16/dflare-block16/epoch_3_step_262419",
            ("dpace", 8): f"{_Q4}/budget8/dpace-block8/epoch_3_step_264135",
            ("dpace", 16): f"{_Q4}/budget16/dpace-block16/epoch_3_step_262419",
            ("domino", 8): f"{_Q4}/budget8/domino-block8/epoch_3_step_264135",
            ("domino", 16): f"{_Q4}/budget16/domino-block16/epoch_3_step_262419",
            ("dspark", 8): f"{_Q4}/budget8/dspark-markov-block7/epoch_3_step_262419",
            ("dspark", 16): f"{_Q4}/budget16/dspark-markov-block15/epoch_3_step_262419",
            # EAGLE3's decode tree (steps/topk/draft) is set via EAGLE3_TREE at serve
            # time, so b8/b16 share the single checkpoint under budget8.
            ("eagle3", 8): f"{_Q4}/budget8/eagle3/epoch_2_step_266418",
            ("eagle3", 16): f"{_Q4}/budget8/eagle3/epoch_2_step_266418",
            ("dcore", 8): f"{_Q4}/budget8/dcore-block8/epoch_3_step_264135",
            ("dcore", 16): f"{_Q4}/budget16/dcore-block16/epoch_3_step_262419",
        },
    },
}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


# Official HF repo per target model (sharded weights, so the whole repo is
# fetched rather than a pattern whitelist).
TARGET_HF = {"Qwen3-8B": "Qwen/Qwen3-8B", "Qwen3-4B": "Qwen/Qwen3-4B"}


def target_path(model: str, download: bool = False) -> Path:
    """Local target dir, or the official Qwen repo via the HF cache.

    Tier 1: the local ``cache/models/<model>`` dir (what the README points at).
    Tier 2: ``snapshot_download`` of the official ``Qwen/Qwen3-*`` repo; the
    weights are sharded (``model-*.safetensors`` + index), so everything is
    fetched and cached, and later launches are offline. ``download=False`` only
    consults the cache, so ``availability()`` never triggers a multi-GB fetch.
    Tier 3: the local path as-is (missing; callers then fail loudly).
    """
    local = resolve(MODELS[model]["target"])
    if local.exists():
        return local
    if model not in TARGET_HF:
        return local
    from huggingface_hub import snapshot_download

    rid = TARGET_HF[model]
    for kw in ({"local_files_only": not download}, {"local_files_only": True}):
        try:
            return Path(snapshot_download(repo_id=rid, **kw))
        except Exception:
            continue
    return local


def _usable(p: Optional[Path]) -> bool:
    """A drafter dir is usable iff both weights and config are present."""
    return p is not None and (p / "config.json").exists() and (p / "model.safetensors").exists()


def checkpoint_path(model: str, method: str, budget: int,
                    download: bool = False) -> Optional[Path]:
    """Resolve the drafter dir for a (model, method, budget) triplet.

    Tier 1: local trained artifacts (skipped when ``SPEC_RACE_USE_RELEASED``).
    Tier 2: the published HF repo via ``snapshot_download``; ``download=True``
    allows a cache miss to hit the network, otherwise only the local HF cache
    is consulted (so ``availability()`` never triggers a multi-GB fetch).
    Tier 3: ``None``.
    """
    rel = MODELS[model]["checkpoints"].get((method, int(budget)))
    if rel is None:
        return None
    local = resolve(rel)
    if not os.environ.get(RELEASED_ENV) and local.exists():
        return local
    if model not in REPO_PREFIX or method not in _REPO_METHOD:
        return local if local.exists() else None
    from huggingface_hub import snapshot_download

    rid = repo_id(model, method, int(budget))
    for kw in ({"local_files_only": not download}, {"local_files_only": True}):
        try:
            d = Path(snapshot_download(repo_id=rid, allow_patterns=[
                "config.json", "model.safetensors", "*.py"], **kw))
            if _usable(d):
                return d
        except Exception:
            continue
    # Released mode never falls back to local: it exists to prove the published
    # weights work (e.g. on a machine without outputs/). Default mode already
    # returned above when local artifacts existed, so reaching here means the
    # repo is unavailable either way.
    return None


def verify_window(method: str, budget: int) -> int:
    """Max tokens a single verify step can commit (accepted drafts + bonus)."""
    return 1 if method == BASELINE else int(budget)


class ServerSpec(NamedTuple):
    """Everything that is fixed at `sglang serve` launch time.

    Target temperature is a per-request sampling param, so DFLASH/DSPARK servers
    are shared between the 0:0 and 1:0 presets. EAGLE3 additionally needs
    --speculative-use-rejection-sampling when the target samples (t > 0).
    """

    model: str
    method: str
    budget: int
    draft_temperature: float
    rejection_sampling: bool

    @property
    def key(self) -> str:
        if self.method == BASELINE:
            return f"{self.model}/baseline"
        rs = "-rs" if self.rejection_sampling else ""
        return f"{self.model}/{self.method}/b{self.budget}/dt{self.draft_temperature:g}{rs}"


def server_spec(model: str, method: str, budget: int, preset: str) -> ServerSpec:
    target_t, draft_t = TEMPERATURE_PRESETS[preset]
    if method == BASELINE:
        return ServerSpec(model, BASELINE, 0, 0.0, False)
    return ServerSpec(
        model,
        method,
        int(budget),
        float(draft_t),
        method == "eagle3" and target_t > 0,
    )


def sampling_params(preset: str, max_new_tokens: int) -> dict:
    # Seeding is done per request via /set_internal_state (see engine.py):
    # `sampling_seed` is only honoured under deterministic inference.
    target_t, _ = TEMPERATURE_PRESETS[preset]
    return {
        "max_new_tokens": int(max_new_tokens),
        "temperature": target_t,
        "top_p": 1.0,
        "top_k": 1 if target_t == 0 else -1,
        "stream_interval": 1,
        "skip_special_tokens": True,
    }


def availability() -> dict:
    """Which targets / checkpoints are usable (for the UI).

    Local-only (``download=False``): does not hit the network, so a demo start
    never pulls several GiB of weights just to render the availability table.
    """
    out = {}
    for model, info in MODELS.items():
        out[model] = {
            "target": str(target_path(model)),
            "target_exists": target_path(model).exists(),
            "checkpoints": {
                f"{m['key']}:{b}": _usable(checkpoint_path(model, m["key"], b))
                for m in METHODS
                for b in BUDGETS
            },
        }
    return out
