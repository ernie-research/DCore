"""Method registry — the single extension point for adding new methods.

Each speculative-decoding method is declared once as a :class:`MethodSpec`.
Harnesses read the spec to decide how to serve/drive the method; they never
hard-code method names.

To add a new method later (e.g. once its SGLang serving code is upstreamed),
register a new :class:`MethodSpec` here and flip ``sglang_supported=True``.
No harness or CLI code needs to change.

Serving reality (SGLang v0.5.14 fork, ``specforge-serving-v0.5.14``):
  * Baseline / EAGLE3 / DFLASH / DSPARK are natively servable.
  * DPace serves as DFLASH (it is a training-loss variant of DFlash, identical
    at serving time).
  * DFlare / Domino serve as DFLASH submodes: the fork's models/dflare.py and
    the Domino projector branch handle their structural differences, so both
    are natively servable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str                          # "baseline" | "eagle3" | "dflash"
    requires_draft: bool

    # --- SGLang serving ---
    sglang_supported: bool
    sglang_speculative_algorithm: Optional[str] = None   # "EAGLE3" | "DFLASH" | "DSPARK" | None
    # EAGLE-family tree params; DFLASH/DSPARK use the draft config's block_size
    # and ignore these, so leave empty for those methods.
    sglang_spec_params: dict = field(default_factory=dict)
    # Per-server process environment; merged with replica-local GPU pinning.
    sglang_env: dict[str, str] = field(default_factory=dict)

    # --- HF-Direct serving ---
    hf_direct_supported: bool = False
    # Detection hint for eval.hf_direct; None => auto-detect from checkpoint.
    hf_direct_model_type: Optional[str] = None

    note: str = ""

    def check_harness(self, harness: str) -> None:
        """Raise a clear error if this method is not runnable on ``harness``."""
        if harness == "sglang" and not self.sglang_supported:
            raise NotImplementedError(
                f"Method '{self.name}' is not yet servable via the SGLang harness "
                f"in this fork. {self.note} "
                f"Use --harness hf_direct, or add serving support and set "
                f"sglang_supported=True in specbench/methods.py."
            )
        if harness == "hf_direct" and not self.hf_direct_supported:
            raise NotImplementedError(
                f"Method '{self.name}' is not supported by the HF-Direct harness "
                f"(e.g. EAGLE3 has no benchmark_dflash path). Use --harness sglang."
            )


# --- Registry --------------------------------------------------------------

_REGISTRY: dict[str, MethodSpec] = {}


def register(spec: MethodSpec) -> MethodSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"Duplicate method registration: {spec.name}")
    _REGISTRY[spec.name] = spec
    return spec


def get_method(name: str) -> MethodSpec:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown method '{name}'. Registered: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key]


def list_methods() -> list[MethodSpec]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


# --- Built-in methods ------------------------------------------------------

register(MethodSpec(
    name="baseline",
    family="baseline",
    requires_draft=False,
    sglang_supported=True,
    sglang_speculative_algorithm=None,
    hf_direct_supported=True,
    note="Target-only autoregressive decoding (no speculation).",
))

register(MethodSpec(
    name="eagle3",
    family="eagle3",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="EAGLE3",
    sglang_spec_params={
        "speculative_num_steps": 15,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 15,
    },
    hf_direct_supported=False,
    note="EAGLE3 has no HF-Direct (benchmark_dflash) path; SGLang harness only.",
))

register(MethodSpec(
    name="eagle3_origin",
    family="eagle3",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="EAGLE3",
    sglang_spec_params={
        "speculative_num_steps": 15,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 15,
    },
    hf_direct_supported=False,
    note="Third-party EAGLE3 checkpoint evaluated through the SGLang harness.",
))

register(MethodSpec(
    name="eagle3_tree",
    family="eagle3",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="EAGLE3",
    sglang_spec_params={
        "speculative_num_steps": 7,
        "speculative_eagle_topk": 10,
        "speculative_num_draft_tokens": 16,
    },
    hf_direct_supported=False,
    note="EAGLE3 evaluated with the standard 7-step, top-k 10, 16-token tree.",
))

register(MethodSpec(
    name="eagle3_origin_tree",
    family="eagle3",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="EAGLE3",
    sglang_spec_params={
        "speculative_num_steps": 7,
        "speculative_eagle_topk": 10,
        "speculative_num_draft_tokens": 16,
    },
    hf_direct_supported=False,
    note="Third-party EAGLE3 checkpoint with 7-step top-k=10 tree decoding.",
))

register(MethodSpec(
    name="eagle3_3l_tree",
    family="eagle3",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="EAGLE3",
    sglang_spec_params={
        "speculative_num_steps": 7,
        "speculative_eagle_topk": 10,
        "speculative_num_draft_tokens": 16,
    },
    hf_direct_supported=False,
    note="EAGLE3 3-layer draft with 7-step top-k=10 tree decoding (ablation).",
))

# EAGLE3 objective ablation (train_eagle3.py --loss-type kl|ce|dpace). The three
# checkpoints share the eagle3 architecture and differ only in training loss, so
# they all serve as EAGLE3 with the standard 7-step top-k=10 tree.
register(MethodSpec(
    name="eagle3_kl_tree",
    family="eagle3",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="EAGLE3",
    sglang_spec_params={
        "speculative_num_steps": 7,
        "speculative_eagle_topk": 10,
        "speculative_num_draft_tokens": 16,
    },
    hf_direct_supported=False,
    note="EAGLE3 with soft-target KL loss (--loss-type kl), 0806 retrain "
         "(outputs/perfectblend/eagle3-0806); kl/ce/dpace ablation baseline.",
))

# Chain-decoding ablation of eagle3_kl_tree: same checkpoint, topk=1 so the
# draft tree degenerates to a chain (num_draft_tokens = num_steps + 1). With
# topk=1 the FlashInfer backend auto-selects the "chain" verify mode
# (mask-free causal paged prefill, bit-identical to the masked path); the
# *_legacy twin pins SGLANG_FLASHINFER_VERIFY_ATTN_MODE=legacy to keep the
# tree-mask verify machinery as the control arm for that comparison.
register(MethodSpec(
    name="eagle3_kl_chain",
    family="eagle3",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="EAGLE3",
    sglang_spec_params={
        "speculative_num_steps": 7,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 8,
    },
    hf_direct_supported=False,
    note="eagle3_kl_tree checkpoint drafted as a chain (topk=1); verify runs "
         "as mask-free causal attention via the auto-selected chain mode.",
))

register(MethodSpec(
    name="eagle3_kl_chain_legacy",
    family="eagle3",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="EAGLE3",
    sglang_spec_params={
        "speculative_num_steps": 7,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 8,
    },
    sglang_env={"SGLANG_FLASHINFER_VERIFY_ATTN_MODE": "legacy"},
    hf_direct_supported=False,
    note="Control arm of eagle3_kl_chain: identical chain topology, but the "
         "verify path is pinned to the legacy tree-mask machinery.",
))

register(MethodSpec(
    name="eagle3_ce_tree",
    family="eagle3",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="EAGLE3",
    sglang_spec_params={
        "speculative_num_steps": 7,
        "speculative_eagle_topk": 10,
        "speculative_num_draft_tokens": 16,
    },
    hf_direct_supported=False,
    note="EAGLE3 with hard-label CE loss (--loss-type ce); ablation pair of "
         "eagle3_kl_tree.",
))

register(MethodSpec(
    name="eagle3_dpace_tree",
    family="eagle3",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="EAGLE3",
    sglang_spec_params={
        "speculative_num_steps": 7,
        "speculative_eagle_topk": 10,
        "speculative_num_draft_tokens": 16,
    },
    hf_direct_supported=False,
    note="EAGLE3 with D-PACE weighted CE loss (--loss-type dpace); ablation "
         "pair of eagle3_kl_tree.",
))

register(MethodSpec(
    name="dflash",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,      # auto-detected from checkpoint keys
    note="DFlash block-parallel speculative decoding.",
))

register(MethodSpec(
    name="dflash_3l",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlash 3-layer variant (deeper draft network, b16 only).",
))

register(MethodSpec(
    name="dflash_7l",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlash 7-layer variant (5 fused target hidden states, b16 only); "
         "depth ablation pair of dflash / dflash_3l.",
))

register(MethodSpec(
    name="dflash_origin",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="Third-party DFlash b16 checkpoint evaluated as DFLASH.",
))

register(MethodSpec(
    name="dpace",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,      # DPace checkpoints detect as plain "dflash"
    note="DPace = DFlash architecture + D-PACE training loss; serves as DFLASH.",
))

# --- DFLASH submodes (served through the DFLASH worker) ---------------------
register(MethodSpec(
    name="dflare",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlare (DFlash variant: separate k/v projections + layer fusion), the "
         "paper-facing recipe: 5 layers fusing 9 target hidden states. Served "
         "via the DFLASH worker; the checkpoint's DFlareDraftModel architecture "
         "selects the per-layer context-KV path.",
))

register(MethodSpec(
    name="dflare_5hs_5l",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlare 5-layer fusing only 5 target hidden states; hidden-state "
         "ablation pair of dflare (which fuses 9).",
))

register(MethodSpec(
    name="dflare_dpace",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlare trained with D-PACE loss (b8 only). Serves as DFLASH.",
))

register(MethodSpec(
    name="dflare_origin",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlare origin 7-layer checkpoint (Huang2020/Qwen3-8b-dflare-b16, 9 target_layer_ids).",
))

register(MethodSpec(
    name="domino",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="Domino (DFlash + GRU logits-correction head). Served via the DFLASH "
         "worker; the checkpoint's projector_type=domino selects the GRU rollout.",
))

register(MethodSpec(
    name="domino_origin",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="Domino origin checkpoint (Huang2020/Qwen3-8B-Domino-b16).",
))

register(MethodSpec(
    name="domino_pool2048",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    sglang_env={"DFLASH_DOMINO_CANDIDATE_POOL": "2048"},
    hf_direct_supported=False,
    note="Domino with candidate pool size 2048 (top-K shortlist from base logits).",
))

register(MethodSpec(
    name="domino_without_curriculum",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="Domino trained without curriculum learning (ablation).",
))

register(MethodSpec(
    name="domino_without_curriculum_dpace",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="Domino + D-PACE loss without curriculum (fixed lambda_base); ablation "
         "pair of domino_without_curriculum.",
))

register(MethodSpec(
    name="eagle3_3layer",
    family="eagle3",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="EAGLE3",
    sglang_spec_params={
        "speculative_num_steps": 15,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 15,
    },
    hf_direct_supported=False,
    note="EAGLE3 with 3-layer draft model (ablation).",
))

register(MethodSpec(
    name="dflash_anchor",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlash with anchor prediction (ablation: predict anchor token position).",
))

register(MethodSpec(
    name="dflare_9hs",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="Legacy alias of dflare (5-layer, 9 fused target hidden states), kept "
         "for the older eval/run_eval_*.sh scripts.",
))

register(MethodSpec(
    name="dflare_9hs_7l",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlare deepened to 7 layers (still 9 fused target hidden states); "
         "depth ablation pair of dflare.",
))

register(MethodSpec(
    name="dflare_dpace_9hs",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlare 5-layer 9-hidden-state + D-PACE loss (ablation).",
))

register(MethodSpec(
    name="dflare_dpace_9hs_anchor",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlare 5-layer 9-hidden-state + D-PACE loss + anchor prediction (ablation).",
))

register(MethodSpec(
    name="dflare_dpace_9hs_b300",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlare 5-layer 9-hidden-state + D-PACE loss, B300 retrain "
         "(block_size 8/16).",
))

register(MethodSpec(
    name="dcore",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DCore = DFlare 5-layer 9-hidden-state + D-PACE loss, B300 retrain "
         "(block_size 8/16). Paper-facing name for the dflare_dpace_9hs_b300 "
         "recipe; identical serving path (DFLASH worker).",
))

register(MethodSpec(
    name="dflare_dpace_9hs_anchor_b300",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlare 5-layer 9-hidden-state + D-PACE + anchor prediction, B300 "
         "retrain. block_size 7/15 keeps the draft budget matched with the "
         "non-anchor block 8/16 runs (anchor adds one draft token).",
))

register(MethodSpec(
    name="dflare_kltv_9hs_b300",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlare 5-layer 9-hidden-state + kltv loss (0.1*KL + 0.9*L1 soft "
         "targets), B300 retrain (block_size 8).",
))

register(MethodSpec(
    name="dflare_kltv_9hs_anchor_b300",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DFLASH",
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DFlare 5-layer 9-hidden-state + kltv loss + anchor prediction, B300 "
         "retrain (block_size 7, budget-matched with block 8).",
))

register(MethodSpec(
    name="dspark_markov",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DSPARK",
    sglang_env={"SGLANG_RAGGED_VERIFY_MODE": "static"},
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DSpark with Markov head (checkpoint-defined block size).",
))

register(MethodSpec(
    name="dspark_origin",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DSPARK",
    sglang_env={"SGLANG_RAGGED_VERIFY_MODE": "static"},
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DSpark origin checkpoint (block_size=7, verify window 8).",
))

register(MethodSpec(
    name="dspark_markov_origin",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DSPARK",
    sglang_env={"SGLANG_RAGGED_VERIFY_MODE": "static"},
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="Legacy alias of dspark_origin, kept for the Qwen3-4B sweep scripts "
         "whose already-collected results.jsonl use this name.",
))

register(MethodSpec(
    name="dspark_rnn",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DSPARK",
    sglang_env={"SGLANG_RAGGED_VERIFY_MODE": "static"},
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DSpark with RNN head (checkpoint-defined block size).",
))

register(MethodSpec(
    name="dspark_markov_bs7",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DSPARK",
    sglang_env={"SGLANG_RAGGED_VERIFY_MODE": "static"},
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DSpark Markov bs7 retrain (checkpoint block_size=7, verify window 8).",
))

register(MethodSpec(
    name="dspark_markov_bs7_noconf",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DSPARK",
    sglang_env={"SGLANG_RAGGED_VERIFY_MODE": "static"},
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DSpark Markov bs7 retrain, confidence-head BCE disabled during training.",
))

register(MethodSpec(
    name="dspark_markov_bs15",
    family="dflash",
    requires_draft=True,
    sglang_supported=True,
    sglang_speculative_algorithm="DSPARK",
    sglang_env={"SGLANG_RAGGED_VERIFY_MODE": "static"},
    hf_direct_supported=True,
    hf_direct_model_type=None,
    note="DSpark Markov bs15 retrain (checkpoint block_size=15, verify window 16).",
))

# Objective ablation for DSpark Markov bs15 / 5 layers / 5 target hidden states:
# <loss> x <position weighting>, checkpoints under outputs/perfectblend/
# qwen3_8b_dspark_markov_5l_bs15_5hs_<loss>_<weighting>/. Serving config is
# identical to dspark_markov_bs15 (block_size=15 -> verify window 16); each is
# registered separately only so ablation result rows in the same run_dir don't
# overwrite each other.
for _loss in ("ce", "cetv", "kl", "kltv"):
    for _weighting, _weighting_note in (
        ("dpace", "D-PACE detached continuation values (alpha 0.5)"),
        ("exp", "exp(-k/7) position decay"),
    ):
        register(MethodSpec(
            name=f"dspark_markov_5l_5hs_{_loss}_{_weighting}",
            family="dflash",
            requires_draft=True,
            sglang_supported=True,
            sglang_speculative_algorithm="DSPARK",
            sglang_env={"SGLANG_RAGGED_VERIFY_MODE": "static"},
            hf_direct_supported=True,
            hf_direct_model_type=None,
            note=(
                f"DSpark Markov bs15, 5 layers fusing 5 target hidden states; "
                f"{_loss.upper()} loss x {_weighting_note}."
            ),
        ))
del _loss, _weighting, _weighting_note
