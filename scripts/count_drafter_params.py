"""Count drafter parameters for the drafter-capacity table.

Instantiates every drafter (Eagle-3, DFlash, DFlare, Domino, DSpark) on CPU
under both targets (Qwen3-4B d=2560, Qwen3-8B d=4096) and reports the
Fusion / Backbone / Corr. / Head breakdown in millions, matching the column
attribution of the paper's drafter-capacity table.

The block-parallel drafters reuse the frozen target LM head (vocab x d).
That head is not trainable but runs in every draft forward, so it is shown
in the Head column (marked as shared) and included in Total.

D-PACE shares the DFlash architecture and DCore shares the DFlare (9hs)
architecture, so their rows equal the DFlash / DFlare rows respectively.

Also prints target-model totals: Qwen3-8B measured from the cached
safetensors shards, Qwen3-4B computed analytically from the official config
(tie_word_embeddings=true, embedding counted once).

Usage: python3 scripts/count_drafter_params.py
"""

import json
import math
import os
import sys

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specforge.modeling.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from specforge.modeling.draft.dflare import DFlareDraftModel
from specforge.modeling.draft.domino import DominoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")

# Excluded from all counts: Eagle-3 loads the target embedding table and
# freezes it (freeze_embedding()); the block-parallel drafters have no
# embedding module at all.
EXCLUDED_PREFIXES = ("embed_tokens.",)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def categorize(model, kind):
    """Split named_parameters() into fusion/backbone/corr/head buckets."""
    counts = {"fusion": 0, "backbone": 0, "corr": 0, "head": 0}
    total = 0
    for name, p in model.named_parameters():
        if name.startswith(EXCLUDED_PREFIXES):
            continue
        total += p.numel()
        if kind == "eagle3":
            if name.startswith("fc."):
                cat = "fusion"
            elif name.startswith("midlayer.") or name.startswith("norm."):
                cat = "backbone"
            elif name.startswith("lm_head."):
                cat = "head"
            else:
                raise ValueError(f"uncategorized eagle3 param: {name}")
        elif kind == "dflare":
            # Fusion = mixing matrix + hidden_norm + per-draft-layer
            # k/v projections of the target hidden states.
            if (
                name == "layer_fusion_weights"
                or name.startswith("hidden_norm.")
                or ".k_proj_target." in name
                or ".v_proj_target." in name
            ):
                cat = "fusion"
            elif name.startswith("layers.") or name.startswith("norm."):
                cat = "backbone"
            else:
                raise ValueError(f"uncategorized dflare param: {name}")
        else:  # dflash family: dflash / domino / dspark
            if name.startswith("fc.") or name.startswith("hidden_norm."):
                cat = "fusion"
            elif name.startswith(
                ("prefix_gru.", "embed_proj.", "markov_head.", "confidence_head.")
            ):
                cat = "corr"
            elif name.startswith("layers.") or name.startswith("norm."):
                cat = "backbone"
            else:
                raise ValueError(f"uncategorized {kind} param: {name}")
        counts[cat] += p.numel()
    assert sum(counts.values()) == total, f"{kind}: bucket sum != total"
    counts["trainable_total"] = total
    return counts


def build_eagle3(size):
    cfg = AutoDraftModelConfig.from_file(
        os.path.join(CONFIG_DIR, f"qwen3-{size}-eagle3.json")
    )
    return AutoEagle3DraftModel.from_config(cfg)


def build_dflash(size):
    cfg = AutoDraftModelConfig.from_file(
        os.path.join(CONFIG_DIR, f"qwen3-{size}-dflash.json")
    )
    return AutoEagle3DraftModel.from_config(cfg)


def build_dflare(size):
    # DFlareDraftModel is not in the draft registry; construct it directly.
    cfg_dict = _load_json(os.path.join(CONFIG_DIR, f"qwen3-{size}-dflare_9hs.json"))
    model = DFlareDraftModel(Qwen3Config.from_dict(cfg_dict))
    assert model.layer_fusion_weights.shape == (5, 9)
    return model


def build_domino(size):
    if size == "8b":
        cfg = AutoDraftModelConfig.from_file(
            os.path.join(CONFIG_DIR, "qwen3-8b-domino.json")
        )
        model = AutoEagle3DraftModel.from_config(cfg)
    else:
        # No 4B Domino config exists; build it in memory from the 4B DFlash
        # config plus the same extras as the 8B Domino config.
        cfg_dict = _load_json(os.path.join(CONFIG_DIR, "qwen3-4b-dflash.json"))
        cfg_dict["architectures"] = ["DominoDraftModel"]
        cfg_dict["dflash_config"].update(
            {"projector_type": "domino", "emb_dim": 256, "gru_hidden_dim": 1024}
        )
        model = DominoDraftModel(Qwen3Config.from_dict(cfg_dict))
    assert isinstance(model, DominoDraftModel)
    assert model.gru_hidden_dim == 1024 and model.emb_dim == 256
    return model


def build_dspark(size):
    cfg_dict = _load_json(os.path.join(CONFIG_DIR, f"qwen3-{size}-dspark.json"))
    # The 8B config keeps the extras under `dspark_config` while the model
    # class reads `dflash_config`; merge before instantiation.
    if "dspark_config" in cfg_dict:
        merged = dict(cfg_dict.get("dflash_config") or {})
        merged.update(cfg_dict.pop("dspark_config"))
        merged["projector_type"] = "dspark"
        cfg_dict["dflash_config"] = merged
    model = DSparkDraftModel(Qwen3Config.from_dict(cfg_dict))
    assert model.markov_head is not None and model.markov_head.markov_rank == 256
    assert model.markov_head.markov_head_type == "vanilla"
    assert model.confidence_head is not None
    return model


def qwen3_4b_total_analytic():
    """Official Qwen3-4B config: tie_word_embeddings=true (embedding counted once)."""
    hidden, layers, inter = 2560, 36, 9728
    n_heads, n_kv, head_dim, vocab = 32, 8, 128, 151936
    per_layer = (
        2 * n_heads * head_dim * hidden  # q_proj + o_proj
        + 2 * n_kv * head_dim * hidden  # k_proj + v_proj
        + 2 * head_dim  # q_norm + k_norm
        + 3 * hidden * inter  # gate/up/down
        + 2 * hidden  # input/post-attention layernorms
    )
    return vocab * hidden + layers * per_layer + hidden  # + final norm


def qwen3_8b_total_measured():
    model_dir = os.path.join(REPO_ROOT, "cache", "models", "Qwen3-8B")
    index = _load_json(os.path.join(model_dir, "model.safetensors.index.json"))
    by_file = {}
    for name, fname in index["weight_map"].items():
        by_file.setdefault(fname, []).append(name)
    from safetensors import safe_open

    total = 0
    for fname, names in by_file.items():
        with safe_open(os.path.join(model_dir, fname), framework="pt") as f:
            for name in names:
                total += math.prod(f.get_slice(name).get_shape())
    return total


def main():
    torch.manual_seed(0)
    builders = {
        "Eagle-3": (build_eagle3, "eagle3"),
        "DFlash": (build_dflash, "dflash"),
        "DFlare": (build_dflare, "dflare"),
        "Domino": (build_domino, "domino"),
        "DSpark": (build_dspark, "dspark"),
    }

    # Frozen target LM head reused by the block-parallel drafters
    # (vocab x hidden). Not trainable, but part of every draft forward,
    # so it is shown in the Head column and included in Total.
    tied_head = {"4b": 151936 * 2560, "8b": 151936 * 4096}

    results = {}
    for size in ("4b", "8b"):
        for row, (builder, kind) in builders.items():
            model = builder(size)
            counts = categorize(model, kind)
            if kind != "eagle3":
                counts["shared_head"] = tied_head[size]
                counts["head"] = tied_head[size]
            counts["total"] = counts["trainable_total"] + counts.get("shared_head", 0)
            results[(row, size)] = counts
            del model

    def fmt(v, mark=""):
        return f"{v / 1e6:.1f}{mark}".rjust(11) if v else "---".rjust(11)

    header = f"{'Drafter':<10}" + "".join(
        f"{c:>11}" for c in ("Fusion", "Backbone", "Corr.", "Head", "Total")
    )
    for size, label in (("4b", "Qwen3-4B (d=2560)"), ("8b", "Qwen3-8B (d=4096)")):
        print(f"\n=== {label} ===")
        print(header)
        for row in builders:
            c = results[(row, size)]
            head_mark = "*" if "shared_head" in c else ""
            print(
                f"{row:<10}{fmt(c['fusion'])}{fmt(c['backbone'])}"
                f"{fmt(c['corr'])}{fmt(c['head'], head_mark)}{fmt(c['total'])}"
            )
        print("(* = frozen target LM head shared with the target; included in Total)")

    print("\n=== Exact counts (params) ===")
    for (row, size), c in sorted(results.items()):
        print(f"{row}-{size}: {c}")

    # Cross-architecture invariants
    for size in ("4b", "8b"):
        assert (
            results[("DFlare", size)]["backbone"] == results[("DFlash", size)]["backbone"]
        ), f"DFlare backbone != DFlash backbone @ {size}"
        assert (
            results[("Domino", size)]["fusion"] == results[("DFlash", size)]["fusion"]
            and results[("Domino", size)]["backbone"]
            == results[("DFlash", size)]["backbone"]
        ), f"Domino fusion/backbone != DFlash @ {size}"
    print("\nInvariant checks passed (DFlare backbone == DFlash; Domino == DFlash + Corr.)")

    total_4b = qwen3_4b_total_analytic()
    total_8b = qwen3_8b_total_measured()
    print(f"\nQwen3-4B total (analytic, tied emb): {total_4b:,} = {total_4b / 1e6:.1f}M")
    print(f"Qwen3-8B total (measured safetensors): {total_8b:,} = {total_8b / 1e6:.1f}M")


if __name__ == "__main__":
    main()
