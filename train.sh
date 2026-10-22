#!/bin/bash
# ============================================================================
# SpecForge one-line training launcher (Qwen3-4B / Qwen3-8B x 7 draft methods)
#
# Usage:
#   ./train.sh <model> <method> [--budget N] [--gpus N] [--no-wandb]
#              [--run-name NAME] [--master-port PORT] [--dry-run] [--list]
#
# Examples:
#   ./train.sh qwen3-8b dflash                 # default budget 16, auto GPUs
#   ./train.sh qwen3-4b dspark --budget 8 --gpus 4
#   ./train.sh qwen3-8b eagle3 --no-wandb
#   ./train.sh --list                          # show all 14 combos
#
# Each <model, method, budget> triple maps to one of the launch scripts under
# examples/colocated/ (the authoritative, paper-matching configurations).
# Everything the scripts read from the environment (RUN_NAME, MASTER_PORT,
# GAMMA_EXP, DPACE_ALPHA, NUM_DRAFT_LAYERS, REPORT_TO, ...) is honoured.
#
# Prerequisites (checked): target model weights, GPU count and (unless --dry-run)
# the project venv. Missing train data is downloaded from Hugging Face
# (scripts/download_training_data.py); see README.md "Prepare the training data".
# ============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR" && pwd)
cd "$ROOT_DIR"

MODELS="qwen3-4b qwen3-8b"
METHODS="dflash dflare dpace domino dspark eagle3 dcore"
DEFAULT_BUDGET=16

# <model>|<method>|<budget> -> launch script (relative to ROOT_DIR).
# EAGLE3 trains a single config (ttt length fixed in the config; b8/b16 both
# share the resulting weights and only differ at serve time).
declare -A SCRIPT=(
  ["qwen3-8b|dflash|8"]="examples/colocated/qwen3_8b/dflash/5l_bs8_5hs_ce_exp.sh"
  ["qwen3-8b|dflash|16"]="examples/colocated/qwen3_8b/dflash/5l_bs16_5hs_ce_exp.sh"
  ["qwen3-8b|dflare|8"]="examples/colocated/qwen3_8b/dflare/5l_bs8_9hs_ce_exp.sh"
  ["qwen3-8b|dflare|16"]="examples/colocated/qwen3_8b/dflare/5l_bs16_9hs_ce_exp.sh"
  ["qwen3-8b|dpace|8"]="examples/colocated/qwen3_8b/dpace/5l_bs8_5hs_ce_dpace.sh"
  ["qwen3-8b|dpace|16"]="examples/colocated/qwen3_8b/dpace/5l_bs16_5hs_ce_dpace.sh"
  ["qwen3-8b|domino|8"]="examples/colocated/qwen3_8b/domino/5l_bs8_5hs_ce_exp.sh"
  ["qwen3-8b|domino|16"]="examples/colocated/qwen3_8b/domino/5l_bs16_5hs_ce_exp.sh"
  ["qwen3-8b|dspark|8"]="examples/colocated/qwen3_8b/dspark/markov_5l_bs7_5hs_ce_exp.sh"
  ["qwen3-8b|dspark|16"]="examples/colocated/qwen3_8b/dspark/markov_5l_bs15_5hs_ce_exp.sh"
  ["qwen3-8b|dcore|8"]="examples/colocated/qwen3_8b/dcore/5l_bs8_9hs_ce_dpace.sh"
  ["qwen3-8b|dcore|16"]="examples/colocated/qwen3_8b/dcore/5l_bs16_9hs_ce_dpace.sh"
  ["qwen3-8b|eagle3|8"]="examples/colocated/qwen3_8b/eagle3/1l_kl_exp.sh"
  ["qwen3-8b|eagle3|16"]="examples/colocated/qwen3_8b/eagle3/1l_kl_exp.sh"
  ["qwen3-4b|dflash|8"]="examples/colocated/qwen3_4b/dflash/5l_bs8_5hs_ce_exp.sh"
  ["qwen3-4b|dflash|16"]="examples/colocated/qwen3_4b/dflash/5l_bs16_5hs_ce_exp.sh"
  ["qwen3-4b|dflare|8"]="examples/colocated/qwen3_4b/dflare/5l_bs8_9hs_ce_exp.sh"
  ["qwen3-4b|dflare|16"]="examples/colocated/qwen3_4b/dflare/5l_bs16_9hs_ce_exp.sh"
  ["qwen3-4b|dpace|8"]="examples/colocated/qwen3_4b/dpace/5l_bs8_5hs_ce_dpace.sh"
  ["qwen3-4b|dpace|16"]="examples/colocated/qwen3_4b/dpace/5l_bs16_5hs_ce_dpace.sh"
  ["qwen3-4b|domino|8"]="examples/colocated/qwen3_4b/domino/5l_bs8_5hs_ce_exp.sh"
  ["qwen3-4b|domino|16"]="examples/colocated/qwen3_4b/domino/5l_bs16_5hs_ce_exp.sh"
  ["qwen3-4b|dspark|8"]="examples/colocated/qwen3_4b/dspark/markov_5l_bs7_5hs_ce_exp.sh"
  ["qwen3-4b|dspark|16"]="examples/colocated/qwen3_4b/dspark/markov_5l_bs15_5hs_ce_exp.sh"
  ["qwen3-4b|dcore|8"]="examples/colocated/qwen3_4b/dcore/5l_bs8_9hs_ce_dpace.sh"
  ["qwen3-4b|dcore|16"]="examples/colocated/qwen3_4b/dcore/5l_bs16_9hs_ce_dpace.sh"
  ["qwen3-4b|eagle3|8"]="examples/colocated/qwen3_4b/eagle3/1l_kl_exp.sh"
  ["qwen3-4b|eagle3|16"]="examples/colocated/qwen3_4b/eagle3/1l_kl_exp.sh"
)

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

list_all() {
    echo "model       method   budget  config"
    local model method b key
    for model in $MODELS; do
        for method in $METHODS; do
            for b in 8 16; do
                key="$model|$method|$b"
                [ -n "${SCRIPT[$key]:-}" ] || continue
                # derive the config name from the script for a compact view
                local cfg
                cfg=$(grep -oE "configs/[A-Za-z0-9_.-]+\.json" "$ROOT_DIR/${SCRIPT[$key]}" | head -1)
                printf "%-11s %-8s %-6s  %s\n" "$model" "$method" "$b" "${cfg:-<auto>}"
            done
        done
    done
    echo
    echo "EAGLE3: one training run per model (budget only affects serving trees)."
}

# ---- defaults / env passthrough -------------------------------------------
GPUS=""            # auto-detect if unset
BUDGET="$DEFAULT_BUDGET"
WANDB=on
RUN_NAME_ARG=""
MASTER_PORT_ARG=""
DRY_RUN=0

# ---- argument parsing ------------------------------------------------------
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --budget) BUDGET="$2"; shift 2 ;;
        --gpus)   GPUS="$2"; shift 2 ;;
        --no-wandb) WANDB=off; shift ;;
        --run-name) RUN_NAME_ARG="$2"; shift 2 ;;
        --master-port) MASTER_PORT_ARG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --list|-l) list_all; exit 0 ;;
        --help|-h) usage ;;
        -*) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
MODEL="${POSITIONAL[0]:-}"
METHOD="${POSITIONAL[1]:-}"

[ -n "$MODEL" ] || { usage >&2; exit 2; }
case " $MODELS " in *" $MODEL "*) ;; *) echo "unknown model: $MODEL (choose from: $MODELS)" >&2; exit 2 ;; esac
case " $METHODS " in *" $METHOD"*) ;; *) echo "unknown method: $METHOD (choose from: $METHODS)" >&2; exit 2 ;; esac
case "$BUDGET" in 8|16) ;; *) echo "--budget must be 8 or 16" >&2; exit 2 ;; esac

KEY="$MODEL|$METHOD|$BUDGET"
SCRIPT_REL="${SCRIPT[$KEY]:-}"
if [ -z "$SCRIPT_REL" ]; then
    echo "no launch script for $KEY" >&2
    exit 2
fi

TARGET_REL="cache/models/Qwen3-$( [ "$MODEL" = "qwen3-8b" ] && echo 8B || echo 4B )"
# Default TRAIN_DATA_PATH of every launch script for this model.
DATA_REL="cache/dataset/perfectblend/perfectblend_train_regen_${MODEL//-/_}.jsonl"
SCRIPT_ABS="$ROOT_DIR/$SCRIPT_REL"

# ---- preflight --------------------------------------------------------------
failed=0
if [ ! -d "$TARGET_REL" ]; then
    echo "ERROR: target model weights not found at $TARGET_REL" >&2
    echo "       run: bash scripts/download_target_models.sh $MODEL" >&2
    failed=1
fi
if [ ! -f "$SCRIPT_ABS" ]; then
    echo "ERROR: launch script missing: $SCRIPT_ABS" >&2
    failed=1
fi
if [ "$DRY_RUN" -eq 0 ] && [ -z "$GPUS" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
    if [ -z "${GPUS:-0}" ] || [ "$GPUS" -eq 0 ]; then
        echo "ERROR: cannot detect GPUs; pass --gpus N" >&2
        failed=1
    fi
fi
[ "$failed" -eq 0 ] || exit 1

# ---- train data (downloaded from Hugging Face on first use) -----------------
if [ ! -f "$DATA_REL" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "NOTE: $DATA_REL not found; a real run downloads it from Hugging Face first"
    else
        VENV=${VENV:-${UV_PROJECT_ENVIRONMENT:-$ROOT_DIR/.venv}}
        PYTHON=${PYTHON:-$( [ -x "$VENV/bin/python" ] && echo "$VENV/bin/python" || echo python3 )}
        echo "==> train data not found at $DATA_REL; downloading from Hugging Face (~6 GB) ..."
        if ! HF_HUB_OFFLINE=0 "$PYTHON" scripts/download_training_data.py "$MODEL"; then
            echo "ERROR: failed to download train data; check network/proxy and rerun," >&2
            echo "       or regenerate it locally: bash scripts/prepare_training_data.sh $MODEL" >&2
            exit 1
        fi
    fi
fi

# ---- assemble env -----------------------------------------------------------
ENV_OVERRIDES=(REPORT_TO=none)
[ "$WANDB" = "on" ] && ENV_OVERRIDES=()
[ -n "$RUN_NAME_ARG" ] && ENV_OVERRIDES+=("RUN_NAME=$RUN_NAME_ARG")
[ -n "$MASTER_PORT_ARG" ] && ENV_OVERRIDES+=("MASTER_PORT=$MASTER_PORT_ARG")

echo "==> model=$MODEL method=$METHOD budget=$BUDGET gpus=${GPUS:-?} wandb=$WANDB"
echo "==> launcher: $SCRIPT_REL"
echo "    (overrides: ${ENV_OVERRIDES[*]:-none})"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN — command would be:"
    echo "    env ${ENV_OVERRIDES[*]} bash $SCRIPT_REL ${GPUS:-<auto>}"
    exit 0
fi

exec env "${ENV_OVERRIDES[@]}" bash "$SCRIPT_ABS" "$GPUS"