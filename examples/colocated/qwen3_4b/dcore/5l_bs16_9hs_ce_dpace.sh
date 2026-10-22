#!/bin/bash
# Qwen3-4B counterpart of examples/colocated/qwen3_8b/dcore/5l_bs16_9hs_ce_dpace.sh.
# DFlare(9 target hs) backbone + D-PACE(CE) loss, block_size 16.
#
# TARGET_MODEL_PATH / TRAIN_DATA_PATH are placeholders -- fill them in below or
# override from the environment.
# MASTER_PORT=29540 nohup bash examples/colocated/qwen3_4b/dcore/5l_bs16_9hs_ce_dpace.sh 8 > outputs/perfectblend/qwen3_4b_dcore_5l_bs16_9hs_ce_dpace.log 2>&1 &
# tail -f outputs/perfectblend/qwen3_4b_dcore_5l_bs16_9hs_ce_dpace.log
# pkill -f "train_dflash.py"
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)

# Load from .env (create .env at project root with WANDB_API_KEY=<your key>)
[ -f "$(dirname "$0")/../../../../.env" ] && source "$(dirname "$0")/../../../../.env"

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# The system python has an older transformers than pyproject requires, so
# specforge fails to import. Pin to the project venv (same convention as
# eval/run_eval_*.sh) instead of relying on the caller's shell.
VENV=${VENV:-${UV_PROJECT_ENVIRONMENT:-$ROOT_DIR/.venv}}
if [ ! -x "$VENV/bin/python" ]; then
    echo "venv not found at $VENV; override with VENV=<path>" >&2
    exit 1
fi
export PATH="$VENV/bin:$PATH"

export NCCL_DEBUG=WARN
export SPECFORGE_DATA_NUM_PROC=32

NUM_GPUS=${1:-8}
ATTENTION_BACKEND=${2:-flex_attention}
NUM_DRAFT_LAYERS=${NUM_DRAFT_LAYERS:-5}

# TODO: fill in the Qwen3-4B target model path and the regenerated train data.
# TRAIN_DATA_PATH is unset first because the surrounding environment may already
# export an unrelated value, which would silently win over the placeholder.
unset TRAIN_DATA_PATH
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-cache/models/Qwen3-4B}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-cache/dataset/perfectblend/perfectblend_train_regen_qwen3_4b.jsonl}
case "$TARGET_MODEL_PATH$TRAIN_DATA_PATH" in
    *FILL_ME_*)
        echo "TARGET_MODEL_PATH / TRAIN_DATA_PATH still hold placeholders; edit this script or export them" >&2
        exit 1
        ;;
esac

DPACE_ALPHA=${DPACE_ALPHA:-0.5}
RUN_NAME=${RUN_NAME:-qwen3_4b_dcore_5l_bs16_9hs_ce_dpace}

MASTER_PORT=${MASTER_PORT:-29500}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
     $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-config-path $ROOT_DIR/configs/qwen3-4b-dflare_9hs.json \
    --train-data-path $TRAIN_DATA_PATH \
    --output-dir $ROOT_DIR/outputs/perfectblend/$RUN_NAME \
    --num-epochs 3 \
    --batch-size 2 \
    --accumulation-steps 2 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 512 \
    --log-interval 20 \
    --save-schedule "0.2,0.5,0.8,1.0,1.5,2.0,2.5,3.0" \
    --report-to ${REPORT_TO:-wandb} \
    --wandb-project qwen3-4b-draft-volley \
    --target-model-backend sglang \
    --sglang-mem-fraction-static 0.40 \
    --block-size 16 \
    --num-draft-layers $NUM_DRAFT_LAYERS \
    --loss-type dpace \
    --dpace-alpha $DPACE_ALPHA \
    --wandb-name $RUN_NAME
