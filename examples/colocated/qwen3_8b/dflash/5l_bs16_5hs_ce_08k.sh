#!/bin/bash
# Qwen3-8B DFlash bs16 loss ablation: 0.8^k position weights x hard-label CE.
# Fills the "0.8^k & CE" cell of figure/loss_interaction.tex (DFlash side).
# Objective: L = mean_w[CE], w = eval_mask * 0.8^k.
# 0.8^k == exp(-k / gamma) with gamma = 1/ln(1.25) = 4.481420117724549.
# MASTER_PORT=29530 nohup bash examples/colocated/qwen3_8b/dflash/5l_bs16_5hs_ce_08k.sh 8 > outputs/perfectblend/qwen3_8b_dflash_5l_bs16_5hs_ce_08k.log 2>&1 &
# tail -f outputs/perfectblend/qwen3_8b_dflash_5l_bs16_5hs_ce_08k.log
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_GPUS=${1:-8}
ATTENTION_BACKEND=${2:-flex_attention}
NUM_DRAFT_LAYERS=${NUM_DRAFT_LAYERS:-5}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-cache/models/Qwen3-8B}
unset TRAIN_DATA_PATH
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-cache/dataset/perfectblend/perfectblend_train_regen_qwen3_8b.jsonl}

# 0.8^k == exp(-k / gamma) with gamma = 1/ln(1.25).
GAMMA_08K=${GAMMA_08K:-4.481420117724549}
RUN_NAME=${RUN_NAME:-qwen3_8b_dflash_5l_bs16_5hs_ce_08k}
MASTER_PORT=${MASTER_PORT:-29530}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflash.json \
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
    --report-to wandb \
    --wandb-project qwen3-8b-draft-volley \
    --target-model-backend sglang \
    --block-size 16 \
    --num-draft-layers $NUM_DRAFT_LAYERS \
    --loss-type dflash \
    --loss-decay-gamma $GAMMA_08K \
    --wandb-name $RUN_NAME
