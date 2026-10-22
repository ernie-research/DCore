#!/bin/bash
# DFlash training loop + anchor prediction + DFlare(9hs) backbone + D-PACE(LK) loss.
# Ablation pair for examples/colocated/qwen3_8b/dcore/5l_bs15_anchor_9hs_ce_dpace.sh: the
# only difference is --loss-type dpace-lk, which keeps the same D-PACE position
# weights but replaces the CE objective with DFly stage-1 LK
# (lambda*KL + (1-lambda)*TV, lambda = exp(-eta*(1-TV))) and normalizes by the
# weight sum instead of the batch size. Requires the sglang backend to return the
# target's final hidden states (target distribution = lm_head(last_hidden)).
#
# Budget note: anchor prediction adds one draft token, so block_size is 15 here to
# stay budget-matched with the block_size 16 baseline -- 15 draft tokens and 16
# verified positions either way, but one fewer block position through the draft
# backbone. At block_size 16 it would draft 16 and verify 17 instead. The 15 must
# live in the draft config: train_dflash.py lets the config's block_size win over
# --block-size (it only prints a warning), hence qwen3-8b-dflare_9hs-bs15.json.
# MASTER_PORT=29533 nohup bash examples/colocated/qwen3_8b/dcore/5l_bs15_anchor_9hs_kltv_dpace.sh 8 > outputs/perfectblend/qwen3_8b_dcore_5l_bs15_anchor_9hs_kltv_dpace.log 2>&1 &
# tail -f outputs/perfectblend/qwen3_8b_dcore_5l_bs15_anchor_9hs_kltv_dpace.log
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
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-cache/models/Qwen3-8B}
unset TRAIN_DATA_PATH
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-cache/dataset/perfectblend/perfectblend_train_regen_qwen3_8b.jsonl}

DPACE_ALPHA=${DPACE_ALPHA:-0.5}
LK_ETA=${LK_ETA:-3.0}
RUN_NAME=${RUN_NAME:-qwen3_8b_dcore_5l_bs15_anchor_9hs_kltv_dpace}

MASTER_PORT=${MASTER_PORT:-29500}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
     $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflare_9hs-bs15.json \
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
    --sglang-mem-fraction-static 0.35 \
    --block-size 15 \
    --num-draft-layers $NUM_DRAFT_LAYERS \
    --loss-type dpace-lk \
    --dpace-alpha $DPACE_ALPHA \
    --lk-eta $LK_ETA \
    --anchor-prediction \
    --wandb-name $RUN_NAME \
    --resume
