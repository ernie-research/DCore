#!/bin/bash
# MASTER_PORT=29502 nohup bash examples/colocated/qwen3_8b/dflash/5l_bs7_anchor_5hs_ce_exp.sh 8 > outputs/perfectblend/qwen3_8b_dflash_5l_bs7_anchor_5hs_ce_exp.log 2>&1 &
# tail -f outputs/perfectblend/qwen3_8b_dflash_5l_bs7_anchor_5hs_ce_exp.log
#
# Budget note: with --anchor-prediction every block position emits a draft token,
# so block_size 7 gives 7 draft tokens and 8 verified positions -- the same budget
# as plain DFlash at block_size 8, but one fewer block position through the draft
# backbone. That saving is the whole point of the flag; running this at block_size
# 8 instead (as an earlier version of this script did) drafts 8 and verifies 9,
# which enlarges the budget rather than making it cheaper. The 7 has to live in
# the draft config: train_dflash.py lets the config's block_size win over
# --block-size (it only prints a warning), hence qwen3-8b-dflash-bs7.json.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)

# Load from .env (create .env at project root with WANDB_API_KEY=<your key>)
[ -f "$(dirname "$0")/../../../../.env" ] && source "$(dirname "$0")/../../../../.env"

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export NCCL_DEBUG=WARN
export SPECFORGE_DATA_NUM_PROC=32

NUM_GPUS=${1:-8}
ATTENTION_BACKEND=${2:-flex_attention}
NUM_DRAFT_LAYERS=${NUM_DRAFT_LAYERS:-5}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-cache/models/Qwen3-8B}
unset TRAIN_DATA_PATH
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-cache/dataset/perfectblend/perfectblend_train_regen_qwen3_8b.jsonl}

DPACE_ALPHA=${DPACE_ALPHA:-0.5}
RUN_NAME=${RUN_NAME:-qwen3_8b_dflash_5l_bs7_anchor_5hs_ce_exp}

MASTER_PORT=${MASTER_PORT:-29500}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
     $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflash-bs7.json \
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
    --loss-decay-gamma 4.0 \
    --log-interval 20 \
    --save-schedule "0.2,0.5,0.8,1.0,1.5,2.0,2.5,3.0" \
    --report-to wandb \
    --wandb-project qwen3-8b-draft-volley \
    --target-model-backend sglang \
    --block-size 7 \
    --num-draft-layers $NUM_DRAFT_LAYERS \
    --loss-type dflash \
    --dpace-alpha $DPACE_ALPHA \
    --anchor-prediction \
    --wandb-name $RUN_NAME
