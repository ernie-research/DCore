#!/bin/bash
# Qwen3-4B DSpark (Markov head), block_size 15 -> verify window 16 (Budget=16).
# 4B port of examples/colocated/qwen3_8b/dspark/markov_5l_bs15_5hs_cetv_exp.sh. Fills the
# budget-matched bs15 weight that the existing 4B DSpark ckpts lack (they were
# trained at block_size 16, i.e. verify window 17). gamma = 7 follows the
# block-16-family convention (exp(-k/gamma) position decay).
#
# Notes vs the 8B script:
#   * --markov-head-type must be "vanilla" (current train_dspark.py choices are
#     vanilla|gated|rnn; the 8B script's "markov" is a stale alias and would
#     fail argparse). vanilla == the Markov head.
#   * The draft config json keeps the head settings under dflash_config --
#     DSparkDraftModel builds from dflash_config only. (The 8B jsons used to keep
#     them under dspark_config, which the builder ignores; that is now fixed.
#     dspark_config still exists in saved checkpoints as sglang's serving mirror.)
#
# TRAIN_DATA_PATH defaults to the Qwen3-4B-regenerated perfectblend data
# pulled from AFS (the 8B regen jsonl is NOT suitable).
# MASTER_PORT=29521 nohup bash examples/colocated/qwen3_4b/dspark/markov_5l_bs15_5hs_cetv_exp.sh 8 > outputs/Qwen3-4B/qwen3_4b_dspark_markov_5l_bs15_5hs_cetv_exp.log 2>&1 &
# tail -f outputs/Qwen3-4B/qwen3_4b_dspark_markov_5l_bs15_5hs_cetv_exp.log
# pkill -f train_dspark.py
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

# TRAIN_DATA_PATH is unset first because the surrounding environment may
# already export an unrelated value, which would silently win over the
# default.
unset TRAIN_DATA_PATH
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-cache/models/Qwen3-4B}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-cache/dataset/perfectblend/perfectblend_train_regen_qwen3_4b.jsonl}

# exp(-k / gamma); gamma = 7 for the block-16 family.
LOSS_DECAY_GAMMA=${LOSS_DECAY_GAMMA:-7.0}
RUN_NAME=${RUN_NAME:-qwen3_4b_dspark_markov_5l_bs15_5hs_cetv_exp}
MASTER_PORT=${MASTER_PORT:-29521}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
    $ROOT_DIR/scripts/train_dspark.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-config-path $ROOT_DIR/configs/qwen3-4b-dspark-bs15.json \
    --train-data-path $TRAIN_DATA_PATH \
    --output-dir $ROOT_DIR/outputs/Qwen3-4B/$RUN_NAME \
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
    --loss-decay-gamma $LOSS_DECAY_GAMMA \
    --ce-loss-alpha 0.1 \
    --l1-loss-alpha 0.9 \
    --confidence-head-alpha 1.0 \
    --markov-rank 256 \
    --markov-head-type vanilla \
    --log-interval 20 \
    --save-schedule "0.2,0.5,0.8,1.0,1.5,2.0,2.5,3.0" \
    --report-to wandb \
    --wandb-project qwen3-4b-draft-volley \
    --target-model-backend sglang \
    --wandb-name $RUN_NAME \
    --resume
