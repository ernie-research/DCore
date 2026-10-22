#!/bin/bash
# Qwen3-8B DSpark (Markov head) bs15 -> verify window 16 loss ablation:
# dpace position weights x hard-label CE.
# Objective: L = mean_w[CE(argmax p, q)] + confidence-head BCE,
# w = eval_mask * <position weight>.
# Position weights: D-PACE detached continuation values (alpha 0.5), i.e.
# each position is weighted by its own predicted acceptance prospect
# instead of a static schedule.
# The confidence head's BCE target is always 1 - 0.5 * L1, so the target's
# final hidden states are fetched even with --l1-loss-alpha 0 and
# --kl-loss-alpha 0, hence the reduced --sglang-mem-fraction-static.
# MASTER_PORT=29638 nohup bash examples/colocated/qwen3_8b/dspark/markov_5l_bs15_5hs_ce_dpace.sh 8 > outputs/perfectblend/qwen3_8b_dspark_markov_5l_bs15_5hs_ce_dpace.log 2>&1 &
# tail -f outputs/perfectblend/qwen3_8b_dspark_markov_5l_bs15_5hs_ce_dpace.log
# pkill -f "train_dspark.py"
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)

# Load from .env (create .env at project root with WANDB_API_KEY=<your key>)
[ -f "$ROOT_DIR/.env" ] && source "$ROOT_DIR/.env"

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
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-cache/models/Qwen3-8B}
# TRAIN_DATA_PATH is unset first because the surrounding environment may
# already export an unrelated value, which would silently win over the
# default.
unset TRAIN_DATA_PATH
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-cache/dataset/perfectblend/perfectblend_train_regen_qwen3_8b.jsonl}

DPACE_ALPHA=${DPACE_ALPHA:-0.5}
RUN_NAME=${RUN_NAME:-qwen3_8b_dspark_markov_5l_bs15_5hs_ce_dpace}
MASTER_PORT=${MASTER_PORT:-29638}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
    $ROOT_DIR/scripts/train_dspark.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dspark-bs15.json \
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
    --position-weighting dpace \
    --dpace-alpha $DPACE_ALPHA \
    --ce-loss-alpha 1.0 \
    --l1-loss-alpha 0.0 \
    --kl-loss-alpha 0.0 \
    --confidence-head-alpha 1.0 \
    --markov-rank 256 \
    --markov-head-type vanilla \
    --log-interval 20 \
    --save-schedule "0.2,0.5,0.8,1.0,1.5,2.0,2.5,3.0" \
    --report-to wandb \
    --wandb-project qwen3-8b-draft-volley \
    --target-model-backend sglang \
    --sglang-mem-fraction-static 0.35 \
    --wandb-name $RUN_NAME \
    --resume
