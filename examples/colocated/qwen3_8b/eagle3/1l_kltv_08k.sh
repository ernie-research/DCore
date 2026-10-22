#!/bin/bash
# Qwen3-8B EAGLE-3 (ttt_length 7 -> budget 8) loss ablation: 08k position weights x KL+TV.
# Objective: L = mean_w[0.1 * KL(p || q) + 0.9 * ||p - q||_1], w = eval_mask * <position weight>.
# Position weights: EAGLE-3's 0.8^k.
# 0.8^k == exp(-k / gamma) with gamma = 1/ln(1.25) = 4.481420117724549.
# MASTER_PORT=29610 nohup bash examples/colocated/qwen3_8b/eagle3/1l_kltv_08k.sh 8 > outputs/perfectblend/qwen3_8b_eagle3_1l_kltv_08k.log 2>&1 &
# tail -f outputs/perfectblend/qwen3_8b_eagle3_1l_kltv_08k.log
# pkill -f "train_eagle3.py"
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

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp8 train eagle3 for Qwen3-4B/8B/32B up to tp_size = 8
NUM_GPUS=${1:-8}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

unset TRAIN_DATA_PATH
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-cache/models/Qwen3-8B}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-cache/dataset/perfectblend/perfectblend_train_regen_qwen3_8b.jsonl}

RUN_NAME=${RUN_NAME:-qwen3_8b_eagle3_1l_kltv_08k}
MASTER_PORT=${MASTER_PORT:-29610}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3.json \
    --train-data-path $TRAIN_DATA_PATH \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/perfectblend/$RUN_NAME \
    --num-epochs 3 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --max-length 3072 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --loss-type kltv \
    --kltv-kl-alpha 0.1 \
    --kltv-l1-alpha 0.9 \
    --position-weighting decay \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-mem-fraction-static 0.4 \
    --log-interval 20 \
    --save-schedule "0.2,0.5,0.8,1.0,1.5,2.0,2.5,3.0" \
    --report-to wandb \
    --wandb-project qwen3-8b-draft-parall_perfectblend \
    --wandb-name $RUN_NAME
