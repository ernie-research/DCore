#!/bin/bash
set -euo pipefail

# nohup bash examples/colocated/qwen3_8b/dflash/perfectblend_5l_bs16_5hs_ce_exp.sh > outputs/perfectblend/qwen3_8b_dflash_perfectblend_5l_bs16_5hs_ce_exp.log 2>&1 &

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)

# Load from .env (create .env at project root with WANDB_API_KEY=<your key>)
[ -f "$(dirname "$0")/../../../../.env" ] && source "$(dirname "$0")/../../../../.env"

export CUDA_HOME=/usr/local/cuda
export CONDA_PREFIX=${CONDA_PREFIX:-$ROOT_DIR/.venv}
export PATH=$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export NCCL_DEBUG=WARN
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32

NUM_GPUS=${1:-8}
RUN_NAME=${RUN_NAME:-qwen3_8b_dflash_perfectblend_5l_bs16_5hs_ce_exp}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path cache/models/Qwen3-8B \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflash.json \
    --train-data-path cache/dataset/perfectblend/perfectblend_train_regen_qwen3_8b.jsonl \
    --output-dir $ROOT_DIR/outputs/perfectblend/$RUN_NAME \
    --num-epochs 3 \
    --batch-size 2 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen \
    --attention-backend flex_attention \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 1000 \
    --report-to wandb \
    --wandb-project specforge-qwen3-8b-draft-parall \
    --wandb-name $RUN_NAME \
    --target-model-backend hf \
    --block-size 16 \
    --num-anchors 512
