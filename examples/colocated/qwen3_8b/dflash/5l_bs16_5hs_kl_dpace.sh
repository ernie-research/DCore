#!/bin/bash
# Qwen3-8B DFlash bs16 loss ablation: D-PAKL (D-PACE App. E) — the KL variant of
# D-PACE. Fills the "D-Pace & KL" cell of figure/loss_interaction.tex (DFlash side).
# Objective: L = mean_w[KL(p||q)], w = eval_mask * dpace_weight(a_hat), where the
# acceptance proxy is the detached a_hat_i = exp(-C_i) with C_i = -sum_x p_i(x)
# log q_psi(x) the soft-label CE (the Jensen lower bound of <p_i, q_psi>), NOT
# the hard-label exp(-CE(z*)) that plain D-PACE uses. No static 0.8^k decay here.
# Requires the sglang backend to return the target's final hidden states.
# MASTER_PORT=29533 nohup bash examples/colocated/qwen3_8b/dflash/5l_bs16_5hs_kl_dpace.sh 8 > outputs/perfectblend/qwen3_8b_dflash_5l_bs16_5hs_kl_dpace.log 2>&1 &
# tail -f outputs/perfectblend/qwen3_8b_dflash_5l_bs16_5hs_kl_dpace.log
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

DPACE_ALPHA=${DPACE_ALPHA:-0.5}
RUN_NAME=${RUN_NAME:-qwen3_8b_dflash_5l_bs16_5hs_kl_dpace}
MASTER_PORT=${MASTER_PORT:-29533}

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
    --sglang-mem-fraction-static 0.35 \
    --block-size 16 \
    --num-draft-layers $NUM_DRAFT_LAYERS \
    --loss-type dpace-kl \
    --dpace-alpha $DPACE_ALPHA \
    --wandb-name $RUN_NAME
