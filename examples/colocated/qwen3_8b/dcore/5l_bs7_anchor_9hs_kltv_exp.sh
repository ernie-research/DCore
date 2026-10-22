#!/bin/bash
# Qwen3-8B, block_size 8 + anchor prediction variant of
# examples/colocated/qwen3_8b/dcore/5l_bs15_anchor_9hs_kltv_08k.sh.
# DFlash training loop + anchor prediction + DFlare(9hs) backbone + fixed-ratio
# KL/L1 distribution loss.
# Objective:  L = mean_w[ 0.1 * KL(p||q) + 0.9 * ||p - q||_1 ],  ||p-q||_1 = 2*TV
# i.e. DSpark's configured 0.1*CE + 0.9*L1 with the hard-label CE replaced by the
# soft-label one (KL against the target distribution p = softmax(lm_head(h_target))).
# Position weights are the exponential decay exp(-k/gamma), with gamma scaled to the
# shorter block the way the dflash scripts do it (7 for block 16, 4 for block 8), so
# gamma = 4 here -- a 0.7788^k decay rather than the exact 0.8^k of the block_size 16
# scripts (gamma = 1/ln(1.25) = 4.481420117724549). Export LOSS_DECAY_GAMMA if you
# would rather have both block sizes share the identical decay base.
# Normalization is the weight sum (same as DSpark's valid_token_count, and as
# --loss-type dpace-lk), NOT the batch size.
#
# Difference vs examples/colocated/qwen3_8b/dcore/5l_bs8_9hs_ce_dpace.sh: same backbone and
# data, but the objective is a soft-label distribution distance with fixed mixing
# weights and static position weights, instead of hard-label CE with D-PACE
# continuation-value weights. No --dpace-alpha here.
# Requires the sglang backend to return the target's final hidden states, and runs
# at a lower --sglang-mem-fraction-static than the dpace scripts (0.35 vs 0.40)
# because the full-vocab fp32 probability tensors need the headroom.
#
# Anchor-prediction half of the ablation pair with
# examples/colocated/qwen3_8b/dcore/5l_bs8_9hs_kltv_exp.sh; the only differences are
# --anchor-prediction and block_size 7 instead of 8, to stay budget-matched
# (7 draft tokens, 8 verified positions either way). The 7 must live in the draft
# config: train_dflash.py lets the config's block_size win over --block-size (it
# only prints a warning), hence qwen3-8b-dflare_9hs-bs7.json. The decay offset
# (0 with anchor prediction, 1 without) makes the two runs see the identical
# weight vector exp(-0/4)..exp(-6/4) over their 7 draft positions.
# MASTER_PORT=29537 nohup bash examples/colocated/qwen3_8b/dcore/5l_bs7_anchor_9hs_kltv_exp.sh 8 > outputs/perfectblend/qwen3_8b_dcore_5l_bs7_anchor_9hs_kltv_exp.log 2>&1 &
# tail -f outputs/perfectblend/qwen3_8b_dcore_5l_bs7_anchor_9hs_kltv_exp.log
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

# exp(-k / gamma); gamma = 4 for block_size 8 (see the header note).
LOSS_DECAY_GAMMA=${LOSS_DECAY_GAMMA:-4.0}
KLTV_KL_ALPHA=${KLTV_KL_ALPHA:-0.1}
KLTV_L1_ALPHA=${KLTV_L1_ALPHA:-0.9}
RUN_NAME=${RUN_NAME:-qwen3_8b_dcore_5l_bs7_anchor_9hs_kltv_exp}

MASTER_PORT=${MASTER_PORT:-29500}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
     $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflare_9hs-bs7.json \
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
    --block-size 7 \
    --num-draft-layers $NUM_DRAFT_LAYERS \
    --loss-type kltv \
    --loss-decay-gamma $LOSS_DECAY_GAMMA \
    --kltv-kl-alpha $KLTV_KL_ALPHA \
    --kltv-l1-alpha $KLTV_L1_ALPHA \
    --anchor-prediction \
    --wandb-name $RUN_NAME
