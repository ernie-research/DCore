#!/usr/bin/env bash
# Batch 2: dflash-block_size_16-dpace-full, domino-16draft-full, domino-b8, dspark-markov-blocksize_15, dspark-markov-blocksize_7
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# --- dflash-block_size_16-dpace-full ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=42000 \
  DRAFT_MODEL="outputs/perfectblend/dflash-block_size_16-dpace-full/epoch_2_step_216000" \
  bash eval/run_dflash_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=42200 \
  DRAFT_MODEL="outputs/perfectblend/dflash-block_size_16-dpace-full/epoch_2_step_216000" \
  bash eval/run_dflash_eval.sh

# --- domino-16draft-full ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=42400 \
  DRAFT_MODEL="outputs/perfectblend/domino-16draft-full/epoch_3_step_262647" \
  bash eval/run_domino_eval_supported_by_dflash.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=42600 \
  DRAFT_MODEL="outputs/perfectblend/domino-16draft-full/epoch_3_step_262647" \
  bash eval/run_domino_eval_supported_by_dflash.sh

# --- domino-b8 ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=42800 \
  DRAFT_MODEL="outputs/perfectblend/domino-b8/epoch_3_step_264279" \
  bash eval/run_domino_eval_supported_by_dflash.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=43000 \
  DRAFT_MODEL="outputs/perfectblend/domino-b8/epoch_3_step_264279" \
  bash eval/run_domino_eval_supported_by_dflash.sh

# --- dspark-markov-blocksize_15 ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=43200 \
  DRAFT_MODEL="outputs/perfectblend/dspark-markov-blocksize_15/epoch_3_step_262647" \
  bash eval/run_dspark_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=43400 \
  DRAFT_MODEL="outputs/perfectblend/dspark-markov-blocksize_15/epoch_3_step_262647" \
  bash eval/run_dspark_eval.sh

# --- dspark-markov-blocksize_7 ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=43600 \
  DRAFT_MODEL="outputs/perfectblend/dspark-markov-blocksize_7/epoch_3_step_262647" \
  bash eval/run_dspark_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=43800 \
  DRAFT_MODEL="outputs/perfectblend/dspark-markov-blocksize_7/epoch_3_step_262647" \
  bash eval/run_dspark_eval.sh

# --- eagle3 (steps=7, topk=1, draft=7) ---
SKIP_BASELINE=1 TEMPERATURE=0.0 \
  DRAFT_MODEL="outputs/perfectblend/eagle3/epoch_2_step_266421" \
  SPECULATIVE_NUM_STEPS=7 SPECULATIVE_EAGLE_TOPK=1 SPECULATIVE_NUM_DRAFT_TOKENS=7 \
  bash eval/run_eagle3_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 \
  DRAFT_MODEL="outputs/perfectblend/eagle3/epoch_2_step_266421" \
  SPECULATIVE_NUM_STEPS=7 SPECULATIVE_EAGLE_TOPK=1 SPECULATIVE_NUM_DRAFT_TOKENS=7 \
  bash eval/run_eagle3_eval.sh

echo "=== Batch 2 (perfectblend) evaluation complete ==="
