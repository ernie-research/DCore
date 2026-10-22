#!/usr/bin/env bash
# Batch 3: dspark-rnn-blocksize_15, dspark-rnn-blocksize_7, eagle3
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# --- dspark-rnn-blocksize_15 ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=44000 \
  DRAFT_MODEL="outputs/perfectblend/dspark-rnn-blocksize_15/epoch_3_step_262647" \
  bash eval/run_dspark_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=44200 \
  DRAFT_MODEL="outputs/perfectblend/dspark-rnn-blocksize_15/epoch_3_step_262647" \
  bash eval/run_dspark_eval.sh

# --- dspark-rnn-blocksize_7 ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=44400 \
  DRAFT_MODEL="outputs/perfectblend/dspark-rnn-blocksize_7/epoch_3_step_262647" \
  bash eval/run_dspark_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=44600 \
  DRAFT_MODEL="outputs/perfectblend/dspark-rnn-blocksize_7/epoch_3_step_262647" \
  bash eval/run_dspark_eval.sh

# --- eagle3 ---
SKIP_BASELINE=1 TEMPERATURE=0.0 \
  DRAFT_MODEL="outputs/perfectblend/eagle3/epoch_2_step_266421" \
  bash eval/run_eagle3_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 \
  DRAFT_MODEL="outputs/perfectblend/eagle3/epoch_2_step_266421" \
  bash eval/run_eagle3_eval.sh

echo "=== Batch 3 (perfectblend) evaluation complete ==="
