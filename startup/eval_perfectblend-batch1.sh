#!/usr/bin/env bash
# Batch 1: dflare-b16, dflare-b8, dflare-block_size_16-dpace-full, dflash-b16, dflash-b8
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# --- dflare-b16 ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=40000 \
  DRAFT_MODEL="outputs/perfectblend/dflare-b16/epoch_3_step_262647" \
  bash eval/run_dflare_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=40200 \
  DRAFT_MODEL="outputs/perfectblend/dflare-b16/epoch_3_step_262647" \
  bash eval/run_dflare_eval.sh

# --- dflare-b8 ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=40400 \
  DRAFT_MODEL="outputs/perfectblend/dflare-b8/epoch_3_step_264279" \
  bash eval/run_dflare_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=40600 \
  DRAFT_MODEL="outputs/perfectblend/dflare-b8/epoch_3_step_264279" \
  bash eval/run_dflare_eval.sh

# --- dflare-block_size_16-dpace-full ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=40800 \
  DRAFT_MODEL="outputs/perfectblend/dflare-block_size_16-dpace-full/epoch_3_step_262647" \
  bash eval/run_dflare_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=41000 \
  DRAFT_MODEL="outputs/perfectblend/dflare-block_size_16-dpace-full/epoch_3_step_262647" \
  bash eval/run_dflare_eval.sh

# --- dflash-b16 ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=41200 \
  DRAFT_MODEL="outputs/perfectblend/dflash-b16/epoch_3_step_262647" \
  bash eval/run_dflash_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=41400 \
  DRAFT_MODEL="outputs/perfectblend/dflash-b16/epoch_3_step_262647" \
  bash eval/run_dflash_eval.sh

# --- dflash-b8 ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=41600 \
  DRAFT_MODEL="outputs/perfectblend/dflash-b8/epoch_3_step_264279" \
  bash eval/run_dflash_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=41800 \
  DRAFT_MODEL="outputs/perfectblend/dflash-b8/epoch_3_step_264279" \
  bash eval/run_dflash_eval.sh


# --- dpace-b16 ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=42000 \
  DRAFT_MODEL="outputs/perfectblend/dflash-block_size_16-dpace-full/epoch_3_step_262647" \
  bash eval/run_dflash_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=42200 \
  DRAFT_MODEL="outputs/perfectblend/dflash-block_size_16-dpace-full/epoch_3_step_262647" \
  bash eval/run_dflash_eval.sh

echo "=== Batch 1 (perfectblend) evaluation complete ==="
