#!/usr/bin/env bash
# Batch 1: evaluate dflash_l1_unnorm-7seg, dflash_l1_unnorm-15seg, dflash_l1-7seg
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# --- dflash_l1_unnorm-7seg ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=29500 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-dflash_l1_unnorm-7seg/epoch_3_step_132255" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_21706.pt \
  bash eval/run_shotgun_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=29550 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-dflash_l1_unnorm-7seg/epoch_3_step_132255" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_21706.pt \
  bash eval/run_shotgun_eval.sh

# --- dflash_l1_unnorm-15seg ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=29600 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-dflash_l1_unnorm-15seg/epoch_3_step_131412" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_10130.pt \
  bash eval/run_shotgun_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=29650 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-dflash_l1_unnorm-15seg/epoch_3_step_131412" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_10130.pt \
  bash eval/run_shotgun_eval.sh

# --- dflash_l1-7seg ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=29700 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-dflash_l1-7seg/epoch_3_step_132255" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_21706.pt \
  bash eval/run_shotgun_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=29750 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-dflash_l1-7seg/epoch_3_step_132255" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_21706.pt \
  bash eval/run_shotgun_eval.sh

echo "=== Batch 1 evaluation complete ==="
