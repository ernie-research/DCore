#!/usr/bin/env bash
# Batch 3: evaluate dpace_l1-7seg, dpace_l1-15seg, refiner-7seg
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# --- dpace_l1-7seg ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=29500 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-dpace_l1-7seg/epoch_3_step_132255" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_21706.pt \
  bash eval/run_shotgun_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=29550 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-dpace_l1-7seg/epoch_3_step_132255" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_21706.pt \
  bash eval/run_shotgun_eval.sh

# --- dpace_l1-15seg ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=29600 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-dpace_l1-15seg/epoch_3_step_131412" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_10130.pt \
  bash eval/run_shotgun_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=29650 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-dpace_l1-15seg/epoch_3_step_131412" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_10130.pt \
  bash eval/run_shotgun_eval.sh

# --- refiner-7seg ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=29700 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-refiner-7seg/epoch_3_step_132255" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_21706.pt \
  bash eval/run_shotgun_eval.sh

SKIP_BASELINE=1 TEMPERATURE=1.0 MASTER_PORT=29750 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-refiner-7seg/epoch_3_step_132255" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_21706.pt \
  bash eval/run_shotgun_eval.sh

echo "=== Batch 3 evaluation complete ==="
