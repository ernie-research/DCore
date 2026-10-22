#!/usr/bin/env bash
# Evaluate all 4 shotgun experiments at temperature=0, skip baseline.
# Each experiment block uses a different MASTER_PORT range to avoid EADDRINUSE.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TASKS=${TASKS:-gsm8k:1319,math500:500,aime25:30,humaneval:164,mbpp:257,livecodebench:700,mt-bench:80,alpaca:500,arena-hard:500,swe-bench:300}
export TASKS

# --- shotgun1.1-7seg ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=29600 \
  DRAFT_MODEL="outputs/shotgun/shotgun1.1-7seg/epoch_*" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_21706.pt \
  bash eval/run_shotgun_eval.sh

# --- shotgun2.0-refiner-7seg ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=29700 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-refiner-7seg/epoch_*" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_21706.pt \
  bash eval/run_shotgun_eval.sh

# --- shotgun1.2-15seg ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=29800 \
  DRAFT_MODEL="outputs/shotgun/shotgun1.2-15seg/epoch_*" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_10130.pt \
  bash eval/run_shotgun_eval.sh

# --- shotgun2.0-refiner-15seg (config has num_segments=7, uses 21706 vocab mapping) ---
SKIP_BASELINE=1 TEMPERATURE=0.0 MASTER_PORT=29900 \
  DRAFT_MODEL="outputs/shotgun/shotgun2.0-refiner-15seg/epoch_*" \
  VOCAB_MAPPING_PATH=cache/models/Qwen3-8B/vocab_mapping_21706.pt \
  bash eval/run_shotgun_eval.sh

echo "=== All t=0.0 evaluations complete ==="
