#!/usr/bin/env bash
# Download the target models used by training / eval / data regeneration:
#   Qwen/Qwen3-8B  -> cache/models/Qwen3-8B
#   Qwen/Qwen3-4B  -> cache/models/Qwen3-4B
#
# Usage:
#   bash scripts/download_target_models.sh                     # both models
#   bash scripts/download_target_models.sh qwen3-8b            # only Qwen3-8B
#   bash scripts/download_target_models.sh qwen3-4b --force    # re-download even if present
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${UV_PROJECT_ENVIRONMENT:-$ROOT_DIR/.venv}/bin/python}"

# name|HF repo|local dir (relative to cache/models/)
MODELS=(
    "qwen3-8b|Qwen/Qwen3-8B|Qwen3-8B"
    "qwen3-4b|Qwen/Qwen3-4B|Qwen3-4B"
)

MODEL="all"
FORCE=0

usage() {
    cat <<EOF
usage: bash scripts/download_target_models.sh [model] [--force]

  model    qwen3-8b | qwen3-4b | all   (default: all)
  --force  re-download even if the directory already exists
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --force) FORCE=1 ;;
        all|qwen3-8b|qwen3-4b) MODEL="$1" ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

mkdir -p "$ROOT_DIR/cache/models"

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name repo local <<< "$entry"
    if [ "$MODEL" != "all" ] && [ "$MODEL" != "$name" ]; then
        continue
    fi
    dest="$ROOT_DIR/cache/models/$local"
    if [ -f "$dest/config.json" ] && [ "$FORCE" -eq 0 ]; then
        echo "==> $name already present at $dest (use --force to re-download)"
        continue
    fi
    echo "==> downloading $repo -> $dest"
    "$PYTHON" -c '
import sys
from huggingface_hub import snapshot_download
repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id, local_dir=local_dir)
' "$repo" "$dest" || {
    echo "ERROR: failed to download $repo" >&2
    echo "       if you are behind a proxy, export http_proxy/https_proxy first;" >&2
    echo "       or mirror via HF_ENDPOINT=https://hf-mirror.com" >&2
    exit 1
}
done

echo "==> done. models live under $ROOT_DIR/cache/models/"