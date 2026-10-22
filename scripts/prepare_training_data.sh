#!/usr/bin/env bash
# Full-service training data pipeline (OpenPerfectBlend -> target-model regeneration).
#
#   1. Download mlabonne/open-perfectblend and convert it to ShareGPT jsonl
#      (scripts/prepare_data.py --dataset perfectblend), idempotent.
#   2. Launch an SGLang server for the target model (or attach to a running one).
#   3. Regenerate every assistant turn with the target model
#      (scripts/regenerate_train_data.py).
#   4. Validate the result (scripts/validate_regenerated_data.py).
#
# The output lands at cache/dataset/perfectblend/perfectblend_train_regen_<model>.jsonl,
# the default TRAIN_DATA_PATH of the launch scripts. The released copy of this data
# is on Hugging Face (scripts/download_training_data.py); use this pipeline only to
# regenerate it yourself.
#
# Usage:
#   bash scripts/prepare_training_data.sh qwen3-8b
#   bash scripts/prepare_training_data.sh qwen3-4b --tp 1 --reasoning disable
#   bash scripts/prepare_training_data.sh qwen3-8b --server-address 10.0.0.1:30000
#
# Requires: target weights at cache/models/Qwen3-{8B,4B} (see
#           scripts/download_target_models.sh), and a GPU (for the server).
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${UV_PROJECT_ENVIRONMENT:-$ROOT_DIR/.venv}/bin/python}"
export PYTHONPATH="$ROOT_DIR/third_party/sglang/python:${PYTHONPATH:-}"

TARGET=""
REASONING="disable"     # none | save | disable (disable = enable_thinking=False)
TEMPERATURE="0.0"
CONCURRENCY="64"
MAX_TOKENS="4096"
PORT="31000"
TP="1"
MEM_FRACTION="0.7"
SERVER_ADDRESS=""
SAMPLE_SIZE=""
NUM_SAMPLES=""
OUTPUT_FILE=""
RESUME=1

usage() {
    cat <<EOF
usage: bash scripts/prepare_training_data.sh <qwen3-8b|qwen3-4b> [options]

optional:
  --reasoning {none|save|disable}  regen mode (default: disable)
  --temperature F                  sampling temperature (default: 0.0, greedy)
  --concurrency N                  client concurrency (default: 64)
  --max-tokens N                   max new tokens per assistant turn (default: 4096)
  --server-address HOST:PORT       reuse a running SGLang server (default: launch one)
  --port N                         port for the launched server (default: $PORT)
  --tp N                           tensor-parallel size for the launched server (default: $TP)
  --mem-fraction F                 mem-fraction-static for the launched server (default: $MEM_FRACTION)
  --sample-size N                  limit the base OpenPerfectBlend rows (for smoke tests)
  --num-samples N                  limit the regenerated samples (for smoke tests)
  --output-file PATH               output jsonl path (default: cache/dataset/perfectblend/...)
  --no-resume                      overwrite instead of resuming from an existing output file
  -h|--help                        show this help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        qwen3-8b|qwen3-4b) TARGET="$1" ;;
        --reasoning) REASONING="$2"; shift ;;
        --temperature) TEMPERATURE="$2"; shift ;;
        --concurrency) CONCURRENCY="$2"; shift ;;
        --max-tokens) MAX_TOKENS="$2"; shift ;;
        --server-address) SERVER_ADDRESS="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --tp) TP="$2"; shift ;;
        --mem-fraction) MEM_FRACTION="$2"; shift ;;
        --sample-size) SAMPLE_SIZE="$2"; shift ;;
        --num-samples) NUM_SAMPLES="$2"; shift ;;
        --output-file) OUTPUT_FILE="$2"; shift ;;
        --no-resume) RESUME=0 ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

case "$REASONING" in none|save|disable) ;; *)
    echo "--reasoning must be one of: none save disable" >&2; exit 2 ;; esac

case "$TARGET" in
    qwen3-8b) TARGET_DIR="$ROOT_DIR/cache/models/Qwen3-8B"; MODEL_LABEL="qwen3_8b" ;;
    qwen3-4b) TARGET_DIR="$ROOT_DIR/cache/models/Qwen3-4B"; MODEL_LABEL="qwen3_4b" ;;
    *) echo "usage: bash scripts/prepare_training_data.sh <qwen3-8b|qwen3-4b>" >&2; usage >&2; exit 2 ;;
esac

BASE_DIR="$ROOT_DIR/cache/dataset/perfectblend"
BASE_INPUT="$BASE_DIR/perfectblend_train.jsonl"
OUTPUT_FILE="${OUTPUT_FILE:-$BASE_DIR/perfectblend_train_regen_${MODEL_LABEL}.jsonl}"
mkdir -p "$BASE_DIR"

# ---- preflight --------------------------------------------------------------
if [ ! -f "$TARGET_DIR/config.json" ]; then
    echo "ERROR: target weights not found at $TARGET_DIR" >&2
    echo "       run: bash scripts/download_target_models.sh $TARGET" >&2
    exit 1
fi

# ---- 1. base dataset (OpenPerfectBlend -> ShareGPT jsonl), idempotent --------
if [ -f "$BASE_INPUT" ]; then
    echo "==> base dataset already present: $BASE_INPUT"
else
    echo "==> downloading mlabonne/open-perfectblend and converting to $BASE_INPUT"
    prepare_args=(--dataset perfectblend --output-path "$BASE_DIR")
    [ -n "$SAMPLE_SIZE" ] && prepare_args+=(--sample-size "$SAMPLE_SIZE")
    "$PYTHON" "$ROOT_DIR/scripts/prepare_data.py" "${prepare_args[@]}"
fi

# ---- 2. SGLang server -------------------------------------------------------
SRV_PID=""
cleanup() {
    if [ -n "$SRV_PID" ] && kill -0 "$SRV_PID" 2>/dev/null; then
        echo "==> stopping SGLang server (pid $SRV_PID)"
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [ -z "$SERVER_ADDRESS" ]; then
    LOG_DIR="$ROOT_DIR/outputs/data_regen"
    LOG_FILE="$LOG_DIR/$(basename "$OUTPUT_FILE" .jsonl).server.log"
    mkdir -p "$LOG_DIR"
    echo "==> launching SGLang server: $TARGET_DIR (tp=$TP, port=$PORT)"
    echo "    log: $LOG_FILE"
    "$PYTHON" -m sglang.launch_server \
        --model-path "$TARGET_DIR" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tp "$TP" \
        --dtype bfloat16 \
        --mem-fraction-static "$MEM_FRACTION" \
        >"$LOG_FILE" 2>&1 &
    SRV_PID=$!
    echo "==> waiting for server to become healthy ..."
    ok=0
    for _ in $(seq 1 1200); do
        if ! kill -0 "$SRV_PID" 2>/dev/null; then
            echo "ERROR: SGLang server exited early; last log lines:" >&2
            tail -n 30 "$LOG_FILE" >&2 || true
            exit 1
        fi
        if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            ok=1
            break
        fi
        sleep 2
    done
    if [ "$ok" -ne 1 ]; then
        echo "ERROR: timed out waiting for SGLang server; last log lines:" >&2
        tail -n 30 "$LOG_FILE" >&2 || true
        exit 1
    fi
    SERVER_ADDRESS="127.0.0.1:$PORT"
fi
echo "==> using server at $SERVER_ADDRESS"

# ---- 3. regenerate ----------------------------------------------------------
regen_args=(
    --model "$TARGET_DIR"
    --server-address "$SERVER_ADDRESS"
    --input-file-path "$BASE_INPUT"
    --output-file-path "$OUTPUT_FILE"
    --concurrency "$CONCURRENCY"
    --temperature "$TEMPERATURE"
    --max-tokens "$MAX_TOKENS"
    --reasoning "$REASONING"
)
[ "$RESUME" -eq 1 ] && regen_args+=(--resume)
[ -n "$NUM_SAMPLES" ] && regen_args+=(--num-samples "$NUM_SAMPLES")

echo "==> regenerating $BASE_INPUT with $TARGET_DIR -> $OUTPUT_FILE"
"$PYTHON" "$ROOT_DIR/scripts/regenerate_train_data.py" "${regen_args[@]}"

# ---- 4. validate ------------------------------------------------------------
val_args=(--data-path "$OUTPUT_FILE")
if [ "$REASONING" = "save" ]; then
    val_args+=(--expect-reasoning)
else
    val_args+=(--expect-non-reasoning)
fi
echo "==> validating $OUTPUT_FILE"
"$PYTHON" "$ROOT_DIR/scripts/validate_regenerated_data.py" "${val_args[@]}"

echo "==> done. training data: $OUTPUT_FILE"
echo "    train.sh will pick it up automatically."