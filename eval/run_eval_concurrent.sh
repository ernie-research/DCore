#!/usr/bin/env bash
# Multi-concurrency eval: measure one method's throughput on the SpecMix mixed-traffic set (concurrency >1 per replica).
#
# This is the simple public entry point, replacing the old run_specbench_cgt1_*_sweep_*.sh.
# It evaluates one method per invocation and measures TPS/TPOT/accept_length on
# SpecMix (3331 mixed-traffic prompts). On first use it is downloaded from
# https://huggingface.co/datasets/wingchiuloong/specmix into cache/dataset/specmix
# (scripts/build_specmix_dataset.py rebuilds the same set from the source benchmarks).
#
# Division of labor with run_eval_single.sh: that script measures accuracy + c1 latency on
# 10 benchmarks; this one measures mixed-traffic throughput under high concurrency. Both
# share the same method table and ckpt discovery rules.
#
# Usage:
#   bash eval/run_eval_concurrent.sh <qwen3-4b|qwen3-8b> <method> [block]
#
# Arguments:
#   model    qwen3-4b | qwen3-8b
#   method   same as run_eval_single.sh (dcore/dflash/.../eagle3_tree/origin variants)
#   block    verify window (default: all blocks of the method; dspark 7/15 maps to 8/16)
#
# Environment variables (all optional):
#   CONCURRENCIES  comma-separated per-replica concurrency (default 64, i.e. global 512)
#   DATASETS       eval set, default specmix:3331
#   TEMPERATURES   target temperatures, default "1,0" (t1rs rejection sampling + t0 greedy)
#   REPEATS        consecutive rounds at t1 (default 3; t0 is always 1)
#   MAX_NEW_TOKENS generation cap for the timed phase (default 2048; specmix averages 557 output tokens)
#   MEM_FRACTION_STATIC   KV pool fraction (default 0.75; must be below sglang's default 0.8 at high concurrency)
#   CHECKPOINT / SWEEP_ID / DP_SIZE / ENABLE_PROFILE same as the single-concurrency script
#
# Examples:
#   bash eval/run_eval_concurrent.sh qwen3-8b dcore
#   CONCURRENCIES=1,8,32,64 bash eval/run_eval_concurrent.sh qwen3-4b dspark_markov 16
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---------- Environment ------------------------------------------------------
PYTHON="${PYTHON:-${UV_PROJECT_ENVIRONMENT:-$ROOT_DIR/.venv}/bin/python}"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT_DIR/third_party/sglang/python:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nccl:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"
export SGLANG_FLASHINFER_VERIFY_ATTN_MODE="${SGLANG_FLASHINFER_VERIFY_ATTN_MODE:-prepacked}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# At high concurrency, verify sampling needs a 2GB+ contiguous block; expandable segments reuse fragments (see the b60 OOM notes).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------- Arguments --------------------------------------------------------
[[ $# -ge 2 ]] || { echo "Usage: $0 <qwen3-4b|qwen3-8b> <method> [block ...]" >&2; exit 2; }
MODEL_LOWER="$1"; shift
METHOD="$1"; shift
case "$MODEL_LOWER" in
    qwen3-4b) MODEL_DIR="Qwen3-4B"; TARGET_MODEL="${TARGET_MODEL:-$ROOT_DIR/cache/models/Qwen3-4B}" ;;
    qwen3-8b) MODEL_DIR="Qwen3-8B"; TARGET_MODEL="${TARGET_MODEL:-$ROOT_DIR/cache/models/Qwen3-8B}" ;;
    *) echo "Unknown model: $MODEL_LOWER (choices: qwen3-4b / qwen3-8b)" >&2; exit 2 ;;
esac

HARNESS="sglang"
DATASETS="${DATASETS:-specmix:3331}"
IFS=', ' read -r -a TEMPERATURES <<< "${TEMPERATURES:-1,0}"
REPEATS="${REPEATS:-3}"
DP_SIZE="${DP_SIZE:-8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
IFS=',' read -r -a CONCURRENCY_LIST <<< "${CONCURRENCIES:-64}"
MAX_CONCURRENCY=0
for _c in "${CONCURRENCY_LIST[@]}"; do
    (( _c > MAX_CONCURRENCY )) && MAX_CONCURRENCY=$_c
done
(( MAX_CONCURRENCY > 0 )) || { echo "CONCURRENCIES parsed as empty: ${CONCURRENCIES:-}" >&2; exit 2; }

MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-$MAX_CONCURRENCY}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
WARMUP_MAX_NEW_TOKENS="${WARMUP_MAX_NEW_TOKENS:-256}"

# ---------- Method table (kept in sync with run_eval_single.sh) --------------
declare -Ar METHOD_NAMES=(
    [dcore]="dcore" [dflash]="dflash" [dflare]="dflare" [dpace]="dpace"
    [domino]="domino" [dspark_markov]="dspark_markov"
    [eagle3_chain]="eagle3" [eagle3_tree]="eagle3_tree"
    [dspark_origin]="dspark_markov_origin"
    [dflash_origin]="dflash_origin" [dflare_origin]="dflare_origin"
    [domino_origin]="domino_origin"
    [eagle3_origin_chain]="eagle3_origin" [eagle3_origin_tree]="eagle3_origin_tree"
)
declare -Ar VARIANT_BLOCKS=(
    [dcore]="8 16" [dflash]="8 16" [dflare]="8 16" [dpace]="8 16"
    [domino]="8 16" [dspark_markov]="8 16"
    [eagle3_chain]="8 16" [eagle3_tree]="8 16 60"
    [dspark_origin]="8" [dflash_origin]="16" [dflare_origin]="16" [domino_origin]="16"
    [eagle3_origin_chain]="8 16" [eagle3_origin_tree]="8 16 60"
)
ckpt_dir_for() {   # <variant> <block>
    case "$1" in
        dspark_markov|dspark_origin)
            [[ "$2" == "8" ]] && echo "dspark-$( [[ "$1" == dspark_origin ]] && echo origin || echo markov )-block7" \
                              || echo "dspark-$( [[ "$1" == dspark_origin ]] && echo origin || echo markov )-block15" ;;
        dflash|dflare|dpace|domino|dcore) echo "$1-block$2" ;;
        eagle3_chain|eagle3_tree) echo "eagle3" ;;
        dflash_origin) echo "dflash-origin-b16" ;;
        dflare_origin) echo "dflare-origin-block16" ;;
        domino_origin) echo "domino-origin-block16" ;;
        eagle3_origin_chain|eagle3_origin_tree) echo "eagle3_origin" ;;
        *) echo "" ;;
    esac
}
find_ckpt() {   # <variant> <block>
    local budget dir
    case "$2" in 8) budget=budget8 ;; 16) budget=budget16 ;; 60) budget=budget8 ;; esac
    dir="$(ckpt_dir_for "$1" "$2")"
    local base="outputs/$MODEL_DIR/recipe-level/$budget/$dir"
    [[ -d "$base" ]] || { echo "checkpoint dir not found: $base" >&2; return 1; }
    local latest
    latest=$(ls -d "$base"/epoch_*_step_* 2>/dev/null | sort -V | tail -1 || true)
    [[ -n "$latest" ]] || latest="$base"
    echo "${latest#./}"
}
declare -Ar EAGLE3_TREE_CONFIGS=([8]="7 4 8" [16]="7 10 16" [60]="7 10 60")
declare -Ar EAGLE3_CHAIN_CONFIGS=([8]="7 1 8" [16]="15 1 16")
# b60 is memory-tight at high concurrency, so the KV pool drops to 0.60 (264k-token pool; measured peak only 67k).
declare -Ar BLOCK_MEM_FRACTION=([60]=0.60)
declare -Ar BLOCK_CUDA_GRAPH_MAX_BS_CAP=([60]=64)

# ---------- Helpers ----------------------------------------------------------
cleanup_stale_sglang() {
    local pids
    pids=$(pgrep -f "sglang serve" || true)
    if [[ -z "$pids" ]]; then
        echo "[cleanup] no stale sglang processes"
        return 0
    fi
    echo "[cleanup] found stale sglang processes: $(echo "$pids" | tr '\n' ' ')"
    for pid in $pids; do kill -TERM "$pid" 2>/dev/null || true; done
    for _ in $(seq 10); do
        pids=$(pgrep -f "sglang serve" || true)
        [[ -z "$pids" ]] && break
        sleep 1
    done
    pids=$(pgrep -f "sglang serve" || true)
    if [[ -n "$pids" ]]; then
        echo "[cleanup] still alive 10s after SIGTERM, sending SIGKILL"
        for pid in $pids; do kill -KILL "$pid" 2>/dev/null || true; done
        sleep 2
    fi
    echo "[cleanup] done"
}

# ---------- Main -------------------------------------------------------------
if [[ $# -gt 0 ]]; then
    BLOCKS=("$@")
else
    read -r -a BLOCKS <<< "${VARIANT_BLOCKS[$METHOD]:-}"
fi
[[ -n "${METHOD_NAMES[$METHOD]:-}" ]] || {
    echo "Unknown method: $METHOD (choices: ${!METHOD_NAMES[*]})" >&2
    exit 2
}
[[ ${#BLOCKS[@]} -gt 0 ]] || { echo "Method $METHOD has no available block" >&2; exit 2; }
[[ -d "$ROOT_DIR/cache/dataset/specmix" ]] || {
    # One-time fetch from Hugging Face (the rest of the run stays offline).
    echo "specmix not cached; downloading wingchiuloong/specmix from Hugging Face ..."
    HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 "$PYTHON" -c \
        "from specbench.datasets import load_and_process_dataset as f; print(len(f('specmix')), 'rows cached')" || {
        echo "Failed to download specmix; check network/proxy, or build it locally: $PYTHON scripts/build_specmix_dataset.py" >&2
        exit 2
    }
}

SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d-%H%M%S)__${MODEL_LOWER}__specmix-${METHOD}-c${MAX_CONCURRENCY}}"
SWEEP_ROOT="$ROOT_DIR/outputs/eval/${MODEL_LOWER}/$SWEEP_ID"
mkdir -p "$SWEEP_ROOT"

echo "==========================================================="
echo " Multi-concurrency eval: model=$MODEL_LOWER method=$METHOD"
echo " Blocks: ${BLOCKS[*]}   Temperatures: ${TEMPERATURES[*]}   Concurrency: ${CONCURRENCY_LIST[*]}"
echo " Datasets: $DATASETS (mixed traffic; see https://huggingface.co/datasets/wingchiuloong/specmix)"
echo " Output: $SWEEP_ROOT"
echo "==========================================================="
cleanup_stale_sglang

RUN_INDEX=0
for block in "${BLOCKS[@]}"; do
    [[ " ${VARIANT_BLOCKS[$METHOD]} " == *" $block "* ]] || {
        echo "--- skip: $METHOD does not support block=$block ---"
        continue
    }
    ckpt="${CHECKPOINT:-$(find_ckpt "$METHOD" "$block")}"
    echo "[$METHOD b$block] ckpt = $ckpt"
    [[ -d "$ROOT_DIR/$ckpt" ]] || { echo "!!! checkpoint not found: $ROOT_DIR/$ckpt" >&2; continue; }

    # Resolve per-block memory params (overwritten every iteration; baseline is unaffected).
    MEM_FRACTION="${BLOCK_MEM_FRACTION[$block]:-$MEM_FRACTION_STATIC}"
    GRAPH_BS="$CUDA_GRAPH_MAX_BS"
    cap="${BLOCK_CUDA_GRAPH_MAX_BS_CAP[$block]:-}"
    [[ -n "$cap" ]] && (( GRAPH_BS > cap )) && GRAPH_BS="$cap"

    for concurrency in "${CONCURRENCY_LIST[@]}"; do
        for temperature in "${TEMPERATURES[@]}"; do
            if [[ "$temperature" == "1" ]]; then
                temperature_tag="t1rs"
                RUN_REPEATS="$REPEATS"
                sampling_args=(--temperature 1.0 --top-k -1 --speculative-use-rejection-sampling)
            else
                temperature_tag="t0"
                RUN_REPEATS=1
                sampling_args=(--temperature 0.0 --top-k 1)
            fi

            run_dir="$SWEEP_ROOT/b${block}/${temperature_tag}/dt00/c${concurrency}"
            mkdir -p "$run_dir/responses" "$run_dir/logs"

            # Baseline depends only on (concurrency, temperature); ideally run once at the lowest
            # concurrency and merged into each run_dir (currently run per concurrency for simplicity;
            # with multiple concurrencies, set BASELINE_DIR to reuse one).
            local_base="$SWEEP_ROOT/baseline/${temperature_tag}/c${concurrency}"
            if [[ "${SKIP_BASELINE:-1}" != "1" && ! -s "$local_base/results.jsonl" ]]; then
                mkdir -p "$local_base/responses" "$local_base/logs"
                echo "--- run: [baseline] $temperature_tag c$concurrency ---"
                "$PYTHON" -m specbench.cli run --run-dir "$local_base" --method baseline \
                    --harness "$HARNESS" --target-model "$TARGET_MODEL" \
                    --datasets "$DATASETS" --concurrencies "$concurrency" \
                    --repeats "$RUN_REPEATS" --dp-size "$DP_SIZE" \
                    --attention-backend "$ATTENTION_BACKEND" \
                    --max-running-requests "$MAX_RUNNING_REQUESTS" \
                    --cuda-graph-max-bs "$GRAPH_BS" \
                    --mem-fraction-static "$MEM_FRACTION" \
                    --max-new-tokens "$MAX_NEW_TOKENS" \
                    --warmup-requests "$WARMUP_REQUESTS" \
                    --warmup-max-new-tokens "$WARMUP_MAX_NEW_TOKENS" \
                    "${sampling_args[@]}"
                "$PYTHON" -m specbench.cli report --run-dir "$local_base" || true
            fi

            RUN_INDEX=$((RUN_INDEX + 1))
            echo "--- run: [$RUN_INDEX] $METHOD / b$block / $temperature_tag / c$concurrency (repeats=$RUN_REPEATS) ---"
            extra=()
            if [[ "$METHOD" == eagle3*_tree ]]; then
                extra=(--spec-num-steps "${EAGLE3_TREE_CONFIGS[$block]%% *}"
                       --spec-eagle-topk "$(awk '{print $2}' <<< "${EAGLE3_TREE_CONFIGS[$block]}")"
                       --spec-num-draft-tokens "$(awk '{print $3}' <<< "${EAGLE3_TREE_CONFIGS[$block]}")")
            elif [[ "$METHOD" == eagle3*_chain* ]]; then
                extra=(--spec-num-steps "${EAGLE3_CHAIN_CONFIGS[$block]%% *}"
                       --spec-eagle-topk "$(awk '{print $2}' <<< "${EAGLE3_CHAIN_CONFIGS[$block]}")"
                       --spec-num-draft-tokens "$(awk '{print $3}' <<< "${EAGLE3_CHAIN_CONFIGS[$block]}")")
            fi

            "$PYTHON" -m specbench.cli run --run-dir "$run_dir" \
                --method "${METHOD_NAMES[$METHOD]}" --harness "$HARNESS" \
                --target-model "$TARGET_MODEL" --draft-model "$ckpt" \
                --datasets "$DATASETS" --concurrencies "$concurrency" \
                --repeats "$RUN_REPEATS" --dp-size "$DP_SIZE" \
                --attention-backend "$ATTENTION_BACKEND" \
                --max-running-requests "$MAX_RUNNING_REQUESTS" \
                --cuda-graph-max-bs "$GRAPH_BS" \
                --mem-fraction-static "$MEM_FRACTION" \
                --max-new-tokens "$MAX_NEW_TOKENS" \
                --warmup-requests "$WARMUP_REQUESTS" \
                --warmup-max-new-tokens "$WARMUP_MAX_NEW_TOKENS" \
                "${sampling_args[@]}" "${extra[@]}"

            if [[ -s "$run_dir/results.jsonl" ]]; then
                report_args=(--run-dir "$run_dir")
                if [[ "${SKIP_BASELINE:-1}" != "1" && -s "$local_base/results.jsonl" ]]; then
                    report_args+=(--baseline-dir "$local_base")
                fi
                "$PYTHON" -m specbench.cli report "${report_args[@]}" || true
                "$PYTHON" -m specbench.cli accuracy --run-dir "$run_dir" || true
                echo "report: $run_dir/report.md"
            fi
        done
    done
done

echo "==========================================================="
echo " Eval finished: $SWEEP_ROOT"
find "$SWEEP_ROOT" -name report.md -print | sort
echo "==========================================================="