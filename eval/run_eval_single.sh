#!/usr/bin/env bash
# Single-concurrency eval: evaluate one method on 10 benchmarks (concurrency 1 per replica).
#
# This is the simple public entry point, replacing the old run_specbench_*_sweep_*.sh
# (full multi-method sweeps). It evaluates one method per invocation and writes
# results to outputs/eval/<model>/.
#
# Usage:
#   bash eval/run_eval_single.sh <qwen3-4b|qwen3-8b> <method> [block ...]
#
# Arguments:
#   model    qwen3-4b | qwen3-8b (checkpoints auto-discovered under outputs/Qwen3-{4B,8B}/recipe-level/)
#   method   dcore | dflash | dflare | dpace | domino | dspark_markov
#            | eagle3_chain | eagle3_tree (EAGLE3 chain / tree decoding)
#            | dspark_origin | dflash_origin | dflare_origin | domino_origin
#            | eagle3_origin_chain | eagle3_origin_tree (third-party released weights, for comparison)
#   block    verify window(s), may pass several (default: all supported by the method, e.g. dcore -> 8 16)
#
# Environment variables (all optional):
#   TEMPERATURES  target temperatures, comma-separated; 0=greedy, 1=rejection sampling. Default "0" (fast, reproducible).
#   DATASETS      eval sets; default is the full set of these 10:
#                 gsm8k:1319,math500:500,aime25:30,humaneval:164,mbpp:257,
#                 livecodebench:700,mt-bench:80,alpaca:500,arena-hard:500,swe-bench:300
#   CHECKPOINT    explicit ckpt dir (overrides auto-discovery; relative to repo root)
#   SWEEP_ID      output dir name (default <timestamp>__<model>__single-<method>)
#   SKIP_BASELINE 1=skip baseline (default; report shows absolute tok/s);
#                 0=first run a drafter-free baseline as the speedup denominator
#   REPEATS       consecutive rounds within one serving session at t1 (default 3; t0 is always 1)
#   DP_SIZE       number of data-parallel replicas (default 8)
#   ENABLE_PROFILE 1=enable sub-stage profiling and emit a latency table (default 0)
#
# Examples:
#   bash eval/run_eval_single.sh qwen3-8b dcore                # all blocks: b8+b16
#   bash eval/run_eval_single.sh qwen3-8b dcore 16             # b16 only
#   TEMPERATURES=1 SKIP_BASELINE=0 bash eval/run_eval_single.sh qwen3-4b dcore
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
# Datasets are cached under cache/dataset/, so evaluation can run fully offline.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

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
DATASETS="${DATASETS:-gsm8k:1319,math500:500,aime25:30,humaneval:164,mbpp:257,livecodebench:700,mt-bench:80,alpaca:500,arena-hard:500,swe-bench:300}"
IFS=', ' read -r -a TEMPERATURES <<< "${TEMPERATURES:-0}"
REPEATS="${REPEATS:-3}"
DP_SIZE="${DP_SIZE:-8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
SKIP_BASELINE="${SKIP_BASELINE:-1}"

if [[ "${ENABLE_PROFILE:-0}" == "1" ]]; then
    export SGLANG_SPEC_DEBUG_DUMP="all"
fi
unset SGLANG_SPEC_DUMP_PQ_PATH

# ---------- Method table -----------------------------------------------------
# Method variant -> registered method name (specbench/methods.py). Chain variants reuse
# the eagle3 entry (topk=1 is the chain config); tree variants use eagle3_tree, so chain
# and tree runs never overwrite each other in the same run_dir.
declare -Ar METHOD_NAMES=(
    [dcore]="dcore" [dflash]="dflash" [dflare]="dflare" [dpace]="dpace"
    [domino]="domino" [dspark_markov]="dspark_markov"
    [eagle3_chain]="eagle3" [eagle3_tree]="eagle3_tree"
    [dspark_origin]="dspark_markov_origin"
    [dflash_origin]="dflash_origin" [dflare_origin]="dflare_origin"
    [domino_origin]="domino_origin"
    [eagle3_origin_chain]="eagle3_origin" [eagle3_origin_tree]="eagle3_origin_tree"
)

# Variant -> supported verify windows (b60 is EAGLE3-tree only; origin weights ship only their own block).
declare -Ar VARIANT_BLOCKS=(
    [dcore]="8 16" [dflash]="8 16" [dflare]="8 16" [dpace]="8 16"
    [domino]="8 16" [dspark_markov]="8 16"
    [eagle3_chain]="8 16" [eagle3_tree]="8 16 60"
    [dspark_origin]="8" [dflash_origin]="16" [dflare_origin]="16" [domino_origin]="16"
    [eagle3_origin_chain]="8 16" [eagle3_origin_tree]="8 16 60"
)

# Variant:block -> directory name under recipe-level (dspark config block_size 7/15 maps to
# verify window 8/16; eagle3 is window-independent, chain/tree share the same weights).
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

# Auto-discover the latest ckpt: pick the highest epoch_*_step_* under
# outputs/<MODEL_DIR>/recipe-level/budget{8,16}/<dir>/; if there is none (origin weights), use the dir itself.
find_ckpt() {   # <variant> <block>
    local budget dir
    case "$2" in 8) budget=budget8 ;; 16) budget=budget16 ;; 60) budget=budget8 ;; esac
    dir="$(ckpt_dir_for "$1" "$2")"
    local base="outputs/$MODEL_DIR/recipe-level/$budget/$dir"
    [[ -d "$base" ]] || { echo "checkpoint dir not found: $base" >&2; return 1; }
    local latest
    latest=$(ls -d "$base"/epoch_*_step_* 2>/dev/null | sort -V | tail -1 || true)
    [[ -n "$latest" ]] || latest="$base"   # origin weights are the model dir itself
    echo "${latest#./}"
}

# Block size -> "num_steps eagle_topk num_draft_tokens".
# Constraint: num_draft_tokens - 1 <= num_steps * eagle_topk.
declare -Ar EAGLE3_TREE_CONFIGS=([8]="7 4 8" [16]="7 10 16" [60]="7 10 60")
declare -Ar EAGLE3_CHAIN_CONFIGS=([8]="7 1 8" [16]="15 1 16")

# b60 memory overrides: verify/extend graphs run at bs x 60 tokens, so defaults OOM on long
# prompts (at c=1 lowering these does not affect throughput or comparability with b8/b16).
declare -Ar BLOCK_MEM_FRACTION=([60]=0.65)
declare -Ar BLOCK_CUDA_GRAPH_MAX_BS=([60]=4)

# ---------- Helpers ----------------------------------------------------------
cleanup_stale_sglang() {
    # Stale sglang serve processes hold GPU memory and cause later CUDA OOM. SIGTERM first, KILL after 10s.
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
# Resolve the list of blocks to evaluate
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

SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d-%H%M%S)__${MODEL_LOWER}__single-${METHOD}}"
SWEEP_ROOT="$ROOT_DIR/outputs/eval/${MODEL_LOWER}/$SWEEP_ID"
mkdir -p "$SWEEP_ROOT"

echo "==========================================================="
echo " Single-concurrency eval: model=$MODEL_LOWER method=$METHOD"
echo " Blocks: ${BLOCKS[*]}   Temperatures: ${TEMPERATURES[*]}"
echo " Datasets: $DATASETS"
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

        run_dir="$SWEEP_ROOT/b${block}/${temperature_tag}/dt00/c1"
        mkdir -p "$run_dir/responses" "$run_dir/logs"

        # Baseline depends only on temperature, so run it once per temperature; skipped when SKIP_BASELINE=1.
        local_base="$SWEEP_ROOT/baseline/${temperature_tag}/c1"
        if [[ "$SKIP_BASELINE" != "1" && ! -s "$local_base/results.jsonl" ]]; then
            mkdir -p "$local_base/responses" "$local_base/logs"
            echo "--- run: [baseline] $temperature_tag ---"
            "$PYTHON" -m specbench.cli run --run-dir "$local_base" --method baseline \
                --harness "$HARNESS" --target-model "$TARGET_MODEL" \
                --datasets "$DATASETS" --concurrencies 1 --repeats "$RUN_REPEATS" \
                --dp-size "$DP_SIZE" --attention-backend "$ATTENTION_BACKEND" \
                "${sampling_args[@]}"
            "$PYTHON" -m specbench.cli report --run-dir "$local_base" || true
        fi

        RUN_INDEX=$((RUN_INDEX + 1))
        echo "--- run: [$RUN_INDEX] $METHOD / b$block / $temperature_tag (repeats=$RUN_REPEATS) ---"
        extra=()
        if [[ "$METHOD" == eagle3*_tree ]]; then
            read -r _depth _topk _tokens <<< "${EAGLE3_TREE_CONFIGS[$block]}"
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
            --datasets "$DATASETS" --concurrencies 1 --repeats "$RUN_REPEATS" \
            --dp-size "$DP_SIZE" --attention-backend "$ATTENTION_BACKEND" \
            --mem-fraction-static "${BLOCK_MEM_FRACTION[$block]:-0.8}" \
            --cuda-graph-max-bs "${BLOCK_CUDA_GRAPH_MAX_BS[$block]:-4}" \
            "${sampling_args[@]}" "${extra[@]}"

        if [[ -s "$run_dir/results.jsonl" ]]; then
            report_args=(--run-dir "$run_dir")
            if [[ "$SKIP_BASELINE" != "1" && -s "$local_base/results.jsonl" ]]; then
                report_args+=(--baseline-dir "$local_base")
            fi
            "$PYTHON" -m specbench.cli report "${report_args[@]}" || true
            "$PYTHON" -m specbench.cli accuracy --run-dir "$run_dir" || true
            echo "report: $run_dir/report.md"
        fi
    done
done

echo "==========================================================="
echo " Eval finished: $SWEEP_ROOT"
find "$SWEEP_ROOT" -name report.md -print | sort
echo "==========================================================="