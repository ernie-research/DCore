#!/usr/bin/env bash
# Speculative decoding race demo (DFlash / DFlare / DPace / Domino / DSpark /
# EAGLE-3 / DCore) with SGLang. One server per GPU when enough devices are
# free, serial on a single device otherwise.
#
#   bash demo/spec_race/run_demo.sh                       # web UI on 127.0.0.1:8099, GPU 0
#   GPU=3 PORT=8080 HOST=0.0.0.0 bash demo/spec_race/run_demo.sh
#   GPUS=0-7 bash demo/spec_race/run_demo.sh              # explicit device pool (parallel)
#   bash demo/spec_race/run_demo.sh record --model Qwen3-8B --budget 16   # offline batch recording
#
# The web UI has no authentication; only use HOST=0.0.0.0 on a trusted network.
# Extra arguments are forwarded to `python -m demo.spec_race.app` (or `.record`).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-${UV_PROJECT_ENVIRONMENT:-$ROOT_DIR/.venv}/bin/python}"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT_DIR/third_party/sglang/python:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nccl:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"
export SGLANG_FLASHINFER_VERIFY_ATTN_MODE="${SGLANG_FLASHINFER_VERIFY_ATTN_MODE:-prepacked}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
# Local HTTP between the demo and the sglang servers must bypass the proxy.
export no_proxy="localhost,127.0.0.1,0.0.0.0${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

GPU="${GPU:-0}"
# Default: the full 8-GPU pool, one device per method (parallel race).
#   GPUS=auto -> check free GPUs, fall back to serial when there are fewer than methods
#   GPUS=0    -> single-GPU serial mode;  GPUS=0-3 / 0,1,2 / 2 -> explicit device list
# A legacy `GPU=N` (without GPUS) keeps serial mode on device N.
if [[ -z "${GPUS:-}" ]]; then
    if [[ "$GPU" == "0" ]]; then
        GPUS="0-7"
    else
        GPUS="$GPU"
    fi
fi
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8099}"
# Set KILL_PREV=0 to skip stopping a previous demo instance on startup.
KILL_PREV="${KILL_PREV:-1}"

# A previous instance may still hold $PORT (e.g. a ^C that left uvicorn waiting
# for in-flight requests) or its sglang children may still pin the GPUs. Stop
# both so the next launch starts clean. Only processes belonging to this demo
# are touched: the app's cmdline contains "demo.spec_race", and sglang servers
# launched here carry the demo's SGLANG_* markers in their environment.
stop_prev_instance() {
    [[ "$KILL_PREV" == "1" ]] || return 0
    local port="${1:-}" pid

    # 1) stop the app bound to $PORT (only if it is one of ours)
    if [[ -n "$port" && "$port" != "0" ]]; then
        for pid in $(ss -tlnp 2>/dev/null | grep -F ":$port " | grep -oP 'pid=\K[0-9]+' | sort -u); do
            if [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -q "demo.spec_race"; then
                echo "[spec_race] stopping previous demo instance (pid $pid, port $port)" >&2
                kill "$pid" 2>/dev/null || true
                for _ in $(seq 1 30); do
                    kill -0 "$pid" 2>/dev/null || break
                    sleep 0.5
                done
                if kill -0 "$pid" 2>/dev/null; then
                    echo "[spec_race] ... pid $pid still alive, sending SIGKILL" >&2
                    kill -9 "$pid" 2>/dev/null || true
                fi
            else
                echo "[spec_race] warning: port $port is held by a process that is not this demo; leaving it alone" >&2
            fi
        done
    fi

    # 2) stop sglang servers launched by a previous instance (they hold GPU
    #    memory even after the app is gone). kill -pid = whole process group.
    for d in /proc/[0-9]*; do
        pid="${d##*/}"
        [[ -r "$d/environ" && -r "$d/cmdline" ]] || continue
        if ! tr '\0' '\n' <"$d/environ" 2>/dev/null | grep -q "^SGLANG_RAGGED_VERIFY_MODE=static$"; then
            continue
        fi
        if ! tr '\0' ' ' <"$d/cmdline" 2>/dev/null | grep -q "sglang"; then
            continue
        fi
        echo "[spec_race] stopping leftover sglang server (pid $pid) from a previous run" >&2
        kill -TERM -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    done
    sleep 1
}

# Leftover `sglang serve` processes hold GPU memory and make the next launch OOM.
if pgrep -f "sglang serve" >/dev/null 2>&1 || pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
    echo "[spec_race] warning: existing sglang processes found (pgrep -af sglang); they may hold GPU memory." >&2
fi

# Stop a stale instance from a previous run before starting, then start fresh.
stop_prev_instance "$PORT"

if [[ "${1:-}" == "record" ]]; then
    shift
    exec "$PYTHON" -m demo.spec_race.record --gpu "$GPU" --gpus "$GPUS" "$@"
fi
exec "$PYTHON" -m demo.spec_race.app --gpu "$GPU" --gpus "$GPUS" --host "$HOST" --port "$PORT" "$@"
