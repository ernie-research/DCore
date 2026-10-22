# Speculative Decoding Race Demo

Runs DFlash, DFlare, DPace, Domino, DSpark, EAGLE-3, DCore, and a target-only baseline on the same prompt with SGLang, shows each method's output token by token (accepted draft tokens are highlighted), and then replays all runs side by side from t=0 as a race.

- Targets: Qwen3-8B / Qwen3-4B (thinking disabled)
- Budgets: 8 / 16
- Temperature presets (target:draft): `0:0` (greedy), `1:0`, `1:1`

## Requirements

- One free GPU per method for parallel mode (default 8), or a single GPU for serial mode.
- The project environment from `build_env.sh` (or set `PYTHON=/path/to/python`); the demo also needs `fastapi`, `uvicorn`, and `aiohttp`.
- Draft checkpoints are downloaded from Hugging Face automatically; target models fall back to the official Qwen repos. Paths and repo ids live in `registry.py`.

## Run

From the repository root:

```bash
bash demo/spec_race/run_demo.sh                    # 8 GPUs in parallel, serves 127.0.0.1:8099
GPUS=0 bash demo/spec_race/run_demo.sh             # single GPU, methods run one after another
GPUS=auto bash demo/spec_race/run_demo.sh          # parallel if enough GPUs are free, else serial
```

Open `http://127.0.0.1:8099`, pick a target, budget, temperature preset, and prompt, then click **Run live**. When all methods finish, the run is saved to `demo/spec_race/recordings/`; click **Replay race** (or press `R`) to replay it, or load a saved recording later without a GPU.

For a remote server, forward the port with `ssh -L 8099:127.0.0.1:8099 <user>@<host>`. The web UI has no authentication, so only use `HOST=0.0.0.0` on a trusted network.

## Offline recording

Record races in batch while GPUs are free, then replay them anywhere:

```bash
bash demo/spec_race/run_demo.sh record                                   # all presets, methods, and prompts
bash demo/spec_race/run_demo.sh record --model Qwen3-4B --budget 8 --presets 0:0 \
    --methods dflash dcore --prompt "Explain speculative decoding in three sentences."
```

Run `python -m demo.spec_race.app --help` or `python -m demo.spec_race.record --help` for all options.

## Troubleshooting

- **OOM at startup**: a leftover SGLang process is probably holding GPU memory; find it with `pgrep -af sglang`, or click **Free GPU** in the UI.
- **A method never becomes ready**: check `demo/spec_race/logs/<server>.log`.
- **Different outputs under `1:0` / `1:1`**: expected, since sampling paths differ per method; use `0:0` to compare speed on identical outputs.
