# When Does Better Acceptance Pay? A Controlled Study of Speculative Decoding

[![arXiv](https://img.shields.io/badge/arXiv-2610.24993-b31b1b.svg)](https://arxiv.org/abs/2610.24993)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-DCore-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/collections/ernie-research/dcore)

https://github.com/user-attachments/assets/0ad6f343-343d-44a2-9c13-20b9651deb3c

This repository accompanies **When Does Better Acceptance Pay? A Controlled Study of Speculative Decoding**. It contains the training, evaluation, and interactive demo code for seven speculative-decoding drafters (EAGLE-3, DFlash, DFlare, D-PACE, Domino, DSpark, and our DCore) on Qwen3-4B and Qwen3-8B:

- **Training**: one-line launch via `./train.sh`, mapped to the tuned configs under `examples/colocated/`.
- **Evaluation**: `specbench/` + `eval/`, online speculative-decoding evaluation on SGLang.
- **Demo**: `demo/spec_race/`, a token-by-token race replay across methods that pulls the released checkpoints automatically.
- **Dependency**: `third_party/sglang`, the only submodule, carrying the speculative-decoding changes DCore needs.

Jump to [Getting Started](#getting-started) to run the code, or read on for the paper's findings.

## More accepted tokens. Faster generation?

Speculative decoding lets a lightweight **drafter** propose tokens for a **target model** to verify in parallel. Accepting more tokens per step sounds like a win. But proposing and verifying them takes time.

**The goal is throughput, not acceptance length.** Our question is: *when is additional acceptance worth the cost of obtaining it?*

We compare six recipes across three families—autoregressive, block-parallel, and semi-autoregressive—using a common training, inference, and evaluation protocol, ten benchmarks, and two target models.

**Why these six drafters?** EAGLE-3, DFlash, DFlare, D-PACE, Domino, and DSpark represent learned drafting families actively explored by mainstream LLMs, making the comparison relevant to practical inference design. The story connects three decisions: **how to compare methods, how to improve them, and when their improvements pay off**.

### 1. Before trusting the ranking, align the conditions

Published rankings mix differences in training data and evaluation protocols. Re-running released checkpoints aligns evaluation, but leaves training conditions different.

**Figure 1** separates results **as reported**, released checkpoints **re-measured** in our runtime, and recipes **re-trained** on shared target-generated data under a common evaluation protocol.

[![Figure 1: Acceptance and speedup across reported, re-measured, and re-trained comparisons.](assets/figure-1.png)](assets/figure-1.png)

*Pale bars show acceptance length; solid bars show speedup. Read from left to right to see the effect of alignment.*

On MT-Bench, the highest-to-lowest speedup gap falls from **1.36 to 0.85 to 0.23**—differences in speedup multiples, not percentages. The apparent separation becomes much smaller.

**Table 8** expands this comparison to seven benchmarks and both targets. Acceptance changes differently across recipes; retraining is not simply a boost for everyone.

[![Table 8: Acceptance length as reported, re-measured, and re-trained across seven benchmarks and both target models.](assets/table-8.png)](assets/table-8.png)

*Each benchmark has three columns: reported, re-measured, and re-trained. Blue/yellow shading marks an increase/decrease from the preceding stage, not a cross-method ranking. Compare methods at matched budgets and temperatures. Click to enlarge.*

For example, on MT-Bench with Qwen3-8B, budget 16, and target|draft temperatures of 1|0, EAGLE-3's acceptance changes **2.70 → 2.53 → 3.59**, while DFlare's changes **4.37 → 3.37 → 3.13**. Their acceptance ordering reverses after retraining.

**A checkpoint comparison alone is not a clean comparison of drafter recipes.** Alignment gives us a clearer starting point for investigating the differences that remain.

### 2. Follow the costs, and a strong baseline emerges

**Figure 2** examines component and training-objective changes together with their execution costs.

[![Figure 2: Design ablation roadmap showing speedup, acceptance length, and proposal latency.](assets/figure-2.png)](assets/figure-2.png)

*Colored bars show speedup, pale bars acceptance length, and stars proposal latency. The gold star marks DCore.*

There are two routes to better acceptance: improve supervision or target context with little additional measured inference overhead, or add computation through deeper networks and stronger token-to-token conditioning. The second route pays only when acceptance gains cover the added cost. Useful components also need testing in combination.

This motivates **DCore**: DFlash's efficient backbone, DFlare's target-layer context mixture, and the D-PACE objective. In the roadmap configuration, speedup rises from **3.004× to 3.261×** at unchanged **1.79 ms** measured proposal latency.

Does that recipe hold up beyond the design search? **Table 1** compares DCore with all six re-trained drafters. It achieves the highest mean throughput in **18 of 24 single-request settings**, defined by target, budget, temperature pair, and task-entropy group.

[![Table 1: Mean throughput and acceptance length for the re-trained drafters and DCore.](assets/table-1.png)](assets/table-1.png)

*Cells show tokens per second, with acceptance length in the subscript. Headers give target|draft temperatures. Throughput and acceptance are ranked separately; bold marks the best result. Click to enlarge.*

DCore is a strong baseline distilled from the analysis. Other recipes lead in some settings, motivating a deeper question: **what makes a setting favorable?**

### 3. Look at uncertainty, not just the task label

**Entropy** measures next-token uncertainty. Even two coding tasks can differ greatly, so domain labels do not fully describe drafting difficulty.

**The full entropy heatmaps** look inside decoding paths across three drafter families, both targets, two budgets, and the temperature settings shown. Accepted prefixes tend to have lower target entropy than the rejected position ending the path.

[![Full target-entropy heatmaps across EAGLE-3 chain decoding, DFlash, and DSpark, with both targets and budgets.](assets/figure-12.png)](assets/figure-12.png)

*Brighter cells indicate greater target uncertainty; black borders mark rejected endpoints. Each panel contains a budget-16 grid and a budget-8 inset. Top rows show Qwen3-4B at 0|0 and 1|1; bottom rows show Qwen3-8B at 0|0 and 1|0. Gray bars show step frequencies. Click to enlarge.*

This recurring pattern makes uncertainty a useful lens for understanding acceptance. But acceptance is only one side of the payoff.

### 4. A larger budget can help—or slow you down

**Figure 4** compares verification budgets of 8 and 16 on Qwen3-8B with sampled target output and greedy drafts.

[![Figure 4: Acceptance gains versus throughput gains when the verification budget increases from 8 to 16.](assets/figure-4.png)](assets/figure-4.png)

*Right means more acceptance; up means more throughput. Solid/hollow markers represent low/high-entropy tasks. Log changes separate throughput gains into acceptance gains and efficiency changes; diagonal guides show the latter.*

Block-parallel recipes best preserve efficiency as the budget grows. Both block-parallel and semi-autoregressive recipes gain up to about **20%** throughput on low-entropy tasks, but at most **2.8%** on high-entropy tasks. EAGLE-3's wider tree gains about **12%** in both groups. Extending its sequential chain instead slows both groups: acceptance cannot repay the extra drafting time.

The block-based comparisons use budget-specific recipes; EAGLE-3's tree and chain comparisons use the same checkpoint. These single-request results lead to the deployment question: **what happens when requests arrive together?**

### 5. The server load changes the answer again

**Figure 9** evaluates a shuffled mixture of requests from all ten benchmarks, called **SpecMix**, across increasing concurrency.

[![Figure 9: Speculative-decoding speedup across concurrency, both target models, two budgets, and three temperature pairs.](assets/figure-9.png)](assets/figure-9.png)

*Rows vary target and budget; columns vary target|draft temperatures. The horizontal axis is concurrent requests per replica. Speedup is relative to the target-only baseline at the same load; the dashed line is 1×. EAGLE-3 uses chain decoding here. Insets clarify nearby curves. Click to enlarge.*

A larger budget can help at low load yet lose to a smaller budget as concurrency grows. DCore remains competitive under greedy drafting, while semi-autoregressive methods become more competitive under sampled drafting. The preferred recipe and budget therefore depend on how the model is served, as well as what it generates.

**The practical lesson:** first establish a common comparison, then seek acceptance improvements with little inference overhead, and finally select the drafter and budget together using end-to-end throughput under the intended workload and serving conditions.

## Getting Started

### 1. Clone and build the environment

```bash
git clone --recurse-submodules https://github.com/ernie-research/DCore.git
cd DCore

# One-shot build: uv virtualenv + rust/protoc + prefetch and compile sglang-kernel
bash build_env.sh

# Activate the environment (the script also appends this to ~/.bashrc)
source ~/.venvs/dcore/bin/activate
```

Prerequisites: Python 3.12, CUDA 12.x or 13.x, uv, and at least 8 GPUs (training is tuned for 8 GPUs).

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

To rebuild sglang-kernel:

```bash
REBUILD_SGLANG_KERNEL=1 bash build_env.sh
```

### 2. Download the target models

```bash
# Download Qwen/Qwen3-8B and Qwen/Qwen3-4B into cache/models/ (skipped if present; --force to re-download)
bash scripts/download_target_models.sh            # both models
bash scripts/download_target_models.sh qwen3-8b   # 8B only
```

You can also download (or symlink) them manually to `cache/models/Qwen3-8B` and `cache/models/Qwen3-4B`.

### 3. Prepare the training data

Training uses **OpenPerfectBlend** (`mlabonne/open-perfectblend`, ShareGPT format), with every assistant turn **regenerated by the target model** (thinking disabled) so the data matches the target's output distribution. The regenerated sets used in the paper are on Hugging Face:

| Target | Dataset |
| --- | --- |
| Qwen3-8B | [`wingchiuloong/OpenPerfectBlend-Qwen3-8B-Regen`](https://huggingface.co/datasets/wingchiuloong/OpenPerfectBlend-Qwen3-8B-Regen) |
| Qwen3-4B | [`wingchiuloong/OpenPerfectBlend-Qwen3-4B-Regen`](https://huggingface.co/datasets/wingchiuloong/OpenPerfectBlend-Qwen3-4B-Regen) |

`train.sh` downloads the matching set automatically on first use (about 6 GB each) and converts it to `cache/dataset/perfectblend/perfectblend_train_regen_<qwen3_8b|qwen3_4b>.jsonl`, the default `TRAIN_DATA_PATH` of every launch script. To fetch it ahead of time:

```bash
python scripts/download_training_data.py qwen3-8b
python scripts/download_training_data.py qwen3-4b
```

#### Regenerating the data yourself (optional)

One command runs download → convert → regenerate → validate and writes to the same path:

```bash
# Needs the target weights (previous step) and at least 1 GPU; an SGLang server is launched automatically
bash scripts/prepare_training_data.sh qwen3-8b
bash scripts/prepare_training_data.sh qwen3-4b --reasoning disable

# Reuse an SGLang server that is already serving the target model:
bash scripts/prepare_training_data.sh qwen3-8b --server-address 10.0.0.1:30000

# Small smoke test (100 samples each, finishes in tens of minutes):
bash scripts/prepare_training_data.sh qwen3-8b --sample-size 100 --num-samples 100
```

Pipeline and outputs:

| Step | Script | Output |
| --- | --- | --- |
| 1. Download and convert OpenPerfectBlend | `scripts/prepare_data.py --dataset perfectblend` (idempotent) | `cache/dataset/perfectblend/perfectblend_train.jsonl` |
| 2. Launch the target SGLang server | `sglang.launch_server` (logs in `outputs/data_regen/`) | `http://127.0.0.1:31000` (adjust with `--port/--tp`) |
| 3. Regenerate turn by turn | `scripts/regenerate_train_data.py` (`--resume` supported) | `cache/dataset/perfectblend/perfectblend_train_regen_<model>.jsonl` |
| 4. Validate | `scripts/validate_regenerated_data.py` | ready for training once it passes |

Common options: `--reasoning {none,save,disable}` (default `disable`, i.e. `enable_thinking=False`; `save` keeps `reasoning_content`), `--temperature` (default 0.0, greedy, matching the released data), `--concurrency` (default 64), `--sample-size` / `--num-samples` (smoke tests), and `--no-resume` (overwrite from scratch).

### 4. Train

```bash
./train.sh qwen3-8b dcore --budget 16 --gpus 8   # budget defaults to 16; GPU count is auto-detected
./train.sh qwen3-4b dspark --budget 8            # only model + method are required
./train.sh --list                                # list all 28 combinations
```

`train.sh` maps `<model> <method> <budget>` to the matching config under `examples/colocated/`; every hyperparameter is tuned for reproducing the paper.

| Method | Draft architecture | Budget 8 config | Budget 16 config |
| --- | --- | --- | --- |
| dflash | DFlashDraftModel (5 layers, 5 hidden states) | `qwen3-{4b,8b}-dflash-bs8.json` | `qwen3-{4b,8b}-dflash.json` |
| dflare | DFlareDraftModel (9 hidden states, layer fusion) | `qwen3-{4b,8b}-dflare_9hs-bs8.json` | `qwen3-{4b,8b}-dflare_9hs.json` |
| dpace | DFlashDraftModel × D-PACE loss | `qwen3-{4b,8b}-dflash-bs8.json` | `qwen3-{4b,8b}-dflash.json` |
| domino | DominoDraftModel (prefix GRU) | `qwen3-{4b,8b}-domino-bs8.json` | `qwen3-{4b,8b}-domino.json` |
| dspark | DSparkDraftModel (Markov head) | `qwen3-{4b,8b}-dspark-bs7.json` | `qwen3-{4b,8b}-dspark-bs15.json` |
| eagle3 | EAGLE-3 (fixed ttt_length; budget only affects inference) | `qwen3-{4b,8b}-eagle3.json` | same (shared weights) |
| dcore | DFlareDraftModel × D-PACE loss (ours) | `qwen3-{4b,8b}-dflare_9hs-bs8.json` | `qwen3-{4b,8b}-dflare_9hs.json` |

Common options:

| Option | Description |
| --- | --- |
| `--budget 8\|16` | Draft window (default 16) |
| `--gpus N` | Number of GPUs (default: auto-detected via `nvidia-smi`) |
| `--no-wandb` | Disable wandb (same as `REPORT_TO=none`) |
| `--run-name X` | Override RUN_NAME (output directory / wandb run name) |
| `--master-port` | torchrun master port (default chosen per script) |
| `--dry-run` | Print the command without running it |

Before launching, `train.sh` checks the target weights, launch script, and GPU count, reports exactly what is missing, and downloads the training data if it is not there yet.

### 5. Logs and monitoring

- Run logs go to `outputs/perfectblend/<RUN_NAME>.log` (launched with nohup; see each script's header comment).
- Checkpoints are saved to `outputs/perfectblend/<RUN_NAME>/epoch_X_step_Y/`.
- Metrics: `--report-to wandb|swanlab|tensorboard|mlflow|none` (default wandb; set `WANDB_API_KEY` in `.env`, see `.env.example`).

### 6. Released checkpoints

The trained drafters are available in the [DCore collection on Hugging Face](https://huggingface.co/collections/ernie-research/dcore) (repo ids are listed in `demo/spec_race/registry.py`). The demo downloads them automatically; you can also fetch them with `snapshot_download` and plug them into the scripts under `examples/colocated/`.

## Evaluation (SpecBench)

Evaluation has two entry points (8 GPUs; the latest checkpoint is discovered automatically under `outputs/<MODEL>/recipe-level/`):

```bash
# Single request: accuracy + throughput on 10 benchmarks (concurrency 1 per replica)
bash eval/run_eval_single.sh qwen3-8b dcore

# High concurrency: throughput on the SpecMix mixed-traffic set
# (downloaded from Hugging Face on first use; default concurrency 64 per replica)
bash eval/run_eval_concurrent.sh qwen3-8b dcore
```

- SpecMix ([`wingchiuloong/specmix`](https://huggingface.co/datasets/wingchiuloong/specmix)) is 3,331 prompts sampled from the ten benchmarks and shuffled with a fixed seed. It is cached to `cache/dataset/specmix` on first run; `scripts/build_specmix_dataset.py` rebuilds it from the source benchmarks.
- Arguments: `<qwen3-4b|qwen3-8b> <method> [block ...]`, where block is the verification window (e.g. `8 16`); omit it to run every budget of that method.
- Methods: `dcore dflash dflare dpace domino dspark_markov eagle3_chain eagle3_tree` (`*_origin` variants evaluate the third-party released weights for comparison).
- The single-request script defaults to temperature 0 (fast and reproducible); the concurrent script defaults to `1,0` (T=1 rejection sampling + T=0 greedy). Override with `TEMPERATURES`; `DATASETS` overrides the default benchmarks.
- Results are written to `outputs/eval/<model>/<SWEEP_ID>/` (`results.jsonl` + `responses/` + `report.md`), reporting acceptance length and tokens/s. EAGLE-3 tree-decoding parameters (steps/topk/draft) are defined in `EAGLE3_TREE` in `demo/spec_race/registry.py`.

## Demo (race replay)

```bash
bash demo/spec_race/run_demo.sh        # 8 GPUs by default, one method per GPU
# then open http://127.0.0.1:8099 in a browser
```

See `demo/spec_race/README.md` for details. Checkpoints, temperature presets, and method order are configured in `demo/spec_race/registry.py`; missing target models are pulled from the official Qwen repos.

## Repository layout

```
train.sh                     # one-line training entry (<model, method, budget> -> launch script)
build_env.sh                 # environment build (uv + rust/protoc + sglang-kernel)
scripts/                     # training entry points (train_dflash/domino/dspark/eagle3.py) and data processing
specforge/                   # core training library (modeling / training / data / eval)
examples/colocated/          # launch scripts for the 7 methods under qwen3_4b and qwen3_8b (per loss/budget variant)
configs/                     # draft model configs (qwen3-4b-*.json / qwen3-8b-*.json)
specbench/                   # SpecBench evaluation harness
eval/                        # evaluation launchers (run_eval_single.sh / run_eval_concurrent.sh)
demo/spec_race/              # race replay demo
third_party/sglang/          # SGLang submodule (with the speculative-decoding changes used by DCore)
```

## Citation

If you find our paper or this repository useful for your research, please consider citing:

```bibtex
@misc{long2026doesbetteracceptancepay,
  title={When Does Better Acceptance Pay? A Controlled Study of Speculative Decoding},
  author={Yongchao Long and Chunrui Zheng and Yuxuan Zhou and Guangkun Nie and Tian-Chi Ma and Chao Pang and Shuohuan Wang and Yuchen Ding and Xianbin Wen and Dianhai Yu and Junyuan Shang and Yuxi Zhou and Shenda Hong},
  year={2026},
  eprint={2610.24993},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2610.24993}
}
```

## Acknowledgments

Our implementation is built on [SpecForge](https://github.com/sgl-project/SpecForge) (MIT License) and [SGLang](https://github.com/sgl-project/sglang). This repository keeps SpecForge's training and evaluation framework and adds DCore and the other drafters on top of it, together with the full data pipeline. If you use this code, please also consider citing SpecForge (see the upstream repository). We thank their authors and maintainers for the training and serving foundations of this work.

We also thank the following projects, whose open-source implementations informed our training and evaluation code:

- [EAGLE](https://github.com/SafeAILab/EAGLE) for EAGLE-3.
- [DFlash](https://github.com/z-lab/dflash).
- [D-PACE](https://github.com/Lucas-TY/D-PACE).
- [AngelSlim](https://github.com/Tencent/AngelSlim) for DFlare.
- [Domino](https://github.com/jianuo-huang/Domino).
- [DeepSpec](https://github.com/deepseek-ai/DeepSpec) for DSpark.

