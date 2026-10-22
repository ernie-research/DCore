"""HF-Direct harness — the in-process HuggingFace protocol.

Loads the target and draft models directly and drives
``draft_model.spec_generate`` (ported from ``eval/benchmark_dflash.py``).
Metrics use the turn-0, hot-pass convention of the original script:
  * tok/s     = new_tokens / wall_time   (per-sample, summed)
  * accept_len = mean of per-step acceptance lengths

This is a DIFFERENT inference stack from the SGLang harness, so its tok/s is
NOT comparable to SGLang tok/s. Only ``accept_length`` is cross-harness
comparable. EAGLE3 has no path here and raises immediately.

Multi-GPU via torchrun (``specbench.distributed``); single-process otherwise.
"""

from __future__ import annotations

import os
import time
from glob import glob
from itertools import chain

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

from specbench import distributed as dist
from specbench.config import BenchConfig
from specbench.datasets import load_and_process_dataset
from specbench.harness.base import Harness
from specbench.methods import MethodSpec
from specbench.metrics import BenchResult, append_jsonl, write_results

# checkpoint weight key -> model type
# Note: a legacy "dflash_full" (depth_proj) variant existed before the 7 released
# methods; it is not part of the public codebase, so no marker is registered.
_TYPE_MARKERS = (
    ("layer_fusion_weights", "dflare"),
    ("markov_w1", "dspark"),
    ("prefix_gru", "domino"),
)


def _detect_model_type(draft_path: str) -> str:
    from safetensors import safe_open

    files = glob(os.path.join(draft_path, "*.safetensors"))
    if files:
        for f in files:
            with safe_open(f, framework="pt") as st:
                keys = list(st.keys())
                for marker, mtype in _TYPE_MARKERS:
                    if any(marker in k for k in keys):
                        return mtype
        return "dflash"
    for f in glob(os.path.join(draft_path, "*.bin")):
        state = torch.load(f, map_location="cpu", weights_only=True)
        for marker, mtype in _TYPE_MARKERS:
            if any(marker in k for k in state.keys()):
                return mtype
    return "dflash"


def _load_draft_model(draft_path: str, model_type: str, attn_impl: str, device):
    config = AutoConfig.from_pretrained(draft_path)
    if model_type == "dflare":
        from specforge.modeling.draft.dflare import DFlareDraftModel as Cls
    elif model_type == "dspark":
        from specforge.modeling.draft.dspark import DSparkDraftModel as Cls
    elif model_type == "domino":
        from specforge.modeling.draft.domino import DominoDraftModel as Cls
    else:
        from specforge.modeling.draft.dflash import DFlashDraftModel as Cls
    model = Cls.from_pretrained(draft_path, config=config,
                                attn_implementation=attn_impl, dtype=torch.bfloat16)
    return model.to(device).eval()


class HFDirectHarness(Harness):
    name = "hf_direct"

    def run(self, method: MethodSpec, cfg: BenchConfig, done=None) -> list[BenchResult]:
        method.check_harness("hf_direct")
        done = done or set()
        is_baseline = method.family == "baseline"
        if not is_baseline and not cfg.draft_model:
            raise ValueError(f"Method '{method.name}' requires --draft-model.")

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        dist.init()
        torch.cuda.set_device(dist.local_rank())
        device = torch.device(f"cuda:{dist.local_rank()}")

        attn_impl = cfg.hf_attn_implementation
        if attn_impl == "flash_attention_2" and not is_flash_attn_2_available():
            attn_impl = "sdpa"

        target = AutoModelForCausalLM.from_pretrained(
            cfg.target_model, attn_implementation=attn_impl, dtype=torch.bfloat16
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg.target_model)

        draft_model = None
        model_type = None
        if not is_baseline:
            model_type = method.hf_direct_model_type or _detect_model_type(cfg.draft_model)
            draft_model = _load_draft_model(cfg.draft_model, model_type, attn_impl, device)
            if model_type == "dspark" and (draft_model.embed_tokens is None or draft_model.lm_head is None):
                draft_model.initialize_embeddings_and_head(
                    target.model.embed_tokens, target.lm_head, freeze=True)
            if dist.is_main():
                print(f"[specbench.hf_direct] method={method.name} detected_type={model_type} "
                      f"block_size={getattr(draft_model, 'block_size', 'n/a')}")

        results: list[BenchResult] = []
        for dataset_name, count in cfg.dataset_specs():
            if (dataset_name, None, 1) in done:
                if dist.is_main():
                    print(f"[specbench.hf_direct] skip dataset {dataset_name} (already done)")
                continue
            res = self._run_dataset(
                dataset_name, count, cfg, method, target, draft_model, tokenizer, device, is_baseline)
            if res is not None:
                results.append(res)
        dist.destroy()
        return results

    def _run_dataset(self, dataset_name, num_samples, cfg, method, target, draft_model, tokenizer,
                     device, is_baseline):
        ds = load_and_process_dataset(dataset_name)
        if num_samples and len(ds) > num_samples:
            ds = ds.shuffle(seed=0).select(range(num_samples))

        def one_sample(idx):
            user_content = ds[idx]["turns"][0]
            messages = [{"role": "user", "content": user_content}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
            n_in = input_ids.shape[1]

            rec = {"spec_tokens": 0, "spec_time": 0.0, "accept": [],
                   "base_tokens": 0, "base_time": 0.0,
                   "sample_index": int(idx), "prompt": user_content, "response": ""}

            if not is_baseline:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out_ids, acc_lens = draft_model.spec_generate(
                    target=target, input_ids=input_ids, max_new_tokens=cfg.max_new_tokens,
                    stop_token_ids=[tokenizer.eos_token_id], temperature=cfg.temperature,
                    confidence_threshold=cfg.confidence_threshold)
                torch.cuda.synchronize()
                rec["spec_time"] = time.perf_counter() - t0
                rec["spec_tokens"] = out_ids.shape[1] - n_in
                rec["accept"] = list(acc_lens)
                rec["response"] = tokenizer.decode(out_ids[0, n_in:], skip_special_tokens=True)

            if is_baseline or cfg.run_baseline:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out_ids = target.generate(
                    input_ids, max_new_tokens=cfg.max_new_tokens,
                    do_sample=(cfg.temperature > 1e-5),
                    temperature=cfg.temperature if cfg.temperature > 1e-5 else None,
                    eos_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize()
                rec["base_time"] = time.perf_counter() - t0
                rec["base_tokens"] = out_ids.shape[1] - n_in
                if is_baseline:
                    rec["response"] = tokenizer.decode(out_ids[0, n_in:], skip_special_tokens=True)
            return rec

        indices = list(range(dist.rank(), len(ds), dist.size()))
        local_recs = []
        for pos, idx in enumerate(indices):
            if pos == 0:
                one_sample(idx)   # warmup pass, discarded
            local_recs.append(one_sample(idx))

        gathered = dist.gather(local_recs, dst=0)
        if not dist.is_main():
            return None
        recs = list(chain(*gathered)) if gathered else []
        if not recs:
            return None

        if cfg.save_responses:
            rpath = cfg.responses_file(dataset_name, method.name)
            if rpath:
                rpath.parent.mkdir(parents=True, exist_ok=True)
                rpath.unlink(missing_ok=True)
                append_jsonl(rpath, ({
                    "harness": self.name, "method": method.name, "dataset": dataset_name,
                    "draft_model": (None if is_baseline else cfg.draft_model),
                    "concurrency": None, "temperature": cfg.temperature,
                    "sample_index": r.get("sample_index"), "prompt": r.get("prompt", ""),
                    "response": r.get("response", ""),
                    "completion_tokens": (r["spec_tokens"] if not is_baseline else r["base_tokens"]),
                    "spec_accept_length": (float(np.mean(r["accept"])) if r.get("accept") else None),
                    "e2e_latency": (r["spec_time"] if not is_baseline else r["base_time"]),
                } for r in sorted(recs, key=lambda x: x.get("sample_index", 0))))

        spec_tok = sum(r["spec_tokens"] for r in recs)
        spec_time = sum(r["spec_time"] for r in recs)
        base_tok = sum(r["base_tokens"] for r in recs)
        base_time = sum(r["base_time"] for r in recs)
        all_acc = list(chain(*[r["accept"] for r in recs if r["accept"]]))

        base_tps = base_tok / base_time if base_time > 0 else None
        if is_baseline:
            tps = base_tps or 0.0
            accept_len = None
            speedup = None
            base_out = None
        else:
            tps = spec_tok / spec_time if spec_time > 0 else 0.0
            accept_len = float(np.mean(all_acc)) if all_acc else None
            speedup = (tps / base_tps) if (base_tps and base_tps > 0) else None
            base_out = base_tps

        br = BenchResult(
            harness=self.name, method=method.name, dataset=dataset_name,
            tokens_per_second=float(tps), accept_length=accept_len,
            baseline_tokens_per_second=base_out, speedup=speedup,
            num_samples=len(recs), concurrency=None,
            target_model=cfg.target_model,
            draft_model=(None if is_baseline else cfg.draft_model),
            backend=cfg.hf_attn_implementation, temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            latency_s=float(spec_time if not is_baseline else base_time),
            extra={"detected_type": (None if is_baseline else (method.hf_direct_model_type or "auto"))})
        write_results(cfg.results_file(), [br], append=True)
        return br
