#!/usr/bin/env python3
"""Analyze draft model acceptance rate for OOV vs in-vocab tokens.

Supports both Domino (third_party) and DFlash (specforge) draft models.

Usage:
    python eval/analyze_domino_oov_acceptance.py \
        --model-name-or-path cache/models/Qwen3-8B \
        --draft-name-or-path <checkpoint> \
        --vocab-mappings cache/models/Qwen3-8B/vocab_mapping.pt,cache/models/Qwen3-8B/vocab_mapping_16seg.pt \
        --dataset gsm8k --max-samples 32 \
        --model-type domino  # or dflash
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from itertools import chain
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../third_party/Domino/code"))
sys.path.insert(0, os.path.dirname(__file__))  # for benchmark_dflash
from dflash import DFlashDraftModel, extract_context_feature, is_domino_projector, sample, cuda_time
from model import load_and_process_dataset
import distributed as dist

from benchmark import load_draft_model_for_benchmark, normalize_draft_config_for_benchmark


def spec_generate_with_token_log(
    draft_model, target, input_ids, max_new_tokens, temperature, block_size, stop_token_ids, use_bias,
):
    """Patched spec_generate that returns per-step token-level acceptance info."""
    device = draft_model.device
    input_ids = input_ids.to(device)
    block_size = int(block_size or draft_model.block_size)
    mask_token_id = draft_model.mask_token_id

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + int(max_new_tokens)
    shift_label = bool(getattr(draft_model.config, "dflash_config", {}).get("shift_label", False))
    extra_buffer = block_size + 1 if shift_label else block_size
    prefix_len = int(draft_model.pure_draft_prefix_len)

    output_ids = torch.full((1, max_length + extra_buffer), mask_token_id, dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    from transformers.cache_utils import DynamicCache
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    output = target(
        input_ids, position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target, use_cache=True,
        logits_to_keep=1, output_hidden_states=block_size > 1,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, draft_model.target_layer_ids)

    start = num_input_tokens
    draft_prefill = True
    # Per-step log: list of dicts with draft_tokens, target_tokens, accepted mask
    step_logs = []

    while start < max_length:
        block_output_ids = output_ids[:, start:start + block_size].clone()
        k_draft = block_size if shift_label else block_size - 1
        verify_ids = torch.full((1, k_draft + 1), mask_token_id, dtype=torch.long, device=device)
        verify_ids[:, 0] = output_ids[:, start]
        verify_position_ids = position_ids[:, start:start + k_draft + 1]

        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            parallel_hiddens = draft_model(
                target_hidden=target_hidden, noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length():start + block_size],
                past_key_values=past_key_values_draft, use_cache=True, is_causal=False,
            )
            if not shift_label:
                parallel_hiddens = parallel_hiddens[:, -block_size + 1:, :]
            past_key_values_draft.crop(start)

            base_logits = target.lm_head(parallel_hiddens)
            if prefix_len > 0:
                prefix_token_ids = sample(base_logits[:, :prefix_len], temperature)
                verify_ids[:, 1:1 + prefix_len] = prefix_token_ids

            realized_prefix_ids = verify_ids[:, :1 + prefix_len]
            realized_prefix_embeds = target.model.embed_tokens(realized_prefix_ids)
            _, gru_hidden = draft_model.prefix_gru(realized_prefix_embeds)

            for i in range(prefix_len, k_draft):
                z_i = parallel_hiddens[:, i:i + 1, :]
                s_i = gru_hidden.transpose(0, 1)
                bias = draft_model.embed_proj(torch.cat([z_i, s_i], dim=-1))
                current_token_id = sample(base_logits[:, i:i + 1, :] + bias, temperature)
                verify_ids[:, i + 1:i + 2] = current_token_id
                if i + 1 < k_draft:
                    new_embed = target.model.embed_tokens(current_token_id)
                    _, gru_hidden = draft_model.prefix_gru(new_embed, gru_hidden)

            if draft_prefill:
                draft_prefill = False

        output = target(
            verify_ids, position_ids=verify_position_ids,
            past_key_values=past_key_values_target, use_cache=True,
            output_hidden_states=block_size > 1,
        )
        posterior = sample(output.logits, temperature)
        # acceptance check
        matches = (verify_ids[:, 1:] == posterior[:, :-1])
        acceptance_length = matches.cumprod(dim=1).sum(dim=1)[0].item()

        # Log: draft proposed tokens vs target chosen tokens at each position
        draft_tokens = verify_ids[0, 1:k_draft + 1].cpu().tolist()
        target_tokens = posterior[0, :k_draft].cpu().tolist()
        accepted = matches[0, :k_draft].cpu().tolist()

        step_logs.append({
            "draft_tokens": draft_tokens,
            "target_tokens": target_tokens,
            "accepted": accepted,
            "acceptance_length": int(acceptance_length),
        })

        output_ids[:, start:start + acceptance_length + 1] = verify_ids[:, :acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += int(acceptance_length) + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(
                output.hidden_states, draft_model.target_layer_ids
            )[:, :acceptance_length + 1, :]

        if stop_token_ids is not None:
            stop_tensor = torch.tensor(stop_token_ids, device=device)
            if torch.isin(output_ids[:, num_input_tokens:start], stop_tensor).any():
                break

    return step_logs


def spec_generate_with_token_log_dflash(
    draft_model, target, input_ids, max_new_tokens, temperature, stop_token_ids,
):
    """Token-level acceptance log for DFlash (specforge) draft models."""
    from transformers.cache_utils import DynamicCache

    device = target.device
    input_ids = input_ids.to(device)
    block_size = draft_model.block_size
    mask_token_id = draft_model.mask_token_id

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + int(max_new_tokens)

    output_ids = torch.full((1, max_length + block_size), mask_token_id, dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    output = target(
        input_ids, position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target, use_cache=True,
        logits_to_keep=1, output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, draft_model.target_layer_ids)

    start = num_input_tokens
    step_logs = []

    while start < max_length:
        block_output_ids = output_ids[:, start:start + block_size].clone()
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(
            draft_model(
                target_hidden=target_hidden, noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length():start + block_size],
                past_key_values=past_key_values_draft, use_cache=True, is_causal=False,
            )[:, -block_size + 1:, :]
        )
        past_key_values_draft.crop(start)
        block_output_ids[:, 1:] = sample(draft_logits)

        output = target(
            block_output_ids, position_ids=position_ids[:, start:start + block_size],
            past_key_values=past_key_values_target, use_cache=True, output_hidden_states=True,
        )
        posterior = sample(output.logits, temperature)

        matches = (block_output_ids[:, 1:] == posterior[:, :-1])
        acceptance_length = matches.cumprod(dim=1).sum(dim=1)[0].item()

        k_draft = block_size - 1
        draft_tokens = block_output_ids[0, 1:].cpu().tolist()
        target_tokens = posterior[0, :k_draft].cpu().tolist()
        accepted = matches[0, :k_draft].cpu().tolist()

        step_logs.append({
            "draft_tokens": draft_tokens,
            "target_tokens": target_tokens,
            "accepted": accepted,
            "acceptance_length": int(acceptance_length),
        })

        output_ids[:, start:start + acceptance_length + 1] = block_output_ids[:, :acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        target_hidden = extract_context_feature(
            output.hidden_states, draft_model.target_layer_ids
        )[:, :acceptance_length + 1, :]

        if stop_token_ids is not None:
            stop_tensor = torch.tensor(stop_token_ids, device=device)
            if torch.isin(output_ids[:, num_input_tokens:start], stop_tensor).any():
                break

    return step_logs


def analyze_oov_acceptance(step_logs, t2d_tensor, mapping_name):
    """Compute acceptance rates split by OOV/in-vocab for target tokens."""
    oov_mask = (t2d_tensor == -1)  # shape [vocab_size]

    stats = defaultdict(lambda: {"accept": 0, "total": 0})
    # Per-position stats
    pos_stats = defaultdict(lambda: {"oov_accept": 0, "oov_total": 0, "iv_accept": 0, "iv_total": 0})

    for step in step_logs:
        for pos, (target_tok, acc) in enumerate(zip(step["target_tokens"], step["accepted"])):
            is_oov = oov_mask[target_tok].item() if target_tok < len(oov_mask) else True
            key = "oov" if is_oov else "in_vocab"
            stats[key]["total"] += 1
            if acc:
                stats[key]["accept"] += 1
            if is_oov:
                pos_stats[pos]["oov_total"] += 1
                if acc:
                    pos_stats[pos]["oov_accept"] += 1
            else:
                pos_stats[pos]["iv_total"] += 1
                if acc:
                    pos_stats[pos]["iv_accept"] += 1

    return stats, pos_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--vocab-mappings", type=str, required=True,
                        help="Comma-separated paths to vocab_mapping .pt files")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--use-bias", action="store_true", default=True)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="domino", choices=["domino", "dflash"],
                        help="Draft model type: domino (third_party) or dflash (specforge)")
    args = parser.parse_args()

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    torch.manual_seed(0)

    # Load models
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, attn_implementation=args.attn_implementation, dtype=torch.bfloat16,
    ).to(device).eval()

    if args.model_type == "dflash":
        from benchmark_dflash import load_draft_model
        draft_model, _ = load_draft_model(args.draft_name_or_path, args.attn_implementation, device)
    else:
        draft_model = load_draft_model_for_benchmark(args.draft_name_or_path, args.attn_implementation).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    block_size = args.block_size

    # Load vocab mappings
    vocab_mappings = {}
    for path in args.vocab_mappings.split(","):
        path = path.strip()
        name = os.path.splitext(os.path.basename(path))[0]
        mapping = torch.load(path, map_location="cpu")
        vocab_mappings[name] = mapping["t2d"]

    # Load dataset
    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    # Distribute samples across ranks
    indices = list(range(dist.rank(), len(dataset), dist.size()))

    # Run inference and collect token logs
    all_step_logs = []
    for local_i, idx in enumerate(indices):
        instance = dataset[idx]
        messages = [{"role": "user", "content": instance["turns"][0]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            if args.model_type == "dflash":
                step_logs = spec_generate_with_token_log_dflash(
                    draft_model, target, input_ids, args.max_new_tokens,
                    args.temperature, [tokenizer.eos_token_id],
                )
            else:
                step_logs = spec_generate_with_token_log(
                    draft_model, target, input_ids, args.max_new_tokens,
                    args.temperature, block_size, [tokenizer.eos_token_id], args.use_bias,
                )
        all_step_logs.extend(step_logs)
        if dist.is_main() and (local_i + 1) % 4 == 0:
            print(f"  [{args.dataset}] rank0: {local_i+1}/{len(indices)} samples done")

    # Gather all step_logs to rank 0
    if dist.size() > 1:
        gathered = dist.gather(all_step_logs, dst=0)
        if not dist.is_main():
            return
        all_step_logs = list(chain(*gathered))
    elif not dist.is_main():
        return

    # Analyze
    results = {}
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset} | Samples: {len(dataset)} | Steps: {len(all_step_logs)}")
    print(f"{'='*60}")

    for name, t2d in vocab_mappings.items():
        stats, pos_stats = analyze_oov_acceptance(all_step_logs, t2d, name)
        oov = stats["oov"]
        iv = stats["in_vocab"]
        oov_rate = oov["accept"] / oov["total"] * 100 if oov["total"] > 0 else 0
        iv_rate = iv["accept"] / iv["total"] * 100 if iv["total"] > 0 else 0

        print(f"\n[{name}] vocab_size={int((t2d != -1).sum())}, OOV_count={int((t2d == -1).sum())}")
        print(f"  In-vocab tokens: {iv['total']:>6} total, {iv['accept']:>6} accepted, rate={iv_rate:.1f}%")
        print(f"  OOV tokens:      {oov['total']:>6} total, {oov['accept']:>6} accepted, rate={oov_rate:.1f}%")
        print(f"  Acceptance gap:  {iv_rate - oov_rate:.1f}pp (in-vocab - OOV)")

        # Per-position breakdown
        print(f"  Per-position (0=first draft token):")
        for pos in sorted(pos_stats.keys()):
            ps = pos_stats[pos]
            oov_r = ps["oov_accept"] / ps["oov_total"] * 100 if ps["oov_total"] > 0 else 0
            iv_r = ps["iv_accept"] / ps["iv_total"] * 100 if ps["iv_total"] > 0 else 0
            print(f"    pos {pos}: in-vocab {iv_r:.1f}% ({ps['iv_accept']}/{ps['iv_total']})"
                  f"  OOV {oov_r:.1f}% ({ps['oov_accept']}/{ps['oov_total']})")

        per_pos = {}
        for pos in sorted(pos_stats.keys()):
            ps = pos_stats[pos]
            per_pos[str(pos)] = {
                "iv_accept": ps["iv_accept"], "iv_total": ps["iv_total"],
                "oov_accept": ps["oov_accept"], "oov_total": ps["oov_total"],
            }

        results[name] = {
            "vocab_size": int((t2d != -1).sum()),
            "oov_count": int((t2d == -1).sum()),
            "in_vocab_total": iv["total"], "in_vocab_accepted": iv["accept"], "in_vocab_rate": round(iv_rate, 2),
            "oov_total": oov["total"], "oov_accepted": oov["accept"], "oov_rate": round(oov_rate, 2),
            "gap_pp": round(iv_rate - oov_rate, 2),
            "per_position": per_pos,
        }

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        record = {"dataset": args.dataset, "num_samples": len(dataset), "num_steps": len(all_step_logs), "mappings": results}
        with open(args.output_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"\nSaved to {args.output_file}")


if __name__ == "__main__":
    main()
