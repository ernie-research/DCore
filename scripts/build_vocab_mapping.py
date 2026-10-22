#!/usr/bin/env python3
"""Generate vocab_mapping_top32000.pt from a conversations jsonl file.

Tokenizes assistant responses, counts token frequencies, selects top-K tokens
as draft vocabulary, and produces d2t/t2d mappings in the same format as
existing vocab_mapping.pt files.

Usage:
    python scripts/build_vocab_mapping.py \
        --data cache/dataset/perfectblend/perfectblend_train_regen_qwen3_8b.jsonl \
        --tokenizer cache/models/Qwen3-8B \
        --draft-vocab-size 32000 \
        --output cache/models/Qwen3-8B/vocab_mapping_top32000.pt
"""
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer

# Global tokenizer for worker processes
_tokenizer = None


def _init_worker(tokenizer_path):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def _count_chunk(lines):
    """Count token frequencies for a chunk of jsonl lines."""
    counter = Counter()
    for line in lines:
        item = json.loads(line)
        for turn in item["conversations"]:
            if turn["role"] == "assistant" and isinstance(turn.get("content"), str):
                ids = _tokenizer.encode(turn["content"], add_special_tokens=False)
                counter.update(ids)
    return counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--draft-vocab-size", type=int, default=32000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=5000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    target_vocab_size = max(len(tokenizer), 151936)

    # Read all lines
    print("Reading file...")
    with open(args.data) as f:
        lines = f.readlines()
    total_lines = len(lines)
    print(f"Total lines: {total_lines}, using {args.workers} workers")

    # Split into chunks
    chunks = [lines[i:i + args.chunk_size] for i in range(0, total_lines, args.chunk_size)]
    del lines  # free memory

    # Parallel tokenization
    token_counter = Counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(args.tokenizer,)) as executor:
        futures = [executor.submit(_count_chunk, chunk) for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Counting tokens"):
            token_counter += future.result()

    print(f"Unique tokens seen: {len(token_counter)}")
    total = sum(token_counter.values())

    # Select top-K
    top_k = token_counter.most_common(args.draft_vocab_size)
    top_k_freq = sum(f for _, f in top_k)
    print(f"Top {args.draft_vocab_size} token coverage: {top_k_freq/total:.2%}")

    # Build mappings
    used_tokens = sorted(tok for tok, _ in top_k)

    t2d = torch.full((target_vocab_size,), -1, dtype=torch.int64)
    for draft_id, target_id in enumerate(used_tokens):
        t2d[target_id] = draft_id

    d2t = torch.tensor(used_tokens, dtype=torch.int64)

    torch.save({"d2t": d2t, "t2d": t2d}, args.output)
    print(f"Saved: {args.output} (d2t: {d2t.shape}, t2d: {t2d.shape})")


if __name__ == "__main__":
    main()
