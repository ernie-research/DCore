# coding=utf-8
"""Auto vocab mapping generation for Shotgun training.

Reuses frequency-counting logic from specforge.data.preprocessing but wraps it
with auto-path-resolution and caching keyed by segment_vocab_size.
"""

import math
import os
from collections import Counter
from typing import Optional

import torch
from tqdm import tqdm


def generate_auto_vocab_mapping(
    dataset,
    segment_vocab_size: int,
    target_vocab_size: int,
    output_path: str,
) -> str:
    """Generate a vocab mapping file from training data and save to output_path.

    Args:
        dataset: HF dataset with 'input_ids' and 'loss_mask' columns.
        segment_vocab_size: Number of top-frequency tokens to keep per segment.
        target_vocab_size: Full target vocabulary size.
        output_path: Where to save the .pt file.

    Returns:
        The output_path (for chaining).
    """
    if os.path.exists(output_path):
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Count token frequencies from loss-masked positions
    token_counter = Counter()
    for input_ids, loss_mask in tqdm(
        zip(dataset["input_ids"], dataset["loss_mask"]),
        total=len(dataset),
        desc="Counting tokens for auto vocab mapping",
    ):
        if isinstance(input_ids, torch.Tensor):
            masked_ids = input_ids[loss_mask == 1]
            unique_ids, counts = masked_ids.unique(return_counts=True)
            token_counter.update(dict(zip(unique_ids.tolist(), counts.tolist())))
        else:
            # list/numpy fallback
            for i, m in enumerate(loss_mask):
                if m:
                    token_counter[input_ids[i]] += 1

    # Select top segment_vocab_size tokens
    top_tokens = [tok for tok, _ in token_counter.most_common(segment_vocab_size)]

    # Pad if fewer unique tokens than segment_vocab_size
    if len(top_tokens) < segment_vocab_size:
        existing = set(top_tokens)
        for tid in range(target_vocab_size):
            if tid not in existing:
                top_tokens.append(tid)
            if len(top_tokens) >= segment_vocab_size:
                break

    used_tokens = sorted(top_tokens[:segment_vocab_size])

    # Build d2t and t2d
    d2t = torch.tensor(used_tokens, dtype=torch.int64)
    t2d = torch.full((target_vocab_size,), -1, dtype=torch.int64)
    for draft_id, target_id in enumerate(used_tokens):
        t2d[target_id] = draft_id

    torch.save({"d2t": d2t, "t2d": t2d}, output_path)
    return output_path


def resolve_vocab_mapping_path(
    target_model_path: str,
    segment_vocab_size: int,
    explicit_path: Optional[str] = None,
) -> str:
    """Resolve vocab mapping file path.

    If explicit_path is given, use it directly.
    Otherwise derive: {target_model_path}/vocab_mapping_{segment_vocab_size}.pt
    """
    if explicit_path:
        return explicit_path
    return os.path.join(target_model_path, f"vocab_mapping_{segment_vocab_size}.pt")
