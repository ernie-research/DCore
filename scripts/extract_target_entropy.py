#!/usr/bin/env python3
"""Extract target-model token entropy along sampled continuations.

The script intentionally keeps model loading inside ``main`` so the conversation
and entropy helpers can be imported by lightweight unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

SCHEMA_VERSION = "target_entropy.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "cache/dataset/perfectblend/perfectblend_train_regen_qwen3_8b.jsonl"
DEFAULT_MODEL = REPO_ROOT / "cache/models/Qwen3-8B"
DEFAULT_OUTPUT = REPO_ROOT / "cache/dataset/perfectblend/perfectblend_train_regen_qwen3_8b_target_entropy.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "cache/dataset/perfectblend/perfectblend_train_regen_qwen3_8b_target_entropy.manifest.json"


def normalize_conversation(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the source conversation with every assistant turn removed."""
    messages = row.get("conversations", row.get("messages"))
    if not isinstance(messages, list):
        raise ValueError("row must contain a conversations or messages list")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("conversation messages must be objects")
        role = str(message.get("role", message.get("from", "user"))).lower()
        if role in {"assistant", "gpt", "bot", "chatgpt"}:
            continue
        copy = dict(message)
        copy["role"] = {"human": "user"}.get(role, role)
        copy.pop("from", None)
        normalized.append(copy)
    return normalized


def entropy_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return entropy, log probabilities, and probabilities in float32."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy, log_probs, probs


def aggregate_entropy(entropies: Sequence[float]) -> dict[str, float | int]:
    values = [float(value) for value in entropies]
    if not values:
        return {"H_mean": 0.0, "H_median": 0.0, "H_min": 0.0, "H_max": 0.0, "H_std": 0.0, "token_count": 0}
    return {
        "H_mean": statistics.fmean(values),
        "H_median": statistics.median(values),
        "H_min": min(values),
        "H_max": max(values),
        "H_std": statistics.pstdev(values),
        "token_count": len(values),
    }


def iter_jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    source_index = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} of {path} is not an object")
            yield source_index, row
            source_index += 1


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for _, row in iter_jsonl_rows(path):
        yield row


def partition_rows(
    rows: Iterator[tuple[int, dict[str, Any]]], rank: int, world_size: int
) -> Iterator[tuple[int, dict[str, Any]]]:
    for source_index, row in rows:
        if source_index % world_size == rank:
            yield source_index, row


def rank_path(path: Path, rank: int, world_size: int) -> Path:
    if world_size == 1:
        return path
    return path.with_name(f"{path.stem}.rank{rank}of{world_size}{path.suffix}")


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        paths = sorted(item for item in path.rglob("*") if item.is_file())
        for item in paths:
            digest.update(str(item.relative_to(path)).encode("utf-8"))
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    else:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def sample_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int, generator: torch.Generator) -> torch.Tensor:
    scaled = logits.float() / temperature
    if top_k > 0:
        values, _ = torch.topk(scaled, min(top_k, scaled.shape[-1]), dim=-1)
        scaled = scaled.masked_fill(scaled < values[-1], float("-inf"))
    if top_p < 1.0:
        values, indices = torch.sort(scaled, descending=True)
        probs = torch.softmax(values, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        keep = cumulative - probs < top_p
        keep[0] = True
        scaled = torch.full_like(scaled, float("-inf"))
        scaled.scatter_(0, indices[keep], values[keep])
    return torch.multinomial(torch.softmax(scaled, dim=-1), 1, generator=generator).squeeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--log-top-k", type=int, default=0, help="Store top-k token probabilities per generated token")
    parser.add_argument("--device", default=None, help="Torch device; defaults to CUDA when available")
    return parser.parse_args()


def make_record(source_index: int, row: dict[str, Any], prompt_token_count: int, token_ids: list[int], token_records: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    metadata = {key: value for key, value in row.items() if key not in {"conversations", "messages"}}
    return {
        "schema_version": SCHEMA_VERSION,
        "source_index": source_index,
        "id": row.get("id"),
        "metadata": metadata,
        "prompt_token_count": prompt_token_count,
        "generated_token_ids": token_ids,
        "generated_text": "".join(record["text"] for record in token_records),
        "tokens": token_records,
        "entropy": aggregate_entropy([record["entropy"] for record in token_records]),
        "generation": settings,
    }


def generate_batch(rows: list[tuple[int, dict[str, Any]]], tokenizer: Any, model: Any, args: argparse.Namespace, device: torch.device, generator: torch.Generator) -> list[dict[str, Any]]:
    source_indices = [source_index for source_index, _ in rows]
    row_values = [row for _, row in rows]
    conversations = [normalize_conversation(row) for row in row_values]
    prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) for messages in conversations]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_lengths = attention_mask.sum(dim=1).tolist()
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, logits_to_keep=1)
    past = output.past_key_values
    logits = output.logits[:, -1, :].clone()
    del output
    eos_ids = {int(value) for value in ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])}
    token_ids: list[list[int]] = [[] for _ in rows]
    token_records: list[list[dict[str, Any]]] = [[] for _ in rows]
    cumulative_entropy = [0.0] * len(rows)
    cumulative_log_probability = [0.0] * len(rows)
    finished = [False] * len(rows)
    settings = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed, "log_top_k": args.log_top_k, "rank": args.rank, "world_size": args.world_size, "local_sample_count": args.local_sample_count}
    next_ids = []
    for _ in range(args.max_new_tokens):
        entropy, log_probs, probs = entropy_from_logits(logits)
        chosen = []
        for index in range(len(rows)):
            if finished[index]:
                chosen.append(tokenizer.eos_token_id or 0)
                continue
            token = sample_token(logits[index], args.temperature, args.top_p, args.top_k, generator)
            token_id = int(token.item())
            token_entropy = float(entropy[index].item())
            token_log_probability = float(log_probs[index, token_id].item())
            cumulative_entropy[index] += token_entropy
            cumulative_log_probability[index] += token_log_probability
            record: dict[str, Any] = {
                "position": len(token_records[index]),
                "token_id": token_id,
                "text": tokenizer.decode([token_id], skip_special_tokens=False),
                "entropy": token_entropy,
                "log_probability": token_log_probability,
                "cumulative_entropy": cumulative_entropy[index],
                "cumulative_log_probability": cumulative_log_probability[index],
            }
            if args.log_top_k > 0:
                top_values, top_indices = torch.topk(probs[index], min(args.log_top_k, probs.shape[-1]))
                record["top_k"] = [{"token_id": int(token), "text": tokenizer.decode([int(token)], skip_special_tokens=False), "probability": float(value)} for value, token in zip(top_values.tolist(), top_indices.tolist())]
            token_ids[index].append(token_id)
            token_records[index].append(record)
            finished[index] = token_id in eos_ids
            chosen.append(token_id)
        next_ids = torch.tensor(chosen, device=device, dtype=torch.long).unsqueeze(1)
        if all(finished):
            break
        attention_mask = torch.cat([attention_mask, torch.ones((len(rows), 1), device=device, dtype=attention_mask.dtype)], dim=1)
        with torch.inference_mode():
            output = model(input_ids=next_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = output.past_key_values
        logits = output.logits[:, -1, :].clone()
        del output
    return [make_record(source_indices[index], row_values[index], int(prompt_lengths[index]), token_ids[index], token_records[index], settings) for index in range(len(rows))]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 0 or args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0 or args.log_top_k < 0:
        raise ValueError("invalid generation settings")
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size and world_size >= 1")
    args.output = rank_path(args.output, args.rank, args.world_size)
    args.manifest = rank_path(args.manifest, args.rank, args.world_size)
    args.local_sample_count = sum(
        1
        for source_index, _ in iter_jsonl_rows(args.input)
        if (args.max_samples is None or source_index < args.max_samples)
        and source_index % args.world_size == args.rank
    )
    tokenizer_path = args.tokenizer or args.model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", trust_remote_code=True).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = partition_rows(
        (
            (source_index, row)
            for source_index, row in iter_jsonl_rows(args.input)
            if args.max_samples is None or source_index < args.max_samples
        ),
        args.rank,
        args.world_size,
    )
    count = 0
    temp_output = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    try:
        with temp_output.open("w", encoding="utf-8") as handle:
            while args.max_samples is None or count < args.max_samples:
                batch = []
                for _ in range(args.batch_size):
                    if args.max_samples is not None and count + len(batch) >= args.max_samples:
                        break
                    try:
                        batch.append(next(rows))
                    except StopIteration:
                        break
                if not batch:
                    break
                for record in generate_batch(batch, tokenizer, model, args, device, generator):
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    count += 1
        os.replace(temp_output, args.output)
    finally:
        if temp_output.exists():
            temp_output.unlink()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": str(args.model),
        "tokenizer": str(tokenizer_path),
        "input": str(args.input),
        "output": str(args.output),
        "input_sha256": fingerprint(args.input),
        "model_sha256": fingerprint(args.model),
        "tokenizer_sha256": fingerprint(tokenizer_path),
        "model_config": getattr(model.config, "_name_or_path", str(tokenizer_path)),
        "command": sys.argv,
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()} | {"device": str(device), "tokenizer": str(tokenizer_path)},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "count": count,
        "local_sample_count": args.local_sample_count,
        "rank": args.rank,
        "world_size": args.world_size,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
