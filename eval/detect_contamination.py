#!/usr/bin/env python3
"""
Detect data contamination between training set and evaluation benchmarks
using n-gram overlap and edit distance.

Usage:
    python eval/detect_contamination.py \
        --train_path cache/dataset/perfectblend/perfectblend_train_regen_qwen3_8b.jsonl \
        --eval_dir cache/dataset \
        --output_dir cache/dataset_clean \
        --ngram_size 8 \
        --threshold 0.7 \
        --enable_edit_distance \
        --edit_distance_threshold 0.2 \
        --edit_distance_candidates 50
"""

import argparse
import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from datasets import load_from_disk
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm


def normalize_text(text: str) -> str:
    """Normalize text for fair comparison: lowercase, collapse whitespace, strip punctuation."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_ngrams(text: str, n: int) -> list[str]:
    """Extract word-level n-grams from text."""
    words = text.split()
    if len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def build_train_ngram_index(train_questions: list[str], n: int) -> set[str]:
    """Build a set of all n-grams from training questions."""
    ngram_set = set()
    for q in train_questions:
        norm = normalize_text(q)
        for ng in get_ngrams(norm, n):
            ngram_set.add(ng)
    return ngram_set


def compute_overlap_ratio(text: str, train_ngrams: set[str], n: int) -> float:
    """Compute fraction of eval sample's n-grams that appear in training set."""
    norm = normalize_text(text)
    sample_ngrams = get_ngrams(norm, n)
    if not sample_ngrams:
        return 0.0
    hits = sum(1 for ng in sample_ngrams if ng in train_ngrams)
    return hits / len(sample_ngrams)


def is_exact_match(text: str, train_normalized: set[str]) -> bool:
    """Check if normalized eval text exactly matches any training question."""
    return normalize_text(text) in train_normalized


def compute_min_edit_distance(text: str, candidates: list[str]) -> tuple[float, int]:
    """
    Compute normalized edit distance between text and closest candidate.
    Returns (min_normalized_distance, best_candidate_index).
    Normalized distance = edit_distance / max(len(a), len(b)), range [0, 1].
    0 means identical, 1 means completely different.
    """
    norm_text = normalize_text(text)
    min_dist = 1.0
    best_idx = -1
    for i, cand in enumerate(candidates):
        max_len = max(len(norm_text), len(cand))
        if max_len == 0:
            continue
        dist = Levenshtein.distance(norm_text, cand) / max_len
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return min_dist, best_idx


# --- Dataset-specific question extractors ---

EVAL_EXTRACTORS = {
    "aime25": lambda row: row["problem"],
    "gsm8k": lambda row: row["question"],
    "humaneval": lambda row: row["prompt"],
    "livecodebench": lambda row: row["question_content"],
    "math500": lambda row: row["problem"],
    "mbpp": lambda row: row["prompt"],
    "mt-bench": lambda row: row["prompt"][0] if isinstance(row["prompt"], list) else row["prompt"],
}


def load_train_questions(path: str) -> list[str]:
    """Load questions from training JSONL."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading training data"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            convs = obj.get("conversations", [])
            if convs and convs[0].get("role") == "user":
                questions.append(convs[0]["content"])
    return questions


def detect_contamination(
    train_questions: list[str],
    eval_dataset,
    extractor,
    ngram_size: int,
    threshold: float,
    enable_edit_distance: bool = False,
    edit_distance_threshold: float = 0.2,
    edit_distance_candidates: int = 50,
):
    """
    Returns list of (index, score, match_type) for contaminated samples.
    match_type: "exact", "ngram", or "edit_distance"

    Edit distance check: for samples not caught by exact/ngram, find the closest
    training sample (pre-filtered by n-gram overlap to top-K candidates) and check
    if normalized edit distance < edit_distance_threshold.
    """
    # Build indices
    train_normalized_list = [normalize_text(q) for q in train_questions]
    train_normalized_set = set(train_normalized_list)
    train_ngrams = build_train_ngram_index(train_questions, ngram_size)

    # For edit distance: build per-training-sample ngram sets for candidate filtering
    if enable_edit_distance:
        train_ngram_sets = []
        for norm_q in train_normalized_list:
            train_ngram_sets.append(set(get_ngrams(norm_q, ngram_size)))

    contaminated = []
    for i, row in enumerate(tqdm(eval_dataset, desc="Scanning")):
        text = extractor(row)
        if not text:
            continue
        if is_exact_match(text, train_normalized_set):
            contaminated.append((i, 1.0, "exact"))
            continue

        ratio = compute_overlap_ratio(text, train_ngrams, ngram_size)
        if ratio >= threshold:
            contaminated.append((i, ratio, "ngram"))
            continue

        # Edit distance check on samples not caught above
        if enable_edit_distance and ratio > 0:
            # Pre-filter: use ngram Jaccard to find top-K candidates
            eval_ng_set = set(get_ngrams(normalize_text(text), ngram_size))
            if eval_ng_set:
                scores = []
                for j, t_ng_set in enumerate(train_ngram_sets):
                    if not t_ng_set:
                        continue
                    jaccard = len(eval_ng_set & t_ng_set) / len(eval_ng_set | t_ng_set)
                    if jaccard > 0:
                        scores.append((jaccard, j))
                scores.sort(reverse=True)
                top_candidates = [train_normalized_list[j] for _, j in scores[:edit_distance_candidates]]
                if top_candidates:
                    min_dist, _ = compute_min_edit_distance(text, top_candidates)
                    if min_dist < edit_distance_threshold:
                        contaminated.append((i, 1.0 - min_dist, "edit_distance"))

    return contaminated


def main():
    parser = argparse.ArgumentParser(description="Detect eval-train data contamination via n-gram overlap")
    parser.add_argument("--train_path", required=True, help="Path to training JSONL file")
    parser.add_argument("--eval_dir", required=True, help="Base dir containing eval dataset subdirs")
    parser.add_argument("--output_dir", default=None, help="Dir to save cleaned eval datasets (default: <eval_dir>_clean)")
    parser.add_argument("--ngram_size", type=int, default=8, help="N-gram size (default: 8)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Overlap ratio threshold (default: 0.7)")
    parser.add_argument("--eval_sets", nargs="*", default=None, help="Specific eval sets to check (default: all known)")
    parser.add_argument("--report_path", default=None, help="Path to save JSON contamination report")
    parser.add_argument("--enable_edit_distance", action="store_true", help="Enable edit distance fuzzy matching")
    parser.add_argument("--edit_distance_threshold", type=float, default=0.2,
                        help="Normalized edit distance threshold (default: 0.2, i.e. <20%% edits = contaminated)")
    parser.add_argument("--edit_distance_candidates", type=int, default=50,
                        help="Number of candidate training samples to compare via edit distance (default: 50)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.eval_dir.rstrip("/") + "_clean"
    if args.report_path is None:
        args.report_path = os.path.join(args.output_dir, "contamination_report.json")

    eval_sets = args.eval_sets or list(EVAL_EXTRACTORS.keys())

    print(f"Loading training data from {args.train_path} ...")
    train_questions = load_train_questions(args.train_path)
    print(f"  Loaded {len(train_questions)} training questions")
    print(f"  Building {args.ngram_size}-gram index ...")
    train_ngrams = build_train_ngram_index(train_questions, args.ngram_size)
    print(f"  Index size: {len(train_ngrams)} unique {args.ngram_size}-grams")
    train_normalized = set(normalize_text(q) for q in train_questions)

    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "config": {
            "ngram_size": args.ngram_size,
            "threshold": args.threshold,
            "enable_edit_distance": args.enable_edit_distance,
            "edit_distance_threshold": args.edit_distance_threshold,
            "edit_distance_candidates": args.edit_distance_candidates,
        },
        "results": {},
    }

    if args.enable_edit_distance:
        print(f"  Edit distance enabled: threshold={args.edit_distance_threshold}, candidates={args.edit_distance_candidates}")

    for name in eval_sets:
        eval_path = os.path.join(args.eval_dir, name)
        if not os.path.isdir(eval_path):
            print(f"\n[SKIP] {name}: directory not found at {eval_path}")
            continue
        if name not in EVAL_EXTRACTORS:
            print(f"\n[SKIP] {name}: no extractor defined")
            continue

        print(f"\n{'='*60}")
        print(f"Checking: {name}")
        print(f"{'='*60}")

        ds = load_from_disk(eval_path)
        extractor = EVAL_EXTRACTORS[name]
        contaminated = detect_contamination(
            train_questions, ds, extractor, args.ngram_size, args.threshold,
            enable_edit_distance=args.enable_edit_distance,
            edit_distance_threshold=args.edit_distance_threshold,
            edit_distance_candidates=args.edit_distance_candidates,
        )

        total = len(ds)
        n_contaminated = len(contaminated)
        print(f"  Total samples: {total}")
        print(f"  Contaminated:  {n_contaminated} ({100*n_contaminated/total:.1f}%)")

        if contaminated:
            exact_count = sum(1 for _, _, t in contaminated if t == "exact")
            ngram_count = sum(1 for _, _, t in contaminated if t == "ngram")
            edit_count = sum(1 for _, _, t in contaminated if t == "edit_distance")
            print(f"    - Exact match: {exact_count}")
            print(f"    - N-gram overlap: {ngram_count}")
            if edit_count:
                print(f"    - Edit distance: {edit_count}")
            # Show top-5 worst offenders
            contaminated.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top contaminated samples (by overlap ratio):")
            for idx, ratio, mtype in contaminated[:5]:
                text = extractor(ds[idx])[:80]
                print(f"    [{idx}] ratio={ratio:.3f} type={mtype} | {text}...")

        # Save clean dataset
        contaminated_indices = set(idx for idx, _, _ in contaminated)
        clean_indices = [i for i in range(total) if i not in contaminated_indices]
        clean_ds = ds.select(clean_indices)

        clean_path = os.path.join(args.output_dir, name)
        clean_ds.save_to_disk(clean_path)
        print(f"  Clean dataset saved: {clean_path} ({len(clean_ds)} samples)")

        report["results"][name] = {
            "total": total,
            "contaminated": n_contaminated,
            "exact_match": sum(1 for _, _, t in contaminated if t == "exact"),
            "ngram_overlap": sum(1 for _, _, t in contaminated if t == "ngram"),
            "edit_distance": sum(1 for _, _, t in contaminated if t == "edit_distance"),
            "contaminated_indices": [idx for idx, _, _ in contaminated],
            "details": [{"index": idx, "ratio": ratio, "type": mtype} for idx, ratio, mtype in contaminated],
        }

    # Save report
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nContamination report saved to: {args.report_path}")


if __name__ == "__main__":
    main()
