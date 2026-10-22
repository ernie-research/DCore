"""Accuracy evaluation for specbench responses.

Scores model responses against gold answers for scorable datasets
(gsm8k, math500, aime25). For mixed-traffic sets (``specmix``) the rows carry
their own ``source``/``gold``, so scoring is reported per originating dataset.
Reuses extraction logic from benchmarks/benchmarker/
but inlined here to avoid cross-package dependencies.

Usage:
    python -m specbench.accuracy --run-dir <path>
    python -m specbench.cli accuracy --run-dir <path>
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from specbench.datasets import load_and_process_dataset

SCORABLE_DATASETS = {"gsm8k", "math500", "aime25"}
# Mixed-traffic sets (published on HF as wingchiuloong/specmix; rebuildable with
# scripts/build_specmix_dataset.py): a single
# dataset whose rows carry ``source``/``gold``, so scoring is done per source.
MIXED_DATASETS = {"specmix"}


# ---------------------------------------------------------------------------
# Gold answer extraction (from dataset rows)
# ---------------------------------------------------------------------------

def extract_gold(dataset_name: str, row: dict) -> Optional[str]:
    """Extract gold answer from a dataset row."""
    if dataset_name == "gsm8k":
        # row["answer"] contains "...\n#### <number>"
        answer = row.get("answer", "")
        if "####" in answer:
            return answer.split("####")[-1].strip().replace(",", "")
        return None
    elif dataset_name == "math500":
        # row["answer"] is the final answer string directly
        return str(row["answer"]).strip() if row.get("answer") is not None else None
    elif dataset_name == "aime25":
        # row["answer"] is the integer as string
        return str(row["answer"]).strip() if row.get("answer") is not None else None
    return None


# ---------------------------------------------------------------------------
# Prediction extraction (from model response text)
# ---------------------------------------------------------------------------

def _extract_gsm8k_prediction(response: str) -> Optional[int]:
    """Extract last integer from response (mirrors get_answer_value)."""
    response = response.replace(",", "")
    numbers = re.findall(r"\d+", response)
    if not numbers:
        return None
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return None


def _extract_math500_prediction(response: str) -> Optional[str]:
    """Extract answer from math response (mirrors extract_math_answer)."""
    # \boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        return match.group(1).strip()
    # \boxed X
    match = re.search(r"\\boxed\s+([^\s]+)", response)
    if match:
        return match.group(1).strip()
    # "answer/Answer/ANSWER" followed by number
    for pattern in [
        r"(?:answer|Answer|ANSWER)[\s:]+([-+]?\d*\.?\d+)",
        r"(?:is|equals?|=\s*)([-+]?\d*\.?\d+)\s*$",
    ]:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    # Fallback: last number
    numbers = re.findall(r"[-+]?\d*\.?\d+", response)
    if numbers:
        return numbers[-1]
    return None


def _extract_aime_prediction(response: str) -> Optional[str]:
    """Extract AIME answer (integer 0-999) from response (mirrors extract_aime_answer)."""
    # \boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        answer = match.group(1).strip()
        numbers = re.findall(r"\d+", answer)
        if numbers:
            return numbers[-1]
        return answer
    # \boxed N
    match = re.search(r"\\boxed\s+(\d+)", response)
    if match:
        return match.group(1).strip()
    # Textual patterns
    for pattern in [
        r"(?:answer|Answer|ANSWER)[\s:]+(\d+)",
        r"(?:final\s+answer|Final\s+Answer)[\s:]+(\d+)",
        r"(?:is|equals?|=\s*)(\d+)\s*$",
    ]:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    # Fallback: last integer in [0, 999]
    numbers = re.findall(r"\b(\d+)\b", response)
    if numbers:
        valid = [n for n in numbers if 0 <= int(n) <= 999]
        if valid:
            return valid[-1]
    return None


def extract_prediction(dataset_name: str, response: str):
    """Extract predicted answer from model response text."""
    if dataset_name == "gsm8k":
        return _extract_gsm8k_prediction(response)
    elif dataset_name == "math500":
        return _extract_math500_prediction(response)
    elif dataset_name == "aime25":
        return _extract_aime_prediction(response)
    return None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(dataset_name: str, pred, gold: str) -> bool:
    """Compare prediction against gold answer."""
    if pred is None or gold is None:
        return False
    if dataset_name == "gsm8k":
        # Integer equality
        try:
            return int(pred) == int(gold)
        except (ValueError, TypeError):
            return False
    elif dataset_name == "math500":
        # String match or float match within 1e-6
        pred_s = str(pred).strip().lower()
        gold_s = str(gold).strip().lower()
        if pred_s == gold_s:
            return True
        try:
            return abs(float(pred_s) - float(gold_s)) < 1e-6
        except (ValueError, TypeError):
            return False
    elif dataset_name == "aime25":
        # Integer equality
        try:
            return int(pred) == int(gold)
        except (ValueError, TypeError):
            return False
    return False


# ---------------------------------------------------------------------------
# Main scoring logic
# ---------------------------------------------------------------------------

def _read_by_method(resp_files: list[Path]) -> dict[str, list[dict]]:
    method_responses: dict[str, list[dict]] = defaultdict(list)
    for resp_file in resp_files:
        with open(resp_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                method_responses[rec.get("method", "unknown")].append(rec)
    return method_responses


def _score_mixed(ds_name: str, resp_files: list[Path]) -> list[dict]:
    """Score a mixed-traffic set per originating source dataset.

    Rows whose ``source`` is not scorable (or that carry no gold answer) are
    ignored, so only the math subsets contribute. Reported dataset names are
    ``"<mixed set>:<source>"``.
    """
    dataset = load_and_process_dataset(ds_name)
    meta = {}
    for idx, row in enumerate(dataset):
        if row["source"] in SCORABLE_DATASETS and row["gold"]:
            meta[idx] = (row["source"], row["gold"])

    results: list[dict] = []
    for method, responses in sorted(_read_by_method(resp_files).items()):
        tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # source -> [correct, total]
        draft_model = None
        for rec in responses:
            if draft_model is None:
                draft_model = rec.get("draft_model")
            entry = meta.get(rec.get("sample_index"))
            if entry is None:
                continue
            source, gold = entry
            pred = extract_prediction(source, rec.get("response", ""))
            counts = tally[source]
            counts[1] += 1
            if compare(source, pred, gold):
                counts[0] += 1
        for source in sorted(tally):
            correct, total = tally[source]
            if total == 0:
                continue
            accuracy = correct / total
            results.append({
                "method": method,
                "dataset": f"{ds_name}:{source}",
                "accuracy": round(accuracy, 4),
                "correct": correct,
                "total": total,
                "draft_model": draft_model,
            })
            print(f"  {method:20s} | {ds_name}:{source:8s} | "
                  f"{correct}/{total} = {accuracy:.2%}")
    return results


def score_responses(run_dir: str, datasets: Optional[list[str]] = None) -> list[dict]:
    """Score responses in run_dir against gold answers.

    Returns list of result records (also written to accuracy.jsonl).
    """
    run_path = Path(run_dir)
    responses_dir = run_path / "responses"
    if not responses_dir.exists():
        print(f"[accuracy] No responses/ directory in {run_dir}")
        return []

    target_datasets = datasets if datasets else sorted(SCORABLE_DATASETS | MIXED_DATASETS)
    results = []

    for ds_name in sorted(target_datasets):
        if ds_name not in SCORABLE_DATASETS and ds_name not in MIXED_DATASETS:
            continue

        # New layout: responses/<dataset>/<method>.jsonl
        # Backward compat: responses/<dataset>.jsonl (flat file)
        ds_dir = responses_dir / ds_name
        if ds_dir.is_dir():
            resp_files = sorted(ds_dir.glob("*.jsonl"))
        else:
            flat = responses_dir / f"{ds_name}.jsonl"
            resp_files = [flat] if flat.exists() else []

        if not resp_files:
            continue

        if ds_name in MIXED_DATASETS:
            results.extend(_score_mixed(ds_name, resp_files))
            continue

        # Load gold answers
        dataset = load_and_process_dataset(ds_name)
        gold_answers = {}
        for idx, row in enumerate(dataset):
            gold = extract_gold(ds_name, row)
            if gold is not None:
                gold_answers[idx] = gold

        # Read responses and group by method
        method_responses = _read_by_method(resp_files)

        # Score each method
        for method, responses in sorted(method_responses.items()):
            correct = 0
            total = 0
            draft_model = None
            for rec in responses:
                idx = rec.get("sample_index")
                response_text = rec.get("response", "")
                if draft_model is None:
                    draft_model = rec.get("draft_model")
                if idx is None or idx not in gold_answers:
                    continue
                gold = gold_answers[idx]
                pred = extract_prediction(ds_name, response_text)
                total += 1
                if compare(ds_name, pred, gold):
                    correct += 1

            if total == 0:
                continue
            accuracy = correct / total
            record = {
                "method": method,
                "dataset": ds_name,
                "accuracy": round(accuracy, 4),
                "correct": correct,
                "total": total,
                "draft_model": draft_model,
            }
            results.append(record)
            print(f"  {method:20s} | {ds_name:8s} | {correct}/{total} = {accuracy:.2%}")

    # Write results
    if results:
        out_file = run_path / "accuracy.jsonl"
        with open(out_file, "w") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[accuracy] Wrote {len(results)} records to {out_file}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="specbench.accuracy",
        description="Score specbench responses against gold answers",
    )
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset names (default: all scorable)")
    args = parser.parse_args()
    ds = args.datasets.split(",") if args.datasets else None
    score_responses(args.run_dir, datasets=ds)


if __name__ == "__main__":
    main()
