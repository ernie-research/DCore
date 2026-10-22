"""Dataset loading + prompt formatting.

Ported verbatim (behavior-preserving) from the original
``third_party/Domino/code/model/utils.py::load_and_process_dataset`` so the
new module has no dependency on ``third_party/Domino``. Only the local-cache
and swe-bench paths were re-anchored to the repo root.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Features, Sequence, Value, load_dataset, load_from_disk

from specbench.config import REPO_ROOT

# Published copy of the frozen SpecMix set (see scripts/build_specmix_dataset.py).
SPECMIX_HF_REPO = "wingchiuloong/specmix"


def load_and_process_dataset(data_name: str):
    """Return a HF dataset where each row has a ``turns`` list of user prompts."""
    local_cache = REPO_ROOT / "cache" / "dataset" / data_name
    _ds = load_from_disk(str(local_cache)) if local_cache.exists() else None

    if data_name == "gsm8k":
        dataset = _ds if _ds is not None else load_dataset("openai/gsm8k", "main", split="test")
        fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [fmt.format(**x)]})

    elif data_name == "math500":
        dataset = _ds if _ds is not None else load_dataset("HuggingFaceH4/MATH-500", split="test")
        fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [fmt.format(**x)]})

    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [fmt.format(**x)]})

    elif data_name == "aime25":
        dataset = _ds if _ds is not None else load_dataset("MathArena/aime_2025", split="train")
        fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [fmt.format(**x)]})

    elif data_name == "alpaca":
        dataset = _ds if _ds is not None else load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(lambda x: {"formatted_input": (f"{x['instruction']}\n\nInput:\n{x['input']}" if x['input'] else x['instruction'])})
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        dataset = _ds if _ds is not None else load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    elif data_name == "humaneval":
        dataset = _ds if _ds is not None else load_dataset("openai/openai_humaneval", split="test")
        fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
        dataset = dataset.map(lambda x: {"turns": [fmt.format(**x)]})

    elif data_name == "mbpp":
        dataset = _ds if _ds is not None else load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})

    elif data_name == "lbpp":
        url = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        dataset = load_dataset("parquet", data_files={"test": url})["test"]
        dataset = dataset.map(lambda x: {"turns": [x["instruction"]]})

    elif data_name == "swe-bench":
        jsonl_path = REPO_ROOT / "third_party" / "DeepSpec" / "eval_datasets" / "swe-bench.jsonl"
        if _ds is not None:
            dataset = _ds
        elif jsonl_path.exists():
            dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
        else:
            dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
            fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
            dataset = dataset.map(lambda x: {"turns": [fmt.format(**x)]})
        if "turns" not in dataset.column_names:
            fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
            dataset = dataset.map(lambda x: {"turns": [fmt.format(**x)]})

    elif data_name == "livecodebench":
        if _ds is not None:
            dataset = _ds
        else:
            base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
            files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
            dataset = load_dataset("json", data_files={"test": [base + fn for fn in files]})["test"]

        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"

        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features,
        )

    elif data_name == "specmix":
        # Frozen mixed-traffic set (published on HF, rebuildable with
        # scripts/build_specmix_dataset.py): prompts are already formatted and
        # shuffled, so pass through as-is. The first load from HF is cached to
        # ``local_cache`` so later (offline) runs read it from disk.
        if _ds is None:
            try:
                _ds = load_dataset(SPECMIX_HF_REPO, split="train")
            except Exception as e:
                raise FileNotFoundError(
                    f"{local_cache} not found and downloading {SPECMIX_HF_REPO} "
                    "failed (set HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0, or "
                    "rebuild it with `python scripts/build_specmix_dataset.py`)"
                ) from e
            _ds.save_to_disk(str(local_cache))
        dataset = _ds

    elif data_name == "arena-hard":
        dataset = _ds if _ds is not None else load_dataset("lmarena-ai/arena-hard-auto-v0.1", split="train")
        dataset = dataset.map(lambda x: {"turns": [x["turns"][0]["content"]]})

    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    return dataset
