import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[2] / "scripts" / "extract_target_entropy.py"
SPEC = importlib.util.spec_from_file_location("extract_target_entropy", SCRIPT)
extractor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(extractor)


def test_normalize_conversation_removes_all_assistant_turns_and_preserves_metadata():
    row = {
        "id": "sample-1",
        "source": "unit",
        "conversations": [
            {"role": "system", "content": "context"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "source answer"},
            {"role": "human", "content": "follow-up"},
            {"role": "gpt", "content": "another answer"},
        ],
    }

    result = extractor.normalize_conversation(row)

    assert result == [
        {"role": "system", "content": "context"},
        {"role": "user", "content": "question"},
        {"role": "user", "content": "follow-up"},
    ]


def test_entropy_uses_full_distribution_in_float32():
    logits = torch.tensor([0.0, 0.0])

    entropy, log_probs, probs = extractor.entropy_from_logits(logits)

    assert entropy.item() == pytest.approx(0.69314718, rel=1e-6)
    assert probs.dtype == torch.float32
    assert log_probs.exp().sum().item() == pytest.approx(1.0)


def test_aggregate_entropy_statistics():
    result = extractor.aggregate_entropy([1.0, 2.0, 4.0])

    assert result == {
        "H_mean": pytest.approx(7 / 3),
        "H_median": 2.0,
        "H_min": 1.0,
        "H_max": 4.0,
        "H_std": pytest.approx((14 / 9) ** 0.5),
        "token_count": 3,
    }


def test_aggregate_entropy_empty():
    assert extractor.aggregate_entropy([])["token_count"] == 0
    assert extractor.aggregate_entropy([])["H_mean"] == 0.0


def test_partition_rows_applies_global_max_samples_before_modulo(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("\n".join('{"id": %d}' % index for index in range(7)) + "\n")
    rows = (
        (source_index, row)
        for source_index, row in extractor.iter_jsonl_rows(input_path)
        if source_index < 5
    )

    selected = list(extractor.partition_rows(rows, rank=1, world_size=2))

    assert [source_index for source_index, _ in selected] == [1, 3]


def test_rank_path_preserves_single_rank_paths_and_inserts_rank_suffix():
    output = Path("results/entropy.jsonl")
    manifest = Path("results/entropy.manifest.json")

    assert extractor.rank_path(output, rank=0, world_size=1) == output
    assert extractor.rank_path(manifest, rank=2, world_size=8) == Path(
        "results/entropy.manifest.rank2of8.json"
    )
    assert extractor.rank_path(output, rank=2, world_size=8) == Path(
        "results/entropy.rank2of8.jsonl"
    )
