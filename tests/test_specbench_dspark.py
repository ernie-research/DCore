import json
import subprocess
import sys
import threading
import time
import types
from unittest.mock import patch

import pytest

from specbench.config import BenchConfig

sglang_utils = types.ModuleType("sglang.srt.utils")
sglang_utils.kill_process_tree = lambda pid: None
sys.modules.setdefault("sglang", types.ModuleType("sglang"))
sys.modules.setdefault("sglang.srt", types.ModuleType("sglang.srt"))
sys.modules["sglang.srt.utils"] = sglang_utils

from specbench.harness.sglang import (
    SGLangHarness,
    _launch_replicas,
    _popen_launch_server,
    _replica_env,
    _run_requests,
)
from specbench.methods import get_method
from specbench.runmeta import update_manifest


def _config(tmp_path, **overrides):
    values = {
        "target_model": "target",
        "draft_model": "draft",
        "datasets": ["gsm8k:2"],
        "output_dir": str(tmp_path),
    }
    values.update(overrides)
    return BenchConfig(**values)


def test_dspark_registry_and_harness_check():
    method = get_method("dspark")

    assert method.family == "dflash"
    assert method.sglang_supported is True
    assert method.sglang_speculative_algorithm == "DSPARK"
    assert method.sglang_spec_params == {}
    assert method.sglang_env == {"SGLANG_RAGGED_VERIFY_MODE": "static"}
    method.check_harness("sglang")


@pytest.mark.parametrize(
    ("method_name", "algorithm"),
    [("eagle3_origin", "EAGLE3"), ("dflash_origin", "DFLASH")],
)
def test_origin_method_registry(method_name, algorithm):
    method = get_method(method_name)

    assert method.sglang_supported is True
    assert method.sglang_speculative_algorithm == algorithm
    method.check_harness("sglang")


def test_dspark_server_args_infer_checkpoint_block_size(tmp_path):
    cfg = _config(tmp_path)
    args = SGLangHarness()._server_args(get_method("dspark"), cfg)

    assert args[args.index("--speculative-algorithm") + 1] == "DSPARK"
    assert args[args.index("--speculative-draft-model-path") + 1] == "draft"
    assert "--speculative-num-draft-tokens" not in args
    assert not any("block-size" in arg for arg in args)


def test_dspark_server_args_forward_generic_draft_token_override(tmp_path):
    cfg = _config(tmp_path, spec_num_draft_tokens=16)
    args = SGLangHarness()._server_args(get_method("dspark"), cfg)

    index = args.index("--speculative-num-draft-tokens")
    assert args[index + 1] == "16"
    assert not any("block-size" in arg for arg in args)


def test_dspark_server_args_forward_draft_temperature(tmp_path):
    cfg = _config(
        tmp_path,
        speculative_draft_temperature=0.0,
        speculative_use_rejection_sampling=True,
    )
    args = SGLangHarness()._server_args(get_method("dspark"), cfg)

    index = args.index("--speculative-draft-temperature")
    assert args[index + 1] == "0.0"
    assert "--speculative-use-rejection-sampling" not in args


def test_method_environment_merges_with_replica_gpu():
    assert _replica_env(get_method("dspark"), "3") == {
        "SGLANG_RAGGED_VERIFY_MODE": "static",
        "CUDA_VISIBLE_DEVICES": "3",
    }
    assert _replica_env(get_method("dflash"), "3") == {
        "CUDA_VISIBLE_DEVICES": "3"
    }
    assert _replica_env(get_method("dflash")) is None


def test_concurrency_is_applied_per_replica(monkeypatch):
    lock = threading.Lock()
    active = 0
    max_active = 0
    active_by_url = {"http://gpu0": 0, "http://gpu1": 0}
    max_active_by_url = {"http://gpu0": 0, "http://gpu1": 0}
    request_count_by_url = {"http://gpu0": 0, "http://gpu1": 0}

    def fake_timed_send(base_url, prompt, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            active_by_url[base_url] += 1
            max_active_by_url[base_url] = max(
                max_active_by_url[base_url], active_by_url[base_url]
            )
            request_count_by_url[base_url] += 1
        time.sleep(0.05 if prompt == "p0" else 0.01)
        with lock:
            active -= 1
            active_by_url[base_url] -= 1
        return 0.05, {
            "text": prompt,
            "meta_info": {"completion_tokens": 1, "e2e_latency": 0.05},
        }

    monkeypatch.setattr("specbench.harness.sglang._timed_send", fake_timed_send)
    _run_requests(
        ["http://gpu0", "http://gpu1"],
        prompts=["p0", "p1", "p2", "p3"],
        max_new_tokens=1,
        concurrency=1,
        timeout_s=30,
        expect_spec=False,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        warmup_prompts=[],
        warmup_max_new_tokens=0,
        skip_prompts=[],
    )

    assert max_active == 2
    assert max_active_by_url == {"http://gpu0": 1, "http://gpu1": 1}
    assert request_count_by_url == {"http://gpu0": 2, "http://gpu1": 2}


def test_replicas_launch_in_parallel_with_separate_logs(tmp_path, monkeypatch):
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    active = 0
    max_active = 0
    calls = []

    class FakeProcess:
        pass

    def fake_launch(model, host, port_base, other_args, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            calls.append((port_base, other_args, kwargs["env_overrides"], kwargs["log_path"]))
        barrier.wait(timeout=1)
        with lock:
            active -= 1
        return FakeProcess(), f"http://{host}:{port_base}"

    monkeypatch.setattr("specbench.harness.sglang._launch_with_retry", fake_launch)
    cfg = _config(tmp_path, run_dir=str(tmp_path / "run"))
    launched = _launch_replicas(
        get_method("dspark"), cfg, ["--arg"], n_rep=2, gpus=["0", "1"]
    )

    assert len(launched) == 2
    assert max_active == 2
    assert sorted(call[0] for call in calls) == [cfg.port_base, cfg.port_base + 2000]
    nccl_ports = {
        int(call[1][call[1].index("--nccl-port") + 1])
        for call in calls
    }
    assert len(nccl_ports) == 2
    assert {call[2]["CUDA_VISIBLE_DEVICES"] for call in calls} == {"0", "1"}
    log_paths = {call[3] for call in calls}
    assert len(log_paths) == 2
    assert all(path.parent == tmp_path / "run" / "logs" for path in log_paths)


def test_server_output_is_redirected_to_log(tmp_path, monkeypatch):
    captured = {}

    class FakeProcess:
        returncode = None

        def poll(self):
            return None

    class HealthyResponse:
        status_code = 200

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        captured["log_name"] = kwargs["stdout"].name
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("specbench.harness.sglang.requests.get", lambda *args, **kwargs: HealthyResponse())
    log_path = tmp_path / "logs" / "gpu0.log"

    _popen_launch_server(
        "target", "http://127.0.0.1:20001", [], {"CUDA_VISIBLE_DEVICES": "0"}, log_path
    )

    assert captured["stderr"] == subprocess.STDOUT
    assert captured["log_name"] == str(log_path)
    assert "sglang serve" in log_path.read_text(encoding="utf-8")


def test_pooled_acceptance_retains_per_request_values(monkeypatch):
    responses = iter([
        {"text": "a", "meta_info": {
            "completion_tokens": 6, "spec_verify_ct": 2,
            "spec_accept_length": 99.0, "e2e_latency": 1.0,
        }},
        {"text": "b", "meta_info": {
            "completion_tokens": 10, "spec_verify_ct": 2,
            "spec_accept_length": 1.0, "e2e_latency": 1.0,
        }},
    ])

    def fake_timed_send(*args, **kwargs):
        return 1.0, next(responses)

    monkeypatch.setattr("specbench.harness.sglang._timed_send", fake_timed_send)
    metrics, samples = _run_requests(
        ["http://server"], prompts=["p0", "p1"], max_new_tokens=16,
        concurrency=1, timeout_s=30, expect_spec=True, temperature=0.0,
        top_p=1.0, top_k=1, warmup_prompts=[], warmup_max_new_tokens=0,
        skip_prompts=[],
    )

    assert metrics.spec_verify_ct_sum == 4
    assert metrics.spec_accept_length == pytest.approx(4.0)
    assert [sample["spec_accept_length"] for sample in samples] == [99.0, 1.0]


@pytest.mark.parametrize(
    ("head_type", "block_size"),
    [("vanilla", 7), ("vanilla", 15), ("rnn", 7), ("rnn", 15)],
)
def test_manifest_extracts_dspark_checkpoint_metadata(tmp_path, head_type, block_size):
    checkpoint = tmp_path / f"{head_type}-block{block_size}"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "architectures": ["DSparkDraftModel"],
        "block_size": block_size,
        "dspark_config": {"markov_head_type": head_type},
    }), encoding="utf-8")
    manifest = tmp_path / "run" / "manifest.json"
    cfg = _config(tmp_path, draft_model=str(checkpoint), run_dir=str(manifest.parent))

    with patch("specbench.runmeta.collect_env", return_value={}):
        update_manifest(
            manifest, cfg=cfg, method=get_method("dspark"), harness="sglang"
        )

    info = json.loads(manifest.read_text(encoding="utf-8"))["methods"]["dspark"]
    assert info["architecture"] == "DSparkDraftModel"
    assert info["block_size"] == block_size
    assert info["markov_head_type"] == head_type
    assert info["ragged_verify_mode"] == "static"
    assert info["method_env"] == {"SGLANG_RAGGED_VERIFY_MODE": "static"}
    assert info["draft_model"] == str(checkpoint)
