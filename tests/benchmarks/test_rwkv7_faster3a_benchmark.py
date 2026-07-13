# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from benchmarks.rwkv7 import benchmark_faster3a as bench


def _config(repo_root: Path, model: str | None = "https://example.test/model") -> bench.BenchmarkConfig:
    return bench.BenchmarkConfig(
        repo_root=repo_root,
        model=model,
        batch_size=16,
        prompt_len=128,
        warmup_tokens=16,
        decode_tokens=128,
    )


def _measurement(tokens_per_s: float = 10000.0, *, wkv_mode: str = "fp16") -> dict[str, Any]:
    env = dict(bench.RUNNER_FP16_THROUGHPUT_REQUIREMENTS)
    env["VLLM_RWKV7_WKV_MODE"] = wkv_mode
    return {
        "runner_steady_decode": {"runner_tokens_per_s": tokens_per_s},
        "config": {"provenance": {"env": env, "raw_env": env}},
    }


def test_git_revision_reads_source_marker(tmp_path: Path) -> None:
    (tmp_path / ".helicopter-source-revision").write_text(
        "abc123-dirty\n", encoding="utf-8"
    )

    assert bench._git_revision(tmp_path) == "abc123-dirty"


def test_environment_metadata_resolves_defaults(monkeypatch) -> None:
    for name in bench.PROVENANCE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    resolved = bench._rwkv_environment_metadata()

    assert resolved["VLLM_RWKV7_WKV_MODE"] == "fp16"
    assert resolved["VLLM_USE_RAPID_SAMPLER"] == "1"
    assert resolved["VLLM_RWKV7_MODEL"] is None


def test_environment_metadata_preserves_explicit_values(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp32io16")
    monkeypatch.setenv("VLLM_RWKV7_MODEL", "/model.pth")

    resolved = bench._rwkv_environment_metadata()

    assert resolved["VLLM_RWKV7_WKV_MODE"] == "fp32io16"
    assert resolved["VLLM_RWKV7_MODEL"] == "/model.pth"


def test_report_records_source_and_runner_check() -> None:
    config = _config(Path.cwd())
    report = bench.build_report(
        config,
        measurements=_measurement(),
        cuda_available=True,
    )

    assert report["overall_status"] == "passed"
    assert set(report["checks"]) == {"runner_steady_decode"}
    assert report["source"]["contracts"]
    assert report["checks"]["runner_steady_decode"]["metrics"]["runner_tokens_per_s"] == 10000.0


@pytest.mark.parametrize(
    ("measurement", "status", "code"),
    [
        (None, "blocked", "missing_measurement_json"),
        (_measurement(wkv_mode="fp32io16"), "blocked", "invalid_runner_throughput_contract"),
        (_measurement(tokens_per_s=0.0), "failed", None),
    ],
)
def test_report_rejects_missing_or_invalid_runner_measurement(
    measurement: dict[str, Any] | None,
    status: str,
    code: str | None,
) -> None:
    report = bench.build_report(
        _config(Path.cwd()),
        measurements=measurement,
        cuda_available=True,
    )
    check = report["checks"]["runner_steady_decode"]

    assert report["overall_status"] == status
    if code is not None:
        assert check["blockers"][0]["code"] == code


def test_report_blocks_missing_runtime_paths() -> None:
    report = bench.build_report(
        _config(Path.cwd(), model=None),
        cuda_available=False,
    )

    assert report["overall_status"] == "blocked"
    assert {item["code"] for item in report["checks"]["runner_steady_decode"]["blockers"]} >= {
        "cuda_unavailable",
        "missing_vllm_model",
    }


def test_cli_writes_runner_measurement_json(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "runner.json"
    calls: list[tuple[Any, ...]] = []

    def fake_generate(config, **kwargs):
        calls.append((config, kwargs))
        return {**_measurement(), "benchmark": bench.BENCHMARK_NAME}

    monkeypatch.setattr(bench, "generate_vllm_runner_measurement", fake_generate)
    rc = bench.main(
        [
            "--repo-root",
            str(tmp_path),
            "--model",
            "https://example.test/model",
            "--measure-vllm-runner",
            "--runner-batch-size",
            "32",
            "--runner-prompt-len",
            "64",
            "--runner-prefill-chunk-tokens",
            "16",
            "--runner-decode-tokens",
            "1280",
            "--runner-warmup",
            "1",
            "--runner-iters",
            "3",
            "--measurement-output",
            str(output),
        ]
    )

    assert rc == 0
    assert json.loads(output.read_text())["benchmark"] == bench.BENCHMARK_NAME
    assert calls[0][1] == {
        "batch_size": 32,
        "prompt_len": 64,
        "decode_tokens": 1280,
        "warmup": 1,
        "iters": 3,
    }


@pytest.mark.parametrize(
    ("measurement", "expected_rc"),
    [
        ({"runner_steady_decode": {"blockers": [{}]}}, 2),
        (_measurement(tokens_per_s=1.0), 1),
        (_measurement(wkv_mode="fp32io16"), 2),
        (
            _measurement(tokens_per_s=bench.RUNNER_BASELINE_TOKENS_PER_S * 0.995),
            0,
        ),
    ],
)
def test_measurement_lane_classifies_result(
    tmp_path: Path,
    monkeypatch,
    measurement: dict[str, Any],
    expected_rc: int,
) -> None:
    output = tmp_path / "runner.json"
    monkeypatch.setattr(bench, "generate_vllm_runner_measurement", lambda *a, **k: measurement)

    rc = bench.main(
        [
            "--repo-root",
            str(tmp_path),
            "--model",
            "https://example.test/model",
            "--measure-vllm-runner",
            "--measurement-output",
            str(output),
        ]
    )

    assert rc == expected_rc


def test_cli_writes_structured_blocked_json(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "report.json"
    monkeypatch.setattr(bench, "_cuda_available", lambda: False)
    monkeypatch.setattr(bench, "_cuda_device_metadata", lambda: {"available": False})

    rc = bench.main(
        [
            "--repo-root",
            str(tmp_path),
            "--model",
            str(tmp_path / "missing-model"),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    assert rc == 2
    assert report["overall_status"] == "blocked"
    assert report["checks"]["runner_steady_decode"]["blockers"]


def test_create_runner_llm_preserves_env_and_capture_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"")
    captured: dict[str, Any] = {}

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            captured["model_env"] = os.environ.get("VLLM_RWKV7_MODEL")

    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.rwkv7_ops", ModuleType("vllm.rwkv7_ops"))
    monkeypatch.setenv("VLLM_RWKV7_MODEL", str(model_path))

    config = replace(
        _config(tmp_path, model=str(model_path)),
        batch_size=1024,
        prompt_len=1,
        decode_tokens=4,
        runner_prefill_chunk_tokens=1,
        runner_cudagraph_capture_sizes=(1024,),
    )
    bench._create_vllm_runner_llm(config)

    assert captured["max_num_seqs"] == 1024
    assert captured["max_num_batched_tokens"] == 1024
    assert captured["compilation_config"] == {"cudagraph_capture_sizes": [1024]}
    assert captured["model_env"] is None
    assert os.environ["VLLM_RWKV7_MODEL"] == str(model_path)


def test_create_runner_llm_omits_capture_for_eager_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"")
    captured: dict[str, Any] = {}

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.rwkv7_ops", ModuleType("vllm.rwkv7_ops"))

    bench._create_vllm_runner_llm(
        replace(
            _config(tmp_path, model=str(model_path)),
            runner_enforce_eager=True,
            runner_cudagraph_capture_sizes=(1024,),
        )
    )

    assert "compilation_config" not in captured


def test_shutdown_runner_uses_engine_core(monkeypatch) -> None:
    calls: list[int] = []
    engine_core = SimpleNamespace(shutdown=lambda **kwargs: calls.append(kwargs["timeout"]))
    llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=engine_core))
    monkeypatch.setattr(bench, "_cuda_available", lambda: False)

    bench._shutdown_vllm_runner_llm(llm)

    assert calls == [30]


def test_finish_execute_without_sampling_postprocesses_state() -> None:
    calls: list[tuple[Any, ...]] = []
    input_batch = SimpleNamespace(idx_mapping="mapping")
    model_state = SimpleNamespace(
        postprocess_state=lambda *args: calls.append(args),
    )
    model_runner = SimpleNamespace(
        execute_model_state=SimpleNamespace(input_batch=input_batch),
        model_state=model_state,
        req_states=SimpleNamespace(
            num_computed_tokens=SimpleNamespace(gpu="computed")
        ),
    )
    worker = SimpleNamespace(model_runner=model_runner)

    bench._worker_finish_execute_without_sampling(worker)

    assert calls == [("mapping", 0, "computed")]
    assert model_runner.execute_model_state is None


def test_phase_summary_reports_average_and_peak() -> None:
    summary = bench._phase_throughput_summary(
        total_tokens=40,
        iteration_durations_s=[0.010, 0.030],
        unit_durations_s=[0.010, 0.020, 0.010],
        unit_tokens=[10, 10, 20],
    )

    assert summary["avg_tokens_per_s"] == pytest.approx(1000.0)
    assert summary["peak_iteration_tokens_per_s"] == pytest.approx(2000.0)
    assert summary["peak_unit_tokens_per_s"] == pytest.approx(2000.0)
    assert summary["p50_ms"] == pytest.approx(10.0)


def test_internal_runner_merge_reports_component_timings() -> None:
    result = bench._merge_worker_internal_runner_results(
        [
            {
                "iteration_durations_s": [0.012],
                "execute_durations_s": [0.003, 0.004],
                "sample_durations_s": [0.001, 0.002],
                "decode_step_durations_s": [0.004, 0.006],
                "decode_steps": 2,
                "timing_clock": "cuda_event",
            }
        ],
        batch_size=2,
        decode_tokens=2,
        iters=1,
    )

    assert result["tokens_per_s"] == pytest.approx(333.333333)
    assert result["execute_model_p50_ms"] == pytest.approx(3.0)
    assert result["sample_tokens_p50_ms"] == pytest.approx(1.0)
    assert result["internal_timing_target"] == bench.VLLM_RUNNER_TIMING_TARGET
