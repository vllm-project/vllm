# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks.rwkv7 import benchmark_faster3a as bench


def _config(
    repo_root: Path,
    *,
    model: str | None = None,
    albatross_root: Path | None = None,
    albatross_checkpoint: Path | None = None,
) -> bench.BenchmarkConfig:
    return bench.BenchmarkConfig(
        repo_root=repo_root,
        model=model,
        albatross_root=albatross_root,
        albatross_impl=bench.ALBATROSS_IMPL,
        albatross_checkpoint=albatross_checkpoint,
        batch_size=16,
        prompt_len=128,
        warmup_tokens=16,
        decode_tokens=128,
    )


def test_report_blocks_without_runtime_paths_and_records_provenance(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    report = bench.build_report(
        _config(
            repo_root,
            model=str(tmp_path / "missing-model"),
            albatross_root=tmp_path / "missing-albatross",
            albatross_checkpoint=tmp_path / "missing.pth",
        ),
        cuda_available=False,
    )

    assert report["overall_status"] == "blocked"
    assert report["source"]["albatross_repo"] == bench.ALBATROSS_REPO
    assert report["source"]["albatross_commit"] == bench.ALBATROSS_COMMIT
    assert report["source"]["albatross_impl"] == bench.ALBATROSS_IMPL
    assert report["source"]["albatross_path"].endswith(
        f"missing-albatross/{bench.ALBATROSS_IMPL}"
    )
    assert {item["target_path"] for item in report["source"]["contracts"]} >= {
        "vllm/model_executor/models/rwkv7.py",
        "csrc/libtorch_stable/rwkv7/rwkv7_v3a_ops.cu",
    }
    blocker_codes = {
        blocker["code"]
        for blocker in report["checks"]["model_only_steady_decode"]["blockers"]
    }
    assert blocker_codes >= {
        "cuda_unavailable",
        "missing_vllm_model_path",
        "missing_albatross_impl_path",
        "missing_albatross_checkpoint_path",
    }
    assert report["checks"]["state_movement"]["metrics"] == {
        "resident_to_decode_copies": None,
        "decode_compactions": None,
        "decode_compaction_rows": None,
    }


def test_report_evaluates_passing_measurements() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 100.0,
            "vllm_tokens_per_s": 96.0,
        },
        "runner_steady_decode": {
            "runner_tokens_per_s": 91.0,
        },
        "state_movement": {
            "resident_to_decode_copies": 0,
            "decode_compactions": 2,
            "decode_compaction_rows": 32,
        },
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=False,
    )

    assert report["overall_status"] in {"blocked", "passed"}
    assert report["checks"]["model_only_steady_decode"]["status"] == "passed"
    assert report["checks"]["runner_steady_decode"]["status"] == "passed"
    assert report["checks"]["state_movement"]["status"] == "passed"
    assert report["checks"]["state_movement"]["metrics"] == {
        "resident_to_decode_copies": 0,
        "decode_compactions": 2,
        "decode_compaction_rows": 32,
    }


def test_report_blocks_only_missing_vllm_when_albatross_measurement_exists() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 3200.0,
            "albatross_batch_size": 2,
            "albatross_seq_len": 4,
            "albatross_p50_ms": 2.5,
        },
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=True,
    )

    model_only_check = report["checks"]["model_only_steady_decode"]
    assert model_only_check["status"] == "blocked"
    assert model_only_check["metrics"]["albatross_tokens_per_s"] == 3200.0
    assert model_only_check["metrics"]["vllm_tokens_per_s"] is None
    assert model_only_check["blockers"] == [
        {
            "code": "missing_vllm_model_only_measurement",
            "message": "Measurement JSON must include vllm_tokens_per_s for "
            "model_only_steady_decode. Albatross model-only measurement is "
            "present; generate vLLM model-only metrics with "
            "--measure-vllm-model-only.",
        }
    ]


def test_cli_writes_albatross_model_only_measurement_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    albatross_root = tmp_path / "albatross"
    impl_dir = albatross_root / bench.ALBATROSS_IMPL
    impl_dir.mkdir(parents=True)
    (impl_dir / "rwkv7_fast_v3a.py").write_text("", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"")
    output_path = tmp_path / "measurement.json"
    calls: list[Any] = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        cmd = args[0]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="\n".join(
                [
                    "[rwkv7_fast_v3a] start model=/tmp/checkpoint.pth",
                    "csv_header,label,B,T,iters,p10_ms,p50_ms,p90_ms,tok_s_p50",
                    "csv,rwkv7_fast_v3a,2,4,7,1.250000,2.500000,4.000000,3200.000000",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = bench.main(
        [
            "--repo-root",
            str(Path(__file__).resolve().parents[2]),
            "--albatross-root",
            str(albatross_root),
            "--albatross-checkpoint",
            str(checkpoint_path),
            "--measure-albatross-model-only",
            "--albatross-case",
            "2x4",
            "--albatross-warmup",
            "3",
            "--albatross-iters",
            "7",
            "--measurement-output",
            str(output_path),
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    model_only = measurement["model_only_steady_decode"]
    assert rc == 0
    assert model_only["albatross_tokens_per_s"] == 3200.0
    assert model_only["albatross_batch_size"] == 2
    assert model_only["albatross_seq_len"] == 4
    assert model_only["albatross_warmup"] == 3
    assert model_only["albatross_iters"] == 7
    assert model_only["albatross_p50_ms"] == 2.5
    assert model_only["albatross_label"] == "rwkv7_fast_v3a"
    assert measurement["config"]["measurement_source"] == "albatross_subprocess"
    cmd = calls[0][0][0]
    assert cmd[:2] == [
        sys.executable,
        str(impl_dir / "rwkv7_fast_v3a.py"),
    ]
    assert "--cases" in cmd
    assert "2x4" in cmd
    assert calls[0][1]["cwd"] == impl_dir


def test_cli_writes_vllm_model_only_measurement_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    output_path = tmp_path / "measurement.json"
    calls: list[Any] = []
    fake_model = object()

    def fake_load(config):
        calls.append(("load", config.model))
        return fake_model

    def fake_time(model, *, batch_size, seq_len, warmup, iters):
        calls.append(("time", model, batch_size, seq_len, warmup, iters))
        return {
            "tokens_per_s": 3040.0,
            "p10_ms": 1.75,
            "p50_ms": 2.0,
            "p90_ms": 2.5,
            "graph": True,
            "measurement_mode": "cuda_graph_replay",
            "distributed_backend": "nccl",
        }

    monkeypatch.setattr(bench, "_load_vllm_rwkv7_model", fake_load)
    monkeypatch.setattr(
        bench,
        "_time_vllm_model_only_steady_decode",
        fake_time,
    )

    rc = bench.main(
        [
            "--repo-root",
            str(Path(__file__).resolve().parents[2]),
            "--model",
            str(model_path),
            "--measure-vllm-model-only",
            "--vllm-case",
            "2x4",
            "--vllm-warmup",
            "3",
            "--vllm-iters",
            "7",
            "--measurement-output",
            str(output_path),
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    model_only = measurement["model_only_steady_decode"]
    assert rc == 0
    assert model_only["vllm_tokens_per_s"] == 3040.0
    assert model_only["vllm_batch_size"] == 2
    assert model_only["vllm_seq_len"] == 4
    assert model_only["vllm_warmup"] == 3
    assert model_only["vllm_iters"] == 7
    assert model_only["vllm_p50_ms"] == 2.0
    assert model_only["vllm_label"] == "RWKV7ForCausalLM.forward_logits"
    assert model_only["vllm_output"] == "logits"
    assert model_only["vllm_logits_included"] is True
    assert model_only["vllm_graph"] is True
    assert model_only["vllm_measurement_mode"] == "cuda_graph_replay"
    assert model_only["vllm_distributed_backend"] == "nccl"
    assert measurement["config"]["measurement_source"] == "vllm_model_direct"
    assert calls == [
        ("load", str(model_path)),
        ("time", fake_model, 2, 4, 3, 7),
    ]


def test_vllm_model_only_timer_includes_logits_for_albatross_parity(
    monkeypatch,
) -> None:
    import torch

    import vllm.model_executor.models.rwkv7 as rwkv7

    real_arange = torch.arange

    class FakeCudaGraph:
        def replay(self) -> None:
            pass

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            self.enable_timing = enable_timing

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, other) -> float:
            return 1.0

    class FakeModel:
        vocab_size = 16
        emb_cpu = True
        _benchmark_distributed_backend = "fake"

        def __init__(self) -> None:
            self.calls: list[Any] = []

        def zero_state(self, batch_size):
            self.calls.append(("zero_state", batch_size))
            return object()

        def embed(self, tokens):
            self.calls.append(("embed", tuple(tokens.shape)))
            return tokens.float()

        def forward_from_x(self, x, state, path):
            self.calls.append(("forward_from_x", tuple(x.shape), path))
            return x + 1.0

        def compute_logits(self, hidden_states):
            self.calls.append(("compute_logits", tuple(hidden_states.shape)))
            return hidden_states

    def fake_arange(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("device", None)
        return real_arange(*args, **kwargs)

    monkeypatch.setattr(bench, "_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeCudaGraph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph: nullcontext())
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch, "arange", fake_arange)
    monkeypatch.setattr(rwkv7, "select_path", lambda batch, seq: "fake-path")

    model = FakeModel()
    parsed = bench._time_vllm_model_only_steady_decode(
        model,
        batch_size=2,
        seq_len=1,
        warmup=0,
        iters=1,
    )

    assert ("compute_logits", (2, 1)) in model.calls
    assert parsed["output"] == "logits"
    assert parsed["logits_included"] is True
    assert parsed["tokens_per_s"] == 2000.0


def test_vllm_model_loader_initializes_distributed_before_model_construction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import torch

    import vllm.model_executor.models.rwkv7 as rwkv7
    import vllm.transformers_utils.configs.rwkv7 as rwkv7_config

    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    calls: list[Any] = []
    fake_weight = object()

    def fake_init():
        calls.append("init")
        return "gloo"

    class FakeRWKV7ForCausalLM:
        def __init__(self, *, vllm_config):
            assert calls == ["init"]
            calls.append(("construct", vllm_config.model_config.hf_config))

        def load_weights(self, weights):
            calls.append(("load_weights", list(weights)))

        def eval(self):
            calls.append("eval")
            return self

    monkeypatch.setattr(bench, "_initialize_vllm_single_process_distributed", fake_init)
    monkeypatch.setattr(
        rwkv7_config,
        "build_rwkv7_config_from_pth",
        lambda path: SimpleNamespace(
            hidden_size=64,
            vocab_size=128,
            head_size=64,
            num_hidden_layers=1,
            num_attention_heads=1,
        ),
    )
    monkeypatch.setattr(rwkv7, "RWKV7ForCausalLM", FakeRWKV7ForCausalLM)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"emb.weight": fake_weight},
    )

    model = bench._load_vllm_rwkv7_model(
        _config(Path(__file__).resolve().parents[2], model=str(model_path))
    )

    assert isinstance(model, FakeRWKV7ForCausalLM)
    assert calls[0] == "init"
    assert calls[1][0] == "construct"
    assert calls[2] == ("load_weights", [("emb.weight", fake_weight)])
    assert calls[3] == "eval"


def test_vllm_single_process_distributed_init_uses_canonical_entrypoints(
    monkeypatch,
) -> None:
    import torch

    import vllm.distributed.parallel_state as parallel_state

    calls: list[Any] = []

    monkeypatch.setattr(bench, "_cuda_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        parallel_state,
        "model_parallel_is_initialized",
        lambda: False,
    )

    def fake_init_distributed_environment(**kwargs):
        calls.append(("init", kwargs))

    def fake_ensure_model_parallel_initialized(*args, **kwargs):
        calls.append(("ensure", args, kwargs))

    monkeypatch.setattr(
        parallel_state,
        "init_distributed_environment",
        fake_init_distributed_environment,
    )
    monkeypatch.setattr(
        parallel_state,
        "ensure_model_parallel_initialized",
        fake_ensure_model_parallel_initialized,
    )

    backend = bench._initialize_vllm_single_process_distributed()

    assert backend == "nccl"
    assert calls[0][0] == "init"
    assert calls[0][1]["world_size"] == 1
    assert calls[0][1]["rank"] == 0
    assert calls[0][1]["local_rank"] == 0
    assert calls[0][1]["backend"] == "nccl"
    assert calls[0][1]["distributed_init_method"].startswith("tcp://127.0.0.1:")
    assert calls[1] == ("ensure", (1, 1), {"backend": "nccl"})


def test_cli_merges_vllm_model_only_measurement_with_albatross_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    input_path = tmp_path / "albatross.json"
    output_path = tmp_path / "combined.json"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": bench.SCHEMA_VERSION,
                "benchmark": bench.BENCHMARK_NAME,
                "model_only_steady_decode": {
                    "albatross_tokens_per_s": 3200.0,
                    "albatross_batch_size": 2,
                    "albatross_seq_len": 4,
                    "albatross_p50_ms": 2.5,
                },
                "config": {"measurement_source": "albatross_subprocess"},
            }
        ),
        encoding="utf-8",
    )
    calls: list[Any] = []

    monkeypatch.setattr(bench, "_load_vllm_rwkv7_model", lambda config: object())

    def fake_time(model, *, batch_size, seq_len, warmup, iters):
        calls.append((batch_size, seq_len, warmup, iters))
        return {
            "tokens_per_s": 3040.0,
            "p10_ms": 1.75,
            "p50_ms": 2.0,
            "p90_ms": 2.5,
            "graph": True,
            "measurement_mode": "cuda_graph_replay",
            "distributed_backend": "gloo",
        }

    monkeypatch.setattr(
        bench,
        "_time_vllm_model_only_steady_decode",
        fake_time,
    )

    rc = bench.main(
        [
            "--repo-root",
            str(repo_root),
            "--model",
            str(model_path),
            "--measurement-json",
            str(input_path),
            "--measure-vllm-model-only",
            "--vllm-warmup",
            "3",
            "--vllm-iters",
            "7",
            "--measurement-output",
            str(output_path),
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    model_only = measurement["model_only_steady_decode"]
    assert rc == 0
    assert model_only["albatross_tokens_per_s"] == 3200.0
    assert model_only["vllm_tokens_per_s"] == 3040.0
    assert model_only["vllm_label"] == "RWKV7ForCausalLM.forward_logits"
    assert model_only["vllm_output"] == "logits"
    assert model_only["vllm_logits_included"] is True
    assert model_only["vllm_graph"] is True
    assert model_only["vllm_measurement_mode"] == "cuda_graph_replay"
    assert model_only["vllm_distributed_backend"] == "gloo"
    assert measurement["config"]["measurement_source"] == "merged_vllm_model_direct"
    assert calls == [(2, 4, 3, 7)]

    report = bench.build_report(
        _config(repo_root),
        measurements=measurement,
        cuda_available=True,
    )
    check = report["checks"]["model_only_steady_decode"]
    assert check["status"] == "passed"
    assert check["metrics"]["vllm_to_albatross_ratio"] == 0.95


def test_cli_writes_vllm_runner_measurement_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    output_path = tmp_path / "runner.json"
    calls: list[Any] = []
    fake_llm = object()

    def fake_create(config):
        calls.append(("create", config.model))
        return fake_llm

    def fake_time(llm, *, batch_size, prompt_len, decode_tokens, warmup, iters):
        calls.append(
            (
                "time",
                llm,
                batch_size,
                prompt_len,
                decode_tokens,
                warmup,
                iters,
            )
        )
        return {
            "tokens_per_s": 91.0,
            "p10_ms": 1.9,
            "p50_ms": 2.0,
            "p90_ms": 2.4,
            "measurement_mode": "worker_execute_model",
            "internal_timing_target": "worker.execute_model",
            "decode_steps": 7,
            "worker_count": 1,
        }

    def fake_state_stats(llm):
        calls.append(("state", llm))
        return {
            "resident_to_decode_copies": 0,
            "decode_compactions": 1,
            "decode_compaction_rows": 2,
        }

    monkeypatch.setattr(bench, "_create_vllm_runner_llm", fake_create)
    monkeypatch.setattr(bench, "_time_vllm_runner_steady_decode", fake_time)
    monkeypatch.setattr(
        bench,
        "_extract_runner_state_movement_stats",
        fake_state_stats,
    )

    rc = bench.main(
        [
            "--repo-root",
            str(repo_root),
            "--model",
            str(model_path),
            "--measure-vllm-runner",
            "--runner-batch-size",
            "3",
            "--runner-prompt-len",
            "5",
            "--runner-decode-tokens",
            "7",
            "--runner-warmup",
            "2",
            "--runner-iters",
            "11",
            "--measurement-output",
            str(output_path),
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    runner = measurement["runner_steady_decode"]
    assert rc == 0
    assert runner["runner_tokens_per_s"] == 91.0
    assert runner["runner_batch_size"] == 3
    assert runner["runner_prompt_len"] == 5
    assert runner["runner_decode_tokens"] == 7
    assert runner["runner_warmup"] == 2
    assert runner["runner_iters"] == 11
    assert runner["runner_p50_ms"] == 2.0
    assert runner["runner_measurement_mode"] == "worker_execute_model"
    assert runner["runner_internal_timing_target"] == "worker.execute_model"
    assert runner["runner_timing_clock"] == "cuda_event"
    assert runner["runner_decode_steps"] == 7
    assert runner["runner_worker_count"] == 1
    assert measurement["state_movement"] == {
        "resident_to_decode_copies": 0,
        "decode_compactions": 1,
        "decode_compaction_rows": 2,
    }
    assert (
        measurement["config"]["measurement_source"]
        == "vllm_runner_worker_execute_model"
    )
    assert calls == [
        ("create", str(model_path)),
        ("time", fake_llm, 3, 5, 7, 2, 11),
        ("state", fake_llm),
    ]


def test_cli_merges_vllm_runner_measurement_with_model_only_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    input_path = tmp_path / "model-only.json"
    output_path = tmp_path / "combined.json"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": bench.SCHEMA_VERSION,
                "benchmark": bench.BENCHMARK_NAME,
                "model_only_steady_decode": {
                    "albatross_tokens_per_s": 100.0,
                    "vllm_tokens_per_s": 96.0,
                    "albatross_batch_size": 2,
                    "albatross_seq_len": 4,
                },
                "config": {"measurement_source": "merged_vllm_model_direct"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(bench, "_create_vllm_runner_llm", lambda config: object())
    monkeypatch.setattr(
        bench,
        "_time_vllm_runner_steady_decode",
        lambda llm, **kwargs: {
            "tokens_per_s": 91.0,
            "p10_ms": 1.9,
            "p50_ms": 2.0,
            "p90_ms": 2.4,
            "measurement_mode": "worker_execute_model",
            "internal_timing_target": "worker.execute_model",
            "decode_steps": 4,
            "worker_count": 1,
        },
    )
    monkeypatch.setattr(
        bench,
        "_extract_runner_state_movement_stats",
        lambda llm: {
            "resident_to_decode_copies": 0,
            "decode_compactions": 2,
            "decode_compaction_rows": 3,
        },
    )

    rc = bench.main(
        [
            "--repo-root",
            str(repo_root),
            "--model",
            str(model_path),
            "--measurement-json",
            str(input_path),
            "--measure-vllm-runner",
            "--runner-batch-size",
            "2",
            "--runner-prompt-len",
            "4",
            "--runner-decode-tokens",
            "4",
            "--runner-warmup",
            "1",
            "--runner-iters",
            "3",
            "--measurement-output",
            str(output_path),
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert measurement["model_only_steady_decode"]["albatross_tokens_per_s"] == 100.0
    assert measurement["model_only_steady_decode"]["vllm_tokens_per_s"] == 96.0
    assert measurement["runner_steady_decode"]["runner_tokens_per_s"] == 91.0
    assert measurement["runner_steady_decode"]["runner_batch_size"] == 2
    assert measurement["runner_steady_decode"]["runner_timing_clock"] == "cuda_event"
    assert measurement["state_movement"]["resident_to_decode_copies"] == 0
    assert (
        measurement["config"]["measurement_source"]
        == "merged_vllm_runner_worker_execute_model"
    )

    report = bench.build_report(
        _config(repo_root),
        measurements=measurement,
        cuda_available=True,
    )
    assert report["checks"]["model_only_steady_decode"]["status"] == "passed"
    assert report["checks"]["runner_steady_decode"]["status"] == "passed"
    assert report["checks"]["state_movement"]["status"] == "passed"
    runner_metrics = report["checks"]["runner_steady_decode"]["metrics"]
    assert runner_metrics["runner_tokens_per_s"] == 91.0
    assert "runner_to_albatross_model_only_ratio" not in runner_metrics


def test_runner_check_does_not_compare_worker_timing_to_logits_baseline() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 1000.0,
            "vllm_tokens_per_s": 950.0,
        },
        "runner_steady_decode": {
            "runner_tokens_per_s": 1.0,
            "runner_measurement_mode": "worker_execute_model",
            "runner_internal_timing_target": "worker.execute_model",
        },
        "state_movement": {
            "resident_to_decode_copies": 0,
            "decode_compactions": 0,
            "decode_compaction_rows": 0,
        },
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=True,
    )

    runner_check = report["checks"]["runner_steady_decode"]
    assert runner_check["status"] == "passed"
    assert runner_check["thresholds"] == {"min_runner_tokens_per_s": 1.0}
    assert runner_check["metrics"]["runner_tokens_per_s"] == 1.0
    assert (
        runner_check["metrics"]["runner_internal_timing_target"]
        == "worker.execute_model"
    )
    assert "runner_to_albatross_model_only_ratio" not in runner_check["metrics"]


def test_cli_writes_blocked_vllm_runner_json_without_fake_tokens(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    output_path = tmp_path / "runner-blocked.json"

    monkeypatch.setattr(bench, "_create_vllm_runner_llm", lambda config: object())
    monkeypatch.setattr(
        bench,
        "_time_vllm_runner_steady_decode",
        lambda llm, **kwargs: {
            "measurement_mode": "worker_execute_model",
            "internal_timing_target": "worker.execute_model",
            "blockers": [
                {
                    "code": "missing_internal_runner_decode_samples",
                    "message": "No internal worker decode timing samples were "
                    "recorded.",
                }
            ],
        },
    )
    monkeypatch.setattr(
        bench,
        "_extract_runner_state_movement_stats",
        lambda llm: {
            "resident_to_decode_copies": 0,
            "decode_compactions": 0,
            "decode_compaction_rows": 0,
        },
    )

    rc = bench.main(
        [
            "--repo-root",
            str(repo_root),
            "--model",
            str(model_path),
            "--measure-vllm-runner",
            "--runner-batch-size",
            "1",
            "--runner-prompt-len",
            "4",
            "--runner-decode-tokens",
            "2",
            "--runner-warmup",
            "0",
            "--runner-iters",
            "1",
            "--measurement-output",
            str(output_path),
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    runner = measurement["runner_steady_decode"]
    assert rc == 0
    assert "runner_tokens_per_s" not in runner
    assert runner["runner_measurement_mode"] == "worker_execute_model"
    assert runner["runner_batch_size"] == 1
    assert runner["runner_prompt_len"] == 4
    assert runner["runner_decode_tokens"] == 2
    assert runner["blockers"] == [
        {
            "code": "missing_internal_runner_decode_samples",
            "message": "No internal worker decode timing samples were recorded.",
        }
    ]

    report = bench.build_report(
        _config(repo_root),
        measurements={
            **measurement,
            "model_only_steady_decode": {
                "albatross_tokens_per_s": 100.0,
                "vllm_tokens_per_s": 96.0,
            },
        },
        cuda_available=True,
    )
    assert report["checks"]["runner_steady_decode"]["status"] == "blocked"
    assert report["checks"]["runner_steady_decode"]["blockers"] == [
        {
            "code": "missing_internal_runner_decode_samples",
            "message": "No internal worker decode timing samples were recorded.",
        }
    ]
    assert report["checks"]["state_movement"]["status"] == "passed"


def test_internal_runner_timing_syncs_once_around_decode_loop(monkeypatch) -> None:
    class FakeWorker:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def execute_model(self, scheduler_output):
            self.calls.append(
                (
                    "execute",
                    scheduler_output.total_num_scheduled_tokens,
                    bool(scheduler_output.finished_req_ids),
                )
            )

        def sample_tokens(self, grammar_output):
            self.calls.append(("sample", grammar_output))

    sync_calls: list[Any] = []
    monkeypatch.setattr(bench, "_worker_cuda_event_pair", lambda: None)
    monkeypatch.setattr(bench, "_worker_cuda_synchronize", lambda: sync_calls.append(1))
    worker = FakeWorker()

    result = bench._run_vllm_worker_internal_steady_decode(
        worker,
        batch_size=2,
        prompt_len=3,
        decode_tokens=4,
        iters=1,
        measure=True,
    )

    assert result["decode_steps"] == 4
    assert len(result["iteration_durations_s"]) == 1
    assert result["timing_clock"] == "wall_clock"
    assert sync_calls == [1] * 13
    assert [call[0] for call in worker.calls].count("execute") == 6
    assert [call[0] for call in worker.calls].count("sample") == 5


def test_internal_runner_uses_cuda_event_timing_when_available(monkeypatch) -> None:
    class FakeEvent:
        def __init__(self, elapsed_ms: float = 2.0) -> None:
            self.elapsed_ms = elapsed_ms
            self.records = 0
            self.synchronizes = 0

        def record(self) -> None:
            self.records += 1

        def synchronize(self) -> None:
            self.synchronizes += 1

        def elapsed_time(self, other) -> float:
            return self.elapsed_ms

    class FakeWorker:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def execute_model(self, scheduler_output):
            self.calls.append(("execute", scheduler_output.total_num_scheduled_tokens))

        def sample_tokens(self, grammar_output):
            self.calls.append(("sample", grammar_output))

    start_event = FakeEvent()
    end_event = FakeEvent()
    sync_calls: list[Any] = []
    monkeypatch.setattr(
        bench,
        "_worker_cuda_event_pair",
        lambda: (start_event, end_event),
    )
    monkeypatch.setattr(bench, "_worker_cuda_synchronize", lambda: sync_calls.append(1))
    worker = FakeWorker()

    result = bench._run_vllm_worker_internal_steady_decode(
        worker,
        batch_size=2,
        prompt_len=3,
        decode_tokens=4,
        iters=1,
        measure=True,
    )

    assert result["timing_clock"] == "cuda_event"
    assert result["iteration_durations_s"] == [0.008]
    assert start_event.records == 4
    assert end_event.records == 4
    assert end_event.synchronizes == 4
    assert sync_calls == [1] * 5
    assert [call[0] for call in worker.calls].count("execute") == 6
    assert [call[0] for call in worker.calls].count("sample") == 5


def test_report_blocks_when_runner_measurement_is_missing() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 100.0,
            "vllm_tokens_per_s": 96.0,
        },
        "state_movement": {
            "resident_to_decode_copies": 0,
            "decode_compactions": 0,
            "decode_compaction_rows": 0,
        },
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=True,
    )

    assert report["checks"]["model_only_steady_decode"]["status"] == "passed"
    assert report["checks"]["runner_steady_decode"]["status"] == "blocked"
    assert report["checks"]["runner_steady_decode"]["blockers"] == [
        {
            "code": "missing_runner_measurement",
            "message": "Measurement JSON must include runner_tokens_per_s for "
            "runner_steady_decode.",
        }
    ]
    assert report["checks"]["state_movement"]["status"] == "passed"


def test_extract_runner_state_movement_stats_uses_collective_rpc() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            self.calls.append((method, timeout, args, kwargs))
            return [
                {
                    "resident_to_decode_copies": 1,
                    "decode_compactions": 2,
                    "decode_compaction_rows": 3,
                }
            ]

    llm = FakeLLM()

    stats = bench._extract_runner_state_movement_stats(llm)

    assert stats == {
        "resident_to_decode_copies": 1,
        "decode_compactions": 2,
        "decode_compaction_rows": 3,
    }
    assert callable(llm.calls[0][0])


def test_report_blocks_without_measurement_json_when_runtime_paths_exist(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    albatross_root = tmp_path / "albatross"
    impl_dir = albatross_root / bench.ALBATROSS_IMPL
    impl_dir.mkdir(parents=True)
    model_path = tmp_path / "model.pth"
    checkpoint_path = tmp_path / "checkpoint.pth"
    model_path.write_bytes(b"")
    checkpoint_path.write_bytes(b"")

    report = bench.build_report(
        _config(
            repo_root,
            model=str(model_path),
            albatross_root=albatross_root,
            albatross_checkpoint=checkpoint_path,
        ),
        cuda_available=True,
    )

    assert report["overall_status"] == "blocked"
    for check_name in (
        "model_only_steady_decode",
        "runner_steady_decode",
        "state_movement",
    ):
        assert report["checks"][check_name]["blockers"] == [
            {
                "code": "missing_measurement_json",
                "message": "Provide --measurement-json with RWKV7 faster3a "
                "benchmark metrics, or run the measurement lane first.",
            }
        ]


def test_report_fails_low_model_only_and_resident_decode_copies() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 100.0,
            "vllm_tokens_per_s": 80.0,
        },
        "runner_steady_decode": {
            "runner_tokens_per_s": 70.0,
        },
        "state_movement": {
            "resident_to_decode_copies": 1,
            "decode_compactions": 0,
            "decode_compaction_rows": 0,
        },
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=True,
    )

    assert report["overall_status"] == "failed"
    assert report["checks"]["model_only_steady_decode"]["status"] == "failed"
    assert report["checks"]["runner_steady_decode"]["status"] == "passed"
    assert report["checks"]["state_movement"]["status"] == "failed"


def test_cli_writes_structured_blocked_json(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"
    rc = bench.main(
        [
            "--repo-root",
            str(Path(__file__).resolve().parents[2]),
            "--model",
            str(tmp_path / "missing-model"),
            "--albatross-root",
            str(tmp_path / "missing-albatross"),
            "--albatross-checkpoint",
            str(tmp_path / "missing.pth"),
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert rc == 2
    assert report["benchmark"] == "rwkv7_faster3a"
    assert report["overall_status"] == "blocked"
    assert "model_only_steady_decode" in report["checks"]
    assert "runner_steady_decode" in report["checks"]


def test_script_entrypoint_writes_structured_blocked_json(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/rwkv7/benchmark_faster3a.py",
            "--repo-root",
            str(repo_root),
            "--model",
            str(tmp_path / "missing-model"),
            "--albatross-root",
            str(tmp_path / "missing-albatross"),
            "--albatross-checkpoint",
            str(tmp_path / "missing.pth"),
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["benchmark"] == "rwkv7_faster3a"
    assert report["overall_status"] == "blocked"
