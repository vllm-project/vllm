# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV7 faster3a performance acceptance harness.

This harness does not claim performance by default. It records Albatross
provenance, reports runtime blockers as structured JSON, and can evaluate an
external measurement JSON against the faster3a acceptance thresholds.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
BENCHMARK_NAME = "rwkv7_faster3a"
ALBATROSS_BENCH_SCRIPT = "rwkv7_fast_v3a.py"
ALBATROSS_REPO = "https://github.com/BlinkDL/Albatross"
ALBATROSS_COMMIT = "5e941fb1eeb7f735a562fb5bbb30fad19adc825b"
ALBATROSS_IMPL = "faster3a_2605"
VLLM_MODEL_ONLY_LABEL = "RWKV7ForCausalLM.forward_logits"
VLLM_RUNNER_MODE = "worker_execute_model"
VLLM_RUNNER_TIMING_TARGET = "worker.execute_model"
VLLM_RUNNER_TIMING_CLOCK = "cuda_event"
STATE_MOVEMENT_COUNTERS = (
    "resident_to_decode_copies",
    "decode_compactions",
    "decode_compaction_rows",
)
ACCEPTANCE_THRESHOLDS = {
    "model_only_steady_decode": {
        "min_vllm_to_albatross_ratio": 0.95,
        "max_latency_slowdown_pct": 5.0,
    },
    "runner_steady_decode": {
        "min_runner_tokens_per_s": 1.0,
    },
    "state_movement": {
        "max_resident_to_decode_copies": 0,
    },
}


@dataclass(frozen=True)
class SourceProvenanceEntry:
    source_path: str
    target_path: str
    correspondence: str


SOURCE_PROVENANCE = (
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/rwkv7_fast_v3a.py",
        target_path="vllm/model_executor/models/rwkv7.py",
        correspondence="model-core-adapted",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_fast_ops_fp16.cpp",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_fast_ops_fp16.cpp",
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_fast_ops_fp16.cu",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_fast_ops_fp16.cu",
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_v3a_ops.cpp",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_v3a_ops.cpp",
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_v3a_ops.cu",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_v3a_ops.cu",
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_wkv_fp16_v2.cpp",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp16_v2.cpp",
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_wkv_fp16_v2.cu",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp16_v2.cu",
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_wkv_fp32_v2.cpp",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp32_v2.cpp",
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_wkv_fp32_v2.cu",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp32_v2.cu",
        correspondence="cuda-source-port",
    ),
)


@dataclass(frozen=True)
class BenchmarkConfig:
    repo_root: Path
    model: str | None
    albatross_root: Path | None
    albatross_impl: str
    albatross_checkpoint: Path | None
    batch_size: int
    prompt_len: int
    warmup_tokens: int
    decode_tokens: int

    @property
    def albatross_impl_dir(self) -> Path | None:
        if self.albatross_root is None:
            return None
        return self.albatross_root / self.albatross_impl


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https")


def _blocker(code: str, message: str, **details: Any) -> dict[str, Any]:
    blocker = {"code": code, "message": message}
    blocker.update({k: v for k, v in details.items() if v is not None})
    return blocker


def _empty_state_movement_metrics() -> dict[str, int | None]:
    return {name: None for name in STATE_MOVEMENT_COUNTERS}


def _measurement_blockers(
    runtime_blockers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if runtime_blockers:
        return runtime_blockers
    return [
        _blocker(
            "missing_measurement_json",
            "Provide --measurement-json with RWKV7 faster3a benchmark metrics, "
            "or run the measurement lane first.",
        )
    ]


def _source_metadata(config: BenchmarkConfig) -> dict[str, Any]:
    impl_dir = config.albatross_impl_dir
    return {
        "albatross_repo": ALBATROSS_REPO,
        "albatross_commit": ALBATROSS_COMMIT,
        "albatross_impl": ALBATROSS_IMPL,
        "albatross_path": str(impl_dir) if impl_dir is not None else None,
        "contracts": [
            {
                "source_path": entry.source_path,
                "target_path": entry.target_path,
                "correspondence": entry.correspondence,
            }
            for entry in SOURCE_PROVENANCE
        ],
    }


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(
        torch.accelerator.is_available()
        and torch.accelerator.current_accelerator().type == "cuda"
    )


def _cuda_module() -> Any:
    import torch

    return torch.cuda


def _runtime_blockers(
    config: BenchmarkConfig,
    *,
    cuda_available: bool,
    require_albatross_checkpoint: bool,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not cuda_available:
        blockers.append(
            _blocker(
                "cuda_unavailable",
                "CUDA is required for RWKV7 faster3a steady decode benchmarking.",
            )
        )

    if not config.model:
        blockers.append(
            _blocker(
                "missing_vllm_model",
                "Set --model or VLLM_RWKV7_MODEL to a vLLM-loadable RWKV7 model.",
            )
        )
    elif not _is_url(config.model) and not Path(config.model).expanduser().exists():
        blockers.append(
            _blocker(
                "missing_vllm_model_path",
                "The configured vLLM model path does not exist.",
                path=config.model,
            )
        )

    impl_dir = config.albatross_impl_dir
    if impl_dir is None:
        blockers.append(
            _blocker(
                "missing_albatross_root",
                "Set --albatross-root or ALBATROSS_ROOT.",
            )
        )
    elif not impl_dir.is_dir():
        blockers.append(
            _blocker(
                "missing_albatross_impl_path",
                "The configured Albatross implementation directory does not exist.",
                path=str(impl_dir),
            )
        )

    if require_albatross_checkpoint:
        checkpoint = config.albatross_checkpoint
        if checkpoint is None:
            blockers.append(
                _blocker(
                    "missing_albatross_checkpoint",
                    "Set --albatross-checkpoint or ALBATROSS_PTH.",
                )
            )
        elif not checkpoint.expanduser().is_file():
            blockers.append(
                _blocker(
                    "missing_albatross_checkpoint_path",
                    "The configured Albatross checkpoint path does not exist.",
                    path=str(checkpoint),
                )
            )
    return blockers


def _get_number(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    return float(value)


def _evaluate_model_only(
    measurements: dict[str, Any] | None,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        "albatross_tokens_per_s": None,
        "vllm_tokens_per_s": None,
        "vllm_to_albatross_ratio": None,
        "latency_slowdown_pct": None,
    }
    check = {
        "status": "blocked",
        "thresholds": ACCEPTANCE_THRESHOLDS["model_only_steady_decode"],
        "metrics": metrics,
        "blockers": blockers,
        "errors": [],
    }
    if measurements is None:
        check["blockers"] = _measurement_blockers(blockers)
        return check

    raw_metrics = measurements.get("model_only_steady_decode", {})
    albatross_tps = _get_number(raw_metrics, "albatross_tokens_per_s")
    vllm_tps = _get_number(raw_metrics, "vllm_tokens_per_s")
    metrics["albatross_tokens_per_s"] = albatross_tps
    metrics["vllm_tokens_per_s"] = vllm_tps
    if albatross_tps is None and vllm_tps is None:
        check["blockers"] = [
            _blocker(
                "missing_model_only_measurement",
                "Measurement JSON must include albatross_tokens_per_s and "
                "vllm_tokens_per_s for model_only_steady_decode.",
            )
        ]
        return check
    if albatross_tps is None:
        check["blockers"] = [
            _blocker(
                "missing_albatross_model_only_measurement",
                "Measurement JSON must include albatross_tokens_per_s for "
                "model_only_steady_decode.",
            )
        ]
        return check
    if vllm_tps is None:
        check["blockers"] = [
            _blocker(
                "missing_vllm_model_only_measurement",
                "Measurement JSON must include vllm_tokens_per_s for "
                "model_only_steady_decode. Albatross model-only measurement is "
                "present; generate vLLM model-only metrics with "
                "--measure-vllm-model-only.",
            )
        ]
        return check
    if albatross_tps <= 0 or vllm_tps <= 0:
        check["status"] = "failed"
        check["errors"] = [
            "model_only_steady_decode token throughput values must be positive"
        ]
        return check

    ratio = vllm_tps / albatross_tps
    slowdown_pct = (albatross_tps / vllm_tps - 1.0) * 100.0
    metrics.update(
        {
            "albatross_tokens_per_s": albatross_tps,
            "vllm_tokens_per_s": vllm_tps,
            "vllm_to_albatross_ratio": ratio,
            "latency_slowdown_pct": slowdown_pct,
        }
    )
    passed = (
        ratio
        >= ACCEPTANCE_THRESHOLDS["model_only_steady_decode"][
            "min_vllm_to_albatross_ratio"
        ]
        or slowdown_pct
        <= ACCEPTANCE_THRESHOLDS["model_only_steady_decode"]["max_latency_slowdown_pct"]
    )
    check["status"] = "passed" if passed else "failed"
    if not passed:
        check["errors"] = [
            "vLLM model-only steady decode is below the albatross threshold"
        ]
    check["blockers"] = []
    return check


def _parse_bxt_case(case: str, label: str) -> tuple[int, int]:
    try:
        batch_text, seq_text = case.lower().split("x", 1)
        batch_size = int(batch_text)
        seq_len = int(seq_text)
    except ValueError as exc:
        raise ValueError(f"{label} must use BxT format, for example 2x4") from exc
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError(f"{label} values must be positive")
    return batch_size, seq_len


def _parse_albatross_case(case: str) -> tuple[int, int]:
    return _parse_bxt_case(case, "albatross case")


def _parse_albatross_csv(
    output: str,
    *,
    expected_batch_size: int,
    expected_seq_len: int,
) -> dict[str, Any]:
    csv_rows = []
    for row in csv.reader(output.splitlines()):
        if row and row[0] == "csv":
            csv_rows.append(row)

    for row in csv_rows:
        if len(row) != 9:
            raise ValueError(f"unexpected albatross csv row shape: {row}")
        (
            _,
            label,
            batch_size,
            seq_len,
            iters,
            p10_ms,
            p50_ms,
            p90_ms,
            tok_s,
        ) = row
        parsed_batch_size = int(batch_size)
        parsed_seq_len = int(seq_len)
        if (
            parsed_batch_size != expected_batch_size
            or parsed_seq_len != expected_seq_len
        ):
            continue
        return {
            "label": label,
            "batch_size": parsed_batch_size,
            "seq_len": parsed_seq_len,
            "iters": int(iters),
            "p10_ms": float(p10_ms),
            "p50_ms": float(p50_ms),
            "p90_ms": float(p90_ms),
            "tokens_per_s": float(tok_s),
        }

    if csv_rows:
        raise ValueError(
            "albatross subprocess did not emit the requested BxT csv row "
            f"({expected_batch_size}x{expected_seq_len})"
        )
    raise ValueError("albatross subprocess did not emit a csv measurement row")


def _required_albatross_script(config: BenchmarkConfig) -> Path:
    impl_dir = config.albatross_impl_dir
    if impl_dir is None:
        raise ValueError("Set --albatross-root or ALBATROSS_ROOT.")
    script = impl_dir / ALBATROSS_BENCH_SCRIPT
    if not script.is_file():
        raise FileNotFoundError(f"missing albatross benchmark script: {script}")
    return script


def _required_albatross_checkpoint(config: BenchmarkConfig) -> Path:
    checkpoint = config.albatross_checkpoint
    if checkpoint is None:
        raise ValueError("Set --albatross-checkpoint or ALBATROSS_PTH.")
    checkpoint = checkpoint.expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing albatross checkpoint: {checkpoint}")
    return checkpoint


def generate_albatross_model_only_measurement(
    config: BenchmarkConfig,
    *,
    case: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    batch_size, seq_len = _parse_albatross_case(case)
    if warmup < 0:
        raise ValueError("albatross warmup must be non-negative")
    if iters <= 0:
        raise ValueError("albatross iters must be positive")

    script = _required_albatross_script(config)
    checkpoint = _required_albatross_checkpoint(config)
    command = [
        sys.executable,
        str(script),
        "--model",
        str(checkpoint),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--cases",
        f"{batch_size}x{seq_len}",
    ]
    result = subprocess.run(
        command,
        cwd=script.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "albatross model-only subprocess failed with exit code "
            f"{result.returncode}: {result.stderr.strip()}"
        )

    parsed = _parse_albatross_csv(
        result.stdout,
        expected_batch_size=batch_size,
        expected_seq_len=seq_len,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "model_only_steady_decode": {
            "albatross_tokens_per_s": parsed["tokens_per_s"],
            "albatross_label": parsed["label"],
            "albatross_batch_size": parsed["batch_size"],
            "albatross_seq_len": parsed["seq_len"],
            "albatross_warmup": warmup,
            "albatross_iters": parsed["iters"],
            "albatross_p10_ms": parsed["p10_ms"],
            "albatross_p50_ms": parsed["p50_ms"],
            "albatross_p90_ms": parsed["p90_ms"],
        },
        "config": {
            "repo_root": str(config.repo_root),
            "albatross_root": (
                str(config.albatross_root)
                if config.albatross_root is not None
                else None
            ),
            "albatross_impl": config.albatross_impl,
            "albatross_checkpoint": str(checkpoint),
            "albatross_command": command,
            "measurement_source": "albatross_subprocess",
        },
    }


def _required_vllm_model_path(config: BenchmarkConfig) -> Path:
    if not config.model:
        raise ValueError("Set --model or VLLM_RWKV7_MODEL.")
    if _is_url(config.model):
        raise ValueError(
            "vLLM model-only measurement currently requires a local RWKV7 "
            f"raw .pth checkpoint, got URL: {config.model}"
        )
    model_path = Path(config.model).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"missing vLLM model checkpoint: {model_path}")
    return model_path


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _initialize_vllm_single_process_distributed() -> str:
    import torch

    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import (
        VllmConfig,
        get_current_vllm_config_or_none,
        set_current_vllm_config,
    )

    preferred_backend = "nccl" if _cuda_available() else "gloo"
    backends = [preferred_backend]
    if preferred_backend == "nccl":
        backends.append("gloo")

    if parallel_state.model_parallel_is_initialized():
        return preferred_backend

    def initialize(backend: str) -> None:
        if not torch.distributed.is_initialized():
            parallel_state.init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{_free_tcp_port()}",
                local_rank=0,
                backend=backend,
            )
        parallel_state.ensure_model_parallel_initialized(1, 1, backend=backend)

    last_error: Exception | None = None
    for backend in backends:
        try:
            if get_current_vllm_config_or_none() is None:
                with set_current_vllm_config(VllmConfig()):
                    initialize(backend)
            else:
                initialize(backend)
            return backend
        except Exception as exc:
            last_error = exc
            can_retry = (
                backend == "nccl"
                and not torch.distributed.is_initialized()
                and not parallel_state.model_parallel_is_initialized()
            )
            if can_retry:
                continue
            raise

    assert last_error is not None
    raise last_error


def _load_vllm_rwkv7_model(config: BenchmarkConfig) -> Any:
    model_path = _required_vllm_model_path(config)

    import torch
    import vllm.rwkv7_ops  # noqa: F401

    from vllm.config.compilation import CompilationConfig, CompilationMode
    from vllm.model_executor.models.rwkv7 import RWKV7ForCausalLM
    from vllm.transformers_utils.configs.rwkv7 import build_rwkv7_config_from_pth

    hf_config = build_rwkv7_config_from_pth(model_path)
    if hf_config is None:
        raise ValueError(
            "vLLM model-only measurement currently supports RWKV7 raw .pth "
            f"checkpoints only: {model_path}"
        )
    vllm_config = SimpleNamespace(
        compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        model_config=SimpleNamespace(enforce_eager=True, hf_config=hf_config),
    )
    distributed_backend = _initialize_vllm_single_process_distributed()
    model = RWKV7ForCausalLM(vllm_config=vllm_config)
    model._benchmark_distributed_backend = distributed_backend
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"RWKV7 checkpoint must contain a state dict: {model_path}")
    model.load_weights(checkpoint.items())
    model.eval()
    return model


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _time_vllm_model_only_steady_decode(
    model: Any,
    *,
    batch_size: int,
    seq_len: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    import torch

    from vllm.model_executor.models.rwkv7 import select_path

    if not _cuda_available():
        raise RuntimeError(
            "CUDA is required for vLLM RWKV7 model-only steady decode measurement."
        )
    cuda = _cuda_module()
    if warmup < 0:
        raise ValueError("vLLM warmup must be non-negative")
    if iters <= 0:
        raise ValueError("vLLM iters must be positive")

    vocab_size = max(1, int(getattr(model, "vocab_size", 0)))
    tokens = torch.arange(
        batch_size * seq_len,
        dtype=torch.long,
        device="cuda",
    ).remainder(vocab_size)
    tokens = tokens.view(batch_size, seq_len)
    state = model.zero_state(batch_size)
    path = select_path(batch_size, seq_len)
    durations_ms: list[float] = []

    if getattr(model, "emb_cpu", False):
        x = model.embed(tokens)

        def run_model() -> Any:
            hidden_states = model.forward_from_x(x, state, path)
            return model.compute_logits(hidden_states)

    else:

        def run_model() -> Any:
            hidden_states = model.forward_tokens(tokens, state)
            return model.compute_logits(hidden_states)

    graph = cuda.CUDAGraph()
    with torch.inference_mode():
        for _ in range(warmup):
            run_model()
        torch.accelerator.synchronize()
        with cuda.graph(graph):
            run_model()
        for _ in range(warmup):
            graph.replay()
        torch.accelerator.synchronize()
        start_event = cuda.Event(enable_timing=True)
        end_event = cuda.Event(enable_timing=True)
        for _ in range(iters):
            start_event.record()
            graph.replay()
            end_event.record()
            end_event.synchronize()
            durations_ms.append(start_event.elapsed_time(end_event))

    p50_ms = _percentile(durations_ms, 0.5)
    return {
        "tokens_per_s": (batch_size * seq_len) / (p50_ms / 1000.0),
        "p10_ms": _percentile(durations_ms, 0.1),
        "p50_ms": p50_ms,
        "p90_ms": _percentile(durations_ms, 0.9),
        "graph": True,
        "measurement_mode": "cuda_graph_replay",
        "output": "logits",
        "logits_included": True,
        "distributed_backend": getattr(model, "_benchmark_distributed_backend", None),
    }


def generate_vllm_model_only_measurement(
    config: BenchmarkConfig,
    *,
    case: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    batch_size, seq_len = _parse_bxt_case(case, "vLLM case")
    if warmup < 0:
        raise ValueError("vLLM warmup must be non-negative")
    if iters <= 0:
        raise ValueError("vLLM iters must be positive")

    model = _load_vllm_rwkv7_model(config)
    parsed = _time_vllm_model_only_steady_decode(
        model,
        batch_size=batch_size,
        seq_len=seq_len,
        warmup=warmup,
        iters=iters,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "model_only_steady_decode": {
            "vllm_tokens_per_s": parsed["tokens_per_s"],
            "vllm_label": VLLM_MODEL_ONLY_LABEL,
            "vllm_batch_size": batch_size,
            "vllm_seq_len": seq_len,
            "vllm_warmup": warmup,
            "vllm_iters": iters,
            "vllm_p10_ms": parsed["p10_ms"],
            "vllm_p50_ms": parsed["p50_ms"],
            "vllm_p90_ms": parsed["p90_ms"],
            "vllm_graph": parsed["graph"],
            "vllm_measurement_mode": parsed["measurement_mode"],
            "vllm_output": "logits",
            "vllm_logits_included": True,
            "vllm_distributed_backend": parsed["distributed_backend"],
        },
        "config": {
            "repo_root": str(config.repo_root),
            "model": config.model,
            "vllm_distributed_backend": parsed["distributed_backend"],
            "measurement_source": "vllm_model_direct",
        },
    }


def _model_only_case_from_measurements(
    measurements: dict[str, Any] | None,
) -> str | None:
    if measurements is None:
        return None
    metrics = measurements.get("model_only_steady_decode", {})
    batch_size = metrics.get("albatross_batch_size")
    seq_len = metrics.get("albatross_seq_len")
    if batch_size is None or seq_len is None:
        return None
    return f"{int(batch_size)}x{int(seq_len)}"


def _merge_vllm_model_only_measurement(
    measurements: dict[str, Any],
    vllm_measurement: dict[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(measurements)
    merged_metrics = merged.setdefault("model_only_steady_decode", {})
    vllm_metrics = vllm_measurement.get("model_only_steady_decode", {})
    albatross_batch_size = merged_metrics.get("albatross_batch_size")
    albatross_seq_len = merged_metrics.get("albatross_seq_len")
    vllm_batch_size = vllm_metrics.get("vllm_batch_size")
    vllm_seq_len = vllm_metrics.get("vllm_seq_len")
    if (
        albatross_batch_size is not None
        and albatross_seq_len is not None
        and vllm_batch_size is not None
        and vllm_seq_len is not None
        and (
            int(albatross_batch_size) != int(vllm_batch_size)
            or int(albatross_seq_len) != int(vllm_seq_len)
        )
    ):
        raise ValueError(
            "vLLM model-only case must match Albatross model-only case: "
            f"albatross={albatross_batch_size}x{albatross_seq_len}, "
            f"vllm={vllm_batch_size}x{vllm_seq_len}"
        )

    merged_metrics.update(vllm_metrics)
    merged["schema_version"] = merged.get("schema_version", SCHEMA_VERSION)
    merged["benchmark"] = merged.get("benchmark", BENCHMARK_NAME)
    merged_config = dict(merged.get("config", {}))
    merged_config.update(
        {
            "model": vllm_measurement.get("config", {}).get("model"),
            "measurement_source": "merged_vllm_model_direct",
        }
    )
    merged["config"] = merged_config
    return merged


def _required_vllm_runner_model(config: BenchmarkConfig) -> str:
    if not config.model:
        raise ValueError("Set --model or VLLM_RWKV7_MODEL.")
    if not _is_url(config.model) and not Path(config.model).expanduser().exists():
        raise FileNotFoundError(f"missing vLLM model path: {config.model}")
    return config.model


def _create_vllm_runner_llm(config: BenchmarkConfig) -> Any:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    import vllm.rwkv7_ops  # noqa: F401

    from vllm import LLM

    max_model_len = max(1, config.prompt_len + config.decode_tokens)
    max_num_seqs = max(1, config.batch_size)
    max_num_batched_tokens = max(
        max_model_len,
        config.batch_size * config.prompt_len,
        config.batch_size,
    )
    return LLM(
        model=_required_vllm_runner_model(config),
        skip_tokenizer_init=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        disable_log_stats=True,
    )


def _synchronize_cuda_if_available() -> None:
    try:
        import torch
    except Exception:
        return
    if _cuda_available():
        torch.accelerator.synchronize()


_RUNNER_STATE_SEARCH_ATTRS = (
    "model_runner",
    "model_state",
    "model",
    "worker",
    "workers",
    "driver_worker",
    "executor",
    "llm_engine",
    "engine_core",
    "engine_core_client",
)


def _normalize_state_movement_stats(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise TypeError("RWKV7 state movement stats must be a dict")
    missing = [name for name in STATE_MOVEMENT_COUNTERS if name not in raw]
    if missing:
        raise ValueError(
            "RWKV7 state movement stats are missing counters: " + ", ".join(missing)
        )
    return {name: int(raw[name]) for name in STATE_MOVEMENT_COUNTERS}


def _merge_state_movement_stat_dicts(raw_stats: list[Any]) -> dict[str, int]:
    stats = [
        _normalize_state_movement_stats(raw) for raw in raw_stats if raw is not None
    ]
    if not stats:
        raise RuntimeError(
            "Could not locate RWKV7ModelState.get_state_movement_stats() "
            "through offline LLM worker/model_runner attributes."
        )
    return {
        name: sum(worker_stats[name] for worker_stats in stats)
        for name in STATE_MOVEMENT_COUNTERS
    }


def _iter_runner_state_children(obj: Any) -> list[Any]:
    children: list[Any] = []
    for attr in _RUNNER_STATE_SEARCH_ATTRS:
        try:
            child = getattr(obj, attr)
        except Exception:
            continue
        if child is not None:
            children.append(child)
    if isinstance(obj, dict):
        children.extend(obj.values())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        children.extend(obj)
    return children


def _collect_runner_state_movement_stats_from_object(
    root: Any,
    *,
    reset: bool,
) -> dict[str, int] | None:
    queue = [root]
    seen: set[int] = set()
    matches: list[dict[str, int]] = []
    while queue and len(seen) < 512:
        obj = queue.pop(0)
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        getter = getattr(obj, "get_state_movement_stats", None)
        if callable(getter):
            if reset:
                resetter = getattr(obj, "reset_state_movement_stats", None)
                if not callable(resetter):
                    raise RuntimeError(
                        "Located RWKV7 state movement stats without "
                        "reset_state_movement_stats()."
                    )
                resetter()
            matches.append(_normalize_state_movement_stats(getter()))
            continue

        queue.extend(_iter_runner_state_children(obj))

    if not matches:
        return None
    return _merge_state_movement_stat_dicts(matches)


def _collect_runner_state_movement_stats_from_worker(
    worker: Any,
    reset: bool = False,
) -> dict[str, int] | None:
    return _collect_runner_state_movement_stats_from_object(worker, reset=reset)


def _extract_runner_state_movement_stats(llm: Any) -> dict[str, int]:
    collective_rpc = getattr(llm, "collective_rpc", None)
    if callable(collective_rpc):
        stats = collective_rpc(_collect_runner_state_movement_stats_from_worker)
        try:
            return _merge_state_movement_stat_dicts(list(stats))
        except RuntimeError:
            local_stats = _collect_runner_state_movement_stats_from_object(
                llm,
                reset=False,
            )
            if local_stats is not None:
                return local_stats
            raise

    local_stats = _collect_runner_state_movement_stats_from_object(
        llm,
        reset=False,
    )
    if local_stats is None:
        return _merge_state_movement_stat_dicts([])
    return local_stats


def _reset_runner_state_movement_stats(llm: Any) -> None:
    collective_rpc = getattr(llm, "collective_rpc", None)
    if callable(collective_rpc):
        stats = collective_rpc(
            _collect_runner_state_movement_stats_from_worker,
            args=(True,),
        )
        _merge_state_movement_stat_dicts(list(stats))
        return

    local_stats = _collect_runner_state_movement_stats_from_object(
        llm,
        reset=True,
    )
    if local_stats is None:
        _merge_state_movement_stat_dicts([])


def _worker_cuda_synchronize() -> None:
    import torch

    if _cuda_available():
        torch.accelerator.synchronize()


def _worker_cuda_event_pair() -> tuple[Any, Any] | None:
    if not _cuda_available():
        return None
    cuda = _cuda_module()
    return (
        cuda.Event(enable_timing=True),
        cuda.Event(enable_timing=True),
    )


def _worker_time_execute_model(
    worker: Any,
    scheduler_output: Any,
    cuda_events: tuple[Any, Any] | None,
) -> float:
    if cuda_events is None:
        import time

        _worker_cuda_synchronize()
        start_s = time.perf_counter()
        worker.execute_model(scheduler_output)
        _worker_cuda_synchronize()
        return time.perf_counter() - start_s

    start_event, end_event = cuda_events
    start_event.record()
    worker.execute_model(scheduler_output)
    end_event.record()
    end_event.synchronize()
    return start_event.elapsed_time(end_event) / 1000.0


def _worker_empty_scheduler_output(finished_req_ids: set[str] | None = None) -> Any:
    from vllm.v1.core.sched.output import SchedulerOutput

    output = SchedulerOutput.make_empty()
    if finished_req_ids:
        output.finished_req_ids = finished_req_ids
    return output


def _worker_new_request_scheduler_output(
    *,
    req_ids: list[str],
    prompt_token_ids: list[list[int]],
    sampling_params: Any,
    prompt_len: int,
) -> Any:
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )

    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt_ids,
                mm_features=[],
                sampling_params=sampling_params,
                pooling_params=None,
                block_ids=(),
                num_computed_tokens=0,
                lora_request=None,
                prefill_token_ids=prompt_ids,
            )
            for req_id, prompt_ids in zip(req_ids, prompt_token_ids)
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: prompt_len for req_id in req_ids},
        total_num_scheduled_tokens=len(req_ids) * prompt_len,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _worker_cached_decode_scheduler_output(
    *,
    req_ids: list[str],
    prompt_len: int,
    step: int,
) -> Any:
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None] * len(req_ids),
            num_computed_tokens=[prompt_len + step] * len(req_ids),
            num_output_tokens=[step + 1] * len(req_ids),
        ),
        num_scheduled_tokens={req_id: 1 for req_id in req_ids},
        total_num_scheduled_tokens=len(req_ids),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _worker_internal_runner_blocker(code: str, message: str) -> dict[str, Any]:
    return {
        "measurement_mode": VLLM_RUNNER_MODE,
        "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
        "timing_clock": VLLM_RUNNER_TIMING_CLOCK,
        "blockers": [_blocker(code, message)],
    }


def _run_vllm_worker_internal_steady_decode(
    worker: Any,
    batch_size: int,
    prompt_len: int,
    decode_tokens: int,
    iters: int,
    measure: bool,
) -> dict[str, Any]:
    if not callable(getattr(worker, "execute_model", None)):
        return _worker_internal_runner_blocker(
            "missing_worker_execute_model",
            "The vLLM worker does not expose execute_model().",
        )
    if not callable(getattr(worker, "sample_tokens", None)):
        return _worker_internal_runner_blocker(
            "missing_worker_sample_tokens",
            "The vLLM worker does not expose sample_tokens().",
        )

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=decode_tokens,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    iteration_durations_s: list[float] = []
    execute_durations_s: list[float] = []
    sample_durations_s: list[float] = []
    prefix = f"rwkv7-runner-{id(worker)}-{time.perf_counter_ns()}"
    cuda_events = _worker_cuda_event_pair() if measure else None
    timing_clock = "cuda_event" if cuda_events is not None else "wall_clock"

    for iteration in range(iters):
        req_ids = [f"{prefix}-{iteration}-{idx}" for idx in range(batch_size)]
        prompt_token_ids = [
            [(idx + position) % 1024 for position in range(prompt_len)]
            for idx in range(batch_size)
        ]
        prefill_output = _worker_new_request_scheduler_output(
            req_ids=req_ids,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            prompt_len=prompt_len,
        )
        worker.execute_model(prefill_output)
        worker.sample_tokens(None)
        _worker_cuda_synchronize()

        if measure:
            iteration_duration_s = 0.0
        for step in range(decode_tokens):
            decode_output = _worker_cached_decode_scheduler_output(
                req_ids=req_ids,
                prompt_len=prompt_len,
                step=step,
            )
            if measure:
                iteration_duration_s += _worker_time_execute_model(
                    worker,
                    decode_output,
                    cuda_events,
                )
            else:
                worker.execute_model(decode_output)
            worker.sample_tokens(None)
            if measure:
                _worker_cuda_synchronize()

        if measure:
            iteration_durations_s.append(iteration_duration_s)

        worker.execute_model(_worker_empty_scheduler_output(set(req_ids)))

    return {
        "measurement_mode": VLLM_RUNNER_MODE,
        "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
        "iteration_durations_s": iteration_durations_s,
        "execute_durations_s": execute_durations_s,
        "sample_durations_s": sample_durations_s,
        "decode_steps": decode_tokens * iters if measure else 0,
        "tokens": batch_size * decode_tokens * iters if measure else 0,
        "timing_clock": timing_clock,
    }


def _merge_worker_internal_runner_results(
    worker_results: list[Any],
    *,
    batch_size: int,
    decode_tokens: int,
    iters: int,
) -> dict[str, Any]:
    if not worker_results:
        return _worker_internal_runner_blocker(
            "missing_internal_runner_worker_results",
            "No vLLM worker returned internal runner timing results.",
        )
    blockers = [
        blocker
        for result in worker_results
        if isinstance(result, dict)
        for blocker in result.get("blockers", [])
    ]
    if blockers:
        return {
            "measurement_mode": VLLM_RUNNER_MODE,
            "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
            "blockers": blockers,
        }

    normalized_results = [
        result for result in worker_results if isinstance(result, dict)
    ]
    if len(normalized_results) != len(worker_results):
        return _worker_internal_runner_blocker(
            "invalid_internal_runner_worker_result",
            "A vLLM worker returned a non-dict internal runner timing result.",
        )

    per_worker_iterations = [
        [float(value) for value in result.get("iteration_durations_s", [])]
        for result in normalized_results
    ]
    if any(len(values) != iters for values in per_worker_iterations):
        return _worker_internal_runner_blocker(
            "missing_internal_runner_decode_samples",
            "No complete internal worker decode timing samples were recorded.",
        )

    iteration_durations_s = [
        max(worker_values[index] for worker_values in per_worker_iterations)
        for index in range(iters)
    ]
    expected_decode_steps = decode_tokens * iters
    decode_steps = sum(
        int(result.get("decode_steps", 0)) for result in normalized_results
    )
    if decode_steps < expected_decode_steps:
        return _worker_internal_runner_blocker(
            "missing_internal_runner_decode_samples",
            "Internal worker decode timing did not record every requested decode step.",
        )
    if iters == 0:
        return {
            "measurement_mode": VLLM_RUNNER_MODE,
            "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
            "decode_steps": decode_steps,
            "worker_count": len(normalized_results),
        }

    p50_s = _percentile(iteration_durations_s, 0.5)
    return {
        "tokens_per_s": (batch_size * decode_tokens) / p50_s,
        "p10_ms": _percentile(iteration_durations_s, 0.1) * 1000.0,
        "p50_ms": p50_s * 1000.0,
        "p90_ms": _percentile(iteration_durations_s, 0.9) * 1000.0,
        "measurement_mode": VLLM_RUNNER_MODE,
        "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
        "decode_steps": decode_steps,
        "worker_count": len(normalized_results),
    }


def _time_vllm_runner_steady_decode(
    llm: Any,
    *,
    batch_size: int,
    prompt_len: int,
    decode_tokens: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("runner batch size must be positive")
    if prompt_len <= 0:
        raise ValueError("runner prompt len must be positive")
    if decode_tokens <= 0:
        raise ValueError("runner decode tokens must be positive")
    if warmup < 0:
        raise ValueError("runner warmup must be non-negative")
    if iters <= 0:
        raise ValueError("runner iters must be positive")

    collective_rpc = getattr(llm, "collective_rpc", None)
    if not callable(collective_rpc):
        return _worker_internal_runner_blocker(
            "missing_collective_rpc",
            "The vLLM LLM object does not expose collective_rpc().",
        )

    if warmup:
        warmup_results = collective_rpc(
            _run_vllm_worker_internal_steady_decode,
            args=(batch_size, prompt_len, decode_tokens, warmup, False),
        )
        warmup_blockers = _merge_worker_internal_runner_results(
            list(warmup_results),
            batch_size=batch_size,
            decode_tokens=decode_tokens,
            iters=0,
        ).get("blockers")
        if warmup_blockers:
            return {
                "measurement_mode": VLLM_RUNNER_MODE,
                "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
                "blockers": warmup_blockers,
            }
    _reset_runner_state_movement_stats(llm)

    timed_results = collective_rpc(
        _run_vllm_worker_internal_steady_decode,
        args=(batch_size, prompt_len, decode_tokens, iters, True),
    )
    return _merge_worker_internal_runner_results(
        list(timed_results),
        batch_size=batch_size,
        decode_tokens=decode_tokens,
        iters=iters,
    )


def _shutdown_vllm_runner_llm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()


def generate_vllm_runner_measurement(
    config: BenchmarkConfig,
    *,
    batch_size: int,
    prompt_len: int,
    decode_tokens: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("runner batch size must be positive")
    if prompt_len <= 0:
        raise ValueError("runner prompt len must be positive")
    if decode_tokens <= 0:
        raise ValueError("runner decode tokens must be positive")
    if warmup < 0:
        raise ValueError("runner warmup must be non-negative")
    if iters <= 0:
        raise ValueError("runner iters must be positive")

    runner_config = replace(
        config,
        batch_size=batch_size,
        prompt_len=prompt_len,
        decode_tokens=decode_tokens,
    )
    llm = _create_vllm_runner_llm(runner_config)
    try:
        parsed = _time_vllm_runner_steady_decode(
            llm,
            batch_size=batch_size,
            prompt_len=prompt_len,
            decode_tokens=decode_tokens,
            warmup=warmup,
            iters=iters,
        )
        state_movement = _extract_runner_state_movement_stats(llm)
    finally:
        _shutdown_vllm_runner_llm(llm)

    runner_metrics: dict[str, Any] = {
        "runner_batch_size": batch_size,
        "runner_prompt_len": prompt_len,
        "runner_decode_tokens": decode_tokens,
        "runner_warmup": warmup,
        "runner_iters": iters,
        "runner_measurement_mode": parsed.get("measurement_mode", VLLM_RUNNER_MODE),
        "runner_internal_timing_target": parsed.get(
            "internal_timing_target",
            VLLM_RUNNER_TIMING_TARGET,
        ),
        "runner_timing_clock": parsed.get(
            "timing_clock",
            VLLM_RUNNER_TIMING_CLOCK,
        ),
        "runner_collective_rpc_serialization": (
            "pickle_enabled"
            if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") == "1"
            else "msgpack_only"
        ),
    }
    if "blockers" in parsed:
        runner_metrics["blockers"] = parsed["blockers"]
    else:
        runner_metrics.update(
            {
                "runner_tokens_per_s": parsed["tokens_per_s"],
                "runner_p10_ms": parsed["p10_ms"],
                "runner_p50_ms": parsed["p50_ms"],
                "runner_p90_ms": parsed["p90_ms"],
                "runner_decode_steps": parsed["decode_steps"],
                "runner_worker_count": parsed["worker_count"],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "runner_steady_decode": runner_metrics,
        "state_movement": state_movement,
        "config": {
            "repo_root": str(config.repo_root),
            "model": config.model,
            "measurement_source": f"vllm_runner_{VLLM_RUNNER_MODE}",
        },
    }


def _merge_vllm_runner_measurement(
    measurements: dict[str, Any],
    runner_measurement: dict[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(measurements)
    merged["schema_version"] = merged.get("schema_version", SCHEMA_VERSION)
    merged["benchmark"] = merged.get("benchmark", BENCHMARK_NAME)
    merged["runner_steady_decode"] = copy.deepcopy(
        runner_measurement.get("runner_steady_decode", {})
    )
    if "state_movement" in runner_measurement:
        merged["state_movement"] = copy.deepcopy(runner_measurement["state_movement"])
    merged_config = dict(merged.get("config", {}))
    merged_config.update(
        {
            "model": runner_measurement.get("config", {}).get("model"),
            "measurement_source": f"merged_vllm_runner_{VLLM_RUNNER_MODE}",
        }
    )
    merged["config"] = merged_config
    return merged


def _evaluate_runner(
    measurements: dict[str, Any] | None,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        "runner_tokens_per_s": None,
        "runner_measurement_mode": None,
        "runner_internal_timing_target": None,
        "runner_timing_clock": None,
    }
    check = {
        "status": "blocked",
        "thresholds": ACCEPTANCE_THRESHOLDS["runner_steady_decode"],
        "metrics": metrics,
        "blockers": blockers,
        "errors": [],
    }
    if measurements is None:
        check["blockers"] = _measurement_blockers(blockers)
        return check

    raw_metrics = measurements.get("runner_steady_decode", {})
    runner_tps = _get_number(raw_metrics, "runner_tokens_per_s")
    if runner_tps is None:
        runner_blockers = raw_metrics.get("blockers")
        if runner_blockers:
            check["blockers"] = runner_blockers
            return check
        check["blockers"] = [
            _blocker(
                "missing_runner_measurement",
                "Measurement JSON must include runner_tokens_per_s for "
                "runner_steady_decode.",
            )
        ]
        return check
    if runner_tps <= 0:
        check["status"] = "failed"
        check["errors"] = ["runner_steady_decode token throughput must be positive"]
        return check

    metrics.update(
        {
            "runner_tokens_per_s": runner_tps,
            "runner_measurement_mode": raw_metrics.get("runner_measurement_mode"),
            "runner_internal_timing_target": raw_metrics.get(
                "runner_internal_timing_target"
            ),
            "runner_timing_clock": raw_metrics.get("runner_timing_clock"),
        }
    )
    passed = (
        runner_tps
        >= ACCEPTANCE_THRESHOLDS["runner_steady_decode"]["min_runner_tokens_per_s"]
    )
    check["status"] = "passed" if passed else "failed"
    if not passed:
        check["errors"] = [
            "vLLM runner steady decode did not produce positive steady throughput"
        ]
    check["blockers"] = []
    return check


def _evaluate_state_movement(
    measurements: dict[str, Any] | None,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = _empty_state_movement_metrics()
    check = {
        "status": "blocked",
        "thresholds": ACCEPTANCE_THRESHOLDS["state_movement"],
        "metrics": metrics,
        "blockers": blockers,
        "errors": [],
    }
    if measurements is None:
        check["blockers"] = _measurement_blockers(blockers)
        return check

    raw_metrics = measurements.get("state_movement", {})
    missing = [name for name in STATE_MOVEMENT_COUNTERS if name not in raw_metrics]
    if missing:
        check["blockers"] = [
            _blocker(
                "missing_state_movement_counters",
                "Measurement JSON must include all RWKV7 state movement counters.",
                missing=missing,
            )
        ]
        return check

    for name in STATE_MOVEMENT_COUNTERS:
        metrics[name] = int(raw_metrics[name])
    passed = (
        metrics["resident_to_decode_copies"]
        <= ACCEPTANCE_THRESHOLDS["state_movement"]["max_resident_to_decode_copies"]
    )
    check["status"] = "passed" if passed else "failed"
    if not passed:
        check["errors"] = ["steady decode resident-to-decode copies must remain zero"]
    check["blockers"] = []
    return check


def build_report(
    config: BenchmarkConfig,
    *,
    measurements: dict[str, Any] | None = None,
    cuda_available: bool | None = None,
) -> dict[str, Any]:
    cuda = _cuda_available() if cuda_available is None else cuda_available
    runtime_blockers = _runtime_blockers(
        config,
        cuda_available=cuda,
        require_albatross_checkpoint=True,
    )
    model_only_check = _evaluate_model_only(measurements, runtime_blockers)
    runner_check = _evaluate_runner(measurements, runtime_blockers)
    state_check = _evaluate_state_movement(measurements, runtime_blockers)
    checks = {
        "model_only_steady_decode": model_only_check,
        "runner_steady_decode": runner_check,
        "state_movement": state_check,
    }
    statuses = [check["status"] for check in checks.values()]
    if "failed" in statuses:
        overall_status = "failed"
    elif "blocked" in statuses:
        overall_status = "blocked"
    else:
        overall_status = "passed"
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "overall_status": overall_status,
        "source": _source_metadata(config),
        "config": {
            "repo_root": str(config.repo_root),
            "model": config.model,
            "albatross_root": (
                str(config.albatross_root)
                if config.albatross_root is not None
                else None
            ),
            "albatross_impl": config.albatross_impl,
            "albatross_checkpoint": (
                str(config.albatross_checkpoint)
                if config.albatross_checkpoint is not None
                else None
            ),
            "batch_size": config.batch_size,
            "prompt_len": config.prompt_len,
            "warmup_tokens": config.warmup_tokens,
            "decode_tokens": config.decode_tokens,
            "cuda_available": cuda,
            "measurement_source": "json" if measurements is not None else None,
        },
        "acceptance": ACCEPTANCE_THRESHOLDS,
        "checks": checks,
    }


def _default_repo_root() -> Path:
    return REPO_ROOT


def _optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    model = args.model or os.environ.get("VLLM_RWKV7_MODEL") or None
    albatross_root = _optional_path(
        args.albatross_root
        or os.environ.get(
            "ALBATROSS_ROOT",
            str(Path.home() / "Projects/MachineLearning/albatross"),
        )
    )
    checkpoint_env = os.environ.get("ALBATROSS_PTH") or os.environ.get(
        "VLLM_RWKV7_MODEL"
    )
    return BenchmarkConfig(
        repo_root=args.repo_root.resolve(),
        model=model,
        albatross_root=albatross_root,
        albatross_impl=args.albatross_impl,
        albatross_checkpoint=_optional_path(
            args.albatross_checkpoint or checkpoint_env
        ),
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        warmup_tokens=args.warmup_tokens,
        decode_tokens=args.decode_tokens,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RWKV7 faster3a performance acceptance harness."
    )
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--model", help="vLLM-loadable RWKV7 model path or URL.")
    parser.add_argument("--albatross-root")
    parser.add_argument(
        "--albatross-impl",
        default=ALBATROSS_IMPL,
    )
    parser.add_argument("--albatross-checkpoint")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument(
        "--measurement-json",
        type=Path,
        help="Optional JSON metrics file to evaluate against thresholds.",
    )
    parser.add_argument(
        "--measure-albatross-model-only",
        action="store_true",
        help="Run the canonical Albatross model-only benchmark for one BxT case.",
    )
    parser.add_argument(
        "--measure-vllm-model-only",
        action="store_true",
        help="Run the vLLM RWKV7 model-only steady decode benchmark.",
    )
    parser.add_argument(
        "--measure-vllm-runner",
        action="store_true",
        help="Run the vLLM offline LLM.generate runner steady decode benchmark.",
    )
    parser.add_argument(
        "--albatross-case",
        help="Single Albatross BxT case for --measure-albatross-model-only.",
    )
    parser.add_argument(
        "--vllm-case",
        help=(
            "Single vLLM BxT case for --measure-vllm-model-only. Defaults to "
            "the Albatross model-only case from --measurement-json, then "
            "--batch-size x --prompt-len."
        ),
    )
    parser.add_argument(
        "--albatross-warmup",
        type=int,
        default=1,
        help="Warmup iterations passed to rwkv7_fast_v3a.py.",
    )
    parser.add_argument(
        "--albatross-iters",
        type=int,
        default=3,
        help="Timed iterations passed to rwkv7_fast_v3a.py.",
    )
    parser.add_argument(
        "--vllm-warmup",
        type=int,
        default=1,
        help="Warmup iterations for vLLM model-only steady decode.",
    )
    parser.add_argument(
        "--vllm-iters",
        type=int,
        default=3,
        help="Timed iterations for vLLM model-only steady decode.",
    )
    parser.add_argument(
        "--runner-batch-size",
        type=int,
        default=16,
        help="Batch size for vLLM runner steady decode measurement.",
    )
    parser.add_argument(
        "--runner-prompt-len",
        type=int,
        default=128,
        help="Prompt token count for vLLM runner steady decode measurement.",
    )
    parser.add_argument(
        "--runner-decode-tokens",
        type=int,
        default=128,
        help="Generated token count for vLLM runner steady decode measurement.",
    )
    parser.add_argument(
        "--runner-warmup",
        type=int,
        default=1,
        help="Warmup iterations for vLLM runner steady decode.",
    )
    parser.add_argument(
        "--runner-iters",
        type=int,
        default=3,
        help="Timed iterations for vLLM runner steady decode.",
    )
    parser.add_argument(
        "--measurement-output",
        type=Path,
        help="Write generated measurement JSON for a measurement mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write structured JSON to this file instead of stdout.",
    )
    args = parser.parse_args(argv)
    measurement_modes = [
        args.measure_albatross_model_only,
        args.measure_vllm_model_only,
        args.measure_vllm_runner,
    ]
    if sum(bool(mode) for mode in measurement_modes) > 1:
        parser.error("choose only one measurement mode")
    if any(measurement_modes) and args.measurement_output is None:
        parser.error("--measurement-output is required with a measurement mode")
    return args


def _load_measurements(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as measurement_file:
        data = json.load(measurement_file)
    if not isinstance(data, dict):
        raise ValueError("measurement JSON must contain an object")
    return data


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(text)
        return
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config = _config_from_args(args)
    if args.measure_albatross_model_only:
        measurement = generate_albatross_model_only_measurement(
            config,
            case=args.albatross_case or f"{config.batch_size}x{config.prompt_len}",
            warmup=args.albatross_warmup,
            iters=args.albatross_iters,
        )
        _write_report(measurement, args.measurement_output)
        return 0
    if args.measure_vllm_model_only:
        existing_measurements = _load_measurements(args.measurement_json)
        measurement = generate_vllm_model_only_measurement(
            config,
            case=args.vllm_case
            or _model_only_case_from_measurements(existing_measurements)
            or f"{config.batch_size}x{config.prompt_len}",
            warmup=args.vllm_warmup,
            iters=args.vllm_iters,
        )
        if existing_measurements is not None:
            measurement = _merge_vllm_model_only_measurement(
                existing_measurements,
                measurement,
            )
        _write_report(measurement, args.measurement_output)
        return 0
    if args.measure_vllm_runner:
        existing_measurements = _load_measurements(args.measurement_json)
        measurement = generate_vllm_runner_measurement(
            config,
            batch_size=args.runner_batch_size,
            prompt_len=args.runner_prompt_len,
            decode_tokens=args.runner_decode_tokens,
            warmup=args.runner_warmup,
            iters=args.runner_iters,
        )
        if existing_measurements is not None:
            measurement = _merge_vllm_runner_measurement(
                existing_measurements,
                measurement,
            )
        _write_report(measurement, args.measurement_output)
        return 0

    report = build_report(
        config,
        measurements=_load_measurements(args.measurement_json),
    )
    _write_report(report, args.output)
    if report["overall_status"] == "passed":
        return 0
    if report["overall_status"] == "blocked":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
