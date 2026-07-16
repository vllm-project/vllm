# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV7 faster3a performance acceptance harness.

This harness records RWKV7 runner provenance and evaluates the canonical
faster3a throughput contract as structured JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
BENCHMARK_NAME = "rwkv7_faster3a"
ALBATROSS_REPO = "https://github.com/BlinkDL/Albatross"
ALBATROSS_COMMIT = "6af325aba3ee477bc1f59ef506375da550c2ef74"
ALBATROSS_RACE_FIX = "ff144b6b11e01ac984ed05ca7f7af4dfdca97180"
ALBATROSS_2607_COMMIT = "63c53f4abf2cd891dd3a18c8f44f5b2cccc8c64b"
ALBATROSS_IMPL = "faster3a_2607"
RUNNER_FP16_THROUGHPUT_REQUIREMENTS = {
    "VLLM_RWKV7_WKV_MODE": "fp16",
    "VLLM_RWKV7_ALLOW_FP16_ACCUMULATION": "1",
    "VLLM_RWKV7_EMB_DEVICE": "gpu",
    "VLLM_RWKV7_SLOT_MAPPED_STATE": "1",
    "VLLM_RWKV7_RKV_MODE": "off",
    "VLLM_RWKV7_CMIX_SPARSE": "no-fc",
    "VLLM_RWKV7_LOW_RANK_WEIGHT": "both",
    "VLLM_RWKV7_ORIG_LINEAR_GROUPS": "none",
    "VLLM_USE_RAPID_SAMPLER": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
}
RUNNER_BASELINE_TOKENS_PER_S = 9712.562
RUNNER_MAX_REGRESSION_PCT = 1.0
RUNNER_MIN_TOKENS_PER_S = RUNNER_BASELINE_TOKENS_PER_S * (
    1.0 - RUNNER_MAX_REGRESSION_PCT / 100.0
)
VLLM_RUNNER_MODE = "worker_execute_model"
VLLM_RUNNER_TIMING_TARGET = "worker.execute_model + worker.sample_tokens"
VLLM_RUNNER_TIMING_CLOCK = "cuda_event"
DEFAULT_RUNNER_PREFILL_CHUNK_TOKENS = 16
VLLM_RUNNER_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "ignore_eos": True,
    "detokenize": False,
}
BENCHMARK_ONLY_VLLM_ENV_VARS = ("VLLM_RWKV7_MODEL",)
PROVENANCE_ENV_VARS = (
    "VLLM_RWKV7_MODEL",
    *RUNNER_FP16_THROUGHPUT_REQUIREMENTS,
)
PROVENANCE_ENV_DEFAULTS = {
    "VLLM_RWKV7_WKV_MODE": "fp16",
    "VLLM_USE_RAPID_SAMPLER": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
}
ACCEPTANCE_THRESHOLDS = {
    "runner_steady_decode": {
        "min_runner_tokens_per_s": RUNNER_MIN_TOKENS_PER_S,
        "max_regression_pct": RUNNER_MAX_REGRESSION_PCT,
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
    batch_size: int
    prompt_len: int
    warmup_tokens: int
    decode_tokens: int
    runner_prefill_chunk_tokens: int = DEFAULT_RUNNER_PREFILL_CHUNK_TOKENS
    runner_enforce_eager: bool = False
    runner_cudagraph_capture_sizes: tuple[int, ...] | None = None


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https")


def _blocker(code: str, message: str, **details: Any) -> dict[str, Any]:
    blocker = {"code": code, "message": message}
    blocker.update({k: v for k, v in details.items() if v is not None})
    return blocker


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
    return {
        "albatross_repo": ALBATROSS_REPO,
        "albatross_commit": ALBATROSS_COMMIT,
        "albatross_changes": {
            "wkv_fp16_race_fix": ALBATROSS_RACE_FIX,
            "faster3a_2607": ALBATROSS_2607_COMMIT,
        },
        "contracts": [
            {
                "source_path": entry.source_path,
                "target_path": entry.target_path,
                "correspondence": entry.correspondence,
            }
            for entry in SOURCE_PROVENANCE
        ],
    }


def _source_revision_file(repo_root: Path) -> Path | None:
    for path in (repo_root, *repo_root.parents):
        marker = path / ".helicopter-source-revision"
        if marker.is_file():
            return marker
    return None


def _synced_revision(repo_root: Path) -> str | None:
    repo_root = repo_root.resolve()
    for workspace in repo_root.parents:
        manifest = workspace / ".helicopter-dev/source-revisions.json"
        if not manifest.is_file():
            continue
        try:
            relative = repo_root.relative_to(workspace).as_posix()
            revisions = json.loads(manifest.read_text(encoding="utf-8"))
            revision = revisions["submodules"].get(relative)
        except (KeyError, OSError, ValueError):
            continue
        if revision:
            return str(revision)
    return None


def _git_revision(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        revision = _synced_revision(repo_root)
        if revision:
            return revision
        marker = _source_revision_file(repo_root)
        if marker is None:
            return None
        revision = marker.read_text(encoding="utf-8").strip()
        return revision or None
    lines = result.stdout.splitlines()
    if len(lines) != 2:
        return None
    git_root, revision = Path(lines[0]).resolve(), lines[1].strip()
    if git_root != repo_root.resolve():
        return _synced_revision(repo_root)
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return f"{revision}-dirty" if status.stdout.strip() else revision


def _cuda_device_metadata() -> dict[str, Any]:
    if not _cuda_available():
        return {"available": False}
    cuda = _cuda_module()
    device_index = cuda.current_device()
    props = cuda.get_device_properties(device_index)
    device_uuid = getattr(props, "uuid", None)
    return {
        "available": True,
        "device_index": int(device_index),
        "device_uuid": str(device_uuid) if device_uuid is not None else None,
        "device_name": cuda.get_device_name(device_index),
        "capability": list(cuda.get_device_capability(device_index)),
        "total_memory": int(props.total_memory),
    }


def _rwkv_environment_raw_metadata() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in PROVENANCE_ENV_VARS}


def _rwkv_environment_metadata() -> dict[str, str | None]:
    raw = _rwkv_environment_raw_metadata()
    resolved = {
        name: raw[name] if raw[name] is not None else PROVENANCE_ENV_DEFAULTS.get(name)
        for name in PROVENANCE_ENV_VARS
    }
    return resolved


@contextmanager
def _without_benchmark_only_vllm_env_vars():
    saved = {
        name: os.environ.pop(name)
        for name in BENCHMARK_ONLY_VLLM_ENV_VARS
        if name in os.environ
    }
    try:
        yield
    finally:
        os.environ.update(saved)


def _benchmark_provenance(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "git_revision": _git_revision(config.repo_root),
        "cuda": _cuda_device_metadata(),
        "env": _rwkv_environment_metadata(),
        "raw_env": _rwkv_environment_raw_metadata(),
        "workload": {
            "batch_size": config.batch_size,
            "prompt_len": config.prompt_len,
            "warmup_tokens": config.warmup_tokens,
            "decode_tokens": config.decode_tokens,
            "runner_prefill_chunk_tokens": config.runner_prefill_chunk_tokens,
            "runner_enforce_eager": config.runner_enforce_eager,
            "runner_cudagraph_capture_sizes": (
                list(config.runner_cudagraph_capture_sizes)
                if config.runner_cudagraph_capture_sizes is not None
                else None
            ),
        },
        "sampling": dict(VLLM_RUNNER_SAMPLING),
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

    return blockers


def _get_number(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    return float(value)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _duration_ms_summary(durations_s: list[float]) -> dict[str, float | None]:
    if not durations_s:
        return {f"p{p}_ms": None for p in (10, 50, 90)}
    return {
        f"p{p}_ms": _percentile(durations_s, p / 100) * 1000.0
        for p in (10, 50, 90)
    }


def _tokens_per_second(tokens: int, duration_s: float) -> float:
    return float("inf") if duration_s <= 0 else tokens / duration_s


def _phase_throughput_summary(
    *,
    total_tokens: int,
    iteration_durations_s: list[float],
    unit_durations_s: list[float],
    unit_tokens: list[int],
) -> dict[str, Any]:
    total_duration_s = sum(iteration_durations_s)
    iteration_tokens = total_tokens // len(iteration_durations_s) if iteration_durations_s else 0
    peak_iteration = (
        max(_tokens_per_second(iteration_tokens, d) for d in iteration_durations_s)
        if iteration_durations_s
        else None
    )
    peak_unit = (
        max(_tokens_per_second(tokens, d) for tokens, d in zip(unit_tokens, unit_durations_s))
        if unit_durations_s
        else None
    )
    summary = _duration_ms_summary(iteration_durations_s)
    unit = _duration_ms_summary(unit_durations_s)
    return {
        "avg_tokens_per_s": _tokens_per_second(total_tokens, total_duration_s),
        "peak_tokens_per_s": peak_iteration,
        "peak_iteration_tokens_per_s": peak_iteration,
        "peak_unit_tokens_per_s": peak_unit,
        "total_tokens": total_tokens,
        "total_duration_ms": total_duration_s * 1000,
        **summary,
        "unit_p10_ms": unit["p10_ms"],
        "unit_p50_ms": unit["p50_ms"],
        "unit_p90_ms": unit["p90_ms"],
    }


def _required_vllm_runner_model(config: BenchmarkConfig) -> str:
    if not config.model:
        raise ValueError("Set --model or VLLM_RWKV7_MODEL.")
    if not _is_url(config.model) and not Path(config.model).expanduser().exists():
        raise FileNotFoundError(f"missing vLLM model path: {config.model}")
    return config.model


def _runner_prefill_chunk_tokens(config: BenchmarkConfig) -> int:
    if config.runner_prefill_chunk_tokens <= 0:
        raise ValueError("runner prefill chunk tokens must be positive")
    return min(config.prompt_len, config.runner_prefill_chunk_tokens)


def _create_vllm_runner_llm(config: BenchmarkConfig) -> Any:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

    import vllm.rwkv7_ops  # noqa: F401

    from vllm import LLM

    max_model_len = max(1, config.prompt_len + config.decode_tokens)
    max_num_seqs = max(1, config.batch_size)
    prefill_chunk_tokens = _runner_prefill_chunk_tokens(config)
    max_num_batched_tokens = max(
        config.batch_size * prefill_chunk_tokens,
        config.batch_size,
    )
    llm_kwargs: dict[str, Any] = {}
    if (
        not config.runner_enforce_eager
        and config.runner_cudagraph_capture_sizes is not None
    ):
        capture_sizes = list(config.runner_cudagraph_capture_sizes)
        if not capture_sizes or any(size <= 0 for size in capture_sizes):
            raise ValueError("runner cudagraph capture sizes must be positive")
        llm_kwargs["compilation_config"] = {
            "cudagraph_capture_sizes": capture_sizes,
        }
    with _without_benchmark_only_vllm_env_vars():
        return LLM(
            model=_required_vllm_runner_model(config),
            skip_tokenizer_init=True,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            long_prefill_token_threshold=prefill_chunk_tokens,
            enable_chunked_prefill=True,
            enforce_eager=config.runner_enforce_eager,
            disable_log_stats=True,
            **llm_kwargs,
        )


def _synchronize_cuda_if_available() -> None:
    try:
        import torch
    except Exception:
        return
    if _cuda_available():
        torch.accelerator.synchronize()


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


def _worker_time_call(
    fn: Any,
    cuda_events: tuple[Any, Any] | None,
) -> float:
    if cuda_events is None:
        import time

        _worker_cuda_synchronize()
        start_s = time.perf_counter()
        fn()
        _worker_cuda_synchronize()
        return time.perf_counter() - start_s

    start_event, end_event = cuda_events
    start_event.record()
    fn()
    end_event.record()
    end_event.synchronize()
    return start_event.elapsed_time(end_event) / 1000.0


def _worker_time_execute_model(
    worker: Any,
    scheduler_output: Any,
    cuda_events: tuple[Any, Any] | None,
) -> float:
    return _worker_time_call(
        lambda: worker.execute_model(scheduler_output),
        cuda_events,
    )


def _worker_time_sample_tokens(
    worker: Any,
    grammar_output: Any,
    cuda_events: tuple[Any, Any] | None,
) -> float:
    return _worker_time_call(
        lambda: worker.sample_tokens(grammar_output),
        cuda_events,
    )


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
    num_scheduled_tokens: int,
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
        num_scheduled_tokens={req_id: num_scheduled_tokens for req_id in req_ids},
        total_num_scheduled_tokens=len(req_ids) * num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _worker_add_decode_requests_scheduler_output(
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
                num_computed_tokens=prompt_len,
                lora_request=None,
                prefill_token_ids=prompt_ids,
            )
            for req_id, prompt_ids in zip(req_ids, prompt_token_ids)
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _worker_cached_prefill_scheduler_output(
    *,
    req_ids: list[str],
    num_computed_tokens: int,
    num_scheduled_tokens: int,
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
            num_computed_tokens=[num_computed_tokens] * len(req_ids),
            num_output_tokens=[0] * len(req_ids),
        ),
        num_scheduled_tokens={req_id: num_scheduled_tokens for req_id in req_ids},
        total_num_scheduled_tokens=len(req_ids) * num_scheduled_tokens,
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


def _worker_prompt_token_ids(batch_size: int, prompt_len: int) -> list[list[int]]:
    return [
        [(idx + position) % 1024 for position in range(prompt_len)]
        for idx in range(batch_size)
    ]


def _worker_execute_prefill_chunks(
    worker: Any,
    *,
    req_ids: list[str],
    prompt_token_ids: list[list[int]],
    sampling_params: Any,
    prompt_len: int,
    prefill_chunk_tokens: int,
    cuda_events: tuple[Any, Any] | None = None,
    measure: bool = False,
) -> tuple[list[float], list[int]]:
    first_chunk_tokens = min(prefill_chunk_tokens, prompt_len)
    prefill_output = _worker_new_request_scheduler_output(
        req_ids=req_ids,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        num_scheduled_tokens=first_chunk_tokens,
    )
    chunk_durations_s: list[float] = []
    chunk_token_counts: list[int] = []
    if measure:
        chunk_durations_s.append(
            _worker_time_execute_model(worker, prefill_output, cuda_events)
        )
        chunk_token_counts.append(len(req_ids) * first_chunk_tokens)
    else:
        worker.execute_model(prefill_output)
    num_prefill_tokens = first_chunk_tokens
    while num_prefill_tokens < prompt_len:
        chunk_tokens = min(prefill_chunk_tokens, prompt_len - num_prefill_tokens)
        prefill_output = _worker_cached_prefill_scheduler_output(
            req_ids=req_ids,
            num_computed_tokens=num_prefill_tokens,
            num_scheduled_tokens=chunk_tokens,
        )
        if measure:
            chunk_durations_s.append(
                _worker_time_execute_model(worker, prefill_output, cuda_events)
            )
            chunk_token_counts.append(len(req_ids) * chunk_tokens)
        else:
            worker.execute_model(prefill_output)
        num_prefill_tokens += chunk_tokens
    return chunk_durations_s, chunk_token_counts


def _worker_finish_execute_without_sampling(worker: Any) -> None:
    model_runner = getattr(worker, "model_runner", None)
    execute_model_state = getattr(model_runner, "execute_model_state", None)
    if execute_model_state is None:
        return
    input_batch = getattr(execute_model_state, "input_batch", None)
    model_state = getattr(model_runner, "model_state", None)
    postprocess_state = getattr(model_state, "postprocess_state", None)
    if input_batch is not None and callable(postprocess_state):
        req_states = getattr(model_runner, "req_states", None)
        num_computed_tokens = getattr(req_states, "num_computed_tokens", None)
        num_computed_gpu = getattr(num_computed_tokens, "gpu", None)
        postprocess_state(input_batch.idx_mapping, 0, num_computed_gpu)
    model_runner.execute_model_state = None


def _run_vllm_worker_internal_prefill(
    worker: Any,
    batch_size: int,
    prompt_len: int,
    prefill_chunk_tokens: int,
    warmup: int,
    iters: int,
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
    if prefill_chunk_tokens <= 0:
        raise ValueError("runner prefill chunk tokens must be positive")
    if warmup < 0:
        raise ValueError("runner warmup must be non-negative")
    prefill_chunk_tokens = min(prefill_chunk_tokens, prompt_len)

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=VLLM_RUNNER_SAMPLING["temperature"],
        top_p=VLLM_RUNNER_SAMPLING["top_p"],
        ignore_eos=VLLM_RUNNER_SAMPLING["ignore_eos"],
        detokenize=VLLM_RUNNER_SAMPLING["detokenize"],
    )
    prefix = f"rwkv7-prefill-{id(worker)}-{time.perf_counter_ns()}"
    cuda_events = _worker_cuda_event_pair()
    timing_clock = "cuda_event" if cuda_events is not None else "wall_clock"
    iteration_durations_s: list[float] = []
    unit_durations_s: list[float] = []
    unit_tokens: list[int] = []

    for warmup_iteration in range(warmup):
        req_ids = [
            f"{prefix}-warmup-{warmup_iteration}-{idx}" for idx in range(batch_size)
        ]
        _worker_execute_prefill_chunks(
            worker,
            req_ids=req_ids,
            prompt_token_ids=_worker_prompt_token_ids(batch_size, prompt_len),
            sampling_params=sampling_params,
            prompt_len=prompt_len,
            prefill_chunk_tokens=prefill_chunk_tokens,
            cuda_events=cuda_events,
            measure=False,
        )
        _worker_finish_execute_without_sampling(worker)
        worker.execute_model(_worker_empty_scheduler_output(set(req_ids)))
    if warmup:
        _worker_cuda_synchronize()

    for iteration in range(iters):
        req_ids = [f"{prefix}-measure-{iteration}-{idx}" for idx in range(batch_size)]
        chunk_durations_s, chunk_token_counts = _worker_execute_prefill_chunks(
            worker,
            req_ids=req_ids,
            prompt_token_ids=_worker_prompt_token_ids(batch_size, prompt_len),
            sampling_params=sampling_params,
            prompt_len=prompt_len,
            prefill_chunk_tokens=prefill_chunk_tokens,
            cuda_events=cuda_events,
            measure=True,
        )
        iteration_durations_s.append(sum(chunk_durations_s))
        unit_durations_s.extend(chunk_durations_s)
        unit_tokens.extend(chunk_token_counts)
        _worker_finish_execute_without_sampling(worker)
        worker.execute_model(_worker_empty_scheduler_output(set(req_ids)))

    return {
        "measurement_mode": VLLM_RUNNER_MODE,
        "internal_timing_target": "worker.execute_model.prefill",
        "timing_clock": timing_clock,
        "iteration_durations_s": iteration_durations_s,
        "unit_durations_s": unit_durations_s,
        "unit_tokens": unit_tokens,
        "tokens": batch_size * prompt_len * iters,
        "warmup_iterations": warmup,
        "worker_count": 1,
    }


def _run_vllm_worker_internal_decode_only(
    worker: Any,
    batch_size: int,
    prompt_len: int,
    prefill_chunk_tokens: int,
    decode_tokens: int,
    warmup_decode_tokens: int,
    iters: int,
    include_sampling: bool,
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
    if prefill_chunk_tokens <= 0:
        raise ValueError("runner prefill chunk tokens must be positive")
    if prompt_len <= 0:
        raise ValueError("runner decode prompt len must be positive")
    if decode_tokens <= 0:
        raise ValueError("runner decode tokens must be positive")
    if warmup_decode_tokens < 0:
        raise ValueError("runner decode warmup tokens must be non-negative")

    from vllm import SamplingParams

    scheduled_decode_tokens = decode_tokens + warmup_decode_tokens
    sampling_params = SamplingParams(
        max_tokens=scheduled_decode_tokens,
        temperature=VLLM_RUNNER_SAMPLING["temperature"],
        top_p=VLLM_RUNNER_SAMPLING["top_p"],
        ignore_eos=VLLM_RUNNER_SAMPLING["ignore_eos"],
        detokenize=VLLM_RUNNER_SAMPLING["detokenize"],
    )
    prefix = f"rwkv7-decode-{id(worker)}-{time.perf_counter_ns()}"
    cuda_events = _worker_cuda_event_pair()
    timing_clock = "cuda_event" if cuda_events is not None else "wall_clock"
    iteration_durations_s: list[float] = []
    unit_durations_s: list[float] = []
    unit_tokens: list[int] = []
    sample_durations_s: list[float] = []

    for iteration in range(iters):
        req_ids = [f"{prefix}-{iteration}-{idx}" for idx in range(batch_size)]
        worker.execute_model(
            _worker_add_decode_requests_scheduler_output(
                req_ids=req_ids,
                prompt_token_ids=_worker_prompt_token_ids(batch_size, prompt_len),
                sampling_params=sampling_params,
                prompt_len=prompt_len,
            )
        )
        _worker_finish_execute_without_sampling(worker)
        _worker_cuda_synchronize()

        for step in range(warmup_decode_tokens):
            worker.execute_model(
                _worker_cached_decode_scheduler_output(
                    req_ids=req_ids,
                    prompt_len=prompt_len,
                    step=step,
                )
            )
            _worker_finish_execute_without_sampling(worker)
        if warmup_decode_tokens:
            _worker_cuda_synchronize()

        iteration_duration_s = 0.0
        for step in range(warmup_decode_tokens, scheduled_decode_tokens):
            decode_output = _worker_cached_decode_scheduler_output(
                req_ids=req_ids,
                prompt_len=prompt_len,
                step=step,
            )
            execute_duration_s = _worker_time_execute_model(
                worker,
                decode_output,
                cuda_events,
            )
            iteration_duration_s += execute_duration_s
            unit_durations_s.append(execute_duration_s)
            unit_tokens.append(batch_size)
            if include_sampling:
                sample_duration_s = _worker_time_sample_tokens(
                    worker,
                    None,
                    cuda_events,
                )
                iteration_duration_s += sample_duration_s
                sample_durations_s.append(sample_duration_s)
            else:
                _worker_finish_execute_without_sampling(worker)
        iteration_durations_s.append(iteration_duration_s)
        worker.execute_model(_worker_empty_scheduler_output(set(req_ids)))

    return {
        "measurement_mode": VLLM_RUNNER_MODE,
        "internal_timing_target": (
            "worker.execute_model+sample_tokens.decode"
            if include_sampling
            else "worker.execute_model.decode"
        ),
        "timing_clock": timing_clock,
        "iteration_durations_s": iteration_durations_s,
        "unit_durations_s": unit_durations_s,
        "unit_tokens": unit_tokens,
        "sample_durations_s": sample_durations_s,
        "tokens": batch_size * decode_tokens * iters,
        "worker_count": 1,
    }


def _run_vllm_worker_internal_steady_decode(
    worker: Any,
    batch_size: int,
    prompt_len: int,
    prefill_chunk_tokens: int,
    decode_tokens: int,
    iters: int,
    measure: bool,
    warmup_decode_tokens: int = 0,
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

    if warmup_decode_tokens < 0:
        raise ValueError("runner decode warmup tokens must be non-negative")
    scheduled_decode_tokens = decode_tokens + warmup_decode_tokens

    sampling_params = SamplingParams(
        max_tokens=scheduled_decode_tokens,
        temperature=VLLM_RUNNER_SAMPLING["temperature"],
        top_p=VLLM_RUNNER_SAMPLING["top_p"],
        ignore_eos=VLLM_RUNNER_SAMPLING["ignore_eos"],
        detokenize=VLLM_RUNNER_SAMPLING["detokenize"],
    )
    iteration_durations_s: list[float] = []
    execute_durations_s: list[float] = []
    sample_durations_s: list[float] = []
    decode_step_durations_s: list[float] = []
    postprocess_durations_s: list[float] = []
    prefix = f"rwkv7-runner-{id(worker)}-{time.perf_counter_ns()}"
    cuda_events = _worker_cuda_event_pair() if measure else None
    timing_clock = "cuda_event" if cuda_events is not None else "wall_clock"
    if prefill_chunk_tokens <= 0:
        raise ValueError("runner prefill chunk tokens must be positive")
    prefill_chunk_tokens = min(prefill_chunk_tokens, prompt_len)

    for iteration in range(iters):
        req_ids = [f"{prefix}-{iteration}-{idx}" for idx in range(batch_size)]
        _worker_execute_prefill_chunks(
            worker,
            req_ids=req_ids,
            prompt_token_ids=_worker_prompt_token_ids(batch_size, prompt_len),
            sampling_params=sampling_params,
            prompt_len=prompt_len,
            prefill_chunk_tokens=prefill_chunk_tokens,
            measure=False,
        )
        worker.sample_tokens(None)
        _worker_cuda_synchronize()

        for step in range(warmup_decode_tokens):
            worker.execute_model(
                _worker_cached_decode_scheduler_output(
                    req_ids=req_ids,
                    prompt_len=prompt_len,
                    step=step,
                )
            )
            worker.sample_tokens(None)
        if warmup_decode_tokens:
            _worker_cuda_synchronize()

        if measure:
            iteration_duration_s = 0.0
        for step in range(warmup_decode_tokens, scheduled_decode_tokens):
            decode_output = _worker_cached_decode_scheduler_output(
                req_ids=req_ids,
                prompt_len=prompt_len,
                step=step,
            )
            if measure:
                execute_duration_s = _worker_time_execute_model(
                    worker,
                    decode_output,
                    cuda_events,
                )
                sample_duration_s = _worker_time_sample_tokens(
                    worker,
                    None,
                    cuda_events,
                )
                step_duration_s = execute_duration_s + sample_duration_s
                execute_durations_s.append(execute_duration_s)
                sample_durations_s.append(sample_duration_s)
                decode_step_durations_s.append(step_duration_s)
                iteration_duration_s += step_duration_s
            else:
                worker.execute_model(decode_output)
                worker.sample_tokens(None)

        if measure:
            iteration_durations_s.append(iteration_duration_s)

        worker.execute_model(_worker_empty_scheduler_output(set(req_ids)))

    return {
        "measurement_mode": VLLM_RUNNER_MODE,
        "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
        "iteration_durations_s": iteration_durations_s,
        "execute_durations_s": execute_durations_s,
        "sample_durations_s": sample_durations_s,
        "decode_step_durations_s": decode_step_durations_s,
        "postprocess_durations_s": postprocess_durations_s,
        "postprocess_timing_available": False,
        "decode_steps": decode_tokens * iters if measure else 0,
        "warmup_decode_steps": warmup_decode_tokens * iters,
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

    expected_decode_steps = decode_tokens * iters
    duration_specs = (
        ("iteration_durations_s", iters),
        ("execute_durations_s", expected_decode_steps),
        ("sample_durations_s", expected_decode_steps),
        ("decode_step_durations_s", expected_decode_steps),
    )
    merged_durations: dict[str, list[float]] = {}
    for key, expected_count in duration_specs:
        per_worker_values = [
            [float(value) for value in result.get(key, [])]
            for result in normalized_results
        ]
        if any(len(values) != expected_count for values in per_worker_values):
            return _worker_internal_runner_blocker(
                "missing_internal_runner_decode_samples",
                "No complete internal worker decode timing samples were recorded.",
            )
        merged_durations[key] = [
            max(worker_values[index] for worker_values in per_worker_values)
            for index in range(expected_count)
        ]

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

    iteration_durations_s = merged_durations["iteration_durations_s"]
    execute_durations_s = merged_durations["execute_durations_s"]
    sample_durations_s = merged_durations["sample_durations_s"]
    decode_step_durations_s = merged_durations["decode_step_durations_s"]
    p50_s = _percentile(iteration_durations_s, 0.5)
    execute_p50_s = _percentile(execute_durations_s, 0.5)
    sample_p50_s = _percentile(sample_durations_s, 0.5)
    decode_step_p50_s = _percentile(decode_step_durations_s, 0.5)
    return {
        "tokens_per_s": (batch_size * decode_tokens) / p50_s,
        "p10_ms": _percentile(iteration_durations_s, 0.1) * 1000.0,
        "p50_ms": p50_s * 1000.0,
        "p90_ms": _percentile(iteration_durations_s, 0.9) * 1000.0,
        "execute_model_p50_ms": execute_p50_s * 1000.0,
        "execute_model_p50_tokens_per_s": batch_size / execute_p50_s,
        "sample_tokens_p50_ms": sample_p50_s * 1000.0,
        "sample_tokens_p50_tokens_per_s": batch_size / sample_p50_s,
        "decode_step_p50_ms": decode_step_p50_s * 1000.0,
        "decode_step_p50_tokens_per_s": batch_size / decode_step_p50_s,
        "postprocess_p50_ms": None,
        "postprocess_timing_available": False,
        "measurement_mode": VLLM_RUNNER_MODE,
        "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
        "decode_steps": decode_steps,
        "worker_count": len(normalized_results),
    }


def _merge_worker_internal_phase_results(
    worker_results: list[Any],
    *,
    total_tokens: int,
    expected_iterations: int,
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
    if any(len(values) != expected_iterations for values in per_worker_iterations):
        return _worker_internal_runner_blocker(
            "missing_internal_runner_phase_samples",
            "No complete internal worker phase timing samples were recorded.",
        )
    iteration_durations_s = [
        max(worker_values[index] for worker_values in per_worker_iterations)
        for index in range(expected_iterations)
    ]

    unit_durations_s: list[float] = []
    unit_tokens: list[int] = []
    worker_count = len(normalized_results)
    max_unit_count = max(
        len(result.get("unit_durations_s", [])) for result in normalized_results
    )
    for index in range(max_unit_count):
        worker_unit_durations: list[float] = []
        worker_unit_tokens: list[int] = []
        for result in normalized_results:
            durations = result.get("unit_durations_s", [])
            tokens = result.get("unit_tokens", [])
            if index < len(durations):
                worker_unit_durations.append(float(durations[index]))
                worker_unit_tokens.append(int(tokens[index]))
        if worker_unit_durations:
            unit_durations_s.append(max(worker_unit_durations))
            unit_tokens.append(sum(worker_unit_tokens) // max(1, worker_count))

    summary = _phase_throughput_summary(
        total_tokens=total_tokens,
        iteration_durations_s=iteration_durations_s,
        unit_durations_s=unit_durations_s,
        unit_tokens=unit_tokens,
    )
    first_result = normalized_results[0]
    summary.update(
        {
            "measurement_mode": first_result.get("measurement_mode", VLLM_RUNNER_MODE),
            "internal_timing_target": first_result.get(
                "internal_timing_target", VLLM_RUNNER_TIMING_TARGET
            ),
            "timing_clock": first_result.get("timing_clock", VLLM_RUNNER_TIMING_CLOCK),
            "worker_count": worker_count,
        }
    )
    if "warmup_iterations" in first_result:
        summary["warmup_iterations"] = first_result["warmup_iterations"]
    sample_durations = [
        float(value)
        for result in normalized_results
        for value in result.get("sample_durations_s", [])
    ]
    if sample_durations:
        sample_summary = _duration_ms_summary(sample_durations)
        summary["sample_tokens_p50_ms"] = sample_summary["p50_ms"]
    return summary


def _time_vllm_runner_steady_decode(
    llm: Any,
    *,
    batch_size: int,
    prompt_len: int,
    prefill_chunk_tokens: int,
    decode_tokens: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("runner batch size must be positive")
    if prompt_len <= 0:
        raise ValueError("runner prompt len must be positive")
    if prefill_chunk_tokens <= 0:
        raise ValueError("runner prefill chunk tokens must be positive")
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

    timed_results = collective_rpc(
        _run_vllm_worker_internal_steady_decode,
        args=(
            batch_size,
            prompt_len,
            prefill_chunk_tokens,
            decode_tokens,
            iters,
            True,
            warmup,
        ),
    )
    return _merge_worker_internal_runner_results(
        list(timed_results),
        batch_size=batch_size,
        decode_tokens=decode_tokens,
        iters=iters,
    )


def _time_vllm_runner_prefill_phase(
    llm: Any,
    *,
    batch_size: int,
    prompt_len: int,
    prefill_chunk_tokens: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    collective_rpc = getattr(llm, "collective_rpc", None)
    if not callable(collective_rpc):
        return _worker_internal_runner_blocker(
            "missing_collective_rpc",
            "The vLLM LLM object does not expose collective_rpc().",
        )
    results = collective_rpc(
        _run_vllm_worker_internal_prefill,
        args=(batch_size, prompt_len, prefill_chunk_tokens, warmup, iters),
    )
    return _merge_worker_internal_phase_results(
        list(results),
        total_tokens=batch_size * prompt_len * iters,
        expected_iterations=iters,
    )


def _time_vllm_runner_decode_phase(
    llm: Any,
    *,
    batch_size: int,
    prompt_len: int,
    prefill_chunk_tokens: int,
    decode_tokens: int,
    warmup: int,
    iters: int,
    include_sampling: bool,
) -> dict[str, Any]:
    collective_rpc = getattr(llm, "collective_rpc", None)
    if not callable(collective_rpc):
        return _worker_internal_runner_blocker(
            "missing_collective_rpc",
            "The vLLM LLM object does not expose collective_rpc().",
        )
    results = collective_rpc(
        _run_vllm_worker_internal_decode_only,
        args=(
            batch_size,
            prompt_len,
            prefill_chunk_tokens,
            decode_tokens,
            warmup,
            iters,
            include_sampling,
        ),
    )
    return _merge_worker_internal_phase_results(
        list(results),
        total_tokens=batch_size * decode_tokens * iters,
        expected_iterations=iters,
    )


def _shutdown_vllm_runner_llm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if not callable(shutdown):
        engine_core = getattr(engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown(timeout=30)
        except TypeError:
            shutdown()
    with suppress(Exception):
        llm.llm_engine = None
    gc.collect()
    if _cuda_available():
        cuda = _cuda_module()
        cuda.empty_cache()


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
    prefill_chunk_tokens = _runner_prefill_chunk_tokens(runner_config)
    capacity_config = replace(
        runner_config,
        decode_tokens=decode_tokens + warmup,
    )
    llm = _create_vllm_runner_llm(capacity_config)
    try:
        parsed = _time_vllm_runner_steady_decode(
            llm,
            batch_size=batch_size,
            prompt_len=prompt_len,
            prefill_chunk_tokens=prefill_chunk_tokens,
            decode_tokens=decode_tokens,
            warmup=warmup,
            iters=iters,
        )
    finally:
        _shutdown_vllm_runner_llm(llm)

    runner_metrics: dict[str, Any] = {
        "runner_batch_size": batch_size,
        "runner_prompt_len": prompt_len,
        "runner_prefill_chunk_tokens": prefill_chunk_tokens,
        "runner_decode_tokens": decode_tokens,
        "runner_warmup": warmup,
        "runner_warmup_mode": "same_request_decode_steps",
        "runner_warmup_decode_tokens": warmup,
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
                "runner_execute_model_p50_ms": parsed.get("execute_model_p50_ms"),
                "runner_execute_model_p50_tokens_per_s": parsed.get(
                    "execute_model_p50_tokens_per_s"
                ),
                "runner_sample_tokens_p50_ms": parsed.get("sample_tokens_p50_ms"),
                "runner_sample_tokens_p50_tokens_per_s": parsed.get(
                    "sample_tokens_p50_tokens_per_s"
                ),
                "runner_decode_step_p50_ms": parsed.get("decode_step_p50_ms"),
                "runner_decode_step_p50_tokens_per_s": parsed.get(
                    "decode_step_p50_tokens_per_s"
                ),
                "runner_postprocess_p50_ms": parsed.get("postprocess_p50_ms"),
                "runner_postprocess_timing_available": parsed.get(
                    "postprocess_timing_available",
                    False,
                ),
                "runner_decode_steps": parsed["decode_steps"],
                "runner_worker_count": parsed["worker_count"],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "runner_steady_decode": runner_metrics,
        "config": {
            "repo_root": str(config.repo_root),
            "model": config.model,
            "measurement_source": f"vllm_runner_{VLLM_RUNNER_MODE}",
            "provenance": _benchmark_provenance(runner_config),
        },
    }


def _runner_throughput_contract_blocker(
    measurements: dict[str, Any],
) -> dict[str, Any] | None:
    config = measurements.get("config")
    provenance = config.get("provenance") if isinstance(config, dict) else None
    raw_env = provenance.get("raw_env") if isinstance(provenance, dict) else None
    if not isinstance(raw_env, dict) or any(
        name not in raw_env for name in RUNNER_FP16_THROUGHPUT_REQUIREMENTS
    ):
        return _blocker(
            "missing_runner_throughput_provenance",
            "Runner performance acceptance requires retained precision provenance.",
        )

    violations = {
        name: {
            "required": required,
            "actual": raw_env[name],
        }
        for name, required in RUNNER_FP16_THROUGHPUT_REQUIREMENTS.items()
        if raw_env[name] != required
    }
    if violations:
        return _blocker(
            "invalid_runner_throughput_contract",
            "Runner performance acceptance requires the FP16 throughput contract.",
            violations=violations,
        )
    return None


def _evaluate_runner(
    measurements: dict[str, Any] | None,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        "runner_tokens_per_s": None,
        "runner_measurement_mode": None,
        "runner_internal_timing_target": None,
        "runner_timing_clock": None,
        "runner_execute_model_p50_ms": None,
        "runner_execute_model_p50_tokens_per_s": None,
        "runner_sample_tokens_p50_ms": None,
        "runner_sample_tokens_p50_tokens_per_s": None,
        "runner_decode_step_p50_ms": None,
        "runner_decode_step_p50_tokens_per_s": None,
        "runner_postprocess_p50_ms": None,
        "runner_postprocess_timing_available": None,
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

    metrics["runner_tokens_per_s"] = runner_tps
    contract_blocker = _runner_throughput_contract_blocker(measurements)
    if contract_blocker is not None:
        check["blockers"] = [contract_blocker]
        return check

    metrics.update(
        {
            "runner_tokens_per_s": runner_tps,
            "runner_measurement_mode": raw_metrics.get("runner_measurement_mode"),
            "runner_internal_timing_target": raw_metrics.get(
                "runner_internal_timing_target"
            ),
            "runner_timing_clock": raw_metrics.get("runner_timing_clock"),
            "runner_execute_model_p50_ms": raw_metrics.get(
                "runner_execute_model_p50_ms"
            ),
            "runner_execute_model_p50_tokens_per_s": raw_metrics.get(
                "runner_execute_model_p50_tokens_per_s"
            ),
            "runner_sample_tokens_p50_ms": raw_metrics.get(
                "runner_sample_tokens_p50_ms"
            ),
            "runner_sample_tokens_p50_tokens_per_s": raw_metrics.get(
                "runner_sample_tokens_p50_tokens_per_s"
            ),
            "runner_decode_step_p50_ms": raw_metrics.get("runner_decode_step_p50_ms"),
            "runner_decode_step_p50_tokens_per_s": raw_metrics.get(
                "runner_decode_step_p50_tokens_per_s"
            ),
            "runner_postprocess_p50_ms": raw_metrics.get("runner_postprocess_p50_ms"),
            "runner_postprocess_timing_available": raw_metrics.get(
                "runner_postprocess_timing_available"
            ),
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
    )
    runner_check = _evaluate_runner(measurements, runtime_blockers)
    status = runner_check["status"]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "overall_status": status,
        "source": _source_metadata(config),
        "config": {
            "repo_root": str(config.repo_root),
            "model": config.model,
            "batch_size": config.batch_size,
            "prompt_len": config.prompt_len,
            "warmup_tokens": config.warmup_tokens,
            "decode_tokens": config.decode_tokens,
            "cuda_available": cuda,
            "measurement_source": "json" if measurements is not None else None,
            "provenance": _benchmark_provenance(config),
        },
        "acceptance": ACCEPTANCE_THRESHOLDS,
        "checks": {"runner_steady_decode": runner_check},
    }


def _default_repo_root() -> Path:
    return REPO_ROOT


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    model = args.model or os.environ.get("VLLM_RWKV7_MODEL") or None
    return BenchmarkConfig(
        repo_root=args.repo_root.resolve(),
        model=model,
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        warmup_tokens=args.warmup_tokens,
        decode_tokens=args.decode_tokens,
        runner_prefill_chunk_tokens=args.runner_prefill_chunk_tokens,
        runner_enforce_eager=args.runner_enforce_eager,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RWKV7 faster3a performance acceptance harness."
    )
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--model", help="vLLM-loadable RWKV7 model path or URL.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--measurement-json", type=Path)
    parser.add_argument(
        "--measure-vllm-runner",
        action="store_true",
        help="Run the canonical vLLM RWKV7 runner throughput benchmark.",
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
        "--runner-prefill-chunk-tokens",
        type=int,
        default=DEFAULT_RUNNER_PREFILL_CHUNK_TOKENS,
        help=(
            "Maximum prompt tokens scheduled per request for synthetic vLLM "
            "runner prefill."
        ),
    )
    parser.add_argument(
        "--runner-enforce-eager",
        action="store_true",
        help="Disable vLLM CUDA graph capture for runner measurements.",
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
        help=(
            "Unmeasured same-request decode steps before timing each vLLM "
            "runner steady decode iteration."
        ),
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


def _measurement_exit_code(measurement: dict[str, Any]) -> int:
    metrics = measurement.get("runner_steady_decode")
    if not isinstance(metrics, dict):
        return 2
    if metrics.get("blockers"):
        return 2
    if _runner_throughput_contract_blocker(measurement) is not None:
        return 2
    try:
        tokens_per_s = float(metrics["runner_tokens_per_s"])
    except (KeyError, TypeError, ValueError):
        return 2
    return int(
        not math.isfinite(tokens_per_s)
        or tokens_per_s < ACCEPTANCE_THRESHOLDS["runner_steady_decode"][
            "min_runner_tokens_per_s"
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config = _config_from_args(args)
    if args.measure_vllm_runner:
        measurement = generate_vllm_runner_measurement(
            config,
            batch_size=args.runner_batch_size,
            prompt_len=args.runner_prompt_len,
            decode_tokens=args.runner_decode_tokens,
            warmup=args.runner_warmup,
            iters=args.runner_iters,
        )
        _write_report(measurement, args.measurement_output)
        return _measurement_exit_code(measurement)
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
