#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)

FP32_ENV_KEY = "VLLM_USE_LOGITS_FP32_COMPUTATION"
CORRECTNESS_NUM_TOKENS = 16
BENCHMARK_NUM_TOKENS = 32
HIDDEN_SIZE = 4096
VOCAB_SIZE = 262144
CORRECTNESS_WARMUP_ITERATIONS = 1
WARMUP_ITERATIONS = 2
MEASURED_ITERATIONS = 8
ATOL = 5e-1
RTOL = 5e-2


class _IdentityGatherLogitsProcessor(LogitsProcessor):
    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for lm-head fp32 projection checks")
    return torch.device("cuda:0")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_fixture(
    num_tokens: int,
) -> tuple[torch.Tensor, VocabParallelEmbedding, LogitsProcessor]:
    device = _require_cuda()
    torch.manual_seed(1234)

    hidden_states = (torch.randn(num_tokens, HIDDEN_SIZE, device=device) * 0.25).to(
        torch.bfloat16
    )
    with (
        patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        lm_head = VocabParallelEmbedding(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            params_dtype=torch.bfloat16,
        )
    lm_head = lm_head.to(device=device)
    lm_head.weight.data.copy_(
        (torch.randn_like(lm_head.weight, device=device) * 0.25).to(torch.bfloat16)
    )
    logits_processor = _IdentityGatherLogitsProcessor(
        vocab_size=VOCAB_SIZE,
        org_vocab_size=VOCAB_SIZE,
    )
    return hidden_states, lm_head, logits_processor


@contextmanager
def _fp32_logits_env(enabled: bool):
    original_value = os.environ.get(FP32_ENV_KEY)
    os.environ[FP32_ENV_KEY] = "1" if enabled else "0"
    envs.disable_envs_cache()
    try:
        yield
    finally:
        if original_value is None:
            os.environ.pop(FP32_ENV_KEY, None)
        else:
            os.environ[FP32_ENV_KEY] = original_value
        envs.disable_envs_cache()


def _project_logits(
    *,
    fp32_enabled: bool,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    logits_processor: LogitsProcessor,
) -> torch.Tensor:
    with _fp32_logits_env(fp32_enabled):
        return logits_processor(lm_head, hidden_states)


def _warmup_projection(
    *,
    fp32_enabled: bool,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    logits_processor: LogitsProcessor,
    warmup_iterations: int,
) -> None:
    device = hidden_states.device
    with _fp32_logits_env(fp32_enabled):
        for _ in range(warmup_iterations):
            logits_processor(lm_head, hidden_states)
        _synchronize(device)


def _benchmark_projection(
    *,
    fp32_enabled: bool,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    logits_processor: LogitsProcessor,
    warmup_iterations: int,
    measured_iterations: int,
) -> tuple[list[float], torch.Tensor]:
    device = hidden_states.device
    with _fp32_logits_env(fp32_enabled):
        for _ in range(warmup_iterations):
            logits_processor(lm_head, hidden_states)
        _synchronize(device)

        times_ms: list[float] = []
        latest_logits = logits_processor(lm_head, hidden_states).clone()
        for _ in range(measured_iterations):
            _synchronize(device)
            start = time.perf_counter()
            latest_logits = logits_processor(lm_head, hidden_states).clone()
            _synchronize(device)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    return times_ms, latest_logits


def _benchmark_summary_path() -> Path:
    summary_file_name = os.environ.get("UNIT_SUMMARY_FILE_NAME")
    if not summary_file_name:
        raise OSError(
            "UNIT_SUMMARY_FILE_NAME env var must be set "
            "(exported by tests/cohere/scripts/run_tests.sh)"
        )
    output_dir = Path(os.environ.get("OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / summary_file_name


def _write_step_summary(summary: dict[str, object]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines = [
        "## LM Head FP32 Unit Benchmark",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| GPU | {summary['gpu_name']} |",
        f"| Shape | {summary['shape']} |",
        f"| Warmups | {summary['warmup_iterations']} |",
        f"| Measured iterations | {summary['measured_iterations']} |",
        f"| BF16 mean (ms) | {summary['bf16_mean_ms']:.4f} |",
        f"| FP32 mean (ms) | {summary['fp32_mean_ms']:.4f} |",
        f"| Relative slowdown | {summary['relative_slowdown']:.4f} |",
        f"| Max abs diff | {summary['max_abs_diff']:.6f} |",
        f"| Mean abs diff | {summary['mean_abs_diff']:.6f} |",
        "",
    ]
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def test_lm_head_fp32_projection_diff_is_small_but_nonzero(default_vllm_config) -> None:
    del default_vllm_config
    hidden_states, lm_head, logits_processor = _make_fixture(
        num_tokens=CORRECTNESS_NUM_TOKENS
    )

    _warmup_projection(
        fp32_enabled=False,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_processor=logits_processor,
        warmup_iterations=CORRECTNESS_WARMUP_ITERATIONS,
    )
    _warmup_projection(
        fp32_enabled=True,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_processor=logits_processor,
        warmup_iterations=CORRECTNESS_WARMUP_ITERATIONS,
    )
    bf16_logits = _project_logits(
        fp32_enabled=False,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_processor=logits_processor,
    )
    fp32_logits = _project_logits(
        fp32_enabled=True,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_processor=logits_processor,
    )

    assert bf16_logits.dtype == torch.bfloat16
    assert fp32_logits.dtype == torch.float32

    bf16_as_fp32 = bf16_logits.to(torch.float32)
    abs_diff = (fp32_logits - bf16_as_fp32).abs()
    max_abs_diff = float(abs_diff.max().item())

    assert not torch.equal(bf16_as_fp32, fp32_logits)
    assert max_abs_diff > 0.0
    torch.testing.assert_close(bf16_as_fp32, fp32_logits, atol=ATOL, rtol=RTOL)


def test_lm_head_fp32_projection_benchmark_writes_summary(default_vllm_config) -> None:
    del default_vllm_config
    hidden_states, lm_head, logits_processor = _make_fixture(
        num_tokens=BENCHMARK_NUM_TOKENS
    )
    device = hidden_states.device

    bf16_times_ms, bf16_logits = _benchmark_projection(
        fp32_enabled=False,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_processor=logits_processor,
        warmup_iterations=WARMUP_ITERATIONS,
        measured_iterations=MEASURED_ITERATIONS,
    )
    fp32_times_ms, fp32_logits = _benchmark_projection(
        fp32_enabled=True,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_processor=logits_processor,
        warmup_iterations=WARMUP_ITERATIONS,
        measured_iterations=MEASURED_ITERATIONS,
    )

    bf16_as_fp32 = bf16_logits.to(torch.float32)
    abs_diff = (fp32_logits - bf16_as_fp32).abs()
    bf16_mean_ms = statistics.fmean(bf16_times_ms)
    fp32_mean_ms = statistics.fmean(fp32_times_ms)
    summary = {
        "benchmark_name": "lm_head_fp32_projection_unit_benchmark",
        "compiled": False,
        "cudagraph_enabled": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "hidden_states_dtype": str(hidden_states.dtype),
        "weight_dtype": str(lm_head.weight.dtype),
        "bf16_output_dtype": str(bf16_logits.dtype),
        "fp32_output_dtype": str(fp32_logits.dtype),
        "shape": {
            "num_tokens": BENCHMARK_NUM_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
        },
        "warmup_iterations": WARMUP_ITERATIONS,
        "measured_iterations": MEASURED_ITERATIONS,
        "bf16_mean_ms": bf16_mean_ms,
        "bf16_median_ms": statistics.median(bf16_times_ms),
        "fp32_mean_ms": fp32_mean_ms,
        "fp32_median_ms": statistics.median(fp32_times_ms),
        "relative_slowdown": ((fp32_mean_ms / bf16_mean_ms) - 1.0),
        "max_abs_diff": float(abs_diff.max().item()),
        "mean_abs_diff": float(abs_diff.mean().item()),
    }

    summary_path = _benchmark_summary_path()
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_step_summary(summary)

    assert bf16_mean_ms > 0.0
    assert fp32_mean_ms > 0.0
    assert summary_path.exists()
