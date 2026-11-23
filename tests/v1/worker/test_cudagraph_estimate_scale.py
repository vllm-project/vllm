# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_CUDAGRAPH_ESTIMATE_SCALE: scaling of the profiled CUDA graph memory
estimate in Worker.determine_available_memory, with every device call mocked."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.config import CUDAGraphMode
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import (
    cudagraph_estimate_scale,
    log_cudagraph_estimate_scale,
    scale_cudagraph_estimate,
)

TOTAL_MEMORY = 288 * GiB_bytes
REQUESTED_MEMORY = 260 * GiB_bytes
NON_KV_CACHE_MEMORY = 200 * GiB_bytes
CUDAGRAPH_ESTIMATE = 16 * GiB_bytes


@pytest.fixture
def no_scale_env(monkeypatch):
    monkeypatch.delenv("VLLM_CUDAGRAPH_ESTIMATE_SCALE", raising=False)


@pytest.mark.usefixtures("no_scale_env")
def test_scale_defaults_to_one():
    assert cudagraph_estimate_scale() == 1.0


@pytest.mark.parametrize(("raw", "expected"), [("", 1.0), ("0.75", 0.75), ("1.2", 1.2)])
def test_scale_parses_env(monkeypatch, raw, expected):
    monkeypatch.setenv("VLLM_CUDAGRAPH_ESTIMATE_SCALE", raw)
    assert cudagraph_estimate_scale() == expected


@pytest.mark.parametrize("raw", ["abc", "0", "-0.5", "inf", "nan"])
def test_scale_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv("VLLM_CUDAGRAPH_ESTIMATE_SCALE", raw)
    with pytest.raises(ValueError, match="VLLM_CUDAGRAPH_ESTIMATE_SCALE"):
        cudagraph_estimate_scale()


def test_scale_arithmetic():
    assert scale_cudagraph_estimate(CUDAGRAPH_ESTIMATE, 1.0) == CUDAGRAPH_ESTIMATE
    assert scale_cudagraph_estimate(CUDAGRAPH_ESTIMATE, 0.75) == 12 * GiB_bytes
    assert scale_cudagraph_estimate(0, 0.75) == 0
    # Rounds down to whole bytes.
    assert scale_cudagraph_estimate(3, 0.5) == 1


def _worker():
    """The minimal Worker surface determine_available_memory touches."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
            )
        ),
        randomize_dummy_inputs=False,
        use_v2_model_runner=True,
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.9
        ),
        model_config=SimpleNamespace(multimodal_config=None),
        parallel_config=SimpleNamespace(),
        model_runner=SimpleNamespace(
            profile_run=lambda **kwargs: None,
            model_memory_usage=0,
            profile_cudagraph_memory=lambda: CUDAGRAPH_ESTIMATE,
        ),
        init_snapshot=SimpleNamespace(
            free_memory=TOTAL_MEMORY, total_memory=TOTAL_MEMORY
        ),
        requested_memory=REQUESTED_MEMORY,
        _scoped_allocator_max_split=lambda **kwargs: nullcontext(),
    )


@contextmanager
def _fake_memory_profiling(*args, **kwargs):
    yield SimpleNamespace(
        total_consumed=NON_KV_CACHE_MEMORY,
        non_kv_cache_memory=NON_KV_CACHE_MEMORY,
        transient_peak_headroom=0,
        after_profile=SimpleNamespace(free_memory=TOTAL_MEMORY - NON_KV_CACHE_MEMORY),
    )


@pytest.fixture
def mocked_device(monkeypatch):
    monkeypatch.setattr(gpu_worker, "maybe_apply_startup_plan", lambda _: None)
    monkeypatch.setattr(gpu_worker, "memory_profiling", _fake_memory_profiling)
    with patch.object(
        gpu_worker,
        "current_platform",
        SimpleNamespace(
            is_cuda_alike=lambda: True, is_xpu=lambda: False, is_rocm=lambda: False
        ),
    ):
        yield


@pytest.mark.usefixtures("mocked_device")
@pytest.mark.parametrize(
    ("scale_env", "profiler_flag", "expected_applied"),
    [
        # Unset: the full estimate is reserved, as before.
        (None, "1", CUDAGRAPH_ESTIMATE),
        ("1.0", "1", CUDAGRAPH_ESTIMATE),
        ("0.75", "1", 12 * GiB_bytes),
        ("1.25", "1", 20 * GiB_bytes),
        # Profiler estimate disabled: nothing is reserved regardless of scale.
        (None, "0", 0),
        ("0.75", "0", 0),
    ],
)
def test_determine_available_memory_applies_scale(
    monkeypatch, scale_env, profiler_flag, expected_applied
):
    if scale_env is None:
        monkeypatch.delenv("VLLM_CUDAGRAPH_ESTIMATE_SCALE", raising=False)
    else:
        monkeypatch.setenv("VLLM_CUDAGRAPH_ESTIMATE_SCALE", scale_env)
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", profiler_flag)
    worker = _worker()

    available = gpu_worker.Worker.determine_available_memory(worker)

    assert available == REQUESTED_MEMORY - NON_KV_CACHE_MEMORY - expected_applied
    assert worker.available_kv_cache_memory_bytes == available
    assert worker.peak_activation_memory == expected_applied
    assert worker.cudagraph_memory_estimate_applied == expected_applied
    # The unscaled estimate is kept for the post-capture comparison.
    assert worker.cudagraph_memory_estimate == CUDAGRAPH_ESTIMATE


def test_post_capture_log_reports_fitted_scale():
    with (
        patch.object(gpu_worker.logger, "info") as info,
        patch.object(gpu_worker.logger, "warning") as warning,
    ):
        log_cudagraph_estimate_scale(
            actual=10 * GiB_bytes,
            estimate=CUDAGRAPH_ESTIMATE,
            applied=12 * GiB_bytes,
            scale=0.75,
        )
    info.assert_called_once()
    assert info.call_args.args[-1] == pytest.approx(10 / 16)
    warning.assert_not_called()


def test_post_capture_log_warns_when_reservation_too_small():
    with (
        patch.object(gpu_worker.logger, "info"),
        patch.object(gpu_worker.logger, "warning") as warning,
    ):
        log_cudagraph_estimate_scale(
            actual=13 * GiB_bytes,
            estimate=CUDAGRAPH_ESTIMATE,
            applied=8 * GiB_bytes,
            scale=0.5,
        )
    warning.assert_called_once()
    assert warning.call_args.args[-1] == pytest.approx(13 / 16)


def test_post_capture_log_silent_when_nothing_reserved():
    """With the profiler estimate disabled nothing was reserved; no warning."""
    with (
        patch.object(gpu_worker.logger, "info"),
        patch.object(gpu_worker.logger, "warning") as warning,
    ):
        log_cudagraph_estimate_scale(
            actual=13 * GiB_bytes, estimate=CUDAGRAPH_ESTIMATE, applied=0, scale=0.5
        )
    warning.assert_not_called()
