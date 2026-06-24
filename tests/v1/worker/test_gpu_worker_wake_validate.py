# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the post-wake validation forward pass in Worker.wake_up().

These tests assert that a silently-corrupted wake (where the first real forward
pass raises, e.g. cudaErrorIllegalAddress after a cumem remap) surfaces as a
failure from wake_up() itself, instead of wake_up() blindly returning success
and letting the engine report ready over a poisoned state.

They run on CPU with the allocator and model runner mocked; no GPU required.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.worker.gpu_worker import Worker


def _make_worker(monkeypatch, *, dummy_run_side_effect=None):
    """Build a bare Worker with only the attributes wake_up() touches.

    Returns (worker, calls) where calls records which validation/runner methods
    fired so tests can assert ordering.
    """
    calls = {"dummy_run": 0, "post_kv_cache_wake_up": 0, "allocator_wake_up": 0}

    allocator = MagicMock()

    def _allocator_wake_up(tags):
        calls["allocator_wake_up"] += 1

    allocator.wake_up.side_effect = _allocator_wake_up
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.get_mem_allocator_instance",
        lambda: allocator,
    )
    # Avoid a real torch.cuda.synchronize() in the CPU test environment.
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.current_platform.is_cuda_alike",
        lambda: False,
    )

    def _dummy_run(num_tokens, uniform_decode=False, **kwargs):
        calls["dummy_run"] += 1
        if dummy_run_side_effect is not None:
            raise dummy_run_side_effect

    def _post_kv_cache_wake_up():
        calls["post_kv_cache_wake_up"] += 1

    worker = object.__new__(Worker)
    worker._sleep_saved_buffers = {}
    worker.model_runner = SimpleNamespace(
        uniform_decode_query_len=1,
        _dummy_run=_dummy_run,
        post_kv_cache_wake_up=_post_kv_cache_wake_up,
    )
    return worker, calls


def test_wake_runs_validation_forward_pass_on_clean_wake(monkeypatch):
    monkeypatch.setenv("VLLM_WAKE_VALIDATE", "1")
    worker, calls = _make_worker(monkeypatch)

    Worker.wake_up(worker, tags=None)

    assert calls["allocator_wake_up"] == 1
    assert calls["post_kv_cache_wake_up"] == 1
    # A clean wake still runs the validation forward pass exactly once.
    assert calls["dummy_run"] == 1


def test_wake_raises_on_corrupt_wake(monkeypatch):
    """Pre-fix this would silently return success; post-fix it must raise."""
    monkeypatch.setenv("VLLM_WAKE_VALIDATE", "1")
    boom = RuntimeError("CUDA error: an illegal memory access was encountered")
    worker, calls = _make_worker(monkeypatch, dummy_run_side_effect=boom)

    with pytest.raises(RuntimeError, match="illegal memory access"):
        Worker.wake_up(worker, tags=None)

    # Validation ran and failed loud — wake_up does NOT report success.
    assert calls["dummy_run"] == 1


def test_wake_validation_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_WAKE_VALIDATE", "0")
    boom = RuntimeError("CUDA error: an illegal memory access was encountered")
    worker, calls = _make_worker(monkeypatch, dummy_run_side_effect=boom)

    # With the flag off, wake_up must NOT run the validation pass and therefore
    # must not raise (preserving the legacy opt-out behavior).
    Worker.wake_up(worker, tags=None)
    assert calls["dummy_run"] == 0


def test_weights_only_wake_skips_validation(monkeypatch):
    """A weights-only wake does not re-enable real forward passes, so the
    validation pass (which needs KV cache) must be skipped."""
    monkeypatch.setenv("VLLM_WAKE_VALIDATE", "1")
    worker, calls = _make_worker(monkeypatch)

    Worker.wake_up(worker, tags=["weights"])

    assert calls["allocator_wake_up"] == 1
    assert calls["post_kv_cache_wake_up"] == 0
    assert calls["dummy_run"] == 0
