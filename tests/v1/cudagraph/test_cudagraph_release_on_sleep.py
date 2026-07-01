# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for releasing CUDA graphs before a cumem sleep() unmap and
re-capturing them after wake_up().

Background: with CUDA-graph capture pools resident, per-allocation cumem VMM
unmap/remap operations slow dramatically (field-measured ~6-9s for a 32B model
vs ~2s without graphs). Releasing the captured graphs *before* the cumem unmap
restores the fast unmap path; the graphs are re-captured on the next full wake.

These tests assert the ordering contract -- graphs are released *before* the
allocator unmap and re-captured *after* the allocator remap -- and the NONE-mode
no-op behavior. The ordering tests run on CPU (allocator + model interactions
are mocked); the real-graph release test is GPU-gated.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.platforms import current_platform
from vllm.v1.worker.gpu_worker import Worker

from .test_cudagraph_dispatch import SimpleMLP, _create_vllm_config

DEVICE_TYPE = current_platform.device_type


# ---------------------------------------------------------------------------
# Worker sleep/wake ordering contract (CPU; allocator + runner mocked)
# ---------------------------------------------------------------------------


def _make_worker_stub(cudagraph_mode: CUDAGraphMode = CUDAGraphMode.PIECEWISE):
    """A bare object onto which we can bind the unbound Worker.sleep/wake_up
    methods without standing up a real Worker (which needs a GPU + a model)."""
    stub = MagicMock()
    stub._sleep_saved_buffers = {}
    stub._cudagraphs_released = False
    # Wake-path tests assume a prior sleep() has run, which leaves both the
    # weights and the KV cache suspended until a wake restores them. Tests that
    # exercise sleep() itself reset these as needed.
    stub._weights_suspended = True
    stub._kv_cache_suspended = True

    # model_runner.release_cudagraphs returns True iff graphs were present.
    released = cudagraph_mode != CUDAGraphMode.NONE
    stub.model_runner.release_cudagraphs.return_value = released
    return stub


def test_release_happens_before_unmap_on_sleep():
    """sleep() must release CUDA graphs BEFORE the allocator unmaps the VA
    space -- that is the whole point of the optimization."""
    stub = _make_worker_stub(CUDAGraphMode.PIECEWISE)

    order = []
    stub.model_runner.release_cudagraphs.side_effect = lambda: (
        order.append("release"),
        True,
    )[1]

    allocator = MagicMock()
    allocator.sleep.side_effect = lambda *a, **k: order.append("unmap")

    with (
        patch(
            "vllm.v1.worker.gpu_worker.get_mem_allocator_instance",
            return_value=allocator,
        ),
        patch("torch.cuda.mem_get_info", return_value=(10, 100)),
    ):
        Worker.sleep(stub, level=1)

    assert order == ["release", "unmap"], (
        "graphs must be released before the cumem unmap"
    )
    assert stub._cudagraphs_released is True
    # sleep() unmaps both the weights and the KV cache, so both are suspended.
    assert stub._weights_suspended is True
    assert stub._kv_cache_suspended is True


def test_recapture_happens_after_remap_on_wake():
    """wake_up() must re-capture graphs AFTER the allocator remaps the VA
    space (weights + KV cache must be back before capture)."""
    stub = _make_worker_stub(CUDAGraphMode.PIECEWISE)
    stub._cudagraphs_released = True  # as if a prior sleep() released them

    order = []
    allocator = MagicMock()
    allocator.wake_up.side_effect = lambda *a, **k: order.append("remap")
    stub.model_runner.recapture_cudagraphs.side_effect = lambda: order.append(
        "recapture"
    )

    with patch(
        "vllm.v1.worker.gpu_worker.get_mem_allocator_instance",
        return_value=allocator,
    ):
        Worker.wake_up(stub, tags=None)

    assert order == ["remap", "recapture"], (
        "graphs must be re-captured after the cumem remap"
    )
    assert stub._cudagraphs_released is False


def test_weights_only_wake_does_not_recapture():
    """A weights-only wake leaves the KV cache unmapped, so graph capture must
    wait for a full (kv_cache) wake."""
    stub = _make_worker_stub(CUDAGraphMode.PIECEWISE)
    stub._cudagraphs_released = True

    allocator = MagicMock()
    with patch(
        "vllm.v1.worker.gpu_worker.get_mem_allocator_instance",
        return_value=allocator,
    ):
        Worker.wake_up(stub, tags=["weights"])

    stub.model_runner.recapture_cudagraphs.assert_not_called()
    # Still pending: must re-capture on the eventual full wake.
    assert stub._cudagraphs_released is True

    # Now the KV cache comes back -> recapture fires.
    with patch(
        "vllm.v1.worker.gpu_worker.get_mem_allocator_instance",
        return_value=allocator,
    ):
        Worker.wake_up(stub, tags=["kv_cache"])
    stub.model_runner.recapture_cudagraphs.assert_called_once()
    assert stub._cudagraphs_released is False


def test_kv_cache_only_wake_defers_recapture_while_weights_asleep():
    """The RLHF partial-wake pattern: sleep() -> wake_up(tags=["kv_cache"])
    WITHOUT a prior weights wake. The weights are still cumem-unmapped, so a
    recapture (which runs a forward pass through the weights) would fault with
    CUDA_ERROR_ILLEGAL_ADDRESS. recapture_cudagraphs() must therefore be
    deferred until the weights are subsequently woken.

    Pre-fix this faulted: the recapture was gated only on the KV cache being
    awake, so a kv_cache-only wake called capture_model() on unmapped weights.
    """
    stub = _make_worker_stub(CUDAGraphMode.PIECEWISE)
    stub._cudagraphs_released = True
    # Both segments start suspended (post-sleep); no wake has happened yet.
    stub._weights_suspended = True
    stub._kv_cache_suspended = True

    allocator = MagicMock()

    # kv_cache-only wake while weights are still asleep -> must NOT recapture.
    with patch(
        "vllm.v1.worker.gpu_worker.get_mem_allocator_instance",
        return_value=allocator,
    ):
        Worker.wake_up(stub, tags=["kv_cache"])

    stub.model_runner.recapture_cudagraphs.assert_not_called()
    # Still pending: must fire once the weights come back. KV cache is now
    # mapped, but the weights remain unmapped.
    assert stub._cudagraphs_released is True
    assert stub._weights_suspended is True
    assert stub._kv_cache_suspended is False

    # Now the weights are woken -> recapture fires (KV cache already mapped).
    with patch(
        "vllm.v1.worker.gpu_worker.get_mem_allocator_instance",
        return_value=allocator,
    ):
        Worker.wake_up(stub, tags=["weights"])

    stub.model_runner.recapture_cudagraphs.assert_called_once()
    assert stub._cudagraphs_released is False
    assert stub._weights_suspended is False


def test_no_recapture_when_nothing_was_released():
    """If sleep() never released graphs (e.g. cudagraph_mode NONE), wake_up()
    must not attempt a re-capture."""
    stub = _make_worker_stub(CUDAGraphMode.NONE)
    stub._cudagraphs_released = False

    allocator = MagicMock()
    with patch(
        "vllm.v1.worker.gpu_worker.get_mem_allocator_instance",
        return_value=allocator,
    ):
        Worker.wake_up(stub, tags=None)

    stub.model_runner.recapture_cudagraphs.assert_not_called()


# ---------------------------------------------------------------------------
# GPUModelRunner.release_cudagraphs / recapture_cudagraphs unit behavior
# ---------------------------------------------------------------------------


class _RunnerStub:
    """Minimal stand-in exposing just the attributes the two methods touch,
    so we can exercise the real GPUModelRunner methods unbound without a GPU
    model load."""

    def __init__(self, cudagraph_mode: CUDAGraphMode):
        self.compilation_config = MagicMock()
        self.compilation_config.cudagraph_mode = cudagraph_mode
        self.encoder_cudagraph_manager = None


def test_release_cudagraphs_noop_for_none_mode():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    stub = _RunnerStub(CUDAGraphMode.NONE)
    assert GPUModelRunner.release_cudagraphs(stub) is False


def test_recapture_cudagraphs_noop_for_none_mode():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    stub = _RunnerStub(CUDAGraphMode.NONE)
    # Should return without calling capture_model.
    stub.capture_model = MagicMock()
    GPUModelRunner.recapture_cudagraphs(stub)
    stub.capture_model.assert_not_called()


def test_release_cudagraphs_clears_wrappers_for_active_mode():
    """For an active cudagraph_mode, release_cudagraphs must clear all
    CUDAGraphWrapper instances and report that re-capture is required."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    stub = _RunnerStub(CUDAGraphMode.PIECEWISE)
    stub.encoder_cudagraph_manager = MagicMock()

    with (
        patch.object(CUDAGraphWrapper, "clear_all_graphs") as clear_cg,
        patch("vllm.v1.worker.gpu_model_runner.BreakableCUDAGraphWrapper") as breakable,
        patch("torch.accelerator.synchronize"),
        patch("torch.accelerator.empty_cache"),
        patch("gc.collect"),
    ):
        result = GPUModelRunner.release_cudagraphs(stub)

    assert result is True
    clear_cg.assert_called_once()
    breakable.clear_all_graphs.assert_called_once()
    # Encoder graph manager dropped so its capture pool is freed too.
    assert stub.encoder_cudagraph_manager is None


# ---------------------------------------------------------------------------
# Real captured-graph release (GPU only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA graph capture requires a CUDA-like device",
)
def test_clear_all_graphs_releases_captured_entries():
    """End-to-end: capture a real CUDA graph, then assert clear_all_graphs()
    (the primitive release_cudagraphs() relies on) drops the captured entry so
    its capture pool can be reclaimed before a cumem unmap."""
    vllm_config = _create_vllm_config(CompilationConfig())
    model = SimpleMLP().to(DEVICE_TYPE)
    wrapper = CUDAGraphWrapper(model, vllm_config, runtime_mode=CUDAGraphMode.FULL)
    inp = torch.randn(1, 10, device=DEVICE_TYPE)
    batch_descriptor = BatchDescriptor(num_tokens=10)

    # Warmup (NONE mode), then capture (FULL mode).
    with set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
    ):
        wrapper(inp)
    with set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=batch_descriptor,
    ):
        wrapper(inp)

    assert batch_descriptor in wrapper.concrete_cudagraph_entries
    assert wrapper.concrete_cudagraph_entries[batch_descriptor].cudagraph is not None

    # The release primitive used by release_cudagraphs().
    CUDAGraphWrapper.clear_all_graphs()

    assert not wrapper.concrete_cudagraph_entries, (
        "captured graphs must be dropped so the capture pool is reclaimable "
        "before the cumem unmap"
    )
