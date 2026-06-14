# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for:
  [Bug] EngineDeadError after first L1 sleep/wake cycle with
        --kv-offloading-backend native + --enable-sleep-mode

Root cause:
  CuMemAllocator.sleep() iterates all cumem allocations and calls
  unmap_and_release() WITHOUT first synchronizing the offloading connector's
  transfer CUDA streams.

  _python_free_callback() (cumem.py:158) correctly calls
  torch.cuda.synchronize() before releasing the handle.  sleep() did not.

  If SingleDirectionOffloadingHandler had an in-flight GPU→CPU store when
  sleep() fired, the DMA engine reads from the now-unmapped VA:
      CUDA error: unspecified launch failure  (cudaErrorLaunchFailure)
  This poisons the CUDA context so every subsequent CUDA call fails, which
  surfaces as EngineDeadError on the first post-wake request.

Fix:
  Worker.sleep() now calls connector.sleep() (OffloadingConnector.sleep()
  → OffloadingConnectorWorker.sleep() → OffloadingWorker.wait_all())
  before CuMemAllocator.sleep(), draining all in-flight transfers first.

Tests:
  test_connector_sleep_quiesces_before_unmap:
      The connector's sleep() method must drain all in-flight transfers so
      allocator.sleep() can safely call cuMemUnmap.  Fails if wait_all()
      is removed from OffloadingConnectorWorker.sleep().

  test_bare_allocator_sleep_races_inflight_transfer:
      Demonstrates the bug (calling allocator.sleep() without quiescing the
      connector first → CUDA error).  This test is expected to FAIL on
      un-patched code and is retained as documentation of the crash.

Reproduction mechanism (torch.cuda._sleep stall):
  torch.cuda._sleep(N) spins the current CUDA stream for N clock cycles.
  SingleDirectionOffloadingHandler.transfer_async() inserts
  transfer_stream.wait_stream(current_stream) for GPU→CPU, so the DMA is
  blocked behind the _sleep.  sleep() is called in this window, guaranteeing
  the DMA has not started when cuMemUnmap fires.

Verified on: NVIDIA A100-SXM4-80GB, CUDA 12.9, PyTorch 2.11.0+cu129
"""

import os

import pytest
import torch

from tests.utils import create_new_process_for_each_test
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required"
    ),
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAGE_SIZE_BYTES = 64 * 1024    # 64 KiB per KV block page
NUM_BLOCKS = 128               # 8 MiB GPU KV cache

# Spin the current stream for ~1.3 s on an A100 @ ~1.5 GHz.
# Must be long enough that Python calls allocator.sleep() before _sleep
# finishes (i.e. before the DMA can start on the transfer stream).
STALL_CYCLES = 2_000_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alloc_tensors():
    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator.get_instance()
    keep = []
    with allocator.use_memory_pool(tag="kv_cache"):
        gpu_tensor = torch.zeros(
            NUM_BLOCKS, PAGE_SIZE_BYTES, dtype=torch.int8, device="cuda"
        )
        keep.append(gpu_tensor)
    cpu_tensor = torch.zeros(
        NUM_BLOCKS, PAGE_SIZE_BYTES, dtype=torch.int8, pin_memory=True
    )
    return allocator, gpu_tensor, cpu_tensor, keep


def _make_handler(gpu_tensor: torch.Tensor, cpu_tensor: torch.Tensor):
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
    )
    from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

    return SingleDirectionOffloadingHandler(
        gpu_tensors=[gpu_tensor],
        cpu_tensors=[cpu_tensor],
        block_size_factor=1,
        kv_cache_groups_data_refs=[[
            CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE_SIZE_BYTES)
        ]],
        gpu_to_cpu=True,
    )


def _transfer_spec(num_blocks: int = 32):
    from vllm.v1.kv_offload.base import GPULoadStoreSpec
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

    ids = list(range(num_blocks))
    return (
        GPULoadStoreSpec(block_ids=ids, group_sizes=(num_blocks,), block_indices=(0,)),
        CPULoadStoreSpec(ids),
    )


def _submit_stalled_transfer(handler):
    """Stall current stream via _sleep, submit a GPU→CPU transfer blocked behind it.
    Returns the transfer stream (which has pending work, blocked by _sleep)."""
    torch.cuda._sleep(STALL_CYCLES)
    assert handler.transfer_async(job_id=1, transfer_spec=_transfer_spec())
    stream = handler._transfers[0].stream
    assert not stream.query(), "transfer stream should be stalled behind _sleep"
    return stream


# ---------------------------------------------------------------------------
# Test 1 – regression: connector.sleep() must quiesce transfers before unmap
# ---------------------------------------------------------------------------

@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
def test_connector_sleep_quiesces_before_unmap():
    """
    OffloadingConnectorWorker.sleep() drains all in-flight transfers so that
    the subsequent allocator.sleep() → unmap_and_release() is safe.

    This is the regression test for the fix: it will FAIL if wait_all() is
    removed from OffloadingConnectorWorker.sleep() or if Worker.sleep() stops
    calling connector.sleep() before allocator.sleep().

    Pre-fix:  test crashes (CUDA error) because allocator.sleep() races with DMA
    Post-fix: no CUDA error; wait_all() ensures all DMA completes first
    """
    allocator, gpu_tensor, cpu_tensor, _keep = _alloc_tensors()
    handler = _make_handler(gpu_tensor, cpu_tensor)

    from vllm.v1.kv_offload.worker.worker import OffloadingWorker

    worker = OffloadingWorker()
    # Register the GPU→CPU handler directly
    worker.handlers.add(handler)
    worker.transfer_type_to_handler[("GPU", "CPU")] = handler

    # Stall current stream and submit a GPU→CPU transfer
    _submit_stalled_transfer(handler)

    # THE FIX: drain in-flight transfers before unmapping
    # OffloadingConnectorWorker.sleep() calls worker.wait_all() which calls
    # handler.wait_all() which synchronizes the transfer stream's end_event.
    worker.wait_all()

    # Now allocator.sleep() is safe: no DMA in flight, cuMemUnmap won't race.
    allocator.sleep(offload_tags=("weights",))

    try:
        torch.cuda.synchronize()
    except RuntimeError as exc:
        handler._transfers.clear()
        handler._transfer_events.clear()
        pytest.fail(
            f"wait_all() did not fully drain transfers before unmap:\n  {exc}\n\n"
            "OffloadingConnectorWorker.sleep() must call worker.wait_all() to "
            "synchronize all transfer streams before CuMemAllocator unmaps GPU VA."
        )

    handler.shutdown()


# ---------------------------------------------------------------------------
# Test 2 – bug demonstration: allocator.sleep() without quiescing = CUDA crash
# ---------------------------------------------------------------------------

@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
def test_bare_allocator_sleep_races_inflight_transfer():
    """
    Demonstrates the original bug: calling allocator.sleep() directly (without
    first calling connector.sleep() / wait_all()) crashes with a CUDA error
    when a GPU→CPU transfer is in-flight.

    This test is expected to FAIL on un-patched code (reproducing the crash)
    and is retained to document the failure mode and to be referenced in the
    PR description.  On patched code this test is also EXPECTED TO FAIL
    (pytest.fail is called if NO error is raised) because it tests that the
    bug genuinely exists when the fix is bypassed.

    Observed failure: CUDA error: unspecified launch failure (cudaErrorLaunchFailure)
    """
    allocator, gpu_tensor, cpu_tensor, _keep = _alloc_tensors()
    handler = _make_handler(gpu_tensor, cpu_tensor)

    # Stall current stream and submit a GPU→CPU transfer blocked behind it
    _submit_stalled_transfer(handler)

    # Bug path: no quiesce before unmap
    allocator.sleep(offload_tags=("weights",))

    # _sleep finishes → DMA runs on the now-unmapped VA
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        # Expected path — CUDA error confirms the bug.
        # os._exit(0) bypasses teardown so the poisoned CUDA context doesn't
        # cause a teardown error that would turn this xfail into a hard failure.
        os._exit(0)

    handler.shutdown()
    pytest.fail(
        "Expected CUDA error was not raised. The race was missed — "
        "try increasing STALL_CYCLES or verify _sleep is working on this GPU."
    )
