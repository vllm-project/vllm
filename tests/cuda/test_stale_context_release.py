# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that _release_stale_cuda_primary_contexts cleans up inherited contexts.

When a worker process is forked from a parent that has initialized CUDA on
multiple devices, the child inherits active primary contexts for all devices.
This test verifies that the cleanup function correctly releases non-assigned
contexts while preserving the assigned device's context.
"""

import ctypes
from concurrent.futures import ProcessPoolExecutor

import pytest
import torch

from vllm.platforms import current_platform


def _get_primary_ctx_state(dev_id: int) -> bool:
    """Return True if the primary context for dev_id is active."""
    libcuda = ctypes.CDLL("libcuda.so.1")
    libcuda.cuInit(0)
    dev = ctypes.c_int()
    libcuda.cuDeviceGet(ctypes.byref(dev), dev_id)
    flags = ctypes.c_uint()
    state = ctypes.c_int()
    libcuda.cuDevicePrimaryCtxGetState(
        dev, ctypes.byref(flags), ctypes.byref(state)
    )
    return state.value != 0


def _child_test_release(local_rank: int, device_count: int):
    """Run in a subprocess: activate all contexts, release stale ones."""
    import torch

    from vllm.v1.worker.gpu_worker import _release_stale_cuda_primary_contexts

    # Activate primary contexts on ALL devices (simulating fork inheritance)
    for dev_id in range(device_count):
        torch.cuda.synchronize(dev_id)

    # Verify all contexts are active before cleanup
    before = {
        dev_id: _get_primary_ctx_state(dev_id) for dev_id in range(device_count)
    }
    assert all(before.values()), f"Expected all contexts active, got {before}"

    # Set device and release stale contexts
    torch.cuda.set_device(local_rank)
    _release_stale_cuda_primary_contexts(local_rank)

    # Check results
    after = {
        dev_id: _get_primary_ctx_state(dev_id) for dev_id in range(device_count)
    }
    return before, after


@pytest.mark.skipif(
    not current_platform.is_cuda() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA devices",
)
class TestStaleCudaContextRelease:

    def test_release_stale_contexts_preserves_assigned_device(self):
        """Verify that only the assigned device's context survives."""
        device_count = torch.cuda.device_count()
        local_rank = 1

        with ProcessPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_child_test_release, local_rank, device_count)
            before, after = future.result(timeout=30)

        assert after[local_rank], (
            f"Assigned device {local_rank} context should remain active"
        )
        for dev_id in range(device_count):
            if dev_id != local_rank:
                assert not after[dev_id], (
                    f"Device {dev_id} context should have been released"
                )

    def test_release_noop_for_single_gpu(self):
        """On single-GPU systems, nothing should be released."""
        if torch.cuda.device_count() > 1:
            pytest.skip("Test requires exactly 1 GPU")

        from vllm.v1.worker.gpu_worker import _release_stale_cuda_primary_contexts

        torch.cuda.synchronize(0)
        _release_stale_cuda_primary_contexts(0)
        assert _get_primary_ctx_state(0), "Device 0 context should remain"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
