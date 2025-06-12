# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.cuda import set_cuda_context


def check_cuda_context(label=""):
    """Check CUDA driver context status"""
    try:
        cuda = ctypes.CDLL('libcuda.so')
        device = ctypes.c_int()
        result = cuda.cuCtxGetDevice(ctypes.byref(device))

        if result == 0:
            return True, device.value
        else:
            return False, None
    except Exception:
        return False, None


def unset_cuda_context(label=""):
    """Unset/destroy the current CUDA context"""
    try:
        cuda = ctypes.CDLL('libcuda.so')
        # Get current context
        context = ctypes.c_void_p()
        result = cuda.cuCtxGetCurrent(ctypes.byref(context))

        if result == 0 and context.value is not None:
            # Destroy the current context
            result = cuda.cuCtxDestroy(context)
            return result == 0
        else:
            return True
    except Exception:
        return False


class TestSetCudaContext:
    """Test suite for the set_cuda_context function."""

    @pytest.mark.skipif(not current_platform.is_cuda(),
                        reason="CUDA not available")
    @pytest.mark.parametrize("device_input,expected_device_id", [
        (0, 0),
        (torch.device('cuda:0'), 0),
        (torch.device('cuda'), 0),
    ],
                             ids=[
                                 "device_ID_as_int", "torch_device_with_index",
                                 "torch_device_without_index"
                             ])
    def test_set_cuda_context_parametrized(self, device_input,
                                           expected_device_id):
        """Parametrized test for setting CUDA context with various input types.
        """

        # Check context before setting - should now be invalid
        valid_before, device_before = check_cuda_context(
            "BEFORE set_cuda_context")
        assert not valid_before, \
            "CUDA context should not be valid after cleanup"

        # Test setting CUDA context
        result = set_cuda_context(device_input)
        assert result is True, "set_cuda_context should succeed"

        # Check context after setting - should always be valid now
        valid_after, device_id = check_cuda_context("AFTER set_cuda_context")
        assert valid_after is True, \
            "CUDA context should be valid after set_cuda_context"
        assert device_id == expected_device_id, \
            f"Expected device {expected_device_id}, got {device_id}"

        # Unset any existing CUDA context to start with a clean state
        unset_cuda_context("CLEANUP before test")

    def test_set_cuda_context_invalid_device_type(self):
        """Test error handling for invalid device type."""
        cpu_device = torch.device('cpu')
        with pytest.raises(ValueError, match="Expected CUDA device, got cpu"):
            set_cuda_context(cpu_device)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
