# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for MPS platform integration."""

import os
import sys

import pytest
import torch

# Set environment variables before importing vLLM
# Note: VLLM_USE_V1=1 is not yet supported with MPS, so we'll test without it
os.environ["VLLM_TARGET_DEVICE"] = "mps"
# Explicitly disable V1 to test the async output behavior
if "VLLM_USE_V1" in os.environ:
    del os.environ["VLLM_USE_V1"]


def test_torch_mps_availability():
    """Test that PyTorch MPS is available."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this system")
    
    assert torch.backends.mps.is_built(), "MPS should be built"


def test_mps_platform_detection():
    """Test that MPS platform is correctly detected."""
    from vllm.platforms import current_platform
    
    platform_name = current_platform.__class__.__name__
    
    # Check that we're using MPS platform
    assert "Mps" in platform_name, f"Expected MPS platform, but got {platform_name}"


def test_mps_engine_initialization():
    """Test MPS engine initialization up to the point where attention backend is needed."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this system")
    
    from vllm.engine.arg_utils import EngineArgs
    
    # Use a small public model for testing
    model_name = "microsoft/DialoGPT-small"
    
    # Create engine args
    engine_args = EngineArgs(
        model=model_name,
        max_num_seqs=1,
        max_model_len=512,
        trust_remote_code=True,
        enforce_eager=True,
    )
    
    # Test creating the VllmConfig - this should work with our MPS platform
    vllm_config = engine_args.create_engine_config("test_mps")
    
    # Verify MPS-specific configuration
    assert str(vllm_config.device_config.device) == "mps"
    assert vllm_config.cache_config.block_size == 16  # Default block size for MPS
    assert vllm_config.parallel_config.worker_cls == "vllm.worker.worker.Worker"
    assert vllm_config.compilation_config.use_cudagraph is False  # Should be disabled for MPS


def test_mps_platform_methods():
    """Test that MPS platform has all required methods implemented."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this system")
    
    from vllm.platforms import current_platform
    
    # Test required platform methods
    assert hasattr(current_platform, "get_device_name")
    assert hasattr(current_platform, "get_device_total_memory")
    assert hasattr(current_platform, "is_async_output_supported")
    assert hasattr(current_platform, "inference_mode")
    assert hasattr(current_platform, "get_current_memory_usage")
    
    # Test method calls
    device_name = current_platform.get_device_name()
    assert device_name == "Apple MPS"
    
    total_memory = current_platform.get_device_total_memory()
    assert total_memory > 0
    
    # Test async output support behavior
    # The MPS platform should follow the same logic as CUDA:
    # Returns True by default, False only when enforce_eager=True AND VLLM_USE_V1=False
    async_support_default = current_platform.is_async_output_supported(enforce_eager=False)
    assert async_support_default is True, "MPS should support async output by default"
    
    # With enforce_eager=True, behavior depends on VLLM_USE_V1 setting
    async_support_eager = current_platform.is_async_output_supported(enforce_eager=True)
    from vllm import envs
    if envs.VLLM_USE_V1:
        assert async_support_eager is True, "MPS should support async output when VLLM_USE_V1=True"
    else:
        assert async_support_eager is False, "MPS should not support async output when enforce_eager=True and VLLM_USE_V1=False"
    
    # Test memory usage (returns 0 for MPS as placeholder)
    memory_usage = current_platform.get_current_memory_usage(torch.device("mps"))
    assert memory_usage == 0


def test_mps_attention_backend_not_implemented():
    """Test that MPS attention backend correctly reports as not implemented."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this system")
    
    from vllm.platforms import current_platform
    from vllm.platforms.interface import _Backend
    
    # This should raise NotImplementedError since attention backend is not yet implemented
    with pytest.raises(NotImplementedError, match="MPS attention backend not yet implemented"):
        current_platform.get_attn_backend_cls(
            selected_backend=_Backend.FLASH_ATTN,
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype=None,
            block_size=16,
            use_v1=False,
            use_mla=False
        )


if __name__ == "__main__":
    # Allow running the test file directly
    pytest.main([__file__, "-v"])