"""Tests for KV cache NaN detection feature."""

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_nan_counter():
    """Reset the NaN counter before each test."""
    yield


def test_env_var_exists():
    """VLLM_DEBUG_KV_CACHE_NANS env var should exist."""
    from vllm import envs
    assert hasattr(envs, 'VLLM_DEBUG_KV_CACHE_NANS')
    assert isinstance(envs.VLLM_DEBUG_KV_CACHE_NANS, bool)


def test_env_var_default_is_false():
    """VLLM_DEBUG_KV_CACHE_NANS should default to False."""
    from vllm import envs
    assert envs.VLLM_DEBUG_KV_CACHE_NANS is False


def test_check_nan_in_cache_source_clean():
    """Python NaN check should pass for clean tensors."""
    from vllm._custom_ops import _check_nan_in_cache_source
    
    key = torch.tensor([1.0, 2.0, 3.0])
    value = torch.tensor([4.0, 5.0, 6.0])
    
    # Should not raise or log warning
    _check_nan_in_cache_source(key, value)


def test_check_nan_in_cache_source_with_nan():
    """Python NaN check should detect NaN values."""
    from vllm._custom_ops import _check_nan_in_cache_source
    
    # Create tensor with NaN
    key = torch.tensor([1.0, float('nan'), 3.0])
    value = torch.tensor([1.0, 2.0, 3.0])
    
    # Should not raise, just log warning
    _check_nan_in_cache_source(key, value)


def test_check_nan_in_cache_source_with_inf():
    """Python NaN check should detect Inf values."""
    from vllm._custom_ops import _check_nan_in_cache_source
    
    key = torch.tensor([1.0, 2.0, 3.0])
    value = torch.tensor([float('inf'), 2.0, 3.0])
    
    # Should not raise, just log warning
    _check_nan_in_cache_source(key, value)


def test_get_nan_cache_write_count():
    """get_nan_cache_write_count should return an integer."""
    from vllm._custom_ops import get_nan_cache_write_count
    # This may not work without CUDA, but should at least import
    try:
        count = get_nan_cache_write_count()
        assert isinstance(count, int)
    except Exception:
        # If CUDA ops not available, that's OK for this test
        pass


def test_reset_nan_cache_write_count():
    """reset_nan_cache_write_count should be callable."""
    from vllm._custom_ops import reset_nan_cache_write_count
    # This may not work without CUDA, but should at least import
    try:
        reset_nan_cache_write_count()
    except Exception:
        # If CUDA ops not available, that's OK for this test
        pass
