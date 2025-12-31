"""Tests for vllm.utils.device_utils module."""

import pytest
import torch

from vllm.utils.device_utils import (
    get_device_property,
    get_gpu_name,
    get_gpu_memory_info,
    get_gpu_utilization,
    get_available_gpu_memory,
    clear_gpu_caches,
    device_memory_tracing,
    get_device_count,
    is_using_gpu,
    get_current_device,
    get_device_name,
    estimate_model_memory_requirements,
)


class TestGetDeviceProperty:
    """Tests for get_device_property function."""
    
    def test_unavailable_cuda(self):
        """Test when CUDA is not available."""
        if not torch.cuda.is_available():
            result = get_device_property(0, 'name')
            assert result is None
    
    def test_invalid_device(self):
        """Test with invalid device index."""
        result = get_device_property(999, 'name')
        assert result is None
    
    def test_unknown_property_raises(self):
        """Test that unknown property raises ValueError."""
        with pytest.raises(ValueError):
            get_device_property(0, 'unknown_property')
    
    def test_valid_properties(self):
        """Test retrieving valid properties."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # These should not raise
        name = get_device_property(0, 'name')
        assert isinstance(name, str)
        
        total_memory = get_device_property(0, 'total_memory')
        assert isinstance(total_memory, int)
        
        compute_cap = get_device_property(0, 'compute_cap')
        assert isinstance(compute_cap, tuple)


class TestGetGpuName:
    """Tests for get_gpu_name function."""
    
    def test_basic(self):
        """Test basic GPU name retrieval."""
        if not torch.cuda.is_available():
            result = get_gpu_name()
            assert result == "Unknown"
        
        result = get_gpu_name(0)
        assert isinstance(result, str)
        assert len(result) > 0


class TestGetGpuMemoryInfo:
    """Tests for get_gpu_memory_info function."""
    
    def test_unavailable_cuda(self):
        """Test when CUDA is not available."""
        result = get_gpu_memory_info(0)
        assert result['total_gb'] == 0.0
        assert result['allocated_gb'] == 0.0
        assert result['utilization_percent'] == 0.0
    
    def test_invalid_device(self):
        """Test with invalid device index."""
        result = get_gpu_memory_info(999)
        assert result['total_gb'] == 0.0
    
    def test_keys_exist(self):
        """Test that all expected keys exist."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        result = get_gpu_memory_info(0)
        expected_keys = ['total_gb', 'allocated_gb', 'reserved_gb', 
                        'free_gb', 'utilization_percent']
        for key in expected_keys:
            assert key in result


class TestGetGpuUtilization:
    """Tests for get_gpu_utilization function."""
    
    def test_unavailable_cuda(self):
        """Test when CUDA is not available."""
        result = get_gpu_utilization()
        assert result == -1.0
    
    def test_valid_utilization(self):
        """Test getting valid utilization."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        result = get_gpu_utilization(0)
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0


class TestGetAvailableGpuMemory:
    """Tests for get_available_gpu_memory function."""
    
    def test_unavailable_cuda(self):
        """Test when CUDA is not available."""
        result = get_available_gpu_memory()
        assert result == -1.0
    
    def test_valid_memory(self):
        """Test getting available memory."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        result = get_available_gpu_memory(0)
        assert isinstance(result, float)
        assert result > 0


class TestClearGpuCaches:
    """Tests for clear_gpu_caches function."""
    
    def test_no_cuda(self):
        """Test when CUDA is not available."""
        # Should not raise
        clear_gpu_caches()
        clear_gpu_caches(device=0)
        clear_gpu_caches(device=None)
    
    def test_with_cuda(self):
        """Test cache clearing with CUDA available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Should not raise
        clear_gpu_caches()
        clear_gpu_caches(device=0)
        clear_gpu_caches(device=None)


class TestDeviceMemoryTracing:
    """Tests for device_memory_tracing context manager."""
    
    def test_basic_usage(self):
        """Test basic context manager usage."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        with device_memory_tracing() as mem_info:
            assert 'device' in mem_info
            assert 'before_allocated_gb' in mem_info
    
    def test_delta_calculation(self):
        """Test that delta is calculated after exit."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        with device_memory_tracing() as mem_info:
            pass
        
        # After context exit, delta should be present
        assert 'delta_gb' in mem_info
        assert 'after_allocated_gb' in mem_info


class TestDeviceCount:
    """Tests for get_device_count function."""
    
    def test_no_cuda(self):
        """Test when CUDA is not available."""
        result = get_device_count()
        if not torch.cuda.is_available():
            assert result == 0
    
    def test_with_cuda(self):
        """Test when CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        result = get_device_count()
        assert result >= 1


class TestIsUsingGpu:
    """Tests for is_using_gpu function."""
    
    def test_returns_bool(self):
        """Test that function returns a boolean."""
        result = is_using_gpu()
        assert isinstance(result, bool)


class TestGetCurrentDevice:
    """Tests for get_current_device function."""
    
    def test_returns_int(self):
        """Test that function returns an integer."""
        result = get_current_device()
        assert isinstance(result, int)


class TestGetDeviceName:
    """Tests for get_device_name function."""
    
    def test_no_device(self):
        """Test when no GPU available."""
        result = get_device_name()
        if not torch.cuda.is_available():
            assert result == "CPU"
    
    def test_with_device(self):
        """Test getting device name."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        result = get_device_name(0)
        assert isinstance(result, str)
        assert len(result) > 0


class TestEstimateModelMemoryRequirements:
    """Tests for estimate_model_memory_requirements function."""
    
    def test_basic_estimation(self):
        """Test basic memory estimation."""
        est = estimate_model_memory_requirements(
            num_parameters=7_000_000_000,
            precision="bf16",
        )
        
        assert "weights_gb" in est
        assert est["weights_gb"] > 0
        assert est["weights_gb"] < 100
    
    def test_all_precisions(self):
        """Test all supported precisions."""
        for precision in ["fp32", "fp16", "bf16", "int8"]:
            est = estimate_model_memory_requirements(
                num_parameters=7_000_000_000,
                precision=precision,
            )
            assert est["weights_gb"] > 0
    
    def test_with_layers(self):
        """Test estimation with layer count."""
        est = estimate_model_memory_requirements(
            num_parameters=7_000_000_000,
            precision="bf16",
            num_layers=32,
        )
        
        assert "activations_gb" in est
        assert "total_estimate_gb" in est
    
    def test_with_full_config(self):
        """Test estimation with full configuration."""
        est = estimate_model_memory_requirements(
            num_parameters=7_000_000_000,
            precision="bf16",
            num_layers=32,
            hidden_size=4096,
            vocab_size=32000,
        )
        
        assert "kv_cache_per_token_gb" in est
        assert "embeddings_gb" in est
    
    def test_auto_precision(self):
        """Test auto precision detection."""
        est = estimate_model_memory_requirements(
            num_parameters=1_000_000_000,
            precision="auto",
        )
        
        assert "weights_gb" in est
        assert est["weights_gb"] > 0
