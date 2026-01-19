# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GPU UUID support in CUDA_VISIBLE_DEVICES."""

import os
import unittest
from unittest.mock import Mock, patch

import pytest

from vllm.platforms.cuda import NvmlCudaPlatform


class TestCudaUUIDSupport(unittest.TestCase):
    """Test suite for GPU UUID support in CUDA_VISIBLE_DEVICES."""

    def setUp(self):
        """Clear cache between tests."""
        if hasattr(NvmlCudaPlatform._build_uuid_to_index_map, "cache_clear"):
            NvmlCudaPlatform._build_uuid_to_index_map.cache_clear()

    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"})
    def test_integer_format_backward_compatibility(self):
        """Test backward compatibility with integer format."""
        assert NvmlCudaPlatform.device_id_to_physical_device_id(0) == 0
        assert NvmlCudaPlatform.device_id_to_physical_device_id(1) == 1
        assert NvmlCudaPlatform.device_id_to_physical_device_id(2) == 2

    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5,7"})
    def test_integer_format_with_offset(self):
        """Test integer format with non-zero offset."""
        assert NvmlCudaPlatform.device_id_to_physical_device_id(0) == 2
        assert NvmlCudaPlatform.device_id_to_physical_device_id(1) == 5
        assert NvmlCudaPlatform.device_id_to_physical_device_id(2) == 7

    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""})
    def test_empty_env_var_ray_case(self):
        """Test Ray CPU placement group case (empty CUDA_VISIBLE_DEVICES)."""
        # When CUDA_VISIBLE_DEVICES is empty, should return device_id as-is
        assert NvmlCudaPlatform.device_id_to_physical_device_id(0) == 0
        assert NvmlCudaPlatform.device_id_to_physical_device_id(5) == 5
        assert NvmlCudaPlatform.device_id_to_physical_device_id(10) == 10

    @patch.dict(os.environ, {}, clear=True)
    def test_unset_env_var(self):
        """Test when CUDA_VISIBLE_DEVICES is not set."""
        # Remove CUDA_VISIBLE_DEVICES if it exists
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        # When CUDA_VISIBLE_DEVICES is unset, should return device_id as-is
        assert NvmlCudaPlatform.device_id_to_physical_device_id(0) == 0
        assert NvmlCudaPlatform.device_id_to_physical_device_id(3) == 3

    @patch("vllm.platforms.cuda.pynvml")
    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-123,GPU-456"})
    def test_uuid_format_resolution(self, mock_pynvml):
        """Test UUID resolution."""
        # Mock NVML responses
        mock_pynvml.nvmlDeviceGetCount.return_value = 2

        mock_handle_0 = Mock()
        mock_handle_1 = Mock()

        def get_handle_by_index(idx):
            return [mock_handle_0, mock_handle_1][idx]

        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = get_handle_by_index
        mock_pynvml.nvmlDeviceGetUUID.side_effect = lambda h: (
            "GPU-123" if h == mock_handle_0 else "GPU-456"
        )
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlShutdown.return_value = None

        # Test resolution
        assert NvmlCudaPlatform.device_id_to_physical_device_id(0) == 0
        assert NvmlCudaPlatform.device_id_to_physical_device_id(1) == 1

    @patch("vllm.platforms.cuda.pynvml")
    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,GPU-123,2"})
    def test_mixed_format(self, mock_pynvml):
        """Test mixed integer and UUID format."""
        # Setup mocks
        mock_pynvml.nvmlDeviceGetCount.return_value = 3

        mock_handle_0 = Mock()
        mock_handle_1 = Mock()
        mock_handle_2 = Mock()

        def get_handle_by_index(idx):
            return [mock_handle_0, mock_handle_1, mock_handle_2][idx]

        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = get_handle_by_index
        mock_pynvml.nvmlDeviceGetUUID.side_effect = lambda h: (
            "GPU-000" if h == mock_handle_0
            else "GPU-123" if h == mock_handle_1
            else "GPU-222"
        )
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlShutdown.return_value = None

        # Test mixed format
        assert NvmlCudaPlatform.device_id_to_physical_device_id(0) == 0
        assert NvmlCudaPlatform.device_id_to_physical_device_id(1) == 1  # GPU-123 maps to index 1
        assert NvmlCudaPlatform.device_id_to_physical_device_id(2) == 2

    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
    def test_out_of_range_error(self):
        """Test IndexError for out of range device_id."""
        with pytest.raises(IndexError) as exc_info:
            NvmlCudaPlatform.device_id_to_physical_device_id(5)

        assert "out of range" in str(exc_info.value)
        assert "expected 0-1" in str(exc_info.value)

    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
    def test_negative_device_id(self):
        """Test IndexError for negative device_id."""
        with pytest.raises(IndexError) as exc_info:
            NvmlCudaPlatform.device_id_to_physical_device_id(-1)

        assert "out of range" in str(exc_info.value)

    @patch("vllm.platforms.cuda.pynvml")
    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-nonexistent"})
    def test_uuid_not_found(self, mock_pynvml):
        """Test ValueError for non-existent UUID."""
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_handle = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetUUID.return_value = "GPU-different"
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlShutdown.return_value = None

        with pytest.raises(ValueError) as exc_info:
            NvmlCudaPlatform.device_id_to_physical_device_id(0)

        assert "not found" in str(exc_info.value)
        assert "Available UUIDs" in str(exc_info.value)

    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "invalid-format"})
    def test_invalid_format(self):
        """Test ValueError for invalid UUID format."""
        with pytest.raises(ValueError) as exc_info:
            NvmlCudaPlatform.device_id_to_physical_device_id(0)

        assert "Invalid device ID format" in str(exc_info.value)
        assert "GPU-" in str(exc_info.value)
        assert "MIG-" in str(exc_info.value)

    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "abc123"})
    def test_invalid_non_numeric_format(self):
        """Test ValueError for non-numeric non-UUID format."""
        with pytest.raises(ValueError) as exc_info:
            NvmlCudaPlatform.device_id_to_physical_device_id(0)

        assert "Invalid device ID format" in str(exc_info.value)

    def test_is_uuid_gpu_format(self):
        """Test _is_uuid recognizes GPU-xxx format."""
        assert NvmlCudaPlatform._is_uuid("GPU-123") is True
        assert NvmlCudaPlatform._is_uuid("GPU-441f29f8-b53a-1c18-174a-dd2066ebd468") is True

    def test_is_uuid_mig_format(self):
        """Test _is_uuid recognizes MIG-xxx format."""
        assert NvmlCudaPlatform._is_uuid("MIG-123") is True
        assert NvmlCudaPlatform._is_uuid("MIG-GPU-441f29f8-b53a-1c18-174a-dd2066ebd468") is True

    def test_is_uuid_rejects_invalid_formats(self):
        """Test _is_uuid rejects non-UUID formats."""
        assert NvmlCudaPlatform._is_uuid("0") is False
        assert NvmlCudaPlatform._is_uuid("123") is False
        assert NvmlCudaPlatform._is_uuid("invalid") is False
        assert NvmlCudaPlatform._is_uuid("gpu-123") is False  # lowercase
        assert NvmlCudaPlatform._is_uuid("mig-123") is False  # lowercase

    @patch("vllm.platforms.cuda.pynvml")
    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-123"})
    def test_nvml_error_handling(self, mock_pynvml):
        """Test graceful handling of NVML errors."""
        # Mock NVML to raise an error
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetCount.side_effect = Exception("NVML Error")
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlShutdown.return_value = None

        # Should get ValueError because UUID not found (empty map due to NVML error)
        with pytest.raises(ValueError) as exc_info:
            NvmlCudaPlatform.device_id_to_physical_device_id(0)

        assert "not found" in str(exc_info.value)

    @patch("vllm.platforms.cuda.pynvml")
    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-123,GPU-456,GPU-789"})
    def test_uuid_map_caching(self, mock_pynvml):
        """Test that UUID map is cached and not rebuilt on every call."""
        # Setup mocks
        mock_pynvml.nvmlDeviceGetCount.return_value = 3

        mock_handles = [Mock(), Mock(), Mock()]
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = lambda idx: mock_handles[idx]
        mock_pynvml.nvmlDeviceGetUUID.side_effect = lambda h: (
            f"GPU-{mock_handles.index(h) + 1}23"
        )
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlShutdown.return_value = None

        # First call should build the map
        NvmlCudaPlatform.device_id_to_physical_device_id(0)
        first_call_count = mock_pynvml.nvmlDeviceGetCount.call_count

        # Second call should use cached map
        NvmlCudaPlatform.device_id_to_physical_device_id(1)
        second_call_count = mock_pynvml.nvmlDeviceGetCount.call_count

        # nvmlDeviceGetCount should only be called once due to caching
        assert second_call_count == first_call_count

    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": " 0 , 1 , 2 "})
    def test_whitespace_handling(self):
        """Test that whitespace in CUDA_VISIBLE_DEVICES is handled correctly."""
        assert NvmlCudaPlatform.device_id_to_physical_device_id(0) == 0
        assert NvmlCudaPlatform.device_id_to_physical_device_id(1) == 1
        assert NvmlCudaPlatform.device_id_to_physical_device_id(2) == 2

    @patch("vllm.platforms.cuda.pynvml")
    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": " GPU-123 , GPU-456 "})
    def test_whitespace_with_uuid(self, mock_pynvml):
        """Test whitespace handling with UUIDs."""
        # Setup mocks
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_handles = [Mock(), Mock()]
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = lambda idx: mock_handles[idx]
        mock_pynvml.nvmlDeviceGetUUID.side_effect = lambda h: (
            "GPU-123" if h == mock_handles[0] else "GPU-456"
        )
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlShutdown.return_value = None

        # Should handle whitespace correctly
        assert NvmlCudaPlatform.device_id_to_physical_device_id(0) == 0
        assert NvmlCudaPlatform.device_id_to_physical_device_id(1) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
