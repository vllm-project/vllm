# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vLLM V1 initialization error handling."""

from unittest.mock import Mock, patch

import pytest
import torch

from vllm.utils import GiB_bytes
from vllm.v1.engine.initialization_errors import (
    InsufficientKVCacheMemoryError, InsufficientMemoryError, ModelLoadingError,
    V1InitializationError, get_cuda_error_suggestions, get_memory_suggestions,
    log_initialization_info)


class TestV1InitializationError:
    """Test the base V1InitializationError class."""

    def test_base_error_creation(self):
        """Test that base error can be created and raised."""
        error = V1InitializationError("Test error message")
        assert str(error) == "Test error message"

        with pytest.raises(V1InitializationError):
            raise error


class TestInsufficientMemoryError:
    """Test the InsufficientMemoryError class."""

    def test_memory_error_creation(self):
        """Test basic memory error creation with required parameters."""
        required = 8 * GiB_bytes  # 8 GiB
        available = 6 * GiB_bytes  # 6 GiB

        error = InsufficientMemoryError(required, available)

        assert error.required_memory == required
        assert error.available_memory == available
        assert error.memory_type == "GPU"
        assert error.suggestions == []

        error_msg = str(error)
        assert "Insufficient GPU memory to load the model" in error_msg
        assert "Required: 8.00 GiB" in error_msg
        assert "Available: 6.00 GiB" in error_msg
        assert "Shortage: 2.00 GiB" in error_msg

    def test_memory_error_with_custom_type(self):
        """Test memory error with custom memory type."""
        required = 4 * GiB_bytes
        available = 2 * GiB_bytes

        error = InsufficientMemoryError(required, available, memory_type="CPU")

        assert error.memory_type == "CPU"
        error_msg = str(error)
        assert "Insufficient CPU memory to load the model" in error_msg

    def test_memory_error_with_suggestions(self):
        """Test memory error with suggestions."""
        required = 16 * GiB_bytes
        available = 12 * GiB_bytes
        suggestions = [
            "Use quantization to reduce memory usage",
            "Increase GPU memory utilization",
            "Consider tensor parallelism",
        ]

        error = InsufficientMemoryError(required,
                                        available,
                                        suggestions=suggestions)

        assert error.suggestions == suggestions
        error_msg = str(error)
        assert "Suggestions to resolve this issue:" in error_msg
        assert "1. Use quantization to reduce memory usage" in error_msg
        assert "2. Increase GPU memory utilization" in error_msg
        assert "3. Consider tensor parallelism" in error_msg

    def test_memory_error_inheritance(self):
        """Test that InsufficientMemoryError inherits from 
        V1InitializationError."""
        error = InsufficientMemoryError(1000, 500)
        assert isinstance(error, V1InitializationError)
        assert isinstance(error, Exception)

    def test_memory_error_zero_required(self):
        """Test memory error when required memory is 0."""
        error = InsufficientMemoryError(0, 1000)

        assert error.required_memory == 0
        assert error.available_memory == 1000

        error_msg = str(error)
        assert ("Invalid GPU memory configuration: required memory is 0"
                in error_msg)
        assert ("This may indicate a configuration or profiling error"
                in error_msg)
        assert "Available: 0.00 GiB" in error_msg

    def test_memory_error_negative_values(self):
        """Test that negative memory values raise ValueError."""
        with pytest.raises(ValueError,
                           match="Required memory cannot be negative"):
            InsufficientMemoryError(-1000, 500)

        with pytest.raises(ValueError,
                           match="Available memory cannot be negative"):
            InsufficientMemoryError(1000, -500)


class TestInsufficientKVCacheMemoryError:
    """Test the InsufficientKVCacheMemoryError class."""

    def test_kv_cache_error_creation(self):
        """Test basic KV cache error creation."""
        required_kv = 4 * GiB_bytes
        available_kv = 2 * GiB_bytes
        max_model_len = 4096

        error = InsufficientKVCacheMemoryError(required_kv, available_kv,
                                               max_model_len)

        assert error.required_kv_memory == required_kv
        assert error.available_kv_memory == available_kv
        assert error.max_model_len == max_model_len
        assert error.estimated_max_len is None
        assert error.suggestions == []

        error_msg = str(error)
        assert "Insufficient memory for KV cache to serve requests" in error_msg
        assert ("Required KV cache memory: 4.00 GiB (for max_model_len=4096)"
                in error_msg)
        assert "Available KV cache memory: 2.00 GiB" in error_msg
        assert "Shortage: 2.00 GiB" in error_msg

    def test_kv_cache_error_with_estimated_length(self):
        """Test KV cache error with estimated maximum length."""
        required_kv = 8 * GiB_bytes
        available_kv = 4 * GiB_bytes
        max_model_len = 8192
        estimated_max_len = 4096

        error = InsufficientKVCacheMemoryError(required_kv, available_kv,
                                               max_model_len,
                                               estimated_max_len)

        assert error.estimated_max_len == estimated_max_len
        error_msg = str(error)
        assert (
            "Based on available memory, estimated maximum model length: 4096"
            in error_msg)

    def test_kv_cache_error_with_suggestions(self):
        """Test KV cache error with suggestions."""
        required_kv = 6 * GiB_bytes
        available_kv = 3 * GiB_bytes
        max_model_len = 2048
        suggestions = [
            "Reduce max_model_len to a smaller value",
            "Increase gpu_memory_utilization",
        ]

        error = InsufficientKVCacheMemoryError(required_kv,
                                               available_kv,
                                               max_model_len,
                                               suggestions=suggestions)

        assert error.suggestions == suggestions
        error_msg = str(error)
        assert "Suggestions to resolve this issue:" in error_msg
        assert "1. Reduce max_model_len to a smaller value" in error_msg
        assert "2. Increase gpu_memory_utilization" in error_msg

    def test_kv_cache_error_inheritance(self):
        """Test that InsufficientKVCacheMemoryError inherits from 
        V1InitializationError."""
        error = InsufficientKVCacheMemoryError(1000, 500, 1024)
        assert isinstance(error, V1InitializationError)
        assert isinstance(error, Exception)

    def test_kv_cache_error_zero_required(self):
        """Test KV cache error when required memory is 0."""
        error = InsufficientKVCacheMemoryError(0, 1000, 2048)

        assert error.required_kv_memory == 0
        assert error.available_kv_memory == 1000
        assert error.max_model_len == 2048

        error_msg = str(error)
        assert ("Invalid KV cache memory configuration: required memory is 0"
                in error_msg)
        assert ("This may indicate a configuration or calculation error"
                in error_msg)
        assert "Available KV cache memory: 0.00 GiB" in error_msg
        assert "Max model length: 2048" in error_msg

    def test_kv_cache_error_negative_values(self):
        """Test that negative memory values raise ValueError."""
        with pytest.raises(
                ValueError,
                match="Required KV cache memory cannot be negative"):
            InsufficientKVCacheMemoryError(-1000, 500, 1024)

        with pytest.raises(
                ValueError,
                match="Available KV cache memory cannot be negative"):
            InsufficientKVCacheMemoryError(1000, -500, 1024)

        with pytest.raises(ValueError,
                           match="Max model length must be positive"):
            InsufficientKVCacheMemoryError(1000, 500, 0)

        with pytest.raises(ValueError,
                           match="Max model length must be positive"):
            InsufficientKVCacheMemoryError(1000, 500, -1024)


class TestModelLoadingError:
    """Test the ModelLoadingError class."""

    def test_model_loading_error_creation(self):
        """Test basic model loading error creation."""
        model_name = "test-model"
        error_details = "Model file not found"

        error = ModelLoadingError(model_name, error_details)

        assert error.model_name == model_name
        assert error.error_details == error_details
        assert error.suggestions == []

        error_msg = str(error)
        assert ("Failed to load model 'test-model' during initialization"
                in error_msg)
        assert "Error details: Model file not found" in error_msg

    def test_model_loading_error_with_suggestions(self):
        """Test model loading error with suggestions."""
        model_name = "llama-3.2-1b"
        error_details = "CUDA out of memory"
        suggestions = [
            "Check if the model path is correct",
            "Verify CUDA drivers are installed",
            "Use a smaller model variant",
        ]

        error = ModelLoadingError(model_name, error_details, suggestions)

        assert error.suggestions == suggestions
        error_msg = str(error)
        assert "Suggestions to resolve this issue:" in error_msg
        assert "1. Check if the model path is correct" in error_msg
        assert "2. Verify CUDA drivers are installed" in error_msg
        assert "3. Use a smaller model variant" in error_msg

    def test_model_loading_error_inheritance(self):
        """Test that ModelLoadingError inherits from V1InitializationError."""
        error = ModelLoadingError("test", "error")
        assert isinstance(error, V1InitializationError)
        assert isinstance(error, Exception)


class TestLogInitializationInfo:
    """Test the log_initialization_info function."""

    def create_mock_vllm_config(self):
        """Create a mock VllmConfig for testing."""
        model_config = Mock()
        model_config.model = "test-model"
        model_config.max_model_len = 2048
        model_config.dtype = torch.float16

        cache_config = Mock()
        cache_config.gpu_memory_utilization = 0.9

        parallel_config = Mock()
        parallel_config.tensor_parallel_size = 1
        parallel_config.pipeline_parallel_size = 1

        vllm_config = Mock()
        vllm_config.model_config = model_config
        vllm_config.cache_config = cache_config
        vllm_config.parallel_config = parallel_config

        return vllm_config

    @patch("vllm.v1.engine.initialization_errors.logger")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.mem_get_info")
    def test_log_initialization_info_with_cuda(self, mock_mem_info,
                                               mock_cuda_available,
                                               mock_logger):
        """Test logging initialization info when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_mem_info.return_value = (
            8 * GiB_bytes,
            16 * GiB_bytes,
        )  # (free, total)

        vllm_config = self.create_mock_vllm_config()

        log_initialization_info(vllm_config)

        # Verify logger.info was called with expected information
        assert mock_logger.info.call_count >= 6  # At least 6 info calls

        # Check specific logged information
        logged_messages = [
            call[0][0] for call in mock_logger.info.call_args_list
        ]
        assert any("vLLM V1 Initialization Details" in msg
                   for msg in logged_messages)
        assert any("Model: test-model" in msg for msg in logged_messages)
        assert any("Max model length: 2048" in msg for msg in logged_messages)
        assert any("GPU memory utilization: 0.9" in msg
                   for msg in logged_messages)
        assert any("GPU memory - Total: 16.00 GiB, Free: 8.00 GiB" in msg
                   for msg in logged_messages)
        assert any("Tensor parallel size: 1" in msg for msg in logged_messages)
        assert any("Pipeline parallel size: 1" in msg
                   for msg in logged_messages)

    @patch("vllm.v1.engine.initialization_errors.logger")
    @patch("torch.cuda.is_available")
    def test_log_initialization_info_without_cuda(self, mock_cuda_available,
                                                  mock_logger):
        """Test logging initialization info when CUDA is not available."""
        mock_cuda_available.return_value = False

        vllm_config = self.create_mock_vllm_config()

        log_initialization_info(vllm_config)

        # Verify logger.info was called
        assert (mock_logger.info.call_count
                >= 5)  # At least 5 info calls (no GPU memory info)

        # Check that GPU memory info is not logged
        logged_messages = [
            call[0][0] for call in mock_logger.info.call_args_list
        ]
        assert not any("GPU memory - Total:" in msg for msg in logged_messages)


class TestGetMemorySuggestions:
    """Test the get_memory_suggestions function."""

    def test_general_memory_suggestions(self):
        """Test general memory suggestions for model loading."""
        required = 16 * GiB_bytes
        available = 12 * GiB_bytes
        current_utilization = 0.8
        max_model_len = 4096

        suggestions = get_memory_suggestions(
            required,
            available,
            current_utilization,
            max_model_len,
            is_kv_cache=False,
        )

        assert len(suggestions) > 0
        assert any("gpu_memory_utilization" in suggestion
                   for suggestion in suggestions)
        assert any("quantization" in suggestion for suggestion in suggestions)
        assert any("tensor parallelism" in suggestion
                   for suggestion in suggestions)
        assert any("GPU processes" in suggestion for suggestion in suggestions)

    def test_kv_cache_memory_suggestions(self):
        """Test memory suggestions specific to KV cache."""
        required = 8 * GiB_bytes
        available = 6 * GiB_bytes
        current_utilization = 0.9
        max_model_len = 8192

        suggestions = get_memory_suggestions(
            required,
            available,
            current_utilization,
            max_model_len,
            is_kv_cache=True,
        )

        assert len(suggestions) > 0
        assert any("max_model_len" in suggestion for suggestion in suggestions)
        assert any("gpu_memory_utilization" in suggestion
                   for suggestion in suggestions)
        assert any("quantization" in suggestion for suggestion in suggestions)
        assert any("tensor parallelism" in suggestion
                   for suggestion in suggestions)

    def test_large_shortage_suggestions(self):
        """Test suggestions when shortage is more than 50%."""
        required = 16 * GiB_bytes
        available = 4 * GiB_bytes  # 75% shortage
        current_utilization = 0.9
        max_model_len = 2048

        suggestions = get_memory_suggestions(
            required,
            available,
            current_utilization,
            max_model_len,
            is_kv_cache=False,
        )

        # Should suggest using a smaller model variant as first suggestion
        assert "smaller model variant" in suggestions[0]

    def test_low_utilization_suggestions(self):
        """Test suggestions when GPU utilization is low."""
        required = 10 * GiB_bytes
        available = 8 * GiB_bytes
        current_utilization = 0.6  # Low utilization
        max_model_len = 2048

        suggestions = get_memory_suggestions(
            required,
            available,
            current_utilization,
            max_model_len,
            is_kv_cache=False,
        )

        # Should suggest increasing utilization as first suggestion
        assert "increasing gpu_memory_utilization first" in suggestions[0]

    def test_zero_required_memory_suggestions(self):
        """Test suggestions when required memory is 0 (edge case)."""
        required = 0
        available = 8 * GiB_bytes
        current_utilization = 0.8
        max_model_len = 2048

        suggestions = get_memory_suggestions(
            required,
            available,
            current_utilization,
            max_model_len,
            is_kv_cache=False,
        )

        # Should still return suggestions, but shortage_ratio should be 0
        assert len(suggestions) > 0
        # Should not suggest smaller model variant since shortage_ratio is 0
        assert not any("smaller model variant" in suggestion
                       for suggestion in suggestions)

    def test_negative_shortage_suggestions(self):
        """Test suggestions when available > required (negative shortage)."""
        required = 8 * GiB_bytes
        available = 12 * GiB_bytes  # More than required
        current_utilization = 0.8
        max_model_len = 2048

        suggestions = get_memory_suggestions(
            required,
            available,
            current_utilization,
            max_model_len,
            is_kv_cache=False,
        )

        # Should still return suggestions even when there's no shortage
        assert len(suggestions) > 0
        # Should not suggest smaller model variant since shortage is negative
        assert not any("smaller model variant" in suggestion
                       for suggestion in suggestions)


class TestGetCudaErrorSuggestions:
    """Test the get_cuda_error_suggestions function."""

    def test_out_of_memory_suggestions(self):
        """Test suggestions for CUDA out of memory errors."""
        error_msg = "CUDA out of memory. Tried to allocate 4.00 GiB"

        suggestions = get_cuda_error_suggestions(error_msg)

        assert len(suggestions) > 0
        assert any("gpu_memory_utilization" in suggestion
                   for suggestion in suggestions)
        assert any("max_model_len" in suggestion for suggestion in suggestions)
        assert any("quantization" in suggestion for suggestion in suggestions)
        assert any("tensor parallelism" in suggestion
                   for suggestion in suggestions)
        assert any("GPU processes" in suggestion for suggestion in suggestions)

    def test_cuda_out_of_memory_suggestions(self):
        """Test suggestions for specific CUDA_OUT_OF_MEMORY errors."""
        error_msg = "RuntimeError: CUDA_OUT_OF_MEMORY: out of memory"

        suggestions = get_cuda_error_suggestions(error_msg)

        assert len(suggestions) > 0
        assert any("gpu_memory_utilization" in suggestion
                   for suggestion in suggestions)

    def test_device_assert_suggestions(self):
        """Test suggestions for device-side assert errors."""
        error_msg = "RuntimeError: CUDA error: device-side assert triggered"

        suggestions = get_cuda_error_suggestions(error_msg)

        assert len(suggestions) > 0
        assert any("CUDA version" in suggestion for suggestion in suggestions)
        assert any("configuration parameters" in suggestion
                   for suggestion in suggestions)
        assert any("eager execution" in suggestion
                   for suggestion in suggestions)

    def test_invalid_device_suggestions(self):
        """Test suggestions for invalid device errors."""
        error_msg = "RuntimeError: CUDA error: invalid device ordinal"

        suggestions = get_cuda_error_suggestions(error_msg)

        assert len(suggestions) > 0
        assert any("CUDA devices" in suggestion for suggestion in suggestions)
        assert any("tensor_parallel_size" in suggestion
                   for suggestion in suggestions)
        assert any("CUDA_VISIBLE_DEVICES" in suggestion
                   for suggestion in suggestions)

    def test_unknown_error_suggestions(self):
        """Test suggestions for unknown CUDA errors."""
        error_msg = "Some unknown CUDA error occurred"

        suggestions = get_cuda_error_suggestions(error_msg)

        # Should return empty list for unknown errors
        assert suggestions == []

    def test_case_insensitive_matching(self):
        """Test that error matching is case insensitive."""
        error_msg = "RUNTIME ERROR: OUT OF MEMORY occurred"

        suggestions = get_cuda_error_suggestions(error_msg)

        assert len(suggestions) > 0
        assert any("gpu_memory_utilization" in suggestion
                   for suggestion in suggestions)


class TestErrorMessageFormatting:
    """Test that error messages are properly formatted and readable."""

    def test_memory_error_formatting(self):
        """Test that memory error messages are well-formatted."""
        error = InsufficientMemoryError(
            required_memory=10 * GiB_bytes,
            available_memory=6 * GiB_bytes,
            memory_type="GPU",
            suggestions=["Use quantization", "Increase memory utilization"],
        )

        message = str(error)

        # Check that message is properly formatted with newlines
        lines = message.split("\n")
        assert len(lines) > 5
        assert lines[0] == "Insufficient GPU memory to load the model."
        assert "Required: 10.00 GiB" in lines[1]
        assert "Available: 6.00 GiB" in lines[2]
        assert "Shortage: 4.00 GiB" in lines[3]
        assert "Suggestions to resolve this issue:" in lines[5]
        assert "  1. Use quantization" in lines[6]
        assert "  2. Increase memory utilization" in lines[7]

    def test_kv_cache_error_formatting(self):
        """Test that KV cache error messages are well-formatted."""
        error = InsufficientKVCacheMemoryError(
            required_kv_memory=8 * GiB_bytes,
            available_kv_memory=4 * GiB_bytes,
            max_model_len=4096,
            estimated_max_len=2048,
            suggestions=["Reduce max_model_len"],
        )

        message = str(error)
        lines = message.split("\n")

        assert "Insufficient memory for KV cache to serve requests." in lines[
            0]
        assert ("Required KV cache memory: 8.00 GiB (for max_model_len=4096)"
                in lines[1])
        assert "Available KV cache memory: 4.00 GiB" in lines[2]
        assert "Shortage: 4.00 GiB" in lines[3]
        assert (
            "Based on available memory, estimated maximum model length: 2048"
            in lines[4])
        assert "Suggestions to resolve this issue:" in lines[6]
        assert "  1. Reduce max_model_len" in lines[7]


if __name__ == "__main__":
    pytest.main([__file__])
