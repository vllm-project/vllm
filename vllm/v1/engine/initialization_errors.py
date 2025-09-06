# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enhanced error handling and logging for vLLM V1 initialization."""

from typing import Optional

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import GiB_bytes

logger = init_logger(__name__)


class V1InitializationError(Exception):
    """Base class for vLLM V1 initialization errors with enhanced messaging."""

    pass


class InsufficientMemoryError(V1InitializationError):
    """Raised when there is insufficient GPU memory to initialize the model."""

    def __init__(
        self,
        required_memory: int,
        available_memory: int,
        memory_type: str = "GPU",
        suggestions: Optional[list[str]] = None,
    ):
        # Validate memory values to prevent invalid states
        if required_memory < 0:
            raise ValueError(
                f"Required memory cannot be negative: {required_memory}")
        if available_memory < 0:
            raise ValueError(
                f"Available memory cannot be negative: {available_memory}")

        self.required_memory = required_memory
        self.available_memory = available_memory
        self.memory_type = memory_type
        self.suggestions = suggestions or []

        # Handle edge case where required_memory is 0
        if required_memory == 0:
            message = (
                f"Invalid {memory_type} memory configuration: "
                f"required memory is 0.\n"
                f"This may indicate a configuration or profiling error.\n"
                f"Available: {available_memory / GiB_bytes:.2f} GiB\n")
        else:
            required_gib = required_memory / GiB_bytes
            available_gib = available_memory / GiB_bytes
            shortage_gib = (required_memory - available_memory) / GiB_bytes

            message = (
                f"Insufficient {memory_type} memory to load the model.\n"
                f"Required: {required_gib:.2f} GiB\n"
                f"Available: {available_gib:.2f} GiB\n"
                f"Shortage: {shortage_gib:.2f} GiB\n")

        if self.suggestions:
            message += "\nSuggestions to resolve this issue:\n"
            for i, suggestion in enumerate(self.suggestions, 1):
                message += f"  {i}. {suggestion}\n"

        super().__init__(message)


class InsufficientKVCacheMemoryError(V1InitializationError):
    """Raised when there is insufficient memory for KV cache."""

    def __init__(
        self,
        required_kv_memory: int,
        available_kv_memory: int,
        max_model_len: int,
        estimated_max_len: Optional[int] = None,
        suggestions: Optional[list[str]] = None,
    ):
        # Validate memory values to prevent invalid states
        if required_kv_memory < 0:
            raise ValueError(f"Required KV cache memory cannot be negative: "
                             f"{required_kv_memory}")
        if available_kv_memory < 0:
            raise ValueError(f"Available KV cache memory cannot be negative: "
                             f"{available_kv_memory}")
        if max_model_len <= 0:
            raise ValueError(
                f"Max model length must be positive: {max_model_len}")

        self.required_kv_memory = required_kv_memory
        self.available_kv_memory = available_kv_memory
        self.max_model_len = max_model_len
        self.estimated_max_len = estimated_max_len
        self.suggestions = suggestions or []

        # Handle edge case where required_kv_memory is 0
        if required_kv_memory == 0:
            message = (
                f"Invalid KV cache memory configuration: "
                f"required memory is 0.\n"
                f"This may indicate a configuration or calculation error.\n"
                f"Available KV cache memory: "
                f"{available_kv_memory / GiB_bytes:.2f} GiB\n"
                f"Max model length: {max_model_len}\n")
        else:
            required_gib = required_kv_memory / GiB_bytes
            available_gib = available_kv_memory / GiB_bytes
            shortage_gib = (required_kv_memory -
                            available_kv_memory) / GiB_bytes

            message = (f"Insufficient memory for KV cache to serve requests.\n"
                       f"Required KV cache memory: {required_gib:.2f} GiB "
                       f"(for max_model_len={max_model_len})\n"
                       f"Available KV cache memory: {available_gib:.2f} GiB\n"
                       f"Shortage: {shortage_gib:.2f} GiB\n")

        if self.estimated_max_len and self.estimated_max_len > 0:
            message += (f"Based on available memory, estimated maximum "
                        f"model length: {self.estimated_max_len}\n")

        if self.suggestions:
            message += "\nSuggestions to resolve this issue:\n"
            for i, suggestion in enumerate(self.suggestions, 1):
                message += f"  {i}. {suggestion}\n"

        super().__init__(message)


class ModelLoadingError(V1InitializationError):
    """Raised when model loading fails during initialization."""

    def __init__(
        self,
        model_name: str,
        error_details: str,
        suggestions: Optional[list[str]] = None,
    ):
        self.model_name = model_name
        self.error_details = error_details
        self.suggestions = suggestions or []

        message = (
            f"Failed to load model '{model_name}' during initialization.\n"
            f"Error details: {error_details}\n")

        if self.suggestions:
            message += "\nSuggestions to resolve this issue:\n"
            for i, suggestion in enumerate(self.suggestions, 1):
                message += f"  {i}. {suggestion}\n"

        super().__init__(message)


def log_initialization_info(vllm_config: VllmConfig) -> None:
    """Log detailed initialization information for debugging."""
    logger.info("=== vLLM V1 Initialization Details ===")
    logger.info("Model: %s", vllm_config.model_config.model)
    logger.info("Max model length: %s", vllm_config.model_config.max_model_len)
    logger.info("Data type: %s", vllm_config.model_config.dtype)
    logger.info(
        "GPU memory utilization: %s",
        vllm_config.cache_config.gpu_memory_utilization,
    )

    if torch.cuda.is_available():
        free_memory, total_memory = torch.cuda.mem_get_info()
        logger.info(
            "GPU memory - Total: %.2f GiB, Free: %.2f GiB",
            total_memory / GiB_bytes,
            free_memory / GiB_bytes,
        )

    logger.info(
        "Tensor parallel size: %s",
        vllm_config.parallel_config.tensor_parallel_size,
    )
    logger.info(
        "Pipeline parallel size: %s",
        vllm_config.parallel_config.pipeline_parallel_size,
    )


def get_memory_suggestions(
    required_memory: int,
    available_memory: int,
    current_gpu_utilization: float,
    max_model_len: int,
    is_kv_cache: bool = False,
) -> list[str]:
    """Generate helpful suggestions for memory-related errors."""
    suggestions = []

    # Avoid division by zero if required_memory is 0
    if required_memory > 0:
        shortage_ratio = (required_memory - available_memory) / required_memory
    else:
        # If required memory is 0, treat as no shortage
        # (shouldn't trigger suggestions)
        shortage_ratio = 0.0

    if is_kv_cache:
        suggestions.extend([
            f"Reduce max_model_len from {max_model_len} to a smaller value",
            f"Increase gpu_memory_utilization from "
            f"{current_gpu_utilization:.2f} (e.g., to "
            f"{min(current_gpu_utilization + 0.1, 0.95):.2f})",
            "Consider using quantization (GPTQ, AWQ, FP8) to reduce "
            "memory usage",
            "Use tensor parallelism to distribute the model across "
            "multiple GPUs",
        ])
    else:
        suggestions.extend([
            f"Increase gpu_memory_utilization from "
            f"{current_gpu_utilization:.2f} (e.g., to "
            f"{min(current_gpu_utilization + 0.1, 0.95):.2f})",
            "Consider using quantization (GPTQ, AWQ, FP8) to reduce "
            "model memory usage",
            "Use tensor parallelism to distribute the model across "
            "multiple GPUs",
            "Close other GPU processes to free up memory",
        ])

    if shortage_ratio > 0.5:
        suggestions.insert(0, "Consider using a smaller model variant")

    if current_gpu_utilization < 0.8:
        suggestions.insert(
            0, "Try increasing gpu_memory_utilization first "
            "(safest option)")

    return suggestions


def get_cuda_error_suggestions(error_msg: str) -> list[str]:
    """Generate suggestions based on CUDA error messages."""
    suggestions = []

    error_lower = error_msg.lower()

    if "out of memory" in error_lower or "cuda_out_of_memory" in error_lower:
        suggestions.extend([
            "Reduce gpu_memory_utilization to leave more memory for "
            "CUDA operations",
            "Reduce max_model_len to decrease KV cache memory usage",
            "Use quantization to reduce model memory footprint",
            "Consider tensor parallelism to distribute memory across GPUs",
            "Close other GPU processes that might be using memory",
        ])
    elif "device-side assert" in error_lower:
        suggestions.extend([
            "Check if the model is compatible with your CUDA version",
            "Verify model configuration parameters are correct",
            "Try using eager execution mode (set enforce_eager=True)",
        ])
    elif "invalid device" in error_lower:
        suggestions.extend([
            "Verify CUDA devices are available and accessible",
            "Check if tensor_parallel_size matches available GPUs",
            "Ensure CUDA_VISIBLE_DEVICES is set correctly",
        ])

    return suggestions
