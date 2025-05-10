# SPDX-License-Identifier: Apache-2.0

"""Common utilities for TPU support across vllm.

This module provides reusable components for TPU integration, including:
- Memory management utilities
- XLA compilation helpers
- TPU-specific initialization
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.profiler as xp
    import torch_xla.runtime as xr
    _HAS_TPU = True
except ImportError:
    _HAS_TPU = False


def is_tpu_available() -> bool:
    """Check if TPU support is available."""
    return _HAS_TPU


def get_xla_device() -> Optional[torch.device]:
    """Get the XLA device if TPU support is available.

    Returns:
        Optional[torch.device]: The XLA device if available, otherwise None.
    """
    if not _HAS_TPU:
        return None
    return xm.xla_device()


def init_tpu_runtime(cache_path: Optional[str] = None,
                     global_seed: Optional[int] = None) -> None:
    """Initialize TPU runtime.

    Args:
        cache_path: Path to cache XLA compilation artifacts.
        global_seed: Seed for XLA RNG state.
    """
    if not _HAS_TPU:
        return

    # Set environment variables for TPU.
    os.environ["PJRT_DEVICE"] = "TPU"

    # XLA compilation cache configuration.
    if cache_path:
        world_size = xr.global_replica_count()
        rank = xr.global_ordinal()
        per_rank_path = os.path.join(cache_path, f"tp{world_size}_rank_{rank}")
        xr.initialize_cache(per_rank_path, readonly = False)

    # Set random seed if provided
    if global_seed is not None:
        device = xm.xla_device()
        xm.set_rng_state(global_seed, device)


def compile_model_for_tpu(model: nn.Module,
                          dynamic: bool = False) -> nn.Module:
    """Compile a PyTorch model for TPU execution.

    Args:
        model: The PyTorch model to compile.
        dynamic: Whether to use dynamic shapes.

    Returns:
        The compiled model.
    """
    if not _HAS_TPU:
        return model

    return torch.compile(model,
                         backend="openxla",
                         fullgraph=True,
                         dynamic=dynamic)


def wait_for_tpu_operations() -> None:
    """Wait for TPU operations to complete."""
    if not _HAS_TPU:
        return

    xm.wait_device_ops()


def get_tpu_memory_info() -> Dict[str, int]:
    """Get TPU memory usage information.

    Returns:
        Dict containing memory info with keys:
        - bytes_limit: Total available memory.
        - peak_bytes: Peak memory usage.
        - bytes_used: Current memory usage.
    """
    if not _HAS_TPU:
        return {"bytes_limit": 0, "peak_bytes": 0, "bytes_used": 0}

    device = xm.xla_device()
    return xm.get_memory_info(device)


def update_weight_efficiently(param: torch.Tensor,
                              loaded_weight: torch.Tensor) -> torch.Tensor:
    """Update model parameter weight efficiently for TPU.

    TPU requires special handling for weight loading to avoid memory duplication.

    Args:
        param: Target parameter tensor.
        loaded_weight: Source weight tensor.

    Returns:
        Updated parameter tensor.
    """
    if not _HAS_TPU:
        param.copy_(loaded_weight)
        return param

    # On TPU, copy_() is not in-place and can cause memory issues
    # Instead, we replace the parameter data directly
    return loaded_weight.to(param.device)


def start_tpu_profiling(profile_dir: str) -> Optional[Any]:
    """Start TPU profiling.

    Args:
        profile_dir: Directory to save profiling traces.

    Returns:
        TPU profiler instance or None if not available.
    """
    if not _HAS_TPU:
        return None

    profiler = xp.start_server(9012)
    xp.start_trace(profile_dir)
    return profiler


def stop_tpu_profiling() -> None:
    """Stop active TPU profiling session."""
    if not _HAS_TPU:
        return

    xp.stop_trace()


def pad_to_multiple(value: int, multiple: int) -> int:
    """Pad a value to the next multiple.

    Args:
        value: The value to pad.
        multiple: The multiple to pad to.

    Returns:
        Padded value.
    """
    return ((value + multiple - 1) // multiple) * multiple


def get_padded_power_of_two(x: int, min_value: int = 16) -> int:
    """Pad to the next power of 2, with a minimum value.

    Args:
        x: The value to pad.
        min_value: The minimum value (must be power of 2).

    Returns:
        Next power of 2 >= max(x, min_value).
    """
    if x <= min_value:
        return min_value
    return 1 << (x - 1).bit_length()