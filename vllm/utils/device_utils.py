"""Device and GPU utilities for vLLM.

This module provides utilities for GPU device management, memory
monitoring, and hardware-specific operations in vLLM.
"""

from contextlib import contextmanager
from typing import Optional, Dict, Generator, List, Union
import os

import torch


# Cache for device properties to avoid repeated API calls
_device_properties_cache: Dict[int, torch.cuda.device_prop] = {}


def get_device_property(device: int, property_name: str):
    """Get a property of a CUDA device.
    
    This function provides a unified interface to query various
    properties of CUDA devices, with caching to reduce overhead.
    
    Args:
        device: The CUDA device ID (0-indexed)
        property_name: One of 'name', 'total_memory', 'compute_cap',
                      'multiprocessor_count', 'max_threads_per_block',
                      'max_threads_per_multiprocessor'
    
    Returns:
        The requested property value, or None if the property doesn't
        exist or CUDA is unavailable.
    
    Raises:
        ValueError: If an unknown property_name is requested.
    
    Example:
        >>> name = get_device_property(0, 'name')
        >>> print(f"GPU 0: {name}")
        >>> memory_gb = get_device_property(0, 'total_memory') / (1024**3)
    """
    if not torch.cuda.is_available():
        return None
    
    if device < 0 or device >= torch.cuda.device_count():
        return None
    
    # Use cached properties
    if device not in _device_properties_cache:
        _device_properties_cache[device] = torch.cuda.get_device_properties(device)
    
    props = _device_properties_cache[device]
    
    property_map = {
        'name': props.name,
        'total_memory': props.total_memory,
        'compute_cap': (props.major, props.minor),
        'multiprocessor_count': props.multi_processor_count,
        'max_threads_per_block': props.max_threads_per_block,
        'max_threads_per_multiprocessor': props.max_threads_per_multiprocessor,
        'total_memory_gb': props.total_memory / (1024**3),
        'warp_size': props.warp_size,
    }
    
    if property_name not in property_map:
        raise ValueError(f"Unknown property '{property_name}'. Valid options: {list(property_map.keys())}")
    
    return property_map[property_name]


def get_gpu_name(device: int = 0) -> str:
    """Get the name of a GPU device.
    
    Args:
        device: The CUDA device ID (default: 0)
    
    Returns:
        The GPU name as a string, or "Unknown" if unavailable.
    
    Example:
        >>> print(f"Using GPU: {get_gpu_name()}")
    """
    name = get_device_property(device, 'name')
    return name if name is not None else "Unknown"


def get_gpu_memory_info(device: int = 0) -> Dict[str, float]:
    """Get comprehensive memory information for a GPU device.
    
    Args:
        device: The CUDA device ID (default: 0)
    
    Returns:
        Dictionary containing:
        - 'total_gb': Total GPU memory in GB
        - 'allocated_gb': Currently allocated memory in GB
        - 'reserved_gb': Currently reserved memory in GB
        - 'free_gb': Available (free) memory in GB
        - 'utilization_percent': Memory utilization percentage
    
    Example:
        >>> info = get_gpu_memory_info(0)
        >>> print(f"Used: {info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB")
    """
    if not torch.cuda.is_available():
        return {
            'total_gb': 0.0,
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'free_gb': 0.0,
            'utilization_percent': 0.0,
        }
    
    if device < 0 or device >= torch.cuda.device_count():
        return {
            'total_gb': 0.0,
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'free_gb': 0.0,
            'utilization_percent': 0.0,
        }
    
    torch.cuda.synchronize(device)
    
    props = get_device_property(device, 'total_memory')
    total_memory = props if props is not None else 0
    
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free = total_memory - reserved
    
    return {
        'total_gb': total_memory / (1024**3),
        'allocated_gb': allocated / (1024**3),
        'reserved_gb': reserved / (1024**3),
        'free_gb': free / (1024**3),
        'utilization_percent': (allocated / total_memory * 100) if total_memory > 0 else 0.0,
    }


def get_gpu_utilization(device: int = 0) -> float:
    """Get the current GPU memory utilization percentage.
    
    Args:
        device: The CUDA device ID (default: 0)
    
    Returns:
        Memory utilization as a percentage (0.0 to 100.0),
        or -1.0 if CUDA is unavailable.
    
    Example:
        >>> util = get_gpu_utilization()
        >>> print(f"GPU utilization: {util:.1f}%")
    """
    info = get_gpu_memory_info(device)
    return info['utilization_percent']


def get_available_gpu_memory(device: int = 0) -> float:
    """Get the available (free) GPU memory in GB.
    
    Args:
        device: The CUDA device ID (default: 0)
    
    Returns:
        Available memory in GB, or -1.0 if CUDA is unavailable
        or the device doesn't exist.
    
    Example:
        >>> free_memory = get_available_gpu_memory()
        >>> if free_memory > 10:
        ...     print("Sufficient memory for large batch")
    """
    if not torch.cuda.is_available():
        return -1.0
    
    if device < 0 or device >= torch.cuda.device_count():
        return -1.0
    
    torch.cuda.synchronize(device)
    props = torch.cuda.get_device_properties(device)
    reserved = torch.cuda.memory_reserved(device)
    free = props.total_memory - reserved
    
    return free / (1024**3)


def clear_gpu_caches(device: Optional[int] = None) -> None:
    """Clear GPU memory caches for specified device(s).
    
    This function releases cached memory back to the GPU allocator,
    which can help when switching between different workloads or
    when troubleshooting memory issues.
    
    Args:
        device: Specific GPU ID, or None to clear all devices
    
    Example:
        >>> # Clear cache on current device
        >>> clear_gpu_caches()
        >>> # Clear cache on specific device
        >>> clear_gpu_caches(device=1)
        >>> # Clear cache on all devices
        >>> clear_gpu_caches(device=None)
    """
    if not torch.cuda.is_available():
        return
    
    if device is not None:
        if 0 <= device < torch.cuda.device_count():
            torch.cuda.empty_cache()
    else:
        for _ in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()


@contextmanager
def device_memory_tracing(device: int = 0) -> Generator[Dict[str, float], None, None]:
    """Context manager to trace memory usage before and after a code block.
    
    This provides a convenient way to measure the memory impact of
    specific operations or code sections.
    
    Args:
        device: The CUDA device ID to trace (default: 0)
    
    Yields:
        A dictionary containing memory statistics that gets updated
        with delta information after the context exits.
    
    Example:
        >>> with device_memory_tracing() as mem_before:
        ...     # Memory stats at entry
        ...     pass
        >>> # After exit, mem_before contains delta information
        >>> print(f"Memory delta: {mem_before.get('delta_gb', 0):.3f} GB")
    """
    if not torch.cuda.is_available():
        yield {
            'device': device,
            'before_allocated_gb': 0.0,
            'after_allocated_gb': 0.0,
            'delta_gb': 0.0,
        }
        return
    
    torch.cuda.synchronize(device)
    before_allocated = torch.cuda.memory_allocated(device)
    
    yield {
        'device': device,
        'before_allocated_gb': before_allocated / (1024**3),
    }
    
    # After the context, calculate delta
    torch.cuda.synchronize(device)
    after_allocated = torch.cuda.memory_allocated(device)
    
    yield {
        'device': device,
        'before_allocated_gb': before_allocated / (1024**3),
        'after_allocated_gb': after_allocated / (1024**3),
        'delta_gb': (after_allocated - before_allocated) / (1024**3),
    }


def get_device_count() -> int:
    """Get the number of available CUDA devices.
    
    Returns:
        The number of CUDA devices, or 0 if CUDA is unavailable.
    
    Example:
        >>> n_gpus = get_device_count()
        >>> if n_gpus > 1:
        ...     print(f"Running on {n_gpus} GPUs")
    """
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def is_using_gpu() -> bool:
    """Check if the system has CUDA available.
    
    Returns:
        True if CUDA is available and at least one GPU exists.
    
    Example:
        >>> if is_using_gpu():
        ...     print("GPU acceleration available")
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def get_current_device() -> int:
    """Get the index of the current CUDA device.
    
    Returns:
        The index of the current device, or -1 if no CUDA device.
    
    Example:
        >>> current = get_current_device()
        >>> print(f"Currently using GPU {current}")
    """
    if not torch.cuda.is_available():
        return -1
    return torch.cuda.current_device()


def get_device_name(device: Optional[int] = None) -> str:
    """Get the name of a CUDA device by index or current device.
    
    Args:
        device: The device index, or None to use current device
    
    Returns:
        The GPU name, or "CPU" if no GPU available.
    
    Example:
        >>> print(f"Running on: {get_device_name()}")
    """
    if device is None:
        device = get_current_device()
    
    if device < 0:
        return "CPU"
    
    return get_gpu_name(device)


def estimate_model_memory_requirements(
    num_parameters: int,
    precision: str = "auto",
    num_layers: Optional[int] = None,
    hidden_size: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate memory requirements for a model based on parameters.
    
    This provides rough estimates for planning resource allocation.
    
    Args:
        num_parameters: Total number of parameters in the model
        precision: Data precision ("fp32", "fp16", "bf16", "int8", "auto")
        num_layers: Number of transformer layers (for better estimates)
        hidden_size: Hidden size dimension (for KV cache estimates)
        vocab_size: Vocabulary size (for embedding estimates)
    
    Returns:
        Dictionary with estimated memory requirements in GB:
        - 'weights_gb': Estimated weight memory
        - 'activations_gb': Estimated activation memory
        - 'kv_cache_per_token_gb': KV cache per token
        - 'total_estimate_gb': Total estimated memory
    
    Example:
        >>> est = estimate_model_memory_requirements(
        ...     num_parameters=7_000_000_000,
        ...     precision="bf16"
        ... )
        >>> print(f"Weights: {est['weights_gb']:.2f} GB")
    """
    # Determine bytes per parameter
    if precision == "auto":
        if torch.cuda.is_available():
            # Check current precision
            try:
                current_dtype = torch.cuda.get_device_properties(0).major
                if current_dtype >= 8:  # Ampere or newer
                    precision = "bf16"
                else:
                    precision = "fp16"
            except:
                precision = "bf16"
        else:
            precision = "fp32"
    
    precision_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    
    bytes_per_param = precision_bytes.get(precision.lower(), 2)
    
    # Weight memory
    weight_memory = (num_parameters * bytes_per_param) / (1024**3)
    
    estimates: Dict[str, float] = {
        "weights_gb": weight_memory,
    }
    
    # Activation memory estimate (rough: ~2x params for full activation)
    if num_layers is not None:
        activation_memory = (num_parameters * bytes_per_param * 2) / (1024**3)
        estimates["activations_gb"] = activation_memory
    
    # KV cache estimate (rough per-token estimate)
    if num_layers is not None and hidden_size is not None:
        # For each layer: 2 * hidden_size * bytes_per_param
        kv_per_token = (2 * num_layers * hidden_size * bytes_per_param) / (1024**3)
        estimates["kv_cache_per_token_gb"] = kv_per_token
    
    # Embedding memory (if vocab size known)
    if vocab_size is not None and hidden_size is not None:
        embedding_memory = (vocab_size * hidden_size * bytes_per_param) / (1024**3)
        estimates["embeddings_gb"] = embedding_memory
    
    # Total estimate
    total = weight_memory
    if "activations_gb" in estimates:
        total += estimates["activations_gb"]
    estimates["total_estimate_gb"] = total
    
    return estimates
