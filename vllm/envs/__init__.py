# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Environment variable management for vLLM.

This module provides access to all vLLM environment variables with lazy evaluation
and type conversion. Environment variables are looked up from the actual OS
environment using os.getenv() with appropriate type conversion and default values.

The module maintains backwards compatibility with the original vllm.envs interface
while providing a cleaner separation between variable definitions and environment
lookups.

Usage:
    import vllm.envs as envs
    
    # Access environment variables
    device = envs.VLLM_TARGET_DEVICE  # Returns string value
    port = envs.VLLM_PORT            # Returns int or None
    
    # Check if variable is explicitly set
    if envs.is_set("VLLM_USE_V1"):
        print("V1 is explicitly configured")
    
    # Get all available variables
    all_vars = dir(envs)
"""

import hashlib
import os
import tempfile
from typing import TYPE_CHECKING, get_type_hints, Type, Union, get_origin, get_args, Optional
from urllib.parse import urlparse

from . import _variables
from ._variables import __defaults as _env_defaults

if TYPE_CHECKING:
    # This way IDEs & type checkers get the declarations directly
    from ._variables import *


def get_default_cache_root() -> str:
    """Get the default cache root directory."""
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root() -> str:
    """Get the default config root directory."""
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def _unwrap_optional(type_: Type) -> Type:
    """Unwrap Optional[T] to get T."""
    origin = get_origin(type_)
    if origin is not Union:
        return type_

    args = get_args(type_)
    if len(args) != 2 or type(None) not in args:
        raise ValueError("Unions not currently supported")

    return next(arg for arg in args if arg is not type(None))


def _get_vllm_port() -> Optional[int]:
    """Get the port from VLLM_PORT environment variable with special validation."""
    if 'VLLM_PORT' not in os.environ:
        return None

    port = os.getenv('VLLM_PORT', '0')

    try:
        return int(port)
    except ValueError as err:
        parsed = urlparse(port)
        if parsed.scheme:
            raise ValueError(
                f"VLLM_PORT '{port}' appears to be a URI. "
                "This may be caused by a Kubernetes service discovery issue,"
                "check the warning in: https://docs.vllm.ai/en/stable/serving/env_vars.html"
            ) from None
        raise ValueError(
            f"VLLM_PORT '{port}' must be a valid integer") from err


def _parse_list_value(value: str) -> list[str]:
    """Parse comma-separated string into list."""
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


_type_hints = get_type_hints(_variables)


def __getattr__(name: str):
    """Lazy evaluation of environment variables with standardized parsing."""
    if name not in _env_defaults:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    # Special handling for complex variables
    if name == "VLLM_PORT":
        return _get_vllm_port()
    
    # Handle variables that need path expansion
    if name == "VLLM_CONFIG_ROOT":
        return os.path.expanduser(
            os.getenv(
                "VLLM_CONFIG_ROOT",
                os.path.join(get_default_config_root(), "vllm"),
            )
        )
    
    if name == "VLLM_CACHE_ROOT":
        return os.path.expanduser(
            os.getenv(
                "VLLM_CACHE_ROOT",
                os.path.join(get_default_cache_root(), "vllm"),
            )
        )
    
    if name == "VLLM_ASSETS_CACHE":
        return os.path.expanduser(
            os.getenv(
                "VLLM_ASSETS_CACHE",
                os.path.join(get_default_cache_root(), "vllm", "assets"),
            )
        )
    
    if name == "VLLM_XLA_CACHE_PATH":
        return os.path.expanduser(
            os.getenv(
                "VLLM_XLA_CACHE_PATH",
                os.path.join(get_default_cache_root(), "vllm", "xla_cache"),
            )
        )
    
    if name == "VLLM_RPC_BASE_PATH":
        return os.getenv('VLLM_RPC_BASE_PATH', tempfile.gettempdir())
    
    # Handle special cases for compound logic
    if name == "VLLM_USE_PRECOMPILED":
        return (os.environ.get("VLLM_USE_PRECOMPILED", "").strip().lower() in
                ("1", "true") or bool(os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION")))
    
    if name == "VLLM_DO_NOT_TRACK":
        return (os.environ.get("VLLM_DO_NOT_TRACK", None) or 
                os.environ.get("DO_NOT_TRACK", None) or "0") == "1"
    
    if name == "VLLM_DP_RANK_LOCAL":
        return int(os.getenv("VLLM_DP_RANK_LOCAL", os.getenv("VLLM_DP_RANK", "0")))
    
    if name == "VLLM_TPU_USING_PATHWAYS":
        return bool("proxy" in os.getenv("JAX_PLATFORMS", "").lower())
    
    if name == "VLLM_TORCH_PROFILER_DIR":
        value = os.getenv("VLLM_TORCH_PROFILER_DIR", None)
        return None if value is None else os.path.abspath(os.path.expanduser(value))

    # Get environment value
    env_value = os.getenv(name)
    if env_value is None:
        return _env_defaults[name]

    # Get type for this variable
    var_type = _type_hints[name]
    var_type = _unwrap_optional(var_type)

    # Parse based on type
    if var_type is str:
        # Handle special string parsing
        if name == "VLLM_TARGET_DEVICE":
            return env_value.lower()
        if name == "VLLM_LOGGING_LEVEL":
            return env_value.upper()
        if name == "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION":
            return env_value.upper()
        if name == "VLLM_MOE_ROUTING_SIMULATION_STRATEGY":
            return env_value.lower()
        return env_value
    
    if var_type is bool:
        return env_value.lower() in ("1", "true")
    
    if var_type in (int, float):
        return var_type(env_value)
    
    if var_type == list[str] or (hasattr(var_type, '__origin__') and var_type.__origin__ is list):
        return _parse_list_value(env_value)

    raise ValueError(f"Unsupported type {var_type} for environment variable {name}")


def __dir__():
    """Return list of available environment variables."""
    return list(_env_defaults.keys())


def is_set(name: str) -> bool:
    """Check if an environment variable is explicitly set."""
    if name not in _env_defaults:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return name in os.environ


def set_vllm_use_v1(use_v1: bool) -> None:
    """Set VLLM_USE_V1 environment variable."""
    if is_set("VLLM_USE_V1"):
        raise ValueError(
            "Should not call set_vllm_use_v1() if VLLM_USE_V1 is set "
            "explicitly by the user. Please raise this as a Github "
            "Issue and explicitly set VLLM_USE_V1=0 or 1.")
    os.environ["VLLM_USE_V1"] = "1" if use_v1 else "0"


def compute_hash() -> str:
    """
    Compute hash of environment variables that affect computation graph.
    
    WARNING: Whenever a new key is added to the environment variables, 
    ensure that it is included in the factors list if it affects the 
    computation graph. For example, different values of 
    VLLM_PP_LAYER_PARTITION will generate different computation graphs, 
    so it is included in the factors list. The env vars that affect 
    the choice of different kernels or attention backends should also 
    be included in the factors list.
    """
    # The values of envs may affects the computation graph.
    environment_variables_to_hash = [
        "VLLM_PP_LAYER_PARTITION",
        "VLLM_MLA_DISABLE",
        "VLLM_USE_TRITON_FLASH_ATTN",
        "VLLM_USE_TRITON_AWQ",
        "VLLM_DP_RANK",
        "VLLM_DP_SIZE",
        "VLLM_USE_STANDALONE_COMPILE",
        "VLLM_FUSED_MOE_CHUNK_SIZE",
        "VLLM_FLASHINFER_MOE_BACKEND",
        "VLLM_V1_USE_PREFILL_DECODE_ATTENTION",
        "VLLM_USE_AITER_UNIFIED_ATTENTION",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_USE_FLASHINFER_SAMPLER",
        "VLLM_DISABLED_KERNELS",
        "VLLM_USE_DEEP_GEMM",
        "VLLM_USE_TRTLLM_FP4_GEMM",
        "VLLM_USE_FUSED_MOE_GROUPED_TOPK",
        "VLLM_USE_FLASHINFER_MOE_FP8",
        "VLLM_USE_FLASHINFER_MOE_FP4",
        "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8",
        "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16",
        "VLLM_USE_CUDNN_PREFILL",
        "VLLM_USE_TRTLLM_ATTENTION",
        "VLLM_ROCM_USE_AITER",
        "VLLM_ROCM_USE_AITER_PAGED_ATTN",
        "VLLM_ROCM_USE_AITER_LINEAR",
        "VLLM_ROCM_USE_AITER_MOE",
        "VLLM_ROCM_USE_AITER_RMSNORM",
        "VLLM_ROCM_USE_AITER_MLA",
        "VLLM_ROCM_USE_AITER_MHA",
        "VLLM_ROCM_USE_SKINNY_GEMM",
        "VLLM_ROCM_FP8_PADDING",
        "VLLM_ROCM_MOE_PADDING",
        "VLLM_ROCM_CUSTOM_PAGED_ATTN",
        "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION",
        "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16",
        "VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB",
    ]
    
    for key in environment_variables_to_hash:
        # if this goes out of sync with _env_defaults,
        # it's not a user error, it's a bug
        assert key in _env_defaults, \
            f"Please update environment_variables_to_hash in envs/__init__.py. Missing: {key}"

    factors = [
        getattr(__import__(__name__), key) for key in environment_variables_to_hash
    ]

    hash_str = hashlib.md5(str(factors).encode(),
                           usedforsecurity=False).hexdigest()

    return hash_str