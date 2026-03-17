# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion integration for vLLM."""

import vllm.kernels.helion.ops  # noqa: F401  Auto-register all Helion ops
from vllm.kernels.helion.config_manager import (
    ConfigManager,
    ConfigSet,
)
from vllm.kernels.helion.platforms import (
    CORE_PLATFORMS,
)
from vllm.kernels.helion.register import (
    ConfiguredHelionKernel,
    HelionKernelWrapper,
    get_kernel,
    get_registered_kernels,
    register_kernel,
    resolve_kernel,
    vllm_helion_lib,
)
from vllm.kernels.helion.utils import canonicalize_gpu_name, get_canonical_gpu_name

__all__ = [
    # Config management
    "ConfigManager",
    "ConfigSet",
    # Kernel registration
    "ConfiguredHelionKernel",
    "HelionKernelWrapper",
    "get_kernel",
    "get_registered_kernels",
    "register_kernel",
    "resolve_kernel",
    "vllm_helion_lib",
    # Platform tiers
    "CORE_PLATFORMS",
    # Utilities
    "canonicalize_gpu_name",
    "get_canonical_gpu_name",
]
