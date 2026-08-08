# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import os

from .aiter import *
from .tilelang import *
from .torch import *
from .triton import *

logger = logging.getLogger(__name__)

# Environment variable to control optimized fusions
# VLLM_MHC_FUSED_KERNELS=1 (default): Enable optimized fusions
# VLLM_MHC_FUSED_KERNELS=0: Disable optimized fusions, use original implementation
_MHC_FUSED_KERNELS_ENABLED = os.environ.get("VLLM_MHC_FUSED_KERNELS", "1") == "1"

# Import optimized fused kernels if TileLang is available and enabled
if _MHC_FUSED_KERNELS_ENABLED:
    try:
        from .optimized_wrappers import (
            mhc_post_hc_head_fused,
            mhc_post_hc_head_norm_fused,
            mhc_post_mean_fused,
        )

        _HAS_OPTIMIZED_FUSIONS = True
        logger.info("MHC optimized fused kernels enabled (VLLM_MHC_FUSED_KERNELS=1)")
    except ImportError as e:
        _HAS_OPTIMIZED_FUSIONS = False
        mhc_post_hc_head_fused = None  # type: ignore
        mhc_post_hc_head_norm_fused = None  # type: ignore
        mhc_post_mean_fused = None  # type: ignore
        logger.info(
            "MHC optimized fused kernels not available (import failed: %s)", str(e)
        )
else:
    _HAS_OPTIMIZED_FUSIONS = False
    mhc_post_hc_head_fused = None  # type: ignore
    mhc_post_hc_head_norm_fused = None  # type: ignore
    mhc_post_mean_fused = None  # type: ignore
    logger.info(
        "MHC optimized fused kernels disabled by environment variable "
        "(VLLM_MHC_FUSED_KERNELS=0)"
    )

__all__ = [
    "mhc_pre_cuda",
    "mhc_post_cuda",
    "mhc_fused_post_pre_cuda",
    "hc_head_fused_kernel_cuda",
    "mhc_pre_aiter",
    "mhc_post_aiter",
    "mhc_fused_post_pre_aiter",
    "hc_head_fused_aiter",
    "mhc_pre_tilelang",
    "mhc_post_tilelang",
    "mhc_fused_post_pre_tilelang",
    "hc_head_fused_tilelang",
    "mhc_pre_torch",
    "mhc_post_torch",
    "mhc_fused_post_pre_torch",
    "hc_head_fused_torch",
    "mhc_pre_triton",
    "mhc_post_triton",
    "mhc_fused_post_pre_triton",
    "hc_head_fused_triton",
    "mhc_post_hc_head_fused",
    "mhc_post_hc_head_norm_fused",
    "mhc_post_mean_fused",
    "_HAS_OPTIMIZED_FUSIONS",
    "_MHC_FUSED_KERNELS_ENABLED",
]
