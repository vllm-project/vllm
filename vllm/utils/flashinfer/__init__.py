# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer utilities package.

This package provides a unified interface to FlashInfer functionality,
consolidating utilities that were previously spread across multiple modules.

Submodules:
    - core: General FlashInfer compatibility and availability checks
    - moe: MoE-specific utilities (kernels, weight manipulation, backends)

For backwards compatibility, all public symbols are re-exported from this
module, so existing imports like `from vllm.utils.flashinfer import ...`
will continue to work.
"""

# Re-export everything from core module (general flashinfer utilities)
from vllm.utils.flashinfer.core import (
    autotune,
    can_use_trtllm_attention,
    flashinfer_cutedsl_grouped_gemm_nt_masked,
    flashinfer_cutlass_fused_moe,
    flashinfer_fp4_quantize,
    flashinfer_scaled_fp4_mm,
    flashinfer_scaled_fp8_mm,
    flashinfer_trtllm_fp8_block_scale_moe,
    flashinfer_trtllm_fp8_per_tensor_scale_moe,
    force_use_trtllm_attention,
    has_flashinfer,
    has_flashinfer_all2all,
    has_flashinfer_comm,
    has_flashinfer_cubin,
    has_flashinfer_cutedsl,
    has_flashinfer_cutedsl_grouped_gemm_nt_masked,
    has_flashinfer_cutlass_fused_moe,
    has_flashinfer_moe,
    has_flashinfer_trtllm_fused_moe,
    has_nvidia_artifactory,
    nvfp4_batched_quantize,
    nvfp4_block_scale_interleave,
    scaled_fp4_grouped_quantize,
    silu_and_mul_scaled_nvfp4_experts_quantize,
    supports_trtllm_attention,
    trtllm_fp4_block_scale_moe,
    use_trtllm_attention,
)

# Re-export everything from moe module (MoE-specific utilities)
from vllm.utils.flashinfer.moe import (
    FlashinferMoeBackend,
    apply_flashinfer_per_tensor_scale_fp8,
    build_flashinfer_fp8_cutlass_moe_prepare_finalize,
    calculate_tile_tokens_dim,
    flashinfer_cutlass_moe_fp8,
    get_flashinfer_moe_backend,
    get_moe_scaling_factors,
    is_flashinfer_supporting_global_sf,
    register_moe_scaling_factors,
    rotate_flashinfer_fp8_moe_weights,
    select_cutlass_fp8_gemm_impl,
    swap_w13_to_w31,
)

__all__ = [
    # Core module - availability checks
    "has_flashinfer",
    "has_flashinfer_cubin",
    "has_flashinfer_moe",
    "has_flashinfer_comm",
    "has_flashinfer_all2all",
    "has_flashinfer_cutedsl",
    "has_flashinfer_trtllm_fused_moe",
    "has_flashinfer_cutlass_fused_moe",
    "has_flashinfer_cutedsl_grouped_gemm_nt_masked",
    "has_nvidia_artifactory",
    # Core module - lazy import wrappers
    "flashinfer_trtllm_fp8_block_scale_moe",
    "flashinfer_trtllm_fp8_per_tensor_scale_moe",
    "flashinfer_cutlass_fused_moe",
    "flashinfer_cutedsl_grouped_gemm_nt_masked",
    "flashinfer_fp4_quantize",
    "nvfp4_batched_quantize",
    "silu_and_mul_scaled_nvfp4_experts_quantize",
    "scaled_fp4_grouped_quantize",
    "nvfp4_block_scale_interleave",
    "trtllm_fp4_block_scale_moe",
    "autotune",
    # Core module - TRTLLM attention
    "supports_trtllm_attention",
    "force_use_trtllm_attention",
    "can_use_trtllm_attention",
    "use_trtllm_attention",
    # Core module - helper functions
    "flashinfer_scaled_fp4_mm",
    "flashinfer_scaled_fp8_mm",
    # MoE module
    "FlashinferMoeBackend",
    "calculate_tile_tokens_dim",
    "swap_w13_to_w31",
    "rotate_flashinfer_fp8_moe_weights",
    "apply_flashinfer_per_tensor_scale_fp8",
    "get_moe_scaling_factors",
    "register_moe_scaling_factors",
    "build_flashinfer_fp8_cutlass_moe_prepare_finalize",
    "select_cutlass_fp8_gemm_impl",
    "flashinfer_cutlass_moe_fp8",
    "get_flashinfer_moe_backend",
    "is_flashinfer_supporting_global_sf",
]
