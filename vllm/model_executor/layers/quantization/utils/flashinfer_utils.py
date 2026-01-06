# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Import everything directly from the real utils
import warnings

from vllm.utils.flashinfer_utils import (
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

warnings.warn(
    "vllm.model_executor.layers.quantization.utils.flashinfer_utils is deprecated. "
    "Please import from vllm.utils.flashinfer_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Make mypy happy with __all__
__all__ = [
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
