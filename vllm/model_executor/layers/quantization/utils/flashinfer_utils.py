# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatible re-export stub.

All utilities that used to live here have been moved to
:mod:`vllm.utils.flashinfer` so that every FlashInfer-related helper
lives in one canonical place.  This module re-exports them so that
existing ``from ...flashinfer_utils import ...`` statements keep working.
"""

from vllm.utils.flashinfer import (  # noqa: F401
    FlashinferMoeBackend,
    align_fp8_moe_weights_for_fi,
    apply_fi_trtllm_fp8_per_tensor_moe,
    convert_moe_weights_to_flashinfer_trtllm_block_layout,
    get_flashinfer_moe_backend,
    is_flashinfer_supporting_global_sf,
    make_fp8_moe_alpha_scales_for_fi,
    prepare_fp8_moe_layer_for_fi,
    register_scales_for_trtllm_fp8_per_tensor_moe,
    rotate_weights_for_fi_trtllm_fp8_per_tensor_moe,
    swap_w13_to_w31,
)
