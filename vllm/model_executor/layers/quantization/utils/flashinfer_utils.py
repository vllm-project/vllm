# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatible re-export stub.

All utilities that used to live here have been moved to
:mod:`vllm.utils.flashinfer` (generic helpers) and
:mod:`vllm.model_executor.layers.fused_moe.flashinfer_utils`
(layer-coupled helpers) so that every FlashInfer-related helper
lives in a canonical place.  This module re-exports them so that
existing ``from ...flashinfer_utils import ...`` statements keep working.
"""

from vllm.model_executor.layers.fused_moe.flashinfer_utils import (  # noqa: F401
    apply_fi_trtllm_fp8_per_tensor_moe,
    prepare_fp8_moe_layer_for_fi,
    register_scales_for_trtllm_fp8_per_tensor_moe,
)
from vllm.utils.flashinfer import (  # noqa: F401
    FlashinferMoeBackend,
    activation_to_flashinfer_int,
    align_fp4_moe_weights_for_fi,
    align_fp8_moe_weights_for_fi,
    convert_moe_weights_to_flashinfer_trtllm_block_layout,
    get_flashinfer_moe_backend,
    is_flashinfer_supporting_global_sf,
    make_fp8_moe_alpha_scales_for_fi,
    rotate_weights_for_fi_trtllm_fp8_per_tensor_moe,
    swap_w13_to_w31,
)
