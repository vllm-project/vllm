# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch/combine dtype selection for the MoRI EP all2all backend.

Kept free of ``import mori`` so it can be imported from the device communicator
without pulling in the (optional) mori package.
"""

from enum import Enum

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

logger = init_logger(__name__)

# Blockwise quantization group sizes: number of elements sharing one scale.
FP8_BLOCK_SIZE = 128
MXFP4_BLOCK_SIZE = 32


class DispatchDtype(Enum):
    bf16 = "bf16"
    fp8 = "fp8"
    fp4 = "fp4"


class CombineDtype(Enum):
    bf16 = "bf16"
    fp8 = "fp8"
    fp8_direct_cast = "fp8_direct_cast"


def combine_quant_type(combine_dtype: CombineDtype) -> str:
    """Map a CombineDtype to the mori ``EpDispatchCombineConfig.quant_type``."""
    if combine_dtype == CombineDtype.fp8:
        return "fp8_blockwise"
    if combine_dtype == CombineDtype.fp8_direct_cast:
        return "fp8_direct_cast"
    return "none"


def _auto_dtypes(
    quant_config: FusedMoEQuantConfig | None,
) -> tuple[DispatchDtype, CombineDtype]:
    """Pick dispatch/combine dtypes from the model weight dtype."""
    if quant_config is None:
        return DispatchDtype.bf16, CombineDtype.bf16

    is_mxfp4_weight = (
        quant_config.use_mxfp4_w4a16
        or quant_config.use_mxfp4_w4a4
        or quant_config.use_mxfp4_w4a8
    )
    if is_mxfp4_weight:
        # FP4 models: fp4 dispatch + fp8 (blockwise) combine is the best
        # accuracy/throughput trade-off (see SGLang amd/DeepSeek-R1 results).
        return DispatchDtype.fp4, CombineDtype.fp8
    if quant_config.use_fp8_w8a8:
        return DispatchDtype.fp8, CombineDtype.bf16
    return DispatchDtype.bf16, CombineDtype.bf16


def resolve_mori_dtypes(
    quant_config: FusedMoEQuantConfig | None,
) -> tuple[DispatchDtype, CombineDtype]:
    """Resolve (dispatch_dtype, combine_dtype) from weights + env overrides.

    Auto-detected from the weight dtype, then overridden by
    ``VLLM_ROCM_MORI_DISPATCH_DTYPE`` / ``VLLM_ROCM_MORI_COMBINE_DTYPE`` when
    those are set to something other than ``"auto"``.
    """
    dispatch_dtype, combine_dtype = _auto_dtypes(quant_config)

    disp_override = envs.VLLM_ROCM_MORI_DISPATCH_DTYPE
    if disp_override != "auto":
        try:
            dispatch_dtype = DispatchDtype(disp_override)
        except ValueError:
            logger.warning(
                "Ignoring invalid VLLM_ROCM_MORI_DISPATCH_DTYPE=%r "
                "(expected auto|bf16|fp8|fp4)",
                disp_override,
            )

    comb_override = envs.VLLM_ROCM_MORI_COMBINE_DTYPE
    if comb_override != "auto":
        try:
            combine_dtype = CombineDtype(comb_override)
        except ValueError:
            logger.warning(
                "Ignoring invalid VLLM_ROCM_MORI_COMBINE_DTYPE=%r "
                "(expected auto|bf16|fp8|fp8_direct_cast)",
                comb_override,
            )

    return dispatch_dtype, combine_dtype
