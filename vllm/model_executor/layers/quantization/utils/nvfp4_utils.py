# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NVFP4 Scheme utilities and enum definitions for fused MoE emulation fallback.
"""
from enum import Enum

from vllm.logger import init_logger

logger = init_logger(__name__)

NVFP4_BLOCK_SIZE = 16

NVFP4_DTYPES = {
    "nvfp4",
}

SUPPORTED_NVFP4_DTYPES = {"nvfp4"}


class NVFP4_Scheme(str, Enum):
    w_nvfp4_a_nvfp4 = "w_nvfp4_a_nvfp4"

    @classmethod
    def from_quant_dtype(cls, input_dtype: str | None, weight_dtype: str | None):
        if input_dtype not in NVFP4_DTYPES or weight_dtype not in NVFP4_DTYPES:
            return None
        elif input_dtype == "nvfp4" and weight_dtype == "nvfp4":
            return cls.w_nvfp4_a_nvfp4
        else:
            logger.warning(
                "input_dtype='%s' and"
                " weight_dtype='%s' is not supported "
                "in NVFP4_Scheme at the moment.",
                input_dtype,
                weight_dtype,
            )
            return None
