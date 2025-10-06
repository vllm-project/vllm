# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import Union

from vllm.logger import init_logger

logger = init_logger(__name__)

OCP_MX_BLOCK_SIZE = 32

OCP_MX_DTYPES = {"fp4", "fp6_e3m2", "fp6_e2m3", "fp8_e4m3", "fp8_e5m2", "int8"}
SUPPORTED_OCP_MX_DTYPES = {"fp4", "fp6_e3m2", "fp6_e2m3"}


class OCP_MX_Scheme(str, Enum):
    w_fp4_a_fp4 = "w_fp4_a_fp4"
    w_fp4_a_fp6_e3m2 = "w_fp4_a_fp6_e3m2"
    w_fp4_a_fp6_e2m3 = "w_fp4_a_fp6_e2m3"
    w_fp6_e3m2_a_fp6_e3m2 = "w_fp6_e3m2_a_fp6_e3m2"
    w_fp6_e2m3_a_fp6_e2m3 = "w_fp6_e2m3_a_fp6_e2m3"

    @classmethod
    def from_quant_dtype(
        cls, input_dtype: Union[str, None], weight_dtype: Union[str, None]
    ):
        if input_dtype not in OCP_MX_DTYPES or weight_dtype not in OCP_MX_DTYPES:
            return None
        elif input_dtype == "fp4" and weight_dtype == "fp4":
            return cls.w_fp4_a_fp4
        elif input_dtype == "fp6_e3m2" and weight_dtype == "fp4":
            return cls.w_fp4_a_fp6_e3m2
        elif input_dtype == "fp6_e2m3" and weight_dtype == "fp4":
            return cls.w_fp4_a_fp6_e2m3
        elif input_dtype == "fp6_e3m2" and weight_dtype == "fp6_e3m2":
            return cls.w_fp6_e3m2_a_fp6_e3m2
        elif input_dtype == "fp6_e2m3" and weight_dtype == "fp6_e2m3":
            return cls.w_fp6_e2m3_a_fp6_e2m3
        else:
            logger.warning(
                "input_dtype='%s' and"
                " weight_dtype='%s' is not supported "
                "in OCP_MX_Scheme at the moment.",
                input_dtype,
                weight_dtype,
            )
            return None
