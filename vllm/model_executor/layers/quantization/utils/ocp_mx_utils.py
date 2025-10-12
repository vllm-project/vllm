# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

from vllm.logger import init_logger

logger = init_logger(__name__)

OCP_MX_BLOCK_SIZE = 32

OCP_MX_DTYPES = {
    "mxfp4",
    "mxfp6_e3m2",
    "mxfp6_e2m3",
    "mxfp8_e4m3",
    "mxfp8_e5m2",
    "mxint8",
}
SUPPORTED_OCP_MX_DTYPES = {"mxfp4", "mxfp6_e3m2", "mxfp6_e2m3"}


class OCP_MX_Scheme(str, Enum):
    w_mxfp4_a_mxfp4 = "w_mxfp4_a_mxfp4"
    w_mxfp4_a_mxfp6_e3m2 = "w_mxfp4_a_mxfp6_e3m2"
    w_mxfp4_a_mxfp6_e2m3 = "w_mxfp4_a_mxfp6_e2m3"
    w_mxfp6_e3m2_a_mxfp6_e3m2 = "w_mxfp6_e3m2_a_mxfp6_e3m2"
    w_mxfp6_e2m3_a_mxfp6_e2m3 = "w_mxfp6_e2m3_a_mxfp6_e2m3"

    @classmethod
    def from_quant_dtype(cls, input_dtype: str | None, weight_dtype: str | None):
        if input_dtype not in OCP_MX_DTYPES or weight_dtype not in OCP_MX_DTYPES:
            return None
        elif input_dtype == "mxfp4" and weight_dtype == "mxfp4":
            return cls.w_mxfp4_a_mxfp4
        elif input_dtype == "mxfp6_e3m2" and weight_dtype == "mxfp4":
            return cls.w_mxfp4_a_mxfp6_e3m2
        elif input_dtype == "mxfp6_e2m3" and weight_dtype == "mxfp4":
            return cls.w_mxfp4_a_mxfp6_e2m3
        elif input_dtype == "mxfp6_e3m2" and weight_dtype == "mxfp6_e3m2":
            return cls.w_mxfp6_e3m2_a_mxfp6_e3m2
        elif input_dtype == "mxfp6_e2m3" and weight_dtype == "mxfp6_e2m3":
            return cls.w_mxfp6_e2m3_a_mxfp6_e2m3
        else:
            logger.warning(
                "input_dtype='%s' and"
                " weight_dtype='%s' is not supported "
                "in OCP_MX_Scheme at the moment.",
                input_dtype,
                weight_dtype,
            )
            return None
