# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

OCP_MX_BLOCK_SIZE = 32


class OCP_MX_Scheme(str, Enum):
    w_fp4_a_fp4 = "w_fp4_a_fp4"
    w_fp4_a_fp6_e3m2 = "w_fp4_a_fp6_e3m2"
    w_fp4_a_fp6_e2m3 = "w_fp4_a_fp6_e2m3"
    w_fp6_e3m2_a_fp6_e3m2 = "w_fp6_e3m2_a_fp6_e3m2"
    w_fp6_e2m3_a_fp6_e2m3 = "w_fp6_e2m3_a_fp6_e2m3"

    @classmethod
    def from_quant_dtype(cls, input_dtype: str, weight_dtype: str):
        if input_dtype == "fp4" and weight_dtype == "fp4":
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
            raise NotImplementedError(
                f"input_dtype='{input_dtype}' and"
                f" weight_dtype='{weight_dtype}' is not supported.")
