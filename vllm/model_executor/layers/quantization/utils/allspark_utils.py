# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD = 1024
ALLSPARK_SUPPORTED_QUANT_TYPES = [scalar_types.uint8b128]
ALLSPARK_AMPERE_N_ALIGN = 16
ALLSPARK_AMPERE_K_ALIGN = 16


def check_allspark_supported_dtype_shape(input_size_per_partition: int,
                                         output_size_per_partition: int,
                                         group_size: int,
                                         weight_dtype: ScalarType,
                                         act_dtype: torch.dtype):
    capability_tuple = current_platform.get_device_capability()
    device_capability = (-1 if capability_tuple is None else
                         capability_tuple.to_int())

    # For Ampere GPU
    if device_capability >= 80 and device_capability < 90:
        if group_size != -1:
            return False, \
                "For Ampere GPU, AllSpark does not support group_size "\
                f"= {group_size}. Only group_size = -1 are supported."

        if weight_dtype not in ALLSPARK_SUPPORTED_QUANT_TYPES:
            return False, "For Ampere GPU, AllSpark does not support "\
                f"quant type ({weight_dtype}). Only quant type "\
                f"({ALLSPARK_SUPPORTED_QUANT_TYPES}) are supported."

        if input_size_per_partition % ALLSPARK_AMPERE_K_ALIGN != 0 \
            or output_size_per_partition % ALLSPARK_AMPERE_N_ALIGN != 0:
            return False, \
                "AllSpark needs input_size_per_partition % "\
                f"{ALLSPARK_AMPERE_K_ALIGN} = 0 and "\
                f"output_size_per_partition % {ALLSPARK_AMPERE_N_ALIGN} = 0 "\
                "for Ampere GPU optimized kernels."

        if act_dtype != torch.float16 and act_dtype != torch.bfloat16:
            return False, \
                "AllSpark only supports act_dtype = float16 or bfloat16,"\
                f"for Ampere GPU, but got act_dtype = {act_dtype}."
    else:
        return False, "AllSpark currently does not support "\
            f"device_capability = {device_capability}."

    return True, None
