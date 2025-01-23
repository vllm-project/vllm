# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple

import torch

from vllm.platforms import current_platform

ALLSPARK_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128, 256]
ALLSPARK_AMPERE_N_ALIGN = 16
ALLSPARK_AMPERE_K_ALIGN = 16
ALLSPARK_VOLTA_N_ALIGN = 8
ALLSPARK_VOLTA_K_ALIGN = 8


def check_allspark_supported(num_bits: int, group_size: int,
                             desc_act: bool) -> Tuple[bool, Optional[str]]:
    if desc_act:
        return (False, "AllSpark does not support desc_act = True.")

    capability_tuple = current_platform.get_device_capability()
    device_capability = (-1 if capability_tuple is None else
                         capability_tuple.to_int())

    # For Ampere GPU
    if device_capability >= 80 and device_capability < 90:
        if num_bits != 8 or group_size != -1:
            return (False, f"For Ampere GPU, AllSpark does not support "
                    f"weight_bits = {num_bits} and group_size = {group_size}."
                    f"Only weight_bits = 8, group_size = -1 are supported.")
    # For Volta GPU， later add
    # elif device_capability == 70:
    #     if (num_bits != 8 and num_bits != 4) or \
    #         group_size not in ALLSPARK_SUPPORTED_GROUP_SIZES:
    #         return (False, f"For Volta GPU, AllSpark does not support"
    #                 f"weight_bits = {num_bits} and group_size = {group_size}."
    #                 f"Only weight_bits = 4 or 8, group_size = "
    #                 f"{ALLSPARK_SUPPORTED_GROUP_SIZES} are supported.")
    else:
        return (False, f"AllSpark currently does not support "
                f"device_capability = {device_capability}.")

    return True, None


def check_allspark_supported_dtype_shape(input_size_per_partition: int,
                                         output_size_per_partition: int,
                                         group_size: int,
                                         act_dtype: torch.dtype):
    capability_tuple = current_platform.get_device_capability()
    device_capability = (-1 if capability_tuple is None else
                         capability_tuple.to_int())
    # For Ampere GPU
    if device_capability >= 80 and device_capability < 90:
        if input_size_per_partition % ALLSPARK_AMPERE_K_ALIGN != 0 \
            or output_size_per_partition % ALLSPARK_AMPERE_N_ALIGN != 0:
            raise ValueError(
                "AllSpark needs input_size_per_partition % "
                f"{ALLSPARK_AMPERE_K_ALIGN} = 0 and "
                f"output_size_per_partition % {ALLSPARK_AMPERE_N_ALIGN} = 0 "
                "for Ampere GPU optimized kernels.")
        if act_dtype != torch.float16 and act_dtype != torch.bfloat16:
            raise ValueError(
                "AllSpark only supports act_dtype = float16 or bfloat16,"
                f"for Ampere GPU, but got act_dtype = {act_dtype}.")

    # For Volta GPU
    if device_capability == 70:
        if input_size_per_partition % ALLSPARK_VOLTA_K_ALIGN != 0 \
            or output_size_per_partition % ALLSPARK_VOLTA_N_ALIGN != 0:
            raise ValueError(
                "AllSpark needs input_size_per_partition "
                f"% {ALLSPARK_VOLTA_K_ALIGN} = 0 and "
                f"output_size_per_partition % {ALLSPARK_VOLTA_N_ALIGN} = 0 "
                "for Volta GPU optimized kernels.")
        if act_dtype != torch.float16:
            raise ValueError(
                "AllSpark only supports act_dtype = float16 for Volta GPU"
                f"but got act_dtype = {act_dtype}.")
