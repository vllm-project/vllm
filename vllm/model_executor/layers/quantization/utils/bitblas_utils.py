# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple

import torch

from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

MINIMUM_BITBLAS_VERSION = "0.1.0"

BITBLAS_MIN_WEIGHT_SIZE_N = 16
BITBLAS_MIN_WEIGHT_SIZE_K = 16
GPTQ_BITBLAS_MAX_PARALLEL = 16

BITBLAS_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

# For dynamic shape code generation
BITBLAS_OPTIMIZE_FEATURES = [1, 16, 32, 64, 128, 256, 512, 1024]
# If want to enable high performance for contiguous batching
# Please use the following values
BITBLAS_OPTIMIZE_FEATURES_CONTIGUOUS = [16, 32, 64, 128, 256, 512, 1024]

BITBLAS_SUPPORTED_NUM_BITS = [1, 2, 4, 8]
BITBLAS_SUPPORTED_SYM = [False, True]


# Determines the supported quantization types for BitBLAS based on the
# device's capability and whether zero-point (zp) is used.
def query_bitblas_supported_quant_types(has_zp: bool,
                                        device_capability: Optional[int] = None
                                        ):
    if device_capability is None:
        capability_tuple = current_platform.get_device_capability()
        device_capability = (-1 if capability_tuple is None else
                             capability_tuple.to_int())

    if device_capability < 70:
        return []

    if has_zp:
        # AWQ style, unsigned + runtime zero-point
        return [scalar_types.uint4, scalar_types.uint8]
    else:
        # GPTQ style, unsigned + symmetric bias
        # TODO: once fp8_bitblas is merged into "gptq_bitblas" we should be able
        #  to add `scalar_types.float8_e4m3fn` here
        return [scalar_types.uint4b8, scalar_types.uint8b128]


def _check_bitblas_supported(
        quant_type: ScalarType,
        group_size: Optional[int],
        has_zp: bool,
        device_capability: Optional[int] = None) -> Tuple[bool, Optional[str]]:

    if device_capability is None:
        capability_tuple = current_platform.get_device_capability()
        device_capability = (-1 if capability_tuple is None else
                             capability_tuple.to_int())

    supported_types = query_bitblas_supported_quant_types(
        has_zp, device_capability)

    if quant_type not in supported_types:
        return (False, f"BitBLAS does not support weight_bits = {quant_type}. "
                f"Only types = {supported_types} "
                f"are supported (for group_size = {group_size}, "
                f"device_capability = {device_capability}, zp = {has_zp}).")
    if (group_size is None or group_size not in BITBLAS_SUPPORTED_GROUP_SIZES):
        return (False, f"BitBLAS does not support group_size = {group_size}. "
                f"Only group_sizes = {BITBLAS_SUPPORTED_GROUP_SIZES} "
                "are supported.")

    return True, None


def check_bitblas_supported(quant_type: ScalarType,
                            group_size: int,
                            has_zp: bool = False,
                            device_capability: Optional[int] = None) -> bool:
    cond, _ = _check_bitblas_supported(quant_type, group_size, has_zp,
                                       device_capability)
    return cond


def verify_bitblas_supported(quant_type: ScalarType,
                             group_size: int,
                             has_zp: bool = False) -> None:
    cond, err_msg = _check_bitblas_supported(quant_type, group_size, has_zp)
    if not cond:
        assert err_msg is not None
        raise ValueError(err_msg)


def verify_bitblas_supports_shape(output_size_per_partition: int,
                                  input_size_per_partition: int,
                                  input_size: int, group_size: int) -> None:

    # Validate output_size_per_partition
    if output_size_per_partition % BITBLAS_MIN_WEIGHT_SIZE_N != 0:
        raise ValueError(f"Weight output_size_per_partition = "
                         f"{output_size_per_partition} is not divisible by "
                         f" min_thread_n = {BITBLAS_MIN_WEIGHT_SIZE_N}. "
                         "Consider reducing tensor_parallel_size or running "
                         "with --quantization gptq.")

    # Validate input_size_per_partition
    if input_size_per_partition % BITBLAS_MIN_WEIGHT_SIZE_K != 0:
        raise ValueError(f"Weight input_size_per_partition = "
                         f"{input_size_per_partition} is not divisible "
                         f"by min_thread_k = {BITBLAS_MIN_WEIGHT_SIZE_K}. "
                         "Consider reducing tensor_parallel_size or running "
                         "with --quantization gptq.")

    if (group_size < input_size
            and input_size_per_partition % group_size != 0):
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition}"
            f" is not divisible by group_size = {group_size}."
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq.")


def check_bitblas_supports_shape(output_size_per_partition: int,
                                input_size_per_partition: int,
                                input_size: int, group_size: int) \
                                    -> Tuple[bool, Optional[str]]:
    try:
        verify_bitblas_supports_shape(output_size_per_partition,
                                      input_size_per_partition, input_size,
                                      group_size)
    except ValueError as e:
        return False, e.__str__()
    return True, None


def bitblas_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def bitblas_repeat_scales_on_all_ranks(act_order: bool, group_size: int,
                                       is_row_parallel: bool) -> bool:
    # Need to repeat scales on every rank if act_ordering or
    # channelwise and RowParallelLinear
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)


def bitblas_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                              requires_grad=False)


def bitblas_make_empty_zp(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                              requires_grad=False)


def bitblas_sort_g_idx(
        g_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def unpack_gptq_qzeros(qzeros, bits, is_gptq_v2=False) -> torch.Tensor:
    qzeros = qzeros.view(torch.int32)
    elems_per_int32 = 32 // bits
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * elems_per_int32),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    for col in range(unpacked_zeros.shape[1]):
        i = col % elems_per_int32
        unpacked_zeros[:, col] = (qzeros[:, col // elems_per_int32] >>
                                  (bits * i)) & 0xF
    if not is_gptq_v2:
        return unpacked_zeros + 1
    return unpacked_zeros


def unpack_gptq_qweight(qweight, bits):
    qweight = qweight.view(torch.int8)
    elems_per_int8 = 8 // bits
    unpacked_weight = torch.zeros(
        (qweight.shape[0], qweight.shape[1] * elems_per_int8),
        dtype=torch.int8,
        device=qweight.device,
        requires_grad=False,
    )
    for col in range(unpacked_weight.shape[1]):
        i = col % elems_per_int8
        unpacked_weight[:, col] = (qweight[:, col // elems_per_int8] >>
                                   (bits * i))

    return torch.bitwise_and(unpacked_weight, 2**bits - 1)
