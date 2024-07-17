from typing import List, Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

GPTQ_MARLIN_SUPPORTED_NUM_BITS = [4, 8]
GPTQ_MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
GPTQ_MARLIN_SUPPORTED_SYM = [True]
GTPQ_MARLIN_UNSUPPORTED_GROUP_SIZE_ACT_ORDER = [-1]


def check_marlin_supported(num_bits: int, group_size: int, is_sym: bool,
                           min_capability: int) -> bool:

    # If the capability of the device is too low, cannot convert.
    major, minor = current_platform.get_device_capability()
    device_capability = major * 10 + minor
    if device_capability < min_capability:
        return False

    return (device_capability >= min_capability
            and num_bits in GPTQ_MARLIN_SUPPORTED_NUM_BITS
            and group_size in GPTQ_MARLIN_SUPPORTED_GROUP_SIZES
            and is_sym in GPTQ_MARLIN_SUPPORTED_SYM)


def verify_marlin_supported(num_bits: int, group_size: Optional[int],
                            is_sym: bool) -> None:

    if num_bits not in GPTQ_MARLIN_SUPPORTED_NUM_BITS:
        raise ValueError(
            f"Marlin does not support weight_bits = {num_bits}. "
            f"Only weight_bits = {GPTQ_MARLIN_SUPPORTED_NUM_BITS} "
            "are supported.")
    if (group_size is None
            or group_size not in GPTQ_MARLIN_SUPPORTED_GROUP_SIZES):
        raise ValueError(
            f"Marlin does not support group_size = {group_size}. "
            f"Only group_sizes = {GPTQ_MARLIN_SUPPORTED_GROUP_SIZES} "
            "are supported.")
    if is_sym not in GPTQ_MARLIN_SUPPORTED_SYM:
        raise ValueError(
            f"Marlin does not support is_sym = is_sym. "
            f"Only sym = {GPTQ_MARLIN_SUPPORTED_SYM} are supported.")


def verify_marlin_supports_shape(output_size_per_partition: int,
                                 input_size_per_partition: int,
                                 input_size: int, group_size: int) -> None:

    # Validate output_size_per_partition
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(f"Weight output_size_per_partition = "
                         f"{output_size_per_partition} is not divisible by "
                         f" min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. "
                         "Consider reducing tensor_parallel_size or running "
                         "with --quantization gptq.")

    # Validate input_size_per_partition
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(f"Weight input_size_per_partition = "
                         f"{input_size_per_partition} is not divisible "
                         f"by min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}. "
                         "Consider reducing tensor_parallel_size or running "
                         "with --quantization gptq.")

    if (group_size < input_size
            and input_size_per_partition % group_size != 0):
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition}"
            f" is not divisible by group_size = {group_size}."
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq.")


def marlin_make_workspace(output_size_per_partition: int,
                          device: torch.device) -> torch.Tensor:
    max_workspace_size = (output_size_per_partition //
                          GPTQ_MARLIN_MIN_THREAD_N) * GPTQ_MARLIN_MAX_PARALLEL

    return torch.zeros(max_workspace_size,
                       dtype=torch.int,
                       device=device,
                       requires_grad=False)


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_repeat_scales_on_all_ranks(act_order: bool, group_size: int,
                                      is_row_parallel: bool) -> bool:
    # Need to repeat scales on every rank if act_ordering or
    # channelwise and RowParallelLinear
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                              requires_grad=False)


def marlin_sort_g_idx(
        g_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def get_scale_perms():
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                          group_size: int) -> torch.Tensor:

    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by vLLM (and won't be freed)
def replace_tensor(layer: torch.nn.Module, name: str,
                   new_t: torch.Tensor) -> None:
    # It is important to use resize_() here since it ensures
    # the same buffer is reused
    getattr(layer, name).resize_(new_t.shape)
    getattr(layer, name).copy_(new_t)
    del new_t


def apply_marlin_linear(input: torch.Tensor,
                        weight: torch.Tensor,
                        weight_scale: torch.Tensor,
                        g_idx: torch.Tensor,
                        g_idx_sort_indices: torch.Tensor,
                        workspace: torch.Tensor,
                        num_bits: int,
                        output_size_per_partition: int,
                        input_size_per_partition: int,
                        is_k_full: bool,
                        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition, )

    output = ops.gptq_marlin_gemm(reshaped_x,
                                  weight,
                                  weight_scale,
                                  g_idx,
                                  g_idx_sort_indices,
                                  workspace,
                                  num_bits,
                                  size_m=reshaped_x.shape[0],
                                  size_n=output_size_per_partition,
                                  size_k=input_size_per_partition,
                                  is_k_full=is_k_full)

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)
