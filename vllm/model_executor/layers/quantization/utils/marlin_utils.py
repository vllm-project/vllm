# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import numpy
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

from .quant_utils import pack_cols, unpack_cols

logger = init_logger(__name__)

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

# In case there is a performance issue with Marlin, the variable below can be
# changed to False, which allows Marlin to perform global reductions in fp16
# precision (instead of fp32), and therefore, save on some memory movements.
USE_FP32_REDUCE_DEFAULT = True


# For binary size and compile time, we don't support the same types for with and
#  without runtime zero-point. We support common cases, i.e. AWQ and GPTQ.
#  TODO: we may want to move this into the C++ so its closer to the actual impl
def query_marlin_supported_quant_types(
    has_zp: bool | None = None,
    include_fp_type: bool = True,
    device_capability: int | None = None,
):
    if device_capability is None:
        capability_tuple = current_platform.get_device_capability()
        device_capability = (
            -1 if capability_tuple is None else capability_tuple.to_int()
        )

    if device_capability < 80:
        return []

    # - has_zp is True: return quant_types that has zero points
    # - has_zp is False: return quant_types that has not zero points
    # - has_zp is None: both
    if has_zp is None:
        types0 = query_marlin_supported_quant_types(
            False, include_fp_type, device_capability
        )
        types1 = query_marlin_supported_quant_types(
            True, include_fp_type, device_capability
        )
        return types0 + types1

    if has_zp:
        # AWQ style, unsigned + runtime zero-point
        return [scalar_types.uint4]
    else:
        # GPTQ style, unsigned + symmetric bias
        res = [scalar_types.uint4b8, scalar_types.uint8b128]
        if include_fp_type:
            res += [scalar_types.float8_e4m3fn, scalar_types.float4_e2m1f]
        return res


def _check_marlin_supported(
    quant_type: ScalarType,
    group_size: int | None,
    has_zp: bool,
    device_capability: int | None = None,
) -> tuple[bool, str | None]:
    if device_capability is None:
        capability_tuple = current_platform.get_device_capability()
        device_capability = (
            -1 if capability_tuple is None else capability_tuple.to_int()
        )

    supported_types = query_marlin_supported_quant_types(
        has_zp, True, device_capability
    )

    if quant_type not in supported_types:
        return (
            False,
            f"Marlin does not support weight_bits = {quant_type}. "
            f"Only types = {supported_types} "
            f"are supported (for group_size = {group_size}, "
            f"device_capability = {device_capability}, zp = {has_zp}).",
        )
    if group_size is None or group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
        return (
            False,
            f"Marlin does not support group_size = {group_size}. "
            f"Only group_sizes = {MARLIN_SUPPORTED_GROUP_SIZES} "
            "are supported.",
        )

    return True, None


def check_marlin_supported(
    quant_type: ScalarType,
    group_size: int,
    has_zp: bool = False,
    device_capability: int | None = None,
) -> bool:
    cond, _ = _check_marlin_supported(quant_type, group_size, has_zp, device_capability)
    return cond


def verify_marlin_supported(
    quant_type: ScalarType, group_size: int, has_zp: bool = False
) -> None:
    cond, err_msg = _check_marlin_supported(quant_type, group_size, has_zp)
    if not cond:
        assert err_msg is not None
        raise ValueError(err_msg)


def verify_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> None:
    # Validate output_size_per_partition
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(
            f"Weight output_size_per_partition = "
            f"{output_size_per_partition} is not divisible by "
            f" min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )

    # Validate input_size_per_partition
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(
            f"Weight input_size_per_partition = "
            f"{input_size_per_partition} is not divisible "
            f"by min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )

    if group_size < input_size and input_size_per_partition % group_size != 0:
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition}"
            f" is not divisible by group_size = {group_size}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )


def check_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> tuple[bool, str | None]:
    try:
        verify_marlin_supports_shape(
            output_size_per_partition, input_size_per_partition, input_size, group_size
        )
    except ValueError as e:
        return False, e.__str__()
    return True, None


def check_marlin_supports_layer(layer: LinearBase, group_size: int) -> bool:
    output_size_per_partition = (
        getattr(layer, "output_size_per_partition", None) or layer.output_size
    )
    input_size_per_partition = (
        getattr(layer, "input_size_per_partition", None) or layer.input_size
    )

    return check_marlin_supports_shape(
        output_size_per_partition=output_size_per_partition,
        input_size_per_partition=input_size_per_partition,
        input_size=layer.input_size,
        group_size=group_size,
    )[0]


def check_moe_marlin_supports_layer(layer: LinearBase, group_size: int) -> bool:
    hidden_size = layer.hidden_size
    intermediate_size_per_partition = layer.intermediate_size_per_partition
    # apply_router_weight_on_input is not supported for moe marlin
    supports_router_weight = not layer.apply_router_weight_on_input
    # moe marlin requires the activation to be silu
    supports_activation = layer.activation == "silu"

    # gate-up: (n, k) = (intermediate_size_per_partition * 2, hidden_size)
    # down: (n, k) = (hidden_size, intermediate_size_per_partition)
    # moe marlin requires n % 128 == 0 and k % 64 == 0
    supports_shape = (
        hidden_size % 128 == 0
        and intermediate_size_per_partition % max(64, group_size) == 0
    )
    supports_group_size = group_size in [-1, 32, 64, 128]
    return (
        supports_shape
        and supports_group_size
        and supports_router_weight
        and supports_activation
    )


def marlin_moe_intermediate_size(w1_packed: torch.Tensor, w2_packed: torch.Tensor):
    """
    Given Marlin packed weight matrices w1_packed, and w2_packed,
    return the MoE intermediate size N
    """
    marlin_tile_size = 16
    return w2_packed.size(1) * marlin_tile_size


def marlin_make_workspace(
    output_size_per_partition: int, device: torch.device
) -> torch.Tensor:
    max_workspace_size = (
        output_size_per_partition // GPTQ_MARLIN_MIN_THREAD_N
    ) * GPTQ_MARLIN_MAX_PARALLEL

    return torch.zeros(
        max_workspace_size, dtype=torch.int, device=device, requires_grad=False
    )


def marlin_make_workspace_new(
    device: torch.device, max_blocks_per_sm: int = 1
) -> torch.Tensor:
    # In the new marlin kernel, we use the num of threadblocks as workspace
    # size. The num of threadblocks is sms_count * max_blocks_per_sm.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(
        sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False
    )


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_repeat_scales_on_all_ranks(
    act_order: bool, group_size: int, is_row_parallel: bool
) -> bool:
    # Need to repeat scales on every rank if act_ordering or
    # channelwise and RowParallelLinear
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


def marlin_make_empty_zp(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


def marlin_sort_g_idx(g_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def get_scale_perms():
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def marlin_permute_bias(s: torch.Tensor) -> torch.Tensor:
    origin_shape = s.shape
    _, scale_perm_single = get_scale_perms()
    s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape(*origin_shape).contiguous()


def marlin_moe_permute_scales(
    s: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
):
    num_experts = s.shape[0]
    output = torch.empty(
        (num_experts, s.shape[1], s.shape[2]),
        device=s.device,
        dtype=s.dtype,
    )

    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size)
    return output


def marlin_zero_points(
    zp: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    # Permute zero-points in a similar way to scales, but do not use the
    # "single" permutation, since zero-points are applied on every MMA
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp


def awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    # AWQ zero-points are quantized and packed on the column dim.
    # In addition, the values are permuted based on dequantizer.
    # Here we undo both of these, and then apply marlin permutation
    # and pack it back.
    q_zp = unpack_cols(q_zp_packed, num_bits, size_k, size_n)

    # Undo interleaving (use argsort(..) to get inverse perm)
    if num_bits == 4:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))
    elif num_bits == 8:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    q_zp = q_zp.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    q_zp = q_zp.reshape((-1, size_n)).contiguous()

    marlin_zp = marlin_zero_points(q_zp, size_k, size_n, num_bits)
    return marlin_zp


def moe_awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor, size_k: int, size_n: int, num_bits: int
):
    num_experts = q_zp_packed.shape[0]
    output = torch.empty(
        (num_experts, q_zp_packed.shape[1], q_zp_packed.shape[2]),
        device=q_zp_packed.device,
        dtype=q_zp_packed.dtype,
    )
    for e in range(num_experts):
        output[e] = awq_to_marlin_zero_points(q_zp_packed[e], size_k, size_n, num_bits)
    return output


def maybe_warn_marlin_atomic_add(device, dtype):
    if torch.compiler.is_dynamo_compiling():
        return
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        logger.info_once(
            "You are running Marlin kernel with bf16 on GPUs before SM90. "
            "You can consider change to fp16 to achieve better performance "
            "if possible."
        )


def maybe_warn_marlin_atomic_add_env():
    if torch.compiler.is_dynamo_compiling():
        return
    if envs.VLLM_MARLIN_USE_ATOMIC_ADD:
        return
    logger.info_once(
        "Marlin kernel can achieve better performance for small size_n "
        "with experimental use_atomic_add feature. "
        "You can consider set environment variable "
        "VLLM_MARLIN_USE_ATOMIC_ADD to 1 if possible."
    )


def should_use_atomic_add_reduce(
    m: int, n: int, k: int, device: torch.device, dtype: torch.dtype
) -> bool:
    # the performance of atomicAdd is better than global reduce
    # only when m*n is small and k is large
    if n >= 2048 or k < 2048 or device.type != "cuda":
        return False

    # disable atomicAdd reduce by default,
    # one can enable it with VLLM_MARLIN_USE_ATOMIC_ADD=1
    if not envs.VLLM_MARLIN_USE_ATOMIC_ADD:
        maybe_warn_marlin_atomic_add_env()
        return False

    # sm8x doesn't support atomicAdd + bfloat16 natively
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        maybe_warn_marlin_atomic_add(device, dtype)
        return False

    return True


def apply_gptq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    wtype: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    is_k_full: bool,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )

    output = ops.gptq_marlin_gemm(
        reshaped_x,
        None,
        weight,
        bias,
        weight_scale,
        None,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        wtype,
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )

    return output.reshape(out_shape)


def apply_awq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    quant_type: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )

    output = ops.gptq_marlin_gemm(
        reshaped_x,
        None,
        weight,
        bias,
        weight_scale,
        None,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        quant_type,
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )

    return output.reshape(out_shape)


def apply_rtn_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    quant_type: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )

    output = ops.gptq_marlin_gemm(
        reshaped_x,
        None,
        weight,
        bias,
        weight_scale,
        None,
        None,
        None,
        None,
        workspace,
        quant_type,
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )

    return output.reshape(out_shape)
