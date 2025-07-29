# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

import vllm._custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT, marlin_make_workspace_new, marlin_permute_scales,
    should_use_atomic_add_reduce)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

FP4_MARLIN_SUPPORTED_GROUP_SIZES = [16]

logger = init_logger(__name__)


def is_fp4_marlin_supported():
    return current_platform.has_device_capability(80)


def fp4_marlin_process_scales(marlin_scales):
    if not (marlin_scales >= 0).all():
        logger.warning_once(
            "NVFP4 Marlin assumes the scales to be >=0, but has encountered "
            "negative scales. Accuracy will likely be degraded. This is "
            "because it changes the scales from FP8-S1E4M3 to a special "
            "FP8-S0E5M3 format to speedup the dequantization.")

    # convert to half first, we would convert to fp8 later
    marlin_scales = marlin_scales.to(torch.half)

    # 8 is the number of scale number using by one thread
    marlin_scales = marlin_scales.view(marlin_scales.size(0) // 2, 2, -1, 8)
    marlin_scales = marlin_scales.permute(0, 2, 1, 3).reshape(
        marlin_scales.size(0) * 2, -1)

    # fit the layout of fp8 dequantization
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1)

    # We assume that weight_scale (FP8-S1E4M3) is always greater
    # than or equal to 0. So we can convert
    # (weight_scale * (2 ** 7) to a special FP8-S0E5M3 format.
    # After multiplying by 2 ** 7, the top bit of FP8-S0E5M3 would always be 1
    # when weight_scale > 0. This allows us to have an exponent bias
    # closer to zero after dequantization.

    marlin_scales = (marlin_scales * (2**7)).view(torch.int16) << 1
    marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
    marlin_scales = marlin_scales[:, 1::2].contiguous()

    return marlin_scales


def fp4_marlin_process_global_scale(global_scale):
    assert global_scale.dtype in [torch.half, torch.bfloat16]
    fp4_exponent = 2
    if global_scale.dtype == torch.half:
        target_exponent = 5
    elif global_scale.dtype == torch.bfloat16:
        target_exponent = 8
    # exponent_bias_fp16 = 2 ** 4 - 2 ** 1 = 14
    # exponent_bias_bf16 = 2 ** 7 - 2 ** 1 = 126
    exponent_bias = 2**(target_exponent - 1) - 2**(fp4_exponent - 1)
    return global_scale * (2.0**(exponent_bias - 7))


def apply_fp4_marlin_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        workspace: torch.Tensor,
        size_n: int,
        size_k: int,
        bias: Optional[torch.Tensor] = None,
        use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT) -> torch.Tensor:
    # For GPUs that lack FP4 hardware support, we can leverage the
    # Marlin kernel for fast weight-only FP4 quantization

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n, )

    use_atomic_add = should_use_atomic_add_reduce(m=reshaped_x.size(0),
                                                  n=size_n,
                                                  k=size_k,
                                                  device=input.device,
                                                  dtype=input.dtype)

    output = ops.gptq_marlin_gemm(a=reshaped_x,
                                  c=None,
                                  b_q_weight=weight,
                                  b_scales=weight_scale,
                                  global_scale=weight_scale_2,
                                  b_zeros=None,
                                  g_idx=None,
                                  perm=None,
                                  workspace=workspace,
                                  b_q_type=scalar_types.float4_e2m1f,
                                  size_m=reshaped_x.size(0),
                                  size_n=size_n,
                                  size_k=size_k,
                                  use_atomic_add=use_atomic_add,
                                  use_fp32_reduce=use_fp32_reduce)

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)


def prepare_fp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    logger.warning_once(
        "Your GPU does not have native support for FP4 computation but "
        "FP4 quantization is being used. Weight-only FP4 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads.")

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    param_dtype = layer.params_dtype

    assert layer.weight.shape == (part_size_n, part_size_k // 2)

    device = layer.weight.device

    # WORKSPACE
    layer.workspace = marlin_make_workspace_new(device)

    # WEIGHT
    # Repack weights to marlin format
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = layer.weight.view(torch.int32).T.contiguous()

    marlin_qweight = ops.gptq_marlin_repack(b_q_weight=qweight,
                                            perm=perm,
                                            size_k=part_size_k,
                                            size_n=part_size_n,
                                            num_bits=4)
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    # WEIGHT SCALES
    # Permute scales
    weight_scale = layer.weight_scale.T.to(param_dtype)
    weight_scale = marlin_permute_scales(s=weight_scale,
                                         size_k=part_size_k,
                                         size_n=part_size_n,
                                         group_size=16)
    weight_scale = fp4_marlin_process_scales(weight_scale)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

    weight_scale_2 = layer.weight_scale_2.to(param_dtype)
    weight_scale_2 = fp4_marlin_process_global_scale(weight_scale_2)
    layer.weight_scale_2 = torch.nn.Parameter(weight_scale_2,
                                              requires_grad=False)

    return


def prepare_moe_fp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    logger.warning_once(
        "Your GPU does not have native support for FP4 computation but "
        "FP4 quantization is being used. Weight-only FP4 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads.")

    e = layer.num_experts
    k = layer.hidden_size
    n = layer.intermediate_size_per_partition

    # WORKSPACE
    device = layer.w13_weight.device
    param_dtype = layer.params_dtype
    layer.workspace = marlin_make_workspace_new(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    # WEIGHT
    # Repack weights to marlin format
    for name in ["w13_weight", "w2_weight"]:
        weight = getattr(layer, name)
        tensor_list = []
        if "w13" in name:
            size_n, size_k = n * 2, k
        else:
            size_n, size_k = k, n

        assert weight.shape == (e, size_n, size_k // 2)

        for i in range(e):
            qweight = weight[i].view(torch.int32).T.contiguous()

            marlin_qweight = ops.gptq_marlin_repack(b_q_weight=qweight,
                                                    perm=perm,
                                                    size_k=size_k,
                                                    size_n=size_n,
                                                    num_bits=4)
            tensor_list.append(marlin_qweight)

        weight = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        weight = torch.nn.Parameter(weight, requires_grad=False)

        setattr(layer, name, weight)

    # WEIGHT SCALES
    # Permute scales
    for name in ["w13", "w2"]:
        scales = getattr(layer, name + "_weight_scale").to(param_dtype)
        global_scale = getattr(layer, name + "_weight_scale_2").to(param_dtype)

        tensor_list = []
        if "w13" in name:
            size_n, size_k = n * 2, k
        else:
            size_n, size_k = k, n

        for i in range(e):
            marlin_scales = marlin_permute_scales(s=scales[i].T,
                                                  size_k=size_k,
                                                  size_n=size_n,
                                                  group_size=16)
            marlin_scales = fp4_marlin_process_scales(marlin_scales)
            tensor_list.append(marlin_scales)

        scales = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        scales = torch.nn.Parameter(scales, requires_grad=False)
        setattr(layer, name + "_weight_scale", scales)

        global_scale = fp4_marlin_process_global_scale(global_scale)
        global_scale = torch.nn.Parameter(global_scale, requires_grad=False)
        setattr(layer, name + "_weight_scale_2", global_scale)


def rand_marlin_weight_fp4_like(weight, group_size):
    assert group_size > 0
    size_n, size_k = weight.shape
    device = weight.device

    scales = weight.view(size_n, -1, group_size).abs().max(-1)[0] / 6
    global_scale = scales.max() / 448
    scales = (scales / global_scale).to(torch.float8_e4m3fn)

    fp4_weight = torch.randint(0,
                               256, (size_n, size_k // 2),
                               dtype=torch.uint8,
                               device=weight.device)
    fp4_weight_part_1 = ((fp4_weight & 0b10000000) |
                         ((fp4_weight & 0b01110000) >> 2))
    fp4_weight_part_1 = fp4_weight_part_1.view(torch.float8_e4m3fn)
    fp4_weight_part_1 = fp4_weight_part_1.to(weight.dtype) * (2**6)

    fp4_weight2 = fp4_weight << 4
    fp4_weight_part_2 = ((fp4_weight2 & 0b10000000) |
                         ((fp4_weight2 & 0b01110000) >> 2))
    fp4_weight_part_2 = fp4_weight_part_2.view(torch.float8_e4m3fn)
    fp4_weight_part_2 = fp4_weight_part_2.to(weight.dtype) * (2**6)

    weight_ref = torch.cat(
        [fp4_weight_part_2.unsqueeze(2),
         fp4_weight_part_1.unsqueeze(2)], 2).view(size_n, size_k)
    weight_ref = weight_ref * global_scale.to(weight.dtype) * \
        scales.repeat_interleave(group_size, 1).to(weight.dtype)

    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=fp4_weight.view(torch.int32).T.contiguous(),
        perm=torch.empty(0, dtype=torch.int, device=device),
        size_k=size_k,
        size_n=size_n,
        num_bits=4,
    )

    marlin_scales = marlin_permute_scales(s=scales.T.to(weight.dtype),
                                          size_k=size_k,
                                          size_n=size_n,
                                          group_size=group_size)
    marlin_scales = fp4_marlin_process_scales(marlin_scales)

    global_scale = fp4_marlin_process_global_scale(global_scale)

    return weight_ref.T, marlin_qweight, marlin_scales, global_scale
