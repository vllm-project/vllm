# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm._custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    get_marlin_input_dtype,
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_quant_input,
    should_use_atomic_add_reduce,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

FP4_MARLIN_SUPPORTED_GROUP_SIZES = [16]

logger = init_logger(__name__)


def is_fp4_marlin_supported():
    return current_platform.has_device_capability(75)


def nvfp4_marlin_process_scales(marlin_scales):
    if not (marlin_scales >= 0).all():
        logger.warning_once(
            "NVFP4 Marlin assumes the scales to be >=0, but has encountered "
            "negative scales. Accuracy will likely be degraded. This is "
            "because it changes the scales from FP8-S1E4M3 to a special "
            "FP8-S0E5M3 format to speedup the dequantization."
        )

    # convert to half first, we would convert to fp8 later
    marlin_scales = marlin_scales.to(torch.half)

    # fit the layout of fp8 dequantization
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1
    )

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


def mxfp4_marlin_process_scales(marlin_scales, input_dtype=None):
    # fit the layout of fp8 dequantization
    if input_dtype is None or input_dtype.itemsize == 2:
        marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
            marlin_scales.size(0), -1
        )

    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
    if input_dtype == torch.float8_e4m3fn:
        marlin_scales = marlin_scales.view(torch.uint8)
        assert marlin_scales.max() <= 249
        # exponent_bias (fp4->fp8) = 2 ** 3 - 2 ** 1 = 6
        marlin_scales = marlin_scales + 6
        marlin_scales = marlin_scales.view(torch.float8_e8m0fnu)
    return marlin_scales


def nvfp4_marlin_process_global_scale(global_scale):
    assert global_scale.dtype in [torch.half, torch.bfloat16]
    fp4_exponent = 2
    if global_scale.dtype == torch.half:
        target_exponent = 5
    elif global_scale.dtype == torch.bfloat16:
        target_exponent = 8
    # exponent_bias_fp16 = 2 ** 4 - 2 ** 1 = 14
    # exponent_bias_bf16 = 2 ** 7 - 2 ** 1 = 126
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return global_scale * (2.0 ** (exponent_bias - 7))


def apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor | None,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    # For GPUs that lack FP4 hardware support, we can leverage the
    # Marlin kernel for fast weight-only FP4 quantization

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0), n=size_n, k=size_k, device=input.device, dtype=input.dtype
    )

    inputs = reshaped_x
    a_scales = None
    is_nvfp4 = weight_global_scale is not None
    if input_dtype is not None and input_dtype.itemsize == 1:
        if is_nvfp4:
            raise RuntimeError("NVFP4 weight + INT8/FP8 activation is not supported.")
        elif input_dtype != torch.float8_e4m3fn:
            raise RuntimeError("MXFP4 weight + INT8 activation is not supported.")

        inputs, a_scales = marlin_quant_input(inputs, torch.float8_e4m3fn)

    output = ops.marlin_gemm(
        a=inputs,
        c=None,
        b_q_weight=weight,
        b_bias=bias,
        b_scales=weight_scale,
        a_scales=a_scales,
        global_scale=weight_global_scale,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float4_e2m1f,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    return output.reshape(out_shape)


def prepare_fp4_layer_for_marlin(
    layer: torch.nn.Module, input_dtype: torch.dtype | None = None
) -> None:
    logger.warning_once(
        "Your GPU does not have native support for FP4 computation but "
        "FP4 quantization is being used. Weight-only FP4 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )

    is_nvfp4 = hasattr(layer, "weight_global_scale")
    if input_dtype is not None and input_dtype.itemsize == 1:
        if is_nvfp4:
            raise RuntimeError("NVFP4 weight + INT8/FP8 activation is not supported.")
        elif input_dtype != torch.float8_e4m3fn:
            raise RuntimeError("MXFP4 weight + INT8 activation is not supported.")

    group_size = 16 if is_nvfp4 else 32

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

    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1
    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=4,
        is_a_8bit=is_a_8bit,
    )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    # WEIGHT SCALES
    # Permute scales
    weight_scale = layer.weight_scale.T.contiguous()

    if not is_nvfp4:
        weight_scale = weight_scale.view(torch.float8_e8m0fnu)

    weight_scale = weight_scale.to(param_dtype)
    weight_scale = marlin_permute_scales(
        s=weight_scale,
        size_k=part_size_k,
        size_n=part_size_n,
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )

    if is_nvfp4:
        weight_scale = nvfp4_marlin_process_scales(weight_scale)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

        weight_global_scale = layer.weight_global_scale.to(param_dtype)
        weight_global_scale = nvfp4_marlin_process_global_scale(weight_global_scale)
        layer.weight_global_scale = torch.nn.Parameter(
            weight_global_scale, requires_grad=False
        )
    else:
        weight_scale = mxfp4_marlin_process_scales(
            weight_scale, input_dtype=input_dtype
        )
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        bias = marlin_permute_bias(layer.bias)
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)

    return


def prepare_nvfp4_moe_layer_for_marlin(
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    is_act_and_mul: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    logger.warning_once(
        "Your GPU does not have native support for FP4 computation but "
        "FP4 quantization is being used. Weight-only FP4 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )

    input_dtype = get_marlin_input_dtype(prefix="")
    if input_dtype is not None and input_dtype.itemsize == 1:
        raise RuntimeError("NVFP4 weight + INT8/FP8 activation is not supported.")

    GROUP_SIZE = 16
    E = layer.num_experts
    K = layer.hidden_size
    N = layer.intermediate_size_per_partition

    device = w13.device
    param_dtype = layer.params_dtype
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1

    # WORKSPACE
    layer.workspace = marlin_make_workspace_new(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    # WEIGHT
    # Repack weights to marlin format
    def repack_weight(weight: torch.Tensor, name: str) -> torch.Tensor:
        tensor_list = []
        num_shards = 2 if is_act_and_mul else 1
        if "w13" in name:
            size_n, size_k = N * num_shards, K
        else:
            size_n, size_k = K, N

        assert weight.shape == (E, size_n, size_k // 2)

        for i in range(E):
            qweight = weight[i].view(torch.int32).T.contiguous()

            marlin_qweight = ops.gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
                is_a_8bit=is_a_8bit,
            )
            tensor_list.append(marlin_qweight)

        return torch.cat([x.unsqueeze(0) for x in tensor_list], 0)

    w13 = repack_weight(w13, "w13")
    w2 = repack_weight(w2, "w2")

    # WEIGHT SCALES
    # Permute scales
    def premute_scales(
        scales: torch.Tensor, g_scales: torch.Tensor, name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scales = scales.to(param_dtype)
        g_scales = g_scales.to(param_dtype)

        tensor_list = []
        num_shards = 2 if is_act_and_mul else 1
        if "w13" in name:
            size_n, size_k = N * num_shards, K
        else:
            size_n, size_k = K, N

        for i in range(E):
            scale = scales[i].T
            marlin_scales = marlin_permute_scales(
                s=scale,
                size_k=size_k,
                size_n=size_n,
                group_size=GROUP_SIZE,
                is_a_8bit=is_a_8bit,
            )
            marlin_scales = nvfp4_marlin_process_scales(marlin_scales)
            tensor_list.append(marlin_scales)

        scales = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        g_scales = nvfp4_marlin_process_global_scale(g_scales)
        return scales, g_scales

    w13_scale, w13_scale_2 = premute_scales(w13_scale, w13_scale_2, "w13")
    w2_scale, w2_scale_2 = premute_scales(w2_scale, w2_scale_2, "w2")

    return w13, w13_scale, w13_scale_2, w2, w2_scale, w2_scale_2


def prepare_moe_fp4_layer_for_marlin(
    layer: torch.nn.Module, input_dtype: torch.dtype | None = None
) -> None:
    logger.warning_once(
        "Your GPU does not have native support for FP4 computation but "
        "FP4 quantization is being used. Weight-only FP4 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )

    is_nvfp4 = hasattr(layer, "w13_weight_scale_2")
    if input_dtype is not None and input_dtype.itemsize == 1:
        if is_nvfp4:
            raise RuntimeError("NVFP4 weight + INT8/FP8 activation is not supported.")
        elif input_dtype != torch.float8_e4m3fn:
            raise RuntimeError("MXFP4 weight + INT8 activation is not supported.")

    group_size = 16 if is_nvfp4 else 32

    e = layer.num_experts
    k = layer.hidden_size
    n = layer.intermediate_size_per_partition

    # WORKSPACE
    device = layer.w13_weight.device
    param_dtype = layer.params_dtype
    layer.workspace = marlin_make_workspace_new(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1

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

            marlin_qweight = ops.gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
                is_a_8bit=is_a_8bit,
            )
            tensor_list.append(marlin_qweight)

        weight = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        weight = torch.nn.Parameter(weight, requires_grad=False)

        setattr(layer, name, weight)

    # WEIGHT SCALES
    # Permute scales
    for name in ["w13", "w2"]:
        scales = getattr(layer, name + "_weight_scale")
        if not is_nvfp4:
            scales = scales.view(torch.float8_e8m0fnu)
        scales = scales.to(param_dtype)
        if is_nvfp4:
            global_scale = getattr(layer, name + "_weight_scale_2").to(param_dtype)

        tensor_list = []
        if "w13" in name:
            size_n, size_k = n * 2, k
        else:
            size_n, size_k = k, n

        for i in range(e):
            scale = scales[i].T

            marlin_scales = marlin_permute_scales(
                s=scale,
                size_k=size_k,
                size_n=size_n,
                group_size=group_size,
                is_a_8bit=is_a_8bit,
            )
            if is_nvfp4:
                marlin_scales = nvfp4_marlin_process_scales(marlin_scales)
            else:
                marlin_scales = mxfp4_marlin_process_scales(
                    marlin_scales, input_dtype=input_dtype
                )
            tensor_list.append(marlin_scales)

        scales = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        scales = torch.nn.Parameter(scales, requires_grad=False)
        setattr(layer, name + "_weight_scale", scales)

        if is_nvfp4:
            global_scale = nvfp4_marlin_process_global_scale(global_scale)
            global_scale = torch.nn.Parameter(global_scale, requires_grad=False)
            setattr(layer, name + "_weight_scale_2", global_scale)

    # BIAS
    # Permute bias
    for name in ["w13_bias", "w2_bias"]:
        if not hasattr(layer, name):
            continue
        bias = getattr(layer, name).to(param_dtype)

        tensor_list = []
        for i in range(e):
            expert_bias = bias[i]

            tensor_list.append(marlin_permute_bias(expert_bias))

        bias = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        bias = torch.nn.Parameter(bias, requires_grad=False)
        setattr(layer, name, bias)


def rand_marlin_weight_nvfp4_like(weight, group_size, input_dtype=None):
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1

    assert not is_a_8bit, "NVFP4 weight + INT8/FP8 activation is not supported."
    assert group_size > 0
    size_n, size_k = weight.shape
    device = weight.device

    scales = weight.view(size_n, -1, group_size).abs().max(-1)[0] / 6
    global_scale = scales.max() / 448
    scales = (scales / global_scale).to(torch.float8_e4m3fn)

    fp4_weight = torch.randint(
        0, 256, (size_n, size_k // 2), dtype=torch.uint8, device=weight.device
    )
    fp4_weight_part_1 = (fp4_weight & 0b10000000) | ((fp4_weight & 0b01110000) >> 2)
    fp4_weight_part_1 = fp4_weight_part_1.view(torch.float8_e4m3fn)
    fp4_weight_part_1 = fp4_weight_part_1.to(weight.dtype) * (2**6)

    fp4_weight2 = fp4_weight << 4
    fp4_weight_part_2 = (fp4_weight2 & 0b10000000) | ((fp4_weight2 & 0b01110000) >> 2)
    fp4_weight_part_2 = fp4_weight_part_2.view(torch.float8_e4m3fn)
    fp4_weight_part_2 = fp4_weight_part_2.to(weight.dtype) * (2**6)

    weight_ref = torch.cat(
        [fp4_weight_part_2.unsqueeze(2), fp4_weight_part_1.unsqueeze(2)], 2
    ).view(size_n, size_k)
    weight_ref = (
        weight_ref
        * global_scale.to(weight.dtype)
        * scales.repeat_interleave(group_size, 1).to(weight.dtype)
    )

    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=fp4_weight.view(torch.int32).T.contiguous(),
        perm=torch.empty(0, dtype=torch.int, device=device),
        size_k=size_k,
        size_n=size_n,
        num_bits=4,
        is_a_8bit=is_a_8bit,
    )

    marlin_scales = marlin_permute_scales(
        s=scales.T.to(weight.dtype),
        size_k=size_k,
        size_n=size_n,
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )
    marlin_scales = nvfp4_marlin_process_scales(marlin_scales)

    global_scale = nvfp4_marlin_process_global_scale(global_scale)

    return weight_ref.T, marlin_qweight, marlin_scales, global_scale


def rand_marlin_weight_mxfp4_like(weight, group_size, input_dtype=None):
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1
    if is_a_8bit:
        assert input_dtype == torch.float8_e4m3fn, (
            "MXFP4 weight + INT8 activation is not supported."
        )

    assert group_size > 0
    size_n, size_k = weight.shape
    device = weight.device

    scales = torch.randint(
        110,
        120,
        (size_n, size_k // group_size),
        dtype=torch.uint8,
        device=weight.device,
    )
    scales = scales.view(torch.float8_e8m0fnu)

    fp4_weight = torch.randint(
        0, 256, (size_n, size_k // 2), dtype=torch.uint8, device=weight.device
    )
    fp4_weight_part_1 = (fp4_weight & 0b10000000) | ((fp4_weight & 0b01110000) >> 2)
    fp4_weight_part_1 = fp4_weight_part_1.view(torch.float8_e4m3fn)
    fp4_weight_part_1 = fp4_weight_part_1.to(weight.dtype) * (2**6)

    fp4_weight2 = fp4_weight << 4
    fp4_weight_part_2 = (fp4_weight2 & 0b10000000) | ((fp4_weight2 & 0b01110000) >> 2)
    fp4_weight_part_2 = fp4_weight_part_2.view(torch.float8_e4m3fn)
    fp4_weight_part_2 = fp4_weight_part_2.to(weight.dtype) * (2**6)

    weight_ref = torch.cat(
        [fp4_weight_part_2.unsqueeze(2), fp4_weight_part_1.unsqueeze(2)], 2
    ).view(size_n, size_k)
    weight_ref = weight_ref * scales.repeat_interleave(group_size, 1).to(weight.dtype)

    perm = torch.empty(0, dtype=torch.int, device=device)
    fp4_weight = fp4_weight.view(torch.int32).T.contiguous()
    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=fp4_weight,
        perm=perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=4,
        is_a_8bit=is_a_8bit,
    )

    marlin_scales = marlin_permute_scales(
        s=scales.T.to(weight.dtype),
        size_k=size_k,
        size_n=size_n,
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )

    marlin_scales = mxfp4_marlin_process_scales(marlin_scales, input_dtype=input_dtype)

    return weight_ref.T, marlin_qweight, marlin_scales.to(torch.float8_e8m0fnu)
