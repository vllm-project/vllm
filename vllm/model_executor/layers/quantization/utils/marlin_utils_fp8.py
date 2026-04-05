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
    should_use_atomic_add_reduce,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


def is_fp8_marlin_supported():
    return current_platform.has_device_capability(75)


def fp8_fused_exponent_bias_into_scales(scales):
    fp8_exponent = 4
    if scales.dtype == torch.half:
        target_exponent = 5
    elif scales.dtype == torch.bfloat16:
        target_exponent = 8
    # exponent_bias_fp16 = 2 ** 4 - 2 ** 3 = 8
    # exponent_bias_bf16 = 2 ** 7 - 2 ** 3 = 120
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp8_exponent - 1)
    s = torch.ones_like(scales) * 2
    s = s**exponent_bias
    return scales * s


def marlin_pad(size_n: int, size_k: int, is_a_8bit: bool = False) -> tuple[int, int]:
    """
    Calc padded shape [N, K].
    """
    tile_size = 16
    tile_k_size = tile_size
    tile_n_size = tile_k_size * 4
    target_tile_n_size = tile_n_size // (2 if is_a_8bit else 1)
    target_tile_k_size = tile_k_size * (2 if is_a_8bit else 1)
    return (
        ((size_n + target_tile_n_size - 1) // target_tile_n_size) * target_tile_n_size,
        ((size_k + target_tile_k_size - 1) // target_tile_k_size) * target_tile_k_size,
    )


def apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None,
    input_dtype: torch.dtype | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    # For GPUs that lack FP8 hardware support, we can leverage the
    # Marlin kernel for fast weight-only FP8 quantization

    n_size, _ = marlin_pad(size_n, size_k)
    pad_n = n_size - size_n

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (n_size,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0), n=n_size, k=size_k, device=input.device, dtype=input.dtype
    )

    inputs = reshaped_x
    a_scales = None
    if input_dtype is not None and input_dtype.itemsize == 1:
        # inputs, a_scales = marlin_quant_input(inputs, torch.float8_e4m3fn)
        raise RuntimeError("Marlin W8A8 is not supported.")

    if pad_n != 0 and bias is not None:
        padding = (0, pad_n)
        bias = torch.nn.functional.pad(bias, padding, mode="constant", value=0)

    output = ops.marlin_gemm(
        a=inputs,
        c=None,
        b_q_weight=weight,
        b_bias=bias,
        b_scales=weight_scale,
        a_scales=a_scales,
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float8_e4m3fn,
        size_m=reshaped_x.size(0),
        size_n=n_size,
        size_k=size_k,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    return output.reshape(out_shape)[..., :size_n]


def prepare_fp8_layer_for_marlin(
    layer: torch.nn.Module,
    size_k_first: bool = True,
    input_dtype: torch.dtype | None = None,
) -> None:
    logger.warning_once(
        "Your GPU does not have native support for FP8 computation but "
        "FP8 quantization is being used. Weight-only FP8 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )
    if input_dtype is not None and input_dtype.itemsize == 1:
        raise RuntimeError("Marlin W8A8 is not supported.")

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    weight_block_size = getattr(layer, "weight_block_size", None)

    # Only 0-padding N dim if requires.
    # The K dimension padding involves complex transformations,
    # and since no models currently requiring K dimension padding
    # have been identified, we will omit K dimension padding.

    n_size, _ = marlin_pad(part_size_n, part_size_k)
    pad_n = n_size - part_size_n

    if size_k_first:
        assert layer.weight.shape == (part_size_k, part_size_n)
        padding = (0, pad_n, 0, 0)
    else:
        assert layer.weight.shape == (part_size_n, part_size_k)
        padding = (0, 0, 0, pad_n)

    # WEIGHT SCALES
    # Permute scales
    scales = None
    orig_dtype = getattr(layer, "orig_dtype", None)
    if orig_dtype is None:
        orig_dtype = getattr(layer, "params_dtype", None)
    if orig_dtype is None:
        raise RuntimeError(
            "Marlin requires either orig_dtype or params_dtype to be set."
        )
    if "weight_scale" in dir(layer):
        scales = layer.weight_scale.to(orig_dtype)
    elif "weight_scale_inv" in dir(layer):
        scales = layer.weight_scale_inv.to(orig_dtype)
    if scales is None:
        raise RuntimeError("Marlin fp8 requires scales")

    if pad_n != 0:
        if weight_block_size is not None or scales.nelement() != 1:
            raise RuntimeError(
                "Marlin imposes alignment requirements for N-dimensional tensors."
                " However, currently supports zero-padding only for weight"
                " with tensor scale."
            )
        padded_weight = torch.nn.functional.pad(
            layer.weight, padding, mode="constant", value=0
        )
        replace_parameter(layer, "weight", padded_weight)
        part_size_n = n_size

    device = layer.weight.device

    # WORKSPACE
    layer.workspace = marlin_make_workspace_new(device)

    # WEIGHT
    # Repack weights to marlin format
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = pack_fp8_to_int32(layer.weight, size_k_first)
    if not size_k_first:
        qweight = qweight.T.contiguous()

    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=8,
    )
    replace_parameter(layer, "weight", marlin_qweight)

    group_size = -1 if weight_block_size is None else weight_block_size[1]

    # marlin kernel only support channel-wise and group-wise quantization
    # we need to convert the scales
    if weight_block_size is None:
        logical_widths = getattr(layer, "logical_widths", [])
        if scales.nelement() == 1:
            # tensor-wise quantization -> channel-wise quantization
            # (1, 1) =>(repeat)=> (1, size_n)
            scales = scales.view(1, 1).repeat_interleave(part_size_n, 1)
        elif scales.nelement() == len(logical_widths):
            # tensor-wise quantization with logical_widths ->
            #    channel-wise quantization
            assert sum(logical_widths) == part_size_n, (
                f"Sum of logical_widths ({sum(logical_widths)}) must be equal "
                f"to part_size_n ({part_size_n})"
            )
            lw_tensor = scales.new_tensor(logical_widths, dtype=torch.int64)
            scales = scales.view(1, -1).repeat_interleave(lw_tensor, dim=1)
        elif scales.nelement() > 1 and scales.nelement() != part_size_n:
            assert part_size_n % scales.nelement() == 0
            s_size = scales.nelement()
            # tensor-wise quantization (for gate-up proj)
            #     -> channel-wise quantization
            # (1, s_size) =>(repeat)=> (1, size_n)
            scales = scales.view(1, s_size)
            scales = scales.repeat_interleave(part_size_n // s_size, 1)
        else:
            # channel-wise quantization
            # (1, size_n)
            scales = scales.view(1, part_size_n)
    else:
        # block-wise quantization -> group-wise quantization
        # (size_k // block_size[1], ceil(size_n / block_size[0]))
        #  =>(repeat)=> (size_k // block_size[1], size_n)
        if not size_k_first:
            scales = scales.T.contiguous()
        block_n = weight_block_size[0]
        scales = scales.repeat_interleave(block_n, 1)
        # size_n may not divisible by block_size[0]
        scales = scales[:, :part_size_n]

    marlin_scales = marlin_permute_scales(
        s=scales, size_k=part_size_k, size_n=part_size_n, group_size=group_size
    )
    if input_dtype != torch.float8_e4m3fn:
        marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)
    if hasattr(layer, "weight_scale"):
        replace_parameter(layer, "weight_scale", marlin_scales)
    elif hasattr(layer, "weight_scale_inv"):
        replace_parameter(layer, "weight_scale_inv", marlin_scales)

    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        bias = marlin_permute_bias(layer.bias)
        replace_parameter(layer, "bias", bias)


def prepare_fp8_moe_layer_for_marlin(
    layer: torch.nn.Module,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shuffle weights and scales into marlin format.

    Note that this function has the side effect of adding a `workspace`
    attribute to the layer. This `workspace` does not need to be
    registered as a Parameter as it is not used during weight reloading.
    """

    logger.warning_once(
        "Your GPU does not have native support for FP8 computation but "
        "FP8 quantization is being used. Weight-only FP8 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )
    input_dtype = get_marlin_input_dtype()
    if input_dtype is not None and input_dtype.itemsize == 1:
        raise NotImplementedError("Marlin W8A8 is not supported.")

    e = layer.num_experts
    k = layer.hidden_size
    n = layer.intermediate_size_per_partition
    w13_n = w13_weight.size(1)
    weight_block_size = getattr(layer, "weight_block_size", None)

    # WORKSPACE
    device = layer.w13_weight.device
    # NOTE(rob): we do not need to register the workspace as a param
    # because it is not used as part of the weight reloading process.
    layer.workspace = marlin_make_workspace_new(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    # WEIGHT
    # Repack weights to marlin format
    def repack_weight(name: str, weight: torch.Tensor) -> torch.Tensor:
        tensor_list = []
        if "w13" in name:
            size_n, size_k = w13_n, k
        else:
            size_n, size_k = k, n

        assert weight.shape == (e, size_n, size_k)

        for i in range(e):
            qweight = pack_fp8_to_int32(weight[i], size_k_first=False)
            qweight = qweight.T.contiguous()

            marlin_qweight = ops.gptq_marlin_repack(
                b_q_weight=qweight, perm=perm, size_k=size_k, size_n=size_n, num_bits=8
            )
            tensor_list.append(marlin_qweight)

        return torch.cat([x.unsqueeze(0) for x in tensor_list], 0)

    w13_weight = repack_weight("w13", w13_weight)
    w2_weight = repack_weight("w2", w2_weight)

    # WEIGHT SCALES
    # Permute scales
    group_size = -1 if weight_block_size is None else weight_block_size[1]

    def permute_scales(scales: torch.Tensor, name: str) -> torch.Tensor:
        scales = scales.to(layer.orig_dtype)
        tensor_list = []
        if "w13" in name:
            size_n, size_k = w13_n, k
        else:
            size_n, size_k = k, n

        # marlin kernel only support channel-wise and group-wise quantization
        # we need to convert the scales
        if weight_block_size is None:
            if scales.nelement() == e:
                # tensor-wise quantization -> channel-wise quantization
                # (e, 1, 1) =>(repeat)=> (e, 1, size_n)
                scales = scales.view(e, 1, 1).repeat_interleave(size_n, 2)
            elif scales.nelement() > e and scales.nelement() != e * size_n:
                assert (e * size_n) % scales.nelement() == 0
                s_size = scales.nelement() // e
                # tensor-wise quantization (for gate-up proj)
                #     -> channel-wise quantization
                # (e, 1, s_size) =>(repeat)=> (e, 1, size_n)
                scales = scales.view(e, 1, s_size)
                scales = scales.repeat_interleave(size_n // s_size, 2)
            else:
                # channel-wise quantization
                # (e, 1, size_n)
                scales = scales.view(e, 1, size_n)
        else:
            # block-wise quantization -> group-wise quantization
            # (e, size_k // block_size[1], ceil(size_n / block_size[0]))
            #  =>(repeat)=> (e, size_k // block_size[1], size_n)
            scales = scales.permute(0, 2, 1)
            block_n = weight_block_size[0]
            scales = scales.repeat_interleave(block_n, 2)
            # size_n may not divisible by block_size[0]
            scales = scales[..., :size_n].contiguous()

        for i in range(e):
            marlin_scales = marlin_permute_scales(
                s=scales[i], size_k=size_k, size_n=size_n, group_size=group_size
            )
            tensor_list.append(marlin_scales)

        scales = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        if input_dtype != torch.float8_e4m3fn:
            scales = fp8_fused_exponent_bias_into_scales(scales)
        return scales

    w13_weight_scale = permute_scales(w13_weight_scale, "w13")
    w2_weight_scale = permute_scales(w2_weight_scale, "w2")

    return w13_weight, w2_weight, w13_weight_scale, w2_weight_scale


def pack_fp8_to_int32(
    fp8_tensor: torch.Tensor, size_k_first: bool = True
) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements)
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert fp8_tensor.ndim == 2

    fp8_tensor = fp8_tensor.T if size_k_first else fp8_tensor
    fp8_tensor = fp8_tensor.contiguous()
    # fp8_tensor is contiguous and have shape (N, K) now
    # with `.view(torch.int32)`, it become (N, K // 4)
    int32_tensor = fp8_tensor.view(torch.int32)
    return int32_tensor.T.contiguous() if size_k_first else int32_tensor


def mxfp8_marlin_process_scales(marlin_scales: torch.Tensor) -> torch.Tensor:
    """Reorder scales for e8m0 kernel layout and convert to float8_e8m0fnu."""
    # fit the layout of fp8 dequantization
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1
    )
    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
    return marlin_scales


def apply_mxfp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=size_n,
        k=size_k,
        device=input.device,
        dtype=input.dtype,
    )

    output = ops.marlin_gemm(
        a=reshaped_x,
        c=None,
        b_q_weight=weight,
        b_bias=bias,
        b_scales=weight_scale,
        a_scales=None,
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float8_e4m3fn,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    return output.reshape(out_shape)


def prepare_mxfp8_layer_for_marlin(layer: torch.nn.Module) -> None:
    """Repack MXFP8 weights and scales into Marlin kernel format.

    Expects the layer to have:
      - weight: [N, K] float8_e4m3fn
      - weight_scale: [N, K//32] uint8 (e8m0 encoded)
      - input_size_per_partition / output_size_per_partition
    """
    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    group_size = 32  # MX standard block size

    device = layer.weight.device

    # WORKSPACE
    layer.workspace = marlin_make_workspace_new(device)

    # WEIGHT - repack FP8 weights to Marlin format
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = pack_fp8_to_int32(layer.weight, size_k_first=False)
    qweight = qweight.T.contiguous()

    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=8,
    )
    replace_parameter(layer, "weight", marlin_qweight)

    # WEIGHT SCALES
    # Convert uint8 scales -> e8m0fnu -> param_dtype for permutation
    # Scales are [N, K//32], need [K//32, N] for marlin_permute_scales
    param_dtype = torch.get_default_dtype()
    scales = layer.weight_scale.data[:part_size_n, : part_size_k // group_size]
    scales = scales.contiguous()
    scales = scales.view(torch.float8_e8m0fnu).to(param_dtype)
    scales = scales.T.contiguous()

    # Permute scales to Marlin layout
    marlin_scales = marlin_permute_scales(
        s=scales,
        size_k=part_size_k,
        size_n=part_size_n,
        group_size=group_size,
    )

    # Reorder for e8m0 kernel layout and convert back to e8m0fnu
    marlin_scales = mxfp8_marlin_process_scales(marlin_scales)
    replace_parameter(layer, "weight_scale", marlin_scales)

    # BIAS
    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        bias = marlin_permute_bias(layer.bias)
        replace_parameter(layer, "bias", bias)


def prepare_mxfp8_moe_layer_for_marlin(
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repack MXFP8 MoE weights and scales into Marlin kernel format.

    Args:
        layer: MoE layer (used to read params_dtype and attach workspace).
        w13: [E, 2*N, K] float8_e4m3fn weights.
        w2:  [E, K, N] float8_e4m3fn weights.
        w13_scale: [E, 2*N, K//32] uint8 e8m0 scales.
        w2_scale:  [E, K, N//32] uint8 e8m0 scales.

    Returns:
        (w13, w2, w13_scale, w2_scale) in Marlin format.
    """
    group_size = 32
    e = w13.shape[0]
    w13_n = w13.shape[1]
    k = w13.shape[2]
    n = w2.shape[2]

    device = w13.device
    param_dtype = torch.get_default_dtype()
    perm = torch.empty(0, dtype=torch.int, device=device)

    layer.workspace = marlin_make_workspace_new(device, 4)

    def repack_weight(weight: torch.Tensor, name: str) -> torch.Tensor:
        if "w13" in name:
            size_n, size_k = w13_n, k
        else:
            size_n, size_k = k, n

        assert weight.shape == (e, size_n, size_k)

        tensor_list = []
        for i in range(e):
            qweight = pack_fp8_to_int32(weight[i], size_k_first=False)
            qweight = qweight.T.contiguous()
            marlin_qweight = ops.gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=8,
            )
            tensor_list.append(marlin_qweight)
        return torch.cat([x.unsqueeze(0) for x in tensor_list], 0)

    w13 = repack_weight(w13, "w13")
    w2 = repack_weight(w2, "w2")

    def permute_scales(scales: torch.Tensor, name: str) -> torch.Tensor:
        if "w13" in name:
            size_n, size_k = w13_n, k
        else:
            size_n, size_k = k, n

        tensor_list = []
        for i in range(e):
            s = scales[i][:size_n, : size_k // group_size].contiguous()
            s = s.view(torch.float8_e8m0fnu).to(param_dtype)
            s = s.T.contiguous()
            marlin_s = marlin_permute_scales(
                s=s,
                size_k=size_k,
                size_n=size_n,
                group_size=group_size,
            )
            marlin_s = mxfp8_marlin_process_scales(marlin_s)
            tensor_list.append(marlin_s)
        return torch.cat([x.unsqueeze(0) for x in tensor_list], 0)

    w13_scale = permute_scales(w13_scale, "w13")
    w2_scale = permute_scales(w2_scale, "w2")

    return w13, w2, w13_scale, w2_scale


def marlin_quant_fp8_torch(weight, group_size, input_dtype=None):
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1
    if is_a_8bit:
        assert input_dtype == torch.float8_e4m3fn

    size_n, size_k = weight.shape
    device = weight.device

    if group_size != -1:
        scales = weight.view(size_n, -1, group_size).abs().max(-1)[0] / 448
        repeated_scales = scales.repeat_interleave(group_size, 1)
        fp8_weight = (weight / repeated_scales).to(torch.float8_e4m3fn)
        weight_ref = fp8_weight.to(weight.dtype) * repeated_scales
    else:
        scales = weight.view(size_n, 1, group_size).abs().max(-1)[0] / 448
        repeated_scales = scales.repeat_interleave(size_k, 1)
        fp8_weight = (weight / repeated_scales).to(torch.float8_e4m3fn)
        weight_ref = fp8_weight.to(weight.dtype) * repeated_scales

    packed_weight = pack_fp8_to_int32(fp8_weight, False).T.contiguous()
    perm = torch.empty(0, dtype=torch.int, device=device)
    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=packed_weight,
        perm=perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=8,
        is_a_8bit=is_a_8bit,
    )

    marlin_scales = marlin_permute_scales(
        s=scales.T,
        size_k=size_k,
        size_n=size_n,
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )

    marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)

    return weight_ref.T, marlin_qweight, marlin_scales
