# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union

import torch

from vllm import _custom_ops as ops
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.platforms import current_platform

# Input scaling factors are no longer optional in _scaled_mm starting
# from pytorch 2.5. Allocating a dummy tensor to pass as input_scale
TORCH_DEVICE_IDENTITY = None

# The condition to determine if it is on a platform that supports
# torch._scaled_mm rowwise feature.
# The condition is determined once as the operations
# are time consuming.
USE_ROWWISE_TORCH_SCALED_MM = (current_platform.is_rocm()
                               and current_platform.has_device_capability(94))


def sparse_cutlass_supported() -> bool:
    if not current_platform.is_cuda():
        return False

    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()

    return ops.cutlass_sparse_scaled_mm_supported(capability)


def cutlass_fp8_supported() -> bool:
    if not current_platform.is_cuda():
        return False

    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()

    return ops.cutlass_scaled_mm_supports_fp8(capability)


def cutlass_block_fp8_supported() -> bool:
    if not current_platform.is_cuda():
        return False

    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()

    return ops.cutlass_scaled_mm_supports_block_fp8(capability)


CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
CUTLASS_BLOCK_FP8_SUPPORTED = cutlass_block_fp8_supported()


def per_tensor_dequantize(
        tensor: torch.Tensor, inv_scale: Union[float,
                                               torch.Tensor]) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))


def convert_to_channelwise(
        weight_scale: torch.Tensor,
        logical_widths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create channelwise buffer
    weight_scale_channel = torch.empty((sum(logical_widths), 1),
                                       dtype=torch.float32,
                                       device=weight_scale.device)

    # Expand each scale to match the size of each logical matrix.
    start = 0
    for idx, logical_width in enumerate(logical_widths):
        end = start + logical_width
        weight_scale_channel[start:end, :] = weight_scale[idx]
        start = end

    return weight_scale_channel


def requantize_with_max_scale(
        weight: torch.Tensor, weight_scale: torch.Tensor,
        logical_widths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    # Max scale to be used for requanitzation.
    max_w_scale = weight_scale.max()

    # QKV / MLP is fused in the on disk checkpoint if any of the
    # weight scales are still set to the default since we initialize
    # N weight scales for N shards but we only load 1 weight scale
    # from disk in this case. Skip requantization in this case (since)
    # we already are quantized with the single scale.
    # * Sample Model: nm-testing/Phi-3-mini-128k-instruct-FP8
    unfused_module_in_checkpoint = (weight_scale[-1]
                                    > torch.finfo(torch.float8_e4m3fn).min)

    # If unfused checkpoint, need requanize with the single scale.
    if unfused_module_in_checkpoint:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            end = start + logical_width
            weight_dq = per_tensor_dequantize(weight[start:end, :],
                                              weight_scale[idx])
            weight[start:end, :], _ = ops.scaled_fp8_quant(
                weight_dq, max_w_scale)
            start = end

    return max_w_scale, weight


def maybe_create_device_identity():
    # Allocate dummy ones tensor for torch._scaled_mm
    global TORCH_DEVICE_IDENTITY
    if TORCH_DEVICE_IDENTITY is None:
        TORCH_DEVICE_IDENTITY = torch.ones(1, dtype=torch.float32)


def apply_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_fp8_supported: bool = CUTLASS_FP8_SUPPORTED,
    use_per_token_if_dynamic: bool = False,
) -> torch.Tensor:
    # ops.scaled_fp8_quant supports both dynamic and static quant.
    #   If dynamic, layer.input_scale is None and x_scale computed from x.
    #   If static, layer.input_scale is scalar and x_scale is input_scale.

    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[1]]

    # cutlass_scaled_mm supports per tensor/channel W and per tensor/token A
    if cutlass_fp8_supported:
        qinput, x_scale = ops.scaled_fp8_quant(
            input_2d,
            input_scale,
            scale_ub=input_scale_ub,
            use_per_token_if_dynamic=use_per_token_if_dynamic)

        # Fused GEMM_DQ
        output = ops.cutlass_scaled_mm(qinput,
                                       weight,
                                       out_dtype=input.dtype,
                                       scale_a=x_scale,
                                       scale_b=weight_scale,
                                       bias=bias)
        return output.view(*output_shape)

    # torch.scaled_mm supports per tensor weights + activations only
    # so fallback to naive if per channel or per token
    else:
        # Note: we pad the input because torch._scaled_mm is more performant
        # for matrices with batch dimension > 16.
        # This could change in the future.
        # We also don't pad when using torch.compile,
        # as it breaks with dynamic shapes.
        config = get_current_vllm_config().compilation_config
        do_pad = config.level < CompilationLevel.PIECEWISE
        qinput, x_scale = ops.scaled_fp8_quant(
            input_2d,
            input_scale,
            num_token_padding=17 if do_pad else None,
            use_per_token_if_dynamic=use_per_token_if_dynamic)

        per_tensor_weights = (weight_scale.numel() == 1)
        per_tensor_activations = (x_scale.numel() == 1)

        if per_tensor_weights and per_tensor_activations:
            # Fused GEMM_DQ
            output = torch._scaled_mm(qinput,
                                      weight,
                                      out_dtype=input.dtype,
                                      scale_a=x_scale,
                                      scale_b=weight_scale,
                                      bias=bias)
            # A fix for discrepancy in scaled_mm which returns tuple
            # for torch < 2.5 and a single value in torch >= 2.5
            if type(output) is tuple and len(output) == 2:
                output = output[0]

            return torch.narrow(output, 0, 0,
                                input_2d.shape[0]).view(*output_shape)

        elif (use_per_token_if_dynamic and not per_tensor_weights
              and not per_tensor_activations and USE_ROWWISE_TORCH_SCALED_MM):
            # For now validated on ROCm platform
            # fp8 rowwise scaling in torch._scaled_mm is introduced in
            # https://github.com/pytorch/pytorch/pull/144432 using
            # hipBLASLt and ROCm 6.3, which only exists in torch 2.7 and above.
            # For CUDA platform please validate if the
            # torch._scaled_mm support rowwise scaled GEMM
            # Fused GEMM_DQ Rowwise GEMM
            output = torch._scaled_mm(qinput,
                                      weight,
                                      out_dtype=input.dtype,
                                      scale_a=x_scale,
                                      scale_b=weight_scale.t(),
                                      bias=bias)

            output = torch.narrow(output, 0, 0, input_2d.shape[0])
            output = output.view(*output_shape)
            return output

        else:
            # Fallback for channelwise case, where we use unfused DQ
            # due to limitations with scaled_mm

            # Symmetric quantized GEMM by definition computes the following:
            #   C = (s_x * X) (s_w * W) + bias
            # This is equivalent to dequantizing the weights and activations
            # before applying a GEMM.
            #
            # In order to compute quantized operands, a quantized kernel
            # will rewrite the above like so:
            #   C = s_w * s_x * (X * W) + bias
            #
            # For the scaled_mm fallback case, we break this down, since it
            # does not support s_w being a vector.

            # GEMM
            # This computes C = (X * W).
            # Output in fp32 to allow subsequent ops to happen in-place
            output = torch._scaled_mm(qinput,
                                      weight,
                                      scale_a=TORCH_DEVICE_IDENTITY,
                                      scale_b=TORCH_DEVICE_IDENTITY,
                                      out_dtype=torch.float32)
            # A fix for discrepancy in scaled_mm which returns tuple
            # for torch < 2.5 and a single value in torch >= 2.5
            if type(output) is tuple and len(output) == 2:
                output = output[0]
            # Unpad (undo num_token_padding)
            output = torch.narrow(output, 0, 0, input_2d.shape[0])
            x_scale = torch.narrow(x_scale, 0, 0, input_2d.shape[0])

            # DQ
            # C = sw * sx * (X * W) + bias
            output = output * x_scale * weight_scale.t()
            if bias is not None:
                output = output + bias
            return output.to(dtype=input.dtype).view(*output_shape)


def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert weight.dtype == torch.float8_e4m3fn
    # The bits pattern 10000000(-128) represents zero in e4m3fn
    # but NaN in e4m3fnuz. So here we set it to 0.
    # https://onnx.ai/onnx/technical/float8.html
    weight_as_int8 = weight.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    # https://onnx.ai/onnx/technical/float8.html
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale
