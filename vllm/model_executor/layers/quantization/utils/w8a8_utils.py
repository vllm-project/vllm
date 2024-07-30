from typing import List, Optional, Tuple, Union

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform


def cutlass_fp8_supported() -> bool:
    capability = current_platform.get_device_capability()
    capability = capability[0] * 10 + capability[1]

    return ops.cutlass_scaled_mm_supports_fp8(capability)


def per_tensor_dequantize(
        tensor: torch.Tensor, inv_scale: Union[float,
                                               torch.Tensor]) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))


def create_per_tensor_scale_param(
    output_partition_sizes: List[int],
    **extra_weight_attrs,
) -> Parameter:
    scale = Parameter(torch.empty(len(output_partition_sizes),
                                  dtype=torch.float32),
                      requires_grad=False)
    scale[:] = torch.finfo(torch.float32).min
    set_weight_attrs(scale, {
        "needs_scalar_to_array": True,
        **extra_weight_attrs
    })
    return scale


def create_per_channel_scale_param(output_partition_sizes: List[int],
                                   **extra_weight_attrs) -> Parameter:
    scale = Parameter(torch.empty((sum(output_partition_sizes), 1),
                                  dtype=torch.float32),
                      requires_grad=False)
    scale[:] = torch.finfo(torch.float32).min
    set_weight_attrs(scale, {"output_dim": 0, **extra_weight_attrs})
    return scale


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
    unfused_module_in_checkpoint = (weight_scale[-1] > torch.finfo(
        torch.float8_e4m3fn).min)

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


def apply_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_fp8_supported: bool = True,
    use_per_token_if_dynamic: bool = False,
) -> torch.Tensor:
    # ops.scaled_fp8_quant supports both dynamic and static quant.
    #   If dynamic, layer.input_scale is None and x_scale computed from x.
    #   If static, layer.input_scale is scalar and x_scale is input_scale.

    # cutlass_scaled_mm supports per tensor/channel W and per tensor/token A
    if cutlass_fp8_supported:
        qinput, x_scale = ops.scaled_fp8_quant(
            input,
            input_scale,
            scale_ub=input_scale_ub,
            use_per_token_if_dynamic=use_per_token_if_dynamic)

        # Fused GEMM_DQ
        return ops.cutlass_scaled_mm(qinput,
                                     weight,
                                     out_dtype=input.dtype,
                                     scale_a=x_scale,
                                     scale_b=weight_scale,
                                     bias=bias)

    # torch.scaled_mm supports per tensor weights + activations only
    # so fallback to naive if per channel or per token
    else:
        # Note: we pad the input because torch._scaled_mm is more performant
        # for matrices with batch dimension > 16.
        # This could change in the future.
        qinput, x_scale = ops.scaled_fp8_quant(
            input,
            input_scale,
            num_token_padding=17,
            use_per_token_if_dynamic=use_per_token_if_dynamic)

        per_tensor_weights = (weight_scale.numel() == 1)
        per_tensor_activations = (x_scale.numel() == 1)

        if per_tensor_weights and per_tensor_activations:
            # Fused GEMM_DQ
            output, _ = torch._scaled_mm(qinput,
                                         weight,
                                         out_dtype=input.dtype,
                                         scale_a=x_scale,
                                         scale_b=weight_scale,
                                         bias=bias)
            return torch.narrow(output, 0, 0, input.shape[0])

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
            output, _ = torch._scaled_mm(qinput,
                                         weight,
                                         out_dtype=torch.float32)
            # Unpad (undo num_token_padding)
            output = torch.narrow(output, 0, 0, input.shape[0])
            x_scale = torch.narrow(x_scale, 0, 0, input.shape[0])

            # DQ
            # C = sw * sx * (X * W) + bias
            output = output * x_scale * weight_scale.t()
            if bias is not None:
                output = output + bias
            return output.to(dtype=input.dtype)


def apply_int8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
):
    # ops.scaled_int8_quant supports both dynamic and static quant.
    # * dynamic, layer.input_scale is None and x_scale computed from x.
    # * static, layer.input_scale is scalar and x_scale is input_scale.
    x_q, x_scale = ops.scaled_int8_quant(input, input_scale)

    return ops.cutlass_scaled_mm(x_q,
                                 weight,
                                 scale_a=x_scale,
                                 scale_b=weight_scale,
                                 out_dtype=input.dtype,
                                 bias=bias)
