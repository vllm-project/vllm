# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

from .layer_utils import replace_parameter

if TYPE_CHECKING:
    from vllm.model_executor.kernels.linear.int8.base import (
        Int8LinearLayerConfig,
    )


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


def cutlass_group_gemm_supported() -> bool:
    if not current_platform.is_cuda():
        return False

    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()

    return ops.cutlass_group_gemm_supported(capability)


CUTLASS_BLOCK_FP8_SUPPORTED = cutlass_block_fp8_supported()


def per_tensor_dequantize(
    tensor: torch.Tensor, inv_scale: float | torch.Tensor
) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))


def convert_to_channelwise(
    weight_scale: torch.Tensor, logical_widths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    # Create channelwise buffer
    weight_scale_channel = torch.empty(
        (sum(logical_widths), 1), dtype=torch.float32, device=weight_scale.device
    )

    # Expand each scale to match the size of each logical matrix.
    start = 0
    for idx, logical_width in enumerate(logical_widths):
        end = start + logical_width
        weight_scale_channel[start:end, :] = weight_scale[idx]
        start = end

    return weight_scale_channel


def prepare_int8_scaled_mm_weights(
    layer: torch.nn.Module,
    config: "Int8LinearLayerConfig",
    param_names: Sequence[str],
) -> None:
    """Convert int8 W8A8 weights on `layer` to the `cutlass_scaled_mm` layout.

    The weight becomes `[K, N]`, fused-module weight scales become per-channel,
    static asymmetric input scales and zero points collapse to one of each, and
    `azp_adj` is computed for asymmetric activations (see
    `csrc/quantization/w8a8/cutlass/Epilogues.md`). Input parameters the scheme
    does not use are cleared.
    """
    w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = param_names
    weight = getattr(layer, w_q_name)
    replace_parameter(
        layer, w_q_name, torch.nn.Parameter(weight.t().data, requires_grad=False)
    )

    weight_scale = getattr(layer, w_s_name)
    if len(layer.logical_widths) > 1 and not config.is_channelwise:
        weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
    replace_parameter(
        layer, w_s_name, torch.nn.Parameter(weight_scale.data, requires_grad=False)
    )

    if config.is_static_input_scheme:
        input_scale = getattr(layer, i_s_name)
        if config.input_symmetric:
            replace_parameter(
                layer,
                i_s_name,
                torch.nn.Parameter(input_scale.max(), requires_grad=False),
            )
        else:
            # Reconstruct the ranges to find a single scale and zero point.
            int8_traits = torch.iinfo(torch.int8)
            azps = getattr(layer, i_zp_name).to(dtype=torch.int32)
            range_max = (input_scale * (int8_traits.max - azps)).max()
            range_min = (input_scale * (int8_traits.min - azps)).min()
            scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
            replace_parameter(
                layer, i_s_name, torch.nn.Parameter(scale, requires_grad=False)
            )
            # The zero point loads as int8 but is used as int32.
            azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)
            replace_parameter(
                layer, i_zp_name, torch.nn.Parameter(azp, requires_grad=False)
            )
    else:
        setattr(layer, i_s_name, None)
        setattr(layer, i_zp_name, None)

    if config.input_symmetric:
        setattr(layer, azp_adj_name, None)
    else:
        weight = getattr(layer, w_q_name)
        azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
        if config.is_static_input_scheme:
            # Per-tensor zero points are folded into azp_adj.
            azp_adj = getattr(layer, i_zp_name) * azp_adj
        replace_parameter(layer, azp_adj_name, azp_adj)


def requantize_with_max_scale(
    weight: torch.Tensor, weight_scale: torch.Tensor, logical_widths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    # Max scale to be used for requanitzation.
    max_w_scale = weight_scale.max()

    # QKV / MLP is fused in the on disk checkpoint if any of the
    # weight scales are still set to the default since we initialize
    # N weight scales for N shards but we only load 1 weight scale
    # from disk in this case. Skip requantization in this case (since)
    # we already are quantized with the single scale.
    # * Sample Model: nm-testing/Phi-3-mini-128k-instruct-FP8
    #
    # Extra note: upon weight reloading weight_scale.ndim == 0
    unfused_module_in_checkpoint = (
        weight_scale.ndim != 0
        and weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min
    )

    # If unfused checkpoint, need requanize with the single scale.
    if unfused_module_in_checkpoint:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            # Skip any component with zero width.
            if logical_width == 0:
                continue
            end = start + logical_width
            weight_dq = per_tensor_dequantize(weight[start:end, :], weight_scale[idx])
            weight[start:end, :], _ = ops.scaled_fp8_quant(weight_dq, max_w_scale)
            start = end

    return max_w_scale, weight


def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
