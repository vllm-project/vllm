# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter

BFLOAT16_EXP_BIAS = 127
BFLOAT16_MANTISSA_BITS = 7
BFLOAT16_EXP_BITS = 8

FLOAT16_EXP_BIAS = 15
FLOAT16_MANTISSA_BITS = 10
FLOAT16_EXP_BITS = 5

FLOAT8_E8M0_MAX_EXP = 127
FLOAT4_EXP_BIAS = 1
FLOAT4_MANTISSA_BITS = 1

FLOAT16_VAL_TO_ADD = 1 << (FLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1)
FLOAT16_SIGN_EXPONENT_MASK = (
    (1 << (FLOAT16_EXP_BITS + 1)) - 1
) << FLOAT16_MANTISSA_BITS

BFLOAT16_VAL_TO_ADD = 1 << (BFLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1)
BFLOAT16_SIGN_EXPONENT_MASK = (
    (1 << (BFLOAT16_EXP_BITS + 1)) - 1
) << BFLOAT16_MANTISSA_BITS


def e8m0_to_half(scale, half_dtype: torch.dtype):
    assert scale.dtype == torch.uint8

    scale_exp = scale.to(torch.int16) - 127

    # This can be implemented with bitwise operations in a proper kernel.
    scale_half = 2.0 ** (scale_exp.to(torch.float))

    return scale_half.to(half_dtype)


def upcast_fp4_to_fp16_or_bf16(
    val, float_dtype: torch.dtype, half_exp_bias: int, half_mantissa_bits: int
):
    assert val.dtype == torch.uint8

    unpacked = torch.zeros(
        *val.shape[:-1], val.shape[-1] * 2, dtype=torch.uint8, device=val.device
    )
    unpacked[..., 1::2] = (val >> 4) & 0x0F  # Extract high 4 bits.
    unpacked[..., ::2] = val & 0x0F  # Extract low 4 bits.

    # Takes one float4 values represented as b0000xxxx,
    # and converts it to the corresponding float16 value.

    sign = unpacked >> 3

    exp = (unpacked >> 1) & 3
    new_mantissa = unpacked & 1

    # if exp == 0 and new_mantissa == 0:
    #     new_exp = 0
    # else:
    #     new_exp = exp - FLOAT4_EXP_BIAS + FLOAT16_EXP_BIAS

    # int8_t works with float16, but may overflow with bfloat16.
    new_exp = exp - FLOAT4_EXP_BIAS + half_exp_bias

    # Cast b0000 to 0. in fp16/bf16.
    new_exp = new_exp * torch.logical_or(exp > 0, new_mantissa > 0)

    # Cast b0001 to 0.5 in fp16/bf16.
    new_mantissa = torch.logical_and(new_mantissa, exp > 0)

    new_mantissa = new_mantissa.to(torch.int32)
    new_exp = new_exp.to(torch.int32)
    sign = sign.to(torch.int32)

    qdq_val = (
        (sign << 15)
        + (new_exp << half_mantissa_bits)
        + (new_mantissa << (half_mantissa_bits - 1))
    )

    assert qdq_val.max() <= 65535
    assert qdq_val.min() >= 0
    qdq_val = qdq_val.to(torch.uint16)

    result = qdq_val.view(float_dtype)

    return result


def dq_mxfp4_torch(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype
) -> torch.Tensor:
    assert x.dtype == torch.uint8
    assert scale.dtype == torch.uint8

    if float_dtype == torch.float16:
        half_exp_bias = FLOAT16_EXP_BIAS
        half_mantissa_bits = FLOAT16_MANTISSA_BITS
    elif float_dtype == torch.bfloat16:
        half_exp_bias = BFLOAT16_EXP_BIAS
        half_mantissa_bits = BFLOAT16_MANTISSA_BITS

    scale_half = e8m0_to_half(scale, half_dtype=float_dtype)

    x_half = upcast_fp4_to_fp16_or_bf16(
        x,
        float_dtype=float_dtype,
        half_exp_bias=half_exp_bias,
        half_mantissa_bits=half_mantissa_bits,
    )

    x_half = x_half.reshape(*x_half.shape[:-1], -1, 32)
    x_half = x_half * scale_half[..., None]
    x_half = x_half.reshape(*x_half.shape[:-2], -1)

    return x_half


class CompressedTensorsW4A4MXFp4(CompressedTensorsScheme):
    def __init__(self):
        self.group_size = 32

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # WEIGHT SCALE
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer) -> None:
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dq_w = dq_mxfp4_torch(layer.weight_packed, layer.weight_scale, x.dtype)
        # qdq_x = quant_dequant_mxfp4(x)
        return torch.nn.functional.linear(x, dq_w, bias)
