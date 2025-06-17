# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

BFLOAT16_EXP_BIAS = 127
BFLOAT16_MANTISSA_BITS = 7
BFLOAT16_EXP_BITS = 8

FLOAT16_EXP_BIAS = 15
FLOAT16_MANTISSA_BITS = 10
FLOAT16_EXP_BITS = 5

FLOAT8_E8M0_MAX_EXP = 127
FLOAT4_EXP_BIAS = 1
FLOAT4_MANTISSA_BITS = 1

FLOAT16_VAL_TO_ADD = (1 << (FLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1))
FLOAT16_SIGN_EXPONENT_MASK = ((
    (1 << (FLOAT16_EXP_BITS + 1)) - 1) << FLOAT16_MANTISSA_BITS)

BFLOAT16_VAL_TO_ADD = (1 <<
                       (BFLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1))
BFLOAT16_SIGN_EXPONENT_MASK = ((
    (1 << (BFLOAT16_EXP_BITS + 1)) - 1) << BFLOAT16_MANTISSA_BITS)


def e8m0_to_half(scale, half_dtype: torch.dtype):
    assert scale.dtype == torch.uint8

    scale_exp = scale.to(torch.int16) - 127

    # This can be implemented with bitwise operations in a proper kernel.
    scale_half = 2.0**(scale_exp.to(torch.float))

    return scale_half.to(half_dtype)


def upcast_fp4_to_fp16_or_bf16(val, float_dtype: torch.dtype,
                               half_exp_bias: int, half_mantissa_bits: int):
    assert val.dtype == torch.uint8

    unpacked = torch.zeros(*val.shape[:-1],
                           val.shape[-1] * 2,
                           dtype=torch.uint8,
                           device=val.device)
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

    qdq_val = (sign << 15) + (new_exp << half_mantissa_bits) + (
        new_mantissa << (half_mantissa_bits - 1))

    assert qdq_val.max() <= 65535
    assert qdq_val.min() >= 0
    qdq_val = qdq_val.to(torch.uint16)

    result = qdq_val.view(float_dtype)

    return result


def dq_mxfp4_torch(x: torch.Tensor, scale: torch.Tensor,
                   float_dtype: torch.dtype) -> torch.Tensor:
    assert x.dtype == torch.uint8
    assert scale.dtype == torch.uint8

    if float_dtype == torch.float16:
        half_exp_bias = FLOAT16_EXP_BIAS
        half_mantissa_bits = FLOAT16_MANTISSA_BITS
    elif float_dtype == torch.bfloat16:
        half_exp_bias = BFLOAT16_EXP_BIAS
        half_mantissa_bits = BFLOAT16_MANTISSA_BITS

    scale_half = e8m0_to_half(scale, half_dtype=float_dtype)

    x_half = upcast_fp4_to_fp16_or_bf16(x,
                                        float_dtype=float_dtype,
                                        half_exp_bias=half_exp_bias,
                                        half_mantissa_bits=half_mantissa_bits)

    x_half = x_half.reshape(*x_half.shape[:-1], -1, 32)
    x_half = x_half * scale_half[..., None]
    x_half = x_half.reshape(*x_half.shape[:-2], -1)

    return x_half


def fp16_to_fp4_simulate(val, half_mantissa_bits: int, half_exp_bits: int,
                         half_exp_bias: int):
    # Casts an fp16/bf16 input to the restricted values of float4_e2m1,
    # that is to say [0., 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0,
    # -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0].

    float_type = val.dtype

    # "rshift_cuda" not implemented for 'UInt16'
    val_view = val.view(torch.int16)  #.to(torch.int32)

    exp = val_view >> half_mantissa_bits
    exp = exp & ((1 << half_exp_bits) - 1)

    exp = exp.view(torch.uint16).to(torch.int32)

    sign = (val_view >> (half_mantissa_bits + half_exp_bits)) & 1

    mantissa_last = (val_view >> (half_mantissa_bits - 1)) & 1

    exp_unbias = exp - half_exp_bias
    new_exp = exp_unbias + FLOAT4_EXP_BIAS

    exp_shift = (new_exp <= 0) * (1 - new_exp)

    # Typically 9.
    # Take the min to prevent overflow on `uint16_t half`. This is the case for
    # very small values, correctly mapped to `round_close`.
    tail_bits = half_mantissa_bits - FLOAT4_MANTISSA_BITS + exp_shift
    tail_bits[tail_bits >= 16] = 16

    mantissa_plus_one = val_view & ((1 << (half_mantissa_bits + 1)) - 1)

    half = 1 << (tail_bits - 1)

    tail = mantissa_plus_one & ((1 << tail_bits) - 1)

    round_close = (tail < half)  # round towards 0
    round_away = (tail > half)  # round away from 0
    tie = tail == half

    new_mantissa_close = torch.zeros(val.shape,
                                     device=val.device,
                                     dtype=torch.bool)
    new_exp_close = torch.zeros(val.shape,
                                device=val.device,
                                dtype=torch.uint16)

    new_mantissa_away = torch.zeros(val.shape,
                                    device=val.device,
                                    dtype=torch.bool)
    new_exp_away = torch.zeros(val.shape,
                               device=val.device,
                               dtype=torch.uint16)

    new_exp_tie = torch.zeros(val.shape, device=val.device, dtype=torch.uint16)

    # 1. round down
    # if new_exp == 0: # case [0.5, 0.749999]
    #     new_mantissa = 0
    # elif new_exp < 0:  # case [0, 0.24999]
    #     new_mantissa = 0
    # else:
    #     new_mantissa = mantissa_last

    new_mantissa_close = (new_exp > 0) * mantissa_last
    new_exp_close = exp

    # # 2. round up
    # if new_exp <= 0:  # case [0.250001, 0.499999] and [0.75001, 0.99999]
    #     new_mantissa = 0
    #     new_exp += 1
    # elif mantissa_last == 0:
    #     new_mantissa = 1
    # else:
    #     new_mantissa = 0
    #     new_exp += 1

    new_mantissa_away = torch.logical_and(new_exp > 0, mantissa_last == 0)
    new_exp_away = exp + torch.logical_or(new_exp <= 0, mantissa_last == 1)

    # # 3. tie
    # 0.25 -> 0. (handled by `exp > (half_exp_bias - 2)`)
    # 0.75 -> 1.
    # 1.25 -> 1.
    # 1.75 -> 2.
    # 2.5 -> 2.
    # 3.5 -> 4.
    # 5. -> 4.
    new_exp_tie = (exp > (half_exp_bias - 2)) * (exp + (mantissa_last == 1))

    # Gather round up, round down and tie.
    new_exp = round_away * new_exp_away \
        + round_close * new_exp_close \
        + tie * new_exp_tie

    new_mantissa = round_away * new_mantissa_away \
        + round_close * new_mantissa_close

    # if new_exp > 3:
    #     new_mantissa = 1
    new_mantissa = new_mantissa + (new_exp >
                                   (2 + half_exp_bias)) * (new_mantissa == 0)

    # Clamp the exponent to acceptable values.
    new_exp = (new_exp >= (half_exp_bias - 2)) * torch.clamp(
        new_exp, half_exp_bias - 2, half_exp_bias + 2)

    sign = sign.to(torch.int32)
    new_mantissa = new_mantissa.to(torch.int32)

    qdq_val = (sign << 15) + (new_exp << half_mantissa_bits) + (
        new_mantissa << (half_mantissa_bits - 1))

    assert qdq_val.max() <= 65535
    assert qdq_val.min() >= 0
    assert qdq_val.dtype == torch.int32
    qdq_val = qdq_val.to(torch.uint16)

    result = qdq_val.view(float_type)
    return result


def qdq_mxfp4_torch(x: torch.Tensor,
                    scale_calculation_mode: str = "even") -> torch.Tensor:
    half_dtype = x.dtype

    if half_dtype == torch.float16:
        half_mantissa_bits = FLOAT16_MANTISSA_BITS
        half_exp_bits = FLOAT16_EXP_BITS
        half_exp_bias = FLOAT16_EXP_BIAS
        val_to_add = FLOAT16_VAL_TO_ADD
        sign_exponent_mask = FLOAT16_SIGN_EXPONENT_MASK
    elif half_dtype == torch.bfloat16:
        half_mantissa_bits = BFLOAT16_MANTISSA_BITS
        half_exp_bits = BFLOAT16_EXP_BITS
        half_exp_bias = BFLOAT16_EXP_BIAS
        val_to_add = BFLOAT16_VAL_TO_ADD
        sign_exponent_mask = BFLOAT16_SIGN_EXPONENT_MASK
    else:
        raise ValueError("not implemented")

    x = x.reshape(*x.shape[:-1], -1, 32)

    block_max = torch.max(torch.abs(x), dim=-1).values

    block_max = block_max.view(torch.uint16).to(torch.int32)

    block_max_uint = torch.bitwise_and(block_max + val_to_add,
                                       sign_exponent_mask)

    assert block_max_uint.max() <= 65535
    assert block_max_uint.min() >= 0
    assert block_max_uint.dtype == torch.int32
    block_max_uint = block_max_uint.to(torch.uint16)

    block_max = block_max_uint.view(half_dtype)

    scale_exp = FLOAT8_E8M0_MAX_EXP + torch.floor(torch.log2(block_max)).to(
        torch.int32) - 2

    scale_exp = torch.clamp(scale_exp, 0, 2 * FLOAT8_E8M0_MAX_EXP)

    scale = 2.0**(scale_exp - FLOAT8_E8M0_MAX_EXP)
    scale = scale.to(half_dtype)

    x = x / scale[..., None]

    x_fp4 = fp16_to_fp4_simulate(x,
                                 half_exp_bits=half_exp_bits,
                                 half_mantissa_bits=half_mantissa_bits,
                                 half_exp_bias=half_exp_bias)

    x_fp4 = x_fp4 * scale[..., None]
    return x_fp4.reshape(*x_fp4.shape[:-2], -1)
