# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_triton_kernels
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer

logger = init_logger(__name__)

# MXFP4 constants
MXFP4_BLOCK_SIZE = 32
MXFP4_VALUE_DTYPE = torch.uint8
MXFP4_SCALE_DTYPE = torch.uint8

# CK's pre-compiled MXFP4 MoE GEMM kernel instances require the
# intermediate_size (after TP split) to be a multiple of this value.
# This arises from FP4 packing (2 values per byte) combined with CK
# tile size constraints. When violated, AITER raises:
# "device_gemm ... does not support this GEMM problem".
CK_MXFP4_MOE_DIM_ALIGNMENT = 256

_BFLOAT16_EXP_BIAS = 127
_BFLOAT16_MANTISSA_BITS = 7
_BFLOAT16_EXP_BITS = 8

_FLOAT16_EXP_BIAS = 15
_FLOAT16_MANTISSA_BITS = 10
_FLOAT16_EXP_BITS = 5

_FLOAT8_E8M0_MAX_EXP = 127
_FLOAT4_EXP_BIAS = 1
_FLOAT4_MANTISSA_BITS = 1

_FLOAT16_VAL_TO_ADD = 1 << (_FLOAT16_MANTISSA_BITS - _FLOAT4_MANTISSA_BITS - 1)
_FLOAT16_SIGN_EXPONENT_MASK = (
    (1 << (_FLOAT16_EXP_BITS + 1)) - 1
) << _FLOAT16_MANTISSA_BITS

_BFLOAT16_VAL_TO_ADD = 1 << (_BFLOAT16_MANTISSA_BITS - _FLOAT4_MANTISSA_BITS - 1)
_BFLOAT16_SIGN_EXPONENT_MASK = (
    (1 << (_BFLOAT16_EXP_BITS + 1)) - 1
) << _BFLOAT16_MANTISSA_BITS


def _swizzle_mxfp4(quant_tensor, scale, num_warps=8):
    """weight swizzle for mxfp4 moe, used for OAI mxfp4 kernel"""
    assert has_triton_kernels()
    import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout
    from triton_kernels.tensor_details.layout import StridedLayout

    value_layout_opts: dict[str, Any] = {}
    scale_layout_opts: dict[str, Any] = {}

    if (
        current_platform.is_cuda()
        and current_platform.is_device_capability(90)
        and not is_torch_equal_or_newer("2.8.1")
    ):
        logger.warning_once(
            "Mxfp4 on hopper is running on torch < 2.8.1, "
            "this cause swizling to be disabled, which may "
            "cause performance degradation. Please upgrade to torch nightly"
        )
        value_layout = StridedLayout
        scale_layout = StridedLayout
    elif current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950

        value_layout = StridedLayout
        if on_gfx950():
            try:
                # triton < 3.6
                from triton_kernels.tensor_details.layout import GFX950MXScaleLayout

                scale_layout = GFX950MXScaleLayout
            except ImportError:
                # triton >= 3.6
                from triton_kernels.tensor_details.layout import CDNA4MXScaleLayout

                scale_layout = CDNA4MXScaleLayout
        else:
            scale_layout = StridedLayout
    else:
        value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
            mx_axis=1
        )
        scale_layout, scale_layout_opts = (
            layout.make_default_matmul_mxfp4_w_scale_layout(
                mx_axis=1, num_warps=num_warps
            )
        )
    if current_platform.is_cuda():
        if current_platform.is_device_capability(90):
            constraints = {
                "split_k": 1,
            }
            opt_flags.update_opt_flags_constraints(constraints)
        elif current_platform.is_device_capability_family(100):
            constraints = {
                "is_persistent": True,
                "epilogue_subtile": 1,
            }
            opt_flags.update_opt_flags_constraints(constraints)
    # transpose the tensor so that the quantization axis is on dim1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout, **value_layout_opts
    )
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout, **scale_layout_opts)
    return quant_tensor, InFlexData(), scale


def _e8m0_to_half(scale, half_dtype: torch.dtype):
    assert scale.dtype == torch.uint8

    scale_exp = scale.to(torch.int16) - 127

    # This can be implemented with bitwise operations in a proper kernel.
    scale_half = 2.0 ** (scale_exp.to(torch.float))

    return scale_half.to(half_dtype)


def _upcast_fp4_to_fp16_or_bf16(
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
    new_exp = exp - _FLOAT4_EXP_BIAS + half_exp_bias

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
        half_exp_bias = _FLOAT16_EXP_BIAS
        half_mantissa_bits = _FLOAT16_MANTISSA_BITS
    elif float_dtype == torch.bfloat16:
        half_exp_bias = _BFLOAT16_EXP_BIAS
        half_mantissa_bits = _BFLOAT16_MANTISSA_BITS

    scale_half = _e8m0_to_half(scale, half_dtype=float_dtype)

    x_half = _upcast_fp4_to_fp16_or_bf16(
        x,
        float_dtype=float_dtype,
        half_exp_bias=half_exp_bias,
        half_mantissa_bits=half_mantissa_bits,
    )

    x_half = x_half.reshape(*x_half.shape[:-1], -1, 32)
    x_half = x_half * scale_half[..., None]
    x_half = x_half.reshape(*x_half.shape[:-2], -1)

    return x_half


def _fp16_to_fp4_simulate(
    val, half_mantissa_bits: int, half_exp_bits: int, half_exp_bias: int
):
    # Casts an fp16/bf16 input to the restricted values of float4_e2m1,
    # that is to say [0., 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0,
    # -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0].

    float_type = val.dtype

    # "rshift_cuda" not implemented for 'UInt16'
    val_view = val.view(torch.int16)  # .to(torch.int32)

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

    round_close = tail < half  # round towards 0
    round_away = tail > half  # round away from 0
    tie = tail == half

    new_mantissa_close = torch.zeros(val.shape, device=val.device, dtype=torch.bool)
    new_exp_close = torch.zeros(val.shape, device=val.device, dtype=torch.uint16)

    new_mantissa_away = torch.zeros(val.shape, device=val.device, dtype=torch.bool)
    new_exp_away = torch.zeros(val.shape, device=val.device, dtype=torch.uint16)

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
    new_exp = (
        round_away * new_exp_away + round_close * new_exp_close + tie * new_exp_tie
    )

    new_mantissa = round_away * new_mantissa_away + round_close * new_mantissa_close

    # if new_exp > 3:
    #     new_mantissa = 1
    new_mantissa = new_mantissa + (new_exp > (2 + half_exp_bias)) * (new_mantissa == 0)

    # Clamp the exponent to acceptable values.
    new_exp = (new_exp >= (half_exp_bias - 2)) * torch.clamp(
        new_exp, half_exp_bias - 2, half_exp_bias + 2
    )

    sign = sign.to(torch.int32)
    new_mantissa = new_mantissa.to(torch.int32)

    qdq_val = (
        (sign << 15)
        + (new_exp << half_mantissa_bits)
        + (new_mantissa << (half_mantissa_bits - 1))
    )

    assert qdq_val.max() <= 65535
    assert qdq_val.min() >= 0
    assert qdq_val.dtype == torch.int32
    qdq_val = qdq_val.to(torch.uint16)

    result = qdq_val.view(float_type)
    return result


def qdq_mxfp4_torch(
    x: torch.Tensor, scale_calculation_mode: str = "even"
) -> torch.Tensor:
    half_dtype = x.dtype

    if half_dtype == torch.float16:
        half_mantissa_bits = _FLOAT16_MANTISSA_BITS
        half_exp_bits = _FLOAT16_EXP_BITS
        half_exp_bias = _FLOAT16_EXP_BIAS
        val_to_add = _FLOAT16_VAL_TO_ADD
        sign_exponent_mask = _FLOAT16_SIGN_EXPONENT_MASK
    elif half_dtype == torch.bfloat16:
        half_mantissa_bits = _BFLOAT16_MANTISSA_BITS
        half_exp_bits = _BFLOAT16_EXP_BITS
        half_exp_bias = _BFLOAT16_EXP_BIAS
        val_to_add = _BFLOAT16_VAL_TO_ADD
        sign_exponent_mask = _BFLOAT16_SIGN_EXPONENT_MASK
    else:
        raise ValueError("not implemented")

    x = x.reshape(*x.shape[:-1], -1, 32)

    block_max = torch.max(torch.abs(x), dim=-1).values

    block_max = block_max.view(torch.uint16).to(torch.int32)

    block_max_uint = torch.bitwise_and(block_max + val_to_add, sign_exponent_mask)

    assert block_max_uint.max() <= 65535
    assert block_max_uint.min() >= 0
    assert block_max_uint.dtype == torch.int32
    block_max_uint = block_max_uint.to(torch.uint16)

    block_max = block_max_uint.view(half_dtype)

    scale_exp = (
        _FLOAT8_E8M0_MAX_EXP + torch.floor(torch.log2(block_max)).to(torch.int32) - 2
    )

    scale_exp = torch.clamp(scale_exp, 0, 2 * _FLOAT8_E8M0_MAX_EXP)

    scale = 2.0 ** (scale_exp - _FLOAT8_E8M0_MAX_EXP)
    scale = scale.to(half_dtype)

    x = x / scale[..., None]

    x_fp4 = _fp16_to_fp4_simulate(
        x,
        half_exp_bits=half_exp_bits,
        half_mantissa_bits=half_mantissa_bits,
        half_exp_bias=half_exp_bias,
    )

    x_fp4 = x_fp4 * scale[..., None]
    return x_fp4.reshape(*x_fp4.shape[:-2], -1)


def _dequant_mxfp4(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype
) -> torch.Tensor:
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError(
            "The package `amd-quark` is required to use "
            "MX-FP4 models. Please install it with `pip install "
            "amd-quark`."
        ) from err

    return mx.dq_mxfp4(x, scale, float_dtype)


def _dequant_mxfp4_fake(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty(
        (*x.shape[:-1], x.shape[-1] * 2), dtype=float_dtype, device=x.device
    )


def _quant_dequant_mxfp4(
    x: torch.Tensor, scale_calculation_mode: str = "even"
) -> torch.Tensor:
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError(
            "The package `amd-quark` is required to use "
            "MX-FP4 models. Please install it with `pip install "
            "amd-quark`."
        ) from err

    return mx.qdq_mxfp4(x, scale_calculation_mode)


def _quant_dequant_mxfp4_fake(
    x: torch.Tensor, scale_calculation_mode: str = "even"
) -> torch.Tensor:
    return torch.empty_like(x)


# Protect these operations into a torch custom op to avoid errors as
# torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
# Explanation: Dynamo does not know how to trace the builtin
# `kernel_ext.PyCapsule.dq_uint8_mxfp4_to_half.` This function is either a
# Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python
# extension (perhaps created with pybind).
# TODO: Make sure there is no way to avoid having these functions
# marked as skipped by dynamo.
try:
    direct_register_custom_op(
        op_name="dequant_mxfp4",
        op_func=_dequant_mxfp4,
        fake_impl=_dequant_mxfp4_fake,
    )
    dequant_mxfp4 = torch.ops.vllm.dequant_mxfp4
except AttributeError as error:
    raise error

try:
    direct_register_custom_op(
        op_name="quant_dequant_mxfp4",
        op_func=_quant_dequant_mxfp4,
        fake_impl=_quant_dequant_mxfp4_fake,
    )
    quant_dequant_mxfp4 = torch.ops.vllm.quant_dequant_mxfp4
except AttributeError as error:
    raise error


def xpu_mxfp4_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.xpu_mxfp4_quantize(x)
