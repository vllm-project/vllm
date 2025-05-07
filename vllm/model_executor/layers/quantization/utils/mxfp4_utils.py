# SPDX-License-Identifier: Apache-2.0
from typing import Tuple

import triton
import triton.language as tl

import torch
from enum import Enum

import vllm.envs as envs

OCP_MX_BLOCK_SIZE = 32
SUPPORTED_IMPLEMS = {"hip", "torch", "triton"}

def per_token_group_dequant_mxfp4_hip(x: torch.Tensor, scale: torch.Tensor,
                                  block_k: int,
                                  float_dtype: torch.dtype) -> torch.Tensor:
    try:
        from quark.torch.kernel.hw_emulation.extensions import kernel_ext
    except ImportError as err:
        raise ImportError("The package `amd-quark` is required to use "
                        "MX-FP4 models. Please install it with `pip install "
                        "amd-quark`.") from err

    dequant_weight_shape = (*x.shape[:-1], x.shape[-1] * 2)

    dq_w = torch.empty(dequant_weight_shape, device=x.device, dtype=float_dtype)
    kernel_ext.dq_uint8_mxfp4_to_half(x, scale, dq_w, OCP_MX_BLOCK_SIZE)

    return dq_w


def per_token_group_dequant_mxfp4_torch(x: torch.Tensor, scale: torch.Tensor,
                                  block_k: int,
                                  float_dtype: torch.dtype) -> torch.Tensor:
    try:
        from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
            dequantize_fp4_fp6_per_group)
        from quark.torch.utils import pack
    except ImportError as e:
        raise ImportError("The package `amd-quark` is required to use "
                        "MX-FP4 models. Please install it with `pip install "
                        "amd-quark`.") from e

    # TODO: Both arguments are unused.
    pack_method = pack.Pack_fp4(None, dtype="fp4")
    # TODO: Both 'reorder' and 'origin_packed_axis_size' are unused.
    unpacked_x = pack_method.unpack(x, reorder=False)

    scale = 2**(scale.view(torch.uint8).to(torch.int16) - 127).to(float_dtype)

    # TODO: `dequantize_fp4_fp6_per_group` and `prepare_inputs_per_group` always return fp32.
    return dequantize_fp4_fp6_per_group(unpacked_x,
                                        scale,
                                        axis=-1,
                                        group_size=block_k,
                                        quant_dtype="fp4").to(float_dtype)

def per_token_group_quant_mxfp4_hip(x: torch.Tensor,
                                block_k: int,
                                scale_calculation_mode: str = "even"
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        from quark.torch.kernel.hw_emulation.extensions import kernel_ext
    except ImportError as err:
        raise ImportError("The package `amd-quark` is required to use "
                        "MX-FP4 models. Please install it with `pip install "
                        "amd-quark`.") from err

    x = kernel_ext.qdq_mxfp4(x, OCP_MX_BLOCK_SIZE)

    return x

def per_token_group_quant_mxfp4_torch(x: torch.Tensor,
                                block_k: int,
                                scale_calculation_mode: str = "even"
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
            fake_quantize_fp4_fp6_per_group_with_scale)
        from quark.torch.quantization.utils import (even_round,
                                                    reshape_to_blocks)
    except ImportError as err:
        raise ImportError("The package `amd-quark` is required to use "
                        "MX-FP4 models. Please install it with `pip install "
                        "amd-quark`.") from err

    axis = -1
    block_x = reshape_to_blocks(x, block_k, axis)
    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    amax = amax.squeeze(-1)

    # TODO: there are other rounding strategies supported in quark and in the
    # config.json that we do not check for here!
    if scale_calculation_mode != "even":
        raise NotImplementedError(
            f"Scale calculation mode {scale_calculation_mode} is not yet "
            "supported in MX-FP4 quantization")
    scale = even_round(amax, "fp4")

    # Apply dequantize(quantize(x)).
    x = fake_quantize_fp4_fp6_per_group_with_scale(
        x,
        scale.to(x.device),
        axis=axis,
        group_size=block_k,
        quant_dtype="fp4",
    )

    return x

def get_max_quant_val(dtype: torch.dtype):
    d = {torch.uint8: 6.0, torch.float8_e5m2: 57344.0, torch.float8_e4m3fn: 448.0}
    assert dtype in d
    return d[dtype]


@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 6.0
    elif dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")


@triton.jit
def _get_max_quant_exp(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 2
    else:
        tl.static_assert(False, f"Invalid {dtype=}")


# fmt: off
@triton.jit
def _compute_quant_and_scale(src_tensor, valid_src_mask, mx_tensor_dtype: tl.constexpr,
                             DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr = 0):
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 32

    # Explicit cast to fp32 since most ops are not supported on bfloat16. We avoid needless conversions to and from bf16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)  # Don't consider padding tensors in scale computation
    abs_tensor = tl.reshape(abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32])
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    if DEQUANT_SCALE_ROUNDING_MODE == 0:
        # DequantScaleRoundingMode.ROUND_UP
        # compute 2 ** ceil(log2(dequant_scale))
        # Adding 0x007FFFFF adds exponent by 1 unless mantissa is all zeros
        # A corner case: exponent is 0xFF that will overflow but that's already
        # NaN so assume we don't care.
        dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
        dequant_scale_exponent = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    elif DEQUANT_SCALE_ROUNDING_MODE == 1:
        # DequantScaleRoundingMode.ROUND_DOWN
        # compute 2 ** floor(log2(dequant_scale))
        dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000
    else:
        # DequantScaleRoundingMode.EVEN
        # compute 2 ** (floor(log2(rounding(max_abs(v)))-max_exp))
        assert DEQUANT_SCALE_ROUNDING_MODE == 2
        # eps =  tl.where(max_val == 0.0, 2**(-126), 0.0)
        max_val = max_val.to(tl.int32, bitcast=True)
        max_val = (max_val + 0x200000).to(tl.uint32, bitcast=True) & 0x7F800000
        max_val = max_val.to(tl.float32, bitcast=True)
        # scale_e8m0_unbiased = tl.log2(max_val + eps).floor() - _get_max_quant_exp(mx_tensor_dtype)
        scale_e8m0_unbiased = tl.log2(max_val).floor() - _get_max_quant_exp(mx_tensor_dtype)  # no eps
        scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
        dequant_scale_rounded = tl.exp2(scale_e8m0_unbiased)
        dequant_scale_exponent = dequant_scale_rounded.to(tl.uint32, bitcast=True)

    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32])
    quant_tensor = f32_tensor * quant_scale

    # Reshape the tensors after scaling
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set the invalid portions of the tensor to 0. This will ensure that any padding tensors are 0 in the mx format.
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])

    # First, we simply extract the exponent part of the scales and store the result
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    # Now we must convert the tensors to the mx format.
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    else:
        quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
        signs = quant_tensor & 0x80000000
        exponents = (quant_tensor >> 23) & 0xFF
        mantissas = (quant_tensor & 0x7FFFFF)

        # 0.25 <= x < 0.75 maps to 0.5, a denormal number
        E8_BIAS = 127
        E2_BIAS = 1
        # Move implicit bit 1 at the beginning to mantissa for denormals
        adjusted_exponents = tl.core.sub(E8_BIAS, exponents + 1, sanitize_overflow=False)
        mantissas = tl.where(exponents < E8_BIAS, (0x400000 | (mantissas >> 1)) >> adjusted_exponents, mantissas)

        # For normal numbers, we change the bias from 127 to 1, and for subnormals, we keep exponent as 0.
        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
        e2m1_tmp = tl.minimum((((exponents << 2) | (mantissas >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale_exponent

@triton.jit
def _downcast_to_mxfp(mx_tensor_ptr, stride_mxt_outer, stride_mxt_quant: tl.constexpr,
                      mx_scale_ptr, stride_mx_scale_outer, stride_mx_scale_quant,
                      src_ptr, stride_src_outer, stride_src_quant,
                      outer_dim, quant_dim,
                      BLOCK_SIZE_OUT_DIM: tl.constexpr, BLOCK_SIZE_QUANT_DIM: tl.constexpr,
                      DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr):

    tl.static_assert(stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1.")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % 32 == 0, f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32")

    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(mx_tensor_dtype == tl.uint8 or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
                     f"Invalid {mx_tensor_dtype=}. Must be uint8 or float8.")

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, f"{mx_scale_ptr.dtype.element_ty=} must be uint8")
    tl.static_assert((src_dtype == tl.bfloat16) or (src_dtype == tl.float16), f"{src_dtype=} must be bfloat16 or float16")
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    K_DIVISOR: tl.constexpr = 1 if is_fp8 else 2
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant and mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_mxt = mask_mxt_quant and mask_n

    scale_mask_k = start_mx_scale_quant + offs_scale_quant < tl.cdiv(quant_dim, 32)
    full_scale_mask = scale_mask_k and mask_n

    src_tensor_offsets = offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    mx_scale_offsets = offs_scale_quant * stride_mx_scale_quant + offs_outer * stride_mx_scale_outer
    mx_tensor_offsets = offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    out_tensor, scale_tensor = _compute_quant_and_scale(src_tensor, full_mask_src, mx_tensor_dtype,
                                                        DEQUANT_SCALE_ROUNDING_MODE)

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


@triton.jit
def _upcast_from_mxfp(out_ptr, stride_o_outer, stride_o_quant: tl.constexpr,
                      mx_scale_ptr, stride_scale_outer, stride_scale_quant,
                      mx_tensor_ptr, stride_tensor_outer, stride_tensor_quant: tl.constexpr,
                      outer_dim, quant_dim,
                      BLOCK_SIZE_OUT_DIM: tl.constexpr, BLOCK_SIZE_QUANT_DIM: tl.constexpr):

    tl.static_assert(stride_o_quant == 1, "the weight must be contiguous in the k dimension for mx")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % 32 == 0, "BLOCK_SIZE_K must be a multiple of 32")
    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    dst_dtype: tl.constexpr = out_ptr.dtype.element_ty
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16)
    tl.static_assert(mx_tensor_dtype == tl.uint8 or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
                     "mx_tensor_ptr must be uint8")
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")

    # Determine if we are dealing with fp8 types.
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    K_DIVISOR: tl.constexpr = 1 if is_fp8 else 2
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    # Compute starting indices for the quantized (packed) dimension and the outer dimension.
    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    start_mxt_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    mx_tensor_ptr += start_mxt_quant * stride_tensor_quant + start_out * stride_tensor_outer
    mx_scale_ptr += start_mx_scale_quant * stride_scale_quant + start_out * stride_scale_outer
    out_ptr += start_out * stride_o_outer + start_out_quant * stride_o_quant

    # Compute offsets and masks.
    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_out_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)
    offs_scale = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)

    mask_outer = start_out + offs_outer < outer_dim
    mask_out_quant = start_out_quant + offs_out_quant < quant_dim
    full_mask_out = mask_out_quant and mask_outer

    mask_src_quant = start_mxt_quant + offs_src_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_src = mask_src_quant and mask_outer

    mask_scale = start_mx_scale_quant + offs_scale < tl.cdiv(quant_dim, 32)
    full_scale_mask = mask_scale and mask_outer

    tensor_offsets = offs_src_quant * stride_tensor_quant + offs_outer * stride_tensor_outer
    scale_offsets = offs_scale * stride_scale_quant + offs_outer * stride_scale_outer
    out_offsets = offs_out_quant * stride_o_quant + offs_outer * stride_o_outer

    # Load the packed tensor and scale.
    tensor = tl.load(mx_tensor_ptr + tensor_offsets, mask=full_mask_src)
    scale = tl.load(mx_scale_ptr + scale_offsets, mask=full_scale_mask)

    # Upcast the scale to the destination type.
    if dst_dtype == tl.bfloat16:
        dst_scale = (scale.to(tl.uint16) << 7).to(dst_dtype, bitcast=True)
    else:
        tl.static_assert(dst_dtype == tl.float16)
        dst_scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        dst_scale = dst_scale.to(tl.float16)

    # Now upcast the tensor.
    if is_fp8:
        dst_tensor = tensor.to(dst_dtype)
        if tensor.dtype == tl.float8e5:
            from_e_bits: tl.constexpr = 5
            from_m_bits: tl.constexpr = 2
            to_e_bits: tl.constexpr = 8 if dst_dtype == tl.bfloat16 else 5
            to_m_bits: tl.constexpr = 7 if dst_dtype == tl.bfloat16 else 10

            # Preserve infs and nans. FIXME Fp8E5M2_to_Bf16 doesn't preserve them!
            non_finite_mask_src: tl.constexpr = ((1 << from_e_bits) - 1) << from_m_bits
            non_finite_mask_dst: tl.constexpr = ((1 << to_e_bits) - 1) << to_m_bits
            dst_tensor = tl.where(
                (tensor.to(tl.uint8, bitcast=True) & non_finite_mask_src) == non_finite_mask_src,
                (dst_tensor.to(tl.uint16, bitcast=True) | non_finite_mask_dst).to(dst_dtype, bitcast=True),
                dst_tensor,
            )
    else:
        dst_bias: tl.constexpr = 127 if dst_dtype == tl.bfloat16 else 15
        dst_0p5: tl.constexpr = 16128 if dst_dtype == tl.bfloat16 else 0x3800
        dst_m_bits: tl.constexpr = 7 if dst_dtype == tl.bfloat16 else 10
        # e2m1
        em0 = tensor & 0x07
        em1 = tensor & 0x70
        x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | ((tensor & 0x08).to(tl.uint16) << 12)
        x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | ((tensor & 0x80).to(tl.uint16) << 8)
        # Three cases:
        # 1) x is normal and non-zero: Correct bias
        x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
        x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
        # 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in the dst type
        x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
        # 3) x is zero, do nothing
        dst_tensor = tl.interleave(x0, x1).to(dst_dtype, bitcast=True)

    # Reshape for proper broadcasting: the scale was stored with a 32‐sized “inner” grouping.
    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    scale = scale.reshape(dst_scale.shape)

    out_tensor = dst_tensor * dst_scale
    # Correct any NaNs encoded via the scale.
    out_tensor = tl.where(scale == 0xFF, float("nan"), out_tensor)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    tl.store(out_ptr + out_offsets, out_tensor, mask=full_mask_out)


class DequantScaleRoundingMode(Enum):
    ROUND_UP = 0
    ROUND_DOWN = 1
    EVEN = 2


SWIZZLE_ALIGN_INNER = 8
SWIZZLE_SIZE_INNER = 4
SWIZZLE_SIZE_OUTER = 128

@triton.jit
def _unswizzle_mx_block(x,
                        SIZE_OUTER: tl.constexpr = SWIZZLE_SIZE_OUTER,
                        SIZE_INNER: tl.constexpr = SWIZZLE_SIZE_INNER,
                        ALIGN_INNER: tl.constexpr = SWIZZLE_ALIGN_INNER):
    shape_0: tl.constexpr = x.shape[0]
    shape_1: tl.constexpr = x.shape[1]
    tl.static_assert(shape_1 % SIZE_OUTER == 0)
    tl.static_assert(shape_1 // SIZE_OUTER <= ALIGN_INNER)
    x = x.reshape(shape_0, (shape_1 // SIZE_OUTER) // SIZE_INNER, 32, SIZE_OUTER // 32, SIZE_INNER)
    x = x.trans(0, 3, 2, 1, 4).reshape(shape_0 * SIZE_OUTER, shape_1 // SIZE_OUTER)
    return x


def axis_permute_order(ndim: int, axis: int, swizzle_axis: int | None = None) -> list[int]:
    permute_order = list(range(ndim))
    permute_order[axis], permute_order[-1] = permute_order[-1], permute_order[axis]

    scale_permute_order = permute_order.copy()
    if swizzle_axis is not None:
        axis = axis if axis >= 0 else axis + ndim
        swizzle_axis = swizzle_axis if swizzle_axis >= 0 else swizzle_axis + ndim
        if swizzle_axis == ndim - 1:
            swizzle_axis = axis
        scale_permute_order[swizzle_axis], scale_permute_order[-2] = scale_permute_order[-2], scale_permute_order[swizzle_axis]

    convert_order = [i for i, (a, b) in enumerate(zip(permute_order, scale_permute_order)) if a != b]
    assert len(convert_order) == 0 or len(convert_order) == 2, "Exactly 0 or 1 swap should be required to transform permute_order to scale_permute_order."
    return permute_order, scale_permute_order, convert_order


def transpose_shape(shape: tuple[int, ...], i: int, j: int) -> tuple[int, ...]:
    shape = list(shape)
    shape[i], shape[j] = shape[j], shape[i]
    return tuple(shape)


def permute_shape(shape: tuple[int, ...], permute_order: list[int]) -> tuple[int, ...]:
    return tuple(shape[i] for i in permute_order)


def downcast_to_mxfp(src_tensor: torch.Tensor, out_quant_type: torch.dtype, axis: int, swizzle_axis: int | None = None,
                     out_quant_tensor: torch.Tensor | None = None, out_scale: torch.Tensor | None = None,
                     DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode=DequantScaleRoundingMode.ROUND_UP,
                     BLOCK_OUT_DIM: int = 128, BLOCK_QUANT_DIM: int = 32):
    """
         Convert the src weights to mx format. The src weight is quantized along the axis dimension.

         If weight_quant_type is torch.uint8, we output mxfp4 where two e2m1 values are packed into a single byte.
         Note that this means the k_dim of the tensor will be half of the logical k_dim.

         If weight_quant_type is torch.float8_e4m3fn or torch.float8_e5m2, we output mxfp8 with the float8s are stored
         in their respective formats.

         When swizzle_axis is provided, the downcast will quantize along the quantization axis and swizzle these values
         with the swizzle_axis from layout (A, B, ..., N, K) to (A, B, ..., N // 128, K // 4, 32, 4, 4), where N is the
         swizzle dimension and K is the quantization dimension. The swizzled scales are then reinterpreted back as
         (A, B, ..., N, K), contiguous along K, and permuted back to the original input layout.
         In order to swizzle in the target layout, the scales are padded to be divisible by 128 and 4 along the
         swizzle and quantization dimensions, respectively.
    """
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    if swizzle_axis is not None:
        assert -ndim <= swizzle_axis < ndim, f"Invalid swizzle axis {swizzle_axis=}"
        swizzle_axis = swizzle_axis if swizzle_axis >= 0 else swizzle_axis + ndim

    L = src_tensor.shape[axis]
    if out_quant_type == torch.uint8:
        # We make this assertion since we can't track if the "real" shape was odd, and we padded it to be even.
        # We want to maintain the property dequant(quant(x)).shape == x.shape
        assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"

    is_fp8 = out_quant_type == torch.float8_e4m3fn or out_quant_type == torch.float8_e5m2
    divisor = 1 if is_fp8 else 2
    device = src_tensor.device

    packed_quant_dim = triton.cdiv(L, divisor)
    out_scale_dim = triton.cdiv(L, 32)

    permute_order, scale_permute_order, convert_order = axis_permute_order(ndim, axis, swizzle_axis)

    prmted_quant_tensor_shape = permute_shape(src_tensor.shape, permute_order)[:-1] + (packed_quant_dim,)
    prmted_scale_shape = permute_shape(src_tensor.shape, scale_permute_order)[:-1] + (out_scale_dim,)
    prmted_src_tensor = src_tensor.permute(permute_order)

    if out_quant_tensor is None:
        out_quant_tensor = torch.empty(prmted_quant_tensor_shape, dtype=out_quant_type, device=device)
    else:
        expected_shape = src_tensor.shape[:axis] + (packed_quant_dim,) + src_tensor.shape[axis + 1:]
        assert out_quant_tensor.shape == expected_shape, f"{out_quant_tensor.shape=} != {expected_shape=}"
        assert out_quant_tensor.dtype == out_quant_type, f"{out_quant_tensor.dtype=} != {out_quant_type=}"
        assert out_quant_tensor.stride(axis) == 1, f"{out_quant_tensor.stride(axis)=} != 1"
        # We expect the axis dimension to be last, so permute the tensor
        out_quant_tensor = out_quant_tensor.permute(permute_order)

    if out_scale is None:
        allocation_shape = prmted_scale_shape
        if swizzle_axis is not None:
            allocation_shape = list(prmted_scale_shape)
            allocation_shape[-1] = triton.cdiv(allocation_shape[-1], SWIZZLE_ALIGN_INNER) * SWIZZLE_ALIGN_INNER
            allocation_shape[-2] = triton.cdiv(allocation_shape[-2], SWIZZLE_SIZE_OUTER) * SWIZZLE_SIZE_OUTER
        out_scale = torch.empty(allocation_shape, dtype=torch.uint8, device=device)
    else:
        if swizzle_axis is not None:
            expected_scale_shape = list(prmted_scale_shape)
            # Pad then unpermute the expected shape
            expected_scale_shape[-1] = triton.cdiv(expected_scale_shape[-1], SWIZZLE_ALIGN_INNER) * SWIZZLE_ALIGN_INNER
            expected_scale_shape[-2] = triton.cdiv(expected_scale_shape[-2], SWIZZLE_SIZE_OUTER) * SWIZZLE_SIZE_OUTER
            expected_scale_shape = permute_shape(expected_scale_shape, scale_permute_order)
        else:
            expected_scale_shape = permute_shape(prmted_scale_shape, scale_permute_order)

        assert out_scale.shape == expected_scale_shape, f"{out_scale.shape=} {expected_scale_shape=}"
        assert out_scale.dtype == torch.uint8, f"{out_scale.dtype=} != torch.uint8"
        out_scale = out_scale.permute(scale_permute_order)

    if convert_order or prmted_scale_shape != out_scale.shape:
        # Output shape is padded. Make a new unpadded tensor.
        assert swizzle_axis is not None  # padding only occurs in the swizzled case.
        # scales should be produced in `permute_order`.
        unpadded_out_scale = torch.empty(transpose_shape(prmted_scale_shape, *convert_order) if convert_order else prmted_scale_shape, dtype=torch.uint8, device=device)
    else:
        unpadded_out_scale = out_scale

    # Flatten input tensor for kernel. This will typically make a copy
    reshaped_src_tensor = prmted_src_tensor.reshape(-1, L)
    blocks_quant_dim = triton.cdiv(reshaped_src_tensor.shape[-1], BLOCK_QUANT_DIM)
    blocks_out_dim = triton.cdiv(reshaped_src_tensor.shape[0], BLOCK_OUT_DIM)

    # Flatten the output tensors for the kernel, this should be a view always
    kernel_quant_tensor = out_quant_tensor.reshape(-1, packed_quant_dim)
    kernel_scale = unpadded_out_scale.reshape(-1, out_scale_dim)
    assert kernel_quant_tensor.data_ptr() == out_quant_tensor.data_ptr()
    assert kernel_scale.data_ptr() == unpadded_out_scale.data_ptr()

    _downcast_to_mxfp[(blocks_out_dim, blocks_quant_dim)](
        kernel_quant_tensor, *kernel_quant_tensor.stride(),
        kernel_scale, *kernel_scale.stride(),
        reshaped_src_tensor, *reshaped_src_tensor.stride(),
        *reshaped_src_tensor.shape,
        BLOCK_OUT_DIM, BLOCK_QUANT_DIM, DEQUANT_SCALE_ROUNDING_MODE.value,
        num_warps=8
    )

    if convert_order or prmted_scale_shape != out_scale.shape:
        if convert_order:
            # convert scales from `permute_order` to `scale_permute_order`
            unpadded_out_scale = unpadded_out_scale.transpose(*convert_order)
        # Copy from the unpadded shape into the padded one.
        out_scale[tuple(slice(0, size) for size in unpadded_out_scale.shape)] = unpadded_out_scale

        # Zero out any padding. `tcgen05.mma` yields MAX_FINITE for the entire block if any
        # scales are not finite (0xFF).
        slices = [slice(None) for _ in unpadded_out_scale.shape]
        for i, size in enumerate(unpadded_out_scale.shape):
            slices[i] = slice(size, None)
            out_scale[slices] = 0
            slices[i] = slice(None)

    out_quant_tensor = out_quant_tensor.permute(permute_order)

    if swizzle_axis is not None:
        out_scale = swizzle_mx(out_scale, allow_pad=False).contiguous().permute(scale_permute_order)
    else:
        out_scale = out_scale.permute(permute_order).contiguous()
    return out_quant_tensor, out_scale, permute_shape(prmted_scale_shape, scale_permute_order)


def upcast_from_mxfp(tensor: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, axis: int, swizzle_axis: int | None = None,
                     BLOCK_OUT_DIM: int = 128, BLOCK_QUANT_DIM: int = 32):
    """
    Upcasts an mxfp (packed) weight tensor back to float16 or bfloat16.

    The function assumes that the tensors were quantized along the given axis.
    It permutes the tensor so that the quantized axis is last, reshapes to 2D,
    launches the Triton upcast kernel, and then unpermutes back to the original order.
    """
    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    if swizzle_axis is not None:
        assert -ndim <= swizzle_axis < ndim, f"Invalid swizzle axis {swizzle_axis=}"
        swizzle_axis = swizzle_axis if swizzle_axis >= 0 else swizzle_axis + ndim

    multiplier = 1 if "float8" in str(tensor.dtype) else 2
    logical_quant_dim_shape = tensor.shape[axis] * multiplier
    assert tensor.ndim == scale.ndim, (f"Weight and scale must have the same number of dimensions. "
                                       f"Got {tensor.ndim=} and {scale.ndim=}")
    quant_dim_align = SWIZZLE_ALIGN_INNER if swizzle_axis is not None else 1
    assert triton.cdiv(triton.cdiv(logical_quant_dim_shape, 32), quant_dim_align) * quant_dim_align == scale.shape[axis], \
        f"Tensor and scale mismatch along quantization axis. Got {tensor.shape[axis]=} and {scale.shape[axis]=}"
    assert tensor.dtype in {torch.uint8, torch.float8_e5m2, torch.float8_e4m3fn}, \
        f"Invalid tensor dtype {tensor.dtype=}"
    assert scale.dtype == torch.uint8, f"Invalid scale dtype {scale.dtype=}"
    assert dtype in {torch.float16, torch.bfloat16}, f"Invalid output dtype {dtype=}"

    # Bring the quantized axis to the end.
    # For the scales, bring the swizzle axis second to last.
    permute_order, scale_permute_order, convert_order = axis_permute_order(ndim, axis, swizzle_axis)
    prmt_tensor = tensor.permute(permute_order).contiguous()
    prmt_scale = scale.permute(scale_permute_order)

    # Unswizzle the scale tensor and slice off padding.
    if swizzle_axis is not None:
        prmt_scale = unswizzle_mx(prmt_scale)

        unpadded_scale_shape = (*prmt_tensor.shape[:-1], triton.cdiv(logical_quant_dim_shape, 32))
        # The kernel expects scales in `permute_order`, not `scale_permute_order`. Transpose if needed.
        if convert_order:
            prmt_scale = prmt_scale.transpose(*convert_order)

        slices = tuple(slice(0, size) for size in unpadded_scale_shape)
        prmt_scale = prmt_scale[slices]

    prmt_scale = prmt_scale.contiguous()

    quant_dim = prmt_tensor.shape[-1]
    reshaped_tensor = prmt_tensor.reshape(-1, quant_dim)
    reshaped_scale = prmt_scale.reshape(-1, prmt_scale.shape[-1])

    outer_dim = reshaped_tensor.shape[0]
    blocks_out_dim = triton.cdiv(outer_dim, BLOCK_OUT_DIM)
    blocks_quant_dim = triton.cdiv(logical_quant_dim_shape, BLOCK_QUANT_DIM)

    out = torch.empty((outer_dim, logical_quant_dim_shape), dtype=dtype, device=tensor.device)
    _upcast_from_mxfp[(blocks_out_dim, blocks_quant_dim)](
        out, out.stride(0), out.stride(1),
        reshaped_scale, reshaped_scale.stride(0), reshaped_scale.stride(1),
        reshaped_tensor, reshaped_tensor.stride(0), reshaped_tensor.stride(1),
        outer_dim, logical_quant_dim_shape, BLOCK_OUT_DIM, BLOCK_QUANT_DIM, num_warps=8
    )
    # Reshape back to the permuted shape.
    out = out.view(*prmt_tensor.shape[:-1], logical_quant_dim_shape)
    out = out.permute(permute_order)
    return out


def right_shift_unsigned(x, shift):
    # CUDA torch does not support bit ops on uint32, so we need to mask to get unsigned right shift
    return (x >> shift) & ((1 << (32 - shift)) - 1)


def downcast_to_mxfp_torch(src_tensor: torch.Tensor, out_quant_type: torch.dtype, axis: int, swizzle_axis: int | None = None,
                           out_quant_tensor: torch.Tensor | None = None, out_scale: torch.Tensor | None = None,
                           DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_UP):
    """
    Converts the src tensor to the output format specified by out_quant_type.
      axis: The axis along which the tensors are contiguous and quantization is applied.
      DEQUANT_SCALE_ROUNDING_MODE: 0 for ROUND_UP, 1 for ROUND_DOWN.

    Returns:
      out_quant_tensor: Quantized tensor in mx format.
         • For mxfp8, the output has the same shape as src_tensor.
         • For mxfp4, the size along the axis is halved, and the tensor is returned as a torch.uint8.
      scale: Scale tensor (stored as uint8) computed per group of 32 elements along the axis.
             Its shape is the same as src_tensor except that the axis is replaced by ceil(L/32),
             where L is the original length along that axis.
    """

    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    assert src_tensor.dtype in {torch.float32, torch.bfloat16, torch.float16}, f"Invalid input tensor dtype {src_tensor.dtype}"

    axis = axis if axis >= 0 else axis + ndim
    if swizzle_axis is not None:
        assert -ndim <= swizzle_axis < ndim, f"Invalid swizzle axis {swizzle_axis=}"
        swizzle_axis = swizzle_axis if swizzle_axis >= 0 else swizzle_axis + ndim
    is_fp4 = out_quant_type == torch.uint8
    is_fp8 = "float8" in str(out_quant_type)
    assert is_fp4 or is_fp8, f"Invalid input tensor dtype {out_quant_type}"

    device = src_tensor.device

    # For mxfp4 conversion, we assume the contiguous axis length is even.
    if is_fp4:
        axis_shape = src_tensor.size(axis)
        assert axis_shape % 2 == 0, "For mxfp4 conversion the contiguous axis length must be even."

    # Permute the tensor so that the contiguous axis becomes the last dimension.
    # For the scales, make the swizzle axis is second to last.
    permute_order, scale_permute_order, convert_order = axis_permute_order(ndim, axis, swizzle_axis)
    src = src_tensor.permute(permute_order).to(torch.float32)  # now shape: (..., axis_shape)
    axis_shape = src.shape[-1]

    # Pad the axis to be divisible by 32, in case it is not.
    next_multiple = (axis_shape + 31) // 32 * 32
    pad_amount = next_multiple - axis_shape
    padded_src = F.pad(src, (0, pad_amount))
    valid_mask = F.pad(torch.ones_like(src, dtype=torch.bool), (0, pad_amount))
    padded_axis_shape = padded_src.size(-1)  # now divisible by 32

    # --- Compute per-group maximums for scale ---
    # Set padded entries to -1 so they don’t affect the max.
    abs_f = torch.abs(padded_src)
    abs_f = torch.where(valid_mask, abs_f, torch.tensor(-1.0, device=device, dtype=padded_src.dtype))
    # Reshape the last dimension into groups of 32.
    new_shape = padded_src.shape[:-1] + (padded_axis_shape // 32, 32)
    abs_groups = abs_f.view(*new_shape)
    # Compute maximum along the group dimension (of size 32).
    max_val, _ = abs_groups.max(dim=-1, keepdim=True)

    # Choose a max quantization value depending on type.
    max_quant_val = get_max_quant_val(out_quant_type)
    dequant_scale = max_val / max_quant_val  # shape: (..., padded_axis_shape//32, 1)

    # Convert to int to round the FP32 scale, prior to quantization!
    ds_int = dequant_scale.view(torch.int32)
    if DEQUANT_SCALE_ROUNDING_MODE == DequantScaleRoundingMode.ROUND_UP:
        ds_int_rounded = (ds_int + 0x007FFFFF) & 0x7F800000
    else:
        ds_int_rounded = ds_int & 0x7F800000
    # Reinterpret back as float32.
    dequant_scale_rounded = ds_int_rounded.view(torch.float32)

    # Compute the quantization scale.
    quant_scale = torch.where(dequant_scale_rounded == 0,
                              torch.tensor(0.0, device=device),
                              1.0 / dequant_scale_rounded)

    # Quantize the tensor
    orig_padded_shape = padded_src.shape
    padded_src_groups = padded_src.view(*new_shape)
    quant_tensor = padded_src_groups * quant_scale
    # Reshape back to the original shape and trim padding
    quant_tensor = quant_tensor.view(orig_padded_shape)
    quant_tensor = quant_tensor[..., :axis_shape]

    # Finally, convert the quantized tensor to the target format
    if is_fp8:
        # Conversion must use satfinite PTX, so clamp before the conversion in torch to emulate this behavior
        quant_tensor = torch.clamp(quant_tensor, -max_quant_val, max_quant_val)
        out_weight = quant_tensor.to(out_quant_type)
    else:
        assert is_fp4, f"Invalid output quantization type {out_quant_type}"
        # For mxfp4, perform bit-level manipulation and pack two 4-bit values per uint8.
        # First, reinterpret the quantized tensor bits.
        q_int = quant_tensor.contiguous().view(torch.int32)
        # Extract sign, exponent, and mantissa.
        signs = q_int & 0x80000000
        exponents = right_shift_unsigned(q_int, 23) & 0xFF
        mantissas = q_int & 0x7FFFFF

        E8_BIAS = 127
        E2_BIAS = 1
        # Adjust mantissas for subnormals.
        mantissas = torch.where(exponents < E8_BIAS,
                                (0x400000 | right_shift_unsigned(mantissas, 1)) >> (E8_BIAS - exponents - 1),
                                mantissas)
        exponents = torch.maximum(exponents,
                                  torch.tensor(E8_BIAS - E2_BIAS, device=device)) - (E8_BIAS - E2_BIAS)
        e2m1_tmp = right_shift_unsigned(((exponents << 2) | right_shift_unsigned(mantissas, 21)) + 1, 1)
        e2m1_tmp = torch.minimum(e2m1_tmp, torch.tensor(0x7, device=device))
        e2m1_value = (right_shift_unsigned(signs, 28) | e2m1_tmp).to(torch.uint8)  # shape: (..., even_axis_shape)

        # Pack pairs of 4-bit values along the last dimension.
        e2m1_value = e2m1_value.view(*e2m1_value.shape[:-1], axis_shape // 2, 2)
        evens = e2m1_value[..., 0]
        odds = e2m1_value[..., 1]
        out_weight = evens | (odds << 4)  # shape: (..., axis_shape//2)

    # --- Process and output the scale ---
    dq_scale = (ds_int_rounded.view(*dequant_scale.shape) >> 23).to(torch.uint8)  # shape: (..., axis_shape//32, 1)
    dq_scale = dq_scale.squeeze(-1)

    if convert_order:
        # dq_scale was produced in `permute_order`, but we want it to be in `scale_permute_order`.
        dq_scale = dq_scale.transpose(*convert_order)

    if swizzle_axis is not None:
        dq_scale = swizzle_mx(dq_scale)

    # Now, invert the permutation so that the contiguous axis returns to its original position.
    out_weight = out_weight.permute(permute_order)
    dq_scale = dq_scale.permute(scale_permute_order)

    if out_quant_tensor is not None:
        assert out_quant_tensor.shape == out_weight.shape, f"Invalid shape {out_quant_tensor.shape} != {out_weight.shape}"
        assert out_quant_tensor.dtype == out_weight.dtype, f"Invalid dtype {out_quant_tensor.dtype} != {out_weight.dtype}"
        out_quant_tensor.copy_(out_weight)
    else:
        out_quant_tensor = out_weight

    if out_scale is not None:
        assert out_scale.shape == dq_scale.shape, f"Invalid shape {out_scale.shape} != {dq_scale.shape}"
        assert out_scale.dtype == dq_scale.dtype, f"Invalid dtype {out_scale.dtype} != {dq_scale.dtype}"
        out_scale.copy_(dq_scale)
    else:
        out_scale = dq_scale

    return out_quant_tensor, out_scale.contiguous()


def cvt_e2m1_to_fp32(input_tensor):
    assert input_tensor.dtype == torch.uint8

    input_tensor = input_tensor.to(torch.int32)
    evens = input_tensor & 0xF
    odds = (input_tensor >> 4) & 0xF

    vals = [0.0, 0.5, 1, 1.5, 2, 3, 4, 6]
    outputs = torch.tensor(vals, dtype=torch.float32, device=input_tensor.device)
    outputs = torch.cat([outputs, -outputs])

    even_floats = outputs[evens]
    odd_floats = outputs[odds]
    output_tensor = torch.stack([even_floats, odd_floats], dim=-1)
    output_tensor = output_tensor.view(*input_tensor.shape[:-1], -1)
    return output_tensor


def upcast_from_mxfp_torch(tensor: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype, axis: int, swizzle_axis: int | None = None):
    """
    Converts the mxfp4/mxfp8 tensor to the target format specified by target_dtype.
      axis: The axis along which dequantization is applied.

    Returns:
      out_weight: Tensor in the target format.
    """

    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    is_fp8 = tensor.dtype == torch.float8_e4m3fn or tensor.dtype == torch.float8_e5m2
    assert is_fp8 or tensor.dtype == torch.uint8, f"Invalid input quantization type {tensor.dtype}"

    # Permute the tensor and scale so that the quantization axis becomes the last dimension
    # For the scales, also permute so the swizzle axis is second to last.
    axis = axis if axis >= 0 else axis + ndim
    if swizzle_axis is not None:
        assert -ndim <= swizzle_axis < ndim, f"Invalid swizzle axis {swizzle_axis=}"
        swizzle_axis = swizzle_axis if swizzle_axis >= 0 else swizzle_axis + ndim
    permute_order, scale_permute_order, convert_order = axis_permute_order(ndim, axis, swizzle_axis)

    tensor = tensor.permute(permute_order)
    scale = scale.permute(scale_permute_order)

    if swizzle_axis is not None:
        scale = unswizzle_mx(scale)

    dq_scale = (scale.to(torch.int32) << 23).view(torch.float32)  # Shift to the exponent and bitcast to fp32

    if is_fp8:
        fp32_tensor = tensor.to(torch.float32)
    else:
        assert tensor.dtype == torch.uint8
        fp32_tensor = cvt_e2m1_to_fp32(tensor)

    fp_tensor_shape = fp32_tensor.shape
    if convert_order:
        fp_tensor_shape = transpose_shape(fp_tensor_shape, *convert_order)

    # Trim padding
    dq_scale = dq_scale[..., :fp_tensor_shape[-2], :(fp_tensor_shape[-1] + 31) // 32]
    if convert_order:
        dq_scale = dq_scale.transpose(*convert_order)

    axis_shape = fp32_tensor.size(-1)
    padded_axis_shape = dq_scale.size(-1) * 32
    pad_size = padded_axis_shape - axis_shape
    padded_tensor = F.pad(fp32_tensor, (0, pad_size))

    new_axis_shape = padded_tensor.shape[-1]
    new_shape = padded_tensor.shape[:-1] + (new_axis_shape // 32, 32)
    padded_tensor = padded_tensor.view(*new_shape)
    dq_scale_padded = dq_scale.unsqueeze(-1)  # shape: [..., ceil(axis_shape/32), 1]
    out_padded = padded_tensor * dq_scale_padded

    # Flatten back and remove the padded tail
    out_padded = out_padded.view(*fp32_tensor.shape[:-1], new_axis_shape)
    out_tensor = out_padded[..., :axis_shape]

    out_tensor = out_tensor.permute(permute_order).to(target_dtype)
    return out_tensor


def swizzle_mx(tensor: torch.Tensor, allow_pad=True):
    """
    Swizzle the input tensor of shape (A, B, ... N, K) to (A, B, ... N // 128, K // 4, 32, 4, 4).
    Padding is applied if N and K are not multiples of 128 and 4 respectively.
    Returns the swizzled tensor repacked as (A, B, ... N, K), with padding.
    """
    *leading_shape, N, K, = tensor.shape
    pad_k = (SWIZZLE_ALIGN_INNER - (K % SWIZZLE_ALIGN_INNER)) % SWIZZLE_ALIGN_INNER
    pad_n = (SWIZZLE_SIZE_OUTER - (N % SWIZZLE_SIZE_OUTER)) % SWIZZLE_SIZE_OUTER
    if pad_k or pad_n > 0:
        assert allow_pad, "Padding is required for swizzling, but it was explicitly disabled."
        tensor = torch.nn.functional.pad(tensor, (0, pad_k, 0, pad_n))
    padded_shape = tensor.shape
    tensor = tensor.reshape(*leading_shape, padded_shape[-2] // SWIZZLE_SIZE_OUTER, SWIZZLE_SIZE_OUTER // 32, 32, padded_shape[-1] // SWIZZLE_SIZE_INNER, SWIZZLE_SIZE_INNER)
    permute_order = list(range(len(tensor.shape)))
    permute_order[-2], permute_order[-4] = permute_order[-4], permute_order[-2]
    return tensor.permute(permute_order).reshape(*padded_shape)


def unswizzle_mx(tensor: torch.Tensor):
    """
    Unswizzle the input tensor of shape (A, B, ... N // 128, K // 4, 32, 4, 4) packed as (A, B, ... N, K).
    """
    assert tensor.shape[-1] % SWIZZLE_SIZE_INNER == 0, f"{tensor.shape[-1]=} must be a multiple of {SWIZZLE_SIZE_INNER}"
    assert tensor.shape[-2] % SWIZZLE_SIZE_OUTER == 0, f"{tensor.shape[-2]=} must be a multiple of {SWIZZLE_SIZE_OUTER}"
    *leading_shape, N, K, = tensor.shape
    tensor = tensor.reshape(*leading_shape, N // SWIZZLE_SIZE_OUTER, K // SWIZZLE_SIZE_INNER, 32, SWIZZLE_SIZE_OUTER // 32, SWIZZLE_SIZE_INNER)
    permute_order = list(range(len(tensor.shape)))
    permute_order[-2], permute_order[-4] = permute_order[-4], permute_order[-2]
    return tensor.permute(permute_order).reshape(*leading_shape, N, K)

def per_token_group_dequant_mxfp4_triton(x: torch.Tensor, scale: torch.Tensor,
                                  block_k: int,
                                  float_dtype: torch.dtype) -> torch.Tensor:
    return upcast_from_mxfp(x, scale, float_dtype, axis=-1, swizzle_axis=None)

def per_token_group_quant_mxfp4_triton(x: torch.Tensor, block_k: int):
    x_mxfp4, scale_e8m0, _ = downcast_to_mxfp(
        x,
        torch.uint8,
        axis=-1,
        swizzle_axis=None,
        out_quant_tensor=None,
        out_scale=None,
        DEQUANT_SCALE_ROUNDING_MODE=DequantScaleRoundingMode.EVEN,  # 0: ceil, 1: floor, 2: even
    )
    x_qdq = upcast_from_mxfp(x_mxfp4, scale_e8m0, x.dtype, axis=-1, swizzle_axis=None)
    return x_qdq

PER_TOKEN_GROUP_QUANT_IMPLEM = {
    "torch": per_token_group_quant_mxfp4_torch,
    "hip": per_token_group_quant_mxfp4_hip,
    "triton": per_token_group_quant_mxfp4_triton,
}

PER_TOKEN_GROUP_DEQUANT_IMPLEM = {
    "torch": per_token_group_dequant_mxfp4_torch,
    "hip": per_token_group_dequant_mxfp4_hip,
    "triton": per_token_group_dequant_mxfp4_triton,
}

per_token_group_quant_mxfp4 = PER_TOKEN_GROUP_QUANT_IMPLEM[envs.VLLM_QUARK_MXFP4_Q_DQ_QDQ_IMPLEM]
per_token_group_dequant_mxfp4 = PER_TOKEN_GROUP_DEQUANT_IMPLEM[envs.VLLM_QUARK_MXFP4_Q_DQ_QDQ_IMPLEM]
