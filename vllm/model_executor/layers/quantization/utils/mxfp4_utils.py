# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_triton_kernels
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer

logger = init_logger(__name__)

# CK's pre-compiled MXFP4 MoE GEMM kernel instances require the
# intermediate_size (after TP split) to be a multiple of this value.
# This arises from FP4 packing (2 values per byte) combined with CK
# tile size constraints. When violated, AITER raises:
# "device_gemm ... does not support this GEMM problem".
CK_MXFP4_MOE_DIM_ALIGNMENT = 256


def should_use_cdna4_mx_scale_swizzle() -> bool:
    """Whether to use the CDNA4 swizzled scale layout for mxfp4 on gfx950.

    CDNA4 swizzle requires BLOCK_K%256==0; at TP>=4 the A8W4 dispatch
    picks BK<256 tiles for the smaller per-rank shapes, so swizzle must
    be off. Used by both the weight-load swizzle in `_swizzle_mxfp4` and
    the kernel-argument gate in `aiter_mxfp4_w4a8_moe`; they must agree.
    """
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950() and get_tensor_model_parallel_world_size() <= 2


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
        value_layout = StridedLayout
        if should_use_cdna4_mx_scale_swizzle():
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
            # Patches #47303: pad K (num scale groups) to 0 mod 4
            # TODO: Remove once we upgrade to Triton 3.8.0+ kernels
            if scale.numel() > 0:
                K = scale.shape[-1]
                pad_k = -K % 4
                scale = torch.nn.functional.pad(scale, (0, pad_k))
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


def permute_shape(shape: tuple[int, ...], permute_order: list[int]) -> tuple[int, ...]:
    return tuple(shape[i] for i in permute_order)


@triton.jit
def _compute_quant_and_scale(src_tensor, valid_src_mask):
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 32

    # Explicit cast to fp32 since most ops are not supported on bfloat16. We
    # avoid needless conversions to and from bf16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(
        valid_src_mask, abs_tensor, -1.0
    )  # Don't consider padding tensors in scale computation
    abs_tensor = tl.reshape(
        abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    # Round-to-nearest-even block scale (OCP MX spec), matching hardware/aiter
    # MXFP4 quantizers: compute 2 ** (floor(log2(rounding(max_abs(v)))) - max_exp).
    max_val = max_val.to(tl.int32, bitcast=True)
    max_val = (max_val + 0x200000).to(tl.uint32, bitcast=True) & 0x7F800000
    max_val = max_val.to(tl.float32, bitcast=True)
    # Add epsilon to prevent log2(0) = -inf when max_val is zero (all-zero input block)
    eps = tl.where(max_val == 0.0, 2 ** (-126), 0.0)
    scale_e8m0_unbiased = tl.log2(max_val + eps).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    dequant_scale_rounded = tl.exp2(scale_e8m0_unbiased)
    dequant_scale_exponent = dequant_scale_rounded.to(tl.uint32, bitcast=True)

    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(
        f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    quant_tensor = f32_tensor * quant_scale

    # Reshape the tensors after scaling
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set the invalid portions of the tensor to 0. This will ensure that any
    # padding tensors are 0 in the mx format.
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape(
        [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE]
    )

    # First, we simply extract the exponent part of the scales and store the result
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    # Now we must convert the tensor to packed e2m1 (mxfp4).
    quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
    signs = quant_tensor & 0x80000000
    exponents = (quant_tensor >> 23) & 0xFF
    mantissas = quant_tensor & 0x7FFFFF

    # 0.25 <= x < 0.75 maps to 0.5, a denormal number
    E8_BIAS = 127
    E2_BIAS = 1
    # Move implicit bit 1 at the beginning to mantissa for denormals. Bits
    # shifted out are OR-reduced into the LSB as a sticky bit, so the
    # round-to-even logic below can tell an exact tie from a value that is
    # merely close to one.
    adjusted_exponents = tl.core.sub(E8_BIAS, exponents + 1, sanitize_overflow=False)
    folded_mantissas = 0x400000 | (mantissas >> 1)
    denormal_mantissas = folded_mantissas >> adjusted_exponents
    denormal_mantissas |= (
        (denormal_mantissas << adjusted_exponents) != folded_mantissas
    ).to(tl.uint32)
    mantissas = tl.where(exponents < E8_BIAS, denormal_mantissas, mantissas)

    # For normal numbers, we change the bias from 127 to 1, and for subnormals,
    # we keep exponent as 0.
    exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    # Combine sign, exponent, and mantissa, rounding to nearest with ties
    # broken to even (OCP MX spec, matching hardware/aiter MXFP4
    # quantizers) via the standard "add a rounding bias, then truncate"
    # trick: the bias's LSB is the kept mantissa bit, so an exact tie
    # rounds to whichever candidate is even, while any lower (sticky) bit
    # being set always rounds up.
    rounding_bias = ((mantissas >> 22) & 1) + 0x1FFFFF
    rounded_mantissa = (mantissas + rounding_bias) >> 22  # in {0, 1, 2}
    e2m1_tmp = tl.minimum((exponents << 1) + rounded_mantissa, 0x7)
    e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale_exponent


@triton.jit
def _downcast_to_mxfp(
    mx_tensor_ptr,
    stride_mxt_outer,
    stride_mxt_quant: tl.constexpr,
    mx_scale_ptr,
    stride_mx_scale_outer,
    stride_mx_scale_quant,
    src_ptr,
    stride_src_outer,
    stride_src_quant,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
):
    tl.static_assert(
        stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1."
    )
    tl.static_assert(
        BLOCK_SIZE_QUANT_DIM % 32 == 0,
        f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32",
    )

    # uint8 signifies two fp4 e2m1 values packed into a single byte
    tl.static_assert(
        mx_tensor_ptr.dtype.element_ty == tl.uint8,
        f"Invalid {mx_tensor_ptr.dtype.element_ty=}. Must be uint8.",
    )

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint8,
        f"{mx_scale_ptr.dtype.element_ty=} must be uint8",
    )
    tl.static_assert(
        (src_dtype == tl.bfloat16) or (src_dtype == tl.float16),
        f"{src_dtype=} must be bfloat16 or float16",
    )

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 2

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += (
        start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    )
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant and mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < tl.cdiv(quant_dim, 2)
    full_mask_mxt = mask_mxt_quant and mask_n

    scale_mask_k = start_mx_scale_quant + offs_scale_quant < tl.cdiv(quant_dim, 32)
    full_scale_mask = scale_mask_k and mask_n

    src_tensor_offsets = (
        offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    )
    mx_scale_offsets = (
        offs_scale_quant * stride_mx_scale_quant + offs_outer * stride_mx_scale_outer
    )
    mx_tensor_offsets = (
        offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    )
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    out_tensor, scale_tensor = _compute_quant_and_scale(src_tensor, full_mask_src)

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


def downcast_to_mxfp(
    src_tensor: torch.Tensor,
    axis: int,
    out_quant_tensor: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    BLOCK_OUT_DIM: int = 128,
    BLOCK_QUANT_DIM: int = 32,
):
    """
    Convert the src weights to MXFP4. The src weight is quantized along the
    axis dimension into packed e2m1 values (torch.uint8, two values per
    byte), so the size of that dimension in the output is half of the
    logical (unpacked) size.
    """
    out_quant_type = torch.uint8
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim

    L = src_tensor.shape[axis]
    # We make this assertion since we can't track if the "real" shape was odd,
    # and we padded it to be even.
    # We want to maintain the property dequant(quant(x)).shape == x.shape
    assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"

    device = src_tensor.device

    packed_quant_dim = triton.cdiv(L, 2)
    out_scale_dim = triton.cdiv(L, 32)

    # Move the quantization axis to the end for the kernel, then permute back.
    permute_order = list(range(ndim))
    permute_order[axis], permute_order[-1] = permute_order[-1], permute_order[axis]

    prmted_quant_tensor_shape = permute_shape(src_tensor.shape, permute_order)[:-1] + (
        packed_quant_dim,
    )
    prmted_scale_shape = permute_shape(src_tensor.shape, permute_order)[:-1] + (
        out_scale_dim,
    )
    prmted_src_tensor = src_tensor.permute(permute_order)

    if out_quant_tensor is None:
        out_quant_tensor = torch.empty(
            prmted_quant_tensor_shape, dtype=out_quant_type, device=device
        )
    else:
        expected_shape = (
            src_tensor.shape[:axis] + (packed_quant_dim,) + src_tensor.shape[axis + 1 :]
        )
        assert out_quant_tensor.shape == expected_shape, (
            f"{out_quant_tensor.shape=} != {expected_shape=}"
        )
        assert out_quant_tensor.dtype == out_quant_type, (
            f"{out_quant_tensor.dtype=} != {out_quant_type=}"
        )
        assert out_quant_tensor.stride(axis) == 1, (
            f"{out_quant_tensor.stride(axis)=} != 1"
        )
        # We expect the axis dimension to be last, so permute the tensor
        out_quant_tensor = out_quant_tensor.permute(permute_order)

    if out_scale is None:
        out_scale = torch.empty(prmted_scale_shape, dtype=torch.uint8, device=device)
    else:
        expected_scale_shape = permute_shape(prmted_scale_shape, permute_order)
        assert out_scale.shape == expected_scale_shape, (
            f"{out_scale.shape=} {expected_scale_shape=}"
        )
        assert out_scale.dtype == torch.uint8, f"{out_scale.dtype=} != torch.uint8"
        out_scale = out_scale.permute(permute_order)

    # Flatten input tensor for kernel. This will typically make a copy
    reshaped_src_tensor = prmted_src_tensor.reshape(-1, L)
    blocks_quant_dim = triton.cdiv(reshaped_src_tensor.shape[-1], BLOCK_QUANT_DIM)
    blocks_out_dim = triton.cdiv(reshaped_src_tensor.shape[0], BLOCK_OUT_DIM)

    # Flatten the output tensors for the kernel, this should be a view always
    kernel_quant_tensor = out_quant_tensor.reshape(-1, packed_quant_dim)
    kernel_scale = out_scale.reshape(-1, out_scale_dim)
    assert kernel_quant_tensor.data_ptr() == out_quant_tensor.data_ptr()
    assert kernel_scale.data_ptr() == out_scale.data_ptr()

    _downcast_to_mxfp[(blocks_out_dim, blocks_quant_dim)](
        kernel_quant_tensor,
        *kernel_quant_tensor.stride(),
        kernel_scale,
        *kernel_scale.stride(),
        reshaped_src_tensor,
        *reshaped_src_tensor.stride(),
        *reshaped_src_tensor.shape,
        BLOCK_OUT_DIM,
        BLOCK_QUANT_DIM,
        num_warps=8,
    )

    out_quant_tensor = out_quant_tensor.permute(permute_order)
    out_scale = out_scale.permute(permute_order).contiguous()
    return out_quant_tensor, out_scale, permute_shape(prmted_scale_shape, permute_order)


def mxfp4_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a bf16/fp16 tensor to MXFP4 along its last dimension.

    Dispatches to the fastest backend available on the current platform:

    - the native XPU custom op on XPU,
    - FlashInfer on CUDA when installed,
    - aiter on ROCm when `aiter` is installed,
    - and the portable Triton kernel (``downcast_to_mxfp``) otherwise.

    Returns packed FP4 values (uint8, two values per byte) and per-block
    (group-32) e8m0 scales (uint8).
    """
    if current_platform.is_xpu():
        return xpu_mxfp4_quantize(x)

    if current_platform.is_cuda():
        from vllm.utils.flashinfer import has_flashinfer

        if has_flashinfer():
            from vllm.utils.flashinfer import flashinfer_mxfp4_quantize

            return flashinfer_mxfp4_quantize(x, backend="cute-dsl")

    from vllm._aiter_ops import is_aiter_found_and_supported

    if is_aiter_found_and_supported() and x.dtype == torch.bfloat16:
        from vllm.model_executor.layers.quantization.quark.utils import (
            quark_quantize_weight_to_mxfp4,
        )

        return quark_quantize_weight_to_mxfp4(x)

    quant_tensor, scale, _ = downcast_to_mxfp(x, axis=-1)
    return quant_tensor, scale
