from ..numerics import MAX_FINITE_FLOAT8E4B8, MAX_FINITE_FLOAT8E4NV, MAX_FINITE_FLOAT8E5
import triton
import triton.language as tl
from triton_kernels.target_info import cuda_capability_geq

# -------------------------------
# Kernels stuff
# -------------------------------

TL_MAX_FINITE_FLOAT8E5 = tl.constexpr(MAX_FINITE_FLOAT8E5)
TL_MAX_FINITE_FLOAT8E4NV = tl.constexpr(MAX_FINITE_FLOAT8E4NV)
TL_MAX_FINITE_FLOAT8E4B8 = tl.constexpr(MAX_FINITE_FLOAT8E4B8)
TL_MAX_FINITE_FLOAT8E4B15 = tl.constexpr(1.750)
TL_MAX_FINITE_FLOAT16 = tl.constexpr(65472.0)

TL_RCP_MAX_FINITE_FLOAT8E5 = tl.constexpr(0x37924925)  # 0x1.24924Ap-16
TL_RCP_MAX_FINITE_FLOAT8E4NV = tl.constexpr(0x3B124925)  # 0x1.24924Ap-9
TL_RCP_MAX_FINITE_FLOAT8E4B8 = tl.constexpr(0x3B888889)  # 0x1.111112p-8
TL_RCP_MAX_FINITE_FLOAT8E4B15 = tl.constexpr(0x3F124925)  # 0x1.24924Ap-1
TL_RCP_MAX_FINITE_FLOAT16 = tl.constexpr(0x37802008)  # 0x1.004010p-16


@triton.jit
def max_finite(dtype):
    if dtype == tl.constexpr(tl.float8e5):
        return TL_MAX_FINITE_FLOAT8E5
    elif dtype == tl.constexpr(tl.float8e4nv):
        return TL_MAX_FINITE_FLOAT8E4NV
    elif dtype == tl.constexpr(tl.float8e4b8):
        return TL_MAX_FINITE_FLOAT8E4B8
    elif dtype == tl.constexpr(tl.float8e4b15):
        return TL_MAX_FINITE_FLOAT8E4B15
    elif dtype == tl.constexpr(tl.float16):
        return TL_MAX_FINITE_FLOAT16
    else:
        tl.static_assert(tl.constexpr(False), f"{dtype} not supported in flexpoint")


@triton.jit
def rcp_max_finite(dtype):
    if dtype == tl.constexpr(tl.float8e5):
        return TL_RCP_MAX_FINITE_FLOAT8E5
    elif dtype == tl.constexpr(tl.float8e4nv):
        return TL_RCP_MAX_FINITE_FLOAT8E4NV
    elif dtype == tl.constexpr(tl.float8e4b8):
        return TL_RCP_MAX_FINITE_FLOAT8E4B8
    elif dtype == tl.constexpr(tl.float8e4b15):
        return TL_RCP_MAX_FINITE_FLOAT8E4B15
    elif dtype == tl.constexpr(tl.float16):
        return TL_RCP_MAX_FINITE_FLOAT16
    else:
        tl.static_assert(tl.constexpr(False), f"{dtype} not supported in flexpoint")


@triton.jit
def sm86_min_nan_xorsign_abs_f32(a, b):
    """Wrapper for min.NaN.xorsign.abs.f32 PTX instruction.

    Computes the minimum of the absolute values of the two inputs and sets its sign to the XOR of the signs of the inputs.
    NaN inputs are propagated to the output.

    Requires CUDA compute capability 8.6+ (A100 and A30 Ampere GPUs don't support it, but A40/A16/A10/A2, Ada, and Hopper GPUs do).
    """
    tl.static_assert(cuda_capability_geq(8, 6), "min.NaN.xorsign.abs.f32 requires CUDA compute capability 8.6+")
    tl.static_assert(a.dtype == tl.float32, "min.NaN.xorsign.abs.f32 requires float32 inputs")
    tl.static_assert(b.dtype == tl.float32, "min.NaN.xorsign.abs.f32 requires float32 inputs")

    return tl.inline_asm_elementwise(
        """{
    min.NaN.xorsign.abs.f32 $0, $1, $2;
    }""",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sm86_max_nan_xorsign_abs_f32(a, b):
    """Wrapper for max.NaN.xorsign.abs.f32 PTX instruction.

    Computes the maximum of the absolute values of the two inputs and sets its sign to the XOR of the signs of the inputs.
    NaN inputs are propagated to the output.

    Requires CUDA compute capability 8.6+ (A100 and A30 Ampere GPUs don't support it, but A40/A16/A10/A2, Ada, and Hopper GPUs do).
    """
    tl.static_assert(cuda_capability_geq(8, 6), "max.NaN.xorsign.abs.f32 requires CUDA compute capability 8.6+")
    tl.static_assert(a.dtype == tl.float32, "max.NaN.xorsign.abs.f32 requires float32 inputs")
    tl.static_assert(b.dtype == tl.float32, "max.NaN.xorsign.abs.f32 requires float32 inputs")

    return tl.inline_asm_elementwise(
        """{
    max.NaN.xorsign.abs.f32 $0, $1, $2;
    }""",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def load_scale(scale_ptr):
    return 1.0 if scale_ptr is None else tl.load(scale_ptr)


@triton.jit
def flex_to_float(x, scale_ptr):
    scale = load_scale(scale_ptr)
    return x.to(tl.float32) * scale


@triton.jit
def clip(x, limit):
    res = tl.minimum(x, limit)
    res = tl.maximum(-limit, res)
    return res


@triton.jit
def nan_propagating_absmax_reduce(x, axis=None):
    if cuda_capability_geq(8, 6):
        # abs-max-reduce as floating-point if `max.NaN.xorsign.abs.f32` is supported.
        x_absmax = tl.reduce(x, axis, sm86_max_nan_xorsign_abs_f32)
        # Note: sign of reduction result is the xor of signs of all inputs, explicitly clear the sign bit to fix it.
        x_absmax = x_absmax.to(tl.uint32, bitcast=True) & 0x7FFFFFFF
    else:
        # Clear the sign bit, max-reduce as integer (same as NaN-propagating max-reduce as float)
        masked_abs_x = x.to(tl.uint32, bitcast=True) & 0x7FFFFFFF
        x_absmax = tl.max(masked_abs_x, axis)

    return x_absmax


@triton.jit
def compute_scale(x, Out):
    x_absmax = nan_propagating_absmax_reduce(tl.ravel(x, can_reorder=True))

    # atomic_max does not propagate NaNs, so we replace them with +inf (0x7f800000).
    # We use integer minimum because NaNs are above +inf in integer representation.
    x_absmax = tl.minimum(x_absmax, 0x7F800000).to(tl.float32, bitcast=True)
    RCP_MAX_VALUE = rcp_max_finite(Out.dtype.element_ty)
    return tl.fma(x_absmax, RCP_MAX_VALUE.to(tl.float32, bitcast=True), 1.0e-30)


@triton.jit
def update_scale(x, scale_ptr, Out) -> None:
    if scale_ptr is not None:
        scale = compute_scale(x, Out)
        tl.atomic_max(scale_ptr, scale, sem="relaxed")


@triton.jit
def float_to_flex(
    x,
    expected_scale_ptr_or_val,
    actual_scale_ptr,
    checksum_scale_ptr,
    mask,
    Out,
    saturate_infs: tl.constexpr,
):
    if expected_scale_ptr_or_val is not None:
        if expected_scale_ptr_or_val.dtype.is_ptr():
            invscale = 1.0 / tl.load(expected_scale_ptr_or_val)
        else:
            invscale = 1.0 / expected_scale_ptr_or_val
    else:
        invscale = 1.0
    if checksum_scale_ptr is not None:
        x_int32 = x.to(tl.int32, bitcast=True)
        zero = tl.cast(0.0, tl.int32)
        if mask is not None:
            x_int32 = tl.where(mask, x_int32, zero)
        checksum_local = tl.xor_sum(tl.ravel(x_int32, can_reorder=True), 0)
        tl.atomic_add(checksum_scale_ptr, checksum_local)
    if mask is not None:
        if actual_scale_ptr is not None:
            x = tl.where(mask, x, 0.0)
    update_scale(x, actual_scale_ptr, Out)
    x = x * invscale
    # if expected_scale_ptr is not None, we applied flexpoint scale. We only want to clip in this case.
    if expected_scale_ptr_or_val is not None:
        if saturate_infs:
            CLIP_VALUE = max_finite(Out.dtype.element_ty)
            x = clip(x, CLIP_VALUE)
    return x
