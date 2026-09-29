# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused dynamic per-tensor FP8 quantization of up to three attention inputs.

Each ``[tokens, heads, dim]`` input (possibly strided, e.g. V as a view of the
``kv_b_proj`` output) gets its own amax and FP32 descale. A partial-amax pass
and a reduce-and-quantize pass cover all inputs in two launches; the kernel
boundary is the only global barrier, so no atomics or zero-initialization are
needed and the ops are CUDA-graph safe. Compile-time specialization does not
depend on the token counts (only the grid size does), so varying batch shapes
do not recompile.

Adapted from the fused QKV quantizer in ROCm/ATOM (atom/model_ops/
triton_fused_qkv_quant.py, MIT license).
"""

import torch

from vllm.triton_utils import tl, triton

_BLOCK = 4096
_PARTS = 256
# Descale for all-zero or empty inputs. A positive value keeps downstream
# attention kernels from producing NaNs.
_MIN_DESCALE = 1e-6


@triton.jit
def _load(X, offsets, N, SHAPE: tl.constexpr, STRIDE: tl.constexpr):
    heads, dim = SHAPE
    offsets = offsets.to(tl.int64)
    physical = (
        offsets // (heads * dim) * STRIDE[0]
        + offsets // dim % heads * STRIDE[1]
        + offsets % dim * STRIDE[2]
    )
    return tl.load(X + physical, offsets < N, other=0.0).to(tl.float32)


@triton.jit
def _partial_amax(
    X,
    Partial,
    N,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for step in range(tl.cdiv(N, PARTS * BLOCK)):
        x = _load(X, offsets + step * PARTS * BLOCK, N, SHAPE, STRIDE)
        acc = tl.maximum(acc, tl.abs(x))
    tl.store(Partial + tl.program_id(0), tl.max(acc, 0))


@triton.jit
def _fused_amax_kernel(
    X0,
    X1,
    X2,
    Partial,
    N0,
    N1,
    N2,
    SHAPES: tl.constexpr,
    STRIDES: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(1)
    if idx == 0:
        _partial_amax(X0, Partial, N0, SHAPES[0], STRIDES[0], PARTS, BLOCK)
    elif idx == 1:
        _partial_amax(X1, Partial + PARTS, N1, SHAPES[1], STRIDES[1], PARTS, BLOCK)
    else:
        _partial_amax(X2, Partial + 2 * PARTS, N2, SHAPES[2], STRIDES[2], PARTS, BLOCK)


@triton.jit
def _quant(
    X,
    Y,
    Partial,
    Scale,
    N,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MIN_DESCALE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = _load(X, offsets, N, SHAPE, STRIDE)
    amax = tl.max(tl.load(Partial + tl.arange(0, PARTS)), 0)
    descale = amax / FP8_MAX
    descale = tl.where(descale > 0, descale, MIN_DESCALE)
    y = tl.clamp(x * (1.0 / descale), -FP8_MAX, FP8_MAX)
    tl.store(Y + offsets, y.to(Y.dtype.element_ty), offsets < N)
    if tl.program_id(0) == 0:
        tl.store(Scale, descale)


@triton.jit
def _fused_quant_kernel(
    X0,
    X1,
    X2,
    Y0,
    Y1,
    Y2,
    Partial,
    Scales,
    N0,
    N1,
    N2,
    SHAPES: tl.constexpr,
    STRIDES: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MIN_DESCALE: tl.constexpr,
):
    idx = tl.program_id(1)
    if idx == 0:
        _quant(
            X0,
            Y0,
            Partial,
            Scales,
            N0,
            SHAPES[0],
            STRIDES[0],
            PARTS,
            BLOCK,
            FP8_MAX,
            MIN_DESCALE,
        )
    elif idx == 1:
        _quant(
            X1,
            Y1,
            Partial + PARTS,
            Scales + 1,
            N1,
            SHAPES[1],
            STRIDES[1],
            PARTS,
            BLOCK,
            FP8_MAX,
            MIN_DESCALE,
        )
    else:
        _quant(
            X2,
            Y2,
            Partial + 2 * PARTS,
            Scales + 2,
            N2,
            SHAPES[2],
            STRIDES[2],
            PARTS,
            BLOCK,
            FP8_MAX,
            MIN_DESCALE,
        )


def fused_per_tensor_fp8_quant(
    *tensors: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Quantize each input to FP8 with its own dynamic per-tensor scale.

    Args:
        tensors: One to three BF16/FP16 ``[tokens, heads, dim]`` tensors on the
            same device. Arbitrary strides are supported; token counts may
            differ and may be zero.
        fp8_dtype: The FP8 output dtype.

    Returns:
        ``(outputs, descales)``: contiguous FP8 tensors with the input shapes
        and FP32 shape-``[1]`` descales (``amax / fp8_max``, or 1e-6 for
        all-zero or empty inputs).

    """
    if not 1 <= len(tensors) <= 3:
        raise ValueError(f"Expected 1 to 3 tensors, got {len(tensors)}")
    first = tensors[0]
    for x in tensors:
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] == 0:
            raise ValueError(
                f"Expected [tokens, nonzero heads, nonzero dim], got {x.shape}"
            )
        if x.device != first.device or x.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise ValueError("Inputs must be BF16/FP16 tensors on the same device")

    outputs = tuple(
        torch.empty(x.shape, dtype=fp8_dtype, device=x.device) for x in tensors
    )
    scales = torch.empty(len(tensors), dtype=torch.float32, device=first.device)
    numels = [x.numel() for x in tensors]
    partial = torch.empty(
        (len(tensors), _PARTS), dtype=torch.float32, device=first.device
    )

    # Unused slots alias the first tensor; their programs never run.
    pad = 3 - len(tensors)
    xs = (*tensors, *(first,) * pad)
    ys = (*outputs, *(outputs[0],) * pad)
    ns = (*numels, *(0,) * pad)
    shapes = tuple(tuple(x.shape[1:]) for x in xs)
    strides = tuple(x.stride() for x in xs)

    _fused_amax_kernel[(_PARTS, len(tensors))](
        *xs, partial, *ns, shapes, strides, _PARTS, _BLOCK, num_warps=4
    )
    _fused_quant_kernel[(max(1, triton.cdiv(max(numels), _BLOCK)), len(tensors))](
        *xs,
        *ys,
        partial,
        scales,
        *ns,
        shapes,
        strides,
        _PARTS,
        _BLOCK,
        torch.finfo(fp8_dtype).max,
        _MIN_DESCALE,
        num_warps=4,
    )
    return outputs, tuple(scales[i : i + 1] for i in range(len(tensors)))
