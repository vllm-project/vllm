# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions adapted from ATOM under the MIT License:
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Fused QKV and KV dynamic FP8 quantization and MLA descale preparation."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _load_tensor(
    X,
    offsets,
    N,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    Rope=None,
    ROPE_DIM: tl.constexpr = 0,
    ROPE_STRIDE: tl.constexpr = (0, 0, 0),
):
    heads, dim = SHAPE
    if ROPE_DIM:
        # K is logically cat(k_nope, k_rope), without a BF16 intermediate.
        token = offsets // (heads * dim)
        head = offsets // dim % heads
        col = offsets % dim
        nope_dim = dim - ROPE_DIM
        nope = tl.load(
            X + token * STRIDE[0] + head * STRIDE[1] + col * STRIDE[2],
            (offsets < N) & (col < nope_dim),
            0,
        ).to(tl.float32)
        rope = tl.load(
            Rope
            + token * ROPE_STRIDE[0]
            + head * ROPE_STRIDE[1]
            + (col - nope_dim) * ROPE_STRIDE[2],
            (offsets < N) & (col >= nope_dim),
            0,
        ).to(tl.float32)
        return tl.where(col < nope_dim, nope, rope)
    physical = (
        offsets // (heads * dim) * STRIDE[0]
        + offsets // dim % heads * STRIDE[1]
        + offsets % dim * STRIDE[2]
    )
    return tl.load(X + physical, offsets < N, 0).to(tl.float32)


@triton.jit
def _partial_amax(
    X,
    Partial,
    N,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    Rope=None,
    ROPE_DIM: tl.constexpr = 0,
    ROPE_STRIDE: tl.constexpr = (0, 0, 0),
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for step in range(tl.cdiv(N, PARTS * BLOCK)):
        x = _load_tensor(
            X,
            offsets + step * PARTS * BLOCK,
            N,
            SHAPE,
            STRIDE,
            Rope,
            ROPE_DIM,
            ROPE_STRIDE,
        )
        acc = tl.maximum(acc, tl.abs(x))
    tl.store(Partial + tl.program_id(0), tl.max(acc, 0))


@triton.jit
def _fused_qkv_amax(
    Q,
    K,
    V,
    Partial,
    NQ,
    NK,
    NV,
    SHAPES: tl.constexpr,
    STRIDES: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    K_ROPE=None,
    ROPE_DIM: tl.constexpr = 0,
    ROPE_STRIDE: tl.constexpr = (0, 0, 0),
    SKIP_Q: tl.constexpr = False,
):
    kind = tl.program_id(1) + (1 if SKIP_Q else 0)
    if kind == 0:
        _partial_amax(Q, Partial, NQ, SHAPES[0], STRIDES[0], PARTS, BLOCK)
    elif kind == 1:
        _partial_amax(
            K,
            Partial + PARTS,
            NK,
            SHAPES[1],
            STRIDES[1],
            PARTS,
            BLOCK,
            K_ROPE,
            ROPE_DIM,
            ROPE_STRIDE,
        )
    else:
        _partial_amax(V, Partial + 2 * PARTS, NV, SHAPES[2], STRIDES[2], PARTS, BLOCK)


@triton.jit
def _quant_tensor(
    X,
    Y,
    Partial,
    Scale,
    GatherScale,
    N,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    GATHER: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    Rope=None,
    ROPE_DIM: tl.constexpr = 0,
    ROPE_STRIDE: tl.constexpr = (0, 0, 0),
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = _load_tensor(X, offsets, N, SHAPE, STRIDE, Rope, ROPE_DIM, ROPE_STRIDE)
    if SINGLE_PASS:
        amax = tl.max(tl.abs(x), 0)
    else:
        amax = tl.max(tl.load(Partial + tl.arange(0, PARTS)), 0)
    # Match AITER for nonzero inputs; positive zero-input descales avoid FMHA NaNs.
    raw_descale = amax * (1.0 / 448.0)
    descale = tl.where(raw_descale > 0, raw_descale, 1e-6)
    inv = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1;",
        constraints="=v,v",
        args=[descale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    y = tl.minimum(tl.maximum(x * inv, -448.0), 448.0)
    tl.store(Y + offsets, y, offsets < N)
    if tl.program_id(0) == 0:
        tl.store(Scale, descale)
        if GATHER:
            tl.store(GatherScale, tl.maximum(descale, 1e-6) * 2.0)


@triton.jit
def _fused_qkv_quant(
    Q,
    K,
    V,
    Q8,
    K8,
    V8,
    Partial,
    Scales,
    NQ,
    NK,
    NV,
    SHAPES: tl.constexpr,
    STRIDES: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    K_ROPE=None,
    ROPE_DIM: tl.constexpr = 0,
    ROPE_STRIDE: tl.constexpr = (0, 0, 0),
    SKIP_Q: tl.constexpr = False,
):
    kind = tl.program_id(1) + (1 if SKIP_Q else 0)
    if kind == 0:
        _quant_tensor(
            Q,
            Q8,
            Partial,
            Scales,
            Scales,
            NQ,
            SHAPES[0],
            STRIDES[0],
            PARTS,
            BLOCK,
            False,
            SINGLE_PASS,
        )
    elif kind == 1:
        _quant_tensor(
            K,
            K8,
            Partial + PARTS,
            Scales + 1,
            Scales + 3,
            NK,
            SHAPES[1],
            STRIDES[1],
            PARTS,
            BLOCK,
            True,
            SINGLE_PASS,
            K_ROPE,
            ROPE_DIM,
            ROPE_STRIDE,
        )
    else:
        _quant_tensor(
            V,
            V8,
            Partial + 2 * PARTS,
            Scales + 2,
            Scales + 4,
            NV,
            SHAPES[2],
            STRIDES[2],
            PARTS,
            BLOCK,
            True,
            SINGLE_PASS,
        )


def fused_qkv_per_tensor_quant(q, k, v, *, k_rope=None, quantized_q=None):
    """Quantize 3-D Q/K/V to E4M3 with independent per-tensor FP32 descales.

    Returns ``(q8, k8, v8, qs, ks, vs, gather_ks, gather_vs)`` with contiguous
    outputs and shape-[1] scales. Gather descales are ``max(ks/vs, 1e-6) * 2``.
    Reads strided inputs in one launch for small tensors, otherwise two.
    Zero/empty inputs use descale 1e-6 and gather descale 2e-6.

    With ``quantized_q``, reuse Q8 and its scale from the same immutable
    forward and launch only K/V work; cross-forward reuse is forbidden.

    With ``k_rope``, ``k`` is the NoPE part and K8 is the quantized logical
    concatenation along the last dimension. K's amax includes both parts.
    ``k_rope`` may have one head (broadcast) or the same head count as K.
    """
    tensors = (q, k, v)
    if any(x.ndim != 3 or x.shape[1] == 0 or x.shape[2] == 0 for x in tensors):
        raise ValueError("Q/K/V must have shape [tokens, nonzero heads, nonzero dim]")
    if not q.is_cuda or any(
        x.device != q.device or x.dtype != q.dtype for x in tensors
    ):
        raise ValueError("Q/K/V must have the same GPU device and dtype")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Q/K/V must be BF16 or FP16")
    rope_dim = 0
    rope_stride = (0, 0, 0)
    k_shape = tuple(k.shape)
    if k_rope is not None:
        if (
            k_rope.ndim != 3
            or k_rope.shape[0] != k.shape[0]
            or k_rope.shape[1] not in (1, k.shape[1])
            or k_rope.shape[2] == 0
        ):
            raise ValueError("K RoPE must match K tokens and have one or K heads")
        if k_rope.device != k.device or k_rope.dtype != k.dtype:
            raise ValueError("K RoPE must have the same GPU device and dtype as K")
        rope_dim = k_rope.shape[2]
        rope_stride = (
            k_rope.stride(0),
            0 if k_rope.shape[1] == 1 else k_rope.stride(1),
            k_rope.stride(2),
        )
        k_shape = (*k.shape[:2], k.shape[2] + rope_dim)
    output_shapes = (tuple(q.shape), k_shape, tuple(v.shape))
    shapes = tuple(shape[1:] for shape in output_shapes)
    strides = tuple(x.stride() for x in tensors)
    skip_q = quantized_q is not None
    if skip_q:
        cached, cached_scale = quantized_q
        if (
            cached.shape != q.shape
            or cached.device != q.device
            or cached.dtype != torch.float8_e4m3fn
        ):
            raise ValueError("Cached FP8 Q must match this forward's Q tensor")
    outputs = tuple(
        quantized_q[0]
        if i == 0 and skip_q
        else torch.empty(shape, device=q.device, dtype=torch.float8_e4m3fn)
        for i, shape in enumerate(output_shapes)
    )
    scales = torch.empty(5, device=q.device, dtype=torch.float32)
    sizes = tuple(x.numel() for x in outputs)
    n = max(sizes[1:] if skip_q else sizes)
    single_pass = n <= 8192
    block = (
        max(256, triton.next_power_of_2(n))
        if single_pass
        else (8192 if n >= 4 * 1024 * 1024 else 4096)
    )
    parts = min(128, triton.next_power_of_2(triton.cdiv(n, block))) if n else 1
    partial = torch.empty((3, parts), device=q.device, dtype=torch.float32)
    if not single_pass:
        _fused_qkv_amax[(parts, 2 if skip_q else 3)](
            *tensors,
            partial,
            *sizes,
            shapes,
            strides,
            parts,
            block,
            k_rope,
            rope_dim,
            rope_stride,
            SKIP_Q=skip_q,
            num_warps=4,
        )
    _fused_qkv_quant[(max(1, triton.cdiv(n, block)), 2 if skip_q else 3)](
        *tensors,
        *outputs,
        partial,
        scales,
        *sizes,
        shapes,
        strides,
        parts,
        block,
        single_pass,
        k_rope,
        rope_dim,
        rope_stride,
        SKIP_Q=skip_q,
        num_warps=4,
    )
    return (
        *outputs,
        quantized_q[1] if skip_q else scales[0:1],
        *(scales[i : i + 1] for i in range(1, 5)),
    )
