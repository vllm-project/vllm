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

"""FP8 query quantization and cache-derived MLA projection scales."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _load_tensor(
    X,
    offsets,
    N,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
):
    heads, dim = SHAPE
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
        )
        acc = tl.maximum(acc, tl.abs(x))
    tl.store(Partial + tl.program_id(0), tl.max(acc, 0))


@triton.jit
def _quant_tensor(
    X,
    Y,
    Partial,
    Scale,
    N,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = _load_tensor(X, offsets, N, SHAPE, STRIDE)
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


@triton.jit
def _cache_bound_partials(
    Cache,
    Indices,
    Partial,
    ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    indices = tl.load(Indices + rows, rows < ROWS, other=0).to(tl.int64)
    x = tl.load(
        Cache + indices[:, None] * 576 + cols[None, :],
        (rows[:, None] < ROWS) & (cols[None, :] < 576),
        other=0.0,
    ).to(tl.float32)
    norm_sq = tl.sum(tl.where(cols[None, :] < 512, x * x, 0), 1)
    rope = tl.max(tl.where(cols[None, :] >= 512, tl.abs(x), 0), 1)
    tl.store(Partial + block * 2, tl.max(norm_sq, 0))
    tl.store(Partial + block * 2 + 1, tl.max(rope, 0))


@triton.jit
def _cache_bound_reduce(
    Partial, WeightNorms, CacheScale, Output, PARTS: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    squares = tl.load(Partial + offsets * 2, offsets < PARTS, other=0)
    rope = tl.load(Partial + offsets * 2 + 1, offsets < PARTS, other=0)
    latent_norm = tl.sqrt(tl.max(squares, 0))
    rope_amax = tl.max(rope, 0)
    scale = tl.load(CacheScale)
    k_norm = tl.load(WeightNorms)
    v_norm = tl.load(WeightNorms + 1)
    # Cauchy-Schwarz bounds every channel; allow for FP32 reduction rounding.
    margin = 1.0 + 1.0 / 1024.0
    ks = tl.maximum(latent_norm * k_norm, rope_amax) * scale * margin / 448.0
    vs = latent_norm * v_norm * scale * margin / 448.0
    tl.store(Output, tl.maximum(ks, 1e-6))
    tl.store(Output + 1, tl.maximum(vs, 1e-6))


@triton.jit
def _q_and_cache_partials(
    Q,
    QPartial,
    Cache,
    Indices,
    CachePartial,
    NQ,
    ROWS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    QPARTS: tl.constexpr,
    QBLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    program = tl.program_id(0)
    if program < QPARTS:
        _partial_amax(Q, QPartial, NQ, SHAPE, STRIDE, QPARTS, QBLOCK)
    else:
        tile = program - QPARTS
        rows = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_COLS)
        indices = tl.load(Indices + rows, rows < ROWS, other=0).to(tl.int64)
        x = tl.load(
            Cache + indices[:, None] * 576 + cols[None, :],
            (rows[:, None] < ROWS) & (cols[None, :] < 576),
            other=0.0,
        ).to(tl.float32)
        squares = tl.sum(tl.where(cols[None, :] < 512, x * x, 0), 1)
        rope = tl.max(tl.where(cols[None, :] >= 512, tl.abs(x), 0), 1)
        tl.store(CachePartial + tile * 2, tl.max(squares, 0))
        tl.store(CachePartial + tile * 2 + 1, tl.max(rope, 0))


@triton.jit
def _q_quant_and_cache_reduce(
    Q,
    Q8,
    QPartial,
    QScale,
    CachePartial,
    WeightNorms,
    CacheScale,
    OutputScales,
    NQ,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    QPARTS: tl.constexpr,
    QBLOCK: tl.constexpr,
    CACHE_PARTS: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
):
    _quant_tensor(
        Q,
        Q8,
        QPartial,
        QScale,
        NQ,
        SHAPE,
        STRIDE,
        QPARTS,
        QBLOCK,
        False,
    )
    if tl.program_id(0) == 0:
        _cache_bound_reduce(
            CachePartial,
            WeightNorms,
            CacheScale,
            OutputScales,
            CACHE_PARTS,
            CACHE_BLOCK,
        )


def _validate_query(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] == 0 or q.shape[2] == 0:
        raise ValueError("Q must have shape [tokens, nonzero heads, nonzero dim]")
    if not q.is_cuda or q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Q must be a BF16 or FP16 GPU tensor")


def _validate_cache(cache, cache_scale, indices, weight_norms) -> None:
    if (
        cache.ndim != 2
        or cache.shape[1] != 576
        or cache.dtype != torch.float8_e4m3fn
        or not cache.is_contiguous()
    ):
        raise ValueError("Expected contiguous OCP E4M3 cache [rows,576]")
    if indices.ndim != 1 or indices.dtype != torch.int32 or not indices.is_contiguous():
        raise ValueError("Expected contiguous int32 cache row indices")
    if (
        cache_scale.dtype != torch.float32
        or cache_scale.numel() != 1
        or not cache_scale.is_contiguous()
        or weight_norms.dtype != torch.float32
        or weight_norms.shape != (2,)
        or not weight_norms.is_contiguous()
    ):
        raise ValueError("Expected scalar FP32 cache scale and two FP32 weight norms")
    if not cache.is_cuda or any(
        t.device != cache.device for t in (cache_scale, indices, weight_norms)
    ):
        raise ValueError("Cache preparation tensors must share a GPU")


def quantize_query(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize a strided BF16 query to E4M3."""
    _validate_query(q)
    n = q.numel()
    single = n <= 8192
    block = (
        max(256, triton.next_power_of_2(n))
        if single
        else (8192 if n >= 4 * 1024 * 1024 else 4096)
    )
    parts = min(128, triton.next_power_of_2(triton.cdiv(n, block))) if n else 1
    partial = torch.empty(parts, device=q.device, dtype=torch.float32)
    scale = torch.empty(1, device=q.device, dtype=torch.float32)
    output = torch.empty(q.shape, device=q.device, dtype=torch.float8_e4m3fn)
    if not single:
        _partial_amax[(parts,)](
            q,
            partial,
            n,
            tuple(q.shape[1:]),
            q.stride(),
            parts,
            block,
            num_warps=4,
        )
    _quant_tensor[(max(1, triton.cdiv(n, block)),)](
        q,
        output,
        partial,
        scale,
        n,
        tuple(q.shape[1:]),
        q.stride(),
        parts,
        block,
        single,
        num_warps=4,
    )
    return output, scale


def cached_kv_scales(cache, cache_scale, indices, weight_norms):
    """Bound projected K/V from the selected cached rows and native weights."""
    _validate_cache(cache, cache_scale, indices, weight_norms)
    rows = indices.numel()
    if not rows:
        return torch.full((2,), 1e-6, device=cache.device, dtype=torch.float32)
    parts = triton.cdiv(rows, 16)
    partial = torch.empty((parts, 2), device=cache.device, dtype=torch.float32)
    output = torch.empty(2, device=cache.device, dtype=torch.float32)
    _cache_bound_partials[(parts,)](
        cache,
        indices,
        partial,
        rows,
        16,
        1024,
        num_warps=4,
    )
    _cache_bound_reduce[(1,)](
        partial,
        weight_norms,
        cache_scale,
        output,
        parts,
        triton.next_power_of_2(parts),
        num_warps=4,
    )
    return output


def prepare_fp8_context(q, cache, cache_scale, indices, weight_norms):
    """Prepare FP8 Q and cache-derived K/V scales in two GPU launches."""
    _validate_query(q)
    _validate_cache(cache, cache_scale, indices, weight_norms)
    if q.device != cache.device:
        raise ValueError("Q and cache must share a GPU")
    n = q.numel()
    block = 8192 if n >= 4 * 1024 * 1024 else 4096
    qparts = max(1, min(128, triton.next_power_of_2(triton.cdiv(n, block))))
    cparts = max(1, triton.cdiv(indices.numel(), 16))
    qpartial = torch.empty(qparts, device=q.device, dtype=torch.float32)
    cpartial = torch.empty((cparts, 2), device=q.device, dtype=torch.float32)
    qscale = torch.empty(1, device=q.device, dtype=torch.float32)
    cscales = torch.empty(2, device=q.device, dtype=torch.float32)
    q8 = torch.empty(q.shape, device=q.device, dtype=torch.float8_e4m3fn)
    _q_and_cache_partials[(qparts + cparts,)](
        q,
        qpartial,
        cache,
        indices,
        cpartial,
        n,
        indices.numel(),
        tuple(q.shape[1:]),
        q.stride(),
        qparts,
        block,
        16,
        1024,
        num_warps=4,
    )
    _q_quant_and_cache_reduce[(max(1, triton.cdiv(n, block)),)](
        q,
        q8,
        qpartial,
        qscale,
        cpartial,
        weight_norms,
        cache_scale,
        cscales,
        n,
        tuple(q.shape[1:]),
        q.stride(),
        qparts,
        block,
        cparts,
        triton.next_power_of_2(cparts),
        num_warps=4,
    )
    return q8, qscale, cscales
