# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton W8A8 GEMM with per-row MX32 weight scales.

Uses block-scaled FP8 accumulation as in SGLang's Hopper FP8 path (#39657).
Activation quantization is part of the operation and changes W8A16 numerics.
"""

from functools import cache

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@cache
def _num_compute_units(device_index: int) -> int:
    return current_platform.num_compute_units(device_index)


@triton.jit(do_not_specialize=["M"])
def _triton_mxfp8_gemm(
    A,
    W,
    AS,
    WS,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pm = tl.program_id(0) // tl.cdiv(N, BN)
    pn = tl.program_id(0) % tl.cdiv(N, BN)
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, 32)
    acc = tl.full((BM, BN), 0, tl.float32)
    for group in range(K // 32):
        a = tl.load(
            A + m[:, None] * K + group * 32 + k[None, :],
            m[:, None] < M,
            0.0,
        )
        w = tl.load(
            W + n[None, :] * K + group * 32 + k[:, None],
            n[None, :] < N,
            0.0,
        )
        sa = tl.load(AS + m * (K // 32) + group, m < M, 0.0)
        sw = tl.load(WS + n * (K // 32) + group, n < N, 0.0)
        acc += tl.dot(a, w) * sa[:, None] * sw[None, :]
    tl.store(
        C + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _triton_mxfp8_gemv(
    X,
    W,
    S,
    Out,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # Retain row addressing to preserve the tuned Triton lowering.
    tile = tl.program_id(0)
    n = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    row = tile // tl.cdiv(N, BN)
    acc = tl.full((BN, BK), 0, tl.float32)
    for i in range(tl.cdiv(K, BK)):
        k = i * BK + tl.arange(0, BK)
        x = tl.load(X + row * K + k, k < K, 0.0).to(tl.float32)
        groups = tl.reshape(x, (BK // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(groups), 1), 1e-10)
        scale = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
        quantized = (groups / scale[:, None]).to(tl.float8e4nv)
        activation = tl.reshape(quantized.to(tl.float32) * scale[:, None], (BK,))
        weight = tl.load(
            W + n[:, None] * K + k[None, :],
            (n[:, None] < N) & (k[None, :] < K),
            0.0,
        ).to(tl.float32)
        weight_scale = tl.load(
            S + n[:, None] * (K // 32) + k[None, :] // 32,
            (n[:, None] < N) & (k[None, :] < K),
            0.0,
        )
        acc += weight * weight_scale * activation[None, :]
    tl.store(Out + row * N + n, tl.sum(acc, 1), n < N)


@triton.jit(do_not_specialize=["M"])
def _triton_mxfp8_small_gemm(
    X,
    W,
    S,
    Out,
    Partial,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pn = pid % tl.cdiv(N, BN)
    split = (pid // tl.cdiv(N, BN)) % SPLIT_K
    pm = pid // (tl.cdiv(N, BN) * SPLIT_K)
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, 32)
    acc = tl.full((BN, BM), 0, tl.float32)
    groups = tl.cdiv(K // 32, SPLIT_K)
    for i in range(groups):
        group = split * groups + i
        x = tl.load(
            X + m[:, None] * K + group * 32 + k[None, :],
            (m[:, None] < M) & (group < K // 32),
            0.0,
        ).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-10)
        a_scale = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
        a = (x / a_scale[:, None]).to(tl.float8e4nv)
        w = tl.load(
            W + n[:, None] * K + group * 32 + k[None, :],
            (n[:, None] < N) & (group < K // 32),
            0.0,
        )
        w_scale = tl.load(
            S + n * (K // 32) + group,
            (n < N) & (group < K // 32),
            0.0,
        )
        acc += tl.dot(w, tl.trans(a)) * w_scale[:, None] * a_scale[None, :]
    offsets = m[:, None] * N + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(Out + offsets, tl.trans(acc), mask)
    else:
        tl.store(Partial + split.to(tl.int64) * M * N + offsets, tl.trans(acc), mask)


@triton.jit(do_not_specialize=["M"])
def _triton_mxfp8_chunked_gemm(
    X,
    W,
    S,
    Out,
    Partial,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pn = pid % tl.cdiv(N, BN)
    split = (pid // tl.cdiv(N, BN)) % SPLIT_K
    pm = pid // (tl.cdiv(N, BN) * SPLIT_K)
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, 128)
    g = tl.arange(0, 4)
    steps = tl.cdiv(K, SPLIT_K * 128)
    acc = tl.full((BN, BM), 0, tl.float32)
    for i in range(steps):
        base = (split * steps + i) * 128
        offsets = base + k
        groups = base // 32 + g
        x = tl.load(
            X + m[:, None] * K + offsets[None, :],
            (m[:, None] < M) & (offsets[None, :] < K),
            0.0,
        ).to(tl.float32)
        x = tl.reshape(x, (BM, 4, 32))
        amax = tl.maximum(tl.max(tl.abs(x), 2), 1e-10)
        sa = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
        a = (x / sa[:, :, None]).to(tl.float8e4nv)
        w = tl.load(
            W + n[:, None] * K + offsets[None, :],
            (n[:, None] < N) & (offsets[None, :] < K),
            0.0,
        )
        sw = tl.load(
            S + n[:, None] * (K // 32) + groups[None, :],
            (n[:, None] < N) & (groups[None, :] < K // 32),
            0.0,
        )
        # Read four adjacent MX32 groups together, retaining independent scales.
        a_groups = tl.reshape(tl.trans(a, (0, 2, 1)), (BM, 32, 2, 2))
        w_groups = tl.reshape(
            tl.trans(tl.reshape(w, (BN, 4, 32)), (0, 2, 1)), (BN, 32, 2, 2)
        )
        ae, ao = tl.split(a_groups)
        a0, a2 = tl.split(ae)
        a1, a3 = tl.split(ao)
        we, wo = tl.split(w_groups)
        w0, w2 = tl.split(we)
        w1, w3 = tl.split(wo)
        sae, sao = tl.split(tl.reshape(sa, (BM, 2, 2)))
        sa0, sa2 = tl.split(sae)
        sa1, sa3 = tl.split(sao)
        swe, swo = tl.split(tl.reshape(sw, (BN, 2, 2)))
        sw0, sw2 = tl.split(swe)
        sw1, sw3 = tl.split(swo)
        acc += tl.dot(w0, tl.trans(a0)) * sw0[:, None] * sa0[None, :]
        acc += tl.dot(w1, tl.trans(a1)) * sw1[:, None] * sa1[None, :]
        acc += tl.dot(w2, tl.trans(a2)) * sw2[:, None] * sa2[None, :]
        acc += tl.dot(w3, tl.trans(a3)) * sw3[:, None] * sa3[None, :]
    offsets = m[:, None] * N + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(Out + offsets, tl.trans(acc), mask)
    else:
        tl.store(Partial + split.to(tl.int64) * M * N + offsets, tl.trans(acc), mask)


@triton.jit
def _triton_mxfp8_reduce(
    Partial,
    Out,
    MN,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K).to(tl.int64)
    values = tl.load(
        Partial + splits[:, None] * MN + offsets[None, :],
        offsets[None, :] < MN,
        0.0,
    )
    tl.store(Out + offsets, tl.sum(values, 0), offsets < MN)


def triton_mxfp8_linear(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Apply an MX32 linear projection with FP8-quantized activations."""
    if weight.ndim != 2 or x.ndim < 2:
        raise ValueError(
            "Expected a matrix weight and activations with at least 2 axes"
        )
    n, k = weight.shape
    if not n or not k or k % 32 or x.shape[-1] != k:
        raise ValueError("MX32 requires matching positive K divisible by 32 and N > 0")
    if scales.shape != (n, k // 32):
        raise ValueError("Expected one weight scale per output row and 32 K channels")
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.float8_e4m3fn
        or scales.dtype != torch.float32
    ):
        raise ValueError("Expected BF16 activations, E4M3 weights, and FP32 scales")
    if not weight.is_contiguous() or not scales.is_contiguous():
        raise ValueError("Weight and scale tensors must be contiguous")
    if (
        x.device.type != "cuda"
        or weight.device != x.device
        or scales.device != x.device
    ):
        raise ValueError("All operands must be on the same CUDA device")
    m = x.numel() // k
    if max(m * k, m * n, n * k) >= 2**31:
        raise ValueError("Triton MXFP8 requires tensor offsets below 2**31")
    x2d = x.reshape(-1, k).contiguous()
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 1:
        bn = 4 if n < 2048 else 8
        bk = min(1024, triton.next_power_of_2(k))
        if k > bk and k % bk:
            bk = max(256, k & -k)
        num_warps = 8 if n < 1024 and k >= 1024 else 4
        _triton_mxfp8_gemv[(triton.cdiv(n, bn),)](
            x2d, weight, scales, out, n, k, bn, bk, num_warps=num_warps
        )
    elif m and (m <= 32 or k >= 2 * n or (m <= 128 and n <= 8192)):
        if m <= 32:
            bm = max(8, triton.next_power_of_2(m))
            long_k = k >= 2 * n
            bn = 128 if n > 8192 or (bm <= 16 and n >= 1024 and k > 1024) else 64
            max_splits = 8 if n > 8192 else 16
            if long_k and (bm <= 16 or n < 1024):
                max_splits = 32
            split_k = min(max_splits, triton.next_power_of_2(triton.cdiv(k, 128)))
        else:
            long_k = k >= 2 * n
            bm = 64 if long_k else 32
            bn = 64 if n < 1024 or (not long_k and m >= 64) else 128
            tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
            target = _num_compute_units(x.device.index or 0) * (
                16 if long_k and n < 1024 else 4
            )
            split_k = 1 << (max(1, target // tiles).bit_length() - 1)
            if split_k == 2:
                split_k = 4
            if not long_k and m >= 64 and k > 1024:
                split_k = max(4, split_k)
            split_k = min(16, split_k, triton.next_power_of_2(triton.cdiv(k, 128)))
        short_k = (
            8 < m <= 128
            and k <= 320
            and triton.cdiv(m, bm) * triton.cdiv(n, 64)
            >= _num_compute_units(x.device.index or 0)
        )
        if short_k:
            bn = 64
            split_k = 1
        partial = (
            torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
            if split_k > 1
            else out
        )
        chunked = m <= 8 and k <= 2048
        if chunked and k > 1024 and n >= 1024:
            bn = 128
        kernel = _triton_mxfp8_chunked_gemm if chunked else _triton_mxfp8_small_gemm
        kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn) * split_k,)](
            x2d,
            weight,
            scales,
            out,
            partial,
            m,
            n,
            k,
            bm,
            bn,
            split_k,
            num_warps=4,
            num_stages=3 if not chunked and (short_k or bm >= 32 or n > 8192) else 2,
        )
        if split_k > 1:
            _triton_mxfp8_reduce[(triton.cdiv(m * n, 256),)](
                partial, out, m * n, split_k, 256
            )
    elif m:
        a, sa = per_token_group_quant_fp8(x2d, 32, use_ue8m0=True)
        _triton_mxfp8_gemm[(triton.cdiv(m, 64) * triton.cdiv(n, 128),)](
            a,
            weight,
            sa,
            scales,
            out,
            m,
            n,
            k,
            64,
            128,
            num_warps=4,
            num_stages=3,
        )
    return out.reshape(*x.shape[:-1], n)
