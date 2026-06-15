# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Q/KV-LoRA RMSNorm + key RoPE CuTe DSL kernel for Kimi-K2.5 NVFP4.

This Blackwell (SM10x) kernel runs once per layer, right after the fused QKV-A
projection, fusing the Q-LoRA RMSNorm, KV-LoRA RMSNorm, and the key (``k_pe``)
RoPE into a single launch (all updates in place).

The public entry point is :func:`_run_kimik25_qkv_rmsnorm_k_pe_fused`
(torch tensors in, in-place out). Compilation is cached per constexpr
configuration by :func:`_compile_qkv_rmsnorm_k_pe_fused`, which builds fake
tensors with symbolic shapes/strides and calls ``cute.compile``, following the
convention in ``vllm/v1/attention/ops/deepseek_v4_ops``. The compiled executor
is then called directly with torch tensors and sources its launch stream from
the TVM-FFI environment.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Int64

from .cutedsl_utils import _fake, _fake_stream


@cute.kernel
def kimik25_qkv_rmsnorm_k_pe_fused_kernel(
    data: cute.Tensor,  # (Sp, (2, lora_dim_kv // 2, 4))
    positions: cute.Tensor,  # (Sp,)
    k_pe: cute.Tensor,  # (Sp, (2, pe_dim // 2))
    cos_sin_cache: cute.Tensor,  # (max_position_embeddings, pe_dim)
    weights_q: cute.Tensor,  # (2, lora_dim_q // 2)
    weights_kv: cute.Tensor,  # (2, lora_dim_kv // 2)
    lora_dim_q: cutlass.Constexpr,  # must be lora_dim_kv * 3
    lora_dim_kv: cutlass.Constexpr,
    pe_dim: cutlass.Constexpr,
    eps_q: cutlass.Constexpr,
    eps_kv: cutlass.Constexpr,
):
    """
    This kernel performs RMSNorm of the low-rank Q and KV, as well
    as doing RoPE on the part of K subject to RoPE. All updates are in place.

    This assumes that lora_dim_q = 3 * lora_dim_kv.

    This is divided into three groups of blocks:
     - [0, Sp): Calculate Q-side RMSNorm. Each block handles 3 * lora_dim_kv values
     - [Sp, 2*Sp): Calculate KV-side RMSNorm. Each block handles lora_dim_kv values
     - [2*Sp, 3*Sp): RoPE on k_pe. Only the first pe_dim // 2 threads are active,
       each handling one rotary pair.
    """
    nwarps = lora_dim_kv // 64
    allocator = cutlass.utils.SmemAllocator()
    sdata = allocator.allocate_tensor(
        cutlass.Float32,
        layout=cute.make_layout(nwarps),
        byte_alignment=16,
        swizzle=None,
    )

    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    Sp = data.shape[0]
    if bid < Sp:
        x0 = data[bid, (None, tid, 0)].load().to(cutlass.Float32)
        x1 = data[bid, (None, tid, 1)].load().to(cutlass.Float32)
        x2 = data[bid, (None, tid, 2)].load().to(cutlass.Float32)
        w0 = weights_q[None, tid].load()
        w1 = weights_q[None, tid + lora_dim_kv // 2].load()
        w2 = weights_q[None, tid + lora_dim_kv].load()
        sum = x0 * x0 + x1 * x1 + x2 * x2
        sum = sum[0] + sum[1]

        sum = cute.arch.warp_reduction_sum(sum, threads_in_group=32)
        if tid % 32 == 0:
            sdata[tid // 32] = sum

        cute.arch.sync_threads()
        ssum: cutlass.Float32 = 0.0
        if tid < nwarps:
            ssum = sdata[tid]

        ssum = cute.arch.warp_reduction_sum(ssum, threads_in_group=nwarps)
        if tid == 0:
            sdata[0] = cute.math.rsqrt(ssum / lora_dim_q + eps_q)

        cute.arch.sync_threads()
        invnorm = sdata[0]
        data[bid, (None, tid, 0)] = (x0 * invnorm).to(cutlass.BFloat16) * w0
        data[bid, (None, tid, 1)] = (x1 * invnorm).to(cutlass.BFloat16) * w1
        data[bid, (None, tid, 2)] = (x2 * invnorm).to(cutlass.BFloat16) * w2
    elif bid < Sp * 2:
        x3 = data[bid - Sp, (None, tid, 3)].load().to(cutlass.Float32)
        w3 = weights_kv[None, tid].load()
        sum = x3 * x3
        sum = sum[0] + sum[1]

        sum = cute.arch.warp_reduction_sum(sum, threads_in_group=32)
        if tid % 32 == 0:
            sdata[tid // 32] = sum

        cute.arch.sync_threads()
        ssum: cutlass.Float32 = 0.0
        if tid < nwarps:
            ssum = sdata[tid]

        ssum = cute.arch.warp_reduction_sum(ssum, threads_in_group=nwarps)
        if tid == 0:
            sdata[0] = cute.math.rsqrt(ssum / lora_dim_kv + eps_kv)

        cute.arch.sync_threads()
        invnorm = sdata[0]
        data[bid - Sp, (None, tid, 3)] = (x3 * invnorm).to(cutlass.BFloat16) * w3
    else:
        token_idx = bid - Sp * 2
        half_pe_dim: cutlass.Constexpr = pe_dim // 2
        if tid < half_pe_dim:
            pos = positions[token_idx]
            cos = cos_sin_cache[pos, tid].to(cutlass.Float32)
            sin = cos_sin_cache[pos, tid + half_pe_dim].to(cutlass.Float32)
            in_scratch = cute.make_rmem_tensor(2, dtype=cutlass.BFloat16)
            cute.autovec_copy(k_pe[token_idx, (None, tid)], in_scratch)
            a = in_scratch[0].to(cutlass.Float32)
            b = in_scratch[1].to(cutlass.Float32)
            in_scratch[0] = (a * cos - b * sin).to(cutlass.BFloat16)
            in_scratch[1] = (a * sin + b * cos).to(cutlass.BFloat16)
            cute.autovec_copy(in_scratch, k_pe[token_idx, (None, tid)])


@cute.jit
def kimik25_qkv_rmsnorm_k_pe_fused(
    data: cute.Tensor,
    positions: cute.Tensor,
    k_pe: cute.Tensor,
    cos_sin_cache: cute.Tensor,
    weights_q: cute.Tensor,
    weights_kv: cute.Tensor,
    lora_dim_q: cutlass.Constexpr,
    lora_dim_kv: cutlass.Constexpr,
    pe_dim: cutlass.Constexpr,
    eps_q: cutlass.Constexpr,
    eps_kv: cutlass.Constexpr,
    stream: CUstream,
):
    row_stride = cute.assume(data.stride[0], divby=2)
    data = cute.make_tensor(
        data.iterator,
        cute.make_layout(
            (data.shape[0], (2, lora_dim_kv // 2, 4)),
            stride=(row_stride, (1, 2, lora_dim_kv)),
        ),
    )
    weights_q = cute.make_tensor(
        weights_q.iterator, cute.make_layout((2, lora_dim_q // 2))
    )
    weights_kv = cute.make_tensor(
        weights_kv.iterator, cute.make_layout((2, lora_dim_kv // 2))
    )
    k_pe = cute.make_tensor(
        k_pe.iterator,
        cute.make_layout(
            (k_pe.shape[0], (2, pe_dim // 2)),
            stride=(cute.assume(k_pe.stride[0], divby=2), (1, 2)),
        ),
    )
    grid = (data.shape[0] * 3, 1, 1)
    block = (lora_dim_kv // 2, 1, 1)
    kimik25_qkv_rmsnorm_k_pe_fused_kernel(
        data,
        positions,
        k_pe,
        cos_sin_cache,
        weights_q,
        weights_kv,
        lora_dim_q,
        lora_dim_kv,
        pe_dim,
        eps_q,
        eps_kv,
    ).launch(grid=grid, block=block, stream=stream)


@cache
def _compile_qkv_rmsnorm_k_pe_fused(
    lora_dim_q: int,
    lora_dim_kv: int,
    pe_dim: int,
    eps_q: float,
    eps_kv: float,
):
    data = _fake(
        BFloat16,
        (cute.sym_int(), lora_dim_q + lora_dim_kv),
        (cute.sym_int64(divisibility=2), 1),
        align=16,
    )
    positions = _fake(Int64, (cute.sym_int(),), (cute.sym_int64(),), align=16)
    k_pe = _fake(
        BFloat16,
        (cute.sym_int(), pe_dim),
        (cute.sym_int64(divisibility=2), 1),
        align=16,
    )
    cos_sin_cache = _fake(
        BFloat16, (cute.sym_int(), pe_dim), (cute.sym_int64(), 1), align=16
    )
    weights_q = _fake(BFloat16, (lora_dim_q,), (1,), align=16)
    weights_kv = _fake(BFloat16, (lora_dim_kv,), (1,), align=16)
    return cute.compile(
        kimik25_qkv_rmsnorm_k_pe_fused,
        data,
        positions,
        k_pe,
        cos_sin_cache,
        weights_q,
        weights_kv,
        lora_dim_q,
        lora_dim_kv,
        pe_dim,
        eps_q,
        eps_kv,
        _fake_stream(),
        options="--enable-tvm-ffi",
    )


def _run_kimik25_qkv_rmsnorm_k_pe_fused(
    *,
    data: torch.Tensor,
    positions: torch.Tensor,
    k_pe: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights_q: torch.Tensor,
    weights_kv: torch.Tensor,
    lora_dim_q: int,
    lora_dim_kv: int,
    pe_dim: int,
    eps_q: float,
    eps_kv: float,
) -> None:
    _compile_qkv_rmsnorm_k_pe_fused(
        lora_dim_q, lora_dim_kv, pe_dim, float(eps_q), float(eps_kv)
    )(data, positions, k_pe, cos_sin_cache, weights_q, weights_kv)
