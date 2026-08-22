# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Q/KV-LoRA RMSNorm + key RoPE CuTe DSL kernel for Kimi-K2.5 NVFP4.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Int64


def qkv_rmsnorm_k_pe_fused(
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
    """
    Perform RMSNorm of the low-rank Q and KV, as well as RoPE on the part of K
    subject to RoPE. All updates are in place.

    The underlying kernel is divided into three groups of blocks:
     - [0, Sp): Calculate Q-side RMSNorm. Each block handles
       3 * lora_dim_kv values.
     - [Sp, 2*Sp): Calculate KV-side RMSNorm. Each block handles
       lora_dim_kv values.
     - [2*Sp, 3*Sp): RoPE on k_pe. Only the first pe_dim // 2 threads are
       active, each handling one rotary pair.
    """
    QKVRMSNormKPeFusedKernel.compile(
        lora_dim_q, lora_dim_kv, pe_dim, float(eps_q), float(eps_kv)
    )(data, positions, k_pe, cos_sin_cache, weights_q, weights_kv)


class QKVRMSNormKPeFusedKernel:
    """Fused Q/KV-LoRA RMSNorm + key RoPE kernel."""

    def __init__(
        self,
        lora_dim_q: int,
        lora_dim_kv: int,
        pe_dim: int,
        eps_q: float,
        eps_kv: float,
    ):
        self.lora_dim_q = lora_dim_q
        self.lora_dim_kv = lora_dim_kv
        self.pe_dim = pe_dim
        self.eps_q = eps_q
        self.eps_kv = eps_kv

    @cute.kernel
    def kernel(
        self,
        data: cute.Tensor,  # (Sp, (2, lora_dim_kv // 2, 4))
        positions: cute.Tensor,  # (Sp,)
        k_pe: cute.Tensor,  # (Sp, (2, pe_dim // 2))
        cos_sin_cache: cute.Tensor,  # (max_position_embeddings, pe_dim)
        weights_q: cute.Tensor,  # (2, lora_dim_q // 2)
        weights_kv: cute.Tensor,  # (2, lora_dim_kv // 2)
    ):
        nwarps = self.lora_dim_kv // 64
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
            w1 = weights_q[None, tid + self.lora_dim_kv // 2].load()
            w2 = weights_q[None, tid + self.lora_dim_kv].load()
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
                sdata[0] = cute.math.rsqrt(ssum / self.lora_dim_q + self.eps_q)

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
                sdata[0] = cute.math.rsqrt(ssum / self.lora_dim_kv + self.eps_kv)

            cute.arch.sync_threads()
            invnorm = sdata[0]
            data[bid - Sp, (None, tid, 3)] = (
                x3 * invnorm
            ).to(cutlass.BFloat16) * w3
        else:
            token_idx = bid - Sp * 2
            half_pe_dim: cutlass.Constexpr = self.pe_dim // 2
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
    def __call__(
        self,
        data: cute.Tensor,
        positions: cute.Tensor,
        k_pe: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        weights_q: cute.Tensor,
        weights_kv: cute.Tensor,
        stream: CUstream,
    ):
        row_stride = cute.assume(data.stride[0], divby=2)
        data = cute.make_tensor(
            data.iterator,
            cute.make_layout(
                (data.shape[0], (2, self.lora_dim_kv // 2, 4)),
                stride=(row_stride, (1, 2, self.lora_dim_kv)),
            ),
        )
        weights_q = cute.make_tensor(
            weights_q.iterator, cute.make_layout((2, self.lora_dim_q // 2))
        )
        weights_kv = cute.make_tensor(
            weights_kv.iterator, cute.make_layout((2, self.lora_dim_kv // 2))
        )
        k_pe = cute.make_tensor(
            k_pe.iterator,
            cute.make_layout(
                (k_pe.shape[0], (2, self.pe_dim // 2)),
                stride=(cute.assume(k_pe.stride[0], divby=2), (1, 2)),
            ),
        )
        grid = (data.shape[0] * 3, 1, 1)
        block = (self.lora_dim_kv // 2, 1, 1)
        self.kernel(
            data,
            positions,
            k_pe,
            cos_sin_cache,
            weights_q,
            weights_kv,
        ).launch(grid=grid, block=block, stream=stream)

    @cache
    @staticmethod
    def compile(
        lora_dim_q: int,
        lora_dim_kv: int,
        pe_dim: int,
        eps_q: float,
        eps_kv: float,
    ):
        if lora_dim_q != lora_dim_kv * 3:
            raise ValueError("This kernel expects lora_dim_q=3*lora_dim_kv.")
        if lora_dim_kv % 64 != 0:
            raise ValueError("lora_dim_kv must be divisible by 64.")
        if pe_dim % 2 != 0:
            raise ValueError("pe_dim must be even.")

        data = cute.runtime.make_fake_tensor(
            BFloat16,
            (cute.sym_int(), lora_dim_q + lora_dim_kv),
            stride=(cute.sym_int64(divisibility=2), 1),
            assumed_align=16,
        )
        positions = cute.runtime.make_fake_tensor(
            Int64,
            (cute.sym_int(),),
            stride=(cute.sym_int64(),),
            assumed_align=16,
        )
        k_pe = cute.runtime.make_fake_tensor(
            BFloat16,
            (cute.sym_int(), pe_dim),
            stride=(cute.sym_int64(divisibility=2), 1),
            assumed_align=16,
        )
        cos_sin_cache = cute.runtime.make_fake_tensor(
            BFloat16,
            (cute.sym_int(), pe_dim),
            stride=(cute.sym_int64(), 1),
            assumed_align=16,
        )
        weights_q = cute.runtime.make_fake_tensor(
            BFloat16,
            (lora_dim_q,),
            stride=(1,),
            assumed_align=16,
        )
        weights_kv = cute.runtime.make_fake_tensor(
            BFloat16,
            (lora_dim_kv,),
            stride=(1,),
            assumed_align=16,
        )

        kernel = QKVRMSNormKPeFusedKernel(
            lora_dim_q, lora_dim_kv, pe_dim, eps_q, eps_kv
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            data,
            positions,
            k_pe,
            cos_sin_cache,
            weights_q,
            weights_kv,
            stream,
            options="--enable-tvm-ffi",
        )
