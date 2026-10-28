# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dynamic-M direct BF16 GEMM adapted from FlashInfer and the Raon kit.

Unlike FlashInfer run_direct_dense, this accepts M > 32 and partial row tiles.
Source: flashinfer/gemm/kernels/dense_bf16_gemm_direct.py (Apache-2.0).
"""

from __future__ import annotations

import dataclasses
import functools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch as _torch
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack

_COMPILE_OPTIONS = "--ptxas-options -maxrregcount=64"
_SUPPORTED_BLOCK_SIZES = (32, 64, 96, 128, 192, 256, 384)
_SUPPORTED_OUTPUTS_PER_BLOCK = (1, 2, 4)
_VECTOR_WIDTH = 8


@dataclasses.dataclass(frozen=True)
class DynMTactic:
    """One dynamic-M specialisation (same fields as the donor's DirectTactic;
    rows_per_block no longer has to divide M)."""

    block_size: int
    outputs_per_block: int
    rows_per_block: int


_SUPPORTED_ROWS_PER_BLOCK = (1, 2, 4, 8, 16)


def validate_dynm_tactic(tactic: DynMTactic, n: int, k: int) -> None:
    """Raise ``ValueError`` when *tactic* cannot serve ``(*, n, k)``."""
    if tactic.block_size not in _SUPPORTED_BLOCK_SIZES:
        raise ValueError(f"unsupported block_size={tactic.block_size}")
    if tactic.outputs_per_block not in _SUPPORTED_OUTPUTS_PER_BLOCK:
        raise ValueError(f"unsupported outputs_per_block={tactic.outputs_per_block}")
    if tactic.rows_per_block not in _SUPPORTED_ROWS_PER_BLOCK:
        raise ValueError(f"unsupported rows_per_block={tactic.rows_per_block}")
    if n <= 0 or n % tactic.outputs_per_block:
        raise ValueError(
            f"N={n} must be divisible by outputs_per_block={tactic.outputs_per_block}"
        )
    k_tile = tactic.block_size * _VECTOR_WIDTH
    if k <= 0 or k % k_tile:
        raise ValueError(
            f"K={k} must be divisible by block_size×{_VECTOR_WIDTH}={k_tile}"
        )


class _DirectDenseGemmKernelDynM:
    """Donor ``_DirectDenseGemmKernel`` with a runtime M (see module doc)."""

    def __init__(self, *, element_type, k_extent: int, tactic: DynMTactic) -> None:
        validate_dynm_tactic(tactic, tactic.outputs_per_block, k_extent)
        self.element_type = element_type
        self.rows_per_block = tactic.rows_per_block
        self.k_extent = k_extent
        self.block_size = tactic.block_size
        self.outputs_per_block = tactic.outputs_per_block
        self.vector_width = _VECTOR_WIDTH
        self.num_warps = tactic.block_size // cute.arch.WARP_SIZE
        self.num_k_tiles = k_extent // (tactic.block_size * _VECTOR_WIDTH)

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        stream: _cuda.CUstream,
    ) -> None:
        n = cute.size(gB, mode=[0])
        m = cute.size(gA, mode=[0])  # CHANGED: runtime M (donor: self.num_rows)
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(gA, gB, gC, copy_a, copy_b).launch(
            grid=[
                cute.ceil_div(n, self.outputs_per_block),
                cute.ceil_div(m, self.rows_per_block),  # CHANGED: ceil, tail guarded
                1,
            ],
            block=[self.block_size, 1, 1],
            smem=self.rows_per_block * self.outputs_per_block * self.num_warps * 4,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, block_m, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        num_rows: cutlass.Constexpr = self.rows_per_block
        outputs_per_block: cutlass.Constexpr = self.outputs_per_block
        vector_width: cutlass.Constexpr = self.vector_width
        block_size: cutlass.Constexpr = self.block_size
        num_warps: cutlass.Constexpr = self.num_warps
        num_k_tiles: cutlass.Constexpr = self.num_k_tiles

        m_total = cute.size(gA, mode=[0])  # CHANGED: runtime M for the tail guard

        # FP32 accumulators in registers — one per (row, output) pair.
        acc = cute.make_rmem_tensor(
            cute.make_layout(
                (num_rows, outputs_per_block), stride=(outputs_per_block, 1)
            ),
            cutlass.Float32,
        )
        acc.fill(0.0)

        n_base = block_idx * outputs_per_block
        m_base = block_m * num_rows

        # Vectorised view of A and B for G2R copy.
        gA_vec = cute.logical_divide(gA, (None, vector_width))
        gB_vec = cute.logical_divide(gB, (None, vector_width))
        tA_all = cute.logical_divide(gA_vec, (None, (None, block_size)))
        tB_all = cute.logical_divide(gB_vec, (None, (None, block_size)))
        tA = tA_all[None, (None, (tidx, None))]

        # Prefetch the entire K dimension of each requested B (weight) row
        # into registers before accumulation begins.
        b_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (outputs_per_block, num_k_tiles, vector_width),
                stride=(num_k_tiles * vector_width, vector_width, 1),
            ),
            self.element_type,
        )
        for ni in cutlass.range_constexpr(outputs_per_block):
            tB = tB_all[n_base + ni, (None, (tidx, None))]
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                cute.copy(copy_b, tB[None, k_tile], b_regs[ni, k_tile, None])

        # Stream each A row from global memory and accumulate against cached B.
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_k_tiles, vector_width), stride=(vector_width, 1)),
            self.element_type,
        )
        for mi in cutlass.range_constexpr(num_rows):
            # CHANGED: clamp the row so the tail block never reads past M; the
            # clamped rows' results are discarded by the store guard below.
            row = cutlass.min(m_base + mi, m_total - 1)
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                cute.copy(copy_a, tA[row, None, k_tile], a_regs[k_tile, None])
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                for vi in cutlass.range_constexpr(vector_width):
                    a_value = a_regs[k_tile, vi].to(cutlass.Float32)
                    for ni in cutlass.range_constexpr(outputs_per_block):
                        acc[mi, ni] = acc[mi, ni] + a_value * b_regs[ni, k_tile, vi].to(
                            cutlass.Float32
                        )

        # Warp-level reduction then cross-warp reduction via shared memory.
        for mi in cutlass.range_constexpr(num_rows):
            for ni in cutlass.range_constexpr(outputs_per_block):
                acc[mi, ni] = cute.arch.warp_reduction_sum(acc[mi, ni])

        smem_layout = cute.make_layout(
            (num_rows, outputs_per_block, num_warps),
            stride=(outputs_per_block * num_warps, num_warps, 1),
        )
        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(cutlass.Float32, smem_layout, byte_alignment=16)
        with cute.arch.elect_one():
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    partials[mi, ni, warp_idx] = acc[mi, ni]

        cute.arch.sync_threads()
        if tidx == 0:
            for mi in cutlass.range_constexpr(num_rows):
                if m_base + mi < m_total:  # CHANGED: tail store guard
                    for ni in cutlass.range_constexpr(outputs_per_block):
                        total = cutlass.Float32(0.0)
                        for warp in cutlass.range_constexpr(num_warps):
                            total = total + partials[mi, ni, warp]
                        gC[m_base + mi, n_base + ni] = total.to(self.element_type)


def _wrap(tensor: _torch.Tensor, dynamic_m: bool):
    t = from_dlpack(tensor, assumed_align=32)
    if dynamic_m:
        t = t.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    return t


@functools.cache
def _get_compiled_dynm_kernel(dtype: _torch.dtype, n: int, k: int, tactic: DynMTactic):
    """JIT-compile once per (n, k, tactic); M is a runtime extent."""
    if dtype != _torch.bfloat16:
        raise ValueError(f"direct GEMM supports BF16; got {dtype}")
    kernel = _DirectDenseGemmKernelDynM(
        element_type=cutlass.BFloat16, k_extent=k, tactic=tactic
    )
    m_repr = 64  # any M: mode 0 is marked dynamic
    tensors = (
        _wrap(_torch.empty((m_repr, k), dtype=dtype, device="cuda"), True),
        _wrap(_torch.empty((n, k), dtype=dtype, device="cuda"), False),
        _wrap(_torch.empty((m_repr, n), dtype=dtype, device="cuda"), True),
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(kernel, *tensors, stream, options=_COMPILE_OPTIONS)


def run_direct_dense_dynm(
    a: _torch.Tensor,
    b: _torch.Tensor,
    out: _torch.Tensor,
    tactic: DynMTactic,
) -> _torch.Tensor:
    """``out = A[M,K] @ B[K,N]`` for any M >= 1 (b column-major, as donor)."""
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("direct GEMM accepts only 2D tensors")
    if a.dtype != _torch.bfloat16 or b.dtype != a.dtype or out.dtype != a.dtype:
        raise ValueError("a, b, and out must share BF16 dtype")
    if not a.is_contiguous() or not b.T.is_contiguous() or not out.is_contiguous():
        raise ValueError("requires row-major a and out, column-major b")
    if any(t.data_ptr() % 32 for t in (a, b, out)):
        raise ValueError("a, b, and out must be 32-byte aligned")
    m, k = a.shape
    n = b.shape[1]
    if b.shape[0] != k or out.shape != (m, n) or m < 1:
        raise ValueError(
            f"bad shapes a={tuple(a.shape)} b={tuple(b.shape)} out={tuple(out.shape)}"
        )
    validate_dynm_tactic(tactic, n, k)
    compiled = _get_compiled_dynm_kernel(a.dtype, n, k, tactic)
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    compiled(_wrap(a, True), _wrap(b.T, False), _wrap(out, True), stream)
    return out


__all__ = ["DynMTactic", "run_direct_dense_dynm", "validate_dynm_tactic"]
