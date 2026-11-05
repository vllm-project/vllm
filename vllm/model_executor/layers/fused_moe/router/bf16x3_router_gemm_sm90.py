# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuteDSL BF16x3 router GEMM for Hopper (SM90).

Computes fp32 ``x @ w.T`` for bf16 ``x`` [M, K] and fp32 router weights ``w``
[E, K]. The MMA warpgroup splits each fp32 weight fragment into three bf16
terms in registers and issues register-A wgmma with experts as the MMA M
dimension. The first term accumulates per K-tile and is promoted into an fp32
register accumulator; the other two accumulate across K. Split-K reduces in
rank order over the cluster's distributed shared memory.
"""

import math
from dataclasses import dataclass
from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from quack.compile_utils import make_fake_tensor

from vllm.model_executor.warmup.jit_warmup import WarmupIntRange, kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op

__all__ = ["bf16x3_router_gemm"]

_BLOCK_E = 64  # experts per CTA: one wgmma M


@dsl_user_op
def _split_fp32x2_to_3xbf16x2(
    w0: Float32,
    w1: Float32,
    *,
    loc=None,
    ip=None,
) -> tuple[Uint32, Uint32, Uint32]:
    # Each term keeps the top 16 bits of what is left. The remainders are exact
    # and three 8-bit significands cover fp32's 24, so the terms sum back to w.
    # Returns bf16x2 registers for the first, second and third terms (low
    # half = w0).
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32()]),
        [w0.ir_value(loc=loc, ip=ip), w1.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 u0, u1, m0, m1, r0, r1, n0, n1, l0, l1;\n\t"
        "mov.b32 u0, $3;\n\t"
        "mov.b32 u1, $4;\n\t"
        "and.b32 m0, u0, 0xffff0000;\n\t"
        "and.b32 m1, u1, 0xffff0000;\n\t"
        "sub.f32 r0, u0, m0;\n\t"
        "sub.f32 r1, u1, m1;\n\t"
        "prmt.b32 $0, u0, u1, 0x7632;\n\t"
        "and.b32 n0, r0, 0xffff0000;\n\t"
        "and.b32 n1, r1, 0xffff0000;\n\t"
        "sub.f32 l0, r0, n0;\n\t"
        "sub.f32 l1, r1, n1;\n\t"
        "prmt.b32 $1, r0, r1, 0x7632;\n\t"
        "prmt.b32 $2, l0, l1, 0x7632;\n\t"
        "}\n",
        "=r,=r,=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), out, [2], loc=loc, ip=ip)),
    )


@dsl_user_op
def _ld_dsmem_v4_f32(smem_ptr, peer_rank: Int32, *, loc=None, ip=None):
    """Load four fp32 at the same shared-memory offset of a cluster peer."""
    addr = Int32(
        cute.arch.map_dsmem_ptr(smem_ptr, peer_rank, loc=loc, ip=ip).toint(
            loc=loc, ip=ip
        )
    )
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.shared::cluster.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    )


class _Sm90BF16x3RouterGemm:
    """One MMA warpgroup (64 experts x block_t tokens) and one TMA warp."""

    def __init__(
        self,
        block_t: int,
        block_k: int,
        num_stages: int,
        split_k: int,
    ) -> None:
        # Slices alternate between two register buffers, so a K-tile holds an
        # even number of k16 slices.
        assert block_k in (64, 128)
        assert block_t % 8 == 0 and block_t <= 256
        # Portable cluster sizes; every split reduces whole float4 chunks.
        assert split_k in (1, 2, 4, 8)
        assert _BLOCK_E * block_t % (4 * split_k) == 0
        self.block_t = block_t
        self.block_k = block_k
        self.num_stages = num_stages
        self.split_k = split_k
        self.num_mma_threads = 128
        self.producer_warp = 4  # after the MMA warpgroup, which must be aligned
        self.threads = self.num_mma_threads + 32

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mOut: cute.Tensor,
        stream: CUstream,
    ):
        bE, bT, bK = _BLOCK_E, self.block_t, self.block_k
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            (1, 1, 1),
            (bE, bT),
            warpgroup.OperandSource.RMEM,
        )
        w_layout = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.ROW_MAJOR, (bE, bT, bK), Float32, self.num_stages
        )
        x_layout = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.ROW_MAJOR, (bE, bT, bK), BFloat16, self.num_stages
        )
        tma_w, mW_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mW,
            cute.slice_(w_layout, (None, None, 0)),
            (bE, bK),
        )
        tma_x, mX_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mX,
            cute.slice_(x_layout, (None, None, 0)),
            (bT, bK),
        )
        # The split-K partial tile reuses the weight stages after the mainloop.
        assert self.split_k == 1 or cute.cosize(w_layout) >= bE * bT
        if const_expr(isinstance(mW.shape[1], int)):
            assert self.split_k <= cute.ceil_div(mW.shape[1], bK)

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sW: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(w_layout)], 1024
            ]
            sX: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(x_layout)], 1024
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_w, mW_tma, tma_x, mX_tma, mOut, tiled_mma, w_layout, x_layout
        ).launch(
            grid=[
                cute.ceil_div(mW.shape[0], bE),
                cute.ceil_div(mX.shape[0], bT),
                self.split_k,
            ],
            block=[self.threads, 1, 1],
            cluster=[1, 1, self.split_k],
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def _split_slice(
        self,
        sW_slice: cute.Tensor,
        rW: cute.Tensor,
        w_hi: cute.Tensor,
        w_mid: cute.Tensor,
        w_lo: cute.Tensor,
    ):
        """Read one k16 slice of the fp32 weight fragment and split it."""
        cute.autovec_copy(sW_slice, rW)
        hi = cute.recast_tensor(w_hi, Uint32)
        mid = cute.recast_tensor(w_mid, Uint32)
        lo = cute.recast_tensor(w_lo, Uint32)
        for i in cutlass.range_constexpr(cute.size(rW) // 2):
            terms = _split_fp32x2_to_3xbf16x2(rW[2 * i], rW[2 * i + 1])
            hi[i] = terms[0]
            mid[i] = terms[1]
            lo[i] = terms[2]

    @cute.jit
    def _tile_mma(
        self,
        tiled_mma: cute.TiledMma,
        acc_hi: cute.Tensor,
        acc_res: cute.Tensor,
        rW: cute.Tensor,
        hi0: cute.Tensor,
        mid0: cute.Tensor,
        lo0: cute.Tensor,
        hi1: cute.Tensor,
        mid1: cute.Tensor,
        lo1: cute.Tensor,
        tCsW: cute.Tensor,
        tCrX: cute.Tensor,
    ):
        """Issue every k16 slice of one K-tile.

        Slice kb uses register buffer kb % 2; the caller has split slice 0. The
        next slice is split while the current one runs, and the last slice is
        left in flight so the caller can prepare the next K-tile.
        """
        num_kb = cute.size(tCsW, mode=[2])
        bufs = [(hi0, mid0, lo0), (hi1, mid1, lo1)]
        for kb in cutlass.range_constexpr(num_kb):
            w_hi, w_mid, w_lo = bufs[kb % 2]
            x_slice = tCrX[(None, None, kb)]
            warpgroup.fence()
            tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            cute.gemm(tiled_mma, acc_res, w_lo, x_slice, acc_res)
            cute.gemm(tiled_mma, acc_res, w_mid, x_slice, acc_res)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, kb != 0)
            cute.gemm(tiled_mma, acc_hi, w_hi, x_slice, acc_hi)
            warpgroup.commit_group()
            if const_expr(kb + 1 < num_kb):
                # Slice kb - 1 is done, so its buffer can take slice kb + 1.
                warpgroup.wait_group(1)
                n_hi, n_mid, n_lo = bufs[(kb + 1) % 2]
                self._split_slice(tCsW[(None, None, kb + 1)], rW, n_hi, n_mid, n_lo)
        return tiled_mma

    @cute.jit
    def _advance(self, state: pipeline.PipelineState):
        # A fresh state keeps it a plain local across staged control flow.
        state = state.clone()
        state.advance()
        return state

    @cute.jit
    def _add(self, acc: cute.Tensor, part: cute.Tensor):
        for i in cutlass.range_constexpr(cute.size(acc)):
            acc[i] = acc[i] + part[i]

    @cute.kernel
    def kernel(
        self,
        tma_w: cute.CopyAtom,
        mW: cute.Tensor,
        tma_x: cute.CopyAtom,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        tiled_mma: cute.TiledMma,
        w_layout: cute.ComposedLayout,
        x_layout: cute.ComposedLayout,
    ):
        bE, bT, bK = _BLOCK_E, self.block_t, self.block_k
        tidx, _, _ = cute.arch.thread_idx()
        bid_e, bid_t, bid_s = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tx_bytes = cute.size_in_bytes(
            Float32, cute.slice_(w_layout, (None, None, 0))
        ) + cute.size_in_bytes(BFloat16, cute.slice_(x_layout, (None, None, 0)))
        mainloop = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
            tx_count=tx_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sW = storage.sW.get_tensor(w_layout.outer, swizzle=w_layout.inner)
        sX = storage.sX.get_tensor(x_layout.outer, swizzle=x_layout.inner)
        gW = cute.local_tile(mW, (bE, bK), (bid_e, None))  # (bE, bK, k_tiles)
        gX = cute.local_tile(mX, (bT, bK), (bid_t, None))  # (bT, bK, k_tiles)
        tWsW, tWgW = cpasync.tma_partition(
            tma_w,
            0,
            cute.make_layout(1),
            cute.group_modes(sW, 0, 2),
            cute.group_modes(gW, 0, 2),
        )
        tXsX, tXgX = cpasync.tma_partition(
            tma_x,
            0,
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(gX, 0, 2),
        )
        # Even K-tile ranges; the host keeps split_k <= k_tiles, so none is empty.
        k_tiles = cute.size(gW, mode=[2])
        k_begin = bid_s * k_tiles // self.split_k
        k_end = (bid_s + 1) * k_tiles // self.split_k
        pipeline_init_wait(cluster_shape_mn=(1, 1))
        cute.arch.griddepcontrol_wait()

        if warp_idx == self.producer_warp:
            cpasync.prefetch_descriptor(tma_w)
            cpasync.prefetch_descriptor(tma_x)
            state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            for kt in range(k_begin, k_end):
                mainloop.producer_acquire(state)
                bar = mainloop.producer_get_barrier(state)
                cute.copy(
                    tma_w, tWgW[(None, kt)], tWsW[(None, state.index)], tma_bar_ptr=bar
                )
                cute.copy(
                    tma_x, tXgX[(None, kt)], tXsX[(None, state.index)], tma_bar_ptr=bar
                )
                mainloop.producer_commit(state)
                state.advance()
            mainloop.producer_tail(state)

        if warp_idx < self.producer_warp:
            thr_mma = tiled_mma.get_slice(tidx)
            tCsW = thr_mma.partition_A(sW)  # fp32 view in A-fragment order
            tCrX = tiled_mma.make_fragment_B(thr_mma.partition_B(sX))
            a_shape = tCsW[(None, None, 0, 0)].shape
            rW = cute.make_rmem_tensor(a_shape, Float32)
            hi0 = cute.make_rmem_tensor(a_shape, BFloat16)
            mid0 = cute.make_rmem_tensor(a_shape, BFloat16)
            lo0 = cute.make_rmem_tensor(a_shape, BFloat16)
            hi1 = cute.make_rmem_tensor(a_shape, BFloat16)
            mid1 = cute.make_rmem_tensor(a_shape, BFloat16)
            lo1 = cute.make_rmem_tensor(a_shape, BFloat16)

            # The accumulator holds out^T [E, M]; view the output that way.
            num_t, num_e = mOut.shape
            mOutT = cute.make_tensor(
                mOut.iterator, cute.make_layout((num_e, num_t), stride=(1, num_e))
            )
            tCgOut = thr_mma.partition_C(
                cute.local_tile(mOutT, (bE, bT), (bid_e, bid_t))
            )
            tCcOut = thr_mma.partition_C(
                cute.local_tile(
                    cute.make_identity_tensor(mOutT.shape), (bE, bT), (bid_e, bid_t)
                )
            )
            acc = cute.make_rmem_tensor(tCgOut.shape, Float32)
            acc_hi = cute.make_rmem_tensor(tCgOut.shape, Float32)
            acc_res = cute.make_rmem_tensor(tCgOut.shape, Float32)
            acc.fill(0.0)
            acc_res.fill(0.0)

            read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )
            release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )
            mainloop.consumer_wait(read_state)
            stage = read_state.index
            read_state = self._advance(read_state)
            self._split_slice(tCsW[(None, None, 0, stage)], rW, hi0, mid0, lo0)
            for _ in range(k_end - k_begin - 1):
                tiled_mma = self._tile_mma(
                    tiled_mma,
                    acc_hi,
                    acc_res,
                    rW,
                    hi0,
                    mid0,
                    lo0,
                    hi1,
                    mid1,
                    lo1,
                    tCsW[(None, None, None, stage)],
                    tCrX[(None, None, None, stage)],
                )
                # Split the next K-tile's first slice while this tile's last
                # slice runs, then promote this tile's first-term partial.
                warpgroup.wait_group(1)
                mainloop.consumer_wait(read_state)
                next_stage = read_state.index
                read_state = self._advance(read_state)
                self._split_slice(tCsW[(None, None, 0, next_stage)], rW, hi0, mid0, lo0)
                warpgroup.wait_group(0)
                mainloop.consumer_release(release_state)
                release_state = self._advance(release_state)
                self._add(acc, acc_hi)
                stage = next_stage
            tiled_mma = self._tile_mma(
                tiled_mma,
                acc_hi,
                acc_res,
                rW,
                hi0,
                mid0,
                lo0,
                hi1,
                mid1,
                lo1,
                tCsW[(None, None, None, stage)],
                tCrX[(None, None, None, stage)],
            )
            warpgroup.wait_group(0)
            mainloop.consumer_release(release_state)
            self._add(acc, acc_hi)
            self._add(acc, acc_res)

            if const_expr(self.split_k == 1):
                for i in cutlass.range_constexpr(cute.size(acc)):
                    if cute.elem_less(tCcOut[i], mOutT.shape):
                        tCgOut[i] = acc[i]
            else:
                # Partial tile, expert-contiguous, for the cluster reduction.
                sRed = cute.make_tensor(
                    storage.sW.data_ptr(), cute.make_layout((bE, bT), stride=(1, bE))
                )
                tCsRed = thr_mma.partition_C(sRed)
                for i in cutlass.range_constexpr(cute.size(acc)):
                    tCsRed[i] = acc[i]

        if const_expr(self.split_k > 1):
            # Each CTA sums one chunk of the tile over all peers, in rank order.
            # The second barrier keeps every CTA's smem alive until its peers
            # have read it.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if warp_idx < self.producer_warp:
                rank = cute.arch.block_idx_in_cluster()
                chunk = bE * bT // self.split_k
                red_base = storage.sW.data_ptr()
                num_t, num_e = mOut.shape
                for v in cutlass.range_constexpr(
                    (chunk // 4 + self.num_mma_threads - 1) // self.num_mma_threads
                ):
                    local = (v * self.num_mma_threads + tidx) * 4
                    if local < chunk:
                        idx = rank * chunk + local
                        s0 = Float32(0.0)
                        s1 = Float32(0.0)
                        s2 = Float32(0.0)
                        s3 = Float32(0.0)
                        for peer in cutlass.range_constexpr(self.split_k):
                            a0, a1, a2, a3 = _ld_dsmem_v4_f32(
                                red_base + idx, Int32(peer)
                            )
                            s0 = s0 + a0
                            s1 = s1 + a1
                            s2 = s2 + a2
                            s3 = s3 + a3
                        e = bid_e * bE + idx % bE
                        t = bid_t * bT + idx // bE
                        if t < num_t:
                            if e < num_e:
                                mOut[t, e] = s0
                            if e + 1 < num_e:
                                mOut[t, e + 1] = s1
                            if e + 2 < num_e:
                                mOut[t, e + 2] = s2
                            if e + 3 < num_e:
                                mOut[t, e + 3] = s3
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        cute.arch.griddepcontrol_launch_dependents()


# Candidate tiles as (block_t, block_k, num_stages) -> (fixed_us, us_per_k64):
# the time of one CTA running alone on an SM, fitted to cold-L2 CUDA-graph
# timings on H200. Every tile is offered with split_k 1, 2, 4 and 8.
_TILE_COST = {
    (32, 128, 3): (3.72, 0.383),
    (64, 64, 4): (4.18, 0.475),
    (96, 128, 2): (5.11, 0.513),
    (128, 128, 2): (5.88, 0.568),
}


@cache
def _pick_config(num_tokens: int, K: int, E: int, num_sms: int) -> tuple:
    """Return the (block_t, block_k, num_stages, split_k) with the lowest
    estimated time: waves times the per-CTA time.

    A wave is one CTA per SM. Clusters of 4 or 8 CTAs only fill 15/16 of the
    SMs, which on H100/H200 is 120 of 132.
    """
    e_tiles = math.ceil(E / _BLOCK_E)
    best = None
    for tile, (fixed_us, us_per_k64) in _TILE_COST.items():
        block_t, block_k, num_stages = tile
        k_tiles = math.ceil(K / block_k)
        for split_k in (1, 2, 4, 8):
            if split_k > k_tiles:
                continue
            slots = num_sms if split_k <= 2 else num_sms * 15 // 16 // split_k * split_k
            ctas = e_tiles * math.ceil(num_tokens / block_t) * split_k
            k64_per_cta = math.ceil(k_tiles / split_k) * block_k // 64
            cost = math.ceil(ctas / slots) * (fixed_us + us_per_k64 * k64_per_cta)
            if best is None or cost < best[0]:
                best = (cost, (block_t, block_k, num_stages, split_k))
    assert best is not None
    return best[1]


class Sm90BF16x3RouterGemmKernel(
    VllmCuTeDSLJitKernel["Sm90BF16x3RouterGemmKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        k: int
        block_t: int
        block_k: int
        num_stages: int
        split_k: int

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        return _Sm90BF16x3RouterGemm(
            block_t=compile_key.block_t,
            block_k=compile_key.block_k,
            num_stages=compile_key.num_stages,
            split_k=compile_key.split_k,
        )

    def dispatch(  # type: ignore[override]
        self,
        *,
        num_tokens: int,
        K: int,
        E: int,
        num_sms: int,
    ) -> CompileKey:
        config = _pick_config(num_tokens, K, E, num_sms)
        return self.CompileKey(
            k=K,
            block_t=config[0],
            block_k=config[1],
            num_stages=config[2],
            split_k=config[3],
        )

    def get_warmup_keys(
        self,
        *,
        K: int,
        E: int,
        max_tokens: int,
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            num_tokens=WarmupIntRange(1, max_tokens + 1),
            K=K,
            E=E,
            num_sms=num_compute_units(torch.accelerator.current_device_index()),
        )

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        M, E = cute.sym_int(), cute.sym_int()
        return (
            make_fake_tensor(BFloat16, (M, compile_key.k), divisibility=8),
            make_fake_tensor(Float32, (E, compile_key.k), divisibility=4),
            make_fake_tensor(Float32, (M, E), divisibility=1),
        )

    @kernel_launcher
    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        M, K = x.shape
        E = weight.shape[0]
        compile_key = self.dispatch(
            num_tokens=M,
            K=K,
            E=E,
            num_sms=num_compute_units(x.device.index),
        )
        out = x.new_empty((M, E), dtype=torch.float32)
        return compile_key, (x, weight, out), out


_SM90_BF16X3_ROUTER_GEMM_KERNEL = Sm90BF16x3RouterGemmKernel()


def _bf16x3_router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.shape[0] == 0:
        return x.new_empty((0, weight.shape[0]), dtype=torch.float32)
    return _SM90_BF16X3_ROUTER_GEMM_KERNEL(x.contiguous(), weight)


def _bf16x3_router_gemm_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


direct_register_custom_op(
    op_name="sm90_bf16x3_router_gemm",
    op_func=_bf16x3_router_gemm,
    fake_impl=_bf16x3_router_gemm_fake,
)


def bf16x3_router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return ``x @ weight.T`` in fp32 using the SM90 BF16x3 router GEMM."""
    return torch.ops.vllm.sm90_bf16x3_router_gemm(x, weight)
