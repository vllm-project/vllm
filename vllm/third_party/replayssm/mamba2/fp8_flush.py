# SPDX-License-Identifier: Apache-2.0
"""MAMBA2 fp8 flush; fixed L=16 specialization."""

from __future__ import annotations

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float16, Float32, Int8, Int32, Int64, Uint32, Uint64, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op
from vllm.third_party.replayssm.common import make_fake_tensor

from vllm.third_party.replayssm.common import _tcgen05, simple_tma_copy
from vllm.third_party.replayssm.common.dsl import (
    alloc_tmem as _alloc_tmem,
)
from vllm.third_party.replayssm.common.dsl import (
    dealloc_tmem as _dealloc_tmem,
)
from vllm.third_party.replayssm.common.dsl import (
    make_rmem,
)
from vllm.third_party.replayssm.common.dsl import (
    make_sdesc_32B_swizzle as _make_sdesc_32B_swizzle,
)
from vllm.third_party.replayssm.common.philox import (
    cvt_f32x2_f16x2_rs,
    philox4x32,
)

E4M3 = cutlass.Float8E4M3FN
E4M3_MAX = 448.0


@dsl_user_op
def f32x4_to_e4m3x4(v0, v1, v2, v3, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        (
            "{\n"
            "\t.reg .b16 t0, t1;\n"
            "\tcvt.rn.satfinite.e4m3x2.f32 t0, $2, $1;\n"
            "\tcvt.rn.satfinite.e4m3x2.f32 t1, $4, $3;\n"
            "\tmov.b32 $0, {t0, t1};\n"
            "\t}\n"
        ),
        "=r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


@dsl_user_op
def div_rn_f32(a, b, *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


@dsl_user_op
def rcp_rn_f32(a, *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip)],
        "rcp.rn.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


class Sm103Mamba2Tp8FlushKernel:
    def __init__(
        self,
        nheads: int,
        ngroups: int,
        head_dim: int = 64,
        dstate: int = 128,
        cache_len: int = 16,
    ) -> None:
        assert head_dim == 64 and dstate == 128 and (cache_len == 16)
        assert nheads % ngroups == 0
        hpg = nheads // ngroups
        assert hpg in (8, 16)
        self.nheads = nheads
        self.ngroups = ngroups
        self.hpg = hpg
        self.pairs = hpg // 2
        self.head_dim = head_dim
        self.dstate = dstate
        self.cache_len = cache_len
        self.dot_k = 16
        self.num_warps = 6
        self.elems = 64
        self.n_span = dstate // self.elems
        self.n_slots = self.pairs * self.n_span
        self.unit = 8
        self.tmem_cols = 128
        self.n_acc = 2

    @cute.jit
    def _make_anchor_tma(self, state: cute.Tensor, op):
        num_elems = 64
        swizzle = cute.make_swizzle(3, 4, 3)
        smem_layout = cute.make_layout(
            (1, 1, self.head_dim, (num_elems, 1), 1),
            stride=(
                0,
                0,
                num_elems,
                (1, self.head_dim * num_elems),
                self.head_dim * num_elems,
            ),
        )
        smem_layout = cute.make_composed_layout(swizzle, 0, smem_layout)
        return cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(state, (None, None, None, num_elems)),
            smem_layout,
            cta_tiler=(1, 1, self.head_dim, num_elems),
        )

    @cute.jit
    def _make_fp8_tma(self, state_lo: cute.Tensor, op):
        num_elems = self.dstate
        swizzle = cute.make_swizzle(3, 4, 3)
        smem_layout = cute.make_layout(
            (1, 1, self.head_dim, (num_elems, 1), 1),
            stride=(
                0,
                0,
                num_elems,
                (1, self.head_dim * num_elems),
                self.head_dim * num_elems,
            ),
        )
        smem_layout = cute.make_composed_layout(swizzle, 0, smem_layout)
        return cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(state_lo, (None, None, None, num_elems)),
            smem_layout,
            cta_tiler=(1, 1, self.head_dim, num_elems),
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        dt: cute.Tensor,
        A: cute.Tensor,
        B: cute.Tensor,
        C: cute.Tensor,
        D: cute.Tensor,
        dt_bias: cute.Tensor,
        state: cute.Tensor,
        state_lo: cute.Tensor,
        state_lo_scale: cute.Tensor,
        x_cache: cute.Tensor,
        dt_cache: cute.Tensor,
        B_cache: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        is_flush: cute.Tensor,
        out: cute.Tensor,
        sr_key0: Uint32,
        sr_key1: Uint32,
        stream: CUstream,
    ):
        anchor_tma = self._make_anchor_tma(state, cpasync.CopyBulkTensorTileG2SOp())
        anchor_tma_store = self._make_anchor_tma(
            state, cpasync.CopyBulkTensorTileS2GOp()
        )
        fp8_tma_store = self._make_fp8_tma(state_lo, cpasync.CopyBulkTensorTileS2GOp())
        self.kernel(
            anchor_tma,
            anchor_tma_store,
            fp8_tma_store,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            state_lo_scale,
            x_cache,
            dt_cache,
            B_cache,
            state_indices,
            write_pos,
            is_flush,
            out,
            sr_key0,
            sr_key1,
        ).launch(
            grid=(x.shape[0], self.ngroups, 1),
            block=(self.num_warps * 32, 1, 1),
            min_blocks_per_mp=3,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        anchor_tma: cpasync.TmaInfo,
        anchor_tma_store: cpasync.TmaInfo,
        fp8_tma_store: cpasync.TmaInfo,
        x: cute.Tensor,
        dt: cute.Tensor,
        A: cute.Tensor,
        B: cute.Tensor,
        C: cute.Tensor,
        D: cute.Tensor,
        dt_bias: cute.Tensor,
        state_lo_scale: cute.Tensor,
        x_cache: cute.Tensor,
        dt_cache: cute.Tensor,
        B_cache: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        is_flush: cute.Tensor,
        out: cute.Tensor,
        sr_key0: Uint32,
        sr_key1: Uint32,
    ):
        state_t = Float16
        elems = self.elems
        n_span = self.n_span
        n_slots = self.n_slots
        n_acc = self.n_acc
        pairs = self.pairs
        uw = self.unit
        L = self.cache_len
        tid, _, _ = cute.arch.thread_idx()
        batch_id, group, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32
        first_head = group * self.hpg

        def allocate_tensor(smem, dtype, layout):
            return smem.allocate_tensor(
                dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )

        smem = cutlass.utils.SmemAllocator()
        state_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                ((self.head_dim, 2), (elems, 2)),
                stride=((elems, self.head_dim * elems), (1, 2 * self.head_dim * elems)),
            ),
        )
        s_state = allocate_tensor(smem, state_t, state_layout)
        state_unit_layout = cute.make_layout(
            (2 * self.head_dim, ((uw, elems // uw), 2)),
            stride=(elems, ((1, uw), 2 * self.head_dim * elems)),
        )
        s_state_u = cute.make_tensor(s_state.iterator, state_unit_layout)
        fp8_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                ((self.head_dim, 2), (self.dstate, 1)),
                stride=(
                    (self.dstate, self.head_dim * self.dstate),
                    (1, 2 * self.head_dim * self.dstate),
                ),
            ),
        )
        s_fp8 = allocate_tensor(smem, E4M3, fp8_layout)
        fp8_unit_layout = cute.make_layout(
            (2 * self.head_dim, (16, self.dstate // 16)), stride=(self.dstate, (1, 16))
        )
        s_fp8_u = cute.make_tensor(s_fp8.iterator, fp8_unit_layout)
        rows = 2 * self.head_dim
        dot_layout = cute.make_layout((rows, self.dot_k), stride=(self.dot_k, 1))
        dot_unit_layout = cute.make_layout(
            (rows, (8, self.dot_k // 8)), stride=(self.dot_k, (1, 8))
        )
        s_xall0 = smem.allocate_tensor(
            BFloat16, cute.make_layout(2048), byte_alignment=1024
        )
        s_xall1 = smem.allocate_tensor(
            BFloat16, cute.make_layout(2048), byte_alignment=1024
        )
        s_ball_extra = smem.allocate_tensor(
            BFloat16, cute.make_layout(2048), byte_alignment=1024
        )
        xall_bases = [
            cute.recast_ptr(
                s_xall0.iterator, cute.make_swizzle(1, 4, 3), dtype=BFloat16
            ),
            cute.recast_ptr(
                s_xall1.iterator, cute.make_swizzle(1, 4, 3), dtype=BFloat16
            ),
        ]
        ball_base = cute.recast_ptr(
            s_ball_extra.iterator, cute.make_swizzle(1, 4, 3), dtype=BFloat16
        )
        s_xall = [cute.make_tensor(b, dot_layout) for b in xall_bases]
        s_xall_st = cute.make_tensor(
            xall_bases[0],
            cute.make_layout((rows, self.dot_k, 2), stride=(self.dot_k, 1, 2048)),
        )
        s_ball = cute.make_tensor(ball_base, dot_layout)
        s_ball_u = cute.make_tensor(ball_base, dot_unit_layout)
        s_c = smem.allocate_array(Float32, self.dstate)
        s_scale = smem.allocate_array(Float32, self.hpg * L)
        s_tdecay = smem.allocate_array(Float32, self.hpg)
        state_slot_a = smem.allocate_array(Int64, n_slots)
        f_mma_a = smem.allocate_array(Int64, n_slots)
        f_gdone_a = smem.allocate_array(Int64, n_slots)
        fd_a = smem.allocate_array(Int64, pairs)
        state_slot = [state_slot_a + i for i in range(n_slots)]
        f_mma = [f_mma_a + i for i in range(n_slots)]
        f_gdone = [f_gdone_a + i for i in range(n_slots)]
        fd = [fd_a + i for i in range(pairs)]
        fk_full = smem.allocate_array(Int64, 1)
        fp8_free_a = smem.allocate_array(Int64, pairs)
        fp8_free = [fp8_free_a + i for i in range(pairs)]
        tmem_addr = smem.allocate(Int32, 4)

        def anchor_load(
            slot, n_span, first_head, state_idx, anchor_tma, s_state, state_slot
        ):
            p = slot // n_span
            g = slot % n_span
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(state_slot[slot], 16384)
            for hp in cutlass.range_constexpr(2):
                head = first_head + p * 2 + hp
                g_state = anchor_tma.tma_tensor[state_idx, head, None, (None, g)]
                simple_tma_copy(
                    anchor_tma.atom,
                    g_state,
                    s_state[(None, hp), (None, slot % 2)],
                    state_slot[slot],
                )

        def stage_x(
            p, cidx, wp, state_idx, first_head, s_scale, x_cache, s_xall_st_, fd_bar, L_
        ):
            hp = cidx // 64
            lg = cidx % 64 // 8
            r0 = cidx % 64 % 8 * 8
            st_hh = p * 2 + hp
            st_h = first_head + st_hh
            st = p % 2
            xrow0 = cute.logical_divide(
                x_cache[state_idx, st_h, lg, None], cute.make_layout(8)
            )
            xrow1 = cute.logical_divide(
                x_cache[state_idx, st_h, lg + 8, None], cute.make_layout(8)
            )
            f0 = make_rmem(8, BFloat16)
            f1 = make_rmem(8, BFloat16)
            cute.autovec_copy(xrow0[None, r0 // 8], f0)
            cute.autovec_copy(xrow1[None, r0 // 8], f1)
            s0 = Float32(0.0)
            if lg <= wp:
                s0 = s_scale[st_hh * L_ + lg]
            s1 = Float32(0.0)
            if lg + 8 <= wp:
                s1 = s_scale[st_hh * L_ + lg + 8]
            for j in cutlass.range_constexpr(8):
                row_j = hp * 64 + r0 + j
                s_xall_st_[row_j, lg, st] = BFloat16(Float32(f0[j]) * s0)
                s_xall_st_[row_j, lg + 8, st] = BFloat16(Float32(f1[j]) * s1)
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(fd_bar)

        state_idx = state_indices[batch_id]
        tag = Uint64(write_pos[batch_id])
        wp = Int32(tag & Uint64(self.cache_len - 1))
        sr_event_lo = Uint32(tag >> Uint64(4))
        sr_event_hi = Uint32(tag >> Uint64(36))
        if Int32(is_flush[batch_id]) != 0 and state_idx > 0:
            if warp_id == 0:
                with cute.arch.elect_one():
                    for slot in cutlass.range_constexpr(2):
                        cute.arch.mbarrier_init(state_slot[slot], 1)
                    cute.arch.mbarrier_init_fence()
                cpasync.prefetch_descriptor(anchor_tma.atom)
                for slot in cutlass.range_constexpr(2):
                    anchor_load(
                        slot,
                        n_span,
                        first_head,
                        state_idx,
                        anchor_tma,
                        s_state,
                        state_slot,
                    )
                with cute.arch.elect_one():
                    for slot in cutlass.range_constexpr(2, n_slots):
                        cute.arch.mbarrier_init(state_slot[slot], 1)
                    for slot in cutlass.range_constexpr(n_slots):
                        cute.arch.mbarrier_init(f_mma[slot], 1)
                        cute.arch.mbarrier_init(f_gdone[slot], 128)
                    cute.arch.mbarrier_init(fk_full, 128)
                    for i in cutlass.range_constexpr(pairs):
                        cute.arch.mbarrier_init(fd[i], 128)
                        cute.arch.mbarrier_init(fp8_free[i], 1)
                    cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            if warp_id == 0:
                cpasync.prefetch_descriptor(anchor_tma_store.atom)
                cpasync.prefetch_descriptor(fp8_tma_store.atom)
                for p in cutlass.range_constexpr(pairs):
                    for g in cutlass.range_constexpr(n_span):
                        slot = p * n_span + g
                        cute.arch.mbarrier_wait(f_gdone[slot], 0)
                        for hp in cutlass.range_constexpr(2):
                            head = first_head + p * 2 + hp
                            g_out = anchor_tma_store.tma_tensor[
                                state_idx, head, None, (None, g)
                            ]
                            simple_tma_copy(
                                anchor_tma_store.atom,
                                s_state[(None, hp), (None, slot % 2)],
                                g_out,
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        if cutlass.const_expr(p + 1 < pairs):
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                            anchor_load(
                                slot + n_span,
                                n_span,
                                first_head,
                                state_idx,
                                anchor_tma,
                                s_state,
                                state_slot,
                            )
                    for hp in cutlass.range_constexpr(2):
                        head = first_head + p * 2 + hp
                        g_lo = fp8_tma_store.tma_tensor[
                            state_idx, head, None, (None, 0)
                        ]
                        simple_tma_copy(
                            fp8_tma_store.atom, s_fp8[(None, hp), (None, 0)], g_lo
                        )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(fp8_free[p])
                cute.arch.cp_async_bulk_wait_group(0)
            elif warp_id == 1:
                _alloc_tmem(tmem_addr, self.tmem_cols)
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.mbarrier_wait(fk_full, 0)
                d_bases = [
                    _make_sdesc_32B_swizzle(Int64(s.iterator.toint() >> 4))
                    for s in s_xall
                ]
                b_base = _make_sdesc_32B_swizzle(Int64(s_ball.iterator.toint() >> 4))
                idesc_up = _tcgen05.make_bf16_idesc(2 * self.head_dim, elems)
                for p in cutlass.range_constexpr(pairs):
                    cute.arch.mbarrier_wait(fd[p], 0)
                    for g in cutlass.range_constexpr(n_span):
                        slot = p * n_span + g
                        if cutlass.const_expr(slot >= n_acc):
                            cute.arch.mbarrier_wait(f_gdone[slot - n_acc], 0)
                        _tcgen05.fence_after_thread_sync()
                        _tcgen05.mma_f16(
                            tmem_addr[0] + slot % n_acc * elems,
                            d_bases[p % 2],
                            b_base + (g * elems * 2 * self.dot_k >> 4),
                            idesc_up,
                            False,
                        )
                        _tcgen05.commit(f_mma[slot])
                cute.arch.mbarrier_wait(f_gdone[n_slots - 1], 0)
                _dealloc_tmem(tmem_addr[0], self.tmem_cols)
            else:
                consumer_warp = warp_id - 2
                tmem_row = consumer_warp * 32 + lane_id
                row = tmem_row ^ 64
                head_par = row // self.head_dim
                r = row % self.head_dim
                n_idx = tmem_row
                live = state_idx != 0
                s_c[n_idx] = Float32(C[batch_id, group, n_idx])
                b_val = B[batch_id, group, n_idx]
                if live:
                    B_cache[state_idx, group, wp, n_idx] = b_val
                if consumer_warp == 0 and lane_id < self.hpg:
                    h_glob = first_head + lane_id
                    dtv = Float32(dt[batch_id, h_glob]) + Float32(dt_bias[h_glob])
                    if dtv <= Float32(20.0):
                        dtv = cute.math.log(
                            Float32(1.0) + cute.math.exp(dtv, fastmath=True),
                            fastmath=True,
                        )
                    if live:
                        dt_cache[state_idx, h_glob, wp] = dtv
                for p in cutlass.range_constexpr(pairs):
                    h_glob = first_head + p * 2 + head_par
                    if live:
                        x_cache[state_idx, h_glob, wp, r] = x[batch_id, h_glob, r]
                cute.arch.barrier(barrier_id=6, number_of_threads=128)
                for rnd in cutlass.range_constexpr(self.hpg // 8):
                    hh = rnd * 8 + consumer_warp * 2 + lane_id // 16
                    seg = lane_id % 16
                    h_glob = first_head + hh
                    dtv = Float32(0.0)
                    if seg <= wp:
                        dtv = dt_cache[state_idx, h_glob, seg]
                    prefix = dtv
                    for off in (1, 2, 4, 8):
                        lower = cute.arch.shuffle_sync_up(prefix, off, mask_and_clamp=0)
                        if seg >= off:
                            prefix += lower
                    t15 = cute.arch.shuffle_sync(prefix, 15)
                    t31 = cute.arch.shuffle_sync(prefix, 31)
                    total = t15
                    if lane_id >= 16:
                        total = t31
                    A_h = Float32(A[h_glob])
                    dA_total = A_h * total
                    sc = dtv * cute.math.exp(dA_total - A_h * prefix, fastmath=True)
                    if seg > wp:
                        sc = Float32(0.0)
                    s_scale[hh * L + seg] = sc
                    if seg == 0:
                        s_tdecay[hh] = cute.math.exp(dA_total, fastmath=True)
                cute.arch.barrier(barrier_id=4, number_of_threads=128)
                for unit in cutlass.range_constexpr(self.dot_k // 8):
                    frag = make_rmem(8, BFloat16)
                    for j in cutlass.range_constexpr(8):
                        c_idx = unit * 8 + j
                        bv = BFloat16(0.0)
                        if c_idx <= wp:
                            bv = B_cache[state_idx, group, c_idx, tmem_row]
                        frag[j] = bv
                    cute.autovec_copy(frag, s_ball_u[tmem_row, (None, unit)])
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(fk_full)
                stage_x(
                    0,
                    tmem_row,
                    wp,
                    state_idx,
                    first_head,
                    s_scale,
                    x_cache,
                    s_xall_st,
                    fd_a + 0,
                    L,
                )
                for p in cutlass.range(pairs, unroll=1):
                    if p + 1 < pairs:
                        stage_x(
                            p + 1,
                            tmem_row,
                            wp,
                            state_idx,
                            first_head,
                            s_scale,
                            x_cache,
                            s_xall_st,
                            fd_a + (p + 1),
                            L,
                        )
                    rmw_hh = p * 2 + head_par
                    h_glob = first_head + rmw_hh
                    total_decay = s_tdecay[rmw_hh]
                    oacc = Float32(x[batch_id, h_glob, r]) * Float32(D[h_glob, r])
                    row_amax = Float32(0.0)
                    for g in cutlass.range_constexpr(n_span):
                        cute.arch.mbarrier_wait(f_mma_a + (p * n_span + g), 0)
                        cute.arch.mbarrier_wait(state_slot_a + (p * n_span + g), 0)
                        _tcgen05.fence_after_thread_sync()
                        acc_col = tmem_addr[0] + g * elems
                        for chunk in cutlass.range(elems // 16, unroll=1):
                            update = _tcgen05.ld(
                                tmem_row, acc_col + chunk * 16, "32x32b", 16
                            )
                            _tcgen05.wait_ld()
                            for u in cutlass.range_constexpr(16 // uw):
                                unit_idx = chunk * (16 // uw) + u
                                col0 = g * elems + unit_idx * uw
                                s_unit = s_state_u[row, ((None, unit_idx), g)]
                                old = make_rmem(uw, state_t)
                                cute.autovec_copy(s_unit, old)
                                for j in cutlass.range_constexpr(uw):
                                    v = (
                                        Float32(old[j]) * total_decay
                                        + update[u * uw + j]
                                    )
                                    oacc += v * s_c[col0 + j]
                                    a_v = cute.math.absf(v)
                                    if a_v > row_amax:
                                        row_amax = a_v
                    s_row = div_rn_f32(row_amax, Float32(E4M3_MAX))
                    if s_row <= Float32(0.0):
                        s_row = Float32(1.0)
                    inv = rcp_rn_f32(s_row)
                    state_lo_scale[state_idx, h_glob, r] = s_row
                    if p > 0:
                        cute.arch.mbarrier_wait(fp8_free_a + (p - 1), 0)
                    for g in cutlass.range_constexpr(n_span):
                        acc_col = tmem_addr[0] + g * elems
                        for chunk in cutlass.range(elems // 16, unroll=1):
                            update = _tcgen05.ld(
                                tmem_row, acc_col + chunk * 16, "32x32b", 16
                            )
                            _tcgen05.wait_ld()
                            qw = make_rmem(4, Uint32)
                            for u in cutlass.range_constexpr(16 // uw):
                                unit_idx = chunk * (16 // uw) + u
                                col0 = g * elems + unit_idx * uw
                                s_unit = s_state_u[row, ((None, unit_idx), g)]
                                old = make_rmem(uw, state_t)
                                cute.autovec_copy(s_unit, old)
                                vals = make_rmem(uw, Float32)
                                for j in cutlass.range_constexpr(uw):
                                    vals[j] = (
                                        Float32(old[j]) * total_decay
                                        + update[u * uw + j]
                                    )
                                packed = make_rmem(uw // 2, Uint32)
                                spatial = Uint32(
                                    (h_glob * self.head_dim + r) * (self.dstate // uw)
                                    + col0 // uw
                                )
                                w = philox4x32(
                                    Uint32(state_idx),
                                    spatial,
                                    sr_event_lo,
                                    sr_event_hi,
                                    sr_key0,
                                    sr_key1,
                                    7,
                                )
                                for pr in cutlass.range_constexpr(uw // 2):
                                    packed[pr] = cvt_f32x2_f16x2_rs(
                                        vals[2 * pr], vals[2 * pr + 1], w[pr]
                                    )
                                cute.autovec_copy(
                                    cute.recast_tensor(packed, Float16), s_unit
                                )
                                qw[2 * u] = f32x4_to_e4m3x4(
                                    vals[0] * inv,
                                    vals[1] * inv,
                                    vals[2] * inv,
                                    vals[3] * inv,
                                )
                                qw[2 * u + 1] = f32x4_to_e4m3x4(
                                    vals[4] * inv,
                                    vals[5] * inv,
                                    vals[6] * inv,
                                    vals[7] * inv,
                                )
                            cute.autovec_copy(
                                cute.recast_tensor(qw, E4M3),
                                s_fp8_u[row, (None, g * (elems // 16) + chunk)],
                            )
                        _tcgen05.fence_before_thread_sync()
                        cute.arch.fence_view_async_shared()
                        cute.arch.mbarrier_arrive(f_gdone_a + (p * n_span + g))
                    out[batch_id, h_glob, r] = BFloat16(oacc)

    @cache
    @staticmethod
    def compile(
        nheads: int,
        ngroups: int,
        head_dim: int = 64,
        dstate: int = 128,
        cache_len: int = 16,
        slot_strides: tuple[int, ...] = (),
    ):
        batch = cute.sym_int()
        slots = cute.sym_int()
        x = make_fake_tensor(BFloat16, (batch, nheads, head_dim), divisibility=16)
        dt = make_fake_tensor(Float32, (batch, nheads), divisibility=16)
        A = make_fake_tensor(Float32, (nheads,), divisibility=16)
        B = make_fake_tensor(BFloat16, (batch, ngroups, dstate), divisibility=16)
        C = make_fake_tensor(BFloat16, (batch, ngroups, dstate), divisibility=16)
        D = make_fake_tensor(Float32, (nheads, head_dim), divisibility=16)
        dt_bias = make_fake_tensor(Float32, (nheads,), divisibility=16)
        state = make_fake_tensor(
            Float16,
            (slots, nheads, head_dim, dstate),
            divisibility=16,
            slot_stride=slot_strides[0],
        )
        state_lo = make_fake_tensor(
            E4M3,
            (slots, nheads, head_dim, dstate),
            divisibility=16,
            slot_stride=slot_strides[1],
        )
        state_lo_scale = make_fake_tensor(
            Float32,
            (slots, nheads, head_dim),
            divisibility=16,
            slot_stride=slot_strides[2],
        )
        x_cache = make_fake_tensor(
            BFloat16,
            (slots, nheads, cache_len, head_dim),
            divisibility=16,
            slot_stride=slot_strides[3],
        )
        dt_cache = make_fake_tensor(
            Float32,
            (slots, nheads, cache_len),
            divisibility=16,
            slot_stride=slot_strides[4],
        )
        B_cache = make_fake_tensor(
            BFloat16,
            (slots, ngroups, cache_len, dstate),
            divisibility=16,
            slot_stride=slot_strides[5],
        )
        state_indices = make_fake_tensor(Int32, (batch,), divisibility=1)
        write_pos = make_fake_tensor(Int64, (batch,), divisibility=1)
        is_flush = make_fake_tensor(Int8, (batch,), divisibility=1)
        out = make_fake_tensor(BFloat16, (batch, nheads, head_dim), divisibility=16)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Sm103Mamba2Tp8FlushKernel(nheads, ngroups, head_dim, dstate, cache_len),
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            state,
            state_lo,
            state_lo_scale,
            x_cache,
            dt_cache,
            B_cache,
            state_indices,
            write_pos,
            is_flush,
            out,
            Uint32(0),
            Uint32(0),
            stream,
            options="--enable-tvm-ffi",
        )


def mamba2_fp8_flush(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    x_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    B_cache: torch.Tensor,
    state_indices: torch.Tensor,
    write_pos: torch.Tensor,
    is_flush: torch.Tensor,
    out: torch.Tensor,
    *,
    state_lo: torch.Tensor,
    state_lo_scale: torch.Tensor,
    stochastic_rounding_seed: int = 0,
    stochastic_rounding_salt: int = 0,
) -> torch.Tensor:
    slots, H, P, N = state.shape
    G = B.shape[1]
    L = x_cache.shape[2]
    if state.dtype is not torch.float16:
        raise TypeError("the anchor must be FP16")
    if write_pos.dtype != torch.int64:
        raise TypeError("stochastic rounding requires int64 packed tags")
    if is_flush.dtype == torch.bool:
        is_flush = is_flush.view(torch.int8)
    Sm103Mamba2Tp8FlushKernel.compile(
        H,
        G,
        P,
        N,
        L,
        (
            state.stride(0),
            state_lo.stride(0),
            state_lo_scale.stride(0),
            x_cache.stride(0),
            dt_cache.stride(0),
            B_cache.stride(0),
        ),
    )(
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        state,
        state_lo.view(torch.float8_e4m3fn),
        state_lo_scale,
        x_cache,
        dt_cache,
        B_cache,
        state_indices,
        write_pos,
        is_flush.view(torch.int8),
        out,
        stochastic_rounding_seed & 4294967295,
        stochastic_rounding_salt & 4294967295,
    )
    return out
