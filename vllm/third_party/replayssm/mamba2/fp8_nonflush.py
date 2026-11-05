# SPDX-License-Identifier: Apache-2.0
"""MAMBA2 fp8 nonflush; fixed L=16 specialization."""

from __future__ import annotations

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int8, Int32, Int64, Uint64, cute
from vllm.third_party.replayssm.common import make_fake_tensor

from vllm.third_party.replayssm.common.nf_ops import (
    bf16x4_bits_to_f32x4,
    cp_async_bulk_g2s,
    e4m3x16_dot_f32,
    lds_bf16_f32,
    lds_f32,
    lds_v4_u32,
)


class Sm103Mamba2Fp8CheckpointFused:
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
        self.hpg = nheads // ngroups
        assert self.hpg in (8, 16)
        self.nheads = nheads
        self.ngroups = ngroups
        self.pairs = self.hpg // 2
        self.nstage = 2
        self.head_dim = head_dim
        self.dstate = dstate
        self.cache_len = cache_len
        self.num_warps = 4
        self.pair_bytes = 2 * head_dim * dstate
        self.pair_scale_bytes = 2 * head_dim * 4
        self.head_x_bytes = cache_len * head_dim * 2
        self.pair_x_bytes = 2 * self.head_x_bytes
        self.stage_tx = self.pair_bytes + self.pair_scale_bytes + self.pair_x_bytes

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
        x_cache: cute.Tensor,
        dt_cache: cute.Tensor,
        B_cache: cute.Tensor,
        state_lo: cute.Tensor,
        state_lo_scale: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        is_flush: cute.Tensor,
        out: cute.Tensor,
        stream: CUstream,
    ):
        self.kernel(
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            x_cache,
            dt_cache,
            B_cache,
            state_lo,
            state_lo_scale,
            state_indices,
            write_pos,
            is_flush,
            out,
        ).launch(
            grid=(x.shape[0], self.ngroups, 1),
            block=(self.num_warps * 32, 1, 1),
            stream=stream,
            min_blocks_per_mp=5,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        dt: cute.Tensor,
        A: cute.Tensor,
        B: cute.Tensor,
        C: cute.Tensor,
        D: cute.Tensor,
        dt_bias: cute.Tensor,
        x_cache: cute.Tensor,
        dt_cache: cute.Tensor,
        B_cache: cute.Tensor,
        state_lo: cute.Tensor,
        state_lo_scale: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        is_flush: cute.Tensor,
        out: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        batch_id, group, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        s_state = smem.allocate_tensor(
            Int8, cute.make_layout(2 * self.pair_bytes), byte_alignment=128
        )
        s_x = smem.allocate_tensor(
            Int8, cute.make_layout(2 * self.pair_x_bytes), byte_alignment=128
        )
        s_scale = smem.allocate_tensor(
            Float32, cute.make_layout(2 * 2 * self.head_dim), byte_alignment=16
        )
        s_c = smem.allocate_array(Float32, self.dstate)
        s_bnew = smem.allocate_array(Float32, self.dstate)
        s_bc = smem.allocate_array(Float32, self.cache_len)
        s_replay = smem.allocate_array(Float32, self.hpg * self.cache_len)
        s_decay = smem.allocate_array(Float32, self.hpg)
        s_ckpt = smem.allocate_array(Float32, 2 * self.head_dim)
        bars = smem.allocate_array(Int64, self.nstage)
        if Int32(is_flush[batch_id]) == 0 and state_indices[batch_id] > 0:
            warp = tid // 32
            lane = tid % 32
            subgroup = lane // 4
            lig = lane % 4
            par = subgroup % 2
            first_head = group * self.hpg
            state_idx = state_indices[batch_id]
            wp = write_pos[batch_id]
            live = state_idx != 0
            B_cache_u64 = cute.recast_tensor(B_cache, Uint64)
            bq = cute.make_rmem_tensor((4,), Uint64)
            for j in cutlass.range_constexpr(4):
                bq[j] = B_cache_u64[state_idx, group, warp + 4 * j, lane]
            dtq = cute.make_rmem_tensor((self.hpg // 8,), Float32)
            dtv = cute.make_rmem_tensor((self.hpg // 8,), Float32)
            av = cute.make_rmem_tensor((self.hpg // 8,), Float32)
            for hr in cutlass.range_constexpr(self.hpg // 8):
                head_r = hr * 8 + warp * 2 + lane // 16
                h_r = first_head + head_r
                dtq[hr] = dt_cache[state_idx, h_r, lane % 16]
                dtv[hr] = Float32(dt[batch_id, h_r]) + Float32(dt_bias[h_r])
                av[hr] = Float32(A[h_r])
            b_value = B[batch_id, group, tid]
            c_value = Float32(C[batch_id, group, tid])
            hp_t = tid // self.head_dim
            r_t = tid % self.head_dim
            x_next = x[batch_id, first_head + hp_t, r_t]
            d_next = Float32(D[first_head + hp_t, r_t])
            if tid == 0:
                for s in cutlass.range_constexpr(self.nstage):
                    cute.arch.mbarrier_init(bars + s, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            state_g = (
                state_lo.iterator.toint()
                + Int64(state_idx) * state_lo.stride[0]
                + Int64(first_head * self.head_dim * self.dstate)
            )
            scale_g = (
                state_lo_scale.iterator.toint()
                + (
                    Int64(state_idx) * state_lo_scale.stride[0]
                    + first_head * self.head_dim
                )
                * 4
            )
            x_g = (
                x_cache.iterator.toint()
                + Int64(state_idx) * x_cache.stride[0] * 2
                + Int64(first_head * self.head_x_bytes)
            )
            state_s = s_state.iterator.toint()
            x_s = s_x.iterator.toint()
            scale_s = s_scale.iterator.toint()
            bar_s = bars.toint()
            if tid == 0:
                for s in cutlass.range_constexpr(self.nstage - 1):
                    cute.arch.mbarrier_arrive_and_expect_tx(bars + s, self.stage_tx)
                    cp_async_bulk_g2s(
                        state_s + s * self.pair_bytes,
                        state_g + s * self.pair_bytes,
                        self.pair_bytes,
                        bar_s + s * 8,
                    )
                    cp_async_bulk_g2s(
                        scale_s + s * self.pair_scale_bytes,
                        scale_g + s * self.pair_scale_bytes,
                        self.pair_scale_bytes,
                        bar_s + s * 8,
                    )
                    cp_async_bulk_g2s(
                        x_s + s * self.pair_x_bytes,
                        x_g + s * self.pair_x_bytes,
                        self.pair_x_bytes,
                        bar_s + s * 8,
                    )
            s_c[tid] = c_value
            s_bnew[tid] = Float32(b_value)
            if live:
                B_cache[state_idx, group, wp, tid] = b_value
            dt_new = cute.make_rmem_tensor((self.hpg // 8,), Float32)
            for hr in cutlass.range_constexpr(self.hpg // 8):
                v = dtv[hr]
                if v <= Float32(20.0):
                    v = cute.math.log(
                        Float32(1.0) + cute.math.exp(v, fastmath=True), fastmath=True
                    )
                dt_new[hr] = v
            if tid < self.hpg:
                h_glob0 = first_head + tid
                v0 = Float32(dt[batch_id, h_glob0]) + Float32(dt_bias[h_glob0])
                if v0 <= Float32(20.0):
                    v0 = cute.math.log(
                        Float32(1.0) + cute.math.exp(v0, fastmath=True), fastmath=True
                    )
                if live:
                    dt_cache[state_idx, h_glob0, wp] = v0
            chunk_a = lig + 4 * par
            chunk_b = lig + 4 * (1 - par)
            C_u64 = cute.recast_tensor(C, Uint64)
            c_vals = cute.make_rmem_tensor((32,), Float32)
            for k in cutlass.range_constexpr(4):
                a0, a1, a2, a3 = bf16x4_bits_to_f32x4(
                    C_u64[batch_id, group, chunk_a * 4 + k]
                )
                c_vals[4 * k] = a0
                c_vals[4 * k + 1] = a1
                c_vals[4 * k + 2] = a2
                c_vals[4 * k + 3] = a3
                b0, b1, b2, b3 = bf16x4_bits_to_f32x4(
                    C_u64[batch_id, group, chunk_b * 4 + k]
                )
                c_vals[16 + 4 * k] = b0
                c_vals[16 + 4 * k + 1] = b1
                c_vals[16 + 4 * k + 2] = b2
                c_vals[16 + 4 * k + 3] = b3
            cute.arch.sync_threads()
            for pair in cutlass.range(self.pairs, unroll=1):
                stage = pair % self.nstage
                phase = pair // self.nstage % 2
                nxt = pair + self.nstage - 1
                if nxt < self.pairs:
                    if tid == 0:
                        nst = nxt % self.nstage
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bars + nst, self.stage_tx
                        )
                        cp_async_bulk_g2s(
                            state_s + nst * self.pair_bytes,
                            state_g + nxt * self.pair_bytes,
                            self.pair_bytes,
                            bar_s + nst * 8,
                        )
                        cp_async_bulk_g2s(
                            scale_s + nst * self.pair_scale_bytes,
                            scale_g + nxt * self.pair_scale_bytes,
                            self.pair_scale_bytes,
                            bar_s + nst * 8,
                        )
                        cp_async_bulk_g2s(
                            x_s + nst * self.pair_x_bytes,
                            x_g + nxt * self.pair_x_bytes,
                            self.pair_x_bytes,
                            bar_s + nst * 8,
                        )
                head = pair * 2 + hp_t
                h_glob = first_head + head
                x_val = x_next
                d_val = d_next
                if pair < self.pairs - 1:
                    x_next = x[batch_id, h_glob + 2, r_t]
                    d_next = Float32(D[h_glob + 2, r_t])
                cute.arch.mbarrier_wait(bars + stage, phase)
                tile_s = state_s + stage * self.pair_bytes
                tile_scale_s = scale_s + stage * self.pair_scale_bytes
                for rr in cutlass.range_constexpr(4):
                    row = rr * 32 + warp * 8 + subgroup
                    row_s = tile_s + row * self.dstate
                    w0, w1, w2, w3 = lds_v4_u32(row_s + chunk_a * 16)
                    w4, w5, w6, w7 = lds_v4_u32(row_s + chunk_b * 16)
                    scale = lds_f32(tile_scale_s + row * 4)
                    d0 = e4m3x16_dot_f32(
                        w0,
                        w1,
                        w2,
                        w3,
                        c_vals[0],
                        c_vals[1],
                        c_vals[2],
                        c_vals[3],
                        c_vals[4],
                        c_vals[5],
                        c_vals[6],
                        c_vals[7],
                        c_vals[8],
                        c_vals[9],
                        c_vals[10],
                        c_vals[11],
                        c_vals[12],
                        c_vals[13],
                        c_vals[14],
                        c_vals[15],
                    )
                    d1 = e4m3x16_dot_f32(
                        w4,
                        w5,
                        w6,
                        w7,
                        c_vals[16],
                        c_vals[17],
                        c_vals[18],
                        c_vals[19],
                        c_vals[20],
                        c_vals[21],
                        c_vals[22],
                        c_vals[23],
                        c_vals[24],
                        c_vals[25],
                        c_vals[26],
                        c_vals[27],
                        c_vals[28],
                        c_vals[29],
                        c_vals[30],
                        c_vals[31],
                    )
                    partial = d0 + d1
                    partial = cute.arch.warp_reduction_sum(partial, threads_in_group=4)
                    if lig == 0:
                        s_ckpt[row] = partial * scale
                if pair == 0:
                    for j in cutlass.range_constexpr(4):
                        cache_pos = warp + 4 * j
                        bc_partial = Float32(0.0)
                        if cache_pos <= wp:
                            q0, q1, q2, q3 = bf16x4_bits_to_f32x4(bq[j])
                            if cache_pos == wp:
                                q0 = s_bnew[4 * lane]
                                q1 = s_bnew[4 * lane + 1]
                                q2 = s_bnew[4 * lane + 2]
                                q3 = s_bnew[4 * lane + 3]
                            bc_partial = (
                                q0 * s_c[4 * lane]
                                + q1 * s_c[4 * lane + 1]
                                + q2 * s_c[4 * lane + 2]
                                + q3 * s_c[4 * lane + 3]
                            )
                        bc_partial = cute.arch.warp_reduction_sum(bc_partial)
                        if lane == 0:
                            s_bc[cache_pos] = bc_partial
                    cute.arch.sync_threads()
                    for hr in cutlass.range_constexpr(self.hpg // 8):
                        head_r = hr * 8 + warp * 2 + lane // 16
                        cache_pos = lane % 16
                        dt_value = Float32(0.0)
                        if cache_pos < wp:
                            dt_value = dtq[hr]
                        if cache_pos == wp:
                            dt_value = dt_new[hr]
                        prefix = dt_value
                        for offset in (1, 2, 4, 8):
                            lower = cute.arch.shuffle_sync_up(
                                prefix, offset, mask_and_clamp=0
                            )
                            if cache_pos >= offset:
                                prefix += lower
                        total_lo = cute.arch.shuffle_sync(prefix, 15)
                        total_hi = cute.arch.shuffle_sync(prefix, 31)
                        total = total_lo
                        if lane >= 16:
                            total = total_hi
                        a_h = av[hr]
                        d_a_total = a_h * total
                        value = Float32(0.0)
                        if cache_pos <= wp:
                            value = dt_value * cute.math.exp(
                                d_a_total - a_h * prefix, fastmath=True
                            )
                        s_replay[head_r * self.cache_len + cache_pos] = (
                            value * s_bc[cache_pos]
                        )
                        if cache_pos == 0:
                            s_decay[head_r] = cute.math.exp(d_a_total, fastmath=True)
                cute.arch.sync_threads()
                x_row_s = (
                    x_s + stage * self.pair_x_bytes + hp_t * self.head_x_bytes + r_t * 2
                )
                cached = Float32(0.0)
                for c in cutlass.range_constexpr(self.cache_len):
                    xv = lds_bf16_f32(x_row_s + c * (self.head_dim * 2))
                    if c == wp:
                        xv = Float32(0.0)
                    cached += xv * s_replay[head * self.cache_len + c]
                x_cur = Float32(x_val)
                value = s_ckpt[tid] * s_decay[head]
                value += cached + x_cur * s_replay[head * self.cache_len + wp]
                value += x_cur * d_val
                out[batch_id, h_glob, r_t] = BFloat16(value)
                if live:
                    x_cache[state_idx, h_glob, wp, r_t] = x_val
                cute.arch.sync_threads()

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
        x_cache = make_fake_tensor(
            BFloat16,
            (slots, nheads, cache_len, head_dim),
            divisibility=16,
            slot_stride=slot_strides[0],
        )
        dt_cache = make_fake_tensor(
            Float32,
            (slots, nheads, cache_len),
            divisibility=16,
            slot_stride=slot_strides[1],
        )
        B_cache = make_fake_tensor(
            BFloat16,
            (slots, ngroups, cache_len, dstate),
            divisibility=16,
            slot_stride=slot_strides[2],
        )
        state_lo = make_fake_tensor(
            Int8,
            (slots, nheads, head_dim, dstate),
            divisibility=16,
            slot_stride=slot_strides[3],
        )
        state_lo_scale = make_fake_tensor(
            Float32,
            (slots, nheads, head_dim),
            divisibility=16,
            slot_stride=slot_strides[4],
        )
        state_indices = make_fake_tensor(Int32, (batch,), divisibility=1)
        write_pos = make_fake_tensor(Int32, (batch,), divisibility=1)
        is_flush = make_fake_tensor(Int8, (batch,), divisibility=1)
        out = make_fake_tensor(BFloat16, (batch, nheads, head_dim), divisibility=16)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Sm103Mamba2Fp8CheckpointFused(nheads, ngroups, head_dim, dstate, cache_len),
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            x_cache,
            dt_cache,
            B_cache,
            state_lo,
            state_lo_scale,
            state_indices,
            write_pos,
            is_flush,
            out,
            stream,
            options="--enable-tvm-ffi",
        )


def mamba2_fp8_nonflush(
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
) -> torch.Tensor:
    nheads = x.shape[1]
    head_dim = x.shape[2]
    ngroups = B.shape[1]
    dstate = B.shape[2]
    cache_len = x_cache.shape[2]
    Sm103Mamba2Fp8CheckpointFused.compile(
        nheads,
        ngroups,
        head_dim,
        dstate,
        cache_len,
        (
            x_cache.stride(0),
            dt_cache.stride(0),
            B_cache.stride(0),
            state_lo.stride(0),
            state_lo_scale.stride(0),
        ),
    )(
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        x_cache,
        dt_cache,
        B_cache,
        state_lo.view(torch.int8),
        state_lo_scale,
        state_indices,
        write_pos,
        is_flush.view(torch.int8),
        out,
    )
    return out
