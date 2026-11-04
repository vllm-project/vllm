# SPDX-License-Identifier: Apache-2.0
"""MAMBA2 nvfp4 nonflush; fixed L=16 specialization."""

from __future__ import annotations

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int8, Int32, Int64, Uint32, Uint64, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from vllm.third_party.replayssm.common import make_fake_tensor

from vllm.third_party.replayssm.common.cvt_fp4 import (
    e2m1_bytes8_fma_f16x2_to_f32,
    f32x2_to_f16x2_rn,
)
from vllm.third_party.replayssm.common.sync import _exit_if_nonzero
from vllm.third_party.replayssm.common.nf_ops import (
    bf16x4_bits_to_f32x4,
    cp_async_bulk_g2s,
    lds_bf16_f32,
    lds_v4_u32,
)


@dsl_user_op
def lds_e4m3x2_to_f32x2(smem_addr, *, loc=None, ip=None):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        (
            "{\n"
            "\t.reg .b16 v, lo, hi;\n"
            "\t.reg .b32 p;\n"
            "\tld.shared.u16 v, [$2];\n"
            "\tcvt.rn.f16x2.e4m3x2 p, v;\n"
            "\tmov.b32 {lo, hi}, p;\n"
            "\tcvt.f32.f16 $0, lo;\n"
            "\tcvt.f32.f16 $1, hi;\n"
            "\t}\n"
        ),
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    lo = Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip))
    hi = Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip))
    return (lo, hi)


class Sm103Mamba2Tp4CheckpointFused:
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
        self.nstage = 3
        self.total_pairs = nheads // 2
        self.head_dim = head_dim
        self.dstate = dstate
        self.cache_len = cache_len
        self.num_warps = 4
        self.row_bytes = dstate // 2
        self.pair_bytes = 2 * head_dim * self.row_bytes
        self.pair_sf_bytes = 4 * 256
        self.head_x_bytes = cache_len * head_dim * 2
        self.pair_x_bytes = 2 * self.head_x_bytes
        self.stage_tx = self.pair_bytes + self.pair_sf_bytes + self.pair_x_bytes

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
        sf_lo: cute.Tensor,
        sg_meta: cute.Tensor,
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
            sf_lo,
            sg_meta,
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
        sf_lo: cute.Tensor,
        sg_meta: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        is_flush: cute.Tensor,
        out: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        batch_id, group, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        s_state = smem.allocate_tensor(
            Int8, cute.make_layout(3 * self.pair_bytes), byte_alignment=128
        )
        s_sf = smem.allocate_tensor(
            Int8, cute.make_layout(3 * self.pair_sf_bytes), byte_alignment=128
        )
        s_x = smem.allocate_tensor(
            Int8, cute.make_layout(3 * self.pair_x_bytes), byte_alignment=128
        )
        s_c = smem.allocate_array(Float32, self.dstate)
        s_bnew = smem.allocate_array(Float32, self.dstate)
        s_bc = smem.allocate_array(Float32, self.cache_len)
        s_replay = smem.allocate_array(Float32, self.hpg * self.cache_len)
        s_decay = smem.allocate_array(Float32, self.hpg)
        s_ckpt = smem.allocate_array(Float32, 2 * self.head_dim)
        bars = smem.allocate_array(Int64, self.nstage)
        flag = Int32(is_flush[batch_id])
        warp = tid // 32
        lane = tid % 32
        subgroup = lane // 4
        lig = lane % 4
        first_head = group * self.hpg
        first_pair = group * self.pairs
        state_idx = state_indices[batch_id]
        _exit_if_nonzero(Int32(state_idx <= 0))
        wp = write_pos[batch_id]
        hp_t = tid // self.head_dim
        r_t = tid % self.head_dim
        sg_state = Float32(sg_meta[state_idx, 2, group])
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
        C_u64 = cute.recast_tensor(C, Uint64)
        q_words = cute.make_rmem_tensor((16,), Uint32)
        for k in cutlass.range_constexpr(8):
            a0, a1, a2, a3 = bf16x4_bits_to_f32x4(C_u64[batch_id, group, lig * 8 + k])
            q_words[2 * k] = f32x2_to_f16x2_rn(a0, a1)
            q_words[2 * k + 1] = f32x2_to_f16x2_rn(a2, a3)
        x_next = x[batch_id, first_head + hp_t, r_t]
        d_next = Float32(D[first_head + hp_t, r_t])
        x_next2 = x[batch_id, first_head + 2 + hp_t, r_t]
        d_next2 = Float32(D[first_head + 2 + hp_t, r_t])
        sf_lane = 2 * lig % 4 + 2 * lig // 4 * 512
        if flag == 0:
            live = state_idx != 0
            if tid == 0:
                for s in cutlass.range_constexpr(self.nstage):
                    cute.arch.mbarrier_init(bars + s, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            state_g = (
                state_lo.iterator.toint()
                + Int64(state_idx) * state_lo.stride[0]
                + Int64(first_pair * self.pair_bytes)
            )
            sf_g = (
                sf_lo.iterator.toint()
                + Int64(state_idx) * sf_lo.stride[0]
                + Int64(first_pair * self.pair_sf_bytes)
            )
            x_g = (
                x_cache.iterator.toint()
                + Int64(state_idx) * x_cache.stride[0] * 2
                + Int64(first_head * self.head_x_bytes)
            )
            state_s = s_state.iterator.toint()
            sf_s = s_sf.iterator.toint()
            x_s = s_x.iterator.toint()
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
                        sf_s + s * self.pair_sf_bytes,
                        sf_g + s * self.pair_sf_bytes,
                        self.pair_sf_bytes,
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
                            sf_s + nst * self.pair_sf_bytes,
                            sf_g + nxt * self.pair_sf_bytes,
                            self.pair_sf_bytes,
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
                x_next = x_next2
                d_next = d_next2
                if pair < self.pairs - 2:
                    x_next2 = x[batch_id, h_glob + 4, r_t]
                    d_next2 = Float32(D[h_glob + 4, r_t])
                cute.arch.mbarrier_wait(bars + stage, phase)
                tile_s = state_s + stage * self.pair_bytes
                tile_sf_s = sf_s + stage * self.pair_sf_bytes
                for rr in cutlass.range_constexpr(4):
                    row = rr * 32 + warp * 8 + subgroup
                    w0, w1, w2, w3 = lds_v4_u32(
                        tile_s + row * self.row_bytes + lig * 16
                    )
                    sf_off = row % 32 * 16 + row // 32 % 4 * 4 + sf_lane
                    sa, sb = lds_e4m3x2_to_f32x2(tile_sf_s + sf_off)
                    d0 = e2m1_bytes8_fma_f16x2_to_f32(
                        w0,
                        w1,
                        q_words[0],
                        q_words[1],
                        q_words[2],
                        q_words[3],
                        q_words[4],
                        q_words[5],
                        q_words[6],
                        q_words[7],
                    )
                    d1 = e2m1_bytes8_fma_f16x2_to_f32(
                        w2,
                        w3,
                        q_words[8],
                        q_words[9],
                        q_words[10],
                        q_words[11],
                        q_words[12],
                        q_words[13],
                        q_words[14],
                        q_words[15],
                    )
                    partial = d0 * (sa * sg_state) + d1 * (sb * sg_state)
                    partial = cute.arch.warp_reduction_sum(partial, threads_in_group=4)
                    if lig == 0:
                        s_ckpt[row] = partial
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
        pairs = nheads // 2
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
            (slots, pairs, 2 * head_dim, dstate // 2),
            divisibility=16,
            slot_stride=slot_strides[3],
        )
        sf_lo = make_fake_tensor(
            Int8, (slots, pairs, 4, 256), divisibility=16, slot_stride=slot_strides[4]
        )
        sg_meta = make_fake_tensor(
            Float32, (slots, 3, ngroups), divisibility=1, slot_stride=slot_strides[5]
        )
        state_indices = make_fake_tensor(Int32, (batch,), divisibility=1)
        write_pos = make_fake_tensor(Int32, (batch,), divisibility=1)
        is_flush = make_fake_tensor(Int8, (batch,), divisibility=1)
        out = make_fake_tensor(BFloat16, (batch, nheads, head_dim), divisibility=16)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Sm103Mamba2Tp4CheckpointFused(nheads, ngroups, head_dim, dstate, cache_len),
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
            sf_lo,
            sg_meta,
            state_indices,
            write_pos,
            is_flush,
            out,
            stream,
            options="--enable-tvm-ffi",
        )


def mamba2_nvfp4_nonflush(
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
    sf_lo: torch.Tensor,
    sg_meta: torch.Tensor,
) -> torch.Tensor:
    slots = state_lo.shape[0]
    nheads = x.shape[1]
    head_dim = x.shape[2]
    ngroups = B.shape[1]
    dstate = B.shape[2]
    cache_len = x_cache.shape[2]
    Sm103Mamba2Tp4CheckpointFused.compile(
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
            sf_lo.stride(0),
            sg_meta.stride(0),
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
        sf_lo.view(torch.int8).view(slots, nheads // 2, 4, 256),
        sg_meta,
        state_indices,
        write_pos,
        is_flush.view(torch.int8),
        out,
    )
    return out
