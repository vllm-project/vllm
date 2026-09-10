# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B008 -- FlyDSL stream defaults are typed launch arguments
"""Bank-balanced wave32 FP8 GEMM for RDNA4.

Raw row-major operands are staged in LDS. Every K128 partial is completed in
FP32 before its independent block scales are applied.
"""

from dataclasses import dataclass
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Vector as Vec

from .gfx12_sync import lds_fence_signal, lds_fence_wait
from .rdna4_fp8_blockscale_common import (
    SCALE_K,
    WMMA_K,
    WMMA_M,
    WMMA_N,
    _f32_to_bf16_rne,
    _load_f32,
    _load_fp8_fragment_ptr,
    _make_buffer,
)


@dataclass(frozen=True)
class Wave32Config:
    tile_m: int
    group_m: int
    a_prefetch: int
    b_prefetch: int
    k_rotate: int = 0
    reg_m: int = 2
    split_k: int = 1
    tile_n: int = 128
    reg_n: int = 4
    uniform_barrier: bool = False
    lds_lookahead: bool = False
    cu_mode: bool = False
    n_major: bool = False


def select_wave32_config(m: int, n: int, k: int) -> Wave32Config:
    """Choose CU-local tiles without requiring a compiler wave-size override."""
    thin_ctas = ((m + 63) // 64) * (n // 128)
    if thin_ctas <= 32 and k >= 1024:
        # More, smaller tiles fill the CUs on narrow grids. Rotate K to spread
        # concurrent weight reads; each 64-column tile shares one scale row.
        return Wave32Config(
            tile_m=32,
            tile_n=64,
            reg_m=2,
            reg_n=2,
            group_m=8,
            a_prefetch=4,
            b_prefetch=8,
            k_rotate=2,
            split_k=2 if k >= 4096 and k % 256 == 0 else 1,
            uniform_barrier=True,
            cu_mode=True,
            lds_lookahead=True,
        )
    if n < 4096:
        return Wave32Config(
            tile_m=64,
            tile_n=64,
            reg_m=2,
            group_m=8,
            a_prefetch=2,
            b_prefetch=8,
            k_rotate=2,
            uniform_barrier=True,
            cu_mode=True,
        )
    tile_m = 128 if m >= 1024 or m % 128 == 0 else 64
    return Wave32Config(
        tile_m=tile_m,
        group_m=8,
        a_prefetch=0,
        b_prefetch=1 if tile_m == 128 else 0,
        k_rotate=0 if m >= 1024 else 2,
        reg_m=4 if m >= 4096 else 2,
        reg_n=2 if m >= 4096 else 4,
        uniform_barrier=True,
        cu_mode=True,
        lds_lookahead=m >= 1024,
        n_major=1024 <= m < 4096,
    )


def _row_swizzle(row):
    return (row // fx.Int32(2)) % fx.Int32(8)


def _store_fp8x16(shared, index, fragment):
    words = Vec(fragment.load())
    fx.ptr_store(
        words, fx.recast_iter(fx.Uint32, fx.get_iter(shared)) + index // fx.Int32(4)
    )


def _load_wordsx4(buffer, index, fragment):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Uint32)
    fx.copy(atom, fx.slice(buffer, (None, index // fx.Int32(4))), fragment)


def _store_bf16x8(buffer, index, values):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    fragment = fx.make_rmem_tensor(8, fx.BFloat16)
    fragment.store(Vec.from_elements(values, fx.BFloat16))
    fx.copy(atom, fragment, fx.slice(buffer, (None, index)))


def _create_wave32_module(n: int, k: int, stride_b: int, config: Wave32Config):
    """Create a single-stage LDS/WMMA prefill kernel."""
    bm = config.tile_m
    rm = config.reg_m
    rn = config.reg_n
    bn = config.tile_n
    wn = rn * 16
    nwaves = bn // wn
    assert bn % wn == 0 and rn in (2, 4)
    wm = rm * 16
    accs = rm * rn
    wave_size = 32
    assert bm % (rm * 16) == 0
    fragment_size = 8
    threads = (bm // (rm * 16)) * nwaves * wave_size
    load_fragment = _load_fp8_fragment_ptr
    assert bm in (32, 64, 128) and rm in (1, 2, 4)
    assert k % 128 == 0
    assert config.group_m > 0
    a_loads = bm * 128 // (16 * threads)
    b_loads = bn * 128 // (16 * threads)
    scale_blocks = k // SCALE_K
    splits = config.split_k

    assert splits in (1, 2)
    assert scale_blocks % splits == 0
    partition_blocks = scale_blocks // splits
    grid_n = n // bn
    lds_a_elems = bm * 128
    lds_b_elems = bn * 128
    output_rows = wm
    lds_partition_elems = max(lds_a_elems + lds_b_elems, output_rows * bn * 4)
    lds_elems = lds_partition_elems * splits
    assert lds_elems <= 65536
    cache_tag = (
        bm,
        bn,
        rn,
        config.group_m,
        config.a_prefetch,
        config.b_prefetch,
        config.k_rotate,
        rm,
        splits,
        wave_size,
        config.uniform_barrier,
        config.lds_lookahead,
        config.cu_mode,
        config.n_major,
    )
    assert n % bn == 0
    assert 0 <= config.a_prefetch <= a_loads
    assert 0 <= config.b_prefetch <= b_loads
    fp8 = fx.Float8E4M3FN
    f32 = fx.Float32
    bf16 = fx.BFloat16
    kernel_attrs = {
        "rocdl.waves_per_eu": 1,
        "rocdl.flat_work_group_size": f"{threads * splits},{threads * splits}",
    }

    if config.cu_mode:
        # Keep each workgroup local to a CU and overlap independent LDS/WMMA
        # instructions. These kernel attributes also work on FlyDSL main.
        kernel_attrs["llvm.passthrough"] = [
            ["target-features", "+cumode,+wavefrontsize32,-wavefrontsize64"],
            ["amdgpu-sched-strategy", "max-ilp"],
        ]

    @fx.struct
    class SharedStorage:
        data: fx.Array[fp8, lds_elems, 16]

    if config.group_m == 1:

        def map_pid(pid, grid_m):
            del grid_m
            return (pid // fx.Int32(grid_n), pid % fx.Int32(grid_n))
    else:

        def map_pid(pid, grid_m):
            num_pid_in_group = fx.Int32(config.group_m * grid_n)
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * fx.Int32(config.group_m)
            remaining_pid_m = grid_m - first_pid_m
            group_size_m = (remaining_pid_m < fx.Int32(config.group_m)).select(
                remaining_pid_m, fx.Int32(config.group_m)
            )
            pid_in_group = pid % num_pid_in_group
            pid_m = first_pid_m + pid_in_group % group_size_m
            pid_n = pid_in_group // group_size_m
            return (pid_m, pid_n)

    @flyc.kernel
    def wave32_kernel(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_as: fx.Tensor,
        arg_bs: fx.Tensor,
        arg_out: fx.Tensor,
        arg_m: fx.Int32,
    ):
        m_extent = arg_m
        grid_m = (m_extent + fx.Int32(bm - 1)) // fx.Int32(bm)
        tid = fx.thread_idx.x % fx.Int32(threads)
        partition_id = fx.thread_idx.x // fx.Int32(threads)
        wave = tid // fx.Int32(wave_size)
        wave_m = wave // fx.Int32(nwaves)
        wave_n = wave % fx.Int32(nwaves)
        lane = tid % fx.Int32(wave_size)
        lane16 = lane % fx.Int32(WMMA_N)
        lane_half = lane // fx.Int32(WMMA_N)
        lane_offset = lane_half * fx.Int32(8)
        pid = fx.block_idx.x
        pid_m, pid_n = map_pid(pid, grid_m)
        m0 = pid_m * fx.Int32(bm)
        n0 = pid_n * fx.Int32(bn)
        partition_start = partition_id * fx.Int32(partition_blocks)
        first_local_kb = pid_n * fx.Int32(config.k_rotate) % fx.Int32(partition_blocks)
        first_kb = partition_start + first_local_kb
        a_buf = _make_buffer(arg_a, fx.Uint32, 4, m_extent * fx.Int32(k))
        b_buf = _make_buffer(arg_b, fx.Uint32, 4, n * stride_b)
        as_buf = _make_buffer(arg_as, f32, 1, m_extent * fx.Int32(scale_blocks * 4))
        bs_ptr = fx.recast_iter(f32, fx.get_iter(arg_bs))
        out_buf = _make_buffer(
            arg_out,
            bf16,
            fragment_size if splits == 1 else 8,
            m_extent * fx.Int32(n * 2),
        )
        shared = fx.SharedAllocator().allocate(SharedStorage).peek().data
        aligned_lds_ty = fx.PointerType.get(fp8.ir_type, fx.AddressSpace.Shared, 16)
        lds_a_ptr = fx.recast_iter(
            aligned_lds_ty, shared.ptr + partition_id * fx.Int32(lds_partition_elems)
        )
        lds_b_ptr = lds_a_ptr + fx.Int32(lds_a_elems)
        lds_a = fx.make_view(lds_a_ptr, fx.make_layout((16, 1), (1, 1)))
        lds_b = fx.make_view(lds_b_ptr, fx.make_layout((16, 1), (1, 1)))
        mma = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fp8, f32))
        totals = [
            fx.make_rmem_tensor(fragment_size, f32) for _ in range_constexpr(accs)
        ]
        for ni in range_constexpr(accs):
            totals[ni].fill(0)
        staged_a = [
            fx.make_rmem_tensor(4, fx.Uint32)
            for _ in range_constexpr(config.a_prefetch)
        ]
        v = row = ko = global_row = fx.Int32(0)
        for q in range_constexpr(config.a_prefetch):
            v = tid + fx.Int32(q * threads)
            row = v // fx.Int32(128 // 16)
            ko = v % fx.Int32(128 // 16) * fx.Int32(16)
            global_row = m0 + row
            _load_wordsx4(
                a_buf,
                global_row * fx.Int32(k) + first_kb * fx.Int32(SCALE_K) + ko,
                staged_a[q],
            )
        staged_b = [
            fx.make_rmem_tensor(4, fx.Uint32)
            for _ in range_constexpr(config.b_prefetch)
        ]
        for q in range_constexpr(config.b_prefetch):
            v = tid + fx.Int32(q * threads)
            row = v // fx.Int32(128 // 16)
            ko = v % fx.Int32(128 // 16) * fx.Int32(16)
            _load_wordsx4(
                b_buf,
                (n0 + row) * fx.Int32(stride_b) + first_kb * fx.Int32(SCALE_K) + ko,
                staged_b[q],
            )
        for k_iter in range(0, partition_blocks, 1):
            local_kb = fx.Int32(k_iter)
            if const_expr(config.k_rotate != 0):
                local_kb = (local_kb + first_local_kb) % fx.Int32(partition_blocks)
            kb = partition_start + local_kb
            # Uniform startup avoids loop peeling on fully prefetched thin tiles.
            if const_expr(config.uniform_barrier):
                lds_fence_signal()
                lds_fence_wait()
            else:
                if k_iter != 0:
                    lds_fence_signal()
                    lds_fence_wait()
            for q in range_constexpr(config.a_prefetch):
                v = tid + fx.Int32(q * threads)
                row = v // fx.Int32(128 // 16)
                ko = v % fx.Int32(128 // 16) * fx.Int32(16)
                sw_ko = (ko // fx.Int32(16) ^ _row_swizzle(row)) * fx.Int32(16)
                _store_fp8x16(lds_a, row * fx.Int32(128) + sw_ko, staged_a[q])
            for q in range_constexpr(config.a_prefetch, a_loads):
                v = tid + fx.Int32(q * threads)
                row = v // fx.Int32(128 // 16)
                ko = v % fx.Int32(128 // 16) * fx.Int32(16)
                fragment = fx.make_rmem_tensor(4, fx.Uint32)
                global_row = m0 + row
                _load_wordsx4(
                    a_buf,
                    global_row * fx.Int32(k) + fx.Int32(kb * SCALE_K) + ko,
                    fragment,
                )
                sw_ko = (ko // fx.Int32(16) ^ _row_swizzle(row)) * fx.Int32(16)
                _store_fp8x16(lds_a, row * fx.Int32(128) + sw_ko, fragment)
            for q in range_constexpr(config.b_prefetch):
                v = tid + fx.Int32(q * threads)
                row = v // fx.Int32(128 // 16)
                ko = v % fx.Int32(128 // 16) * fx.Int32(16)
                swizzled_ko = (ko // fx.Int32(16) ^ _row_swizzle(row)) * fx.Int32(16)
                _store_fp8x16(lds_b, row * fx.Int32(128) + swizzled_ko, staged_b[q])
            for q in range_constexpr(config.b_prefetch, b_loads):
                v = tid + fx.Int32(q * threads)
                row = v // fx.Int32(128 // 16)
                ko = v % fx.Int32(128 // 16) * fx.Int32(16)
                fragment = fx.make_rmem_tensor(4, fx.Uint32)
                _load_wordsx4(
                    b_buf,
                    (n0 + row) * fx.Int32(stride_b) + fx.Int32(kb * SCALE_K) + ko,
                    fragment,
                )
                swizzled_ko = (ko // fx.Int32(16) ^ _row_swizzle(row)) * fx.Int32(16)
                _store_fp8x16(lds_b, row * fx.Int32(128) + swizzled_ko, fragment)
            lds_fence_signal()
            if k_iter + 1 < partition_blocks:
                next_local = local_kb + fx.Int32(1)
                if const_expr(config.k_rotate != 0):
                    next_local = next_local % fx.Int32(partition_blocks)
                next_k0 = (partition_start + next_local) * fx.Int32(SCALE_K)
                for q in range_constexpr(config.a_prefetch):
                    v = tid + fx.Int32(q * threads)
                    row = v // fx.Int32(128 // 16)
                    ko = v % fx.Int32(128 // 16) * fx.Int32(16)
                    global_row = m0 + row
                    _load_wordsx4(
                        a_buf, global_row * fx.Int32(k) + next_k0 + ko, staged_a[q]
                    )
                for q in range_constexpr(config.b_prefetch):
                    v = tid + fx.Int32(q * threads)
                    row = v // fx.Int32(128 // 16)
                    ko = v % fx.Int32(128 // 16) * fx.Int32(16)
                    _load_wordsx4(
                        b_buf,
                        (n0 + row) * fx.Int32(stride_b) + next_k0 + ko,
                        staged_b[q],
                    )
            lds_fence_wait()
            for bi in range_constexpr(1):
                b_group = (n0 + wave_n * fx.Int32(wn)) // fx.Int32(SCALE_K)
                if const_expr(bn <= SCALE_K):
                    b_group = n0 // fx.Int32(SCALE_K)
                b_scale = f32(
                    fx.ptr_load(
                        bs_ptr + b_group * fx.Int32(scale_blocks) + fx.Int32(kb + bi)
                    )
                )
                row_scales = []
                for mi in range_constexpr(rm):
                    scales = []
                    for ri in range_constexpr(fragment_size):
                        row = (
                            m0
                            + wave_m * fx.Int32(wm)
                            + fx.Int32(mi * WMMA_M)
                            + lane_offset
                            + fx.Int32(ri)
                        )
                        row = (
                            m0 + wave_m * fx.Int32(wm) + fx.Int32(mi * WMMA_M) + lane16
                        )
                        scales.append(
                            _load_f32(
                                as_buf, row * fx.Int32(scale_blocks) + fx.Int32(kb + bi)
                            )
                            * f32(1.0)
                        )
                    row_scales.append(scales)
                for mg in range_constexpr(rm // rm):
                    partials = [
                        fx.make_rmem_tensor(fragment_size, f32)
                        for _ in range_constexpr(rm * rn)
                    ]
                    for ni in range_constexpr(rm * rn):
                        partials[ni].fill(0)
                    a_row_base = wave_m * fx.Int32(wm) + lane16
                    b_row_base = wave_n * fx.Int32(wn) + lane16
                    b_lane_base = b_row_base * fx.Int32(128)
                    word_flip = fx.Int32(0)
                    if const_expr(config.lds_lookahead):
                        staged_a_lds = [
                            fx.make_rmem_tensor(fragment_size, fp8)
                            for _ in range_constexpr(rm)
                        ]
                        for mi in range_constexpr(rm):
                            a_sw_k = (
                                fx.Int32(bi * 8) ^ _row_swizzle(a_row_base)
                            ) * fx.Int32(16) + lane_offset ^ word_flip
                            a_index = (
                                a_row_base * fx.Int32(128)
                                + fx.Int32((mg * rm + mi) * WMMA_M * 128)
                                + a_sw_k
                            )
                            load_fragment(lds_a_ptr, a_index, staged_a_lds[mi])
                        staged_b_lds = [
                            fx.make_rmem_tensor(fragment_size, fp8)
                            for _ in range_constexpr(rn)
                        ]
                        swizzled_k = (
                            fx.Int32(bi * 8) ^ _row_swizzle(b_row_base)
                        ) * fx.Int32(16) + lane_offset
                        swizzled_k = swizzled_k ^ word_flip
                        for ni in range_constexpr(rn):
                            load_fragment(
                                lds_b_ptr,
                                b_lane_base + fx.Int32(ni * WMMA_N * 128) + swizzled_k,
                                staged_b_lds[ni],
                            )
                    for ki in range_constexpr(SCALE_K // WMMA_K):
                        a_frags = [
                            fx.make_rmem_tensor(fragment_size, fp8)
                            for _ in range_constexpr(rm)
                        ]
                        b_frags = [
                            fx.make_rmem_tensor(fragment_size, fp8)
                            for _ in range_constexpr(rn)
                        ]
                        if const_expr(config.lds_lookahead):
                            for mi in range_constexpr(rm):
                                a_frags[mi].store(staged_a_lds[mi].load())
                            for ni in range_constexpr(rn):
                                b_frags[ni].store(staged_b_lds[ni].load())
                            if const_expr(ki + 1 < SCALE_K // WMMA_K):
                                staged_a_lds = [
                                    fx.make_rmem_tensor(fragment_size, fp8)
                                    for _ in range_constexpr(rm)
                                ]
                                for mi in range_constexpr(rm):
                                    a_sw_k = (
                                        fx.Int32(bi * 8 + ki + 1)
                                        ^ _row_swizzle(a_row_base)
                                    ) * fx.Int32(16) + lane_offset ^ word_flip
                                    a_index = (
                                        a_row_base * fx.Int32(128)
                                        + fx.Int32((mg * rm + mi) * WMMA_M * 128)
                                        + a_sw_k
                                    )
                                    load_fragment(lds_a_ptr, a_index, staged_a_lds[mi])
                                staged_b_lds = [
                                    fx.make_rmem_tensor(fragment_size, fp8)
                                    for _ in range_constexpr(rn)
                                ]
                                swizzled_k = (
                                    fx.Int32(bi * 8 + ki + 1) ^ _row_swizzle(b_row_base)
                                ) * fx.Int32(16) + lane_offset ^ word_flip
                                for ni in range_constexpr(rn):
                                    load_fragment(
                                        lds_b_ptr,
                                        b_lane_base
                                        + fx.Int32(ni * WMMA_N * 128)
                                        + swizzled_k,
                                        staged_b_lds[ni],
                                    )
                        else:
                            for mi in range_constexpr(rm):
                                a_sw_k = (
                                    fx.Int32(bi * 8 + ki) ^ _row_swizzle(a_row_base)
                                ) * fx.Int32(16) + lane_offset ^ word_flip
                                a_index = (
                                    a_row_base * fx.Int32(128)
                                    + fx.Int32((mg * rm + mi) * WMMA_M * 128)
                                    + a_sw_k
                                )
                                load_fragment(lds_a_ptr, a_index, a_frags[mi])
                            swizzled_k = (
                                fx.Int32(bi * 8 + ki) ^ _row_swizzle(b_row_base)
                            ) * fx.Int32(16) + lane_offset
                            swizzled_k = swizzled_k ^ word_flip
                            for ni in range_constexpr(rn):
                                load_fragment(
                                    lds_b_ptr,
                                    b_lane_base
                                    + fx.Int32(ni * WMMA_N * 128)
                                    + swizzled_k,
                                    b_frags[ni],
                                )
                        for outer in range_constexpr(rn if config.n_major else rm):
                            for inner in range_constexpr(rm if config.n_major else rn):
                                mi = inner if const_expr(config.n_major) else outer
                                ni = outer if const_expr(config.n_major) else inner
                                idx = mi * rn + ni
                                fx.gemm(
                                    mma,
                                    partials[idx],
                                    b_frags[ni],
                                    a_frags[mi],
                                    partials[idx],
                                )
                        fx.rocdl.sched_barrier(0)
                    for mi in range_constexpr(rm):
                        for ni in range_constexpr(rn):
                            idx = mi * rn + ni
                            total_idx = (mg * rm + mi) * rn + ni
                            total_v = Vec(totals[total_idx].load())
                            partial_v = Vec(partials[idx].load())
                            totals[total_idx].store(
                                Vec.from_elements(
                                    [
                                        fx.fma(
                                            partial_v[ri],
                                            row_scales[mg * rm + mi][ri] * b_scale,
                                            total_v[ri],
                                        )
                                        for ri in range_constexpr(fragment_size)
                                    ],
                                    f32,
                                )
                            )
        if const_expr(splits == 1):
            for mi in range_constexpr(rm):
                row = m0 + wave_m * fx.Int32(wm) + fx.Int32(mi * 16) + lane16
                if row < m_extent:
                    for ni in range_constexpr(rn):
                        values = Vec(totals[mi * rn + ni].load())
                        col = (
                            n0 + wave_n * fx.Int32(wn) + fx.Int32(ni * 16) + lane_offset
                        )
                        converted = [
                            _f32_to_bf16_rne(values[ri])
                            for ri in range_constexpr(fragment_size)
                        ]
                        _store_bf16x8(out_buf, row * fx.Int32(n) + col, converted)
        else:
            gpu.barrier()
            lds_out_ptr = fx.recast_iter(f32, lds_a_ptr)
            output_wave_rows = output_rows // wm
            for group in range_constexpr(bm // output_rows):
                if wave_m // fx.Int32(output_wave_rows) == fx.Int32(group):
                    local_wave_m = wave_m % fx.Int32(output_wave_rows)
                    for mi in range_constexpr(rm):
                        for ni in range_constexpr(rn):
                            idx = mi * rn + ni
                            local_row = (
                                local_wave_m * fx.Int32(wm)
                                + fx.Int32(mi * WMMA_M)
                                + lane16
                            )
                            local_col = (
                                wave_n * fx.Int32(wn)
                                + fx.Int32(ni * WMMA_N)
                                + lane_offset
                            )
                            fx.ptr_store(
                                Vec(totals[idx].load()),
                                lds_out_ptr + local_row * fx.Int32(bn) + local_col,
                            )
                gpu.barrier()
                for q in range_constexpr(output_rows * bn // 8 // threads):
                    v = tid + fx.Int32(q * threads)
                    local_row = v // fx.Int32(bn // 8)
                    col = v % fx.Int32(bn // 8) * fx.Int32(8)
                    row = m0 + fx.Int32(group * output_rows) + local_row
                    if (row < m_extent) & (partition_id == fx.Int32(0)):
                        values = fx.ptr_load(
                            lds_out_ptr + local_row * fx.Int32(bn) + col,
                            result_type=fx.Vector.make_type(8, f32),
                        )
                        values_v = Vec(values)
                        if const_expr(splits > 1):
                            other_values = fx.ptr_load(
                                lds_out_ptr
                                + fx.Int32(lds_partition_elems // 4)
                                + local_row * fx.Int32(bn)
                                + col,
                                result_type=fx.Vector.make_type(8, f32),
                            )
                            values_v = values_v + Vec(other_values)
                        _store_bf16x8(
                            out_buf,
                            row * fx.Int32(n) + n0 + col,
                            [_f32_to_bf16_rne(values_v[i]) for i in range_constexpr(8)],
                        )
                gpu.barrier()

    @flyc.jit
    def launch(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_as: fx.Tensor,
        arg_bs: fx.Tensor,
        arg_out: fx.Tensor,
        arg_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = cache_tag
        launch_m = arg_m
        total_blocks = (launch_m + fx.Int32(bm - 1)) // fx.Int32(bm) * fx.Int32(grid_n)
        wave32_kernel(
            arg_a, arg_b, arg_as, arg_bs, arg_out, arg_m, value_attrs=kernel_attrs
        ).launch(
            grid=(total_blocks, 1, 1), block=(threads * splits, 1, 1), stream=stream
        )

    return launch


@lru_cache(maxsize=128)
def _get_wave32_module(n, k, stride_b, config):
    return _create_wave32_module(n, k, stride_b, config)
