# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B008 -- FlyDSL launch signatures require typed stream defaults

"""LDS-tiled RDNA4 block-scaled FP8 GEMM for broad prefill shapes.

Provides wave32 tiles for wide decode and short prefill. A and raw row-major B
are staged in LDS for cross-wave reuse. Every K=128 partial is completed in FP32
before its independent activation and weight scales are applied.
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
    WAVE_SIZE,
    WMMA_K,
    WMMA_M,
    WMMA_N,
    _f32_to_bf16_rne,
    _load_f32,
    _load_fp8_fragment_ptr,
    _make_buffer,
)
from .runtime import run_compiled

LDA = SCALE_K + 8
LDB = SCALE_K


@dataclass(frozen=True)
class PrefillConfig:
    tile_m: int
    tile_n: int
    group_m: int
    a_prefetch: int
    b_prefetch: int
    static_m: int = 0
    k_rotate: int = 0

    @property
    def threads(self) -> int:
        return (self.tile_m // 32) * (self.tile_n // 64) * WAVE_SIZE


def select_prefill_config(m: int, n: int, k: int, stride_b: int) -> PrefillConfig:
    """Select broad-M tiles and measured short-prefill specializations."""
    if (m, n, k, stride_b) == (39, 16384, 8192, 8192):
        return PrefillConfig(64, 128, 1, 2, 2, static_m=39, k_rotate=2)
    if m <= 64:
        raise ValueError(f"RDNA4 prefill route requires M > 64, got {m}")
    if n <= 0 or n % SCALE_K:
        raise ValueError(
            f"RDNA4 prefill route requires positive N divisible by 128, got {n}"
        )
    if k <= 0 or k % SCALE_K:
        raise ValueError(
            f"RDNA4 prefill route requires positive K divisible by 128, got {k}"
        )

    balanced_reuse = m >= 128 and (
        (n >= 16384 and k >= 4096)
        or (n >= 10240 and k >= 6144)
        or (n >= 7168 and k >= 10240)
        or (m >= 4096 and n >= 5120)
        or (m >= 896 and n >= 12288 and k >= 3072)
        or (m >= 768 and n >= 6144 and k >= 6144)
        or (384 <= m <= 640 and 5120 <= n < 8192)
    )
    short_m_weight_reuse = 96 <= m < 384 and (m & 127) != 0 and n >= 10240 and k >= 6144
    small_n_long_k = 128 <= m < 4096 and n <= 4096 and k >= 6144

    if (
        n % 256
        or short_m_weight_reuse
        or small_n_long_k
        or (192 <= m <= 384 and 4096 <= n <= 7168 and k >= 6144)
    ):
        return PrefillConfig(
            tile_m=32 if n == 128 else 64,
            tile_n=128,
            group_m=1 if n == 128 else 32,
            a_prefetch=2,
            b_prefetch=5,
        )
    if (m, n, k, stride_b) == (256, 8192, 5120, 5376):
        return PrefillConfig(
            tile_m=64,
            tile_n=256,
            group_m=2,
            a_prefetch=2,
            b_prefetch=2,
            static_m=256,
        )
    if balanced_reuse and n >= 16384:
        # The native HIP compiler favors its 128x128/256-thread geometry here,
        # but current FlyDSL gfx120x lowering pays a large occupancy cost for
        # that shape. Use the measured narrow tile for very wide N.
        return PrefillConfig(
            tile_m=128,
            tile_n=128,
            group_m=8,
            a_prefetch=4,
            b_prefetch=4,
        )
    return PrefillConfig(
        tile_m=64,
        tile_n=256,
        group_m=8,
        a_prefetch=2,
        b_prefetch=2,
    )


def _load_fp8x16(buffer, index, fragment):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FN)
    fx.copy(atom, fx.slice(buffer, (None, index)), fragment)


def _store_fp8x16(shared, index, fragment):
    atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float8E4M3FN)
    fx.copy(atom, fragment, fx.slice(shared, (None, index)))


def _store_bf16x8(buffer, index, values):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    fragment = fx.make_rmem_tensor(8, fx.BFloat16)
    fragment.store(Vec.from_elements(values, fx.BFloat16))
    fx.copy(atom, fragment, fx.slice(buffer, (None, index)))


def _create_prefill_module(
    n: int,
    k: int,
    stride_b: int,
    config: PrefillConfig,
):
    """Create a single-stage LDS/WMMA prefill kernel."""
    bm = config.tile_m
    bn = config.tile_n
    static_m = config.static_m
    threads = config.threads
    a_waves = bn // 64
    a_loads = (bm * SCALE_K) // (16 * threads)
    b_loads = (bn * SCALE_K) // (16 * threads)
    scale_blocks = k // SCALE_K
    grid_n = n // bn
    lds_a_elems = bm * LDA
    lds_b_elems = bn * LDB
    lds_elems = lds_a_elems + lds_b_elems
    # FlyDSL fingerprints scalar/tuple closures, but not dataclass fields.
    cache_tag = (
        bm,
        bn,
        config.group_m,
        config.a_prefetch,
        config.b_prefetch,
        static_m,
        config.k_rotate,
    )

    assert threads in (64, 128, 256)
    assert n % bn == 0
    assert config.a_prefetch <= a_loads
    assert config.b_prefetch <= b_loads

    fp8 = fx.Float8E4M3FN
    f32 = fx.Float32
    bf16 = fx.BFloat16
    kernel_attrs = {
        "rocdl.waves_per_eu": 1,
        "rocdl.flat_work_group_size": f"{threads},{threads}",
    }

    @fx.struct
    class SharedStorage:
        data: fx.Array[fp8, lds_elems, 16]

    if config.group_m == 1:

        def map_pid(pid, grid_m):
            del grid_m
            return pid // fx.Int32(grid_n), pid % fx.Int32(grid_n)

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
            return pid_m, pid_n

    @flyc.kernel
    def prefill_kernel(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_as: fx.Tensor,
        arg_bs: fx.Tensor,
        arg_out: fx.Tensor,
        arg_m: fx.Int32,
    ):
        m_extent = fx.Int32(static_m) if static_m else arg_m
        grid_m = (m_extent + fx.Int32(bm - 1)) // fx.Int32(bm)
        tid = fx.thread_idx.x
        wave = tid // fx.Int32(WAVE_SIZE)
        wave_m = wave // fx.Int32(a_waves)
        wave_n = wave % fx.Int32(a_waves)
        lane = tid % fx.Int32(WAVE_SIZE)
        lane16 = lane % fx.Int32(WMMA_N)
        lane_half = lane // fx.Int32(WMMA_N)

        pid = fx.block_idx.x
        # A grouped mapping with one M tile is just row-major. Specialize it
        # before tracing so gfx120x lowering sees only division/remainder by
        # the compile-time grid_n. The generic formulation computes a dynamic
        # group_size_m and currently lowers even its always-one remainder and
        # division through an FP32 reciprocal correction chain.
        pid_m, pid_n = map_pid(pid, grid_m)
        m0 = pid_m * fx.Int32(bm)
        n0 = pid_n * fx.Int32(bn)
        # Stagger strided weight reads across N tiles; visit every K block once.
        first_kb = (pid_n * fx.Int32(config.k_rotate)) % fx.Int32(scale_blocks)

        a_buf = _make_buffer(
            arg_a,
            fp8,
            16,
            m_extent * fx.Int32(k),
        )
        b_buf = _make_buffer(arg_b, fp8, 16, n * stride_b)
        as_buf = _make_buffer(
            arg_as,
            f32,
            1,
            m_extent * fx.Int32(scale_blocks * 4),
        )
        bs_ptr = fx.recast_iter(f32, fx.get_iter(arg_bs))
        out_buf = _make_buffer(arg_out, bf16, 8, m_extent * fx.Int32(n * 2))

        shared = fx.SharedAllocator().allocate(SharedStorage).peek().data
        lds_a_ptr = shared.ptr
        lds_b_ptr = shared.ptr + fx.Int32(lds_a_elems)
        lds_a = fx.make_view(lds_a_ptr, fx.make_layout((16, 1), (1, 1)))
        lds_b = fx.make_view(lds_b_ptr, fx.make_layout((16, 1), (1, 1)))

        mma = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fp8, f32))
        totals = [fx.make_rmem_tensor(8, f32) for _ in range_constexpr(8)]
        for ni in range_constexpr(8):
            totals[ni].fill(0)

        staged_a = [
            fx.make_rmem_tensor(16, fp8) for _ in range_constexpr(config.a_prefetch)
        ]
        for q in range_constexpr(config.a_prefetch):
            v = tid + fx.Int32(q * threads)
            row = v // fx.Int32(8)
            ko = (v % fx.Int32(8)) * fx.Int32(16)
            global_row = m0 + row
            _load_fp8x16(
                a_buf,
                global_row * fx.Int32(k) + first_kb * fx.Int32(SCALE_K) + ko,
                staged_a[q],
            )

        staged_b = [
            fx.make_rmem_tensor(16, fp8) for _ in range_constexpr(config.b_prefetch)
        ]
        for q in range_constexpr(config.b_prefetch):
            v = tid + fx.Int32(q * threads)
            row = v // fx.Int32(8)
            ko = (v % fx.Int32(8)) * fx.Int32(16)
            _load_fp8x16(
                b_buf,
                (n0 + row) * fx.Int32(stride_b) + first_kb * fx.Int32(SCALE_K) + ko,
                staged_b[q],
            )

        for k_iter in range(0, scale_blocks, 1):
            kb = fx.Int32(k_iter)
            if const_expr(config.k_rotate != 0):
                kb = (kb + first_kb) % fx.Int32(scale_blocks)
            if k_iter != 0:
                gpu.barrier()

            for q in range_constexpr(config.a_prefetch):
                v = tid + fx.Int32(q * threads)
                row = v // fx.Int32(8)
                ko = (v % fx.Int32(8)) * fx.Int32(16)
                _store_fp8x16(lds_a, row * fx.Int32(LDA) + ko, staged_a[q])
            for q in range_constexpr(config.a_prefetch, a_loads):
                v = tid + fx.Int32(q * threads)
                row = v // fx.Int32(8)
                ko = (v % fx.Int32(8)) * fx.Int32(16)
                fragment = fx.make_rmem_tensor(16, fp8)
                global_row = m0 + row
                _load_fp8x16(
                    a_buf,
                    global_row * fx.Int32(k) + fx.Int32(kb * SCALE_K) + ko,
                    fragment,
                )
                _store_fp8x16(lds_a, row * fx.Int32(LDA) + ko, fragment)

            for q in range_constexpr(config.b_prefetch):
                v = tid + fx.Int32(q * threads)
                row = v // fx.Int32(8)
                ko = (v % fx.Int32(8)) * fx.Int32(16)
                swizzled_ko = ((ko // fx.Int32(16)) ^ (row % fx.Int32(8))) * fx.Int32(
                    16
                )
                _store_fp8x16(lds_b, row * fx.Int32(LDB) + swizzled_ko, staged_b[q])
            for q in range_constexpr(config.b_prefetch, b_loads):
                v = tid + fx.Int32(q * threads)
                row = v // fx.Int32(8)
                ko = (v % fx.Int32(8)) * fx.Int32(16)
                fragment = fx.make_rmem_tensor(16, fp8)
                _load_fp8x16(
                    b_buf,
                    (n0 + row) * fx.Int32(stride_b) + fx.Int32(kb * SCALE_K) + ko,
                    fragment,
                )
                swizzled_ko = ((ko // fx.Int32(16)) ^ (row % fx.Int32(8))) * fx.Int32(
                    16
                )
                _store_fp8x16(lds_b, row * fx.Int32(LDB) + swizzled_ko, fragment)

            lds_fence_signal()

            # Do not issue an unused next-tile vector load after the final
            # K128 block. In vLLM the graph-pool activation can end exactly at
            # a page boundary; relying on raw-buffer OOB suppression for that
            # speculative load caused a gfx1201 page fault at M1568.
            if k_iter + 1 < scale_blocks:
                next_k0 = (kb + fx.Int32(1)) * fx.Int32(SCALE_K)
                if const_expr(config.k_rotate != 0):
                    next_k0 = ((kb + fx.Int32(1)) % fx.Int32(scale_blocks)) * fx.Int32(
                        SCALE_K
                    )
                for q in range_constexpr(config.a_prefetch):
                    v = tid + fx.Int32(q * threads)
                    row = v // fx.Int32(8)
                    ko = (v % fx.Int32(8)) * fx.Int32(16)
                    global_row = m0 + row
                    _load_fp8x16(
                        a_buf,
                        global_row * fx.Int32(k) + next_k0 + ko,
                        staged_a[q],
                    )
                for q in range_constexpr(config.b_prefetch):
                    v = tid + fx.Int32(q * threads)
                    row = v // fx.Int32(8)
                    ko = (v % fx.Int32(8)) * fx.Int32(16)
                    _load_fp8x16(
                        b_buf,
                        (n0 + row) * fx.Int32(stride_b) + next_k0 + ko,
                        staged_b[q],
                    )

            lds_fence_wait()
            b_group = (n0 + wave_n * fx.Int32(64)) // fx.Int32(SCALE_K)
            b_scale = f32(
                fx.ptr_load(bs_ptr + b_group * fx.Int32(scale_blocks) + fx.Int32(kb))
            )
            row_scales = []
            for mi in range_constexpr(2):
                scales = []
                for ri in range_constexpr(8):
                    row = (
                        m0
                        + wave_m * fx.Int32(32)
                        + fx.Int32(mi * WMMA_M)
                        + lane_half * fx.Int32(8)
                        + fx.Int32(ri)
                    )
                    scales.append(
                        _load_f32(
                            as_buf,
                            row * fx.Int32(scale_blocks) + fx.Int32(kb),
                        )
                        * b_scale
                    )
                row_scales.append(scales)

            partials = [fx.make_rmem_tensor(8, f32) for _ in range_constexpr(8)]
            for ni in range_constexpr(8):
                partials[ni].fill(0)

            # Keep the lane-varying LDS row bases explicit and express each
            # unrolled matrix slice as a static offset from them. This gives
            # LLVM the base+immediate form it needs to combine paired 64-bit
            # reads into gfx12 ds_read2/ds_read2st64 instructions.
            a_lane_base = (wave_m * fx.Int32(32) + lane16) * fx.Int32(
                LDA
            ) + lane_half * fx.Int32(8)
            b_row_base = wave_n * fx.Int32(64) + lane16
            b_lane_base = b_row_base * fx.Int32(LDB)
            for ki in range_constexpr(SCALE_K // WMMA_K):
                a_frags = [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(2)]
                for mi in range_constexpr(2):
                    _load_fp8_fragment_ptr(
                        lds_a_ptr,
                        a_lane_base + fx.Int32(mi * WMMA_M * LDA + ki * WMMA_K),
                        a_frags[mi],
                    )

                b_frags = [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(4)]
                swizzled_k = (fx.Int32(ki) ^ (b_row_base % fx.Int32(8))) * fx.Int32(
                    16
                ) + lane_half * fx.Int32(8)
                for ni in range_constexpr(4):
                    _load_fp8_fragment_ptr(
                        lds_b_ptr,
                        b_lane_base + fx.Int32(ni * WMMA_N * LDB) + swizzled_k,
                        b_frags[ni],
                    )

                for mi in range_constexpr(2):
                    for ni in range_constexpr(4):
                        idx = mi * 4 + ni
                        fx.gemm(
                            mma, partials[idx], a_frags[mi], b_frags[ni], partials[idx]
                        )

            for mi in range_constexpr(2):
                for ni in range_constexpr(4):
                    idx = mi * 4 + ni
                    total_v = Vec(totals[idx].load())
                    partial_v = Vec(partials[idx].load())
                    totals[idx].store(
                        Vec.from_elements(
                            [
                                total_v[ri] + partial_v[ri] * row_scales[mi][ri]
                                for ri in range_constexpr(8)
                            ],
                            f32,
                        )
                    )

        # Reuse the dead input stage as a row-major FP32 exchange. This turns
        # the WMMA fragment layout into aligned 16-byte BF16 global stores.
        gpu.barrier()
        lds_out_ptr = fx.recast_iter(f32, lds_a_ptr)
        output_rows = 32 if bm <= 64 else 64
        output_wave_rows = output_rows // 32
        for group in range_constexpr(bm // output_rows):
            if wave_m // fx.Int32(output_wave_rows) == fx.Int32(group):
                local_wave_m = wave_m % fx.Int32(output_wave_rows)
                for mi in range_constexpr(2):
                    for ni in range_constexpr(4):
                        idx = mi * 4 + ni
                        total_v = Vec(totals[idx].load())
                        for ri in range_constexpr(8):
                            local_row = (
                                local_wave_m * fx.Int32(32)
                                + fx.Int32(mi * WMMA_M)
                                + lane_half * fx.Int32(8)
                                + fx.Int32(ri)
                            )
                            local_col = (
                                wave_n * fx.Int32(64) + fx.Int32(ni * WMMA_N) + lane16
                            )
                            fx.ptr_store(
                                total_v[ri],
                                lds_out_ptr + local_row * fx.Int32(bn) + local_col,
                            )
            gpu.barrier()

            for q in range_constexpr((output_rows * bn // 8) // threads):
                v = tid + fx.Int32(q * threads)
                local_row = v // fx.Int32(bn // 8)
                col = (v % fx.Int32(bn // 8)) * fx.Int32(8)
                row = m0 + fx.Int32(group * output_rows) + local_row
                if row < m_extent:
                    values = fx.ptr_load(
                        lds_out_ptr + local_row * fx.Int32(bn) + col,
                        result_type=fx.Vector.make_type(8, f32),
                    )
                    values_v = Vec(values)
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
        launch_m = fx.Int32(static_m) if static_m else arg_m
        total_blocks = ((launch_m + fx.Int32(bm - 1)) // fx.Int32(bm)) * fx.Int32(
            grid_n
        )
        prefill_kernel(
            arg_a,
            arg_b,
            arg_as,
            arg_bs,
            arg_out,
            arg_m,
            value_attrs=kernel_attrs,
        ).launch(grid=(total_blocks, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


@lru_cache(maxsize=128)
def _get_prefill_module_cached(
    n: int,
    k: int,
    stride_b: int,
    tile_m: int,
    tile_n: int,
    group_m: int,
    a_prefetch: int,
    b_prefetch: int,
    static_m: int,
    k_rotate: int = 0,
):
    config = PrefillConfig(
        tile_m,
        tile_n,
        group_m,
        a_prefetch,
        b_prefetch,
        static_m,
        k_rotate,
    )
    return _create_prefill_module(n, k, stride_b, config)


def _get_prefill_module(
    n: int,
    k: int,
    stride_b: int,
    config: PrefillConfig,
):
    # Only lowering-affecting geometry and specialization belong in the cache
    # key. The M=8192 startup profile can therefore populate the executable
    # reused by ragged prompt sizes, while the static-M256 specialization stays
    # cache-isolated.
    return _get_prefill_module_cached(
        n,
        k,
        stride_b,
        config.tile_m,
        config.tile_n,
        config.group_m,
        config.a_prefetch,
        config.b_prefetch,
        config.static_m,
        config.k_rotate,
    )


def _run_prefill(a, weight, a_scale, weight_scale, out, stream, m, n, k):
    """Select and launch the broad-M implementation."""
    config = select_prefill_config(m, n, k, weight.stride(0))
    module = _get_prefill_module(n, k, weight.stride(0), config)
    run_compiled(module, a, weight, a_scale, weight_scale, out, m, stream)
    return out
