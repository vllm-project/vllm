# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B008 -- FlyDSL launch signatures require typed stream defaults
# ruff: noqa: SIM114 -- Keep compile-time and device predicates separate

"""RDNA4 block-scaled FP8 GEMM for decode and narrow short-prefill shapes.

This is a raw-weight kernel: ``weight`` is the ordinary row-major ``[N, K]``
tensor consumed by vLLM.  No preshuffle or persistent workspace is part of the
interface.  The numerical contract is intentionally identical to
the public RDNA4 block-scaled operator in vLLM::

    out[m, n] = sum_kb(
        dot_fp32(a[m, kb * 128 : (kb + 1) * 128], weight[n, kb * 128 : (kb + 1) * 128])
        * a_scale[m, kb]
        * weight_scale[n // 128, kb]
    )

Each K=128 dot product is completed in FP32 before its two scales are applied.
The scaled blocks are accumulated in FP32 and converted to BF16 once.
"""

from dataclasses import dataclass
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from .rdna4_fp8_blockscale_common import (
    SCALE_K,
    WAVE_SIZE,
    WMMA_K,
    WMMA_M,
    WMMA_N,
    _f32_to_bf16_rne,
    _load_f32,
    _load_fp8_fragment_buffer,
    _make_buffer,
    _store_bf16,
)
from .runtime import run_compiled


@dataclass(frozen=True)
class KernelConfig:
    block_n: int
    row_tiles: int
    threads: int
    split_k: int = 1
    rotate_k: int = 0
    load_batch: int = 0
    m_fast: bool = False
    hardware_bounds: bool = False
    capped_split: bool = False


def _use_tiled_decode_split(m: int, n: int, k: int) -> bool:
    """Keep larger split grids within their measured low-K reuse regime."""
    tiles = ((m + 15) // 16) * (n // 16)
    return (
        16 < m <= 256
        and n <= 1024
        and k >= 1024
        and (tiles <= 256 or (m > 64 and tiles <= 512 and k <= 4096))
    )


def select_kernel_config(m: int, n: int, k: int) -> KernelConfig:
    """Select packed rows or a single-launch split for underfilled grids."""
    if not 1 <= m <= 256:
        raise ValueError(f"RDNA4 small-M route requires 1 <= M <= 256, got {m}")
    if n <= 0 or n % 128:
        raise ValueError(
            f"RDNA4 small-M route requires positive N divisible by 128, got {n}"
        )
    if k <= 0 or k % SCALE_K:
        raise ValueError(
            f"RDNA4 small-M route requires positive K divisible by 128, got {k}"
        )

    if m <= 16:
        if n <= 3072 and k >= 1024:
            splits = 8 if n <= 1024 else 4
            return KernelConfig(
                16, 1, splits * WAVE_SIZE, split_k=splits, rotate_k=2, load_batch=8
            )
        return KernelConfig(64, 1, 64, rotate_k=2, load_batch=8)
    if _use_tiled_decode_split(m, n, k):
        return KernelConfig(
            16,
            1,
            4 * WAVE_SIZE,
            split_k=4,
            rotate_k=2,
            load_batch=8,
            capped_split=((m + 15) // 16) * (n // 16) > 256,
        )
    if m > 64:
        raise ValueError(
            "RDNA4 M > 64 split route requires N <= 1024, K >= 1024, "
            "and at most 256 output tiles (512 when K <= 4096)"
        )
    return KernelConfig(64, 2, 64, rotate_k=2, load_batch=8, m_fast=True)


def _create_packed_module(m: int, n: int, k: int, stride_b: int, config: KernelConfig):
    """Create the shared packed-row implementation for all non-split-K routes."""
    assert config.split_k == 1
    assert config.block_n == config.threads
    assert config.block_n in (64, 128)
    assert config.row_tiles in (1, 2, 4)
    assert config.load_batch in (0, 1, 2, 4, 8)
    config_signature = (
        config.block_n,
        config.row_tiles,
        config.threads,
        config.rotate_k,
        config.load_batch,
        config.m_fast,
        config.hardware_bounds,
    )

    fp8 = fx.Float8E4M3FN
    f32 = fx.Float32
    bf16 = fx.BFloat16
    scale_blocks = k // SCALE_K
    grid_m = (
        1
        if m <= 16
        else (m + config.row_tiles * WMMA_M - 1) // (config.row_tiles * WMMA_M)
    )

    @flyc.kernel
    def packed_kernel(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_as: fx.Tensor,
        arg_bs: fx.Tensor,
        arg_out: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        lane = tid % fx.Int32(WAVE_SIZE)
        wave = tid // fx.Int32(WAVE_SIZE)
        lane_col = lane % fx.Int32(WMMA_N)
        lane_row_base = (lane // fx.Int32(WMMA_N)) * fx.Int32(8)
        pid_n, pid_m = fx.block_idx.x, fx.block_idx.y
        if const_expr(config.m_fast):
            pid_n = fx.block_idx.x // fx.Int32(grid_m)
            pid_m = fx.block_idx.x % fx.Int32(grid_m)
        block_row = pid_m * fx.Int32(config.row_tiles * WMMA_M)
        n0 = pid_n * fx.Int32(config.block_n) + wave * fx.Int32(2 * WMMA_N)

        a_buf = _make_buffer(arg_a, fp8, 8, m * k)
        b_buf = _make_buffer(arg_b, fp8, 8, n * stride_b)
        as_buf = _make_buffer(arg_as, f32, 1, m * scale_blocks * 4)
        bs_ptr = fx.recast_iter(f32, fx.get_iter(arg_bs))
        out_buf = _make_buffer(arg_out, bf16, 1, m * n * 2)

        mma = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fp8, f32))
        totals = [
            [fx.make_rmem_tensor(8, f32) for _ in range_constexpr(2)]
            for _ in range_constexpr(config.row_tiles)
        ]
        for rt in range_constexpr(config.row_tiles):
            for j in range_constexpr(2):
                totals[rt][j].fill(0)

        for iteration in range(0, scale_blocks, 1):
            kb = iteration
            if const_expr(config.rotate_k):
                kb = (iteration + pid_n * fx.Int32(config.rotate_k)) % fx.Int32(
                    scale_blocks
                )
            partials = [
                [fx.make_rmem_tensor(8, f32) for _ in range_constexpr(2)]
                for _ in range_constexpr(config.row_tiles)
            ]
            for rt in range_constexpr(config.row_tiles):
                for j in range_constexpr(2):
                    partials[rt][j].fill(0)

            if const_expr(config.load_batch or (m == 1 and n >= 16384)):
                batch = const_expr(config.load_batch or 4)
                for ks_group in range_constexpr(8 // batch):
                    a_frags = [
                        [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(batch)]
                        for _ in range_constexpr(config.row_tiles)
                    ]
                    b_frags = [
                        [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(2)]
                        for _ in range_constexpr(batch)
                    ]
                    if const_expr(config.row_tiles == 1):
                        for ks_local in range_constexpr(batch):
                            ks = ks_group * batch + ks_local
                            k_lane = kb * fx.Int32(SCALE_K) + (
                                lane // fx.Int32(16)
                            ) * fx.Int32(8)
                            for rt in range_constexpr(config.row_tiles):
                                a_frags[rt][ks_local].fill(0)
                                global_row = (
                                    block_row
                                    + fx.Int32(rt * WMMA_M)
                                    + (lane % fx.Int32(WMMA_M))
                                )
                                if const_expr(config.hardware_bounds):
                                    _load_fp8_fragment_buffer(
                                        a_buf,
                                        global_row * fx.Int32(k)
                                        + k_lane
                                        + fx.Int32(ks * WMMA_K),
                                        a_frags[rt][ks_local],
                                    )
                                elif global_row < fx.Int32(m):
                                    _load_fp8_fragment_buffer(
                                        a_buf,
                                        global_row * fx.Int32(k)
                                        + k_lane
                                        + fx.Int32(ks * WMMA_K),
                                        a_frags[rt][ks_local],
                                    )
                        for ks_local in range_constexpr(batch):
                            ks = ks_group * batch + ks_local
                            k_lane = kb * fx.Int32(SCALE_K) + (
                                lane // fx.Int32(16)
                            ) * fx.Int32(8)
                            for j in range_constexpr(2):
                                b_row = n0 + fx.Int32(j * WMMA_N) + lane_col
                                _load_fp8_fragment_buffer(
                                    b_buf,
                                    b_row * fx.Int32(stride_b)
                                    + k_lane
                                    + fx.Int32(ks * WMMA_K),
                                    b_frags[ks_local][j],
                                )
                    else:
                        for ks_local in range_constexpr(batch):
                            ks = ks_group * batch + ks_local
                            k_lane = (
                                kb * fx.Int32(SCALE_K)
                                + fx.Int32(ks * WMMA_K)
                                + (lane // fx.Int32(16)) * fx.Int32(8)
                            )
                            for j in range_constexpr(2):
                                b_row = n0 + fx.Int32(j * WMMA_N) + lane_col
                                _load_fp8_fragment_buffer(
                                    b_buf,
                                    b_row * fx.Int32(stride_b) + k_lane,
                                    b_frags[ks_local][j],
                                )
                            for rt in range_constexpr(config.row_tiles):
                                a_frags[rt][ks_local].fill(0)
                                global_row = (
                                    block_row
                                    + fx.Int32(rt * WMMA_M)
                                    + (lane % fx.Int32(WMMA_M))
                                )
                                if const_expr(config.hardware_bounds):
                                    _load_fp8_fragment_buffer(
                                        a_buf,
                                        global_row * fx.Int32(k) + k_lane,
                                        a_frags[rt][ks_local],
                                    )
                                elif global_row < fx.Int32(m):
                                    _load_fp8_fragment_buffer(
                                        a_buf,
                                        global_row * fx.Int32(k) + k_lane,
                                        a_frags[rt][ks_local],
                                    )
                    for ks_local in range_constexpr(batch):
                        for rt in range_constexpr(config.row_tiles):
                            for j in range_constexpr(2):
                                fx.gemm(
                                    mma,
                                    partials[rt][j],
                                    a_frags[rt][ks_local],
                                    b_frags[ks_local][j],
                                    partials[rt][j],
                                )
                    if const_expr(config.row_tiles > 1):
                        fx.rocdl.sched_vmem((2 + config.row_tiles) * batch)
                        fx.rocdl.sched_mfma(2 * config.row_tiles * batch)
                        fx.rocdl.sched_barrier(0)
            else:
                for ks in range_constexpr(SCALE_K // WMMA_K):
                    b_frags = [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(2)]
                    k_lane = (
                        kb * fx.Int32(SCALE_K)
                        + fx.Int32(ks * WMMA_K)
                        + (lane // fx.Int32(16)) * fx.Int32(8)
                    )
                    for j in range_constexpr(2):
                        b_row = n0 + fx.Int32(j * WMMA_N) + lane_col
                        _load_fp8_fragment_buffer(
                            b_buf,
                            b_row * fx.Int32(stride_b) + k_lane,
                            b_frags[j],
                        )

                    for rt in range_constexpr(config.row_tiles):
                        a_frag = fx.make_rmem_tensor(8, fp8)
                        a_frag.fill(0)
                        local_row = fx.Int32(rt * WMMA_M) + (lane % fx.Int32(WMMA_M))
                        global_row = block_row + local_row
                        if const_expr(config.hardware_bounds):
                            _load_fp8_fragment_buffer(
                                a_buf,
                                global_row * fx.Int32(k) + k_lane,
                                a_frag,
                            )
                        elif global_row < fx.Int32(m):
                            _load_fp8_fragment_buffer(
                                a_buf,
                                global_row * fx.Int32(k) + k_lane,
                                a_frag,
                            )
                        for j in range_constexpr(2):
                            fx.gemm(
                                mma,
                                partials[rt][j],
                                a_frag,
                                b_frags[j],
                                partials[rt][j],
                            )

            b_group = n0 // fx.Int32(SCALE_K)
            b_scale = f32(fx.ptr_load(bs_ptr + b_group * fx.Int32(scale_blocks) + kb))
            for rt in range_constexpr(config.row_tiles):
                if const_expr(m == 1):
                    a_scale = _load_f32(as_buf, kb)
                    scales = [
                        (
                            (lane < fx.Int32(16)).select(a_scale * b_scale, f32(0.0))
                            if value_idx == 0
                            else f32(0.0)
                        )
                        for value_idx in range_constexpr(8)
                    ]
                else:
                    scales = []
                    for value_idx in range_constexpr(8):
                        global_row = (
                            block_row
                            + fx.Int32(rt * WMMA_M)
                            + lane_row_base
                            + fx.Int32(value_idx)
                        )
                        if const_expr(value_idx >= m):
                            scales.append(f32(0.0))
                        elif const_expr(config.hardware_bounds):
                            a_scale = _load_f32(
                                as_buf, global_row * fx.Int32(scale_blocks) + kb
                            )
                            scales.append(a_scale * b_scale)
                        else:
                            valid = global_row < fx.Int32(m)
                            safe_row = valid.select(global_row, fx.Int32(0))
                            a_scale = _load_f32(
                                as_buf, safe_row * fx.Int32(scale_blocks) + kb
                            )
                            scales.append(valid.select(a_scale * b_scale, f32(0.0)))
                for j in range_constexpr(2):
                    total_v = Vec(totals[rt][j].load())
                    partial_v = Vec(partials[rt][j].load())
                    updated = [
                        total_v[x] + partial_v[x] * scales[x]
                        for x in range_constexpr(8)
                    ]
                    totals[rt][j].store(Vec.from_elements(updated, f32))

        if const_expr(m == 1):
            if lane < fx.Int32(16):
                for j in range_constexpr(2):
                    col = n0 + fx.Int32(j * WMMA_N) + lane_col
                    value = _f32_to_bf16_rne(Vec(totals[0][j].load())[0])
                    _store_bf16(out_buf, col, value)
        else:
            for rt in range_constexpr(config.row_tiles):
                for value_idx in range_constexpr(min(8, m)):
                    global_row = (
                        block_row
                        + fx.Int32(rt * WMMA_M)
                        + lane_row_base
                        + fx.Int32(value_idx)
                    )
                    for j in range_constexpr(2):
                        col = n0 + fx.Int32(j * WMMA_N) + lane_col
                        value = _f32_to_bf16_rne(Vec(totals[rt][j].load())[value_idx])
                        if const_expr(config.hardware_bounds):
                            _store_bf16(out_buf, global_row * fx.Int32(n) + col, value)
                        elif global_row < fx.Int32(m):
                            _store_bf16(out_buf, global_row * fx.Int32(n) + col, value)

    @flyc.jit
    def launch(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_as: fx.Tensor,
        arg_bs: fx.Tensor,
        arg_out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        assert const_expr(
            config_signature
            == (
                config.block_n,
                config.row_tiles,
                config.threads,
                config.rotate_k,
                config.load_batch,
                config.m_fast,
                config.hardware_bounds,
            )
        )
        packed_kernel(arg_a, arg_b, arg_as, arg_bs, arg_out).launch(
            grid=((n // config.block_n) * grid_m, 1, 1)
            if config.m_fast
            else (n // config.block_n, grid_m, 1),
            block=(config.threads, 1, 1),
            stream=stream,
        )

    return launch


@lru_cache(maxsize=128)
def _get_module(m: int, n: int, k: int, stride_b: int, config: KernelConfig):
    if config.split_k in (4, 8):
        from .rdna4_fp8_blockscale_decode_split import create_decode_split

        return create_decode_split(
            m,
            n,
            k,
            stride_b,
            config.block_n,
            config.split_k,
            config.load_batch,
            config.rotate_k,
            capped=config.capped_split,
        )
    return _create_packed_module(m, n, k, stride_b, config)


def _run_small_m(a, weight, a_scale, weight_scale, out, stream, m, n, k):
    """Launch an already validated packed-row or narrow split input."""
    config = select_kernel_config(m, n, k)
    module = _get_module(m, n, k, weight.stride(0), config)
    run_compiled(module, a, weight, a_scale, weight_scale, out, stream)
    return out
