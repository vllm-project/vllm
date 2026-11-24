# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 the FlashTopK authors.
# Licensed under the Apache License, Version 2.0.
"""What the kernels of every architecture share: the indexer's shape, the tiling the
host side plans with, and PTX helpers. Every float operation names its rounding, so no
compiler choice (contraction, fast math) can change a bit."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import arith, llvm, math
from cutlass.cutlass_dsl import dsl_user_op

NUM_HEADS = 32
HEAD_DIM = 128
NEXT_N = 8  # query rows of a scan tile
BLOCK_KV = (
    128  # keys of a KV block: KV splits and the pruning mask are counted in these
)
PRUNE_BLOCK_Q = 16  # query rows of a pruning tile
PRUNE_COLS = 16  # columns of the pruning operand
REFINE_BLOCK_Q = 8  # query rows of a refine tile
REFINE_STAGES = 4  # compact tiles in flight in the refine kernels
REFINE_MAX_BLOCKS = 16384  # mask blocks a refine CTA's range may span
GATHER_THREADS = 128  # four warps compacting surviving keys in the refine kernels


@dsl_user_op
def store_global_b32(address, value, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(address).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_add_global_i32(address, value, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [
                cutlass.Int64(address).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def select_i32(predicate, true_value, false_value, *, loc=None, ip=None):
    return Int32(
        arith.select(
            cutlass.Boolean(predicate).ir_value(loc=loc, ip=ip),
            Int32(true_value).ir_value(loc=loc, ip=ip),
            Int32(false_value).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fp32_to_i32_bits(value, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.bitcast(
            Int32.mlir_type, Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )


@dsl_user_op
def abs_f32(value, *, loc=None, ip=None) -> Float32:
    return Float32(math.absf(Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def _f32_asm(op, *values, loc=None, ip=None) -> Float32:
    operands = ",".join(["=f"] + ["f"] * len(values))
    args = ", ".join(f"${i + 1}" for i in range(len(values)))
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [Float32(v).ir_value(loc=loc, ip=ip) for v in values],
            f"{op} $0, {args};",
            operands,
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul_rn(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_asm("mul.rn.f32", a, b, loc=loc, ip=ip)


@dsl_user_op
def sub_rn(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_asm("sub.rn.f32", a, b, loc=loc, ip=ip)


@dsl_user_op
def add_rn(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_asm("add.rn.f32", a, b, loc=loc, ip=ip)


@dsl_user_op
def div_rn(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_asm("div.rn.f32", a, b, loc=loc, ip=ip)


@dsl_user_op
def fma_rn(a, b, c, *, loc=None, ip=None) -> Float32:
    return _f32_asm("fma.rn.f32", a, b, c, loc=loc, ip=ip)


@dsl_user_op
def fmax(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_asm("max.f32", a, b, loc=loc, ip=ip)


@dsl_user_op
def fmin(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_asm("min.f32", a, b, loc=loc, ip=ip)


@dsl_user_op
def i32_to_f32_rn(value, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [Int32(value).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f32.s32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def select_f32(predicate, true_value, false_value, *, loc=None, ip=None):
    return Float32(
        arith.select(
            cutlass.Boolean(predicate).ir_value(loc=loc, ip=ip),
            Float32(true_value).ir_value(loc=loc, ip=ip),
            Float32(false_value).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def loose_raw_gate(inv, origin, th_bucket) -> Float32:
    """A raw score below which no key passes the bucket gate fma(-inv, raw, c0) < th + 1
    (c0 = -origin * inv), a little below the exact edge so rounding never excludes one;
    -inf where the bucket map is degenerate, at most FLT_MAX."""
    c0 = mul_rn(-origin, inv)
    o = i32_to_f32_rn(th_bucket + 1)
    valid = (
        (inv > Float32(0.0))
        & (inv <= Float32(3.4e38))
        & (abs_f32(c0) <= Float32(3.4e38))
    )
    slack = mul_rn(Float32(2.0**-16), fmax(fmax(abs_f32(c0), abs_f32(o)), Float32(1.0)))
    gate = fmin(
        div_rn(sub_rn(sub_rn(c0, o), slack), inv), Float32(3.4028234663852886e38)
    )
    return select_f32(valid, gate, Float32(float("-inf")))


@dsl_user_op
def find_nth_set(word, n, *, loc=None, ip=None) -> Int32:
    """The position of the n-th (1-based) set bit of word, from bit 0."""
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [Int32(word).ir_value(loc=loc, ip=ip), Int32(n).ir_value(loc=loc, ip=ip)],
            "fns.b32 $0, $1, 0, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cp_async_16(smem_addr, gmem_addr, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_4(smem_addr, gmem_addr, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.ca.shared.global [$0], [$1], 4;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_global_v4_b32(gmem_addr, *, loc=None, ip=None):
    """A 128-key block's four mask words with one 16-byte load."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([Int32.mlir_type] * 4),
        [cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Int32(llvm.extractvalue(Int32.mlir_type, result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


def swizzled_k_offset(slot, chunk: int):
    """Byte offset of 16-byte chunk `chunk` (< 8) of row `slot` in a K-major tile of
    128-byte FP8 rows with 128-byte swizzle."""
    return slot * 128 + (((slot % 8) ^ (chunk % 8)) * 16)


@cute.jit
def _locate_key(g, b, kv_start, word_quad, sPrefix) -> Int32:
    r = g - sPrefix[b]
    part = Int32(0)
    word = word_quad[0]
    for i in cutlass.range_constexpr(3):
        cnt = Int32(cute.arch.popc(word))
        step = (part == i) & (r >= cnt)
        r = select_i32(step, r - cnt, r)
        part = select_i32(step, part + 1, part)
        word = select_i32(step, word_quad[i + 1], word)
    return kv_start + b * BLOCK_KV + part * 32 + find_nth_set(word, r + 1)


@cute.jit
def _locate_tile_64(t, lane, nb, kv_start, total, mask_base, sPrefix):
    g0 = min(t * 64 + lane, total - 1)
    g1 = min(t * 64 + 32 + lane, total - 1)
    b0, b1 = Int32(0), Int32(0)
    length = nb
    while length > 1:
        half = length // 2
        b0 = select_i32(sPrefix[b0 + half] <= g0, b0 + half, b0)
        b1 = select_i32(sPrefix[b1 + half] <= g1, b1 + half, b1)
        length = length - half
    w0 = load_global_v4_b32(mask_base + cutlass.Int64(b0) * 16)
    w1 = load_global_v4_b32(mask_base + cutlass.Int64(b1) * 16)
    return (
        _locate_key(g0, b0, kv_start, w0, sPrefix),
        _locate_key(g1, b1, kv_start, w1, sPrefix),
    )


@cute.jit
def _locate_tile_128(t, lane, nb, kv_start, total, mask_base, sPrefix):
    g0 = min(t * 128 + lane, total - 1)
    g1 = min(t * 128 + 32 + lane, total - 1)
    g2 = min(t * 128 + 64 + lane, total - 1)
    g3 = min(t * 128 + 96 + lane, total - 1)
    b0, b1, b2, b3 = Int32(0), Int32(0), Int32(0), Int32(0)
    length = nb
    while length > 1:
        half = length // 2
        b0 = select_i32(sPrefix[b0 + half] <= g0, b0 + half, b0)
        b1 = select_i32(sPrefix[b1 + half] <= g1, b1 + half, b1)
        b2 = select_i32(sPrefix[b2 + half] <= g2, b2 + half, b2)
        b3 = select_i32(sPrefix[b3 + half] <= g3, b3 + half, b3)
        length = length - half
    blks = (b0, b1, b2, b3)
    words = [load_global_v4_b32(mask_base + cutlass.Int64(b) * 16) for b in blks]
    return tuple(
        _locate_key(g, blks[j], kv_start, words[j], sPrefix)
        for j, g in enumerate((g0, g1, g2, g3))
    )


@cute.jit
def refine_gather(
    gt,
    gw,
    lane,
    nb,
    kv_start,
    q_tile,
    mask_blocks,
    mMask,
    mKV,
    mScales,
    sPrefix,
    sTile,
    sMetaIdx,
    sMetaScale,
    sCounts,
    count_bar,
    full_tile,
    empty_tile,
    tile_keys: cutlass.Constexpr,
    gather_barrier: cutlass.Constexpr,
):
    """Count survivors per 128-key mask block into an exclusive prefix sum in sPrefix,
    publish the total on count_bar, and compact surviving K rows and scales into
    tile_keys-row ring stages by cp.async."""
    mask_base = (
        mMask.iterator.toint()
        + (cutlass.Int64(q_tile) * mask_blocks + kv_start // BLOCK_KV) * 16
    )
    run = (nb + (GATHER_THREADS - 1)) // GATHER_THREADS
    b_lo = min(gt * run, nb)
    b_hi = min(b_lo + run, nb)
    local = Int32(0)
    for blk in cutlass.range(b_lo, b_hi, 4, unroll=1):
        words = [
            load_global_v4_b32(mask_base + cutlass.Int64(min(blk + u, b_hi - 1)) * 16)
            for u in range(4)
        ]
        for u in cutlass.range_constexpr(4):
            c = Int32(0)
            for word in words[u]:
                c = c + Int32(cute.arch.popc(word))
            in_run = blk + u < b_hi
            if in_run:
                sPrefix[blk + u] = c
            local = local + select_i32(in_run, c, 0)
    incl = local
    for off in cutlass.range_constexpr(5):
        up = cute.arch.shuffle_sync_up(incl, 1 << off, mask_and_clamp=0)
        if lane >= (1 << off):
            incl = incl + up
    if lane == 31:
        sCounts[1 + gw] = incl
    cute.arch.barrier(barrier_id=gather_barrier, number_of_threads=GATHER_THREADS)
    base = Int32(0)
    total = Int32(0)
    for i in cutlass.range_constexpr(4):
        v = sCounts[1 + i]
        base = base + select_i32(i < gw, v, 0)
        total = total + v
    running = base + incl - local
    for blk in cutlass.range(b_lo, b_hi, unroll=1):
        c = sPrefix[blk]
        sPrefix[blk] = running
        running = running + c
    if gt == 0:
        sPrefix[nb] = total
    cute.arch.barrier(barrier_id=gather_barrier, number_of_threads=GATHER_THREADS)
    if gt == 0:
        sCounts[0] = total
        cute.arch.mbarrier_arrive(count_bar)

    tiles = (total + tile_keys - 1) // tile_keys
    tile_bytes = tile_keys * HEAD_DIM
    smem_base = sTile.iterator.toint()
    scale_smem = sMetaScale.iterator.toint()
    kv_base = mKV.iterator.toint()
    scale_base = mScales.iterator.toint()
    if cutlass.const_expr(tile_keys == 64):
        i0, i1 = Int32(0), Int32(0)
        if gw < tiles:
            i0, i1 = _locate_tile_64(gw, lane, nb, kv_start, total, mask_base, sPrefix)
        for t in cutlass.range(gw, tiles, REFINE_STAGES, unroll=1):
            s = t % REFINE_STAGES
            cute.arch.mbarrier_wait(empty_tile + s, (t // REFINE_STAGES) % 2 ^ 1)
            keys2 = (i0, i1)
            for j in cutlass.range_constexpr(2):
                slot = j * 32 + lane
                if t * tile_keys + slot < total:
                    src = kv_base + cutlass.Int64(keys2[j]) * HEAD_DIM
                    dst = smem_base + s * tile_bytes
                    for chunk in cutlass.range_constexpr(HEAD_DIM // 16):
                        cp_async_16(
                            dst + swizzled_k_offset(slot, chunk), src + chunk * 16
                        )
                    sMetaIdx[(s, slot)] = keys2[j]
                    cp_async_4(
                        scale_smem + (s * tile_keys + slot) * 4,
                        scale_base + cutlass.Int64(keys2[j]) * 4,
                    )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(full_tile + s)
            if t + REFINE_STAGES < tiles:
                i0, i1 = _locate_tile_64(
                    t + REFINE_STAGES, lane, nb, kv_start, total, mask_base, sPrefix
                )
    else:
        k0, k1, k2, k3 = Int32(0), Int32(0), Int32(0), Int32(0)
        if gw < tiles:
            k0, k1, k2, k3 = _locate_tile_128(
                gw, lane, nb, kv_start, total, mask_base, sPrefix
            )
        for t in cutlass.range(gw, tiles, REFINE_STAGES, unroll=1):
            s = t % REFINE_STAGES
            cute.arch.mbarrier_wait(empty_tile + s, (t // REFINE_STAGES) % 2 ^ 1)
            keys4 = (k0, k1, k2, k3)
            for j in cutlass.range_constexpr(4):
                slot = j * 32 + lane
                if t * tile_keys + slot < total:
                    src = kv_base + cutlass.Int64(keys4[j]) * HEAD_DIM
                    dst = smem_base + s * tile_bytes
                    for chunk in cutlass.range_constexpr(HEAD_DIM // 16):
                        cp_async_16(
                            dst + swizzled_k_offset(slot, chunk), src + chunk * 16
                        )
                    sMetaIdx[(s, slot)] = keys4[j]
                    cp_async_4(
                        scale_smem + (s * tile_keys + slot) * 4,
                        scale_base + cutlass.Int64(keys4[j]) * 4,
                    )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(full_tile + s)
            if t + REFINE_STAGES < tiles:
                k0, k1, k2, k3 = _locate_tile_128(
                    t + REFINE_STAGES, lane, nb, kv_start, total, mask_base, sPrefix
                )
