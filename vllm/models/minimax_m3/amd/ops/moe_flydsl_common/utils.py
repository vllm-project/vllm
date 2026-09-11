# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""MiniMax-M3 FlyDSL MoE utils helpers."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

LOG2E = 1.4426950408889634


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _global_i32_buffer_view(addr_i64, num_bytes):
    """Buffer-tensor view of ``num_bytes`` i32 dwords at ``addr_i64`` (OOB-clamped)."""
    num_bytes_i64 = fx.Int64(num_bytes)
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    view = fx.Tensor(fx.make_view(ptr, fx.make_layout(num_bytes_i64 // fx.Int64(4), 1)))
    return fx.rocdl.make_buffer_tensor(
        view, max_size=False, num_records_bytes=num_bytes_i64
    )


def _lds_ptr3(base_i32, byte_off_i32):
    addr_i64 = fx.Int64(base_i32 + byte_off_i32)
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(addr_i64))


def _gep3(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8
    )


def _global_base_ptr1(addr_i64):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def _gep1(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8
    )


def _global_i32_ptr(addr_i64):
    """Typed global i32 pointer at a raw device address (``p[i]`` loads / stores)."""
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _lds_atomic_add_i32(ptr, value):
    """``old = *ptr; *ptr += value`` on an LDS i32 pointer (workgroup scope);
    returns ``old``. fx has no atomic wrapper, so this is the one raw LLVM op."""
    return fx.Int32(
        llvm.AtomicRMWOp(
            llvm.AtomicBinOp.add,
            ptr.llvm_ptr,
            fx.Int32(value).ir_value(),
            llvm.AtomicOrdering.monotonic,
            syncscope="workgroup",
            alignment=4,
        ).result
    )


def _global_i32_at(addr_i64, idx):
    return _global_i32_ptr(addr_i64)[idx]


def _e8m0_byte_to_f32(packed_i32, byte_pos):
    shift = byte_pos * fx.Int32(8)
    b = packed_i32.shrui(shift) & fx.Int32(0xFF)
    return fx.Float32(_raw(b << fx.Int32(23)).bitcast(T.f32))


def _sigmoid_f32(g):
    e = fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E))))
    return fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + e)))


def _swigluoai_f32(g, u, alpha, neg_limit):
    """gpt-oss / MiniMax-M3 swiglu (aiter ``ActivationType.Swiglu`` with swiglu_limit,
    CK-tile ``moe::Swiglu``): y = min(g, L) * sigmoid(alpha * min(g, L)) *
    (clamp(u, -L, L) + 1). ``neg_limit`` is -L (host-negated)."""
    g_c = -((-g).maximumf(neg_limit))  # min(g, L)
    u_c = (-((-u).maximumf(neg_limit))).maximumf(neg_limit)  # clamp(u, -L, L)
    sig = _sigmoid_f32(g_c * alpha)
    return g_c * sig * (u_c + fx.Float32(1.0))


def inline_sort_max_pairs(n_tokens, topk, bm):
    """Routing pairs the inline-sort table scans (64 per wave pass): one pass when
    n_tokens*topk <= 64 (M <= 12 at topk 5), else the full bm*topk (80 -> two
    passes). Compile-time (part of the kernel name)."""
    return 64 if int(n_tokens) * int(topk) <= 64 else int(bm) * int(topk)


def inline_sort_table(arg_topk, i32_ntok, TOPK, p_i32, lane, tab, max_pairs=64, bm=16):
    """Inline sort (n_tokens <= BM), no sort kernel: routing pair q = token*TOPK + slot;
    block p owns expert e = topk_ids[p] iff p is the first pair with that expert,
    its rows are all pairs with expert e in pair order (one m-block per expert).
    ``tab`` (i32 LDS, ``bm + 1`` entries) receives token | slot<<24 per row, token
    = n_tokens for padding, i.e. what moe_sorting would write for this block;
    non-matching pairs are parked in slot ``bm``. Returns (expert id, owner, row
    count, build_table); non-owner blocks must exit. ``max_pairs`` = BM*TOPK.
    """
    n_chunks = (int(max_pairs) + 63) // 64
    n_pairs = i32_ntok * fx.Int32(TOPK)
    last = n_pairs - fx.Int32(1)
    p_lane = p_i32 % fx.Int32(64)
    p_chunk = p_i32 // fx.Int32(64)
    qs, pvs = [], []
    for c in range_constexpr(n_chunks):
        q = lane + fx.Int32(c * 64)
        idx = fx.Int32(arith.minsi(_raw(q), _raw(last)))
        v = fx.Int32(_global_i32_at(arg_topk, idx))
        qs.append(q)
        pvs.append((q < n_pairs).select(v, fx.Int32(-1)))
    # my expert = value of pair p (uniform)
    e = fx.Int32(rocdl.readlane(T.i32, _raw(pvs[0]), _raw(p_lane)))
    for c in range_constexpr(1, n_chunks):
        e_c = fx.Int32(rocdl.readlane(T.i32, _raw(pvs[c]), _raw(p_lane)))
        e = (p_chunk == fx.Int32(c)).select(e_c, e)
    base = fx.Int32(0)
    rank_p = fx.Int32(0)
    slots, fuseds = [], []
    for c in range_constexpr(n_chunks):
        is_match = pvs[c] == e
        mask = rocdl.ballot(T.i64, _raw(is_match))
        mask_lo = arith.trunci(T.i32, mask)
        mask_hi = arith.trunci(T.i32, arith.shrui(mask, arith.constant(32, type=T.i64)))
        below = fx.Int32(
            rocdl.mbcnt_hi(
                T.i32, mask_hi, rocdl.mbcnt_lo(T.i32, mask_lo, _raw(fx.Int32(0)))
            )
        )
        rank = base + below  # row of pair q among this expert's pairs (pair order)
        fuseds.append(
            (qs[c] // fx.Int32(TOPK)) | ((qs[c] % fx.Int32(TOPK)) << fx.Int32(24))
        )
        slots.append(is_match.select(rank, fx.Int32(bm)))  # non-matching -> slot bm
        r_p = fx.Int32(rocdl.readlane(T.i32, _raw(rank), _raw(p_lane)))
        rank_p = (p_chunk == fx.Int32(c)).select(r_p, rank_p)
        # matches so far = rank at lane 63 + lane 63's own match bit
        tot = rank + is_match.select(fx.Int32(1), fx.Int32(0))
        base = fx.Int32(rocdl.readlane(T.i32, _raw(tot), _raw(fx.Int32(63))))
    owner = rank_p == fx.Int32(0)

    def build_table():
        # padding sentinel in every row slot first (0..31 covers bm <= 32), then the
        # rows
        tab[lane % 32] = i32_ntok
        gpu.barrier()
        for c in range_constexpr(n_chunks):
            tab[slots[c]] = fuseds[c]
        gpu.barrier()

    # The kernel calls build_table() under `if owner:` (a uniform branch the kernel's
    # AST rewriter turns into scf.if), so duplicate-expert blocks leave after the loads
    # and ballots without the two barriers.
    return e, owner, base, build_table


def _swizzle_xor16(row, col_bytes, k_blocks16):
    """A-LDS bank-conflict XOR swizzle (aiter swizzle_xor16: col ^ ((row&(kb16-1))*16));
    the DMA write and the LDS read both go through it."""
    rem = row & fx.Int32(k_blocks16 - 1)
    return col_bytes ^ (rem * fx.Int32(16))
