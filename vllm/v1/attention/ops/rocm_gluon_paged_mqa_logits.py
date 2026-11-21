# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Portions adapted from ROCm/aiter (aiter/ops/triton/gluon/pa_mqa_logits.py):
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 paged MQA FP8 logits for the DSA indexer's MTP verify rows.

Derived from aiter's gluon `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`
(aiter e7d2453) and its host wrapper, reduced to the flattened decode path of
the ROCm DSA indexer (gfx950, preshuffled 16/32/64-token pages, 32 heads x 128, one
query per row) and re-tuned:

- the MTP verify rows of one request share a page table, so one program scores
  two sibling rows and reads each K page once (the pairing is checked on the
  device, so any input is exact);
- groups are dispatched longest first, and program ids are swizzled so that
  the KV-split pieces of one pair share an XCD (L2);
- gfx950's full-rate scaled fp8 MFMA and a uniform-SGPR row index.

Only `rocm_aiter_mla_sparse.rocm_fp8_paged_mqa_logits` calls into this
module, and only when `supported` holds; every other call keeps aiter's
kernel.
"""

from __future__ import annotations

from functools import cache

import torch

from vllm.triton_utils import gl, gluon, tl, triton


@cache
def get_gfx() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


# Number of XCDs the phase-1 grid is round-robined over (gfx950 SPX = 8).
# Used to relabel program ids so that the KV-split pieces that share page-table
# entries land on the same XCD and hit its private L2 instead of the fabric.
_NUM_XCD = tl.constexpr(8)


@triton.jit
def _sum_combine(a, b):
    return a + b


@gluon.jit
def _uniform_i32(x):
    # v_readfirstlane into an SGPR ("=s" output): the value is then provably
    # uniform, so buffer descriptors / loop bounds derived from it stay scalar
    # (no waterfall loops). The s_nops cover the VALU/DPP-write ->
    # readfirstlane-read and the VALU-SGPR-write -> use hazards, which the
    # compiler cannot see through inline asm (without the leading nop the read
    # returned a stale lane value, measured).
    return gl.inline_asm_elementwise(
        "s_nop 4\n v_readfirstlane_b32 $0, $1\n s_nop 4",
        "=s,v",
        [x],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _dot_fp8(a, b, acc, CDNA_VERSION: gl.constexpr):
    # gfx950: the full-rate v_mfma_*_f8f6f4 instructions are only reachable through the
    # scaled-MFMA op (plain tt.dot has no fp8 mapping for K=64/128 shapes); unit e8m0
    # scales (0x7F == 2^0) make it a plain fp8 dot. The legacy v_mfma_f32_16x16x32_fp8
    # runs at half of gfx950's fp8 rate (16 cycles each, measured).
    # Epilogue relu: gl.maximum(.., propagate_nan=ALL) lowers to one v_maximum3_f32 per
    # element; the default maxnum form costs a canonicalize v_max plus the v_max.
    # Bitwise identical for finite inputs; a NaN MFMA output (only from fp8 NaN codes,
    # which the ue8m0 cache writer does not produce from finite K) now propagates
    # instead of being clipped to 0.
    if CDNA_VERSION == 4:
        out = gl.amd.cdna4.mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)
    else:
        out = gl.amd.cdna3.mfma(a, b, acc)
    return out


# ===========================================================================
# Sibling-fused paged MQA logits (gfx950, preshuffle, page 16/32/64, 32 heads x
# 128, one query per row). The MTP verify rows of one request (GROUP consecutive
# rows, lengths L+1..L+GROUP) share one page table, so every K page is read by
# GROUP rows. Here one workgroup computes a PAIR of sibling rows (an odd group's
# last row pairs with itself): each K / scale register tile is loaded once and
# fed to two MFMA pairs (one per row's Q); only
# the KV partition is taken from the longer row. The pairing is verified at run
# time: the two rows' page-table entries over their whole common prefix must
# match, else the workgroup runs two single-row passes instead (each row with its
# own pages and partition). Dispatch order inside each XCD slab: longest group
# first.
# ===========================================================================
@gluon.jit
def _gluon_paged_mqa_logits_fused2(
    batch_size,
    Q_buffer,
    stride_q_batch,
    stride_q_heads,
    KV_buffer,
    stride_k_seq,
    scale_buffer,
    stride_scale_seq,
    context_len_ptr,
    kv_indices,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    max_block_len,
    SplitKV,
    ChunkQ: gl.constexpr,
    ChunkK: gl.constexpr,
    KVBlockSize: gl.constexpr,
    HiddenDim: gl.constexpr,
    GROUP: gl.constexpr,
    NG2: gl.constexpr,
    NUM_XCD: gl.constexpr,
):
    NumWarps: gl.constexpr = 4
    ThreadsPerWarp: gl.constexpr = 64
    ValQMPerThread: gl.constexpr = ChunkQ // (
        NumWarps * ThreadsPerWarp // (HiddenDim // 16)
    )
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[ValQMPerThread, 16],
        threads_per_warp=[ThreadsPerWarp // (HiddenDim // 16), HiddenDim // 16],
        warps_per_cta=[NumWarps, 1],
        order=[1, 0],
    )
    ChunkKPerStage: gl.constexpr = ChunkK // 2
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=False,
        warps_per_cta=[1, NumWarps],
        tiles_per_warp=[1, 1],
    )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    layout_scale: gl.constexpr = gl.SliceLayout(1, mfma_layout)
    layout_cols: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    layout_kcols: gl.constexpr = gl.SliceLayout(0, mfma_layout_b)
    ContextBlockPerChunkK: gl.constexpr = ChunkK // KVBlockSize
    ShuffleRows: gl.constexpr = min(KVBlockSize, 16)
    PAIRS: gl.constexpr = (GROUP + 1) // 2

    # ---- work tile: (group rank, split, pair) inside each XCD slab -------------------
    pid = gl.program_id(0)
    SX = SplitKV // NUM_XCD
    GS = PAIRS * SX
    x_id = pid % NUM_XCD
    j = pid // NUM_XCD
    q_rank = j // GS
    t_split = (j % GS) // PAIRS
    p_pair = j % PAIRS
    ngroups = batch_size // GROUP
    L_RANK: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, NumWarps], [1, 0])
    gi = gl.arange(0, NG2, layout=gl.SliceLayout(1, L_RANK))
    gj = gl.arange(0, NG2, layout=gl.SliceLayout(0, L_RANK))
    Li = gl.load(context_len_ptr + gi * GROUP, mask=gi < ngroups, other=-1)
    Lj = gl.load(context_len_ptr + gj * GROUP, mask=gj < ngroups, other=-1)
    ahead = (Lj[None, :] > Li[:, None]) | (
        (Lj[None, :] == Li[:, None]) & (gj[None, :] < gi[:, None])
    )
    rank = gl.sum(ahead.to(gl.int32), axis=1)
    g_sel = gl.sum(gl.where((rank == q_rank) & (gi < ngroups), gi, 0), axis=0)
    g_sel = _uniform_i32(g_sel)
    r0 = g_sel * GROUP + 2 * p_pair
    r1 = gl.minimum(r0 + 1, g_sel * GROUP + GROUP - 1)
    L0 = gl.load(context_len_ptr + r0)
    L1 = gl.load(context_len_ptr + r1)
    pid_split_kv = x_id * SX + t_split
    lead_is_1 = (L1 >= L0).to(gl.int32)
    Lmax = gl.maximum(L0, L1)
    Lmin = gl.minimum(L0, L1)
    lead = r0 + lead_is_1 * (r1 - r0)

    # ---- pairing check: do the two rows share every page of their common prefix?
    # Checked over the WHOLE common prefix, not just this workgroup's window, so that
    # every split of a pair takes the same branch: the fused pass partitions by the
    # longer row and the single-row passes by each row's own length, and mixing the
    # two across splits leaves columns of the shorter row unwritten.
    cnum = tl.cdiv(Lmax, ChunkK)
    scnum = cnum // SplitKV
    rcnum = cnum % SplitKV
    s_start = pid_split_kv * scnum * ChunkK + min(pid_split_kv, rcnum) * ChunkK
    s_len = min(
        Lmax - s_start, scnum * ChunkK + (ChunkK if pid_split_kv < rcnum else 0)
    )
    if s_len <= 0:
        return
    CHK: gl.constexpr = 64 * NumWarps
    L_CHK: gl.constexpr = gl.BlockedLayout([1], [64], [NumWarps], [0])
    pg_lo = 0
    pg_hi = tl.cdiv(Lmin, KVBlockSize)
    bad = gl.zeros([CHK], gl.int32, layout=L_CHK)
    for pb in range(pg_lo, pg_hi, CHK):
        offs = pb + gl.arange(0, CHK, layout=L_CHK)
        mchk = offs < pg_hi
        pa = gl.amd.cdna3.buffer_load(
            ptr=kv_indices, offsets=r0 * max_block_len + offs, mask=mchk, other=0
        )
        pbv = gl.amd.cdna3.buffer_load(
            ptr=kv_indices, offsets=r1 * max_block_len + offs, mask=mchk, other=0
        )
        bad = bad | (pa != pbv).to(gl.int32)
    fbi = _uniform_i32(gl.max(bad, axis=0))  # 0: fused pass, 1: two single-row passes

    n_pass = 1 + fbi
    for ip in range(n_pass):
        rowA = r0 + ip * fbi
        rowB = r1 - fbi + ip * fbi
        prow = lead * (1 - fbi) + rowA * fbi
        LA = L0 + ip * fbi * (L1 - L0)
        LB = L1 * (1 - fbi) + LA * fbi
        Lp = Lmax * (1 - fbi) + LA * fbi

        context_chunk_num = tl.cdiv(Lp, ChunkK)
        split_context_chunk_num = context_chunk_num // SplitKV
        residual_context_chunks = context_chunk_num % SplitKV
        split_context_start = (
            pid_split_kv * split_context_chunk_num * ChunkK
            + min(pid_split_kv, residual_context_chunks) * ChunkK
        )
        split_context_length = min(
            Lp - split_context_start,
            split_context_chunk_num * ChunkK
            + (ChunkK if pid_split_kv < residual_context_chunks else 0),
        )
        if split_context_length > 0:
            split_context_block = tl.cdiv(split_context_length, KVBlockSize)
            split_context_length = split_context_block * KVBlockSize
            residual_context_blocks = (
                ContextBlockPerChunkK - split_context_block % ContextBlockPerChunkK
            ) % ContextBlockPerChunkK
            residual_context = residual_context_blocks * KVBlockSize

            q_off = (
                gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))
                * stride_q_heads
            )[:, None] + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[
                None, :
            ]
            qA = gl.amd.cdna3.buffer_load(
                ptr=Q_buffer, offsets=rowA * stride_q_batch + q_off
            )
            qB = gl.amd.cdna3.buffer_load(
                ptr=Q_buffer, offsets=rowB * stride_q_batch + q_off
            )

            context_idx = split_context_start - residual_context
            kcol = gl.arange(0, ChunkKPerStage, layout=layout_kcols)
            mask_kv_next_0 = (
                context_idx // KVBlockSize + kcol // KVBlockSize
            ) >= split_context_start // KVBlockSize
            context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                ptr=kv_indices,
                offsets=prow * max_block_len
                + context_idx // KVBlockSize
                + kcol // KVBlockSize,
                mask=mask_kv_next_0,
            )
            mask_kv_next_1 = (
                (context_idx + ChunkKPerStage) // KVBlockSize + kcol // KVBlockSize
            ) >= split_context_start // KVBlockSize
            context_kv_idx_next_1 = gl.amd.cdna3.buffer_load(
                ptr=kv_indices,
                offsets=prow * max_block_len
                + (context_idx + ChunkKPerStage) // KVBlockSize
                + kcol // KVBlockSize,
                mask=mask_kv_next_1,
            )
            w_off = gl.arange(0, ChunkQ, layout=layout_scale)
            wA = gl.amd.cdna3.buffer_load(
                ptr=weights, offsets=rowA * stride_w_batch + w_off
            )
            wB = gl.amd.cdna3.buffer_load(
                ptr=weights, offsets=rowB * stride_w_batch + w_off
            )

            offset_k_fixed = (
                gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b)) % 16
                + gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b))
                // 16
                * (ShuffleRows * 16)
            )[:, None] + (
                kcol % ShuffleRows * 16
                + kcol % KVBlockSize // ShuffleRows * ShuffleRows * HiddenDim
            )[None, :]
            kscol = kcol % KVBlockSize

            mfma_qA = gl.convert_layout(qA, mfma_layout_a)
            mfma_qB = gl.convert_layout(qB, mfma_layout_a)

            context_kv_idx_next_0 = tl.where(mask_kv_next_0, context_kv_idx_next_0, 0)
            k_next_0 = gl.load(
                KV_buffer
                + offset_k_fixed
                + context_kv_idx_next_0.to(gl.int64)[None, :] * stride_k_seq
            )
            k_scale_f_next_0 = gl.load(
                scale_buffer
                + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
                + kscol
            )
            if context_idx + ChunkK < split_context_start + split_context_length:
                context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                    ptr=kv_indices,
                    offsets=prow * max_block_len
                    + (context_idx + ChunkK) // KVBlockSize
                    + kcol // KVBlockSize,
                )

            zero = gl.zeros(
                (ChunkQ, ChunkKPerStage), dtype=tl.float32, layout=mfma_layout
            )
            outA = OutLogits_buffer + rowA.to(tl.int64) * stride_out_batch
            outB = OutLogits_buffer + rowB.to(tl.int64) * stride_out_batch
            ccol = gl.arange(0, ChunkKPerStage, layout=layout_cols)

            k = k_next_0
            k_scale_f = k_scale_f_next_0
            context_kv_idx_next_1 = tl.where(mask_kv_next_1, context_kv_idx_next_1, 0)
            k_next_1 = gl.load(
                KV_buffer
                + offset_k_fixed
                + context_kv_idx_next_1.to(gl.int64)[None, :] * stride_k_seq
            )
            k_scale_f_next_1 = gl.load(
                scale_buffer
                + context_kv_idx_next_1.to(gl.int64) * stride_scale_seq
                + kscol
            )
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            oA = _dot_fp8(mfma_qA, mfma_k, zero, 4)
            oB = _dot_fp8(mfma_qB, mfma_k, zero, 4)
            k_scale_f = gl.convert_layout(k_scale_f, layout_cols)
            oA = gl.maximum(oA, 0.0, propagate_nan=tl.PropagateNan.ALL) * wA[:, None]
            oB = gl.maximum(oB, 0.0, propagate_nan=tl.PropagateNan.ALL) * wB[:, None]
            logitsA = gl.reduce(oA, axis=0, combine_fn=_sum_combine) * k_scale_f
            logitsB = gl.reduce(oB, axis=0, combine_fn=_sum_combine) * k_scale_f
            store_cols = context_idx + ccol
            smask = (store_cols >= split_context_start) & (store_cols < max_model_len)
            gl.amd.cdna3.buffer_store(logitsA, ptr=outA, offsets=store_cols, mask=smask)
            gl.amd.cdna3.buffer_store(logitsB, ptr=outB, offsets=store_cols, mask=smask)

            for context_idx in range(
                split_context_start - residual_context,
                split_context_start + split_context_length - ChunkK,
                ChunkK,
            ):
                k = k_next_1
                k_scale_f = k_scale_f_next_1
                context_kv_idx_next_1 = gl.amd.cdna3.buffer_load(
                    ptr=kv_indices,
                    offsets=prow * max_block_len
                    + (context_idx + ChunkK + ChunkKPerStage) // KVBlockSize
                    + kcol // KVBlockSize,
                )
                k_next_0 = gl.load(
                    KV_buffer
                    + offset_k_fixed
                    + context_kv_idx_next_0.to(gl.int64)[None, :] * stride_k_seq
                )
                k_scale_f_next_0 = gl.load(
                    scale_buffer
                    + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
                    + kscol
                )
                mfma_k = gl.convert_layout(k, mfma_layout_b)
                oA = _dot_fp8(mfma_qA, mfma_k, zero, 4)
                oB = _dot_fp8(mfma_qB, mfma_k, zero, 4)
                k_scale_f = gl.convert_layout(k_scale_f, layout_cols)
                oA = (
                    gl.maximum(oA, 0.0, propagate_nan=tl.PropagateNan.ALL) * wA[:, None]
                )
                oB = (
                    gl.maximum(oB, 0.0, propagate_nan=tl.PropagateNan.ALL) * wB[:, None]
                )
                logitsA = gl.reduce(oA, axis=0, combine_fn=_sum_combine) * k_scale_f
                logitsB = gl.reduce(oB, axis=0, combine_fn=_sum_combine) * k_scale_f
                store_cols = context_idx + ChunkKPerStage + ccol
                smask = (store_cols >= split_context_start) & (
                    store_cols < max_model_len
                )
                gl.amd.cdna3.buffer_store(
                    logitsA, ptr=outA, offsets=store_cols, mask=smask
                )
                gl.amd.cdna3.buffer_store(
                    logitsB, ptr=outB, offsets=store_cols, mask=smask
                )

                k = k_next_0
                k_scale_f = k_scale_f_next_0
                if (
                    context_idx + ChunkK + ChunkK
                    < split_context_start + split_context_length
                ):
                    context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                        ptr=kv_indices,
                        offsets=prow * max_block_len
                        + (context_idx + ChunkK + ChunkK) // KVBlockSize
                        + kcol // KVBlockSize,
                    )
                k_next_1 = gl.load(
                    KV_buffer
                    + offset_k_fixed
                    + context_kv_idx_next_1.to(gl.int64)[None, :] * stride_k_seq
                )
                k_scale_f_next_1 = gl.load(
                    scale_buffer
                    + context_kv_idx_next_1.to(gl.int64) * stride_scale_seq
                    + kscol
                )
                mfma_k = gl.convert_layout(k, mfma_layout_b)
                oA = _dot_fp8(mfma_qA, mfma_k, zero, 4)
                oB = _dot_fp8(mfma_qB, mfma_k, zero, 4)
                k_scale_f = gl.convert_layout(k_scale_f, layout_cols)
                oA = (
                    gl.maximum(oA, 0.0, propagate_nan=tl.PropagateNan.ALL) * wA[:, None]
                )
                oB = (
                    gl.maximum(oB, 0.0, propagate_nan=tl.PropagateNan.ALL) * wB[:, None]
                )
                logitsA = gl.reduce(oA, axis=0, combine_fn=_sum_combine) * k_scale_f
                logitsB = gl.reduce(oB, axis=0, combine_fn=_sum_combine) * k_scale_f
                store_cols = context_idx + ChunkK + ccol
                smask = store_cols < max_model_len
                gl.amd.cdna3.buffer_store(
                    logitsA, ptr=outA, offsets=store_cols, mask=smask
                )
                gl.amd.cdna3.buffer_store(
                    logitsB, ptr=outB, offsets=store_cols, mask=smask
                )

            context_idx = split_context_start + split_context_length - ChunkK
            k = k_next_1
            k_scale_f = k_scale_f_next_1
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            oA = _dot_fp8(mfma_qA, mfma_k, zero, 4)
            oB = _dot_fp8(mfma_qB, mfma_k, zero, 4)
            k_scale_f = gl.convert_layout(k_scale_f, layout_cols)
            oA = gl.maximum(oA, 0.0, propagate_nan=tl.PropagateNan.ALL) * wA[:, None]
            oB = gl.maximum(oB, 0.0, propagate_nan=tl.PropagateNan.ALL) * wB[:, None]
            store_cols = context_idx + ChunkKPerStage + ccol
            logitsA = gl.reduce(oA, axis=0, combine_fn=_sum_combine) * k_scale_f
            logitsA = tl.where(store_cols <= LA - 1, logitsA, float("-inf"))
            logitsB = gl.reduce(oB, axis=0, combine_fn=_sum_combine) * k_scale_f
            logitsB = tl.where(store_cols <= LB - 1, logitsB, float("-inf"))
            smask = (store_cols >= split_context_start) & (store_cols < max_model_len)
            gl.amd.cdna3.buffer_store(logitsA, ptr=outA, offsets=store_cols, mask=smask)
            gl.amd.cdna3.buffer_store(logitsB, ptr=outB, offsets=store_cols, mask=smask)


@cache
def _compile_paged_mqa_logits_fused2(
    ChunkQ,
    ChunkK,
    KVBlockSize,
    HiddenDim,
    is_padded_mode,
    WavesPerEuHint,
    Group,
    NumGroupsPow2,
    NumXcd,
):
    gfx_version = get_gfx()
    assert gfx_version == "gfx950", gfx_version
    warp_size = 64
    target = triton.backends.compiler.GPUTarget("hip", gfx_version, warp_size)
    fp8_ptr = "*fp8e4nv"
    fn_signature = {
        "batch_size": "i32",
        "Q_buffer": fp8_ptr,
        "stride_q_batch": "i32",
        "stride_q_heads": "i32",
        "KV_buffer": fp8_ptr,
        "stride_k_seq": "i32",
        "scale_buffer": "*fp32",
        "stride_scale_seq": "i32",
        "context_len_ptr": "*i32",
        "kv_indices": "*i32",
        "weights": "*fp32",
        "stride_w_batch": "i32",
        "OutLogits_buffer": "*fp32",
        "stride_out_batch": "i64",
        "max_model_len": "i32",
        "max_block_len": "i32",
        "SplitKV": "i32",
        "ChunkQ": "constexpr",
        "ChunkK": "constexpr",
        "KVBlockSize": "constexpr",
        "HiddenDim": "constexpr",
        "GROUP": "constexpr",
        "NG2": "constexpr",
        "NUM_XCD": "constexpr",
    }
    options = {
        "num_warps": 4,
        "waves_per_eu": WavesPerEuHint,
        "num_stages": 2,
        "num_ctas": 1,
        "cluster_dims": [1, 1, 1],
        "arch": gfx_version,
        "backend_name": "hip",
        "warp_size": warp_size,
        "name": "_gluon_paged_mqa_logits_fused2",
    }
    kv_cache_attr = [["tt.divisibility", 16]] if is_padded_mode else []
    src = gluon._runtime.GluonASTSource(
        fn=_gluon_paged_mqa_logits_fused2,
        signature=fn_signature,
        constexprs={
            "ChunkQ": ChunkQ,
            "ChunkK": ChunkK,
            "KVBlockSize": KVBlockSize,
            "HiddenDim": HiddenDim,
            "GROUP": Group,
            "NG2": NumGroupsPow2,
            "NUM_XCD": NumXcd,
        },
        attrs={
            (1,): [["tt.divisibility", 16], ["tt.pointer_range", 32]],  # Q_buffer
            (2,): [["tt.divisibility", 16]],  # stride_q_batch
            (3,): [["tt.divisibility", 16]],  # stride_q_heads
            (4,): kv_cache_attr,  # KV_buffer
            (5,): kv_cache_attr,  # stride_k_seq
            (6,): kv_cache_attr,  # scale_buffer
            (7,): kv_cache_attr,  # stride_scale_seq
            (8,): [["tt.pointer_range", 32]],  # context_len_ptr
            (9,): [["tt.pointer_range", 32]],  # kv_indices
            (10,): [["tt.divisibility", 16], ["tt.pointer_range", 32]],  # weights
            (11,): [["tt.divisibility", 16]],  # stride_w_batch
        },
    )
    return triton.compile(src, target=target, options=options)


# Groups (one request's verify rows) the kernel ranks for longest-first dispatch:
# a compile-time power of two in [_NG2_SMALL, _NG2_MAX] that covers the batch.
# Ranking costs O(NG2^2) per program.
_NG2_SMALL = 32
_NG2_MAX = 256
_WAVES_PER_EU = 3
_CHUNK_K = 256
# KV splits per pair: about _SPLIT_ROWS / rows, a power of two in
# [_SPLIT_KV_MIN, _SPLIT_KV_MAX] (so a multiple of the XCD count). Swept on 4- and
# 6-row verify groups, 4k-400k contexts: the best split falls from 128 at 4-6 rows
# to 16 at 64-96 rows.
_SPLIT_ROWS = 1536
_SPLIT_KV_MIN = 16
_SPLIT_KV_MAX = 128
# Validated row range (decode and MTP-verify batches of one DP rank).
MAX_ROWS = 1536


def _split_kv(rows: int) -> int:
    target = max(1, _SPLIT_ROWS // rows)
    return min(_SPLIT_KV_MAX, max(_SPLIT_KV_MIN, 1 << (target.bit_length() - 1)))


def gluon_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,  # [rows, 1, heads, hidden] fp8 e4m3fn
    kv_cache: torch.Tensor,  # [num_pages, page, 1, hidden + 4] preshuffled fp8 + scales
    weights: torch.Tensor,  # [rows, heads] fp32
    out_logits: torch.Tensor,  # [rows, >= max_model_len] fp32
    context_lens: torch.Tensor,  # [rows] int32
    kv_indices: torch.Tensor,  # [rows, max_block_len] int32, unit column stride
    max_model_len: int,
    rows_per_request: int,
) -> None:
    """Paged MQA logits of `supported` inputs, written into
    `out_logits[:, :context_lens[row]]`."""
    rows, _, heads, hidden_dim = q_fp8.shape
    _, block_size, _, index_dim = kv_cache.shape
    max_block_len = kv_indices.stride(0)

    kv_cache = kv_cache.view(-1, block_size * index_dim)
    kv_cache_fp8 = kv_cache[..., : block_size * hidden_dim].view(torch.float8_e4m3fn)
    kv_cache_scale = kv_cache[..., block_size * hidden_dim :].view(torch.float32)
    is_padded_mode = kv_cache_fp8.stride(0) % 16 == 0

    split_kv = _split_kv(rows)
    ngroups = rows // rows_per_request
    ng2 = max(_NG2_SMALL, 1 << (ngroups - 1).bit_length())
    kernel = _compile_paged_mqa_logits_fused2(
        heads,
        _CHUNK_K,
        block_size,
        hidden_dim,
        is_padded_mode,
        _WAVES_PER_EU,
        rows_per_request,
        ng2,
        _NUM_XCD.value,
    )
    kernel[(ngroups * ((rows_per_request + 1) // 2) * split_kv, 1, 1)](
        rows,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_logits,
        out_logits.stride(0),
        max_model_len,
        max_block_len,
        split_kv,
        heads,
        _CHUNK_K,
        block_size,
        hidden_dim,
        rows_per_request,
        ng2,
        _NUM_XCD.value,
    )


def supported(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    preshuffle: bool,
    rows_per_request: int,
) -> bool:
    """Whether `gluon_fp8_paged_mqa_logits` serves this call: MTP verify
    rows of the flattened decode path, `rows_per_request` (>= 2) consecutive
    rows per request and at most `_NG2_MAX` requests. `q_fp8` is
    `[rows, 1, 32, 128]`, `kv_cache_fp8` `[num_pages, page, 1, 132]` with a
    page of 16, 32 or 64 tokens, `context_lens` one length per row and
    `block_tables` one page row per row, each contiguous. Plain decode rows
    gain little from this kernel and stay on aiter's."""
    rows = q_fp8.shape[0]
    return (
        preshuffle
        and rows_per_request >= 2
        and rows % rows_per_request == 0
        and rows // rows_per_request <= _NG2_MAX
        and 1 <= rows <= MAX_ROWS
        and q_fp8.dim() == 4
        and q_fp8.shape[1:] == (1, 32, 128)
        and kv_cache_fp8.dim() == 4
        and kv_cache_fp8.shape[1] in (16, 32, 64)
        and kv_cache_fp8.shape[2:] == (1, 132)
        and context_lens.numel() == rows
        and context_lens.dtype == torch.int32
        and context_lens.is_contiguous()
        and block_tables.dim() == 2
        and block_tables.dtype == torch.int32
        and block_tables.shape[0] == rows
        and block_tables.stride(1) == 1
        and get_gfx() == "gfx950"
    )
