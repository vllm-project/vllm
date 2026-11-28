# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand-scheduled Gluon boundary kernel for GQA tiles (four heads x 32 tokens).

Inference only, for GQA ratios that are multiples of 4. Same math as
`boundary._boundary` and bitwise equal to it: own-block QK, the
token-normalized weights W, W·U, and the merge with the bulk partial. Rows are
token-major (row = token * 4 + head), so Q8, the BF16 partial, and the output
move as TMA boxes `[32, 4, 128]` / `[32, 4, 64]`. U is consumed transposed
(`[B, HK, 128, N]`) because the scaled MMA takes its B operand K-contiguous.
"""

from functools import lru_cache

import torch

from vllm.triton_utils import gl, gluon

TensorMemoryLayout = gl.nvidia.blackwell.TensorMemoryLayout
TensorMemoryScalesLayout = gl.nvidia.blackwell.TensorMemoryScalesLayout
allocate_tensor_memory = gl.nvidia.blackwell.allocate_tensor_memory
fence_async_shared = gl.nvidia.blackwell.fence_async_shared
mbarrier = gl.nvidia.blackwell.mbarrier
tcgen05_commit = gl.nvidia.blackwell.tcgen05_commit
tcgen05_mma_scaled = gl.nvidia.blackwell.tcgen05_mma_scaled
tma = gl.nvidia.blackwell.tma
TensorDescriptor = gluon.nvidia.hopper.TensorDescriptor

PACK = 4
KEYS = 32
# One program per SM overlaps two tiles; it needs several tiles per program to
# fill its pipeline. With fewer, the two-CTA Triton kernel is faster (B200
# sweep: Triton wins up to 7 tiles per SM, this kernel from 10).
MIN_TILES_PER_SM = 8


@lru_cache(maxsize=8)
def _sms(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def preferred(b, hq, hk, n, device):
    """Four query heads per K/V head fill the 128 MMA rows, and enough tiles."""
    tiles = n // KEYS * (hq // PACK) * b
    return (hq // hk) % PACK == 0 and tiles >= MIN_TILES_PER_SM * _sms(device)


def _descriptor(x, block, dtype):
    return TensorDescriptor.from_tensor(
        x, block, gl.NVMMASharedLayout.get_default_for(block, dtype)
    )


def boundary(q8, k8, sq, sk, ut, ae, partial, lse, out):
    """Merge the boundary partition with the bulk `partial`/`lse` into `out`.

    `ut` is U transposed, `[B, HK, 128, N]`; the other operands are as
    returned by `boundary.prepare` and the bulk kernel.
    """
    b, hq, n, _ = q8.shape
    hk = k8.shape[1]

    def rows(x, h):
        return x.transpose(1, 2).reshape(b * n, h, 128)

    q_desc = _descriptor(rows(q8, hq), [KEYS, PACK, 128], gl.float8e4nv)
    k_desc = _descriptor(
        k8.transpose(1, 2).reshape(b * n, hk * 128), [KEYS, 128], gl.float8e4nv
    )
    ut_desc = _descriptor(ut.reshape(b * hk * 128, n), [128, KEYS], gl.float8e4nv)
    p_desc = _descriptor(rows(partial, hq), [KEYS, PACK, 64], gl.bfloat16)
    o_desc = _descriptor(rows(out, hq), [KEYS, PACK, 64], gl.bfloat16)
    total = n // KEYS * (hq // PACK) * b
    # One persistent program per SM: the partitions already overlap two tiles.
    return _boundary_ws[(min(total, _sms(q8.device)),)](
        q_desc,
        k_desc,
        ut_desc,
        p_desc,
        o_desc,
        sq,
        sk,
        ae,
        lse,
        n,
        hq,
        hk,
        b,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Warp-specialized kernel: four partitions over double-buffered stages.
#   load    (1 warp)   TMA loads of Q8, K8, U^T, and the BF16 partial
#   mma     (1 warp)   QK(i) is issued before W·U(i - 1), so the next tile's
#                      scores compute while softmax works on this one
#   softmax (2 x 4)    two warpgroups, one per buffer: softmax and W(i), the
#                      scales of tile i + 2, then the merge of O(i) and its TMA store
# Tile i uses buffer i % 2 with mbarrier phase (i // 2) & 1.
# ---------------------------------------------------------------------------


@gluon.jit
def _tile_coords(i, N, HQ: gl.constexpr, HK: gl.constexpr):
    PACK: gl.constexpr = 4
    TILES = N // 32
    GROUPS_H: gl.constexpr = HQ // PACK
    t = gl.program_id(0) + i * gl.num_programs(0)
    # Head groups vary fastest: concurrent programs cover whole token rows.
    packed_h = t % GROUPS_H
    tile = (t // GROUPS_H) % TILES
    b = t // (TILES * GROUPS_H)
    kh = packed_h // ((HQ // HK) // PACK)
    return tile, packed_h, b, kh


@gluon.jit
def _tile_count(N, HQ: gl.constexpr, B):
    total = (N // 32) * (HQ // 4) * B
    return (total - gl.program_id(0) + gl.num_programs(0) - 1) // gl.num_programs(0)


@gluon.jit
def _load_partition(
    descs,
    smem,
    bars,
    N,
    HQ: gl.constexpr,
    HK: gl.constexpr,
    B,
):
    q_desc, k_desc, ut_desc, p_desc, o_desc = descs
    q_s, k_s, ut_s, p_s, w_s, out_s = smem
    load_ready, stage_empty = bars[0], bars[1]
    p_ready, p_empty = bars[9], bars[10]
    op_bytes: gl.constexpr = (
        q_desc.block_type.nbytes + k_desc.block_type.nbytes + ut_desc.block_type.nbytes
    )
    p_bytes: gl.constexpr = 2 * p_desc.block_type.nbytes
    count = _tile_count(N, HQ, B)
    for pair in range((count + 1) // 2):
        phase = pair & 1
        for buf in gl.static_range(2):
            i = pair * 2 + buf
            if i < count:
                tile, packed_h, b, kh = _tile_coords(i, N, HQ, HK)
                row0 = b * N + tile * 32
                # Operands (freed once W·U completes) and the partial (freed once
                # the merge has read it) turn over independently.
                mbarrier.wait(stage_empty.index(buf), phase ^ 1)
                bar = load_ready.index(buf)
                mbarrier.expect(bar, op_bytes)
                tma.async_load(q_desc, [row0, packed_h * 4, 0], bar, q_s.index(buf))
                tma.async_load(k_desc, [row0, kh * 128], bar, k_s.index(buf))
                tma.async_load(
                    ut_desc, [(b * HK + kh) * 128, tile * 32], bar, ut_s.index(buf)
                )
                mbarrier.wait(p_empty.index(buf), phase ^ 1)
                pbar = p_ready.index(buf)
                mbarrier.expect(pbar, p_bytes)
                tma.async_load(
                    p_desc, [row0, packed_h * 4, 0], pbar, p_s.index(buf * 2)
                )
                tma.async_load(
                    p_desc, [row0, packed_h * 4, 64], pbar, p_s.index(buf * 2 + 1)
                )


@gluon.jit
def _mma_partition(
    smem,
    tmem,
    bars,
    N,
    HQ: gl.constexpr,
    HK: gl.constexpr,
    B,
):
    q_s, k_s, ut_s, p_s, w_s, out_s = smem
    s_t, o_t, sq_ts, sk_ts, ws_ts, us_t = tmem
    load_ready, stage_empty = bars[0], bars[1]
    scales_ready, s_ready, s_empty = bars[2], bars[3], bars[4]
    w_ready, w_empty, o_ready, o_empty = bars[5], bars[6], bars[7], bars[8]
    count = _tile_count(N, HQ, B)
    # Covers i = 0 .. count: QK(i) for i < count, then W·U(i - 1) for i >= 1.
    for pair in range(count // 2 + 1):
        for buf in gl.static_range(2):
            i = pair * 2 + buf
            if i < count:
                phase = pair & 1
                mbarrier.wait(load_ready.index(buf), phase)
                mbarrier.wait(scales_ready.index(buf), phase)
                mbarrier.wait(s_empty.index(buf), phase ^ 1)
                tcgen05_mma_scaled(
                    q_s.index(buf).reshape([128, 128]),
                    k_s.index(buf).permute((1, 0)),
                    s_t.index(buf),
                    sq_ts[buf],
                    sk_ts[buf],
                    "e4m3",
                    "e4m3",
                    use_acc=False,
                )
                tcgen05_commit(s_ready.index(buf))
            if (i >= 1) & (i <= count):
                jbuf: gl.constexpr = 1 - buf
                jphase = ((i - 1) // 2) & 1
                mbarrier.wait(w_ready.index(jbuf), jphase)
                mbarrier.wait(o_empty.index(jbuf), jphase ^ 1)
                tcgen05_mma_scaled(
                    w_s.index(jbuf),
                    ut_s.index(jbuf).permute((1, 0)),
                    o_t.index(jbuf),
                    ws_ts[jbuf],
                    us_t,
                    "e4m3",
                    "e4m3",
                    use_acc=False,
                )
                tcgen05_commit(o_ready.index(jbuf))
                tcgen05_commit(w_empty.index(jbuf))
                tcgen05_commit(stage_empty.index(jbuf))


@gluon.jit
def _scale_offsets(
    i,
    layout: gl.constexpr,
    ROWS: gl.constexpr,
    IS_Q: gl.constexpr,
    N,
    HQ: gl.constexpr,
    HK: gl.constexpr,
):
    # Byte offsets of a tile's E8M0 scales in the packed F8_128x4 atoms, built
    # directly in the tensor-memory register layout (no layout conversion).
    tile, packed_h, b, kh = _tile_coords(i, N, HQ, HK)
    srow = gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    sgrp = gl.arange(0, 4, layout=gl.SliceLayout(0, layout))
    if IS_Q:
        r = tile * 32 + srow // 4
        plane = b * HQ + packed_h * 4 + srow % 4
    else:
        r = tile * 32 + srow
        plane = (b * HK + kh) + srow * 0
    atom = (plane * (N // 128) + r // 128) * 512 + (r % 32) * 16 + (r % 128 // 32) * 4
    return atom[:, None] + sgrp[None, :]


@gluon.jit
def _softmax_merge_partition(
    descs,
    smem,
    tmem,
    bars,
    SQ,
    SK,
    AE,
    LSE,
    PAR: gl.constexpr,
    N,
    HQ: gl.constexpr,
    HK: gl.constexpr,
    B,
):
    # One of two warpgroups: tiles i with i % 2 == PAR, all on buffer PAR. Each
    # computes softmax and W(i), then merges O(i) with the bulk partial, so the
    # two groups overlap each other's merge.
    LOG2E: gl.constexpr = 1.4426950408889634
    LOG2_448: gl.constexpr = 8.807354922057604
    q_desc, k_desc, ut_desc, p_desc, o_desc = descs
    q_s, k_s, ut_s, p_s, w_s, out_s = smem
    s_t, o_t, sq_ts, sk_ts, ws_ts, us_t = tmem
    scales_ready, s_ready, s_empty = bars[2], bars[3], bars[4]
    w_ready, w_empty, o_ready, o_empty = bars[5], bars[6], bars[7], bars[8]
    p_ready, p_empty = bars[9], bars[10]
    s_reg: gl.constexpr = s_t.index(0).get_reg_layout()
    rows: gl.constexpr = gl.SliceLayout(1, s_reg)
    cols: gl.constexpr = gl.SliceLayout(0, s_reg)
    o_reg: gl.constexpr = o_t.index(0).slice(0, 64).get_reg_layout()
    o_rows: gl.constexpr = gl.SliceLayout(1, o_reg)
    sq_t = sq_ts[PAR]
    sk_t = sk_ts[PAR]
    ws_t = ws_ts[PAR]
    sq_reg: gl.constexpr = sq_t.get_reg_layout()
    sk_reg: gl.constexpr = sk_t.get_reg_layout()
    count = _tile_count(N, HQ, B)
    if count > PAR:
        sq_t.store(gl.load(SQ + _scale_offsets(PAR, sq_reg, 128, True, N, HQ, HK)))
        sk_t.store(gl.load(SK + _scale_offsets(PAR, sk_reg, 32, False, N, HQ, HK)))
        fence_async_shared()
        mbarrier.arrive(scales_ready.index(PAR), count=1)
    row = gl.arange(0, 128, layout=rows)
    col = gl.arange(0, 32, layout=cols)
    o_row = gl.arange(0, 128, layout=o_rows)
    for pair in range((count + 1) // 2):
        i = pair * 2 + PAR
        if i < count:
            phase = pair & 1
            tile, packed_h, b, kh = _tile_coords(i, N, HQ, HK)
            # Prefetch: AE and the completed-block LSE of this tile, and the
            # scales of this group's next tile.
            ae = gl.load(AE + (b * HK + kh) * N + tile * 32 + col)
            r_o = tile * 32 + o_row // 4
            bulk_lse = gl.load(LSE + (b * HQ + packed_h * 4 + o_row % 4) * N + r_o)
            more = i + 2 < count
            sq_next = gl.load(
                SQ + _scale_offsets(i + 2, sq_reg, 128, True, N, HQ, HK),
                mask=more,
                other=127,
            )
            sk_next = gl.load(
                SK + _scale_offsets(i + 2, sk_reg, 32, False, N, HQ, HK),
                mask=more,
                other=127,
            )
            mbarrier.wait(s_ready.index(PAR), phase)
            score = s_t.index(PAR).load(s_reg) * 0.08838834764831845
            mbarrier.arrive(s_empty.index(PAR), count=1)
            r = tile * 32 + row // 4
            visible = col[None, :] <= (r % 32)[:, None]
            score = gl.where(visible, score, -float("inf"))
            m = gl.max(score, axis=1)
            denominator = gl.sum(gl.exp(score - m[:, None]), axis=1)
            logw = (score - m[:, None]) * LOG2E + ae[None, :].to(gl.float32)
            logw = gl.where(visible & (ae[None, :] != -1000), logw, -float("inf"))
            wmax = gl.max(logw, axis=1)
            we = gl.where(wmax == -float("inf"), 0.0, gl.ceil(wmax - LOG2_448))
            we = gl.minimum(127.0, gl.maximum(-127.0, we))
            w = gl.exp2(logw - we[:, None]).to(gl.float8e4nv)
            ws = gl.expand_dims((we + 127.0).to(gl.uint8), 1)
            mbarrier.wait(w_empty.index(PAR), phase ^ 1)
            w_s.index(PAR).store(w)
            ws_t.store(gl.convert_layout(ws, ws_t.get_reg_layout()))
            fence_async_shared()
            mbarrier.arrive(w_ready.index(PAR), count=1)
            # QK(i) completed before S(i) was read, so this group's scale
            # buffers are free for its next tile.
            if more:
                sq_t.store(sq_next)
                sk_t.store(sk_next)
                fence_async_shared()
                mbarrier.arrive(scales_ready.index(PAR), count=1)
            # Merge with the completed-block partial while W·U(i) runs.
            live = r_o >= 32
            bulk_lse = gl.where(live, bulk_lse, -float("inf"))
            m = gl.convert_layout(m, o_rows)
            denominator = gl.convert_layout(denominator, o_rows)
            common = gl.maximum(m, bulk_lse)
            a = gl.exp(bulk_lse - common)
            z = gl.exp(m - common)
            inverse = 1.0 / (a + z * denominator)
            mbarrier.wait(p_ready.index(PAR), phase)
            mbarrier.wait(o_ready.index(PAR), phase)
            # This group's stores from tile i - 2 must have left out_s.
            tma.store_wait(0)
            for half in gl.static_range(2):
                bulk = p_s.index(PAR * 2 + half).reshape([128, 64]).load(o_reg)
                numerator = o_t.index(PAR).slice(half * 64, 64).load(o_reg)
                bulk = gl.where(live[:, None], bulk.to(gl.float32), 0.0)
                result = (a[:, None] * bulk + z[:, None] * numerator) * inverse[:, None]
                out_s.index(PAR * 2 + half).reshape([128, 64]).store(
                    result.to(gl.bfloat16)
                )
            mbarrier.arrive(p_empty.index(PAR), count=1)
            mbarrier.arrive(o_empty.index(PAR), count=1)
            fence_async_shared()
            row0 = b * N + tile * 32
            tma.async_copy_shared_to_global(
                o_desc, [row0, packed_h * 4, 0], out_s.index(PAR * 2)
            )
            tma.async_copy_shared_to_global(
                o_desc, [row0, packed_h * 4, 64], out_s.index(PAR * 2 + 1)
            )
    tma.store_wait(0)


@gluon.jit(do_not_specialize=["B"])
def _boundary_ws(
    q_desc,
    k_desc,
    ut_desc,
    p_desc,
    o_desc,
    SQ,
    SK,
    AE,
    LSE,
    N,
    HQ: gl.constexpr,
    HK: gl.constexpr,
    B,
):
    q_s = gl.allocate_shared_memory(gl.float8e4nv, [2, 32, 4, 128], q_desc.layout)
    k_s = gl.allocate_shared_memory(gl.float8e4nv, [2, 32, 128], k_desc.layout)
    ut_s = gl.allocate_shared_memory(gl.float8e4nv, [2, 128, 32], ut_desc.layout)
    p_s = gl.allocate_shared_memory(gl.bfloat16, [4, 32, 4, 64], p_desc.layout)
    w_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [128, 32], gl.float8e4nv
    )
    w_s = gl.allocate_shared_memory(gl.float8e4nv, [2, 128, 32], w_layout)
    out_s = gl.allocate_shared_memory(gl.bfloat16, [4, 32, 4, 64], o_desc.layout)

    s_t = allocate_tensor_memory(
        gl.float32, [2, 128, 32], TensorMemoryLayout([128, 32], col_stride=1)
    )
    o_t = allocate_tensor_memory(
        gl.float32, [2, 128, 128], TensorMemoryLayout([128, 128], col_stride=1)
    )
    # Scale buffers cannot be multibuffered: one allocation per stage.
    scales: gl.constexpr = TensorMemoryScalesLayout()
    sq_ts = (
        allocate_tensor_memory(gl.uint8, [128, 4], scales),
        allocate_tensor_memory(gl.uint8, [128, 4], scales),
    )
    sk_ts = (
        allocate_tensor_memory(gl.uint8, [32, 4], scales),
        allocate_tensor_memory(gl.uint8, [32, 4], scales),
    )
    ws_ts = (
        allocate_tensor_memory(gl.uint8, [128, 1], scales),
        allocate_tensor_memory(gl.uint8, [128, 1], scales),
    )
    us_t = allocate_tensor_memory(gl.uint8, [128, 1], scales)
    us_t.store(gl.full([128, 1], 119, gl.uint8, layout=us_t.get_reg_layout()))

    load_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    stage_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    scales_ready = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    s_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    w_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    w_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    o_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    o_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    p_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    p_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    bars = (
        load_ready,
        stage_empty,
        scales_ready,
        s_ready,
        s_empty,
        w_ready,
        w_empty,
        o_ready,
        o_empty,
        p_ready,
        p_empty,
    )
    for k in gl.static_range(11):
        for j in gl.static_range(2):
            mbarrier.init(bars[k].index(j), count=1)
    descs = (q_desc, k_desc, ut_desc, p_desc, o_desc)
    smem = (q_s, k_s, ut_s, p_s, w_s, out_s)
    tmem = (s_t, o_t, sq_ts, sk_ts, ws_ts, us_t)
    gl.warp_specialize(
        [
            (
                _softmax_merge_partition,
                (descs, smem, tmem, bars, SQ, SK, AE, LSE, 0, N, HQ, HK, B),
            ),
            (
                _softmax_merge_partition,
                (descs, smem, tmem, bars, SQ, SK, AE, LSE, 1, N, HQ, HK, B),
            ),
            (_load_partition, (descs, smem, bars, N, HQ, HK, B)),
            (_mma_partition, (smem, tmem, bars, N, HQ, HK, B)),
        ],
        [4, 1, 1],
        [232, 24, 24],
    )
