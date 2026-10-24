# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 ``mono_pre``: one MiniMax-M3 sparse layer, input norm through cache insert.

One launch per layer and rank, ``BLOCKS`` CTAs x ``THREADS`` threads, for S =
``tokens`` decode rows, ``q_len`` consecutive rows a request (each row below is
per token)::

    acc = ar + res (f32)  -> residual out bf16(acc)
      -> gemma RMSNorm -> per-token FP8 (amax / FP8_MAX, true divide)
      -> qkv | index_q | index_k GEMV on the preshuffled ptpc-FP8 weight
      -> per-head gemma norm + partial NeoX RoPE
      -> q, index_q out (bf16); K / V -> page-16 SHUFFLE cache against that
         cache's one scale; index_k -> index cache (E4M3, unit scale)

It reads exactly what the original path leaves in memory (the ``ar`` / ``res``
pair of the custom all-reduce, the loaded weights, the bound caches) and writes
what the original ``fused_qknorm_idxrqknorm`` writes, so either path can run any
layer. The numerical contracts (reduction orders, where values round to bf16)
follow the aiter kernels the original path uses; see each stage.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.minimax_m3.amd.mono.config import (
    BLOCK_PAGES,
    BLOCKS,
    HEAD_DIM,
    HIDDEN,
    LOCAL_Q_HEADS,
    MAX_QKV_ROWS,
    MAX_TOKENS,
    ONE_INDEX_HEAD,
    PAGE16,
    ROTARY_DIM,
    SPARSE_BLOCK,
    THREADS,
    TP,
    WAVES,
    IndexHeads,
    qkv_rows,
)
from vllm.models.minimax_m3.amd.mono.kernels.common import (
    CM_DEV,
    Mailbox,
    ar_block_sums,
    ar_pack_sumsq,
    bf16_pair,
    bf16_round,
    butterfly,
    div_rn,
    fp8_pack4,
    hw_rsq,
    kernel_symbol,
    kv_cache_scale,
    memrealtime,
    rsrc,
    traced,
    uniform,
    wave_max,
)
from vllm.models.minimax_m3.amd.mono.kernels.index_score import (
    emit_index_scores,
    index_scale_log2e,
    step_rows,
)

FP8_MAX = 448.0
ROWS_PER_TASK = 16
K_CHUNKS = HIDDEN // 64  # 64-k chunks: one dwordx4 of weight per lane
CHUNKS_PER_WAVE = K_CHUNKS // WAVES
Q_OFF, K_OFF, V_OFF = 0, LOCAL_Q_HEADS * HEAD_DIM, (LOCAL_Q_HEADS + 1) * HEAD_DIM
N_HEAD_TASKS = LOCAL_Q_HEADS + 4  # q heads, k, v, the rank's index_q, index_k
IK_TASK = LOCAL_Q_HEADS + 3
HEAD_ROW_GROUPS = HEAD_DIM // ROWS_PER_TASK  # GEMV tasks a head
ELEMS = 4  # per lane in a head task: 32 lanes x 4 = 128, the aiter kernel's split
PAGE_BYTES = PAGE16 * HEAD_DIM  # one page-16 of one kv head (fp8)
# GEMV tasks, head tasks (every token of a head on one CTA) and the tokens' norm
# tasks each own a CTA, in this order; a head CTA's half waves hold the tokens.
# The rank's own rows (q | k | v | its index q | index k) are the first ONE_GEMV
# tasks under indexer context parallelism too, and its other index q heads' the
# OTHER_GEMV after the norm tasks, run only in a step that needs them: the CTA
# layout of a step without long requests is the one-head build's.
ONE_GEMV = qkv_rows(1) // ROWS_PER_TASK
OTHER_GEMV = (TP - 1) * HEAD_ROW_GROUPS
HT0 = ONE_GEMV
NT0 = HT0 + N_HEAD_TASKS
XT0 = NT0 + MAX_TOKENS
assert XT0 + OTHER_GEMV <= BLOCKS
assert MAX_TOKENS * 32 <= THREADS
assert HIDDEN // 4 % THREADS == 0

# The launch goes on the caller's current stream (graph capture included).
_CURRENT_STREAM = fx.Stream(None)

SCRATCH_QKV = 0  # (value, tag) pairs, one per qkv row
SCRATCH_X8 = MAX_TOKENS * MAX_QKV_ROWS * 8  # S > 2: the norm tasks' x8 rows, words
SCRATCH_X8S = SCRATCH_X8 + MAX_TOKENS * HIDDEN // 4 * 8  # and their scales
# fused layer kernel only: a head task's / the residual's done flags
SCRATCH_HDONE = SCRATCH_X8S + MAX_TOKENS * 8
SCRATCH_RDONE = SCRATCH_HDONE + MAX_TOKENS * N_HEAD_TASKS * 8
SCRATCH_BYTES = SCRATCH_RDONE + MAX_TOKENS * 8


def _ld_bf16x4(r, k):
    w = fx.Vector(bo.buffer_load(r, k // 2, vec_width=2, dtype=T.i32))
    v = w.bitcast(fx.BFloat16).to(fx.Float32)
    return [v[j] for j in range(4)]


def ld_raw8(r, lt):
    """Logical thread lt's 8 bf16 of a row, as loaded (for ``bf16x8_f32``)."""
    return fx.Vector(bo.buffer_load(r, lt * 4, vec_width=4, dtype=T.i32))


def bf16x8_f32(raw):
    v = raw.bitcast(fx.BFloat16)
    return [fx.Float32(v[j]) for j in range(8)]


def kv_page(slot):
    """A KV slot's (page-16 id, token in the page) in the main cache.

    A block's page ids cover ``PAGE16_SIDES`` sides, so a slot's page is its
    block's base widened by that much; the caller reaches V from the same id
    because its cache view starts one side in. The index cache is paged by whole
    blocks instead, so it keeps the slot itself.
    """
    blk, off = slot // SPARSE_BLOCK, slot % SPARSE_BLOCK
    return blk * BLOCK_PAGES + off // PAGE16, off % PAGE16


def head_rope_inputs(tk, gamma, d0, cos_sin, positions):
    """Token tk's head gamma and cos / sin at dims d0 .. d0 + ELEMS (as f32)."""
    # positions are int64: this token's low word
    pos = fx.Int32(bo.buffer_load(rsrc(positions), 2 * tk, vec_width=1, dtype=T.i32))
    half = ROTARY_DIM // 2
    ib = (d0 < half).select(d0, d0 - half)
    r_cs = rsrc(cos_sin)
    return (
        _ld_bf16x4(rsrc(gamma), d0),
        _ld_bf16x4(r_cs, pos * ROTARY_DIM + fx.min(ib, half - ELEMS)),
        _ld_bf16x4(r_cs, pos * ROTARY_DIM + half + fx.min(ib, half - ELEMS)),
    )


def head_norm_rope(e, lane, gw4, cs, sn, eps):
    """One head's gemma RMSNorm + partial NeoX RoPE on a half wave, lane l of it
    holding dims 4 (l % 32) .. + 4 (the aiter kernel's split and reduction order)."""
    d0 = (lane % 32) * ELEMS
    half = ROTARY_DIM // 2
    first = d0 < half
    ss = fx.Float32(0.0)
    for j in range_constexpr(ELEMS):
        ss = ss + e[j] * e[j]
    ss = butterfly(ss, (16, 8, 4, 2, 1))
    rr = hw_rsq(ss / float(HEAD_DIM) + eps)
    n = [e[j] * rr * (gw4[j] + 1.0) for j in range(ELEMS)]
    # partial NeoX RoPE over the first ROTARY_DIM dims: the partner is in this half
    in_rope = d0 < ROTARY_DIM
    partner_lane = first.select(lane + half // ELEMS, lane - half // ELEMS)
    out = []
    for j in range_constexpr(ELEMS):
        p = fx.Int32(
            rocdl.ds_bpermute(
                T.i32,
                (fx.min(fx.max(partner_lane, 0), 63) * 4).ir_value(),
                n[j].bitcast(fx.Int32).ir_value(),
            )
        ).bitcast(fx.Float32)
        # hipcc contracts ``x*c - p*s`` into fma(x, c, -(p*s)); match it
        ps = p * sn[j]
        rot = fmath.fma(n[j], cs[j], first.select(-ps, ps))
        out.append(in_rope.select(rot, n[j]))
    return out


@traced
def emit_pre_attn(
    tid, bid, lane, wave, x8, red, mb_put, mb_put_words, mb_poll, qkv_mb, x8_mb,
    x8s_mb, stamp, eps, tokens, heads, ptrs, signal=None, long_step=None,
):  # fmt: skip
    """K1's body: the GEMV tasks (ROWS_PER_TASK of the ``heads`` projection's rows
    each: the rank's own on CTAs 0 .. ONE_GEMV, its other index q heads' on CTAs
    XT0 .., in a step where ``long_step()``), the head tasks (CTAs HT0 .., a token
    per half wave) and, with S > 2, one norm task per token (CTAs NT0 ..) feeding
    the GEMV tasks."""
    rows = heads.rows
    (
        ar,
        res,
        h_out, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin, positions,
        slot_mapping, k_cache, v_cache, k_scale, v_scale, index_cache, q_out, iq_out,
    ) = ptrs  # fmt: skip

    # stores K4 reads in the same launch (signal): device-coherent like the
    # mailboxes, so the done flags need no L2 writeback
    cm_out = CM_DEV if signal is not None else 0

    def publish_done(mb_addr, idx, who=None):
        """The CTA's stores done, then flag idx of ``mb_addr`` (from thread 0, or
        from every thread where ``who``)."""
        if const_expr(signal is not None):
            fx.memory_fence(
                syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Release
            )
            gpu.barrier()
            if (tid == 0) if who is None else who:
                mb_put(mb_addr, idx, fx.Int32(1))

    # acc = bf16(sum of partials) + residual, f32; the norm reads acc unrounded.
    # Elements are laid out as the fused all-reduce's logical threads (8 each): this
    # lane holds lt0 and, in lanes < 32, lt1, so rstd reduces in its order.
    lts = [wave * 96 + lane, wave * 96 + 64 + fx.min(lane, 31)]

    def load_norm_inputs(toks):
        """(gamma, [token][lt] -> (all-reduce, residual)) as loaded."""
        raw_g = [ld_raw8(rsrc(g_in), lt) for lt in lts]
        raw_ar = [
            [
                (
                    ld_raw8(rsrc(ar), tk * (HIDDEN // 8) + lt),
                    ld_raw8(rsrc(res), tk * (HIDDEN // 8) + lt),
                )
                for lt in lts
            ]
            for tk in toks
        ]
        return raw_g, raw_ar

    def norm_quant(toks, raw_g, raw_ar, store_residual):
        """Tokens toks: the residual out (where ``store_residual(j)``), gemma RMSNorm
        and per-token FP8 into x8 row j (the j-th of toks) -> the x scales."""
        gs = [bf16x8_f32(raw) for raw in raw_g]
        accs = [  # [j][lt][8]
            [
                [a + r for a, r in zip(bf16x8_f32(ra), bf16x8_f32(rr))]
                for ra, rr in raw_ar[j]
            ]
            for j in range(len(toks))
        ]
        for j in range_constexpr(len(toks)):
            if store_residual(j):
                r_h = rsrc(h_out)
                bo.buffer_store(
                    fx.Vector.from_elements(accs[j][0], fx.Float32).to(fx.BFloat16),
                    r_h,
                    toks[j] * HIDDEN + lts[0] * 8,
                    cache_modifier=cm_out,
                )
                if lane < 32:
                    bo.buffer_store(
                        fx.Vector.from_elements(accs[j][1], fx.Float32).to(fx.BFloat16),
                        r_h,
                        toks[j] * HIDDEN + lts[1] * 8,
                        cache_modifier=cm_out,
                    )
        # every token's sum of squares behind one set of barriers
        tots = ar_block_sums(
            [
                (
                    ar_pack_sumsq(accs[j][0]),
                    (lane < 32).select(ar_pack_sumsq(accs[j][1]), fx.Float32(0.0)),
                )
                for j in range(len(toks))
            ],
            lane,
            wave,
            red,
        )
        stamp(1)
        normed = []
        for j in range_constexpr(len(toks)):
            rstd = hw_rsq(tots[j] / float(HIDDEN) + eps)
            normed.append(
                [
                    [accs[j][i][e] * rstd * (gs[i][e] + 1.0) for e in range(8)]
                    for i in range(2)
                ]
            )
        for j in range_constexpr(len(toks)):
            amax = fx.Float32(0.0)
            for e in range_constexpr(8):
                amax = fx.max(amax, fmath.absf(normed[j][0][e]))
                amax = fx.max(
                    amax,
                    (lane < 32).select(fmath.absf(normed[j][1][e]), fx.Float32(0.0)),
                )
            w_max = wave_max(amax)
            if lane == 0:
                fx.ptr_store(w_max, red + (j * WAVES + wave))
        gpu.barrier()
        x_scales = []
        for j in range_constexpr(len(toks)):
            amax = fx.ptr_load(red + j * WAVES)
            for w in range_constexpr(1, WAVES):
                amax = fx.max(amax, fx.ptr_load(red + (j * WAVES + w)))
            x_scale = (amax == 0.0).select(fx.Float32(1.0), amax / FP8_MAX)
            x_scales.append(x_scale)
            x_rcp = 1.0 / x_scale
            for i in range_constexpr(2):
                q = [div_rn(normed[j][i][e], x_scale, x_rcp) for e in range(8)]
                w01 = [
                    fp8_pack4(q[0], q[1], q[2], q[3]),
                    fp8_pack4(q[4], q[5], q[6], q[7]),
                ]
                xo = x8 + (j * (HIDDEN // 4) + lts[i] * 2)
                if (i == 0) | (lane < 32):
                    fx.ptr_store(w01[0].bitcast(fx.Float32), xo)
                    fx.ptr_store(w01[1].bitcast(fx.Float32), xo + 1)
        gpu.barrier()
        stamp(2)
        return x_scales

    # ============================================ norm tasks (S > 2)
    # (S <= 2 the GEMV tasks normalize every token themselves: S = 2 fused layer
    # 43.8 -> 43.6 us; S = 4 57.5 -> 59.4 us the other way round)
    if const_expr(tokens > 2):  # noqa: SIM102
        if (bid >= NT0) & (bid < NT0 + tokens):
            tk = bid - NT0
            raw_g, raw_ar = load_norm_inputs([tk])
            x_scale = norm_quant([tk], raw_g, raw_ar, lambda j: True)[0]
            if const_expr(signal is not None):
                publish_done(signal[1], tk)  # the residual
            # the x8 row as plain words, then its scale: the scale's pair is the flag
            for i in range_constexpr(2):
                if (i == 0) | (lane < 32):
                    bo.buffer_store(
                        fx.Vector(
                            fx.ptr_load(
                                x8 + lts[i] * 2,
                                result_type=fx.Vector.make_type(2, fx.Int32),
                            )
                        ),
                        rsrc(x8_mb),
                        tk * (HIDDEN // 4) + lts[i] * 2,
                        cache_modifier=CM_DEV,
                    )
            fx.memory_fence(
                syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Release
            )
            gpu.barrier()
            if tid == 0:
                mb_put(x8s_mb, tk, x_scale)

    # ============================================ GEMV tasks (one per CTA)
    # CTA bid < ONE_GEMV: row group bid of q | k | v, then the own index q head's
    # and index k's (the identity in the one-head build)
    own_q = (LOCAL_Q_HEADS + 2) * HEAD_ROW_GROUPS
    rg = (bid < own_q).select(
        bid,
        (bid < own_q + HEAD_ROW_GROUPS).select(
            heads.iq_off // ROWS_PER_TASK + bid - own_q,
            heads.ik_off // ROWS_PER_TASK + bid - own_q - HEAD_ROW_GROUPS,
        ),
    )
    gemv = bid < ONE_GEMV
    if const_expr(heads.count > 1):
        xt = bid - XT0
        if (xt >= 0) & (xt < OTHER_GEMV):
            other = xt // HEAD_ROW_GROUPS  # the other heads in head order, own skipped
            head = other + (other >= heads.own).select(1, 0)
            rg = (LOCAL_Q_HEADS + 2 + head) * HEAD_ROW_GROUPS + xt % HEAD_ROW_GROUPS
            gemv = long_step()
    if gemv:
        if const_expr(tokens <= 2):
            raw_g, raw_ar = load_norm_inputs(list(range(tokens)))
        # weight stream after the norm's inputs: the norm waits on its loads in issue
        # order (vmcnt), so it would otherwise wait for the 96 KB of weights too; the
        # scheduling barrier keeps the compiler from interleaving them again (nothing
        # before it uses a loaded value, so it does not wait either)
        rocdl.sched_barrier(0)
        r_w = rsrc(w_qkv)
        # 64-k chunk kc of row group rg is 1 KB at ((rg*K/32 + 2kc)*512); lane l
        # holds row l%16, k = 64kc + 16(l/16) + [0, 16).
        wts = []
        for c in range_constexpr(CHUNKS_PER_WAVE):
            kc = wave * CHUNKS_PER_WAVE + c
            wts.append(
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        ((rg * (HIDDEN // 32) + kc * 2) * 512 + lane * 16) // 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
            )
        ws = fx.Float32(  # row tid % 16's scale, sampled below by tid < 16 tokens
            bo.buffer_load(
                rsrc(s_qkv),
                rg * ROWS_PER_TASK + tid % ROWS_PER_TASK,
                vec_width=1,
                dtype=T.f32,
            )
        )
        if const_expr(tokens <= 2):
            x_scales = norm_quant(
                list(range(tokens)), raw_g, raw_ar, lambda j: bid == j
            )
            if const_expr(signal is not None):  # noqa: SIM102
                if bid < tokens:
                    publish_done(signal[1], bid)  # the residual
        else:
            # the norm tasks' scales (each published after its row's plain x8 words),
            # then every row, 16 B a load
            got = mb_poll([(x8s_mb, tk, 1) for tk in range(tokens)])
            x_scales = [got[tk][0].bitcast(fx.Float32) for tk in range(tokens)]
            fx.memory_fence(
                syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Acquire
            )
            n_v = tokens * HIDDEN // 16  # 16 B loads
            for i in range_constexpr((n_v + THREADS - 1) // THREADS):
                v = tid + THREADS * i
                if v < n_v:
                    fx.ptr_store(
                        fx.Vector(
                            bo.buffer_load(
                                rsrc(x8_mb),
                                v * 4,
                                vec_width=4,
                                dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        ),
                        x8 + v * 4,
                    )
            gpu.barrier()
            stamp(2)
        # 12 chunks per wave: two 16x16x32 FP8 MFMAs each (k halves of 8 bytes).
        # B column l % 16 is token l % 16 (token 0 past the last): C column t is
        # token t's rows, the weight stream is the same for any token count.
        col_tok = fx.min(lane % 16, tokens - 1)
        c = fx.Vector.filled(4, 0.0, fx.Float32)
        for cc in range_constexpr(CHUNKS_PER_WAVE):
            kc = wave * CHUNKS_PER_WAVE + cc
            xb = fx.Vector(
                fx.ptr_load(
                    x8 + (col_tok * (HIDDEN // 4) + kc * 16 + (lane // 16) * 4),
                    result_type=fx.Vector.make_type(4, fx.Float32),
                )
            ).bitcast(fx.Int64)
            wa = wts[cc].bitcast(fx.Int64)
            for h in range_constexpr(2):
                c = fx.Vector(
                    rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.vec(4, T.f32), [wa[h], xb[h], c, 0, 0, 0]
                    )
                )
        fx.ptr_store(c, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        # sample column t of C: lanes t, 16 + t, 32 + t, 48 + t hold rows 4 (l/16) + e
        if tid < ROWS_PER_TASK * tokens:
            r = tid % ROWS_PER_TASK
            tk = tid // ROWS_PER_TASK
            tot_r = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                tot_r = tot_r + fx.ptr_load(
                    red + ((w * 64 + 16 * (r // 4) + tk) * 4 + r % 4)
                )
            x_scale = x_scales[0]
            for k in range_constexpr(1, tokens):
                x_scale = (tk == k).select(x_scales[k], x_scale)
            mb_put(
                qkv_mb,
                tk * rows + rg * ROWS_PER_TASK + r,
                bf16_round(tot_r * x_scale * ws),
            )

        stamp(3)

    # ======================================= head tasks: norm / rope / cache
    # one CTA per head, token k on the half wave of threads 32 k .. 32 k + 31
    if (bid >= HT0) & (bid < HT0 + N_HEAD_TASKS):
        ht = bid - HT0
        tk = tid // 32
        l32 = tid % 32
        is_q = ht < LOCAL_Q_HEADS
        is_k = ht == LOCAL_Q_HEADS
        is_v = ht == LOCAL_Q_HEADS + 1
        is_iq = ht == LOCAL_Q_HEADS + 2
        base = tk * rows + is_q.select(
            ht * HEAD_DIM,
            is_k.select(
                K_OFF, is_v.select(V_OFF, is_iq.select(heads.iq_off, heads.ik_off))
            ),
        )
        if tk < tokens:
            d0 = l32 * ELEMS
            # slot_mapping is int64: this token's low word
            slot = fx.Int32(
                bo.buffer_load(rsrc(slot_mapping), 2 * tk, vec_width=1, dtype=T.i32)
            )
            # gamma and this position's cos / sin go out before the wait
            gw = is_q.select(g_q, is_k.select(g_k, is_iq.select(g_iq, g_ik)))
            gw4, cs, sn = head_rope_inputs(tk, gw, d0, cos_sin, positions)
            got = mb_poll([(qkv_mb, base + d0, 2), (qkv_mb, base + d0 + 2, 2)])
            stamp(4)
            e = [
                got[0][0].bitcast(fx.Float32),
                got[0][1].bitcast(fx.Float32),
                got[1][0].bitcast(fx.Float32),
                got[1][1].bitcast(fx.Float32),
            ]
            if is_v:
                # raw V against the cache's one scale: no amax pass over the head,
                # and nothing to store back, since the scale is not per token
                vs = kv_cache_scale(v_scale)
                if slot >= 0:
                    b, t = kv_page(slot)
                    r_v = rsrc(v_cache)
                    for j in range_constexpr(ELEMS):
                        q8 = fp8_pack4(e[j] / vs, 0.0, 0.0, 0.0) & 0xFF
                        bo.buffer_store(
                            fx.Int8(q8),
                            r_v,
                            b * PAGE_BYTES + (d0 + j) * PAGE16 + t,
                            cache_modifier=cm_out,
                        )
            else:
                out = head_norm_rope(e, lane, gw4, cs, sn, eps)
                if is_q:
                    bo.buffer_store(
                        fx.Vector.from_elements(out, fx.Float32).to(fx.BFloat16),
                        rsrc(q_out),
                        (tk * LOCAL_Q_HEADS + ht) * HEAD_DIM + d0,
                        cache_modifier=cm_out,
                    )
                if is_iq:
                    bo.buffer_store(
                        fx.Vector.from_elements(out, fx.Float32).to(fx.BFloat16),
                        rsrc(iq_out),
                        tk * HEAD_DIM + d0,
                    )
                if is_k:
                    ksc = kv_cache_scale(k_scale)
                    if slot >= 0:
                        b, t = kv_page(slot)
                        qk = [bf16_round(out[j]) / ksc for j in range(ELEMS)]
                        bo.buffer_store(
                            fp8_pack4(qk[0], qk[1], qk[2], qk[3]),
                            rsrc(k_cache),
                            (
                                b * PAGE_BYTES
                                + (d0 // 16) * (PAGE16 * 16)
                                + t * 16
                                + d0 % 16
                            )
                            // 4,
                            cache_modifier=cm_out,
                        )
                if (ht == IK_TASK) & (slot >= 0):
                    qi = [bf16_round(out[j]) for j in range(ELEMS)]
                    bo.buffer_store(
                        fp8_pack4(qi[0], qi[1], qi[2], qi[3]),
                        rsrc(index_cache),
                        (slot * HEAD_DIM + d0) // 4,
                        cache_modifier=cm_out,
                    )

        if const_expr(signal is not None):
            # every token's flag once the whole CTA's stores are out
            publish_done(signal[0], tid * N_HEAD_TASKS + ht, tid < tokens)


# the fused layer kernel's K1 pointers that do not change per step, in its device
# array (the rest are K4's own, or kernel arguments)
K1_ARGS = (
    "ar", "g_in", "w_qkv", "s_qkv", "g_q", "g_k", "g_iq", "g_ik", "cos_sin",
    "index_cache", "iq_out", "scratch",
)  # fmt: skip


@traced
def emit_k1(
    tid, bid, lane, wave, x8, red, qs, stamp, step, layer, eps, index_scale, tokens,
    q_len, init_blocks, local_blocks, heads, ptrs, bt_width, signal=False,
):  # fmt: skip
    """K1 whole: the GEMV / head / norm tasks, then every CTA's indexer scores.
    ``x8``: tokens * HIDDEN / 4 words of LDS, ``red``: WAVES * 256 floats, ``qs``:
    tokens * heads.count * HEAD_DIM / 2 floats. ``q_len``: tokens per request
    (speculative verify rows). ``heads``: the index q heads of the fused projection
    (``IndexHeads``). ``signal`` (the fused layer kernel): the head tasks and
    residual writers publish done flags (SCRATCH_HDONE / SCRATCH_RDONE)."""
    (
        ar, res, h_out, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin, positions,
        slot_mapping, k_cache, v_cache, k_scale, v_scale, index_cache, q_out, iq_out,
        block_table, seq_lens, iscore, scratch,
    ) = ptrs  # fmt: skip
    mbox = Mailbox(scratch, step, layer)
    qkv_mb = mbox.addr(SCRATCH_QKV)
    x8_mb = mbox.addr(SCRATCH_X8)
    x8s_mb = mbox.addr(SCRATCH_X8S)
    # Plain names, not method calls: flydsl's AST rewriter carries every local whose
    # method is called inside a dynamic if/for as loop state.
    mb_put = mbox.put
    mb_put_words = mbox.put_words
    mb_poll = mbox.poll

    def index_q_row(tk, head):
        """Token tk's index q of the projection's index head ``head`` into qs (row
        tk * heads.count + head, HEAD_DIM / 2 words), made from the GEMV's raw rows
        with the index-q head's math, by this wave's lanes < 32."""
        d0 = lane * ELEMS
        gw4, cs, sn = head_rope_inputs(tk, g_iq, d0, cos_sin, positions)
        base = tk * heads.rows + IndexHeads(heads.count, 0).iq_off + head * HEAD_DIM
        got = mb_poll([(qkv_mb, base + d0, 2), (qkv_mb, base + d0 + 2, 2)])
        e = [got[h][j].bitcast(fx.Float32) for h in range(2) for j in range(2)]
        out = head_norm_rope(e, lane, gw4, cs, sn, eps)
        row = qs + (tk * heads.count + head) * (HEAD_DIM // 2)
        fx.ptr_store(bf16_pair(out[0], out[1]), row + lane * 2)
        fx.ptr_store(bf16_pair(out[2], out[3]), row + (lane * 2 + 1))

    def make_index_q(first, last):
        """Tokens first..last's index q of this rank's head into qs: wave w makes
        tokens w, w + WAVES, ..; run by a CTA with score tasks only."""
        for k0 in range_constexpr(0, tokens, WAVES):
            tk = wave + k0
            if (tk < tokens) & (tk >= first) & (tk <= last) & (lane < 32):
                index_q_row(tk, heads.own)
        gpu.barrier()

    def make_index_q_heads():
        """Every token's index q of every head into qs (indexer context
        parallelism): (token, head) job j on wave j % WAVES."""
        for j0 in range_constexpr(0, tokens * heads.count, WAVES):
            j = wave + j0
            if (j < tokens * heads.count) & (lane < 32):
                index_q_row(j // heads.count, j % heads.count)
        gpu.barrier()

    def q_frag_head(tk, s, head):
        """This lane's MFMA B operand of token tk's index q of head ``head``,
        k-chunk s."""
        row = (tk * heads.count + head) * (HEAD_DIM // 2)
        return fx.Vector(
            fx.ptr_load(
                qs + (row + (32 * s + 8 * (lane // 16)) // 2),
                result_type=fx.Vector.make_type(4, fx.Float32),
            )
        ).bitcast(fx.BFloat16)

    def q_frag(tk, s):
        """This lane's MFMA B operand of token tk's index q (this rank's head),
        k-chunk s."""
        return q_frag_head(tk, s, heads.own)

    def long_step():
        """A request of the step is long (``step_rows``): it scores every index q
        head (indexer context parallelism)."""
        return step_rows(seq_lens, tokens, q_len, heads).any_long()

    emit_pre_attn(
        tid, bid, lane, wave, x8, red, mb_put, mb_put_words, mb_poll, qkv_mb, x8_mb,
        x8s_mb, stamp, eps, tokens, heads,
        (
            ar, res, h_out, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin,
            positions, slot_mapping, k_cache, v_cache, k_scale, v_scale, index_cache,
            q_out, iq_out,
        ),
        (mbox.addr(SCRATCH_HDONE), mbox.addr(SCRATCH_RDONE)) if signal else None,
        long_step,
    )  # fmt: skip
    stamp(5)
    # the scorers: the idle CTAs first (the norm tasks' come first but finish
    # early), then the GEMV CTAs, the head CTAs last
    score_base = ONE_GEMV + N_HEAD_TASKS
    hdone_mb = mbox.addr(SCRATCH_HDONE)

    def wait_new_keys():
        """Every token's index_k head task done, its key visible to this wave."""
        mb_poll([(hdone_mb, fx.min(lane, tokens - 1) * N_HEAD_TASKS + IK_TASK, 1)])
        fx.memory_fence(
            syncscope=rocdl.SyncScope.Agent, ordering=fx.AtomicOrdering.Acquire
        )

    emit_index_scores(
        (bid + (BLOCKS - score_base)) % BLOCKS, lane, uniform(wave), red, mb_put,
        iscore, q_frag, tokens, q_len, init_blocks, local_blocks, index_scale,
        index_cache, block_table, seq_lens, bt_width, wait_new_keys if signal else None,
        make_index_q, heads, q_frag_head, make_index_q_heads,
    )  # fmt: skip
    stamp(6)


TL_POINTS = 8  # timeline stamps per CTA


def build_pre_attn_kernel(
    eps: float,
    sm_scale: float,
    init_blocks: int,
    local_blocks: int,
    tokens: int = 1,
    timeline: bool = False,
    heads: IndexHeads = ONE_INDEX_HEAD,
):
    """``@flyc.jit`` launcher of K1 for this model's RMSNorm epsilon, attention
    scale and sparse pinned blocks, a decode batch of ``tokens`` (<= MAX_TOKENS)
    rows and the fused projection's index q ``heads``."""
    assert 1 <= tokens <= MAX_TOKENS
    index_scale = index_scale_log2e(sm_scale)

    @fx.struct
    class Smem:
        # FP8 activations, natural k order, one row per token
        x8: fx.Array[fx.Int32, tokens * HIDDEN // 4, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS if timeline else 1, 16]  # timeline stamps
        # every token's index q for the scorers, bf16 pairs
        qs: fx.Array[fx.Float32, HEAD_DIM // 2 * tokens * heads.count, 16]

    kernel_name = kernel_symbol(
        "minimax_m3_pre_attn",
        s=tokens,
        ib=init_blocks,
        lb=local_blocks,
        ih=heads.count,
        io=heads.own,
        tl=timeline,
    )

    # the JIT cache key holds the kernel's scalar closure values, not objects: the
    # index heads go in as ints (a shared cache otherwise serves one rank's build
    # to every rank)
    ih_count, ih_own = heads.count, heads.own

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def pre_attn_kernel(
        ar: Int64,
        res: Int64,
        h_out: Int64,
        g_in: Int64,
        w_qkv: Int64,
        s_qkv: Int64,
        g_q: Int64,
        g_k: Int64,
        g_iq: Int64,
        g_ik: Int64,
        cos_sin: Int64,
        positions: Int64,
        slot_mapping: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        index_cache: Int64,
        q_out: Int64,
        iq_out: Int64,
        block_table: Int64,
        seq_lens: Int64,
        bt_width: Int32,
        q_len: Int32,
        iscore: Int64,
        scratch: Int64,
        step: Int64,
        layer: Int32,
        tl: Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        lds = fx.SharedAllocator().allocate(Smem).peek()
        x8 = lds.x8.ptr
        red = lds.red.ptr
        tls = lds.tls.ptr
        qs = lds.qs.ptr

        def stamp(k):
            """``timeline``: s_memrealtime (100 MHz) of this CTA passing point k, kept
            in LDS until the kernel ends."""
            # compile-time gate outside, traced condition inside: they cannot be
            # one `and`
            if const_expr(timeline):  # noqa: SIM102
                if tid == 0:
                    fx.ptr_store(memrealtime(), tls + k)

        if const_expr(timeline):  # noqa: SIM102
            if tid < TL_POINTS:  # same wave as the stamping thread: ordered
                fx.ptr_store(fx.Int64(0), tls + tid)
        stamp(0)
        emit_k1(
            tid, bid, lane, wave, x8, red, qs, stamp, step, layer, eps, index_scale,
            tokens, q_len, init_blocks, local_blocks, IndexHeads(ih_count, ih_own),
            (
                ar, res, h_out, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin,
                positions, slot_mapping, k_cache, v_cache, k_scale, v_scale,
                index_cache, q_out, iq_out, block_table, seq_lens, iscore, scratch,
            ),
            bt_width,
        )  # fmt: skip
        if const_expr(timeline):  # noqa: SIM102
            if tid < TL_POINTS:
                fx.generic_store(
                    fx.inttoptr(
                        fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                        tl + fx.Int64((bid * TL_POINTS + tid) * 8),
                    ),
                    fx.ptr_load(tls + tid),
                )

    @flyc.jit
    def launch_pre_attn(
        ar: Int64,
        res: Int64,
        h_out: Int64,
        g_in: Int64,
        w_qkv: Int64,
        s_qkv: Int64,
        g_q: Int64,
        g_k: Int64,
        g_iq: Int64,
        g_ik: Int64,
        cos_sin: Int64,
        positions: Int64,
        slot_mapping: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        index_cache: Int64,
        q_out: Int64,
        iq_out: Int64,
        block_table: Int64,
        seq_lens: Int64,
        bt_width: Int32,
        q_len: Int32,
        iscore: Int64,
        scratch: Int64,
        step: Int64,
        layer: Int32,
        tl: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        pre_attn_kernel(
            ar,
            res,
            h_out,
            g_in,
            w_qkv,
            s_qkv,
            g_q,
            g_k,
            g_iq,
            g_ik,
            cos_sin,
            positions,
            slot_mapping,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            index_cache,
            q_out,
            iq_out,
            block_table,
            seq_lens,
            bt_width,
            q_len,
            iscore,
            scratch,
            step,
            layer,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch_pre_attn
