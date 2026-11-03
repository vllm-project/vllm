# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: B008, E501, SIM102
"""DSpark draft: the 5 draft layers and the final norm of a draft step as one persistent launch.

One rank of a TP8 group, 256 CTAs x 512 threads, S = 8 / 16 rows: 1 / 2 requests of 7 query rows
(vLLM's rows 7 r + j; rows 7 NR .. S - 1 are padding: computed, never stored to vLLM's tensors).
A draft layer is K3's attention block (k3.py) followed by a dense SwiGLU MLP:

    IN     x = embeds -> the fp32 stream (smid), the input of the first boundary
    BND    (FFN boundary of layer l - 1; the embeddings for l = 0) the owner of column block b
           (rank b // 32) sums the W down_proj partials in rank order (bf16, the all-reduce output),
           post-FFN norm (weight first, eps 1e-8), gated residual, the pre-QKV norm of layer l
           (eps 1e-5) of the bf16-rounded stream; the reduced rows and the norm partials all-gathered
    QKV    layer l's qkv + o-gate GEMV (K in eighths, as K3); the owners finish whole heads: qk-norm,
           RoPE, q, the bf16 K / V into the layer's LBHNC draft KV cache (slot_mapping), the o-gate
    ATT    after every CTA's QKV flag: the causal sliding-window attention over the draft KV
           (attn_frag.py, 7 query rows a request)
    O      o-norm x sigmoid(gate), o_proj (this rank's K = 1024), partial to the block owner rank
    AB     the attention boundary (as BND: post-attention norm weightless eps 1e-8, gated residual,
           pre-FFN norm) -> ffn_in
    GU     gate_up (2 x 2048 rows a rank, K in eighths over 8-CTA groups; the partials go to the
           token owner CTA), bf16 gate / up, SwiGLU -> h (bf16, 2048 a rank)
    DN     down_proj rows 32 b .. (K = 2048), partial to the block owner rank
    after layer 4: BND with the final norm -> the draft hidden states (bf16, rows < 7 NR)

Every norm takes the bf16-rounded stream like the stock layer; its rstd comes from one token
reduction round of the unrounded stream (k3.tok_rstd2's two-norm identity). The norm multipliers
are fp32 (weight + gain center). Peer regions are double buffered by tag parity; the tags of a
launch are epoch * 64 + 1 .. 6 (6 phases, so consecutive phases alternate parity across launches),
and CTA 0 bumps the epoch at the end of the launch.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from . import attn_frag
from .dev import (
    BLOCKS,
    CM_DEV,
    CM_SYS,
    THREADS,
    WAVES,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fresh,
    hw_exp2,
    hw_rsq,
    ld_f32,
    ld_i32,
    ld_i32_if,
    ld_vec,
    ld_vec_if,
    mb_poll,
    mb_put,
    mb_put_sys,
    mfma_bf16,
    rsrc,
    st_if,
    traced,
    uniform,
)
from .k3 import _lay, bf_tile_load_mma, o_gate, sel_peer

HID = 8192
W = 8
HQ, HKV, HD = 128 // W, 16 // W, 64  # q / kv heads a rank
QROWS, KROWS = HQ * HD, HKV * HD
QKV_ROWS = QROWS + 2 * KROWS  # qkvg rows a rank: q 1024 | k 128 | v 128 | gate 1024
QKVG_ROWS = QKV_ROWS + QROWS
OK_K = HID // W  # o_proj K a rank
FF = 16384 // W  # MLP intermediate a rank
LAYERS = 5
NQ = 7  # query rows a request
PRE_EPS = 1e-5
POST_EPS = 1e-8
KCH = 128
XPAD = 8
XQ8 = HID // 8  # K eighth of the qkv / gate_up GEMVs
NQTILES = QKVG_ROWS // 16
NQT = -(-NQTILES // 32)  # qkvg tiles a CTA group (32 groups x 8 K eighths)
NH = HQ + 2 * HKV  # head groups
NQKVT = QKV_ROWS // 16
GUT = 8  # gate_up tiles a CTA group: 4 gate + the 4 matching up tiles
CPRX = BLOCKS // W  # column blocks a rank owns (owner of block b = b // CPRX)
HEALTH_WORDS = 8 + 8 * BLOCKS
# The launch table: TAB_WORDS int64 a row. Row l (0..5), the boundary into layer l (row 5: the
# final norm): alpha, beta, post-FFN norm weight of layer l - 1 (row 0: ones, zeros, ones) and
# layer l's pre-QKV norm weight (row 5: the final norm's). Row l (0..4), layer l: qkv_proj,
# o_proj, add_attn alpha / beta, pre-FFN norm weight, gate_up_proj, down_proj. Norm weights fp32.
# Row T_G: the step-invariant buffers: q, o-gate (bf16 [S][1024]), the RoPE cos / sin cache (fp32),
# then the W peer buffers.
TAB_WORDS = 16
TAB_ROWS = LAYERS + 2
T_BND = 0
T_LAYER = 4
T_G = LAYERS + 1
G_Q = 0
G_PEER = 4
REGION_IDS = {
    "xf": 16,
    "xnf": 17,
    "qx": 18,
    "ssa4": 20,
    "ssg4": 21,
    "orr": 24,
    "ssa4p": 26,
    "op": 13,
    "qf": 50,
    "gx": 51,
    "hf": 52,
    "dp": 53,
    "yrr": 54,
    "ssg4p": 55,
}


def sfresh(v):
    """An opaque s_mov of a wave-uniform i32 (dev.fresh for SGPRs)."""
    return fx.Int32(
        llvm.inline_asm(
            T.i32, [fx.Int32(v).ir_value()], "s_mov_b32 $0, $1", "=s,s",
            has_side_effects=True
        )
    )  # fmt: skip


def sfresh64(v):
    return fx.Int64(
        llvm.inline_asm(
            T.i64, [fx.Int64(v).ir_value()], "s_mov_b64 $0, $1", "=s,s",
            has_side_effects=True
        )
    )  # fmt: skip


@dataclass(frozen=True)
class Build:
    S: int = 8  # rows: 8 (1 request) or 16 (2 requests)


def scratch_layout(S, world=W):
    """This rank's scratch regions."""
    assert world == W, world
    return _lay(
        [
            ("stream", S * HID * 4),  # the stream into a layer (fp32 plain)
            (
                "smid",
                S * HID * 4,
            ),  # the stream after the attention residual (fp32 plain)
            ("xin", S * HID * 2),  # ffn_in (bf16 plain), CTA b's 32 columns each
            ("xf", BLOCKS * 8),  # its flags
            ("xn", S * HID * 2),  # the layer's attention input (bf16 plain)
            ("xnf", BLOCKS * 8),
            # boundary norm rounds: [t][CTA][4] partials (unused: they come from the peers),
            # then (r1, r2)[t]
            ("ssa4", (4 * S * BLOCKS + 2 * S) * 8),
            ("ssg4", (4 * S * BLOCKS + 2 * S) * 8),
            # qkvg K-eighth partials [group][src][tile row][t]
            ("qx", 32 * 8 * NQT * 16 * S * 8),
            ("qf", BLOCKS * 8),  # QKV done flags
            # gate_up K-eighth partials [group][src][tile][row][t]
            ("gx", 32 * 8 * GUT * 16 * S * 8),
            ("h", S * FF * 2),  # SwiGLU output (bf16 plain)
            ("hf", BLOCKS * 8),
        ]
    )


def peer_layout(S, world=W):
    """Every rank's peer buffer (IPC-shared), [parity] first."""
    assert world == W, world
    return _lay(
        [
            # o_proj / down_proj partials [par][src][t][pair] tagged
            ("op", 2 * W * S * (HID // 2) * 8),
            ("dp", 2 * W * S * (HID // 2) * 8),
            # the owners' reduced rows [par][t][pair] tagged (all-gather)
            ("orr", 2 * S * (HID // 2) * 8),
            ("yrr", 2 * S * (HID // 2) * 8),
            # the owners' norm partials [par][t][CTA][4] tagged (all-gather)
            ("ssa4p", 2 * 4 * S * BLOCKS * 8),
            ("ssg4p", 2 * 4 * S * BLOCKS * 8),
        ]
    )


_BUILDS = {}


def build(key: Build):
    if key in _BUILDS:
        return _BUILDS[key]
    # every knob in the disk cache key (see k3.build)
    _KEYSTR = repr(key)
    S = key.S
    assert S in (8, 16), S
    NR = S // 8  # requests
    R = NQ * NR  # real rows
    SO = S // 8  # tokens a QKV / gate_up owner CTA
    lay, _ = scratch_layout(S)
    play, _ = peer_layout(S)
    YROW = OK_K + XPAD
    CO = OK_K // KCH // WAVES  # o_proj chunks a wave
    CD = FF // KCH // WAVES  # down_proj chunks a wave
    assert CO >= 1 and CD >= 1 and NQT == 5 and NH <= 32
    FA = attn_frag.make(NR, True, nq=NQ, qcm=CM_DEV, kvcm=CM_DEV)
    FaLds = FA.Lds
    _KEYSTR = _KEYSTR + FA.key
    name = f"muse_dspark_s{S}"
    tag_par = lambda tag: tag & 1

    @fx.struct
    class Misc:
        rs: fx.Array[fx.Float32, 2 * S + 32, 16]

    @fx.struct
    class StageO:
        y: fx.Array[fx.Int32, S * YROW // 2, 16]
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]

    @fx.struct
    class StageQ:
        red: fx.Array[fx.Float32, WAVES * NQT * 64 * 4, 16]
        own: fx.Array[fx.Float32, NQT * 16 * SO, 16]

    @fx.struct
    class StageU:
        red: fx.Array[fx.Float32, WAVES * GUT * 64 * 4, 16]
        own: fx.Array[fx.Float32, GUT * 16 * SO, 16]

    @fx.struct
    class StageD:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]

    @fx.union
    class StageLds:
        o_: StageO
        q_: StageQ
        u_: StageU
        d_: StageD
        fa_: FaLds

    def reg(base, region):
        return base + fx.Int64(lay[region][0])

    def preg(base, region):
        return base + fx.Int64(play[region][0])

    def p64(words, i):
        lo = uniform(words[2 * i])
        hi = uniform(words[2 * i + 1])
        return (fx.Int64(hi) << fx.Int64(32)) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))

    def tab_ptrs(wtab, row, first, n):
        """Words first .. first + n (n even) of launch-table row ``row`` (wave-uniform)."""
        words = []
        for i in range_constexpr(n // 2):
            v = ld_vec(wtab, row * (2 * TAB_WORDS) + 2 * first + 4 * i, 4)
            words += [v[e] for e in range(4)]
        return [p64(words, j) for j in range(n)]

    # ------------------------------------------------------------ token norm round (k3.tok_rstd2)
    @traced
    def tok_rstd2(scratch, region, tag, health, tid, lane, rs_out, bid, red, pbase):
        """One round for two norms of a token from the owners' partials at pbase
        ((t * BLOCKS + CTA) * 4 + j): P1 = sum o^2, P2 = sum (a s)^2, P3 = sum (a s)(b o),
        P4 = sum (b o)^2 for s' = a s + b (o r1). Reducer CTA 255 - t sums the 256 CTAs in a
        fixed order: r1 = rsqrt(P1 / HID + 1e-8), r2 = rsqrt((P2 + 2 r1 P3 + r1^2 P4) / HID + 1e-5).
        rs_out[t] = r1, rs_out[S + t] = r2 on every CTA."""
        rbase = reg(scratch, region)
        tred = bid >= BLOCKS - S
        if tred:
            t = BLOCKS - 1 - bid
            va = fx.Float32(0.0)
            vb = fx.Float32(0.0)
            if tid < 2 * BLOCKS:
                got = mb_poll(
                    tag,
                    health,
                    [(pbase, (t * BLOCKS + tid // 2) * 4 + 2 * (tid % 2), 2)],
                    REGION_IDS[region],
                    cm=CM_SYS,
                )
                va = got[0][0].bitcast(fx.Float32)
                vb = got[0][1].bitcast(fx.Float32)
            va = butterfly(va, (32, 16, 8, 4, 2))  # lanes of one parity: 32 CTAs
            vb = butterfly(vb, (32, 16, 8, 4, 2))
            if lane < 2:
                fx.ptr_store(va, red + (tid // 64) * 4 + 2 * lane)
                fx.ptr_store(vb, red + (tid // 64) * 4 + 2 * lane + 1)
            gpu.barrier()
            if tid == 0:
                p = [fx.Float32(0.0) for _ in range(4)]
                for w in range_constexpr(WAVES):
                    for j in range_constexpr(4):
                        p[j] = p[j] + fx.ptr_load(red + w * 4 + j)
                r1 = hw_rsq(p[0] * fx.Float32(1.0 / HID) + fx.Float32(POST_EPS))
                q = p[1] + fx.Float32(2.0) * r1 * p[2] + r1 * r1 * p[3]
                r2 = hw_rsq(q * fx.Float32(1.0 / HID) + fx.Float32(PRE_EPS))
                mb_put(tag, rbase, 4 * S * BLOCKS + 2 * t, r1)
                mb_put(tag, rbase, 4 * S * BLOCKS + 2 * t + 1, r2)
        if tid < S:
            got2 = mb_poll(
                tag, health, [(rbase, 4 * S * BLOCKS + 2 * tid, 2)], REGION_IDS[region]
            )
            fx.ptr_store(got2[0][0].bitcast(fx.Float32), rs_out + tid)
            fx.ptr_store(got2[0][1].bitcast(fx.Float32), rs_out + (S + tid))
        gpu.barrier()

    # ------------------------------------------------------------ IN: the embeddings
    @traced
    def stage_in(x_in, scratch, tid, bid):
        """CTA b's 32 columns of the embeddings (rows < R; padding rows 0) -> smid (fp32)."""
        tid = fresh(tid)
        pp = tid % 16
        tt = fx.min(tid // 16, S - 1)
        live = tid < 16 * S
        c = bid * 32 + 2 * pp
        xw = ld_i32_if(live & ((tid // 16) < R), x_in, (tt * HID + c) // 2, R * HID * 2)
        if live:
            bo.buffer_store(
                fx.Vector.from_elements(
                    [bf_lo(xw).bitcast(fx.Int32), bf_hi(xw).bitcast(fx.Int32)], fx.Int32
                ),
                rsrc(reg(scratch, "smid")),
                tt * HID + c,
                cache_modifier=CM_DEV,
            )

    # ------------------------------------------------------------ BND / AB: a residual boundary
    @traced
    def boundary(
        peer,
        pl,
        scratch,
        src,
        rr,
        p4,
        rnd,
        flag,
        st_in,
        st_out,
        x_out,
        alpha,
        beta,
        w_post,
        w_next,
        tag,
        health,
        tid,
        bid,
        lane,
        rank,
        rsl,
        ffn,
        poll_ok=None,
        final=False,
    ):
        """CTA b's hidden columns 32 b ..: the block owner sums the W partials (``src``) in rank
        order (fp32 -> bf16: the all-reduce output y) and all-gathers y (``rr``) and its norm
        partials (``p4``); every CTA: the post-norm (ffn: rms(w_post y), eps 1e-8; else rms(y)),
        the gated residual on the fp32 stream (st_in -> st_out), the next norm of the bf16 stream
        (w_next, eps 1e-5) -> x_out (bf16; final: vLLM's output rows < R, else scratch + flag)."""
        tid = fresh(tid)
        par = tag_par(tag)
        pp = tid % 16
        tt = fx.min(tid // 16, S - 1)
        live = tid < 16 * S
        c = bid * 32 + 2 * pp
        st = ld_vec(st_in, tt * HID + c, 2, CM_DEV)
        al = ld_vec(alpha, c, 2)
        be = ld_vec(beta, c, 2)
        wn = ld_vec(w_next, c, 2)
        wp = None
        if const_expr(ffn):
            wp = ld_vec(w_post, c, 2)
        a0 = al[0].bitcast(fx.Float32) * st[0].bitcast(fx.Float32)
        a1 = al[1].bitcast(fx.Float32) * st[1].bitcast(fx.Float32)
        # the block owner: W partials in rank order -> bf16 -> every rank
        if live & ((bid // CPRX) == rank):
            specs = [
                (
                    preg(peer, src),
                    ((par * W + s_) * S + tt) * (HID // 2) + bid * 16 + pp,
                    1,
                )
                for s_ in range(W)
            ]
            oks = None if poll_ok is None else [poll_ok] * W
            got = mb_poll(tag, health, specs, REGION_IDS[src], oks=oks, cm=CM_SYS)
            y0 = bf_lo(got[0][0])
            y1 = bf_hi(got[0][0])
            for s_ in range_constexpr(1, W):
                y0 = y0 + bf_lo(got[s_][0])
                y1 = y1 + bf_hi(got[s_][0])
            wr = bf16_pair(y0, y1)
            oo0 = bf_lo(wr)
            oo1 = bf_hi(wr)
            if const_expr(ffn):
                oo0 = wp[0].bitcast(fx.Float32) * oo0
                oo1 = wp[1].bitcast(fx.Float32) * oo1
            pb0 = be[0].bitcast(fx.Float32) * oo0
            pb1 = be[1].bitcast(fx.Float32) * oo1
            q1 = butterfly(oo0 * oo0 + oo1 * oo1, (1, 2, 4, 8))
            q2 = butterfly(a0 * a0 + a1 * a1, (1, 2, 4, 8))
            q3 = butterfly(a0 * pb0 + a1 * pb1, (1, 2, 4, 8))
            q4 = butterfly(pb0 * pb0 + pb1 * pb1, (1, 2, 4, 8))
            if pp == 0:
                i0 = ((par * S + tt) * BLOCKS + bid) * 4
                for pb in pl:
                    base = preg(pb, p4)
                    bo.buffer_store(
                        fx.Vector.from_elements(
                            [q1.bitcast(fx.Int32), tag, q2.bitcast(fx.Int32), tag],
                            fx.Int32,
                        ),
                        rsrc(base),
                        i0 * 2,
                        cache_modifier=CM_SYS,
                    )
                    bo.buffer_store(
                        fx.Vector.from_elements(
                            [q3.bitcast(fx.Int32), tag, q4.bitcast(fx.Int32), tag],
                            fx.Int32,
                        ),
                        rsrc(base),
                        (i0 + 2) * 2,
                        cache_modifier=CM_SYS,
                    )
            for pb in pl:
                mb_put_sys(
                    tag,
                    preg(pb, rr),
                    (par * S + tt) * (HID // 2) + bid * 16 + pp,
                    wr,
                )
        o0 = fx.Float32(0.0)
        o1 = fx.Float32(0.0)
        if live:
            got = mb_poll(
                tag,
                health,
                [(preg(peer, rr), (par * S + tt) * (HID // 2) + bid * 16 + pp, 1)],
                REGION_IDS[rr],
                cm=CM_SYS,
            )
            o0 = bf_lo(got[0][0])
            o1 = bf_hi(got[0][0])
            if const_expr(ffn):
                o0 = wp[0].bitcast(fx.Float32) * o0
                o1 = wp[1].bitcast(fx.Float32) * o1
        tok_rstd2(
            scratch,
            rnd,
            tag,
            health,
            tid,
            lane,
            rsl,
            bid,
            rsl + 2 * S,
            preg(peer, p4) + fx.Int64(par * S * BLOCKS * 4 * 8),
        )
        r1 = fx.ptr_load(rsl + tt)
        n0 = bf16_round(o0 * r1)
        n1 = bf16_round(o1 * r1)
        s0 = a0 + be[0].bitcast(fx.Float32) * n0
        s1 = a1 + be[1].bitcast(fx.Float32) * n1
        if live:
            bo.buffer_store(
                fx.Vector.from_elements(
                    [s0.bitcast(fx.Int32), s1.bitcast(fx.Int32)], fx.Int32
                ),
                rsrc(st_out),
                tt * HID + c,
                cache_modifier=CM_DEV,
            )
        r2 = fx.ptr_load(rsl + (S + tt))
        xw = bf16_pair(
            (bf16_round(s0) * r2) * wn[0].bitcast(fx.Float32),
            (bf16_round(s1) * r2) * wn[1].bitcast(fx.Float32),
        )
        if const_expr(final):
            st_if(live & ((tid // 16) < R), xw, x_out, (tt * HID + c) // 2)
        else:
            if live:
                bo.buffer_store(
                    xw, rsrc(x_out), (tt * HID + c) // 2, cache_modifier=CM_DEV
                )
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if const_expr(not final):
            if tid == 0:
                mb_put(tag, reg(scratch, flag), bid, fx.Int32(1))

    # ------------------------------------------------------------ QKV: the layer's qkv + gate
    # Head-aligned tile map (k3 stage_qkv8): group gq < NH holds the 4 tiles of head gq (q heads,
    # then k, then v) + gate tile NQKVT + gq; groups NH .. hold the other gate tiles. After the
    # K-eighth partial sums, CTA k8 of the group owns the tokens t = k8 (mod 8) for all its rows.
    def q8_tile(gq, n):
        head = gq < NH
        if const_expr(n < 4):
            return head.select(4 * gq + n, NQKVT + NH + NQT * (gq - NH) + n)
        return head.select(NQKVT + gq, NQKVT + NH + NQT * (gq - NH) + n)

    def q8_loads(w_qkvg, bid, lane, wave):
        k8 = bid % 8
        gq = bid // 8
        out = []
        for n in range_constexpr(NQT):
            tl = q8_tile(gq, n)
            row16 = fx.min(tl, NQTILES - 1) * 16
            out.append(
                bf_tile_load_mma(w_qkvg, row16, HID, k8 * XQ8 + wave * KCH, lane)
            )
        return out

    @traced
    def finish(w2, hh, tt, p, q_out, kvc, kvs, slots, cos_sin, pos):
        """Head hh (uniform), token tt (uniform in the wave), lane p = pair (2p, 2p + 1): as
        vLLM's fused qk-norm / RoPE / KV insert (k3 vllm_finish), the KV through the draft
        group's slot_mapping (rows >= R and slot < 0 skipped), stores at device scope."""
        x0 = bf_lo(w2)
        x1 = bf_hi(w2)
        ss = butterfly(x0 * x0 + x1 * x1, (1, 2, 4, 8, 16))
        rs = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(PRE_EPS))
        isv = hh >= HQ + HKV
        isq = hh < HQ
        n0 = x0 * rs
        n1 = x1 * rs
        tr = fx.min(tt, R - 1)
        ps = ld_i32(pos, tr * 2)  # int64 positions: low word
        cs = ld_f32(cos_sin, ps * HD + p)
        sn = ld_f32(cos_sin, ps * HD + HD // 2 + p)
        r0 = n0 * cs - n1 * sn
        r1 = n0 * sn + n1 * cs
        o0 = isv.select(x0, r0)
        o1 = isv.select(x1, r1)
        row0 = (hh * HD) // 2 + p
        if isq:
            bo.buffer_store(
                bf16_pair(o0, o1),
                rsrc(q_out),
                (tt * QROWS) // 2 + row0,
                cache_modifier=CM_DEV,
            )
        else:
            kv = hh - HQ - isv.select(fx.Int32(HKV), fx.Int32(0))
            slot = uniform(ld_i32(slots, tr * 2))  # int64 slot_mapping: low word
            slot = (tt < R).select(slot, fx.Int32(-1))
            if slot >= 0:
                eoff = fx.Int64(slot >> 4) * kvs + fx.Int64(
                    (kv * 16 + (slot & 15)) * 128
                    + isv.select(fx.Int32(HD), fx.Int32(0))
                )
                bo.buffer_store(
                    bf16_pair(o0, o1),
                    rsrc(kvc + eoff * fx.Int64(2)),
                    p,
                    cache_modifier=CM_DEV,
                )

    @traced
    def stage_qkv(
        q_ops,
        scratch,
        q_out,
        g_out,
        kvc,
        kvs,
        slots,
        cos_sin,
        pos,
        tag,
        health,
        tid,
        bid,
        lane,
        wave,
        red,
        own,
    ):
        tid = fresh(tid)
        lane = fresh(lane)
        g = lane // 16
        k8 = bid % 8
        gq = bid // 8
        if tid < 32:  # the boundary into the layer: phase layer - 1 (tag - 1)
            mb_poll(
                tag - 1,
                health,
                [(reg(scratch, "xnf"), k8 * 32 + tid, 1)],
                REGION_IDS["xnf"],
            )
        gpu.barrier()
        t = lane % 16
        xb = []
        for qq in range_constexpr(4):
            kk = wave * KCH + 8 * g + 32 * qq
            xb.append(
                ld_vec_if(
                    t < S,
                    reg(scratch, "xn"),
                    (t * HID + k8 * XQ8 + kk) // 2,
                    4,
                    S * HID * 2,
                    CM_DEV,
                )
            )
        rocdl.sched_barrier(0)
        for n in range_constexpr(NQT):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for qq in range_constexpr(4):
                acc = mfma_bf16(
                    q_ops[n][qq].bitcast(fx.BFloat16), xb[qq].bitcast(fx.BFloat16), acc
                )
            fx.ptr_store(acc, red + ((wave * NQT + n) * 64 + lane) * 4)
        gpu.barrier()
        for it_ in range_constexpr((NQT * 8 * S + THREADS - 1) // THREADS):
            e = tid + THREADS * it_
            if e < NQT * 8 * S:
                n = e // (8 * S)
                rp = (e // S) % 8
                tt = e % S
                v = []
                for h2 in range_constexpr(2):
                    rr = 2 * rp + h2
                    s_ = fx.Float32(0.0)
                    for w in range_constexpr(WAVES):
                        s_ = s_ + fx.ptr_load(
                            red
                            + (((w * NQT + n) * 64 + 16 * (rr // 4) + tt) * 4 + rr % 4)
                        )
                    v.append(s_)
                if (tt % 8) == k8:
                    for h2 in range_constexpr(2):
                        fx.ptr_store(
                            v[h2], own + ((n * 16 + 2 * rp + h2) * SO + tt // 8)
                        )
                else:
                    for h2 in range_constexpr(2):
                        mb_put(
                            tag,
                            reg(scratch, "qx"),
                            (((gq * 8 + k8) * NQT + n) * 16 + 2 * rp + h2) * S + tt,
                            v[h2],
                        )
        gpu.barrier()
        # owner wave j: token tt = k8 + 8 j; lane p < 40 = (tile n = p / 8, row pair p % 8)
        if (tid < 64 * SO) & (lane < NQT * 8):
            j = tid // 64
            p = lane
            n = p // 8
            rp = p % 8
            tt = k8 + 8 * j
            specs, oks = [], []
            for src in range_constexpr(8):
                for h2 in range_constexpr(2):
                    specs.append(
                        (
                            reg(scratch, "qx"),
                            (((gq * 8 + src) * NQT + n) * 16 + 2 * rp + h2) * S + tt,
                            1,
                        )
                    )
                    oks.append(fx.Int32(src) != k8)
            got = mb_poll(tag, health, specs, REGION_IDS["qx"], oks=oks)
            tot = []
            for h2 in range_constexpr(2):
                s_ = fx.Float32(0.0)
                mine = fx.ptr_load(own + ((n * 16 + 2 * rp + h2) * SO + j))
                for src in range_constexpr(8):
                    s_ = s_ + (fx.Int32(src) == k8).select(
                        mine, got[2 * src + h2][0].bitcast(fx.Float32)
                    )
                tot.append(s_)
            head = gq < NH
            tl = head.select(4 * gq + n, NQKVT + NH + NQT * (gq - NH) + n)
            tl = (head & (n == 4)).select(NQKVT + gq, tl)
            row = tl * 16 + 2 * rp
            w2 = bf16_pair(tot[0], tot[1])
            st_if(
                (row >= QKV_ROWS) & (row < QKVG_ROWS),
                w2,
                g_out,
                (tt * QROWS + row - QKV_ROWS) // 2,
                CM_DEV,
            )
            if head & (p < 32):
                finish(w2, gq, tt, p, q_out, kvc, kvs, slots, cos_sin, pos)
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if tid == 0:
            mb_put(tag, reg(scratch, "qf"), bid, fx.Int32(1))

    @traced
    def wait_all(scratch, region, tag, health, tid):
        """Every CTA's flag in ``region`` (a grid-wide wait)."""
        tid = fresh(tid)
        if tid < BLOCKS:
            mb_poll(tag, health, [(reg(scratch, region), tid, 1)], REGION_IDS[region])
        gpu.barrier()

    # ------------------------------------------------------------ O: o-norm x gate, o_proj, push
    def o_loads(w_o, bid, lane, wave):
        ops = []
        for i in range_constexpr(2):
            for ci in range_constexpr(CO):
                c = wave + WAVES * ci
                ops.append(
                    bf_tile_load_mma(w_o, bid * 32 + 16 * i, OK_K, c * KCH, lane)
                )
        return ops

    @traced
    def stage_o(
        o_ops, ao, attn_g, pl, tag, health, tid, bid, lane, wave, rank, yl, red
    ):
        """k3 stage_o on the fragment's merged rows (tagged pairs at ``ao``); padding rows take
        row R - 1."""
        tid = fresh(tid)
        lane = fresh(lane)
        par = tag_par(tag)
        g = lane // 16
        npc = S * OK_K // 8
        for i in range_constexpr((npc + THREADS - 1) // THREADS):
            e = tid + THREADS * i
            ts = fx.min(e // (OK_K // 8), S - 1)
            t = fx.min(ts, R - 1)
            kk = (e % (OK_K // 8)) * 8
            got = mb_poll(
                tag,
                health,
                [(ao, (t * OK_K + kk) // 2, 2), (ao, (t * OK_K + kk) // 2 + 2, 2)],
                43,
            )
            xv = got[0] + got[1]
            gv = ld_vec(attn_g, (t * QROWS + kk) // 2, 4, CM_DEV)
            ow = o_gate(xv, gv)
            if e < npc:
                fx.ptr_store(
                    fx.Vector.from_elements(ow, fx.Int32), yl + (ts * YROW + kk) // 2
                )
        gpu.barrier()
        t = fx.min(lane % 16, S - 1)
        accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
        for i in range_constexpr(2):
            for ci in range_constexpr(CO):
                c = wave + WAVES * ci
                for qq in range_constexpr(4):
                    kk = c * KCH + 8 * g + 32 * qq
                    b = fx.Vector(
                        fx.ptr_load(
                            yl + (t * YROW + kk) // 2,
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                    ).bitcast(fx.BFloat16)
                    accs[i] = mfma_bf16(
                        o_ops[i * CO + ci][qq].bitcast(fx.BFloat16), b, accs[i]
                    )
        for i in range_constexpr(2):
            fx.ptr_store(accs[i], red + ((wave * 2 + i) * 64 + lane) * 4)
        gpu.barrier()
        push_rows(pl, "op", tag, par, tid, bid, rank, red)

    @traced
    def push_rows(pl, region, tag, par, tid, bid, rank, red):
        """The CTA's 32 rows x S tokens (red: [wave][2 tiles][64 lanes][4]) summed over the waves,
        bf16 pairs into slot [par][rank] of the block owner's ``region``."""
        if tid < 16 * S:
            pp = tid % 16  # row pair of the CTA's 32 rows
            tt = tid // 16
            v = []
            for h2 in range_constexpr(2):
                rr = 2 * pp + h2
                s_ = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    s_ = s_ + fx.ptr_load(
                        red
                        + (
                            ((w * 2 + rr // 16) * 64 + 16 * ((rr % 16) // 4) + tt) * 4
                            + rr % 4
                        )
                    )
                v.append(s_)
            pd = sel_peer(pl, bid // CPRX)
            mb_put_sys(
                tag,
                preg(pd, region),
                ((par * W + rank) * S + tt) * (HID // 2) + bid * 16 + pp,
                bf16_pair(v[0], v[1]),
            )

    # ------------------------------------------------------------ GU: gate_up + SwiGLU
    # CTA b = (group G = b / 8, K eighth k8 = b % 8): gate rows 64 G .. 64 G + 64 (tiles 0..3) and
    # the matching up rows FF + 64 G .. (tiles 4..7) over K eighth k8; the partials go to the token
    # owner CTA (k8 = t mod 8), which sums the 8 eighths in order: h[t][64 G ..] = SwiGLU.
    def gu_loads(w_gu, bid, lane, wave):
        k8 = bid % 8
        grp = bid // 8
        out = []
        for n in range_constexpr(GUT):
            row16 = grp * 64 + (16 * n if n < 4 else FF + 16 * (n - 4))
            out.append(bf_tile_load_mma(w_gu, row16, HID, k8 * XQ8 + wave * KCH, lane))
        return out

    @traced
    def stage_gu(u_ops, scratch, tag, health, tid, bid, lane, wave, red, own):
        tid = fresh(tid)
        lane = fresh(lane)
        g = lane // 16
        k8 = bid % 8
        grp = bid // 8
        if tid < 32:
            mb_poll(
                tag, health, [(reg(scratch, "xf"), k8 * 32 + tid, 1)], REGION_IDS["xf"]
            )
        gpu.barrier()
        t = lane % 16
        xb = []
        for qq in range_constexpr(4):
            kk = wave * KCH + 8 * g + 32 * qq
            xb.append(
                ld_vec_if(
                    t < S,
                    reg(scratch, "xin"),
                    (t * HID + k8 * XQ8 + kk) // 2,
                    4,
                    S * HID * 2,
                    CM_DEV,
                )
            )
        rocdl.sched_barrier(0)
        for n in range_constexpr(GUT):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for qq in range_constexpr(4):
                acc = mfma_bf16(
                    u_ops[n][qq].bitcast(fx.BFloat16), xb[qq].bitcast(fx.BFloat16), acc
                )
            fx.ptr_store(acc, red + ((wave * GUT + n) * 64 + lane) * 4)
        gpu.barrier()
        for it_ in range_constexpr((GUT * 8 * S + THREADS - 1) // THREADS):
            e = tid + THREADS * it_
            if e < GUT * 8 * S:
                n = e // (8 * S)
                rp = (e // S) % 8
                tt = e % S
                v = []
                for h2 in range_constexpr(2):
                    rr = 2 * rp + h2
                    s_ = fx.Float32(0.0)
                    for w in range_constexpr(WAVES):
                        s_ = s_ + fx.ptr_load(
                            red
                            + (((w * GUT + n) * 64 + 16 * (rr // 4) + tt) * 4 + rr % 4)
                        )
                    v.append(s_)
                if (tt % 8) == k8:
                    for h2 in range_constexpr(2):
                        fx.ptr_store(
                            v[h2], own + ((n * 16 + 2 * rp + h2) * SO + tt // 8)
                        )
                else:
                    for h2 in range_constexpr(2):
                        mb_put(
                            tag,
                            reg(scratch, "gx"),
                            (((grp * 8 + k8) * GUT + n) * 16 + 2 * rp + h2) * S + tt,
                            v[h2],
                        )
        gpu.barrier()
        # owner wave j: token tt = k8 + 8 j; lane p < 32 = (gate tile n = p / 8, row pair p % 8)
        if (tid < 64 * SO) & (lane < 32):
            j = tid // 64
            p = lane
            n = p // 8
            rp = p % 8
            tt = k8 + 8 * j
            specs, oks = [], []
            for half in range_constexpr(2):
                for src in range_constexpr(8):
                    for h2 in range_constexpr(2):
                        specs.append(
                            (
                                reg(scratch, "gx"),
                                (
                                    ((grp * 8 + src) * GUT + n + 4 * half) * 16
                                    + 2 * rp
                                    + h2
                                )
                                * S
                                + tt,
                                1,
                            )
                        )
                        oks.append(fx.Int32(src) != k8)
            got = mb_poll(tag, health, specs, REGION_IDS["gx"], oks=oks)
            tot = []
            for half in range_constexpr(2):
                for h2 in range_constexpr(2):
                    s_ = fx.Float32(0.0)
                    mine = fx.ptr_load(
                        own + (((n + 4 * half) * 16 + 2 * rp + h2) * SO + j)
                    )
                    for src in range_constexpr(8):
                        s_ = s_ + (fx.Int32(src) == k8).select(
                            mine, got[16 * half + 2 * src + h2][0].bitcast(fx.Float32)
                        )
                    tot.append(bf16_round(s_))  # the bf16 gate_up output
            hv = []
            for h2 in range_constexpr(2):
                gt = tot[h2]
                silu = bf16_round(
                    gt / (fx.Float32(1.0) + hw_exp2(gt * fx.Float32(-LOG2E)))
                )
                hv.append(silu * tot[2 + h2])
            col = grp * 64 + 16 * n + 2 * rp
            bo.buffer_store(
                bf16_pair(hv[0], hv[1]),
                rsrc(reg(scratch, "h")),
                (tt * FF + col) // 2,
                cache_modifier=CM_DEV,
            )
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if tid == 0:
            mb_put(tag, reg(scratch, "hf"), bid, fx.Int32(1))

    # ------------------------------------------------------------ DN: down_proj, push
    def dn_loads(w_dn, bid, lane, wave):
        ops = []
        for i in range_constexpr(2):
            for ci in range_constexpr(CD):
                c = wave + WAVES * ci
                ops.append(bf_tile_load_mma(w_dn, bid * 32 + 16 * i, FF, c * KCH, lane))
        return ops

    @traced
    def stage_dn(d_ops, scratch, pl, tag, health, tid, bid, lane, wave, rank, red):
        """down_proj rows 32 b .. 32 b + 32 (2 tiles) over this rank's K = FF; the bf16 partial
        into slot [par][rank] of the block owner's DP."""
        lane = fresh(lane)
        wait_all(scratch, "hf", tag, health, tid)
        tid = fresh(tid)
        par = tag_par(tag)
        g = lane // 16
        t = lane % 16
        hb = []
        for ci in range_constexpr(CD):
            c = wave + WAVES * ci
            for qq in range_constexpr(4):
                hb.append(
                    ld_vec_if(
                        t < S,
                        reg(scratch, "h"),
                        (t * FF + c * KCH + 8 * g + 32 * qq) // 2,
                        4,
                        S * FF * 2,
                        CM_DEV,
                    )
                )
        rocdl.sched_barrier(0)
        accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
        for i in range_constexpr(2):
            for ci in range_constexpr(CD):
                for qq in range_constexpr(4):
                    accs[i] = mfma_bf16(
                        d_ops[i * CD + ci][qq].bitcast(fx.BFloat16),
                        hb[ci * 4 + qq].bitcast(fx.BFloat16),
                        accs[i],
                    )
        for i in range_constexpr(2):
            fx.ptr_store(accs[i], red + ((wave * 2 + i) * 64 + lane) * 4)
        gpu.barrier()
        push_rows(pl, "dp", tag, par, tid, bid, rank, red)

    # ------------------------------------------------------------ one draft layer
    def ids(A):
        """Fresh thread / CTA ids for a stage: no lane or CTA math hoisted out of the layer loop
        (or CSE'd across stages) into long-lived registers."""
        tid = fresh(A["tid"])
        return tid, tid % 64, tid // 64, sfresh(A["bid"])

    def peers(A):
        pl = tab_ptrs(A["wtab"], T_G, G_PEER, W)
        return pl, sel_peer(pl, A["rank"])

    @traced
    def layer_step(layer, A):
        """Layer ``layer``: the boundary into it (the FFN boundary of layer - 1, or the
        embeddings), QKV, attention, O, AB, GU, DN (its partials pushed)."""
        wtab, health, rank, rsl = A["wtab"], A["health"], A["rank"], A["rsl"]
        scratch = sfresh64(A["scratch"])
        tb = A["base"] + layer + 1  # the boundary into the layer (phase layer - 1)
        tl = tb + 1
        dptr = A["dtab"] + fx.Int64(layer * (attn_frag.DESC_WORDS * 8))

        tid, lane, wave, bid = ids(A)
        qo = q8_loads(tab_ptrs(wtab, layer, T_LAYER, 2)[0], bid, lane, wave)
        bw = tab_ptrs(wtab, layer, T_BND, 4)
        pl, peer = peers(A)
        boundary(
            peer,
            pl,
            scratch,
            "dp",
            "yrr",
            "ssg4p",
            "ssg4",
            "xnf",
            reg(scratch, "smid"),
            reg(scratch, "stream"),
            reg(scratch, "xn"),
            bw[0],
            bw[1],
            bw[2],
            bw[3],
            tb,
            health,
            tid,
            bid,
            lane,
            rank,
            rsl,
            True,
            poll_ok=layer > 0,
        )

        tid, lane, wave, bid = ids(A)
        gw = tab_ptrs(wtab, T_G, G_Q, 4)  # q, g, cos_sin
        kw = ld_vec(dptr, 2, 4)  # descriptor words 1 (KV cache), 2 (block stride)
        stage_qkv(
            qo,
            scratch,
            gw[0],
            gw[1],
            p64(kw, 0),
            p64(kw, 1),
            p64(ld_vec(A["slot_tab"], 2 * layer, 2), 0),  # the layer's slot_mapping
            gw[2],
            A["pos"],
            tl,
            health,
            tid,
            bid,
            lane,
            wave,
            A["sq"].red.ptr,
            A["sq"].own.ptr,
        )
        tid, lane, wave, bid = ids(A)
        wait_all(scratch, "qf", tl, health, tid)
        _, ao = FA.run(dptr, tl, health, tid, bid, lane, wave, A["sfa"])

        tid, lane, wave, bid = ids(A)
        o_ops = o_loads(tab_ptrs(wtab, layer, T_LAYER, 2)[1], bid, lane, wave)
        pl, _ = peers(A)
        stage_o(
            o_ops,
            ao,
            tab_ptrs(wtab, T_G, G_Q, 2)[1],
            pl,
            tl,
            health,
            tid,
            bid,
            lane,
            wave,
            rank,
            A["so"].y.ptr,
            A["so"].red.ptr,
        )

        tid, lane, wave, bid = ids(A)
        lw = tab_ptrs(wtab, layer, T_LAYER + 2, 4)  # alpha_a, beta_a, w_pffn, w_gu
        u_ops = gu_loads(lw[3], bid, lane, wave)
        pl, peer = peers(A)
        boundary(
            peer,
            pl,
            scratch,
            "op",
            "orr",
            "ssa4p",
            "ssa4",
            "xf",
            reg(scratch, "stream"),
            reg(scratch, "smid"),
            reg(scratch, "xin"),
            lw[0],
            lw[1],
            None,
            lw[2],
            tl,
            health,
            tid,
            bid,
            lane,
            rank,
            rsl,
            False,
        )
        tid, lane, wave, bid = ids(A)
        stage_gu(
            u_ops,
            scratch,
            tl,
            health,
            tid,
            bid,
            lane,
            wave,
            A["su"].red.ptr,
            A["su"].own.ptr,
        )

        tid, lane, wave, bid = ids(A)
        d_ops = dn_loads(tab_ptrs(wtab, layer, T_LAYER + 6, 2)[0], bid, lane, wave)
        pl, _ = peers(A)
        stage_dn(
            d_ops, scratch, pl, tl, health, tid, bid, lane, wave, rank, A["sd"].red.ptr
        )

    # ------------------------------------------------------------ kernel
    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dspark(
        x_in: Int64,
        out: Int64,
        wtab: Int64,
        dtab: Int64,
        pos: Int64,
        slot_tab: Int64,
        scratch: Int64,
        epoch: Int64,
        health: Int64,
        rank: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        ml = alloc.allocate(Misc).peek()
        sl = alloc.allocate(StageLds)
        ep = uniform(ld_i32(epoch, 0, CM_DEV))
        base = (ep & 0x1FFFFFF) * 64
        stage_in(x_in, scratch, tid, bid)
        A = {
            "tid": tid,
            "bid": bid,
            "scratch": scratch,
            "health": health,
            "rank": rank,
            "base": base,
            "wtab": wtab,
            "dtab": dtab,
            "pos": pos,
            "slot_tab": slot_tab,
            "rsl": ml.rs.ptr,
            "so": sl.o_.peek(),
            "sq": sl.q_.peek(),
            "su": sl.u_.peek(),
            "sd": sl.d_.peek(),
            "sfa": sl.fa_.peek(),
        }
        layer = fx.Int32(0)
        while layer < LAYERS:
            layer_step(layer, A)
            layer = layer + 1
        # the FFN boundary of the last layer with the final norm -> the draft hidden states
        tid, lane, _, bid = ids(A)
        bw = tab_ptrs(wtab, fx.Int32(LAYERS), T_BND, 4)
        pl, peer = peers(A)
        boundary(
            peer,
            pl,
            scratch,
            "dp",
            "yrr",
            "ssg4p",
            "ssg4",
            "xnf",
            scratch + fx.Int64(lay["smid"][0]),
            scratch + fx.Int64(lay["stream"][0]),
            out,
            bw[0],
            bw[1],
            bw[2],
            bw[3],
            base + LAYERS + 1,
            health,
            tid,
            bid,
            lane,
            rank,
            ml.rs.ptr,
            True,
            final=True,
        )
        # every CTA read the epoch before the last norm round completed
        if (bid == 0) & (tid == 0):
            bo.buffer_store(ep + 1, rsrc(epoch), 0, cache_modifier=CM_DEV)

    @flyc.jit
    def launch(
        x_in: Int64,
        out: Int64,
        wtab: Int64,
        dtab: Int64,
        pos: Int64,
        slot_tab: Int64,
        scratch: Int64,
        epoch: Int64,
        health: Int64,
        rank: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _KEYSTR
        dspark(
            x_in, out, wtab, dtab, pos, slot_tab, scratch, epoch, health, rank
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch


def build_vllm(S=8, W=8):
    """The draft build vLLM launches for S rows (8 / 16: 1 / 2 requests of 7 rows), TP = 8."""
    assert W == 8 and S in (8, 16), (S, W)
    return Build(S=S)
