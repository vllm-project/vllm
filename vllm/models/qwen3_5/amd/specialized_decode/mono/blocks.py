# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decoder layer's two launches, each persistent, ``BLOCKS`` x ``THREADS``,
S <= ``MAX_TOKENS`` rows:

    gdn    (a GDN layer's residual add + input_layernorm and ``linear_attn``
           up to out_proj's input): h (+ residual) -> the residual out, x =
           norm(h + residual) -> in_proj_qkvz (MXFP4) / in_proj_ba -> conv1d ->
           the gated delta rule -> the gated norm -> core [S, CORE]
    layer  (out_proj / o_proj, its TP all-reduce, the residual add +
           post_attention_layernorm, the sparse MoE and its TP all-reduce):
           core (x sigmoid(gate)) -> every rank's AR -> h + residual -> the
           residual out, x -> router, sigmoid shared gate, top-10, the MXFP4
           experts with the shared expert fused as expert 512 -> every rank's
           FINAL -> the layer's output [S, H]

Neither keeps x [S, H] in LDS (at 16 rows a bf16 x is 262 KB): the norm tasks
(CTAs 0 .. NRED - 1) put their columns' sums of squares, and every consumer
builds what it reads of x from the residual add's inputs in global memory
(``stages.norm``): the router and in_proj_ba a bf16 K slice, in_proj_qkvz and
the routed experts the rows' MXFP4 codes, the shared gate a dot product. From
``GDN_XQ_MIN`` / ``LAYER_XQ_MIN`` rows the norm tasks quantize their columns
once into a shared MXFP4 image (``xq``) that in_proj_qkvz and the experts copy
instead: every CTA rebuilding all of x would move 256 x 2 [S, H] bf16 through
the L2s each launch. A launch's input is a previous launch's output (plain
loads), except the layer launch's AR sums (``xbuf``), which its norm tasks
write in the launch.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as L
from vllm.models.qwen3_5.amd.specialized_decode.mono import mx_layout as QL
from vllm.models.qwen3_5.amd.specialized_decode.mono.common.debug import region_ids
from vllm.models.qwen3_5.amd.specialized_decode.mono.common.ops import (
    CM_DEV,
    bf16_round,
    ld_i32,
    load_ptr64,
    row_sum,
    traced,
    uniform,
)
from vllm.models.qwen3_5.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
    source_digest,
)
from vllm.models.qwen3_5.amd.specialized_decode.mono.common.ranks import peer_bases
from vllm.models.qwen3_5.amd.specialized_decode.mono.common.sync import (
    Mailbox,
    preg,
    shift,
    sreg,
)
from vllm.models.qwen3_5.amd.specialized_decode.mono.layout import need
from vllm.models.qwen3_5.amd.specialized_decode.mono.stages import gdn as G
from vllm.models.qwen3_5.amd.specialized_decode.mono.stages import gemv, moe
from vllm.models.qwen3_5.amd.specialized_decode.mono.stages import norm as N
from vllm.models.qwen3_5.amd.specialized_decode.mono.stages.gdn import (
    NVCH,
    VCH,
    gdn_tasks,
)
from vllm.models.qwen3_5.amd.specialized_decode.mono.stages.moe import (
    PS_MIN,
    f32_of,
    step_tag,
    xq_scales,
)

H, ROWS, HD, TP = L.HIDDEN, L.ROWS, L.HD, L.TP
E, TOPK, EPR = L.E, L.TOPK, L.EPR
KT = TOPK + 1  # picks a token: the top-10 and the fused shared expert
MAXT = L.MAX_TOKENS
NRED = N.NRED  # 64: the norm / final tasks, CTAs 0 ..
OTASKS = H // ROWS  # 512: o_proj / down row groups, bid and bid + 256
# in_proj_qkvz: row group 255 - bid, and 511 - bid on CTAs QQ0 ..
QTASKS = L.QKVZ // ROWS  # 288
QQ0 = 2 * BLOCKS - QTASKS  # 224
# in_proj_ba: 2 row groups x 8 K parts on CTAs 0 ..
BA_PARTS = 8
BA_K = H // BA_PARTS
BA_TASKS = L.BA // ROWS * BA_PARTS  # 16
# the router: 4 row groups of the rank's 64 experts x 4 K quarters, CTAs R0 ..
R_PARTS = 4
R_K = H // R_PARTS
R_TASKS = EPR // ROWS * R_PARTS  # 16
R0 = NRED  # 64
RT0 = R0 + R_TASKS  # 80: shared gate CTAs, a token each
PS0 = RT0 + MAXT  # 96: the fused shared expert's early ug CTAs
# ``rm``: a router task multiplies 16 rows of w_rm with one of RM_CH chunks
# of RM_K columns: the two halves of the core rows and this rank's slice of
# the residual. The tasks run on CTAs R0 .. R0 + RM_TASKS - 1, and CTA R0 +
# i, i < R_TASKS, sums the logits of RS_PER experts.
CK = L.CORE  # o_proj's K, the old OK
HIDDEN = H
RM_K = 1024
RM_CH = 3
RM_TASKS = RM_CH * E // ROWS  # 96
RS_PER = E // R_TASKS  # 32
# rows below which the layer block takes the router logits through o_proj
RM_MAX = 5
GDN_TASKS = L.NV * NVCH  # GDN tasks a token
# rows from which x goes through MXFP4 once, on the norm tasks (``N.stage_xq``)
GDN_XQ_MIN = 4
# rows up to which every GDN CTA computes the rows' rstd itself (``N.local_rstd``)
GDN_LOCAL_MAX = 3
LAYER_XQ_MIN = 5
# rows from which down runs the slots of run_ug's full rounds first, under its
# last round (needs the fused shared expert's tasks first: ``ps``)
SPLIT_MIN = 8
# mailbox tag slots: gdn ``layer``, layer LAYER_SLOT + ``layer``
LAYER_SLOT = L.MAX_LAYERS

assert SPLIT_MIN >= PS_MIN and GDN_LOCAL_MAX < GDN_XQ_MIN
assert RM_CH * RM_K == CK + H // TP and R0 + RM_TASKS <= BLOCKS
assert OTASKS == 2 * BLOCKS and BLOCKS < QTASKS <= 2 * BLOCKS
assert BA_TASKS <= NRED and PS0 + L.RI // moe.UG_COLS <= BLOCKS
assert LAYER_SLOT + L.MAX_LAYERS <= moe.TAG_SLOTS

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class GdnBuild:
    tokens: int
    first: bool = False  # layer 0: no residual yet (the residual out is h)
    eps: float = 1e-6  # input_layernorm
    norm_eps: float = 1e-6  # the gated norm
    shuffled: bool = True  # in_proj_qkvz in the ASM FP4 GEMM layout, else Triton's
    # in_proj_qkvz in MXFP6 E2M3, in the layout of runtime.mxfp6.fp6_pack,
    # multiplied with the bf16 x rows (W6A16). x is not quantized.
    q6: bool = False

    @property
    def xq(self) -> bool:
        return not self.q6 and self.tokens >= GDN_XQ_MIN

    @property
    def local(self) -> bool:
        return self.tokens <= GDN_LOCAL_MAX


@dataclass(frozen=True)
class LayerBuild:
    tokens: int
    eps: float = 1e-6  # post_attention_layernorm
    oq: bool = True  # out_proj / o_proj MXFP4
    og: bool = False  # o_proj's input is core x sigmoid(gate) (full attention)
    shuffled: bool = True  # with oq: the ASM FP4 GEMM layout, else Triton's
    # out_proj / o_proj in MXFP6 E2M3 (fp6_pack's layout and scales),
    # multiplied with the bf16 core rows (W6A16). It takes the place of oq.
    o6: bool = False
    # The router's logits come through o_proj (rm_pack's w_rm), summed from
    # every rank's partials without the norm, so every CTA routes as soon as
    # its rstd is in. Not with ``oq``, and up to RM_MAX rows.
    rm: bool = False

    @property
    def u1(self) -> bool:
        return self.tokens == 1

    @property
    def up(self) -> bool:
        """2 or 3 rows: a union slot for each pick (no union pass)."""
        return 2 <= self.tokens <= 3

    @property
    def ps(self) -> bool:
        """The fused shared expert's ug tasks run before the route, at the
        widths where the layer stage benchmark found that faster."""
        return self.tokens >= PS_MIN or self.tokens == 2

    @property
    def xq(self) -> bool:
        return self.tokens >= LAYER_XQ_MIN

    @property
    def split(self) -> bool:
        return self.tokens >= SPLIT_MIN


GDN_ABI = KernelAbi(
    (
        "hidden",
        "residual",
        "res_out",
        "ln_w",
        "w_qkvz",
        "w_qkvzs",
        "w_ba",
        "conv_w",
        "conv_st",
        "cs_seq",
        "cs_dim",
        "cs_tok",
        "a_log",
        "dt_bias",
        "norm_w",
        "rstate",
        "rs_seq",
        "st_idx",
        "core",
        "scratch",
        "epoch",
        "health",
        "alarm",
        "layer",
    )
)
LAYER_ABI = KernelAbi(
    (
        "core",
        "gate",
        "residual",
        "w_o",
        "w_os",
        "ln_w",
        "w_gate",
        "w_sg",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "out",
        "res_out",
        "scratch",
        "peers",
        "rank",
        "epoch",
        "health",
        "alarm",
        "layer",
        "w_rm",
    )
)


def fp4_scale_shape(rows: int, k: int, shuffled: bool) -> tuple[int, int]:
    """An MXFP4 linear's scales: [rows, K / 32] swizzled for the ASM FP4 GEMM,
    else transposed [K / 32, rows] for the Triton FP4 GEMM."""
    return (rows, QL.scale_cols(k)) if shuffled else (k // 32, rows)


def gdn_offset(tokens: int) -> int:
    """The first GDN CTA: past the norm CTAs while the tasks fit on the CTAs
    with a single in_proj task, else a full round from CTA 0."""
    return NRED if tokens * GDN_TASKS <= QQ0 - NRED else 0


def _xq_layout(lay, key):
    """With ``key.xq``: the x rows' shared MXFP4 image (16 B aligned) past
    ``lay``."""
    if not key.xq:
        return lay
    return lay | pair_layout(
        [("xq", N.xq_dwords(key.tokens) // 2)], 16, scratch_bytes(lay)
    )


def gdn_scratch_layout(key: GdnBuild) -> dict:
    s = key.tokens
    lay = pair_layout(
        {
            "msq": NRED * s,
            "xqf": NRED,
            "proj": s * L.QKVZ,
            "qk": s * L.NK * 2 * HD,
            "onrm": s * L.NV * NVCH,
            "bapart": BA_TASKS * s * ROWS,
        }.items()
    )
    return _xq_layout(lay, key)


LAYER_REGIONS = (
    "msq",
    "msq2",
    "xar",
    "xqf",
    "rpart",
    "sgate",
    "route",
    "inter",
    "rsum",
)


def layer_scratch_layout(key: LayerBuild) -> dict:
    """The mailbox regions, then ``xbuf``: the AR sums [S, H] bf16 the norm
    tasks write for the consumers' x (16 B aligned)."""
    s = key.tokens
    lay = pair_layout(
        {
            "msq": NRED * s,
            # the norm tasks' sums of squares again, after their xbuf stores
            # are drained, for the consumers that read xbuf
            "msq2": NRED * s,
            # the bf16 AR sums as tagged pairs, a pair a word of xbuf, for
            # the router tasks
            "xar": s * H // 2,
            "xqf": NRED,
            "rpart": R_TASKS * s * ROWS,
            "sgate": s,
            "route": s * 2 * TOPK,
            "inter": s * KT * moe.inter_pairs(L.MOE),
            # (rm) the summed logits before the rstd [token][expert], fp32
            "rsum": s * E,
        }.items()
    )
    lay |= pair_layout([("xbuf", s * H * 2 // 8)], 16, scratch_bytes(lay))
    lay = _xq_layout(lay, key)
    return lay


def scratch_bytes(layout: dict) -> int:
    return max(o + n for o, n in layout.values())


PEER_REGIONS = ("ar", "rlog", "final", "rpre")


def peer_layout(s):
    """One parity's peer regions: the o_proj partials, the router logits and
    the MoE partials of every rank."""
    return pair_layout(
        {
            "ar": TP * s * H // 2,
            "rlog": TP * s * EPR // 2,
            "final": TP * s * H // 2,
            # (rm) the fp32 partial logits [rank][chunk][token][expert]
            "rpre": TP * RM_CH * s * E,
        }.items()
    )


def peer_bytes(s):
    """The peer buffer: both parities' regions at ``s`` rows."""
    return 2 * scratch_bytes(peer_layout(s))


def _peers(peers, rank, epoch):
    """The peer regions at the widest width, this step's parity half: every
    rank's and this rank's own."""
    play = peer_layout(MAXT)
    half = peer_bytes(MAXT) // 2
    par = uniform(ld_i32(epoch, 0)) % 2
    bases = peer_bases(peers, TP)
    own = load_ptr64(peers, rank)
    peer = {}
    for region in PEER_REGIONS:
        off = play[region][0]
        peer[region] = [
            shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
        ]
        peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
    return peer


def _xwords(s):
    """LDS words of the x a CTA holds at once: a bf16 K slice (in_proj_ba,
    the router) or the rows' MXFP4 codes."""
    return max(s * (max(BA_K, R_K) + gemv.LDS_PAD) // 2, xq_scales(s, H) + s * H // 32)


# ---------------------------------------------------------------- gdn
_GDN: dict = {}


def qkvz_loads(c, rg, live=None):
    """An fp4 (``q6``: fp6) in_proj_qkvz task's (rg < 288) operands for this
    wave."""
    a = c["args"]
    if const_expr(c.get("q6", False)):
        return moe.fp6_gemv_loads(
            c["lane"], c["wave"], a["w_qkvz"], a["w_qkvzs"], rg, H
        )
    return moe.fp4_gemv_loads(
        c["lane"],
        c["wave"],
        a["w_qkvz"],
        a["w_qkvzs"],
        rg,
        H,
        L.QKVZ,
        c["shuffled"],
        live,
    )


@traced
def stage_qkvz(c, rg, ops):
    """in_proj_qkvz rows 16 rg .. on the x rows' MXFP4 codes (with ``q6``: an
    MXFP6 task on the bf16 x rows) -> bf16 PROJ."""
    lane, wave = c["lane"], c["wave"]
    if const_expr(c.get("q6", False)):
        xrow = H + gemv.LDS_PAD
        moe.tiled(
            c,
            lambda t0, n: moe.fp6_gemv(
                lane, wave, moe.rows_at(c["xl"], xrow, t0), xrow, H, n, ops
            ),
            lambda t0, n: _qkvz_out(c, rg, t0, n),
        )
        return
    moe.put_acc(
        c, moe.fp4q_gemv(lane, wave, c["xl"], H, c["S"], rg, ops, c["shuffled"])
    )
    _qkvz_out(c, rg, 0, c["S"])


@traced
def _qkvz_out(c, rg, t0, n):
    tid = c["tid"]
    if tid < ROWS * n:
        r = tid % ROWS
        j = tid // ROWS
        v = bf16_round(row_sum(c["red"], r, j))
        c["put"](c["proj"], moe.tok(t0, j) * L.QKVZ + rg * ROWS + r, v)
    gpu.barrier()


def build_gdn(key: GdnBuild):
    if key in _GDN:
        return _GDN[key]
    s = key.tokens
    need(1 <= s <= MAXT, f"{s} rows, the kernels take 1..{MAXT}")
    need(
        not key.q6 or s <= L.Q6_MAX,
        f"q6 holds the bf16 x rows in LDS, {L.Q6_MAX} at most",
    )
    scr = gdn_scratch_layout(key)
    nr = -(-s * GDN_TASKS // BLOCKS)

    xw = s * (H + gemv.LDS_PAD) // 2 if key.q6 else _xwords(s)

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, xw, 16]

    @fx.struct
    class GdnLds:
        qkl: fx.Array[fx.Float32, nr * 2 * HD, 16]
        vloc: fx.Array[fx.Float32, nr * VCH, 16]
        ol: fx.Array[fx.Float32, nr * VCH, 16]
        scal: fx.Array[fx.Float32, nr * 4, 16]

    @fx.union
    class StageLds:
        x: XLds
        gdn: GdnLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        rs: fx.Array[fx.Float32, 16, 16]

    keyed = key_tuple(key, source_digest())
    name = f"qwen38_sd_gdn_s{s}_f{int(key.first)}_w{int(key.shuffled)}_q{int(key.q6)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kgdn(
        hidden: Int64,
        residual: Int64,
        res_out: Int64,
        ln_w: Int64,
        w_qkvz: Int64,
        w_qkvzs: Int64,
        w_ba: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        norm_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        core: Int64,
        scratch: Int64,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        layer: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        st_lds = alloc.allocate(StageLds)
        xl, gl = st_lds.x.peek(), st_lds.gdn.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1, health, alarm, region_ids(scr))
        c = {
            "S": s,
            "gdn": L.GDN,
            "shuffled": key.shuffled,
            "q6": key.q6,
            "ba_of": lambda c, t, h: G.ba_sums(c, t, h, BA_PARTS),
            "ba_parts": BA_PARTS,
            "z_gate": G.silu,
            "goff": gdn_offset(s),
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "eps": key.eps,
            "norm_eps": key.norm_eps,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "rs": lds.rs.ptr,
            "xl": xl.xl.ptr,
            "qkl": gl.qkl.ptr,
            "vloc": gl.vloc.ptr,
            "ol": gl.ol.ptr,
            "scal": gl.scal.ptr,
            "args": {
                "res_out": res_out,
                "ln_w": ln_w,
                "w_qkvz": w_qkvz,
                "w_qkvzs": w_qkvzs,
                "w_ba": w_ba,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "norm_w": norm_w,
                "rstate": rstate,
                "rs_seq": rs_seq,
                "st_idx": st_idx,
                "core": core,
            },
        }
        for region in scr:
            if region != "xq":
                c[region] = sreg(scratch, scr[region][0], region)
        if key.xq:
            c["xq"] = scratch + fx.Int64(scr["xq"][0])
        src = (hidden, 0, None if key.first else residual)
        if bid < NRED:
            N.stage_front(c, bid, src, not key.local)
        if const_expr(key.q6):
            # With q6, in_proj_qkvz multiplies the bf16 rows in LDS, so x is
            # not quantized and stage_ba reads the same full rows.
            def load_x():
                if const_expr(key.local):
                    N.local_rstd(c, src)
                else:
                    N.load_rstd(c)
                N.x_rows_to_lds(c, src, H, 0)

            rg = BLOCKS - 1 - bid
            qops = qkvz_loads(c, rg)
            if bid >= QQ0:
                rg2 = 2 * BLOCKS - 1 - bid
                qops2 = qkvz_loads(c, rg2)
                load_x()
                stage_qkvz(c, rg, qops)
                stage_qkvz(c, rg2, qops2)
            if bid < BA_TASKS:
                bops = G.ba_loads(c, bid, BA_PARTS, BA_K)
                load_x()
                G.stage_ba(c, bid, bops, BA_PARTS, BA_K)
                stage_qkvz(c, rg, qops)
            if (bid >= BA_TASKS) & (bid < QQ0):
                load_x()
                stage_qkvz(c, rg, qops)
        else:
            rg = BLOCKS - 1 - bid
            two = bid >= QQ0
            qops = qkvz_loads(c, rg)
            qops2 = qkvz_loads(c, fx.min(2 * BLOCKS - 1 - bid, QTASKS - 1), two)
            is_ba = bid < BA_TASKS
            bops = G.ba_loads(c, fx.min(bid, BA_TASKS - 1), BA_PARTS, BA_K)
            if const_expr(key.xq):
                if bid < NRED:
                    N.load_rstd(c)
                    N.stage_xq(c, bid, src, key.shuffled)
                if is_ba:
                    N.x_rows_to_lds(c, src, BA_K, bid % BA_PARTS * BA_K)
                    G.stage_ba(c, bid, bops, BA_PARTS, BA_K, sliced=True)
                N.xq_to_lds(c)
            else:
                if const_expr(key.local):
                    N.local_rstd(c, src)
                else:
                    N.load_rstd(c)
                if is_ba:
                    N.x_rows_to_lds(c, src, BA_K, bid % BA_PARTS * BA_K)
                    G.stage_ba(c, bid, bops, BA_PARTS, BA_K, sliced=True)
                N.x_fp4_to_lds(c, src, key.shuffled)
            stage_qkvz(c, rg, qops)
            if two:
                stage_qkvz(c, 2 * BLOCKS - 1 - bid, qops2)
        gdn_tasks(c, nr)

    @flyc.jit
    def launch(
        hidden: Int64,
        residual: Int64,
        res_out: Int64,
        ln_w: Int64,
        w_qkvz: Int64,
        w_qkvzs: Int64,
        w_ba: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        norm_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        core: Int64,
        scratch: Int64,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kgdn(
            hidden,
            residual,
            res_out,
            ln_w,
            w_qkvz,
            w_qkvzs,
            w_ba,
            conv_w,
            conv_st,
            cs_seq,
            cs_dim,
            cs_tok,
            a_log,
            dt_bias,
            norm_w,
            rstate,
            rs_seq,
            st_idx,
            core,
            scratch,
            epoch,
            health,
            alarm,
            layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    GDN_ABI.check(kgdn, launch)
    _GDN[key] = launch
    return launch


def gdn_block(
    key: GdnBuild,
    *,
    hidden,
    residual,
    res_out,
    ln_w,
    w_qkvz,
    w_qkvzs,
    w_ba,
    conv_w,
    conv_state,
    a_log,
    dt_bias,
    norm_w,
    rstate,
    st_idx,
    core,
    scratch,
    epoch,
    health,
    alarm,
    layer,
):
    """A GDN layer's residual add + input_layernorm and ``linear_attn`` up to
    out_proj's input: ``hidden`` [S, H] bf16 (+ ``residual``, None with
    ``key.first``) -> ``res_out`` [S, H] and ``core`` [S, CORE] bf16. Every
    weight as vLLM holds it after loading (in_proj_qkvz online MXFP4 with e8m0
    scales, in the layout of the FP4 GEMM AITER picked: ``key.shuffled``).
    ``conv_state`` [slots, CONV, 3] or a transposed view of [slots, 3, CONV]
    (strides read); ``rstate`` [slots, NV, 128, 128] fp32, a slot dense, slots
    strided; ``st_idx`` [>= S] int32, <= 0 a pad row (no state written, a zero
    core row); ``epoch`` a device int32 bumped once a step, ``layer`` this
    launch's tag slot."""
    s = key.tokens
    bf = torch.bfloat16
    rows = (s, H)
    need(
        hidden.shape == rows and hidden.dtype == bf and hidden.is_contiguous(),
        "hidden: bf16 [S, H]",
    )
    need((residual is None) == key.first, "residual: None for the first layer only")
    need(
        residual is None or (residual.shape == rows and residual.dtype == bf),
        "residual: bf16 [S, H]",
    )
    need(res_out.shape == rows and res_out.dtype == bf, "res_out: bf16 [S, H]")
    need(ln_w.dtype == bf and ln_w.numel() == H, "input_layernorm weight")
    need(core.shape == (s, L.CORE) and core.dtype == bf, f"core: bf16 [S, {L.CORE}]")
    if key.q6:
        need(
            w_qkvz.dtype == torch.uint8 and w_qkvz.numel() == L.QKVZ * H * 3 // 4,
            "in_proj_qkvz: fp6_pack's MXFP6 weight, 4608 x 8192 x 3/4 bytes",
        )
        need(
            w_qkvzs.numel() == L.QKVZ * H // 32 and w_qkvzs.element_size() == 1,
            "in_proj_qkvz scales: e8m0, one a 32-block",
        )
    else:
        need(
            w_qkvz.shape == (L.QKVZ, H // 2)
            and w_qkvz.dtype in (torch.uint8, torch.float4_e2m1fn_x2),
            f"in_proj_qkvz: fp4x2 [{L.QKVZ}, {H // 2}]",
        )
        need(
            w_qkvzs.shape == fp4_scale_shape(L.QKVZ, H, key.shuffled)
            and w_qkvzs.element_size() == 1,
            "in_proj_qkvz scales: e8m0, one a 32-block, in the weight's layout",
        )
    need(
        w_ba.shape == (L.BA, H) and w_ba.dtype == bf, f"in_proj_ba: bf16 [{L.BA}, {H}]"
    )
    need(conv_w.numel() == L.CONV * L.CONV_W and conv_w.dtype == bf, "conv1d weight")
    need(
        conv_state.dtype == bf and conv_state.shape[1:] == (L.CONV, G.SL),
        f"conv state: bf16 [slots, {L.CONV}, {G.SL}]",
    )
    need(
        rstate.dtype == torch.float32 and rstate.shape[1:] == (L.NV, HD, HD),
        f"recurrent state: fp32 [slots, {L.NV}, 128, 128]",
    )
    need(rstate[0].is_contiguous(), "a recurrent state slot must be dense")
    need(a_log.dtype == torch.float32 and dt_bias.dtype == bf, "A_log / dt_bias dtype")
    need(norm_w.dtype == bf and norm_w.numel() == HD, "gated norm weight")
    need(st_idx.dtype == torch.int32 and st_idx.numel() >= s, "state indices")
    used = [hidden, residual, res_out, ln_w, w_qkvz, w_qkvzs, w_ba, conv_w, core]
    need(
        all(t.is_contiguous() for t in used if t is not None),
        "a non-contiguous GDN input",
    )
    vals = {
        "hidden": hidden.data_ptr(),
        "residual": 0 if residual is None else residual.data_ptr(),
        "res_out": res_out.data_ptr(),
        "ln_w": ln_w.data_ptr(),
        "w_qkvz": w_qkvz.data_ptr(),
        "w_qkvzs": w_qkvzs.data_ptr(),
        "w_ba": w_ba.data_ptr(),
        "conv_w": conv_w.data_ptr(),
        "conv_st": conv_state.data_ptr(),
        "cs_seq": conv_state.stride(0),
        "cs_dim": conv_state.stride(1),
        "cs_tok": conv_state.stride(2),
        "a_log": a_log.data_ptr(),
        "dt_bias": dt_bias.data_ptr(),
        "norm_w": norm_w.data_ptr(),
        "rstate": rstate.data_ptr(),
        "rs_seq": rstate.stride(0),
        "st_idx": st_idx.data_ptr(),
        "core": core.data_ptr(),
        "scratch": scratch.data_ptr(),
        "epoch": epoch.data_ptr(),
        "health": health.data_ptr(),
        "alarm": alarm,
        "layer": layer,
    }
    build_gdn(key)(*GDN_ABI.pack(vals), stream=torch.cuda.current_stream())


# ---------------------------------------------------------------- layer
_LAYER: dict = {}


def router_loads(c, i, q):
    """Router task (i, q): the rank's experts 16 i .., K quarter q."""
    return gemv.loads(
        c["lane"],
        c["wave"],
        c["args"]["w_gate"],
        c["rank"] * EPR + i * ROWS,
        R_K,
        ldk=H,
        k0=q * R_K,
    )


@traced
def stage_router(c, i, q, ops):
    """Router task (i, q) on x's K quarter q (``x_tagged_to_lds``). Parts q >
    0 publish fp32 partials; part 0 adds them in part order, rounds to bf16
    (GateLinear's output) and puts the logits to every rank's RLOG."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    N.x_tagged_to_lds(c, R_K, q * R_K)
    moe.put_acc(c, gemv.mfmas(lane, wave, c["xl"], R_K + gemv.LDS_PAD, R_K, s, ops))

    def out(t, r, vs):
        for p in range_constexpr(TP):
            c["put_bf"](
                c["peer"]["rlog"][p], (c["rank"] * s + t) * EPR + i * ROWS + r, vs
            )

    moe.ksplit_rows(
        c,
        c["rpart"],
        lambda qq, t, r: ((i * R_PARTS + qq) * s + t) * ROWS + r,
        q,
        R_PARTS,
        out,
    )


def rm_pack(w_gate, ln_w, w_o, rank):
    """``rm``'s weight w_rm of one rank, bf16 [E, RM_CH * RM_K]. Its first OK
    columns are the router rows times the post-attention norm's (1 + w),
    times this rank's o_proj rows ``w_o`` (bf16 [HIDDEN, OK], the values
    that K2 multiplies). Its last RM_K columns are the same scaled router
    rows on this rank's slice of the residual. Over the 8 ranks, w_rm times
    [core rows | residual slice] sums to the router rows times
    (AR sum + residual) (1 + w): the router's logits before the rstd."""
    wr = w_gate.float() * (1.0 + ln_w.float())
    cols = wr[:, rank * RM_K : (rank + 1) * RM_K]
    return torch.cat([wr @ w_o.float(), cols], dim=1).to(torch.bfloat16).contiguous()


def rm_loads(c, task):
    """``rm`` router task ``task``'s w_rm tiles of this wave: rows
    16 (task % 32) .. 16 (task % 32) + 15 of K chunk task / 32."""
    q, rg = task // (E // ROWS), task % (E // ROWS)
    return gemv.loads(
        c["lane"],
        c["wave"],
        c["args"]["w_rm"],
        rg * ROWS,
        RM_K,
        ldk=RM_CH * RM_K,
        k0=q * RM_K,
    )


@traced
def stage_rm(c, task, ops):
    """``rm`` router task ``task``: rows 16 (task % 32) .. of w_rm times K
    chunk task / 32 of [core rows | this rank's residual slice] -> the fp32
    partial logits -> every rank's RPRE, in this rank's and chunk's slot.
    The core rows are still in LDS from run_oproj, so only the residual
    slice is loaded, into the rows after them, with the same row length."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    q, rg = task // (E // ROWS), task % (E // ROWS)
    xrow = CK + gemv.LDS_PAD
    res_at = s * xrow // 2
    gemv.rows_to_lds(
        tid,
        c["args"]["residual"],
        RM_K,
        s,
        c["xl"] + res_at,
        xrow,
        ldk=HIDDEN,
        k0=c["rank"] * RM_K,
    )
    base = (q == RM_CH - 1).select(fx.Int32(res_at), q * (RM_K // 2))
    acc = gemv.mfmas(lane, wave, c["xl"] + base, xrow, RM_K, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS * s:
        r = tid % ROWS
        t = tid // ROWS
        v = row_sum(red, r, t)
        at = ((c["rank"] * RM_CH + q) * s + t) * E + rg * ROWS + r
        for p in range_constexpr(TP):
            c["put"](c["peer"]["rpre"][p], at, v)
    gpu.barrier()


@traced
def stage_rsum(c, i):
    """``rm``, CTA R0 + i: the logits of experts RS_PER i .. RS_PER i +
    RS_PER - 1 of every token, a thread a logit. Each is the sum of the 8
    ranks' RM_CH partials (RPRE) in rank and chunk order, the same on every
    rank, and goes to RSUM, which every CTA's route polls."""
    s, tid = c["S"], c["tid"]
    if tid < RS_PER * s:
        t = tid // RS_PER
        e = i * RS_PER + tid % RS_PER
        got = c["poll"](
            [
                (c["peer"]["rpre_own"], ((src * RM_CH + q) * s + t) * E + e, 1)
                for src in range(TP)
                for q in range(RM_CH)
            ],
            batch=TP * RM_CH,
        )
        v = f32_of(got[0][0])
        for k in range_constexpr(1, TP * RM_CH):
            v = v + f32_of(got[k][0])
        c["put"](c["rsum"], t * E + e, v)


def _rlog_at(rlog, s, t, e):
    """Token t's logit of expert e: rank e / EPR's slot of every rank's RLOG."""
    return (rlog, ((e // EPR * s + t) * EPR + e % EPR) // 2, 1)


@traced
def _x_fp4(c, src, missing):
    """The routed experts' x (RCEIL MXFP4, as the stock decode MoE quantizes
    it) -> LDS; ``missing``: the CTA has no rstd yet."""
    if missing:
        N.load_rstd(c)
    N.x_fp4_to_lds(c, src)


@traced
def _x_ug(c, src, xq, missing, rm=False):
    """The routed experts' x -> LDS: the shared MXFP4 image (``xq``), else
    built here (``_x_fp4``). With ``rm`` the route needs every CTA's rstd,
    which the image path does not load."""
    if const_expr(xq):
        if const_expr(rm):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if missing & (c["bid"] >= NRED):
                N.load_rstd(c)
        N.xq_to_lds(c)
    else:
        _x_fp4(c, src, missing)


def build_layer(key: LayerBuild):
    if key in _LAYER:
        return _LAYER[key]
    s = key.tokens
    need(1 <= s <= MAXT, f"{s} rows, the kernels take 1..{MAXT}")
    need(not (key.oq and key.o6), "o6 takes the place of oq")
    need(not (key.rm and key.oq), "rm takes a bf16 or MXFP6 o_proj, not oq")
    need(not key.rm or s <= RM_MAX, f"rm runs up to {RM_MAX} rows")
    scr = layer_scratch_layout(key)
    max_u = s * TOPK + 1  # distinct experts the step can touch + the fused shared
    d = L.MOE

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * KT, 16]
        flag: fx.Array[fx.Int32, E, 16]
        expert: fx.Array[fx.Int32, max_u, 16]
        kof: fx.Array[fx.Int32, max_u * s, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class XLds:
        # With rm, the residual slice rows follow the core rows (stage_rm).
        xl: fx.Array[
            fx.Int32,
            max(_xwords(s), (1 + int(key.rm)) * s * (L.CORE + gemv.LDS_PAD) // 2),
            16,
        ]

    @fx.struct
    class DownLds:
        il: fx.Array[fx.Int32, s * KT * (moe.irow(d) // 4 + d.rip // 32), 16]
        contrib: fx.Array[fx.Float32, (1 + int(key.split)) * s * KT * ROWS, 16]

    @fx.union
    class StageLds:
        x: XLds
        down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        rs: fx.Array[fx.Float32, 16, 16]

    keyed = key_tuple(key, source_digest())
    name = (
        f"qwen38_sd_layer_s{s}_o{int(key.oq)}{int(key.o6)}"
        f"_a{int(key.og)}_w{int(key.shuffled)}_r{int(key.rm)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def klayer(
        core: Int64,
        gate: Int64,
        residual: Int64,
        w_o: Int64,
        w_os: Int64,
        ln_w: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        res_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        layer: Int32,
        w_rm: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        rl = alloc.allocate(RouteLds).peek()
        st_lds = alloc.allocate(StageLds)
        xl, dl = st_lds.x.peek(), st_lds.down.peek()
        mb = Mailbox(
            step_tag(epoch, layer) - 1,
            health,
            alarm,
            region_ids((*LAYER_REGIONS, *PEER_REGIONS)),
        )
        peer = _peers(peers, rank, epoch)
        rlog = peer["rlog_own"]
        if const_expr(key.rm):
            # rm: the route polls RSUM, the fp32 sums of the logits before
            # the rstd, which stage_rsum puts after the slowest rank's
            # partials, so there is no last logit to wait on first.
            def rlog_at(t, lane, i):
                return (c["rsum"], t * E + lane + 64 * i, 1)

            rlog_last = (0, None)
        else:

            def rlog_at(t, lane, i):
                return _rlog_at(rlog, s, t, lane + 64 * i)

            rlog_last = (  # type: ignore[assignment]
                TP,
                lambda i: (rlog, ((i * s + s - 1) * EPR + EPR - 2) // 2, 1),
            )
        c = {
            "S": s,
            "TP": TP,
            "moe": d,
            "rlog_at": rlog_at,
            "rlog_last": rlog_last,
            "rm": key.rm,
            "route_cta": RT0,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "eps": key.eps,
            "oq": key.oq,
            "o6": key.o6,
            "og": key.og,
            "shuffled": key.shuffled,
            "u1": key.u1,
            "up": key.up,
            "KT": KT,
            "kof_t": s,
            "max_u": max_u,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "rs": lds.rs.ptr,
            "peer": peer,
            "xl": xl.xl.ptr,
            "il": dl.il.ptr,
            "contrib": dl.contrib.ptr,
            "rt": {
                "route": rl.route.ptr,
                "flag": rl.flag.ptr,
                "expert": rl.expert.ptr,
                "kof": rl.kof.ptr,
                "scan": rl.scan.ptr,
            },
            "xbuf": scratch + fx.Int64(scr["xbuf"][0]),
            "xq": scratch + fx.Int64(scr["xq"][0] if key.xq else 0),
            "args": {
                "core": core,
                "gate": gate,
                "residual": residual,
                "w_o": w_o,
                "w_os": w_os,
                "ln_w": ln_w,
                "w_gate": w_gate,
                "w_sg": w_sg,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "out": out,
                "res_out": res_out,
                "w_rm": w_rm,
            },
        }
        for region in LAYER_REGIONS:
            c[region] = sreg(scratch, scr[region][0], region)
        src = (c["xbuf"], CM_DEV, residual)
        is_r = (bid >= R0) & (bid < RT0)
        is_g = (bid >= RT0) & (bid < RT0 + s)
        ck = L.CORE
        # ---- o_proj (both tasks' weights in flight first), then the AR +
        # residual add + norm sums, the router and the shared gate
        if const_expr(key.rm):
            # rm: every CTA runs its o_proj tasks first. CTAs R0 .. R0 + 95
            # then run one rm router task each, with its weights issued only
            # after run_oproj, for the reason given below for the router
            # weights. CTAs R0 .. R0 + 15 then sum every rank's partials.
            moe.run_oproj(
                c, [moe.o_loads(c, bid + BLOCKS * j, ck) for j in range(2)], ck
            )
            if (bid >= R0) & (bid < R0 + RM_TASKS):
                stage_rm(c, bid - R0, rm_loads(c, bid - R0))
                if bid < R0 + R_TASKS:
                    stage_rsum(c, bid - R0)
        else:
            if is_r:
                oops = [moe.o_loads(c, bid + BLOCKS * j, ck) for j in range(2)]
                ri = bid - R0
                moe.run_oproj(c, oops, ck)
                # A router CTA loads its router weights only now. run_oproj
                # loads the core rows, and a wave's wait for a load also
                # waits for every load that it issued before, so router
                # weights issued before run_oproj would delay this CTA's
                # o_proj, and every norm task waits for the o_proj of every
                # CTA. The weights still land before x does.
                rops = router_loads(c, ri // R_PARTS, ri % R_PARTS)
                stage_router(c, ri // R_PARTS, ri % R_PARTS, rops)
            if ~is_r:
                moe.run_oproj(
                    c, [moe.o_loads(c, bid + BLOCKS * j, ck) for j in range(2)], ck
                )
        if bid < NRED:
            N.stage_norm(c, bid)
            if const_expr(key.xq):
                N.load_rstd(c)
                N.stage_xq(c, bid, src)
        if is_g:
            N.load_rstd(c)
            N.stage_sgate(c, bid - RT0, src)
        # ---- the routed experts, then down: two row groups a CTA
        # The router CTAs built their x slice from the tagged XAR and MSQ
        # pairs, which do not show that xbuf is complete. So they poll MSQ2
        # here like every other CTA that has no rstd yet.
        missing = ~is_g
        nu = moe.run_ug(c, key.ps, PS0, lambda: _x_ug(c, src, key.xq, missing, key.rm))
        dops = moe.down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
        if const_expr(key.split):
            cut = moe.early_slots(c, nu)
            moe.load_inter(c, cut, True)
            b2 = bid + BLOCKS
            moe.down_zero(c, 2)
            dops = moe.down_slots(c, bid, nu, dops, 0, cut, (b2, 0, cut))
            dops = moe.down_slots(c, b2, nu, dops, 0, cut, (bid, cut, nu), 1)
            moe.load_inter(c, cut, False)
            dops = moe.down_slots(c, bid, nu, dops, cut, nu, (b2, cut, nu))
            moe.down_slots(c, b2, nu, dops, cut, nu, None, 1)
            moe.down_put(c, bid)
            moe.down_put(c, b2, 1)
        elif const_expr(s * TOPK + 1 <= WAVES * moe.DB):
            # Up to 3 rows one batch of a wave covers the union in each row
            # group. Then the down rounds run without a prefetch, and the
            # second row group's tiles go out here, behind the INTER polls,
            # so the polls do not wait for them.
            moe.load_inter(c)
            dops2 = moe.down_loads(c, bid + BLOCKS, nu, fx.min(c["wave"], nu - 1))
            moe.stage_down(c, bid, nu, dops, prefetch=False, drain=True)
            moe.stage_down(c, bid + BLOCKS, nu, dops2, prefetch=False)
        else:
            moe.load_inter(c)
            dops2 = moe.stage_down(c, bid, nu, dops, bid + BLOCKS)
            moe.stage_down(c, bid + BLOCKS, nu, dops2)
        if bid < NRED:
            N.stage_final(c, bid)

    @flyc.jit
    def launch(
        core: Int64,
        gate: Int64,
        residual: Int64,
        w_o: Int64,
        w_os: Int64,
        ln_w: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        res_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        layer: Int32,
        w_rm: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        klayer(
            core,
            gate,
            residual,
            w_o,
            w_os,
            ln_w,
            w_gate,
            w_sg,
            w13,
            w13s,
            w2,
            w2s,
            out,
            res_out,
            scratch,
            peers,
            rank,
            epoch,
            health,
            alarm,
            layer,
            w_rm,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    LAYER_ABI.check(klayer, launch)
    _LAYER[key] = launch
    return launch


def layer_block(
    key: LayerBuild,
    *,
    core,
    residual,
    w_o,
    ln_w,
    w_gate,
    w_sg,
    w13,
    w13s,
    w2,
    w2s,
    out,
    res_out,
    scratch,
    peers,
    rank,
    epoch,
    health,
    alarm,
    layer,
    w_os=None,
    gate=None,
    w_rm=None,
):
    """out_proj / o_proj and its TP all-reduce, the residual add +
    post_attention_layernorm, the sparse MoE and its TP all-reduce: ``core``
    [S, CORE] bf16 (x sigmoid(``gate``) with ``key.og``) and ``residual`` [S,
    H] -> ``res_out`` (the sum over the ranks + residual) and ``out`` (the
    MoE's output) [S, H] bf16. ``w_o`` as vLLM holds it after loading
    (``key.oq``: fp4x2 with e8m0 ``w_os``, in the layout of the FP4 GEMM
    AITER picked: ``key.shuffled``); the routed experts as the
    AITER_MXFP4_MXFP4 backend holds them (shuffled, e8m0 scales), the shared
    expert fused as expert 512. With ``key.o6``, ``w_o`` and ``w_os`` are
    fp6_pack's MXFP6 weight and scales. ``w_rm``: rm_pack's router weight,
    with ``key.rm`` only."""
    s = key.tokens
    k = L.CORE
    bf = torch.bfloat16
    fp4s = (torch.uint8, torch.float4_e2m1fn_x2)
    ne = E + 1
    rows = (s, H)
    need(core.shape == (s, k) and core.dtype == bf, f"core: bf16 [{s}, {k}]")
    need(
        (gate is not None) == key.og and (gate is None or gate.shape == core.shape),
        "gate: core's shape, with og only",
    )
    for name, t in (("residual", residual), ("out", out), ("res_out", res_out)):
        need(t.shape == rows and t.dtype == bf, f"{name}: bf16 [S, H]")
    need(ln_w.dtype == bf and ln_w.numel() == H, "post_attention_layernorm weight")
    if key.oq:
        need(
            w_o.shape == (H, k // 2)
            and w_os is not None
            and w_os.shape == fp4_scale_shape(H, k, key.shuffled)
            and w_os.element_size() == 1,
            f"o_proj: fp4x2 [{H}, {k // 2}] + e8m0 scales in its layout",
        )
    elif key.o6:
        need(
            w_o.dtype == torch.uint8 and w_o.numel() == H * k * 3 // 4,
            "o_proj: fp6_pack's MXFP6 weight, 8192 x 2048 x 3/4 bytes",
        )
        need(
            w_os is not None
            and w_os.numel() == H * k // 32
            and w_os.element_size() == 1,
            "o_proj scales: e8m0, one a 32-block",
        )
    else:
        need(w_o.shape == (H, k) and w_o.dtype == bf, f"o_proj: bf16 [{H}, {k}]")
    need(w_gate.shape == (E, H) and w_sg.numel() == H, "router shape")
    need(all(t.dtype == bf for t in (w_gate, w_sg)), "router weights: bf16")
    need(
        (w_rm is not None) == key.rm
        and (w_rm is None or (w_rm.shape == (E, RM_CH * RM_K) and w_rm.dtype == bf)),
        f"w_rm: rm_pack's bf16 [{E}, {RM_CH * RM_K}], with rm only",
    )
    need(
        w13.shape == (ne, 2 * L.RIP, H // 2) and w13.dtype in fp4s,
        f"w13: fp4x2 [{ne}, {2 * L.RIP}, {H // 2}]",
    )
    need(
        w2.shape == (ne, H, L.RIP // 2) and w2.dtype in fp4s,
        f"w2: fp4x2 [{ne}, {H}, {L.RIP // 2}]",
    )
    need(
        w13s.numel() == ne * 2 * L.RIP * QL.scale_cols(H) and w13s.element_size() == 1,
        "w13 scales: e8m0, one a 32-block, shuffled",
    )
    need(
        w2s.numel() == ne * H * QL.scale_cols(L.RIP) and w2s.element_size() == 1,
        "w2 scales: e8m0, one a 32-block, shuffled (K blocks padded to 8)",
    )
    used = [core, gate, residual, w_o, w_os, ln_w, w_gate, w_sg, w13, w13s, w2, w2s]
    used += [out, res_out, w_rm]
    need(
        all(t.is_contiguous() for t in used if t is not None), "a non-contiguous input"
    )
    vals = {
        "core": core.data_ptr(),
        "gate": 0 if gate is None else gate.data_ptr(),
        "residual": residual.data_ptr(),
        "w_o": w_o.data_ptr(),
        "w_os": w_os.data_ptr() if (key.oq or key.o6) and w_os is not None else 0,
        "ln_w": ln_w.data_ptr(),
        "w_gate": w_gate.data_ptr(),
        "w_sg": w_sg.data_ptr(),
        "w13": w13.data_ptr(),
        "w13s": w13s.data_ptr(),
        "w2": w2.data_ptr(),
        "w2s": w2s.data_ptr(),
        "out": out.data_ptr(),
        "res_out": res_out.data_ptr(),
        "scratch": scratch.data_ptr(),
        "peers": peers.data_ptr(),
        "rank": rank,
        "epoch": epoch.data_ptr(),
        "health": health.data_ptr(),
        "alarm": alarm,
        "layer": layer,
        "w_rm": 0 if w_rm is None else w_rm.data_ptr(),
    }
    build_layer(key)(*LAYER_ABI.pack(vals), stream=torch.cuda.current_stream())
