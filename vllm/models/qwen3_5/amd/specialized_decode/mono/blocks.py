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

    @property
    def xq(self) -> bool:
        return self.tokens >= GDN_XQ_MIN

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

    @property
    def u1(self) -> bool:
        return self.tokens == 1

    @property
    def ps(self) -> bool:
        return self.tokens >= PS_MIN

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


LAYER_REGIONS = ("msq", "xqf", "rpart", "sgate", "route", "inter")


def layer_scratch_layout(key: LayerBuild) -> dict:
    """The mailbox regions, then ``xbuf``: the AR sums [S, H] bf16 the norm
    tasks write for the consumers' x (16 B aligned)."""
    s = key.tokens
    lay = pair_layout(
        {
            "msq": NRED * s,
            "xqf": NRED,
            "rpart": R_TASKS * s * ROWS,
            "sgate": s,
            "route": s * 2 * TOPK,
            "inter": s * KT * moe.inter_pairs(L.MOE),
        }.items()
    )
    lay |= pair_layout([("xbuf", s * H * 2 // 8)], 16, scratch_bytes(lay))
    lay = _xq_layout(lay, key)
    return lay


def scratch_bytes(layout: dict) -> int:
    return max(o + n for o, n in layout.values())


PEER_REGIONS = ("ar", "rlog", "final")


def peer_layout(s):
    """One parity's peer regions: the o_proj partials, the router logits and
    the MoE partials of every rank."""
    return pair_layout(
        {
            "ar": TP * s * H // 2,
            "rlog": TP * s * EPR // 2,
            "final": TP * s * H // 2,
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
    a = c["args"]
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
    """in_proj_qkvz rows 16 rg .. on the x rows' MXFP4 codes -> bf16 PROJ."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    moe.put_acc(c, moe.fp4q_gemv(lane, wave, c["xl"], H, s, rg, ops, c["shuffled"]))
    if tid < ROWS * s:
        r = tid % ROWS
        j = tid // ROWS
        v = bf16_round(row_sum(c["red"], r, j))
        c["put"](c["proj"], j * L.QKVZ + rg * ROWS + r, v)
    gpu.barrier()


def build_gdn(key: GdnBuild):
    if key in _GDN:
        return _GDN[key]
    s = key.tokens
    need(1 <= s <= MAXT, f"{s} rows, the kernels take 1..{MAXT}")
    scr = gdn_scratch_layout(key)
    nr = -(-s * GDN_TASKS // BLOCKS)

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, _xwords(s), 16]

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
    name = f"qwen38_sd_gdn_s{s}_f{int(key.first)}_w{int(key.shuffled)}"

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
        rg = BLOCKS - 1 - bid
        two = bid >= QQ0
        qops = qkvz_loads(c, rg)
        qops2 = qkvz_loads(c, fx.min(2 * BLOCKS - 1 - bid, QTASKS - 1), two)
        is_ba = bid < BA_TASKS
        bops = G.ba_loads(c, fx.min(bid, BA_TASKS - 1), BA_PARTS, BA_K)
        if bid < NRED:
            N.stage_front(c, bid, src, not key.local)
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
def stage_router(c, i, q, ops, src):
    """Router task (i, q) on x's K quarter q (``rs`` loaded). Parts q > 0
    publish fp32 partials; part 0 adds them in part order, rounds to bf16
    (GateLinear's output) and puts the logits to every rank's RLOG."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    N.x_rows_to_lds(c, src, R_K, q * R_K)
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
def _x_ug(c, src, xq, missing):
    """The routed experts' x -> LDS: the shared MXFP4 image (``xq``), else
    built here (``_x_fp4``)."""
    if const_expr(xq):
        N.xq_to_lds(c)
    else:
        _x_fp4(c, src, missing)


def build_layer(key: LayerBuild):
    if key in _LAYER:
        return _LAYER[key]
    s = key.tokens
    need(1 <= s <= MAXT, f"{s} rows, the kernels take 1..{MAXT}")
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
        xl: fx.Array[fx.Int32, max(_xwords(s), s * (L.CORE + gemv.LDS_PAD) // 2), 16]

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
    name = f"qwen38_sd_layer_s{s}_o{int(key.oq)}_a{int(key.og)}_w{int(key.shuffled)}"

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
        c = {
            "S": s,
            "TP": TP,
            "moe": d,
            "rlog_at": lambda t, lane, i: _rlog_at(rlog, s, t, lane + 64 * i),
            "rlog_last": (
                TP,
                lambda i: (rlog, ((i * s + s - 1) * EPR + EPR - 2) // 2, 1),
            ),
            "route_cta": RT0,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "eps": key.eps,
            "oq": key.oq,
            "og": key.og,
            "shuffled": key.shuffled,
            "u1": key.u1,
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
            },
        }
        for region in LAYER_REGIONS:
            c[region] = sreg(scratch, scr[region][0], region)
        src = (c["xbuf"], CM_DEV, residual)
        is_r = (bid >= R0) & (bid < RT0)
        is_g = (bid >= RT0) & (bid < RT0 + s)
        ck = L.CORE
        # ---- o_proj (both tasks' weights in flight first; a router CTA's
        # router weights behind them), then the AR + residual add + norm sums,
        # the router and the shared gate
        if is_r:
            oops = [moe.o_loads(c, bid + BLOCKS * j, ck) for j in range(2)]
            ri = bid - R0
            rops = router_loads(c, ri // R_PARTS, ri % R_PARTS)
            moe.run_oproj(c, oops, ck)
            N.load_rstd(c)
            stage_router(c, ri // R_PARTS, ri % R_PARTS, rops, src)
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
        missing = ~(is_r | is_g)
        nu = moe.run_ug(c, key.ps, PS0, lambda: _x_ug(c, src, key.xq, missing))
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
):
    """out_proj / o_proj and its TP all-reduce, the residual add +
    post_attention_layernorm, the sparse MoE and its TP all-reduce: ``core``
    [S, CORE] bf16 (x sigmoid(``gate``) with ``key.og``) and ``residual`` [S,
    H] -> ``res_out`` (the sum over the ranks + residual) and ``out`` (the
    MoE's output) [S, H] bf16. ``w_o`` as vLLM holds it after loading
    (``key.oq``: fp4x2 with e8m0 ``w_os``, in the layout of the FP4 GEMM
    AITER picked: ``key.shuffled``); the routed experts as the
    AITER_MXFP4_MXFP4 backend holds them (shuffled, e8m0 scales), the shared
    expert fused as expert 512."""
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
    else:
        need(w_o.shape == (H, k) and w_o.dtype == bf, f"o_proj: bf16 [{H}, {k}]")
    need(w_gate.shape == (E, H) and w_sg.numel() == H, "router shape")
    need(all(t.dtype == bf for t in (w_gate, w_sg)), "router weights: bf16")
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
    used += [out, res_out]
    need(
        all(t.is_contiguous() for t in used if t is not None), "a non-contiguous input"
    )
    vals = {
        "core": core.data_ptr(),
        "gate": 0 if gate is None else gate.data_ptr(),
        "residual": residual.data_ptr(),
        "w_o": w_o.data_ptr(),
        "w_os": w_os.data_ptr() if key.oq else 0,
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
    }
    build_layer(key)(*LAYER_ABI.pack(vals), stream=torch.cuda.current_stream())
