# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decoder layer's blocks, each one persistent launch, ``BLOCKS`` x
``THREADS``, S <= ``MAX_TOKENS`` rows:

    gdn    (``linear_attn`` up to out_proj's input): x [S, H] -> in_proj_qkvz
           (MXFP4) / in_proj_ba -> conv1d -> the gated delta rule -> the
           sigmoid-gated norm -> core [S, CORE]
    oproj  (``out_proj`` / ``o_proj``, its TP all-reduce and the attention
           HC's combine): core (x sigmoid(gate)) -> bf16 partials -> every
           rank's AR -> the next state [S, HC H] and its sums of squares
    moe    (``mlp``, its TP all-reduce and the MLP HC's combine): x [S, H] ->
           router, sigmoid shared gate, top-10, the MXFP4 experts with the
           shared expert fused as expert 512 -> every rank's FINAL -> the next
           state and its sums of squares

The combine (``hc.stage_combine``) runs in the all-reduce epilogue: the ranks'
bf16 partials summed in rank order in fp32 -> bf16 block output; state_k +
block 2 sigmoid(inj_k / 4) -> the next state, as ``GatedResidual.combine``
rounds it, so the next ``front.hc_front`` reads a plain state and its norm's
sums of squares. A launch's input is a previous launch's output (plain loads).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.qwen4_exp.amd.specialized_decode.mono import hc
from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.mono import mx_layout as QL
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.debug import region_ids
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.ops import (
    bf_hi,
    bf_lo,
    ld_i32,
    load_ptr64,
    rsrc,
    traced,
    uniform,
)
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
    source_digest,
)
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.ranks import peer_bases
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.sync import (
    Mailbox,
    preg,
    shift,
    sreg,
)
from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import need
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages import gdn as G
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages import gemv, moe
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages import in_proj as P
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages import router as R
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages.gdn import (
    NVCH,
    VCH,
    gdn_tasks,
)
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages.moe import (
    PS_MIN,
    quant_x,
    sigmoid,
    step_tag,
)

H, ROWS, HD = L.HIDDEN, L.ROWS, L.HD
E, TOPK, KT = L.E, L.TOPK, R.KT
OTASKS = H // ROWS  # 160: o_proj / down row groups, CTAs 0 ..
COLS = 128  # output columns a reduction task
NRED = H // COLS  # 20
R0 = OTASKS  # 160: router CTAs
RT0 = R0 + E // ROWS  # 192: shared gate CTAs, a token each
MAXT = L.MAX_TOKENS
PS0 = RT0 + MAXT  # 208: the fused shared expert's early ug CTAs
XROW = H + gemv.LDS_PAD
# mailbox tag slots a layer: gdn ``layer``, the others these + ``layer`` (a
# layer's two HC fronts share one, each on its own scratch)
OPROJ_SLOT = moe.TAG_SLOTS // 4
MOE_SLOT = moe.TAG_SLOTS // 2
HC_SLOT = 3 * moe.TAG_SLOTS // 4

assert NRED <= OTASKS and E == THREADS and OPROJ_SLOT == L.MAX_LAYERS

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class GdnBuild:
    tokens: int
    tp: int = 2
    norm_eps: float = 1e-6
    shuffled: bool = True  # in_proj_qkvz in the ASM FP4 GEMM layout, else Triton's


@dataclass(frozen=True)
class OprojBuild:
    tokens: int
    tp: int = 2
    oq: bool = False  # out_proj MXFP4
    og: bool = False  # o_proj's input is core x sigmoid(gate)
    shuffled: bool = True  # with oq: the ASM FP4 GEMM layout, else Triton's


@dataclass(frozen=True)
class MoeBuild:
    tokens: int
    tp: int = 2

    @property
    def u1(self) -> bool:
        return self.tokens == 1

    @property
    def ps(self) -> bool:
        return self.tokens >= PS_MIN


GDN_ABI = KernelAbi(
    (
        "x",
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
OPROJ_ABI = KernelAbi(
    (
        "core",
        "gate",
        "w_o",
        "w_os",
        "out",
        "state",
        "inj",
        "msq_out",
        "peers",
        "rank",
        "epoch",
        "health",
        "alarm",
        "layer",
    )
)
MOE_ABI = KernelAbi(
    (
        "x",
        "w_gate",
        "w_sg",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "out",
        "state",
        "inj",
        "msq_out",
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


def gdn_scratch_layout(key: GdnBuild) -> dict:
    s, lay = key.tokens, L.Layout(key.tp)
    return pair_layout(
        {
            "proj": s * lay.QKVZ,
            "qk": s * lay.NK * 2 * HD,
            "onrm": s * lay.NV * NVCH,
            "bapart": P.ba_tasks(lay) * s * ROWS,
        }.items()
    )


def moe_scratch_layout(key: MoeBuild) -> dict:
    s, lay = key.tokens, L.Layout(key.tp)
    return pair_layout(
        {
            "rlog": s * E // 2,
            "sgate": s,
            "route": s * 2 * TOPK,
            "inter": s * KT * moe.inter_pairs(lay.moe),
        }.items()
    )


def scratch_bytes(layout: dict) -> int:
    return max(o + n for o, n in layout.values())


PEER_REGIONS = ("ar", "final")


def peer_layout(s, tp):
    """One parity's peer regions: the o_proj and MoE partials of every rank."""
    return pair_layout({r: tp * s * H // 2 for r in PEER_REGIONS}.items())


def peer_bytes(s, tp):
    """The peer buffer: both parities' regions at ``s`` rows."""
    return 2 * scratch_bytes(peer_layout(s, tp))


def _peers(peers, rank, epoch, tp):
    """The peer regions at the widest width, this step's parity half: every
    rank's and this rank's own."""
    play = peer_layout(MAXT, tp)
    half = peer_bytes(MAXT, tp) // 2
    par = uniform(ld_i32(epoch, 0)) % 2
    bases = peer_bases(peers, tp)
    own = load_ptr64(peers, rank)
    peer = {}
    for region in PEER_REGIONS:
        off = play[region][0]
        peer[region] = [
            shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
        ]
        peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
    return peer


def _inj_of(c):
    """The injection logits the block's HC front wrote ([S, HC] bf16)."""

    def get(t):
        w = fx.Vector(
            bo.buffer_load(rsrc(c["args"]["inj_a"]), t * 2, vec_width=2, dtype=T.i32)
        )
        return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]

    return get


def stage_out(c, j, own, out):
    """The block's TP all-reduce at peer region ``own``, then the HC combine of
    the input ``state`` [S, HC H] and ``inj_a`` [S, HC] -> ``out`` [S, HC H] +
    ``msq_out`` [NRED, S, HC]."""
    a = c["args"]

    def msq_put(jj, t, k, v):
        bo.buffer_store(v, rsrc(a["msq_out"]), (jj * c["S"] + t) * L.HC + k)

    hc.stage_combine(c, j, own, _inj_of(c), a["state"], 0, out, 0, msq_put)


# ---------------------------------------------------------------- gdn
_GDN: dict = {}


def build_gdn(key: GdnBuild):
    if key in _GDN:
        return _GDN[key]
    s = key.tokens
    need(1 <= s <= MAXT, f"{s} rows, the kernels take 1..{MAXT}")
    lay = L.Layout(key.tp)
    scr = gdn_scratch_layout(key)
    nr = -(-s * P.gdn_tasks_of(lay) // BLOCKS)
    nba = P.ba_tasks(lay)
    need(P.BA0 + nba <= BLOCKS, "in_proj_ba's CTAs")
    need(lay.QKVZ // ROWS == 2 * BLOCKS, "in_proj_qkvz: two row groups a CTA")
    xwords = s * XROW // 2

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, xwords, 16]

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

    keyed = key_tuple(key, source_digest())

    @flyc.kernel(
        name=f"flashnext_split_gdn_s{s}_t{key.tp}_w{int(key.shuffled)}",
        known_block_size=[THREADS, 1, 1],
    )
    def kgdn(
        x: Int64,
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
            "lay": lay,
            "gdn": lay.gdn,
            "shuffled": key.shuffled,
            "ba_of": P.ba_of,
            "ba_parts": P.BA_PARTS,
            "z_gate": sigmoid,
            "goff": P.gdn_offset(s, key.tp),
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "norm_eps": key.norm_eps,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "xl": xl.xl.ptr,
            "qkl": gl.qkl.ptr,
            "vloc": gl.vloc.ptr,
            "ol": gl.ol.ptr,
            "scal": gl.scal.ptr,
            "args": {
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
        for region in ("proj", "qk", "onrm", "bapart"):
            c[region] = sreg(scratch, scr[region][0], region)
        qops = [P.qkvz_loads(c, bid + BLOCKS * j) for j in range(2)]
        is_ba = (bid >= P.BA0) & (bid < P.BA0 + nba)
        if is_ba:
            bops = G.ba_loads(c, bid - P.BA0, P.BA_PARTS, P.BA_K)
            gemv.rows_to_lds(tid, x, H, s, c["xl"], XROW)
            G.stage_ba(c, bid - P.BA0, bops, P.BA_PARTS, P.BA_K)
        if ~is_ba:
            gemv.rows_to_lds(tid, x, H, s, c["xl"], XROW)
        quant_x(c, H)
        for j in range_constexpr(2):
            P.stage_qkvz(c, bid + BLOCKS * j, qops[j])
        gdn_tasks(c, nr)

    @flyc.jit
    def launch(
        x: Int64,
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
            x,
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
    x,
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
    """A GDN layer's ``linear_attn`` up to out_proj's input: ``x`` [S, H] bf16
    (the block input) -> ``core`` [S, CORE] bf16. Every weight as vLLM holds
    it after loading (in_proj_qkvz online MXFP4 with e8m0 scales, in the
    layout of the FP4 GEMM AITER picked: ``key.shuffled``). ``conv_state``
    [slots, CONV, 3] or a transposed view of [slots, 3, CONV] (strides read);
    ``rstate`` [slots, NV, 128, 128] fp32, a slot dense, slots strided;
    ``st_idx`` [>= S] int32, <= 0 a pad row (no state written, a zero core
    row); ``epoch`` a device int32 bumped once a step, ``layer`` this
    launch's tag slot."""
    s = key.tokens
    lay = L.Layout(key.tp)
    bf = torch.bfloat16
    need(x.shape == (s, H) and x.dtype == bf and x.is_contiguous(), "x: bf16 [S, H]")
    need(
        core.shape == (s, lay.CORE) and core.dtype == bf, f"core: bf16 [S, {lay.CORE}]"
    )
    need(
        w_qkvz.shape == (lay.QKVZ, H // 2)
        and w_qkvz.dtype in (torch.uint8, torch.float4_e2m1fn_x2),
        f"in_proj_qkvz: fp4x2 [{lay.QKVZ}, {H // 2}]",
    )
    need(
        w_qkvzs.shape == fp4_scale_shape(lay.QKVZ, H, key.shuffled)
        and w_qkvzs.element_size() == 1,
        "in_proj_qkvz scales: e8m0, one a 32-block, in the weight's layout",
    )
    need(
        w_ba.shape == (lay.BA, H) and w_ba.dtype == bf,
        f"in_proj_ba: bf16 [{lay.BA}, {H}]",
    )
    need(conv_w.numel() == lay.CONV * L.CONV_W and conv_w.dtype == bf, "conv1d weight")
    need(
        conv_state.dtype == bf and conv_state.shape[1:] == (lay.CONV, P.SL),
        f"conv state: bf16 [slots, {lay.CONV}, 3]",
    )
    need(
        rstate.dtype == torch.float32 and rstate.shape[1:] == (lay.NV, HD, HD),
        f"recurrent state: fp32 [slots, {lay.NV}, 128, 128]",
    )
    need(rstate[0].is_contiguous(), "a recurrent state slot must be dense")
    need(a_log.dtype == torch.float32 and dt_bias.dtype == bf, "A_log / dt_bias dtype")
    need(norm_w.dtype == bf and norm_w.numel() == HD, "gated norm weight")
    need(st_idx.dtype == torch.int32 and st_idx.numel() >= s, "state indices")
    need(
        all(t.is_contiguous() for t in (w_qkvz, w_qkvzs, w_ba, conv_w, core)),
        "a non-contiguous GDN input",
    )
    vals = {
        "x": x.data_ptr(),
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


# ---------------------------------------------------------------- oproj
_OPROJ: dict = {}


def build_oproj(key: OprojBuild):
    if key in _OPROJ:
        return _OPROJ[key]
    s, tp = key.tokens, key.tp
    need(1 <= s <= MAXT, f"{s} rows, the kernels take 1..{MAXT}")
    lay = L.Layout(tp)
    xwords = s * (lay.CORE + gemv.LDS_PAD) // 2

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        xl: fx.Array[fx.Int32, xwords, 16]

    keyed = key_tuple(key, source_digest())
    name = (
        f"flashnext_split_oproj_s{s}_t{tp}_o{int(key.oq)}_a{int(key.og)}"
        f"_w{int(key.shuffled)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def koproj(
        core: Int64,
        gate: Int64,
        w_o: Int64,
        w_os: Int64,
        out: Int64,
        state: Int64,
        inj: Int64,
        msq_out: Int64,
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
        lds = fx.SharedAllocator().allocate(Smem).peek()
        mb = Mailbox(
            step_tag(epoch, layer) - 1, health, alarm, region_ids(PEER_REGIONS)
        )
        c = {
            "S": s,
            "TP": tp,
            "lay": lay,
            "moe": lay.moe,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "oq": key.oq,
            "og": key.og,
            "shuffled": key.shuffled,
            "put_bf": mb.put_bf,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "xl": lds.xl.ptr,
            "peer": _peers(peers, rank, epoch, tp),
            "args": {
                "core": core,
                "gate": gate,
                "w_o": w_o,
                "w_os": w_os,
                "state": state,
                "inj_a": inj,
                "msq_out": msq_out,
            },
        }
        if bid < OTASKS:
            oops = moe.o_loads(c, bid, lay.CORE)
            moe.run_oproj(c, [oops], lay.CORE)
        if bid < NRED:
            stage_out(c, bid, c["peer"]["ar_own"], out)

    @flyc.jit
    def launch(
        core: Int64,
        gate: Int64,
        w_o: Int64,
        w_os: Int64,
        out: Int64,
        state: Int64,
        inj: Int64,
        msq_out: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        koproj(
            core,
            gate,
            w_o,
            w_os,
            out,
            state,
            inj,
            msq_out,
            peers,
            rank,
            epoch,
            health,
            alarm,
            layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    OPROJ_ABI.check(koproj, launch)
    _OPROJ[key] = launch
    return launch


def _hc_vals(key, out, state, inj, msq_out) -> dict:
    """The HC combine's tensors -> their ABI values."""
    s, bf = key.tokens, torch.bfloat16
    need(
        out.shape == state.shape == (s, L.HH) and out.dtype == state.dtype == bf,
        "out, state: bf16 [S, HC H]",
    )
    need(inj.shape == (s, L.HC) and inj.dtype == bf, "inj: bf16 [S, HC]")
    need(
        msq_out.shape == (NRED, s, L.HC) and msq_out.dtype == torch.float32,
        "msq_out: fp32 [NRED, S, HC]",
    )
    need(all(t.is_contiguous() for t in (state, inj, msq_out)), "a non-contiguous HC")
    return {
        "state": state.data_ptr(),
        "inj": inj.data_ptr(),
        "msq_out": msq_out.data_ptr(),
    }


def oproj_block(
    key: OprojBuild,
    *,
    core,
    w_o,
    state,
    inj,
    out,
    msq_out,
    peers,
    rank,
    epoch,
    health,
    alarm,
    layer,
    w_os=None,
    gate=None,
):
    """``out_proj`` / ``o_proj``, its TP all-reduce and the attention HC's
    combine: ``core`` [S, CORE] bf16 (x sigmoid(``gate``) with ``key.og``) ->
    the sum over the ranks, combined into ``state`` [S, HC H] with the
    injection logits ``inj`` [S, HC] -> ``out`` [S, HC H] and its sums of
    squares ``msq_out`` [NRED, S, HC] fp32. ``w_o`` as vLLM
    holds it after loading (``key.oq``: fp4x2 with e8m0 ``w_os``, in the
    layout of the FP4 GEMM AITER picked: ``key.shuffled``)."""
    s = key.tokens
    k = L.Layout(key.tp).CORE
    bf = torch.bfloat16
    need(core.shape == (s, k) and core.dtype == bf, f"core: bf16 [{s}, {k}]")
    hcv = _hc_vals(key, out, state, inj, msq_out)
    need(
        (gate is not None) == key.og and (gate is None or gate.shape == core.shape),
        "gate: core's shape, with og only",
    )
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
    used = [t for t in (core, gate, w_o, w_os, out) if t is not None]
    need(all(t.is_contiguous() for t in used), "a non-contiguous input")
    vals = {
        "core": core.data_ptr(),
        "gate": 0 if gate is None else gate.data_ptr(),
        "w_o": w_o.data_ptr(),
        "w_os": w_os.data_ptr() if key.oq else 0,
        "out": out.data_ptr(),
        **hcv,
        "peers": peers.data_ptr(),
        "rank": rank,
        "epoch": epoch.data_ptr(),
        "health": health.data_ptr(),
        "alarm": alarm,
        "layer": layer,
    }
    build_oproj(key)(*OPROJ_ABI.pack(vals), stream=torch.cuda.current_stream())


# ---------------------------------------------------------------- moe
_MOE: dict = {}


@traced
def _x_to_lds(c, missing):
    if missing:
        gemv.rows_to_lds(c["tid"], c["args"]["x"], H, c["S"], c["xl"], XROW)


def build_moe(key: MoeBuild):
    if key in _MOE:
        return _MOE[key]
    s, tp = key.tokens, key.tp
    need(1 <= s <= MAXT, f"{s} rows, the kernels take 1..{MAXT}")
    lay = L.Layout(tp)
    scr = moe_scratch_layout(key)
    xwords = s * XROW // 2
    max_u = s * TOPK + 1  # distinct experts the step can touch + the fused shared
    need(PS0 + lay.RI // moe.UG_COLS <= BLOCKS, "the shared expert's early ug CTAs")

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * KT, 16]
        flag: fx.Array[fx.Int32, E, 16]
        expert: fx.Array[fx.Int32, max_u, 16]
        kof: fx.Array[fx.Int32, max_u * s, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, xwords, 16]

    @fx.struct
    class DownLds:
        il: fx.Array[fx.Int32, s * KT * (moe.irow(lay.moe) // 4 + lay.RIP // 32), 16]
        contrib: fx.Array[fx.Float32, s * KT * ROWS, 16]

    @fx.union
    class StageLds:
        x: XLds
        down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]

    keyed = key_tuple(key, source_digest())

    @flyc.kernel(
        name=f"flashnext_split_moe_s{s}_t{tp}",
        known_block_size=[THREADS, 1, 1],
    )
    def kmoe(
        x: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        state: Int64,
        inj: Int64,
        msq_out: Int64,
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
            step_tag(epoch, layer) - 1, health, alarm, region_ids((*scr, *PEER_REGIONS))
        )
        c = {
            "S": s,
            "TP": tp,
            "lay": lay,
            "moe": lay.moe,
            "route_cta": RT0,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "oq": False,
            "og": False,
            "shuffled": True,
            "u1": key.u1,
            "KT": KT,
            "kof_t": s,
            "max_u": max_u,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "peer": _peers(peers, rank, epoch, tp),
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
            "args": {
                "x": x,
                "w_gate": w_gate,
                "w_sg": w_sg,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "state": state,
                "inj_a": inj,
                "msq_out": msq_out,
            },
        }
        for region in ("rlog", "sgate", "route", "inter"):
            c[region] = sreg(scratch, scr[region][0], region)
        rlog = c["rlog"]
        c["rlog_at"] = lambda t, lane, i: (rlog, (t * E + lane + 64 * i) // 2, 1)
        c["rlog_last"] = (1, lambda i: (rlog, (s * E - 2) // 2, 1))
        is_r = (bid >= R0) & (bid < RT0)
        is_g = (bid >= RT0) & (bid < RT0 + s)
        if is_r:
            rops = R.router_loads(c, bid - R0)
            _x_to_lds(c, is_r)
            R.stage_router(c, bid - R0, rops)
        if is_g:
            _x_to_lds(c, is_g)
            R.stage_sgate(c, bid - RT0)
        nu = moe.run_ug(c, key.ps, PS0, lambda: _x_to_lds(c, ~(is_r | is_g)))
        if bid < OTASKS:
            dops = moe.down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
            moe.load_inter(c)
            moe.stage_down(c, bid, nu, dops)
        if bid < NRED:
            stage_out(c, bid, c["peer"]["final_own"], out)

    @flyc.jit
    def launch(
        x: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        state: Int64,
        inj: Int64,
        msq_out: Int64,
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
        kmoe(
            x,
            w_gate,
            w_sg,
            w13,
            w13s,
            w2,
            w2s,
            out,
            state,
            inj,
            msq_out,
            scratch,
            peers,
            rank,
            epoch,
            health,
            alarm,
            layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    MOE_ABI.check(kmoe, launch)
    _MOE[key] = launch
    return launch


def moe_block(
    key: MoeBuild,
    *,
    x,
    w_gate,
    w_sg,
    w13,
    w13s,
    w2,
    w2s,
    state,
    inj,
    out,
    msq_out,
    scratch,
    peers,
    rank,
    epoch,
    health,
    alarm,
    layer,
):
    """The sparse MoE block, its TP all-reduce and the MLP HC's combine: ``x``
    [S, H] bf16 -> the sum over the ranks, combined into ``state`` with ``inj``
    as ``oproj_block`` does. The routed experts as the AITER_MXFP4_MXFP4
    backend holds them after loading (shuffled, e8m0 scales), the shared
    expert fused as expert 512."""
    s = key.tokens
    lay = L.Layout(key.tp)
    bf = torch.bfloat16
    fp4s = (torch.uint8, torch.float4_e2m1fn_x2)
    ne = E + 1
    need(x.shape == (s, H) and x.dtype == bf, "x: bf16 [S, H]")
    hcv = _hc_vals(key, out, state, inj, msq_out)
    need(w_gate.shape == (E, H) and w_sg.numel() == H, "router shape")
    need(
        w13.shape == (ne, 2 * lay.RIP, H // 2) and w13.dtype in fp4s,
        f"w13: fp4x2 [{ne}, {2 * lay.RIP}, {H // 2}]",
    )
    need(
        w2.shape == (ne, H, lay.RIP // 2) and w2.dtype in fp4s,
        f"w2: fp4x2 [{ne}, {H}, {lay.RIP // 2}]",
    )
    need(
        w13s.numel() == ne * 2 * lay.RIP * QL.scale_cols(H)
        and w13s.element_size() == 1,
        "w13 scales: e8m0, one a 32-block, shuffled",
    )
    need(
        w2s.numel() == ne * H * QL.scale_cols(lay.RIP) and w2s.element_size() == 1,
        "w2 scales: e8m0, one a 32-block, shuffled (K blocks padded to 8)",
    )
    tensors = [x, out, w_gate, w_sg, w13, w13s, w2, w2s]
    need(all(t.is_contiguous() for t in tensors), "a non-contiguous input")
    need(all(t.dtype == bf for t in (w_gate, w_sg)), "router weights: bf16")
    vals = {
        "x": x.data_ptr(),
        "w_gate": w_gate.data_ptr(),
        "w_sg": w_sg.data_ptr(),
        "w13": w13.data_ptr(),
        "w13s": w13s.data_ptr(),
        "w2": w2.data_ptr(),
        "w2s": w2s.data_ptr(),
        "out": out.data_ptr(),
        **hcv,
        "scratch": scratch.data_ptr(),
        "peers": peers.data_ptr(),
        "rank": rank,
        "epoch": epoch.data_ptr(),
        "health": health.data_ptr(),
        "alarm": alarm,
        "layer": layer,
    }
    build_moe(key)(*MOE_ABI.pack(vals), stream=torch.cuda.current_stream())
