# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 ``layer_post``: one Flash-Next decoder layer from the attention's core
output (the out_proj / o_proj input, the same width on GDN and QSA layers) to
the layer's materialized multi-stream state, every TP reduction inside the
kernel. One launch a layer and rank, ``BLOCKS`` x ``THREADS``, S <= 8 rows:

    o_proj   (160 x 16 rows, K 3072 a rank; CTAs 0 .. 159): GEMV of the core
             rows -> bf16 partial -> every rank's AR
    combine  (CTAs 0 .. 19, ``hc.stage_combine``): AR in rank order -> attn
             out; state + attn 2 sigmoid(inj_a / 4) -> STATE1; MSQ
    hc down  (168 tasks, CTAs 88 .. 255) and hc up (160, CTAs 0 .. 159): the
             mlp_hyper_connection's mix of STATE1 -> BIN (block input), its
             injection logits -> INJ
    router   (32 x 16 experts, CTAs 160 .. 191): bf16 logits of BIN -> RLOG
    sgate    (CTAs 192 .. 192+S-1, token t): sigmoid(bf16(x . w_sg)) -> SGATE
    route    (every CTA, wave t token t): the 512 logits' top-10 in descending
             order, softmax-renormalized (``_route_mp``)
    ug       (|U| x 20 tasks, every CTA): the union U's w13 rows (the fused
             shared expert its last slot) against x's MXFP4 codes; silu(g) u
             -> INTER a pick
    down     (160 x 16 hidden rows, CTAs 0 .. 159): w2 of U against the picks'
             INTER through MXFP4, times the route weight, summed in top-k
             order -> bf16 partial -> every rank's FINAL
    final    (CTAs 0 .. 19): FINAL in rank order -> moe out; STATE1 + moe 2
             sigmoid(INJ / 4) -> ``out``; its sums of squares -> ``msq_out``
             (the next layer's front reads them)

The routed experts' numerics are the Qwen3.8 mono's (the stock one-stage
MXFP4 decode MoE: x and silu(g) u through MXFP4, RCEIL scales), with the
intermediate the weights hold padded from RI to RIP: the pad columns' w13 rows
are not run and their INTER codes are zero. The shared expert is fused as
expert 512 (``VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS``), weighted by
sigmoid(bf16(x . w_sg)) in fp32. Peer regions are double-buffered by the step
epoch's parity.

Variants: ``oq`` (an online-MXFP4 out_proj on the ASM FP4 GEMM: the core rows
through MXFP4, as ``per_1x32_f4_quant_hip`` quantizes the GEMM's input) and
``og`` (o_proj's input is core x sigmoid(gate), the QSA output gate).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_DEV,
    bf16_round,
    bf_hi,
    bf_lo,
    ld_i32,
    load_ptr64,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_sum,
)
from vllm.models.kimi_k3.amd.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from vllm.models.kimi_k3.amd.mono.common.ranks import peer_bases
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, preg, shift, sreg
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono import layout as QL
from vllm.models.qwen3_5.amd.mono import moe
from vllm.models.qwen3_5.amd.mono.moe import MAX_U, PS_MIN, f32_of, sigmoid, step_tag
from vllm.models.qwen4_exp.amd.mono import hc
from vllm.models.qwen4_exp.amd.mono import layout as L
from vllm.models.qwen4_exp.amd.mono.layout import need
from vllm.models.qwen4_exp.amd.mono.sources import digest

H, HH, HC, ROWS = L.HIDDEN, L.HH, L.HC, L.ROWS
E, TOPK = L.E, L.TOPK
KT = TOPK + 1  # picks a token: the top-k + the fused shared expert
OTASKS = H // ROWS  # 160 o_proj / down row groups, CTAs 0 ..
D0 = BLOCKS - hc.DOWN_TASKS  # 88: the hc down CTAs, to the last
R_TASKS = E // ROWS  # 32
R0 = OTASKS  # 160: router CTAs
RT0 = R0 + R_TASKS  # 192: shared gate CTAs, a token each
PS0 = RT0 + L.MAX_TOKENS  # 200: the fused shared expert's early ug CTAs
TAG_SLOTS = moe.TAG_SLOTS
XROW = H + gemv.LDS_PAD

assert hc.UP_TASKS == OTASKS and D0 <= R0 and hc.NRED <= D0
assert E == THREADS, "load_route: an expert a thread"
assert (E, TOPK, L.MAX_TOKENS) == (moe.E, moe.TOPK, moe.MAX_TOKENS)

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K2Build:
    tokens: int
    tp: int = 2
    eps: float = 1e-6  # the HC norms
    oq: bool = False  # out_proj MXFP4 (ASM FP4 GEMM layout)
    og: bool = False  # o_proj's input is core x sigmoid(gate)

    @property
    def u1(self) -> bool:
        """One row: U is the token's picks in top-k order (no union pass)."""
        return self.tokens == 1

    @property
    def ps(self) -> bool:
        """The fused shared expert's ug tasks run before the route."""
        return self.tokens >= PS_MIN


ABI = KernelAbi(
    (
        "core",
        "gate",
        "state",
        "inj_a",
        "w_o",
        "w_os",
        "hc_w",
        "w_down",
        "w_up",
        "w_gate",
        "w_sg",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "out",
        "msq_out",
        "scratch",
        "peers",
        "rank",
        "epoch",
        "layer",
    )
)


def peer_pairs(s, tp):
    """One parity's peer regions, pairs: name -> count."""
    return {"ar": tp * s * H // 2, "final": tp * s * H // 2}


def peer_layout(s, tp):
    return pair_layout(peer_pairs(s, tp).items())


def peer_bytes(s, tp):
    lay = peer_layout(s, tp)
    return 2 * max(o + n for o, n in lay.values())


def scratch_layout(key: K2Build) -> dict:
    s = key.tokens
    lay = L.Layout(key.tp)
    pairs = hc.regions(s) | {
        "inj": s * HC,
        "rlog": s * E // 2,
        "sgate": s,
        "route": s * 2 * TOPK,
        "inter": s * KT * lay.RI,
    }
    out = pair_layout(pairs.items())
    end = max(o + n for o, n in out.values())
    for name, n in (("state1", s * HH * 2), ("bin", hc.bin_bytes(s))):
        end = (end + 15) // 16 * 16
        out[name] = (end, n)
        end += n
    return out


def scratch_bytes(key: K2Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


# ---------------------------------------------------------------- combines
def inj_a_of(c):
    """The attention injection logits a previous launch wrote ([S, HC] bf16)."""

    def get(t):
        w = fx.Vector(
            bo.buffer_load(rsrc(c["args"]["inj_a"]), t * 2, vec_width=2, dtype=T.i32)
        )
        return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]

    return get


def inj_m_of(c):
    """The mlp injection logits this launch's hc down put to INJ."""

    def get(t):
        got = c["poll"]([(c["inj"], t * HC, 2), (c["inj"], t * HC + 2, 2)])
        return [f32_of(got[k // 2][k % 2]) for k in range(HC)]

    return get


def stage_attn_combine(c, j):
    a = c["args"]
    hc.stage_combine(
        c,
        j,
        c["peer"]["ar_own"],
        inj_a_of(c),
        a["state"],
        0,
        c["state1"],
        CM_DEV,
        lambda jj, t, k, v: c["put"](c["hc"]["msq"], (jj * c["S"] + t) * HC + k, v),
    )


def stage_final(c, j):
    a = c["args"]

    def msq_put(jj, t, k, v):
        bo.buffer_store(v, rsrc(a["msq_out"]), (jj * c["S"] + t) * HC + k)

    hc.stage_combine(
        c,
        j,
        c["peer"]["final_own"],
        inj_m_of(c),
        c["state1"],
        CM_DEV,
        a["out"],
        0,
        msq_put,
    )


# ---------------------------------------------------------------- router / sgate
def router_loads(c, i):
    return gemv.loads(c["lane"], c["wave"], c["args"]["w_gate"], i * ROWS, H)


@traced
def stage_router(c, i, ops):
    """Experts 16 i ..: bf16 logits of the block-input rows in LDS -> RLOG."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    acc = gemv.mfmas(lane, wave, c["xl"], XROW, H, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        r = 2 * (tid % (ROWS // 2))
        t = tid // (ROWS // 2)
        vs = [bf16_round(row_sum(red, r + d, t)) for d in range(2)]
        c["put_bf"](c["rlog"], t * E + i * ROWS + r, vs)
    gpu.barrier()


@traced
def stage_sgate(c, t):
    """Token t's shared-expert weight: sigmoid(bf16(x . w_sg)) in fp32 (the
    top-k kernel's fused sigmoid on the combined gate's bf16 logit) -> SGATE."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    n = H // 8
    kk = fx.min(tid, n - 1) * 8
    xw = fx.Vector(
        fx.ptr_load(
            c["xl"] + (t * XROW + kk) // 2,
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    ww = fx.Vector(bo.buffer_load(rsrc(a["w_sg"]), kk // 2, vec_width=4, dtype=T.i32))
    acc = fx.Float32(0.0)
    for d in range_constexpr(4):
        acc = acc + bf_lo(xw[d]) * bf_lo(ww[d]) + bf_hi(xw[d]) * bf_hi(ww[d])
    acc = wave_sum((tid < n).select(acc, fx.Float32(0.0)))
    if lane == 0:
        fx.ptr_store(acc, red + wave)
    gpu.barrier()
    if tid == 0:
        tot = fx.ptr_load(red + 0)
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(red + w)
        c["put"](c["sgate"], t, sigmoid(bf16_round(tot)))
    gpu.barrier()


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: K2Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernels take 1..{L.MAX_TOKENS}")
    lay = L.Layout(key.tp)
    tp = key.tp
    scr = scratch_layout(key)
    # every width at the widest's offsets: a step's parity half never overlaps
    # the other half of a step of another width
    play = peer_layout(L.MAX_TOKENS, tp)
    half = peer_bytes(L.MAX_TOKENS, tp) // 2
    xwords = s * (max(lay.CORE, H, hc.DK) + gemv.LDS_PAD) // 2
    need(PS0 + lay.RI // moe.UG_COLS <= BLOCKS, "the shared expert's early ug CTAs")
    need(lay.RI % 32 == 0 and lay.RIP % L.FP4_STEP == 0, "the routed intermediate")

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * KT, 16]
        flag: fx.Array[fx.Int32, E, 16]
        expert: fx.Array[fx.Int32, MAX_U, 16]
        kof: fx.Array[fx.Int32, MAX_U * 8, 16]
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
        rs: fx.Array[fx.Float32, L.MAX_TOKENS * HC, 16]
        ll: fx.Array[fx.Int32, s * hc.LROW // 2, 16]

    keyed = key_tuple(key, digest())
    name = f"flashnext_mono_k2_s{s}_t{tp}_o{int(key.oq)}_a{int(key.og)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64,
        gate: Int64,
        state: Int64,
        inj_a: Int64,
        w_o: Int64,
        w_os: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        msq_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
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
        mb = Mailbox(step_tag(epoch, layer) - 1)
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, tp)
        own = load_ptr64(peers, rank)
        peer = {}
        for region in ("ar", "final"):
            off = play[region][0]
            peer[region] = [
                shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
            ]
            peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
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
            "eps": key.eps,
            "fse": True,
            "oq": key.oq,
            "og": key.og,
            "u1": key.u1,
            "KT": KT,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "rs": lds.rs.ptr,
            "ll": lds.ll.ptr,
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
            "state1": scratch + fx.Int64(scr["state1"][0]),
            "args": {
                "core": core,
                "gate": gate,
                "state": state,
                "inj_a": inj_a,
                "w_o": w_o,
                "w_os": w_os,
                "w_gate": w_gate,
                "w_sg": w_sg,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "out": out,
                "msq_out": msq_out,
            },
        }
        for region in ("inj", "rlog", "sgate", "route", "inter"):
            c[region] = sreg(scratch, scr[region][0], region)
        # every rank's router computes all 512 logits: RLOG is local
        rlog = c["rlog"]
        c["rlog_at"] = lambda t, lane, i: (rlog, (t * E + lane + 64 * i) // 2, 1)
        c["rlog_last"] = (1, lambda i: (rlog, (s * E - 2) // 2, 1))
        c["hc"] = {
            "state": c["state1"],
            "state_cm": CM_DEV,
            "w_norm": hc_w,
            "w_down": w_down,
            "w_up": w_up,
            "msq": sreg(scratch, scr["msq"][0], "msq"),
            "msq_in": None,
            "dpart": sreg(scratch, scr["dpart"][0], "dpart"),
            "lora": sreg(scratch, scr["lora"][0], "lora"),
            "binf": sreg(scratch, scr["binf"][0], "binf"),
            "bin": scratch + fx.Int64(scr["bin"][0]),
            "inj_out": lambda t, r, vs: _put_inj(c, t, r, vs),
        }
        is_r = (bid >= R0) & (bid < RT0)
        is_g = (bid >= RT0) & (bid < RT0 + s)
        # ---- o_proj, the attention combine, the mlp HC mix, router / sgate:
        # a CTA's whole sequence in one branch, its weights in flight first
        if bid < D0:
            oops = moe.o_loads(c, bid, lay.CORE)
            moe.run_oproj(c, [oops], lay.CORE)
            if bid < hc.NRED:
                stage_attn_combine(c, bid)
            uops = hc.up_loads(c, bid)
            hc.load_rrms(c)
            hc.stage_up(c, bid, uops)
        if (bid >= D0) & (bid < OTASKS):
            oops = moe.o_loads(c, bid, lay.CORE)
            dops = hc.down_loads(c, bid - D0)
            moe.run_oproj(c, [oops], lay.CORE)
            uops = hc.up_loads(c, bid)
            hc.load_rrms(c)
            hc.stage_down(c, bid - D0, dops)
            hc.stage_up(c, bid, uops)
        if is_r:
            dops = hc.down_loads(c, bid - D0)
            rops = router_loads(c, bid - R0)
            hc.load_rrms(c)
            hc.stage_down(c, bid - D0, dops)
            hc.bin_to_lds(c, XROW)
            stage_router(c, bid - R0, rops)
        if bid >= RT0:
            dops = hc.down_loads(c, bid - D0)
            hc.load_rrms(c)
            hc.stage_down(c, bid - D0, dops)
            if is_g:
                hc.bin_to_lds(c, XROW)
                stage_sgate(c, bid - RT0)
        # ---- the routed experts (x: the router / sgate CTAs' is in LDS)
        nu = moe.run_ug(c, key.ps, PS0, lambda: _bin_to_lds(c, ~(is_r | is_g)))
        if bid < OTASKS:
            dops = moe.down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
            moe.load_inter(c)
            moe.stage_down(c, bid, nu, dops)
        if bid < hc.NRED:
            stage_final(c, bid)

    @flyc.jit
    def launch(
        core: Int64,
        gate: Int64,
        state: Int64,
        inj_a: Int64,
        w_o: Int64,
        w_os: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        msq_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            core,
            gate,
            state,
            inj_a,
            w_o,
            w_os,
            hc_w,
            w_down,
            w_up,
            w_gate,
            w_sg,
            w13,
            w13s,
            w2,
            w2s,
            out,
            msq_out,
            scratch,
            peers,
            rank,
            epoch,
            layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k2, launch)
    _BUILDS[key] = launch
    return launch


@traced
def _bin_to_lds(c, missing):
    if missing:
        hc.bin_to_lds(c, XROW)


def _put_inj(c, t, r, vs):
    """The mlp injection rows r, r + 1 of token t -> INJ (fp32 of bf16)."""
    c["put"](c["inj"], t * HC + r, vs[0])
    c["put"](c["inj"], t * HC + r + 1, vs[1])


def layer_post(
    key: K2Build,
    *,
    core,
    state,
    inj_a,
    w_o,
    hc_w,
    w_down,
    w_up,
    w_gate,
    w_sg,
    w13,
    w13s,
    w2,
    w2s,
    out,
    msq_out,
    scratch,
    peers,
    rank,
    epoch,
    layer,
    w_os=None,
    gate=None,
):
    """Host launcher; every weight as vLLM holds it after loading, the experts'
    shuffled (``layout``) with the shared expert fused as expert 512. ``state``
    / ``out``: the multi-stream state [S, 4 H] before the layer's attention
    combine and after its mlp combine; ``inj_a`` [S, 4] bf16: the attention
    HC's injection logits; ``msq_out`` [20, S, 4] fp32: ``out``'s sums of
    squares for the next layer's front. ``w_os``: an fp4 o_proj's scales
    (``key.oq``); ``gate``: the attention output gate (``key.og``), core's
    shape. ``peers``: the int64 table of every rank's peer buffer (at least
    ``peer_bytes(MAX_TOKENS, tp)`` each)."""
    s = key.tokens
    lay = L.Layout(key.tp)
    bf = torch.bfloat16
    fp4s = (torch.uint8, torch.float4_e2m1fn_x2)
    ne = E + 1
    k = lay.CORE
    need(core.shape == (s, k), f"core: [{s}, {k}]")
    need(state.shape == (s, HH) and out.shape == (s, HH), "state / out: [S, 4 H]")
    need(inj_a.shape == (s, HC) and inj_a.dtype == bf, "inj_a: bf16 [S, 4]")
    need(
        msq_out.numel() == hc.NRED * s * HC and msq_out.dtype == torch.float32,
        "msq_out: fp32 [20, S, 4]",
    )
    need(
        (gate is not None) == key.og and (gate is None or gate.shape == core.shape),
        "gate: core's shape, with og only",
    )
    if key.oq:
        need(
            w_o.shape == (H, k // 2) and w_o.dtype in fp4s,
            f"o_proj: fp4x2 [{H}, {k // 2}]",
        )
        need(
            w_os is not None
            and w_os.numel() == H * k // 32
            and w_os.element_size() == 1,
            "o_proj scales: e8m0, one a 32-block",
        )
    else:
        need(w_o.shape == (H, k) and w_o.dtype == bf, f"o_proj: bf16 [{H}, {k}]")
    need(hc_w.shape == (HH,), "mlp HC norm weight: [4 H]")
    need(w_down.shape == (L.DOWN_ROWS, HH), "mlp HC down: [336, 4 H]")
    need(w_up.shape == (HH, L.LORA), "mlp HC up: [4 H, 320]")
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
    used = [w_o, w_os, w_gate, w_sg, w13, w13s, w2, w2s, gate]
    tensors = [core, state, inj_a, out, msq_out, hc_w, w_down, w_up]
    tensors += [w for w in used if w is not None]
    need(all(t.is_contiguous() for t in tensors), "a non-contiguous input")
    dense = [core, state, out, hc_w, w_down, w_up, w_gate, w_sg]
    dense += [gate] if gate is not None else []
    need(all(t.dtype == bf for t in dense), "activations and dense weights: bf16")

    def ptr(t):
        return 0 if t is None else t.data_ptr()

    f = build(key)
    f(
        *ABI.pack(
            {
                "core": core.data_ptr(),
                "gate": ptr(gate),
                "state": state.data_ptr(),
                "inj_a": inj_a.data_ptr(),
                "w_o": w_o.data_ptr(),
                "w_os": ptr(w_os) if key.oq else 0,
                "hc_w": hc_w.data_ptr(),
                "w_down": w_down.data_ptr(),
                "w_up": w_up.data_ptr(),
                "w_gate": w_gate.data_ptr(),
                "w_sg": w_sg.data_ptr(),
                "w13": w13.data_ptr(),
                "w13s": w13s.data_ptr(),
                "w2": w2.data_ptr(),
                "w2s": w2s.data_ptr(),
                "out": out.data_ptr(),
                "msq_out": msq_out.data_ptr(),
                "scratch": scratch.data_ptr(),
                "peers": peers.data_ptr(),
                "rank": rank,
                "epoch": epoch.data_ptr(),
                "layer": layer,
            }
        ),
        stream=torch.cuda.current_stream(),
    )
