# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 ``layer_post``: one Qwen3.8 decoder layer from the attention's core
output (the out_proj / o_proj input, the same shape on GDN and full-attention
layers) to the layer's MoE output, every TP reduction inside the kernel.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``, S <= 8 decode rows:

    o_proj   (512 x 16 rows, K 2048 a rank; every CTA, tasks bid, bid + 256):
             bf16 GEMV of the core rows -> bf16 partial -> every rank's AR
    norm     (CTAs 0 .. 63, 128 columns each): the ranks' partials in rank
             order (fp32 -> bf16), + residual (bf16) -> ``res_out``; every
             task's sum of squares -> post_attention_layernorm (Gemma, 1 + w)
             -> x rows (plain, XRDY a task)
    router   (CTAs 64 .. 79: 4 x 16 of this rank's 64 experts x K quarters):
             fp32 parts -> bf16 logits -> every rank's RLOG
    sgu      (CTAs 80 .. 111: 32 x 16 rows of the shared gate | up) -> SGU
    route    (CTAs 112 .. 112+S-1, token t): sigmoid(x . w_sg) -> SGATE; the
             512 logits' top-10 (radix select; ties: the lower expert), in
             descending order, softmax-renormalized -> ROUTE
    ug       (|U| x 16, every CTA): the union U of the step's picks; an
             expert's 16 gate + 16 up rows (fp4 -> bf16 by their e8m0 scales)
             against the x rows through MXFP4; silu(g) u (fp32) -> INTER a pick
    down     (512 x 16 hidden rows, every CTA): w2 of every expert of U
             against its picks' INTER through MXFP4, times the route weight,
             summed in top-k
             order; + the shared down rows of silu(g) u (SGU) times SGATE ->
             bf16 partial -> every rank's FINAL
    final    (CTAs 0 .. 63, 128 columns): FINAL in rank order -> bf16 ``out``

The routed experts quantize their activations as the stock decode path does
(AITER's one-stage ``fmoe_bf16_pertokenMXfp4_g1u1_flat`` below 32 rows takes bf16
in, but runs x and silu(g) u through MXFP4, 1 x 32 blocks, RCEIL scales); the
quantized values are exact in bf16, so the MFMAs stay bf16. Their weights are
``AITER_MXFP4_MXFP4``'s shuffled ones (``layout``). Peer regions are
double-buffered by the step epoch's parity.

Two variants follow the stock flags:

- ``fse`` (``VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS``): the shared expert is
  expert 512 of w13 / w2, a pick of every token at top-k index 10 with weight
  sigmoid(bf16(x . w_sg)) (fp32, the top-k kernel's fused sigmoid); no sgu
  stage, no shared down rows.
- ``oq`` (an online-MXFP4 out_proj on the ASM FP4 GEMM): o_proj's weight is
  fp4x2 through ``shuffle_weight(layout=(16, 16))``, its scales through
  ``e8m0_shuffle``; the core rows go through MXFP4 (RCEIL), as
  ``per_1x32_f4_quant_hip`` quantizes the GEMM's input.
"""

import math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from vllm.models.kimi_k3.amd.mono.common.debug import (
    region_ids,
    stamp,
    stamp_begin,
    stamp_flush,
)
from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_DEV,
    CM_NT,
    ballot,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    block_excl_scan,
    div_rn,
    hw_exp2,
    hw_rsq,
    lanes_below,
    ld_i32,
    load_ptr64,
    mfma_bf16,
    popcount,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_max,
    wave_sum,
    xred,
)
from vllm.models.kimi_k3.amd.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from vllm.models.kimi_k3.amd.mono.common.ranks import peer_bases, sum_partials
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, preg, publish, shift, sreg
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono import layout as L
from vllm.models.qwen3_5.amd.mono.layout import need
from vllm.models.qwen3_5.amd.mono.sources import digest

TP, HIDDEN, ROWS = L.TP, L.HIDDEN, L.ROWS
E, TOPK, RI, SI, EPR = L.E, L.TOPK, L.RI, L.SI, L.EPR
OK = L.CORE  # o_proj K a rank: 2048
OTASKS = HIDDEN // ROWS  # 512
COLS = 128  # columns a norm / final task
NRED = HIDDEN // COLS  # 64
R_GROUPS = EPR // ROWS  # 4
R_PARTS = 4  # K quarters: the router is on the path to the route
R_K = HIDDEN // R_PARTS
R_TASKS = R_GROUPS * R_PARTS  # 16
G_TASKS = 2 * SI // ROWS  # 32
R0 = NRED  # 64: router CTAs
G0 = R0 + R_TASKS  # 80: shared gate | up CTAs
RT0 = G0 + G_TASKS  # 112: route CTAs
PS0 = G0  # ``ps``: the shared expert's ug CTAs (the gate | up ones, idle fused)
UG_COLS = 16  # intermediate columns an ug task
UG_GROUPS = RI // UG_COLS  # 16
W13_STEPS = HIDDEN // L.FP4_STEP  # 64
UG_STEPS = W13_STEPS // 4  # a wave's K quarter: 16 steps
W2_STEPS = RI // L.FP4_STEP  # 2
MAX_U = L.MAX_U + 1  # + the fused shared expert
LOG2E = 1.0 / math.log(2.0)
TAG_SLOTS = 256


assert RT0 + L.MAX_TOKENS <= BLOCKS and OTASKS == 2 * BLOCKS and L.BLOCKS == BLOCKS

REGIONS = (
    "msq",
    "xrdy",
    "rpart",
    "sgu",
    "sgate",
    "route",
    "inter",
    "ar",
    "rlog",
    "final",
)

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K2Build:
    tokens: int
    eps: float = 1e-6  # post_attention_layernorm
    diag: bool = False  # bounded waits recording into ``diag`` (debug)
    # debug: run phases < stop only (1 o_proj, 2 norm, 3 router / route,
    # 4 ug, 5 down)
    stop: int = 99
    fse: bool = False  # the shared expert fused as expert 512
    oq: bool = False  # o_proj MXFP4 (ASM FP4 GEMM layout)
    # debug: each CTA's TL_POINTS phase stamps -> ``diag`` [CTA][point] int64
    timeline: bool = False
    # every CTA routes its tokens itself (no ROUTE hand-off)
    lr: bool = False
    ugd: int = 2  # ug tasks in flight a CTA (the running one and its prefetches)
    # the consumers build x (AR sum + residual, rstd from the norm tasks' sums
    # of squares): no x hand-off after the rstd one
    cx: bool = False
    # ug on the scaled fp4 MFMA, x as MXFP4 codes in LDS
    f4m: bool = False
    db: int = 4  # a wave's down slots whose w2 is in flight together
    # down on the scaled fp4 MFMA, INTER as MXFP4 codes in LDS
    f4d: bool = False
    # lr / cx: a few lanes wait on the last token's words before the full polls
    sp: bool = False
    # MXFP4 quantization on v_cvt_scalef32_pk_fp4 (bitwise the software one)
    hq: bool = False
    # (``f4m``, not ``cx``) the norm tasks publish x's MXFP4 codes too: the
    # experts load them (no x quantization a CTA)
    xq: bool = False
    # (``fse``) the shared expert's ug tasks run before the route (its picks
    # are known): off the routed ug phase
    ps: bool = False
    # both down row groups' first batches in flight before INTER and the first
    # group's FINAL puts (vmcnt counts stores too: a load issued after the puts
    # waits on their acks)
    dp: bool = False
    # (``lr``) one token: U is its picks in top-k order, straight from the
    # route (distinct; INTER and the down sums go by pick, not slot)
    u1: bool = False
    # o_proj's input is core x sigmoid(gate) (bf16; the attention output gate
    # as Qwen3NextAttention applies it)
    og: bool = False
    # the route's top-10 by wave max pops (no bit search, no rank pass)
    mp: bool = False
    # (``fse``, ``lr``, ``cx``) every CTA its tokens' shared gate itself, from
    # its LDS x (no route CTAs, no SGATE hand-off)
    ls: bool = False
    # a CTA's ug tasks past the last: loads without memory access
    ul: bool = False

    @property
    def picks(self) -> int:
        """Expert picks a token: the top-k (+ the fused shared expert)."""
        return TOPK + int(self.fse)


def tuned(s: int, fse: bool) -> dict:
    """The fastest variant flags at width s (K2 stage benchmark): x's codes
    from the norm tasks pay their extra hand-off back at the wide widths."""
    return {
        "lr": True,
        "cx": s < 6,
        "xq": s >= 6,
        "sp": True,
        "f4m": True,
        "f4d": True,
        "hq": True,
        "mp": True,
        "u1": s == 1,
        "ps": fse and s >= 4,
        "ul": True,
    }


ABI = KernelAbi(
    (
        "core",
        "residual",
        "w_o",
        "w_os",
        "ln_w",
        "w_gate",
        "w_sg",
        "w_sgu",
        "w_sd",
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
        "layer",
        "diag",
        "gate",
    )
)


# timeline points: 0 start, 1 o_proj done, 2 norm / router / shared gate | up,
# 3 route, 4 union + x quantized, 5 ug, 6 INTER loaded, 7 / 8 down rounds,
# 9 final reduce; 10 core rows quantized, 11 the norm's AR sums in, 12 the
# router's x in, 13 the experts' x in, 14 x quantized, 15 local route; 16 the
# router's parts in (part 0), 17 its logits put, 18 the last logits in (``lr``
# / ``sp``), 19 routed (``lr``; the route CTAs: ROUTE put); in route_token: 20
# the logits in, 21 the 10th's key found, 22 the 10 picked; in stage_ug: 23
# its MFMAs done (the weights in), 24 its INTER put
TL_POINTS = 25


def peer_pairs(s):
    """One parity's peer regions, pairs: name -> count."""
    return {
        "ar": TP * s * HIDDEN // 2,
        "rlog": TP * s * EPR,
        "final": TP * s * HIDDEN // 2,
    }


def peer_layout(s):
    return pair_layout(peer_pairs(s).items())


def peer_bytes(s):
    lay = peer_layout(s)
    return 2 * max(o + n for o, n in lay.values())


def scratch_layout(key: K2Build) -> dict:
    s = key.tokens
    pairs = {
        "msq": NRED * s,
        "xrdy": NRED,
        "rpart": R_TASKS * s * ROWS,
        "sgu": s * SI,  # 2 SI bf16 a token, two a pair
        "sgate": s,
        "route": s * 2 * TOPK,
        "inter": s * key.picks * RI,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    lay["xrow"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    if key.xq:
        at = (max(o + n for o, n in lay.values()) + 255) // 256 * 256
        lay["xq"] = (at, xq_bytes(s))
    return lay


def xq_bytes(s):
    """``xq``'s x codes: token-major fp4 rows (HIDDEN / 2 bytes), then a dword a
    1 x 32 block's e8m0 code."""
    return s * HIDDEN // 2 + s * HIDDEN // 32 * 4


def scratch_bytes(key: K2Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def step_tag(epoch, slot):
    """This launch's mailbox tag: the step's epoch (bumped once a step, in the
    graph) and the launch's slot, so no pair of another step or launch passes."""
    e = uniform(ld_i32(epoch, 0))
    return (e % (1 << 22)) * TAG_SLOTS + slot + 1


def f32_of(w):
    return w.bitcast(fx.Float32)


def ld_bf(ptr, i):
    return fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16))
    )


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def silu_mul(g, u):
    return g * sigmoid(g) * u


def pow2(code):
    """The fp32 power of two an e8m0 code stands for."""
    return (code << 23).bitcast(fx.Float32)


def mx_scale(amax):
    """A 1 x 32 block's MXFP4 e8m0 code from its abs max: ceil_pow2(amax / 6)
    (RCEIL, AITER's activation quant), clamped to codes 1 .. 253."""
    d = div_rn(amax, fx.Float32(6.0), fx.Float32(1.0 / 6.0))
    bits = d.bitcast(fx.Int32)
    e = (bits >> 23) & 0xFF
    e = e + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
    return fx.min(fx.max(e, fx.Int32(1)), fx.Int32(253))


def mx_qdq(v, code):
    """An fp32 through e2m1 at scale ``code`` (nearest even, saturating at 6)
    and back: a power of two times an e2m1 value, exact in bf16."""
    a = fx.min(fx.Float32(fmath.absf(v * pow2(fx.Int32(254) - code))), fx.Float32(6.0))
    q = (a < fx.Float32(2.0)).select(
        fx.Float32(fmath.roundeven(a * fx.Float32(2.0))) * fx.Float32(0.5),
        (a < fx.Float32(4.0)).select(
            fx.Float32(fmath.roundeven(a)),
            fx.Float32(fmath.roundeven(a * fx.Float32(0.5))) * fx.Float32(2.0),
        ),
    )
    return (v < fx.Float32(0.0)).select(-q, q) * pow2(code)


def mx_nibble(v, code):
    """``mx_qdq``'s e2m1 value of an fp32 at scale ``code`` as its fp4 code
    (sign bit 3)."""
    a = fx.min(fx.Float32(fmath.absf(v * pow2(fx.Int32(254) - code))), fx.Float32(6.0))
    small = fx.Float32(fmath.roundeven(a * fx.Float32(2.0)))
    mid = fx.Float32(fmath.roundeven(a)) + fx.Float32(2.0)
    big = fx.Float32(fmath.roundeven(a * fx.Float32(0.5))) + fx.Float32(4.0)
    m = (a < fx.Float32(2.0)).select(small, (a < fx.Float32(4.0)).select(mid, big))
    # an integral fp32 below 2^23 -> its int: add 2^23, take the mantissa
    n = (m + fx.Float32(8388608.0)).bitcast(fx.Int32) & 0x7
    return n | (v < fx.Float32(0.0)).select(fx.Int32(8), fx.Int32(0))


def mx_div(code):
    """An e8m0 code as the fp32 the scaled converts take (2^(code - 127))."""
    return (code << 23).bitcast(fx.Float32)


def pk_fp4_bf16(old, word, scale, j):
    """A packed bf16 pair / ``scale`` -> e2m1 codes into byte j of ``old``
    (nearest even, saturating: ``mx_nibble``'s)."""
    v = fx.Vector.from_elements([fx.Int32(word)], fx.Int32).bitcast(fx.BFloat16)
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_bf16(
            T.i32,
            as_ir_value(fx.Int32(old)),
            as_ir_value(v),
            as_ir_value(fx.Float32(scale)),
            j,
        )
    )


def pk_fp4_f32(old, a, b, scale, j):
    """Two fp32 / ``scale`` -> e2m1 codes into byte j of ``old``."""
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32,
            as_ir_value(fx.Int32(old)),
            as_ir_value(fx.Float32(a)),
            as_ir_value(fx.Float32(b)),
            as_ir_value(fx.Float32(scale)),
            j,
        )
    )


def pk_bf16_fp4(word, scale, j):
    """Byte j of ``word``'s two e2m1 codes times ``scale`` -> a bf16 pair."""
    v = rocdl.cvt_scalef32_pk_bf16_fp4(
        T.vec(2, T.bf16),
        as_ir_value(fx.Int32(word)),
        as_ir_value(fx.Float32(scale)),
        j,
    )
    return fx.Vector(v).bitcast(fx.Int32)[0]


def mfma_f4(a, b, acc, sa, sb):
    """16x16x128 fp4 x fp4 (e8m0 scales ``sa``, ``sb``: a lane's 32-block)."""
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32), [a, b, acc, 4, 4, 0, sa, 0, sb]
        )
    )


def fp4x8_bf16(word, scale):
    """The 8 fp4 of ``word`` times ``scale`` -> an MFMA operand (8 bf16 as 4 i32,
    nibble order: element 2 j, 2 j + 1 from byte j)."""
    out = []
    for j in range(4):
        v = rocdl.cvt_scalef32_pk_bf16_fp4(
            T.vec(2, T.bf16),
            as_ir_value(fx.Int32(word)),
            as_ir_value(fx.Float32(scale)),
            j,
        )
        out.append(fx.Vector(v).bitcast(fx.Int32)[0])
    return fx.Vector.from_elements(out, fx.Int32)


def fp4_mfmas(tile, scale, xl, xoff, acc):
    """One 128-K step: a lane's 16 B of fp4 (4 dwords, K 32 (lane / 16) + 8 q ..
    for dword q) against the bf16 B rows at LDS ``xl`` + ``xoff`` (the same K)."""
    for q in range_constexpr(4):
        xb = fx.Vector(
            fx.ptr_load(
                xl + (xoff + 8 * q) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        a = fp4x8_bf16(tile[q], scale)
        acc = mfma_bf16(a.bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc)
    return acc


def fp4_tile(r_w, at):
    return fx.Vector(
        bo.buffer_load(r_w, at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )


def scale_word(r_s, byte_index):
    return fx.Int32(bo.buffer_load(r_s, byte_index // 4, vec_width=1, dtype=T.i32))


def fp4_gemv_loads(lane, wave, w, ws, rg, k):
    """An online-MXFP4 linear's rows 16 rg .. (K ``k``, the ASM FP4 GEMM's
    shuffled weight and scales) for this wave: its K steps' tiles (wave w:
    steps w n .. w n + n - 1) and their scale words (a word: two steps)."""
    n = k // L.FP4_STEP // WAVES
    r_w, r_s = rsrc(w), rsrc(ws)
    row = rg * ROWS + lane % ROWS
    tiles = [
        fp4_tile(r_w, L.fp4_tile_dword(0, rg, k, wave * n + i, lane)) for i in range(n)
    ]
    words = [
        scale_word(
            r_s, L.scale_index(row, 4 * (wave * n + 2 * i) + lane // 16, k // 32)
        )
        for i in range(n // 2)
    ]
    return tiles + words


def fp4_gemv(lane, wave, xl, xrow, k, s, rg, ops):
    """The ``fp4_gemv_loads`` operands against the MXFP4-exact bf16 rows at LDS
    ``xl`` (row stride ``xrow``) -> this wave's fp32 partial MFMA tile."""
    n = k // L.FP4_STEP // WAVES
    t = fx.min(lane % 16, s - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(n):
        # the code is byte (step % 2) 2 + rg % 2 of its word (e8m0_shuffle)
        code = (ops[n + i // 2] >> ((i % 2) * 16 + (rg % 2) * 8)) & 0xFF
        xoff = t * xrow + (wave * n + i) * L.FP4_STEP + 32 * (lane // 16)
        acc = fp4_mfmas(ops[i], pow2(code), xl, xoff, acc)
    return acc


# ---------------------------------------------------------------- o_proj
@traced
def stage_o(c, rg, ops):
    """Rows 16 rg ..: GEMV of the LDS core rows -> bf16 partials -> every rank's
    AR region, this rank's slot (thread: token, row pair)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    if const_expr(c["oq"]):
        acc = fp4_gemv(lane, wave, c["xl"], OK + gemv.LDS_PAD, OK, s, rg, ops)
    else:
        acc = gemv.mfmas(lane, wave, c["xl"], OK + gemv.LDS_PAD, OK, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < s * (ROWS // 2):
        tt = tid // (ROWS // 2)
        rp = tid % (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, tt))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tt))
        at = (c["rank"] * s + tt) * HIDDEN + rg * ROWS + 2 * rp
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["ar"][p], at, [v0, v1])
    gpu.barrier()


def o_loads(c, j):
    """o_proj task bid + 256 j's weights for this wave (fp4: its two K steps'
    tiles and their scale word)."""
    lane, wave, a = c["lane"], c["wave"], c["args"]
    rg = c["bid"] + BLOCKS * j
    if const_expr(c["oq"]):
        return fp4_gemv_loads(lane, wave, a["w_o"], a["w_os"], rg, OK)
    return gemv.loads(lane, wave, a["w_o"], rg * ROWS, OK)


@traced
def gated_rows_to_lds(c, core, gate):
    """``og``: bf16(core x sigmoid(gate)) (fp32 math, as the compiled
    ``o * torch.sigmoid(gate)``) -> the o_proj input rows in LDS."""
    tid, s, xl = c["tid"], c["S"], c["xl"]
    words = s * OK // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (OK // 8)
        kk = e % (OK // 8) * 8
        at = (t * OK + kk) // 2
        v = fx.Vector(bo.buffer_load(rsrc(core), at, vec_width=4, dtype=T.i32))
        g = fx.Vector(bo.buffer_load(rsrc(gate), at, vec_width=4, dtype=T.i32))
        y = fx.Vector.from_elements(
            [
                bf16_pair(
                    bf_lo(v[d]) * sigmoid(bf_lo(g[d])),
                    bf_hi(v[d]) * sigmoid(bf_hi(g[d])),
                ).bitcast(fx.Int32)
                for d in range(4)
            ],
            fx.Int32,
        )
        if tid + THREADS * i < words:
            fx.ptr_store(y, xl + (t * (OK + gemv.LDS_PAD) + kk) // 2)
    gpu.barrier()


@traced
def run_oproj(c, oops):
    """This CTA's o_proj tasks (bid, bid + 256) on the core rows (plain loads:
    the previous launch wrote them), their weights ``oops`` already in flight."""
    core = c["args"]["core"]
    if const_expr(c["og"]):
        gated_rows_to_lds(c, core, c["args"]["gate"])
    else:
        gemv.rows_to_lds(c["tid"], core, OK, c["S"], c["xl"], OK + gemv.LDS_PAD)
    if const_expr(c["oq"]):
        quant_x(c, OK)
    c["st"](10)
    for j in range_constexpr(OTASKS // BLOCKS):
        stage_o(c, c["bid"] + BLOCKS * j, oops[j])


# ---------------------------------------------------------------- norm
@traced
def stage_norm(c, j):
    """Columns 128 j ..: the AR sum (rank order, fp32 -> bf16) + residual (fp32;
    bf16 -> ``res_out``); every task's sum of squares -> rstd -> bf16(v rstd
    (1 + w)) -> the x rows, XRDY j. The norm reads the unrounded sum, as
    ``fused_add_rms_norm`` does."""
    s, tid, lane = c["S"], c["tid"], c["lane"]
    a = c["args"]
    mine = tid < s * 64
    t = fx.min(tid // 64, s - 1)
    cp = tid % 64
    col = j * COLS + 2 * cp
    lo, hi = sum_partials(
        c["poll"],
        c["peer"]["ar_own"],
        lambda src: ((src * s + t) * HIDDEN + col) // 2,
        TP,
    )
    c["st"](11)
    rw = fx.Int32(
        bo.buffer_load(
            rsrc(a["residual"]), (t * HIDDEN + col) // 2, vec_width=1, dtype=T.i32
        )
    )
    z0 = bf16_round(lo) + bf_lo(rw)
    z1 = bf16_round(hi) + bf_hi(rw)
    zw = (
        fx.Vector.from_elements([z0, z1], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    if mine:
        bo.buffer_store(zw, rsrc(a["res_out"]), (t * HIDDEN + col) // 2)
    if const_expr(c["cx"]):
        # the bf16 AR sums, drained before this token's (this wave's) MSQ
        if mine:
            aw = bf16_pair(lo, hi).bitcast(fx.Int32)
            at = (t * HIDDEN + col) // 2
            bo.buffer_store(aw, rsrc(c["xbuf"]), at, cache_modifier=CM_DEV)
        sq = wave_sum(z0 * z0 + z1 * z1)
        rocdl.s_waitcnt(vmcnt=0)
        if mine & (lane == 0):
            c["put"](c["msq"], j * s + t, sq)
    else:
        stage_norm_x(c, j, z0, z1, mine, t, col)


@traced
def stage_norm_x(c, j, z0, z1, mine, t, col):
    """The norm task's x columns -> XRDY j."""
    s, tid, lane, red = c["S"], c["tid"], c["lane"], c["red"]
    a = c["args"]
    sq = wave_sum(z0 * z0 + z1 * z1)
    if mine & (lane == 0):
        c["put"](c["msq"], j * s + t, sq)
    gpu.barrier()
    if mine:
        v = c["poll"]([(c["msq"], lane * s + t, 1)])[0][0]
        fx.ptr_store(f32_of(v), red + (t * 64 + lane))
    gpu.barrier()
    tot = fx.ptr_load(red + t * 64)
    for q in range_constexpr(1, NRED):
        tot = tot + fx.ptr_load(red + (t * 64 + q))
    r = hw_rsq(tot * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
    w0 = fx.Float32(1.0) + ld_bf(a["ln_w"], col)
    w1 = fx.Float32(1.0) + ld_bf(a["ln_w"], col + 1)
    y = (
        fx.Vector.from_elements([z0 * r * w0, z1 * r * w1], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    if mine:
        at = (t * HIDDEN + col) // 2
        bo.buffer_store(y, rsrc(c["xbuf"]), at, cache_modifier=CM_DEV)
    if const_expr(c["xq"] is not None):
        put_xq(c, y, mine, t, col)
    publish(c["put"], c["xrdy"], j, fx.Int32(1), tid == 0)


@traced
def put_xq(c, y, mine, t, col):
    """``xq``: a norm task lane's x pair ``y`` (token t, columns col, col + 1;
    a 1 x 32 block on 16 lanes) -> its MXFP4 codes in XQ (``quant_x``'s)."""
    s, lane = c["S"], c["lane"]
    y0, y1 = bf_lo(y), bf_hi(y)
    amax = fx.max(fx.Float32(fmath.absf(y0)), fx.Float32(fmath.absf(y1)))
    for off in range_constexpr(4):
        amax = xred(amax, 1 << off, fx.max)
    code = mx_scale(amax)
    if const_expr(c["hq"]):
        v = pk_fp4_bf16(fx.Int32(0), y, mx_div(code), 0)
    else:
        v = mx_nibble(y0, code) | (mx_nibble(y1, code) << 4)
    v = v << (8 * (lane % 4))
    for off in range_constexpr(2):
        v = xred(v, 1 << off, lambda a, b: a | b)
    at = t * HIDDEN + col
    if mine & (lane % 4 == 0):
        bo.buffer_store(v, rsrc(c["xq"]), at // 8, cache_modifier=CM_DEV)
    if mine & (lane % 16 == 0):
        bo.buffer_store(
            code, rsrc(c["xq"]), s * HIDDEN // 8 + at // 32, cache_modifier=CM_DEV
        )
    gpu.barrier()


@traced
def load_rstd(c):
    """``cx``: every norm task's sums of squares -> each token's rstd -> LDS
    ``rs`` (the norm tasks' order and arithmetic)."""
    s, tid, lane, red = c["S"], c["tid"], c["lane"], c["red"]
    if const_expr(c["sp"]):
        if tid < 64:
            c["poll"]([(c["msq"], tid * s + s - 1, 1)])
        gpu.barrier()
    if tid < 64 * s:
        t = tid // 64
        v = c["poll"]([(c["msq"], lane * s + t, 1)])[0][0]
        fx.ptr_store(f32_of(v), red + (t * 64 + lane))
    gpu.barrier()
    if tid < s:
        tot = fx.ptr_load(red + tid * 64)
        for q in range_constexpr(1, NRED):
            tot = tot + fx.ptr_load(red + (tid * 64 + q))
        r = hw_rsq(tot * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        fx.ptr_store(r, c["rs"] + tid)
    gpu.barrier()


@traced
def x_rows_to_lds(c, klen, k0):
    """``cx``: x columns k0 .. k0 + klen = bf16((AR sum + residual) rstd (1 + w))
    -> LDS, 8 columns a lane (``stage_norm``'s arithmetic)."""
    s, tid, a = c["S"], c["tid"], c["args"]
    xrow = klen + gemv.LDS_PAD
    words = s * klen // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (klen // 8)
        kk = e % (klen // 8) * 8
        g = (t * HIDDEN + k0 + kk) // 2
        av = fx.Vector(
            bo.buffer_load(
                rsrc(c["xbuf"]), g, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV
            )
        )
        rv = fx.Vector(bo.buffer_load(rsrc(a["residual"]), g, vec_width=4, dtype=T.i32))
        wv = fx.Vector(
            bo.buffer_load(rsrc(a["ln_w"]), (k0 + kk) // 2, vec_width=4, dtype=T.i32)
        )
        r = fx.ptr_load(c["rs"] + t)
        ys = []
        for d in range_constexpr(4):
            z0 = bf_lo(av[d]) + bf_lo(rv[d])
            z1 = bf_hi(av[d]) + bf_hi(rv[d])
            w0 = fx.Float32(1.0) + bf_lo(wv[d])
            w1 = fx.Float32(1.0) + bf_hi(wv[d])
            ys.append(bf16_pair(z0 * r * w0, z1 * r * w1).bitcast(fx.Int32))
        if tid + THREADS * i < words:
            fx.ptr_store(
                fx.Vector.from_elements(ys, fx.Int32), c["xl"] + (t * xrow + kk) // 2
            )
    gpu.barrier()


@traced
def wait_x(c, klen, k0):
    """Every norm task's x columns ready -> x columns k0 .. k0 + klen into LDS."""
    if const_expr(c["cx"]):
        load_rstd(c)
        x_rows_to_lds(c, klen, k0)
    else:
        wait_x_rows(c, klen, k0)


@traced
def wait_x_rows(c, klen, k0):
    tid = c["tid"]
    if tid < NRED:
        c["poll"]([(c["xrdy"], tid, 1)])
    gpu.barrier()
    gemv.rows_to_lds(
        tid,
        c["xbuf"],
        klen,
        c["S"],
        c["xl"],
        klen + gemv.LDS_PAD,
        CM_DEV,
        ldk=HIDDEN,
        k0=k0,
    )


@traced
def load_xq(c):
    """``xq``: every norm task's x ready -> its MXFP4 codes into LDS in
    ``quant_x_fp4``'s layout."""
    s, tid, xl = c["S"], c["tid"], c["xl"]
    if tid < NRED:
        c["poll"]([(c["xrdy"], tid, 1)])
    gpu.barrier()
    row = HIDDEN // 32  # 16 B chunks a token's codes
    n = s * row + s * HIDDEN // 128  # then the e8m0 dwords, 4 a chunk
    for i in range_constexpr((n + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, n - 1)
        v = bo.buffer_load(
            rsrc(c["xq"]), 4 * e, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV
        )
        codes = e < s * row
        at = codes.select(
            (e // row) * (XQ_ROW // 4) + 4 * (e % row),
            xq_scales(s) + 4 * (e - s * row),
        )
        if tid + THREADS * i < n:
            fx.ptr_store(fx.Vector(v), xl + at)
    gpu.barrier()


@traced
def quant_x(c, k=HIDDEN):
    """The rows of K ``k`` in LDS through MXFP4 in place, 1 x 32 blocks (RCEIL
    scale), 8 elements a lane (a block on 4 adjacent lanes: one 16 B LDS access
    a lane, conflict-free): the routed experts' activations, as the stock
    decode kernel takes them (the router and the shared expert read x
    unquantized, before this), or an fp4 o_proj's core rows."""
    s, tid, xl = c["S"], c["tid"], c["xl"]
    xrow = k + gemv.LDS_PAD
    nq = s * k // 8
    for i in range_constexpr((nq + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, nq - 1)
        at = ((e // (k // 8)) * xrow + 8 * (e % (k // 8))) // 2
        wv, vs, code = _quad_scale(xl + at)
        if const_expr(c["hq"]):
            sc = mx_div(code)
            q = fx.Int32(0)
            for d in range_constexpr(4):
                q = pk_fp4_bf16(q, wv[d], sc, d)
            ys = [pk_bf16_fp4(q, sc, d) for d in range(4)]
        else:
            ys = [
                bf16_pair(mx_qdq(vs[2 * d], code), mx_qdq(vs[2 * d + 1], code)).bitcast(
                    fx.Int32
                )
                for d in range(4)
            ]
        if tid + THREADS * i < nq:
            fx.ptr_store(fx.Vector.from_elements(ys, fx.Int32), xl + at)
    gpu.barrier()


def _quad_scale(ptr):
    """A lane's 8 bf16 at LDS ``ptr`` (4 words, as f32) and their 1 x 32
    block's e8m0 code (the amax over the block's 4 lanes)."""
    wv = fx.Vector(fx.ptr_load(ptr, result_type=fx.Vector.make_type(4, fx.Int32)))
    vs = []
    for d in range_constexpr(4):
        vs += [bf_lo(wv[d]), bf_hi(wv[d])]
    amax = fx.Float32(0.0)
    for j in range_constexpr(8):
        amax = fx.max(amax, fx.Float32(fmath.absf(vs[j])))
    for off in range_constexpr(2):
        amax = xred(amax, 1 << off, fx.max)
    return wv, vs, mx_scale(amax)


XQ_ROW = HIDDEN // 2 + 16  # bytes a token's fp4 x row in LDS (padded)
IQ_ROW = RI // 2 + 16  # bytes a pick's fp4 INTER row in LDS (padded)


def iq_scales(s, kt):
    """Dword of LDS ``il`` where the fp4 INTER rows' e8m0 codes start (a dword
    a 1 x 32 block, 8 a pick)."""
    return s * kt * IQ_ROW // 4


def xq_scales(s):
    """Dword of LDS ``xl`` where the fp4 x rows' e8m0 codes start (a dword a
    1 x 32 block, token-major)."""
    return s * XQ_ROW // 4


@traced
def quant_x_fp4(c):
    """``f4m``: the x rows in LDS (bf16, K ``HIDDEN``) -> MXFP4 codes over them,
    ``quant_x``'s arithmetic and lanes: token t's block b at byte t ``XQ_ROW`` +
    16 b (element 2 j in byte j's low nibble; a lane's 8 elements one word), its
    code at dword ``xq_scales`` + t (HIDDEN / 32) + b. Every thread reads
    before any writes."""
    s, tid, xl = c["S"], c["tid"], c["xl"]
    xrow = HIDDEN + gemv.LDS_PAD
    kq = HIDDEN // 8
    nq = s * kq
    rounds = (nq + THREADS - 1) // THREADS
    # four rounds read, then their words written: a word lands below every
    # later round's x, so only its own group's reads must precede it; the
    # codes (four rounds' a register) after every read
    codes = []
    for g0 in range_constexpr(0, rounds, 4):
        words = []
        for i in range_constexpr(g0, min(g0 + 4, rounds)):
            e = fx.min(tid + THREADS * i, nq - 1)
            wv, vs, code = _quad_scale(xl + ((e // kq) * xrow + 8 * (e % kq)) // 2)
            w = fx.Int32(0)
            if const_expr(c["hq"]):
                for d in range_constexpr(4):
                    w = pk_fp4_bf16(w, wv[d], mx_div(code), d)
            else:
                for n in range_constexpr(8):
                    w = w | (mx_nibble(vs[n], code) << (4 * n))
            words.append(w)
            if const_expr(i % 4 == 0):
                codes.append(code)
            else:
                codes[-1] = codes[-1] | (code << (8 * (i % 4)))
        gpu.barrier()
        for i in range_constexpr(g0, min(g0 + 4, rounds)):
            e = fx.min(tid + THREADS * i, nq - 1)
            if tid + THREADS * i < nq:
                fx.ptr_store(words[i - g0], xl + ((e // kq) * XQ_ROW // 4 + e % kq))
    gpu.barrier()
    for i in range_constexpr(rounds):
        e = fx.min(tid + THREADS * i, nq - 1)
        if (tid + THREADS * i < nq) & (e % 4 == 0):
            code = (codes[i // 4] >> (8 * (i % 4))) & 0xFF
            fx.ptr_store(code, xl + (xq_scales(s) + e // 4))
    gpu.barrier()


# ---------------------------------------------------------------- router / sgu
def router_loads(c, i, q):
    a = c["args"]
    return gemv.loads(
        c["lane"],
        c["wave"],
        a["w_gate"],
        c["rank"] * EPR + i * ROWS,
        R_K,
        ldk=HIDDEN,
        k0=q * R_K,
    )


@traced
def stage_router(c, i, q, ops):
    """Router task (i, q): this rank's experts 16 i .., K quarter q. Parts
    q > 0 publish fp32 partials; part 0 adds them in part order, rounds to bf16
    (GateLinear's output) and puts the logits to every rank's RLOG."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    wait_x(c, R_K, q * R_K)
    c["st"](12)
    acc = gemv.mfmas(lane, wave, c["xl"], R_K + gemv.LDS_PAD, R_K, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    at = lambda qq, t, r: ((i * R_PARTS + qq) * s + t) * ROWS + r  # noqa: E731
    if q > 0:  # noqa: SIM102 (traced: `and` does not combine traced values)
        if tid < ROWS * s:
            r = tid % ROWS
            t = tid // ROWS
            c["put"](c["rpart"], at(q, t, r), row_sum(red, r, t))
    if q == 0:  # noqa: SIM102
        if tid < (ROWS // 2) * s:
            r = 2 * (tid % (ROWS // 2))
            t = tid // (ROWS // 2)
            got = c["poll"](
                [
                    (c["rpart"], at(qq, t, r + d), 1)
                    for d in range(2)
                    for qq in range(1, R_PARTS)
                ]
            )
            c["st"](16)
            vs = []
            for d in range_constexpr(2):
                v = row_sum(red, r + d, t)
                for qq in range_constexpr(R_PARTS - 1):
                    v = v + f32_of(got[d * (R_PARTS - 1) + qq][0])
                vs.append(bf16_round(v))
            for p in range_constexpr(TP):
                c["put_bf"](
                    c["peer"]["rlog"][p], (c["rank"] * s + t) * EPR + i * ROWS + r, vs
                )
    gpu.barrier()
    c["st"](17)


@traced
def stage_sgu(c, g, ops):
    """Shared gate | up rows 16 g .. (gate 0 .. SI, up SI .. 2 SI) -> bf16 SGU."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    wait_x(c, HIDDEN, 0)
    acc = gemv.mfmas(lane, wave, c["xl"], HIDDEN + gemv.LDS_PAD, HIDDEN, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        t = tid // (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, t))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, t))
        c["put_bf"](c["sgu"], t * 2 * SI + g * ROWS + 2 * rp, [v0, v1])
    gpu.barrier()


# ---------------------------------------------------------------- route
@traced
def stage_sgate(c, t):
    """Token t's shared-expert gate: sigmoid(bf16(x . w_sg)) as bf16 -> SGATE
    (ReplicatedLinear then torch.sigmoid, both bf16 out); fused (``fse``), the
    sigmoid stays fp32: the top-k kernel's on the combined gate's bf16 logit."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    if const_expr(c["cx"]):
        wait_x(c, HIDDEN, 0)
    else:
        if tid < NRED:
            c["poll"]([(c["xrdy"], tid, 1)])
        gpu.barrier()
    per = HIDDEN // THREADS  # 16: two 16 B loads of x and of w a thread
    acc = fx.Float32(0.0)
    for h in range_constexpr(per // 8):
        k = tid * per + 8 * h
        if const_expr(c["cx"]):
            xw = fx.Vector(
                fx.ptr_load(
                    c["xl"] + (t * (HIDDEN + gemv.LDS_PAD) + k) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
        else:
            xw = fx.Vector(
                bo.buffer_load(
                    rsrc(c["xbuf"]),
                    (t * HIDDEN + k) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
        ww = fx.Vector(
            bo.buffer_load(rsrc(a["w_sg"]), k // 2, vec_width=4, dtype=T.i32)
        )
        for d in range_constexpr(4):
            acc = acc + bf_lo(xw[d]) * bf_lo(ww[d]) + bf_hi(xw[d]) * bf_hi(ww[d])
    acc = wave_sum(acc)
    if lane == 0:
        fx.ptr_store(acc, red + wave)
    gpu.barrier()
    if tid == 0:
        tot = fx.ptr_load(red + 0)
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(red + w)
        g = sigmoid(bf16_round(tot))
        c["put"](c["sgate"], t, g if c["fse"] else bf16_round(g))
    gpu.barrier()


# ``ls``: red's last words (8 tokens' gate partials)
SG_RED = WAVES * 64 * 4 - 8 * WAVES


@traced
def local_sgate(c):
    """``ls``: every CTA's own stage_sgate of its tokens (its LDS bf16 x, before
    ``quant_x_fp4`` overwrites it; the same sums in the same order) -> the LDS
    route's shared-expert pick (``load_route``'s fused entries)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    per = HIDDEN // THREADS
    ws = [
        fx.Vector(
            bo.buffer_load(
                rsrc(c["args"]["w_sg"]),
                (tid * per + 8 * h) // 2,
                vec_width=4,
                dtype=T.i32,
            )
        )
        for h in range_constexpr(per // 8)
    ]
    for t in range_constexpr(c["S"]):
        acc = fx.Float32(0.0)
        for h in range_constexpr(per // 8):
            k = tid * per + 8 * h
            xw = fx.Vector(
                fx.ptr_load(
                    c["xl"] + (t * (HIDDEN + gemv.LDS_PAD) + k) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            ww = ws[h]
            for d in range_constexpr(4):
                acc = acc + bf_lo(xw[d]) * bf_lo(ww[d]) + bf_hi(xw[d]) * bf_hi(ww[d])
        acc = wave_sum(acc)
        if lane == 0:
            fx.ptr_store(acc, red + (SG_RED + t * WAVES + wave))
    gpu.barrier()
    if tid < c["S"]:
        tot = fx.ptr_load(red + (SG_RED + tid * WAVES))
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(red + (SG_RED + tid * WAVES + w))
        g = sigmoid(bf16_round(tot)).bitcast(fx.Int32)
        at = c["rt"]["route"] + tid * 2 * c["KT"] + TOPK
        fx.ptr_store(fx.Int32(E), at)
        fx.ptr_store(g, at + c["KT"])
    gpu.barrier()


def _okey(v):
    """An order-preserving uint32 key of an f32 (larger float, larger key)."""
    b = v.bitcast(fx.Int32)
    neg = b < fx.Int32(0)
    return fx.Uint32(neg.select(~b, b ^ fx.Int32(-2147483648)))


_SORT8 = (
    (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7), (0, 1),
    (2, 3), (4, 5), (6, 7), (2, 4), (3, 5), (1, 4), (3, 6), (1, 2), (3, 4), (5, 6),
)  # fmt: skip


def _route_mp(c, lgs):
    """``mp``: route_token's top-10 as 10 wave max pops. A pick's key is its
    logit's key (16 bits: bf16) over 511 - expert, sign-flipped for a signed max:
    distinct, ordered as route_token's (equal logits: the lower expert), so the
    pops come out ranked; each lane's 8 keys sorted first, a pop shifts the
    winner's list."""
    lane = c["lane"]
    imin = fx.Int32(-2147483648)
    hi16 = fx.Int32(-65536)
    ks = []
    for i in range_constexpr(E // 64):
        b = lgs[i].bitcast(fx.Int32)
        ok = (b < fx.Int32(0)).select(~b, b ^ imin) & hi16
        ks.append((ok | (fx.Int32(E - 1) - (lane + 64 * i))) ^ imin)
    for i, j in _SORT8:
        ks[i], ks[j] = fx.max(ks[i], ks[j]), fx.min(ks[i], ks[j])

    def logit(k):
        u = k ^ imin
        h = u & hi16
        return (h < fx.Int32(0)).select(h ^ imin, ~h & hi16).bitcast(fx.Float32), u

    mine = imin
    top = fx.Float32(0.0)
    total = fx.Float32(0.0)
    mex = fx.Float32(0.0)
    for j in range_constexpr(TOPK):
        m = wave_max(ks[0])
        hit = ks[0] == m
        for i in range_constexpr(E // 64 - 1):
            ks[i] = hit.select(ks[i + 1], ks[i])
        ks[E // 64 - 1] = hit.select(imin, ks[E // 64 - 1])
        lg, _ = logit(m)
        if j == 0:
            top = lg
        ex = hw_exp2((lg - top) * fx.Float32(LOG2E))
        total = total + ex
        sel = lane == j
        mine = sel.select(m, mine)
        mex = sel.select(ex, mex)
    _, u = logit(mine)
    e = fx.Int32(E - 1) - (u & fx.Int32(0xFFFF))
    inv = fx.Float32(1.0) / total
    return fx.min(lane, TOPK - 1), fx.min(fx.max(e, 0), fx.Int32(E - 1)), mex * inv


@traced
def route_token(c, t, rb):
    """Token t on this wave: the 512 bf16 logits (packed pairs), 8 a lane (expert
    lane + 64 i) -> the top-10 by a radix select on their keys (the 10th largest
    key by a bit search of ballot counts; equal keys: the lower experts) -> the
    10 ranked by logit, descending -> exp(l - l_max) renormalized over the 10
    (the renormalized softmax). ``rb``: 128 f32 of LDS for this wave. Returns
    (rank, expert, weight), lanes < ``TOPK``."""
    lane = c["lane"]
    s = c["S"]
    per = E // 64
    specs = []
    for i in range_constexpr(per):
        e = lane + 64 * i
        src = e // EPR
        at = ((src * s + t) * EPR + e % EPR) // 2
        specs.append((c["peer"]["rlog_own"], at, 1))
    got = c["poll"](specs, batch=len(specs))
    odd = (lane % 2) == 1
    keys, lgs = [], []
    for i in range_constexpr(per):
        w = got[i][0]
        lg = odd.select(bf_hi(w), bf_lo(w))
        # a NaN logit (a warmup's garbage row) ranks last: -2^127, bf16
        lg = (lg == lg).select(lg, fx.Float32(-(2.0**127)))
        keys.append(_okey(lg))
        lgs.append(lg)
    c["st"](20)
    if const_expr(c["mp"]):
        return _route_mp(c, lgs)

    def count_ge(th):
        n = fx.Int32(0)
        for i in range_constexpr(per):
            n = n + popcount(ballot(keys[i] >= th))
        return n

    # bf16 logits: a key's low 16 bits are its sign's (0 or 0xffff), so the
    # 16 high bits order them; the low bits of th follow the 10th's sign
    th = fx.Uint32(0)
    for bit in range_constexpr(31, 15, -1):
        cand = th | fx.Uint32(1 << bit)
        th = (count_ge(cand) >= TOPK).select(cand, th)
    lo = fx.Uint32(0xFFFF)
    th = (count_ge(th | lo) >= TOPK).select(th | lo, th)
    c["st"](21)
    n_gt = fx.Int32(0)
    for i in range_constexpr(per):
        n_gt = n_gt + popcount(ballot(keys[i] > th))
    need = fx.Int32(TOPK) - n_gt
    eq_before = fx.Int32(0)
    sel_before = fx.Int32(0)
    for i in range_constexpr(per):
        eqb = ballot(keys[i] == th)
        eq_rank = eq_before + lanes_below(eqb, lane)
        sel = (keys[i] > th) | ((keys[i] == th) & (eq_rank < need))
        selb = ballot(sel)
        pos = sel_before + lanes_below(selb, lane)
        if sel:
            fx.ptr_store(keys[i].bitcast(fx.Float32), rb + pos * 3)
            fx.ptr_store(
                fx.Int32(lane + 64 * i).bitcast(fx.Float32), rb + (pos * 3 + 1)
            )
            fx.ptr_store(lgs[i], rb + (pos * 3 + 2))
        eq_before = eq_before + popcount(eqb)
        sel_before = sel_before + popcount(selb)
    c["st"](22)
    q = fx.min(lane, TOPK - 1)
    mk = fx.Uint32(fx.ptr_load(rb + q * 3).bitcast(fx.Int32))
    me = fx.ptr_load(rb + (q * 3 + 1)).bitcast(fx.Int32)
    ml = fx.ptr_load(rb + (q * 3 + 2))
    rank = fx.Int32(0)
    for j in range_constexpr(TOPK):
        ok = fx.Uint32(fx.ptr_load(rb + j * 3).bitcast(fx.Int32))
        oe = fx.ptr_load(rb + (j * 3 + 1)).bitcast(fx.Int32)
        before = (ok > mk) | ((ok == mk) & (oe < me))
        rank = rank + before.select(fx.Int32(1), fx.Int32(0))
    if lane < TOPK:
        fx.ptr_store(ml, rb + (64 + rank))
    top = fx.ptr_load(rb + 64)
    ex = hw_exp2((ml - top) * fx.Float32(LOG2E))
    if lane < TOPK:
        fx.ptr_store(ex, rb + (96 + rank))
    total = fx.Float32(0.0)
    for j in range_constexpr(TOPK):
        total = total + fx.ptr_load(rb + (96 + j))
    inv = fx.Float32(1.0) / total
    return rank, fx.min(fx.max(me, 0), fx.Int32(E - 1)), ex * inv


def put_route(c, t, rank, e, w):
    c["put"](c["route"], t * 2 * TOPK + rank, e)
    c["put"](c["route"], t * 2 * TOPK + TOPK + rank, w)


@traced
def stage_route(c, t):
    """Token t (wave 0) -> ROUTE (every rank alike)."""
    if c["wave"] == 0:
        rank, e, w = route_token(c, t, c["red"])
        if c["lane"] < TOPK:
            put_route(c, t, rank, e, w)
    c["st"](19)
    gpu.barrier()


@traced
def local_route(c):
    """``lr``: every CTA routes its tokens itself, wave t token t, into its LDS
    route (``load_route``'s layout; ``load_route``'s barrier follows); CTA RT0
    also puts ROUTE, for the tests."""
    lane, wave, rt, kt = c["lane"], c["wave"], c["rt"], c["KT"]
    if const_expr(c["sp"]):
        # every rank's last router group, last token
        s = c["S"]
        if c["tid"] < TP:
            at = ((c["tid"] * s + s - 1) * EPR + EPR - 2) // 2
            c["poll"]([(c["peer"]["rlog_own"], at, 1)])
        gpu.barrier()
    c["st"](18)
    if wave < c["S"]:
        rank, e, w = route_token(c, wave, c["red"] + wave * 128)
        if lane < TOPK:
            fx.ptr_store(e, rt["route"] + (wave * 2 * kt + rank))
            fx.ptr_store(w.bitcast(fx.Int32), rt["route"] + (wave * 2 * kt + kt + rank))
            if c["bid"] == RT0:
                put_route(c, wave, rank, e, w)
    c["st"](19)


@traced
def load_route(c):
    """ROUTE -> LDS (ids, weights; a token's ``KT`` picks); the union U in expert
    order: flag[e] -> slot, expert[slot], kof[slot][t] (the token's pick index,
    -1 if not picked); fused, the shared expert is U's last slot, every token's
    pick ``TOPK`` with its SGATE weight. Returns |U|."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt, kt = c["rt"], c["KT"]
    n = 0 if c["lr"] else s * 2 * TOPK
    if tid < n:
        v = c["poll"]([(c["route"], tid, 1)])[0][0]
        t, j = tid // (2 * TOPK), tid % (2 * TOPK)
        at = t * 2 * kt + j
        if const_expr(c["fse"]):
            at = at + (j >= TOPK).select(fx.Int32(1), fx.Int32(0))
        fx.ptr_store(v, rt["route"] + at)
    if const_expr(c["fse"]):  # noqa: SIM102 (traced: `and` does not combine traced values)
        if const_expr(not c["ls"]):  # noqa: SIM102
            if tid < s:
                g = c["poll"]([(c["sgate"], tid, 1)])[0][0]
                fx.ptr_store(fx.Int32(E), rt["route"] + (tid * 2 * kt + TOPK))
                fx.ptr_store(g, rt["route"] + (tid * 2 * kt + kt + TOPK))
    if const_expr(c["u1"] and s == 1):
        # local_route's stores are this wave's (wave 0): no barrier first
        if tid < TOPK:
            e = fx.max(fx.min(fx.ptr_load(rt["route"] + tid), E - 1), 0)
            fx.ptr_store(e, rt["expert"] + tid)
            fx.ptr_store(fx.Int32(tid), rt["kof"] + tid * 8)
        if const_expr(c["fse"]):  # noqa: SIM102
            if tid == 0:
                fx.ptr_store(fx.Int32(E), rt["expert"] + TOPK)
                fx.ptr_store(fx.Int32(TOPK), rt["kof"] + TOPK * 8)
        gpu.barrier()
        return fx.Int32(kt)
    for i in range_constexpr((E + THREADS - 1) // THREADS):
        e = tid + THREADS * i
        if e < E:
            fx.ptr_store(fx.Int32(0), rt["flag"] + e)
    for i in range_constexpr((MAX_U * 8 + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < MAX_U * 8:
            fx.ptr_store(fx.Int32(-1), rt["kof"] + j)
    gpu.barrier()
    if tid < s * TOPK:
        t = tid // TOPK
        k = tid % TOPK
        e = fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * kt + k)), E - 1), 0)
        fx.ptr_store(fx.Int32(1), rt["flag"] + e)
    gpu.barrier()
    # E = 512 = THREADS: an expert a thread
    f = fx.ptr_load(rt["flag"] + tid)
    off, tot = block_excl_scan(f, lane, wave, rt["scan"])
    gpu.barrier()
    if f == 1:
        fx.ptr_store(off, rt["flag"] + tid)
        fx.ptr_store(fx.Int32(tid), rt["expert"] + off)
    gpu.barrier()
    if tid < s * TOPK:
        t = tid // TOPK
        k = tid % TOPK
        e = fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * kt + k)), E - 1), 0)
        slot = fx.ptr_load(rt["flag"] + e)
        fx.ptr_store(fx.Int32(k), rt["kof"] + (slot * 8 + t))
    if const_expr(c["fse"]):
        if tid < s:
            fx.ptr_store(fx.Int32(TOPK), rt["kof"] + (tot * 8 + tid))
        if tid == 0:
            fx.ptr_store(fx.Int32(E), rt["expert"] + tot)
        tot = tot + 1
    gpu.barrier()
    return uniform(tot)


# ---------------------------------------------------------------- ug
def ug_loads(c, task, on=None, live=None):
    """Task ``task``'s w13 operands for this wave (gate or up by wave % 2, K
    quarter wave / 2: 16 steps) and its scale words (a word: two steps). With
    ``on``: the shared expert's task ``task`` (no union yet), its loads out of
    bounds (no memory access, zeros) where ``on`` is false; ``live`` likewise
    for a routed task."""
    lane, wave = c["lane"], c["wave"]
    a, rt = c["args"], c["rt"]
    slot = task // UG_GROUPS
    g = task % UG_GROUPS
    gu = wave % 2
    kq = wave // 2
    rg = L.w13_group(gu, g)
    wb, sb = RI * HIDDEN, 2 * RI * HIDDEN // 32
    if on is None:
        e = uniform(fx.ptr_load(rt["expert"] + slot))
        if live is None:
            r_w = rsrc(a["w13"])
            r_s = rsrc(a["w13s"])
        else:
            # + the fused shared expert, expert E
            n_w, n_s = (E + 1) * wb, (E + 1) * sb
            r_w = rsrc(a["w13"], live.select(fx.Int32(n_w), fx.Int32(0)))
            r_s = rsrc(a["w13s"], live.select(fx.Int32(n_s), fx.Int32(0)))
    else:
        e = 0
        r_w = rsrc(a["w13"] + fx.Int64(E * wb), on.select(fx.Int32(wb), fx.Int32(0)))
        r_s = rsrc(a["w13s"] + fx.Int64(E * sb), on.select(fx.Int32(sb), fx.Int32(0)))
    base = L.w13_base(e)
    tiles = [
        fp4_tile(r_w, L.fp4_tile_dword(base, rg, HIDDEN, kq * UG_STEPS + i, lane))
        for i in range(UG_STEPS)
    ]
    row = rg * ROWS + lane % ROWS
    words = [
        scale_word(
            r_s, L.w13_scale_index(e, row, 4 * (kq * UG_STEPS + 2 * i) + lane // 16)
        )
        for i in range(UG_STEPS // 2)
    ]
    return tiles + words


@traced
def stage_ug(c, task, ops, shared=False):
    """Slot task / 16, columns 16 (task % 16) ..: the gate and up rows against
    the quantized x rows in LDS (``ops``: ``ug_loads``); silu(g) u -> fp32 ->
    INTER of each picking token (``shared``: every token's pick ``TOPK``)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt = c["rt"]
    slot = task // UG_GROUPS
    g = task % UG_GROUPS
    kq = wave // 2
    rg = L.w13_group(wave % 2, g)
    tiles = ops[:UG_STEPS]
    words = ops[UG_STEPS:]
    t = fx.min(lane % 16, s - 1)
    xrow = HIDDEN + gemv.LDS_PAD
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(UG_STEPS):
        st = kq * UG_STEPS + i
        # the code is byte (st % 2) 2 + rg % 2 of its word (e8m0_shuffle)
        code = (words[i // 2] >> ((st % 2) * 16 + (rg % 2) * 8)) & 0xFF
        xoff = t * xrow + st * L.FP4_STEP + 32 * (lane // 16)
        if const_expr(c["f4m"]):
            xq = (t * XQ_ROW + st * (L.FP4_STEP // 2)) // 4 + 4 * (lane // 16)
            xb = fx.Vector(
                fx.ptr_load(c["xl"] + xq, result_type=fx.Vector.make_type(4, fx.Int32))
            )
            sb = fx.ptr_load(
                c["xl"] + (xq_scales(s) + t * (HIDDEN // 32) + 4 * st + lane // 16)
            )
            acc = mfma_f4(tiles[i], xb, acc, code, sb)
        else:
            acc = fp4_mfmas(tiles[i], pow2(code), c["xl"], xoff, acc)
    c["st"](23)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    # thread (token, column pair): g, u summed over the K quarters
    if tid < s * (UG_COLS // 2):
        tt = tid // (UG_COLS // 2)
        cp = tid % (UG_COLS // 2)
        if const_expr(shared):
            k = fx.Int32(TOPK)
        else:
            k = fx.ptr_load(rt["kof"] + (slot * 8 + tt))
        gw = (0, 2, 4, 6)
        uw = (1, 3, 5, 7)
        v = [
            silu_mul(
                row_sum(red, 2 * cp + d, tt, waves=gw),
                row_sum(red, 2 * cp + d, tt, waves=uw),
            ).bitcast(fx.Int32)
            for d in range(2)
        ]
        if k >= 0:
            pick = tt * c["KT"] + k
            at = pick * RI + g * UG_COLS + 2 * cp
            c["put_words"](c["inter"], at, v)
    gpu.barrier()
    c["st"](24)


# ---------------------------------------------------------------- down
@traced
def load_inter(c, nu):
    """Every pick's INTER (fp32) through MXFP4 by 32 columns (RCEIL scale, the
    block's 16 column pairs on 16 lanes of a row), as the stock decode kernel
    quantizes its intermediate -> LDS (bf16 rows of RI, padded); one poll batch."""
    s, tid = c["S"], c["tid"]
    n = s * c["KT"] * RI // 2
    per = (n + THREADS - 1) // THREADS
    irow = RI + gemv.LDS_PAD
    got = c["poll"](
        [(c["inter"], 2 * fx.min(tid + THREADS * i, n - 1), 2) for i in range(per)]
    )
    for i in range_constexpr(per):
        p = fx.min(tid + THREADS * i, n - 1)
        pick = p // (RI // 2)
        cp = p % (RI // 2)
        h0, h1 = f32_of(got[i][0]), f32_of(got[i][1])
        amax = fx.max(fx.Float32(fmath.absf(h0)), fx.Float32(fmath.absf(h1)))
        for off in range_constexpr(4):
            amax = xred(amax, 1 << off, fx.max)
        e = mx_scale(amax)
        if const_expr(c["f4d"]):
            # a lane's byte; four lanes' bytes -> a word, the block's code
            if const_expr(c["hq"]):
                v = pk_fp4_f32(fx.Int32(0), h0, h1, mx_div(e), 0)
            else:
                v = mx_nibble(h0, e) | (mx_nibble(h1, e) << 4)
            v = v << (8 * (cp % 4))
            for off in range_constexpr(2):
                v = xred(v, 1 << off, lambda a, b: a | b)
            ok = tid + THREADS * i < n
            if ok & (cp % 4 == 0):
                fx.ptr_store(v, c["il"] + (pick * IQ_ROW + cp) // 4)
            if ok & (cp % 16 == 0):
                fx.ptr_store(e, c["il"] + (iq_scales(s, c["KT"]) + pick * 8 + cp // 16))
        else:
            w = bf16_pair(mx_qdq(h0, e), mx_qdq(h1, e)).bitcast(fx.Int32)
            if tid + THREADS * i < n:
                fx.ptr_store(w, c["il"] + (pick * irow + 2 * cp) // 2)
    gpu.barrier()


@traced
def load_h(c):
    """SGU -> h = bf16(silu(g) u) (SiluAndMul on bf16) -> LDS rows of SI."""
    s, tid = c["S"], c["tid"]
    n = s * SI // 2
    per = (n + THREADS - 1) // THREADS
    hrow = SI + gemv.LDS_PAD
    specs = []
    for i in range_constexpr(per):
        p = fx.min(tid + THREADS * i, n - 1)
        t = p // (SI // 2)
        cp = p % (SI // 2)
        specs += [
            (c["sgu"], (t * 2 * SI) // 2 + cp, 1),
            (c["sgu"], (t * 2 * SI + SI) // 2 + cp, 1),
        ]
    got = c["poll"](specs)
    for i in range_constexpr(per):
        p = fx.min(tid + THREADS * i, n - 1)
        t = p // (SI // 2)
        cp = p % (SI // 2)
        g, u = got[2 * i][0], got[2 * i + 1][0]
        h0 = silu_mul(bf_lo(g), bf_lo(u))
        h1 = silu_mul(bf_hi(g), bf_hi(u))
        hw = (
            fx.Vector.from_elements([h0, h1], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        if tid + THREADS * i < n:
            fx.ptr_store(hw, c["hl"] + (t * hrow + 2 * cp) // 2)
    gpu.barrier()


def down_loads(c, rgd, nu, b0):
    """The w2 tiles and scale words of a wave's batch from slot b0 (slots b0,
    b0 + 8, ...; past |U| the last slot's)."""
    lane = c["lane"]
    a, rt = c["args"], c["rt"]
    r_w = rsrc(a["w2"])
    r_s = rsrc(a["w2s"])
    row = rgd * ROWS + lane % ROWS
    tiles, words = [], []
    for j in range(c["db"]):
        sl = fx.min(b0 + WAVES * j, nu - 1)
        e = uniform(fx.ptr_load(rt["expert"] + sl))
        base = L.w2_base(e)
        tiles += [
            fp4_tile(r_w, L.fp4_tile_dword(base, rgd, RI, st, lane))
            for st in range(W2_STEPS)
        ]
        words.append(scale_word(r_s, L.w2_scale_index(e, row, lane // 16)))
    return tiles + words


@traced
def stage_down(c, rgd, nu, ops0, rgd_next=None):
    """Hidden rows 16 rgd ..: every slot's w2 against its picks' INTER (wave w
    the slots w, w + 8, ...; a batch in flight under the last one's MFMAs)
    -> fp32 (t, k) contributions times the route weight in LDS -> summed in
    top-k order -> bf16; + the shared down rows against h, times SGATE ->
    bf16 partial -> every rank's FINAL. The last batch's prefetch is row group
    ``rgd_next``'s first batch, returned (the loop runs: |U| >= TOPK > WAVES)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt, kt, fse = c["rt"], c["KT"], c["fse"]
    t = fx.min(lane % 16, s - 1)
    irow = RI + gemv.LDS_PAD
    nt = c["db"] * W2_STEPS
    if const_expr(not fse):
        # the shared down's operands first: K 256 is two chunks (waves 0, 1)
        sops = gemv.loads(lane, wave, c["args"]["w_sd"], rgd * ROWS, SI)
    for i in range_constexpr((s * kt * ROWS + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < s * kt * ROWS:
            fx.ptr_store(fx.Float32(0.0), c["contrib"] + j)
    gpu.barrier()
    b0 = wave
    ops = ops0
    while b0 < nu:
        nb = b0 + WAVES * c["db"]
        if const_expr(rgd_next is None):
            ops_n = down_loads(c, rgd, nu, fx.min(nb, nu - 1))
        else:
            more = nb < nu
            ops_n = down_loads(
                c,
                more.select(rgd, rgd_next),
                nu,
                more.select(nb, fx.min(wave, nu - 1)),
            )
        for j in range_constexpr(c["db"]):
            sl = fx.min(b0 + WAVES * j, nu - 1)
            live = (b0 + WAVES * j) < nu
            k = fx.ptr_load(rt["kof"] + (sl * 8 + t))
            picked = (k >= 0) & (lane % 16 < s)
            pick = t * kt + fx.max(k, 0)
            word = ops[nt + j]
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(W2_STEPS):
                code = (word >> (st * 16 + (rgd % 2) * 8)) & 0xFF
                xoff = pick * irow + st * L.FP4_STEP + 32 * (lane // 16)
                tile = ops[j * W2_STEPS + st]
                if const_expr(c["f4d"]):
                    xq = (pick * IQ_ROW + st * (L.FP4_STEP // 2)) // 4 + 4 * (
                        lane // 16
                    )
                    xb = fx.Vector(
                        fx.ptr_load(
                            c["il"] + xq, result_type=fx.Vector.make_type(4, fx.Int32)
                        )
                    )
                    xb = fx.Vector.from_elements(
                        [picked.select(xb[d], fx.Int32(0)) for d in range(4)],
                        fx.Int32,
                    )
                    sb = fx.ptr_load(
                        c["il"] + (iq_scales(s, kt) + pick * 8 + 4 * st + lane // 16)
                    )
                    acc = mfma_f4(tile, xb, acc, code, sb)
                for q in range_constexpr(0 if c["f4d"] else 4):
                    xb = fx.Vector(
                        fx.ptr_load(
                            c["il"] + (xoff + 8 * q) // 2,
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                    )
                    xb = fx.Vector.from_elements(
                        [picked.select(xb[d], fx.Int32(0)) for d in range(4)],
                        fx.Int32,
                    )
                    a = fp4x8_bf16(tile[q], pow2(code))
                    acc = mfma_bf16(
                        a.bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
                    )
            if live & picked:
                wt = f32_of(fx.ptr_load(rt["route"] + (t * 2 * kt + kt + fx.max(k, 0))))
                for i in range_constexpr(4):
                    row = 4 * (lane // 16) + i
                    fx.ptr_store(
                        acc[i] * wt,
                        c["contrib"] + ((t * kt + fx.max(k, 0)) * ROWS + row),
                    )
        b0 = nb
        ops = ops_n
    if const_expr(not fse):
        # the shared down: rows 16 rgd .. against h
        acc = gemv.mfmas(lane, wave, c["hl"], SI + gemv.LDS_PAD, SI, s, sops)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        if const_expr(not fse):
            sg = f32_of(c["poll"]([(c["sgate"], tt, 1)])[0][0])
        ys = []
        for d in range_constexpr(2):
            r = 2 * rp + d
            y = fx.Float32(0.0)
            for k in range_constexpr(kt):
                y = y + fx.ptr_load(c["contrib"] + ((tt * kt + k) * ROWS + r))
            if const_expr(fse):
                ys.append(bf16_round(y))
            else:
                sd = bf16_round(bf16_round(row_sum(red, r, tt)) * sg)
                ys.append(bf16_round(bf16_round(y) + sd))
        for p in range_constexpr(TP):
            c["put_bf"](
                c["peer"]["final"][p],
                (c["rank"] * s + tt) * HIDDEN + rgd * ROWS + 2 * rp,
                ys,
            )
    gpu.barrier()
    return ops


# ---------------------------------------------------------------- final
@traced
def stage_final(c, j):
    """Columns 128 j ..: FINAL in rank order -> bf16 out."""
    s, tid = c["S"], c["tid"]
    if tid < s * 64:
        t = tid // 64
        cp = tid % 64
        col = j * COLS + 2 * cp
        lo, hi = sum_partials(
            c["poll"],
            c["peer"]["final_own"],
            lambda src: ((src * s + t) * HIDDEN + col) // 2,
            TP,
        )
        w = (
            fx.Vector.from_elements([lo, hi], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        bo.buffer_store(w, rsrc(c["args"]["out"]), (t * HIDDEN + col) // 2)
    gpu.barrier()


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: K2Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernels take 1..{L.MAX_TOKENS}")
    need(not key.xq or (key.f4m and not key.cx), "xq takes f4m without cx")
    need(not key.ps or key.fse, "ps takes fse")
    need(not key.u1 or key.lr, "u1 takes lr")
    need(not key.ls or (key.fse and key.lr and key.cx), "ls takes fse, lr, cx")
    lay = scratch_layout(key)
    # every width at the widest's offsets: a step's parity half never overlaps
    # the other half of a step of another width
    play = peer_layout(L.MAX_TOKENS)
    half = peer_bytes(L.MAX_TOKENS) // 2

    kt = key.picks

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * kt, 16]
        flag: fx.Array[fx.Int32, E, 16]
        expert: fx.Array[fx.Int32, MAX_U, 16]
        kof: fx.Array[fx.Int32, MAX_U * 8, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, s * (HIDDEN + gemv.LDS_PAD) // 2, 16]

    @fx.struct
    class DownLds:
        il: fx.Array[fx.Int32, s * kt * (RI + gemv.LDS_PAD) // 2, 16]
        hl: fx.Array[fx.Int32, 16 if key.fse else s * (SI + gemv.LDS_PAD) // 2, 16]
        contrib: fx.Array[fx.Float32, s * kt * ROWS, 16]

    @fx.union
    class StageLds:
        x: XLds
        down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]
        rs: fx.Array[fx.Float32, 16, 16]

    keyed = key_tuple(key, digest())
    name = (
        f"qwen38_mono_k2_s{s}_g{int(key.diag)}_p{key.stop}"
        f"_f{int(key.fse)}_o{int(key.oq)}_t{int(key.timeline)}_l{int(key.lr)}_u{key.ugd}_x{int(key.cx)}_m{int(key.f4m)}_d{key.db}_q{int(key.f4d)}_w{int(key.sp)}_h{int(key.hq)}_z{int(key.xq)}_p{int(key.ps)}_e{int(key.dp)}_k{int(key.u1)}_a{int(key.og)}_y{int(key.mp)}_g{int(key.ls)}_v{int(key.ul)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64,
        residual: Int64,
        w_o: Int64,
        w_os: Int64,
        ln_w: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w_sgu: Int64,
        w_sd: Int64,
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
        layer: Int32,
        diag: Int64,
        gate: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        rl = alloc.allocate(RouteLds).peek()
        st_lds = alloc.allocate(StageLds)
        xl, dl = st_lds.x.peek(), st_lds.down.peek()
        if const_expr(key.diag):
            mb = Mailbox(step_tag(epoch, layer) - 1, diag, region_ids(REGIONS))
        else:
            mb = Mailbox(step_tag(epoch, layer) - 1)
        on, tls = key.timeline, lds.tls.ptr
        stamp_begin(on, tls, tid, TL_POINTS)
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, TP)
        own = load_ptr64(peers, rank)
        peer = {}
        for region in ("ar", "rlog", "final"):
            off = play[region][0]
            peer[region] = [
                shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
            ]
            peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
        c = {
            "st": lambda k: stamp(on, tls, tid, k),
            "S": s,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "eps": key.eps,
            "fse": key.fse,
            "oq": key.oq,
            "lr": key.lr,
            "cx": key.cx,
            "f4m": key.f4m,
            "db": key.db,
            "f4d": key.f4d,
            "sp": key.sp,
            "KT": kt,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "rs": lds.rs.ptr,
            "peer": peer,
            "xl": xl.xl.ptr,
            "il": dl.il.ptr,
            "hl": dl.hl.ptr,
            "contrib": dl.contrib.ptr,
            "rt": {
                "route": rl.route.ptr,
                "flag": rl.flag.ptr,
                "expert": rl.expert.ptr,
                "kof": rl.kof.ptr,
                "scan": rl.scan.ptr,
            },
            "xbuf": scratch + fx.Int64(lay["xrow"][0]),
            "hq": key.hq,
            "u1": key.u1,
            "mp": key.mp,
            "ls": key.ls,
            "og": key.og,
            "xq": scratch + fx.Int64(lay["xq"][0]) if key.xq else None,
            "args": {
                "core": core,
                "gate": gate,
                "residual": residual,
                "w_o": w_o,
                "w_os": w_os,
                "ln_w": ln_w,
                "w_gate": w_gate,
                "w_sg": w_sg,
                "w_sgu": w_sgu,
                "w_sd": w_sd,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "out": out,
                "res_out": res_out,
            },
        }
        for region in (
            "msq",
            "xrdy",
            "rpart",
            "sgu",
            "sgate",
            "route",
            "inter",
        ):
            c[region] = sreg(scratch, lay[region][0], region)
        r0, r_tasks = R0, R_TASKS
        is_r = (bid >= r0) & (bid < r0 + r_tasks)
        is_g = (bid >= G0) & (bid < G0 + G_TASKS)
        nj = OTASKS // BLOCKS
        # ---- o_proj (both tasks' weights in flight first; a router CTA's
        # router weights behind them, they land before x does), then the
        # post-attention add + norm, router, shared gate | up
        if is_r:
            oops = [o_loads(c, j) for j in range(nj)]
            ri = bid - r0
            rops = router_loads(c, ri // R_PARTS, ri % R_PARTS)
            run_oproj(c, oops)
            stamp(on, tls, tid, 1)
            if const_expr(key.stop > 2):
                stage_router(c, ri // R_PARTS, ri % R_PARTS, rops)
        if ~is_r:
            run_oproj(c, [o_loads(c, j) for j in range(nj)])
            stamp(on, tls, tid, 1)
        if const_expr(key.stop > 2):
            if bid < NRED:
                stage_norm(c, bid)
            if const_expr(not key.fse):  # noqa: SIM102
                if is_g:
                    gi = bid - G0
                    gops = gemv.loads(c["lane"], c["wave"], w_sgu, gi * ROWS, HIDDEN)
                    stage_sgu(c, gi, gops)
        stamp(on, tls, tid, 2)
        if const_expr(key.stop > 3):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if const_expr(not key.ls):  # noqa: SIM102
                if (bid >= RT0) & (bid < RT0 + s):
                    stage_sgate(c, bid - RT0)
                    if const_expr(not key.lr):
                        stage_route(c, bid - RT0)
        stamp(on, tls, tid, 3)
        # ---- the routed experts
        if const_expr(key.stop > 4):
            # x is ready a route's latency before ROUTE: load and quantize it
            # while the route CTAs run
            is_ps = (bid >= PS0) & (bid < PS0 + UG_GROUPS)
            if const_expr(key.ps):
                sops = ug_loads(c, fx.min(fx.max(bid - PS0, 0), UG_GROUPS - 1), is_ps)
            if const_expr(key.xq):
                load_xq(c)
            else:
                wait_x(c, HIDDEN, 0)
            stamp(on, tls, tid, 13)
            if const_expr(key.ls):
                local_sgate(c)
            if const_expr(key.xq):
                pass
            elif const_expr(key.f4m):
                quant_x_fp4(c)
            else:
                quant_x(c)
            stamp(on, tls, tid, 14)
            if const_expr(key.ps):  # noqa: SIM102
                if is_ps:
                    stage_ug(c, bid - PS0, sops, True)
            if const_expr(key.lr):
                local_route(c)
            stamp(on, tls, tid, 15)
            nu = load_route(c)
            stamp(on, tls, tid, 4)
            total = (nu - int(key.ps)) * UG_GROUPS
            last = total - 1
            # reversed placement: the round's leftover tasks go to the high
            # CTAs, past the norm / router / route ones
            task = BLOCKS - 1 - bid
            ahead = key.ugd - 1

            def task_loads(n):
                live = (n <= last) if key.ul else None
                return ug_loads(c, fx.min(n, last), live=live)

            ops = [task_loads(task + BLOCKS * j) for j in range(ahead)]
            while task < total:
                ops_n = task_loads(task + BLOCKS * ahead)
                stage_ug(c, task, ops[0])
                task = task + BLOCKS
                ops = ops[1:] + [ops_n]
            stamp(on, tls, tid, 5)
            if const_expr(key.stop > 5):
                dops = down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
                if const_expr(key.dp):
                    dops2 = down_loads(c, bid + BLOCKS, nu, fx.min(c["wave"], nu - 1))
                if const_expr(not key.fse):
                    load_h(c)
                load_inter(c, nu)
                stamp(on, tls, tid, 6)
                if const_expr(key.dp):
                    stage_down(c, bid, nu, dops)
                else:
                    dops2 = stage_down(c, bid, nu, dops, bid + BLOCKS)
                stamp(on, tls, tid, 7)
                stage_down(c, bid + BLOCKS, nu, dops2)
                stamp(on, tls, tid, 8)
                if bid < NRED:
                    stage_final(c, bid)
                stamp(on, tls, tid, 9)
        stamp_flush(on, tls, diag, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        core: Int64,
        residual: Int64,
        w_o: Int64,
        w_os: Int64,
        ln_w: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w_sgu: Int64,
        w_sd: Int64,
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
        layer: Int32,
        diag: Int64,
        gate: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            core,
            residual,
            w_o,
            w_os,
            ln_w,
            w_gate,
            w_sg,
            w_sgu,
            w_sd,
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
            layer,
            diag,
            gate,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k2, launch)
    _BUILDS[key] = launch
    return launch


def layer_post(
    key: K2Build,
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
    layer,
    w_os=None,
    w_sgu=None,
    w_sd=None,
    diag=None,
    gate=None,
):
    """Host launcher; every weight as vLLM holds it after loading, the experts'
    shuffled (``layout``). ``w_os``: an fp4 o_proj's scales (``key.oq``);
    ``w_sgu`` / ``w_sd``: the unfused shared expert (not ``key.fse``); ``gate``:
    the attention output gate (``key.og``), core's shape.
    ``peers``: the int64 table of every rank's peer buffer (at least
    ``peer_bytes(S)`` each)."""
    s = key.tokens
    bf = torch.bfloat16
    fp4 = (torch.uint8, torch.float4_e2m1fn_x2)
    ne = E + int(key.fse)
    need(not (key.diag and key.timeline), "diag and timeline share ``diag``")
    if key.timeline:
        need(
            diag is not None
            and diag.numel() * diag.element_size() >= BLOCKS * TL_POINTS * 8,
            "timeline: diag holds [BLOCKS][TL_POINTS] int64",
        )
    need(
        core.shape == (s, OK) and residual.shape == (s, HIDDEN),
        "core / residual shape",
    )
    need(out.shape == (s, HIDDEN) and res_out.shape == (s, HIDDEN), "output shape")
    need(
        (gate is not None) == key.og and (gate is None or gate.shape == core.shape),
        "gate: core's shape, with og only",
    )
    need(ln_w.shape == (HIDDEN,), "norm shape")
    if key.oq:
        need(
            w_o.shape == (HIDDEN, OK // 2) and w_o.dtype in fp4,
            "o_proj: fp4x2 [8192, 1024]",
        )
        need(
            w_os is not None
            and w_os.numel() == HIDDEN * OK // 32
            and w_os.element_size() == 1,
            "o_proj scales: e8m0, one a 32-block",
        )
    else:
        need(w_o.shape == (HIDDEN, OK) and w_o.dtype == bf, "o_proj: bf16 [8192, 2048]")
    need(w_gate.shape == (E, HIDDEN) and w_sg.numel() == HIDDEN, "router shape")
    if not key.fse:
        need(
            w_sgu is not None
            and w_sd is not None
            and w_sgu.shape == (2 * SI, HIDDEN)
            and w_sd.shape == (HIDDEN, SI),
            "shared expert shape",
        )
    need(
        w13.shape == (ne, 2 * RI, HIDDEN // 2) and w13.dtype in fp4,
        f"w13: fp4x2 [{ne}, 512, 4096]",
    )
    need(
        w2.shape == (ne, HIDDEN, RI // 2) and w2.dtype in fp4,
        f"w2: fp4x2 [{ne}, 8192, 128]",
    )
    need(
        w13s.numel() == ne * 2 * RI * HIDDEN // 32 and w13s.element_size() == 1,
        "w13 scales: e8m0, one a 32-block",
    )
    need(
        w2s.numel() == ne * HIDDEN * RI // 32 and w2s.element_size() == 1,
        "w2 scales: e8m0, one a 32-block",
    )
    used = [w_o, w_os, w_gate, w_sg, w_sgu, w_sd, w13, w13s, w2, w2s]
    used += [gate]
    tensors = [core, residual, out, res_out, ln_w] + [w for w in used if w is not None]
    need(all(t.is_contiguous() for t in tensors), "a non-contiguous input")
    dense = [core, residual, out, res_out, ln_w, w_gate, w_sg]
    dense += [gate] if gate is not None else []
    dense += [w for w in (w_sgu, w_sd) if w is not None and not key.fse]
    need(all(t.dtype == bf for t in dense), "activations and dense weights: bf16")

    def ptr(t):
        return 0 if t is None else t.data_ptr()

    f = build(key)
    f(
        *ABI.pack(
            {
                "core": core.data_ptr(),
                "residual": residual.data_ptr(),
                "w_o": w_o.data_ptr(),
                "w_os": ptr(w_os) if key.oq else 0,
                "ln_w": ln_w.data_ptr(),
                "w_gate": w_gate.data_ptr(),
                "w_sg": w_sg.data_ptr(),
                "w_sgu": 0 if key.fse else ptr(w_sgu),
                "w_sd": 0 if key.fse else ptr(w_sd),
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
                "layer": layer,
                "diag": 0 if diag is None else diag.data_ptr(),
                "gate": ptr(gate),
            }
        ),
        stream=torch.cuda.current_stream(),
    )
