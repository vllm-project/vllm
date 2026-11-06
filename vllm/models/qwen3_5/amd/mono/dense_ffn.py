# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``dense_ffn``: the FFN half of a dense Qwen3.8-27B decoder layer as one
persistent launch, ``grid`` x ``THREADS``, S <= 8 decode rows (TP1 below):

    norm     (every CTA, redundantly; the first gate_up weights already in
             flight): h + residual (fp32) -> ``res_out`` (bf16, spread over the
             CTAs); post_attention_layernorm (Gemma, 1 + w) -> bf16 x -> MXFP4
             qdq (1 x 32 blocks, Quark "even" scales, as aiter's Triton
             ``dynamic_mxfp4_quant``) -> LDS
    gate_up  (1088 groups of 16 intermediate columns: CTA b groups 4 b .. 4 b + 3
             whole, then 4 columns of group 1024 + b / 4; wave w: gate / up by
             w % 2, K quarter w / 2): MXFP4 weights against the x rows in the
             stock GEMM's K chains (below) -> bf16 g, u -> bf16(bf16(silu(g)) u);
             each pair of whole groups (a 32-column block) through MXFP4 qdq ->
             HQ, the 4 columns bf16 -> H; one ready flag a CTA
    down     (CTA b: hidden row group b whole, then 4 rows of group 256 + b / 4;
             wave w: the group by w % 2, K quarter w / 2): HQ, and H's other
             columns through MXFP4 qdq, -> LDS; the weights against them in the
             stock GEMM's K chains -> bf16 ``out``

At TP2 (``FfnBuild.tp``, ``normed``) a rank runs its half of the intermediate
(``FGeo``) from the stock post_attention_layernorm's output, qdq'd per CTA, and
writes down's partial before the all-reduce: 544 groups, 2 whole a CTA and 2
columns of the 32 others.

The CTA counts above are the 256-CTA grid (``FfnBuild.grid``, MI355X). On 128
CTAs (MI350P, ``tiling``) a CTA takes 8 whole gate_up groups and 8 columns of
a split one [TP2: 4 and 4], down's whole row groups b and 128 + b and 8 rows of
split group 256 + b / 2 in two passes (the two whole tiles, then the split one
on the even waves), and o_proj's owners 128 + b (b < 32) in a second pass. Up to
``LOCAL_NORM_ROWS`` rows there the owners publish their bf16 o_proj outputs and
every CTA runs the norm on the whole rows itself (``run_norm``: the residual
from memory, the owners' reduction order), one grid-wide hand-off in place of
two. Each task's arithmetic is the same at either width.

Every GEMM is bit-exact with the stock ``gemm_afp4wfp4`` by construction: each
output takes the stock kernel's K splits (``gu_splits``, ``d_splits``,
``O_SPLITS``), each a chain of scaled MFMAs over the 128-K steps in K order from
a zero accumulator (a wave's quarter continues the previous quarter's
accumulator), then the split partials added as the stock split-K reduce
(``split_total``) and one bf16 rounding. The MFMA shape (16x16x128 here,
32x32x64 in stock up to 8 rows) does not change the bits; any other K grouping
does, on rows whose exact sums span more than 24 bits.

Weights are the checkpoint's as SGLang keeps them for ``gemm_afp4wfp4``: fp4x2
[N, K / 2] row-major (low nibble the even column), e8m0 scales [N, K / 32].
A load past a weight's end (a dead slot) reads zeros at no memory traffic.
"""

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.qwen3_5.amd.mono.common.mx import FP4, mfma_scaled
from vllm.models.qwen3_5.amd.mono.common.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    memrealtime,
    plus,
    rsrc,
    traced,
    xred,
)
from vllm.models.qwen3_5.amd.mono.common.plan import (
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from vllm.models.qwen3_5.amd.mono.common.sync import Mailbox, publish, sreg
from vllm.models.qwen3_5.amd.mono.helpers import (
    pow2,
    sigmoid,
    step_tag,
)
from vllm.models.qwen3_5.amd.mono.sources import digest
from vllm.models.qwen3_5.amd.mono.tiling import (  # noqa: F401 (re-exported)
    CORE,
    DEFAULT_GRID,
    HIDDEN,
    INTER,
    MAX_TOKENS,
    MAX_TOKENS_TP2,
    O_TASKS,
    QUARTERS,
    ROWS,
    STEP,
    TP_SIZES,
    FGeo,
    check_resident,
    max_tokens,
)

# STEP: K an fp4 tile step covers (a lane: 32 fp4 of one row). TP2 takes up to
# 16 rows: a token is lane % 16 of the MFMA tile (one N-tile). A tile's K chain
# is cut into QUARTERS, wave 2 q + (gate / up or tile) the q-th.

# gate_up
GU_STEPS = HIDDEN // STEP  # 40
GU_Q = GU_STEPS // QUARTERS  # 10: a wave's K quarter
W13_DW = HIDDEN // 8  # dwords a w13 row
W13S_DW = HIDDEN // 32 // 4  # scale dwords a w13 row
XQROW = HIDDEN // 8 + 4  # i32 (8 fp4 each) an LDS x row (padded)
XSROW = HIDDEN // 32  # e8m0 codes (an i32 each) a row

# down: a CTA's whole row groups of 16 hidden rows, then rows of a split one
# (``FGeo.d_per``, ``d_sl_rows``: 1 and 4 on 256 CTAs, 2 and 8 on 128)

# The stock GEMMs' K splits on this image. These shapes have no tuned
# gemm_afp4wfp4 config, so M<=8 uses DEFAULT.json (NUM_KSPLIT 4, BLOCK_K 512)
# after get_splitk, and wider steps use NUM_KSPLIT 1. o_proj (K 6144) stays 4.
# Other AITER versions split differently (``get_splitk`` of AITER main at
# 59fae207 gives gate_up 2, down 2 at TP1 and 1 at TP2): a build takes the
# installed AITER's counts through ``FfnBuild.gu_split`` / ``d_split`` /
# ``o_split`` (``stock_split_fields``), these defaults where they agree.
O_SPLITS = 4
SPLIT_COUNTS = (1, 2, 4)  # K splits a tile's chain takes: whole, halves, quarters


def gu_splits(s: int) -> int:
    """Stock gate_up K splits (K 5120): 4 up to 8 rows, 1 above."""
    return 4 if s <= MAX_TOKENS else 1


def d_splits(s: int, tp: int) -> int:
    """Stock down K splits up to 8 rows: 4 at TP1 (K 17408), 2 at TP2 (K 8704).
    Wider steps are 1 split."""
    if s > MAX_TOKENS:
        return 1
    return 4 if tp == 1 else 2


@functools.cache
def stock_splits(m: int, n: int, k: int) -> int:
    """K splits of the stock MXFP4 GEMM (AITER Triton ``gemm_afp4wfp4``) at M
    ``m`` for an [n, k] weight: the partials it returns with ``skip_reduce``.
    Runs one small GEMM on the GPU, once per shape."""
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    xq, xs = dynamic_mxfp4_quant(torch.zeros(m, k, dtype=torch.bfloat16, device="cuda"))
    w = torch.zeros(n, k // 2, dtype=torch.uint8, device="cuda")
    ws = torch.full((n, k // 32), 127, dtype=torch.uint8, device="cuda")
    p = gemm_afp4wfp4(xq, w, xs, ws, torch.bfloat16, skip_reduce=True)
    return p.size(0) if p.dim() == 3 else 1


def stock_split_fields(s: int, tp: int, oproj: bool) -> dict:
    """``FfnBuild`` fields for the installed AITER's K splits at S rows: each 0
    where it equals this module's default (so the build is #61573's), else the
    count. ValueError for a count K2 cannot take."""
    inter = FGeo(tp).inter
    got = {
        "gu_split": (stock_splits(s, 2 * inter, HIDDEN), gu_splits(s)),
        "d_split": (stock_splits(s, HIDDEN, inter), d_splits(s, tp)),
    }
    if oproj:
        got["o_split"] = (stock_splits(s, HIDDEN, CORE), O_SPLITS)
    for name, (n, _) in got.items():
        if n not in SPLIT_COUNTS:
            raise ValueError(
                f"the stock MXFP4 GEMM splits {name[:-6]}'s K in {n} at {s} rows; "
                f"K2 reproduces {SPLIT_COUNTS} splits only"
            )
    return {name: 0 if n == default else n for name, (n, default) in got.items()}


def split_total(vals):
    """fp32 split partials added as the stock split-K reduce: ``tl.sum`` over
    the split axis padded with zeros to a power of 2 (adjacent pairs, then
    pairs of pairs); one split is the chain itself."""
    v = list(vals)
    while len(v) & (len(v) - 1):
        v.append(fx.Float32(0.0))
    while len(v) > 1:
        v = [v[2 * i] + v[2 * i + 1] for i in range(len(v) // 2)]
    return v[0]


# FGeo (``tiling``): a TP rank's share of the FFN on the build's grid. TP1: 1088
# groups, down's K 136 steps; TP2: 544, 68 steps.

# o_proj (FfnBuild.oproj): owner task o < O_CTAS has hidden columns 32 o .. 32 o
# + 31; CTA b runs owners b, b + grid, ... (one pass on 256 CTAs, two on 128)
O_CTAS = O_TASKS  # 160
OQROW = CORE // 8 + 4
OSROW = CORE // 32
X_PAIRS = 5  # a (token, owner) x hand-off: 4 fp4 words, the code

# norm
CHUNKS = HIDDEN // 8  # 640 chunks of 8 columns a row
CH_WAVES = CHUNKS // 64  # 10 wave-sized runs of chunks a row
# The stock Triton Gemma norm (minimax_m3_rmsnorm, 8192-wide block): 16 warps,
# a thread 8 columns; DPP row_shr 8..1, row_bcast 15 / 31 in a warp.
NORM_WARPS = 16
WAVE_ORDER = (8, 4, 2, 1, 16, 32)
MAX_ROUNDS = (MAX_TOKENS * CHUNKS + THREADS - 1) // THREADS  # 10

# the grid-dependent invariants (whole-group pairs, column / row pairs) are
# checked for every validated grid by ``tiling.check_ffn``
assert CHUNKS % 64 == 0

_STREAM = fx.Stream(None)
STAMPS = 8  # debug stamps a CTA


def stamp(c, k):
    """Thread 0: the 100 MHz clock -> stamp k of this CTA (debug builds)."""
    if c["stamps"] is not None:
        _stamp(c, k)


@traced
def _stamp(c, k):
    t = memrealtime()
    if c["tid"] == 0:
        bo.buffer_store(
            fx.Vector.from_elements(
                [fx.Int32(t & 0xFFFFFFFF), fx.Int32(t >> 32)], fx.Int32
            ),
            rsrc(c["stamps"]),
            (c["bid"] * STAMPS + k) * 2,
        )


@dataclass(frozen=True)
class FfnBuild:
    tokens: int
    eps: float = 1e-6
    # debug: phases < stop only (1 norm, 2 gate_up, 3 down)
    stop: int = 99
    # debug: per-CTA s_memrealtime stamps at the phase ends -> scratch "stamps"
    stamps: bool = False
    # weights in aiter's shuffle_weight(layout=(16, 16)) order: a 16-row
    # group's 16 B K chunks as [chunk][row % 16][16 B] (``shuffle_w``)
    shuffled: bool = False
    # o_proj / out_proj folded in: ``hidden`` is its input (the attention's
    # core, bf16 [S, CORE]); MXFP4 qdq + the GEMV by the owner CTAs, then the
    # norm distributed (sum-of-squares and fp4 x hand-offs). Needs ``shuffled``.
    oproj: bool = False
    # the rank's share at TP ``tp`` (``FGeo``)
    tp: int = 1
    # ``hidden`` is post_attention_layernorm's bf16 output: no residual, no
    # res_out; ``out`` gets down's output (at TP2 the rank's partial)
    normed: bool = False
    # CTAs, one a CU, all resident (``tiling.resolve_grid``): 256 MI355X, 128
    # MI350P
    grid: int = DEFAULT_GRID
    # the stock GEMMs' K splits of gate_up, down and o_proj (1, 2 or 4); 0: the
    # defaults ``gu_splits`` / ``d_splits`` / ``O_SPLITS`` (``stock_split_fields``)
    gu_split: int = 0
    d_split: int = 0
    o_split: int = 0


# Up to LOCAL_NORM_ROWS rows on a grid narrower than 256, the o_proj owners
# publish their outputs once and every CTA runs the norm on the whole rows itself
# (``run_norm``, the owners' reduction order) in place of the owners' two
# grid-wide hand-offs (sum-of-squares partials, then x). Wider, the rows kept
# next to the in-flight gate_up weights spill.
LOCAL_NORM_ROWS = 4


def local_norm(key: "FfnBuild") -> bool:
    return key.oproj and key.grid != DEFAULT_GRID and key.tokens <= LOCAL_NORM_ROWS


def key_splits(key: "FfnBuild") -> tuple[int, int, int]:
    """(gate_up, down, o_proj) K splits of a build."""
    s = key.tokens
    return (
        key.gu_split or gu_splits(s),
        key.d_split or d_splits(s, key.tp),
        key.o_split or O_SPLITS,
    )


ABI = KernelAbi(
    (
        "hidden",
        "residual",
        "w_o",
        "w_os",
        "ln_w",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "out",
        "res_out",
        "scratch",
        "epoch",
        "layer",
    )
)


def scratch_layout(key: FfnBuild) -> dict:
    s = key.tokens
    geo = FGeo(key.tp, key.grid)
    lay = pair_layout(
        (
            ("hrdy", key.grid),
            ("nsum", CHUNKS * s),
            ("xmb", s * O_CTAS * X_PAIRS),
            ("omb", s * HIDDEN // 2 if local_norm(key) else 0),
        )
    )
    end = max(o + n for o, n in lay.values())
    # H: bf16 [S, inter] (the split groups' columns); HQ: its whole groups'
    # blocks as fp4 words [S, blocks, 4] and e8m0 codes [S, blocks], an i32 each
    lay["h"] = ((end + 255) // 256 * 256, s * geo.inter * 2)
    lay["hq"] = (lay["h"][0] + lay["h"][1], s * geo.inter // 2)
    lay["hqs"] = (lay["hq"][0] + lay["hq"][1], s * geo.blocks * 4)
    lay["stamps"] = (lay["hqs"][0] + lay["hqs"][1], key.grid * STAMPS * 8)
    return lay


def scratch_bytes(key: FfnBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def mx_code_even(amax):
    """A 1 x 32 block's e8m0 code from its abs max, Quark "even" (aiter's Triton
    ``_mxfp4_scale_from_amax``): the exponent of amax rounded up past 1.75,
    minus 2, clamped to 0 .. 254."""
    bits = amax.bitcast(fx.Int32)
    e = ((bits + fx.Int32(0x200000)) >> 23) & 0xFF
    return fx.min(fx.max(e - 2, fx.Int32(0)), fx.Int32(254))


def pack8(vs, code):
    """8 fp32 -> e2m1 at e8m0 ``code`` (v_cvt_scalef32_pk_fp4_f32: nearest even,
    saturating, bit-exact with aiter's Triton quant) -> an i32, element 2 j / 2 j
    + 1 the low / high nibble of byte j. ``code`` >= 1 (``mx_code``)."""
    sc = pow2(code)
    w = fx.Int32(0)
    for j in range(4):
        w = fx.Int32(
            rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, w, fx.Float32(vs[2 * j]), fx.Float32(vs[2 * j + 1]), sc, j
            )
        )
    return w


def mx_code(amax):
    """``mx_code_even`` with code 0 (an all-zero block: its fp4 are 0 at any
    scale) taken as 1, a scale the conversion can divide by."""
    return fx.max(mx_code_even(amax), fx.Int32(1))


def ld4(r, at, cm=0):
    return fx.Vector(bo.buffer_load(r, at, vec_width=4, dtype=T.i32, cache_modifier=cm))


def pack_bf(vs):
    """f32 list (even length) -> i32 list of packed bf16 pairs."""
    return [
        fx.Vector.from_elements([vs[2 * j], vs[2 * j + 1]], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
        for j in range(len(vs) // 2)
    ]


def tl_sqrt(x):
    """Triton's ``tl.sqrt``: the raw v_sqrt_f32."""
    return fx.Float32(
        llvm.call_intrinsic(
            T.f32, "llvm.amdgcn.sqrt.f32", [fx.Float32(x).ir_value()], [], []
        )
    )


def chunk_ss(z):
    """A chunk's sum of squares as the stock norm's ISA forms it: packed pairs,
    its mul / add tree fp-contracted."""
    sq = lambda e: fx.Float32(fmath.fma(z[e], z[e], z[e + 2] * z[e + 2]))  # noqa: E731
    return (sq(0) + sq(4)) + (sq(1) + sq(5))


def _add_live(x, y):
    """X + y, None standing for a zero partial."""
    return x if y is None else y if x is None else x + y


def runs_total(runs):
    """A row's CH_WAVES run sums combined as the stock norm's NORM_WARPS warp
    partials (xor 8, 4, 2, 1; warps past the row hold zeros)."""
    v = list(runs) + [None] * (NORM_WARPS - len(runs))
    for off in (8, 4, 2, 1):
        v = [_add_live(v[i], v[i + off]) for i in range(off)]
    return v[0]


def stock_rstd(tot, eps):
    """The stock norm's 1 / sqrt(sum / n + eps): IEEE divisions, raw sqrt."""
    return fx.Float32(1.0) / tl_sqrt(tot / fx.Float32(HIDDEN) + fx.Float32(eps))


def silu_mul_bf(g, u):
    """SiluAndMul on ROCm: silu(g) rounded to bf16 before the multiply."""
    return bf16_round(g * sigmoid(g)) * u


# ---------------------------------------------------------------- norm
def norm_loads(c, first=False):
    """Chunk ``tid + 512 i`` of the S rows (8 columns; every round's live chunks
    of a wave in one row): raw h / residual words and the 1 + w words. ``first``:
    no residual (the model's first layer)."""
    s, tid, a = c["S"], c["tid"], c["args"]
    n = s * CHUNKS
    out = []
    for i in range((n + THREADS - 1) // THREADS):
        ch = fx.min(tid + THREADS * i, n - 1)
        hw = ld4(rsrc(a["hidden"]), ch * 4)
        rw = None if first else ld4(rsrc(a["residual"]), ch * 4)
        out.append((hw, rw, ld4(rsrc(a["ln_w"]), ch % CHUNKS * 4)))
    return out


@traced
def run_norm(c, raw, marks=True):
    """Z = h + residual (h alone without a residual) -> res_out (chunk run
    ch / 16 by CTA run % grid); every row's sum of squares (a wave's 64 chunks a
    run, as the stock norm) -> rstd -> bf16 x -> qdq by 32 columns (four lanes
    a block) -> the LDS x rows. ``marks``: debug stamps 5 and 6."""
    s, tid, lane, wave, bid = c["S"], c["tid"], c["lane"], c["wave"], c["bid"]
    nrm, rst = c["nrm"], c["rst"]
    a = c["args"]
    grid = c["G"].grid
    n = s * CHUNKS
    zs = []
    for i in range_constexpr(len(raw)):
        hw, rw, _ = raw[i]
        z = []
        for d in range_constexpr(4):
            if const_expr(rw is None):
                z += [bf_lo(hw[d]), bf_hi(hw[d])]
            else:
                z += [bf_lo(hw[d]) + bf_lo(rw[d]), bf_hi(hw[d]) + bf_hi(rw[d])]
        zs.append(z)
        ss = butterfly(chunk_ss(z), WAVE_ORDER)
        if lane == 0:
            fx.ptr_store(ss, nrm + (i * WAVES + wave))
    gpu.barrier()
    if const_expr(marks):
        stamp(c, 5)
    if tid < s:
        tot = runs_total(
            [fx.ptr_load(nrm + (tid * CH_WAVES + k)) for k in range(CH_WAVES)]
        )
        fx.ptr_store(stock_rstd(tot, c["eps"]), rst + tid)
    gpu.barrier()
    if const_expr(marks):
        stamp(c, 6)
    for i in range_constexpr(len(raw)):
        ch_raw = tid + THREADS * i
        live = ch_raw < n
        ch = fx.min(ch_raw, n - 1)
        t = ch // CHUNKS
        col = ch % CHUNKS * 8
        z = zs[i]
        wv = raw[i][2]
        lnw = []
        for d in range_constexpr(4):
            lnw += [fx.Float32(1.0) + bf_lo(wv[d]), fx.Float32(1.0) + bf_hi(wv[d])]
        if live & ((ch // 16) % grid == bid):
            bo.buffer_store(
                fx.Vector.from_elements(pack_bf(z), fx.Int32),
                rsrc(a["res_out"]),
                ch * 4,
            )
        r = fx.ptr_load(rst + t)
        x = [bf16_round(z[e] * r * lnw[e]) for e in range(8)]
        amax = fx.Float32(0.0)
        for e in range_constexpr(8):
            amax = fx.max(amax, fx.Float32(fmath.absf(x[e])))
        amax = xred(xred(amax, 1, fx.max), 2, fx.max)
        code = mx_code(amax)
        w = pack8(x, code)
        if live:
            fx.ptr_store(w, c["xq"] + (t * XQROW + col // 8))
            if ch % 4 == 0:
                fx.ptr_store(code, c["xs"] + (t * XSROW + col // 32))
    gpu.barrier()


# ---------------------------------------------------------------- o_proj
def owner(c, own=None):
    """Owner task ``own``, the CTA's first (``bid``) by default."""
    return c["bid"] if own is None else own


def o_owner(c, own=None):
    """Owner task ``own`` (the CTA's first by default), clamped to a real one."""
    return fx.min(owner(c, own), O_CTAS - 1)


def o_more(c):
    """The CTA's owner tasks past its first: ``bid + p grid`` for passes p >= 1
    (none on 256 CTAs; 128 + b on CTAs b < 32 of 128)."""
    return [c["bid"] + p * c["G"].grid for p in range(1, c["G"].o_passes)]


def block_loads(c, width):
    """1 x 32 block ``tid + 512 i`` of the S ``hidden`` rows (bf16, ``width``
    wide): its 64 B."""
    s, tid, a = c["S"], c["tid"], c["args"]
    nblk = width // 32
    nb = s * nblk
    out = []
    for i in range((nb + THREADS - 1) // THREADS):
        b = fx.min(tid + THREADS * i, nb - 1)
        at = (b // nblk * width + 32 * (b % nblk)) // 2
        out.append([ld4(rsrc(a["hidden"]), at + 4 * k) for k in range(4)])
    return out


@traced
def block_quant(c, raw, width, dq, ds, qrow, srow):
    """``block_loads``' blocks through MXFP4 qdq (a thread a 1 x 32 block) ->
    LDS rows (fp4 at ``dq`` + ``qrow`` i32 a row, codes at ``ds`` + ``srow``)."""
    s, tid = c["S"], c["tid"]
    nblk = width // 32
    nb = s * nblk
    for i in range_constexpr(len(raw)):
        b = fx.min(tid + THREADS * i, nb - 1)
        t = b // nblk
        blk = b % nblk
        vs = []
        for k in range_constexpr(4):
            for d in range_constexpr(4):
                vs += [bf_lo(raw[i][k][d]), bf_hi(raw[i][k][d])]
        amax = fx.Float32(0.0)
        for e in range_constexpr(32):
            amax = fx.max(amax, fx.Float32(fmath.absf(vs[e])))
        code = mx_code(amax)
        qs = [pack8(vs[8 * k : 8 * k + 8], code) for k in range(4)]
        if tid + THREADS * i < nb:
            fx.ptr_store(
                fx.Vector.from_elements(qs, fx.Int32), dq + (t * qrow + 4 * blk)
            )
            fx.ptr_store(code, ds + (t * srow + blk))
    gpu.barrier()


def core_loads(c):
    return block_loads(c, CORE)


def o_lane(c):
    """The norm thread (token t, run j of 8 of the owner's 32 columns)."""
    tid = c["tid"]
    return fx.min(tid // 4, c["S"] - 1), tid % 4


def o_res_loads(c, own=None):
    """The norm thread's residual and 1 + w words (8 columns of owner ``own``)."""
    a = c["args"]
    t, j = o_lane(c)
    col = 2 * ROWS * o_owner(c, own) + 8 * j
    return ld4(rsrc(a["residual"]), (t * HIDDEN + col) // 2), ld4(
        rsrc(a["ln_w"]), col // 2
    )


def o_loads(c, width=CORE, splits=4, own=None):
    """Wave w: owner ``own``'s (the CTA's first by default) row group w % 2, K
    split w / 2 of ``splits`` (of o_proj's K, ``width``: the stock GEMM's split-K
    partition); past the weight (no traffic) for an owner past the last, or a
    wave past the splits."""
    lane, wave, a = c["lane"], c["wave"], c["args"]
    bid = owner(c, own)
    o_q = width // STEP // splits
    row = (2 * bid + wave % 2) * ROWS + lane % ROWS
    live = bid < O_CTAS
    if const_expr(2 * splits < WAVES):
        live = live & (wave < 2 * splits)
    row = live.select(row, fx.Int32(HIDDEN))
    r_w = rsrc(a["w_o"], HIDDEN * width // 2)
    r_s = rsrc(a["w_os"], HIDDEN * width // 32)
    steps = [wave // 2 * o_q + i for i in range(o_q)]
    tiles = [
        ld4(r_w, row // ROWS * (ROWS * width // 8) + st * 256 + lane * 4, CM_NT)
        for st in steps
    ]
    words = [
        fx.Int32(
            bo.buffer_load(r_s, row * (width // 128) + st, vec_width=1, dtype=T.i32)
        )
        for st in steps
    ]
    return tiles, words


def o_quant(c, craw):
    """The core rows through MXFP4 qdq -> LDS."""
    block_quant(c, craw, CORE, c["oq"], c["os"], OQROW, OSROW)


def x_quant(c, xraw):
    """``normed``: the x rows (the stock norm's output) through MXFP4 qdq ->
    the LDS x rows."""
    block_quant(c, xraw, HIDDEN, c["xq"], c["xs"], XQROW, XSROW)


@traced
def o_stage(c, ops, width=CORE, splits=4):
    """The wave's K split of the owner's row group, its steps chained in order
    -> red (the core rows' qdq in LDS, ``width // 8 + 4`` i32 a row)."""
    lane, wave = c["lane"], c["wave"]
    o_q = width // STEP // splits
    gl = lane // ROWS
    t = fx.min(lane % ROWS, c["S"] - 1)
    tiles, words = ops
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(o_q):
        st = wave // 2 * o_q + i
        acc = scaled_step(
            c["oq"],
            c["os"],
            t * (width // 8 + 4),
            t * (width // 32),
            fx.min(st, width // STEP - 1),
            gl,
            tiles[i],
            words[i],
            acc,
        )
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()


@traced
def o_finish(c, rraw, own=None):
    """Norm thread (t, j) of owner ``own`` (the CTA's first by default): the
    o_proj rows' K splits added as stock's reduce -> bf16 (the stock GEMM's
    output) + residual = z -> res_out; the chunk's sum of squares -> NSUM.
    Returns z, 1 + w."""
    s, tid = c["S"], c["tid"]
    bid = owner(c, own)
    t, j = o_lane(c)
    rw, wv = rraw
    res, lnw = [], []
    for d in range_constexpr(4):
        res += [bf_lo(rw[d]), bf_hi(rw[d])]
        lnw += [fx.Float32(1.0) + bf_lo(wv[d]), fx.Float32(1.0) + bf_hi(wv[d])]
    z = [
        bf16_round(
            split_total(
                red_at(c["red"], 2 * k + j // 2, 8 * (j % 2) + e, t)
                for k in range(c["o_n"])
            )
        )
        + res[e]
        for e in range(8)
    ]
    live = (tid < s * 4) & (bid < O_CTAS)
    if live:
        bo.buffer_store(
            fx.Vector.from_elements(pack_bf(z), fx.Int32),
            rsrc(c["args"]["res_out"]),
            (t * HIDDEN + 2 * ROWS * bid + 8 * j) // 2,
        )
        c["put"](c["nsum"], t * CHUNKS + 4 * bid + j, chunk_ss(z))
    return z, lnw


@traced
def o_publish(c, own=None):
    """``o_finish`` for the local norm: norm thread (t, j) of owner ``own`` (the
    CTA's first by default) puts its 8 bf16 o_proj outputs (the stock GEMM's
    output, before the residual) for every CTA."""
    s, tid = c["S"], c["tid"]
    bid = owner(c, own)
    t, j = o_lane(c)
    o = [
        bf16_round(
            split_total(
                red_at(c["red"], 2 * k + j // 2, 8 * (j % 2) + e, t)
                for k in range(c["o_n"])
            )
        )
        for e in range(8)
    ]
    if (tid < s * 4) & (bid < O_CTAS):
        at = t * HIDDEN + 2 * ROWS * bid + 8 * j
        c["put_bf"](c["omb"], at, o[:4])
        c["put_bf"](c["omb"], at + 4, o[4:])


def o_norm_loads(c):
    """``norm_loads`` with h taken from the owners' o_proj outputs (``o_publish``,
    polled) instead of memory: chunk ``tid + 512 i``'s o words, residual and
    1 + w words. The residual and 1 + w loads go first, then the polls."""
    s, tid, a = c["S"], c["tid"], c["args"]
    n = s * CHUNKS
    rounds = (n + THREADS - 1) // THREADS
    chs = [fx.min(tid + THREADS * i, n - 1) for i in range(rounds)]
    mem = [
        (ld4(rsrc(a["residual"]), ch * 4), ld4(rsrc(a["ln_w"]), ch % CHUNKS * 4))
        for ch in chs
    ]
    specs = []
    for ch in chs:
        specs += [(c["omb"], ch * 4, 2), (c["omb"], ch * 4 + 2, 2)]
    got = c["poll"](specs)
    return [
        (got[2 * i] + got[2 * i + 1], mem[i][0], mem[i][1]) for i in range(len(chs))
    ]


def o_more_loads(c, residual=True):
    """(owner, residual words, o weights) of each of the CTA's owner tasks past
    its first (``o_more``; none on 256 CTAs). No residual words for the local
    norm (``residual=False``)."""
    return [
        (
            own,
            o_res_loads(c, own) if residual else None,
            o_loads(c, splits=c["o_n"], own=own),
        )
        for own in o_more(c)
    ]


def o_more_publish(c, more, width=CORE, splits=4):
    """``o_more_passes`` for the local norm: ``o_stage`` and ``o_publish`` of
    each owner of ``more``."""
    for own, _, ops in more:
        gpu.barrier()
        o_stage(c, ops, width, splits)
        o_publish(c, own)


def o_more_passes(c, more, width=CORE, splits=4):
    """``o_stage`` and ``o_finish`` of each owner of ``more`` (``o_more_loads``),
    a barrier first: the previous ``o_finish`` reads RED, which the next chains
    overwrite. Returns (z, 1 + w, owner) each."""
    out = []
    for own, rraw, ops in more:
        gpu.barrier()
        o_stage(c, ops, width, splits)
        z, lnw = o_finish(c, rraw, own)
        out.append((z, lnw, own))
    return out


def o_more_x(c, zs):
    """``o_x`` of each owner of ``zs`` (``o_more_passes``)."""
    for z, lnw, own in zs:
        o_x(c, z, lnw, own)


@traced
def o_rstd(c):
    """Owner CTAs: wave t reduces token t's chunk sums as the stock norm (lane l:
    chunk l of each run of 64) -> rstd -> LDS."""
    s, tid, lane, bid = c["S"], c["tid"], c["lane"], c["bid"]
    if (tid < s * 64) & (bid < O_CTAS):
        t = tid // 64
        got = c["poll"](
            [(c["nsum"], t * CHUNKS + 64 * k + lane, 1) for k in range(CH_WAVES)]
        )
        runs = [butterfly(g[0].bitcast(fx.Float32), WAVE_ORDER) for g in got]
        if lane == 0:
            fx.ptr_store(stock_rstd(runs_total(runs), c["eps"]), c["rst"] + t)
    gpu.barrier()


@traced
def o_x(c, z, lnw, own=None):
    """Norm thread (t, j) of owner ``own`` (the CTA's first by default): bf16 x
    -> the 1 x 32 block's code (over the 4 threads) -> its fp4 word (and the
    code) put for every CTA."""
    s, tid = c["S"], c["tid"]
    bid = owner(c, own)
    t, j = o_lane(c)
    r = fx.ptr_load(c["rst"] + t)
    x = [bf16_round(z[e] * r * lnw[e]) for e in range(8)]
    amax = fx.Float32(0.0)
    for e in range_constexpr(8):
        amax = fx.max(amax, fx.Float32(fmath.absf(x[e])))
    amax = xred(xred(amax, 1, fx.max), 2, fx.max)
    code = mx_code(amax)
    w = pack8(x, code)
    if (tid < s * 4) & (bid < O_CTAS):
        base = (t * O_CTAS + bid) * X_PAIRS
        c["put_words"](c["xmb"], base + j, [w])
        if j == 0:
            c["put_words"](c["xmb"], base + 4, [code])


@traced
def o_gather(c):
    """Every owner's fp4 x words and codes -> the LDS x rows."""
    s, tid = c["S"], c["tid"]
    n = s * O_CTAS
    rounds = (n + THREADS - 1) // THREADS
    units = [fx.min(tid + THREADS * i, n - 1) for i in range(rounds)]
    specs = []
    for u in units:
        base = u * X_PAIRS
        specs += [(c["xmb"], base, 2), (c["xmb"], base + 2, 2), (c["xmb"], base + 4, 1)]
    got = c["poll"](specs)
    for i in range_constexpr(rounds):
        u = units[i]
        t = u // O_CTAS
        b = u % O_CTAS
        ws = [got[3 * i][0], got[3 * i][1], got[3 * i + 1][0], got[3 * i + 1][1]]
        if tid + THREADS * i < n:
            fx.ptr_store(
                fx.Vector.from_elements(ws, fx.Int32), c["xq"] + (t * XQROW + 4 * b)
            )
            fx.ptr_store(got[3 * i + 2][0], c["xs"] + (t * XSROW + b))
    gpu.barrier()


# ---------------------------------------------------------------- chains
def red_at(red, w, r, t):
    """Row r, token t of wave w's 16x16 MFMA accumulator (at ``red`` + (w * 64 +
    lane) * 4)."""
    return fx.ptr_load(red + ((w * 64 + 16 * (r // 4) + t) * 4 + r % 4))


@traced
def chained(c, buf, q, per, run):
    """The tiles' K chains, a quarter a wave (wave 2 q + its tile, quarter q
    traced): quarter q is position q % ``per`` of its split's chain. Phase p:
    the waves at position p continue the accumulator of wave - 2 (zero at a
    split's start) in ``buf`` (wave w's at (w * 64 + lane) * 4) by ``run`` and
    store theirs there."""
    lane, wave = c["lane"], c["wave"]
    vec4 = fx.Vector.make_type(4, fx.Float32)
    for p in range_constexpr(per):
        if q % per == p:
            if const_expr(p == 0):
                acc = fx.Vector.filled(4, 0.0, fx.Float32)
            else:
                acc = fx.Vector(
                    fx.ptr_load(buf + ((wave - 2) * 64 + lane) * 4, result_type=vec4)
                )
            fx.ptr_store(run(acc), buf + (wave * 64 + lane) * 4)
        gpu.barrier()


def split_ends(n, per):
    """The quarters ending the ``n`` splits (``per`` quarters each)."""
    return [per * k + per - 1 for k in range(n)]


# ---------------------------------------------------------------- gate_up
def w13_loads(c, row, steps):
    """This lane's w13 tiles of ``row`` at K steps ``steps`` (traced, each) and
    their scale words (byte lane / 16 of each)."""
    lane, a, geo = c["lane"], c["args"], c["G"]
    gl = lane // ROWS
    r_w = rsrc(a["w13"], 2 * geo.inter * HIDDEN // 2)
    r_s = rsrc(a["w13s"], 2 * geo.inter * HIDDEN // 32)
    if c["shuffled"]:
        at = [row // ROWS * (ROWS * W13_DW) + st * 256 + lane * 4 for st in steps]
    else:
        at = [row * W13_DW + st * 16 + 4 * gl for st in steps]
    tiles = [ld4(r_w, a_, CM_NT) for a_ in at]
    words = [
        fx.Int32(bo.buffer_load(r_s, row * W13S_DW + st, vec_width=1, dtype=T.i32))
        for st in steps
    ]
    return tiles, words


def gu_row(c, g, sliver):
    """This lane's w13 row of group g, gate / up by wave % 2; of the split group
    (``sliver``) only the CTA's ``sl_cols`` columns, the others dead."""
    lane, wave, geo = c["lane"], c["wave"], c["G"]
    row = ((wave % 2) * geo.ng + g) * ROWS + lane % ROWS
    if const_expr(sliver):
        off = sl_off(c)
        live = (lane % ROWS >= off) & (lane % ROWS < off + geo.sl_cols)
        row = live.select(row, fx.Int32(2 * geo.inter))
    return row


def sl_off(c):
    """The first of the CTA's columns in its split group."""
    geo = c["G"]
    return c["bid"] % (ROWS // geo.sl_cols) * geo.sl_cols


def sl_group(c):
    geo = c["G"]
    return geo.g_full * geo.grid + c["bid"] // (ROWS // geo.sl_cols)


def gu_loads(c, g, sliver=False):
    """Group g: wave w's gate / up rows (w % 2), its K quarter w / 2."""
    kq = c["wave"] // 2
    return w13_loads(c, gu_row(c, g, sliver), [kq * GU_Q + i for i in range(GU_Q)])


def gu_mfmas(c, tiles, words, steps, acc=None):
    """The steps chained in order onto ``acc`` (None: a chain from zero)."""
    if acc is None:
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
    lane = c["lane"]
    gl = lane // ROWS
    t = fx.min(lane % ROWS, c["S"] - 1)
    for i in range_constexpr(len(tiles)):
        acc = scaled_step(
            c["xq"],
            c["xs"],
            t * XQROW,
            t * XSROW,
            steps[i],
            gl,
            tiles[i],
            words[i],
            acc,
        )
    return acc


def scaled_step(aq, as_, qrow, srow, st, gl, tile, word, acc):
    """One 128-K step: the lane's 32 fp4 of its weight row (byte gl of ``word``
    its e8m0) against the 32 fp4 of its token's activation row in LDS (``aq`` +
    ``qrow`` i32, its code at ``as_`` + ``srow``), one scaled MFMA."""
    bw = fx.Vector(
        fx.ptr_load(
            aq + (qrow + st * 16 + gl * 4), result_type=fx.Vector.make_type(4, fx.Int32)
        )
    )
    sb = fx.ptr_load(as_ + (srow + st * 4 + gl))
    sa = (word >> (gl * 8)) & 0xFF
    return mfma_scaled(tile, bw, acc, sa, sb, FP4, FP4, 0, 0)


@traced
def stage_gu(c, j, g, ops, sliver):
    """Group g, the CTA's j-th: the K chains (``chained``) -> red; thread (token,
    column pair): gate and up, the splits added as stock's reduce, bf16 each ->
    h (``h_out``)."""
    s, tid, wave, red = c["S"], c["tid"], c["wave"], c["red"]
    n = c["gu_n"]
    per = QUARTERS // n
    kq = wave // 2
    steps = [kq * GU_Q + i for i in range(GU_Q)]
    chained(c, red, kq, per, lambda acc: gu_mfmas(c, ops[0], ops[1], steps, acc))
    if tid < s * (ROWS // 2):
        tt = tid // (ROWS // 2)
        cp = tid % (ROWS // 2)
        ends = split_ends(n, per)
        gv = [
            bf16_round(split_total(red_at(red, 2 * q, 2 * cp + d, tt) for q in ends))
            for d in range(2)
        ]
        uv = [
            bf16_round(
                split_total(red_at(red, 2 * q + 1, 2 * cp + d, tt) for q in ends)
            )
            for d in range(2)
        ]
        h_out(c, j, g, tt, cp, gv, uv, sliver)
    gpu.barrier()


@traced
def h_out(c, j, g, tt, cp, gv, uv, sliver):
    """bf16 h of columns 16 g + 2 cp + (0, 1), token tt: a whole group's -> the
    LDS pair (HP, its half j % 2); the split group's live columns -> H."""
    v = [silu_mul_bf(gv[d], uv[d]) for d in range(2)]
    if const_expr(sliver):
        off = sl_off(c)
        if (2 * cp >= off) & (2 * cp < off + c["G"].sl_cols):
            bo.buffer_store(
                pack_bf(v)[0],
                rsrc(c["hbuf"]),
                (tt * c["G"].inter + g * ROWS + 2 * cp) // 2,
                cache_modifier=CM_DEV,
            )
    else:
        for d in range_constexpr(2):
            fx.ptr_store(
                bf16_round(v[d]), c["hp"] + (tt * 32 + (j % 2) * ROWS + 2 * cp + d)
            )


@traced
def h_quant(c, g):
    """Thread t < S: the h of whole groups g - 1, g (one 1 x 32 block, in HP)
    through MXFP4 qdq -> its fp4 words and code -> HQ."""
    s, tid, geo = c["S"], c["tid"], c["G"]
    if tid < s:
        vs = [fx.ptr_load(c["hp"] + (tid * 32 + e)) for e in range(32)]
        amax = fx.Float32(0.0)
        for e in range_constexpr(32):
            amax = fx.max(amax, fx.Float32(fmath.absf(vs[e])))
        code = mx_code(amax)
        qs = [pack8(vs[8 * k : 8 * k + 8], code) for k in range(4)]
        blk = (g - 1) * ROWS // 32
        bo.buffer_store(
            fx.Vector.from_elements(qs, fx.Int32),
            rsrc(c["hq4"]),
            (tid * geo.blocks + blk) * 4,
            cache_modifier=CM_DEV,
        )
        bo.buffer_store(
            code, rsrc(c["hq4s"]), tid * geo.blocks + blk, cache_modifier=CM_DEV
        )


@traced
def run_gate_up(c, ops):
    """The whole groups (a group's weights in flight under the previous one,
    ``ops`` the first's since the start) and each pair's qdq, then the split
    group's columns; one flag."""
    bid, tid, geo = c["bid"], c["tid"], c["G"]
    g0 = geo.g_full * bid
    for j in range_constexpr(geo.g_full + 1):
        sliver = j == geo.g_full
        if const_expr(j + 1 < geo.g_full):
            ops_n = gu_loads(c, g0 + j + 1)
        elif const_expr(j + 1 == geo.g_full):
            ops_n = gu_loads(c, sl_group(c), sliver=True)
        stage_gu(c, j, sl_group(c) if sliver else g0 + j, ops, sliver)
        if const_expr(j % 2 == 1 and not sliver):
            h_quant(c, g0 + j)
        if const_expr(not sliver):
            ops = ops_n
    publish(c["put"], c["hrdy"], bid, fx.Int32(1), tid == 0)


def row_sum_waves(red, r, t, gu, n=4):
    """Row r, token t summed over the waves of gate (gu 0: 0, 2, 4, 6) or up
    (1, 3, 5, 7), the first ``n`` of them, in order (gu traced)."""
    tot = fx.Float32(0.0)
    for k in range(n):
        w = 2 * k + gu
        tot = tot + fx.ptr_load(red + ((w * 64 + 16 * (r // 4) + t) * 4 + r % 4))
    return tot


# ---------------------------------------------------------------- down
def d_tile_row(c, i):
    """This lane's w2 row of the CTA's tile ``i`` (``FGeo.d_pass_tiles``): of
    whole row group i grid + b, of the CTA's ``d_sl_rows`` rows of split group
    d_full + b / (16 / d_sl_rows) (the group's other rows dead), or (None) past
    the weight."""
    lane, bid, geo = c["lane"], c["bid"], c["G"]
    if i is None:
        return fx.Int32(HIDDEN)
    if i < geo.d_per:
        return plus(bid, i * geo.grid) * ROWS + lane % ROWS
    off = d_off(c)
    live = (lane % ROWS >= off) & (lane % ROWS < off + geo.d_sl_rows)
    return live.select(
        (geo.d_full + bid // (ROWS // geo.d_sl_rows)) * ROWS + lane % ROWS,
        fx.Int32(HIDDEN),
    )


def d_row(c, p=0):
    """This lane's w2 row: pass ``p``'s tile on the wave's parity. On 256 CTAs
    the one pass is whole row group b on the even waves, 4 rows of split group
    256 + b / 4 on the odd; on 128, groups b and 128 + b, then 8 rows of split
    group 256 + b / 2 on the even waves (the odd ones dead)."""
    even, odd = c["G"].d_pass_tiles(p)
    odd_row = d_tile_row(c, odd)
    return (c["wave"] % 2 == 0).select(d_tile_row(c, even), odd_row)


def d_off(c):
    """The first of the CTA's rows in its split row group."""
    geo = c["G"]
    return c["bid"] % (ROWS // geo.d_sl_rows) * geo.d_sl_rows


def d_out_base(c, i):
    """The first hidden row of the CTA's tile ``i`` (0 for none)."""
    bid, geo = c["bid"], c["G"]
    if i is None:
        return fx.Int32(0)
    if i < geo.d_per:
        return plus(bid, i * geo.grid) * ROWS
    return (geo.d_full + bid // (ROWS // geo.d_sl_rows)) * ROWS


def d_out_live(c, tile, rp, tiles):
    """Whether output thread (``tile`` parity, row pair ``rp``) writes: every row
    pair of a whole group, the CTA's of the split group, none of no tile."""
    geo = c["G"]
    out = None
    for parity, i in enumerate(tiles):
        if i is None:
            continue
        on = tile == parity
        if i == geo.d_per:
            off = d_off(c)
            on = on & ((2 * rp >= off) & (2 * rp < off + geo.d_sl_rows))
        out = on if out is None else out | on
    return out


def down_loads(c, p=0):
    """The wave's K quarter (wave / 2) of its row in pass ``p``: tiles and
    scale words."""
    lane, wave, geo, a = c["lane"], c["wave"], c["G"], c["args"]
    gl = lane // ROWS
    r_w = rsrc(a["w2"], HIDDEN * geo.inter // 2)
    r_s = rsrc(a["w2s"], HIDDEN * geo.inter // 32)
    row = d_row(c, p)
    kq = wave // 2
    tiles, words = [], []
    for i in range(geo.d_q):
        st = kq * geo.d_q + i
        if c["shuffled"]:
            at = row // ROWS * (ROWS * geo.w2_dw) + st * 256 + lane * 4
        else:
            at = row * geo.w2_dw + st * 16 + 4 * gl
        tiles.append(ld4(r_w, at, CM_NT))
        words.append(
            fx.Int32(
                bo.buffer_load(r_s, row * geo.w2s_dw + st, vec_width=1, dtype=T.i32)
            )
        )
    return tiles, words


@traced
def load_h(c):
    """Every CTA's H ready -> HQ's blocks, and the split groups' columns of H
    through MXFP4 qdq (a thread a 1 x 32 block), -> LDS rows of hqrow / hsrow."""
    s, tid, geo = c["S"], c["tid"], c["G"]
    if tid < geo.grid:
        c["poll"]([(c["hrdy"], tid, 1)])
    gpu.barrier()
    nb = s * geo.bulk_blocks
    for i in range_constexpr((nb + THREADS - 1) // THREADS):
        b = fx.min(tid + THREADS * i, nb - 1)
        t = b // geo.bulk_blocks
        blk = b % geo.bulk_blocks
        w = ld4(rsrc(c["hq4"]), (t * geo.blocks + blk) * 4, CM_DEV)
        code = fx.Int32(
            bo.buffer_load(
                rsrc(c["hq4s"]),
                t * geo.blocks + blk,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        if tid + THREADS * i < nb:
            fx.ptr_store(w, c["hq"] + (t * geo.hqrow + 4 * blk))
            fx.ptr_store(code, c["hs"] + (t * geo.hsrow + blk))
    sl_blocks = geo.blocks - geo.bulk_blocks
    ns = s * sl_blocks
    r_h = rsrc(c["hbuf"])
    for i in range_constexpr((ns + THREADS - 1) // THREADS):
        b = fx.min(tid + THREADS * i, ns - 1)
        t = b // sl_blocks
        blk = geo.bulk_blocks + b % sl_blocks
        at = (t * geo.inter + 32 * blk) // 2
        ws = [ld4(r_h, at + 4 * k, CM_DEV) for k in range(4)]
        vs = []
        for k in range_constexpr(4):
            for d in range_constexpr(4):
                vs += [bf_lo(ws[k][d]), bf_hi(ws[k][d])]
        amax = fx.Float32(0.0)
        for e in range_constexpr(32):
            amax = fx.max(amax, fx.Float32(fmath.absf(vs[e])))
        code = mx_code(amax)
        qs = [pack8(vs[8 * k : 8 * k + 8], code) for k in range(4)]
        if tid + THREADS * i < ns:
            fx.ptr_store(
                fx.Vector.from_elements(qs, fx.Int32),
                c["hq"] + (t * geo.hqrow + 4 * blk),
            )
            fx.ptr_store(code, c["hs"] + (t * geo.hsrow + blk))
    gpu.barrier()


@traced
def stage_down(c, ops, p=0, nxt=None):
    """Pass ``p``: the K chains (``chained``) -> DRED, then ``nxt()`` (the next
    pass's loads, returned); thread (tile, token, row pair): the splits added as
    stock's reduce -> bf16 ``out`` (of the split row group, the CTA's rows
    only)."""
    s, lane, wave, geo = c["S"], c["lane"], c["wave"], c["G"]
    dred = c["dred"]
    tiles, words = ops
    gl = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    kq = wave // 2
    n = c["d_n"]
    per = QUARTERS // n

    def run(acc):
        for i in range_constexpr(geo.d_q):
            acc = scaled_step(
                c["hq"],
                c["hs"],
                t * geo.hqrow,
                t * geo.hsrow,
                kq * geo.d_q + i,
                gl,
                tiles[i],
                words[i],
                acc,
            )
        return acc

    chained(c, dred, kq, per, run)
    ops_n = None
    if const_expr(nxt is not None):
        ops_n = nxt()
    tiles_p = geo.d_pass_tiles(p)
    if const_expr(tiles_p == (0, geo.d_per)):
        down_out_split(c, n, per)
    else:
        down_out(c, n, per, tiles_p)
    return ops_n


@traced
def down_out_split(c, n, per):
    """The pass of whole row group b (even waves) and the split group's rows
    (odd): thread (tile, token, row pair) -> ``out`` (256 CTAs)."""
    s, tid, geo, bid, dred = c["S"], c["tid"], c["G"], c["bid"], c["dred"]
    if tid < 2 * s * (ROWS // 2):
        tile = tid // (s * (ROWS // 2))
        tt = tid % (s * (ROWS // 2)) // (ROWS // 2)
        rp = tid % (ROWS // 2)
        y = [
            split_total(
                red_at(dred, 2 * q + tile, 2 * rp + d, tt) for q in split_ends(n, per)
            )
            for d in range(2)
        ]
        off = d_off(c)
        whole = tile == 0
        row = (
            whole.select(
                bid * ROWS, (geo.d_full + bid // (ROWS // geo.d_sl_rows)) * ROWS
            )
            + 2 * rp
        )
        if whole | ((2 * rp >= off) & (2 * rp < off + geo.d_sl_rows)):
            bo.buffer_store(
                pack_bf(y)[0], rsrc(c["args"]["out"]), (tt * HIDDEN + row) // 2
            )


@traced
def down_out(c, n, per, tiles):
    """Any other pass (``tiles``: its even and odd waves' tiles, 128 CTAs):
    thread (tile, token, row pair) -> ``out``, rows the CTA owns only."""
    s, tid, dred = c["S"], c["tid"], c["dred"]
    if tid < 2 * s * (ROWS // 2):
        tile = tid // (s * (ROWS // 2))
        tt = tid % (s * (ROWS // 2)) // (ROWS // 2)
        rp = tid % (ROWS // 2)
        y = [
            split_total(
                red_at(dred, 2 * q + tile, 2 * rp + d, tt) for q in split_ends(n, per)
            )
            for d in range(2)
        ]
        row = (tile == 0).select(
            d_out_base(c, tiles[0]), d_out_base(c, tiles[1])
        ) + 2 * rp
        if d_out_live(c, tile, rp, tiles):
            bo.buffer_store(
                pack_bf(y)[0], rsrc(c["args"]["out"]), (tt * HIDDEN + row) // 2
            )


def run_down(c, ops):
    """Down's passes (``FGeo.d_passes``, ``ops`` pass 0's loads): pass p + 1's
    weights in flight under pass p's reduce, and a barrier between passes (the
    next chains overwrite DRED). One pass on 256 CTAs."""
    n = c["G"].d_passes
    for p in range(n):
        if p:
            gpu.barrier()
        nxt = (lambda p=p: down_loads(c, p + 1)) if p + 1 < n else None
        ops = stage_down(c, ops, p, nxt)


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: FfnBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    assert key.tp in TP_SIZES and 1 <= s <= max_tokens(key.tp)
    # TP2 runs from the stock norm's output only
    assert key.normed == (key.tp > 1) and not (key.normed and key.oproj)
    # every CTA resident, or the launch hangs
    check_resident(key.grid)
    gu_n, d_n, o_n = key_splits(key)
    assert {gu_n, d_n, o_n} <= set(SPLIT_COUNTS), key
    geo = FGeo(key.tp, key.grid)
    nblk = key.grid
    lay = scratch_layout(key)

    @fx.struct
    class XLds:
        xq: fx.Array[fx.Int32, s * XQROW, 16]
        xs: fx.Array[fx.Int32, s * XSROW, 16]

    @fx.struct
    class DLds:
        hq: fx.Array[fx.Int32, s * geo.hqrow, 16]
        hs: fx.Array[fx.Int32, s * geo.hsrow, 16]
        dred: fx.Array[fx.Float32, WAVES * 64 * 4, 16]

    @fx.struct
    class OLds:
        oq: fx.Array[fx.Int32, s * OQROW, 16]
        os: fx.Array[fx.Int32, s * OSROW, 16]

    @fx.union
    class StageLds:
        x: XLds
        d: DLds
        o: OLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        nrm: fx.Array[fx.Float32, MAX_ROUNDS * WAVES, 16]
        rst: fx.Array[fx.Float32, max_tokens(key.tp), 16]
        hp: fx.Array[fx.Float32, s * 2 * ROWS, 16]

    keyed = key_tuple(key, digest())
    name = (
        f"qwen38_dense_ffn_s{s}_p{key.stop}_t{int(key.stamps)}_w{int(key.shuffled)}"
        f"_o{int(key.oproj)}_tp{key.tp}_n{int(key.normed)}_v6"
    )
    if key.grid != DEFAULT_GRID:
        name += f"_g{key.grid}"
    if key.gu_split or key.d_split or key.o_split:
        name += f"_k{gu_n}{d_n}{o_n}"
    local = local_norm(key)
    if local:
        name += "_l"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kd(
        hidden: Int64,
        residual: Int64,
        w_o: Int64,
        w_os: Int64,
        ln_w: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        res_out: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        st_lds = alloc.allocate(StageLds)
        xl, dl, ol = st_lds.x.peek(), st_lds.d.peek(), st_lds.o.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1)
        c = {
            "S": s,
            "G": geo,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "eps": key.eps,
            "shuffled": key.shuffled,
            "gu_n": gu_n,
            "d_n": d_n,
            "o_n": o_n,
            "stamps": scratch + fx.Int64(lay["stamps"][0]) if key.stamps else None,
            "put": mb.put,
            "put_words": mb.put_words,
            "put_bf": mb.put_bf,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "nrm": lds.nrm.ptr,
            "rst": lds.rst.ptr,
            "hp": lds.hp.ptr,
            "xq": xl.xq.ptr,
            "xs": xl.xs.ptr,
            "hq": dl.hq.ptr,
            "hs": dl.hs.ptr,
            "dred": dl.dred.ptr,
            "oq": ol.oq.ptr,
            "os": ol.os.ptr,
            "nsum": sreg(scratch, lay["nsum"][0], "nsum"),
            "xmb": sreg(scratch, lay["xmb"][0], "xmb"),
            "omb": sreg(scratch, lay["omb"][0], "omb"),
            "hbuf": scratch + fx.Int64(lay["h"][0]),
            "hq4": scratch + fx.Int64(lay["hq"][0]),
            "hq4s": scratch + fx.Int64(lay["hqs"][0]),
            "hrdy": sreg(scratch, lay["hrdy"][0], "hrdy"),
            "args": {
                "hidden": hidden,
                "residual": residual,
                "w_o": w_o,
                "w_os": w_os,
                "ln_w": ln_w,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "out": out,
                "res_out": res_out,
            },
        }
        stamp(c, 0)
        if const_expr(local):
            # the core first, then the o weights: vmcnt drains in order
            craw = core_loads(c)
            oops = o_loads(c, splits=o_n)
            o_quant(c, craw)
            o_stage(c, oops, splits=o_n)
            # owners past the CTA's first (128 CTAs): loads under its reduce
            more = o_more_loads(c, residual=False)
            o_publish(c)
            o_more_publish(c, more, splits=o_n)
            stamp(c, 5)
            # the first gate_up weights before the norm's one hand-off
            gops = gu_loads(c, geo.g_full * bid)
            run_norm(c, o_norm_loads(c), marks=False)
        elif const_expr(key.oproj):
            # the core first, then the residual and o weights: vmcnt drains in order
            craw = core_loads(c)
            rraw = o_res_loads(c)
            oops = o_loads(c, splits=o_n)
            o_quant(c, craw)
            o_stage(c, oops, splits=o_n)
            # owners past the CTA's first (128 CTAs): loads under its reduce
            more = o_more_loads(c)
            z, lnw = o_finish(c, rraw)
            zs = o_more_passes(c, more, splits=o_n)
            stamp(c, 5)
            # the first gate_up weights before the norm's hand-offs
            gops = gu_loads(c, geo.g_full * bid)
            o_rstd(c)
            o_x(c, z, lnw)
            o_more_x(c, zs)
            o_gather(c)
        elif const_expr(key.normed):
            # x first: vmcnt drains in issue order
            xraw = block_loads(c, HIDDEN)
            gops = gu_loads(c, geo.g_full * bid)
            x_quant(c, xraw)
        else:
            # the norm's loads first: vmcnt drains in issue order
            nraw = norm_loads(c)
            run_norm(c, nraw)
            gops = gu_loads(c, geo.g_full * bid)
        stamp(c, 1)
        if const_expr(key.stop > 2):
            run_gate_up(c, gops)
            stamp(c, 2)
        if const_expr(key.stop > 3):
            dops = down_loads(c)
            load_h(c)
            stamp(c, 3)
            run_down(c, dops)
            stamp(c, 4)

    @flyc.jit
    def launch(
        hidden: Int64,
        residual: Int64,
        w_o: Int64,
        w_os: Int64,
        ln_w: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        res_out: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kd(
            hidden, residual, w_o, w_os, ln_w, w13, w13s, w2, w2s, out, res_out,
            scratch, epoch, layer,
        ).launch(grid=(nblk,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(kd, launch)
    _BUILDS[key] = launch
    return launch


def shuffle_w(w):
    """fp4x2 [N, K / 2] -> the (16, 16) shuffled order ``FfnBuild.shuffled`` reads."""
    n, kb = w.shape
    return (
        w.view(n // ROWS, ROWS, kb // 16, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(n, kb)
    )


def dense_ffn(
    key: FfnBuild,
    *,
    hidden,
    residual,
    ln_w,
    w13,
    w13s,
    w2,
    w2s,
    out,
    res_out,
    scratch,
    epoch,
    layer,
    w_o=None,
    w_os=None,
):
    """Host launcher. ``hidden`` / ``residual`` / ``out`` / ``res_out``: bf16
    [S, HIDDEN] (``out`` / ``res_out`` may be views of larger buffers' first S
    rows); ``w13`` = [gate; up] rows of the rank's share (``FGeo``), ``w2`` its
    K columns. ``key.oproj``: ``hidden`` is o_proj's input, bf16 [S, CORE], and
    ``w_o`` / ``w_os`` its shuffled weights. ``key.normed``: ``hidden`` is x;
    ``residual`` / ``ln_w`` / ``res_out`` None."""
    s = key.tokens
    inter = FGeo(key.tp).inter
    if key.oproj:
        assert key.shuffled and hidden.shape == (s, CORE)
        assert w_o.shape == (HIDDEN, CORE // 2) and w_os.shape == (HIDDEN, CORE // 32)
    else:
        assert hidden.shape == (s, HIDDEN)
        w_o = w_os = hidden
    if key.normed:
        assert residual is None and res_out is None and ln_w is None
        residual = res_out = ln_w = hidden
    assert residual.shape == (s, HIDDEN)
    assert w13.shape == (2 * inter, HIDDEN // 2) and w13s.shape == (
        2 * inter,
        HIDDEN // 32,
    )
    assert w2.shape == (HIDDEN, inter // 2) and w2s.shape == (HIDDEN, inter // 32)
    assert scratch.numel() >= scratch_bytes(key)
    for t in (hidden, residual, out, res_out, w13, w13s, w2, w2s):
        assert t.is_contiguous()
    f = build(key)
    f(
        *ABI.pack(
            {
                "hidden": hidden.data_ptr(),
                "residual": residual.data_ptr(),
                "w_o": w_o.data_ptr(),
                "w_os": w_os.data_ptr(),
                "ln_w": ln_w.data_ptr(),
                "w13": w13.data_ptr(),
                "w13s": w13s.data_ptr(),
                "w2": w2.data_ptr(),
                "w2s": w2s.data_ptr(),
                "out": out.data_ptr(),
                "res_out": res_out.data_ptr(),
                "scratch": scratch.data_ptr(),
                "epoch": epoch.data_ptr(),
                "layer": layer,
            }
        ),
        stream=torch.cuda.current_stream(),
    )
