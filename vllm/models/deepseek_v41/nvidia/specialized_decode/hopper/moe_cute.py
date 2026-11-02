# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-batch MoE of DeepSeek-V4.1 at TP4 on SM90, in CuTe DSL: one persistent
launch per MoE layer (``LowBatchMoEKernel``), reading the layer's stock Humming
tensors in place (``moe`` builds the tensor maps and runs it).

Roles per CTA (320 threads, 2 CTAs per SM): 8 consumer warps (two wgmma
warpgroups), a TMA producer warp, a signal warp.

* Front: row CTA ``t < M`` routes token ``t`` (from the router's logits:
  sqrt-softplus plus the correction bias, top-6, renormalized, times the routed
  scaling factor; image sentinel tokens with ``bias_vl``; or from given top-k
  ids and weights), block-FP8 (group 128) quantizes it and publishes it with
  one counter. Every CTA builds the expert-major row-chunk table in shared
  memory.

  - Dense mode (``M <= DENSE_MAX_ROWS``): one row chunk per active expert
    covering all ``M`` tokens (a token mask marks the routed ones), on the
    token-major quantized rows: no gather.
  - Entry mode: 8-aligned row chunks of at most 32 routed entries; row CTAs
    scatter their row into its 6 entry slots.

* GEMM1 (``w13``, MXFP4 x FP8 wgmma): dynamic split-K chunks of (row chunk,
  gate/up pair) tiles. Each chunk's FP32 partial goes to global memory; the
  signal warp detects the CTA that completes a tile, whose consumers run the
  tile's clamped SwiGLU and FP8 intermediate, and publishes a row chunk ready
  after its 9 pairs.
* GEMM2 (``w2``): the producer streams ``w2`` tiles into a second ring once its
  GEMM1 ring is drained; each item waits only for its own row chunk.
* Shared expert (``shared``): the W8A16 FP8 weights read in place, as f16
  wgmma A fragments (exact e4m3 to f16 conversion) with the per-32 UE8M0
  scale applied on each k32 partial, on f16 activations converted from x; an
  extra row chunk of weight 1 covering every token, written to its own output
  in token order. Its first weight stages are issued before the grid wait.
* Output: unweighted BF16 rows per (token, expert) and the
  route-to-row map in the record: an ``UnfinalizedMoEOutput``, which the next
  boundary's fused all-reduce + mHC finalizes.

Counters are epoch-stamped or reset by their last user, so repeated calls and
CUDA graph replays need no host-side reset. The kernel launches with
programmatic dependent launch.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from .moe_ptx import (
    atom_add_acqrel_gpu,
    atom_add_relaxed,
    bf16x2_to_f16x2,
    bf16x2_to_f32x2,
    bulk_g2s,
    cluster_sync_all,
    cvt_bf16x2,
    cvt_e4m3_word,
    cvt_e4m3x4,
    dequant_e2m1x8,
    e8m0_to_f32,
    evict_first_policy,
    f16x2,
    fabs,
    fence_gpu,
    fence_mbar_init,
    fence_mbar_init_cluster,
    fence_proxy_async_global,
    fence_proxy_async_smem,
    fence_regs,
    fmax,
    griddep_launch,
    griddep_wait,
    ld_acquire_i32,
    ldg128,
    ldg_cg_f32,
    ldg_cg_f32x2,
    ldg_cg_i32,
    ldg_f32,
    ldg_i64,
    lds32,
    lds64,
    lds128,
    lds_acquire,
    lds_f32,
    lds_u16,
    make_desc_nosw,
    mapa,
    mbar_arrive,
    mbar_arrive_cluster,
    mbar_arrive_expect_tx,
    mbar_init,
    mbar_wait,
    mbar_wait_cluster,
    named_barrier,
    nanosleep,
    popc,
    prefetch_l2_bulk,
    prefetch_tmap,
    rcp,
    red_add_gpu,
    round_bf16,
    score_of,
    shfl_idx_f32,
    shfl_up_i32,
    shfl_xor_f32,
    shfl_xor_i32,
    smem_add_release,
    smem_addr,
    smem_atom_add,
    smem_atom_or,
    st_cluster_f32,
    stg_f32,
    stg_i32,
    stg_u16,
    stg_u32,
    stg_u32x4,
    sts32,
    sts128,
    sts_f32,
    sts_release,
    swiglu,
    tma_load_2d,
    tma_load_3d,
    tma_load_4d,
    wgmma_commit,
    wgmma_fence,
    wgmma_rs,
    wgmma_rs_f16,
    wgmma_wait_n,
)

HIDDEN = 5120
INTER = 576
NPAIR = INTER // 64  # 9 gate/up 64-row pairs = GEMM2 k64 tiles
KT1 = HIDDEN // 64  # 80 k64 tiles in GEMM1
NBLK2 = HIDDEN // 64  # 80 64-row output blocks in GEMM2
NBP2 = NBLK2 // 2  # 40 output block pairs (GEMM2 work unit width 128)
MAX_EXPERTS = 384  # table capacity
MAX_TOPK = 6  # table capacity
SWIGLU_LIMIT = 10.0
ROUTED_SCALING = 1.5
MAXM = 256
ENT_CAP = MAX_TOPK * MAXM
RC_ROWS = 32  # entries per row chunk (4 n8 groups)
RC_CAP = 512  # >= MAX_EXPERTS + ENT_CAP // RC_ROWS
ENTP = 4608  # padded entry capacity: row chunks start 8-aligned
REC_RC_EXP = 16
REC_RC_OFF = REC_RC_EXP + RC_CAP
REC_RC_CNT = REC_RC_OFF + RC_CAP
REC_ENT_TOK = REC_RC_CNT + RC_CAP
REC_ENT_W = REC_ENT_TOK + ENTP
REC_POS = REC_ENT_W + ENTP
REC_ENT_RC = REC_POS + ENT_CAP  # row chunk of each padded entry (-1 = pad)
REC_WORDS = REC_ENT_RC + ENTP
MASKW = MAXM // 32  # bitmap words per expert
NG = ENTP // 8
K1S = 4  # k64 tiles per stage
G1_W = 0
G1_WS = 16384
G1_B = G1_WS + 1024
G1_AS = G1_B + 4 * 2048
K2S = 3
G2_W = 0
G2_WS = 12288
G2_B = G2_WS + 768
G2_AS = G2_B + 4 * 1536
SB2 = 19712  # GEMM2 stage + 128 B header
HDR2 = SB2 - 128
CHUNK_CAP = 8192  # partial slots
SPT = KT1 // K1S  # 20 k256 units per GEMM1 tile
SB1 = 26112  # GEMM1 stage + header, 256 B aligned (32 B swizzle pattern)
HDR1 = SB1 - 128
NCW = 8
PROD = 8
SIG = 9
NTHR = 32 * (NCW + 2)
NCT = 32 * NCW
END = 0x7FFFFFFF
CL = 256  # chunk / finalize list capacity per CTA
FIN_FLAG = 0x40000000
E_SH = 384  # the shared expert's id in the row-chunk table
SH_U1 = 40  # k128 stages per shared GEMM1 tile
SH_V2 = 6  # k96 stages per shared GEMM2 item
NGS = MAXM // 8  # 8-token groups of the shared expert's buffers
C_BAR = 0  # rows published
C_EXIT = 1
C_WORK1 = 2
C_WORK2 = 3
C_GEN = 5
N_RESET = 5
S_TILE = 0
S_READY = RC_CAP * NPAIR
S_ALL = S_READY + RC_CAP  # finalized tiles of the call (reset by the last CTA out)
SYNC_WORDS = S_ALL + 4
FRONT_SCRATCH = MAXM * MAX_TOPK * 4 + MAX_EXPERTS * MASKW * 4 + MAX_EXPERTS * 16 + 64
TAB_BYTES = 4 * RC_CAP * 4
SIG_BYTES = 80 + 2 * CL * 4 + 80
NCHS_ENTRY = 5


def _i64(x):
    return Int64(x)


@cute.jit
def _wgmma_k64(
    N: cutlass.Constexpr,
    bt: Int32,
    sbo: cutlass.Constexpr,
    af: cute.Tensor,
    d: cute.Tensor,
    base: cutlass.Constexpr,
    first: Int32,
):
    """Two k32 wgmmas; B k64 tile at ``bt`` (k-chunks 128 B apart, 8-entry
    groups ``sbo`` apart). Accumulators d[base : base + N/2]."""
    vals = [d[base + i] for i in range(N // 2)]
    wgmma_fence()
    vals = wgmma_rs(
        N, vals, af[0], af[1], af[2], af[3], make_desc_nosw(bt, 128, sbo), first
    )
    vals = wgmma_rs(
        N,
        vals,
        af[4],
        af[5],
        af[6],
        af[7],
        make_desc_nosw(bt + 256, 128, sbo),
        Int32(1),
    )
    for i in cutlass.range_constexpr(N // 2):
        d[base + i] = vals[i]


@cute.jit
def _pin(N: cutlass.Constexpr, d: cute.Tensor, base: cutlass.Constexpr):
    vals = fence_regs([d[base + i] for i in range(N // 2)])
    for i in cutlass.range_constexpr(N // 2):
        d[base + i] = vals[i]


@cute.jit
def _promote(
    N: cutlass.Constexpr,
    d: cute.Tensor,
    base: cutlass.Constexpr,
    acc: cute.Tensor,
    asb: Int32,
    stride: cutlass.Constexpr,
    t: Int32,
):
    """Acc += d * scale[col]; scale of entry column c at asb + c * stride."""
    for i in cutlass.range_constexpr(N // 8):
        sa = lds_f32(asb + (8 * i + 2 * t) * stride)
        sb = lds_f32(asb + (8 * i + 2 * t + 1) * stride)
        acc[4 * i] = acc[4 * i] + d[base + 4 * i] * sa
        acc[4 * i + 1] = acc[4 * i + 1] + d[base + 4 * i + 1] * sb
        acc[4 * i + 2] = acc[4 * i + 2] + d[base + 4 * i + 2] * sa
        acc[4 * i + 3] = acc[4 * i + 3] + d[base + 4 * i + 3] * sb


@cute.jit
def _route_row(
    logits: Int64,
    bias: Int64,
    ids: Int64,
    wts: Int64,
    r: Int32,
    lane: Int32,
    experts: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
):
    """DSv4 top-k of row r (one warp): score = sqrt(softplus), key = score + bias,
    lowest id on ties; weights = scores / sum x routed scaling."""
    per_lane = experts // 32
    sc = cute.make_rmem_tensor((per_lane,), Float32)
    ky = cute.make_rmem_tensor((per_lane,), Float32)
    bb = cute.make_rmem_tensor((per_lane,), Float32)
    for j in cutlass.range_constexpr(per_lane):
        bj0 = ldg_f32(bias + Int64((lane + 32 * j) * 4))
        lg = ldg_f32(logits + (Int64(r) * experts + lane + 32 * j) * 4)
        sc[j] = score_of(lg)
        ky[j] = sc[j] + bj0
        bb[j] = bj0
    tot = Float32(0.0)
    my_id = Int32(0)
    my_s = Float32(0.0)
    for k in cutlass.range_constexpr(top_k):
        bk = ky[0]
        bj = Int32(0)
        for j in cutlass.range_constexpr(1, per_lane):
            if ky[j] > bk:
                bk = ky[j]
                bj = Int32(j)
        bidx = lane + 32 * bj
        for sh in cutlass.range_constexpr(5):
            ok = shfl_xor_f32(bk, 1 << sh)
            oi = shfl_xor_i32(bidx, 1 << sh)
            if ok > bk:
                bk = ok
                bidx = oi
            elif ok == bk:
                if oi < bidx:
                    bidx = oi
        owner = bidx % 32
        s_sel = Float32(0.0)
        if lane == owner:
            for j in cutlass.range_constexpr(per_lane):
                if j == bidx // 32:
                    s_sel = sc[j]
                    ky[j] = Float32(-3.0e38)
        s_sel = shfl_idx_f32(s_sel, owner)
        tot = tot + s_sel
        if lane == k:
            my_id = bidx
            my_s = s_sel
    if lane < top_k:
        wv = my_s * (Float32(ROUTED_SCALING) * rcp(tot))
        stg_i32(ids + (Int64(r) * top_k + lane) * 4, my_id)
        stg_f32(wts + (Int64(r) * top_k + lane) * 4, wv)


def red_bytes(ns2):
    """cluster: rank-1 partial of one tile, 128 rows x 8 (ns2 = 5) or 16 (ns2 <= 4) columns"""  # noqa: E501
    return 128 * (8 if ns2 >= 5 else 16) * 4


def smem_bytes(ns1, ns2, cluster=False):
    u = max(
        ns1 * SB1 + FRONT_SCRATCH,
        max(ns1 * SB1, ns2 * SB2) + (red_bytes(ns2) if cluster else 0),
    )
    return u + TAB_BYTES + 16 * (ns1 + ns2) + SIG_BYTES


@cute.jit
def _min_count(base: Int32) -> Int32:
    m = lds_acquire(base)
    for w in cutlass.range_constexpr(1, NCW):
        v = lds_acquire(base + w * 4)
        if v < m:
            m = v
    return m


@cute.jit
def _ch_tile(ch: Int32, NCH: Int32, tshc: Int32):
    """(tile, k-chunk index) of chunk ch: the shared tiles' NCHS chunks first,
    then the routed tiles' NCH chunks."""
    tsh = tshc // 32
    nchs = tshc % 32
    cs = tsh * nchs
    tile = ch // nchs
    h = ch % nchs
    if ch >= cs:
        tile = tsh + (ch - cs) // NCH
        h = (ch - cs) % NCH
    return tile, h


@cute.jit
def _tile_nch(tile: Int32, NCH: Int32, tshc: Int32) -> Int32:
    n = NCH
    if tile < tshc // 32:
        n = tshc % 32
    return n


@cute.jit
def _tile_ch(tile: Int32, h: Int32, NCH: Int32, tshc: Int32) -> Int32:
    tsh = tshc // 32
    nchs = tshc % 32
    ch = tile * nchs + h
    if tile >= tsh:
        ch = tsh * nchs + (tile - tsh) * NCH + h
    return ch


@cute.jit
def _route_row2(
    plog: Int64,
    bias: Int64,
    bias_vl: Int64,
    in_ids: Int64,
    lo: Int32,
    ids: Int64,
    wts: Int64,
    r: Int32,
    lane: Int32,
    experts: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    image: cutlass.Constexpr,
    halves: cutlass.Constexpr = True,
):
    """_route_row on the two published partial-logit halves; image sentinel
    tokens (lo <= id < lo + 5) use ``bias_vl``."""
    use_vl = Int32(0)
    if cutlass.const_expr(image):
        tok = ldg_cg_i32(in_ids + Int64(r) * 4)
        if tok >= lo:  # noqa: SIM102
            if tok < lo + 5:
                use_vl = Int32(1)
    per_lane = experts // 32
    sc = cute.make_rmem_tensor((per_lane,), Float32)
    ky = cute.make_rmem_tensor((per_lane,), Float32)
    bb = cute.make_rmem_tensor((per_lane,), Float32)
    for j in cutlass.range_constexpr(per_lane):
        e = lane + 32 * j
        bj0 = ldg_f32(bias + Int64(e) * 4)
        if cutlass.const_expr(image):  # noqa: SIM102
            if use_vl == 1:
                bj0 = ldg_f32(bias_vl + Int64(e) * 4)
        lg = ldg_cg_f32(plog + (Int64(r) * experts + e) * 4)
        if cutlass.const_expr(halves):
            lg = lg + ldg_cg_f32(plog + (Int64(MAXM + r) * experts + e) * 4)
        sc[j] = score_of(lg)
        ky[j] = sc[j] + bj0
        bb[j] = bj0
    tot = Float32(0.0)
    my_id = Int32(0)
    my_s = Float32(0.0)
    for k in cutlass.range_constexpr(top_k):
        bk = ky[0]
        bj = Int32(0)
        for j in cutlass.range_constexpr(1, per_lane):
            if ky[j] > bk:
                bk = ky[j]
                bj = Int32(j)
        bidx = lane + 32 * bj
        for sh in cutlass.range_constexpr(5):
            ok = shfl_xor_f32(bk, 1 << sh)
            oi = shfl_xor_i32(bidx, 1 << sh)
            if ok > bk:
                bk = ok
                bidx = oi
            elif ok == bk:
                if oi < bidx:
                    bidx = oi
        owner = bidx % 32
        s_sel = Float32(0.0)
        if lane == owner:
            for j in cutlass.range_constexpr(per_lane):
                if j == bidx // 32:
                    s_sel = sc[j]
                    ky[j] = Float32(-3.0e38)
        s_sel = shfl_idx_f32(s_sel, owner)
        tot = tot + s_sel
        if lane == k:
            my_id = bidx
            my_s = s_sel
    if lane < top_k:
        wv = my_s * (Float32(ROUTED_SCALING) * rcp(tot))
        stg_i32(ids + (Int64(r) * top_k + lane) * 4, my_id)
        stg_f32(wts + (Int64(r) * top_k + lane) * 4, wv)


@cute.jit
def _dq_load(
    wt: Int32, wsa0: Int32, wsa1: Int32, lane: Int32, R: Int32, q: cute.Tensor
):
    """Shared-memory half of _dequant: weights word quad + both k32 scales."""
    v = lds128(wt + R * 512 + lane * 16)
    q[0] = v[0]
    q[1] = v[1]
    q[2] = v[2]
    q[3] = v[3]
    q[4] = lds_u16(wsa0)
    q[5] = lds_u16(wsa1)


@cute.jit
def _dq_conv(q: cute.Tensor, af: cute.Tensor):
    lo0 = dequant_e2m1x8(q[0], q[4] & Uint32(0xFF))
    hi0 = dequant_e2m1x8(q[1], q[4] >> Uint32(8))
    lo1 = dequant_e2m1x8(q[2], q[5] & Uint32(0xFF))
    hi1 = dequant_e2m1x8(q[3], q[5] >> Uint32(8))
    af[0] = lo0[0]
    af[1] = hi0[0]
    af[2] = lo0[1]
    af[3] = hi0[1]
    af[4] = lo1[0]
    af[5] = hi1[0]
    af[6] = lo1[1]
    af[7] = hi1[1]


@cute.jit
def _wgmma_k64_sw32(
    N: cutlass.Constexpr,
    bt: Int32,
    af: cute.Tensor,
    d: cute.Tensor,
    base: cutlass.Constexpr,
    first: Int32,
):
    """_wgmma_k64 on the 32 B-swizzled K-major B (k32 slices 256 B apart,
    8-entry groups 2048 B apart)."""
    vals = [d[base + i] for i in range(N // 2)]
    sw = Int64(3) << Int64(62)
    wgmma_fence()
    vals = wgmma_rs(
        N, vals, af[0], af[1], af[2], af[3], make_desc_nosw(bt, 16, 2048) | sw, first
    )
    vals = wgmma_rs(
        N,
        vals,
        af[4],
        af[5],
        af[6],
        af[7],
        make_desc_nosw(bt + 256, 16, 2048) | sw,
        Int32(1),
    )
    for i in cutlass.range_constexpr(N // 2):
        d[base + i] = vals[i]


class LowBatchMoEKernel:
    def __init__(
        self,
        ns1: int = 3,
        ns2: int = 5,
        chunk_target: int = 3,
        ids_mode: bool = False,
        dense: bool = True,
        shared: bool = False,
        image: bool = False,
        cluster: bool = False,
        nchs: int = 20,
        experts: int = MAX_EXPERTS,
        top_k: int = MAX_TOPK,
    ):
        assert experts % 32 == 0 and 0 < experts <= MAX_EXPERTS
        assert 0 < top_k <= MAX_TOPK
        self.experts = experts
        self.top_k = top_k
        self.per_lane = experts // 32
        self.cluster = cluster and dense
        self.nchs_d = nchs
        assert not self.cluster or nchs % 2 == 0  # chunk parity == cluster rank
        self.shared = shared
        self.image = image and not ids_mode
        self.ns1 = ns1
        self.ns2 = ns2
        self.chunk_target = chunk_target
        self.ids_mode = ids_mode
        self.dense = dense
        self.smem = smem_bytes(ns1, ns2, self.cluster)

    @cute.jit
    def __call__(
        self,
        maps: Int64,
        g13: Int64,
        g2: Int64,
        xg: Int64,
        xsg: Int64,
        rec: Int64,
        iq: Int64,
        isc: Int64,
        part: Int64,
        cnt: Int64,
        sync: Int64,
        x: Int64,
        ids: Int64,
        wts: Int64,
        logits: Int64,
        bias: Int64,
        y: Int64,
        out: Int64,
        xgs: Int64,
        xsgs: Int64,
        iqs: Int64,
        iscs: Int64,
        sh: Int64,
        M: Int32,
        grid: Int32,
        stream,
    ):
        self.kernel(
            maps,
            g13,
            g2,
            xg,
            xsg,
            rec,
            iq,
            isc,
            part,
            cnt,
            sync,
            x,
            ids,
            wts,
            logits,
            bias,
            y,
            out,
            xgs,
            xsgs,
            iqs,
            iscs,
            sh,
            M,
        ).launch(
            grid=[grid, 1, 1],
            block=[NTHR, 1, 1],
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=2,
            preferred_smem_carveout=100,
            cluster=[2, 1, 1] if self.cluster else None,
        )

    # ================================================================ front
    @cute.jit
    def _quant_row(
        self, x: Int64, r: Int32, tid: Int32, qw: cute.Tensor, qs: cute.Tensor
    ):
        """Block-FP8 g128 of row r: thread tid < 160 holds k [32 tid, 32 tid + 32)."""
        k0 = tid * 32
        f = cute.make_rmem_tensor((32,), Float32)
        for c in cutlass.range_constexpr(4):
            v = ldg128(x + (Int64(r) * HIDDEN + k0 + 8 * c) * 2)
            for i in cutlass.range_constexpr(4):
                lo, hi = bf16x2_to_f32x2(v[i])
                f[8 * c + 2 * i] = lo
                f[8 * c + 2 * i + 1] = hi
        amax = fabs(f[0])
        for i in cutlass.range_constexpr(1, 32):
            amax = fmax(amax, fabs(f[i]))
        for sh in cutlass.range_constexpr(2):
            amax = fmax(amax, shfl_xor_f32(amax, 1 << sh))
        amax = fmax(amax, Float32(1e-12))
        qs[0] = amax * Float32(1.0 / 448.0)
        inv = Float32(448.0) / amax
        for w in cutlass.range_constexpr(8):
            qw[w] = cvt_e4m3x4(
                f[4 * w] * inv,
                f[4 * w + 1] * inv,
                f[4 * w + 2] * inv,
                f[4 * w + 3] * inv,
            )

    @cute.jit
    def _store_xrow(
        self,
        xg: Int64,
        xsg: Int64,
        ent: Int32,
        tid: Int32,
        qw: cute.Tensor,
        qs: cute.Tensor,
    ):
        """Quantized row slice of thread tid into GEMM1 B slot ``ent``."""
        k0 = tid * 32
        kt = k0 // 64
        kt4 = kt // 4  # noqa: F841
        kti = kt % 4  # noqa: F841
        kc = (k0 % 64) // 16  # noqa: F841
        b128 = k0 // 128
        # [stage (k256)][entry/8][k32 slice][8 entries][32 B]; halves swapped
        # for entries 4-7 (32 B swizzle). Lane pair (2j, 2j + 1) = slices
        # (s, s + 1) of one stage: each instruction writes one whole slice row.
        n = ent % 8
        sw = (n // 4) % 2
        se = (tid % 8) - (tid % 2)
        row = xg + ((Int64(tid // 8) * NG + ent // 8) * 8 + se) * 256 + n * 32
        odd = tid % 2
        s0 = qw[0]
        s1 = qw[1]
        s2 = qw[2]
        s3 = qw[3]
        if (odd ^ sw) == 0:
            s0 = qw[4]
            s1 = qw[5]
            s2 = qw[6]
            s3 = qw[7]
        r0 = Uint32(shfl_xor_i32(Int32(s0), 1))
        r1 = Uint32(shfl_xor_i32(Int32(s1), 1))
        r2 = Uint32(shfl_xor_i32(Int32(s2), 1))
        r3 = Uint32(shfl_xor_i32(Int32(s3), 1))
        o0 = qw[0]
        o1 = qw[1]
        o2 = qw[2]
        o3 = qw[3]
        if (odd ^ sw) == 1:
            o0 = qw[4]
            o1 = qw[5]
            o2 = qw[6]
            o3 = qw[7]
        if odd == 0:
            stg_u32x4(row, o0, o1, o2, o3)
            stg_u32x4(row + 256, r0, r1, r2, r3)
        else:
            stg_u32x4(row + 16, r0, r1, r2, r3)
            stg_u32x4(row + 256 + 16, o0, o1, o2, o3)
        if tid % 4 == 0:
            stg_f32(xsg + ((Int64(b128 // 2) * ENTP + ent) * 2 + b128 % 2) * 4, qs[0])

    @cute.jit
    def _publish_row(self, cnt: Int64, tid: Int32):
        """Consumer warps: make this CTA's row writes visible, count it."""
        fence_proxy_async_global()
        named_barrier(1, NCT)
        if tid == 0:
            fence_gpu()
            atom_add_relaxed(cnt + C_BAR * 4, 1)

    @cute.jit
    def _wait_rows(self, cnt: Int64, M: Int32):
        while ld_acquire_i32(cnt + C_BAR * 4) < M:
            nanosleep(32)

    @cute.jit
    def _front_dense(
        self,
        ids: Int64,
        cnt: Int64,
        M: Int32,
        tid: Int32,
        warp: Int32,
        lane: Int32,
        tab: Int32,
        s_tmask: Int32,
        s_rcof: Int32,
        s_tot: Int32,
        r0: Int32,
    ):
        """Active experts in increasing id order -> one row chunk each with all
        Npad token columns and the routed tokens' mask."""
        npad = (M + 7) // 8 * 8
        for i in cutlass.range(tid, self.experts, NTHR, unroll=1):
            sts32(s_tmask + i * 4, Uint32(0))
        if cutlass.const_expr(not self.ids_mode):  # noqa: SIM102
            if tid == 0:
                self._wait_rows(cnt, M)
        cute.arch.sync_threads()
        if tid < M * self.top_k:
            ex = ldg_cg_i32(ids + Int64(tid) * 4)
            smem_atom_or(s_tmask + ex * 4, Uint32(1) << Uint32(tid // self.top_k))
        cute.arch.sync_threads()
        if warp == 0:
            c = Int32(0)
            for j in cutlass.range_constexpr(self.per_lane):
                if lds32(s_tmask + (lane * self.per_lane + j) * 4) != Uint32(0):
                    c = c + 1
            inc = c
            for sh in cutlass.range_constexpr(5):
                o = shfl_up_i32(inc, 1 << sh)
                if lane >= (1 << sh):
                    inc = inc + o
            r = inc - c + r0
            for j in cutlass.range_constexpr(self.per_lane):
                ex = lane * self.per_lane + j
                m = lds32(s_tmask + ex * 4)
                if m != Uint32(0):
                    sts32(tab + r * 4, Uint32(ex))
                    sts32(tab + (RC_CAP + r) * 4, Uint32(r * npad))
                    sts32(tab + (2 * RC_CAP + r) * 4, Uint32(npad))
                    sts32(tab + (3 * RC_CAP + r) * 4, m)
                    sts32(s_rcof + ex * 4, Uint32(r))
                    r = r + 1
            if lane == 31:
                sts32(s_tot, Uint32(inc + r0))
        cute.arch.sync_threads()
        return Int32(lds32(s_tot))

    @cute.jit
    def _rank_slot(self, t: Int32, k: Int32, s_ids: Int32, s_mask: Int32, s_off: Int32):
        ex = Int32(lds32(s_ids + (t * self.top_k + k) * 4))
        rank = Int32(0)
        for wd in cutlass.range(t // 32, unroll=1):
            rank = rank + popc(lds32(s_mask + (ex * MASKW + wd) * 4))
        low = (Uint32(1) << Uint32(t % 32)) - Uint32(1)
        rank = rank + popc(lds32(s_mask + (ex * MASKW + t // 32) * 4) & low)
        return ex, rank, Int32(lds32(s_off + ex * 4)) + rank

    @cute.jit
    def _front_entry(
        self,
        ids: Int64,
        cnt: Int64,
        M: Int32,
        tid: Int32,
        warp: Int32,
        lane: Int32,
        tab: Int32,
        s_ids: Int32,
        s_mask: Int32,
        s_cnt: Int32,
        s_off: Int32,
        s_rcb: Int32,
        s_tot: Int32,
        r0: Int32,
    ):
        """v4's expert-major layout (bitmaps, counts, scan, 8-aligned row chunks
        of <= 32 entries); returns the row-chunk count."""
        for i in cutlass.range(tid, self.experts * MASKW, NTHR, unroll=1):
            sts32(s_mask + i * 4, Uint32(0))
        for i in cutlass.range(tid, self.experts, NTHR, unroll=1):
            sts32(s_cnt + i * 4, Uint32(0))
        if cutlass.const_expr(not self.ids_mode):  # noqa: SIM102
            if tid == 0:
                self._wait_rows(cnt, M)
        cute.arch.sync_threads()
        for i in cutlass.range(tid, M * self.top_k, NTHR, unroll=1):
            ex = ldg_cg_i32(ids + Int64(i) * 4)
            r = i // self.top_k
            sts32(s_ids + i * 4, Uint32(ex))
            smem_atom_or(
                s_mask + (ex * MASKW + r // 32) * 4, Uint32(1) << Uint32(r % 32)
            )
            smem_atom_add(s_cnt + ex * 4, 1)
        cute.arch.sync_threads()
        if warp == 0:
            pe = cute.make_rmem_tensor((self.per_lane,), Int32)
            pr = cute.make_rmem_tensor((self.per_lane,), Int32)
            se = Int32(0)
            sr = Int32(0)
            for j in cutlass.range_constexpr(self.per_lane):
                c = Int32(lds32(s_cnt + (lane * self.per_lane + j) * 4))
                pe[j] = se
                pr[j] = sr
                se = se + c + (8 - c % 8) % 8
                sr = sr + (c + RC_ROWS - 1) // RC_ROWS
            ie = se
            ir = sr
            for sh in cutlass.range_constexpr(5):
                oe = shfl_up_i32(ie, 1 << sh)
                orr = shfl_up_i32(ir, 1 << sh)
                if lane >= (1 << sh):
                    ie = ie + oe
                    ir = ir + orr
            for j in cutlass.range_constexpr(self.per_lane):
                sts32(s_off + (lane * self.per_lane + j) * 4, Uint32(ie - se + pe[j]))
                sts32(
                    s_rcb + (lane * self.per_lane + j) * 4, Uint32(ir - sr + pr[j] + r0)
                )
            if lane == 31:
                sts32(s_tot, Uint32(ir + r0))
                sts32(s_tot + 4, Uint32(ie))
        cute.arch.sync_threads()
        for ex in cutlass.range(tid, self.experts, NTHR, unroll=1):
            c = Int32(lds32(s_cnt + ex * 4))
            o = Int32(lds32(s_off + ex * 4))
            rb = Int32(lds32(s_rcb + ex * 4))
            for j in cutlass.range((c + RC_ROWS - 1) // RC_ROWS, unroll=1):
                rem = c - RC_ROWS * j
                if rem > RC_ROWS:
                    rem = Int32(RC_ROWS)
                sts32(tab + (rb + j) * 4, Uint32(ex))
                sts32(tab + (RC_CAP + rb + j) * 4, Uint32(o + RC_ROWS * j))
                sts32(tab + (2 * RC_CAP + rb + j) * 4, Uint32(rem))
                sts32(tab + (3 * RC_CAP + rb + j) * 4, Uint32(0xFFFFFFFF))
        cute.arch.sync_threads()
        return Int32(lds32(s_tot))

    # ================================================================ shared expert

    @cute.jit
    def _promote_w(
        self,
        N: cutlass.Constexpr,
        d: cute.Tensor,
        base: cutlass.Constexpr,
        acc: cute.Tensor,
        wsa: Int32,
    ):
        """Acc += d * 2^(e_row - 127): one k32 group's UE8M0 weight scale."""
        w = lds_u16(wsa)
        f_lo = e8m0_to_f32(w)
        f_hi = e8m0_to_f32(w >> Uint32(8))
        for i in cutlass.range_constexpr(N // 8):
            acc[4 * i] = acc[4 * i] + d[base + 4 * i] * f_lo
            acc[4 * i + 1] = acc[4 * i + 1] + d[base + 4 * i + 1] * f_lo
            acc[4 * i + 2] = acc[4 * i + 2] + d[base + 4 * i + 2] * f_hi
            acc[4 * i + 3] = acc[4 * i + 3] + d[base + 4 * i + 3] * f_hi

    @cute.jit
    def _wgmma_k32_f16(
        self,
        N: cutlass.Constexpr,
        wt: Int32,
        bt: Int32,
        sbo: cutlass.Constexpr,
        R: Int32,
        lane: Int32,
        d: cute.Tensor,
        base: cutlass.Constexpr,
    ):
        """One k32 scale group = two k16 f16 wgmmas. ``wt``: the k32's first 1 KB
        W8A16 block of this warpgroup's 64 rows (the second is 2 KB further)."""
        vals = [d[base + i] for i in range(N // 2)]
        for kb in cutlass.range_constexpr(2):
            w = lds64(wt + kb * 2048 + lane * 32 + 8 * R)
            lo_g, hi_g = cvt_e4m3_word(w[0])
            lo_h, hi_h = cvt_e4m3_word(w[1])
            wgmma_fence()
            vals = wgmma_rs_f16(
                N,
                vals,
                lo_g,
                lo_h,
                hi_g,
                hi_h,
                make_desc_nosw(bt + kb * 256, 128, sbo),
                Int32(kb),
            )
        for i in cutlass.range_constexpr(N // 2):
            d[base + i] = vals[i]

    @cute.jit
    def _consume1_sh(
        self,
        N: cutlass.Constexpr,
        nst: Int32,
        stage: Int32,
        phase: Int32,
        sbase: Int32,
        full: Int32,
        empty: Int32,
        blk: Int32,
        R: Int32,
        lane: Int32,
        acc: cute.Tensor,
        d: cute.Tensor,
    ):
        """Shared GEMM1 stages (f16): [k16 (8)][gate/up][1 KB] W8A16 e4m3 weights,
        [kg (4)][gate/up][64 B] UE8M0 scales, B f16 [8-token group][k8 (16)][8][16 B]."""  # noqa: E501
        NS = self.ns1
        g = lane // 4
        for i in cutlass.range_constexpr(N // 2):
            acc[i] = Float32(0.0)
        for js in cutlass.range(nst, unroll=1):
            if js > 0:
                mbar_wait(full + stage * 8, phase)
            st = sbase + stage * SB1
            for q in cutlass.range_constexpr(4):
                self._wgmma_k32_f16(
                    N,
                    st + G1_W + ((2 * q) * 2 + blk) * 1024,
                    st + G1_B + 4 * q * 128,
                    2048,
                    R,
                    lane,
                    d,
                    (q % 2) * 16,
                )
                wgmma_commit()
                if cutlass.const_expr(q > 0):
                    wgmma_wait_n(1)
                    _pin(N, d, ((q - 1) % 2) * 16)
                    self._promote_w(
                        N,
                        d,
                        ((q - 1) % 2) * 16,
                        acc,
                        st + G1_WS + ((q - 1) * 2 + blk) * 64 + 8 * g + 2 * R,
                    )
            wgmma_wait_n(0)
            _pin(N, d, 16)
            self._promote_w(
                N, d, 16, acc, st + G1_WS + (3 * 2 + blk) * 64 + 8 * g + 2 * R
            )
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(empty + stage * 8)
            stage = stage + 1
            if stage == NS:
                stage = Int32(0)
                phase = phase ^ 1
        return stage, phase

    @cute.jit
    def _consume2_sh(
        self,
        N: cutlass.Constexpr,
        stage: Int32,
        phase: Int32,
        sbase: Int32,
        full: Int32,
        empty: Int32,
        blk: Int32,
        R: Int32,
        lane: Int32,
        acc: cute.Tensor,
        d: cute.Tensor,
    ):
        """Shared GEMM2 stages (f16): [k16 (6)][n64 block (2)][1 KB] weights,
        [kg (3)][128 B] scales, B f16 [8-token group][k8 (12)][8][16 B]."""
        NS = self.ns2
        g = lane // 4
        for i in cutlass.range_constexpr(N // 2):
            acc[i] = Float32(0.0)
        for js in cutlass.range(SH_V2, unroll=1):
            if js > 0:
                mbar_wait(full + stage * 8, phase)
            st = sbase + stage * SB2
            for q in cutlass.range_constexpr(3):
                self._wgmma_k32_f16(
                    N,
                    st + G2_W + ((2 * q) * 2 + blk) * 1024,
                    st + G2_B + 4 * q * 128,
                    1536,
                    R,
                    lane,
                    d,
                    (q % 2) * 16,
                )
                wgmma_commit()
                if cutlass.const_expr(q > 0):
                    wgmma_wait_n(1)
                    _pin(N, d, ((q - 1) % 2) * 16)
                    self._promote_w(
                        N,
                        d,
                        ((q - 1) % 2) * 16,
                        acc,
                        st + G2_WS + (q - 1) * 128 + 64 * blk + 8 * g + 2 * R,
                    )
            wgmma_wait_n(0)
            _pin(N, d, 0)
            self._promote_w(
                N, d, 0, acc, st + G2_WS + 2 * 128 + 64 * blk + 8 * g + 2 * R
            )
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(empty + stage * 8)
            stage = stage + 1
            if stage == NS:
                stage = Int32(0)
                phase = phase ^ 1
        return stage, phase

    @cute.jit
    def _xconv_sh(
        self,
        st: Int32,
        x: Int64,
        M: Int32,
        t0: Int32,
        ncol: Int32,
        u: Int32,
        lane: Int32,
    ):
        """Signal warp: the shared GEMM1 B tile of stage (k128 unit u) straight from
        x: bf16 -> f16, [8-token group][k8 (16)][8][16 B]. |x| <= sqrt(5120) *
        max|gamma| (~190 on DSV4.1-Flash) fits f16 without a prescale."""
        G = (ncol + 7) // 8
        tl = lane // 4
        ks = (lane % 4) * 32
        for gi in cutlass.range(G, unroll=1):
            tok = gi * 8 + tl
            t = t0 + tok
            msk = Uint32(0)
            tc = Int32(0)
            if tok < ncol:  # noqa: SIM102
                if t < M:
                    msk = Uint32(0xFFFFFFFF)
                    tc = t
            for c in cutlass.range_constexpr(4):
                v = ldg128(x + (Int64(tc) * HIDDEN + 128 * u + ks + 8 * c) * 2)
                sts128(
                    st + G1_B + gi * 2048 + ((ks + 8 * c) // 8) * 128 + tl * 16,
                    bf16x2_to_f16x2(v[0]) & msk,
                    bf16x2_to_f16x2(v[1]) & msk,
                    bf16x2_to_f16x2(v[2]) & msk,
                    bf16x2_to_f16x2(v[3]) & msk,
                )

    # ================================================================ GEMM1
    @cute.jit
    def _consume1(
        self,
        N: cutlass.Constexpr,
        nst: Int32,
        stage: Int32,
        phase: Int32,
        sbase: Int32,
        full: Int32,
        empty: Int32,
        blk: Int32,
        R: Int32,
        lane: Int32,
        acc: cute.Tensor,
        d: cute.Tensor,
        af: cute.Tensor,
    ):
        NS = self.ns1
        g = lane // 4
        t = lane % 4
        for i in cutlass.range_constexpr(N // 2):
            acc[i] = Float32(0.0)
        for js in cutlass.range(nst, unroll=1):
            if js > 0:
                mbar_wait(full + stage * 8, phase)
            st = sbase + stage * SB1
            qa = cute.make_rmem_tensor((6,), Uint32)
            qb = cute.make_rmem_tensor((6,), Uint32)
            ws0 = st + G1_WS + blk * 64 + 8 * g + 2 * R
            _dq_load(st + G1_W + blk * 2048, ws0, ws0 + 128, lane, R, qa)
            for kb in cutlass.range_constexpr(2):
                for kk in cutlass.range_constexpr(2):
                    ki = 2 * kb + kk
                    if cutlass.const_expr(ki < 3):
                        wn = st + G1_W + ((ki + 1) * 2 + blk) * 2048
                        wsn = (
                            st + G1_WS + ((2 * (ki + 1)) * 2 + blk) * 64 + 8 * g + 2 * R
                        )
                        if cutlass.const_expr(ki % 2 == 0):
                            _dq_load(wn, wsn, wsn + 128, lane, R, qb)
                        else:
                            _dq_load(wn, wsn, wsn + 128, lane, R, qa)
                    if cutlass.const_expr(ki % 2 == 0):
                        _dq_conv(qa, af)
                    else:
                        _dq_conv(qb, af)
                    _wgmma_k64_sw32(N, st + G1_B + ki * 512, af, d, kb * 16, Int32(kk))
                wgmma_commit()
                if cutlass.const_expr(kb == 1):
                    wgmma_wait_n(1)
                    _pin(N, d, 0)
                    _promote(N, d, 0, acc, st + G1_AS, 8, t)
            wgmma_wait_n(0)
            _pin(N, d, 16)
            _promote(N, d, 16, acc, st + G1_AS + 4, 8, t)
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(empty + stage * 8)
            stage = stage + 1
            if stage == NS:
                stage = Int32(0)
                phase = phase ^ 1
        return stage, phase

    @cute.jit
    def _issue_b1(
        self,
        st: Int32,
        fb: Int32,
        tab: Int32,
        rc: Int32,
        js_u: Int32,
        xg: Int64,
        xsg: Int64,
        xgs: Int64,
        xsgs: Int64,
        pol: Int64,
    ):
        G = (Int32(lds32(tab + (2 * RC_CAP + rc) * 4)) + 7) // 8
        is_sh = Int32(0)
        if cutlass.const_expr(self.shared):  # noqa: SIM102
            if Int32(lds32(tab + rc * 4)) == E_SH:
                is_sh = Int32(1)
        if is_sh == 0:
            off = Int32(0)
            if cutlass.const_expr(not self.dense):
                off = Int32(lds32(tab + (RC_CAP + rc) * 4))
            bulk_g2s(
                st + G1_B, xg + (_i64(js_u) * NG + off // 8) * 2048, G * 2048, fb, pol
            )
            bulk_g2s(st + G1_AS, xsg + (_i64(js_u) * ENTP + off) * 8, G * 64, fb, pol)

    @cute.jit
    def _produce1(
        self,
        tw13: Int64,
        ts13: Int64,
        sw13: Int64,
        ss13: Int64,
        xg: Int64,
        xsg: Int64,
        xgs: Int64,
        xsgs: Int64,
        cnt: Int64,
        btgt: Int32,
        n_chunks: Int32,
        NCH: Int32,
        UPC: Int32,
        sbase: Int32,
        full: Int32,
        empty: Int32,
        tab: Int32,
        nblk: Int32,
        bid: Int32,
        tsh: Int32,
        hb: Int32,
        pre: Int32,
    ):
        """GEMM1 weight tiles only; the signal warp issues the B tiles of each
        stage once its header is posted (``hb``), so waiting for the activations
        (and the proxy fence that would also wait for these weight loads) stays
        off this thread."""
        NS = self.ns1
        pol = evict_first_policy()
        work = cnt + C_WORK1 * 4
        stage = pre % NS  # stages of the first chunk already issued before the PDL wait
        phase = pre // NS
        skip = pre
        kst = Int32(0)
        nxt = bid
        running = Int32(1)
        while running == 1:
            ch = nxt
            if ch < n_chunks:
                if cutlass.const_expr(self.cluster):
                    nxt = ch + nblk
                else:
                    nxt = nblk + atom_add_relaxed(work, 1)
            tl, h = _ch_tile(ch, NCH, tsh)
            rc = tl // NPAIR
            p = tl % NPAIR
            e = Int32(lds32(tab + rc * 4))
            G = (Int32(lds32(tab + (2 * RC_CAP + rc) * 4)) + 7) // 8
            nst = UPC
            if e == E_SH:
                nst = SH_U1 // (tsh % 32)
            if ch >= n_chunks:
                nst = Int32(1)
                running = Int32(0)
            for js in cutlass.range(skip, nst, unroll=1):
                mbar_wait(empty + stage * 8, phase ^ 1)
                st = sbase + stage * SB1
                fb = full + stage * 8
                if running == 0:
                    sts32(st + HDR1, Uint32(END))
                    mbar_arrive_expect_tx(fb, 0)
                    mbar_arrive(hb + stage * 8)
                else:
                    js_u = h * nst + js
                    sts32(st + HDR1, Uint32(ch))
                    sts32(st + HDR1 + 4, Uint32(js_u))
                    if e == E_SH:
                        mbar_arrive_expect_tx(fb, 16384 + 512)
                        tma_load_4d(st + G1_W, sw13, 0, p, 0, js_u * 8, fb, pol)
                        tma_load_4d(st + G1_WS, ss13, 0, p, 0, js_u * 4, fb, pol)
                    else:
                        kt = js_u * 4
                        mbar_arrive_expect_tx(fb, G1_B + G * 2048 + G * 64)
                        tma_load_4d(st + G1_W, tw13, 0, p, 0, e * KT1 + kt, fb, pol)
                        tma_load_4d(
                            st + G1_WS, ts13, 0, p, 0, e * (2 * KT1) + 2 * kt, fb, pol
                        )
                    mbar_arrive(hb + stage * 8)
                    kst = kst + 1
                stage = stage + 1
                if stage == NS:
                    stage = Int32(0)
                    phase = phase ^ 1
            skip = Int32(0)
        # drain: every GEMM1 stage released before ring 2 reuses the smem
        for j in cutlass.range(NS, unroll=1):
            mbar_wait(empty + stage * 8, phase ^ 1)
            stage = stage + 1
            if stage == NS:
                stage = Int32(0)
                phase = phase ^ 1

    # ================================================================ signal warp
    @cute.jit
    def _b_loop(
        self,
        x: Int64,
        M: Int32,
        xg: Int64,
        xsg: Int64,
        xgs: Int64,
        xsgs: Int64,
        cnt: Int64,
        btgt: Int32,
        NCH: Int32,
        tsh: Int32,
        sbase: Int32,
        full: Int32,
        hb: Int32,
        tab: Int32,
        bid: Int32,
    ):
        """B tiles (activations) of every GEMM1 stage, in ring order, after the
        rows are published."""
        NS = self.ns1
        lane = cute.arch.lane_idx()
        pol = evict_first_policy()
        ready = Int32(0)
        stage = Int32(0)
        phase = Int32(0)
        running = Int32(1)
        while running == 1:
            mbar_wait(hb + stage * 8, phase)
            st = sbase + stage * SB1
            fb = full + stage * 8
            ch = Int32(lds32(st + HDR1))
            if ch == END:
                running = Int32(0)
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(fb)
            else:
                tl, h = _ch_tile(ch, NCH, tsh)
                rc = tl // NPAIR
                js_u = Int32(lds32(st + HDR1 + 4))
                if Int32(lds32(tab + rc * 4)) == E_SH:
                    self._xconv_sh(
                        st,
                        x,
                        M,
                        Int32(lds32(tab + (RC_CAP + rc) * 4)),
                        Int32(lds32(tab + (2 * RC_CAP + rc) * 4)),
                        js_u,
                        lane,
                    )
                    fence_proxy_async_smem()
                    cute.arch.sync_warp()
                    if lane == 0:
                        mbar_arrive(fb)
                else:
                    if lane == 0:
                        if ready == 0:
                            self._wait_rows(cnt, btgt)
                            fence_proxy_async_global()
                            ready = Int32(1)
                        self._issue_b1(st, fb, tab, rc, js_u, xg, xsg, xgs, xsgs, pol)
                        mbar_arrive(fb)
                    cute.arch.sync_warp()
                stage = stage + 1
                if stage == NS:
                    stage = Int32(0)
                    phase = phase ^ 1

    @cute.jit
    def _signal(
        self,
        sync: Int64,
        stamp: Int32,
        NCH: Int32,
        tsh: Int32,
        s_done: Int32,
        s_fin: Int32,
        s_fdone: Int32,
        clist: Int32,
        flist: Int32,
    ):
        processed = Int32(0)
        fin_n = Int32(0)
        ended = Int32(0)
        while ended == 0:
            avail = _min_count(s_done)
            if avail > processed:
                fence_gpu()
                for s in cutlass.range(processed, avail, unroll=1):
                    ch = Int32(lds32(clist + (s % CL) * 4))
                    if ch == END:
                        ended = Int32(1)
                    else:
                        tile, hh = _ch_tile(ch, NCH, tsh)
                        nch = _tile_nch(tile, NCH, tsh)
                        if hh == nch - 1:
                            sts32(flist + (fin_n % CL) * 4, Uint32(tile))
                            fin_n = fin_n + 1
                        else:
                            red_add_gpu(sync + _i64(S_TILE + tile) * 4, 1)
                processed = avail
            else:
                nanosleep(20)
        # this CTA finalizes the tiles whose last k-chunk it ran, once the
        # other chunks are counted
        for i in cutlass.range(fin_n, unroll=1):
            tile = Int32(lds32(flist + (i % CL) * 4))
            nch = _tile_nch(tile, NCH, tsh)
            while ld_acquire_i32(sync + _i64(S_TILE + tile) * 4) < nch - 1:
                nanosleep(32)
            stg_i32(sync + _i64(S_TILE + tile) * 4, 0)
        sts_release(s_fin, fin_n | FIN_FLAG)
        if fin_n > 0:
            while _min_count(s_fdone) == 0:
                nanosleep(20)
            fence_gpu()
            for i in cutlass.range(fin_n, unroll=1):
                tile = Int32(lds32(flist + (i % CL) * 4))
                red_add_gpu(sync + _i64(S_READY + tile // NPAIR) * 4, 1)
            red_add_gpu(sync + _i64(S_ALL) * 4, fin_n)

    # ================================================================ SwiGLU
    @cute.jit
    def _swiglu_one(
        self,
        NQ: cutlass.Constexpr,
        HM: cutlass.Constexpr,
        tile: Int32,
        c0: Int32,
        ncol: Int32,
        mask: Uint32,
        NCH: Int32,
        tsh: Int32,
        tab: Int32,
        g2: Int64,
        part: Int64,
        iq: Int64,
        isc: Int64,
        iqs: Int64,
        iscs: Int64,
        lane: Int32,
    ):
        """SwiGLU + FP8 of columns c0 + 8 q (q < NQ) of one finalized tile, with
        the partial loads of all NQ columns (<= HM k-chunks each) issued at once."""
        rc = tile // NPAIR
        p = tile % NPAIR
        e = Int32(lds32(tab + rc * 4))
        off = Int32(lds32(tab + (RC_CAP + rc) * 4))
        f2 = Float32(1.0)
        if e != E_SH:
            f2 = ldg_f32(g2 + _i64(e) * 4) * Float32(64.0)
        nch = _tile_nch(tile, NCH, tsh)
        pv = cute.make_rmem_tensor((4 * NQ * HM,), Float32)
        for qq in cutlass.range_constexpr(NQ):
            cq = c0 + 8 * qq
            if cq < ncol:  # noqa: SIM102
                if ((mask >> Uint32(cq)) & Uint32(1)) != Uint32(0):
                    for h in cutlass.range_constexpr(HM):
                        if h < nch:
                            a = (
                                part
                                + (
                                    (_i64(_tile_ch(tile, Int32(h), NCH, tsh)) * 32 + cq)
                                    * 128
                                    + 2 * lane
                                )
                                * 4
                            )
                            gv = ldg_cg_f32x2(a)
                            uv = ldg_cg_f32x2(a + 64 * 4)
                            b = (qq * HM + h) * 4
                            pv[b] = gv[0]
                            pv[b + 1] = gv[1]
                            pv[b + 2] = uv[0]
                            pv[b + 3] = uv[1]
        for qq in cutlass.range_constexpr(NQ):
            cc = c0 + 8 * qq
            okc = Int32(0)
            if cc < ncol:  # noqa: SIM102
                if ((mask >> Uint32(cc)) & Uint32(1)) != Uint32(0):
                    okc = Int32(1)
            if okc == 1:
                ent = off + cc
                b0 = qq * HM * 4
                g0 = pv[b0]
                g1 = pv[b0 + 1]
                u0 = pv[b0 + 2]
                u1 = pv[b0 + 3]
                for h in cutlass.range_constexpr(1, HM):
                    if h < nch:
                        g0 = g0 + pv[b0 + 4 * h]
                        g1 = g1 + pv[b0 + 4 * h + 1]
                        u0 = u0 + pv[b0 + 4 * h + 2]
                        u1 = u1 + pv[b0 + 4 * h + 3]
                if e == E_SH:
                    g0 = round_bf16(g0)
                    g1 = round_bf16(g1)
                    u0 = round_bf16(u0)
                    u1 = round_bf16(u1)
                h0 = swiglu(g0, u0, Float32(SWIGLU_LIMIT))
                h1 = swiglu(g1, u1, Float32(SWIGLU_LIMIT))
                amax = fmax(fabs(h0), fabs(h1))
                for sh in cutlass.range_constexpr(5):
                    amax = fmax(amax, shfl_xor_f32(amax, 1 << sh))
                amax = fmax(amax, Float32(1e-12))
                inv = Float32(448.0) * rcp(amax)
                q = cvt_e4m3x4(h0 * inv, h1 * inv, Float32(0.0), Float32(0.0))
                if e == E_SH:
                    # shared GEMM2 B (f16): [k96 v][t/8][k8 (12)][t%8][16 B]
                    kq = 64 * p + 2 * lane
                    dst = (
                        iqs
                        + ((_i64(kq // 96) * NGS + ent // 8) * 12 + (kq % 96) // 8)
                        * 128
                        + (ent % 8) * 16
                        + (kq % 8) * 2
                    )
                    stg_u32(dst, f16x2(round_bf16(h0), round_bf16(h1)))
                else:
                    p3 = p // K2S
                    ki = p % K2S
                    r = 2 * lane
                    dst = (
                        iq
                        + (((_i64(p3) * NG + ent // 8) * K2S + ki) * 4 + r // 16) * 128
                        + (ent % 8) * 16
                        + r % 16
                    )
                    stg_u16(dst, q & Uint32(0xFFFF))
                    if lane == 0:
                        stg_f32(
                            isc + ((_i64(p3) * ENTP + ent) * K2S + ki) * 4,
                            amax * Float32(1.0 / 448.0) * f2,
                        )

    @cute.jit
    def _cluster_tile(
        self,
        tile: Int32,
        hq: Int32,
        rc: Int32,
        e: Int32,
        ncol: Int32,
        mask: Uint32,
        f13: Float32,
        acc: cute.Tensor,
        r0: Int32,
        t: Int32,
        lane: Int32,
        warp: Int32,
        tid: Int32,
        cph: Int32,
        slot: Int32,
        cfull: Int32,
        cfree: Int32,
        tab: Int32,
        g2: Int64,
        iq: Int64,
        isc: Int64,
        sync: Int64,
    ) -> Int32:
        """Routed tile, k-half ``hq`` = cluster rank (N = 8 columns). Rank 1:
        partial -> rank 0's ``slot`` (DSMEM). Rank 0: own + rank-1 partial ->
        SwiGLU + FP8 intermediate of the tile, then publish it (S_READY)."""
        G = (ncol + 7) // 8
        if hq == 1:
            mbar_wait_cluster(cfree, cph)
            rs = mapa(slot, 0)
            for j in cutlass.range_constexpr(2):
                if j < G:
                    for i2 in cutlass.range_constexpr(2):
                        cc = 8 * j + 2 * t + i2
                        st_cluster_f32(rs + (cc * 128 + r0) * 4, acc[4 * j + i2] * f13)
                        st_cluster_f32(
                            rs + (cc * 128 + r0 + 8) * 4, acc[4 * j + 2 + i2] * f13
                        )
            mbar_arrive_cluster(mapa(cfull, 0))
        else:
            mbar_wait_cluster(cfull, cph)
            for j in cutlass.range_constexpr(2):
                if j < G:
                    for i2 in cutlass.range_constexpr(2):
                        cc = 8 * j + 2 * t + i2
                        a0 = slot + (cc * 128 + r0) * 4
                        a1 = slot + (cc * 128 + r0 + 8) * 4
                        v0 = acc[4 * j + i2] * f13 + lds_f32(a0)
                        v1 = acc[4 * j + 2 + i2] * f13 + lds_f32(a1)
                        sts_f32(a0, v0)
                        sts_f32(a1, v1)
            named_barrier(3, NCT)
            for q2 in cutlass.range_constexpr(2):
                self._cluster_swiglu(
                    warp + 8 * q2, tile, rc, e, ncol, mask, lane, slot, tab, g2, iq, isc
                )
            fence_proxy_async_global()
            named_barrier(3, NCT)
            if tid == 0:
                mbar_arrive_cluster(mapa(cfree, 1))
                fence_gpu()
                red_add_gpu(sync + _i64(S_READY + rc) * 4, 1)
                red_add_gpu(sync + _i64(S_ALL) * 4, 1)
        return cph ^ 1

    @cute.jit
    def _cluster_swiglu(
        self,
        cc: Int32,
        tile: Int32,
        rc: Int32,
        e: Int32,
        ncol: Int32,
        mask: Uint32,
        lane: Int32,
        slot: Int32,
        tab: Int32,
        g2: Int64,
        iq: Int64,
        isc: Int64,
    ):
        if cc < ncol:  # noqa: SIM102
            if ((mask >> Uint32(cc)) & Uint32(1)) != Uint32(0):
                p = tile % NPAIR
                off = Int32(lds32(tab + (RC_CAP + rc) * 4))
                f2 = ldg_f32(g2 + _i64(e) * 4) * Float32(64.0)
                a = slot + (cc * 128 + 2 * lane) * 4
                g0 = lds_f32(a)
                g1 = lds_f32(a + 4)
                u0 = lds_f32(a + 256)
                u1 = lds_f32(a + 260)
                h0 = swiglu(g0, u0, Float32(SWIGLU_LIMIT))
                h1 = swiglu(g1, u1, Float32(SWIGLU_LIMIT))
                amax = fmax(fabs(h0), fabs(h1))
                for sh in cutlass.range_constexpr(5):
                    amax = fmax(amax, shfl_xor_f32(amax, 1 << sh))
                amax = fmax(amax, Float32(1e-12))
                inv = Float32(448.0) * rcp(amax)
                q = cvt_e4m3x4(h0 * inv, h1 * inv, Float32(0.0), Float32(0.0))
                ent = off + cc
                p3 = p // K2S
                ki = p % K2S
                r = 2 * lane
                dst = (
                    iq
                    + (((_i64(p3) * NG + ent // 8) * K2S + ki) * 4 + r // 16) * 128
                    + (ent % 8) * 16
                    + r % 16
                )
                stg_u16(dst, q & Uint32(0xFFFF))
                if lane == 0:
                    stg_f32(
                        isc + ((_i64(p3) * ENTP + ent) * K2S + ki) * 4,
                        amax * Float32(1.0 / 448.0) * f2,
                    )

    # ================================================================ GEMM2
    @cute.jit
    def _consume2(
        self,
        N: cutlass.Constexpr,
        stage: Int32,
        phase: Int32,
        sbase: Int32,
        full: Int32,
        empty: Int32,
        blk: Int32,
        R: Int32,
        lane: Int32,
        acc: cute.Tensor,
        d: cute.Tensor,
        af: cute.Tensor,
    ):
        NS = self.ns2
        t = lane % 4
        g = lane // 4
        for i in cutlass.range_constexpr(N // 2):
            acc[i] = Float32(0.0)
        for js in cutlass.range(NPAIR // K2S, unroll=1):
            if js > 0:
                mbar_wait(full + stage * 8, phase)
            st = sbase + stage * SB2
            qa = cute.make_rmem_tensor((6,), Uint32)
            qb = cute.make_rmem_tensor((6,), Uint32)
            ws0 = st + G2_WS + blk * 64 + 8 * g + 2 * R
            _dq_load(st + G2_W + blk * 2048, ws0, ws0 + 128, lane, R, qa)
            for ki in cutlass.range_constexpr(K2S):
                if cutlass.const_expr(ki < K2S - 1):
                    wn = st + G2_W + ((ki + 1) * 2 + blk) * 2048
                    wsn = st + G2_WS + (2 * (ki + 1)) * 128 + blk * 64 + 8 * g + 2 * R
                    if cutlass.const_expr(ki % 2 == 0):
                        _dq_load(wn, wsn, wsn + 128, lane, R, qb)
                    else:
                        _dq_load(wn, wsn, wsn + 128, lane, R, qa)
                if cutlass.const_expr(ki % 2 == 0):
                    _dq_conv(qa, af)
                else:
                    _dq_conv(qb, af)
                _wgmma_k64(
                    N, st + G2_B + ki * 512, 1536, af, d, (ki % 2) * 16, Int32(0)
                )
                wgmma_commit()
                if cutlass.const_expr(ki > 0):
                    wgmma_wait_n(1)
                    _pin(N, d, ((ki - 1) % 2) * 16)
                    _promote(
                        N, d, ((ki - 1) % 2) * 16, acc, st + G2_AS + (ki - 1) * 4, 12, t
                    )
            wgmma_wait_n(0)
            _pin(N, d, ((K2S - 1) % 2) * 16)
            _promote(N, d, ((K2S - 1) % 2) * 16, acc, st + G2_AS + (K2S - 1) * 4, 12, t)
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(empty + stage * 8)
            stage = stage + 1
            if stage == NS:
                stage = Int32(0)
                phase = phase ^ 1
        return stage, phase

    @cute.jit
    def _issue_b2(
        self,
        it: Int32,
        js: Int32,
        st: Int32,
        fb: Int32,
        tab: Int32,
        iq: Int64,
        isc: Int64,
        iqs: Int64,
        iscs: Int64,
        pol: Int64,
    ):
        rc = it // NBP2
        off = Int32(lds32(tab + (RC_CAP + rc) * 4))
        G = (Int32(lds32(tab + (2 * RC_CAP + rc) * 4)) + 7) // 8
        if Int32(lds32(tab + rc * 4)) == E_SH:
            bulk_g2s(
                st + G2_B, iqs + (_i64(js) * NGS + off // 8) * 1536, G * 1536, fb, pol
            )
        else:
            bulk_g2s(
                st + G2_B, iq + (_i64(js) * NG + off // 8) * 1536, G * 1536, fb, pol
            )
            bulk_g2s(st + G2_AS, isc + (_i64(js) * ENTP + off) * 12, G * 96, fb, pol)

    @cute.jit
    def _wait_tiles(
        self,
        sync: Int64,
        rc: Int32,
        p_lo: Int32,
        p_hi: Int32,
        stamp: Int32,
        all_ok: Int32,
        n_tiles_all: Int32,
    ) -> Int32:
        """All 9 intermediate tiles of row chunk rc are finalized; returns 1 once
        every tile of the call is (then no more polls are needed)."""
        if all_ok == 0:
            if ld_acquire_i32(sync + _i64(S_ALL) * 4) >= n_tiles_all:
                all_ok = Int32(1)
            else:
                while ld_acquire_i32(sync + _i64(S_READY + rc) * 4) < NPAIR:
                    nanosleep(32)
            fence_proxy_async_global()
        return all_ok

    @cute.jit
    def _stage_pairs(self, e: Int32, js: Int32):
        """First / last intermediate pair (64 columns) a GEMM2 stage reads."""
        p_lo = js * K2S
        p_hi = js * K2S + K2S - 1
        if e == E_SH:
            p_lo = (3 * js) // 2
            p_hi = (3 * js + 2) // 2
        return p_lo, p_hi

    @cute.jit
    def _produce2(
        self,
        tw2: Int64,
        ts2: Int64,
        sw2: Int64,
        ss2: Int64,
        iq: Int64,
        isc: Int64,
        iqs: Int64,
        iscs: Int64,
        cnt: Int64,
        sync: Int64,
        stamp: Int32,
        n_rc: Int32,
        sbase: Int32,
        full: Int32,
        empty: Int32,
        tab: Int32,
        nblk: Int32,
        bid: Int32,
    ):
        """w2 tiles of the first NS2 stages at once; each stage's B tile after
        its row chunk's intermediate is published."""
        NS = self.ns2
        SPI = NPAIR // K2S
        pol = evict_first_policy()
        n_items = n_rc * NBP2
        work = cnt + C_WORK2 * 4
        nxt = bid
        k = Int32(0)
        flushed = Int32(0)
        last_rc = Int32(-1)
        last_p = Int32(-1)
        all_ok = Int32(0)
        running = Int32(1)
        while running == 1:
            it = nxt
            if it < n_items:
                nxt = nblk + atom_add_relaxed(work, 1)
            rc = it // NBP2
            nbp = it % NBP2
            e = Int32(lds32(tab + rc * 4))
            G = (Int32(lds32(tab + (2 * RC_CAP + rc) * 4)) + 7) // 8
            nst = Int32(SPI)
            if e == E_SH:
                nst = Int32(SH_V2)
            if it >= n_items:
                nst = Int32(1)
                running = Int32(0)
            for js in cutlass.range(nst, unroll=1):
                stage = k % NS
                phase = (k // NS) % 2
                mbar_wait(empty + stage * 8, phase ^ 1)
                st = sbase + stage * SB2
                fb = full + stage * 8
                if running == 0:
                    sts32(st + HDR2, Uint32(END))
                    mbar_arrive_expect_tx(fb, 0)
                else:
                    sts32(st + HDR2, Uint32(it))
                    sts32(st + HDR2 + 4, Uint32(js))
                    if e == E_SH:
                        mbar_arrive_expect_tx(fb, 12288 + 384 + G * 1536)
                        tma_load_3d(st + G2_W, sw2, 0, 2 * nbp, 6 * js, fb, pol)
                        tma_load_2d(st + G2_WS, ss2, 128 * nbp, 3 * js, fb, pol)
                    else:
                        mbar_arrive_expect_tx(fb, G2_B + G * 1536 + G * 96)
                        tma_load_3d(
                            st + G2_W, tw2, 0, 2 * nbp, e * 10 + js * K2S, fb, pol
                        )
                        tma_load_3d(
                            st + G2_WS, ts2, 0, nbp, e * 20 + js * 2 * K2S, fb, pol
                        )
                    if flushed == 1:
                        p_lo, p_hi = self._stage_pairs(e, js)
                        if rc != last_rc:
                            all_ok = self._wait_tiles(
                                sync, rc, p_lo, p_hi, stamp, all_ok, n_rc * NPAIR
                            )
                            last_rc = rc
                            last_p = p_hi
                        self._issue_b2(it, js, st, fb, tab, iq, isc, iqs, iscs, pol)
                    elif k == NS - 1:
                        flushed = Int32(1)
                        for k2 in cutlass.range(NS, unroll=1):
                            st2 = sbase + k2 * SB2
                            it2 = Int32(lds32(st2 + HDR2))
                            rc2 = it2 // NBP2
                            js2 = Int32(lds32(st2 + HDR2 + 4))
                            p_lo, p_hi = self._stage_pairs(
                                Int32(lds32(tab + rc2 * 4)), js2
                            )
                            if rc2 != last_rc:
                                all_ok = self._wait_tiles(
                                    sync, rc2, p_lo, p_hi, stamp, all_ok, n_rc * NPAIR
                                )
                                last_rc = rc2
                                last_p = p_hi
                            self._issue_b2(
                                it2,
                                Int32(lds32(st2 + HDR2 + 4)),
                                st2,
                                full + k2 * 8,
                                tab,
                                iq,
                                isc,
                                iqs,
                                iscs,
                                pol,
                            )
                k = k + 1
        if flushed == 0:
            for k2 in cutlass.range(k - 1, unroll=1):
                st2 = sbase + k2 * SB2
                it2 = Int32(lds32(st2 + HDR2))
                rc2 = it2 // NBP2
                js2 = Int32(lds32(st2 + HDR2 + 4))
                p_lo, p_hi = self._stage_pairs(Int32(lds32(tab + rc2 * 4)), js2)
                if rc2 != last_rc:
                    all_ok = self._wait_tiles(
                        sync, rc2, p_lo, p_hi, stamp, all_ok, n_rc * NPAIR
                    )
                    last_rc = rc2
                    last_p = p_hi  # noqa: F841
                self._issue_b2(
                    it2,
                    Int32(lds32(st2 + HDR2 + 4)),
                    st2,
                    full + k2 * 8,
                    tab,
                    iq,
                    isc,
                    iqs,
                    iscs,
                    pol,
                )

    @cute.jit
    def _store_rows(
        self,
        N: cutlass.Constexpr,
        acc: cute.Tensor,
        y: Int64,
        off: Int32,
        ncol: Int32,
        mask: Uint32,
        n0: Int32,
        lane: Int32,
    ):
        """Rows of 16 output columns per lane pair; finalize: fp32, else bf16
        pairs via lane^4. Only the routed columns (mask) are written."""
        t = lane % 4
        g = lane // 4
        for j in cutlass.range_constexpr(N // 8):
            odd = g % 2
            s_lo = acc[4 * j + 1]
            s_hi = acc[4 * j + 3]
            if odd == 1:
                s_lo = acc[4 * j]
                s_hi = acc[4 * j + 2]
            r_lo = shfl_xor_f32(s_lo, 4)
            r_hi = shfl_xor_f32(s_hi, 4)
            cc = 8 * j + 2 * t + odd
            if cc < ncol:  # noqa: SIM102
                if ((mask >> Uint32(cc)) & Uint32(1)) != Uint32(0):
                    rowp = y + (_i64(off + cc) * HIDDEN + n0 - odd) * 2
                    if odd == 0:
                        stg_u32(rowp, cvt_bf16x2(acc[4 * j], r_lo))
                        stg_u32(rowp + 16, cvt_bf16x2(acc[4 * j + 2], r_hi))
                    else:
                        stg_u32(rowp, cvt_bf16x2(r_lo, acc[4 * j + 1]))
                        stg_u32(rowp + 16, cvt_bf16x2(r_hi, acc[4 * j + 3]))

    # ================================================================ kernel
    @cute.kernel
    def kernel(
        self,
        maps: Int64,
        g13: Int64,
        g2: Int64,
        xg: Int64,
        xsg: Int64,
        rec: Int64,
        iq: Int64,
        isc: Int64,
        part: Int64,
        cnt: Int64,
        sync: Int64,
        x: Int64,
        ids: Int64,
        wts: Int64,
        logits: Int64,
        bias: Int64,
        y: Int64,
        out: Int64,
        xgs: Int64,
        xsgs: Int64,
        iqs: Int64,
        iscs: Int64,
        sh: Int64,
        M: Int32,
    ):
        NS = self.ns1
        NS2 = self.ns2
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        warp = tid // 32
        lane = tid % 32
        tw13 = maps
        ts13 = maps + 128
        tw2 = maps + 256
        ts2 = maps + 384
        sw13 = maps + 512
        ss13 = maps + 640
        sw2 = maps + 768
        ss2 = maps + 896
        smem = cute.arch.alloc_smem(cutlass.Uint8, self.smem, alignment=1024)
        sbase = smem_addr(smem)
        U = max(
            NS * SB1 + FRONT_SCRATCH,
            max(NS * SB1, NS2 * SB2) + (red_bytes(NS2) if self.cluster else 0),
        )
        slot = sbase + max(
            NS * SB1, NS2 * SB2
        )  # cluster: rank-1 partial (free of both rings)
        s_ids = sbase + NS * SB1
        s_mask = s_ids + MAXM * MAX_TOPK * 4
        s_cnt = s_mask + MAX_EXPERTS * MASKW * 4
        s_off = s_cnt + MAX_EXPERTS * 4
        s_rcb = s_off + MAX_EXPERTS * 4
        s_tot = s_rcb + MAX_EXPERTS * 4 * 2
        tab = sbase + U
        full = tab + TAB_BYTES
        empty = full + NS * 8
        full2 = empty + NS * 8
        empty2 = full2 + NS2 * 8
        s_done = empty2 + NS2 * 8  # [8] per consumer warp: chunks stored
        s_fdone = s_done + 32  # [8] per consumer warp: tiles finalized
        s_fin = s_done + 64
        clist = s_done + 80
        flist = clist + CL * 4
        hb = flist + CL * 4  # GEMM1 stage header posted (producer -> signal warp)
        cfull = hb + 56  # cluster rank 0: rank-1 partial landed in ``slot``
        cfree = hb + 64  # cluster rank 1: rank 0 consumed ``slot``
        if tid == 0:
            for s in cutlass.range_constexpr(NS):
                mbar_init(full + s * 8, 2)  # producer (weights tx) + signal warp (B)
                mbar_init(empty + s * 8, NCW)
            for s in cutlass.range_constexpr(NS2):
                mbar_init(full2 + s * 8, 1)
                mbar_init(empty2 + s * 8, NCW)
            for w in cutlass.range_constexpr(20):
                sts32(s_done + w * 4, Uint32(0))
            for s in cutlass.range_constexpr(NS):
                mbar_init(hb + s * 8, 1)
            mbar_init(cfull, NCT)
            mbar_init(cfree, 1)
            fence_mbar_init()
            if cutlass.const_expr(self.cluster):
                fence_mbar_init_cluster()
        if warp == PROD:  # noqa: SIM102
            if lane == 0:
                prefetch_tmap(tw13)
                prefetch_tmap(ts13)
                prefetch_tmap(tw2)
                prefetch_tmap(ts2)
                if cutlass.const_expr(self.shared):
                    prefetch_tmap(sw13)
                    prefetch_tmap(ss13)
                    prefetch_tmap(sw2)
                    prefetch_tmap(ss2)
        if cutlass.const_expr(self.shared):  # noqa: SIM102
            # PDL: the shared expert's weights do not depend on the previous
            # kernel -- pull this CTA's slice of them into L2 before the wait
            if warp == SIG:  # noqa: SIM102
                if lane < 4:
                    base = ldg_i64(maps + 1024 + lane * 16)
                    nb = Int32(ldg_i64(maps + 1024 + lane * 16 + 8))
                    chunk = ((nb + nblk - 1) // nblk + 15) // 16 * 16
                    start = bid * chunk
                    if start < nb:
                        n = nb - start
                        if n > chunk:
                            n = chunk
                        prefetch_l2_bulk(base + Int64(start), n)
        cute.arch.sync_threads()
        if cutlass.const_expr(self.cluster):
            cluster_sync_all()  # the peer's barriers are initialized
        pre = Int32(0)
        if cutlass.const_expr(self.shared and self.dense):  # noqa: SIM102
            # chunks 0 .. 9 * NCHS - 1 are the shared expert's (rc 0), statically
            # chunk = blockIdx: their weights do not depend on the previous kernel
            if warp == PROD:  # noqa: SIM102
                if lane == 0:  # noqa: SIM102
                    if bid < NPAIR * self.nchs_d:
                        pol0 = evict_first_policy()
                        tl0 = bid // self.nchs_d
                        h0 = bid % self.nchs_d
                        nst0 = SH_U1 // self.nchs_d
                        for js in cutlass.range_constexpr(
                            min(SH_U1 // self.nchs_d, NS)
                        ):
                            st0 = sbase + js * SB1
                            js_u0 = h0 * nst0 + js
                            sts32(st0 + HDR1, Uint32(bid))
                            sts32(st0 + HDR1 + 4, Uint32(js_u0))
                            mbar_arrive_expect_tx(full + js * 8, 16384 + 512)
                            tma_load_4d(
                                st0 + G1_W,
                                sw13,
                                0,
                                tl0 % NPAIR,
                                0,
                                js_u0 * 8,
                                full + js * 8,
                                pol0,
                            )
                            tma_load_4d(
                                st0 + G1_WS,
                                ss13,
                                0,
                                tl0 % NPAIR,
                                0,
                                js_u0 * 4,
                                full + js * 8,
                                pol0,
                            )
                            mbar_arrive(hb + js * 8)
                        pre = Int32(min(SH_U1 // self.nchs_d, NS))
        griddep_wait()
        # ---------------- front
        qw = cute.make_rmem_tensor((8,), Uint32)
        qs = cute.make_rmem_tensor((1,), Float32)
        n_rc = Int32(0)
        nsh_rc = Int32(0)  # shared-expert row chunks (rc 0 .. nsh_rc - 1)
        if cutlass.const_expr(self.shared):
            nsh_rc = Int32(1) if self.dense else (M + 31) // 32
        if cutlass.const_expr(self.dense):
            if bid < M:
                if cutlass.const_expr(self.image):
                    if warp == 0:
                        _route_row2(
                            logits,
                            bias,
                            ldg_i64(maps + 1024 + 9 * 8),
                            xgs,
                            Int32(ldg_i64(maps + 1024 + 10 * 8)),
                            ids,
                            wts,
                            bid,
                            lane,
                            self.experts,
                            self.top_k,
                            True,
                            False,
                        )
                elif cutlass.const_expr(not self.ids_mode):  # noqa: SIM102
                    if warp == 0:
                        _route_row(
                            logits, bias, ids, wts, bid, lane, self.experts, self.top_k
                        )
                if tid < 160:
                    self._quant_row(x, bid, tid, qw, qs)
                    self._store_xrow(xg, xsg, bid, tid, qw, qs)
                if warp < NCW:
                    self._publish_row(cnt, tid)
            n_rc = self._front_dense(
                ids, cnt, M, tid, warp, lane, tab, s_mask, s_off, s_tot, nsh_rc
            )
            if cutlass.const_expr(self.shared):
                npad = (M + 7) // 8 * 8
                if tid == 0:
                    sts32(tab, Uint32(E_SH))
                    sts32(tab + RC_CAP * 4, Uint32(0))
                    sts32(tab + 2 * RC_CAP * 4, Uint32(npad))
                    sts32(tab + 3 * RC_CAP * 4, (Uint32(1) << Uint32(M)) - Uint32(1))
                cute.arch.sync_threads()
            if bid < M:  # noqa: SIM102
                # the boundary's finalize reads the map before its griddepcontrol.wait
                if warp == SIG:  # noqa: SIM102
                    if lane < self.top_k:
                        ex = ldg_cg_i32(ids + Int64(bid * self.top_k + lane) * 4)
                        stg_i32(
                            rec + Int64(REC_POS + bid * self.top_k + lane) * 4,
                            Int32(lds32(s_off + ex * 4)) * ((M + 7) // 8 * 8) + bid,
                        )
                        fence_gpu()
        else:
            if cutlass.const_expr(not self.ids_mode):  # noqa: SIM102
                if bid < M:
                    if warp == 0:
                        if cutlass.const_expr(self.image):
                            _route_row2(
                                logits,
                                bias,
                                ldg_i64(maps + 1024 + 9 * 8),
                                xgs,
                                Int32(ldg_i64(maps + 1024 + 10 * 8)),
                                ids,
                                wts,
                                bid,
                                lane,
                                self.experts,
                                self.top_k,
                                True,
                                False,
                            )
                        else:
                            _route_row(
                                logits,
                                bias,
                                ids,
                                wts,
                                bid,
                                lane,
                                self.experts,
                                self.top_k,
                            )
                    if warp < NCW:
                        named_barrier(1, NCT)
                        if tid == 0:
                            fence_gpu()
                            atom_add_relaxed(cnt + C_BAR * 4, 1)
            if bid < M:  # noqa: SIM102
                if tid < 160:
                    self._quant_row(x, bid, tid, qw, qs)
            n_rc = self._front_entry(
                ids,
                cnt,
                M,
                tid,
                warp,
                lane,
                tab,
                s_ids,
                s_mask,
                s_cnt,
                s_off,
                s_rcb,
                s_tot,
                nsh_rc,
            )
            if cutlass.const_expr(self.shared):
                if tid < nsh_rc:
                    rem = M - 32 * tid
                    if rem > 32:
                        rem = Int32(32)
                    sts32(tab + tid * 4, Uint32(E_SH))
                    sts32(tab + (RC_CAP + tid) * 4, Uint32(32 * tid))
                    sts32(tab + (2 * RC_CAP + tid) * 4, Uint32(rem))
                    sts32(tab + (3 * RC_CAP + tid) * 4, Uint32(0xFFFFFFFF))
                cute.arch.sync_threads()
            if bid < M:
                if tid < 160:
                    for k in cutlass.range_constexpr(self.top_k):
                        ex, rank, ent = self._rank_slot(
                            bid, Int32(k), s_ids, s_mask, s_off
                        )
                        self._store_xrow(xg, xsg, ent, tid, qw, qs)
                        if tid == k:
                            stg_i32(
                                rec + Int64(REC_POS + bid * self.top_k + k) * 4, ent
                            )
                if warp < NCW:
                    self._publish_row(cnt, tid)
        nchs = self.nchs_d if self.dense else NCHS_ENTRY
        tsh = nsh_rc * NPAIR * 32 + nchs  # packed (shared tiles, their k-chunks)
        n_tiles = (n_rc - nsh_rc) * NPAIR
        tgt = self.chunk_target * nblk
        NCH = Int32(20)
        if n_tiles * 10 >= tgt:
            NCH = Int32(10)
        if n_tiles * 5 >= tgt:
            NCH = Int32(5)
        if n_tiles * 4 >= tgt:
            NCH = Int32(4)
        if n_tiles * 2 >= tgt:
            NCH = Int32(2)
        if n_tiles >= tgt:
            NCH = Int32(1)
        if cutlass.const_expr(self.cluster):
            NCH = Int32(2)
        UPC = SPT // NCH
        n_chunks = nsh_rc * NPAIR * nchs + n_tiles * NCH
        # entry mode: the GEMM1 bar counts row CTAs; dense too (logits mode
        # already waited in the front)
        if warp == PROD:  # noqa: SIM102
            if lane == 0:
                btgt = M
                if cutlass.const_expr(not self.dense and not self.ids_mode):
                    btgt = 2 * M
                self._produce1(
                    tw13,
                    ts13,
                    sw13,
                    ss13,
                    xg,
                    xsg,
                    xgs,
                    xsgs,
                    cnt,
                    btgt,
                    n_chunks,
                    NCH,
                    UPC,
                    sbase,
                    full,
                    empty,
                    tab,
                    nblk,
                    bid,
                    tsh,
                    hb,
                    pre,
                )
                stamp = ldg_cg_i32(cnt + C_GEN * 4) + 1
                self._produce2(
                    tw2,
                    ts2,
                    sw2,
                    ss2,
                    iq,
                    isc,
                    iqs,
                    iscs,
                    cnt,
                    sync,
                    stamp,
                    n_rc,
                    sbase,
                    full2,
                    empty2,
                    tab,
                    nblk,
                    bid,
                )
        if warp == SIG:
            btgt = M
            if cutlass.const_expr(not self.dense and not self.ids_mode):
                btgt = 2 * M
            self._b_loop(
                x, M, xg, xsg, xgs, xsgs, cnt, btgt, NCH, tsh, sbase, full, hb, tab, bid
            )
            if lane == 0:
                stamp = ldg_cg_i32(cnt + C_GEN * 4) + 1
                self._signal(
                    sync, stamp, NCH, tsh, s_done, s_fin, s_fdone, clist, flist
                )
        if warp < NCW:
            blk = warp // 4
            R = warp % 4
            g = lane // 4
            t = lane % 4
            acc = cute.make_rmem_tensor((16,), Float32)
            d = cute.make_rmem_tensor((32,), Float32)
            af = cute.make_rmem_tensor((8,), Uint32)
            stage = Int32(0)
            phase = Int32(0)
            seq = Int32(0)
            running = Int32(1)
            rank = bid % 2
            cph = Int32(0)
            if cutlass.const_expr(self.cluster):  # noqa: SIM102
                if rank == 0:  # noqa: SIM102
                    if tid == 0:  # the slot is free once this CTA's front is done
                        mbar_arrive_cluster(mapa(cfree, 1))
            while running == 1:
                mbar_wait(full + stage * 8, phase)
                ch = Int32(lds32(sbase + stage * SB1 + HDR1))
                if ch >= n_chunks:
                    running = Int32(0)
                    cute.arch.sync_warp()
                    if lane == 0:
                        mbar_arrive(empty + stage * 8)
                else:
                    tl, hq = _ch_tile(ch, NCH, tsh)
                    rc = tl // NPAIR
                    e = Int32(lds32(tab + rc * 4))
                    ncol = Int32(lds32(tab + (2 * RC_CAP + rc) * 4))
                    mask = lds32(tab + (3 * RC_CAP + rc) * 4)
                    G = (ncol + 7) // 8
                    f13 = Float32(1.0)
                    ush = SH_U1 // (tsh % 32)
                    if e == E_SH:
                        if G == 1:
                            stage, phase = self._consume1_sh(
                                8,
                                ush,
                                stage,
                                phase,
                                sbase,
                                full,
                                empty,
                                blk,
                                R,
                                lane,
                                acc,
                                d,
                            )
                        elif G == 2:
                            stage, phase = self._consume1_sh(
                                16,
                                ush,
                                stage,
                                phase,
                                sbase,
                                full,
                                empty,
                                blk,
                                R,
                                lane,
                                acc,
                                d,
                            )
                        elif G == 3:
                            stage, phase = self._consume1_sh(
                                24,
                                ush,
                                stage,
                                phase,
                                sbase,
                                full,
                                empty,
                                blk,
                                R,
                                lane,
                                acc,
                                d,
                            )
                        else:
                            stage, phase = self._consume1_sh(
                                32,
                                ush,
                                stage,
                                phase,
                                sbase,
                                full,
                                empty,
                                blk,
                                R,
                                lane,
                                acc,
                                d,
                            )
                    else:
                        f13 = ldg_f32(g13 + _i64(e) * 4) * Float32(64.0)
                    if e == E_SH:
                        pass
                    elif G == 1:
                        stage, phase = self._consume1(
                            8,
                            UPC,
                            stage,
                            phase,
                            sbase,
                            full,
                            empty,
                            blk,
                            R,
                            lane,
                            acc,
                            d,
                            af,
                        )
                    elif G == 2:
                        stage, phase = self._consume1(
                            16,
                            UPC,
                            stage,
                            phase,
                            sbase,
                            full,
                            empty,
                            blk,
                            R,
                            lane,
                            acc,
                            d,
                            af,
                        )
                    elif G == 3:
                        stage, phase = self._consume1(
                            24,
                            UPC,
                            stage,
                            phase,
                            sbase,
                            full,
                            empty,
                            blk,
                            R,
                            lane,
                            acc,
                            d,
                            af,
                        )
                    else:
                        stage, phase = self._consume1(
                            32,
                            UPC,
                            stage,
                            phase,
                            sbase,
                            full,
                            empty,
                            blk,
                            R,
                            lane,
                            acc,
                            d,
                            af,
                        )
                    r0 = blk * 64 + 16 * R + g
                    dsm = Int32(0)
                    if cutlass.const_expr(self.cluster):  # noqa: SIM102
                        if e != E_SH:
                            dsm = Int32(1)
                            cph = self._cluster_tile(
                                tl,
                                hq,
                                rc,
                                e,
                                ncol,
                                mask,
                                f13,
                                acc,
                                r0,
                                t,
                                lane,
                                warp,
                                tid,
                                cph,
                                slot,
                                cfull,
                                cfree,
                                tab,
                                g2,
                                iq,
                                isc,
                                sync,
                            )
                    Gs = G
                    if dsm == 1:
                        Gs = Int32(0)
                    for j in cutlass.range_constexpr(4):
                        if j < Gs:
                            for i2 in cutlass.range_constexpr(2):
                                cc = 8 * j + 2 * t + i2
                                if cc < ncol:  # noqa: SIM102
                                    if ((mask >> Uint32(cc)) & Uint32(1)) != Uint32(0):
                                        a = part + ((_i64(ch) * 32 + cc) * 128 + r0) * 4
                                        stg_f32(a, acc[4 * j + i2] * f13)
                                        stg_f32(a + 32, acc[4 * j + 2 + i2] * f13)
                    if dsm == 0:
                        if tid == 0:
                            sts32(clist + (seq % CL) * 4, Uint32(ch))
                        cute.arch.sync_warp()
                        if lane == 0:
                            smem_add_release(s_done + warp * 4, 1)
                        seq = seq + 1
                # (the stage pointer/phase advanced inside _consume1)
            if tid == 0:
                sts32(clist + (seq % CL) * 4, Uint32(END))
            cute.arch.sync_warp()
            if lane == 0:
                smem_add_release(s_done + warp * 4, 1)
            # ---------------- SwiGLU of the tiles this CTA completed
            if tid == 0:
                while (lds_acquire(s_fin) & FIN_FLAG) == 0:
                    nanosleep(20)
            named_barrier(1, NCT)
            fin_n = lds_acquire(s_fin) & (FIN_FLAG - 1)
            j = Int32(0)
            for i in cutlass.range(fin_n, unroll=1):
                tile = Int32(lds32(flist + (i % CL) * 4))
                rcf = tile // NPAIR
                ncf = Int32(lds32(tab + (2 * RC_CAP + rcf) * 4))
                mkf = lds32(tab + (3 * RC_CAP + rcf) * 4)
                if _tile_nch(tile, NCH, tsh) <= 5:
                    # every warp takes 4 of the tile's (<= 32) columns
                    self._swiglu_one(
                        4,
                        5,
                        tile,
                        warp,
                        ncf,
                        mkf,
                        NCH,
                        tsh,
                        tab,
                        g2,
                        part,
                        iq,
                        isc,
                        iqs,
                        iscs,
                        lane,
                    )
                else:
                    for cc in cutlass.range(ncf, unroll=1):
                        if ((mkf >> Uint32(cc)) & Uint32(1)) != Uint32(0):
                            if j % NCW == warp:
                                self._swiglu_one(
                                    1,
                                    20,
                                    tile,
                                    cc,
                                    cc + 1,
                                    mkf,
                                    NCH,
                                    tsh,
                                    tab,
                                    g2,
                                    part,
                                    iq,
                                    isc,
                                    iqs,
                                    iscs,
                                    lane,
                                )
                            j = j + 1
            fence_proxy_async_global()
            cute.arch.sync_warp()
            if lane == 0:
                smem_add_release(s_fdone + warp * 4, 1)
            # ---------------- GEMM2 consumers
            n_items = n_rc * NBP2
            stage = Int32(0)
            phase = Int32(0)
            running = Int32(1)
            while running == 1:
                mbar_wait(full2 + stage * 8, phase)
                it = Int32(lds32(sbase + stage * SB2 + HDR2))
                if it >= n_items:
                    running = Int32(0)
                    cute.arch.sync_warp()
                    if lane == 0:
                        mbar_arrive(empty2 + stage * 8)
                else:
                    rc = it // NBP2
                    nbp = it % NBP2
                    off = Int32(lds32(tab + (RC_CAP + rc) * 4))
                    ncol = Int32(lds32(tab + (2 * RC_CAP + rc) * 4))
                    mask = lds32(tab + (3 * RC_CAP + rc) * 4)
                    G = (ncol + 7) // 8
                    n0 = 128 * nbp + 64 * blk + 16 * R + g
                    e = Int32(lds32(tab + rc * 4))
                    if e == E_SH:
                        if G == 1:
                            stage, phase = self._consume2_sh(
                                8,
                                stage,
                                phase,
                                sbase,
                                full2,
                                empty2,
                                blk,
                                R,
                                lane,
                                acc,
                                d,
                            )
                            self._store_rows(8, acc, sh, off, ncol, mask, n0, lane)
                        elif G == 2:
                            stage, phase = self._consume2_sh(
                                16,
                                stage,
                                phase,
                                sbase,
                                full2,
                                empty2,
                                blk,
                                R,
                                lane,
                                acc,
                                d,
                            )
                            self._store_rows(16, acc, sh, off, ncol, mask, n0, lane)
                        elif G == 3:
                            stage, phase = self._consume2_sh(
                                24,
                                stage,
                                phase,
                                sbase,
                                full2,
                                empty2,
                                blk,
                                R,
                                lane,
                                acc,
                                d,
                            )
                            self._store_rows(24, acc, sh, off, ncol, mask, n0, lane)
                        else:
                            stage, phase = self._consume2_sh(
                                32,
                                stage,
                                phase,
                                sbase,
                                full2,
                                empty2,
                                blk,
                                R,
                                lane,
                                acc,
                                d,
                            )
                            self._store_rows(32, acc, sh, off, ncol, mask, n0, lane)
                    elif G == 1:
                        stage, phase = self._consume2(
                            8,
                            stage,
                            phase,
                            sbase,
                            full2,
                            empty2,
                            blk,
                            R,
                            lane,
                            acc,
                            d,
                            af,
                        )
                        self._store_rows(8, acc, y, off, ncol, mask, n0, lane)
                    elif G == 2:
                        stage, phase = self._consume2(
                            16,
                            stage,
                            phase,
                            sbase,
                            full2,
                            empty2,
                            blk,
                            R,
                            lane,
                            acc,
                            d,
                            af,
                        )
                        self._store_rows(16, acc, y, off, ncol, mask, n0, lane)
                    elif G == 3:
                        stage, phase = self._consume2(
                            24,
                            stage,
                            phase,
                            sbase,
                            full2,
                            empty2,
                            blk,
                            R,
                            lane,
                            acc,
                            d,
                            af,
                        )
                        self._store_rows(24, acc, y, off, ncol, mask, n0, lane)
                    else:
                        stage, phase = self._consume2(
                            32,
                            stage,
                            phase,
                            sbase,
                            full2,
                            empty2,
                            blk,
                            R,
                            lane,
                            acc,
                            d,
                            af,
                        )
                        self._store_rows(32, acc, y, off, ncol, mask, n0, lane)
            griddep_launch()
        cute.arch.sync_threads()
        if tid == 0:
            old = atom_add_acqrel_gpu(cnt + C_EXIT * 4, 1)
            last = Uint32(0)
            if old == nblk - 1:
                last = Uint32(1)
                gen = ldg_cg_i32(cnt + C_GEN * 4)
                for s in cutlass.range_constexpr(N_RESET):
                    stg_i32(cnt + s * 4, 0)
                stg_i32(cnt + C_GEN * 4, gen + 1)
            sts32(s_fin, last)
        cute.arch.sync_threads()
        if lds32(s_fin) == Uint32(1):
            for i in cutlass.range(tid, n_rc, NTHR, unroll=1):
                stg_i32(sync + _i64(S_READY + i) * 4, 0)
            if tid == 0:
                stg_i32(sync + _i64(S_ALL) * 4, 0)
        if cutlass.const_expr(self.cluster):
            cluster_sync_all()
