# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, SIM102
"""The fused MiniMax-M3 MoE engine for 1..8 decode tokens on one TP rank, one CuTe DSL kernel (m3_moe_kernel).

Routing (MiniMax2 top-4, or routes polled from the post), FC1 (NVFP4 x NVFP4, SwiGLU-OAI, NVFP4 requantization), FC2,
the MXFP8 shared expert on extra CTAs, the combine and the push into the next layer's all-reduce slots, in one launch.
The routed part follows the TRTLLM-Gen fused MoE's math. The build is configured by M3_CONFIG (see _cfg below);
fc1_per_token selects the per-token FC1 scale (mode Q)."""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op

try:
    import cutlass.experimental.cuda as cuda
    from cutlass.experimental import primitives as prims
except ImportError as _err:  # pragma: no cover - depends on the installed DSL
    raise ImportError(
        "the MiniMax-M3 fused MoE engine needs cutlass.experimental (cuda, primitives: tcgen05_mma_block_scale, "
        "Tcgen05MMABlockScale, ...) from nvidia-cutlass-dsl; vLLM pins nvidia-cutlass-dsl==4.8.0 "
        "(requirements/cuda.txt). Install that version or run the stock MoE."
    ) from _err

_CFG = globals().get("M3_CONFIG") or {}


def _cfg(key: str, default):
    return type(default)(_CFG.get(key, default))


# =============================================================================
# Problem shape (trace-time constants).
# =============================================================================
H = 6144
# routed intermediate per rank: 768 at TP4, 384 at TP8 (FC2 K = 1.5 256-K tiles: the
# last tile's A box is half out of bounds, zero-filled by TMA, and only its first
# KB2_LAST K-64 MMAs / scale atoms are issued; no padding, no extra bytes)
I_TP = _cfg("i_tp", 768)
TWO_I = 2 * I_TP
assert I_TP % 128 == 0 and 128 <= I_TP <= 1536, I_TP  # TP2 (1536) admitted
E_LOCAL = 128
TOP_K = 4
ROUTE_SCALE = 2.0
SUM_EPS = 1e-20
M_MAX = _cfg("m_max", 1)
# Routed CTAs = live SMs - shared CTAs, always set by the host (build_config); 1 only
# lets the module import without a config.
NCTA = _cfg("num_ctas", 1)
FUSED_ROUTER = bool(_cfg("fused_router", 0))
NROUTE = E_LOCAL if FUSED_ROUTER else 0  # experts whose logits the router GEMV computes
NGEMV = (NROUTE + NCTA - 1) // NCTA  # experts per CTA: bx, bx + NCTA, ..
LAG = _cfg("lag", 2)  # phase p runs FC1 of group p and FC2 of group p - LAG
G_CAP = TOP_K * M_MAX  # distinct experts per call
NG_CAP = M_MAX  # groups of 4 slots
XW = _cfg(
    "xw", 5 if M_MAX > 4 else (4 if M_MAX > 2 else 1)
)  # FC2 B builder warps: 5, then 8 ..
THREADS = 256 + 32 * (XW - 1)
GEMV_THREADS = 256
# ROUTER_VLLM: the router GEMV in vLLM ops.fp32_router_gemm's order (csrc/libtorch_stable/fp32_router_gemm.cu at
# E 128, H 6144, M <= 10: 384 threads per token, 8 consecutive K per thread per 3072-K pass, fma in K order, warp
# xor tree 16..1, then 0 + warp 0 + .. + warp 11), so the logits are bit-identical to the stock router's. 384 virtual
# threads on the 256 GEMV threads: thread i runs virtual i and, for i < 128, virtual 256 + i.
ROUTER_VLLM = bool(_cfg("router_vllm", 0))
# SILU_VLLM: the shared expert's activation as vLLM torch.ops._C.silu_and_mul_with_clamp computes it
# (csrc/libtorch_stable/activation_kernels.cu, packed act_first + clamp): s = bf16(g / (1 + expf(-g * alpha))),
# h = bf16(s * (clamp(up) + beta)), expf / IEEE div without fast-math. The default form (one fused fp32 product,
# ex2.approx, div.full) can differ from it in the last bf16 bit of h.
SILU_VLLM = bool(_cfg("silu_vllm", 0))
# SHARED: the MoE's shared expert (MXFP8 gate_up [1536, 6144] -> SwiGLU-OAI -> MXFP8 down [6144, 768]) on NSH more
# CTAs (warps 0-3), its output added in finalize: out = bf16(shared + routed) (shared_output.add_(routed_output)).
# The math is mxfp8_low_m_gemm's (native MXFP8 activation quant, mma.sync m16n8k32 e4m3 per 32-K block, block scales
# as one fma, its partial-sum order, bf16) and the Triton silu_and_mul's, operation for operation. Shared CTA s owns
# intermediate positions 32 s .. 32 s + 31 (gate and up rows: MMA tiles = warps 0-3) and down rows 256 s .. + 255;
# 16-row x 768-byte weight chunks (+ their scale atoms) stream through NBW buffers per warp, the first NBW before
# the grid dependency. h crosses the shared CTAs, y_sh reaches the routed CTAs' finalize (both tagged words).
SHARED = bool(_cfg("shared", 0))
# LATE_TRIGGER: release the dependents at teardown, not after the grid dependency. An early release lets the next
# kernels' CTAs (the all-reduce, then the attention it releases at entry) become resident during this kernel, packed
# onto the SMs it leaves free, where the attention then runs on a fraction of the GPU.
LATE_TRIGGER = bool(_cfg("late_trigger", 1))
# EARLY_TRIG: with the push on, release the dependents once routing is done, so the prepushed front (no grid
# wait; it polls the pushed slots) lands on SMs as the shared CTAs retire
EARLY_TRIG = bool(_cfg("early_trig", 0))
NOFENCE1 = bool(
    _cfg("nofence1", 0)
)  # FC1 stages are all bulk-async with sfa_bulk: no proxy fence after ab_full
SFA_ARR1 = bool(
    _cfg("sfa_arr1", 0)
)  # (SFA_BULK) one count-32 arrive by the issuing lane, not 32 noinc arrivals + a sync
# AR_PRE: the finalize's all-reduce push uses the call parities and descriptor fields loaded while the last group's
# FC2 runs (not two dependent global loads per push on the kernel's tail).
AR_PRE = bool(_cfg("ar_pre", 1))
# AR_G3: the finalize pushes into boundary-exchange's unicast boundary slots (slot
# bufs[r] + ((call % 3 * W + rank) * AR_ROWS + row) * H, every rank's buffer, -0
# canonicalised) instead of the multicast set A. arp = [1, &bufs[0..W-1] (int64 device
# array), &flags (call at +0), rank, W, AR_ROWS, 0, 0].
AR_G3 = bool(_cfg("ar_g3", 0))
# PF_NEXT: at teardown the routed CTAs L2-prefetch the next layer's routing-independent bytes (shared expert,
# router weight) so its shared CTAs read them from L2. arp[6] = &table (int64 pairs ptr, bytes), arp[7] = pairs.
PF_NEXT = bool(_cfg("pf_next", 0))
# PF_FC2: once routing is done, the producer warp L2-prefetches this CTA's FC2 weight
# rows (TMA tensor prefetch) and scale atoms for every slot, so the FC2 phase reads L2
# instead of paying HBM latency per tile
PF_FC2 = bool(_cfg("pf_fc2", 0))
B1_QB = _cfg("b1_qb", 0)
# SF2_W6: the FC2 stages' scale gather (16 x 4-byte cp.async per lane, the producer's
# per-stage issue cost after roll_fc2) moves to warp 8 (idle in B2IMG mode after routing
# / early issue); the producer only issues the TMA boxes and posts (stage, g, blk, t2)
# into a QD-deep smem queue
SF2_W6 = bool(_cfg("sf2_w6", 0))
# PW2 (window 4): a second weights producer, warp 9 (idle once routing is done when
# B2IMG replaces the builders), issues every other FC1 stage; FC2 stages, the SF2 queue
# and the early-stage hand-off stay on warp 4
PW2 = bool(_cfg("pw2", 0))
PW2_WARP = 9
# PW2F (window 4): warp 9 owns the whole FC2 weight / SF2 stream on its own ring slice
# [S1R, STAGES) and issues it as early as its slots allow (FC2 weights do not depend on
# the FC1 output); warp 4 keeps all of FC1 on [0, S1R)
PW2F = bool(_cfg("pw2f", 0))
PW2F_S2 = _cfg("pw2f_s2", 3)
PWX = PW2 or PW2F
Q2D = 8
SF2_WARP = 8
_B1_NBLK = (
    M_MAX * (H // 16) // 128
)  # (EPI_THREADS) the B1 build's 16-element blocks per thread
B1_QB_EFF = (
    B1_QB
    if (B1_QB > 0 and _B1_NBLK % B1_QB == 0)
    else (3 if _B1_NBLK % 3 == 0 else (2 if _B1_NBLK % 2 == 0 else 1))
)
# ROLL_FC2: the FC2 k-tile loops of the producer and MMA warps rolled (one code copy,
# instruction-cache hot) instead of KT2 unrolled copies that each run once per block and
# group (TP2: KT2 = 6, all tiles full)
ROLL_FC2 = bool(_cfg("roll_fc2", 0))
assert I_TP % 256 == 0, "FC2 k-tile loops take a dynamic t2 (full 256-K tiles only)"
PF_CHUNK = _cfg("pf_chunk", 16384)
PF_WHERE = _cfg(
    "pf_where", 1
)  # 0 routed CTAs at teardown, 1 shared CTAs when their work is done
AR_W = _cfg("ar_w", 2)
AR_ROWS = _cfg("ar_rows", 48)
assert not AR_G3 or (AR_PRE and AR_W in (2, 4) and M_MAX <= AR_ROWS)
# AR_V3: push into boundary-exchange prx slots (the boundary exchange's push_chunk
# layout, _exchange.py): flags per (row, H / AR_CL slice) and AR_ND destinations (world
# unicast peers, or 1 = the multicast mapping). arp = [1, &peers[0..ND-1] (int64 device
# array), flags, rank, world, maxt, 0, 0].
AR_V3 = bool(_cfg("ar_v3", 0))
AR_CL = _cfg("ar_cl", 8)
AR_ND = _cfg("ar_nd", 1)
assert not AR_V3 or (AR_PRE and not AR_G3 and 1 <= AR_ND <= 16)
# XSTAGE (fused router): the hidden states land in the (still idle) ring by one bulk
# copy per token right after the grid dependency; the NVFP4 quantization of B1 reads
# them from shared memory (one HBM / L2 round trip, not ~8).
XSTAGE = bool(_cfg("xstage", 1)) and bool(_cfg("fused_router", 0))
# EXT_ROUTE (router hoist): the routing comes from outside, read by the routing warps
# before griddepcontrol.wait; no router GEMV, logit exchange or top-k. 1 = poll the
# post's tagged route words (route_lines, int64 [ROUTE_REPL][ROUTE_MAXT][16], words 0..3
# = tag << 32 | id << 16 | bf16 w, tag = epochs[bx] + 1, replica bx % ROUTE_REPL; passed
# in the lgbuf slot), 2 = a static table (rt_tab, int32 [t][4] = id << 16 | bf16 w;
# passed in the x32 slot). 0 = the base program.
EXT_ROUTE = _cfg("ext_route", 0)
ROUTE_REPL = _cfg("route_repl", 16)
ROUTE_POLL_NS = _cfg("route_poll_ns", 64)
ROUTE_MAXT = _cfg("route_maxt", M_MAX)  # rows per route-line replica (the post's maxt)
ROUTER_GEMV = FUSED_ROUTER and not EXT_ROUTE
# B1 from the post's NVFP4 bytes (non-fused _build_b1, post-wait) instead of quantizing x here
B1_X4 = bool(_cfg("b1_x4", 0))
# fc1_scale="per_token" (mode Q): the post's x4 is per-token quantized; its fp32 s_t[M]
# sits in row max(M, 16) of the x4 scale buffer (word S_T_WORD) and the FC1 epilogue
# multiplies each token's sums by s_t. FC2 stays static.
FC1_PER_TOKEN = bool(_cfg("fc1_per_token", 0) or _cfg("pt5", 0))
PT5 = FC1_PER_TOKEN  # compat: route_warp_tp2_fc1_per_token_host.make_moe_pt checks the module's PT5
assert not _cfg("pt5_red", 0), (
    "M3_CONFIG pt5_red must be 0: the engine has no pt5_red build"
)
S_T_WORD = max(M_MAX, 16) * (H // 16) // 4
assert not FC1_PER_TOKEN or B1_X4, (
    "fc1_per_token needs b1_x4 (s_t comes with the post's x4)"
)
assert not FC1_PER_TOKEN or M_MAX <= 8, "fc1_per_token: s_pts holds 8 tokens"
assert EXT_ROUTE in (0, 1, 2), EXT_ROUTE
NSH = (
    _cfg("nsh", 24) if SHARED else 0
)  # 48 at TP2 (32 positions, 128 down rows per CTA)
# the shared intermediate per rank: 768 (TP4), 384 (TP8). A shared CTA owns SH_ROWS
# intermediate positions: its gate_up MMA tile is SH_SPLIT K slices x (SH_ROWS gate +
# SH_ROWS up rows) = 128 rows (TP4: 2 x (32 + 32), the order of m3_mx_gemv split 2; TP8:
# 4 x (16 + 16), m3_mx_gemv split 4), and 6144 / NSH = 256 down rows over the full K.
SH_I = _cfg("sh_i", 768)
SH_ROWS = SH_I // (NSH or 24)
SH_DROWS = H // (NSH or 24)  # down rows per shared CTA
SH_DT = SH_DROWS // 128  # down tiles per shared CTA
assert SH_DROWS % 128 == 0 and SH_I % (NSH or 24) == 0
# gate_up K slices (= m3_mx_gemv's cluster ranks, contiguous K ranges)
SH_SPLIT = 128 // (2 * SH_ROWS)
SH_NB = 8 * SH_SPLIT  # gate_up B rows (8 q + t: token t's slice-q k-tile) = its MMA N
SH_KB = H // 32  # gate_up K blocks
SH_DKB = SH_I // 32  # down K blocks
SH_DKT = SH_I // 128  # down k-tiles
SH_GT = SH_KB // 4 // SH_SPLIT  # gate_up k-tiles (128 K) per slice
SH_STAGES_GU = SH_GT
SH_STAGES = SH_GT + SH_DT * SH_DKT  # + SH_DT down tiles x SH_DKT k-tiles
SHX_BAR = 6
# SH_PRETILE: the shared weights come pre-tiled (op._pretile_shared): per shared CTA its SH_STAGES stages, each the
# 16 KB shared-memory image of its 128-row MMA tile (128-byte swizzle) followed by the stage's 512-byte instruction
# scale atom, so a stage is two contiguous bulk copies (not 128 row segments of 128 bytes) and needs no splicing.
SH_PRETILE = bool(
    _cfg("sh_pretile", 1 if M_MAX <= 2 or SH_SPLIT != 2 else 0)
)  # (in-chain: a win at M <= 2, not 4..8)
# SH_ARR1 (with SH_PRETILE: the MMA warp splices nothing, its generic reads of a slot
# are none): the shared ring's empty barrier completes on the MMAs' commit alone (count
# 1), not commit + 32 lane arrivals
SH_ARR1 = bool(_cfg("sh_arr1", 0))
# SH_LATE: the shared CTAs issue nothing (no ring TMA, no L2 prefetch) before griddepcontrol.wait; the first SH_NST
# stages go out right after it (in-chain the engine is resident while the post / o_proj still run)
SH_LATE = bool(_cfg("sh_late", 0))
# SH_TRIG: with the push on, the shared CTAs release the dependents right after their own grid wait (as the
# routed CTAs' EARLY_TRIG does after ROUTE_BAR), so the next front's launch is not gated by the last shared exit
SH_TRIG = bool(_cfg("sh_trig", 0))
assert not SH_TRIG or (SHARED and LATE_TRIGGER and EARLY_TRIG), (
    "sh_trig: shared CTAs, late_trigger=1, early_trig=1"
)
if SHARED:
    assert SH_I in (384, 768, 1536), (
        SH_I
    )  # 1536 = TP2 (one K slice, 64 positions per CTA)
    assert not SH_ARR1 or SH_PRETILE, (
        "sh_arr1: only with the pre-tiled (pre-spliced) shared stages"
    )
    assert SH_PRETILE or SH_SPLIT == 2, (
        "the un-pretiled shared stream is the 32-row (TP4) layout only"
    )
SH_STAGE_BYTES = 16384 + 512
MMA_M = 128
KT_BYTES = 128
INC_K = 2
NUM_KB = 4
# FC1
N1 = (
    16 if M_MAX <= 4 else 32
)  # B rows: QS q + t (token t at the stage's k-tile 4 j + q)
QS = N1 // 4
R1 = 32
C1 = 4
U1 = TWO_I // R1  # 48 units per slot
K1_TILES = H // 256
S1 = K1_TILES // C1  # 6 stages per unit
NUM_BYTES_A = MMA_M * KT_BYTES
NUM_BYTES_BOX1 = R1 * KT_BYTES
SF1_KA = H // 64
SF1_RB = TWO_I // 128
NUM_BYTES_SFA_RAW = 16 * 512
NUM_WORDS_SFA_RAW = NUM_BYTES_SFA_RAW // 4
NUM_BYTES_SFA = NUM_KB * 512
NUM_WORDS_SFA = NUM_BYTES_SFA // 4
NUM_BYTES_B1 = N1 * KT_BYTES
NUM_SFB1_ATOMS = S1 * NUM_KB
# FC2
R2 = 32
NB2 = H // R2  # 192
KT2 = (I_TP + 255) // 256  # 3 (TP4), 2 (TP8)
# K-64 instructions of the last FC2 k-tile: 4 (TP4), 2 (TP8)
KB2_LAST = (I_TP - 256 * (KT2 - 1)) // 64


def _nkb2(t2: int) -> int:
    """K-64 MMA instructions (= scale atoms) of FC2 k-tile t2."""
    return NUM_KB if t2 < KT2 - 1 else KB2_LAST


SF2_KA = I_TP // 64
SF2_RB = H // 128
N2 = max(8, (4 * M_MAX + 7) // 8 * 8)  # B rows j M + t
NUM_BYTES_B2 = N2 * KT_BYTES
NB2_CAP = _cfg("nb2_cap", 2)
UNITS_ROW = TWO_I // R1  # 48 records per (slot, token) row
H_ROW = UNITS_ROW * 16
# HB_REPL: the tagged FC1 -> FC2 records written to HB_REPL copies (HB_STRIDE int32 words apart); CTA bx polls copy
# bx % HB_REPL. Every routed CTA polls every record, so with one copy the hand-off latency depends on where the few
# hot lines land in L2; copies spread the polls.
HB_REPL = _cfg("hb_repl", 1)
assert 1 <= HB_REPL <= 32, HB_REPL  # (copy i is written by lane i)
HB_STRIDE = (G_CAP * M_MAX * UNITS_ROW * 4 + 255) // 256 * 256 + 32
PARTIAL_EMPTY = -1
EPI_THREADS = 128
EPI_BAR = 1
ROUTE_BAR = 2
GEMV_BAR = 3
B2_BAR = 5
TD_STRIDE = 128  # tdbg columns per CTA (64+ = per-stage MMA issue time / count)
HANG_CAP = 0x7FFFFFFF  # global polls are unbounded
TOPK_BAR = 4
B1_AFTER_TOPK = bool(
    _cfg("b1_after_topk", 0)
)  # (experiment) the B1 build waits for the routing warps' top-k
TOPK2_BAR = 7
EARLY0 = bool(
    _cfg("early0", 1)
)  # warp 4 streams group 0 (token 0's experts) right after its own top-k
EARLY0_N = _cfg("early0_n", 99)  # stages issued early (capped at STAGES)
# routing warps: 4..7, then the FC2 B builder warps 8.. (idle until the routing is done) up to one warp per token
RW_EXTRA = (
    max(0, min(M_MAX - 4, (THREADS // 32) - 8)) if bool(_cfg("rw_extra", 1)) else 0
)
NRW0 = (
    min(4, M_MAX) if bool(_cfg("rw_trim", 1)) else 4
)  # warps 4.. with a token (no idle pollers in the barrier)
NRW = NRW0 + RW_EXTRA
LG_REPL = _cfg("lg_repl", 32 if M_MAX > 4 else 128)
GEMV_SMEM = bool(
    _cfg("gemv_smem", 1)
)  # (XSTAGE) the router GEMV reads the staged hidden states from smem
LG_STRIDE = M_MAX * E_LOCAL + 32
POLL_NS = _cfg("poll_ns", 150)
B2_POLL_NS = _cfg("b2_poll_ns", 200)
# B2IMG (needs SFA_CP: warp 6 is free): the FC1 epilogue writes each finished unit's FP4
# bytes and scale byte straight into a per-group global image laid out as the FC2 B tile
# + its scale atoms (exactly what the builder warps assembled from the tagged records);
# per group, warp 6 joins the epilogue (fc1_done), fences the async proxy and does one
# release add of this CTA's finished units to the group counter; warp 5 acquires the
# counter (every group gains CNT_PER_CALL per call, CTA 0 pads absent slots), fences,
# and bulk-copies the image into the B2 buffer.
B2IMG = bool(_cfg("b2img", 0)) and bool(_cfg("sfa_cp", 0))
IMG_B2 = KT2 * NUM_BYTES_B2
IMG_SF = KT2 * NUM_KB * 512
IMG_BYTES = IMG_B2 + IMG_SF
CNT_PER_CALL = 4 * (TWO_I // R1)
B2IMG_NS = _cfg("b2img_ns", 16)
# PAD_X (B2IMG): CTA 0's group-counter pad is issued by its B2 loader warp after the
# route barrier, not by routing warp 4 before it (the release add would hold CTA 0's
# routing behind every other CTA's)
PAD_X = bool(_cfg("pad_x", 0))  # (B2IMG) counter poll back-off
# B2_REPL (B2IMG, HB_REPL's idea applied to the counter): every routed CTA polls the one group-counter word,
# so the hand-off latency depends on where that hot line lands in L2. Releasers add to B2_REPL copies (lane r -> copy
# r, B2_CSTRIDE words apart); CTA bx polls copy bx % B2_REPL. B2_GS: words between group counters (8 = 4 groups per
# 128-byte line; 32 = one line each). Defaults (1, 8) are the base layout.
B2_REPL = _cfg("b2_repl", 1)
B2_GS = _cfg("b2_gs", 8)
# B2_SHARD (write side, the other half of the hot line): releasers add to shard bx %
# B2_SHARD of the counter; the B2 loader warp's lanes acquire one shard each and sum (a
# group is complete when the shards sum to the target)
B2_SHARD = _cfg("b2_shard", 1)
assert 1 <= B2_REPL <= 32 and 1 <= B2_SHARD <= 32 and B2_GS in (8, 16, 32), (
    B2_REPL,
    B2_SHARD,
    B2_GS,
)
assert B2_REPL == 1 or B2_SHARD == 1, "b2_repl and b2_shard are alternatives"
B2_NCOPY = max(B2_REPL, B2_SHARD)
B2_CSTRIDE = (NG_CAP * B2_GS + 8 + 63) // 64 * 64 + 32 if B2_NCOPY > 1 else 0
B2CNT_WORDS = (B2_NCOPY - 1) * B2_CSTRIDE + NG_CAP * B2_GS + 8
W_WARP = 4
X_WARP = 5
S_WARP = 6
MMA_WARP = 7
# at M = 4 (four routing warps, no extra ones) the idle FC2 B builder warp 8 issues the early stages instead
E0_W8 = (
    NRW0 == 4
    and RW_EXTRA == 0
    and THREADS // 32 > 8
    and I_TP > 768
    and bool(_cfg("e0_w8", 1))
)  # (hangs at TP4: )
E0_WARP = (
    8 if E0_W8 else MMA_WARP
)  # (M <= 3) the early stages are issued by the idle MMA warp, not by warp 4 (routing's critical path)
assert not PWX or (
    B2IMG and THREADS // 32 > PW2_WARP and PW2_WARP not in (E0_WARP, SF2_WARP)
), "pw2: needs B2IMG and an idle warp 9"
assert not (PW2 and PW2F), "pw2 / pw2f: one or the other"
assert not PW2F or bool(_cfg("sfa_cp", 0)), (
    "pw2f: SFA_CP (the raw-scale splicer walks one shared ring cursor)"
)
E1_N = (
    96 if PWX else 64
)  # (PW2) warp 9 also waits for the early stages before reading the skip count
E0_BAR = 8
E1_BAR = 9
E0_ARRIVE = bool(_cfg("e0_arrive", 1))
E0_OFFLOAD = EARLY0 and (NRW0 <= 3 or E0_W8) and bool(_cfg("e0_off", 1))
EVICT_FIRST = 0x12F0000000000000
_LOG2E = 1.4426950408889634
LD_POW2 = (
    1 if M_MAX == 1 else 2 if M_MAX == 2 else 4 if M_MAX <= 4 else 8
)  # tcgen05.ld .num covering M columns

# B2BUF: FC2 B image (smem tile + scale atoms in smem and TMEM) buffers. 2 = double-buffered (default); 1 frees
# KT2 * NUM_BYTES_B2 + KT2 * 2 KB of smem and KT2 * 16 TMEM columns for more ring stages (8 rows: 6 -> 8), at the cost
# of loading group g + 1's image only after group g's FC2 has drained.
NB2BUF = _cfg("b2buf", 2)
# reserve for the small smem arrays (barriers, plan, slots, queues): 6 KB default; the
# compiled usage is about 2.7 KB at 4 rows, so v5 trims it (cfg smem_slack, bytes) to
# fit one more 18 KB stage; dsl_check verifies
SMEM_SLACK = _cfg("smem_slack", 6 * 1024)
assert NB2BUF in (1, 2)
_FIXED_SMEM = (
    S1 * NUM_BYTES_B1
    + NB2BUF * KT2 * NUM_BYTES_B2
    + NB2BUF * KT2 * NUM_KB * 512
    + NB2_CAP * M_MAX * TOP_K * R2 * 4
    + 2 * M_MAX * C1 * R1 * 4
    + (M_MAX * (H // 16) if FUSED_ROUTER else 0)
    + SMEM_SLACK
)
# SFA_CP: the A scale words go by 4-byte cp.async straight into the instruction atom layout (no raw scale box, no
# splicer warp; the MMA warp tcgen05.cp's them), completing on the stage's full barrier: 18 KB stages instead of 26 KB
SFA_CP = bool(_cfg("sfa_cp", 0))
# SFA_BULK (needs SFA_CP): s1w is op's pre-spliced FC1 scale image; a stage's 2 KB of
# instruction atoms is one bulk copy completing the stage's full barrier (count 1, + 2
# KB tx)
SFA_BULK = bool(_cfg("sfa_bulk", 0)) and SFA_CP
assert not NOFENCE1 or SFA_BULK, (
    "nofence1 needs the bulk SFA path (every FC1 stage byte is async-proxy)"
)
assert not bool(_cfg("sf2_w6", 0)) or (
    SFA_CP and bool(_cfg("b2img", 0)) and THREADS // 32 > 8
), "sf2_w6 needs SFA_CP, B2IMG and warp 8"
_RAW_STAGE = 0 if SFA_CP else NUM_BYTES_SFA_RAW
_STAGE_SMEM = NUM_BYTES_A + _RAW_STAGE + NUM_BYTES_SFA
_STAGES_FIT = (227 * 1024 - _FIXED_SMEM - (128 if SHARED else 0)) // _STAGE_SMEM
ACC1_COL = 0
ACC2_COL = 2 * N1
SFB1_COL = ACC2_COL + 2 * N2
SFB2_COL = SFB1_COL + NUM_SFB1_ATOMS * 4
SFA1_COL = SFB2_COL + NB2BUF * KT2 * NUM_KB * 4
STAGES = min(_cfg("stages", 12 if SFA_CP else 8), _STAGES_FIT, (512 - SFA1_COL) // 16)
S1R = (
    STAGES - PW2F_S2 if PW2F else STAGES
)  # (PW2F) FC1 ring [0, S1R); FC2 ring [S1R, STAGES)
assert not PW2F or 2 <= S1R < STAGES, ("pw2f: ring split", STAGES, PW2F_S2)
TMEM_COLS = 512
# AR_IN: the engine is the post-attention boundary. The attention producer pushed its partial into boundary-exchange prx slots
# every routed CTA polls both ranks' slots, sums them, adds the residual and applies the Gemma RMSNorm
# into its x staging (no boundary kernel, no x copy). CTA 0 writes residual_out / normed x and releases the shared
# CTAs; at teardown the CTAs re-arm the slots and CTA 0 flips the call parities. arp[6] = &desc (int64: slot buffer,
# flags, rank, maxt, residual_in, residual_out, gamma, eps bits, counter, xready); arp[6] == 0 runs the x copy as before.
AR_IN = bool(_cfg("ar_in", 0))
AR_IN_W = _cfg("ar_in_w", 2)
AR_IN_CL = _cfg("ar_in_cl", 4)
SH_PF_ON = bool(_cfg("sh_pf", 1))
ARIN_NCC = H // 8  # 16-byte chunks per row
ARIN_J = (ARIN_NCC + THREADS - 1) // THREADS
ARIN_TC = M_MAX * ARIN_NCC
ARIN_PER = (ARIN_TC + NCTA - 1) // NCTA  # chunks per routed CTA in phase A
assert not AR_IN or (ARIN_PER <= THREADS and M_MAX % 4 == 0)
ARIN_EMPTY = -2147483648
assert not AR_IN or (XSTAGE and GEMV_SMEM and (not AR_V3 or AR_CL == AR_IN_CL))
# ext_route: the ring must be free at entry (staged x aliases it), CTA 0's counter pad
# runs after the wait (pad_x), the early warps' path to the routing block has no trigger
# (late_trigger) and no AR_IN prologue
assert not EXT_ROUTE or (
    FUSED_ROUTER and not XSTAGE and PAD_X and LATE_TRIGGER and not AR_IN
), "ext_route needs fused_router=1, xstage=0, pad_x=1, late_trigger=1, ar_in=0"
assert not EXT_ROUTE or (1 <= ROUTE_REPL <= 32 and ROUTE_MAXT >= M_MAX), (
    ROUTE_REPL,
    ROUTE_MAXT,
)
assert not B1_X4 or (EXT_ROUTE and FUSED_ROUTER and not XSTAGE), (
    "b1_x4 needs ext_route, fused_router=1, xstage=0"
)
# the early (pre-wait) warps never include the finalize / AR-flag warps 0-3
assert not EXT_ROUTE or not (
    (
        set(range(W_WARP, W_WARP + NRW0))
        | set(range(8, 8 + RW_EXTRA))
        | ({E0_WARP} if E0_OFFLOAD else set())
    )
    & {0, 1, 2, 3}
), (
    "ext_route early warps overlap warps 0-3",
    W_WARP,
    NRW0,
    RW_EXTRA,
    E0_WARP,
    E0_OFFLOAD,
)
assert 1 <= M_MAX <= 8, M_MAX
assert STAGES >= 3, (STAGES, _STAGES_FIT, SFA1_COL)
assert not XSTAGE or M_MAX * H * 2 <= STAGES * (
    NUM_BYTES_A + _RAW_STAGE + NUM_BYTES_SFA
), (  # only staged x
    "x staging must fit the ring"
)
# one shared-memory region for the ring and the FC1 / FC2 B operands (a shared CTA reuses all of it)
_OFF_SFAR = STAGES * NUM_BYTES_A
_OFF_SFA = _OFF_SFAR + STAGES * _RAW_STAGE
_OFF_B1 = _OFF_SFA + STAGES * NUM_BYTES_SFA
_OFF_B2 = _OFF_B1 + S1 * NUM_BYTES_B1
_OFF_SFB2 = _OFF_B2 + NB2BUF * KT2 * NUM_BYTES_B2
BIG = _OFF_SFB2 + NB2BUF * KT2 * NUM_KB * 512
# shared CTA layout in BIG: weight buffers, x quarters (2), x scales [M, 192], gate / up
# [4, 16, M], h, h scales shared CTA layout in BIG: A ring (16 KB stages), raw scale
# atoms (4 per stage), spliced atoms, B (x: [stage][16 rows] [128 B]; h: [6][8][128 B]),
# x / h scale codes, gate_up partial exchange
_SB_A = 0
_SB_RAW = None
# slices 1.. gate / up partials [slice - 1][u][p][8 tokens] f32
_SH_XCH_BYTES = (SH_SPLIT - 1) * 2 * SH_ROWS * 8 * 4
_SH_FIX = (
    SH_GT * SH_NB * 128
    + SH_DKT * 8 * 128
    + M_MAX * SH_KB
    + M_MAX * SH_DKB
    + _SH_XCH_BYTES
    + 2 * SH_ROWS * 8 * 4
    + 1024
)
# (cfg sh_nst: a lower cap on the shared ring depth; fewer shared bytes in flight leaves more DRAM to the routed stream)
SH_NST = (
    min(_cfg("sh_nst", 8), 8, (BIG - _SH_FIX) // (NUM_BYTES_A + 4 * 512 + 512))
    if SHARED
    else 1
)
_SB_RAW = SH_NST * NUM_BYTES_A
_SB_SPL = _SB_RAW + SH_NST * 2048
_SB_BX = (_SB_SPL + SH_NST * 512 + 1023) // 1024 * 1024
_SB_BH = _SB_BX + SH_GT * SH_NB * 128
_SB_XC = _SB_BH + SH_DKT * 8 * 128
_SB_HC = _SB_XC + M_MAX * SH_KB
_SB_XCH = (_SB_HC + M_MAX * SH_DKB + 15) // 16 * 16
_SB_GU = _SB_XCH + _SH_XCH_BYTES
_SB_END = _SB_GU + 2 * SH_ROWS * 8 * 4
# shared CTA TMEM columns: gate_up accumulator (N SH_NB), down accumulators (2 x N 8), A scales per stage, B scales
SH_ACC_GU = 0
SH_ACC_DN = SH_NB
SH_SFA = SH_ACC_DN + 16
SH_SFB_GU = SH_SFA + 4 * 8
SH_SFB_DN = SH_SFB_GU + 4 * SH_GT
SH_PF = max(
    2, (SH_STAGES - SH_NST + 31) // 32
)  # L2-prefetched stages per lane beyond the ring
SH_TMEM = (
    256 if SH_SFB_DN + 24 <= 256 else 512
)  # TP2 needs 4 x 48 gate_up B-scale columns
if SHARED:
    assert FUSED_ROUTER, "shared expert: fused router only"
    assert SH_STAGES - SH_NST <= 32 * SH_PF, SH_STAGES
    assert SH_NST >= 3 and _SB_END <= BIG and SH_SFB_DN + 24 <= SH_TMEM, (
        SH_NST,
        _SB_END,
        BIG,
    )


# =============================================================================
# Inline-PTX helpers.
# =============================================================================
def _asm(ret, args, ptx, cons, side=False, loc=None, ip=None):
    return _llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        ptx,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---- all-reduce push (the MoE boundary's multicast exchange, set A): arp = int64
# [enable, mc, flags, rank, world, maxt]
@dsl_user_op
def _ld_volatile_s32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)], "ld.volatile.global.s32 $0, [$1];", "=r,l",
            has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _st_global_b32_if(addr, value, pred, *, loc=None, ip=None):
    """st.global.b32 when pred != 0."""
    _llvm.inline_asm(
        None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(value).ir_value(loc=loc, ip=ip),
               cutlass.Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.global.b32 [$0], $1;\n\t}", "l,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _bf16x2_word(lo, hi, *, loc=None, ip=None):
    """bf16(lo) | bf16(hi) << 16 (RN), with -0.0 halves turned into +0.0 (0x80000000 marks an empty exchange word)."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Float32(hi).ir_value(loc=loc, ip=ip), cutlass.Float32(lo).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 w, l, h;\n\t.reg .pred pl, ph;\n\tcvt.rn.bf16x2.f32 w, $1, $2;\n\t"
            "and.b32 l, w, 0xFFFF;\n\tsetp.eq.u32 pl, l, 0x8000;\n\t@pl and.b32 w, w, 0xFFFF0000;\n\t"
            "shr.u32 h, w, 16;\n\tsetp.eq.u32 ph, h, 0x8000;\n\t@ph and.b32 w, w, 0x0000FFFF;\n\t"
            "mov.b32 $0, w;\n\t}", "=r,f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _ar_push(arp, v, low, t, cl):
    """Push bf16 channel pair (cl, cl + 1) of token t (v = channel cl's bf16 value on this lane, its neighbour's on
    lane ^ 8) into this rank's set-A slot of the exchange; the slot half is the all-reduce CTA (t, cl // 768)'s call
    parity. Only lanes with `low` store; every lane of the warp must call it (shuffle)."""
    hi = cute.arch.shuffle_sync_bfly(v, 8)
    pred = cutlass.select_(low, cutlass.Int32(1), cutlass.Int32(0))
    cs = cutlass.select_(low, cl, cutlass.Int32(0))
    world = arp.load(idx=4)
    parity = cutlass.Int64(
        _ld_volatile_s32(
            arp.load(idx=2)
            + cutlass.Int64(4)
            * cutlass.Int64(t * cutlass.Int32(8) + cs // cutlass.Int32(H // 8))
        )
        & cutlass.Int32(1)
    )
    off = (
        parity * arp.load(idx=5) * world * cutlass.Int64(H // 2)
        + (cutlass.Int64(t) * world + arp.load(idx=3)) * (cutlass.Int64(H // 2))
        + cutlass.Int64(cs // cutlass.Int32(2))
    )
    _st_global_b32_if(
        arp.load(idx=1) + off * cutlass.Int64(4), _bf16x2_word(v, hi), pred
    )


def _ar_parity(arp, t, cl):
    """The call parity _ar_push reads for token t's channel cl (fixed between this kernel's grid dependency and the
    next all-reduce, which this kernel releases at teardown): loaded ahead of the finalize (AR_PRE)."""
    return _ld_volatile_s32(
        arp.load(idx=2)
        + cutlass.Int64(4)
        * cutlass.Int64(t * cutlass.Int32(AR_CL) + cl // cutlass.Int32(H // AR_CL))
    )


def _ar_push_pre(ard, parity32, v, low, t, cl):
    """_ar_push with the parity (_ar_parity) and the descriptor fields ard = (mc, rank, world, maxt) preloaded."""
    hi = cute.arch.shuffle_sync_bfly(v, 8)
    pred = cutlass.select_(low, cutlass.Int32(1), cutlass.Int32(0))
    cs = cutlass.select_(low, cl, cutlass.Int32(0))
    rk, world, maxt = ard[-3], ard[-2], ard[-1]
    parity = cutlass.Int64(parity32 & cutlass.Int32(1))
    off = (
        parity * maxt * world * cutlass.Int64(H // 2)
        + (cutlass.Int64(t) * world + rk) * (cutlass.Int64(H // 2))
        + cutlass.Int64(cs // cutlass.Int32(2))
    )
    w = _bf16x2_word(v, hi)
    for r_ in range(len(ard) - 3):
        _st_global_b32_if(ard[r_] + off * cutlass.Int64(4), w, pred)


def _ar_push_g3(ard, v, low, t, cl):
    """Push bf16 channel pair (cl, cl + 1) of token t into boundary-exchange's slot (buf, rank, t) on every rank's buffer;
    ard = (bufs[0], .., bufs[W - 1], rank, buf) preloaded after the grid dependency (AR_PRE)."""
    hi = cute.arch.shuffle_sync_bfly(v, 8)
    pred = cutlass.select_(low, cutlass.Int32(1), cutlass.Int32(0))
    cs = cutlass.select_(low, cl, cutlass.Int32(0))
    w = _bf16x2_word(v, hi)
    rk = ard[AR_W]
    bufi = ard[AR_W + 1]
    off = (
        (bufi * cutlass.Int64(AR_W) + rk) * cutlass.Int64(AR_ROWS) + cutlass.Int64(t)
    ) * cutlass.Int64(H) + cutlass.Int64(cs)
    for r in range(AR_W):
        _st_global_b32_if(ard[r] + off * cutlass.Int64(2), w, pred)


@dsl_user_op
def _bar_arrive(bar_id: int, threads: int, *, loc=None, ip=None):
    """bar.arrive: signal a named barrier without waiting (its memory ordering is bar.sync's)."""
    _llvm.inline_asm(
        None,
        [],
        f"bar.arrive {bar_id}, {threads};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_relaxed_u32(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(val)],
        "st.relaxed.gpu.global.u32 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _ld_relaxed_u32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int64(addr)],
            "ld.relaxed.gpu.global.u32 $0, [$1];",
            "=r,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_relaxed_u64(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int64(val)],
        "st.relaxed.gpu.global.u64 [$0], $1;",
        "l,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _ld_relaxed_u64(addr, *, loc=None, ip=None):
    return cutlass.Int64(
        _asm(
            _T.i64(),
            [cutlass.Int64(addr)],
            "ld.relaxed.gpu.global.u64 $0, [$1];",
            "=l,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _nanosleep(ns: int, *, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [],
        f"nanosleep.u32 {ns};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _rcp(a, *, loc=None, ip=None):
    """rcp.approx.ftz.f32 (the reciprocal TRT-LLM's FP4 quantization uses)."""
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b), cutlass.Float32(c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _sub_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "sub.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _e4m3_bits(x, *, loc=None, ip=None):
    """e4m3 (round to nearest even, saturating) of x as an int 0..255."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(x)],
            "{\n.reg .b16 t;\ncvt.rn.satfinite.e4m3x2.f32 t, 0f00000000, $1;\ncvt.u32.u16 $0, t;\n}",
            "=r,f",
            False,
            loc,
            ip,
        )
    ) & cutlass.Int32(0xFF)


@dsl_user_op
def _e4m3_to_f32(b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Int32(b)],
            "{\n.reg .b16 t, l, u;\n.reg .b32 h;\ncvt.u16.u32 t, $1;\ncvt.rn.f16x2.e4m3x2 h, t;\nmov.b32 {l, u}, h;\n"
            "cvt.f32.f16 $0, l;\n}",
            "=f,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _e2m1x2(hi, lo, *, loc=None, ip=None):
    """One byte of two E2M1 values (lo in the low nibble), round to nearest even, saturating."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(hi), cutlass.Float32(lo)],
            "{\n.reg .b8 t;\ncvt.rn.satfinite.e2m1x2.f32 t, $1, $2;\ncvt.u32.u8 $0, t;\n}",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ex2(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(x)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16_round(x, *, loc=None, ip=None):
    """X rounded to bf16 (nearest even), as f32."""
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(x)],
            "{\n.reg .b16 t;\ncvt.rn.bf16.f32 t, $1;\nmov.b32 $0, {0, t};\n}",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _redux_max_u32(v, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int32(v)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16_bits(x, *, loc=None, ip=None):
    """bf16 (nearest even) bits of x in the low 16 bits."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(x)],
            "{\n.reg .b16 t;\ncvt.rn.bf16.f32 t, $1;\ncvt.u32.u16 $0, t;\n}",
            "=r,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ue8m0_rp(x, *, loc=None, ip=None):
    """UE8M0 code of x rounded toward +inf, saturating (__nv_cvt_float_to_e8m0(x, SATFINITE, RoundPosInf))."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(x)],
            "{\n.reg .b16 t;\ncvt.rp.satfinite.ue8m0x2.f32 t, $1, $1;\ncvt.u32.u16 $0, t;\n}",
            "=r,f",
            False,
            loc,
            ip,
        )
    ) & cutlass.Int32(0xFF)


@dsl_user_op
def _e4m3x2(hi, lo, *, loc=None, ip=None):
    """Two e4m3 (nearest even, saturating) in 16 bits, lo in the low byte."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(hi), cutlass.Float32(lo)],
            "{\n.reg .b16 t;\ncvt.rn.satfinite.e4m3x2.f32 t, $1, $2;\ncvt.u32.u16 $0, t;\n}",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _min_xorsign_abs(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "min.xorsign.abs.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _div_rn(a, b, *, loc=None, ip=None):
    """IEEE division (div.rn.f32)."""
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _div_full(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "div.full.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ex2_full(x, *, loc=None, ip=None):
    """ex2.approx.f32 (no flush to zero), Triton's tl.exp."""
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(x)],
            "ex2.approx.f32 $0, $1;",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _l2_prefetch(addr, nbytes, *, loc=None, ip=None):
    """cp.async.bulk.prefetch.L2 of nbytes (a multiple of 16) at global address addr (16-byte aligned)."""
    _llvm.inline_asm(
        None,
        [
            cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _mx_osc(amax):
    """mxfp8 quantization of a block with absolute max amax: (UE8M0 code, the element multiplier)."""
    code = _ue8m0_rp(_mul_rn(amax, _rcp(cutlass.Float32(448.0))))
    cf = cutlass.Int32(
        cutlass.select_(
            code == cutlass.Int32(255),
            cutlass.Int32(0x7FFFFFFF),
            cutlass.select_(
                code == cutlass.Int32(0),
                cutlass.Int32(0x00400000),
                code << cutlass.Int32(23),
            ),
        )
    ).bitcast(cutlass.Float32)
    return code, cutlass.select_(
        amax != cutlass.Float32(0.0), _rcp(cf), cutlass.Float32(0.0)
    )


def _mx_words(v, osc):
    """e4m3 words of len(v) (a multiple of 4) values times osc, in element order."""
    words = []
    for t in range(len(v) // 4):
        lo = _e4m3x2(_mul_rn(v[4 * t + 1], osc), _mul_rn(v[4 * t], osc))
        hi = _e4m3x2(_mul_rn(v[4 * t + 3], osc), _mul_rn(v[4 * t + 2], osc))
        words.append(lo | (hi << cutlass.Int32(16)))
    return words


def _twiddle(score):
    """Float bits mapped so that unsigned order is float order (cub TwiddleIn)."""
    bits = (
        score.bitcast(cutlass.Int32)
        if hasattr(score, "bitcast")
        else cutlass.Float32(score).bitcast(cutlass.Int32)
    )
    return cutlass.select_(
        bits < cutlass.Int32(0), ~bits, bits | cutlass.Int32(-2147483648)
    )


@cute.jit
def _wait(bar: cutlass.Array, parity):
    while not cute.arch.mbarrier_try_wait(bar.data_ptr(), parity):
        pass
    cute.arch.sync_warp()


def _tptr(base_row, col, dtype=cutlass.Int32):
    return cutlass.inttoptr((base_row << 16) | col, 6, dtype)


@dsl_user_op
def _match_any(value, *, loc=None, ip=None):
    """The mask of the warp's lanes holding the same 32-bit value."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int32(value)],
            "match.any.sync.b32 $0, $1, 0xffffffff;",
            "=r,r",
            False,
            loc,
            ip,
        )
    )


@cute.jit
def _waitd(bar: cutlass.Array, parity, tdbg: cutlass.Array, site: cutlass.Constexpr):
    """Wait on `bar` at `parity`, then sync the warp."""
    while not cute.arch.mbarrier_try_wait(bar.data_ptr(), parity):
        pass
    cute.arch.sync_warp()


@cute.jit
def _adv1(stage, phase):
    """FC1 ring cursor: [0, S1R) (= _adv unless PW2F)."""
    stage = stage + cutlass.Int32(1)
    wrap = stage == cutlass.Int32(S1R)
    return cutlass.select_(wrap, cutlass.Int32(0), stage), cutlass.select_(
        wrap, phase ^ cutlass.Int32(1), phase
    )


@cute.jit
def _adv2(stage, phase):
    """FC2 ring cursor: [S1R if PW2F else 0, STAGES) (= _adv unless PW2F)."""
    stage = stage + cutlass.Int32(1)
    wrap = stage == cutlass.Int32(STAGES)
    return cutlass.select_(
        wrap, cutlass.Int32(S1R if PW2F else 0), stage
    ), cutlass.select_(wrap, phase ^ cutlass.Int32(1), phase)


@cute.jit
def _adv(stage, phase):
    """The next ring position's (stage, phase)."""
    stage = stage + cutlass.Int32(1)
    wrap = stage == cutlass.Int32(STAGES)
    return cutlass.select_(wrap, cutlass.Int32(0), stage), cutlass.select_(
        wrap, phase ^ cutlass.Int32(1), phase
    )


@cute.jit
def _seg(i0, i1, k):
    """Segment k of the FC1 item range [i0, i1) (items = unit-major stages): (unit, first stage, end stage). Order:
    the last unit (its head, the part a split unit's next CTA does not cover), then the first unit, then the rest."""
    uf = i0 // cutlass.Int32(S1)
    ul = (i1 - cutlass.Int32(1)) // cutlass.Int32(S1)
    unit = uf + k - cutlass.Int32(1)
    unit = cutlass.select_(k == cutlass.Int32(0), ul, unit)
    unit = cutlass.select_(ul == uf, uf, unit)
    j0 = i0 - unit * cutlass.Int32(S1)
    j0 = cutlass.select_(j0 < cutlass.Int32(0), cutlass.Int32(0), j0)
    j1 = i1 - unit * cutlass.Int32(S1)
    j1 = cutlass.select_(j1 > cutlass.Int32(S1), cutlass.Int32(S1), j1)
    return unit, j0, j1


@cute.jit
def _nsegs(i0, i1):
    n = (
        (i1 - cutlass.Int32(1)) // cutlass.Int32(S1)
        - i0 // cutlass.Int32(S1)
        + cutlass.Int32(1)
    )
    return cutlass.select_(i1 > i0, n, cutlass.Int32(0))


@cute.jit
def _ug_slot(ug):
    """Slot (in its group) of FC1 unit index ug (slot-major)."""
    return ug // cutlass.Int32(U1)


@cute.jit
def _ug_c(ug):
    """Unit (in its expert) of FC1 unit index ug (see _ug_slot)."""
    return ug - (ug // cutlass.Int32(U1)) * cutlass.Int32(U1)


@cute.jit
def _sfb1_to_tmem(
    warp, lane, base_row, base_col, src: cutlass.Array, b1_ready: cutlass.Array
):
    """B1 scales to TMEM: instruction (j, kb)'s column holds, at lane n = QS q + t, token t's 4 scales of k-tile
    4 j + q, K 64 kb .. (src: int32 words [M, 96] of the token scales); each warp writes its sub-partition."""
    q = lane // cutlass.Int32(QS)
    tl = lane - q * cutlass.Int32(QS)
    sf_ok = (lane < cutlass.Int32(N1)) & (tl < cutlass.Int32(M_MAX))
    sfv = []
    for a in cutlass.range_constexpr(NUM_SFB1_ATOMS):
        j = a // NUM_KB
        kb = a % NUM_KB
        wsf = src.load(
            idx=cutlass.select_(
                sf_ok,
                tl * cutlass.Int32(H // 64)
                + (cutlass.Int32(4 * j) + q) * cutlass.Int32(4)
                + cutlass.Int32(kb),
                cutlass.Int32(0),
            )
        )
        sfv.append(cutlass.select_(sf_ok, cutlass.Int32(wsf), cutlass.Int32(0)))
    for a in cutlass.range_constexpr(NUM_SFB1_ATOMS):
        prims.tcgen05_st(
            "32x32b",
            _tptr(base_row + warp * cutlass.Int32(32), base_col + SFB1_COL + a * 4),
            sfv[a],
        )
    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
    prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.mbarrier_arrive(b1_ready)


@cute.jit
def _b1_addr(t, kb16):
    """Word index in the resident B tiles of the 8 FP4 bytes of token t's 16-value block kb16."""
    kt = kb16 >> cutlass.Int32(4)
    j = kt >> cutlass.Int32(2)
    n = (kt & cutlass.Int32(3)) * cutlass.Int32(QS) + t
    cc = (kb16 >> cutlass.Int32(1)) & cutlass.Int32(7)
    return (
        j * cutlass.Int32(NUM_BYTES_B1)
        + n * cutlass.Int32(KT_BYTES)
        + (cc ^ (n & cutlass.Int32(7))) * cutlass.Int32(16)
        + (kb16 & cutlass.Int32(1)) * cutlass.Int32(8)
    ) // cutlass.Int32(4)


@cute.jit
def _build_b1(
    tidx,
    warp,
    lane,
    base_row,
    base_col,
    x32: cutlass.Array,
    xsf32: cutlass.Array,
    sB1: cutlass.Array,
    b1_ready: cutlass.Array,
):
    """B (the M tokens' whole K) into the resident tiles from the NVFP4 activation, its scales to TMEM."""
    sB1_32 = cutlass.Array(
        sB1.data_ptr(0),
        shape=(S1 * NUM_BYTES_B1 // 4,),
        dtype=cutlass.Int32,
        alignment=16,
    )
    for r in cutlass.range_constexpr(
        (M_MAX * (H // 16) + EPI_THREADS - 1) // EPI_THREADS
    ):
        b = tidx + cutlass.Int32(r * EPI_THREADS)
        if b < cutlass.Int32(M_MAX * (H // 16)):
            t = b // cutlass.Int32(H // 16)
            kb = b - t * cutlass.Int32(H // 16)
            v2 = x32.load(idx=b * cutlass.Int32(2), vector_size=2, alignment=8)
            sB1_32.store(
                (cutlass.Int32(v2[0]), cutlass.Int32(v2[1])),
                idx=_b1_addr(t, kb),
                alignment=8,
            )
    _sfb1_to_tmem(warp, lane, base_row, base_col, xsf32, b1_ready)


@cute.jit
def _build_b1_fused(
    tidx,
    warp,
    lane,
    base_row,
    base_col,
    xb32: cutlass.Array,
    sfsc: cutlass.Array,
    sB1: cutlass.Array,
    s_xsf: cutlass.Array,
    b1_ready: cutlass.Array,
):
    """_build_b1 from the bf16 hidden states, quantized exactly as fp4_quantize(x, sf_scale, 16)."""
    sB1_32 = cutlass.Array(
        sB1.data_ptr(0),
        shape=(S1 * NUM_BYTES_B1 // 4,),
        dtype=cutlass.Int32,
        alignment=16,
    )
    s_xsf8 = cutlass.Array(
        s_xsf.data_ptr(0), shape=(M_MAX * (H // 16),), dtype=cutlass.Int8, alignment=16
    )
    sc = cutlass.Float32(sfsc.load(idx=0))
    rsc = _rcp(sc)
    rsix = _rcp(cutlass.Float32(6.0))
    NBLK = M_MAX * (H // 16) // EPI_THREADS
    QB = B1_QB_EFF  # blocks whose loads are in flight together (cfg b1_qb)
    # a rolled loop over groups of QB blocks (a fully unrolled one is ~50 KB of SASS at M = 8: instruction fetch bound)
    for rg in cutlass.range(NBLK // QB, unroll=1):
        wall = []
        for rr in cutlass.range_constexpr(QB):
            bq = tidx + (rg * cutlass.Int32(QB) + cutlass.Int32(rr)) * cutlass.Int32(
                EPI_THREADS
            )
            for h in cutlass.range_constexpr(2):
                wall.append(
                    xb32.load(
                        idx=bq * cutlass.Int32(8) + cutlass.Int32(4 * h),
                        vector_size=4,
                        alignment=16,
                    )
                )
        for rr in cutlass.range_constexpr(QB):
            b = tidx + (rg * cutlass.Int32(QB) + cutlass.Int32(rr)) * cutlass.Int32(
                EPI_THREADS
            )
            t = b // cutlass.Int32(H // 16)
            kb = b - t * cutlass.Int32(H // 16)
            w = []
            for h in cutlass.range_constexpr(2):
                v4 = wall[rr * 2 + h]
                for z in cutlass.range_constexpr(4):
                    w.append(cutlass.Int32(v4[z]))
            xs = []
            for z in cutlass.range_constexpr(8):
                xs.append((w[z] << cutlass.Int32(16)).bitcast(cutlass.Float32))
                xs.append((w[z] & cutlass.Int32(-65536)).bitcast(cutlass.Float32))
            vmax = cute.math.abs(xs[0])
            for z in cutlass.range_constexpr(1, 16):
                vmax = cute.arch.fmax(vmax, cute.math.abs(xs[z]))
            sfb = _e4m3_bits(_mul_rn(sc, _mul_rn(vmax, rsix)))
            osc = cutlass.select_(
                vmax != cutlass.Float32(0.0),
                _rcp(_mul_rn(_e4m3_to_f32(sfb), rsc)),
                cutlass.Float32(0.0),
            )
            by = [
                _e2m1x2(_mul_rn(xs[2 * z + 1], osc), _mul_rn(xs[2 * z], osc))
                & cutlass.Int32(0xFF)
                for z in range(8)
            ]
            lo = (
                by[0]
                | (by[1] << cutlass.Int32(8))
                | (by[2] << cutlass.Int32(16))
                | (by[3] << cutlass.Int32(24))
            )
            hi = (
                by[4]
                | (by[5] << cutlass.Int32(8))
                | (by[6] << cutlass.Int32(16))
                | (by[7] << cutlass.Int32(24))
            )
            sB1_32.store((lo, hi), idx=_b1_addr(t, kb), alignment=8)
            s_xsf8.store(cutlass.Int8(sfb), idx=b)
    cute.arch.sync_warp()
    cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=EPI_THREADS)
    _sfb1_to_tmem(warp, lane, base_row, base_col, s_xsf, b1_ready)


@cute.jit
def _swiglu_quant(xv, gate_raw, is_up, a, c, alpha, beta, lim):
    """TRTLLM-Gen's FC1 epilogue (the swiGlu Bmm E2m1 output), operation for operation: lane value xv is the up row,
    gate_raw its gate row (fp32 sums). Returns (q, the scale's e4m3 bits) of the 16-row NVFP4 block (lanes r, r ^ 16
    and r ^ 8 cover it; amax over the warp's up lanes of a half). a = fc31_alpha, c = fc31_scale_c."""
    neg_max = cutlass.Float32(-3.40282346638528859812e38)
    gc = cutlass.select_(gate_raw > lim, lim, gate_raw)
    gc = cutlass.select_(gate_raw < neg_max, neg_max, gc)
    uc = cutlass.select_(xv > lim, lim, xv)
    uc = cutlass.select_(xv < -lim, -lim, uc)
    c1 = _mul_rn(a, _mul_rn(alpha, cutlass.Float32(1.4426950216293334961)))
    sgm = _rcp(_add_rn(_ex2(-_mul_rn(gc, c1)), cutlass.Float32(1.0)))
    gs = _mul_rn(gc, sgm)
    h = _mul_rn(_fma_rn(uc, a, _mul_rn(a, beta)), gs)
    h = cutlass.select_(is_up, h, cutlass.Float32(0.0))
    amax = cute.arch.fmax(_mul_rn(cute.math.abs(h), c), cutlass.Float32(0.0))
    for off in (16, 8, 4, 2, 1):
        amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, off))
    sfb = _e4m3_bits(_mul_rn(amax, cutlass.Float32(0.16666667163372039795)))
    sff = _e4m3_to_f32(sfb)
    osc = cutlass.select_(sff != cutlass.Float32(0.0), _rcp(sff), cutlass.Float32(0.0))
    return _mul_rn(_mul_rn(h, osc), c), sfb


@cute.jit
def _plan_range(s_plan: cutlass.Array, p):
    w = s_plan.load(idx=p)
    return w & cutlass.Int32(0xFFFF), w >> cutlass.Int32(16)


@cute.jit
def _sh_issue(
    lane,
    sid,
    g,
    s_big: cutlass.Array,
    sh_full: cutlass.Array,
    tma_gu,
    tma_dn,
    gusf: cutlass.Array,
    dnsf: cutlass.Array,
):
    """Stage g of the shared CTA's stream into ring slot g % SH_NST (warp 0). g < SH_GT: gate_up k-tile g of both
    slices: quadrant Q = 2 q + u (q slice, u 0 gate / 1 up) = rows 32 sid .. (+ 768 for up) at k-tile SH_GT q + g, its
    raw scale atom at raw slot Q. Else down tile j = (g - SH_GT) / 6 (rows 256 sid + 128 j ..), k-tile
    (g - SH_GT) % 6."""
    st = g % cutlass.Int32(SH_NST)
    bar = sh_full.subview(st)
    if cutlass.const_expr(SH_PRETILE):
        if lane == 0:
            src = (sid * cutlass.Int32(SH_STAGES) + g) * cutlass.Int32(SH_STAGE_BYTES)
            prims.mbarrier_arrive_expect_tx(bar, SH_STAGE_BYTES)
            prims.cp_async_bulk_shared_cluster_global(
                s_big.subview(st * cutlass.Int32(NUM_BYTES_A)),
                gusf.subview(src),
                bar,
                16384,
            )
            prims.cp_async_bulk_shared_cluster_global(
                s_big.subview(cutlass.Int32(_SB_SPL) + st * cutlass.Int32(512)),
                gusf.subview(src + cutlass.Int32(16384)),
                bar,
                512,
            )
        cute.arch.sync_warp()
    else:
        if g < cutlass.Int32(SH_GT):
            if lane == 0:
                prims.mbarrier_arrive_expect_tx(bar, NUM_BYTES_A + 4 * 512)
            cute.arch.sync_warp()
            if lane < 4:
                q = lane >> cutlass.Int32(1)
                u = lane & cutlass.Int32(1)
                row0 = sid * cutlass.Int32(32) + u * cutlass.Int32(SH_I)
                kt = q * cutlass.Int32(SH_GT) + g
                prims.cp_async_bulk_tensor_shared_cta_global(
                    s_big.subview(
                        st * cutlass.Int32(NUM_BYTES_A) + lane * cutlass.Int32(4096)
                    ),
                    tma_gu,
                    (kt * cutlass.Int32(128), row0),
                    bar,
                    l2_cache_hint=EVICT_FIRST,
                )
                prims.cp_async_bulk_shared_cluster_global(
                    s_big.subview(
                        cutlass.Int32(_SB_RAW)
                        + st * cutlass.Int32(2048)
                        + lane * cutlass.Int32(512)
                    ),
                    gusf.subview(
                        ((row0 >> cutlass.Int32(7)) * cutlass.Int32(SH_KB // 4) + kt)
                        * cutlass.Int32(512)
                    ),
                    bar,
                    512,
                )
        else:
            gd = g - cutlass.Int32(SH_GT)
            jt = gd // cutlass.Int32(SH_DKT)
            kt = gd - jt * cutlass.Int32(SH_DKT)
            row0 = sid * cutlass.Int32(SH_DROWS) + jt * cutlass.Int32(128)
            if lane == 0:
                prims.mbarrier_arrive_expect_tx(bar, NUM_BYTES_A + 512)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    s_big.subview(st * cutlass.Int32(NUM_BYTES_A)),
                    tma_dn,
                    (kt * cutlass.Int32(128), row0),
                    bar,
                    l2_cache_hint=EVICT_FIRST,
                )
                prims.cp_async_bulk_shared_cluster_global(
                    s_big.subview(cutlass.Int32(_SB_RAW) + st * cutlass.Int32(2048)),
                    dnsf.subview(
                        ((row0 >> cutlass.Int32(7)) * cutlass.Int32(SH_DKB // 4) + kt)
                        * cutlass.Int32(512)
                    ),
                    bar,
                    512,
                )
            cute.arch.sync_warp()


@dsl_user_op
def _red_release_add(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(val)],
        "red.release.gpu.global.add.u32 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _ld_acquire_u32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int64(addr)],
            "ld.acquire.gpu.global.u32 $0, [$1];",
            "=r,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _fence_proxy_async_global(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.global;", "", True, loc, ip)


@cute.jit
def _sfa_to_tmem(sSFA: cutlass.Array, stage, base_row, base_col):
    """(SFA_CP, elected MMA thread) the stage's 4 scale atoms smem -> TMEM (what the splicer warp does otherwise)."""
    s2t_shape, s2t_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
    sd0 = prims.Tcgen05SmemDesc.build(
        sSFA.subview(stage * cutlass.Int32(NUM_WORDS_SFA)),
        leading_byte_offset=16,
        stride_byte_offset=128,
        base_offset=0,
        layout=0,
    )
    for kb in cutlass.range_constexpr(NUM_KB):
        prims.tcgen05_cp(
            s2t_shape,
            _tptr(base_row, base_col + SFA1_COL + stage * cutlass.Int32(16) + kb * 4),
            sd0 + (kb * 512 >> 4),
            multicast=s2t_mc,
        )


@cute.jit
def _sf1_cp(
    lane,
    stage,
    sf_line,
    c,
    sSFA: cutlass.Array,
    s1w: cutlass.Array,
    ab_full: cutlass.Array,
):
    """(SFA_CP) FC1 stage scale atoms: word `[r][q]` of atom kb <- raw atom 4 q + kb (raw line sf_line = 4 x atom)
    word `[r][c % 4]` (what the splicer copies out of the raw box), 4-byte cp.async by the warp's 32 lanes; their
    arrivals (noinc) complete the stage's full barrier together with the A boxes' bytes."""
    src0 = (
        (sf_line >> cutlass.Int32(2)) * cutlass.Int32(128)
        + lane * cutlass.Int32(4)
        + (c & cutlass.Int32(3))
    )
    dst0 = stage * cutlass.Int32(NUM_WORDS_SFA) + lane * cutlass.Int32(4)
    if cutlass.const_expr(SFA_BULK):
        if lane == 0:
            blk_ = ((sf_line >> cutlass.Int32(2)) >> cutlass.Int32(4)) * cutlass.Int32(
                4
            ) + (c & cutlass.Int32(3))
            _bulk_g2s(
                sSFA.subview(stage * cutlass.Int32(NUM_WORDS_SFA)).data_ptr().toint(),
                cutlass.Int64(s1w.data_ptr().toint())
                + cutlass.Int64(blk_) * cutlass.Int64(NUM_BYTES_SFA),
                NUM_BYTES_SFA,
                ab_full.subview(stage).data_ptr().toint(),
            )
            if cutlass.const_expr(
                SFA_ARR1
            ):  # the 32 lanes' arrivals the barrier counts (no cp.async is pending)
                _mbar_arrive_n(ab_full.subview(stage).data_ptr().toint(), 32)
        if cutlass.const_expr(not SFA_ARR1):
            cute.arch.sync_warp()
    else:
        for q in cutlass.range_constexpr(C1):
            for kb in cutlass.range_constexpr(NUM_KB):
                prims.cp_async_shared_global(
                    sSFA.subview(dst0 + cutlass.Int32(kb * 128 + q)),
                    s1w.subview(src0 + cutlass.Int32((4 * q + kb) * 128)),
                    4,
                    "ca",
                )
    if cutlass.const_expr(not (SFA_BULK and SFA_ARR1)):
        prims.cp_async_mbarrier_arrive(
            ab_full.subview(stage), noinc=True
        )  # (SFA_BULK: no cp.async, arrives now)
    cute.arch.sync_warp()


@cute.jit
def _sf2_cp(
    lane,
    stage,
    g,
    blk,
    t2,
    n_here,
    s_el: cutlass.Array,
    sSFA: cutlass.Array,
    s2w: cutlass.Array,
    ab_full: cutlass.Array,
):
    """(SFA_CP) FC2 k-tile t2 scale atoms: word `[r][jj]` of atom kb <- raw atom (rows / 128, K block 4 t2 + kb) of slot
    jj's expert, word `[r][blk % 4]`."""
    dst0 = stage * cutlass.Int32(NUM_WORDS_SFA) + lane * cutlass.Int32(4)
    for jj in cutlass.range_constexpr(4):
        if cutlass.Int32(jj) < n_here:
            e = s_el.load(idx=g * cutlass.Int32(4) + cutlass.Int32(jj))
            src0 = (
                (
                    (e * cutlass.Int32(SF2_RB) + (blk >> cutlass.Int32(2)))
                    * cutlass.Int32(SF2_KA)
                    + cutlass.Int32(4 * t2)
                )
                * cutlass.Int32(128)
                + lane * cutlass.Int32(4)
                + (blk & cutlass.Int32(3))
            )
            # (a partial last k-tile has no atoms past I_TP)
            for kb in cutlass.range_constexpr(NUM_KB):
                prims.cp_async_shared_global(
                    sSFA.subview(dst0 + cutlass.Int32(kb * 128 + jj)),
                    s2w.subview(src0 + cutlass.Int32(kb * 128)),
                    4,
                    "ca",
                )
    prims.cp_async_mbarrier_arrive(ab_full.subview(stage), noinc=True)
    cute.arch.sync_warp()


@dsl_user_op
def _atom_add_release(addr, val, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int64(addr), cutlass.Int32(val)],
            "atom.release.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_release_u32(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(val)],
        "st.release.gpu.global.u32 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _rsqrt_approx(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(x)],
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_v4_u32(addr, volatile: bool = False, cg: bool = False, *, loc=None, ip=None):
    """Four int32 words at a 16-byte-aligned global address (ld.volatile for the polled exchange slots, .cg for data
    other CTAs of this grid wrote)."""
    op = (
        "ld.volatile.global.v4.u32"
        if volatile
        else ("ld.global.cg.v4.u32" if cg else "ld.global.v4.u32")
    )
    out = _llvm.inline_asm(
        _llvm.StructType.get_literal([_T.i32()] * 4), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        op + " {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip
    return [
        cutlass.Int32(_llvm.extractvalue(_T.i32(), out, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def _st_v4_u32(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(w0).ir_value(loc=loc, ip=ip),
         cutlass.Int32(w1).ir_value(loc=loc, ip=ip), cutlass.Int32(w2).ir_value(loc=loc, ip=ip),
         cutlass.Int32(w3).ir_value(loc=loc, ip=ip)],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _mbar_arrive_n(mbar_smem, n, *, loc=None, ip=None):
    """mbarrier.arrive with count n on a shared-memory barrier (release, CTA scope)."""
    _llvm.inline_asm(
        None,
        [cutlass.Int32(mbar_smem).ir_value(loc=loc, ip=ip), cutlass.Int32(n).ir_value(loc=loc, ip=ip)],
        "{ .reg .b64 st; mbarrier.arrive.release.cta.shared::cta.b64 st, [$0], $1; }", "r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _bulk_g2s(dst_smem, src_gaddr, nbytes, mbar_smem, *, loc=None, ip=None):
    """cp.async.bulk of nbytes from a raw global address into shared memory, completing mbar (shared addresses)."""
    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst_smem).ir_value(loc=loc, ip=ip), cutlass.Int64(src_gaddr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip), cutlass.Int32(mbar_smem).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];", "r,l,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@cute.jit
def _ar_in_reduce(
    tidx,
    warp,
    lane,
    bx,
    desc,
    s_big: cutlass.Array,
    s_par: cutlass.Array,
    xb32: cutlass.Array,
    s_misc: cutlass.Array,
    tdbg: cutlass.Array,
    ar_bar: cutlass.Array,
):
    """AR_IN prologue (all THREADS threads of a routed CTA), distributed like the prx boundary: CTA c owns ARIN_PER
    16-byte chunks of the [M, H] rows (each slot word is polled by one thread of the grid, then re-armed by it), sums
    the ranks, adds the residual, writes residual_out and its rows' sum-of-squares partials; one grid-wide counter;
    then every CTA normalizes the full rows (read back from residual_out) into its x staging. CTA 0 also writes the
    normed x to global (xb32) for the shared CTAs."""
    buf = cutlass.Int64(_ld_relaxed_u64(desc))
    flg = cutlass.Int64(_ld_relaxed_u64(desc + cutlass.Int64(8)))
    maxt = cutlass.Int64(_ld_relaxed_u64(desc + cutlass.Int64(24)))
    res_in = cutlass.Int64(_ld_relaxed_u64(desc + cutlass.Int64(32)))
    rout = cutlass.Int64(_ld_relaxed_u64(desc + cutlass.Int64(40)))
    gamma_ptr = cutlass.Int64(_ld_relaxed_u64(desc + cutlass.Int64(48)))
    eps = _ld_relaxed_u32(desc + cutlass.Int64(56)).bitcast(cutlass.Float32)
    cnt = cutlass.Int64(_ld_relaxed_u64(desc + cutlass.Int64(64)))
    psq = cutlass.Int64(_ld_relaxed_u64(desc + cutlass.Int64(80)))
    half = maxt * cutlass.Int64(AR_IN_W * (H // 2))
    xs32 = cutlass.Array(
        s_big.data_ptr(0), shape=(M_MAX * H // 2,), dtype=cutlass.Int32, alignment=16
    )
    scr = cutlass.Array(
        s_big.data_ptr(_OFF_B2),
        shape=((THREADS // 32) * M_MAX,),
        dtype=cutlass.Float32,
        alignment=16,
    )
    if tidx < cutlass.Int32(M_MAX * AR_IN_CL):
        s_par.store(
            _ld_volatile_s32(flg + cutlass.Int64(4) * cutlass.Int64(tidx))
            & cutlass.Int32(1),
            idx=tidx,
        )
    prims.barrier_cta_sync(0)
    # ---- phase A: this CTA's chunks
    E = cutlass.Int32(ARIN_EMPTY)
    q0 = bx * cutlass.Int32(ARIN_PER) + tidx
    act = (tidx < cutlass.Int32(ARIN_PER)) & (q0 < cutlass.Int32(ARIN_TC))
    q = cutlass.select_(act, q0, cutlass.Int32(0))
    t = q // cutlass.Int32(ARIN_NCC)
    cc = q - t * cutlass.Int32(ARIN_NCC)
    sl = cc // cutlass.Int32(ARIN_NCC // AR_IN_CL)
    par = cutlass.Int64(s_par.load(idx=t * cutlass.Int32(AR_IN_CL) + sl))
    wb = (
        par * half
        + cutlass.Int64(t) * cutlass.Int64(AR_IN_W * (H // 2))
        + cutlass.Int64(cc) * cutlass.Int64(4)
    )
    sqt = cutlass.Float32(0.0)
    if act:
        acc = [cutlass.Float32(0.0) for _ in range(8)]
        addrs = [
            buf + (wb + cutlass.Int64(r * (H // 2))) * cutlass.Int64(4)
            for r in range(AR_IN_W)
        ]
        first = [_ld_v4_u32(addrs[r], True) for r in range(AR_IN_W)]
        rres = _ld_v4_u32(
            res_in
            + (
                cutlass.Int64(t) * cutlass.Int64(H)
                + cutlass.Int64(cc) * cutlass.Int64(8)
            )
            * cutlass.Int64(2)
        )
        for r in cutlass.range_constexpr(AR_IN_W):
            v0, v1, v2, v3 = first[r][0], first[r][1], first[r][2], first[r][3]
            while (v0 == E) | (v1 == E) | (v2 == E) | (v3 == E):
                vr = _ld_v4_u32(addrs[r], True)
                v0, v1, v2, v3 = vr[0], vr[1], vr[2], vr[3]
            _st_v4_u32(
                addrs[r], E, E, E, E
            )  # re-arm (prx: the reader re-arms what it read)
            vv = [v0, v1, v2, v3]
            for z in cutlass.range_constexpr(4):
                lo = (vv[z] << cutlass.Int32(16)).bitcast(cutlass.Float32)
                hi = (vv[z] & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                if cutlass.const_expr(r == 0):
                    acc[2 * z] = lo
                    acc[2 * z + 1] = hi
                else:
                    acc[2 * z] = _add_rn(acc[2 * z], lo)
                    acc[2 * z + 1] = _add_rn(acc[2 * z + 1], hi)
        words = []
        for z in cutlass.range_constexpr(4):
            rw = rres[z]
            ulo = _bf16_round(
                _add_rn(
                    _bf16_round(acc[2 * z]),
                    (rw << cutlass.Int32(16)).bitcast(cutlass.Float32),
                )
            )
            uhi = _bf16_round(
                _add_rn(
                    _bf16_round(acc[2 * z + 1]),
                    (rw & cutlass.Int32(-65536)).bitcast(cutlass.Float32),
                )
            )
            sqt = _fma_rn(uhi, uhi, _fma_rn(ulo, ulo, sqt))
            words.append(
                (
                    (ulo.bitcast(cutlass.Int32) >> cutlass.Int32(16))
                    & cutlass.Int32(0xFFFF)
                )
                | (uhi.bitcast(cutlass.Int32) & cutlass.Int32(-65536))
            )
        _st_v4_u32(
            rout + cutlass.Int64(q) * cutlass.Int64(16),
            words[0],
            words[1],
            words[2],
            words[3],
        )
    for tt in cutlass.range_constexpr(M_MAX):
        v = cutlass.select_(act & (t == cutlass.Int32(tt)), sqt, cutlass.Float32(0.0))
        for off in cutlass.range_constexpr(5):
            v = _add_rn(v, cute.arch.shuffle_sync_bfly(v, 16 >> off))
        if lane == 0:
            scr.store(v, idx=warp * cutlass.Int32(M_MAX) + cutlass.Int32(tt))
    prims.barrier_cta_sync(0)
    if tidx < cutlass.Int32(M_MAX):
        tot = cutlass.Float32(scr.load(idx=tidx))
        for w_ in cutlass.range_constexpr(1, THREADS // 32):
            tot = _add_rn(
                tot, cutlass.Float32(scr.load(idx=cutlass.Int32(w_ * M_MAX) + tidx))
            )
        _st_relaxed_u32(
            psq + cutlass.Int64(bx * cutlass.Int32(M_MAX) + tidx) * cutlass.Int64(4),
            tot.bitcast(cutlass.Int32),
        )
    prims.barrier_cta_sync(0)
    # ---- grid-wide: every CTA's chunks and partials are out
    if tidx == 0:
        _fence_proxy_async_global()  # residual_out rows are read back by TMA in other CTAs
        old_cnt = _atom_add_release(cnt, 1)
        s_misc.store(old_cnt, idx=1)
        tgt = (old_cnt // cutlass.Int32(NCTA) + cutlass.Int32(1)) * cutlass.Int32(NCTA)
        cv = _ld_acquire_u32(cnt)
        while cv - tgt < cutlass.Int32(0):
            _nanosleep(40)
            cv = _ld_acquire_u32(cnt)
        _fence_proxy_async_global()
        prims.mbarrier_arrive_expect_tx(ar_bar.subview(0), M_MAX * H * 2)
        for tt in cutlass.range_constexpr(M_MAX):
            _bulk_g2s(
                xs32.subview(cutlass.Int32(tt * (H // 2))).data_ptr().toint(),
                rout + cutlass.Int64(tt * H * 2),
                H * 2,
                ar_bar.subview(0).data_ptr().toint(),
            )
    prims.barrier_cta_sync(0)
    # ---- phase B: row sums of squares, then the full normed rows into the x staging
    # (every load of a phase is issued before its first use: the loads are volatile asm,
    # one round trip per phase, not per value)
    PV = (M_MAX + 3) // 4
    pls = []
    for k in cutlass.range_constexpr((NCTA + THREADS - 1) // THREADS):
        ci = tidx + cutlass.Int32(k * THREADS)
        cs = cutlass.select_(ci < cutlass.Int32(NCTA), ci, cutlass.Int32(0))
        for h in cutlass.range_constexpr(PV):
            pls.append(
                _ld_v4_u32(
                    psq
                    + (cutlass.Int64(cs) * cutlass.Int64(M_MAX) + cutlass.Int64(4 * h))
                    * cutlass.Int64(4),
                    False,
                    True,
                )
            )
    gvs = []
    for j in cutlass.range_constexpr(ARIN_J):
        cq = cutlass.select_(
            tidx + cutlass.Int32(j * THREADS) < cutlass.Int32(ARIN_NCC),
            tidx + cutlass.Int32(j * THREADS),
            cutlass.Int32(ARIN_NCC - 1),
        )
        gvs.append(_ld_v4_u32(gamma_ptr + cutlass.Int64(cq) * cutlass.Int64(16)))
    for tt in cutlass.range_constexpr(M_MAX):
        v = cutlass.Float32(0.0)
        for k in cutlass.range_constexpr((NCTA + THREADS - 1) // THREADS):
            ok = tidx + cutlass.Int32(k * THREADS) < cutlass.Int32(NCTA)
            pv = pls[k * PV + tt // 4][tt % 4]
            v = _add_rn(
                v,
                cutlass.select_(ok, pv.bitcast(cutlass.Float32), cutlass.Float32(0.0)),
            )
        for off in cutlass.range_constexpr(5):
            v = _add_rn(v, cute.arch.shuffle_sync_bfly(v, 16 >> off))
        if lane == 0:
            scr.store(v, idx=warp * cutlass.Int32(M_MAX) + cutlass.Int32(tt))
    _wait(ar_bar.subview(0), 0)  # the u rows landed in the x staging
    prims.barrier_cta_sync(0)
    for tt in cutlass.range_constexpr(M_MAX):
        tot = cutlass.Float32(scr.load(idx=cutlass.Int32(tt)))
        for w_ in cutlass.range_constexpr(1, THREADS // 32):
            tot = _add_rn(
                tot, cutlass.Float32(scr.load(idx=cutlass.Int32(w_ * M_MAX + tt)))
            )
        inv = _rsqrt_approx(_add_rn(_div_full(tot, cutlass.Float32(float(H))), eps))
        for j in cutlass.range_constexpr(ARIN_J):
            cc0 = tidx + cutlass.Int32(j * THREADS)
            valid = cc0 < cutlass.Int32(ARIN_NCC)
            cq = cutlass.select_(valid, cc0, cutlass.Int32(ARIN_NCC - 1))
            uv = [
                xs32.load(
                    idx=cutlass.Int32(tt * (H // 2))
                    + cq * cutlass.Int32(4)
                    + cutlass.Int32(z)
                )
                for z in range(4)
            ]
            gv = gvs[j]
            nw = []
            for z in cutlass.range_constexpr(4):
                u = uv[z]
                gw = gv[z]
                glo = _add_rn(
                    cutlass.Float32(1.0),
                    (gw << cutlass.Int32(16)).bitcast(cutlass.Float32),
                )
                ghi = _add_rn(
                    cutlass.Float32(1.0),
                    (gw & cutlass.Int32(-65536)).bitcast(cutlass.Float32),
                )
                nlo = _bf16_round(
                    _mul_rn(
                        _mul_rn((u << cutlass.Int32(16)).bitcast(cutlass.Float32), inv),
                        glo,
                    )
                )
                nhi = _bf16_round(
                    _mul_rn(
                        _mul_rn(
                            (u & cutlass.Int32(-65536)).bitcast(cutlass.Float32), inv
                        ),
                        ghi,
                    )
                )
                nw.append(
                    (
                        (nlo.bitcast(cutlass.Int32) >> cutlass.Int32(16))
                        & cutlass.Int32(0xFFFF)
                    )
                    | (nhi.bitcast(cutlass.Int32) & cutlass.Int32(-65536))
                )
            if valid:
                w0i = cutlass.Int32(tt * (H // 2)) + cq * cutlass.Int32(4)
                for z in cutlass.range_constexpr(4):
                    xs32.store(nw[z], idx=w0i + cutlass.Int32(z))
                if bx == 0:
                    _st_v4_u32(
                        cutlass.Int64(xb32.data_ptr().toint())
                        + cutlass.Int64(w0i) * cutlass.Int64(4),
                        nw[0],
                        nw[1],
                        nw[2],
                        nw[3],
                    )


@cute.jit
def _ar_in_rearm(tidx, bx, desc, s_par: cutlass.Array):
    """AR_IN teardown: re-arm this CTA's share of the consumed slots (every CTA has read them: counter)."""
    buf = cutlass.Int64(_ld_relaxed_u64(desc))
    maxt = cutlass.Int64(_ld_relaxed_u64(desc + cutlass.Int64(24)))
    half = maxt * cutlass.Int64(AR_IN_W * (H // 2))
    total = cutlass.Int32(M_MAX * AR_IN_W * ARIN_NCC)
    q = bx * cutlass.Int32(THREADS) + tidx
    E = cutlass.Int32(ARIN_EMPTY)
    while q < total:
        t = q // cutlass.Int32(AR_IN_W * ARIN_NCC)
        rem = q - t * cutlass.Int32(AR_IN_W * ARIN_NCC)
        r = rem // cutlass.Int32(ARIN_NCC)
        cc = rem - r * cutlass.Int32(ARIN_NCC)
        sl = cc // cutlass.Int32(ARIN_NCC // AR_IN_CL)
        par = cutlass.Int64(s_par.load(idx=t * cutlass.Int32(AR_IN_CL) + sl))
        a0 = buf + (
            par * half
            + (cutlass.Int64(t) * cutlass.Int64(AR_IN_W) + cutlass.Int64(r))
            * cutlass.Int64(H // 2)
            + cutlass.Int64(cc) * cutlass.Int64(4)
        ) * cutlass.Int64(4)
        for z in cutlass.range_constexpr(4):
            _st_relaxed_u32(a0 + cutlass.Int64(4 * z), E)
        q = q + cutlass.Int32(NCTA * THREADS)


@cute.jit
def _pf_next(arp: cutlass.Array, cid, lane, nthr_ctas: cutlass.Constexpr):
    """PF_NEXT: this lane's share of the arp[6] table's L2 prefetches (PF_CHUNK-byte bulk prefetches)."""
    npf = cutlass.Int32(arp.load(idx=7))
    tbl = arp.load(idx=6)
    gid = cutlass.Int64(cid) * cutlass.Int64(32) + cutlass.Int64(lane)
    for si in cutlass.range(0, npf, unroll=1):
        pbase = _ld_relaxed_u64(tbl + cutlass.Int64(si) * cutlass.Int64(16))
        pbytes = _ld_relaxed_u64(
            tbl + cutlass.Int64(si) * cutlass.Int64(16) + cutlass.Int64(8)
        )
        pf_nch = (pbytes + cutlass.Int64(PF_CHUNK - 1)) // cutlass.Int64(PF_CHUNK)
        pf_ci = gid
        while pf_ci < pf_nch:
            pf_rem = pbytes - pf_ci * cutlass.Int64(PF_CHUNK)
            pf_nb = cutlass.Int32(
                cutlass.select_(
                    pf_rem > cutlass.Int64(PF_CHUNK), cutlass.Int64(PF_CHUNK), pf_rem
                )
            )
            _l2_prefetch(pbase + pf_ci * cutlass.Int64(PF_CHUNK), pf_nb)
            pf_ci = pf_ci + cutlass.Int64(nthr_ctas * 32)


@cute.jit
def _shared_cta(
    tidx,
    warp,
    lane,
    sid,
    s_big: cutlass.Array,
    sh_bar: cutlass.Array,
    epochs: cutlass.Array,
    tdbg: cutlass.Array,
    bx,
    xb32: cutlass.Array,
    tma_gu,
    tma_dn,
    gusf: cutlass.Array,
    dnsf: cutlass.Array,
    hx: cutlass.Array,
    dnx: cutlass.Array,
    arp: cutlass.Array,
    tmem_ptr_i32: cutlass.Array,
    gu8: cutlass.Array,
    dn8: cutlass.Array,
):
    """Shared CTA sid (see SHARED): tcgen05 MXF8 (E4M3 x E4M3, UE8M0 per 32 K), m3_mx_gemv's tile order. gate_up: one
    128-row MMA tile = 2 K slices x (32 gate + 32 up rows), B rows 8 q + t = token t's slice-q k-tile; down: 2 tiles
    of 128 rows, full K. Warps: 0 weights (TMA), 1 TMEM + scale splice + MMA, 4-7 epilogue; 2.. activation quant."""
    big32 = cutlass.Array(
        s_big.data_ptr(0), shape=(BIG // 4,), dtype=cutlass.Int32, alignment=16
    )
    bigf = cutlass.Array(
        s_big.data_ptr(0), shape=(BIG // 4,), dtype=cutlass.Float32, alignment=16
    )
    big8 = cutlass.Array(
        s_big.data_ptr(0), shape=(BIG,), dtype=cutlass.Int8, alignment=16
    )
    sh_full = sh_bar  # [0, SH_NST): stage full
    sh_empty = cutlass.Array(
        sh_bar.data_ptr(8), shape=(8,), dtype=cutlass.Int64, alignment=8
    )
    b_gu_rdy = sh_bar.subview(16)
    b_dn_rdy = sh_bar.subview(17)
    acc_gu = sh_bar.subview(18)
    acc_dn = sh_bar.subview(19)
    if warp == 0:
        prims.prefetch_tensormap(tma_gu)
        prims.prefetch_tensormap(tma_dn)
        if lane < SH_NST:
            prims.mbarrier_init(sh_full.subview(lane), 1)
            prims.mbarrier_init(
                sh_empty.subview(lane), 1 if SH_ARR1 else 33
            )  # the MMAs' commit + each splicer lane's reads done (SH_ARR1: the commit only)
        if lane == 0:
            prims.mbarrier_init(b_gu_rdy, 1)
            prims.mbarrier_init(b_dn_rdy, 1)
            prims.mbarrier_init(acc_gu, 1)
            prims.mbarrier_init(acc_dn, 1)
    if warp == 1:
        prims.tcgen05_alloc(tmem_ptr_i32, SH_TMEM)
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.sync_warp()
    prims.barrier_cta_sync(0)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    tmem_raw = tmem_ptr_i32.load()
    base_col = tmem_raw & 0xFFFF
    base_row = tmem_raw >> 16
    if (warp == 0) & cutlass.Boolean(not SH_LATE):
        for g in cutlass.range_constexpr(
            SH_NST
        ):  # layer constants: before the grid dependency
            _sh_issue(
                lane, sid, cutlass.Int32(g), s_big, sh_full, tma_gu, tma_dn, gusf, dnsf
            )
        # the rest of this CTA's weights into L2 (64 gate / up rows of 6 KB, 256 down
        # rows of 768 B) (AR_IN, cfg sh_pf=0: skipped, so the 29 MB flood does not delay
        # the boundary reduce's loads)
        if cutlass.const_expr(not SH_PF_ON):
            pass
        elif cutlass.const_expr(SH_PRETILE):
            # this CTA's stages beyond the ring, 4 per lane
            for r in cutlass.range_constexpr(SH_PF):
                pg = (
                    cutlass.Int32(SH_NST)
                    + lane * cutlass.Int32(SH_PF)
                    + cutlass.Int32(r)
                )
                if pg < cutlass.Int32(SH_STAGES):
                    _l2_prefetch(
                        gusf.subview(
                            (sid * cutlass.Int32(SH_STAGES) + pg)
                            * cutlass.Int32(SH_STAGE_BYTES)
                        )
                        .data_ptr()
                        .toint(),
                        SH_STAGE_BYTES,
                    )
        else:
            for r in cutlass.range_constexpr(2):
                prw = sid * cutlass.Int32(32) + cutlass.Int32(r * SH_I) + lane
                _l2_prefetch(gu8.subview(prw * cutlass.Int32(H)).data_ptr().toint(), H)
            if lane == 0:
                _l2_prefetch(
                    dn8.subview(sid * cutlass.Int32(SH_DROWS * SH_I))
                    .data_ptr()
                    .toint(),
                    SH_DROWS * SH_I,
                )
    cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(SH_LATE):
        if warp == 0:
            for g in cutlass.range_constexpr(
                SH_NST
            ):  # SH_LATE: the ring's first stages after the grid dependency
                _sh_issue(
                    lane,
                    sid,
                    cutlass.Int32(g),
                    s_big,
                    sh_full,
                    tma_gu,
                    tma_dn,
                    gusf,
                    dnsf,
                )
    if cutlass.const_expr(not LATE_TRIGGER):
        if arp.load(idx=0) == cutlass.Int64(0):
            cute.arch.griddepcontrol_launch_dependents()
    if cutlass.const_expr(SH_TRIG):
        if arp.load(idx=0) != cutlass.Int64(
            0
        ):  # E-T: every thread triggers after its own wait above
            cute.arch.griddepcontrol_launch_dependents()
    ep = epochs.load(idx=bx)
    tag16 = (ep & cutlass.Int32(0x7FFF)) | cutlass.Int32(0x8000)
    NQ = THREADS - 64  # activation quant threads (warps 2 ..)
    if warp == 0:
        for g in cutlass.range(SH_NST, SH_STAGES, unroll=1):
            _wait(
                sh_empty.subview(g % cutlass.Int32(SH_NST)),
                ((g // cutlass.Int32(SH_NST)) - cutlass.Int32(1)) & cutlass.Int32(1),
            )
            _sh_issue(lane, sid, g, s_big, sh_full, tma_gu, tma_dn, gusf, dnsf)
    elif warp == 1:
        # ---- MMA: gate_up (scale atoms spliced to the tile's row order), then down
        idesc_gu = prims.Tcgen05MxInstrDesc.build(
            a_dtype=cutlass.Float8E4M3FN,
            b_dtype=cutlass.Float8E4M3FN,
            scale_format=1,
            n_dim=SH_NB,
            m_dim=128,
        )
        idesc_dn = prims.Tcgen05MxInstrDesc.build(
            a_dtype=cutlass.Float8E4M3FN,
            b_dtype=cutlass.Float8E4M3FN,
            scale_format=1,
            n_dim=8,
            m_dim=128,
        )
        s2t_shape, s2t_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
        desc_a0 = prims.Tcgen05SmemDesc.build(
            s_big.subview(0),
            leading_byte_offset=16,
            stride_byte_offset=1024,
            base_offset=0,
            layout=2,
        )
        desc_bx = prims.Tcgen05SmemDesc.build(
            s_big.subview(_SB_BX),
            leading_byte_offset=16,
            stride_byte_offset=1024,
            base_offset=0,
            layout=2,
        )
        desc_bh = prims.Tcgen05SmemDesc.build(
            s_big.subview(_SB_BH),
            leading_byte_offset=16,
            stride_byte_offset=1024,
            base_offset=0,
            layout=2,
        )
        tcol = (base_row << 16) | base_col
        sq = sid & cutlass.Int32(3)
        _wait(b_gu_rdy, 0)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        for g in cutlass.range(SH_STAGES, unroll=1):
            st = g % cutlass.Int32(SH_NST)
            if g == cutlass.Int32(SH_GT):
                _wait(b_dn_rdy, 0)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            _wait(sh_full.subview(st), (g // cutlass.Int32(SH_NST)) & cutlass.Int32(1))
            raw = cutlass.Int32(_SB_RAW // 4) + st * cutlass.Int32(512)  # type: ignore[operator]
            if (g < cutlass.Int32(SH_GT)) & cutlass.Boolean(not SH_PRETILE):
                # MMA row 32 Q + i = quadrant Q's row i: word 4 i + Q <- raw atom Q's word 4 i + sid % 4
                w4 = [
                    big32.load(
                        idx=raw + cutlass.Int32(128 * qq) + lane * cutlass.Int32(4) + sq
                    )
                    for qq in range(4)
                ]
                big32.store(
                    (
                        cutlass.Int32(w4[0]),
                        cutlass.Int32(w4[1]),
                        cutlass.Int32(w4[2]),
                        cutlass.Int32(w4[3]),
                    ),
                    idx=cutlass.Int32(_SB_SPL // 4)
                    + st * cutlass.Int32(128)
                    + lane * cutlass.Int32(4),
                    alignment=16,
                )
                prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
            cute.arch.sync_warp()
            if cutlass.const_expr(not SH_ARR1):
                prims.mbarrier_arrive(
                    sh_empty.subview(st)
                )  # this lane's generic reads of the slot are done
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if prims.elect_sync():
                if g < cutlass.Int32(SH_GT):
                    dsf = prims.Tcgen05SmemDesc.build(
                        s_big.subview(cutlass.Int32(_SB_SPL) + st * cutlass.Int32(512)),
                        leading_byte_offset=16,
                        stride_byte_offset=128,
                        base_offset=0,
                        layout=0,
                    )
                    prims.tcgen05_cp(
                        s2t_shape,
                        _tptr(base_row, base_col + SH_SFA + st * cutlass.Int32(4)),
                        dsf,
                        multicast=s2t_mc,
                    )
                    da = desc_a0 + cutlass.Int64(st * cutlass.Int32(NUM_BYTES_A >> 4))
                    db = desc_bx + cutlass.Int64(g * cutlass.Int32(SH_NB * 128 >> 4))
                    for kb in cutlass.range_constexpr(4):
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF8F6F4,
                            prims.CTAGroup.CTA_1,
                            cutlass.inttoptr(tcol + SH_ACC_GU, 6, cutlass.Float32),
                            da + 2 * kb,
                            db + 2 * kb,
                            idesc_gu.set_sf_ids(a_sf_id=kb, b_sf_id=kb),
                            (g > cutlass.Int32(0)) | cutlass.Boolean(kb > 0),
                            cutlass.inttoptr(
                                tcol + SH_SFA + st * cutlass.Int32(4), 6, cutlass.Int32
                            ),
                            cutlass.inttoptr(
                                tcol + SH_SFB_GU + g * cutlass.Int32(4),
                                6,
                                cutlass.Int32,
                            ),
                        )
                    prims.tcgen05_commit(sh_empty.subview(st))
                    if g == cutlass.Int32(SH_GT - 1):
                        prims.tcgen05_commit(acc_gu)
                else:
                    gd = g - cutlass.Int32(SH_GT)
                    jt = gd // cutlass.Int32(SH_DKT)
                    kt = gd - jt * cutlass.Int32(SH_DKT)
                    dsf = prims.Tcgen05SmemDesc.build(
                        s_big.subview(
                            cutlass.Int32(_SB_SPL if SH_PRETILE else _SB_RAW)
                            + st * cutlass.Int32(512 if SH_PRETILE else 2048)
                        ),
                        leading_byte_offset=16,
                        stride_byte_offset=128,
                        base_offset=0,
                        layout=0,
                    )
                    prims.tcgen05_cp(
                        s2t_shape,
                        _tptr(base_row, base_col + SH_SFA + st * cutlass.Int32(4)),
                        dsf,
                        multicast=s2t_mc,
                    )
                    da = desc_a0 + cutlass.Int64(st * cutlass.Int32(NUM_BYTES_A >> 4))
                    db = desc_bh + cutlass.Int64(kt * cutlass.Int32(1024 >> 4))
                    for kb in cutlass.range_constexpr(4):
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF8F6F4,
                            prims.CTAGroup.CTA_1,
                            cutlass.inttoptr(
                                tcol + SH_ACC_DN + jt * cutlass.Int32(8),
                                6,
                                cutlass.Float32,
                            ),
                            da + 2 * kb,
                            db + 2 * kb,
                            idesc_dn.set_sf_ids(a_sf_id=kb, b_sf_id=kb),
                            (kt > cutlass.Int32(0)) | cutlass.Boolean(kb > 0),
                            cutlass.inttoptr(
                                tcol + SH_SFA + st * cutlass.Int32(4), 6, cutlass.Int32
                            ),
                            cutlass.inttoptr(
                                tcol + SH_SFB_DN + kt * cutlass.Int32(4),
                                6,
                                cutlass.Int32,
                            ),
                        )
                    prims.tcgen05_commit(sh_empty.subview(st))
                    if g == cutlass.Int32(SH_STAGES - 1):
                        prims.tcgen05_commit(acc_dn)
            cute.arch.sync_warp()
    if warp >= 2:
        if cutlass.const_expr(AR_IN):
            if arp.load(idx=6) != cutlass.Int64(0):
                # AR_IN: routed CTA 0 writes the normed hidden states; wait for this call's release
                xra = cutlass.Int64(
                    _ld_relaxed_u64(arp.load(idx=6) + cutlass.Int64(72))
                )
                if (
                    lane == 0
                ):  # one poller per warp, backed off (the counter's L2 line is shared by every routed CTA)
                    xv = _ld_acquire_u32(xra)
                    while xv - (ep + cutlass.Int32(1)) < cutlass.Int32(0):
                        _nanosleep(100)
                        xv = _ld_acquire_u32(xra)
                cute.arch.sync_warp()
        # ---- the activation, MXFP8 per 32-K block (native mxfp8_quantize), into B (row
        # 8 q + t of stage tile i = token t's k-tile SH_GT q + i, 128-byte swizzle) and
        # its codes
        qt = tidx - cutlass.Int32(64)
        for r in cutlass.range_constexpr((M_MAX * SH_KB + NQ - 1) // NQ):
            q_ = qt + cutlass.Int32(r * NQ)
            if q_ < cutlass.Int32(M_MAX * SH_KB):
                t = q_ // cutlass.Int32(SH_KB)
                b = q_ - t * cutlass.Int32(SH_KB)
                wv = [
                    xb32.load(
                        idx=t * cutlass.Int32(H // 2)
                        + b * cutlass.Int32(16)
                        + cutlass.Int32(4 * z),
                        vector_size=4,
                        alignment=16,
                    )
                    for z in range(4)
                ]
                v = []
                for z in cutlass.range_constexpr(4):
                    for zz in cutlass.range_constexpr(4):
                        w = cutlass.Int32(wv[z][zz])
                        v.append((w << cutlass.Int32(16)).bitcast(cutlass.Float32))
                        v.append((w & cutlass.Int32(-65536)).bitcast(cutlass.Float32))
                amax = cutlass.Float32(0.0)
                for z in cutlass.range_constexpr(32):
                    amax = cute.arch.fmax(amax, cute.math.abs(v[z]))
                code, osc = _mx_osc(amax)
                words = _mx_words(v, osc)
                kt = b >> cutlass.Int32(2)
                sl = kt // cutlass.Int32(SH_GT)
                it = kt - sl * cutlass.Int32(SH_GT)
                n = sl * cutlass.Int32(8) + t
                c0 = (b & cutlass.Int32(3)) * cutlass.Int32(2)
                rw = (
                    cutlass.Int32(_SB_BX)
                    + it * cutlass.Int32(SH_NB * 128)
                    + n * cutlass.Int32(128)
                ) >> cutlass.Int32(2)
                big32.store(
                    (words[0], words[1], words[2], words[3]),
                    idx=rw + ((c0 ^ (n & cutlass.Int32(7))) << cutlass.Int32(2)),
                    alignment=16,
                )
                big32.store(
                    (words[4], words[5], words[6], words[7]),
                    idx=rw
                    + (
                        ((c0 + cutlass.Int32(1)) ^ (n & cutlass.Int32(7)))
                        << cutlass.Int32(2)
                    ),
                    alignment=16,
                )
                big8.store(cutlass.Int8(code), idx=cutlass.Int32(_SB_XC) + q_)
        prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
        cute.arch.sync_warp()
        cute.arch.barrier(barrier_id=SHX_BAR, number_of_threads=NQ)
    if warp >= 4:
        if warp < 8:
            qd = warp - cutlass.Int32(4)
            # B scales of gate_up to TMEM: column of stage i, lane n = 8 q + t: token t's codes of k-tile SH_GT q + i
            n = lane
            t = n & cutlass.Int32(7)
            ok = (n < cutlass.Int32(SH_NB)) & (t < cutlass.Int32(M_MAX))
            for i in cutlass.range_constexpr(SH_GT):
                kt = (n >> cutlass.Int32(3)) * cutlass.Int32(SH_GT) + cutlass.Int32(i)
                wsf = big32.load(
                    idx=(
                        cutlass.Int32(_SB_XC)
                        + cutlass.select_(ok, t, cutlass.Int32(0))
                        * cutlass.Int32(SH_KB)
                        + kt * cutlass.Int32(4)
                    )
                    >> cutlass.Int32(2)
                )
                prims.tcgen05_st(
                    "32x32b",
                    _tptr(
                        base_row + qd * cutlass.Int32(32), base_col + SH_SFB_GU + i * 4
                    ),
                    cutlass.select_(ok, cutlass.Int32(wsf), cutlass.Int32(0)),
                )
            prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
            et = tidx - cutlass.Int32(128)
            if et == 0:
                prims.mbarrier_arrive(b_gu_rdy)
            # ---- gate_up out: quadrant qd lane l = MMA row 32 qd + l; slice q = qd / 2 holds token t at column 8 q + t
            _wait(acc_gu, 0)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if cutlass.const_expr(SH_SPLIT == 1):
                # (TP2): one K slice; warp qd holds MMA rows 32 qd + l = gate (qd < 2) /
                # up (qd >= 2) position 32 (qd % 2) + l, i.e. GU row 32 qd + l;
                # m3_mx_gemv's owner sum (0 + rank 0), then bf16
                av = prims.tcgen05_ld(
                    "32x32b",
                    _tptr(
                        base_row + qd * cutlass.Int32(32),
                        base_col + SH_ACC_GU,
                        cutlass.Float32,
                    ),
                    num=8,
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                for tt in cutlass.range_constexpr(8):
                    bigf.store(
                        _bf16_round(
                            _add_rn(cutlass.Float32(0.0), cutlass.Float32(av[tt]))
                        ),
                        idx=cutlass.Int32(_SB_GU // 4)
                        + (qd * cutlass.Int32(32) + lane) * cutlass.Int32(8)
                        + cutlass.Int32(tt),
                    )
                cute.arch.sync_warp()
                cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
            elif cutlass.const_expr(SH_SPLIT == 2):
                av = prims.tcgen05_ld(
                    "32x32b",
                    _tptr(
                        base_row + qd * cutlass.Int32(32),
                        base_col + SH_ACC_GU,
                        cutlass.Float32,
                    ),
                    num=16,
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                if qd >= cutlass.Int32(2):
                    for tt in cutlass.range_constexpr(8):
                        bigf.store(
                            cutlass.Float32(av[8 + tt]),
                            idx=cutlass.Int32(_SB_XCH // 4)
                            + ((qd - cutlass.Int32(2)) * cutlass.Int32(32) + lane)
                            * cutlass.Int32(8)
                            + cutlass.Int32(tt),
                        )
                cute.arch.sync_warp()
                cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
                if qd < cutlass.Int32(2):
                    # m3_mx_gemv's owner sum: (0 + rank 0) + rank 1, then bf16
                    for tt in cutlass.range_constexpr(8):
                        tot = _add_rn(cutlass.Float32(0.0), cutlass.Float32(av[tt]))
                        tot = _add_rn(
                            tot,
                            bigf.load(
                                idx=cutlass.Int32(_SB_XCH // 4)
                                + (qd * cutlass.Int32(32) + lane) * cutlass.Int32(8)
                                + cutlass.Int32(tt)
                            ),
                        )
                        bigf.store(
                            _bf16_round(tot),
                            idx=cutlass.Int32(_SB_GU // 4)
                            + (qd * cutlass.Int32(32) + lane) * cutlass.Int32(8)
                            + cutlass.Int32(tt),
                        )
                cute.arch.sync_warp()
                cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
            else:
                # 16-row octants: warp qd = slice qd, lane l = (u = l / 16 gate / up,
                # position l % 16); its tokens are accumulator columns 8 qd + t
                av = prims.tcgen05_ld(
                    "32x32b",
                    _tptr(
                        base_row + qd * cutlass.Int32(32),
                        base_col + SH_ACC_GU + qd * cutlass.Int32(8),
                        cutlass.Float32,
                    ),
                    num=8,
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                if qd >= cutlass.Int32(1):
                    for tt in cutlass.range_constexpr(8):
                        bigf.store(
                            cutlass.Float32(av[tt]),
                            idx=cutlass.Int32(_SB_XCH // 4)
                            + ((qd - cutlass.Int32(1)) * cutlass.Int32(32) + lane)
                            * cutlass.Int32(8)
                            + cutlass.Int32(tt),
                        )
                cute.arch.sync_warp()
                cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
                if qd == cutlass.Int32(0):
                    # m3_mx_gemv's owner sum: (((0 + rank 0) + rank 1) + rank 2) + rank 3, then bf16
                    for tt in cutlass.range_constexpr(8):
                        tot = _add_rn(cutlass.Float32(0.0), cutlass.Float32(av[tt]))
                        for s_ in cutlass.range_constexpr(1, SH_SPLIT):
                            tot = _add_rn(
                                tot,
                                bigf.load(
                                    idx=cutlass.Int32(_SB_XCH // 4)
                                    + (cutlass.Int32((s_ - 1) * 32) + lane)
                                    * cutlass.Int32(8)
                                    + cutlass.Int32(tt)
                                ),
                            )
                        bigf.store(
                            _bf16_round(tot),
                            idx=cutlass.Int32(_SB_GU // 4)
                            + lane * cutlass.Int32(8)
                            + cutlass.Int32(tt),
                        )
                cute.arch.sync_warp()
                cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
            # h = silu_and_mul(gate, up) of positions 32 sid + p, published under the call's tag
            for r in cutlass.range_constexpr((SH_ROWS * M_MAX + 127) // 128):
                pq = et + cutlass.Int32(r * 128)
                if pq < cutlass.Int32(SH_ROWS * M_MAX):
                    p = pq // cutlass.Int32(M_MAX)
                    tt = pq - p * cutlass.Int32(M_MAX)
                    ga = bigf.load(
                        idx=cutlass.Int32(_SB_GU // 4) + p * cutlass.Int32(8) + tt
                    )
                    up = bigf.load(
                        idx=cutlass.Int32(_SB_GU // 4)
                        + (cutlass.Int32(SH_ROWS) + p) * cutlass.Int32(8)
                        + tt
                    )
                    q_a = cute.arch.fmin(cutlass.Float32(ga), cutlass.Float32(7.0))
                    q_bu = _min_xorsign_abs(cutlass.Float32(up), cutlass.Float32(7.0))
                    q_ub = _add_rn(q_bu, cutlass.Float32(1.0))
                    if cutlass.const_expr(SILU_VLLM):
                        q_ex = cute.math.exp(
                            _mul_rn(q_a, cutlass.Float32(-1.70200002193450927734)),
                            fastmath=False,
                        )
                        q_s = _bf16_round(
                            _div_rn(
                                q_a,
                                _add_rn(cutlass.Float32(1.0), cutlass.Float32(q_ex)),
                            )
                        )
                        q_h = _mul_rn(q_s, q_ub)
                    else:
                        q_e2 = _ex2_full(
                            _mul_rn(
                                _mul_rn(q_a, cutlass.Float32(-1.70200002193450927734)),
                                cutlass.Float32(1.4426950216293334961),
                            )
                        )
                        q_sg = _div_full(
                            cutlass.Float32(1.0), _add_rn(q_e2, cutlass.Float32(1.0))
                        )
                        q_h = _mul_rn(q_ub, _mul_rn(q_sg, q_a))
                    _st_relaxed_u32(
                        hx.subview(
                            tt * cutlass.Int32(SH_I) + sid * cutlass.Int32(SH_ROWS) + p
                        )
                        .data_ptr()
                        .toint(),
                        (tag16 << cutlass.Int32(16)) | _bf16_bits(q_h),
                    )
            # every shared CTA's h: (token, K block) per thread, 32 tagged words, MXFP8 into B (h) and its codes
            for rr in cutlass.range_constexpr((M_MAX * SH_DKB + 127) // 128):
                q_ = et + cutlass.Int32(rr * 128)
                if q_ < cutlass.Int32(M_MAX * SH_DKB):
                    t = q_ // cutlass.Int32(SH_DKB)
                    kb = q_ - t * cutlass.Int32(SH_DKB)
                    base = t * cutlass.Int32(SH_I) + kb * cutlass.Int32(32)
                    hv = [cutlass.Int32(0) for _ in range(32)]
                    pend = cutlass.Int32(1)
                    hd0 = cutlass.Int32(0)
                    while (pend != cutlass.Int32(0)) & (hd0 < cutlass.Int32(HANG_CAP)):
                        hd0 = hd0 + cutlass.Int32(1)
                        for q4 in cutlass.range_constexpr(8):
                            v4 = prims.load_ext(
                                hx.subview(base + cutlass.Int32(4 * q4)),
                                dtype=cutlass.Int32,
                                count=4,
                                order="relaxed",
                                scope="gpu",
                            )
                            for z in cutlass.range_constexpr(4):
                                hv[4 * q4 + z] = cutlass.Int32(v4[z])
                        okw = cutlass.Boolean(True)
                        for z in cutlass.range_constexpr(32):
                            okw = okw & (
                                ((hv[z] >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF))
                                == tag16
                            )
                        pend = cutlass.select_(okw, cutlass.Int32(0), cutlass.Int32(1))
                    vals = [
                        (hv[z] << cutlass.Int32(16)).bitcast(cutlass.Float32)
                        for z in range(32)
                    ]
                    amax = cutlass.Float32(0.0)
                    for z in cutlass.range_constexpr(32):
                        amax = cute.arch.fmax(amax, cute.math.abs(vals[z]))
                    code, osc = _mx_osc(amax)
                    words = _mx_words(vals, osc)
                    kt = kb >> cutlass.Int32(2)
                    c0 = (kb & cutlass.Int32(3)) * cutlass.Int32(2)
                    rw = (
                        cutlass.Int32(_SB_BH)
                        + kt * cutlass.Int32(1024)
                        + t * cutlass.Int32(128)
                    ) >> cutlass.Int32(2)
                    big32.store(
                        (words[0], words[1], words[2], words[3]),
                        idx=rw + ((c0 ^ t) << cutlass.Int32(2)),
                        alignment=16,
                    )
                    big32.store(
                        (words[4], words[5], words[6], words[7]),
                        idx=rw + (((c0 + cutlass.Int32(1)) ^ t) << cutlass.Int32(2)),
                        alignment=16,
                    )
                    big8.store(cutlass.Int8(code), idx=cutlass.Int32(_SB_HC) + q_)
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
            okh = lane < cutlass.Int32(M_MAX)
            for kt in cutlass.range_constexpr(SH_DKT):
                wsf = big32.load(
                    idx=(
                        cutlass.Int32(_SB_HC)
                        + cutlass.select_(okh, lane, cutlass.Int32(0))
                        * cutlass.Int32(SH_DKB)
                        + cutlass.Int32(4 * kt)
                    )
                    >> cutlass.Int32(2)
                )
                prims.tcgen05_st(
                    "32x32b",
                    _tptr(
                        base_row + qd * cutlass.Int32(32), base_col + SH_SFB_DN + kt * 4
                    ),
                    cutlass.select_(okh, cutlass.Int32(wsf), cutlass.Int32(0)),
                )
            prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
            if et == 0:
                prims.mbarrier_arrive(b_dn_rdy)
            # ---- down out: tile j row 32 qd + l -> y_sh row 256 sid + 128 j + 32 qd + l
            _wait(acc_dn, 0)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for jt in cutlass.range_constexpr(SH_DT):
                dv = prims.tcgen05_ld(
                    "32x32b",
                    _tptr(
                        base_row + qd * cutlass.Int32(32),
                        base_col + SH_ACC_DN + jt * 8,
                        cutlass.Float32,
                    ),
                    num=8,
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                rowo = (
                    sid * cutlass.Int32(SH_DROWS)
                    + cutlass.Int32(128 * jt)
                    + qd * cutlass.Int32(32)
                    + lane
                )
                for tt in cutlass.range_constexpr(M_MAX):
                    _st_relaxed_u32(
                        dnx.subview(cutlass.Int32(tt * H) + rowo).data_ptr().toint(),
                        (tag16 << cutlass.Int32(16))
                        | _bf16_bits(cutlass.Float32(dv[tt])),
                    )
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.sync_warp()
    prims.barrier_cta_sync(0)
    if warp == 1:
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        prims.tcgen05_dealloc(cutlass.inttoptr(tmem_raw, 6, cutlass.Float32), SH_TMEM)
    if tidx == 0:
        epochs.store(ep + cutlass.Int32(1), idx=bx)
    if cutlass.const_expr(PF_NEXT and PF_WHERE == 1):
        if warp == 0:
            _pf_next(arp, sid, lane, NSH)


@cute.kernel
def m3_moe_kernel(
    tma_a1_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_a2_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_s1_desc: cutlass.GridConstant[
        cuda.TensorMap
    ],  # w3_w1_weight_scale as 128-byte lines
    tma_s2_desc: cutlass.GridConstant[
        cuda.TensorMap
    ],  # w2_weight_scale as 128-byte lines
    tma_gu_desc: cutlass.GridConstant[
        cuda.TensorMap
    ],  # SHARED: gate_up e4m3 [1536, 6144], box 128 B x 32 rows
    tma_dn_desc: cutlass.GridConstant[
        cuda.TensorMap
    ],  # SHARED: down e4m3 [6144, 768], box 128 B x 128 rows
    logits: cutlass.Array,  # f32 [M, 128] (not FUSED_ROUTER); B1_X4: int32 words of the post's x4 values [M, 3072 B]
    bias: cutlass.Array,  # f32 [128]
    x32: cutlass.Array,  # int32 words of x_fp4 [M, 3072 B] (not FUSED_ROUTER)
    xsf32: cutlass.Array,  # int32 words of x_sf [M, 384 B] (not FUSED_ROUTER; B1_X4: the post's x4 scales)
    escal: cutlass.Array,  # f32 [6, 128]: fc31_alpha, fc31_scale_c, act_alpha, act_beta, act_clamp, fc2_alpha
    plan: cutlass.Array,  # int32 [4 M + 1, NCTA, NG_CAP + 1] by G: per group i0 | i1 << 16, then b0 | nb << 16
    partials: cutlass.Array,  # int32 bits of f32 [G_CAP * 48, M, 32]: split units' head sums
    hbuf32: cutlass.Array,  # int32 [G_CAP * M, 48, 4]: tagged intermediate records
    epochs: cutlass.Array,  # int32 [NCTA]
    out: cutlass.Array,  # bf16 [M, 6144]
    route_dbg: cutlass.Array,  # int32 [M, 8]: ids, weight bits (CTA 0)
    tdbg: cutlass.Array,  # int64 [NCTA, TD_STRIDE] scratch (unused in shipped builds)
    xb32: cutlass.Array,  # FUSED_ROUTER: int32 words (bf16 pairs) of the hidden states [M, 6144]
    rw: cutlass.Array,  # FUSED_ROUTER: f32 router weight [128, 6144]
    sfsc: cutlass.Array,  # FUSED_ROUTER: f32 [1] fc31_input_scale
    lgbuf: cutlass.Array,  # FUSED_ROUTER: int64 [LG_REPL, LG_STRIDE] logits: tag << 32 | f32 bits
    guw: cutlass.Array,  # SHARED: uint8 gate_up e4m3 [1536, 6144]
    gusf: cutlass.Array,  # SHARED: uint8 its swizzled UE8M0 scales
    dnw: cutlass.Array,  # SHARED: uint8 down e4m3 [6144, 768]
    dnsf: cutlass.Array,  # SHARED: uint8 its swizzled UE8M0 scales
    hx: cutlass.Array,  # SHARED: int32 [M, 768]: h as (call tag << 16 | bf16), tag 0 = never written
    dnx: cutlass.Array,  # SHARED: int32 [M, 6144]: y_sh as (call tag << 16 | bf16), tag 0 = never written
    arp: cutlass.Array,  # int64 [8]: all-reduce push (enable, mc, flags, rank, world, maxt); enable 0 = off
    s1w: cutlass.Array,  # SFA_CP: int32 words of w3_w1_weight_scale
    s2w: cutlass.Array,  # SFA_CP: int32 words of w2_weight_scale
    img32: cutlass.Array,  # B2IMG: int32 [NG_CAP, IMG_BYTES / 4] per-group FC2 B image (tile + scale atoms)
    b2cnt: cutlass.Array,  # B2IMG: int32 [NG_CAP * 8 + 8] per-group unit counters (index 8 g)
):
    tidx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    s_big = cutlass.Array(
        cutlass.Int8, BIG, space=cutlass.AddressSpace.smem, alignment=1024
    )
    sA = cutlass.Array(
        s_big.data_ptr(0),
        shape=(NUM_BYTES_A * STAGES,),
        dtype=cutlass.Int8,
        alignment=1024,
    )
    sSFAraw = cutlass.Array(
        s_big.data_ptr(_OFF_SFAR),
        shape=(max(1, _RAW_STAGE // 4) * STAGES,),
        dtype=cutlass.Int32,
        alignment=1024,
    )
    sSFA = cutlass.Array(
        s_big.data_ptr(_OFF_SFA),
        shape=(NUM_WORDS_SFA * STAGES,),
        dtype=cutlass.Int32,
        alignment=1024,
    )
    sB1 = cutlass.Array(
        s_big.data_ptr(_OFF_B1),
        shape=(NUM_BYTES_B1 * S1,),
        dtype=cutlass.Int8,
        alignment=1024,
    )
    sB2 = cutlass.Array(
        s_big.data_ptr(_OFF_B2),
        shape=(NB2BUF * KT2 * NUM_BYTES_B2,),
        dtype=cutlass.Int8,
        alignment=1024,
    )
    sSFB2 = cutlass.Array(
        s_big.data_ptr(_OFF_SFB2),
        shape=(NB2BUF * KT2 * NUM_KB * 128,),
        dtype=cutlass.Int32,
        alignment=1024,
    )
    ys = cutlass.Array(
        cutlass.Float32,
        NB2_CAP * M_MAX * TOP_K * R2,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    s_part = cutlass.Array(
        cutlass.Float32,
        2 * M_MAX * C1 * R1,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    s_bias = cutlass.Array(
        cutlass.Float32, E_LOCAL, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_red = cutlass.Array(
        cutlass.Float32,
        (13 if ROUTER_VLLM else 9) * M_MAX,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    s_xsf = cutlass.Array(
        cutlass.Int32,
        M_MAX * (H // 64) if FUSED_ROUTER else 4,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    ab_full = cutlass.Array(
        cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8
    )
    ab_empty = cutlass.Array(
        cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8
    )
    sfa_ready = cutlass.Array(
        cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8
    )
    acc_full = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    acc_empty = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    acc2_full = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    acc2_empty = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    b2_ready = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    b2_free = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    # this CTA's FC1 share of group g is out: one barrier per group parity (a single
    # barrier could complete twice while warp 5 still works on an earlier group, and its
    # parity wait would then miss them)
    fc1_done = cutlass.Array(
        cutlass.Int64, 4, space=cutlass.AddressSpace.smem, alignment=8
    )
    b1_ready = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    x_bar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )  # x landed, x consumed
    ar_bar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )  # AR_IN u rows
    q2_bar = cutlass.Array(
        cutlass.Int64, 2 * Q2D, space=cutlass.AddressSpace.smem, alignment=8
    )  # SF2_W6 full/empty
    q2_ent = cutlass.Array(
        cutlass.Int32, 4 * Q2D, space=cutlass.AddressSpace.smem, alignment=16
    )
    img_bar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )  # (B2IMG) image landed
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    s_el = cutlass.Array(
        cutlass.Int32, G_CAP + 1, space=cutlass.AddressSpace.smem, alignment=16
    )  # slots, then G
    s_kpos = cutlass.Array(
        cutlass.Int32, G_CAP * M_MAX, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_w = cutlass.Array(
        cutlass.Float32, M_MAX * TOP_K, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_ids = cutlass.Array(
        cutlass.Int32, M_MAX * TOP_K, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_plan = cutlass.Array(
        cutlass.Int32, NG_CAP + 1, space=cutlass.AddressSpace.smem, alignment=16
    )
    # this CTA's plan rows for every G (static data: loaded before the grid dependency)
    s_misc = cutlass.Array(
        cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_par = cutlass.Array(
        cutlass.Int32, 32, space=cutlass.AddressSpace.smem, alignment=16
    )  # AR_IN parities
    s_plan_all = cutlass.Array(
        cutlass.Int32,
        (G_CAP + 1) * (NG_CAP + 1),
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    if cutlass.const_expr(FC1_PER_TOKEN):
        s_pts = cutlass.Array(
            cutlass.Float32, 8, space=cutlass.AddressSpace.smem, alignment=16
        )  # s_t
    s_esc = cutlass.Array(
        cutlass.Float32, 8 * G_CAP, space=cutlass.AddressSpace.smem, alignment=32
    )  # slot s at 8 s

    is_routed = cutlass.Boolean(True)
    if cutlass.const_expr(SHARED):
        sh_full = cutlass.Array(
            cutlass.Int64, 32, space=cutlass.AddressSpace.smem, alignment=8
        )
        is_routed = bx < cutlass.Int32(NCTA)
        if bx >= cutlass.Int32(NCTA):
            _shared_cta(
                tidx,
                warp,
                lane,
                bx - cutlass.Int32(NCTA),
                s_big,
                sh_full,
                epochs,
                tdbg,
                bx,
                xb32,
                tma_gu_desc.get_ptr(),
                tma_dn_desc.get_ptr(),
                gusf,
                dnsf,
                hx,
                dnx,
                arp,
                tmem_ptr_i32,
                guw,
                dnw,
            )
    if is_routed:
        # ---- before the grid dependency
        if warp == 0:
            if tidx < STAGES:
                prims.mbarrier_init(ab_full.subview(tidx), 33 if SFA_CP else 1)
                prims.mbarrier_init(ab_empty.subview(tidx), 1)
                prims.mbarrier_init(sfa_ready.subview(tidx), 1)
            if tidx < 2:
                prims.mbarrier_init(acc_full.subview(tidx), 1)
                prims.mbarrier_init(acc_empty.subview(tidx), EPI_THREADS)
                prims.mbarrier_init(acc2_full.subview(tidx), 1)
                prims.mbarrier_init(acc2_empty.subview(tidx), EPI_THREADS)
                prims.mbarrier_init(b2_ready.subview(tidx), 1)
                prims.mbarrier_init(b2_free.subview(tidx), EPI_THREADS)
                if cutlass.const_expr(B2IMG):
                    prims.mbarrier_init(img_bar.subview(tidx), 1)
            if tidx < 4:
                prims.mbarrier_init(fc1_done.subview(tidx), EPI_THREADS)
            if cutlass.const_expr(SF2_W6):
                if tidx < 2 * Q2D:
                    prims.mbarrier_init(q2_bar.subview(tidx), 1)
            if tidx == 0:
                prims.mbarrier_init(b1_ready.subview(0), EPI_THREADS)
                if cutlass.const_expr(XSTAGE):
                    prims.mbarrier_init(x_bar.subview(0), 1)
                    prims.mbarrier_init(x_bar.subview(1), EPI_THREADS)
                    prims.mbarrier_init(ar_bar.subview(0), 1)
        if warp == MMA_WARP:
            prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
            prims.tcgen05_relinquish_alloc_permit()
        if warp == W_WARP:
            prims.prefetch_tensormap(tma_a1_desc.get_ptr())
            prims.prefetch_tensormap(tma_a2_desc.get_ptr())
            prims.prefetch_tensormap(tma_s1_desc.get_ptr())
            prims.prefetch_tensormap(tma_s2_desc.get_ptr())
            for i in cutlass.range_constexpr(
                0 if EXT_ROUTE else E_LOCAL // 32
            ):  # (ext_route) no top-k here
                s_bias.store(
                    cutlass.Float32(bias.load(idx=lane + i * 32)), idx=lane + i * 32
                )
            for i in cutlass.range_constexpr(((G_CAP + 1) * (NG_CAP + 1) + 31) // 32):
                q = lane + cutlass.Int32(32 * i)
                if q < cutlass.Int32((G_CAP + 1) * (NG_CAP + 1)):
                    gq = q // cutlass.Int32(NG_CAP + 1)
                    s_plan_all.store(
                        plan.load(
                            idx=(gq * cutlass.Int32(NCTA) + bx)
                            * cutlass.Int32(NG_CAP + 1)
                            + q
                            - gq * cutlass.Int32(NG_CAP + 1)
                        ),
                        idx=q,
                    )
        rwv = []
        if cutlass.const_expr(ROUTER_GEMV):
            tg = cutlass.select_(
                tidx < cutlass.Int32(GEMV_THREADS), tidx, cutlass.Int32(0)
            )
            for rg in cutlass.range_constexpr(NGEMV):
                eg = bx + cutlass.Int32(rg * NCTA)
                rbx = cutlass.select_(eg < cutlass.Int32(NROUTE), eg, cutlass.Int32(0))
                if cutlass.const_expr(ROUTER_VLLM):
                    # virtual threads tg (and 256 + tg for tg < 128): K = 3072 ki + 8 v + 4 h .. + 3
                    tg2 = cutlass.select_(
                        tg < cutlass.Int32(128), tg + cutlass.Int32(256), tg
                    )
                    rwv.append(
                        [
                            rw.load(
                                idx=rbx * cutlass.Int32(H)
                                + vv * cutlass.Int32(8)
                                + cutlass.Int32(3072 * ki + 4 * h4),
                                vector_size=4,
                                alignment=16,
                            )
                            for vv in (tg, tg2)
                            for ki in range(2)
                            for h4 in range(2)
                        ]
                    )
                else:
                    rwv.append(
                        [
                            rw.load(
                                idx=rbx * cutlass.Int32(H)
                                + tg * cutlass.Int32(4)
                                + cutlass.Int32(1024 * j),
                                vector_size=4,
                                alignment=16,
                            )
                            for j in range(H // 1024)
                        ]
                    )
        prims.fence_mbarrier_init()
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.sync_warp()
        prims.barrier_cta_sync(0)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        tmem_raw = tmem_ptr_i32.load()
        base_col = tmem_raw & 0xFFFF
        base_row = tmem_raw >> 16

        if cutlass.const_expr(EXT_ROUTE):
            # the routing warps and the early-issue warp wait later (after the route read / the early issue)
            ext_early = ((warp >= W_WARP) & (warp < cutlass.Int32(W_WARP + NRW0))) | (
                (warp >= cutlass.Int32(8)) & (warp < cutlass.Int32(8 + RW_EXTRA))
            )
            if cutlass.const_expr(E0_OFFLOAD):
                ext_early = ext_early | (warp == cutlass.Int32(E0_WARP))
            if ext_early == cutlass.Boolean(False):
                cute.arch.griddepcontrol_wait()
        else:
            cute.arch.griddepcontrol_wait()
        arin_on = cutlass.Boolean(False)
        if cutlass.const_expr(AR_IN):
            arin_on = arp.load(idx=6) != cutlass.Int64(0)
        if arin_on:
            if cutlass.const_expr(AR_IN):
                ardesc = arp.load(idx=6)
                _ar_in_reduce(
                    tidx,
                    warp,
                    lane,
                    bx,
                    ardesc,
                    s_big,
                    s_par,
                    xb32,
                    s_misc,
                    tdbg,
                    ar_bar,
                )
                prims.barrier_cta_sync(0)
                if tidx == 0:
                    if bx == 0:
                        _st_release_u32(
                            cutlass.Int64(_ld_relaxed_u64(ardesc + cutlass.Int64(72))),
                            epochs.load(idx=bx) + cutlass.Int32(1),
                        )
                    prims.mbarrier_arrive(x_bar.subview(0))
        elif cutlass.const_expr(XSTAGE):
            if warp == 0:
                if lane == 0:
                    prims.mbarrier_arrive_expect_tx(x_bar.subview(0), M_MAX * H * 2)
                cute.arch.sync_warp()
                if lane < M_MAX:
                    prims.cp_async_bulk_shared_cluster_global(
                        s_big.subview(lane * cutlass.Int32(H * 2)),
                        xb32.subview(lane * cutlass.Int32(H // 2)),
                        x_bar.subview(0),
                        H * 2,
                    )
        ar_on = arp.load(idx=0) != cutlass.Int64(0)
        ar_off = arp.load(idx=0) == cutlass.Int64(0)
        if cutlass.const_expr(not LATE_TRIGGER):
            if ar_off:
                cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(EXT_ROUTE):
            ep = _ld_relaxed_u32(
                epochs.subview(bx).data_ptr().toint()
            )  # (ext_route) pre-wait for the early warps: L2
        else:
            ep = epochs.load(idx=bx)
        tagv = ((ep & cutlass.Int32(0x7F)) | cutlass.Int32(0x80)) << cutlass.Int32(24)
        tag16 = (ep & cutlass.Int32(0x7FFF)) | cutlass.Int32(0x8000)
        ltag = cutlass.Int64(ep + cutlass.Int32(1)) << cutlass.Int64(32)
        if cutlass.const_expr(ROUTER_GEMV):
            # ---- router GEMV (experts bx, bx + NCTA, .., threads 0..255), m3_router_gemv's order exactly
            for rg in cutlass.range_constexpr(NGEMV):
                eg = bx + cutlass.Int32(rg * NCTA)
                if (eg < cutlass.Int32(NROUTE)) & (tidx < cutlass.Int32(GEMV_THREADS)):
                    accs = [cutlass.Float32(0.0) for _ in range(M_MAX)]
                    xsrc = xb32
                    if cutlass.const_expr(GEMV_SMEM and XSTAGE):
                        _wait(x_bar.subview(0), 0)
                        xsrc = cutlass.Array(
                            s_big.data_ptr(0),
                            shape=(M_MAX * H // 2,),
                            dtype=cutlass.Int32,
                            alignment=16,
                        )
                    if cutlass.const_expr(ROUTER_VLLM):
                        accb = [cutlass.Float32(0.0) for _ in range(M_MAX)]
                        for half in cutlass.range_constexpr(2):
                            vth = tidx + cutlass.Int32(256 * half)
                            vok = (half == 0) | (tidx < cutlass.Int32(128))
                            vsafe = cutlass.select_(vok, vth, tidx)
                            for ki in cutlass.range_constexpr(2):
                                for h4 in cutlass.range_constexpr(2):
                                    wq = rwv[rg][half * 4 + ki * 2 + h4]
                                    for t in cutlass.range_constexpr(M_MAX):
                                        xw = xsrc.load(
                                            idx=cutlass.Int32(
                                                t * (H // 2) + 1536 * ki + 2 * h4
                                            )
                                            + vsafe * cutlass.Int32(4),
                                            vector_size=2,
                                            alignment=8,
                                        )
                                        x0 = cutlass.Int32(xw[0])
                                        x1 = cutlass.Int32(xw[1])
                                        a = accs[t] if half == 0 else accb[t]
                                        a = _fma_rn(
                                            (x0 << cutlass.Int32(16)).bitcast(
                                                cutlass.Float32
                                            ),
                                            cutlass.Float32(wq[0]),
                                            a,
                                        )
                                        a = _fma_rn(
                                            (x0 & cutlass.Int32(-65536)).bitcast(
                                                cutlass.Float32
                                            ),
                                            cutlass.Float32(wq[1]),
                                            a,
                                        )
                                        a = _fma_rn(
                                            (x1 << cutlass.Int32(16)).bitcast(
                                                cutlass.Float32
                                            ),
                                            cutlass.Float32(wq[2]),
                                            a,
                                        )
                                        a = _fma_rn(
                                            (x1 & cutlass.Int32(-65536)).bitcast(
                                                cutlass.Float32
                                            ),
                                            cutlass.Float32(wq[3]),
                                            a,
                                        )
                                        if half == 0:
                                            accs[t] = a
                                        else:
                                            accb[t] = a
                        for t in cutlass.range_constexpr(M_MAX):
                            v = accs[t]
                            vb = accb[t]
                            for off in (16, 8, 4, 2, 1):
                                v = _add_rn(v, cute.arch.shuffle_sync_bfly(v, off))
                                vb = _add_rn(vb, cute.arch.shuffle_sync_bfly(vb, off))
                            if lane == 0:
                                s_red.store(
                                    v,
                                    idx=warp * cutlass.Int32(M_MAX) + cutlass.Int32(t),
                                )
                                if warp < cutlass.Int32(4):
                                    s_red.store(
                                        vb,
                                        idx=(warp + cutlass.Int32(8))
                                        * cutlass.Int32(M_MAX)
                                        + cutlass.Int32(t),
                                    )
                        cute.arch.sync_warp()
                        cute.arch.barrier(
                            barrier_id=GEMV_BAR, number_of_threads=GEMV_THREADS
                        )
                        if tidx < M_MAX:
                            sacc = cutlass.Float32(0.0)
                            for w in cutlass.range_constexpr(12):
                                sacc = _add_rn(
                                    sacc,
                                    s_red.load(idx=cutlass.Int32(w * M_MAX) + tidx),
                                )
                            s_red.store(sacc, idx=cutlass.Int32(12 * M_MAX) + tidx)
                        cute.arch.sync_warp()
                        cute.arch.barrier(
                            barrier_id=GEMV_BAR, number_of_threads=GEMV_THREADS
                        )
                    else:
                        for j in cutlass.range_constexpr(H // 1024):
                            for t in cutlass.range_constexpr(M_MAX):
                                xw = xsrc.load(
                                    idx=cutlass.Int32(t * (H // 2) + 512 * j)
                                    + tidx * cutlass.Int32(2),
                                    vector_size=2,
                                    alignment=8,
                                )
                                x0 = cutlass.Int32(xw[0])
                                x1 = cutlass.Int32(xw[1])
                                a = accs[t]
                                a = _fma_rn(
                                    (x0 << cutlass.Int32(16)).bitcast(cutlass.Float32),
                                    cutlass.Float32(rwv[rg][j][0]),
                                    a,
                                )
                                a = _fma_rn(
                                    (x0 & cutlass.Int32(-65536)).bitcast(
                                        cutlass.Float32
                                    ),
                                    cutlass.Float32(rwv[rg][j][1]),
                                    a,
                                )
                                a = _fma_rn(
                                    (x1 << cutlass.Int32(16)).bitcast(cutlass.Float32),
                                    cutlass.Float32(rwv[rg][j][2]),
                                    a,
                                )
                                a = _fma_rn(
                                    (x1 & cutlass.Int32(-65536)).bitcast(
                                        cutlass.Float32
                                    ),
                                    cutlass.Float32(rwv[rg][j][3]),
                                    a,
                                )
                                accs[t] = a
                        for t in cutlass.range_constexpr(M_MAX):
                            v = accs[t]
                            for off in (16, 8, 4, 2, 1):
                                v = _add_rn(v, cute.arch.shuffle_sync_bfly(v, off))
                            if lane == 0:
                                s_red.store(
                                    v,
                                    idx=warp * cutlass.Int32(M_MAX) + cutlass.Int32(t),
                                )
                        cute.arch.sync_warp()
                        cute.arch.barrier(
                            barrier_id=GEMV_BAR, number_of_threads=GEMV_THREADS
                        )
                        if tidx < M_MAX:
                            sacc = cutlass.Float32(0.0)
                            for w in cutlass.range_constexpr(8):
                                sacc = _add_rn(
                                    sacc,
                                    s_red.load(idx=cutlass.Int32(w * M_MAX) + tidx),
                                )
                            s_red.store(sacc, idx=cutlass.Int32(8 * M_MAX) + tidx)
                        cute.arch.sync_warp()
                        cute.arch.barrier(
                            barrier_id=GEMV_BAR, number_of_threads=GEMV_THREADS
                        )
                    if tidx < LG_REPL:
                        for t in cutlass.range_constexpr(M_MAX):
                            lw = ltag | (
                                cutlass.Int64(
                                    cutlass.Float32(
                                        s_red.load(
                                            idx=(12 if ROUTER_VLLM else 8) * M_MAX + t
                                        )
                                    ).bitcast(cutlass.Int32)
                                )
                                & cutlass.Int64(0xFFFFFFFF)
                            )
                            _st_relaxed_u64(
                                lgbuf.subview(
                                    tidx * cutlass.Int32(LG_STRIDE)
                                    + cutlass.Int32(t * E_LOCAL)
                                    + eg
                                )
                                .data_ptr()
                                .toint(),
                                lw,
                            )
                    if cutlass.const_expr(NGEMV > 1):
                        cute.arch.sync_warp()
                        cute.arch.barrier(
                            barrier_id=GEMV_BAR, number_of_threads=GEMV_THREADS
                        )  # s_red reused next

        # ---- routing (warps 4..7, token t on warp 4 + t % 4); the epilogue warps build B meanwhile
        if ((warp >= W_WARP) & (warp < cutlass.Int32(W_WARP + NRW0))) | (
            (warp >= cutlass.Int32(8)) & (warp < cutlass.Int32(8 + RW_EXTRA))
        ):
            rw_i = cutlass.select_(
                warp < cutlass.Int32(8),
                warp - cutlass.Int32(W_WARP),
                warp - cutlass.Int32(4),
            )
            # ILP: poll every token's logits first, then run the independent top-k chains back to back
            lgv_all = []
            t_all = []
            for tr in cutlass.range_constexpr((M_MAX + NRW - 1) // NRW):
                t = rw_i + cutlass.Int32(NRW * tr)
                if cutlass.const_expr(M_MAX % NRW != 0):
                    t = cutlass.select_(
                        t < cutlass.Int32(M_MAX), t, cutlass.Int32(M_MAX - 1)
                    )
                t_all.append(t)
                lgv = []
                if cutlass.const_expr(EXT_ROUTE):
                    # lane l reads top-k entry l % 4 of token t (pre-wait: the word's tag, or a static table)
                    kq = lane & cutlass.Int32(TOP_K - 1)
                    if cutlass.const_expr(EXT_ROUTE == 2):
                        rlo = x32.load(idx=t * cutlass.Int32(TOP_K) + kq)
                    else:
                        want = ltag >> cutlass.Int64(32)
                        raddr = (
                            lgbuf.subview(
                                (
                                    (bx % cutlass.Int32(ROUTE_REPL))
                                    * cutlass.Int32(ROUTE_MAXT)
                                    + t
                                )
                                * cutlass.Int32(16)
                                + kq
                            )
                            .data_ptr()
                            .toint()
                        )
                        done = cutlass.Int32(0)
                        rwd = cutlass.Int64(0)
                        hd1 = cutlass.Int32(0)
                        while (done == cutlass.Int32(0)) & (
                            hd1 < cutlass.Int32(HANG_CAP)
                        ):
                            hd1 = hd1 + cutlass.Int32(1)
                            rwd = _ld_relaxed_u64(raddr)
                            if prims.vote_sync(
                                0xFFFFFFFF,
                                (rwd >> cutlass.Int64(32)) == want,
                                prims.VoteSync.BALLOT,
                            ) == cutlass.Int32(-1):
                                done = cutlass.Int32(1)
                            else:
                                if cutlass.const_expr(ROUTE_POLL_NS > 0):
                                    _nanosleep(ROUTE_POLL_NS)
                        # a given-up poll routes to experts 0..3 with weight 0: distinct ids, a valid plan row
                        rlo = cutlass.select_(
                            done == cutlass.Int32(0),
                            kq << cutlass.Int32(16),
                            cutlass.Int32(rwd),
                        )
                    lgv.append(rlo)
                elif cutlass.const_expr(FUSED_ROUTER):
                    want = ltag >> cutlass.Int64(32)
                    rbase = (
                        (bx % cutlass.Int32(LG_REPL)) * cutlass.Int32(LG_STRIDE)
                        + t * cutlass.Int32(E_LOCAL)
                        + lane
                    )
                    done = cutlass.Int32(0)
                    res = [cutlass.Int64(0) for _ in range(4)]
                    hd1 = cutlass.Int32(0)
                    while (done == cutlass.Int32(0)) & (hd1 < cutlass.Int32(HANG_CAP)):
                        hd1 = hd1 + cutlass.Int32(1)
                        ok = cutlass.Boolean(True)
                        for z in cutlass.range_constexpr(4):
                            res[z] = _ld_relaxed_u64(
                                lgbuf.subview(cutlass.Int32(z * 32) + rbase)
                                .data_ptr()
                                .toint()
                            )
                            ok = ok & ((res[z] >> cutlass.Int64(32)) == want)
                        if prims.vote_sync(
                            0xFFFFFFFF, ok, prims.VoteSync.BALLOT
                        ) == cutlass.Int32(-1):
                            done = cutlass.Int32(1)
                        else:
                            _nanosleep(POLL_NS)
                    for z in cutlass.range_constexpr(4):
                        lgv.append(cutlass.Int32(res[z]).bitcast(cutlass.Float32))
                else:
                    for i in cutlass.range_constexpr(4):
                        lgv.append(
                            cutlass.Float32(
                                logits.load(
                                    idx=t * cutlass.Int32(E_LOCAL)
                                    + cutlass.Int32(i * 32)
                                    + lane
                                )
                            )
                        )
                lgv_all.append(lgv)
            res_all = []
            if cutlass.const_expr(EXT_ROUTE):
                # (t, own, w, id) exactly as the top-k leaves them: lane k < 4 holds entry k
                for tr in cutlass.range_constexpr((M_MAX + NRW - 1) // NRW):
                    rlo = lgv_all[tr][0]
                    res_all.append(
                        (
                            t_all[tr],
                            (rw_i + cutlass.Int32(NRW * tr)) < cutlass.Int32(M_MAX),
                            cutlass.Int32(rlo << cutlass.Int32(16)).bitcast(
                                cutlass.Float32
                            ),
                            (rlo >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF),
                        )
                    )
            for tr in cutlass.range_constexpr(
                0 if EXT_ROUTE else (M_MAX + NRW - 1) // NRW
            ):
                t = t_all[tr]
                lgv = lgv_all[tr]
                own = (rw_i + cutlass.Int32(NRW * tr)) < cutlass.Int32(M_MAX)
                # top-4 (keys sigmoid + bias, ties to the lower id)
                keys = []
                for i in cutlass.range_constexpr(4):
                    sg = _fma_rn(
                        cutlass.Float32(0.5),
                        cute.math.tanh(_mul_rn(cutlass.Float32(0.5), lgv[i])),
                        cutlass.Float32(0.5),
                    )
                    keys.append(_add_rn(sg, s_bias.load(idx=i * 32 + lane)))
                hi = [_twiddle(keys[i]) for i in range(4)]
                taken = [cutlass.Boolean(False) for _ in range(4)]
                ids_t = []
                kt_t = []
                for k in cutlass.range_constexpr(TOP_K):  # noqa: B007
                    bh = cutlass.Int32(0)
                    bl = cutlass.Int32(0)
                    for i in cutlass.range_constexpr(4):
                        lo_i = cutlass.Int32(65535 - i * 32) - lane
                        hv = cutlass.select_(taken[i], cutlass.Int32(0), hi[i])
                        lv = cutlass.select_(taken[i], cutlass.Int32(0), lo_i)
                        better = (cutlass.Uint32(hv) > cutlass.Uint32(bh)) | (
                            (hv == bh) & (cutlass.Uint32(lv) > cutlass.Uint32(bl))
                        )
                        bh = cutlass.select_(better, hv, bh)
                        bl = cutlass.select_(better, lv, bl)
                    mh = _redux_max_u32(bh)
                    ml = _redux_max_u32(cutlass.select_(bh == mh, bl, cutlass.Int32(0)))
                    e_k = cutlass.Int32(65535) - ml
                    for i in cutlass.range_constexpr(4):
                        taken[i] = taken[i] | (e_k == cutlass.Int32(i * 32) + lane)
                    ids_t.append(e_k)
                    kv = cutlass.Float32(0.0)
                    for i in cutlass.range_constexpr(4):
                        kv = cutlass.select_(
                            e_k == cutlass.Int32(i * 32) + lane, keys[i], kv
                        )
                    kt_t.append(cute.arch.shuffle_sync(kv, e_k % cutlass.Int32(32)))
                sig = cutlass.Float32(0.0)
                e_l = cutlass.Int32(0)
                for k in cutlass.range_constexpr(TOP_K):
                    sk = _sub_rn(kt_t[k], s_bias.load(idx=ids_t[k]))
                    sig = cutlass.select_(lane == cutlass.Int32(k), sk, sig)
                    e_l = cutlass.select_(lane == cutlass.Int32(k), ids_t[k], e_l)
                total = sig
                for off in (16, 8, 4, 2, 1):
                    total = _add_rn(total, cute.arch.shuffle_sync_bfly(total, off))
                wv = _bf16_round(
                    _div_rn(
                        _mul_rn(sig, cutlass.Float32(ROUTE_SCALE)),
                        _add_rn(total, cutlass.Float32(SUM_EPS)),
                    )
                )
                res_all.append((t, own, wv, e_l))
            for tr in cutlass.range_constexpr(
                (M_MAX + NRW - 1) // NRW
            ):  # stores after every chain (one branch-free block)
                t, own, wv, e_l = res_all[tr]
                if (lane < TOP_K) & own:
                    s_w.store(wv, idx=t * cutlass.Int32(TOP_K) + lane)
                    s_ids.store(e_l, idx=t * cutlass.Int32(TOP_K) + lane)
                    if cutlass.const_expr(
                        not EXT_ROUTE
                    ):  # (ext_route) a global write: after the wait below
                        if bx == 0:
                            route_dbg.store(
                                wv.bitcast(cutlass.Int32),
                                idx=t * cutlass.Int32(8) + cutlass.Int32(4) + lane,
                            )
            cute.arch.sync_warp()
            if cutlass.const_expr(EARLY0 and not E0_OFFLOAD):
                if warp == cutlass.Int32(W_WARP):
                    # slots 0..3 are always token 0's four (distinct) experts and group
                    # 0's stage ranges are the same for every G (the plan guarantees
                    # it): fill the ring with this CTA's first group-0 stages now
                    if cutlass.const_expr(XSTAGE):
                        _wait(x_bar.subview(1), 0)
                    n0 = cutlass.Int32(0)
                    w0 = s_plan_all.load(idx=G_CAP * (NG_CAP + 1))
                    ei0 = w0 & cutlass.Int32(0xFFFF)
                    ei1 = w0 >> cutlass.Int32(16)
                    for k in range(_nsegs(ei0, ei1)):
                        ug, j0, j1 = _seg(ei0, ei1, k)
                        c = _ug_c(ug)
                        e = s_ids.load(idx=_ug_slot(ug))
                        sf_line = (
                            (e * cutlass.Int32(SF1_RB) + (c >> cutlass.Int32(2)))
                            * cutlass.Int32(SF1_KA)
                            + j0 * cutlass.Int32(C1 * NUM_KB)
                        ) * cutlass.Int32(4)
                        kx = j0 * cutlass.Int32(C1 * KT_BYTES)
                        row = c * cutlass.Int32(R1)
                        rowu = row
                        dsc1 = tma_a1_desc.get_ptr()
                        for j in cutlass.range(j0, j1, unroll=1):  # noqa: B007
                            if n0 < cutlass.Int32(min(S1R, EARLY0_N)):
                                if prims.elect_sync():
                                    prims.mbarrier_arrive_expect_tx(
                                        ab_full.subview(n0),
                                        (
                                            NUM_BYTES_A
                                            + (NUM_BYTES_SFA if SFA_BULK else 0)
                                        )
                                        if SFA_CP
                                        else NUM_BYTES_A + NUM_BYTES_SFA_RAW,
                                    )
                                    for q in cutlass.range_constexpr(C1):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sA.subview(
                                                n0 * cutlass.Int32(NUM_BYTES_A)
                                                + cutlass.Int32(q * NUM_BYTES_BOX1)
                                            ),
                                            dsc1,
                                            (kx + cutlass.Int32(q * KT_BYTES), rowu, e),
                                            ab_full.subview(n0),
                                            l2_cache_hint=EVICT_FIRST,
                                        )
                                    if cutlass.const_expr(not SFA_CP):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sSFAraw.subview(
                                                n0 * cutlass.Int32(NUM_WORDS_SFA_RAW)
                                            ),
                                            tma_s1_desc.get_ptr(),
                                            (cutlass.Int32(0), sf_line),
                                            ab_full.subview(n0),
                                        )
                                if cutlass.const_expr(SFA_CP):
                                    _sf1_cp(lane, n0, sf_line, c, sSFA, s1w, ab_full)
                                n0 = n0 + cutlass.Int32(1)
                            kx = kx + cutlass.Int32(C1 * KT_BYTES)
                            sf_line = sf_line + cutlass.Int32(NUM_BYTES_SFA_RAW // 128)
                    if lane == 0:
                        s_misc.store(n0, idx=0)
                    cute.arch.sync_warp()
            if cutlass.const_expr(E0_OFFLOAD):
                if warp == cutlass.Int32(W_WARP):
                    _bar_arrive(
                        E0_BAR, 64
                    )  # token 0's experts are in s_ids: warp 7 issues the early stages
            if cutlass.const_expr(B1_AFTER_TOPK):
                _bar_arrive(TOPK2_BAR, 128 + 32 * NRW)
            cute.arch.barrier(barrier_id=TOPK_BAR, number_of_threads=32 * NRW)
            if cutlass.const_expr(EXT_ROUTE):
                cute.arch.griddepcontrol_wait()
                for tr in cutlass.range_constexpr((M_MAX + NRW - 1) // NRW):
                    t, own, wv, e_l = res_all[tr]
                    if (lane < TOP_K) & own & (bx == 0):
                        route_dbg.store(
                            wv.bitcast(cutlass.Int32),
                            idx=t * cutlass.Int32(8) + cutlass.Int32(4) + lane,
                        )
        if cutlass.const_expr(E0_OFFLOAD):
            if warp == cutlass.Int32(E0_WARP):
                if cutlass.const_expr(E0_W8 and E0_ARRIVE):
                    # arrive at the routing barrier now; the early issue (the scale cp.async) must not gate it
                    cute.arch.sync_warp()
                    _bar_arrive(ROUTE_BAR, THREADS)
                cute.arch.barrier(barrier_id=E0_BAR, number_of_threads=64)
                # slots 0..3 are always token 0's four (distinct) experts and group 0's
                # stage ranges are the same for every G (the plan guarantees it): fill
                # the ring with this CTA's first group-0 stages now
                if cutlass.const_expr(XSTAGE):
                    _wait(x_bar.subview(1), 0)
                n0 = cutlass.Int32(0)
                w0 = s_plan_all.load(idx=G_CAP * (NG_CAP + 1))
                ei0 = w0 & cutlass.Int32(0xFFFF)
                ei1 = w0 >> cutlass.Int32(16)
                for k in range(_nsegs(ei0, ei1)):
                    ug, j0, j1 = _seg(ei0, ei1, k)
                    c = _ug_c(ug)
                    e = s_ids.load(idx=_ug_slot(ug))
                    sf_line = (
                        (e * cutlass.Int32(SF1_RB) + (c >> cutlass.Int32(2)))
                        * cutlass.Int32(SF1_KA)
                        + j0 * cutlass.Int32(C1 * NUM_KB)
                    ) * cutlass.Int32(4)
                    kx = j0 * cutlass.Int32(C1 * KT_BYTES)
                    row = c * cutlass.Int32(R1)
                    rowu = row
                    dsc1 = tma_a1_desc.get_ptr()
                    for j in cutlass.range(j0, j1, unroll=1):  # noqa: B007
                        if n0 < cutlass.Int32(min(S1R, EARLY0_N)):
                            if prims.elect_sync():
                                prims.mbarrier_arrive_expect_tx(
                                    ab_full.subview(n0),
                                    (NUM_BYTES_A + (NUM_BYTES_SFA if SFA_BULK else 0))
                                    if SFA_CP
                                    else NUM_BYTES_A + NUM_BYTES_SFA_RAW,
                                )
                                for q in cutlass.range_constexpr(C1):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        sA.subview(
                                            n0 * cutlass.Int32(NUM_BYTES_A)
                                            + cutlass.Int32(q * NUM_BYTES_BOX1)
                                        ),
                                        dsc1,
                                        (kx + cutlass.Int32(q * KT_BYTES), rowu, e),
                                        ab_full.subview(n0),
                                        l2_cache_hint=EVICT_FIRST,
                                    )
                                if cutlass.const_expr(not SFA_CP):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        sSFAraw.subview(
                                            n0 * cutlass.Int32(NUM_WORDS_SFA_RAW)
                                        ),
                                        tma_s1_desc.get_ptr(),
                                        (cutlass.Int32(0), sf_line),
                                        ab_full.subview(n0),
                                    )
                            if cutlass.const_expr(SFA_CP):
                                _sf1_cp(lane, n0, sf_line, c, sSFA, s1w, ab_full)
                            n0 = n0 + cutlass.Int32(1)
                        kx = kx + cutlass.Int32(C1 * KT_BYTES)
                        sf_line = sf_line + cutlass.Int32(NUM_BYTES_SFA_RAW // 128)
                if lane == 0:
                    s_misc.store(n0, idx=0)
                cute.arch.sync_warp()
                _bar_arrive(E1_BAR, E1_N)
                if cutlass.const_expr(EXT_ROUTE):
                    cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(EXT_ROUTE):
            # every warp is past its own grid wait here: the post-wait protocols take a post-wait epoch
            ep_pre = ep
            ep = epochs.load(idx=bx)
            tagv = ((ep & cutlass.Int32(0x7F)) | cutlass.Int32(0x80)) << cutlass.Int32(
                24
            )
            tag16 = (ep & cutlass.Int32(0x7FFF)) | cutlass.Int32(0x8000)
            ltag = cutlass.Int64(ep + cutlass.Int32(1)) << cutlass.Int64(32)
            if cutlass.const_expr(EXT_ROUTE == 1):
                if (
                    ep != ep_pre
                ):  # the route tag came from a stale epoch, so the words read may be a stale call's
                    _asm(None, [], "trap;", "", True)
        if warp == W_WARP:
            # slots: lane l < 4 M is (token l / 4, top-k l % 4); the distinct experts in lane order (match.any)
            valid = lane < cutlass.Int32(TOP_K * M_MAX)
            my_e = cutlass.select_(
                valid,
                s_ids.load(idx=cutlass.select_(valid, lane, cutlass.Int32(0))),
                cutlass.Int32(-1) - lane,
            )
            same = _match_any(my_e)
            leader = cute.arch.popc(
                (same & (cutlass.Int32(0) - same)) - cutlass.Int32(1)
            )
            first = valid & (lane == leader)
            bal = prims.vote_sync(0xFFFFFFFF, first, prims.VoteSync.BALLOT)
            rank = cute.arch.popc(bal & cutlass.Int32(cute.arch.lanemask_lt()))
            slot_l = cute.arch.shuffle_sync(rank, leader)
            for z in cutlass.range_constexpr((G_CAP * M_MAX + 31) // 32):
                if lane + cutlass.Int32(32 * z) < cutlass.Int32(G_CAP * M_MAX):
                    s_kpos.store(cutlass.Int32(-1), idx=lane + cutlass.Int32(32 * z))
            cute.arch.sync_warp()
            if first:
                s_el.store(my_e, idx=rank)
            if valid:
                s_kpos.store(
                    lane % cutlass.Int32(TOP_K),
                    idx=slot_l * cutlass.Int32(M_MAX) + lane // cutlass.Int32(TOP_K),
                )
                if bx == 0:
                    route_dbg.store(
                        my_e,
                        idx=(lane // cutlass.Int32(TOP_K)) * cutlass.Int32(8)
                        + lane % cutlass.Int32(TOP_K),
                    )
            gcnt = cute.arch.popc(bal)
            if lane == 0:
                s_el.store(gcnt, idx=G_CAP)
            if lane < cutlass.Int32(NG_CAP + 1):  # the plan row of this G (preloaded)
                s_plan.store(
                    s_plan_all.load(idx=gcnt * cutlass.Int32(NG_CAP + 1) + lane),
                    idx=lane,
                )
            cute.arch.sync_warp()
            if cutlass.const_expr(B2IMG and not PAD_X):
                # CTA 0 pads every group counter to exactly CNT_PER_CALL for this call (absent slots, unused groups)
                if (bx == 0) & (lane < cutlass.Int32(NG_CAP)):
                    nh = gcnt - lane * cutlass.Int32(4)
                    nh = cutlass.select_(nh > cutlass.Int32(4), cutlass.Int32(4), nh)
                    nh = cutlass.select_(nh < cutlass.Int32(0), cutlass.Int32(0), nh)
                    pad = cutlass.Int32(CNT_PER_CALL) - nh * cutlass.Int32(UNITS_ROW)
                    if pad > cutlass.Int32(0):
                        for rr in cutlass.range_constexpr(B2_REPL):
                            _red_release_add(
                                b2cnt.subview(
                                    lane * cutlass.Int32(B2_GS)
                                    + cutlass.Int32(rr * B2_CSTRIDE)
                                )
                                .data_ptr()
                                .toint(),
                                pad,
                            )
                cute.arch.sync_warp()
            cute.arch.sync_warp()
            _bar_arrive(ROUTE_BAR, THREADS)
        if warp < 4:
            if cutlass.const_expr(EXT_ROUTE):
                if cutlass.const_expr(EXT_ROUTE == 1):
                    # (post-wait only: the grid wait covered the post's completion):
                    # lane l loads row l's counter (rctr in the x32 slot, 32-word rows;
                    # lanes >= M re-read row 0), checked after the B1 build
                    rcv = _ld_relaxed_u32(
                        x32.subview(
                            cutlass.select_(
                                lane < cutlass.Int32(M_MAX), lane, cutlass.Int32(0)
                            )
                            * cutlass.Int32(32)
                        )
                        .data_ptr()
                        .toint()
                    )
            if cutlass.const_expr(B1_AFTER_TOPK):
                cute.arch.barrier(
                    barrier_id=TOPK2_BAR, number_of_threads=128 + 32 * NRW
                )
            if cutlass.const_expr(FUSED_ROUTER):
                if cutlass.const_expr(B1_X4):
                    if cutlass.const_expr(FC1_PER_TOKEN):
                        if tidx < cutlass.Int32(M_MAX):
                            s_pts.store(
                                cutlass.Int32(
                                    xsf32.load(idx=cutlass.Int32(S_T_WORD) + tidx)
                                ).bitcast(cutlass.Float32),
                                idx=tidx,
                            )
                    _build_b1(
                        tidx,
                        warp,
                        lane,
                        base_row,
                        base_col,
                        logits,
                        xsf32,
                        sB1,
                        b1_ready,
                    )
                    if cutlass.const_expr(
                        FC1_PER_TOKEN
                    ):  # s_pts is read by the same four warps in the FC1 epilogue
                        cute.arch.barrier(
                            barrier_id=EPI_BAR, number_of_threads=EPI_THREADS
                        )
                elif cutlass.const_expr(XSTAGE):
                    _wait(x_bar.subview(0), 0)
                    xs32 = cutlass.Array(
                        s_big.data_ptr(0),
                        shape=(M_MAX * H // 2,),
                        dtype=cutlass.Int32,
                        alignment=16,
                    )
                    _build_b1_fused(
                        tidx,
                        warp,
                        lane,
                        base_row,
                        base_col,
                        xs32,
                        sfsc,
                        sB1,
                        s_xsf,
                        b1_ready,
                    )
                    prims.mbarrier_arrive(
                        x_bar.subview(1)
                    )  # the ring is free for the weights
                else:
                    _build_b1_fused(
                        tidx,
                        warp,
                        lane,
                        base_row,
                        base_col,
                        xb32,
                        sfsc,
                        sB1,
                        s_xsf,
                        b1_ready,
                    )
            else:
                _build_b1(
                    tidx, warp, lane, base_row, base_col, x32, xsf32, sB1, b1_ready
                )
            if cutlass.const_expr(EXT_ROUTE):
                if cutlass.const_expr(EXT_ROUTE == 1):
                    # exactly one publish per engine call; an extra or a missing one would route from stale words
                    if prims.vote_sync(
                        0xFFFFFFFF, rcv != ep + cutlass.Int32(1), prims.VoteSync.BALLOT
                    ) != cutlass.Int32(0):
                        _asm(None, [], "trap;", "", True)
        if (warp != W_WARP) & (
            (warp != cutlass.Int32(E0_WARP))
            | cutlass.Boolean(not (E0_OFFLOAD and E0_W8 and E0_ARRIVE))
        ):  # one sync site for every thread but warp 4's (which arrived)
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=ROUTE_BAR, number_of_threads=THREADS)
        if cutlass.const_expr(EARLY_TRIG):
            if ar_on:
                cute.arch.griddepcontrol_launch_dependents()
        g_n = s_el.load(idx=G_CAP)
        ng = (g_n + cutlass.Int32(3)) >> cutlass.Int32(2)
        pr2 = s_plan.load(idx=NG_CAP)
        blk0 = pr2 & cutlass.Int32(0xFFFF)
        nb = pr2 >> cutlass.Int32(16)

        # ---- (PF_FC2) warp 6 (idle with SFA_CP once routing is done): L2-prefetch this
        # CTA's FC2 rows, full 768 B rows per (slot, block) as one contiguous bulk
        # prefetch (FC2 tiles read 128 B per row per stage, which costs DRAM page
        # locality), plus the rows' scale atoms. arp[7] = w2_weight base (0: off).
        if cutlass.const_expr(PF_FC2 and SFA_CP):
            if warp == cutlass.Int32(S_WARP):
                w2b = arp.load(idx=7)
                if w2b != cutlass.Int64(0):
                    npairs = g_n * nb
                    pq = lane
                    while pq < npairs:
                        ps_ = pq // nb
                        pblk = blk0 + (pq - ps_ * nb)
                        pe = s_el.load(idx=ps_)
                        _l2_prefetch(
                            w2b
                            + (
                                cutlass.Int64(pe) * cutlass.Int64(H)
                                + cutlass.Int64(pblk * cutlass.Int32(R2))
                            )
                            * cutlass.Int64(I_TP // 2),
                            R2 * (I_TP // 2),
                        )
                        _l2_prefetch(
                            s2w.subview(
                                (
                                    pe * cutlass.Int32(SF2_RB)
                                    + (pblk >> cutlass.Int32(2))
                                )
                                * cutlass.Int32(SF2_KA * 128)
                            )
                            .data_ptr()
                            .toint(),
                            SF2_KA * 512,
                        )
                        pq = pq + cutlass.Int32(32)

        # ---- weights producer (warp 4): phase p = FC1 stages of group p, then FC2
        # tiles of group p - LAG (PW2: warp 9 runs the same walk and issues the odd FC1
        # stages)
        if (warp == W_WARP) | (
            (warp == cutlass.Int32(PW2_WARP)) & cutlass.Boolean(PWX)
        ):
            isw4 = warp == cutlass.Int32(W_WARP)
            fc2own = isw4 != cutlass.Boolean(
                PW2F
            )  # (PW2F) warp 9 owns FC2, else warp 4
            nis = cutlass.Int32(0)
            q2n = cutlass.Int32(0)
            if cutlass.const_expr(XSTAGE):
                _wait(x_bar.subview(1), 0)  # the staged hidden states are quantized
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(1)
            skip = cutlass.Int32(0)
            if cutlass.const_expr(E0_OFFLOAD):
                cute.arch.barrier(barrier_id=E1_BAR, number_of_threads=E1_N)
            if cutlass.const_expr(EARLY0):
                skip = s_misc.load(idx=0)  # group-0 stages already in the ring
                stage = cutlass.select_(
                    skip == cutlass.Int32(S1R), cutlass.Int32(0), skip
                )
                phase = cutlass.select_(
                    skip == cutlass.Int32(S1R), cutlass.Int32(0), cutlass.Int32(1)
                )
            if cutlass.const_expr(
                PW2F
            ):  # warp 9: the FC2 ring slice from its start, no early FC1 stages
                stage = cutlass.select_(isw4, stage, cutlass.Int32(S1R))
                phase = cutlass.select_(isw4, phase, cutlass.Int32(1))
                skip = cutlass.select_(isw4, skip, cutlass.Int32(0))
            if cutlass.const_expr(SF2_W6):
                # header entry 0: this CTA's FC2 stage count. Warp 8 arrived at
                # ROUTE_BAR early (E0_W8), so it must not read the routing (g_n) or the
                # plan (nb) before this post (release / acquire via q2_bar)
                if (lane == 0) & fc2own:
                    q2_ent.store(cutlass.Int32(-1), idx=0)
                    q2_ent.store(
                        cutlass.select_(
                            nb > cutlass.Int32(0),
                            ng * nb * cutlass.Int32(KT2),
                            cutlass.Int32(0),
                        ),
                        idx=1,
                    )
                    prims.mbarrier_arrive(q2_bar.subview(0))
                cute.arch.sync_warp()
                q2n = cutlass.Int32(1)
            for p in range(ng + cutlass.Int32(LAG)):
                if p < ng:
                    i0, i1 = _plan_range(s_plan, p)
                    for k in range(_nsegs(i0, i1)):
                        ug, j0, j1 = _seg(i0, i1, k)
                        slot = p * cutlass.Int32(4) + _ug_slot(ug)
                        c = _ug_c(ug)
                        e = s_el.load(idx=slot)
                        sf_line = (
                            (e * cutlass.Int32(SF1_RB) + (c >> cutlass.Int32(2)))
                            * cutlass.Int32(SF1_KA)
                            + j0 * cutlass.Int32(C1 * NUM_KB)
                        ) * cutlass.Int32(4)
                        kx = j0 * cutlass.Int32(C1 * KT_BYTES)
                        row = c * cutlass.Int32(R1)
                        rowu = row
                        dsc1 = tma_a1_desc.get_ptr()
                        for j in cutlass.range(j0, j1, unroll=1):  # noqa: B007
                            if skip > cutlass.Int32(0):
                                skip = skip - cutlass.Int32(1)
                            else:
                                own1 = cutlass.Boolean(True)
                                if cutlass.const_expr(PW2):
                                    own1 = (
                                        (nis & cutlass.Int32(1)) == cutlass.Int32(1)
                                    ) == (warp == cutlass.Int32(PW2_WARP))
                                    nis = nis + cutlass.Int32(1)
                                if cutlass.const_expr(PW2F):
                                    own1 = isw4
                                if (
                                    own1 | cutlass.Boolean(PW2)
                                ):  # PW2: wait every position (the other warp's too), so
                                    # neither producer runs a lap ahead and aliases an empty-slot parity
                                    _waitd(ab_empty.subview(stage), phase, tdbg, 1)
                                if own1:
                                    if prims.elect_sync():
                                        prims.mbarrier_arrive_expect_tx(
                                            ab_full.subview(stage),
                                            (
                                                NUM_BYTES_A
                                                + (NUM_BYTES_SFA if SFA_BULK else 0)
                                            )
                                            if SFA_CP
                                            else NUM_BYTES_A + NUM_BYTES_SFA_RAW,
                                        )
                                        for q in cutlass.range_constexpr(C1):
                                            prims.cp_async_bulk_tensor_shared_cta_global(
                                                sA.subview(
                                                    stage * cutlass.Int32(NUM_BYTES_A)
                                                    + cutlass.Int32(q * NUM_BYTES_BOX1)
                                                ),
                                                dsc1,
                                                (
                                                    kx + cutlass.Int32(q * KT_BYTES),
                                                    rowu,
                                                    e,
                                                ),
                                                ab_full.subview(stage),
                                                l2_cache_hint=EVICT_FIRST,
                                            )
                                        if cutlass.const_expr(not SFA_CP):
                                            prims.cp_async_bulk_tensor_shared_cta_global(
                                                sSFAraw.subview(
                                                    stage
                                                    * cutlass.Int32(NUM_WORDS_SFA_RAW)
                                                ),
                                                tma_s1_desc.get_ptr(),
                                                (cutlass.Int32(0), sf_line),
                                                ab_full.subview(stage),
                                            )
                                    if cutlass.const_expr(SFA_CP):
                                        _sf1_cp(
                                            lane, stage, sf_line, c, sSFA, s1w, ab_full
                                        )
                                if cutlass.const_expr(
                                    PW2F
                                ):  # warp 9's cursor is on the FC2 slice
                                    if isw4:
                                        stage, phase = _adv1(stage, phase)
                                else:
                                    stage, phase = _adv1(stage, phase)
                            kx = kx + cutlass.Int32(C1 * KT_BYTES)
                            sf_line = sf_line + cutlass.Int32(NUM_BYTES_SFA_RAW // 128)
                if p >= cutlass.Int32(LAG):
                    g = p - cutlass.Int32(LAG)
                    n_here = g_n - g * cutlass.Int32(4)
                    n_here = cutlass.select_(
                        n_here > cutlass.Int32(4), cutlass.Int32(4), n_here
                    )
                    for bb in range(nb):
                        blk = blk0 + bb
                        dsc2 = tma_a2_desc.get_ptr()
                        rowb = blk * cutlass.Int32(R2)
                        for t2 in cutlass.range(KT2, unroll=1 if ROLL_FC2 else KT2):
                            if cutlass.const_expr(
                                PW2
                            ):  # the non-owner waits too (no lapping, see FC1)
                                if fc2own == cutlass.Boolean(False):
                                    _waitd(ab_empty.subview(stage), phase, tdbg, 2)
                            if fc2own:  # FC2 stages and their SF2 queue entries: warp 4's (PW2F: warp 9's)
                                _waitd(ab_empty.subview(stage), phase, tdbg, 2)
                                if prims.elect_sync():
                                    prims.mbarrier_arrive_expect_tx(
                                        ab_full.subview(stage),
                                        n_here
                                        * cutlass.Int32(
                                            NUM_BYTES_BOX1
                                            if SFA_CP
                                            else NUM_BYTES_BOX1 + NUM_BYTES_SFA_RAW // 4
                                        ),
                                    )
                                    for jj in cutlass.range_constexpr(4):
                                        if cutlass.Int32(jj) < n_here:
                                            e = s_el.load(
                                                idx=g * cutlass.Int32(4)
                                                + cutlass.Int32(jj)
                                            )
                                            prims.cp_async_bulk_tensor_shared_cta_global(
                                                sA.subview(
                                                    stage * cutlass.Int32(NUM_BYTES_A)
                                                    + cutlass.Int32(jj * NUM_BYTES_BOX1)
                                                ),
                                                dsc2,
                                                (
                                                    cutlass.Int32(t2 * KT_BYTES),
                                                    rowb,
                                                    e,
                                                ),
                                                ab_full.subview(stage),
                                                l2_cache_hint=EVICT_FIRST,
                                            )
                                            if cutlass.const_expr(not SFA_CP):
                                                prims.cp_async_bulk_tensor_shared_cta_global(
                                                    sSFAraw.subview(
                                                        stage
                                                        * cutlass.Int32(
                                                            NUM_WORDS_SFA_RAW
                                                        )
                                                        + cutlass.Int32(
                                                            jj * NUM_WORDS_SFA_RAW // 4
                                                        )
                                                    ),
                                                    tma_s2_desc.get_ptr(),
                                                    (
                                                        cutlass.Int32(0),
                                                        (
                                                            (
                                                                e
                                                                * cutlass.Int32(SF2_RB)
                                                                + (
                                                                    blk
                                                                    >> cutlass.Int32(2)
                                                                )
                                                            )
                                                            * cutlass.Int32(SF2_KA)
                                                            + cutlass.Int32(4 * t2)
                                                        )
                                                        * cutlass.Int32(4),
                                                    ),
                                                    ab_full.subview(stage),
                                                )
                                if cutlass.const_expr(SF2_W6):
                                    # post the stage to warp 6, which gathers its scale atoms and does the 32 arrivals
                                    qi = q2n - (
                                        q2n // cutlass.Int32(Q2D)
                                    ) * cutlass.Int32(Q2D)
                                    _wait(
                                        q2_bar.subview(Q2D + qi),
                                        ((q2n // cutlass.Int32(Q2D)) & cutlass.Int32(1))
                                        ^ cutlass.Int32(1),
                                    )
                                    if lane == 0:
                                        q2_ent.store(stage, idx=qi * cutlass.Int32(4))
                                        q2_ent.store(
                                            g,
                                            idx=qi * cutlass.Int32(4)
                                            + cutlass.Int32(1),
                                        )
                                        q2_ent.store(
                                            blk,
                                            idx=qi * cutlass.Int32(4)
                                            + cutlass.Int32(2),
                                        )
                                        q2_ent.store(
                                            t2,
                                            idx=qi * cutlass.Int32(4)
                                            + cutlass.Int32(3),
                                        )
                                        prims.mbarrier_arrive(q2_bar.subview(qi))
                                    cute.arch.sync_warp()
                                    q2n = q2n + cutlass.Int32(1)
                                elif cutlass.const_expr(SFA_CP):
                                    _sf2_cp(
                                        lane,
                                        stage,
                                        g,
                                        blk,
                                        t2,
                                        n_here,
                                        s_el,
                                        sSFA,
                                        s2w,
                                        ab_full,
                                    )
                            if cutlass.const_expr(
                                PW2F
                            ):  # warp 4's cursor stays on the FC1 slice
                                if fc2own:
                                    stage, phase = _adv2(stage, phase)
                            else:
                                stage, phase = _adv2(stage, phase)
        # ---- scale splicer (warp 6): every stage, atom kb word [r][q] = raw atom 4 q + kb word [r][m1]
        if (warp == S_WARP) & cutlass.Boolean(not SFA_CP):  # (SFA_CP: no splicer)
            s2t_shape, s2t_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)
            for p in range(ng + cutlass.Int32(LAG)):
                i0 = cutlass.Int32(0)
                i1 = cutlass.Int32(0)
                for part in cutlass.range_constexpr(2):
                    cnt = cutlass.Int32(0)
                    m1 = cutlass.Int32(0)
                    nk = cutlass.Int32(0)
                    if cutlass.const_expr(part == 0):
                        if p < ng:
                            i0, i1 = _plan_range(s_plan, p)
                            nk = _nsegs(i0, i1)
                    else:
                        if p >= cutlass.Int32(LAG):
                            nk = nb
                    for k in range(nk):
                        if cutlass.const_expr(part == 0):
                            ug, j0, j1 = _seg(i0, i1, k)
                            m1 = (_ug_c(ug)) & cutlass.Int32(3)
                            cnt = j1 - j0
                        else:
                            m1 = (blk0 + k) & cutlass.Int32(3)
                            cnt = cutlass.Int32(KT2)
                        for _j in cutlass.range(cnt, unroll=1):
                            _waitd(ab_full.subview(stage), phase, tdbg, 3)
                            raw0 = (
                                stage * cutlass.Int32(NUM_WORDS_SFA_RAW)
                                + lane * cutlass.Int32(4)
                                + m1
                            )
                            dst0 = stage * cutlass.Int32(
                                NUM_WORDS_SFA
                            ) + lane * cutlass.Int32(4)
                            for kb in cutlass.range_constexpr(NUM_KB):
                                w0 = sSFAraw.load(
                                    idx=raw0 + cutlass.Int32((0 * 4 + kb) * 128)
                                )
                                w1 = sSFAraw.load(
                                    idx=raw0 + cutlass.Int32((1 * 4 + kb) * 128)
                                )
                                w2 = sSFAraw.load(
                                    idx=raw0 + cutlass.Int32((2 * 4 + kb) * 128)
                                )
                                w3 = sSFAraw.load(
                                    idx=raw0 + cutlass.Int32((3 * 4 + kb) * 128)
                                )
                                sSFA.store(
                                    (
                                        cutlass.Int32(w0),
                                        cutlass.Int32(w1),
                                        cutlass.Int32(w2),
                                        cutlass.Int32(w3),
                                    ),
                                    idx=dst0 + cutlass.Int32(kb * 128),
                                    alignment=16,
                                )
                            prims.fence_proxy(
                                "async_shared", space=prims.SharedSpace.shared_cta
                            )
                            cute.arch.sync_warp()
                            if prims.elect_sync():
                                sd0 = prims.Tcgen05SmemDesc.build(
                                    sSFA.subview(stage * cutlass.Int32(NUM_WORDS_SFA)),
                                    leading_byte_offset=16,
                                    stride_byte_offset=128,
                                    base_offset=0,
                                    layout=0,
                                )
                                for kb in cutlass.range_constexpr(NUM_KB):
                                    prims.tcgen05_cp(
                                        s2t_shape,
                                        _tptr(
                                            base_row,
                                            base_col
                                            + SFA1_COL
                                            + stage * cutlass.Int32(16)
                                            + kb * 4,
                                        ),
                                        sd0 + (kb * 512 >> 4),
                                        multicast=s2t_mc,
                                    )
                                prims.tcgen05_commit(sfa_ready.subview(stage))
                            stage, phase = _adv(stage, phase)

        if cutlass.const_expr(B2IMG):
            # ---- (B2IMG) group releases (warp 6): this CTA's finished units of group p, one release add per group
            if warp == cutlass.Int32(S_WARP):
                for p in range(ng):
                    i0, i1 = _plan_range(s_plan, p)
                    n_done = cutlass.Int32(0)
                    for k in range(_nsegs(i0, i1)):
                        ug, j0, j1 = _seg(i0, i1, k)
                        n_done = n_done + cutlass.select_(
                            (j0 != cutlass.Int32(0)) | (j1 == cutlass.Int32(S1)),
                            cutlass.Int32(1),
                            cutlass.Int32(0),
                        )
                    _wait(
                        fc1_done.subview(p & cutlass.Int32(3)),
                        (p >> cutlass.Int32(2)) & cutlass.Int32(1),
                    )
                    if (lane < cutlass.Int32(B2_REPL)) & (n_done > cutlass.Int32(0)):
                        _fence_proxy_async_global()
                        _red_release_add(
                            b2cnt.subview(
                                (lane + bx % cutlass.Int32(B2_SHARD))
                                * cutlass.Int32(B2_CSTRIDE)
                                + p * cutlass.Int32(B2_GS)
                            )
                            .data_ptr()
                            .toint(),
                            n_done,
                        )
                    cute.arch.sync_warp()
            if cutlass.const_expr(SF2_W6):
                # ---- (SF2_W6) FC2 scale gather (warp 8: idle in B2IMG mode once
                # routing / early issue is done): every FC2 stage of this CTA, in the
                # producer's order (posted through q2)
                if warp == cutlass.Int32(SF2_WARP):
                    _wait(q2_bar.subview(0), 0)
                    n2 = q2_ent.load(idx=1)
                    gn8 = s_el.load(idx=G_CAP)  # read only after the header's acquire
                    cute.arch.sync_warp()
                    if lane == 0:
                        prims.mbarrier_arrive(q2_bar.subview(Q2D))
                    for i2 in cutlass.range(1, n2 + cutlass.Int32(1), unroll=1):
                        qi = i2 - (i2 // cutlass.Int32(Q2D)) * cutlass.Int32(Q2D)
                        _wait(
                            q2_bar.subview(qi),
                            (i2 // cutlass.Int32(Q2D)) & cutlass.Int32(1),
                        )
                        e_stage = q2_ent.load(idx=qi * cutlass.Int32(4))
                        e_g = q2_ent.load(idx=qi * cutlass.Int32(4) + cutlass.Int32(1))
                        e_blk = q2_ent.load(
                            idx=qi * cutlass.Int32(4) + cutlass.Int32(2)
                        )
                        e_t2 = q2_ent.load(idx=qi * cutlass.Int32(4) + cutlass.Int32(3))
                        cute.arch.sync_warp()
                        if lane == 0:
                            prims.mbarrier_arrive(q2_bar.subview(Q2D + qi))
                        e_here = gn8 - e_g * cutlass.Int32(4)
                        e_here = cutlass.select_(
                            e_here > cutlass.Int32(4), cutlass.Int32(4), e_here
                        )
                        _sf2_cp(
                            lane,
                            e_stage,
                            e_g,
                            e_blk,
                            e_t2,
                            e_here,
                            s_el,
                            sSFA,
                            s2w,
                            ab_full,
                        )
            # ---- (B2IMG) FC2 B loader (warp 5): group g's image once its counter is complete
            if warp == cutlass.Int32(X_WARP):
                if cutlass.const_expr(
                    PAD_X
                ):  # CTA 0 pads every group counter (as warp 4 did before the route bar)
                    if (bx == 0) & (lane < cutlass.Int32(NG_CAP)):
                        nh = g_n - lane * cutlass.Int32(4)
                        nh = cutlass.select_(
                            nh > cutlass.Int32(4), cutlass.Int32(4), nh
                        )
                        nh = cutlass.select_(
                            nh < cutlass.Int32(0), cutlass.Int32(0), nh
                        )
                        pad = cutlass.Int32(CNT_PER_CALL) - nh * cutlass.Int32(
                            UNITS_ROW
                        )
                        if pad > cutlass.Int32(0):
                            for rr in cutlass.range_constexpr(B2_REPL):
                                _red_release_add(
                                    b2cnt.subview(
                                        lane * cutlass.Int32(B2_GS)
                                        + cutlass.Int32(rr * B2_CSTRIDE)
                                    )
                                    .data_ptr()
                                    .toint(),
                                    pad,
                                )
                    cute.arch.sync_warp()
                s2t_shape, s2t_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
                cnt_target = (ep + cutlass.Int32(1)) * cutlass.Int32(CNT_PER_CALL)
                for g in range(
                    cutlass.select_(nb > cutlass.Int32(0), ng, cutlass.Int32(0))
                ):
                    buf = g % cutlass.Int32(NB2BUF)
                    if g >= cutlass.Int32(NB2BUF):
                        _waitd(
                            b2_free.subview(buf),
                            ((g // cutlass.Int32(NB2BUF)) - cutlass.Int32(1))
                            & cutlass.Int32(1),
                            tdbg,
                            101,
                        )
                    caddr = (
                        b2cnt.subview(
                            g * cutlass.Int32(B2_GS)
                            + (bx % cutlass.Int32(B2_REPL)) * cutlass.Int32(B2_CSTRIDE)
                        )
                        .data_ptr()
                        .toint()
                    )
                    if cutlass.const_expr(B2_SHARD > 1):
                        # every lane: acquire shard `lane` (lanes >= B2_SHARD add 0), butterfly sum (warp-uniform)
                        saddr = (
                            b2cnt.subview(
                                g * cutlass.Int32(B2_GS)
                                + lane * cutlass.Int32(B2_CSTRIDE)
                            )
                            .data_ptr()
                            .toint()
                        )
                        sv = cutlass.Int32(0)
                        if lane < cutlass.Int32(B2_SHARD):
                            sv = _ld_acquire_u32(saddr)
                        for off in cutlass.range_constexpr(5):
                            sv = sv + cute.arch.shuffle_sync_bfly(sv, 16 >> off)
                        hs = cutlass.Int32(0)
                        while ((sv - cnt_target) < cutlass.Int32(0)) & (
                            hs < cutlass.Int32(HANG_CAP)
                        ):
                            hs = hs + cutlass.Int32(1)
                            _nanosleep(B2IMG_NS)
                            sv = cutlass.Int32(0)
                            if lane < cutlass.Int32(B2_SHARD):
                                sv = _ld_acquire_u32(saddr)
                            for off in cutlass.range_constexpr(5):
                                sv = sv + cute.arch.shuffle_sync_bfly(sv, 16 >> off)
                        # order the other lanes' acquires before lane 0's async-proxy fence and image copy
                        cute.arch.sync_warp()
                    if lane == 0:
                        if cutlass.const_expr(B2_SHARD == 1):
                            cv = _ld_acquire_u32(caddr)
                            hd2 = cutlass.Int32(0)
                            while ((cv - cnt_target) < cutlass.Int32(0)) & (
                                hd2 < cutlass.Int32(HANG_CAP)
                            ):
                                hd2 = hd2 + cutlass.Int32(1)
                                _nanosleep(B2IMG_NS)
                                cv = _ld_acquire_u32(caddr)
                        _fence_proxy_async_global()
                        prims.mbarrier_arrive_expect_tx(img_bar.subview(buf), IMG_BYTES)
                        prims.cp_async_bulk_shared_cluster_global(
                            sB2.subview(buf * cutlass.Int32(IMG_B2)),
                            img32.subview(g * cutlass.Int32(IMG_BYTES // 4)),
                            img_bar.subview(buf),
                            IMG_B2,
                        )
                        prims.cp_async_bulk_shared_cluster_global(
                            sSFB2.subview(buf * cutlass.Int32(IMG_SF // 4)),
                            img32.subview(
                                g * cutlass.Int32(IMG_BYTES // 4)
                                + cutlass.Int32(IMG_B2 // 4)
                            ),
                            img_bar.subview(buf),
                            IMG_SF,
                        )
                    cute.arch.sync_warp()
                    _waitd(
                        img_bar.subview(buf),
                        (g // cutlass.Int32(NB2BUF)) & cutlass.Int32(1),
                        tdbg,
                        108,
                    )
                    if prims.elect_sync():
                        sd0 = prims.Tcgen05SmemDesc.build(
                            sSFB2.subview(0),
                            leading_byte_offset=16,
                            stride_byte_offset=128,
                            base_offset=0,
                            layout=0,
                        )
                        for a in cutlass.range_constexpr(KT2 * NUM_KB):
                            prims.tcgen05_cp(
                                s2t_shape,
                                _tptr(
                                    base_row,
                                    base_col
                                    + SFB2_COL
                                    + buf * cutlass.Int32(KT2 * NUM_KB * 4)
                                    + a * 4,
                                ),
                                sd0
                                + cutlass.Int64(
                                    (
                                        buf * cutlass.Int32(KT2 * NUM_KB)
                                        + cutlass.Int32(a)
                                    )
                                    * cutlass.Int32(512 >> 4)
                                ),
                                multicast=s2t_mc,
                            )
                        prims.tcgen05_commit(b2_ready.subview(buf))
                    cute.arch.sync_warp()
        # ---- FC2 B builder (warps 5, 8 ..): group g's records (all 48 per (slot,
        # token), the call's tag on every word) into B2 buffer g % 2 (row j M + t,
        # k-tile t2 = record / 16) and its scale atoms, then tcgen05.cp to TMEM.
        if ((warp == X_WARP) | (warp >= cutlass.Int32(8))) & cutlass.Boolean(not B2IMG):
            xi = cutlass.select_(
                warp == cutlass.Int32(X_WARP), cutlass.Int32(0), warp - cutlass.Int32(7)
            )
            if cutlass.const_expr(G_CAP > TOP_K):
                if (bx == 0) & (xi == cutlass.Int32(0)):
                    # rows of slots this call does not use: tag 0, never taken for a later call's
                    for i in range(
                        g_n * cutlass.Int32(M_MAX * UNITS_ROW) + lane,
                        cutlass.Int32(G_CAP * M_MAX * UNITS_ROW),
                        32,
                    ):
                        for rr in cutlass.range_constexpr(HB_REPL):
                            hbuf32.store(
                                (
                                    cutlass.Int32(0),
                                    cutlass.Int32(0),
                                    cutlass.Int32(0),
                                    cutlass.Int32(0),
                                ),
                                idx=i * cutlass.Int32(4)
                                + (cutlass.Int32(rr * HB_STRIDE) if rr else 0),
                                alignment=16,
                            )
            s2t_shape, s2t_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
            sSFB2_8 = cutlass.Array(
                sSFB2.data_ptr(0),
                shape=(NB2BUF * KT2 * NUM_KB * 512,),
                dtype=cutlass.Int8,
                alignment=16,
            )
            sB2_32 = cutlass.Array(
                sB2.data_ptr(0),
                shape=(NB2BUF * KT2 * NUM_BYTES_B2 // 4,),
                dtype=cutlass.Int32,
                alignment=16,
            )
            nrec_cap = 4 * M_MAX * UNITS_ROW  # records of one group
            hb_off = cutlass.Int32(0)
            if cutlass.const_expr(HB_REPL > 1):
                # this CTA's copy of the records
                hb_off = (bx % cutlass.Int32(HB_REPL)) * cutlass.Int32(HB_STRIDE)
            NZ = (nrec_cap + 32 * XW - 1) // (
                32 * XW
            )  # records per lane: lane + 32 (XW z + xi)
            for g in range(
                cutlass.select_(nb > cutlass.Int32(0), ng, cutlass.Int32(0))
            ):
                buf = g % cutlass.Int32(NB2BUF)
                if g >= cutlass.Int32(NB2BUF):
                    _waitd(
                        b2_free.subview(buf),
                        ((g // cutlass.Int32(NB2BUF)) - cutlass.Int32(1))
                        & cutlass.Int32(1),
                        tdbg,
                        101,
                    )
                _waitd(
                    fc1_done.subview(g & cutlass.Int32(3)),
                    (g >> cutlass.Int32(2)) & cutlass.Int32(1),
                    tdbg,
                    4,
                )  # this CTA's own share of group g is out
                n_here = g_n - g * cutlass.Int32(4)
                n_here = cutlass.select_(
                    n_here > cutlass.Int32(4), cutlass.Int32(4), n_here
                )
                nrec = n_here * cutlass.Int32(M_MAX * UNITS_ROW)
                rec0 = g * cutlass.Int32(
                    4 * M_MAX * UNITS_ROW
                )  # rows (4 g + j) M + t are contiguous
                done = cutlass.Int32(0)
                npoll = cutlass.Int32(0)
                got = [[cutlass.Int32(0) for _ in range(3)] for _ in range(NZ)]
                hd3 = cutlass.Int32(0)
                while (done == cutlass.Int32(0)) & (hd3 < cutlass.Int32(HANG_CAP)):
                    hd3 = hd3 + cutlass.Int32(1)
                    ok = cutlass.Boolean(True)
                    for z in cutlass.range_constexpr(NZ):
                        qi = lane + cutlass.Int32(32 * XW * z) + cutlass.Int32(32) * xi
                        if qi < nrec:
                            v4 = prims.load_ext(
                                hbuf32.subview(
                                    (rec0 + qi) * cutlass.Int32(4)
                                    + (hb_off if HB_REPL > 1 else 0)
                                ),
                                dtype=cutlass.Int32,
                                count=4,
                                order="relaxed",
                                scope="gpu",
                            )
                            for wq in cutlass.range_constexpr(3):
                                got[z][wq] = cutlass.Int32(v4[wq])
                            ok = ok & (
                                (
                                    (
                                        (got[z][0] ^ tagv)
                                        | (got[z][1] ^ tagv)
                                        | (got[z][2] ^ tagv)
                                    )
                                    & cutlass.Int32(-16777216)
                                )
                                == cutlass.Int32(0)
                            )
                    npoll = npoll + cutlass.Int32(1)
                    if prims.vote_sync(
                        0xFFFFFFFF, ok, prims.VoteSync.BALLOT
                    ) == cutlass.Int32(-1):
                        done = cutlass.Int32(1)
                    else:
                        _nanosleep(B2_POLL_NS)
                for z in cutlass.range_constexpr(NZ):
                    qi = lane + cutlass.Int32(32 * XW * z) + cutlass.Int32(32) * xi
                    if qi < nrec:
                        w0, w1, w2 = got[z][0], got[z][1], got[z][2]
                        rr = qi // cutlass.Int32(
                            UNITS_ROW
                        )  # row within the group: j M + t = B row n
                        c = qi - rr * cutlass.Int32(UNITS_ROW)
                        t2 = c >> cutlass.Int32(4)
                        cc = (c >> cutlass.Int32(1)) & cutlass.Int32(7)
                        lo = (w0 & cutlass.Int32(0xFFFFFF)) | (w1 << cutlass.Int32(24))
                        hi = ((w1 >> cutlass.Int32(8)) & cutlass.Int32(0xFFFF)) | (
                            (w2 & cutlass.Int32(0xFFFF)) << cutlass.Int32(16)
                        )
                        sB2_32.store(
                            (lo, hi),
                            idx=(
                                (buf * cutlass.Int32(KT2) + t2)
                                * cutlass.Int32(NUM_BYTES_B2)
                                + rr * cutlass.Int32(KT_BYTES)
                                + (cc ^ (rr & cutlass.Int32(7))) * cutlass.Int32(16)
                                + (c & cutlass.Int32(1)) * cutlass.Int32(8)
                            )
                            // cutlass.Int32(4),
                            alignment=8,
                        )
                        # scale of block c: atom (t2, kb = (c % 16) / 4), row rr, byte c % 4
                        sSFB2_8.store(
                            cutlass.Int8(
                                (w2 >> cutlass.Int32(16)) & cutlass.Int32(0xFF)
                            ),
                            idx=(
                                (
                                    (buf * cutlass.Int32(KT2) + t2)
                                    * cutlass.Int32(NUM_KB)
                                    + ((c & cutlass.Int32(15)) >> cutlass.Int32(2))
                                )
                                * cutlass.Int32(128)
                                + rr * cutlass.Int32(4)
                            )
                            * cutlass.Int32(4)
                            + (c & cutlass.Int32(3)),
                        )
                prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
                if cutlass.const_expr(XW > 1):
                    cute.arch.sync_warp()
                    cute.arch.barrier(barrier_id=B2_BAR, number_of_threads=32 * XW)
                cute.arch.sync_warp()
                if (xi == cutlass.Int32(0)) & prims.elect_sync():
                    sd0 = prims.Tcgen05SmemDesc.build(
                        sSFB2.subview(0),
                        leading_byte_offset=16,
                        stride_byte_offset=128,
                        base_offset=0,
                        layout=0,
                    )
                    for a in cutlass.range_constexpr(KT2 * NUM_KB):
                        prims.tcgen05_cp(
                            s2t_shape,
                            _tptr(
                                base_row,
                                base_col
                                + SFB2_COL
                                + buf * cutlass.Int32(KT2 * NUM_KB * 4)
                                + a * 4,
                            ),
                            sd0
                            + cutlass.Int64(
                                (buf * cutlass.Int32(KT2 * NUM_KB) + cutlass.Int32(a))
                                * cutlass.Int32(512 >> 4)
                            ),
                            multicast=s2t_mc,
                        )
                    prims.tcgen05_commit(b2_ready.subview(buf))
        # ---- MMA (warp 7)
        if warp == MMA_WARP:
            idesc1 = prims.Tcgen05MxOmmaInstrDesc.build(
                a_dtype=cutlass.Float4E2M1FN,
                b_dtype=cutlass.Float4E2M1FN,
                scale_format=0,
                n_dim=N1,
                m_dim=MMA_M,
                sparsity_version=0,
            )
            idesc2 = prims.Tcgen05MxOmmaInstrDesc.build(
                a_dtype=cutlass.Float4E2M1FN,
                b_dtype=cutlass.Float4E2M1FN,
                scale_format=0,
                n_dim=N2,
                m_dim=MMA_M,
                sparsity_version=0,
            )
            desc_a0 = prims.Tcgen05SmemDesc.build(
                sA.subview(0),
                leading_byte_offset=16,
                stride_byte_offset=1024,
                base_offset=0,
                layout=2,
            )
            desc_b10 = prims.Tcgen05SmemDesc.build(
                sB1.subview(0),
                leading_byte_offset=16,
                stride_byte_offset=1024,
                base_offset=0,
                layout=2,
            )
            desc_b20 = prims.Tcgen05SmemDesc.build(
                sB2.subview(0),
                leading_byte_offset=16,
                stride_byte_offset=1024,
                base_offset=0,
                layout=2,
            )
            tcol = (base_row << 16) | base_col
            _waitd(b1_ready, 0, tdbg, 5)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)
            stage2 = cutlass.Int32(
                S1R if PW2F else 0
            )  # (PW2F) FC2 ring cursor, swapped in around each FC2 block
            phase2 = cutlass.Int32(0)
            k1 = cutlass.Int32(0)
            k2 = cutlass.Int32(0)
            for p in range(ng + cutlass.Int32(LAG)):
                if p < ng:
                    i0, i1 = _plan_range(s_plan, p)
                    for k in range(_nsegs(i0, i1)):
                        ug, j0, j1 = _seg(i0, i1, k)
                        ab = k1 & cutlass.Int32(1)
                        _waitd(
                            acc_empty.subview(ab),
                            ((k1 >> cutlass.Int32(1)) & cutlass.Int32(1))
                            ^ cutlass.Int32(1),
                            tdbg,
                            102,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        acc_ptr = cutlass.inttoptr(
                            tcol + ACC1_COL + ab * cutlass.Int32(N1), 6, cutlass.Float32
                        )
                        for j in cutlass.range(j0, j1, unroll=1):
                            if cutlass.const_expr(SFA_CP):
                                _waitd(ab_full.subview(stage), phase, tdbg, 6)
                                if cutlass.const_expr(not NOFENCE1):
                                    prims.fence_proxy(
                                        "async_shared",
                                        space=prims.SharedSpace.shared_cta,
                                    )
                            else:
                                _waitd(sfa_ready.subview(stage), phase, tdbg, 6)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            da = desc_a0 + cutlass.Int64(
                                stage * cutlass.Int32(NUM_BYTES_A >> 4)
                            )
                            db = desc_b10 + cutlass.Int64(
                                j * cutlass.Int32(NUM_BYTES_B1 >> 4)
                            )
                            sfa_c = tcol + SFA1_COL + stage * cutlass.Int32(16)
                            sfb_c = tcol + SFB1_COL + j * cutlass.Int32(NUM_KB * 4)
                            if prims.elect_sync():
                                if cutlass.const_expr(SFA_CP):
                                    _sfa_to_tmem(sSFA, stage, base_row, base_col)
                                for kb in cutlass.range_constexpr(NUM_KB):
                                    prims.tcgen05_mma_block_scale(
                                        prims.MMABlockScaleKind.MXF4NVF4,
                                        prims.CTAGroup.CTA_1,
                                        acc_ptr,
                                        da + INC_K * kb,
                                        db + INC_K * kb,
                                        idesc1,
                                        (j > j0) | cutlass.Boolean(kb > 0),
                                        cutlass.inttoptr(
                                            sfa_c + kb * 4, 6, cutlass.Int32
                                        ),
                                        cutlass.inttoptr(
                                            sfb_c + kb * 4, 6, cutlass.Int32
                                        ),
                                        scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                                    )
                                prims.tcgen05_commit(ab_empty.subview(stage))
                            stage, phase = _adv1(stage, phase)
                        if prims.elect_sync():
                            prims.tcgen05_commit(acc_full.subview(ab))
                        k1 = k1 + cutlass.Int32(1)
                # a CTA with no FC2 blocks (nb 0: NCTA > 192) has no FC2 B pipeline at all
                if (p >= cutlass.Int32(LAG)) & (nb > cutlass.Int32(0)):
                    if cutlass.const_expr(PW2F):
                        stage, stage2 = stage2, stage
                        phase, phase2 = phase2, phase
                    g = p - cutlass.Int32(LAG)
                    buf = g % cutlass.Int32(NB2BUF)
                    _waitd(
                        b2_ready.subview(buf),
                        (g // cutlass.Int32(NB2BUF)) & cutlass.Int32(1),
                        tdbg,
                        103,
                    )
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    for bb in range(nb):  # noqa: B007
                        a2 = k2 & cutlass.Int32(1)
                        _waitd(
                            acc2_empty.subview(a2),
                            ((k2 >> cutlass.Int32(1)) & cutlass.Int32(1))
                            ^ cutlass.Int32(1),
                            tdbg,
                            104,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        acc2_ptr = cutlass.inttoptr(
                            tcol + ACC2_COL + a2 * cutlass.Int32(N2), 6, cutlass.Float32
                        )
                        for t2 in cutlass.range(KT2, unroll=1 if ROLL_FC2 else KT2):
                            if cutlass.const_expr(SFA_CP):
                                _waitd(ab_full.subview(stage), phase, tdbg, 7)
                                prims.fence_proxy(
                                    "async_shared", space=prims.SharedSpace.shared_cta
                                )
                            else:
                                _waitd(sfa_ready.subview(stage), phase, tdbg, 7)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            da = desc_a0 + cutlass.Int64(
                                stage * cutlass.Int32(NUM_BYTES_A >> 4)
                            )
                            db = desc_b20 + cutlass.Int64(
                                (buf * cutlass.Int32(KT2) + cutlass.Int32(t2))
                                * cutlass.Int32(NUM_BYTES_B2 >> 4)
                            )
                            sfa_c = tcol + SFA1_COL + stage * cutlass.Int32(16)
                            sfb_c = (
                                tcol
                                + SFB2_COL
                                + buf * cutlass.Int32(KT2 * NUM_KB * 4)
                                + cutlass.Int32(t2 * NUM_KB * 4)
                            )
                            if prims.elect_sync():
                                if cutlass.const_expr(SFA_CP):
                                    _sfa_to_tmem(sSFA, stage, base_row, base_col)
                                for kb in cutlass.range_constexpr(NUM_KB):
                                    prims.tcgen05_mma_block_scale(
                                        prims.MMABlockScaleKind.MXF4NVF4,
                                        prims.CTAGroup.CTA_1,
                                        acc2_ptr,
                                        da + INC_K * kb,
                                        db + INC_K * kb,
                                        idesc2,
                                        cutlass.Boolean(t2 > cutlass.Int32(0))
                                        | cutlass.Boolean(kb > 0),
                                        cutlass.inttoptr(
                                            sfa_c + kb * 4, 6, cutlass.Int32
                                        ),
                                        cutlass.inttoptr(
                                            sfb_c + kb * 4, 6, cutlass.Int32
                                        ),
                                        scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                                    )
                                prims.tcgen05_commit(ab_empty.subview(stage))
                            stage, phase = _adv2(stage, phase)
                        if prims.elect_sync():
                            prims.tcgen05_commit(acc2_full.subview(a2))
                        k2 = k2 + cutlass.Int32(1)
                    if cutlass.const_expr(PW2F):
                        stage, stage2 = stage2, stage
                        phase, phase2 = phase2, phase
        # ---- epilogue warps (0-3)
        if warp < 4:
            row_w = base_row + warp * 32
            for r in cutlass.range_constexpr(
                (6 * G_CAP + EPI_THREADS - 1) // EPI_THREADS
            ):
                qi = tidx + cutlass.Int32(r * EPI_THREADS)
                sl_e = qi // cutlass.Int32(6)
                if (qi < cutlass.Int32(6 * G_CAP)) & (sl_e < g_n):
                    fld = qi - sl_e * cutlass.Int32(6)
                    s_esc.store(
                        cutlass.Float32(
                            escal.load(
                                idx=fld * cutlass.Int32(E_LOCAL) + s_el.load(idx=sl_e)
                            )
                        ),
                        idx=sl_e * cutlass.Int32(8) + fld,
                    )
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=EPI_THREADS)
            k1 = cutlass.Int32(0)
            k2 = cutlass.Int32(0)
            SH_ADD = SHARED
            wys = [cutlass.Int32(0) for _ in range(NB2_CAP * M_MAX // 4 + 1)]
            pars = [cutlass.Int32(0) for _ in range(NB2_CAP * M_MAX // 4 + 1)]
            ard = tuple(
                cutlass.Int64(0)
                for _ in range(AR_W + 2 if AR_G3 else (AR_ND + 3 if AR_V3 else 4))
            )
            for p in range(ng + cutlass.Int32(LAG)):
                if cutlass.const_expr(AR_G3):
                    if (p == ng + cutlass.Int32(LAG - 1)) & ar_on:
                        # after the grid dependency: this call's slot buffer (call % 3), then every rank's base
                        callv = _ld_volatile_s32(
                            cutlass.Int64(
                                _ld_relaxed_u64(arp.subview(2).data_ptr().toint())
                            )
                        )
                        bp = cutlass.Int64(
                            _ld_relaxed_u64(arp.subview(1).data_ptr().toint())
                        )
                        ard = tuple(
                            cutlass.Int64(_ld_relaxed_u64(bp + cutlass.Int64(8 * r_)))
                            for r_ in range(AR_W)
                        ) + (
                            cutlass.Int64(
                                _ld_relaxed_u64(arp.subview(3).data_ptr().toint())
                            ),
                            cutlass.Int64(
                                callv - (callv // cutlass.Int32(3)) * cutlass.Int32(3)
                            ),
                        )
                elif cutlass.const_expr(AR_PRE):
                    if (p == ng + cutlass.Int32(LAG - 1)) & ar_on:
                        # one 8-byte load each (the vectorizer merges adjacent
                        # arp.load()s into 16-byte reads at 8-byte offsets: misaligned)
                        if cutlass.const_expr(AR_V3):
                            pp = cutlass.Int64(
                                _ld_relaxed_u64(arp.subview(1).data_ptr().toint())
                            )
                            ard = tuple(
                                cutlass.Int64(
                                    _ld_relaxed_u64(pp + cutlass.Int64(8 * r_))
                                )
                                for r_ in range(AR_ND)
                            ) + tuple(
                                cutlass.Int64(
                                    _ld_relaxed_u64(arp.subview(i_).data_ptr().toint())
                                )
                                for i_ in (3, 4, 5)
                            )
                        else:
                            ard = tuple(
                                cutlass.Int64(
                                    _ld_relaxed_u64(arp.subview(i_).data_ptr().toint())
                                )
                                for i_ in (1, 3, 4, 5)
                            )
                        for bt in cutlass.range_constexpr((NB2_CAP * M_MAX + 3) // 4):
                            f_pair = warp + cutlass.Int32(bt * 4)
                            f_bb = f_pair // cutlass.Int32(M_MAX)
                            f_t = f_pair - f_bb * cutlass.Int32(M_MAX)
                            f_cl = cutlass.select_(
                                f_bb < nb,
                                (blk0 + f_bb) * cutlass.Int32(R2),
                                cutlass.Int32(0),
                            )
                            if cutlass.const_expr(AR_IN):
                                # this call consumed parity s_par of the shared
                                # workspace; the next boundary's is its complement (CTA
                                # 0 flips the flags only at teardown)
                                pars[bt] = cutlass.select_(
                                    arp.load(idx=6) != cutlass.Int64(0),
                                    s_par.load(
                                        idx=f_t * cutlass.Int32(AR_IN_CL)
                                        + f_cl // cutlass.Int32(H // AR_IN_CL)
                                    )
                                    ^ cutlass.Int32(1),
                                    _ar_parity(arp, f_t, f_cl),
                                )
                            else:
                                pars[bt] = _ar_parity(arp, f_t, f_cl)
                if cutlass.const_expr(SH_ADD):
                    if p == ng + cutlass.Int32(LAG - 1):
                        # this CTA's y_sh words, loaded while the last group's FC2 runs (re-polled in finalize)
                        for bt in cutlass.range_constexpr((NB2_CAP * M_MAX + 3) // 4):
                            f_pair = warp + cutlass.Int32(bt * 4)
                            f_bb = f_pair // cutlass.Int32(M_MAX)
                            f_t = f_pair - f_bb * cutlass.Int32(M_MAX)
                            f_ch = cutlass.select_(
                                f_bb < nb,
                                f_t * cutlass.Int32(H)
                                + (blk0 + f_bb) * cutlass.Int32(R2)
                                + cutlass.Int32(4) * (lane & cutlass.Int32(7))
                                + (lane >> cutlass.Int32(3)),
                                cutlass.Int32(0),
                            )
                            wys[bt] = _ld_relaxed_u32(
                                dnx.subview(f_ch).data_ptr().toint()
                            )
                if p < ng:
                    i0, i1 = _plan_range(s_plan, p)
                    for k in range(_nsegs(i0, i1)):
                        ug, j0, j1 = _seg(i0, i1, k)
                        ab = k1 & cutlass.Int32(1)
                        slot = p * cutlass.Int32(4) + _ug_slot(ug)
                        cblk = _ug_c(ug)
                        unit = slot * cutlass.Int32(U1) + cblk
                        _waitd(
                            acc_full.subview(ab),
                            (k1 >> cutlass.Int32(1)) & cutlass.Int32(1),
                            tdbg,
                            105,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        acc = prims.tcgen05_ld(
                            "32x32b",
                            _tptr(
                                row_w,
                                base_col + ACC1_COL + ab * cutlass.Int32(N1),
                                cutlass.Float32,
                            ),
                            num=N1,
                        )
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.mbarrier_arrive(acc_empty.subview(ab))
                        pb = ab * cutlass.Int32(M_MAX * C1 * R1)
                        for t in cutlass.range_constexpr(M_MAX):
                            part = cutlass.Float32(0.0)
                            for w in cutlass.range_constexpr(4):
                                part = cutlass.select_(
                                    warp == cutlass.Int32(w),
                                    cutlass.Float32(acc[QS * w + t]),
                                    part,
                                )
                            s_part.store(part, idx=pb + (t * C1 + warp) * R1 + lane)
                        cute.arch.sync_warp()
                        cute.arch.barrier(
                            barrier_id=EPI_BAR, number_of_threads=EPI_THREADS
                        )
                        for tr in cutlass.range_constexpr((M_MAX + 3) // 4):
                            t = warp + cutlass.Int32(4 * tr)
                            if t < cutlass.Int32(M_MAX):
                                xv = _add_rn(
                                    _add_rn(
                                        _add_rn(
                                            s_part.load(
                                                idx=pb + (t * C1 + 0) * R1 + lane
                                            ),
                                            s_part.load(
                                                idx=pb + (t * C1 + 1) * R1 + lane
                                            ),
                                        ),
                                        s_part.load(idx=pb + (t * C1 + 2) * R1 + lane),
                                    ),
                                    s_part.load(idx=pb + (t * C1 + 3) * R1 + lane),
                                )
                                finish = cutlass.Int32(1)
                                if (j0 != cutlass.Int32(0)) | (j1 != cutlass.Int32(S1)):
                                    paddr = (
                                        partials.subview(
                                            (unit * cutlass.Int32(M_MAX) + t)
                                            * cutlass.Int32(R1)
                                            + lane
                                        )
                                        .data_ptr()
                                        .toint()
                                    )
                                    if j0 == cutlass.Int32(0):
                                        _st_relaxed_u32(
                                            paddr, xv.bitcast(cutlass.Int32)
                                        )
                                        finish = cutlass.Int32(0)
                                    else:
                                        pbits = _ld_relaxed_u32(paddr)
                                        spins = cutlass.Int32(0)
                                        hd4 = cutlass.Int32(0)
                                        while (
                                            pbits == cutlass.Int32(PARTIAL_EMPTY)
                                        ) & (hd4 < cutlass.Int32(HANG_CAP)):
                                            hd4 = hd4 + cutlass.Int32(1)
                                            pbits = _ld_relaxed_u32(paddr)
                                            spins = spins + cutlass.Int32(1)
                                        _st_relaxed_u32(
                                            paddr, cutlass.Int32(PARTIAL_EMPTY)
                                        )
                                        xv = _add_rn(pbits.bitcast(cutlass.Float32), xv)
                                        cute.arch.sync_warp()
                                if finish == cutlass.Int32(1):
                                    if cutlass.const_expr(
                                        FC1_PER_TOKEN
                                    ):  # per-token FC1 dequant (acc * s_t, then a)
                                        xv = _mul_rn(xv, s_pts.load(idx=t))
                                    gate_raw = cute.arch.shuffle_sync_bfly(xv, 8)
                                    is_up = (
                                        (lane >> cutlass.Int32(3)) & cutlass.Int32(1)
                                    ) == cutlass.Int32(0)
                                    es = slot * cutlass.Int32(8)
                                    qv, sfb = _swiglu_quant(
                                        xv,
                                        gate_raw,
                                        is_up,
                                        s_esc.load(idx=es),
                                        s_esc.load(idx=es + cutlass.Int32(1)),
                                        s_esc.load(idx=es + cutlass.Int32(2)),
                                        s_esc.load(idx=es + cutlass.Int32(3)),
                                        s_esc.load(idx=es + cutlass.Int32(4)),
                                    )
                                    byte = _e2m1x2(
                                        cute.arch.shuffle_sync_down(qv, 16), qv
                                    )
                                    bt = [
                                        cute.arch.shuffle_sync(byte, i)
                                        & cutlass.Int32(0xFF)
                                        for i in range(8)
                                    ]
                                    rw0 = (
                                        bt[0]
                                        | (bt[1] << cutlass.Int32(8))
                                        | (bt[2] << cutlass.Int32(16))
                                        | tagv
                                    )
                                    rw1 = (
                                        bt[3]
                                        | (bt[4] << cutlass.Int32(8))
                                        | (bt[5] << cutlass.Int32(16))
                                        | tagv
                                    )
                                    rw2 = (
                                        bt[6]
                                        | (bt[7] << cutlass.Int32(8))
                                        | (
                                            (sfb & cutlass.Int32(0xFF))
                                            << cutlass.Int32(16)
                                        )
                                        | tagv
                                    )
                                    if cutlass.const_expr(B2IMG):
                                        if lane == 0:
                                            # B2 row n = j M + t, k-tile c / 16, 16-byte
                                            # chunk (c / 2) % 8 (swizzled), half c % 2;
                                            # scale: atom (t2, kb = (c % 16) / 4), row n
                                            # at byte 16 n, byte c % 4
                                            n = _ug_slot(ug) * cutlass.Int32(M_MAX) + t
                                            t2i = cblk >> cutlass.Int32(4)
                                            cci = (
                                                cblk >> cutlass.Int32(1)
                                            ) & cutlass.Int32(7)
                                            gbase = p * cutlass.Int32(IMG_BYTES)
                                            lo = (
                                                bt[0]
                                                | (bt[1] << cutlass.Int32(8))
                                                | (bt[2] << cutlass.Int32(16))
                                                | (bt[3] << cutlass.Int32(24))
                                            )
                                            hi = (
                                                bt[4]
                                                | (bt[5] << cutlass.Int32(8))
                                                | (bt[6] << cutlass.Int32(16))
                                                | (bt[7] << cutlass.Int32(24))
                                            )
                                            img32.store(
                                                (lo, hi),
                                                idx=(
                                                    gbase
                                                    + t2i * cutlass.Int32(NUM_BYTES_B2)
                                                    + n * cutlass.Int32(KT_BYTES)
                                                    + (cci ^ (n & cutlass.Int32(7)))
                                                    * cutlass.Int32(16)
                                                    + (cblk & cutlass.Int32(1))
                                                    * cutlass.Int32(8)
                                                )
                                                // cutlass.Int32(4),
                                                alignment=8,
                                            )
                                            img8 = cutlass.Array(
                                                img32.data_ptr(0),
                                                shape=(NG_CAP * IMG_BYTES,),
                                                dtype=cutlass.Int8,
                                                alignment=16,
                                            )
                                            img8.store(
                                                cutlass.Int8(sfb & cutlass.Int32(0xFF)),
                                                idx=gbase
                                                + cutlass.Int32(IMG_B2)
                                                + (
                                                    (
                                                        t2i * cutlass.Int32(NUM_KB)
                                                        + (
                                                            (cblk & cutlass.Int32(15))
                                                            >> cutlass.Int32(2)
                                                        )
                                                    )
                                                    * cutlass.Int32(512)
                                                )
                                                + n * cutlass.Int32(16)
                                                + (cblk & cutlass.Int32(3)),
                                            )
                                    else:
                                        if cutlass.const_expr(HB_REPL == 1):
                                            if lane == 0:
                                                hbuf32.store(
                                                    (rw0, rw1, rw2, tagv),
                                                    idx=(
                                                        (
                                                            slot * cutlass.Int32(M_MAX)
                                                            + t
                                                        )
                                                        * cutlass.Int32(UNITS_ROW)
                                                        + cblk
                                                    )
                                                    * cutlass.Int32(4),
                                                    alignment=16,
                                                )
                                        else:
                                            if lane < HB_REPL:  # copy `lane`
                                                hbuf32.store(
                                                    (rw0, rw1, rw2, tagv),
                                                    idx=(
                                                        (
                                                            slot * cutlass.Int32(M_MAX)
                                                            + t
                                                        )
                                                        * cutlass.Int32(UNITS_ROW)
                                                        + cblk
                                                    )
                                                    * cutlass.Int32(4)
                                                    + lane * cutlass.Int32(HB_STRIDE),
                                                    alignment=16,
                                                )
                        k1 = k1 + cutlass.Int32(1)
                    prims.mbarrier_arrive(
                        fc1_done.subview(p & cutlass.Int32(3))
                    )  # warp 5 may poll group p
                if p >= cutlass.Int32(LAG):
                    g = p - cutlass.Int32(LAG)
                    for bb in range(nb):
                        a2 = k2 & cutlass.Int32(1)
                        _waitd(
                            acc2_full.subview(a2),
                            (k2 >> cutlass.Int32(1)) & cutlass.Int32(1),
                            tdbg,
                            106,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        yv = prims.tcgen05_ld(
                            "32x32b",
                            _tptr(
                                row_w,
                                base_col
                                + ACC2_COL
                                + a2 * cutlass.Int32(N2)
                                + warp * cutlass.Int32(M_MAX),
                                cutlass.Float32,
                            ),
                            num=LD_POW2,
                        )
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.mbarrier_arrive(acc2_empty.subview(a2))
                        sl = g * cutlass.Int32(4) + warp
                        if sl < g_n:
                            a2s = s_esc.load(
                                idx=sl * cutlass.Int32(8) + cutlass.Int32(5)
                            )
                            for t in cutlass.range_constexpr(M_MAX):
                                kp = s_kpos.load(
                                    idx=sl * cutlass.Int32(M_MAX) + cutlass.Int32(t)
                                )
                                if kp >= cutlass.Int32(0):
                                    ys.store(
                                        _bf16_round(
                                            _mul_rn(cutlass.Float32(yv[t]), a2s)
                                        ),
                                        idx=(
                                            (
                                                bb * cutlass.Int32(M_MAX)
                                                + cutlass.Int32(t)
                                            )
                                            * cutlass.Int32(TOP_K)
                                            + kp
                                        )
                                        * cutlass.Int32(R2)
                                        + lane,
                                    )
                        k2 = k2 + cutlass.Int32(1)
                    prims.mbarrier_arrive(b2_free.subview(g % cutlass.Int32(NB2BUF)))
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=EPI_THREADS)
            # finalize: (block bb, token t) per warp round: the fma chain over t's top-4 in order
            for bt in cutlass.range_constexpr((NB2_CAP * M_MAX + 3) // 4):
                pair = warp + cutlass.Int32(bt * 4)
                bb = pair // cutlass.Int32(M_MAX)
                t = pair - bb * cutlass.Int32(M_MAX)
                if bb < nb:
                    acc_o = cutlass.Float32(0.0)
                    for k in cutlass.range_constexpr(TOP_K):
                        acc_o = _fma_rn(
                            s_w.load(idx=t * cutlass.Int32(TOP_K) + cutlass.Int32(k)),
                            ys.load(
                                idx=(
                                    (bb * cutlass.Int32(M_MAX) + t)
                                    * cutlass.Int32(TOP_K)
                                    + cutlass.Int32(k)
                                )
                                * cutlass.Int32(R2)
                                + lane
                            ),
                            acc_o,
                        )
                    ch = (
                        t * cutlass.Int32(H)
                        + (blk0 + bb) * cutlass.Int32(R2)
                        + cutlass.Int32(4) * (lane & cutlass.Int32(7))
                        + (lane >> cutlass.Int32(3))
                    )
                    if cutlass.const_expr(SH_ADD):
                        # + y_sh (this row's shared output, from its shared CTA), as shared_output.add_(routed_output)
                        wy = wys[bt]
                        hd5 = cutlass.Int32(0)
                        while (
                            ((wy >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)) != tag16
                        ) & (hd5 < cutlass.Int32(HANG_CAP)):
                            hd5 = hd5 + cutlass.Int32(1)
                            wy = _ld_relaxed_u32(dnx.subview(ch).data_ptr().toint())
                        acc_o = _add_rn(
                            (wy << cutlass.Int32(16)).bitcast(cutlass.Float32),
                            _bf16_round(acc_o),
                        )
                    out.store(cutlass.BFloat16(acc_o), idx=ch)
                    if ar_on:
                        cl = (
                            (blk0 + bb) * cutlass.Int32(R2)
                            + cutlass.Int32(4) * (lane & cutlass.Int32(7))
                            + (lane >> cutlass.Int32(3))
                        )
                        if cutlass.const_expr(AR_G3):
                            _ar_push_g3(
                                ard,
                                _bf16_round(acc_o),
                                ((lane >> cutlass.Int32(3)) & cutlass.Int32(1))
                                == cutlass.Int32(0),
                                t,
                                cl,
                            )
                        elif cutlass.const_expr(AR_PRE):
                            _ar_push_pre(
                                ard,
                                pars[bt],
                                _bf16_round(acc_o),
                                ((lane >> cutlass.Int32(3)) & cutlass.Int32(1))
                                == cutlass.Int32(0),
                                t,
                                cl,
                            )
                        else:
                            _ar_push(
                                arp,
                                _bf16_round(acc_o),
                                ((lane >> cutlass.Int32(3)) & cutlass.Int32(1))
                                == cutlass.Int32(0),
                                t,
                                cl,
                            )

        # ---- teardown
        if cutlass.const_expr(AR_IN):
            if arp.load(idx=6) != cutlass.Int64(0):
                ardesc = arp.load(idx=6)
                if (bx == 0) & (tidx < cutlass.Int32(M_MAX * AR_IN_CL)):
                    _st_relaxed_u32(
                        cutlass.Int64(_ld_relaxed_u64(ardesc + cutlass.Int64(8)))
                        + cutlass.Int64(4) * cutlass.Int64(tidx),
                        s_par.load(idx=tidx) ^ cutlass.Int32(1),
                    )
        if tidx == 0:
            epochs.store(ep + cutlass.Int32(1), idx=bx)
        if cutlass.const_expr(PF_NEXT and PF_WHERE == 0):
            if warp == 0:
                _pf_next(arp, bx, lane, NCTA)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.sync_warp()
        prims.barrier_cta_sync(0)
        if ar_on | cutlass.Boolean(LATE_TRIGGER):
            # the all-reduce polls the pushed slots without a grid wait: release it after every push of this CTA
            cute.arch.griddepcontrol_launch_dependents()
        if warp == MMA_WARP:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            prims.tcgen05_dealloc(
                cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Float32), TMEM_COLS
            )


@cute.jit
def m3_moe(
    a1_tensor: cute.Tensor,
    sfa1_tensor: cute.Tensor,
    a2_tensor: cute.Tensor,
    sfa2_tensor: cute.Tensor,
    logits: cute.Tensor,
    bias: cute.Tensor,
    x32: cute.Tensor,
    xsf32: cute.Tensor,
    escal: cute.Tensor,
    plan: cute.Tensor,
    partials: cute.Tensor,
    hbuf32: cute.Tensor,
    epochs: cute.Tensor,
    out: cute.Tensor,
    route_dbg: cute.Tensor,
    tdbg: cute.Tensor,
    xb32: cute.Tensor,
    rw: cute.Tensor,
    sfsc: cute.Tensor,
    lgbuf: cute.Tensor,
    guw: cute.Tensor,
    gusf: cute.Tensor,
    dnw: cute.Tensor,
    dnsf: cute.Tensor,
    hx: cute.Tensor,
    dnx: cute.Tensor,
    arp: cute.Tensor,
    s1w: cute.Tensor,
    s2w: cute.Tensor,
    img32: cute.Tensor,
    b2cnt: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    R1S = TWO_I  # FC1 rows per expert
    R2S = H  # FC2 rows per expert
    R1E = TWO_I  # FC1 rows between experts
    R2E = H
    tma_a1_desc = cuda.create_tensor_map_tiled(
        global_address=a1_tensor.iterator.toint(),
        dtype=cutlass.Uint8,
        global_dims=[H // 2, R1S, E_LOCAL],
        global_strides=[H // 2 // 16, R1E * (H // 2) // 16],
        box_dims=(KT_BYTES, R1, 1),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_a2_desc = cuda.create_tensor_map_tiled(
        global_address=a2_tensor.iterator.toint(),
        dtype=cutlass.Uint8,
        global_dims=[I_TP // 2, R2S, E_LOCAL],
        global_strides=[I_TP // 2 // 16, R2E * (I_TP // 2) // 16],
        box_dims=(KT_BYTES, R2, 1),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_s1_desc = cuda.create_tensor_map_tiled(
        global_address=sfa1_tensor.iterator.toint(),
        dtype=cutlass.Uint8,
        global_dims=[128, E_LOCAL * TWO_I * (H // 16) // 128],
        global_strides=[128 // 16],
        box_dims=(128, NUM_BYTES_SFA_RAW // 128),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_s2_desc = cuda.create_tensor_map_tiled(
        global_address=sfa2_tensor.iterator.toint(),
        dtype=cutlass.Uint8,
        global_dims=[128, E_LOCAL * H * (I_TP // 16) // 128],
        global_strides=[128 // 16],
        box_dims=(128, NUM_BYTES_SFA_RAW // 4 // 128),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_gu_desc = tma_a1_desc
    tma_dn_desc = tma_a2_desc
    if cutlass.const_expr(SHARED and not SH_PRETILE):
        tma_gu_desc = cuda.create_tensor_map_tiled(
            global_address=guw.iterator.toint(),
            dtype=cutlass.Uint8,
            global_dims=[H, 2 * SH_I],
            global_strides=[H // 16],
            box_dims=(KT_BYTES, 32),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )
        tma_dn_desc = cuda.create_tensor_map_tiled(
            global_address=dnw.iterator.toint(),
            dtype=cutlass.Uint8,
            global_dims=[SH_I, H],
            global_strides=[SH_I // 16],
            box_dims=(KT_BYTES, 128),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )
    m3_moe_kernel(
        tma_a1_desc,
        tma_a2_desc,
        tma_s1_desc,
        tma_s2_desc,
        tma_gu_desc,
        tma_dn_desc,
        logits,
        bias,
        x32,
        xsf32,
        escal,
        plan,
        partials,
        hbuf32,
        epochs,
        out,
        route_dbg,
        tdbg,
        xb32,
        rw,
        sfsc,
        lgbuf,
        guw,
        gusf,
        dnw,
        dnsf,
        hx,
        dnx,
        arp,
        s1w,
        s2w,
        img32,
        b2cnt,
    ).launch(
        grid=(NCTA + NSH, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(1, 1, 1),
        stream=stream,
        use_pdl=True,
    )
