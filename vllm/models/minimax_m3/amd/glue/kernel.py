# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MiniMax-M3.1 decode glue (TP4, gfx950, 17 .. 192 rows): one persistent FlyDSL launch
from the unreduced o_proj partial to the MoE inputs, ahead of AITER's A16W4 gemm1 /
gemm2.

Stages of the default build (`policy`; stage walk, BLOCKS x THREADS, S rows, G = S / 4
rows a TP owner):

- rs push (CTAs 0 .. 127, 48 columns a task): this rank's partial, rows of owner g ->
  rank g's RS inbox [this rank]. Transport "lamport" (default): plain bf16 16-byte
  stores (lane = (row, vector)) into the owner's sentinel inbox, the 3 peers only;
  lpmix 4: a lane's j-th store goes to peer (j + row // 4) % 3, so each destination
  gets 384-byte (3-line) chunks. "tagged": tagged triple pairs (3 bf16 + 16-bit tag in
  one 64-bit store); mixdst: each store instruction's four 16-lane groups go to four
  different ranks.
- rs reduce (CTAs 0 .. 127): this rank's G rows, the 4 ranks' partials summed in rank
  order (f32) -> bf16 h (the own partial read straight from `part` in lamport builds),
  pushed to every rank's AG slot [this rank]. lpag (lamport, rows >= LP_AG_MIN_ROWS):
  plain bf16 16-byte stores into the AG sentinel inboxes, this rank's own first
  (ownfirst); else tagged triples. Mode 2 (the residual sharded by owner rows):
  bf16(h + this rank's residual rows).
- own rows (orouter: CTAs 128 .. 128 + G - 1, which issued no remote stores): this
  rank's G owner rows (reduced here, landing first): h + residual -> residual out,
  Gemma RMSNorm -> normed (gemm1 input), MXFP8 copy; xfirst: normed stored, drained and
  XRDY published before residual out and the MXFP8 copy.
- router (CTAs 128 .. 255: 8 tiles of 16 experts x 16 K parts) on the G owner rows
  only: the FP32 weight (loaded at kernel entry) as bf16 hi + mid + lo (exact; rbf16) x
  normed on 16x16x16 bf16 MFMAs -> partial logits.
- route (CTAs 128 .. 128 + G - 1, a CTA an owner row): parts summed in order, sigmoid,
  + bias (prefetched) for selection, top-4 (ties to the lower expert), weights =
  sigmoid / sum x 2.0 (the routed scale) -> tagged (id, weight) pairs into every rank's
  peer heap (the only cross-rank routing exchange).
- peer rows (CTAs 0 .. 127, and 128 + G .. past 128 rows): the 3G gathered rows:
  residual out, normed, MXFP8.
- sort (CTA 255): every rank's top-k pairs (one poll batch) -> top-k ids / weights for
  all rows + AITER moe_sorting's outputs (block 16); fastsort: one-wave ds_bpermute
  scan, (expert, slot) padding threads.
- re-arm (lamport): every CTA, after a barrier at its end, writes the sentinel back over
  the inbox slots it consumed (seam CTAs: their task's RS segments; row CTAs: their
  row's AG segments).
- finish (afinish): the CTA whose device-counter add completes the launch's BLOCKS
  moves the epoch.

Every option is a GlueBuild field (None = `policy(rows)`, the measured per-width choice,
MI355X TP4; options measured and left off: rowvec, agdefer, rnorm, prepoll outside 48 ..
64 rows; lamport: lpmix 1, lpprobe, lpack).

mode: 0 = load only (compile), 1 = full residual (every row on every rank), 2 = residual
sharded by owner rows.

Lamport transport (specialized_decode/mono/lamport.py): a payload is its own flag. The
inboxes (peer buffer past the tagged halves, two parities by epoch & 1) hold the
sentinel 0xFFFF in every half-word until a push lands; a reader re-loads until none of
the half-words it uses is the sentinel, and the CTA re-arms them after a barrier at its
end. The host fills the inboxes with 0xFF at setup and at every reset (`arm_peer`).
Numerics are the tagged build's: the same rank-order sums, the same row-stage lane
layout (outputs bitwise equal to the tagged kernel's).

Sync, tags, the device-side epoch and fail-stop are the upstream M3 monokernel's
(specialized_decode/mono/sync.py, deepseek_v41 mono/common): every wait (tagged pair,
triple or Lamport slot) is bounded by WAIT_TICKS; a timeout sets the error record [1,
rank, epoch, site] on every rank and every waiter that sees it gives up."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from vllm.models.deepseek_v41.amd.mono.common.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    buf_load,
    buf_store,
    butterfly,
    fp8_pack4,
    hw_exp2,
    hw_rcp,
    lane_gather,
    memrealtime,
    peer_bases,
    traced,
)
from vllm.models.deepseek_v41.amd.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
    pair_layout,
    sel,
)
from vllm.models.deepseek_v41.amd.mono.common.sync import Addr, Space, preg, sreg
from vllm.models.minimax_m3.amd.specialized_decode.mono import lamport as lp
from vllm.models.minimax_m3.amd.specialized_decode.mono import sync as msync
from vllm.models.minimax_m3.amd.specialized_decode.mono.sync import (
    WAIT_TICKS,
    Mailbox,
    cta_sync,
    error_word,
    load32,
    load_dev,
    store16,
    store32,
    store64,
    sync_acquire,
    sync_release,
)

_STREAM = fx.Stream(None)
HIDDEN = 6144
EXPERTS = 128
TOPK = 4
ROUTED_SCALE = 2.0
EPS = 1e-6
LOG2E = 1.4426950408889634
TP = 4
MIN_ROWS, MAX_ROWS = 17, 192
GMAX = MAX_ROWS // TP  # rows a TP owner

TRIPLES = HIDDEN // 3  # 2048 a row
TASK_TRIPLES = 16  # a seam task's triples (48 columns)
SEAM_TASKS = TRIPLES // TASK_TRIPLES  # 128: CTAs 0 .. 127, a row's SSQ parts
ROW_LANES = THREADS // TASK_TRIPLES  # 32 rows a seam pass
ROUTER_PARTS = 16
ROUTER_K = HIDDEN // ROUTER_PARTS  # 384
ROUTER_TASKS = EXPERTS // 16 * ROUTER_PARTS  # 128
SEAM0, NORM0, ROUTER0, ROUTE0 = 0, 0, 128, 0

ERROR_BYTES = 256
RS_PAIRS = TP * GMAX * TRIPLES
AG_PAIRS = TP * GMAX * TRIPLES
ROUTE_P0 = (
    RS_PAIRS + AG_PAIRS
)  # orouter: the top-k exchange, [row][4 ids, 4 weights] tagged pairs
ROUTE_PAIRS = TP * GMAX * 2 * TOPK
HALF_BYTES = (RS_PAIRS + AG_PAIRS + ROUTE_PAIRS) * 8
# payload + flag layout of a parity half (bytes): RS / AG bf16 rows [rank][row][col],
# then a flag pair a (rank, task)
TASK_VEC = 48 // 8  # 16-byte vectors of a task's 48 columns
EPOCH_MARKS = 4
EPOCH_WORDS = EPOCH_MARKS + BLOCKS
SITE_RS, SITE_AG, SITE_NORM, SITE_ROUTER, SITE_ROUTE, SITE_EPOCH = 1, 2, 3, 4, 5, 9
EPOCH_LIMIT = (1 << 17) - (1 << 12)

# the kernel and the common transport / sync code it inlines
_SOURCES = hashlib.sha256(
    b"".join(Path(f).read_bytes() for f in (__file__, lp.__file__, msync.__file__))
).hexdigest()[:16]


# Lamport (sentinel) inboxes (transport "lamport_rs" / "lamport"), past the tagged
# halves: plain bf16 payload, no tags, no flags. Every half-word holds the sentinel
# 0xFFFF until a push lands (finite bf16 never has it; a 0xFFFF in pushed data is
# rewritten 0x7FFF, still a NaN). Layout of one inbox, one parity: [source / owner
# rank][seam task][row][the task's 48 columns] (a (rank, task) block of G rows is
# contiguous: a wave's pushes coalesce). Two parities (epoch & 1).
TRANSPORTS = (
    "tagged",
    "lamport",
)  # lamport: the RS untagged; the AG untagged too when GlueBuild.lpag
LP_SEG = 48 * 2  # bytes of a seam task's 48 columns of one row
LP_VEC = LP_SEG // 16  # 6 16-byte vectors
LP_ROWS = THREADS // LP_VEC  # 85 rows a pass (vector layout: lane = (row, vector))
LP_PAR = TP * SEAM_TASKS * GMAX * LP_SEG  # bytes of one inbox parity
LP_RS = ERROR_BYTES + 2 * HALF_BYTES  # RS inbox [2][src][task][row][48]
LP_AG = LP_RS + 2 * LP_PAR  # AG inbox [2][owner][task][row][48]
LP_END = LP_AG + 2 * LP_PAR
SENT_SIGN = -0x7FFF8000  # 0x80008000


def peer_bytes() -> int:
    return LP_END


def lamport_span() -> tuple[int, int]:
    """(byte offset, bytes) of the Lamport inboxes in the peer buffer: 0xFF at setup and
    at every reset."""
    return LP_RS, LP_END - LP_RS


def arm_peer(peer_bytes_tensor) -> None:
    """Host side, no launch in flight: every Lamport inbox half-word := the sentinel
    (the tagged regions untouched)."""
    lp.arm(peer_bytes_tensor, *lamport_span())


@dataclass(frozen=True)
class GlueBuild:
    rows: int
    sort: bool = True  # stage C: AITER moe_sorting outputs + zeroed moe_buf
    # s_memrealtime at every stage boundary of every CTA -> tl [BLOCKS, TL_POINTS] i64
    timeline: bool = False
    wait_ticks: int = WAIT_TICKS
    plain: bool | None = (
        # peer data pushes (RS, AG) as plain stores (wavefront-scope monotonic: no cache
        # bits)
        None
    )
    rsbatch: bool | None = (
        # rs reduce: every pass's poll in one batch (G > 32 rows: no second round trip)
        None
    )
    rowvec: bool | None = (
        # row stage: h + residual through LDS, 12 consecutive columns a thread -> 64-bit
        # res_out /
        None
    )
    # normed stores (not 2-byte ones); the MXFP8 pass reads a padded LDS row (no 32-way
    # bank conflicts)
    rbf16: bool | None = (
        # router MFMAs as bf16 x 3: W = hi + mid + lo bf16 exactly, 9 16x16x16 bf16
        # MFMAs a tile and
        None
    )
    # wave instead of 12 16x16x4 f32 (products exact; only the f32 accumulation order
    # changes)
    prepoll: bool | None = (
        # bulk polls (rs reduce, row) start with one lane a wave spinning on one pair:
        # far less
        None
    )
    # uncached poll traffic while the data is in flight
    mixdst: bool | None = (
        # every push instruction's four 16-lane groups go to four different ranks
        # (rotated by
        None
    )
    # lane group), not all lanes to one rank: the three XGMI links fill together instead
    # of one after another
    fastsort: bool | None = (
        # sort: one-wave scan (ds_bpermute Hillis-Steele) instead of the serial 2-level
        None
    )
    # scan; padding and expert ids by (expert, slot) threads instead of a serial loop an
    # expert
    agdefer: bool | None = (
        # rs reduce (rsbatch): own AG slot at once, the remote AG pushes only after
        # every seam
        None
    )
    # CTA of this rank has reduced (a device counter): the all-gather does not compete
    # with the reduce-scatter
    xfirst: bool | None = (
        # row stage: normed rows stored, drained and XRDY published first; residual out
        # and the
        None
    )
    # MXFP8 copy after (no in-kernel reader waits for them)
    ownfirst: bool | None = (
        # AG push: this rank's own slot (the owner rows' norm input) before the peers'
        # slots
        None
    )
    rnorm: bool | None = (
        # orouter + rsbatch: the router normalizes its own K slice of the owner rows
        # from the
        None
    )
    # owner's reduced rows (own AG slot) + residual and rstd from per-task SSQ partials
    # the seam CTAs publish: no wait for the norm CTAs' drained stores + XRDY (the owner
    # rows' outputs are written after the route instead)
    afinish: bool | None = (
        # epoch moved by the last CTA through a device atomic counter (no CTA-0 mark
        # poll)
        None
    )
    orouter: bool | None = (
        # router + top-k on this rank's G owner rows only (normed first, while the
        # all-gather is in
        None
    )
    # flight); the top-k pairs go to every rank's peer heap, the sort CTA gathers all
    # rows' picks from there
    transport: str | None = (
        None  # TRANSPORTS: "tagged" (tagged triples) or "lamport" (the RS as
    )
    # a plain bf16 payload into sentinel inboxes; the AG too when lpag). "lamport_rs" =
    # lamport with lpag off.
    lpag: bool | None = (
        # lamport: the all-gather of h as plain payload too (else tagged triples);
        # policy by width
        None
    )
    lpprobe: bool | None = (
        # Lamport polls: one probe load a wave (lane 63's word, one request) spins
        # before the
        None
    )
    # full poll (prepoll's analogue)
    lpack: bool | None = (
        # Lamport: a seam CTA waits for its RS pushes' acks (vmcnt 0) and counts itself
        # on a device
        None
    )
    # counter; the reducers push the peers' AG only once all SEAM_TASKS CTAs of this
    # rank counted (the own AG store first, not gated): no AG store of this rank
    # competes with its RS stores on the links
    # debug: re-arm passes at the CTA end (2 = twice: the re-arm's cost sensitivity; 1 =
    # normal)
    lprearm: int = 1
    lpmix: int | None = (
        # Lamport pushes (mixdst): rows of a destination-rotation group (a lane's j-th
        # store goes
        None
    )
    # to peer (j + row // lpmix) % 3): 1 = every row (96-byte chunks a destination), 4 =
    # 384-byte (3 full 128-byte lines) chunks; 0 = no rotation (a store instruction to
    # one peer)


OPT_FLAGS = (
    "plain",
    "rsbatch",
    "rowvec",
    "rbf16",
    "prepoll",
    "afinish",
    "orouter",
    "mixdst",
    "fastsort",
    "agdefer",
    "xfirst",
    "ownfirst",
    "rnorm",
    "lpprobe",
    "lpack",
)  # transport, lpag, lpmix: resolved after these (they depend on the transport)
# lamport: 384-byte (3 x 128-byte line) chunks a destination; per-row rotation (96-byte
# chunks) lost
LP_MIX_DEFAULT = 4
# 1.8-2.9 us at 64-192 rows
# lamport: the AG untagged from this width (tagged triples below: 0.5-1 us faster at
# 24-40 rows)
LP_AG_MIN_ROWS = 44


POLICY_ON = (
    "plain",
    "rsbatch",
    "orouter",
    "mixdst",
    "afinish",
    "fastsort",
    "xfirst",
    "rbf16",
)
PREPOLL_ROWS = (
    48,
    64,
)  # prepoll measured -0.7 / -1.0 us at 48 / 64 rows, neutral or worse at 24-32 and 96+
# GlueBuild.transport None -> this. The runtime (decode.py) passes "lamport" explicitly;
# the module default stays "tagged", because a lamport launch on inboxes the host did
# not arm (arm_peer) would read zeros as landed data.
TRANSPORT_DEFAULT = "tagged"


def policy(rows: int) -> dict[str, bool]:
    """The options a width builds with when its GlueBuild leaves them None (measured,
    MI355X TP4). Options off: rowvec (+0..0.8 us), agdefer (+1.5..2.5: its RS finishes 7
    us earlier but the deferred all-gather then blocks the owner rows' local stores),
    rnorm (+1..2.5: its local hand-offs queue behind the all-gather's remote stores),
    ownfirst (unmeasured), prepoll outside PREPOLL_ROWS."""
    pol = {f: f in POLICY_ON for f in OPT_FLAGS}
    pol["prepoll"] = PREPOLL_ROWS[0] <= rows <= PREPOLL_ROWS[1]
    return pol


def resolve(key: GlueBuild) -> GlueBuild:
    """`key` with every None option set by `policy`."""
    from dataclasses import replace

    pol = policy(key.rows)
    of_free = key.ownfirst is None
    key = replace(key, **{f: pol[f] for f in OPT_FLAGS if getattr(key, f) is None})  # type: ignore[arg-type]
    if key.transport is None:
        key = replace(key, transport=TRANSPORT_DEFAULT)
    if key.transport == "lamport_rs":  # alias: the RS untagged, the AG tagged
        key = replace(key, transport="lamport", lpag=False)
    if key.lpag is None or key.transport == "tagged":
        key = replace(
            key, lpag=key.transport == "lamport" and key.rows >= LP_AG_MIN_ROWS
        )
    if key.lpmix is None:
        key = replace(key, lpmix=LP_MIX_DEFAULT if key.mixdst else 0)
    if (
        of_free and key.lpag
        # lamport AG: this rank's own inbox first (its owner rows start the routing
        # chain)
    ):
        key = replace(key, ownfirst=True)
    assert key.transport in TRANSPORTS, key.transport
    if key.transport != "tagged":
        bad = [f for f in ("rnorm", "agdefer") if getattr(key, f)]
        if key.lpag and key.rowvec:
            bad.append("rowvec")
        assert not bad, f"transport {key.transport} does not support {bad}"
    return key


# timeline stamps (timeline builds): s_memrealtime per CTA at each named point -> tl
# [BLOCKS, TL_POINTS] i64
TL_NAMES = (
    "entry",
    "zero",
    "push",
    "rs_land",
    "reduce",
    "ag_land",
    "norm",
    "r_rows",
    "r_mfma",
    "router",
    "rt_land",
    "route",
    "s_picks",
    "s_scan",
    "s_placed",
    "sort",
    "done",
    "epoch",
)
TL = {n: i for i, n in enumerate(TL_NAMES)}
TL.update(
    own_land=TL["ag_land"], own_norm=TL["norm"]
)  # orouter: the owner rows' stamps, reported apart
TL_NAMES = TL_NAMES[:5] + ("own_land", "own_norm") + TL_NAMES[5:]  # type: ignore[assignment]
TL_POINTS = 24


def tl_participants(
    s: int, orouter: bool = False, afinish: bool = False
) -> dict[str, list[int]]:
    """The CTAs that do the work a stamp closes (the harness reduces min / median / max
    over them)."""
    seam = list(range(SEAM0, SEAM0 + SEAM_TASKS))
    rows = orouter_row_ctas(s) if orouter else list(range(s))
    route = list(range(ROUTER0, ROUTER0 + s // TP)) if orouter else rows
    router = list(range(ROUTER0, ROUTER0 + ROUTER_TASKS))
    out = {
        "entry": list(range(BLOCKS)),
        "zero": router,
        "push": seam,
        "rs_land": seam,
        "reduce": seam,
        "ag_land": rows,
        "norm": rows,
        "r_rows": router,
        "r_mfma": router,
        "router": router,
        "rt_land": route,
        "route": route,
        "done": list(range(BLOCKS)),
        "epoch": [0],
    }
    for k in ("s_picks", "s_scan", "s_placed", "sort"):
        out[k] = [SORT_CTA]
    if afinish:
        out["epoch"] = list(range(BLOCKS))
    if orouter:
        own = list(range(ROUTER0, ROUTER0 + s // TP))
        out["own_land"] = out["own_norm"] = own
        out["ag_land"] = out["norm"] = [b for b in rows if b not in own]
    return out


SORT_CTA = BLOCKS - 1  # a router CTA, free after its router task
SORT_BLOCK = 16  # AITER A16W4 decode block_m
MAX_PICKS = MAX_ROWS * TOPK  # 768
ZERO_CTAS = (
    BLOCKS - ROUTER0
)  # 128: the router CTAs zero moe_buf right after their weight loads


def max_pad(s: int) -> int:
    """sorted_ids length, as aiter.fused_moe.moe_sorting allocates it."""
    n = s * TOPK + EXPERTS * SORT_BLOCK - TOPK
    return -(-n // SORT_BLOCK) * SORT_BLOCK


def scratch_layout(s: int) -> dict[str, tuple[int, int]]:
    tagged = pair_layout(
        [
            ("xrdy", s),
            ("logit", s * EXPERTS * ROUTER_PARTS),
            ("route", s * 2 * TOPK),
            (
                "ssq",
                (s // TP) * SEAM_TASKS,
            ),  # rnorm: [owner row][seam task] partial sums of squares
        ],
        align=256,
    )
    return dict(tagged)


def scratch_bytes(s: int) -> int:
    return max(o + n for o, n in scratch_layout(s).values())


# ---------------------------------------------------------------- helpers
def rs_pair(src, r, tr):
    return (src * GMAX + r) * TRIPLES + tr


def ag_pair(owner, r, tr):
    return RS_PAIRS + (owner * GMAX + r) * TRIPLES + tr


def triple_bits(buf, t, triple):
    """Row t's bf16 at columns 3 triple .. of a [*, HIDDEN] bf16 buffer, in bits
    0 .. 47."""
    v = lp.load_dwords(buf, t * (HIDDEN // 2) + (3 * triple) // 2, 2)
    w = (fx.Int64(fx.Uint32(fx.Int32(v[1]))) << 32) | fx.Int64(
        fx.Uint32(fx.Int32(v[0]))
    )
    return w >> fx.Int64(16 * (triple & 1))


def bf16_bits_f32(bits):
    return ((bits & 0xFFFF) << 16).bitcast(fx.Float32)


def bf16_bits(v):
    """The 16 bits of bf16(v) (round to nearest even), in an Int64."""
    return fx.Int64(fx.Uint32(bf16_round(v).bitcast(fx.Int32)) >> 16)


def triple_value(lo, hi, k):
    """Value k (0 .. 2) of a triple pair's (low word, high word)."""
    if const_expr(k == 2):
        return bf16_bits_f32(hi)
    return bf16_bits_f32(lo >> (16 * k))


@traced
def block_sum(c, v, slot):
    lane, wave, red = c["lane"], c["wave"], c["red"]
    v = butterfly(v, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(v, red + (slot * WAVES + wave))
    gpu.barrier()
    tot = fx.Float32(0.0)
    for w in range_constexpr(WAVES):
        tot = tot + fx.ptr_load(red + (slot * WAVES + w))
    return tot


# ---------------------------------------------------------------- seam: reduce-scatter
@traced
def stage_rs_push(c, task):
    """Triples 16 task ..: every row of this rank's partial to its owner's RS slot [this
    rank]; peers' rows first."""
    g_rows, rank, tid = c["G"], c["rank"], c["tid"]
    tr = task * TASK_TRIPLES + tid % TASK_TRIPLES
    loads = []  # every row's load in flight before the first store
    for d in range_constexpr(1, TP + 1):
        g = (rank + d) % TP  # the three peers first, this rank last
        base = c["peer"][0].value
        for p in range_constexpr(1, TP):
            base = sel(g == p, c["peer"][p].value, base)
        for it in range_constexpr(-(-g_rows // ROW_LANES)):
            r = it * ROW_LANES + tid // TASK_TRIPLES
            rr = fx.min(r, g_rows - 1)
            loads.append((base, r, triple_bits(c["args"]["part"], g * g_rows + rr, tr)))
    if const_expr(c["mixdst"]):
        # store j of a lane group q goes to destination (q + j) % 4 of the four: each
        # instruction feeds every link
        q = c["lane"] // 16
        passes = -(-g_rows // ROW_LANES)
        for it in range_constexpr(passes):
            r = it * ROW_LANES + tid // TASK_TRIPLES
            if r < g_rows:
                for j in range_constexpr(TP):
                    dsel = (q + j) % TP
                    base, _, bits = loads[it]
                    for k in range_constexpr(1, TP):
                        bk, _, vk = loads[k * passes + it]
                        base = sel(dsel == k, bk, base)
                        bits = sel(dsel == k, vk, bits)
                    push(c, Addr("peer", Space.PEER, base), rs_pair(rank, r, tr), bits)
    else:
        for base, r, bits in loads:
            if r < g_rows:
                push(c, Addr("peer", Space.PEER, base), rs_pair(rank, r, tr), bits)


@traced
def prepoll(c, addr, idx, site, triples=True):
    """One lane a wave spins on one pair before the CTA's full poll (a hint only: the
    full poll checks every tag)."""
    if const_expr(c["prepoll"]):  # noqa: SIM102 (compile-time branch)
        if c["lane"] == 0:
            c["mb"].poll([(addr, idx, 1)], site=site, triples=triples)


# agdefer: the epoch block's count of reduced seam CTAs (SEAM_TASKS a launch, monotonic)
RS_DONE_WORD = 2


@traced
def rs_done_wait(c):
    """Every seam CTA of this launch has reduced: one add each, then wait for the
    launch's multiple of SEAM_TASKS."""
    epoch = c["epoch"]
    cta_sync()
    if c["tid"] == 0:
        ptr = fx.inttoptr(
            fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
            epoch + fx.Int64(4 * RS_DONE_WORD),
        )
        old = fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent")
        target = (old // SEAM_TASKS + 1) * SEAM_TASKS
        t0 = memrealtime()
        n = load32(epoch, RS_DONE_WORD, "agent")
        late = fx.Int64(0)
        while (n < target) & (late < c["mb"].ticks):
            rocdl.s_sleep(1)
            n = load32(epoch, RS_DONE_WORD, "agent")
            late = memrealtime() - t0
        if n < target:
            c["mb"].fail(SITE_RS)
    cta_sync()


def ag_push_peers(c, i, bits):
    """Pair i of the three peers' AG regions := bits (mixdst: rotated by lane group)."""
    q = c["lane"] // 16 if c["mixdst"] else 0
    for j in range_constexpr(TP - 1):
        psel = (c["rank"] + 1 + (q + j) % (TP - 1)) % TP
        base = c["peer"][0].value
        for k in range_constexpr(1, TP):
            base = sel(psel == k, c["peer"][k].value, base)
        push(c, Addr("peer", Space.PEER, base), i, bits)


def ag_push_all(c, i, bits):
    """Pair i of every rank's AG region := bits (mixdst: lane group q's j-th store to
    rank (rank + 1 + q + j) % 4; ownfirst: this rank's own slot first, then the three
    peers rotated by lane group)."""
    if const_expr(c["ownfirst"]):
        push(c, c["own"], i, bits)
        ag_push_peers(c, i, bits)
    elif const_expr(c["mixdst"]):
        q = c["lane"] // 16
        for j in range_constexpr(TP):
            psel = (c["rank"] + 1 + q + j) % TP
            base = c["peer"][0].value
            for k in range_constexpr(1, TP):
                base = sel(psel == k, c["peer"][k].value, base)
            push(c, Addr("peer", Space.PEER, base), i, bits)
    else:
        for p in range_constexpr(TP):
            push(c, c["peer"][p], i, bits)


def push(c, a, i, bits):
    if const_expr(c["plain"]):
        c["mb"].put_triple(a, i, bits, plain=True)
    else:
        c["mb"].put_triple(a, i, bits)


def _rs_sum(c, got, rb):
    bits = fx.Int64(0)
    for k in range_constexpr(3):
        acc = triple_value(got[0][0][0], got[0][0][1], k)
        for src in range_constexpr(1, TP):
            acc = acc + triple_value(got[src][0][0], got[src][0][1], k)
        s_k = bf16_round(acc) + bf16_bits_f32(
            fx.Int32(rb >> (16 * k))
        )  # the stock add of bf16 h
        bits = bits | (c["sharded"].select(bf16_bits(s_k), bf16_bits(acc)) << (16 * k))
    return bits


@traced
def stage_rs_reduce_batched(c, task):
    """`stage_rs_reduce` with every pass's poll in one batch (rows past G poll row G - 1
    again)."""
    g_rows, rank, tid = c["G"], c["rank"], c["tid"]
    tr = task * TASK_TRIPLES + tid % TASK_TRIPLES
    passes = -(-g_rows // ROW_LANES)
    rs = [
        fx.min(it * ROW_LANES + tid // TASK_TRIPLES, g_rows - 1) for it in range(passes)
    ]
    rbs = [triple_bits(c["args"]["res_in"], rank * g_rows + r, tr) for r in rs]
    prepoll(
        c, c["own"], rs_pair((rank + 1) % TP, rs[0], tr), SITE_RS
    )  # (rank + 1) pushes to this owner last
    got = c["mb"].poll(
        [(c["own"], rs_pair(src, r, tr), 1) for r in rs for src in range(TP)],
        site=SITE_RS,
        triples=True,
    )
    stamp(c, "rs_land")
    if const_expr(c["rnorm"]):
        hs = [_rs_sum(c, got[it * TP : (it + 1) * TP], rbs[it]) for it in range(passes)]
        for it in range_constexpr(
            passes
        ):  # this rank's AG slot first (the owner rows' input, local)
            r = it * ROW_LANES + tid // TASK_TRIPLES
            if r < g_rows:
                push(c, c["own"], ag_pair(rank, r, tr), hs[it])
        for it in range_constexpr(
            passes
        ):  # the row's partial sum of squares over this task's 48 columns
            r = it * ROW_LANES + tid // TASK_TRIPLES
            sq = fx.Float32(0.0)
            for k in range_constexpr(3):
                hv = bf16_bits_f32(fx.Int32(hs[it] >> (16 * k)))
                v = c["sharded"].select(
                    hv, hv + bf16_bits_f32(fx.Int32(rbs[it] >> (16 * k)))
                )
                sq = sq + v * v
            sq = butterfly(sq, (8, 4, 2, 1))
            if (r < g_rows) & (tid % TASK_TRIPLES == 0):
                c["mb"].put(c["ssq"], r * SEAM_TASKS + task, sq)
        for it in range_constexpr(passes):
            r = it * ROW_LANES + tid // TASK_TRIPLES
            if r < g_rows:
                ag_push_peers(c, ag_pair(rank, r, tr), hs[it])
    elif const_expr(c["agdefer"]):
        hs = []
        for it in range_constexpr(passes):
            r = it * ROW_LANES + tid // TASK_TRIPLES
            hs.append(_rs_sum(c, got[it * TP : (it + 1) * TP], rbs[it]))
            if r < g_rows:
                push(
                    c, c["own"], ag_pair(rank, r, tr), hs[it]
                )  # own rows: this rank's AG slot now
        rs_done_wait(c)
        for it in range_constexpr(passes):
            r = it * ROW_LANES + tid // TASK_TRIPLES
            if r < g_rows:
                ag_push_peers(c, ag_pair(rank, r, tr), hs[it])
    else:
        for it in range_constexpr(passes):
            r = it * ROW_LANES + tid // TASK_TRIPLES
            if r < g_rows:
                bits = _rs_sum(c, got[it * TP : (it + 1) * TP], rbs[it])
                ag_push_all(c, ag_pair(rank, r, tr), bits)


@traced
def stage_rs_reduce(c, task):
    """Triples 16 task .. of this rank's G rows: the 4 ranks' triples in rank order
    (f32) -> bf16 h -> every rank's AG slot [this rank]."""
    g_rows, rank, tid = c["G"], c["rank"], c["tid"]
    tr = task * TASK_TRIPLES + tid % TASK_TRIPLES
    for it in range_constexpr(-(-g_rows // ROW_LANES)):
        r = it * ROW_LANES + tid // TASK_TRIPLES
        if r < g_rows:
            rb = triple_bits(
                c["args"]["res_in"], rank * g_rows + r, tr
            )  # mode 2: this rank's residual rows
            prepoll(c, c["own"], rs_pair((rank + 1) % TP, r, tr), SITE_RS)
            got = c["mb"].poll(
                [(c["own"], rs_pair(src, r, tr), 1) for src in range(TP)],
                site=SITE_RS,
                triples=True,
            )
            if const_expr(it == 0):
                stamp(c, "rs_land")
            bits = fx.Int64(0)
            for k in range_constexpr(3):
                acc = triple_value(got[0][0][0], got[0][0][1], k)
                for src in range_constexpr(1, TP):
                    acc = acc + triple_value(got[src][0][0], got[src][0][1], k)
                s_k = bf16_round(acc) + bf16_bits_f32(
                    fx.Int32(rb >> (16 * k))
                )  # the stock add of bf16 h
                bits = bits | (
                    c["sharded"].select(bf16_bits(s_k), bf16_bits(acc)) << (16 * k)
                )
            ag_push_all(c, ag_pair(rank, r, tr), bits)


# A payload is its own flag: the reader re-loads until none of the half-words it uses is
# the sentinel. Each half-word is checked by exactly one lane of one CTA on its rank and
# re-armed only after that CTA's barrier (a slot re-armed by a fast reader before a slow
# one saw it would lose the hand-off). Parity = epoch & 1: a slot of parity p is written
# again two collective launches later, which a peer reaches only after this rank
# finished this launch (it needs this rank's next pushes), i.e. after the re-arm stores
# completed at the kernel boundary. The consumers re-arm at their CTA's end, off the
# critical path (re-arming twice measured no change). Loads, stores and sentinel helpers
# of the common Lamport inbox (specialized_decode/mono/lamport.py).
lp_ld32, lp_ld64, lp_st16 = lp.ld32, lp.ld64, lp.st16
lp_sent, lp_canon, lp_triple_off = lp.sent_mask, lp.canon, lp.triple_off


def _ld16(base, dw):
    """Four dwords at dword `dw` of a (cached) buffer, as Int32 words."""
    return lp.words(lp.load16(base, dw))


def lp_seg(c, base, region, who, task, r):
    """Byte address of segment (this launch's parity, rank `who`, seam task, row r) of
    inbox `region` in the peer buffer at `base`."""
    blk = (c["lpar"] * TP + fx.Int64(who)) * SEAM_TASKS + fx.Int64(task)
    return base + fx.Int64(region) + (blk * GMAX + fx.Int64(r)) * LP_SEG


def lp_peer(c, p):
    """Rank p's peer buffer base (p traced)."""
    b = c["raw"][0]
    for q in range_constexpr(1, TP):
        b = sel(p == q, c["raw"][q], b)
    return b


def _err(c):
    """This launch's error word reader (the glue's peer buffer starts with the error
    record)."""
    return lambda: error_word(c["sym"])


def lp_spin(c, load_all, pending, site):
    return lp.spin(load_all, pending, _err(c), c["mb"].ticks, c["mb"].fail, site)


def lp_probe(c, addr, wave_live):
    lp.probe(addr, wave_live, _err(c), c["mb"].ticks)


def lp_poll_triples(c, addrs, tl, lives, site):
    return lp.poll_triples(addrs, tl, lives, _err(c), c["mb"].ticks, c["mb"].fail, site)


def lp_poll_vecs(c, addrs, live, site):
    return lp.poll_vecs(addrs, live, _err(c), c["mb"].ticks, c["mb"].fail, site)


assert GMAX <= LP_ROWS


@traced
def stage_rs_push_lp(c, task):
    """This rank's partial, owner g's rows, the task's 48 columns -> rank g's RS inbox
    `[this rank][task]` as plain bf16 (lane = (row, 16-byte vector); every load before
    the first store); the 3 peers only: the owner reads its own partial from `part`.
    mixdst: the lane's j-th store goes to peer (j + row) % 3 (after this rank), so one
    store instruction feeds all three links."""
    g_rows, rank, tid, a = c["G"], c["rank"], c["tid"], c["args"]
    r = tid // LP_VEC
    q = tid % LP_VEC
    rr = fx.min(r, g_rows - 1)
    col = task * 48 + 8 * q
    vals = []
    for d in range_constexpr(1, TP):
        g = (rank + d) % TP
        vals.append(_ld16(a["part"], ((g * g_rows + rr) * HIDDEN + col) // 2))
    if r < g_rows:
        for j in range_constexpr(TP - 1):
            k = (j + r // c["lpmix"]) % (TP - 1) if c["lpmix"] else j
            ws = list(vals[0])
            for kk in range_constexpr(1, TP - 1):
                ws = [sel(k == kk, vals[kk][e], ws[e]) for e in range(4)]
            p = (rank + 1 + k) % TP
            lp_st16(
                lp_seg(c, lp_peer(c, p), LP_RS, rank, task, r)
                + fx.Int64(16) * fx.Int64(q),
                lp_canon(ws),
            )
    if const_expr(c["lpack"]):
        lp_count_acked(c)


@traced
def lp_count_acked(c):
    """lpack: this CTA's pushes acked (vmcnt 0 in every wave), then one add to the epoch
    block's RS_DONE_WORD; thread 0 keeps the launch's target (the next multiple of
    SEAM_TASKS) in LDS for lp_wait_acked."""
    cta_sync(drain=True)
    if c["tid"] == 0:
        ptr = fx.inttoptr(
            fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
            c["epoch"] + fx.Int64(4 * RS_DONE_WORD),
        )
        old = fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent")
        fx.ptr_store(
            ((old // SEAM_TASKS + 1) * SEAM_TASKS).bitcast(fx.Float32),
            c["red"] + LP_ACK_LDS,
        )


@traced
def lp_wait_acked(c):
    """lpack: until every seam CTA of this rank counted its acked RS pushes (thread 0
    polls, bounded; then a barrier)."""
    if c["tid"] == 0:
        target = fx.ptr_load(c["red"] + LP_ACK_LDS).bitcast(fx.Int32)
        t0 = memrealtime()
        n = load32(c["epoch"], RS_DONE_WORD, "agent")
        late = fx.Int64(0)
        while (n - target < 0) & (late < c["mb"].ticks):
            rocdl.s_sleep(1)
            n = load32(c["epoch"], RS_DONE_WORD, "agent")
            late = memrealtime() - t0
        if n - target < 0:
            c["mb"].fail(SITE_RS)
    cta_sync()


# a word of LDS ``red`` past the router's MFMA sums (seam CTAs never run the router)
LP_ACK_LDS = 8000


def _by_rank(c, own, got):
    """[own] + got[d - 1] (peer at position d after this rank) -> the same words per
    absolute source rank."""
    rank = c["rank"]
    out = []
    for s in range(TP):
        ws = list(own)
        for d in range(1, TP):
            m = ((rank + d) % TP) == s
            ws = [sel(m, got[d - 1][e], ws[e]) for e in range(len(ws))]
        out.append(ws)
    return out


@traced
def stage_rs_reduce_lp3(c, task):
    """lamport_rs: this rank's G rows of the task's columns in the tagged reducer's
    layout (lane = (row, triple); every pass's poll in one batch): each peer's segment
    read at the lane's triple, plus this rank's own partial (cached), summed in rank
    order (f32) -> _rs_sum -> tagged triples to every rank's AG slot, as the tagged
    build."""
    g_rows, rank, tid, a = c["G"], c["rank"], c["tid"], c["args"]
    tl = tid % TASK_TRIPLES
    tr = task * TASK_TRIPLES + tl
    passes = -(-g_rows // ROW_LANES)
    rows = [it * ROW_LANES + tid // TASK_TRIPLES for it in range(passes)]
    rs = [fx.min(r, g_rows - 1) for r in rows]
    owns = [triple_bits(a["part"], rank * g_rows + rr, tr) for rr in rs]
    rbs = [triple_bits(a["res_in"], rank * g_rows + rr, tr) for rr in rs]
    addrs, lives = [], []
    for it in range_constexpr(passes):
        for d in range_constexpr(1, TP):
            addrs.append(lp_seg(c, c["sym"], LP_RS, (rank + d) % TP, task, rs[it]))
            lives.append(rows[it] < g_rows)
    if const_expr(c["lpprobe"]):
        lp_probe(
            c,
            addrs[-1] + fx.Int64(lp_triple_off(tl)),
            (c["wave"] * (64 // TASK_TRIPLES)) < g_rows,
        )
    got = lp_poll_triples(c, addrs, tl, lives, SITE_RS)
    stamp(c, "rs_land")
    for it in range_constexpr(passes):
        r = rows[it]
        if r < g_rows:
            own = [fx.Int32(owns[it]), fx.Int32(owns[it] >> 32)]
            srcs = _by_rank(
                c, own, [list(got[it * (TP - 1) + d]) for d in range(TP - 1)]
            )
            bits = _rs_sum(c, [[(w[0], w[1])] for w in srcs], rbs[it])
            ag_push_all(c, ag_pair(rank, r, tr), bits)


@traced
def stage_rs_reduce_lpv(c, task):
    """lamport: this rank's G rows of the task's columns, lane = (row, 16-byte vector):
    the 3 peers' slots + this rank's own partial (cached), summed in rank order (f32) ->
    bf16 h (mode 2: bf16(h + this rank's residual rows)) -> every rank's AG inbox
    `[this rank][task]` as plain bf16 (ownfirst: this rank's own inbox first; mixdst:
    the peers rotated by row)."""
    g_rows, rank, tid, a = c["G"], c["rank"], c["tid"], c["args"]
    r = tid // LP_VEC
    q = tid % LP_VEC
    live = r < g_rows
    rr = fx.min(r, g_rows - 1)
    dw = ((rank * g_rows + rr) * HIDDEN + task * 48 + 8 * q) // 2
    own = _ld16(a["part"], dw)
    rb = _ld16(a["res_in"], dw)
    q16 = fx.Int64(16) * fx.Int64(q)
    addrs = [
        lp_seg(c, c["sym"], LP_RS, (rank + d) % TP, task, rr) + q16
        for d in range(1, TP)
    ]
    if const_expr(c["lpprobe"]):
        lp_probe(c, addrs[-1], ((c["wave"] * 64) // LP_VEC) < g_rows)
    got = lp_poll_vecs(c, addrs, live, SITE_RS)
    stamp(c, "rs_land")
    srcs = _by_rank(c, own, got)
    hw = []
    for e in range_constexpr(4):
        lo, hi = bf2_f32(srcs[0][e])
        for s in range_constexpr(1, TP):
            l2, h2 = bf2_f32(srcs[s][e])
            lo = lo + l2
            hi = hi + h2
        rlo, rhi = bf2_f32(rb[e])
        blo = c["sharded"].select(
            bf16_bits(bf16_round(lo) + rlo), bf16_bits(lo)
        )  # the stock add of bf16 h (mode 2)
        bhi = c["sharded"].select(bf16_bits(bf16_round(hi) + rhi), bf16_bits(hi))
        hw.append(fx.Int32(blo | (bhi << 16)))
    hv = lp_canon(hw)
    if const_expr(c["lpack"]):  # own AG first, then the gate, then the peers
        if live:
            lp_st16(lp_seg(c, c["sym"], LP_AG, rank, task, r) + q16, hv)
        lp_wait_acked(c)
    if live:
        if const_expr(c["ownfirst"] and not c["lpack"]):
            lp_st16(lp_seg(c, c["sym"], LP_AG, rank, task, r) + q16, hv)
        for j in range_constexpr(TP - 1):
            k = (j + r // c["lpmix"]) % (TP - 1) if c["lpmix"] else j
            p = (rank + 1 + k) % TP
            lp_st16(lp_seg(c, lp_peer(c, p), LP_AG, rank, task, r) + q16, hv)
        if const_expr(not c["ownfirst"] and not c["lpack"]):
            lp_st16(lp_seg(c, c["sym"], LP_AG, rank, task, r) + q16, hv)


def lp_row_ag(c, t, trs):
    """lamport: row t's 4 triples a lane from this rank's AG inbox (owner t // G) ->
    [[(lo, hi)]] as the tagged poll."""
    g_rows, tid = c["G"], c["tid"]
    owner = t // g_rows
    r = t % g_rows
    tl = tid % TASK_TRIPLES
    addrs = [lp_seg(c, c["sym"], LP_AG, owner, tr // TASK_TRIPLES, r) for tr in trs]
    if const_expr(c["lpprobe"]):
        lp_probe(c, addrs[-1] + fx.Int64(lp_triple_off(tl)), c["tid"] >= 0)
    return [[p] for p in lp_poll_triples(c, addrs, tl, [True] * len(trs), SITE_AG)]


SENT_WORDS = 4


@traced
def lp_rearm_rs(c, task):
    """The RS slots seam task `task` consumed on this rank: 3 peer segments x G rows
    (contiguous), sentinel stores."""
    g_rows, rank, tid = c["G"], c["rank"], c["tid"]
    n = LP_VEC * g_rows
    sent = [fx.Int32(-1)] * SENT_WORDS
    for d in range_constexpr(1, TP):
        base = lp_seg(c, c["sym"], LP_RS, (rank + d) % TP, task, 0)
        for i in range_constexpr(-(-n // THREADS)):
            v = tid + THREADS * i
            if v < n:
                lp_st16(base + fx.Int64(16) * fx.Int64(v), sent)


@traced
def lp_rearm_row(c, t):
    """AG inbox row t (owner t // G, row t % G) on this rank: its 128 task segments, 768
    sentinel stores."""
    g_rows, tid = c["G"], c["tid"]
    owner = t // g_rows
    r = t % g_rows
    sent = [fx.Int32(-1)] * SENT_WORDS
    nv = SEAM_TASKS * LP_VEC
    for i in range_constexpr(-(-nv // THREADS)):
        v = tid + THREADS * i
        if v < nv:
            lp_st16(
                lp_seg(c, c["sym"], LP_AG, owner, v // LP_VEC, r)
                + fx.Int64(16) * fx.Int64(v % LP_VEC),
                sent,
            )


def lp_row_of(c, bid):
    """(has a row, the row t) of the row stage CTA `bid` ran (run_glue's placement)."""
    s, g_rows = c["S"], c["G"]
    if const_expr(not c["orouter"]):
        return bid < s, bid
    npeer = s - g_rows
    own = (bid >= ROUTER0) & (bid < ROUTER0 + g_rows)
    peer = bid < min(npeer, ROUTER0)
    t = sel(
        own, c["rank"] * g_rows + bid - ROUTER0, peer_row(c, fx.min(bid, npeer - 1))
    )
    has = own | peer
    if const_expr(npeer > ROUTER0):
        peer2 = (bid >= ROUTER0 + g_rows) & (bid < ROUTER0 + g_rows + npeer - ROUTER0)
        t = sel(
            peer2, peer_row(c, fx.max(fx.min(bid - g_rows, npeer - 1), fx.Int32(0))), t
        )
        has = has | peer2
    return has, t


@traced
def lp_rearm_end(c, bid):
    """Consumers re-arm at their CTA's end: after a CTA barrier every lane of the CTA
    has consumed its slots."""
    gpu.barrier()
    if bid < SEAM_TASKS:
        lp_rearm_rs(c, bid)
    if const_expr(c["lp_ag"]):
        has, t = lp_row_of(c, bid)
        if has:
            lp_rearm_row(c, t)


# ---------------------------------------------------------------- all-gather in +
# residual + norm, a CTA a row
ROW_TRIPLES = TRIPLES // THREADS  # 4 triples (12 columns) a thread
GROUPS = HIDDEN // 32  # MXFP8 groups a row
XROW_OFF = 1024  # the normed row's f32 copy in LDS ``red`` (past block_sum's words)
FP8_MAX = 448.0
F32_TINY = 1.1754943508222875e-38


@traced
def stage_row(c, t):
    """Row t: its owner's gathered h + residual (f32) -> residual out (bf16); Gemma
    RMSNorm -> normed (bf16), XRDY."""
    g_rows, tid, a = c["G"], c["tid"], c["args"]
    owner = t // g_rows
    trs = [tid + j * THREADS for j in range(ROW_TRIPLES)]
    res = [triple_bits(a["res_in"], t, tr) for tr in trs]  # loads before the poll
    nw = [triple_bits(a["norm_w"], 0, tr) for tr in trs]
    if const_expr(c["lp_ag"]):
        got = lp_row_ag(c, t, trs)
    else:
        prepoll(c, c["own"], ag_pair(owner, t % g_rows, trs[0]), SITE_AG)
        got = c["mb"].poll(
            [(c["own"], ag_pair(owner, t % g_rows, tr), 1) for tr in trs],
            site=SITE_AG,
            triples=True,
        )
    stamp(c, "ag_land")
    vals = []
    sq = fx.Float32(0.0)
    for j in range_constexpr(ROW_TRIPLES):
        for k in range_constexpr(3):
            hv = triple_value(got[j][0][0], got[j][0][1], k)
            # mode 2: the gathered value is already bf16(h + residual)
            v = c["sharded"].select(
                hv, hv + bf16_bits_f32(fx.Int32(res[j] >> (16 * k)))
            )
            if const_expr(not c["xfirst"]):
                buf_store(v.to(fx.BFloat16), a["res_out"], t * HIDDEN + 3 * trs[j] + k)
            vals.append(v)
            sq = sq + v * v
    var = block_sum(c, sq, 0) / fx.Float32(HIDDEN)
    rstd = fx.Float32(1.0) / fx.Float32(fmath.sqrt(var + fx.Float32(EPS)))
    for j in range_constexpr(ROW_TRIPLES):
        for k in range_constexpr(3):
            w = bf16_bits_f32(fx.Int32(nw[j] >> (16 * k)))
            o = vals[3 * j + k] * rstd * (fx.Float32(1.0) + w)
            store16(a["xn"], t * HIDDEN + 3 * trs[j] + k, o.to(fx.BFloat16), "agent")
    if const_expr(c["xfirst"]):
        cta_sync(drain=True)
        if tid == 0:
            c["mb"].publish(c["xrdy"], t, fx.Int32(1))
        for j in range_constexpr(ROW_TRIPLES):
            for k in range_constexpr(3):
                buf_store(
                    vals[3 * j + k].to(fx.BFloat16),
                    a["res_out"],
                    t * HIDDEN + 3 * trs[j] + k,
                )
    # MXFP8 copy of the normed row (the shared expert's gate_up input), vLLM's
    # fused-norm math on bf16(normed): per 32 columns sb = clamp(ceil(log2(amax / 448))
    # + 127, 0, 254), q = y * 2^(127 - sb) as e4m3.
    xrow = c["red"] + XROW_OFF
    for j in range_constexpr(ROW_TRIPLES):
        for k in range_constexpr(3):
            w = bf16_bits_f32(fx.Int32(nw[j] >> (16 * k)))
            o = vals[3 * j + k] * rstd * (fx.Float32(1.0) + w)
            fx.ptr_store(bf16_round(o), xrow + 3 * trs[j] + k)
    gpu.barrier()
    if tid < GROUPS:
        ys = [fx.ptr_load(xrow + tid * 32 + i) for i in range(32)]
        amax = fx.Float32(F32_TINY)
        for y in ys:
            amax = fx.max(amax, fx.Float32(fmath.absf(y)))
        bits = (amax / fx.Float32(FP8_MAX)).bitcast(fx.Int32)
        # ceil(log2 r) of a normal r: its exponent, + 1 unless the mantissa is 0
        k_exp = (
            ((bits >> 23) & 0xFF)
            - 127
            + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
        )
        sb = fx.min(fx.max(k_exp + 127, fx.Int32(0)), fx.Int32(254))
        inv = ((254 - sb) << 23).bitcast(fx.Float32)
        words = [
            fp8_pack4(
                ys[4 * w] * inv,
                ys[4 * w + 1] * inv,
                ys[4 * w + 2] * inv,
                ys[4 * w + 3] * inv,
            )
            for w in range(8)
        ]
        for h in range_constexpr(2):
            buf_store(
                fx.Vector.from_elements(words[4 * h : 4 * h + 4], fx.Int32),
                a["xq"],
                (t * HIDDEN + tid * 32) // 4 + 4 * h,
            )
        fx.ptr_store(sb, c["scl"] + tid)
    gpu.barrier()
    if tid < GROUPS // 4:
        sw = [fx.ptr_load(c["scl"] + 4 * tid + i) for i in range(4)]
        buf_store(
            sw[0] | (sw[1] << 8) | (sw[2] << 16) | (sw[3] << 24),
            a["xsc"],
            t * (GROUPS // 4) + tid,
        )
    if const_expr(not c["xfirst"]):
        cta_sync(drain=True)
        if tid == 0:
            c["mb"].publish(c["xrdy"], t, fx.Int32(1))
    stamp(c, "norm")


def _pad(col):
    """LDS index of a row column with one pad word every 32 (a thread's 32-column MXFP8
    group: distinct banks)."""
    return col + (col >> 5)


@traced
def stage_row_vec(c, t):
    """`stage_row` with vector stores: the f32 sums go through LDS; each thread then
    owns 12 consecutive columns (24 bytes: res_out and normed as 64-bit stores). Same
    math and bits."""
    g_rows, tid, a = c["G"], c["tid"], c["args"]
    owner = t // g_rows
    trs = [tid + j * THREADS for j in range(ROW_TRIPLES)]
    res = [triple_bits(a["res_in"], t, tr) for tr in trs]  # loads before the poll
    c0 = tid * (3 * ROW_TRIPLES)
    nw = [
        fx.Vector(buf_load(a["norm_w"], c0 // 2 + 2 * i, vec_width=2, dtype=T.i32))
        for i in range(3)
    ]
    prepoll(c, c["own"], ag_pair(owner, t % g_rows, trs[0]), SITE_AG)
    got = c["mb"].poll(
        [(c["own"], ag_pair(owner, t % g_rows, tr), 1) for tr in trs],
        site=SITE_AG,
        triples=True,
    )
    stamp(c, "ag_land")
    xrow = c["red"] + XROW_OFF
    sq = fx.Float32(0.0)
    for j in range_constexpr(ROW_TRIPLES):
        for k in range_constexpr(3):
            hv = triple_value(got[j][0][0], got[j][0][1], k)
            v = c["sharded"].select(
                hv, hv + bf16_bits_f32(fx.Int32(res[j] >> (16 * k)))
            )
            fx.ptr_store(v, xrow + _pad(3 * trs[j] + k))
            sq = sq + v * v
    var = block_sum(c, sq, 0) / fx.Float32(
        HIDDEN
    )  # its barrier also publishes the LDS row
    rstd = fx.Float32(1.0) / fx.Float32(fmath.sqrt(var + fx.Float32(EPS)))
    vs = [fx.ptr_load(xrow + _pad(c0 + i)) for i in range(3 * ROW_TRIPLES)]
    ws = []
    for i in range_constexpr(3):
        for h in range_constexpr(2):
            lo, hi = bf2_f32(fx.Int32(nw[i][h]))
            ws += [lo, hi]
    os_ = [vs[i] * rstd * (fx.Float32(1.0) + ws[i]) for i in range(3 * ROW_TRIPLES)]
    rw = [bf16_pair(vs[2 * i], vs[2 * i + 1]).bitcast(fx.Int32) for i in range(6)]
    ow = [bf16_pair(os_[2 * i], os_[2 * i + 1]).bitcast(fx.Int32) for i in range(6)]
    for i in range_constexpr(3):
        buf_store(
            fx.Vector.from_elements(rw[2 * i : 2 * i + 2], fx.Int32),
            a["res_out"],
            (t * HIDDEN + c0) // 2 + 2 * i,
        )
        store64(
            a["xn"] + fx.Int64(t * HIDDEN + c0) * 2,
            i,
            (fx.Int64(fx.Uint32(ow[2 * i + 1])) << 32) | fx.Int64(fx.Uint32(ow[2 * i])),
            "agent",
        )
    for i in range_constexpr(3 * ROW_TRIPLES):
        fx.ptr_store(bf16_round(os_[i]), xrow + _pad(c0 + i))
    gpu.barrier()
    if tid < GROUPS:
        ys = [fx.ptr_load(xrow + tid * 33 + i) for i in range(32)]
        amax = fx.Float32(F32_TINY)
        for y in ys:
            amax = fx.max(amax, fx.Float32(fmath.absf(y)))
        bits = (amax / fx.Float32(FP8_MAX)).bitcast(fx.Int32)
        k_exp = (
            ((bits >> 23) & 0xFF)
            - 127
            + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
        )
        sb = fx.min(fx.max(k_exp + 127, fx.Int32(0)), fx.Int32(254))
        inv = ((254 - sb) << 23).bitcast(fx.Float32)
        words = [
            fp8_pack4(
                ys[4 * w] * inv,
                ys[4 * w + 1] * inv,
                ys[4 * w + 2] * inv,
                ys[4 * w + 3] * inv,
            )
            for w in range(8)
        ]
        for h in range_constexpr(2):
            buf_store(
                fx.Vector.from_elements(words[4 * h : 4 * h + 4], fx.Int32),
                a["xq"],
                (t * HIDDEN + tid * 32) // 4 + 4 * h,
            )
        fx.ptr_store(sb, c["scl"] + tid)
    gpu.barrier()
    if tid < GROUPS // 4:
        sw = [fx.ptr_load(c["scl"] + 4 * tid + i) for i in range(4)]
        buf_store(
            sw[0] | (sw[1] << 8) | (sw[2] << 16) | (sw[3] << 24),
            a["xsc"],
            t * (GROUPS // 4) + tid,
        )
    cta_sync(drain=True)
    if tid == 0:
        c["mb"].publish(c["xrdy"], t, fx.Int32(1))
    stamp(c, "norm")


def peer_row(c, q):
    """Peer row q (0 .. 3G - 1) in owner order rank + 1, rank + 2, rank + 3."""
    g_rows = c["G"]
    return ((c["rank"] + 1 + q // g_rows) % TP) * g_rows + q % g_rows


def orouter_row_ctas(s: int) -> list[int]:
    g = s // TP
    npeer = s - g
    return (
        list(range(ROUTER0, ROUTER0 + g))
        + list(range(min(npeer, ROUTER0)))
        + list(range(ROUTER0 + g, ROUTER0 + g + max(0, npeer - ROUTER0)))
    )


def row_stage(c, t):
    if const_expr(c["rowvec"]):
        stage_row_vec(c, t)
    else:
        stage_row(c, t)


# ---------------------------------------------------------------- router
def mfma_f32(a, b, acc):
    from flydsl.expr.rocdl import _blgp_attr

    return fx.Vector(
        rocdl.mfma_f32_16x16x4f32(
            T.vec(4, T.f32),
            as_ir_value(a),
            as_ir_value(b),
            as_ir_value(acc),
            0,
            0,
            _blgp_attr(0),
        ).result
    )


def mfma_bf16(a64, b64, acc):
    """16x16x16 bf16 MFMA on 64-bit packed operands (4 bf16 each, element 0 =
    lowest k)."""
    a = fx.Vector.from_elements([a64], dtype=fx.Int64).bitcast(fx.Int16)
    b = fx.Vector.from_elements([b64], dtype=fx.Int64).bitcast(fx.Int16)
    return fx.Vector(
        rocdl.mfma_f32_16x16x16bf16_1k(T.vec(4, T.f32), [a, b, acc, 0, 0, 0])
    )


def split3(w4):
    """A float4 -> three 64-bit words of 4 bf16 (hi, mid, lo) with w = hi + mid + lo
    exactly (truncations)."""
    outs: list[list] = [[], [], []]
    for e in range(4):
        r = w4[e]
        for lvl in range(3):
            b = r.bitcast(fx.Int32) & fx.Int32(-65536)
            outs[lvl].append(b)
            if lvl < 2:
                r = r - b.bitcast(fx.Float32)
    words = []
    for lvl in range(3):
        b = outs[lvl]
        lo = fx.Uint32((b[0] >> 16) & 0xFFFF) | fx.Uint32(b[1])
        hi = fx.Uint32((b[2] >> 16) & 0xFFFF) | fx.Uint32(b[3])
        words.append(fx.Int64(lo) | (fx.Int64(hi) << 32))
    return words


def router_weights(c, task):
    """The task's FP32 weights (a lane's float4 x 3 chunks), issued at kernel entry.
    rnorm: a lane group's 12 consecutive columns (4 triples: the own AG slot's
    granularity) instead of 3 x 4 columns 16 apart."""
    lane, wave, a = c["lane"], c["wave"], c["args"]
    tile, part = task // ROUTER_PARTS, task % ROUTER_PARTS
    e = tile * 16 + lane % 16
    chunks = ROUTER_K // WAVES // 16
    if const_expr(c["rnorm"]):
        k0 = part * ROUTER_K + wave * (ROUTER_K // WAVES) + (lane // 16) * 12
        return [
            fx.Vector(
                buf_load(
                    a["router_w"], e * HIDDEN + k0 + 4 * ch, vec_width=4, dtype=T.f32
                )
            )
            for ch in range(chunks)
        ]
    k0 = part * ROUTER_K + wave * (ROUTER_K // WAVES) + (lane // 16) * 4
    return [
        fx.Vector(
            buf_load(a["router_w"], e * HIDDEN + k0 + 16 * ch, vec_width=4, dtype=T.f32)
        )
        for ch in range(chunks)
    ]


@traced
def stage_router(c, task, wv):
    """Experts 16 tile .. over K part `part` for every row (16-row tiles): f32 MFMAs,
    waves summed in order -> `LOGIT[row][expert][part]`. One wait for every row's XRDY,
    then every tile's X loads in flight together."""
    s, tid, lane, wave, red = (
        c["RR"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
    )  # s: the router's rows
    row0 = c["row0"]  # its first row (orouter: this rank's owner rows)
    tile, part = task // ROUTER_PARTS, task % ROUTER_PARTS
    k0 = part * ROUTER_K + wave * (ROUTER_K // WAVES) + (lane // 16) * 4
    chunks = ROUTER_K // WAVES // 16
    ntiles = -(-s // 16)
    if tid < s:
        c["mb"].poll([(c["xrdy"], row0 + tid, 1)], site=SITE_ROUTER)
        c["mb"].acquire(Space.SCRATCH)
    cta_sync()
    stamp(c, "r_rows")
    xs = []
    for rt in range_constexpr(ntiles):
        n = min(16, s - rt * 16)
        tok = row0 + rt * 16 + fx.min(lane % 16, n - 1)
        xs.append(
            [
                load_dev(c["args"]["xn"], (tok * HIDDEN + k0 + 16 * ch) // 2, 2)
                for ch in range(chunks)
            ]
        )
    if const_expr(c["rbf16"]):
        w3 = [split3(wv[ch]) for ch in range(chunks)]
    for rt in range_constexpr(ntiles):
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for ch in range_constexpr(chunks):
            if const_expr(c["rbf16"]):
                x64 = fx.Int64(fx.Uint32(fx.Int32(xs[rt][ch][0]))) | (
                    fx.Int64(fx.Uint32(fx.Int32(xs[rt][ch][1]))) << 32
                )
                for lvl in range_constexpr(3):
                    acc = mfma_bf16(w3[ch][lvl], x64, acc)
            else:
                for h in range_constexpr(2):
                    lo, hi = bf2_f32(fx.Int32(xs[rt][ch][h]))
                    acc = mfma_f32(wv[ch][2 * h], lo, acc)
                    acc = mfma_f32(wv[ch][2 * h + 1], hi, acc)
        fx.ptr_store(acc, red + ((rt * WAVES + wave) * 64 + lane) * 4)
    gpu.barrier()
    stamp(c, "r_mfma")
    for rt in range_constexpr(ntiles):
        n = min(16, s - rt * 16)
        if tid < 16 * n:
            r = tid % 16
            t = tid // 16
            at = (16 * (r // 4) + t) * 4 + r % 4
            tot = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                tot = tot + fx.ptr_load(red + ((rt * WAVES + w) * 64 * 4 + at))
            c["mb"].put(
                c["logit"], (part * s + rt * 16 + t) * EXPERTS + tile * 16 + r, tot
            )


# rnorm: LDS ``red`` offsets of the SSQ partials [G][128] and the rows' rstd (past the
# MFMA sums)
SSQ_LDS = 8192
RSTD_LDS = SSQ_LDS + GMAX * SEAM_TASKS


@traced
def stage_router_rn(c, task, wv):
    """`stage_router` on the owner rows with its own normalization (rnorm): rstd from
    the seam CTAs' SSQ partials (a row's 128 in a fixed order), v = own AG slot (h) +
    residual for the lane group's 12 columns, x = bf16(v * rstd * (1 + w)) as the norm
    CTAs compute it (rstd's summation order differs: a rare 1-ulp x difference)."""
    s, tid, lane, wave, red, a = (
        c["RR"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
        c["args"],
    )
    rank, row0 = c["rank"], c["row0"]
    tile, part = task // ROUTER_PARTS, task % ROUTER_PARTS
    tr0 = (
        part * (ROUTER_K // 3) + wave * (ROUTER_K // WAVES // 3) + (lane // 16) * 4
    )  # the lane group's 4 triples
    ntiles = -(-s // 16)
    # loads before the polls: residual and norm weight triples of the lane's columns
    res, rls = [], []
    for rt in range_constexpr(ntiles):
        n = min(16, s - rt * 16)
        rl = rt * 16 + fx.min(lane % 16, n - 1)
        rls.append(rl)
        res.append([triple_bits(a["res_in"], row0 + rl, tr0 + j) for j in range(4)])
    nw = [triple_bits(a["norm_w"], 0, tr0 + j) for j in range(4)]
    # every owner row's 128 SSQ partials (rows past G poll row G - 1 again)
    its = -(-(s * SEAM_TASKS) // THREADS)
    specs = []
    for it in range_constexpr(its):
        idx = fx.min(it * THREADS + tid, s * SEAM_TASKS - 1)
        specs.append((c["ssq"], idx, 1))
    a0, i0, _ = specs[-1]
    prepoll(
        c, a0, i0, SITE_ROUTER, triples=False
    )  # prepoll builds: a lane a wave on one pair first
    got = c["mb"].poll(specs, batch=16, site=SITE_ROUTER)
    for it in range_constexpr(its):
        idx = it * THREADS + tid
        if idx < s * SEAM_TASKS:
            fx.ptr_store(got[it][0].bitcast(fx.Float32), red + SSQ_LDS + idx)
    gpu.barrier()
    for rr in range_constexpr(
        -(-s // WAVES)
    ):  # wave w: rows w, w + 8, ...: 2 partials a lane, butterfly
        row = rr * WAVES + wave
        if row < s:
            p2 = fx.ptr_load(red + SSQ_LDS + row * SEAM_TASKS + 2 * lane) + fx.ptr_load(
                red + SSQ_LDS + row * SEAM_TASKS + 2 * lane + 1
            )
            tot = butterfly(p2, (1, 2, 4, 8, 16, 32))
            if lane == 0:
                var = tot / fx.Float32(HIDDEN)
                fx.ptr_store(
                    fx.Float32(1.0) / fx.Float32(fmath.sqrt(var + fx.Float32(EPS))),
                    red + RSTD_LDS + row,
                )
    hs = c["mb"].poll(
        [
            (c["own"], ag_pair(rank, rls[rt], tr0 + j), 1)
            for rt in range(ntiles)
            for j in range(4)
        ],
        batch=16,
        site=SITE_ROUTER,
        triples=True,
    )
    gpu.barrier()
    stamp(c, "r_rows")
    for rt in range_constexpr(ntiles):
        rstd = fx.ptr_load(red + RSTD_LDS + rls[rt])
        xs = []
        for j in range_constexpr(4):
            for k in range_constexpr(3):
                hv = triple_value(hs[rt * 4 + j][0][0], hs[rt * 4 + j][0][1], k)
                v = c["sharded"].select(
                    hv, hv + bf16_bits_f32(fx.Int32(res[rt][j] >> (16 * k)))
                )
                w = bf16_bits_f32(fx.Int32(nw[j] >> (16 * k)))
                xs.append(v * rstd * (fx.Float32(1.0) + w))
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        if const_expr(c["rbf16"]):
            for ch in range_constexpr(3):
                lo = bf16_pair(xs[4 * ch], xs[4 * ch + 1]).bitcast(fx.Int32)
                hi = bf16_pair(xs[4 * ch + 2], xs[4 * ch + 3]).bitcast(fx.Int32)
                x64 = fx.Int64(fx.Uint32(lo)) | (fx.Int64(fx.Uint32(hi)) << 32)
                for w3 in split3(wv[ch]):
                    acc = mfma_bf16(w3, x64, acc)
        else:
            for m in range_constexpr(12):
                acc = mfma_f32(wv[m // 4][m % 4], bf16_round(xs[m]), acc)
        fx.ptr_store(acc, red + ((rt * WAVES + wave) * 64 + lane) * 4)
    gpu.barrier()
    stamp(c, "r_mfma")
    for rt in range_constexpr(ntiles):
        n = min(16, s - rt * 16)
        if tid < 16 * n:
            r = tid % 16
            t = tid // 16
            at = (16 * (r // 4) + t) * 4 + r % 4
            tot = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                tot = tot + fx.ptr_load(red + ((rt * WAVES + w) * 64 * 4 + at))
            c["mb"].put(
                c["logit"], (part * s + rt * 16 + t) * EXPERTS + tile * 16 + r, tot
            )


# ---------------------------------------------------------------- route
def _sigmoid(x):
    return hw_rcp(fx.Float32(1.0) + hw_exp2(x * fx.Float32(-LOG2E)))


@traced
def stage_route(c, t):
    """Row t: top-4 of sigmoid + bias (ties to the lower expert); weights = selected
    sigmoids x (2.0 / their sum)."""
    tid, lane, wave, red, a = c["tid"], c["lane"], c["wave"], c["red"], c["args"]
    quarter = ROUTER_PARTS // 4
    e = tid % EXPERTS
    q = tid // EXPERTS
    # the bias before the poll (a global load round trip off the critical path)
    bias = [
        fx.Float32(buf_load(a["bias"], lane + 64 * i, vec_width=1, dtype=T.f32))
        for i in range(2)
    ]
    got = c["mb"].poll(
        [
            (c["logit"], ((q * quarter + p) * c["RR"] + t) * EXPERTS + e, 1)
            for p in range(quarter)
        ],
        site=SITE_ROUTE,
    )
    c["mb"].acquire(Space.SCRATCH)
    stamp(c, "rt_land")
    part = fx.Float32(0.0)
    for p in range_constexpr(quarter):
        part = part + got[p][0].bitcast(fx.Float32)
    fx.ptr_store(part, red + tid)
    gpu.barrier()
    if wave == 0:
        logits = []
        for i in range_constexpr(2):
            tot = fx.Float32(0.0)
            for qq in range_constexpr(4):
                tot = tot + fx.ptr_load(red + (qq * EXPERTS + lane + 64 * i))
            logits.append(tot)
        sc = [_sigmoid(logits[i]) for i in range(2)]
        neg = fx.Float32(float("-inf"))
        val = [sc[i] + bias[i] for i in range(2)]
        val = [fx.isnan(v).select(neg, v) for v in val]
        sc = [fx.isnan(v).select(fx.Float32(0.0), v) for v in sc]
        taken = [fx.Int32(0), fx.Int32(0)]
        total = fx.Float32(0.0)
        my_id, my_w = fx.Int32(0), fx.Float32(0.0)
        for k in range_constexpr(TOPK):
            v0 = (taken[0] == 0).select(val[0], neg)
            v1 = (taken[1] == 0).select(val[1], neg)
            pick1 = v1 > v0
            v = pick1.select(v1, v0)
            idx = pick1.select(fx.Int32(lane + 64), fx.Int32(lane))
            best = butterfly(v, (32, 16, 8, 4, 2, 1), fx.max)
            cand = (v == best).select(idx, fx.Int32(1 << 20))
            win = butterfly(cand, (32, 16, 8, 4, 2, 1), fx.min)
            mine0 = win == lane
            mine1 = win == lane + 64
            w_mine = mine0.select(sc[0], mine1.select(sc[1], fx.Float32(0.0)))
            w = butterfly(w_mine, (32, 16, 8, 4, 2, 1), fx.max)
            taken[0] = mine0.select(fx.Int32(1), taken[0])
            taken[1] = mine1.select(fx.Int32(1), taken[1])
            total = total + w
            my_id = (lane == k).select(win, my_id)
            my_w = (lane == k).select(w, my_w)
        scale = fx.Float32(ROUTED_SCALE) / fx.max(total, fx.Float32(1e-20))
        if const_expr(c["orouter"]):
            if (
                lane < TOPK
            ):  # row t of this rank's owner rows -> every rank's top-k exchange
                tg = c["row0"] + t
                for p in range_constexpr(TP):
                    c["mb"].put(c["peer"][p], ROUTE_P0 + tg * 2 * TOPK + lane, my_id)
                    c["mb"].put(
                        c["peer"][p],
                        ROUTE_P0 + tg * 2 * TOPK + TOPK + lane,
                        my_w * scale,
                    )
        elif lane < TOPK:
            buf_store(my_id, a["topk_ids"], t * TOPK + lane)
            buf_store(my_w * scale, a["topk_w"], t * TOPK + lane)
            c["mb"].put(c["route"], t * 2 * TOPK + lane, my_id)
            c["mb"].put(c["route"], t * 2 * TOPK + TOPK + lane, my_w * scale)


def stamp(c, name):
    """s_memrealtime -> `tl[CTA][point]` (timeline builds); `name` a TL name or a raw
    index (post_kernel.py)."""
    if const_expr(c["timeline"]):
        k = TL[name] if isinstance(name, str) else name
        buf_store(memrealtime(), c["args"]["tl"], fx.block_idx.x * TL_POINTS + k)


# ---------------------------------------------------------------- moe_buf zero, sort
@traced
def stage_zero(c, part):
    """moe_buf [S, HIDDEN] bf16 := 0, slice `part` of ZERO_CTAS (gemm2 adds into it
    atomically)."""
    words = c["S"] * HIDDEN // 2
    per = -(-words // ZERO_CTAS)
    per = -(-per // 4) * 4
    z = fx.Vector.filled(4, 0, fx.Int32)
    for it in range_constexpr(-(-per // (4 * THREADS))):
        w = part * per + (it * THREADS + c["tid"]) * 4
        if (w < (part + 1) * per) & (w < words):
            buf_store(z, c["args"]["moe_buf"], w)


@traced
def stage_sort(c):
    """Every row's top-k -> AITER moe_sorting's outputs (block 16): expert segments in
    ascending expert id, a pick's place in its segment by (row, slot), each segment
    padded to 16 with MOCK = (TOPK << 24) | S at weight 0, an expert id a 16-row block,
    num_valid = [padded rows, S]. Experts without picks get no block."""
    s, tid, a = c["S"], c["tid"], c["args"]
    npick = s * TOPK
    pe, pw, cnt, off, fill = c["pe"], c["pw"], c["cnt"], c["off"], c["fill"]
    if tid < EXPERTS:
        fx.ptr_store(fx.Int32(0), cnt + tid)
        if const_expr(c["fastsort"]):
            fx.ptr_store(fx.Int32(0), fill + tid)
    if const_expr(
        c["orouter"]
    ):  # every pick's pairs in one poll batch (S > 128: one round trip, not two)
        its = -(-npick // THREADS)
        specs = []
        for it in range_constexpr(its):
            pc = fx.min(it * THREADS + tid, npick - 1)
            base = ROUTE_P0 + (pc // TOPK) * 2 * TOPK + pc % TOPK
            specs += [(c["own"], base, 1), (c["own"], base + TOPK, 1)]
        picks = c["mb"].poll(specs, site=SITE_ROUTE)
        for it in range_constexpr(its):
            pp = it * THREADS + tid
            if pp < npick:
                buf_store(picks[2 * it][0], a["topk_ids"], pp)
                buf_store(picks[2 * it + 1][0].bitcast(fx.Float32), a["topk_w"], pp)
                fx.ptr_store(picks[2 * it][0], pe + pp)
                fx.ptr_store(picks[2 * it + 1][0].bitcast(fx.Float32), pw + pp)
    for it in range_constexpr(0 if c["orouter"] else -(-npick // THREADS)):
        p = it * THREADS + tid
        if p < npick:
            t = p // TOPK
            k = p % TOPK
            if const_expr(c["orouter"]):
                got = c["mb"].poll(
                    [
                        (c["own"], ROUTE_P0 + t * 2 * TOPK + k, 1),
                        (c["own"], ROUTE_P0 + t * 2 * TOPK + TOPK + k, 1),
                    ],
                    site=SITE_ROUTE,
                )
                buf_store(got[0][0], a["topk_ids"], p)
                buf_store(got[1][0].bitcast(fx.Float32), a["topk_w"], p)
            else:
                got = c["mb"].poll(
                    [
                        (c["route"], t * 2 * TOPK + k, 1),
                        (c["route"], t * 2 * TOPK + TOPK + k, 1),
                    ],
                    site=SITE_ROUTE,
                )
            fx.ptr_store(got[0][0], pe + p)
            fx.ptr_store(got[1][0].bitcast(fx.Float32), pw + p)
    gpu.barrier()
    stamp(c, "s_picks")
    for it in range_constexpr(-(-npick // THREADS)):
        p = it * THREADS + tid
        if p < npick:
            fx.llvm.atomic_add(
                cnt + fx.ptr_load(pe + p), fx.Int32(1), syncscope="workgroup"
            )
    gpu.barrier()

    def padded(n):
        return ((n + (SORT_BLOCK - 1)) // SORT_BLOCK) * SORT_BLOCK

    if const_expr(c["fastsort"]):
        sort_scan_wave(c, padded)
    else:
        sort_scan_serial(c, padded)
    stamp(c, "s_scan")
    for it in range_constexpr(-(-npick // THREADS)):
        p = it * THREADS + tid
        if p < npick:
            e = fx.ptr_load(pe + p)
            # a slot in the expert's segment (the order inside a segment only moves
            # gemm2's bf16 atomic adds, whose order is not fixed anyway)
            rank = fx.llvm.atomic_add(fill + e, fx.Int32(1), syncscope="workgroup")
            pos = fx.ptr_load(off + e) + rank
            buf_store(fx.Int32(((p % TOPK) << 24) | (p // TOPK)), a["sorted_ids"], pos)
            buf_store(fx.ptr_load(pw + p), a["sorted_w"], pos)
    stamp(c, "s_placed")
    if const_expr(c["fastsort"]):
        # thread (expert e, slot j): pad slot o + n + j of e's segment, expert id of e's
        # block j
        for it in range_constexpr(EXPERTS * SORT_BLOCK // THREADS):
            idx = it * THREADS + tid
            e = idx // SORT_BLOCK
            j = idx % SORT_BLOCK
            n = fx.ptr_load(cnt + e)
            o = fx.ptr_load(off + e)
            o_end = fx.ptr_load(off + e + 1)
            if o + n + j < o_end:
                buf_store(fx.Int32((TOPK << 24) | s), a["sorted_ids"], o + n + j)
                buf_store(fx.Float32(0.0), a["sorted_w"], o + n + j)
            if o // SORT_BLOCK + j < o_end // SORT_BLOCK:
                buf_store(e, a["sorted_eids"], o // SORT_BLOCK + j)
    else:
        sort_pad_serial(c)


@traced
def sort_scan_wave(c, padded):
    """Exclusive scan of the padded counts by wave 0 (two experts a lane, ds_bpermute
    Hillis-Steele) -> off[0 .. E], num_valid. `fill` was zeroed at the start."""
    s, tid, a, cnt, off = c["S"], c["tid"], c["args"], c["cnt"], c["off"]
    if tid < 64:
        p0 = padded(fx.ptr_load(cnt + 2 * tid))
        p1 = padded(fx.ptr_load(cnt + 2 * tid + 1))
        v = p0 + p1
        for d in range_constexpr(6):
            sh = 1 << d
            got = lane_gather(v, fx.max(tid - sh, fx.Int32(0)))
            v = v + (tid >= sh).select(got, fx.Int32(0))
        ex = v - p0 - p1
        fx.ptr_store(ex, off + 2 * tid)
        fx.ptr_store(ex + p0, off + 2 * tid + 1)
        if tid == 63:
            fx.ptr_store(v, off + EXPERTS)
            buf_store(v, a["num_valid"], 0)
            buf_store(fx.Int32(s), a["num_valid"], 1)
    gpu.barrier()


@traced
def sort_scan_serial(c, padded):
    s, tid, a, cnt, off, fill = (
        c["S"],
        c["tid"],
        c["args"],
        c["cnt"],
        c["off"],
        c["fill"],
    )
    # exclusive scan of the padded counts, two levels: 8 groups of 16 experts
    if tid < EXPERTS // 16:
        gs = fx.Int32(0)
        for j in range_constexpr(16):
            gs = gs + padded(fx.ptr_load(cnt + tid * 16 + j))
        fx.ptr_store(
            gs, fill + tid
        )  # fill doubles as the group sums until the placement
    gpu.barrier()
    if tid < EXPERTS:
        grp = tid // 16
        o = fx.Int32(0)
        for g in range_constexpr(EXPERTS // 16):
            o = o + (g < grp).select(fx.ptr_load(fill + g), fx.Int32(0))
        for j in range_constexpr(16):
            o = o + (j < tid % 16).select(
                padded(fx.ptr_load(cnt + grp * 16 + j)), fx.Int32(0)
            )
        fx.ptr_store(o, off + tid)
        if tid == EXPERTS - 1:
            tot = o + padded(fx.ptr_load(cnt + tid))
            fx.ptr_store(tot, off + EXPERTS)
            buf_store(tot, a["num_valid"], 0)
            buf_store(fx.Int32(s), a["num_valid"], 1)
    gpu.barrier()
    if tid < EXPERTS:
        fx.ptr_store(fx.Int32(0), fill + tid)
    gpu.barrier()


@traced
def sort_pad_serial(c):
    s, tid, a, cnt, off = c["S"], c["tid"], c["args"], c["cnt"], c["off"]
    if tid < EXPERTS:
        n = fx.ptr_load(cnt + tid)
        o = fx.ptr_load(off + tid)
        o_end = fx.ptr_load(off + tid + 1)
        for j in range(o + n, o_end):
            buf_store(fx.Int32((TOPK << 24) | s), a["sorted_ids"], j)
            buf_store(fx.Float32(0.0), a["sorted_w"], j)
        for b in range(o // SORT_BLOCK, o_end // SORT_BLOCK):
            buf_store(fx.Int32(tid), a["sorted_eids"], b)


# ---------------------------------------------------------------- walk
@traced
def run_glue(c, bid):
    s = c["S"]
    stamp(c, "entry")
    # Router CTAs (128 .. 255, no seam work) issue their weight loads first; a seam CTA
    # must not, or its polls would queue behind the burst.
    rtask = first_task(bid, ROUTER0)
    z = fx.Vector.filled(4, 0.0, fx.Float32)
    w0, w1, w2 = z, z, z
    if rtask < ROUTER_TASKS:
        w0, w1, w2 = router_weights(c, rtask)
    wv = [w0, w1, w2]
    if const_expr(c["sort"]):  # noqa: SIM102 (compile-time branch)
        if rtask < ZERO_CTAS:
            stage_zero(c, rtask)
    stamp(c, "zero")
    for task in range(first_task(bid, SEAM0), SEAM_TASKS, BLOCKS):
        if const_expr(c["lp_rs"]):
            stage_rs_push_lp(c, task)
        else:
            stage_rs_push(c, task)
    stamp(c, "push")
    for task in range(first_task(bid, SEAM0), SEAM_TASKS, BLOCKS):
        if const_expr(c["lp_ag"]):
            stage_rs_reduce_lpv(c, task)
        elif const_expr(c["lp_rs"]):
            stage_rs_reduce_lp3(c, task)
        elif const_expr(c["rsbatch"]):
            stage_rs_reduce_batched(c, task)
        else:
            stage_rs_reduce(c, task)
    stamp(c, "reduce")
    if const_expr(c["orouter"]):
        # This rank's G owner rows (reduced here: they land first) are normed on router
        # CTAs 128 .. 128 + G - 1, which issued no remote stores (a poll's vmcnt wait
        # would also wait for those stores' acks), then routed there. The 3G peer rows:
        # seam CTAs 0 .. 127, the rest on router CTAs 128 + G .. after their router
        # task.
        g_rows, rank = c["G"], c["rank"]
        npeer = s - g_rows
        if const_expr(not c["rnorm"]):  # noqa: SIM102 (compile-time branch)
            if (bid >= ROUTER0) & (bid < ROUTER0 + g_rows):
                row_stage(c, rank * g_rows + bid - ROUTER0)
        if rtask < ROUTER_TASKS:
            if const_expr(c["rnorm"]):
                stage_router_rn(c, rtask, wv)
            else:
                stage_router(c, rtask, wv)
        stamp(c, "router")
        gpu.barrier()  # LDS ``red`` changes hands between stages
        for j in range(first_task(bid, ROUTER0), g_rows, BLOCKS):
            stage_route(c, j)
        stamp(c, "route")
        if const_expr(c["rnorm"]):  # noqa: SIM102 (compile-time branch) - the owner rows' outputs, after the route
            if (bid >= ROUTER0) & (bid < ROUTER0 + g_rows):
                gpu.barrier()
                row_stage(c, rank * g_rows + bid - ROUTER0)
        if bid < min(npeer, ROUTER0):
            row_stage(c, peer_row(c, bid))
        if const_expr(npeer > ROUTER0):  # noqa: SIM102 (compile-time branch)
            if (bid >= ROUTER0 + g_rows) & (bid < ROUTER0 + g_rows + npeer - ROUTER0):
                gpu.barrier()
                row_stage(c, peer_row(c, bid - g_rows))
    else:
        for t in range(first_task(bid, NORM0), s, BLOCKS):
            row_stage(c, t)
        if rtask < ROUTER_TASKS:
            stage_router(c, rtask, wv)
        stamp(c, "router")
        for t in range(first_task(bid, ROUTE0), s, BLOCKS):
            stage_route(c, t)
        stamp(c, "route")
    if const_expr(c["sort"]):  # noqa: SIM102 (compile-time branch)
        if bid == SORT_CTA:
            stage_sort(c)
    stamp(c, "sort")
    if const_expr(c["lp_rs"]):
        for _ in range_constexpr(c["lprearm"]):
            lp_rearm_end(c, bid)


# afinish: the epoch block's arrival counter (monotonic; zeroed with the epoch at a
# reset)
DONE_WORD = 1


@traced
def finish(c, epoch, tag):
    """Every CTA marks the launch done; CTA 0 waits for the marks, then moves the epoch
    on. afinish: every CTA adds one to a device counter; the CTA whose add completes a
    multiple of BLOCKS (the launch's last) moves the epoch."""
    tid, red = c["tid"], c["red"]
    if const_expr(c.get("afinish", False)):
        cta_sync(drain=True)
        if tid == 0:
            ptr = fx.inttoptr(
                fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
                epoch + fx.Int64(4 * DONE_WORD),
            )
            old = fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent")
            if ((old + 1) & (BLOCKS - 1)) == 0:
                store32(epoch, 0, tag, "agent")
        return
    cta_sync(drain=True)
    if tid == 0:
        sync_release("agent")
        store32(epoch, EPOCH_MARKS + fx.block_idx.x, tag, "agent")
    if fx.block_idx.x == 0:
        if tid == 0:
            fx.ptr_store(fx.Int32(0).bitcast(fx.Float32), red)
        gpu.barrier()
        if tid < BLOCKS:
            t0 = memrealtime()
            mark = load32(epoch, EPOCH_MARKS + tid, "agent")
            late = fx.Int64(0)
            while (mark != tag) & (late < c["mb"].ticks):
                rocdl.s_sleep(1)
                mark = load32(epoch, EPOCH_MARKS + tid, "agent")
                late = memrealtime() - t0
            if mark != tag:
                fx.ptr_store(fx.Int32(1).bitcast(fx.Float32), red)
            sync_acquire("agent")
        cta_sync()
        if tid == 0:
            if fx.ptr_load(red).bitcast(fx.Int32) == 0:
                store32(epoch, 0, tag, "agent")
            else:
                c["mb"].fail(SITE_EPOCH)


@traced
def refuse_wrap(c, bid, ep):
    if (bid == 0) & (c["tid"] == 0) & (ep >= EPOCH_LIMIT):
        c["mb"].fail(10)


def kernel_name(key: GlueBuild) -> str:
    key = resolve(key)
    return (
        f"m31_decode_glue_s{key.rows}"
        + ("" if key.sort else "_ab")
        + ("_tl" if key.timeline else "")
        + ("_pl" if key.plain else "")
        + ("_rb" if key.rsbatch else "")
        + ("_or" if key.orouter else "")
        + ("_af" if key.afinish else "")
        + ("_rv" if key.rowvec else "")
        + ("_rb16" if key.rbf16 else "")
        + ("_pp" if key.prepoll else "")
        + ("_mx" if key.mixdst else "")
        + ("_fs" if key.fastsort else "")
        + ("_ad" if key.agdefer else "")
        + ("_xf" if key.xfirst else "")
        + ("_of" if key.ownfirst else "")
        + ("_rn" if key.rnorm else "")
        + {"tagged": "", "lamport": "_lp" if key.lpag else "_lr"}[str(key.transport)]
        + ("_lq" if key.lpprobe else "")
        + ("_la" if key.lpack else "")
        + (f"_ra{key.lprearm}" if key.lprearm != 1 else "")
        + (f"_m{key.lpmix}" if key.transport != "tagged" and key.lpmix != 1 else "")
    )


def build_glue(key: GlueBuild):
    key = resolve(key)
    s = key.rows
    assert MIN_ROWS <= s <= MAX_ROWS and s % TP == 0, s
    g_rows = s // TP
    layout = scratch_layout(s)
    keyed = (
        s,
        key.sort,
        key.timeline,
        key.wait_ticks,
        key.plain,
        key.rsbatch,
        key.orouter,
        key.afinish,
        key.rowvec,
        key.rbf16,
        key.prepoll,
        key.mixdst,
        key.fastsort,
        key.agdefer,
        key.xfirst,
        key.ownfirst,
        key.rnorm,
        key.transport,
        key.lpag,
        key.lpprobe,
        key.lpmix,
        key.lpack,
        key.lprearm,
        _SOURCES,
    )

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, -(-MAX_ROWS // 16) * WAVES * 64 * 4, 16]
        pe: fx.Array[fx.Int32, MAX_PICKS, 16]
        pw: fx.Array[fx.Float32, MAX_PICKS, 16]
        cnt: fx.Array[fx.Int32, EXPERTS, 16]
        off: fx.Array[fx.Int32, EXPERTS + 16, 16]
        fill: fx.Array[fx.Int32, EXPERTS, 16]
        scl: fx.Array[fx.Int32, HIDDEN // 32, 16]

    @flyc.kernel(name=kernel_name(key), known_block_size=[THREADS, 1, 1])
    def kglue(
        part: Int64,
        res_in: Int64,
        res_out: Int64,
        xn: Int64,
        xq: Int64,
        xsc: Int64,
        topk_ids: Int64,
        topk_w: Int64,
        norm_w: Int64,
        router_w: Int64,
        bias: Int64,
        sorted_ids: Int64,
        sorted_w: Int64,
        sorted_eids: Int64,
        num_valid: Int64,
        moe_buf: Int64,
        tl: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        mode: Int32,
    ):
        _ = keyed
        if mode != 0:
            bid = fx.block_idx.x
            tid = fx.thread_idx.x
            lds = fx.SharedAllocator().allocate(Lds).peek()
            ep = load32(epoch, 0, "agent")
            tag = ep + 1
            raw = peer_bases(peers, TP)
            par = fx.Int64(ep & 1) * fx.Int64(HALF_BYTES) + fx.Int64(ERROR_BYTES)
            tag16 = ((ep >> 1) + 1) & 0xFFFF
            mb = Mailbox(
                tag, sym, raw, rank, ep, key.wait_ticks, tag16, lazy_errors=True
            )
            c = {
                "S": s,
                "G": g_rows,
                "tid": tid,
                "lane": tid % 64,
                "wave": tid // 64,
                "rank": rank,
                "mb": mb,
                "red": lds.red.ptr,
                "pe": lds.pe.ptr,
                "pw": lds.pw.ptr,
                "cnt": lds.cnt.ptr,
                "off": lds.off.ptr,
                "fill": lds.fill.ptr,
                "scl": lds.scl.ptr,
                "sort": key.sort,
                "sharded": mode == 2,
                "timeline": key.timeline,
                "plain": key.plain,
                "rsbatch": key.rsbatch,
                "orouter": key.orouter,
                "afinish": key.afinish,
                "rowvec": key.rowvec,
                "rbf16": key.rbf16,
                "prepoll": key.prepoll,
                "mixdst": key.mixdst,
                "fastsort": key.fastsort,
                "agdefer": key.agdefer,
                "xfirst": key.xfirst,
                "ownfirst": key.ownfirst,
                "rnorm": key.rnorm,
                "lp_rs": key.transport == "lamport",
                "lp_ag": key.lpag,
                "lpprobe": key.lpprobe,
                "lpmix": key.lpmix,
                "lpack": key.lpack,
                "lprearm": key.lprearm,
                "lpar": fx.Int64(ep & 1),
                "sym": sym,
                "raw": raw,
                "epoch": epoch,
                "RR": g_rows if key.orouter else s,
                "row0": rank * g_rows if key.orouter else 0,
                "own": preg(sym + par, 0, "peer"),
                "peer": [preg(b + par, 0, "peer") for b in raw],
                "args": {
                    "part": part,
                    "res_in": res_in,
                    "res_out": res_out,
                    "xn": xn,
                    "xq": xq,
                    "xsc": xsc,
                    "topk_ids": topk_ids,
                    "topk_w": topk_w,
                    "norm_w": norm_w,
                    "router_w": router_w,
                    "bias": bias,
                    "sorted_ids": sorted_ids,
                    "sorted_w": sorted_w,
                    "sorted_eids": sorted_eids,
                    "num_valid": num_valid,
                    "moe_buf": moe_buf,
                    "tl": tl,
                },
            }
            for region, (off, _) in layout.items():
                c[region] = sreg(scratch, off, region)
            refuse_wrap(c, bid, ep)
            run_glue(c, bid)
            stamp(c, "done")
            finish(c, epoch, tag)
            stamp(c, "epoch")

    @flyc.jit
    def launch(
        part: Int64,
        res_in: Int64,
        res_out: Int64,
        xn: Int64,
        xq: Int64,
        xsc: Int64,
        topk_ids: Int64,
        topk_w: Int64,
        norm_w: Int64,
        router_w: Int64,
        bias: Int64,
        sorted_ids: Int64,
        sorted_w: Int64,
        sorted_eids: Int64,
        num_valid: Int64,
        moe_buf: Int64,
        tl: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        mode: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kglue(
            part,
            res_in,
            res_out,
            xn,
            xq,
            xsc,
            topk_ids,
            topk_w,
            norm_w,
            router_w,
            bias,
            sorted_ids,
            sorted_w,
            sorted_eids,
            num_valid,
            moe_buf,
            tl,
            scratch,
            sym,
            peers,
            rank,
            epoch,
            mode,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
