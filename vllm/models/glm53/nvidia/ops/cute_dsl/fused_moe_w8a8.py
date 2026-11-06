# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 fused MoE for 9 to 16 decode rows (the W8A8 row tier), in CuTe DSL."""

from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Any

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, Uint64

from . import _rt as rt
from .fused_moe import _HINT, LAYOUT, MoeRouting
from .w8a8_shape import (
    A_STAGE,
    DOWN_CTAS,
    DOWN_GROUPS,
    DOWN_ROW_BLOCK,
    EXPERTS,
    GRID_CTAS,
    HIDDEN,
    K_BLOCK,
    K_BLOCKS,
    LOW_CTAS,
    OWNER_ROWS,
    SHARD,
    SLICE,
    SLICES,
    TOP_K,
    TierShape,
    _align,
)

cdiv, cmod = rt.cdiv, rt.cmod

THREADS = 384
ROW_CHUNK = 8
ROUTER_ROW_BYTES = 2 * HIDDEN
ROUTER_SPLIT = 4
ROUTER_COLUMNS = LOW_CTAS // ROUTER_SPLIT
ROUTER_EXPERTS = EXPERTS // ROUTER_COLUMNS
ROUTER_GROUP = 4
#: Bytes of one router row's K third (the unit of the router weights' bulk copies).
ROUTER_THIRD = ROUTER_ROW_BYTES // 3
SMEM_BYTES = 228608
MAX_STAGES = 8
LOG2E = 1.4426950408889634
FLT_MAX = 3.4028234663852886e38
FP8_MAX = 448.0
QUANT_EPS = 1e-10
BIG = 1 << 20
#: Named barrier of the Up promotion warps and the signaling warp (warp 10).
SIGNAL_BARRIER = 13
FIN_TOKENS = 16


@cache
def smem_layout(rmax: int) -> type[rt.SmemLayout]:
    """The shared-memory map of the tier with row capacity ``rmax`` (one class per
    tier).
    """
    sh = TierShape(rmax)
    q = rmax
    router_bytes = max(
        ROUTER_EXPERTS * ROUTER_ROW_BYTES, 2 * ROUTER_ROW_BYTES + 2 * q * EXPERTS * 4
    )
    small = (
        256 * 4
        + 3 * q * TOP_K * 4
        + 64
        + q * 4
        + 4096
        + 320
        + q * OWNER_ROWS * 2
        + q * K_BLOCKS * 4
        + 4096
    )
    avail = SMEM_BYTES - _align(small, 1024) - 1024
    stages = min(MAX_STAGES, avail // sh.ring_stage)
    down_stages = 4 if 2 * q * 5 <= 512 and 4 * sh.down_stage <= avail else 3
    assert stages >= 4 and down_stages * sh.down_stage <= avail, (
        rmax,
        stages,
        down_stages,
    )
    ring_bytes = max(stages * sh.ring_stage, down_stages * sh.down_stage, router_bytes)
    assert ring_bytes <= avail, (rmax, ring_bytes, avail)
    tmem_columns = 512 if 2 * q * (down_stages + 1) > 256 else 256

    class W8A8Smem(rt.SmemLayout):
        STAGES = stages
        DOWN_STAGES = down_stages
        TMEM_COLUMNS = tmem_columns
        LOGITS = 2 * ROUTER_ROW_BYTES
        CORRECTED = 2 * ROUTER_ROW_BYTES + q * EXPERTS * 4
        a = rt.SmemAlloc(SMEM_BYTES)
        ring_full = a.mbarriers(MAX_STAGES, arrivals=1)
        ring_empty = a.mbarriers(MAX_STAGES, arrivals=1)
        down_full = a.mbarriers(4, arrivals=1)
        down_ready = a.mbarriers(4, arrivals=1)
        down_free = a.mbarriers(4, arrivals=4)
        acc_full = a.mbarriers(2, arrivals=1)
        acc_empty = a.mbarriers(2, arrivals=4)
        router_full = a.mbarriers(3, arrivals=1)
        union_full = a.mbarriers(1, arrivals=1)
        scores_full = a.mbarriers(1, arrivals=1)
        scales_full = a.mbarriers(1, arrivals=1)
        up_done = a.mbarriers(1, arrivals=1)
        tmem_hold = a.array(Int32, 1)
        epoch_slot = a.array(Uint32, 1)
        union_count = a.array(Int32, 1)
        first_pair = a.array(Int32, EXPERTS)
        route_ids = a.array(Int32, q * TOP_K)
        route_weights = a.array(Float32, q * TOP_K)
        union_ids = a.array(Int32, sh.max_union)
        owners = a.array(Uint32, 16)
        inverse = a.array(Float32, q)
        scratch = a.array(Float32, 1024)
        fin_amax = a.array(Float32, 64)
        fin_scale = a.array(Float32, 16)
        local_o = a.array(BFloat16, q * OWNER_ROWS, also=(Uint32,))
        token_scales = a.array(Float32, q * K_BLOCKS, align=16)
        up_scales = a.array(Float32, 256)
        ring = a.array(
            Uint8,
            ring_bytes,
            align=1024,
            also=(BFloat16, Uint64, Uint32, Float32),
        )

    return W8A8Smem


@rt.record
class MoeW8A8Params(rt.Params):
    """The kernel's parameters: the order of :class:`MoeParams` with ``Activation``
    replaced by ``Work`` (the shared scratch tensor), and the row table last (the
    provider inserts its address).
    """

    X: cute.Pointer = rt.ptr("bfloat16")
    Gamma: cute.Pointer = rt.ptr("bfloat16")
    RouterWeight: cute.Pointer = rt.ptr("bfloat16")
    CorrectionBias: cute.Pointer = rt.ptr("float32")
    RoutedUp: cute.Pointer = rt.ptr("float8_e4m3")
    RoutedDown: cute.Pointer = rt.ptr("float8_e4m3")
    SharedUp: cute.Pointer = rt.ptr("float8_e4m3")
    SharedDown: cute.Pointer = rt.ptr("float8_e4m3")
    RoutedUpScales: cute.Pointer = rt.ptr("float32")
    RoutedDownScales: cute.Pointer = rt.ptr("float32")
    SharedScales: cute.Pointer = rt.ptr("float32")
    RouterLogits: cute.Pointer = rt.ptr("float32")
    Work: cute.Pointer = rt.ptr("uint32")
    RouteIds: cute.Pointer = rt.ptr("int32")
    RouteWeights: cute.Pointer = rt.ptr("float32")
    LocalO: cute.Pointer = rt.ptr("bfloat16")
    Final: cute.Pointer = rt.ptr("bfloat16")
    FullEpoch: cute.Pointer = rt.ptr("uint32")
    Flags: cute.Pointer = rt.ptr("uint32")
    peer_rank: Int32
    peer_base0: Int64
    peer_base1: Int64
    peer_base2: Int64
    peer_base3: Int64
    peer_base4: Int64
    peer_base5: Int64
    peer_base6: Int64
    peer_base7: Int64
    RowTable: cute.Pointer = rt.ptr("int32", align=4)


@rt.record
class MoeW8A8SeamParams(MoeW8A8Params):
    """The seam's parameters: the next layer's RMSNorm weight and its scaled input."""

    GammaNext: cute.Pointer = rt.ptr("float32")
    XG: cute.Pointer = rt.ptr("bfloat16")


@dataclass(frozen=True)
class W8A8State:
    """One CTA's state: thread ids, the shared-memory base and the TMEM base."""

    base: Int32
    tid: Int32 = None
    warp: Int32 = None
    lane: Int32 = None
    bid: Int32 = None
    taddr: Int32 = None

    if TYPE_CHECKING:
        # The shared-memory record's views (rt.SmemLayout.record).
        def __getattr__(self, name: str) -> Any: ...


def _bind(s: W8A8State, ids: rt.ThreadIds) -> W8A8State:
    return replace(s, tid=ids.tid, warp=ids.warp, lane=ids.lane, bid=ids.bid)


class FusedMoeW8A8(MoeRouting, rt.UpDownMapKernel):
    """The W8A8 row-tier MoE of the module docstring; ``RMAX`` is the tier's row
    capacity.
    """

    RMAX = 16
    #: The seam variant hands the next layer's shared worker ``Final x gamma`` as well.
    SEAM = False
    GRID = GRID_CTAS
    Params = MoeW8A8Params
    ROLES = (rt.Role("compute", 0, 7, 208, True), rt.Role("load", 8, 11, 80, False))
    SMEM_BYTES = SMEM_BYTES

    @property
    def shape(self) -> TierShape:
        return TierShape(self.RMAX, self.SEAM)

    @property
    def exchange(self) -> rt.SliceExchange:
        return rt.SliceExchange(self.RMAX, self.SEAM)

    @property
    def Smem(self) -> type[rt.SmemLayout]:  # noqa: N802
        return smem_layout(self.RMAX)

    def mbarriers(self) -> tuple:
        return self.Smem.barriers()

    def smem(self, base):
        return self.Smem.record(base, bases=(W8A8State,))

    def tensor_maps(self, p) -> tuple:
        w13, w2 = LAYOUT.w13_expert_bytes, LAYOUT.w2_expert_bytes
        groups = [LAYOUT.routed_experts * size // (16 * K_BLOCK) for size in (w13, w2)]
        return (
            rt.byte_map(
                p.RoutedUp.toint(),
                [K_BLOCK, 2, 8, groups[0]],
                [1024, K_BLOCK, 2048],
                [K_BLOCK, 2, 8, SLICE // 16],
                True,
            ),
            rt.byte_map(
                p.RoutedDown.toint(),
                [K_BLOCK, 2, 8, groups[1]],
                [1024, K_BLOCK, 2048],
                [K_BLOCK, 2, 8, DOWN_ROW_BLOCK // 16],
                True,
            ),
            rt.byte_map(
                p.SharedUp.toint(),
                [HIDDEN, 2 * SHARD, 1, 1],
                [HIDDEN, w13, w13],
                [K_BLOCK, SLICE, 1, 1],
                True,
            ),
            rt.byte_map(
                p.SharedDown.toint(),
                [SHARD, HIDDEN, 1, 1],
                [SHARD, w2, w2],
                [K_BLOCK, DOWN_ROW_BLOCK, 1, 1],
                True,
            ),
        )

    @cute.kernel
    def kernel(
        self,
        p,
        UpMap: rt.GridConstant[rt.TensorMap],
        DownMap: rt.GridConstant[rt.TensorMap],
        SharedUpMap: rt.GridConstant[rt.TensorMap],
        SharedDownMap: rt.GridConstant[rt.TensorMap],
    ):
        self.up_map, self.down_map = UpMap.get_ptr().toint(), DownMap.get_ptr().toint()
        self.shared_up_map = SharedUpMap.get_ptr().toint()
        self.shared_down_map = SharedDownMap.get_ptr().toint()
        ids, s = self.prologue(p)
        self.run(p, s, ids)

    @cute.jit
    def weight_box(self, maps, kb, row, expert, kblocks, rows):
        shared = expert == LAYOUT.shared_slot
        group = cdiv((expert * kblocks + kb) * rows + row, 16)
        return (
            cutlass.select_(shared, maps[1], maps[0]),
            cutlass.select_(shared, Int32(kb) * K_BLOCK, 0),
            cutlass.select_(shared, row, 0),
            cutlass.select_(shared, 0, group),
        )

    @cute.jit
    def run(self, p, s, ids: rt.ThreadIds) -> None:
        sm = self.Smem
        taddr = rt.Umma.alloc(s.base, sm.tmem_hold.offset, sm.TMEM_COLUMNS, ids.warp)
        rt.RoleKernel.run(self, p, replace(s, taddr=taddr), ids)
        cute.arch.sync_threads()
        rt.Umma.free(s.base, sm.tmem_hold.offset, sm.TMEM_COLUMNS, ids.warp)

    @cute.jit
    def signal(self, p, warp, index):
        with rt.warp_leader(warp, 0):
            rt.Memory.fence_release_gpu()
            rt.Memory.atomic_add_u32(p.Flags, index, 1)

    @cute.jit
    def compute(self, p: MoeW8A8Params, s: W8A8State, ids: rt.ThreadIds):
        p, s = p.views(), _bind(s, ids)
        tid, warp, bid = s.tid, s.warp, s.bid  # noqa: F841  unused here, kept with the other bindings
        if bid < LOW_CTAS:
            self.router_issue(p, s)
        bias = p.CorrectionBias[self.router_column(s)]
        self.set_regs("compute")
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        rows = self.active_rows(p)
        if bid < LOW_CTAS:
            self.front(p, s, rows, bias)
            self.signal(p, s.warp, self.shape.flag_router + bid)
        rt.bar_sync(11, THREADS)
        if bid < LOW_CTAS:
            self.publish_image(p, s, rows)
            self.signal(p, s.warp, self.shape.flag_image + bid)
        rt.mbar_wait(s.scores_full, 0)
        self.select_and_union(p, s, rows)
        self.compute_stages(p, s, rows, warp, bid)

    @cute.jit
    def compute_stages(self, p, s, rows, warp, bid):
        epoch = s.epoch_slot[0]
        units = self.unit_count(s)
        if warp < 4:
            self.up_promote(p, s, units)
            self.down_promote(p, s, rows)
        else:
            self.finalize(p, s, rows, epoch)
        if bid < LOW_CTAS:
            self.owner(p, s, rows, epoch)

    @cute.jit
    def router_issue(self, p, s):
        with rt.warp_leader(s.warp, 0):
            cute.arch.griddepcontrol_wait()
            policy = rt.AsyncCopy.createpolicy_evict_first(
                rt.ga(p.RouterWeight, 0), 3145728, 3145728
            )
            first = cmod(s.bid, ROUTER_COLUMNS) * ROUTER_EXPERTS
            for v in cutlass.range_constexpr(3):
                rt.mbar_expect_tx(s.router_full + v * 8, ROUTER_EXPERTS * ROUTER_THIRD)
                for e in cutlass.range_constexpr(ROUTER_EXPERTS):
                    rt.bulk_g2s_hint(
                        s.ring_addr + (v * ROUTER_EXPERTS + e) * ROUTER_THIRD,
                        rt.ga(p.RouterWeight, (first + e) * HIDDEN + v * 2048),
                        ROUTER_THIRD,
                        s.router_full + v * 8,
                        policy,
                    )

    @cute.jit
    def router_column(self, s):
        return cmod(s.bid, ROUTER_COLUMNS) * ROUTER_EXPERTS + cmod(
            s.tid, ROUTER_EXPERTS
        )

    @cute.jit
    def front(self, p, s, rows, bias):
        rt.mbar_wait(s.router_full, 0)
        gammas = self.router_gammas(p, s)
        self.router_groups(p, s, rows, gammas, bias)
        for v in cutlass.range_constexpr(1, 3):
            rt.mbar_wait(s.router_full + v * 8, 0)

    @cute.jit
    def publish_image(self, p, s, rows):
        self.quantize_rows(p, s, rows)
        rt.Memory.fence_proxy_async_global()
        rt.bar_sync(8, 256)

    @cute.jit
    def router_gammas(self, p, s):
        gammas = []
        for v in cutlass.range_constexpr(3):
            gammas += rt.Convert.bf16x8_to_f32(
                rt.Memory.ld_global_v4(rt.ga(p.Gamma, v * 2048 + s.tid * 8))
            )
        return gammas

    @cute.jit
    def router_groups(self, p, s, rows, gammas, bias):
        slice_ = cdiv(s.bid, ROUTER_COLUMNS)
        groups = cdiv(rows + ROUTER_GROUP - 1, ROUTER_GROUP)
        for group in range(slice_, groups, ROUTER_SPLIT):
            self.router_group(p, s, group, rows, gammas, bias)

    @cute.jit
    def router_group(self, p, s, group, rows, gammas, bias):
        tid, warp, lane = s.tid, s.warp, s.lane
        first = group * ROUTER_GROUP
        squares = [Float32(0.0) for _ in range(ROUTER_GROUP)]
        logits = [
            [Float32(0.0) for _ in range(ROUTER_EXPERTS)] for _ in range(ROUTER_GROUP)
        ]
        for v in cutlass.range_constexpr(3):
            rt.mbar_wait(s.router_full + v * 8, 0)
            scaled = []
            for r in cutlass.range_constexpr(ROUTER_GROUP):
                row = cutlass.min(first + r, rows - 1)
                x = rt.Convert.bf16x8_to_f32(
                    rt.Memory.ld_global_v4(
                        rt.ga(p.X, row * HIDDEN + v * 2048 + tid * 8)
                    )
                )
                values = []
                for i in cutlass.range_constexpr(8):
                    squares[r] = rt.fma(x[i], x[i], squares[r])
                    values.append(rt.mul(x[i], gammas[v * 8 + i]))
                scaled.append(values)
            for e in cutlass.range_constexpr(ROUTER_EXPERTS):
                w = rt.Convert.bf16x8_to_f32(
                    rt.Memory.lds_v4(
                        s.ring_addr + (v * ROUTER_EXPERTS + e) * ROUTER_THIRD + tid * 16
                    )
                )
                for r in cutlass.range_constexpr(ROUTER_GROUP):
                    acc = logits[r][e]
                    for i in cutlass.range_constexpr(8):
                        acc = rt.fma(scaled[r][i], w[i], acc)
                    logits[r][e] = acc
        flat = [x for per_row in logits for x in per_row]
        # Lane ``l`` ends with row ``l // 8``, expert ``l % 8`` of the warp's partials.
        s.scratch[warp * 32 + lane] = rt.Reduce.fold(flat, lane, 16, ())
        total = rt.Reduce.fold(squares, lane, 16, (4, 2, 1))
        if (lane & 7) == 0:
            s.scratch[256 + warp * ROUTER_GROUP + cdiv(lane, 8)] = total
        rt.bar_sync(8, 256)
        if tid < 32:
            row_in, expert = cdiv(tid, ROUTER_EXPERTS), cmod(tid, ROUTER_EXPERTS)
            logit = rt.Reduce.strided_sum(s.scratch, tid, 32)
            inverse = rt.Reduce.rms_inverse(
                rt.Reduce.strided_sum(s.scratch, 256 + row_in, ROUTER_GROUP)
            )
            if first + row_in < rows:
                at = (first + row_in) * EXPERTS + self.router_column(s)
                value = rt.mul(logit, inverse)
                score = rt.FastMath.rcp_rn(
                    rt.add(1.0, cute.math.exp2(rt.mul(-value, LOG2E), fastmath=True))
                )
                biased = rt.add(score, bias)
                finite = rt.FastMath.fcmp(">=", biased, -FLT_MAX) & rt.FastMath.fcmp(
                    "<=", biased, FLT_MAX
                )
                p.RouterLogits[at] = value
                p.Work[self.shape.scores // 4 + at] = rt.Convert.f32_bits(score)
                if finite:
                    p.Work[self.shape.corrected // 4 + at] = rt.Convert.f32_bits(biased)
                else:
                    p.Work[self.shape.corrected // 4 + at] = rt.Convert.f32_bits(
                        Float32(-FLT_MAX)
                    )
            if expert == 0:
                s.inverse[first + row_in] = inverse
        rt.bar_sync(8, 256)

    @cute.jit
    def quantize_rows(self, p, s, rows):
        slice_, column = cdiv(s.bid, ROUTER_COLUMNS), cmod(s.bid, ROUTER_COLUMNS)
        groups = cdiv(rows + ROUTER_GROUP - 1, ROUTER_GROUP)
        mine = cdiv(groups - slice_ + ROUTER_SPLIT - 1, ROUTER_SPLIT)
        first_job = column + ROUTER_COLUMNS * s.warp
        for job in range(first_job, mine * K_BLOCKS, 8 * ROUTER_COLUMNS):
            group = slice_ + ROUTER_SPLIT * cdiv(job, K_BLOCKS)
            self.quantize_job(p, s, group * ROUTER_GROUP, cmod(job, K_BLOCKS))

    @cute.jit
    def quantize_job(self, p, s, first, kb):
        sh = self.shape
        q = self.RMAX
        token = first + cdiv(s.lane, 8)
        c8 = cmod(s.lane, 8)
        column = kb * K_BLOCK + c8 * 16
        inverse = s.inverse[token]
        values = []
        amax = Float32(0.0)
        for half in cutlass.range_constexpr(2):
            x = rt.Convert.bf16x8_to_f32(
                rt.Memory.ld_global_v4(rt.ga(p.X, token * HIDDEN + column + half * 8))
            )
            g = rt.Convert.bf16x8_to_f32(
                rt.Memory.ld_global_v4(rt.ga(p.Gamma, column + half * 8))
            )
            for i in cutlass.range_constexpr(8):
                v = rt.Convert.bf16_round(rt.mul(rt.mul(x[i], inverse), g[i]))
                values.append(v)
                amax = rt.FastMath.fmax(amax, rt.FastMath.abs_ftz(v))
        for bit in cutlass.range_constexpr(3):
            amax = rt.FastMath.fmax(amax, cute.arch.shuffle_sync_bfly(amax, 1 << bit))
        scale = rt.FastMath.div_rn(rt.FastMath.fmax(amax, QUANT_EPS), FP8_MAX)
        words = []
        for w in cutlass.range_constexpr(4):
            words.append(
                rt.Convert.e4m3x4(
                    *[rt.FastMath.div_rn(values[4 * w + i], scale) for i in range(4)]
                )
            )
        row8 = cmod(token, 8)
        byte = (
            sh.image
            + kb * q * K_BLOCK
            + cdiv(token, 8) * 1024
            + row8 * 128
            + ((c8 ^ row8) * 16)
        )
        rt.Memory.stg_u32x4(p.Work, cdiv(byte, 4), words)
        if c8 == 0:
            p.Work[sh.image_scales // 4 + kb * q + token] = rt.Convert.f32_bits(scale)

    @cute.jit
    def select_and_union(self, p, s, rows):
        self.select_reset(p, s)
        self.select_routes(p, s, rows)
        self.select_union(p, s, rows)

    @cute.jit
    def select_reset(self, p, s):
        s.first_pair[s.tid] = BIG
        if s.tid < 16:
            s.owners[s.tid] = Uint32(0)
        rt.bar_sync(8, 256)

    @cute.jit
    def select_routes(self, p, s, rows):
        sm = self.Smem
        logits = rt.Memory.smem_view_n(
            s.base, sm.ring.offset + sm.LOGITS, Float32, self.RMAX * EXPERTS
        )
        corrected = rt.Memory.smem_view_n(
            s.base, sm.ring.offset + sm.CORRECTED, Float32, self.RMAX * EXPERTS
        )
        routes = (
            corrected,
            logits,
            s.route_ids,
            s.route_weights,
            p.RouteIds,
            p.RouteWeights,
        )
        mine = cdiv(rows + 7 - s.warp, 8)
        for pair in range(cdiv(mine + 1, 2)):
            first = s.warp + 16 * pair
            second = first + 8
            if second >= rows:
                second = first
            self.route_pair(routes, s.lane, s.bid, first, second)
        rt.bar_sync(8, 256)

    @cute.jit
    def route_pair(self, routes, lane, bid, row_a, row_b):
        corrected, scores, route_ids, route_weights, RouteIds, RouteWeights = routes
        rows = (row_a, row_b)
        reads = (row_a, row_b)
        local = (
            cute.make_rmem_tensor(8, Float32),
            cute.make_rmem_tensor(8, Float32),
        )
        for r in cutlass.range_constexpr(2):
            for item in cutlass.range_constexpr(8):
                local[r][item] = corrected[reads[r] * EXPERTS + lane + 32 * item]
        for slot in cutlass.range_constexpr(TOP_K):
            best = []
            for r in cutlass.range_constexpr(2):
                max_lo = rt.FastMath.max3(local[r][0], local[r][1], local[r][2])
                max_mid = rt.FastMath.max3(local[r][3], local[r][4], local[r][5])
                max_hi = rt.FastMath.fmax(local[r][6], local[r][7])
                best.append(
                    rt.Sync.redux_max_nan_f32(rt.FastMath.max3(max_lo, max_mid, max_hi))
                )
            owned = []
            for r in cutlass.range_constexpr(2):
                mine = Uint32(4294967295)
                for j in cutlass.range_constexpr(8):
                    if rt.FastMath.fcmp("==", local[r][7 - j], best[r]):
                        mine = Uint32(lane + 32 * (7 - j))
                owned.append(mine)
            for r in cutlass.range_constexpr(2):
                best_id = Int32(rt.Sync.redux_min_u32(owned[r]))
                for item in cutlass.range_constexpr(8):
                    if best_id == lane + 32 * item:
                        local[r][item] = -float("inf")
                if lane == 0:
                    route_ids[rows[r] * 8 + slot] = best_id
        cute.arch.sync_warp()
        picked = []
        for r in cutlass.range_constexpr(2):
            route = rows[r] * 8 + cmod(lane, 8)
            picked.append(scores[reads[r] * EXPERTS + route_ids[route]])
        totals = []
        for r in cutlass.range_constexpr(2):
            selected = Float32(0.0)
            if lane < 8:
                selected = picked[r]
            totals.append(rt.Reduce.fold([selected], lane, 16, (16, 8, 4, 2, 1)))
        for r in cutlass.range_constexpr(2):
            total = totals[r]
            if rt.FastMath.fcmp("<=", total, 0.0):
                total = Float32(1.0)
            route = rows[r] * 8 + cmod(lane, 8)
            weight = rt.FastMath.div_rn(rt.mul(picked[r], 2.5), total)
            if lane < 8:
                route_weights[route] = weight
                if bid == 0:
                    RouteIds[route] = route_ids[route]
                    RouteWeights[route] = weight

    @cute.jit
    def select_union(self, p, s, rows):
        tid, warp, lane = s.tid, s.warp, s.lane
        sh = self.shape
        pairs = TOP_K * rows
        rounds = cdiv(pairs + 255, 256)
        for k in range(rounds):
            pair = tid + 256 * k
            if pair < pairs:
                cute.arch.atomic_min(s.first_pair.iterator + s.route_ids[pair], pair)
        rt.bar_sync(8, 256)
        below = (Uint32(1) << Uint32(lane)) - 1
        for k in range(rounds):
            pair = tid + 256 * k
            safe = cutlass.min(pair, pairs - 1)
            leader = (pair < pairs) & (s.first_pair[s.route_ids[safe]] == safe)
            word = Uint32(cute.arch.vote_ballot_sync(leader))
            if lane == 0:
                s.owners[k * 8 + warp] = word
        rt.bar_sync(8, 256)
        total = Int32(0)
        for w in cutlass.range_constexpr(16):
            total = total + Int32(cute.arch.popc(s.owners[w]))
        for k in range(rounds):
            pair = tid + 256 * k
            if pair < pairs:
                expert = s.route_ids[pair]
                if s.first_pair[expert] == pair:
                    before = Int32(0)
                    for w in cutlass.range_constexpr(16):
                        if w < k * 8 + warp:
                            before = before + Int32(cute.arch.popc(s.owners[w]))
                    index = before + Int32(
                        cute.arch.popc(s.owners[k * 8 + warp] & below)
                    )
                    s.union_ids[index] = expert
        if tid == 0:
            s.union_count[0] = total
        rt.Memory.fence_proxy_async_shared()
        rt.bar_sync(8, 256)
        with rt.warp_leader(warp, 0):
            rt.mbar_arrive(s.union_full)
        if s.bid == 0:
            if tid == 0:
                p.Flags[sh.flag_union] = Uint32(total)
            for i in range(tid, total, 256):
                p.Flags[sh.flag_union + 1 + i] = Uint32(s.union_ids[i])

    @cute.jit
    def load(self, p: MoeW8A8Params, s: W8A8State, ids: rt.ThreadIds):
        p, s = p.views(), _bind(s, ids)
        self.set_regs("load")
        with rt.warp_leader(s.warp, 8):
            rt.AsyncCopy.prefetch_tensormap(self.up_map)
            rt.AsyncCopy.prefetch_tensormap(self.down_map)
            rt.AsyncCopy.prefetch_tensormap(self.shared_up_map)
            rt.AsyncCopy.prefetch_tensormap(self.shared_down_map)
        self.epoch_ticket(s.warp, p.FullEpoch.iterator, s.epoch_slot)
        cute.arch.griddepcontrol_wait()
        rows = self.active_rows(p)
        rt.bar_sync(11, THREADS)
        self.router_flags(p, s, self.shape.flag_router)
        self.gather_scores(p, s, rows)
        self.router_flags(p, s, self.shape.flag_image)
        self.gather_scales(p, s)
        self.load_stages(p, s, rows)

    @cute.jit
    def router_flags(self, p, s, first):
        epoch = s.epoch_slot[0]
        self.poll_all(p.Flags, Int32(1), first + s.tid - 256, epoch)
        rt.Memory.fence_acquire_gpu()
        rt.bar_sync(12, 128)

    @cute.jit
    def gather_scores(self, p, s, rows):
        sm, sh = self.Smem, self.shape
        with rt.warp_leader(s.warp, 8):
            nbytes = rows * (EXPERTS * 4)
            rt.Memory.fence_proxy_async_global()
            rt.mbar_expect_tx(s.scores_full, 2 * nbytes)
            rt.bulk_g2s(
                s.ring_addr + sm.LOGITS,
                rt.ga(p.Work, sh.scores // 4),
                nbytes,
                s.scores_full,
            )
            rt.bulk_g2s(
                s.ring_addr + sm.CORRECTED,
                rt.ga(p.Work, sh.corrected // 4),
                nbytes,
                s.scores_full,
            )

    @cute.jit
    def gather_scales(self, p, s):
        with rt.warp_leader(s.warp, 8):
            nbytes = K_BLOCKS * self.RMAX * 4
            rt.Memory.fence_proxy_async_global()
            rt.mbar_expect_tx(s.scales_full, nbytes)
            rt.bulk_g2s(
                s.token_scales_addr,
                rt.ga(p.Work, self.shape.image_scales // 4),
                nbytes,
                s.scales_full,
            )

    @cute.jit
    def load_stages(self, p, s, rows):
        with rt.warp_leader(s.warp, 8):
            self.up_load(p, s, rows)
            self.down_load(p, s, rows)
        with rt.warp_leader(s.warp, 11):
            self.up_mma(p, s)
            self.down_mma(p, s)
        if s.warp == 10:
            self.up_signal(p, s)

    @cute.jit
    def unit_count(self, s):
        units = (s.union_count[0] + 1) * (SLICES * self.shape.ksplit)
        return cdiv(units - s.bid + GRID_CTAS - 1, GRID_CTAS)

    @cute.jit
    def unit_fields(self, s, unit):
        ks = self.shape.ksplit
        slot = cdiv(unit, SLICES * ks)
        slice_ = cmod(cdiv(unit, ks), SLICES)
        part = cmod(unit, ks)
        expert = Int32(LAYOUT.shared_slot)
        if slot > 0:
            expert = s.union_ids[slot - 1]
        return slot, slice_, part, part * (K_BLOCKS // ks), expert

    @cute.jit
    def up_load(self, p, s, rows):
        sm = self.Smem
        ks = self.shape.ksplit
        per = K_BLOCKS // ks
        rt.mbar_wait(s.union_full, 0)
        rt.Memory.fence_proxy_async_global()
        units = self.unit_count(s)
        tile = cdiv(rows + 7, 8) * 1024
        maps = (self.up_map, self.shared_up_map)
        sequence = Int32(0)
        for local in range(units):
            slot, slice_, part, first_kb, expert = self.unit_fields(
                s, s.bid + local * GRID_CTAS
            )
            routed = Int32(slot > 0)
            for i in range(per):
                kb = first_kb + i
                stage = cmod(sequence, sm.STAGES)
                rt.mbar_wait(
                    s.ring_empty + stage * 8, cmod(cdiv(sequence, sm.STAGES), 2) ^ 1
                )
                full = s.ring_full + stage * 8
                dst = s.ring_addr + stage * self.shape.ring_stage
                rt.mbar_expect_tx(full, A_STAGE + tile)
                for half in cutlass.range_constexpr(2):
                    # Gate rows to TMEM lanes 0-63: rows 256.. of the routed w31.
                    row = (routed ^ half) * SHARD + slice_ * SLICE
                    desc, c0, c1, c3 = self.weight_box(
                        maps, kb, row, expert, K_BLOCKS, 2 * SHARD
                    )
                    rt.AsyncCopy.tma_4d(
                        dst + half * (A_STAGE // 2), desc, c0, c1, 0, c3, full, _HINT
                    )
                rt.bulk_g2s(
                    dst + A_STAGE,
                    rt.ga(p.Work, (self.shape.image + kb * self.RMAX * K_BLOCK) // 4),
                    tile,
                    full,
                )
                sequence = sequence + 1

    @cute.jit
    def up_mma(self, p, s):
        sm = self.Smem
        ks = self.shape.ksplit
        per = K_BLOCKS // ks
        q = self.RMAX
        up_column = sm.DOWN_STAGES * 2 * q
        rt.mbar_wait(s.union_full, 0)
        units = self.unit_count(s)
        sequence = Int32(0)
        for local in range(units):
            for i in range(per):
                acc = cmod(sequence, 2)
                stage = cmod(sequence, sm.STAGES)
                rt.mbar_wait(s.acc_empty + acc * 8, cmod(cdiv(sequence, 2), 2) ^ 1)
                rt.mbar_wait(
                    s.ring_full + stage * 8, cmod(cdiv(sequence, sm.STAGES), 2)
                )
                rt.TensorCore.fence_after_thread_sync()
                base = s.ring_addr + stage * self.shape.ring_stage
                rt.Umma.mma_k128(
                    rt.Umma.operand_low(base),
                    rt.Umma.operand_low(base + A_STAGE),
                    s.taddr + up_column + acc * q,
                    q,
                    0,
                )
                rt.TensorCore.tcgen05_commit(s.ring_empty + stage * 8)
                rt.TensorCore.tcgen05_commit(s.acc_full + acc * 8)
                sequence = sequence + 1
        rt.TensorCore.tcgen05_commit(s.up_done)

    @cute.jit
    def weight_scale(self, p, s, unit, entry):
        per = K_BLOCKS // self.shape.ksplit
        slot, slice_, part, first_kb, expert = self.unit_fields(s, unit)
        half = cdiv(entry, per) ^ Int32(slot > 0)  # routed w31: up blocks first
        scale_row = cdiv(half * SHARD + slice_ * SLICE, K_BLOCK)
        word = scale_row * K_BLOCKS + first_kb + cmod(entry, per)
        routed = Int64(expert) * LAYOUT.w13_scale_words + word
        return self.block_scale(
            slot, rt.ga(p.RoutedUpScales, routed), rt.ga(p.SharedScales, word)
        )

    @cute.jit
    def block_scale(self, slot, routed, shared):
        address = cutlass.select_(slot > 0, routed, shared)
        return rt.Convert.bits_f32(rt.Memory.ldg_nc_u32(address))

    @cute.jit
    def up_promote(self, p, s, units):
        sm = self.Smem
        sh = self.shape
        q, ks = self.RMAX, sh.ksplit
        per = K_BLOCKS // ks
        entries = 2 * per
        row = s.tid
        half = cdiv(row, SLICE)
        lane_base = (cmod(s.warp, 4) * 32) << 16
        up_column = sm.DOWN_STAGES * 2 * q
        upcoming = Float32(0.0)
        if (row < entries) & (units > 0):
            upcoming = self.weight_scale(p, s, s.bid, row)
        sequence = Int32(0)
        rt.mbar_wait(s.scales_full, 0)
        for local in range(units):
            unit = s.bid + local * GRID_CTAS
            slot, slice_, part, first_kb, expert = self.unit_fields(s, unit)
            if row < entries:
                s.up_scales[cmod(local, 2) * entries + row] = upcoming
            rt.bar_sync(9, 128)
            if (row < entries) & (local + 1 < units):
                upcoming = self.weight_scale(p, s, unit + GRID_CTAS, row)
            total = rt.zeros(q)
            for i in range(per):
                acc = cmod(sequence, 2)
                weight = s.up_scales[cmod(local, 2) * entries + half * per + i]
                rt.mbar_wait(s.acc_full + acc * 8, cmod(cdiv(sequence, 2), 2))
                rt.TensorCore.fence_after_thread_sync()
                sums = rt.Umma.tmem_ld_columns(
                    s.taddr + lane_base + up_column + acc * q, q
                )
                rt.Umma.wait_ld()
                rt.TensorCore.fence_before_thread_sync()
                if s.lane == 0:
                    rt.mbar_arrive(s.acc_empty + acc * 8)
                kb = first_kb + i
                for t in cutlass.range_constexpr(q):
                    total[t] = rt.fma(
                        sums[t], rt.mul(weight, s.token_scales[kb * q + t]), total[t]
                    )
                sequence = sequence + 1
            base = (
                ((slot * ks + part) * 2 + half) * SHARD
                + slice_ * SLICE
                + cmod(row, SLICE)
            ) * q
            for chunk in cutlass.range_constexpr(q // ROW_CHUNK):
                if chunk * ROW_CHUNK < self.active_rows(p):
                    for quad in cutlass.range_constexpr(ROW_CHUNK // 4):
                        rt.Memory.stg_u32x4(
                            p.Work,
                            sh.up_partials // 4 + base + chunk * ROW_CHUNK + 4 * quad,
                            [
                                rt.Convert.f32_bits(
                                    total[chunk * ROW_CHUNK + 4 * quad + j]
                                )
                                for j in range(4)
                            ],
                        )
            rt.bar_sync(SIGNAL_BARRIER, 128 + 32)

    @cute.jit
    def up_signal(self, p, s):
        sh = self.shape
        rt.mbar_wait(s.union_full, 0)
        units = self.unit_count(s)
        for local in range(units):
            slot, slice_, part, first_kb, expert = self.unit_fields(
                s, s.bid + local * GRID_CTAS
            )
            rt.bar_sync(SIGNAL_BARRIER, 128 + 32)
            if s.lane == 0:
                rt.Memory.fence_release_gpu()
                rt.Memory.atomic_add_u32(
                    p.Flags, sh.flag_counter + 2 * slot + cdiv(slice_, 2), 1
                )

    @cute.jit
    def finalize(self, p, s, rows, epoch):
        sh = self.shape
        jobs = 2 * (s.union_count[0] + 1)
        for job in range(s.bid, jobs, GRID_CTAS):
            if s.warp == 4:
                done = Int32(0)
                while done == 0:
                    seen = rt.Memory.ld_relaxed_gpu(
                        rt.ga(p.Flags, sh.flag_counter + job)
                    )
                    done = Int32(cute.arch.vote_all_sync(seen == Uint32(2 * sh.ksplit)))
                rt.Memory.fence_acquire_gpu()
                if s.lane == 0:
                    p.Flags[sh.flag_counter + job] = Uint32(0)
            rt.bar_sync(10, 128)
            self.finalize_block(p, s, rows, cdiv(job, 2), cmod(job, 2), epoch)

    @cute.jit
    def route_weight(self, s, slot, token):
        expert = s.union_ids[cutlass.max(slot - 1, 0)]
        weight = Float32(0.0)
        for k in cutlass.range_constexpr(TOP_K):
            weight = rt.add(
                weight,
                cutlass.select_(
                    s.route_ids[token * TOP_K + k] == expert,
                    s.route_weights[token * TOP_K + k],
                    Float32(0.0),
                ),
            )
        return cutlass.select_(slot == 0, Float32(1.0), weight)

    @cute.jit
    def finalize_block(self, p, s, rows, slot, block, epoch):
        sh = self.shape
        q, ks = self.RMAX, sh.ksplit
        t = s.tid - 128
        warp4 = s.warp - 4
        row = block * K_BLOCK + t
        for chunk in range(cdiv(rows + FIN_TOKENS - 1, FIN_TOKENS)):
            first = chunk * FIN_TOKENS
            gate = [Float32(0.0)] * FIN_TOKENS
            up = [Float32(0.0)] * FIN_TOKENS
            for part in cutlass.range_constexpr(ks):
                for half in cutlass.range_constexpr(2):
                    base = (((slot * ks + part) * 2 + half) * SHARD + row) * q + first
                    for quad in cutlass.range_constexpr(FIN_TOKENS // 4):
                        v = rt.Memory.ld_global_v4_f32(
                            rt.ga(p.Work, sh.up_partials // 4 + base + 4 * quad)
                        )
                        for j in cutlass.range_constexpr(4):
                            if half == 0:
                                gate[4 * quad + j] = rt.add(gate[4 * quad + j], v[j])
                            else:
                                up[4 * quad + j] = rt.add(up[4 * quad + j], v[j])
            h = []
            for token in cutlass.range_constexpr(FIN_TOKENS):
                g = gate[token]
                silu = rt.mul(
                    g,
                    rt.FastMath.rcp_rn(
                        rt.add(1.0, cute.math.exp2(rt.mul(-g, LOG2E), fastmath=True))
                    ),
                )
                h.append(rt.mul(silu, up[token]))
            for token in cutlass.range_constexpr(FIN_TOKENS):
                peak = rt.FastMath.abs_ftz(h[token])
                for bit in cutlass.range_constexpr(5):
                    peak = rt.FastMath.fmax(
                        peak, cute.arch.shuffle_sync_bfly(peak, 1 << bit)
                    )
                if s.lane == 0:
                    s.fin_amax[warp4 * FIN_TOKENS + token] = peak
            rt.bar_sync(10, 128)
            if t < FIN_TOKENS:
                peak = rt.FastMath.fmax(
                    rt.FastMath.fmax(s.fin_amax[t], s.fin_amax[FIN_TOKENS + t]),
                    rt.FastMath.fmax(
                        s.fin_amax[2 * FIN_TOKENS + t], s.fin_amax[3 * FIN_TOKENS + t]
                    ),
                )
                scale = rt.FastMath.div_rn(rt.FastMath.fmax(peak, QUANT_EPS), FP8_MAX)
                s.fin_scale[t] = scale
                if first + t < rows:
                    p.Work[sh.token_scale // 4 + (slot * q + first + t) * 2 + block] = (
                        rt.Convert.f32_bits(
                            rt.mul(self.route_weight(s, slot, first + t), scale)
                        )
                    )
            rt.bar_sync(10, 128)
            piece = cdiv(t, 16)
            for token in cutlass.range_constexpr(FIN_TOKENS):
                value = rt.FastMath.div_rn(h[token], s.fin_scale[token])
                second = cute.arch.shuffle_sync_bfly(value, 1)
                third = cute.arch.shuffle_sync_bfly(value, 2)
                fourth = cute.arch.shuffle_sync_bfly(value, 3)
                if cmod(t, 4) == 0:
                    # 8-token tiles of the image: tile ``2 chunk + token / 8``
                    byte = (
                        sh.hq
                        + (slot * 2 + block) * q * K_BLOCK
                        + (chunk * (FIN_TOKENS // 8) + token // 8) * 1024
                        + (token % 8) * 128
                        + ((piece ^ (token % 8)) * 16)
                        + cmod(t, 16)
                    )
                    p.Work[cdiv(byte, 4)] = rt.Convert.e4m3x4(
                        value, second, third, fourth
                    )
            rt.bar_sync(10, 128)
        rt.Memory.fence_proxy_async_global()
        rt.bar_sync(10, 128)
        if t == 0:
            rt.Memory.fence_release_gpu()
            rt.Memory.st_relaxed_sys(
                rt.ga(p.Flags, sh.flag_ready + 2 * slot + block), epoch + 1
            )

    @cute.jit
    def down_slots(self, s):
        group = cmod(s.bid, DOWN_GROUPS)
        last = s.union_count[0]
        count = cutlass.select_(
            (group <= last) & (s.bid < DOWN_CTAS),
            cdiv(last - group, DOWN_GROUPS) + 1,
            0,
        )
        return group, cdiv(s.bid, DOWN_GROUPS), count

    @cute.jit
    def slot_expert(self, s, slot):
        expert = Int32(LAYOUT.shared_slot)
        if slot > 0:
            expert = s.union_ids[slot - 1]
        return expert

    @cute.jit
    def down_load(self, p, s, rows):
        sm, sh = self.Smem, self.shape
        q = self.RMAX
        group, block, count = self.down_slots(s)
        epoch = s.epoch_slot[0]
        tile = cdiv(rows + 7, 8) * 1024
        scale_bytes = cdiv(rows + 1, 2) * 16
        maps = (self.down_map, self.shared_down_map)
        rt.mbar_wait(s.up_done, 0)
        for j in range(count):
            slot = group + DOWN_GROUPS * j
            stage = cmod(j, sm.DOWN_STAGES)
            rt.mbar_wait(s.down_free + stage * 8, cmod(cdiv(j, sm.DOWN_STAGES), 2) ^ 1)
            full = s.down_full + stage * 8
            dst = s.ring_addr + stage * sh.down_stage
            rt.mbar_expect_tx(full, 2 * A_STAGE + 2 * tile + scale_bytes)
            expert = self.slot_expert(s, slot)
            for kb in cutlass.range_constexpr(2):
                desc, c0, c1, c3 = self.weight_box(
                    maps, kb, block * DOWN_ROW_BLOCK, expert, 2, HIDDEN
                )
                rt.AsyncCopy.tma_4d(
                    dst + kb * A_STAGE, desc, c0, c1, 0, c3, full, _HINT
                )
            waiting = Int32(1)
            while waiting != 0:
                first = rt.Memory.ld_relaxed_gpu(
                    rt.ga(p.Flags, sh.flag_ready + 2 * slot)
                )
                second = rt.Memory.ld_relaxed_gpu(
                    rt.ga(p.Flags, sh.flag_ready + 2 * slot + 1)
                )
                if (first >= epoch + 1) & (second >= epoch + 1):
                    waiting = Int32(0)
            rt.Memory.fence_acquire_gpu()
            rt.Memory.fence_proxy_async_global()
            for kb in cutlass.range_constexpr(2):
                rt.bulk_g2s(
                    dst + 2 * A_STAGE + kb * q * K_BLOCK,
                    rt.ga(p.Work, (sh.hq + (slot * 2 + kb) * q * K_BLOCK) // 4),
                    tile,
                    full,
                )
            rt.bulk_g2s(
                dst + sh.down_scales,
                rt.ga(p.Work, (sh.token_scale + slot * q * 8) // 4),
                scale_bytes,
                full,
            )

    @cute.jit
    def down_mma(self, p, s):
        sm, sh = self.Smem, self.shape
        q = self.RMAX
        _, _, count = self.down_slots(s)
        for j in range(count):
            stage = cmod(j, sm.DOWN_STAGES)
            phase = cmod(cdiv(j, sm.DOWN_STAGES), 2)
            rt.mbar_wait(s.down_full + stage * 8, phase)
            rt.mbar_wait(s.down_free + stage * 8, phase ^ 1)
            rt.TensorCore.fence_after_thread_sync()
            base = s.ring_addr + stage * sh.down_stage
            for kb in cutlass.range_constexpr(2):
                rt.Umma.mma_k128(
                    rt.Umma.operand_low(base + kb * A_STAGE),
                    rt.Umma.operand_low(base + 2 * A_STAGE + kb * q * K_BLOCK),
                    s.taddr + (stage * 2 + kb) * q,
                    q,
                    0,
                )
            rt.TensorCore.tcgen05_commit(s.down_ready + stage * 8)

    @cute.jit
    def down_scale(self, p, s, slot, block, kb):
        word = block * 2 + kb
        routed = Int64(self.slot_expert(s, slot)) * LAYOUT.w2_scale_words + word
        shared = LAYOUT.w13_scale_words + word
        return self.block_scale(
            slot, rt.ga(p.RoutedDownScales, routed), rt.ga(p.SharedScales, shared)
        )

    @cute.jit
    def down_promote(self, p, s, rows):
        sm, sh = self.Smem, self.shape
        q = self.RMAX
        chunk = 32 if q % 32 == 0 else 16
        group, block, count = self.down_slots(s)
        row = s.tid
        lane_base = (cmod(s.warp, 4) * 32) << 16
        total = rt.zeros(q)
        scale0, scale1 = Float32(0.0), Float32(0.0)
        if count > 0:
            scale0 = self.down_scale(p, s, group, block, 0)
            scale1 = self.down_scale(p, s, group, block, 1)
        for j in range(count):
            stage = cmod(j, sm.DOWN_STAGES)
            rt.mbar_wait(s.down_ready + stage * 8, cmod(cdiv(j, sm.DOWN_STAGES), 2))
            rt.TensorCore.fence_after_thread_sync()
            tokens = s.ring_addr + stage * sh.down_stage + sh.down_scales
            for c in cutlass.range_constexpr(q // chunk):
                column = s.taddr + lane_base + stage * 2 * q + c * chunk
                first = rt.Umma.tmem_ld_columns(column, chunk)
                second = rt.Umma.tmem_ld_columns(column + q, chunk)
                rt.Umma.wait_ld()
                for t in cutlass.range_constexpr(chunk):
                    index = c * chunk + t
                    total[index] = rt.fma(
                        first[t],
                        rt.mul(scale0, rt.Memory.lds(tokens + index * 8)),
                        total[index],
                    )
                    total[index] = rt.fma(
                        second[t],
                        rt.mul(scale1, rt.Memory.lds(tokens + index * 8 + 4)),
                        total[index],
                    )
            rt.TensorCore.fence_before_thread_sync()
            cute.arch.sync_warp()
            if s.lane == 0:
                rt.mbar_arrive(s.down_free + stage * 8)
            if j + 1 < count:
                scale0 = self.down_scale(p, s, group + DOWN_GROUPS * (j + 1), block, 0)
                scale1 = self.down_scale(p, s, group + DOWN_GROUPS * (j + 1), block, 1)
        if s.bid < DOWN_CTAS:
            for t in cutlass.range_constexpr(q):
                if t < rows:
                    p.Work[
                        sh.down_partials // 4
                        + (group * q + t) * HIDDEN
                        + block * DOWN_ROW_BLOCK
                        + row
                    ] = rt.Convert.f32_bits(total[t])
            rt.bar_sync(9, 128)
            if row == 0:
                rt.Memory.fence_release_gpu()
                rt.Memory.atomic_add_u32(p.Flags, sh.flag_down + block, 1)

    @cute.jit
    def owner(self, p, s, rows, epoch):
        sh = self.shape
        first = s.bid * OWNER_ROWS
        if s.warp == 0:
            target = Uint32(DOWN_GROUPS) * (epoch + 1)
            waiting = Int32(1)
            while waiting != 0:
                low = rt.Memory.ld_relaxed_gpu(
                    rt.ga(p.Flags, sh.flag_down + cdiv(first, DOWN_ROW_BLOCK))
                )
                high = rt.Memory.ld_relaxed_gpu(
                    rt.ga(
                        p.Flags,
                        sh.flag_down + cdiv(first + OWNER_ROWS - 1, DOWN_ROW_BLOCK),
                    )
                )
                waiting = Int32(
                    cute.arch.vote_all_sync((low >= target) & (high >= target)) == 0
                )
            rt.Memory.fence_acquire_gpu()
        rt.bar_sync(8, 256)
        for index in range(s.tid, rows * OWNER_ROWS, 256):
            token, column = cdiv(index, OWNER_ROWS), cmod(index, OWNER_ROWS)
            total = Float32(0.0)
            for g in cutlass.range_constexpr(DOWN_GROUPS):
                part = rt.Convert.bits_f32(
                    rt.Memory.ld_cg_b32(
                        rt.ga(
                            p.Work,
                            sh.down_partials // 4
                            + (g * self.RMAX + token) * HIDDEN
                            + first
                            + column,
                        )
                    )
                )
                total = part if g == 0 else rt.add(total, part)
            value = total.to(BFloat16)
            s.local_o[index] = value
            p.LocalO[token * HIDDEN + first + column] = value
        rt.bar_sync(8, 256)
        self.exchange_slice(p, s, rows, epoch)

    @cute.jit
    def exchange_slice(self, p, s, rows, epoch):
        self.exchange.push_peers(
            p, s.local_o_words, epoch, s.bid, s.tid, s.warp, s.lane
        )
        if cutlass.const_expr(self.SEAM):  # noqa: SIM102 (nested: const_expr)
            if s.warp == 7:  # the own partial goes to the own slot
                peer = rt.Memory.select_peer(p.peer_rank, p.peer_bases)
                for group in cutlass.range_constexpr(self.exchange.groups):
                    self.exchange.push(
                        s.local_o_words,
                        epoch,
                        s.bid,
                        s.lane,
                        peer,
                        self.exchange.OWN,
                        group,
                    )
        for turn in cutlass.range_constexpr(-(-self.exchange.groups // 8)):
            group = s.warp + 8 * turn
            if group * rt.SliceExchange.GROUP_TOKENS < rows:
                self.exchange.combine(
                    p,
                    p.X,
                    s.lane,
                    s.bid,
                    epoch,
                    None if self.SEAM else s.local_o_words,
                    group=group,
                    live=rows,
                )


class FusedMoeW8A8Seam(FusedMoeW8A8):
    """The row-tier MoE of a seam layer: also writes ``XG = Final x GammaNext``."""

    SEAM = True
    Params = MoeW8A8SeamParams  # type: ignore[assignment]
