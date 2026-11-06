# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 fused MoE decode kernels in CuTe DSL."""

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, Uint64

from . import _rt as rt
from .moe_q_shape import Flag, Q8Flag, Q8Shape, flag_layout  # noqa: F401

cdiv, cmod = rt.cdiv, rt.cmod

ROUTED_CTAS = 128
COLUMNS = 48
NORM_ROW = 6152
EXPERTS = 256
TOP_K = 8
SHARD = 256
COMPUTE_THREADS = 256
WARP = 32
RING_SLOTS = 4
SCRATCH_WORDS = 2048
LOG2E = 1.4426950408889634
FLT_MAX = 3.4028234663852886e38
_HINT = 0x12F0000000000000


@dataclass(frozen=True)
class ExpertLayout:
    """Shapes of one rank's experts as the MoE kernels read them: vLLM's tensors in
    place, where vLLM allocated them, plus a small header of our own.
    """

    routed_experts: int = EXPERTS
    hidden: int = rt.HIDDEN
    shard: int = SHARD
    block: int = 128
    prefetch_line: int = 128

    @property
    def slots(self) -> int:
        return self.routed_experts + 1

    @property
    def shared_slot(self) -> int:
        return self.routed_experts

    @property
    def w13_shape(self) -> tuple[int, int]:
        return (2 * self.shard, self.hidden)

    @property
    def w2_shape(self) -> tuple[int, int]:
        return (self.hidden, self.shard)

    @property
    def w13_block_shape(self) -> tuple[int, int, int]:
        return (self.hidden // self.block, 2 * self.shard, self.block)

    @property
    def w2_block_shape(self) -> tuple[int, int, int]:
        return (self.shard // self.block, self.hidden, self.block)

    @property
    def w13_scale_shape(self) -> tuple[int, int]:
        return (2 * self.shard // self.block, self.hidden // self.block)

    @property
    def w2_scale_shape(self) -> tuple[int, int]:
        return (self.hidden // self.block, self.shard // self.block)

    @property
    def w13_expert_bytes(self) -> int:
        return self.w13_shape[0] * self.w13_shape[1]

    @property
    def w2_expert_bytes(self) -> int:
        return self.w2_shape[0] * self.w2_shape[1]

    @property
    def w13_scale_words(self) -> int:
        return self.w13_scale_shape[0] * self.w13_scale_shape[1]

    @property
    def w2_scale_words(self) -> int:
        return self.w2_scale_shape[0] * self.w2_scale_shape[1]

    @property
    def header_words(self) -> int:
        return self.w13_scale_words + self.w2_scale_words

    @property
    def w13_expert_lines(self) -> int:
        return self.w13_expert_bytes // self.prefetch_line


#: The expert layout the MoE kernels and the worker's MoE prefetch read.
LAYOUT = ExpertLayout()


class UpPage:
    """A gate/up ring page: two TMA boxes (gate, up), then their FP32 scale blocks."""

    #: One box: 16 rows by 4 chunks of 256 K.
    BOX_ROWS, BOX_CHUNKS = 16, 4
    BOX = 16384
    SCALES = 2 * BOX
    TX = SCALES + 64


class DownPlane:
    """Down records in the ring: 48 rows x 256 K E4M3 per expert, 1 KiB-aligned, then
    scales.
    """

    BOX = 6144
    RECORD = 2 * BOX
    TX = RECORD + 32
    SCALES = 2 * RECORD

    @staticmethod
    def pad(base, off):
        return 1024 - (base + off & 1023) & 1023

    @staticmethod
    def scales(bid):
        return cdiv(cdiv(bid * COLUMNS, 128), 2) * 4

    @staticmethod
    def lane_word(warp, lane):
        group = cdiv(lane, 4)
        chunk = cmod(warp, 4) + 4 * (cmod(cdiv(lane, 2), 2)) ^ group
        return cdiv(cdiv(warp, 4) * DownPlane.BOX + group * 128 + chunk * 16, 8) + cmod(
            lane, 2
        )

    @staticmethod
    def rows(ring_u64, byte_base, warp, lane) -> list:
        first = (byte_base >> 3) + DownPlane.lane_word(warp, lane)
        return [ring_u64[first + 128 * r] for r in range(6)]

    @staticmethod
    def k_words(warp, lane):
        return (
            cdiv(warp, 4) * 64
            + cmod(cdiv(warp, 2), 2) * 32
            + cmod(cdiv(lane, 2), 2) * 16
            + cmod(warp, 2) * 8
            + cmod(lane, 2) * 2
        )

    @staticmethod
    def scale_word(block, bid, warp, k):
        return (
            block
            + (
                cdiv(bid * COLUMNS + 16 * k, 128)
                - cdiv(cdiv(bid * COLUMNS, 128), 2) * 2
            )
            * 2
            + cdiv(warp, 4)
        )

    @staticmethod
    def accum(total, part, scales, scale0, weights, route):
        for j in range(3):
            first = rt.mul(scales[scale0 + j], weights[route[0]])
            second = rt.mul(scales[scale0 + j], weights[route[1]])
            for q in range(2):
                total[j * 4 + 2 * q] = rt.fma(
                    part[j * 4 + 2 * q], first, total[j * 4 + 2 * q]
                )
                total[j * 4 + 2 * q + 1] = rt.fma(
                    part[j * 4 + 2 * q + 1], second, total[j * 4 + 2 * q + 1]
                )


class ActPlane:
    """Activation planes (bf16): plane 0 the shared expert, 1 + i routed slot i;
    swizzled columns.
    """

    ROW_BYTES = 528
    ROW_WORDS = ROW_BYTES // 4

    @staticmethod
    def col(e):
        return (e & 147) | (e & 64) >> 1 | (e & 32) << 1 | (e & 8) >> 1 | (e & 4) << 1


class Rms:
    """RMSNorm pieces."""

    @staticmethod
    def gamma_load(Gamma, tid) -> list:
        cp16 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Uint32, num_bits_per_copy=128
        )
        gamma_u32 = cute.recast_tensor(Gamma, Uint32)
        raws = []
        for g in range(3):
            raw = cute.make_rmem_tensor(4, Uint32)
            cute.copy(cp16, cute.local_tile(gamma_u32, (4,), (g * 256 + tid,)), raw)
            raws.append(raw)
        return raws

    @staticmethod
    def store(addr, values, inverse, gamma) -> None:
        rt.Memory.sts_v4(addr, rt.Reduce.norm_bf16x8(values, inverse, gamma))

    @staticmethod
    def inverses(scratch, rows: int = 4) -> list:
        totals = [
            rt.FastMath.div_rn(rt.Reduce.strided_sum(scratch, t * 8, 1), 6144.0)
            for t in range(rows)
        ]
        return [rt.FastMath.rsqrt(rt.add(x, 1e-05)) for x in totals]


@rt.record
class MoeParams(rt.Params):
    """The kernel's sole parameter."""

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
    Activation: cute.Pointer = rt.ptr("bfloat16")
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


@rt.record
class MoeSeamParams(MoeParams):
    """Seam parameters: also the next layer's RMSNorm weight and its
    gamma-scaled input.
    """

    GammaNext: cute.Pointer = rt.ptr("float32")
    XG: cute.Pointer = rt.ptr("bfloat16")


@rt.record
class MoeQ8Params(MoeParams):
    """q8 parameters: also the step's row table, whose word 0 is the live
    row count R.
    """

    RowTable: cute.Pointer = rt.ptr("int32", align=4)


@rt.record
class MoeQ8SeamParams(MoeSeamParams):
    """q8 seam parameters: the seam's, then the step's row table."""

    RowTable: cute.Pointer = rt.ptr("int32", align=4)


@dataclass(frozen=True)
class MoeState:
    """One CTA's state: thread ids, the shared-memory base and the weight ring."""

    base: Int32
    pages: rt.Pipeline.Ring
    tid: Int32 = None
    warp: Int32 = None
    lane: Int32 = None
    bid: Int32 = None

    if TYPE_CHECKING:
        # The shared-memory record's views (rt.SmemLayout.record).
        def __getattr__(self, name: str) -> Any: ...


def _bind(s: MoeState, ids: rt.ThreadIds) -> MoeState:
    return replace(s, tid=ids.tid, warp=ids.warp, lane=ids.lane, bid=ids.bid)


class MoeRouting:
    """Flag handshakes and the router's top-8 selection shared by the MoE kernel
    families (the q4 builds and the W8A8 row tiers).
    """

    DEAD_STRIDE = 0
    DEAD_LIVE = 0

    @cute.jit
    def active_rows(self, p):
        return cutlass.min(cutlass.max(Int32(p.RowTable[0]), 1), self.RMAX)  # type: ignore[attr-defined]

    @cute.jit
    def elect_add(
        self,
        Flags,
        warp,
        index,
        scoped: cutlass.Constexpr = True,
    ):
        with rt.warp_leader(warp, 0):
            if cutlass.const_expr(scoped):
                rt.Memory.atomic_add_u32(Flags, index, 1)
            else:
                cute.arch.atomic_add(Flags.iterator + index, Uint32(1))

    @cute.jit
    def poll_all(self, Flags, owned, index, epoch):
        done = Int32(0)
        while done == 0:
            satisfied = Int32(1)
            if owned != 0:
                observed = rt.Memory.ld_relaxed_gpu(rt.ga(Flags, index))
                satisfied = satisfied & Int32(observed >= epoch + 1)
            done = Int32(cute.arch.vote_all_sync(satisfied != 0))

    @cute.jit
    def store_finite(self, dst, idx, value):
        if rt.FastMath.fcmp(">=", value, -FLT_MAX) & rt.FastMath.fcmp(
            "<=", value, FLT_MAX
        ):
            dst[idx] = value
        else:
            dst[idx] = -FLT_MAX

    @cute.jit
    def route_token(self, views, base, lane, warp, bid, route_full_addr, row):
        corrected, scores, route_ids, route_weights, RouteIds, RouteWeights = views
        self.top8(corrected, base, lane, warp, bid, route_ids, route_full_addr, row)
        cute.arch.sync_warp()
        route = row * 8 + cmod(lane, 8)
        picked = scores[base + route_ids[route]]
        selected = Float32(0.0)
        if lane < 8:
            selected = picked
        total = rt.Reduce.fold([selected], lane, 16, (16, 8, 4, 2, 1))
        if rt.FastMath.fcmp("<=", total, 0.0):
            total = Float32(1.0)
        weight = rt.FastMath.div_rn(rt.mul(picked, 2.5), total)
        if lane < 8:
            route_weights[route] = weight
            if bid == 0:
                RouteIds[route] = route_ids[route]
                RouteWeights[route] = weight

    @cute.jit
    def top8_candidates(self, corrected, base, lane):
        local = cute.make_rmem_tensor(8, Float32)
        for item in cutlass.range_constexpr(8):
            local[item] = corrected[base + lane + 32 * item]
        return local

    @cute.jit
    def top8(self, corrected, base, lane, warp, bid, route_ids, route_full_addr, row):
        local = self.top8_candidates(corrected, base, lane)
        for slot in cutlass.range_constexpr(8):
            max_lo = rt.FastMath.max3(local[0], local[1], local[2])
            max_mid = rt.FastMath.max3(local[3], local[4], local[5])
            max_hi = rt.FastMath.fmax(local[6], local[7])
            best = rt.Sync.redux_max_nan_f32(rt.FastMath.max3(max_lo, max_mid, max_hi))
            owned = Uint32(4294967295)
            for j in cutlass.range_constexpr(8):
                if rt.FastMath.fcmp("==", local[7 - j], best):
                    owned = Uint32(lane + 32 * (7 - j))
            best_id = Int32(rt.Sync.redux_min_u32(owned))
            for item in cutlass.range_constexpr(8):
                if best_id == lane + 32 * item:
                    local[item] = -float("inf")
            if lane == 0:
                route_ids[row * 8 + slot] = best_id
            if cutlass.const_expr(route_full_addr is not None):  # noqa: SIM102  nested bare-bool ifs: the DSL traces them
                if (warp == 0) & (cdiv(bid, 16) == slot):
                    with rt.warp_leader(warp, 0):
                        route_ids[slot] = best_id
                        rt.mbar_arrive(route_full_addr)


class FusedMoe(MoeRouting, rt.UpDownMapKernel):
    """Stages shared by every variant; subclasses override the differing ones."""

    ROWS: int
    RING_SLOTS = RING_SLOTS
    Flag = Flag
    SEAM = False
    Params = MoeParams
    shared_up_map = shared_down_map = None
    GRID = 144
    ROLES = (rt.Role("compute", 0, 7, 216, True), rt.Role("load", 8, 11, 72, False))
    Smem: type[rt.SmemLayout]
    SLOT_BYTES: int
    PARTIAL_STRIDE: int
    EARLY_RELEASE = False

    @property
    def mailbox(self) -> rt.SliceExchange:
        return rt.SliceExchange(self.ROWS, self.SEAM)

    def mbarriers(self) -> tuple:
        return self.Smem.barriers()

    def tensor_maps(self, p) -> tuple:
        hidden, block, rows13 = rt.HIDDEN, LAYOUT.block, 2 * SHARD
        routed = LAYOUT.routed_experts
        return (
            rt.byte_map(
                p.RoutedUp.toint(),
                [block, hidden // block * routed, rows13 // 8, 8],
                [rows13 * block, 1024, block],
                [block, UpPage.BOX_CHUNKS * 2, 2, 8],
            ),
            rt.byte_map(
                p.RoutedDown.toint(),
                [block, 2, 8, hidden // 16 * (SHARD // block) * routed],
                [1024, block, 16 * block],
                [block, 2, 8, COLUMNS // 16],
                True,
            ),
            rt.byte_map(
                p.SharedUp.toint(),
                [256, hidden // 256, rows13, 1],
                [256, hidden, LAYOUT.w13_expert_bytes],
                [256, UpPage.BOX_CHUNKS, UpPage.BOX_ROWS, 1],
            ),
            rt.byte_map(
                p.SharedDown.toint(),
                [SHARD, hidden, 1],
                [SHARD, LAYOUT.w2_expert_bytes],
                [128, COLUMNS, 1],
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

    def smem(self, base):
        return self.Smem.record(base, ring="pages", bases=(MoeState,))

    @cute.jit
    def compute(self, p: MoeParams, s: MoeState, ids: rt.ThreadIds):
        self.compute_cta(p.views(), _bind(s, ids))

    @cute.jit
    def load(self, p: MoeParams, s: MoeState, ids: rt.ThreadIds):
        p, s = p.views(), _bind(s, ids)
        self.set_regs("load")
        with rt.warp_leader(s.warp, 8):
            rt.AsyncCopy.prefetch_tensormap(self.up_map)
            rt.AsyncCopy.prefetch_tensormap(self.down_map)
            rt.AsyncCopy.prefetch_tensormap(self.shared_up_map)
            rt.AsyncCopy.prefetch_tensormap(self.shared_down_map)
        self.epoch_ticket(s.warp, p.FullEpoch.iterator, s.epoch_slot)
        self.load_prologue(p, s)
        lane_0 = s.tid - 256
        rt.bar_sync(11, 384)
        epoch_ld = s.epoch_slot[0]
        self.load_cta(p, s, epoch_ld, lane_0)

    @abstractmethod
    def compute_cta(self, p: MoeParams, s: MoeState):
        raise NotImplementedError

    @abstractmethod
    def load_prologue(self, p: MoeParams, s: MoeState):
        raise NotImplementedError

    @abstractmethod
    def load_cta(self, p: MoeParams, s: MoeState, epoch_ld, lane_0):
        raise NotImplementedError

    @abstractmethod
    def shared_up_page(self, p: MoeParams, s: MoeState, page):
        raise NotImplementedError

    @cute.jit
    def shared_up_pages(
        self,
        p: MoeParams,
        s: MoeState,
        first: cutlass.Constexpr,
        last: cutlass.Constexpr,
    ):
        for page in range(first, last):
            self.shared_up_page(p, s, page)

    @cute.jit
    def up_pages(self, p: MoeParams, s: MoeState, first, expert, row16):
        for page in range(0, 6):
            self.up_page(p, s, first + page, page, expert, row16)

    @cute.jit
    def rms_norm(self, p: MoeParams, s: MoeState, gammas):
        rows = self.ROWS
        tid, warp, lane = s.tid, s.warp, s.lane
        values = cute.make_rmem_tensor(24 * rows, Float32)
        squares = cute.make_rmem_tensor(rows, Float32)
        for token in cutlass.range_constexpr(rows):
            squares[token] = Float32(0.0)
            for part in cutlass.range_constexpr(3):
                row_word = cdiv(token * NORM_ROW + part * 2048 + tid * 8, 2)
                words_f32 = rt.Convert.ld_bf16x8_words(s.norm_words, row_word)
                for item in cutlass.range_constexpr(8):
                    values[token * 24 + part * 8 + item] = words_f32[item]
                    squares[token] = rt.fma(
                        words_f32[item], words_f32[item], squares[token]
                    )
        last = rt.Reduce.fold([squares[t] for t in range(4)], lane, 16, (4, 2, 1))
        if (lane & 7) == 0:
            s.scratch[((lane >> 3) & 3) * 8 + warp] = last
        rt.bar_sync(8, 256)
        inverses = Rms.inverses(s.scratch, rows)
        for token in cutlass.range_constexpr(rows):
            for part in cutlass.range_constexpr(3):
                row_value = token * 24 + part * 8
                norm_dst = s.norm_addr + (token * NORM_ROW + part * 2048 + tid * 8) * 2
                row_values = [values[row_value + i] for i in range(8)]
                Rms.store(norm_dst, row_values, inverses[token], gammas[part])
        rt.bar_sync(8, 256)

    @cute.jit
    def score_and_route(self, p: MoeParams, s: MoeState, early_bias):
        tid = s.tid
        denominators = cute.make_rmem_tensor(self.ROWS, Float32)
        for token in cutlass.range_constexpr(self.ROWS):
            denominators[token] = rt.add(
                1.0,
                cute.math.exp2(
                    rt.mul(-s.logits[token * 256 + tid], LOG2E), fastmath=True
                ),
            )
        s.first_lane[tid] = 32
        for token in cutlass.range_constexpr(self.ROWS):
            score = rt.FastMath.rcp_rn(denominators[token])
            s.scores[token * 256 + tid] = score
            self.store_finite(s.corrected, token * 256 + tid, rt.add(score, early_bias))
        rt.bar_sync(8, 256)
        if s.warp < self.ROWS:
            routes = (
                s.corrected,
                s.scores,
                s.route_ids,
                s.route_weights,
                p.RouteIds,
                p.RouteWeights,
            )
            self.route_token(
                routes, s.warp * 256, s.lane, s.warp, s.bid, s.route_full, s.warp
            )
        if tid < 128:
            s.union_route[tid] = 0.0
        rt.bar_sync(8, 256)

    @cute.jit
    def gate_up_slice(
        self, p: MoeParams, s: MoeState, round_index, shared: cutlass.Constexpr
    ):
        accum = rt.zeros(8)
        lane = s.lane
        if cutlass.const_expr(self.ROWS == 8):
            token_words = cdiv(lane, 4) * (NORM_ROW // 2) + cmod(lane, 4) * 8
        else:
            token_words = cdiv(lane, 8) * (NORM_ROW // 2) + cmod(lane, 4) * 8
        for page in range(6):
            self.gate_up_slice_page(p, s, accum, token_words, page, round_index, shared)
        self.publish_slice(p, s, accum, round_index, shared)

    @cute.jit
    def gate_up_slice_page(
        self,
        p: MoeParams,
        s: MoeState,
        accum,
        token_words,
        page,
        round_index,
        shared: cutlass.Constexpr,
    ):
        sequence = page if shared else 1 + 6 * round_index + page
        self.gate_up_page(
            s,
            accum,
            sequence,
            token_words + cdiv(page * 1024 + s.warp * 128, 2),
            s.warp,
            s.lane,
        )

    @cute.jit
    def publish_slice(
        self, p: MoeParams, s: MoeState, accum, round_index, shared: cutlass.Constexpr
    ):
        tid = s.tid
        self.store_partials(s, accum, s.warp, s.lane)
        rt.bar_sync(8, 256)
        if tid < 16 * self.ROWS:
            token, row = cdiv(tid, 16), cmod(tid, 16)
            act = self.silu_up(s.scratch, token, row)
            p.Activation[self.activation_index(s, token, row, round_index, shared)] = (
                act.to(BFloat16)
            )
        rt.bar_sync(8, 256)

    def activation_index(self, s: MoeState, token, row, round_index, shared: bool):
        if shared:
            return token * 256 + ActPlane.col((s.bid - ROUTED_CTAS) * 16 + row)
        plane = 1 + cdiv(s.bid, 16) + 8 * round_index
        return (plane * self.ROWS + token) * 256 + ActPlane.col(
            cmod(s.bid, 16) * 16 + row
        )

    @cute.jit
    def store_partials(self, s: MoeState, accum, warp, lane):
        if cutlass.const_expr(self.ROWS == 8):
            base = warp * self.PARTIAL_STRIDE + cmod(lane, 4) * 64 + cdiv(lane, 4)
            for m in cutlass.range_constexpr(2):
                for column in cutlass.range_constexpr(2):
                    s.scratch[base + column * 32 + m * 16] = accum[m * 4 + column]
                    s.scratch[base + column * 32 + m * 16 + 8] = accum[
                        m * 4 + column + 2
                    ]
        else:
            self.stage_partials(
                s.scratch,
                accum,
                warp * self.PARTIAL_STRIDE + cmod(lane, 4) * 32 + cdiv(lane, 4),
            )

    def silu_up(self, scratch, token, row):
        stride = self.PARTIAL_STRIDE
        gate = scratch[token * 32 + row]
        up = scratch[token * 32 + 16 + row]
        for w in range(1, 8):
            gate = rt.add(gate, scratch[w * stride + token * 32 + row])
            up = rt.add(up, scratch[w * stride + token * 32 + 16 + row])
        return rt.mul(
            rt.mul(
                gate,
                rt.FastMath.rcp_rn(
                    rt.add(1.0, cute.math.exp2(rt.mul(-gate, LOG2E), fastmath=True))
                ),
            ),
            up,
        )

    def stage_partials(self, scratch, accum, base):
        for m in range(2):
            scratch[base + m * 16] = accum[m * 4]
            scratch[base + m * 16 + 8] = accum[m * 4 + 2]

    @cute.jit
    def gate_up_page(self, s: MoeState, accum, sequence, k_words, warp, lane):
        slot_bytes = self.SLOT_BYTES
        slot = cmod(sequence, self.RING_SLOTS)
        rt.mbar_wait(s.ring_full + slot * 8, cmod(cdiv(sequence, self.RING_SLOTS), 2))
        up_rows = cute.make_rmem_tensor(32, Uint32)
        odd = cmod(cdiv(lane, 4), 2)
        chunk = cdiv(
            slot * slot_bytes + cdiv(lane, 4) * 1024 + warp * 128 + cmod(lane, 4) * 16,
            4,
        )
        for h in cutlass.range_constexpr(4):
            r0 = rt.Memory.lds_b128(
                s.ring_words_addr + (chunk + 2048 * h + 16 * odd) * 4
            )
            r1 = rt.Memory.lds_b128(
                s.ring_words_addr + (chunk + 2048 * h + 16 - 16 * odd) * 4
            )
            for i in cutlass.range_constexpr(4):
                up_rows[8 * h + i] = Uint32(cutlass.select_(odd != 0, r1[i], r0[i]))
                up_rows[8 * h + 4 + i] = Uint32(cutlass.select_(odd != 0, r0[i], r1[i]))
        scale_base = slot * (slot_bytes // 4) + 8192
        gate_scale = s.ring_f32[scale_base + warp]
        up_scale = s.ring_f32[scale_base + 8 + warp]
        if cutlass.const_expr(self.EARLY_RELEASE):
            s.pages.release(slot)
        norm = s.norm_words_addr
        b_words = lambda k: rt.Memory.lds_weak_b64(
            norm + (k_words + k % 4 * 2 + k // 4 * 32) * 4
        )
        rt.Mma.gate_up_steps(accum, up_rows, b_words, gate_scale, up_scale)
        if cutlass.const_expr(not self.EARLY_RELEASE):
            s.pages.release(slot)

    @cute.jit
    def exchange(self, p: MoeParams, s: MoeState, epoch, total_r):
        x = self.mailbox
        x.reduce_rows(
            s.scratch, s.local_o, p.LocalO, total_r, s.tid, s.warp, s.lane, s.bid
        )
        self.elect_add(p.Flags, s.warp, self.Flag.LOCAL_O + s.bid, scoped=False)
        x.push_peers(p, s.local_o_words, epoch, s.bid, s.tid, s.warp, s.lane)
        self.exchange_tail(p, s, epoch)

    @cute.jit
    def exchange_tail(self, p: MoeParams, s: MoeState, epoch):
        self.elect_add(p.Flags, s.warp, self.Flag.DONE + s.bid, scoped=False)
        combining = s.warp == 0
        if combining:
            self.mailbox.combine(p, p.X, s.lane, s.bid, epoch, s.local_o_words)

    @cute.jit
    def routed_down(self, s: MoeState, total, count, first_down):
        route_lane = self.route_lane(s.lane)
        for page in range(cdiv(count + 1, 2)):
            self.routed_down_record_pair(s, total, count, first_down, page, route_lane)

    @cute.jit
    def routed_down_record_pair(
        self, s: MoeState, total, count, first_down, page, route_lane
    ):
        bid, lane, warp = s.bid, s.lane, s.warp
        sequence_dn = first_down + page
        slot_dn = cmod(sequence_dn, 4)
        rt.mbar_wait(s.ring_full + slot_dn * 8, cmod(cdiv(sequence_dn, 4), 2))
        er_second = cutlass.min(page * 2 + 1, count - 1)
        er_scales = cute.make_rmem_tensor(6, Float32)
        page_rows = []
        rb = slot_dn * self.SLOT_BYTES + DownPlane.pad(
            s.ring_u64_addr, slot_dn * self.SLOT_BYTES
        )
        for h in cutlass.range_constexpr(2):
            page_rows += DownPlane.rows(
                s.ring_u64, rb + DownPlane.RECORD * h, warp, lane
            )
            for j in cutlass.range_constexpr(3):
                er_scales[3 * h + j] = s.ring_f32[
                    DownPlane.scale_word(
                        rb + DownPlane.SCALES + 32 * h >> 2, bid, warp, j
                    )
                ]
        self.admit_down_page(s, page)
        lane_words = self.down_lane_words(warp, lane)
        er_part = (rt.zeros((4, 3)), rt.zeros((4, 3)))
        expert_row = (page * 2, er_second)
        for kk_dn in cutlass.range_constexpr(2):
            for h in cutlass.range_constexpr(2):
                b_0 = rt.Memory.lds_weak_b64(
                    s.act_words_addr
                    + (self.down_plane_word(expert_row[h]) + lane_words + 4 * kk_dn) * 4
                )
                for m in cutlass.range_constexpr(3):
                    acc_tile = er_part[h][None, m]
                    row_a, row_b = (
                        page_rows[6 * h + 2 * m],
                        page_rows[6 * h + 2 * m + 1],
                    )
                    frag = rt.Convert.a_frag_u64(row_a, row_b, kk_dn)
                    rt.TensorCore.mma_acc(acc_tile, frag, b_0, False)
        s.pages.release(slot_dn)
        DownPlane.accum(
            total, er_part[0], er_scales, 0, *self.route_pair(s, page * 2, route_lane)
        )
        if count > page * 2 + 1:
            DownPlane.accum(
                total,
                er_part[1],
                er_scales,
                3,
                *self.route_pair(s, er_second, route_lane),
            )

    @abstractmethod
    def admit_down_page(self, s: MoeState, page):
        raise NotImplementedError

    @abstractmethod
    def down_plane_word(self, expert):
        raise NotImplementedError

    @abstractmethod
    def down_lane_words(self, warp, lane):
        raise NotImplementedError

    @abstractmethod
    def route_lane(self, lane):
        raise NotImplementedError

    @abstractmethod
    def route_pair(self, s: MoeState, expert, route_lane):
        raise NotImplementedError

    @cute.jit
    def publish_logit(self, p: MoeParams, s: MoeState, logit, epoch):
        tid, bid, Flags = s.tid, s.bid, p.Flags
        p.RouterLogits[cdiv(tid, 2) * 256 + bid * 2 + cmod(tid, 2)] = logit
        bits = logit.bitcast(Uint32)
        if bits == Uint32(2147483648):
            bits = Uint32(0)
        word = cdiv(tid, 2) * 256 + bid * 2 + cmod(tid, 2)
        parity = Int32(cmod(epoch, 2))
        filled = self.Flag.PAD_FIRST + (1 + parity) * self.Flag.PAGE + word
        rt.Memory.st_relaxed_sys(rt.ga(Flags, filled), bits ^ Uint32(2147483648))
        rt.Memory.st_relaxed_sys(
            rt.ga(Flags, filled + self.Flag.MIRROR), bits ^ Uint32(2147483648)
        )

    @cute.jit
    def clear_stale_logits(self, Flags, epoch, bid, i, copy: cutlass.Constexpr):
        word = cdiv(i, 2) * 256 + bid * 2 + cmod(i, 2)
        parity = Int32(cmod(epoch, 2))
        rt.Memory.st_relaxed_sys(
            rt.ga(
                Flags,
                self.Flag.PAD_FIRST
                + (2 - parity) * self.Flag.PAGE
                + copy * self.Flag.MIRROR
                + word,
            ),
            0,
        )

    @cute.jit
    def count_router_slice(self, p: MoeParams, s: MoeState):
        Flags, bid = p.Flags, s.bid
        base = self.Flag.PAD_FIRST + bid
        cute.arch.atomic_add(Flags.iterator + base, Uint32(1))
        cute.arch.atomic_add(Flags.iterator + (base + self.Flag.MIRROR), Uint32(1))
        cute.arch.atomic_add(Flags.iterator + bid, Uint32(1))

    @cute.jit
    def pull_logits(self, p: MoeParams, s: MoeState, epoch_ld, lane_0):
        Flags = p.Flags
        near = Int32(
            rt.Memory.ld_relaxed_gpu(
                rt.ga(Flags, self.Flag.SM_DOMAIN + Int32(rt.Sync.smid()))
            )
        )
        polled = self.Flag.PAD_FIRST + near * self.Flag.MIRROR
        slot_owned = Int32(lane_0 < 128)
        slot_index = polled + cutlass.min(lane_0, 127)
        self.poll_all(Flags, slot_owned, slot_index, epoch_ld)
        rt.bar_sync(9, 128)
        tagged_base = (
            self.Flag.PAD_FIRST
            + (1 + Int32(cmod(epoch_ld, 2))) * self.Flag.PAGE
            + (1 - near) * self.Flag.MIRROR
        )
        self.poll_logit_words(Flags, s, tagged_base, lane_0, self.ROWS // 2)
        return polled

    @cute.jit
    def poll_logit_words(
        self, Flags, s: MoeState, tagged_base, lane_0, halves: cutlass.Constexpr
    ):
        kept = cute.make_rmem_tensor(4 * halves, Uint32)
        done = Int32(0)
        while done == 0:
            present = Int32(1)
            for h_ld in cutlass.range_constexpr(halves):
                w = rt.Memory.ld_volatile_v4(
                    rt.ga(Flags, tagged_base + (lane_0 + 128 * h_ld) * 4)
                )
                for item_ld in cutlass.range_constexpr(4):
                    kept[4 * h_ld + item_ld] = w[item_ld]
                    present = present & Int32(w[item_ld] != 0)
            done = Int32(cute.arch.vote_all_sync(present != 0))
        for half in cutlass.range_constexpr(halves):
            for item_ld in cutlass.range_constexpr(4):
                s.logits_words[lane_0 * 4 + item_ld + 512 * half] = kept[
                    4 * half + item_ld
                ] ^ Uint32(2147483648)

    @cute.jit
    def up_page(
        self,
        p: MoeParams,
        s: MoeState,
        sequence,
        page,
        expert,
        row16,
        shared: cutlass.Constexpr = False,
    ):
        slot_bytes = self.SLOT_BYTES
        slot = cmod(sequence, self.RING_SLOTS)
        with rt.warp_leader(s.warp, 8):
            s.pages.acquire(sequence, UpPage.TX, slot)
            scales = p.SharedScales if shared else p.RoutedUpScales
            first = 0 if shared else Int64(expert) * LAYOUT.w13_scale_words
            block = cdiv(row16, 8) * 48 + page * 8
            for h in cutlass.range_constexpr(2):
                box_dst = s.ring_addr + slot * slot_bytes + UpPage.BOX * h
                full = s.ring_full + slot * 8
                half = h if shared else 1 - h
                if cutlass.const_expr(shared):
                    rt.AsyncCopy.tma_4d(
                        box_dst,
                        self.shared_up_map,
                        0,
                        page * 4,
                        256 * h + row16 * 16,
                        0,
                        full,
                        _HINT,
                    )
                else:
                    rt.AsyncCopy.tma_4d(
                        box_dst,
                        self.up_map,
                        0,
                        expert * 48 + page * 8,
                        32 * half + row16 * 2,
                        0,
                        full,
                        _HINT,
                    )
                scale_dst = s.ring_addr + slot * slot_bytes + UpPage.SCALES + 32 * h
                scale_src = rt.ga(scales, first + block + 96 * half)
                rt.bulk_g2s(scale_dst, scale_src, 32, s.ring_full + slot * 8)

    @cute.jit
    def routed_down_page(self, p: MoeParams, s: MoeState, sequence, page, count, ids):
        slot_bytes = self.SLOT_BYTES
        slot = cmod(sequence, self.RING_SLOTS)
        with rt.warp_leader(s.warp, 8):
            rt.mbar_wait(
                s.ring_empty + slot * 8, cmod(cdiv(sequence, self.RING_SLOTS), 2) ^ 1
            )
            records = cutlass.min(count - page * 2, 2)
            rt.mbar_expect_tx(s.ring_full + slot * 8, records * DownPlane.TX)
            aligned = s.ring_addr + (
                slot * slot_bytes + DownPlane.pad(s.ring_addr, slot * slot_bytes)
            )
            for within in cutlass.range_constexpr(2):
                if records > within:
                    record = ids[page * 2 + within]
                    for h in cutlass.range_constexpr(2):
                        box_dst = (
                            aligned + within * DownPlane.RECORD + DownPlane.BOX * h
                        )
                        groups = (2 * record + h) * (rt.HIDDEN // 16) + s.bid * 3
                        full = s.ring_full + slot * 8
                        rt.AsyncCopy.tma_4d(
                            box_dst, self.down_map, 0, 0, 0, groups, full, _HINT
                        )
                    scale_dst = aligned + DownPlane.SCALES + within * 32
                    scale_src = rt.ga(
                        p.RoutedDownScales,
                        record * LAYOUT.w2_scale_words + DownPlane.scales(s.bid),
                    )
                    rt.bulk_g2s(scale_dst, scale_src, 32, s.ring_full + slot * 8)


class MoeQ4Smem(rt.SmemLayout):
    """Shared memory of q4 and the seam."""

    ROWS = 4
    MAX_UNION = ROWS * TOP_K
    ROUNDS = 4
    PLANES = 1 + MAX_UNION
    SLOT_BYTES = 32896
    SHARED_DOWN = RING_SLOTS * SLOT_BYTES
    SHARED_DOWN_SPAN = 1024 + DownPlane.RECORD
    SMEM_BYTES = 228608

    a = rt.SmemAlloc(SMEM_BYTES)
    ring_full = a.mbarriers(RING_SLOTS, arrivals=1)
    ring_empty = a.mbarriers(RING_SLOTS, arrivals=COMPUTE_THREADS)
    x_full = a.mbarriers(1, arrivals=1)
    route_full = a.mbarriers(1, arrivals=1)
    union_full = a.mbarriers(1, arrivals=1)
    shared_down_full = a.mbarriers(1, arrivals=1)
    planes_full = a.mbarriers(ROUNDS, arrivals=WARP)
    rows_free = a.mbarriers(1, arrivals=1)
    round_done = a.mbarriers(ROUNDS, arrivals=COMPUTE_THREADS)
    with a.union(align=1024):
        scratch = a.array(Float32, SCRATCH_WORDS)
        with a.arm():
            a.pad(SCRATCH_WORDS // 2 * 4)
            first_lane = a.array(Int32, EXPERTS)
    route_ids = a.array(Int32, (ROWS, TOP_K))
    route_weights = a.array(Float32, (ROWS, TOP_K))
    union_ids = a.array(Int32, MAX_UNION)
    union_route = a.array(Float32, (MAX_UNION, ROWS))
    union_count = a.array(Int32, 1)
    epoch_slot = a.array(Uint32, 1)
    local_o = a.array(BFloat16, (ROWS, COLUMNS), also=(Uint32,))
    with a.union(align=512):
        with a.arm():
            norm = a.array(BFloat16, (ROWS, NORM_ROW), also=(Uint32,))
            logits = a.array(Float32, (ROWS, EXPERTS), align=128, also=(Uint32,))
            corrected = a.array(Float32, (ROWS, EXPERTS))
            scores = a.array(Float32, (ROWS, EXPERTS))
        act_words = a.array(Uint32, (PLANES, ROWS, ActPlane.ROW_WORDS))
    ring = a.array(
        Uint8,
        RING_SLOTS * SLOT_BYTES + SHARED_DOWN_SPAN,
        align=128,
        also=(BFloat16, Uint64, Uint32, Float32),
    )


class FusedMoeQ4(FusedMoe):
    """Four token rows (``fused_moe_q4_ckpt_shared``)."""

    ROWS = 4
    Smem = MoeQ4Smem
    SLOT_BYTES = MoeQ4Smem.SLOT_BYTES
    SMEM_BYTES = MoeQ4Smem.SMEM_BYTES
    PARTIAL_STRIDE = 129
    SHARED_DOWN = MoeQ4Smem.SHARED_DOWN

    @cute.jit
    def shared_up_page(self, p: MoeParams, s: MoeState, page):
        self.up_page(p, s, page, page, LAYOUT.shared_slot, s.bid - ROUTED_CTAS, True)

    @cute.jit
    def compute_cta(self, p: MoeParams, s: MoeState):
        tid, warp, bid = s.tid, s.warp, s.bid
        with rt.warp_leader(warp, 0):
            cute.arch.griddepcontrol_wait()
            rt.mbar_expect_tx(s.x_full, self.ROWS * 2 * rt.HIDDEN)
            for token in cutlass.range_constexpr(self.ROWS):
                norm_dst = s.norm_addr + token * NORM_ROW * 2
                rt.bulk_g2s(
                    norm_dst, rt.ga(p.X, token * rt.HIDDEN), 2 * rt.HIDDEN, s.x_full
                )
            if bid < ROUTED_CTAS:
                l2_range_policy = rt.AsyncCopy.createpolicy_evict_first(
                    rt.ga(p.RouterWeight, 0), 3145728, 3145728
                )
                rt.mbar_expect_tx(s.ring_full, 24576)
                router_src = rt.ga(p.RouterWeight, bid * 2 * rt.HIDDEN)
                policy = l2_range_policy
                rt.bulk_g2s_hint(
                    s.ring_bf16_addr, router_src, 24576, s.ring_full, policy
                )
        raw_gammas = Rms.gamma_load(p.Gamma, tid)
        early_bias = p.CorrectionBias[tid]
        self.set_regs("compute")
        cute.arch.griddepcontrol_wait()
        rt.mbar_wait(s.x_full, 0)
        gammas_f32 = [
            rt.bf16x8_to_f32([raw[i] for i in range(4)]) for raw in raw_gammas
        ]
        if bid >= ROUTED_CTAS:
            self.rms_norm(p, s, gammas_f32)
        if bid >= ROUTED_CTAS:
            rt.bar_sync(11, 384)
        epoch = s.epoch_slot[0]
        if bid >= ROUTED_CTAS:
            self.gate_up_slice(p, s, 0, True)
            rt.bar_sync(8, 256)
            with rt.warp_leader(warp, 0):
                rt.Memory.fence_release_gpu()
                padded = self.Flag.PAD_FIRST + bid
                rt.Memory.atomic_add_u32(p.Flags, padded, 1)
                rt.Memory.atomic_add_u32(p.Flags, padded + self.Flag.MIRROR, 1)
                rt.Memory.atomic_add_u32(p.Flags, self.Flag.ACT + bid, 1)
            rt.bar_sync(8, 256)
            self.shared_tail(p, s, epoch)
        else:
            epoch = self.router(p, s, gammas_f32)
            count, rounds = self.route(p, s, early_bias)
            for round_index in range(rounds):
                self.gate_up_slice(p, s, round_index, False)
                rt.mbar_arrive(s.round_done + round_index * 8)
            total = self.down(p, s, count, rounds)
            self.exchange(p, s, epoch, total)

    @cute.jit
    def shared_tail(self, p: MoeParams, s: MoeState, epoch):
        self.elect_add(p.Flags, s.warp, self.Flag.DONE + s.bid)

    @cute.jit
    def router(self, p: MoeParams, s: MoeState, gammas):
        tid, warp, lane = s.tid, s.warp, s.lane
        scratch = s.scratch
        squares_r = cute.make_rmem_tensor(4, Float32)
        sums01 = (Float32(0.0), Float32(0.0))
        sums23 = (Float32(0.0), Float32(0.0))
        muls = []
        for g_r in cutlass.range_constexpr(3):
            nw = []
            for t in cutlass.range_constexpr(4):
                nw.append(
                    rt.Convert.ld_bf16x8_words(
                        s.norm_words, cdiv(g_r * 2048 + t * NORM_ROW + tid * 8, 2)
                    )
                )
            for k in cutlass.range_constexpr(8):
                gk = (gammas[g_r][k], gammas[g_r][k])
                a = (nw[0][k], nw[1][k])
                b = (nw[2][k], nw[3][k])
                sums01 = rt.FastMath.fma2(a, a, sums01)
                sums23 = rt.FastMath.fma2(b, b, sums23)
                muls.append(rt.FastMath.mul2(a, gk))
                muls.append(rt.FastMath.mul2(b, gk))
        squares_r[0], squares_r[1] = sums01
        squares_r[2], squares_r[3] = sums23
        last_r = rt.Reduce.fold([squares_r[t] for t in range(4)], lane, 16, (4, 2, 1))
        if (lane & 7) == 0:
            scratch[(lane >> 3 & 3) * 8 + warp] = last_r
        rt.mbar_wait(s.ring_full, 0)
        acc = cute.make_rmem_tensor(8, Float32)
        fused = [(Float32(0.0), Float32(0.0))] * 4
        for g_r in cutlass.range_constexpr(3):
            column = g_r * 2048 + tid * 8
            r0 = rt.Convert.ld_bf16x8_words(s.ring_words, cdiv(column, 2))
            r1 = rt.Convert.ld_bf16x8_words(s.ring_words, cdiv(rt.HIDDEN + column, 2))
            for k in cutlass.range_constexpr(8):
                pk = (r0[k], r0[k])
                qk = (r1[k], r1[k])
                for j in cutlass.range_constexpr(4):
                    fused[j] = rt.FastMath.fma2(
                        muls[g_r * 16 + 2 * k + j // 2], (pk, qk)[j % 2], fused[j]
                    )
        for j in cutlass.range_constexpr(4):
            acc[j % 2 + 4 * (j // 2)], acc[j % 2 + 2 + 4 * (j // 2)] = fused[j]
        s.pages.release(0)
        last_0 = rt.Reduce.fold([acc[i] for i in range(8)], lane, 16, (2, 1))
        if (lane & 3) == 0:
            scratch[64 + warp * 2 * 4 + (lane >> 2 & 7)] = last_0
        rt.bar_sync(8, 256)
        rt.bar_sync(11, 384)
        fused_epoch = s.epoch_slot[0]
        if tid < 8:
            row_total = rt.Reduce.strided_sum(scratch, cdiv(tid, 2) * 8, 1)
            inverse = rt.Reduce.rms_inverse(row_total)
            total_r = rt.Reduce.strided_sum(scratch, 64 + tid, 8)
            logit_0 = rt.mul(total_r, inverse)
            self.publish_logit(p, s, logit_0, fused_epoch)
        epoch = s.epoch_slot[0]
        if tid == 0:
            self.count_router_slice(p, s)
        inverses_r = Rms.inverses(scratch)
        for token_r in cutlass.range_constexpr(4):
            for g_r in cutlass.range_constexpr(3):
                off = token_r * NORM_ROW + g_r * 2048 + tid * 8
                norm_dst = s.norm_addr + off * 2
                words = rt.Convert.ld_bf16x8_words(s.norm_words, cdiv(off, 2))
                Rms.store(norm_dst, words, inverses_r[token_r], gammas[g_r])
        rt.bar_sync(11, 384)
        return epoch

    @cute.jit
    def route(self, p: MoeParams, s: MoeState, early_bias):
        lane = s.lane
        self.score_and_route(p, s, early_bias)
        if s.warp == 0:
            expert = s.route_ids[lane]
            cute.arch.atomic_min(s.first_lane.iterator + expert, lane)
            cute.arch.sync_warp()
            first = s.first_lane[expert]
            _vote_0 = Uint32(cute.arch.vote_ballot_sync(first == lane))
            below = (Uint32(1) << Uint32(first)) - 1
            index = Int32(cute.arch.popc(_vote_0 & below))
            if first == lane:
                s.union_ids[index] = expert
            s.union_route[index * 4 + cdiv(lane, 8)] = s.route_weights[lane]
            cute.arch.sync_warp()
            if lane == 0:
                count_0 = Int32(cute.arch.popc(_vote_0))
                s.union_count[0] = count_0
                rt.mbar_arrive(s.union_full)
        rt.bar_sync(8, 256)
        count = s.union_count[0]
        rounds = cdiv(count - cdiv(s.bid, 16) + 8 - 1, 8)
        return count, rounds

    @cute.jit
    def down(self, p: MoeParams, s: MoeState, count, rounds):
        bid, lane, warp = s.bid, s.lane, s.warp
        total_r = rt.zeros(12)
        rt.mbar_wait(s.planes_full, 0)
        rt.mbar_wait(s.shared_down_full, 0)
        down_off = DownPlane.pad(s.ring_u64_addr, self.SHARED_DOWN) + self.SHARED_DOWN
        shared_rows = DownPlane.rows(s.ring_u64, down_off, warp, lane)
        shared_scales = [
            s.ring_f32[
                DownPlane.scale_word(down_off + DownPlane.RECORD >> 2, bid, warp, k)
            ]
            for k in range(3)
        ]
        token_words = cmod(cdiv(lane, 4), 4) * ActPlane.ROW_WORDS + DownPlane.k_words(
            warp, lane
        )
        shared_part = rt.zeros(12)
        for kk in cutlass.range_constexpr(2):
            b = rt.Memory.lds_weak_b64(
                s.act_words_addr + (12672 + token_words + 4 * kk) * 4
            )
            for m in cutlass.range_constexpr(3):
                acc_tile = cute.local_tile(shared_part, (4,), (m,))
                frag = rt.Convert.a_frag_u64(
                    shared_rows[2 * m], shared_rows[2 * m + 1], kk
                )
                rt.TensorCore.mma_acc(acc_tile, frag, b, False)
        for i in cutlass.range_constexpr(12):
            total_r[i] = rt.fma(shared_part[i], shared_scales[cdiv(i, 4)], total_r[i])
        self.routed_down(s, total_r, count, 1 + 6 * rounds)
        return total_r

    @cute.jit
    def admit_down_page(self, s: MoeState, page):
        if (page > 0) & (cmod(page, 4) == 0):
            rt.mbar_wait(s.planes_full + (cdiv(page, 4)) * 8, 0)

    def down_plane_word(self, expert):
        return (1 + expert - 9 + 33 * Int32(1 + expert < 9)) * 4 * ActPlane.ROW_WORDS

    def down_lane_words(self, warp, lane):
        return cmod(cdiv(lane, 4), 4) * ActPlane.ROW_WORDS + DownPlane.k_words(
            warp, lane
        )

    def route_lane(self, lane):
        return cmod(cmod(lane, 4) * 2, 4)

    def route_pair(self, s: MoeState, expert, route_lane):
        route = expert * 4 + route_lane
        return s.union_route, (route, route + 1)

    @cute.jit
    def load_prologue(self, p: MoeParams, s: MoeState):
        warp, bid = s.warp, s.bid
        cute.arch.griddepcontrol_wait()
        if bid < ROUTED_CTAS:
            with rt.warp_leader(warp, 8):
                rt.mbar_expect_tx(s.shared_down_full, DownPlane.TX)
                dst = s.ring_addr + (
                    self.SHARED_DOWN + DownPlane.pad(s.ring_addr, self.SHARED_DOWN)
                )
                for h_ld in cutlass.range_constexpr(2):
                    rt.AsyncCopy.tma_3d(
                        dst + DownPlane.BOX * h_ld,
                        self.shared_down_map,
                        128 * h_ld,
                        bid * COLUMNS,
                        0,
                        s.shared_down_full,
                        _HINT,
                    )
                scale_dst = dst + DownPlane.RECORD
                scale_src = rt.ga(
                    p.SharedScales, LAYOUT.w13_scale_words + DownPlane.scales(bid)
                )
                rt.bulk_g2s(scale_dst, scale_src, 32, s.shared_down_full)
        else:
            self.shared_up_pages(p, s, 0, self.RING_SLOTS)

    @cute.jit
    def load_cta(self, p: MoeParams, s: MoeState, epoch_ld, lane_0):
        bid = s.bid
        if bid >= ROUTED_CTAS:
            self.shared_up_pages(p, s, self.RING_SLOTS, 6)
        else:
            polled = self.pull_logits(p, s, epoch_ld, lane_0)
            rt.bar_sync(11, 384)
            row16 = cmod(bid, 16)
            rt.mbar_wait(s.route_full, 0)
            first_expert = s.route_ids[cdiv(bid, 16)]
            self.up_pages(p, s, 1, first_expert, row16)
            rt.mbar_wait(s.union_full, 0)
            count_ld = s.union_count[0]
            rounds_ld = cdiv(count_ld - cdiv(bid, 16) + 8 - 1, 8)
            for round_index_ld in range(1, rounds_ld):
                expert_ld = s.union_ids[cdiv(bid, 16) + 8 * round_index_ld]
                self.up_pages(p, s, 1 + 6 * round_index_ld, expert_ld, row16)
            self.load_rounds(p, s, epoch_ld, lane_0, polled, count_ld, rounds_ld)

    @cute.jit
    def load_rounds(
        self, p: MoeParams, s: MoeState, epoch_ld, lane_0, polled, count_ld, rounds_ld
    ):
        bid, warp, Flags = s.bid, s.warp, p.Flags
        first_down_ld = 1 + 6 * rounds_ld
        down_pages = cdiv(count_ld + 1, 2)
        early_pages = cutlass.min(down_pages, 4)
        loader_warp = warp - 8
        if loader_warp == 0:
            for page_ld in range(0, early_pages):
                self.routed_down_page(
                    p, s, first_down_ld + page_ld, page_ld, count_ld, s.union_ids
                )
            with rt.warp_leader(warp, 8):
                rt.mbar_arrive(s.rows_free)
            for page_ld in range(early_pages, down_pages):
                self.routed_down_page(
                    p, s, first_down_ld + page_ld, page_ld, count_ld, s.union_ids
                )
            if lane_0 < 8:
                self.clear_stale_logits(Flags, epoch_ld, bid, lane_0, 0)
        else:
            if loader_warp == 3:
                for published in range(0, rounds_ld):
                    rt.mbar_wait(s.round_done + published * 8, 0)
                    with rt.warp_leader(warp, 11):
                        rt.Memory.fence_release_gpu()
                        bump = Uint32(1)
                        if published == rounds_ld - 1:
                            bump = Uint32(5 - rounds_ld)
                        base_ld = self.Flag.PAD_FIRST + (self.Flag.PAD_ROUND + bid)
                        cute.arch.atomic_add(Flags.iterator + base_ld, bump)
                        cute.arch.atomic_add(
                            Flags.iterator + (base_ld + self.Flag.MIRROR), bump
                        )
                        cute.arch.atomic_add(
                            Flags.iterator + (self.Flag.ROUND + bid), bump
                        )
                        if published == rounds_ld - 1:
                            cute.arch.atomic_add(
                                Flags.iterator + (self.Flag.ACT + bid), Uint32(1)
                            )
            lane_w = cmod(lane_0, 32)
            union_rounds = cdiv(count_ld + 8 - 1, 8)
            first_round = loader_warp
            if loader_warp == 1:
                first_round = 0
            for round_index_ld in range(
                first_round, cutlass.min(loader_warp + 1, union_rounds)
            ):
                if round_index_ld > 0:
                    rt.mbar_wait(s.rows_free, 0)
                target = epoch_ld * 4 + Uint32(round_index_ld) + 1
                done_poll = Int32(0)
                while done_poll == 0:
                    satisfied = Int32(1)
                    for k in cutlass.range_constexpr(4):
                        cta = lane_w + 32 * k
                        seen = rt.Memory.ld_relaxed_gpu(
                            rt.ga(Flags, polled + self.Flag.PAD_ROUND + cta)
                        )
                        if count_ld > cdiv(cta, 16) + 8 * round_index_ld:
                            satisfied = satisfied & Int32(Int32(seen - target) >= 0)
                    shared_seen = Uint32(epoch_ld + 1)
                    if (round_index_ld == 0) & (lane_w < 16):
                        shared_seen = rt.Memory.ld_relaxed_gpu(
                            rt.ga(Flags, polled + self.Flag.PAD_SHARED + lane_w)
                        )
                    satisfied = satisfied & Int32(
                        Int32(shared_seen - (epoch_ld + 1)) >= 0
                    )
                    done_poll = Int32(cute.arch.vote_all_sync(satisfied != 0))
                rt.Memory.fence_acquire_gpu()
                cute.arch.sync_warp()
                for group in range(0, cutlass.min(count_ld - 8 * round_index_ld, 8)):
                    plane = 1 + group + 8 * round_index_ld
                    for part_ld in cutlass.range_constexpr(4):
                        chunk = plane * 128 + lane_w + 32 * part_ld
                        rt.AsyncCopy.cp_async_cg16(
                            s.act_words_addr
                            + (
                                (plane - 9 + 33 * Int32(plane < 9)) * 4
                                + cmod(cdiv(chunk, 32), 4)
                            )
                            * ActPlane.ROW_BYTES
                            + cmod(chunk, 32) * 16,
                            rt.ga(p.Activation, chunk * 8),
                        )
                if round_index_ld == 0:
                    for part_ld in cutlass.range_constexpr(4):
                        chunk = lane_w + 32 * part_ld
                        rt.AsyncCopy.cp_async_cg16(
                            s.act_words_addr
                            + (96 + cmod(cdiv(chunk, 32), 4)) * ActPlane.ROW_BYTES
                            + cmod(chunk, 32) * 16,
                            rt.ga(p.Activation, chunk * 8),
                        )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                rt.mbar_arrive(s.planes_full + round_index_ld * 8)
                if (round_index_ld == union_rounds - 1) & (lane_w == 0):
                    cute.arch.atomic_add(
                        Flags.iterator + (self.Flag.PLANES + bid), Uint32(1)
                    )
            if (lane_0 >= 32) & (lane_0 < 40):
                self.clear_stale_logits(Flags, epoch_ld, bid, lane_0 - 32, 1)


class FusedMoeSeam(FusedMoeQ4):
    """q4 seam: column-slice combine instead of the per-CTA pull."""

    SEAM = True
    Params = MoeSeamParams  # type: ignore[assignment]

    @cute.jit
    def shared_tail(self, p: MoeParams, s: MoeState, epoch):
        self.elect_add(p.Flags, s.warp, self.Flag.DONE + s.bid)
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.ROWS == 8):
            seam_slice = s.bid - ROUTED_CTAS + 48 * cdiv(s.warp, 2)
            if seam_slice < 128:
                self.mailbox.combine(
                    p,
                    p.X,
                    s.lane,
                    seam_slice,
                    epoch,
                    None,
                    True,
                    group=cmod(s.warp, 2),
                )
        else:
            seam_slice = s.bid - ROUTED_CTAS + 48 * s.warp
            if seam_slice < 128:
                self.mailbox.combine(p, p.X, s.lane, seam_slice, epoch, None, True)

    @cute.jit
    def exchange_tail(self, p: MoeParams, s: MoeState, epoch):
        if s.warp == 7:
            peer = rt.Memory.select_peer(p.peer_rank, p.peer_bases)
            if cutlass.const_expr(self.ROWS == 8):
                for group in cutlass.range_constexpr(self.mailbox.groups):
                    self.mailbox.push(
                        s.local_o_words,
                        epoch,
                        s.bid,
                        s.lane,
                        peer,
                        self.mailbox.OWN,
                        group,
                    )
            else:
                self.mailbox.push(
                    s.local_o_words, epoch, s.bid, s.lane, peer, self.mailbox.OWN
                )
        self.elect_add(p.Flags, s.warp, self.Flag.DONE + s.bid, scoped=False)
        if s.bid >= 96:
            cute.arch.griddepcontrol_launch_dependents()
            if cutlass.const_expr(self.ROWS == 8):
                routed_slice = s.bid - 96 + 16 + 48 * cdiv(s.warp, 2)
                if routed_slice < 128:
                    self.mailbox.combine(
                        p,
                        p.X,
                        s.lane,
                        routed_slice,
                        epoch,
                        group=cmod(s.warp, 2),
                    )
            else:
                routed_slice = s.bid - 96 + 16 + 48 * s.warp
                if routed_slice < 128:
                    self.mailbox.combine(p, p.X, s.lane, routed_slice, epoch)


class MoeQ8Smem(rt.SmemLayout):
    """Eight-row scratch aliases router scores and Up/Down partials by phase."""

    ROWS = Q8Shape.ROWS
    ROUNDS = Q8Shape.ROUNDS
    SLOT_BYTES = MoeQ4Smem.SLOT_BYTES
    SHARED_DOWN = Q8Shape.RING_SLOTS * SLOT_BYTES
    SMEM_BYTES = MoeQ4Smem.SMEM_BYTES
    a = rt.SmemAlloc(SMEM_BYTES)
    ring_full = a.mbarriers(Q8Shape.RING_SLOTS, arrivals=1)
    ring_empty = a.mbarriers(Q8Shape.RING_SLOTS, arrivals=COMPUTE_THREADS)
    x_full = a.mbarriers(1, arrivals=1)
    route_full = a.mbarriers(1, arrivals=1)
    union_full = a.mbarriers(1, arrivals=1)
    shared_down_full = a.mbarriers(1, arrivals=1)
    planes_full = a.mbarriers(ROUNDS, arrivals=WARP)
    round_done = a.mbarriers(ROUNDS, arrivals=COMPUTE_THREADS)
    with a.union(align=1024):
        scratch = a.array(Float32, 8 * (ROWS * COLUMNS + 1))
        with a.arm():
            a.pad(1024)
            logits = a.array(Float32, (ROWS, EXPERTS), also=(Uint32,))
            first_lane = a.array(Int32, EXPERTS)
    route_ids = a.array(Int32, (ROWS, TOP_K))
    route_weights = a.array(Float32, (ROWS, TOP_K))
    union_ids = a.array(Int32, Q8Shape.MAX_UNION)
    union_route = a.array(Float32, (Q8Shape.MAX_UNION, ROWS))
    union_count = a.array(Int32, 1)
    epoch_slot = a.array(Uint32, 1)
    owners = a.array(Uint32, 2)
    local_o = a.array(BFloat16, (ROWS, COLUMNS), also=(Uint32,))
    norm = a.array(BFloat16, (ROWS, NORM_ROW), align=128, also=(Uint32,))
    ring = a.array(
        Uint8,
        SHARED_DOWN + MoeQ4Smem.SHARED_DOWN_SPAN,
        align=128,
        also=(BFloat16, Uint64, Uint32, Float32),
    )


class FusedMoeQ8(FusedMoeQ4):
    """Eight W8A16 token rows, with a 64-expert union and global Down activations."""

    ROWS = Q8Shape.ROWS
    RMAX = Q8Shape.ROWS
    Params = MoeQ8Params  # type: ignore[assignment]
    RING_SLOTS = Q8Shape.RING_SLOTS
    Flag = Q8Flag
    Smem = MoeQ8Smem  # type: ignore[assignment]
    SMEM_BYTES = MoeQ8Smem.SMEM_BYTES
    SHARED_DOWN = MoeQ8Smem.SHARED_DOWN
    PARTIAL_STRIDE = 257
    EARLY_RELEASE = True

    @cute.jit
    def exchange_tail(self, p: MoeParams, s: MoeState, epoch):
        self.elect_add(p.Flags, s.warp, self.Flag.DONE + s.bid, scoped=False)
        if s.warp < self.mailbox.groups:
            self.mailbox.combine(
                p, p.X, s.lane, s.bid, epoch, s.local_o_words, group=s.warp
            )

    @cute.jit
    def rms_norm(self, p: MoeParams, s: MoeState, gammas):
        for group in cutlass.range_constexpr(2):
            values = cute.make_rmem_tensor(96, Float32)
            squares = rt.zeros(4)
            for token in cutlass.range_constexpr(4):
                for part in cutlass.range_constexpr(3):
                    off = (4 * group + token) * NORM_ROW + part * 2048 + s.tid * 8
                    words = rt.Convert.ld_bf16x8_words(s.norm_words, cdiv(off, 2))
                    for item in cutlass.range_constexpr(8):
                        values[token * 24 + part * 8 + item] = words[item]
                        squares[token] = rt.fma(
                            words[item], words[item], squares[token]
                        )
            folded = rt.Reduce.fold(
                [squares[t] for t in range(4)], s.lane, 16, (4, 2, 1)
            )
            if (s.lane & 7) == 0:
                s.scratch[(s.lane >> 3 & 3) * 8 + s.warp] = folded
            rt.bar_sync(8, 256)
            inverses = Rms.inverses(s.scratch)
            for token in cutlass.range_constexpr(4):
                for part in cutlass.range_constexpr(3):
                    off = (4 * group + token) * NORM_ROW + part * 2048 + s.tid * 8
                    words = [values[token * 24 + part * 8 + i] for i in range(8)]
                    Rms.store(
                        s.norm_addr + off * 2, words, inverses[token], gammas[part]
                    )
            rt.bar_sync(8, 256)

    @cute.jit
    def router_group(self, s: MoeState, gammas, group: cutlass.Constexpr):
        tid, warp, lane = s.tid, s.warp, s.lane
        sums01 = (Float32(0.0), Float32(0.0))
        sums23 = (Float32(0.0), Float32(0.0))
        muls = []
        for part in cutlass.range_constexpr(3):
            words = [
                rt.Convert.ld_bf16x8_words(
                    s.norm_words,
                    cdiv(part * 2048 + (4 * group + t) * NORM_ROW + tid * 8, 2),
                )
                for t in range(4)
            ]
            for k in cutlass.range_constexpr(8):
                gamma = (gammas[part][k], gammas[part][k])
                first = (words[0][k], words[1][k])
                second = (words[2][k], words[3][k])
                sums01 = rt.FastMath.fma2(first, first, sums01)
                sums23 = rt.FastMath.fma2(second, second, sums23)
                muls.append(rt.FastMath.mul2(first, gamma))
                muls.append(rt.FastMath.mul2(second, gamma))
        squares = [sums01[0], sums01[1], sums23[0], sums23[1]]
        folded = rt.Reduce.fold(squares, lane, 16, (4, 2, 1))
        if (lane & 7) == 0:
            s.scratch[group * 32 + (lane >> 3 & 3) * 8 + warp] = folded
        fused = [(Float32(0.0), Float32(0.0))] * 4
        for part in cutlass.range_constexpr(3):
            column = part * 2048 + tid * 8
            first = rt.Convert.ld_bf16x8_words(s.ring_words, cdiv(column, 2))  # type: ignore[assignment]
            second = rt.Convert.ld_bf16x8_words(  # type: ignore[assignment]
                s.ring_words, cdiv(rt.HIDDEN + column, 2)
            )
            for k in cutlass.range_constexpr(8):
                pair0, pair1 = (first[k], first[k]), (second[k], second[k])
                for j in cutlass.range_constexpr(4):
                    fused[j] = rt.FastMath.fma2(
                        muls[part * 16 + 2 * k + j // 2],
                        (pair0, pair1)[j % 2],
                        fused[j],
                    )
        acc = cute.make_rmem_tensor(8, Float32)
        for j in cutlass.range_constexpr(4):
            acc[j % 2 + 4 * (j // 2)], acc[j % 2 + 2 + 4 * (j // 2)] = fused[j]
        folded = rt.Reduce.fold([acc[i] for i in range(8)], lane, 16, (2, 1))
        if (lane & 3) == 0:
            s.scratch[64 + warp * 16 + group * 8 + (lane >> 2 & 7)] = folded

    @cute.jit
    def router(self, p: MoeParams, s: MoeState, gammas):
        rt.mbar_wait(s.ring_full, 0)
        for group in cutlass.range_constexpr(2):
            self.router_group(s, gammas, group)
        s.pages.release(0)
        rt.bar_sync(8, 256)
        rt.bar_sync(11, 384)
        epoch = s.epoch_slot[0]
        if s.tid < 16:
            square = rt.Reduce.strided_sum(s.scratch, cdiv(s.tid, 2) * 8, 1)
            total = rt.Reduce.strided_sum(s.scratch, 64 + s.tid, 16)
            self.publish_logit(
                p, s, rt.mul(total, rt.Reduce.rms_inverse(square)), epoch
            )
        if s.tid == 0:
            self.count_router_slice(p, s)
        inverses = Rms.inverses(s.scratch, self.ROWS)
        for token in cutlass.range_constexpr(self.ROWS):
            for part in cutlass.range_constexpr(3):
                off = token * NORM_ROW + part * 2048 + s.tid * 8
                words = rt.Convert.ld_bf16x8_words(s.norm_words, cdiv(off, 2))
                Rms.store(s.norm_addr + off * 2, words, inverses[token], gammas[part])
        rt.bar_sync(11, 384)
        return epoch

    @cute.jit
    def live_rows(self, p: MoeQ8Params):
        if cutlass.const_expr(self.DEAD_STRIDE > 0):
            return Int32(self.DEAD_LIVE)
        return self.active_rows(p)

    @cute.jit
    def score_and_route(self, p: MoeParams, s: MoeState, early_bias, live):
        denominators = cute.make_rmem_tensor(self.ROWS, Float32)
        for token in cutlass.range_constexpr(self.ROWS):
            denominators[token] = rt.add(
                1.0,
                cute.math.exp2(
                    rt.mul(-s.logits[token * 256 + s.tid], LOG2E), fastmath=True
                ),
            )
        s.first_lane[s.tid] = Q8Shape.MAX_UNION
        for token in cutlass.range_constexpr(self.ROWS):
            s.logits[token * 256 + s.tid] = rt.FastMath.rcp_rn(denominators[token])
        for part in cutlass.range_constexpr(2):
            s.union_route[s.tid + part * 256] = 0.0
        rt.bar_sync(8, 256)
        views = (
            (s.logits, p.CorrectionBias),
            s.logits,
            s.route_ids,
            s.route_weights,
            p.RouteIds,
            p.RouteWeights,
        )
        if s.warp < live:
            self.route_token(
                views, s.warp * 256, s.lane, s.warp, s.bid, s.route_full, s.warp
            )
        else:
            if s.lane < TOP_K:
                s.route_ids[s.warp * TOP_K + s.lane] = 0
        rt.bar_sync(8, 256)

    @cute.jit
    def top8_candidates(self, corrected, base, lane):
        scores, bias = corrected
        local = cute.make_rmem_tensor(8, Float32)
        for item in cutlass.range_constexpr(8):
            value = rt.add(scores[base + lane + 32 * item], bias[lane + 32 * item])
            local[item] = -FLT_MAX
            if rt.FastMath.fcmp(">=", value, -FLT_MAX) & rt.FastMath.fcmp(
                "<=", value, FLT_MAX
            ):
                local[item] = value
        return local

    @cute.jit
    def route(self, p: MoeParams, s: MoeState, early_bias):
        live = self.live_rows(p)
        self.score_and_route(p, s, early_bias, live)
        if s.tid < live * TOP_K:
            expert = s.route_ids[s.tid]
            cute.arch.atomic_min(s.first_lane.iterator + expert, s.tid)
        rt.bar_sync(8, 256)
        if s.tid < Q8Shape.MAX_UNION:
            first = s.first_lane[s.route_ids[s.tid]]
            ballot = Uint32(cute.arch.vote_ballot_sync(first == s.tid))
            if s.lane == 0:
                s.owners[s.warp] = ballot
        rt.bar_sync(8, 256)
        if s.tid < Q8Shape.MAX_UNION:
            expert = s.route_ids[s.tid]
            first = s.first_lane[expert]
            index, count = Int32(0), Int32(0)
            for word in cutlass.range_constexpr(2):
                bits = s.owners[word]
                below = first - 32 * word
                if below >= 32:
                    index = index + Int32(cute.arch.popc(bits))
                if (below >= 0) & (below < 32):
                    mask = (Uint32(1) << Uint32(below)) - 1
                    index = index + Int32(cute.arch.popc(bits & mask))
                count = count + Int32(cute.arch.popc(bits))
            if first == s.tid:
                s.union_ids[index] = expert
            if s.tid < live * TOP_K:
                row = cdiv(s.tid, 8)
                s.union_route[index * self.ROWS + row] = s.route_weights[s.tid]
            if s.tid == 0:
                s.union_count[0] = count
        rt.bar_sync(8, 256)
        with rt.warp_leader(s.warp, 0):
            rt.mbar_arrive(s.union_full)
        count = s.union_count[0]
        return count, cdiv(count - cdiv(s.bid, 16) + 7, 8)

    def activation_index(self, s: MoeState, token, row, round_index, shared: bool):
        logical = (
            (s.bid - ROUTED_CTAS) * 16 + row if shared else cmod(s.bid, 16) * 16 + row
        )
        element = ActPlane.col(logical)
        word = element >> 1
        swapped = (
            (word & ~30)
            | ((word >> 3 & 1) << 4)
            | ((word >> 4 & 1) << 3)
            | ((word >> 1 & 1) << 2)
            | ((word >> 2 & 1) << 1)
        )
        plane = 0 if shared else 1 + cdiv(s.bid, 16) + 8 * round_index
        return (plane * self.ROWS + token) * SHARD + (swapped << 1) + (element & 1)

    def global_b(self, p: MoeParams, s: MoeState, plane):
        row = (plane * self.ROWS + cdiv(s.lane, 4)) * SHARD
        chunk = (4 * s.warp + cmod(s.lane, 4)) * 8
        return rt.Memory.ld_cg_v4(rt.ga(p.Activation, row + chunk), memory=True)

    @cute.jit
    def shared_down(self, p: MoeParams, s: MoeState, total):
        pair = self.global_b(p, s, 0)
        rt.mbar_wait(s.shared_down_full, 0)
        off = self.SHARED_DOWN + DownPlane.pad(s.ring_u64_addr, self.SHARED_DOWN)
        rows = DownPlane.rows(s.ring_u64, off, s.warp, s.lane)
        partial = rt.zeros(12)
        for kk in cutlass.range_constexpr(2):
            b = (pair[2 * kk], pair[2 * kk + 1])
            for m in cutlass.range_constexpr(3):
                frag = rt.Convert.a_frag_u64(rows[2 * m], rows[2 * m + 1], kk)
                rt.TensorCore.mma_acc(
                    cute.local_tile(partial, (4,), (m,)), frag, b, False
                )
        for m in cutlass.range_constexpr(3):
            scale = s.ring_f32[
                DownPlane.scale_word(off + DownPlane.RECORD >> 2, s.bid, s.warp, m)
            ]
            for i in cutlass.range_constexpr(4):
                total[4 * m + i] = rt.fma(partial[4 * m + i], scale, total[4 * m + i])

    @cute.jit
    def down(self, p: MoeParams, s: MoeState, count, rounds):
        total = rt.zeros(12)
        rt.mbar_wait(s.planes_full, 0)
        pair0 = cute.make_rmem_tensor(4, Uint32)
        pair1 = cute.make_rmem_tensor(4, Uint32)
        first = self.global_b(p, s, 1)
        second = self.global_b(p, s, 1 + cutlass.min(1, count - 1))
        for item in cutlass.range_constexpr(4):
            pair0[item], pair1[item] = first[item], second[item]
        self.shared_down(p, s, total)
        for page in range(cdiv(count + 1, 2)):
            current0 = cute.make_rmem_tensor(4, Uint32)
            current1 = cute.make_rmem_tensor(4, Uint32)
            for item in cutlass.range_constexpr(4):
                current0[item], current1[item] = pair0[item], pair1[item]
            if page + 1 < cdiv(count + 1, 2):
                if cmod(page + 1, 4) == 0:
                    rt.mbar_wait(s.planes_full + cdiv(page + 1, 4) * 8, 0)
                first = self.global_b(p, s, 2 * page + 3)
                second = self.global_b(p, s, 1 + cutlass.min(2 * page + 3, count - 1))
                for item in cutlass.range_constexpr(4):
                    pair0[item], pair1[item] = first[item], second[item]
            self.down_page(
                s, total, count, 1 + 6 * rounds + page, page, current0, current1
            )
        return total

    @cute.jit
    def down_page(self, s: MoeState, total, count, sequence, page, pair0, pair1):
        slot = cmod(sequence, self.RING_SLOTS)
        rt.mbar_wait(s.ring_full + slot * 8, cmod(cdiv(sequence, self.RING_SLOTS), 2))
        base = slot * self.SLOT_BYTES + DownPlane.pad(
            s.ring_u64_addr, slot * self.SLOT_BYTES
        )
        scales = cute.make_rmem_tensor(6, Float32)
        rows = []
        for half in cutlass.range_constexpr(2):
            rows += DownPlane.rows(
                s.ring_u64, base + half * DownPlane.RECORD, s.warp, s.lane
            )
            for m in cutlass.range_constexpr(3):
                scales[half * 3 + m] = s.ring_f32[
                    DownPlane.scale_word(
                        base + DownPlane.SCALES + 32 * half >> 2, s.bid, s.warp, m
                    )
                ]
        s.pages.release(slot)
        partials = (rt.zeros((4, 3)), rt.zeros((4, 3)))
        for kk in cutlass.range_constexpr(2):
            for half in cutlass.range_constexpr(2):
                pair = (pair0, pair1)[half]
                b = (pair[2 * kk], pair[2 * kk + 1])
                for m in cutlass.range_constexpr(3):
                    frag = rt.Convert.a_frag_u64(
                        rows[6 * half + 2 * m], rows[6 * half + 2 * m + 1], kk
                    )
                    rt.TensorCore.mma_acc(partials[half][None, m], frag, b, False)
        route = page * 2 * self.ROWS + cmod(s.lane, 4) * 2
        DownPlane.accum(
            total, partials[0], scales, 0, s.union_route, (route, route + 1)
        )
        if page * 2 + 1 < count:
            route = route + self.ROWS
            DownPlane.accum(
                total, partials[1], scales, 3, s.union_route, (route, route + 1)
            )

    @cute.jit
    def load_rounds(
        self, p: MoeParams, s: MoeState, epoch, lane_0, polled, count, rounds
    ):
        if s.warp == 8:
            for page in range(cdiv(count + 1, 2)):
                self.routed_down_page(
                    p, s, 1 + 6 * rounds + page, page, count, s.union_ids
                )
            if lane_0 < 16:
                self.clear_stale_logits(p.Flags, epoch, s.bid, lane_0, 0)
        else:
            if s.warp == 11:
                for published in range(rounds):
                    rt.mbar_wait(s.round_done + published * 8, 0)
                    with rt.warp_leader(s.warp, 11):
                        rt.Memory.fence_release_gpu()
                        bump = Uint32(1)
                        if published == rounds - 1:
                            bump = Uint32(Q8Shape.ROUNDS + 1 - rounds)
                        base = self.Flag.PAD_FIRST + self.Flag.PAD_ROUND + s.bid
                        cute.arch.atomic_add(p.Flags.iterator + base, bump)
                        cute.arch.atomic_add(
                            p.Flags.iterator + base + self.Flag.MIRROR, bump
                        )
                        cute.arch.atomic_add(
                            p.Flags.iterator + self.Flag.ROUND + s.bid, bump
                        )
                        if published == rounds - 1:
                            cute.arch.atomic_add(
                                p.Flags.iterator + self.Flag.ACT + s.bid, Uint32(1)
                            )
            if s.warp == 9:
                lane = cmod(lane_0, 32)
                for round_index in range(cdiv(count + 7, 8)):
                    target = epoch * Q8Shape.ROUNDS + Uint32(round_index) + 1
                    done = Int32(0)
                    while done == 0:
                        satisfied = Int32(1)
                        for part in cutlass.range_constexpr(4):
                            cta = lane + 32 * part
                            seen = rt.Memory.ld_relaxed_gpu(
                                rt.ga(p.Flags, polled + self.Flag.PAD_ROUND + cta)
                            )
                            if count > cdiv(cta, 16) + 8 * round_index:
                                satisfied = satisfied & Int32(Int32(seen - target) >= 0)
                        if (round_index == 0) & (lane < 16):
                            seen = rt.Memory.ld_relaxed_gpu(
                                rt.ga(p.Flags, polled + self.Flag.PAD_SHARED + lane)
                            )
                            satisfied = satisfied & Int32(
                                Int32(seen - (epoch + 1)) >= 0
                            )
                        done = Int32(cute.arch.vote_all_sync(satisfied != 0))
                    rt.Memory.fence_acquire_gpu()
                    cute.arch.sync_warp()
                    rt.mbar_arrive(s.planes_full + round_index * 8)
            if (lane_0 >= 32) & (lane_0 < 48):
                self.clear_stale_logits(p.Flags, epoch, s.bid, lane_0 - 32, 1)


class FusedMoeQ8Seam(FusedMoeSeam, FusedMoeQ8):
    """q8 computation with paired seam receivers, one warp per four-row group."""

    Params = MoeQ8SeamParams  # type: ignore[assignment]
