# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 fused MoE decode kernels in CuTe DSL."""

from abc import abstractmethod
from dataclasses import dataclass, replace

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, Uint64

from . import _rt as rt

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
class ColocatedExpertLayout:
    """Byte layout of one rank's colocated experts (routed, then the shared expert's
    slot), as the kernels read it.
    """

    routed_experts: int = EXPERTS
    hidden: int = rt.HIDDEN
    shard: int = SHARD
    block: int = 128
    alignment: int = 128
    prefetch_line: int = 128

    def _aligned(self, offset: int) -> int:
        return -(-offset // self.alignment) * self.alignment

    @property
    def slots(self) -> int:
        """Expert slots: the routed experts and the shared expert."""
        return self.routed_experts + 1

    @property
    def shared_slot(self) -> int:
        """The shared expert's slot."""
        return self.routed_experts

    @property
    def w13_shape(self) -> tuple[int, int]:
        """One expert's ``w13``: gate rows then up rows, by hidden."""
        return (2 * self.shard, self.hidden)

    @property
    def w2_shape(self) -> tuple[int, int]:
        """One expert's ``w2``: hidden rows by the rank's intermediate columns."""
        return (self.hidden, self.shard)

    @property
    def w13_scale_shape(self) -> tuple[int, int]:
        """One expert's ``w13`` block scales."""
        return (2 * self.shard // self.block, self.hidden // self.block)

    @property
    def w2_scale_shape(self) -> tuple[int, int]:
        """One expert's ``w2`` block scales."""
        return (self.hidden // self.block, self.shard // self.block)

    @property
    def w13_expert_bytes(self) -> int:
        """Bytes of one expert's E4M3 ``w13``: the expert stride of ``routed_up``."""
        return self.w13_shape[0] * self.w13_shape[1]

    @property
    def w2_expert_bytes(self) -> int:
        """Bytes of one expert's E4M3 ``w2``: the expert stride of the ``w2`` region."""
        return self.w2_shape[0] * self.w2_shape[1]

    @property
    def w13_scale_expert_bytes(self) -> int:
        """Bytes of one expert's FP32 ``w13`` block scales."""
        return 4 * self.w13_scale_shape[0] * self.w13_scale_shape[1]

    @property
    def w2_scale_expert_bytes(self) -> int:
        """Bytes of one expert's FP32 ``w2`` block scales."""
        return 4 * self.w2_scale_shape[0] * self.w2_scale_shape[1]

    @property
    def w13_expert_lines(self) -> int:
        """Prefetch lines of one expert's ``w13``."""
        return self.w13_expert_bytes // self.prefetch_line

    @property
    def w2_scale_offset(self) -> int:
        """Offset of the FP32 ``w2`` scales ``[slots, 48, 2]`` in ``routed_down`` (its
        start).
        """
        return 0

    @property
    def w13_scale_offset(self) -> int:
        """Offset of the FP32 ``w13`` scales ``[slots, 4, 48]`` in ``routed_down``."""
        return self._aligned(
            self.w2_scale_offset + self.slots * self.w2_scale_expert_bytes
        )

    @property
    def w2_offset(self) -> int:
        """Offset of slot 0's ``w2`` in ``routed_down`` (the header's size)."""
        return self._aligned(
            self.w13_scale_offset + self.slots * self.w13_scale_expert_bytes
        )

    def routed_up_bytes(self, slots: int) -> int:
        """Bytes of a ``routed_up`` buffer of ``slots`` experts."""
        return slots * self.w13_expert_bytes

    def routed_down_bytes(self, slots: int) -> int:
        """Bytes of a ``routed_down`` buffer of ``slots`` experts (header included)."""
        return self.w2_offset + slots * self.w2_expert_bytes


#: The colocated-expert layout the MoE kernels and the worker's MoE prefetch read.
LAYOUT = ColocatedExpertLayout()


class Flag:
    """``Flags`` (u32 words)."""

    ACT = 144
    PLANES = 288
    LOCAL_O = 432
    DONE = 576
    ROUND = 11601
    PAD_FIRST = 11732
    PAGE = 1024
    PAD_SHARED = 128
    PAD_ROUND = 256
    #: Per-SM copy selector.
    SM_DOMAIN = PAD_FIRST + 3 * PAGE
    #: Offset of the second copy.
    MIRROR = 1 << 21


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
        """Bytes from ``base + off`` up to the next 1024-byte boundary."""
        return 1024 - (base + off & 1023) & 1023

    @staticmethod
    def scales(expert):
        """Offset of ``expert``'s FP32 ``w2`` block scales in the ``routed_down``
        header.
        """
        return LAYOUT.w2_scale_offset + expert * LAYOUT.w2_scale_expert_bytes

    @staticmethod
    def lane_word(warp, lane):
        """The lane's first u64 row word in a swizzled 1024-aligned record."""
        group = cdiv(lane, 4)
        chunk = cmod(warp, 4) + 4 * (cmod(cdiv(lane, 2), 2)) ^ group
        return cdiv(cdiv(warp, 4) * DownPlane.BOX + group * 128 + chunk * 16, 8) + cmod(
            lane, 2
        )

    @staticmethod
    def rows(ring_u64, byte_base, warp, lane) -> list:
        """The lane's six u64 row words of the record at ring byte offset
        ``byte_base``.
        """
        first = (byte_base >> 3) + DownPlane.lane_word(warp, lane)
        return [ring_u64[first + 128 * r] for r in range(6)]

    @staticmethod
    def k_words(warp, lane):
        """The lane's activation word offset (mma B fragment) within one down k tile."""
        return (
            cdiv(warp, 4) * 64
            + cmod(cdiv(warp, 2), 2) * 32
            + cmod(cdiv(lane, 2), 2) * 16
            + cmod(warp, 2) * 8
            + cmod(lane, 2) * 2
        )

    @staticmethod
    def scale_word(block, bid, warp, k):
        """Block-scale word of output rows bid * 48 + 16 * k."""
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
        """Total += part * scale * route weight (the lane's two token columns)."""
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
        """Stored column of expert row ``e`` (the swizzle of the down GEMV's B
        fragments).
        """
        return (e & 147) | (e & 64) >> 1 | (e & 32) << 1 | (e & 8) >> 1 | (e & 4) << 1


class Rms:
    """RMSNorm pieces."""

    @staticmethod
    def gamma_load(Gamma, tid) -> list:
        """The thread's three uint4 slices (24 bf16) of the RMSNorm weight row."""
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
        """bf16(x * inverse * gamma) of 8 values stored as one 16-byte shared vector."""
        rt.Memory.sts_v4(addr, rt.Reduce.norm_bf16x8(values, inverse, gamma))

    @staticmethod
    def inverses(scratch, rows: int = 4) -> list:
        """rsqrt(mean square + 1e-5) of the token rows from the 8 warps' staged
        squares.
        """
        totals = [
            rt.FastMath.div_rn(rt.Reduce.strided_sum(scratch, t * 8, 1), 6144.0)
            for t in range(rows)
        ]
        return [rt.FastMath.rsqrt(rt.add(x, 1e-05)) for x in totals]


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class MoeSeamParams(MoeParams):
    """Seam parameters: also the next layer's RMSNorm weight and its gamma-scaled
    input.
    """

    GammaNext: cute.Pointer = rt.ptr("float32")
    XG: cute.Pointer = rt.ptr("bfloat16")


@dataclass(frozen=True)
class MoeState:
    """One CTA's state: thread ids, the shared-memory base and the weight ring."""

    base: Int32
    pages: rt.Pipeline.Ring
    tid: Int32 = None
    warp: Int32 = None
    lane: Int32 = None
    bid: Int32 = None


def _bind(s: MoeState, ids: rt.ThreadIds) -> MoeState:
    return replace(s, tid=ids.tid, warp=ids.warp, lane=ids.lane, bid=ids.bid)


class FusedMoe(rt.UpDownMapKernel):
    """Stages shared by every variant; subclasses override the differing ones."""

    ROWS: int
    #: Expert slots the descriptors cover (slot 256: the colocated shared expert).
    SLOTS = LAYOUT.routed_experts
    SEAM = False
    Params = MoeParams
    GRID = 144
    ROLES = (rt.Role("compute", 0, 7, 216, True), rt.Role("load", 8, 11, 72, False))
    Smem: type[rt.SmemLayout]
    SLOT_BYTES: int
    PARTIAL_STRIDE: int

    @property
    def mailbox(self) -> rt.SliceExchange:
        """The variant's TP8 slice exchange (one token row per ROWS)."""
        return rt.SliceExchange(self.ROWS, self.SEAM)

    def mbarriers(self) -> tuple:
        """The mbarrier init groups, in address order."""
        return self.Smem.barriers()

    def tensor_maps(self, p) -> tuple:
        """Host wrapper: the Up pages' descriptor over ``routed_up``, the Down records'
        over the ``w2`` region.
        """
        hidden, w13, w2 = rt.HIDDEN, LAYOUT.w13_expert_bytes, LAYOUT.w2_expert_bytes
        up_dims, up_box = (
            [256, hidden // 256, 2 * SHARD, self.SLOTS],
            [256, UpPage.BOX_CHUNKS, UpPage.BOX_ROWS, 1],
        )
        down = p.RoutedDown.toint() + LAYOUT.w2_offset
        return (
            rt.byte_map(p.RoutedUp.toint(), up_dims, [256, hidden, w13], up_box),
            rt.byte_map(
                down, [SHARD, hidden, self.SLOTS], [SHARD, w2], [128, COLUMNS, 1], True
            ),
        )

    def smem(self, base):
        """The CTA state record over the shared-memory base (thread ids are bound per
        role).
        """
        return self.Smem.record(base, ring="pages", bases=(MoeState,))

    @cute.jit
    def compute(self, p: MoeParams, s: MoeState, ids: rt.ThreadIds):
        """Compute role (warps 0-7)."""
        self.compute_cta(p.views(), _bind(s, ids))

    @cute.jit
    def load(self, p: MoeParams, s: MoeState, ids: rt.ThreadIds):
        """Load role (warps 8-11); warp 8 prefetches the two descriptors before the PDL
        wait.
        """
        p, s = p.views(), _bind(s, ids)
        self.set_regs("load")
        with rt.warp_leader(s.warp, 8):
            rt.AsyncCopy.prefetch_tensormap(self.up_map)
            rt.AsyncCopy.prefetch_tensormap(self.down_map)
        self.epoch_ticket(s.warp, p.FullEpoch.iterator, s.epoch_slot)
        self.load_prologue(p, s)
        lane_0 = s.tid - 256
        rt.bar_sync(11, 384)
        epoch_ld = s.epoch_slot[0]
        self.load_cta(p, s, epoch_ld, lane_0)

    @abstractmethod
    def compute_cta(self, p: MoeParams, s: MoeState):
        """Compute role body."""
        raise NotImplementedError

    @abstractmethod
    def load_prologue(self, p: MoeParams, s: MoeState):
        """Load role before the epoch."""
        raise NotImplementedError

    @abstractmethod
    def load_cta(self, p: MoeParams, s: MoeState, epoch_ld, lane_0):
        """Load role after the epoch: routed pages, activation planes, cross-CTA
        polls.
        """
        raise NotImplementedError

    @abstractmethod
    def shared_up_page(self, p: MoeParams, s: MoeState, page):
        """Issue the shared expert's gate/up ``page`` into ring slot ``page % 4``
        (shared CTAs).
        """
        raise NotImplementedError

    @cute.jit
    def shared_up_pages(
        self,
        p: MoeParams,
        s: MoeState,
        first: cutlass.Constexpr,
        last: cutlass.Constexpr,
    ):
        """Shared-expert gate/up pages first..last-1."""
        for page in range(first, last):
            self.shared_up_page(p, s, page)

    @cute.jit
    def up_pages(self, p: MoeParams, s: MoeState, first, expert, row16):
        """The six gate/up pages of routed expert ``expert``."""
        for page in range(0, 6):
            self.up_page(p, s, first + page, page, expert, row16)

    @cute.jit
    def rms_norm(self, p: MoeParams, s: MoeState, gammas):
        """RMSNorm of the ROWS rows in place."""
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
        if cutlass.const_expr(rows == 1):
            square = squares[0]
            for step in cutlass.range_constexpr(5):
                square = rt.add(square, cute.arch.shuffle_sync_down(square, 16 >> step))
            if lane == 0:
                s.scratch[warp] = square
        else:
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
        """Sigmoid scores, bias-corrected copies, then per-token top-8 and route
        weights.
        """
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
            self.route_token(routes, s.warp * 256, s.lane, s.warp, s.bid, s.route_full)
        if tid < 128:
            s.union_route[tid] = 0.0
        rt.bar_sync(8, 256)

    @cute.jit
    def gate_up_slice(
        self, p: MoeParams, s: MoeState, round_index, shared: cutlass.Constexpr
    ):
        """Gate/up of one 16-row slice: 6 ring pages, cross-warp sum, silu(gate) *
        up.
        """
        accum = rt.zeros(8)
        lane = s.lane
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
        """Page ``page`` of a gate/up slice (see :meth:`gate_up_slice`)."""
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
        """Sum the warps' gate/up partials and store silu(gate) * up."""
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
        """Activation element of (token, slice row) of this CTA's slice."""
        if shared:
            return token * 256 + ActPlane.col((s.bid - ROUTED_CTAS) * 16 + row)
        plane = 1 + cdiv(s.bid, 16) + 8 * round_index
        return (plane * self.ROWS + token) * 256 + ActPlane.col(
            cmod(s.bid, 16) * 16 + row
        )

    @cute.jit
    def store_partials(self, s: MoeState, accum, warp, lane):
        """Stage the warp's gate/up accumulators for the cross-warp sum."""
        self.stage_partials(
            s.scratch,
            accum,
            warp * self.PARTIAL_STRIDE + cmod(lane, 4) * 32 + cdiv(lane, 4),
        )

    def silu_up(self, scratch, token, row):
        """silu(gate) * up of one (token, row), summing the 8 warps' partials."""
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

    @cute.jit
    def elect_add(
        self,
        Flags,
        warp,
        index,
        release: cutlass.Constexpr = False,
        scoped: cutlass.Constexpr = True,
    ):
        """One elected lane of warp 0 bumps Flags[index]."""
        with rt.warp_leader(warp, 0):
            if cutlass.const_expr(release):
                rt.Memory.fence_release_gpu()
            if cutlass.const_expr(scoped):
                rt.Memory.atomic_add_u32(Flags, index, 1)
            else:
                cute.arch.atomic_add(Flags.iterator + index, Uint32(1))

    @cute.jit
    def poll_all(self, Flags, owned, index, epoch):
        """Warp-wide spin until every owned lane reads Flags[index] > epoch."""
        done = Int32(0)
        while done == 0:
            satisfied = Int32(1)
            if owned != 0:
                observed = rt.Memory.ld_relaxed_gpu(rt.ga(Flags, index))
                satisfied = satisfied & Int32(observed >= epoch + 1)
            done = Int32(cute.arch.vote_all_sync(satisfied != 0))

    def stage_partials(self, scratch, accum, base):
        """Stage one token's gate and up accumulators at scratch[base]."""
        for m in range(2):
            scratch[base + m * 16] = accum[m * 4]
            scratch[base + m * 16 + 8] = accum[m * 4 + 2]

    @cute.jit
    def store_finite(self, dst, idx, value):
        """dst[idx] = value if it is finite, else -FLT_MAX (the router's corrected-score
        guard).
        """
        if rt.FastMath.fcmp(">=", value, -FLT_MAX) & rt.FastMath.fcmp(
            "<=", value, FLT_MAX
        ):
            dst[idx] = value
        else:
            dst[idx] = -FLT_MAX

    @cute.jit
    def route_token(self, views, base, lane, warp, bid, route_full_addr):
        """Top-8 of one token's scores and its normalized route weights."""
        corrected, scores, route_ids, route_weights, RouteIds, RouteWeights = views
        self.top8(corrected, base, lane, warp, bid, route_ids, route_full_addr)
        cute.arch.sync_warp()
        route = warp * 8 + cmod(lane, 8)
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
    def top8(self, corrected, base, lane, warp, bid, route_ids, route_full_addr):
        """Top-8 by repeated warp max (ties to the lowest id); releases the loader's
        slot.
        """
        local = cute.make_rmem_tensor(8, Float32)
        for item in cutlass.range_constexpr(8):
            local[item] = corrected[base + lane + 32 * item]
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
                route_ids[warp * 8 + slot] = best_id
            if (warp == 0) & (cdiv(bid, 16) == slot):
                with rt.warp_leader(warp, 0):
                    route_ids[slot] = best_id
                    rt.mbar_arrive(route_full_addr)

    @cute.jit
    def gate_up_page(self, s: MoeState, accum, sequence, k_words, warp, lane):
        """One K page of a 16-row gate/up slice (8 k-steps x gate, up)."""
        slot_bytes = self.SLOT_BYTES
        slot = cmod(sequence, 4)
        rt.mbar_wait(s.ring_full + slot * 8, cmod(cdiv(sequence, 4), 2))
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
        norm = s.norm_words_addr
        b_words = lambda k: rt.Memory.lds_weak_b64(
            norm + (k_words + k % 4 * 2 + k // 4 * 32) * 4
        )
        rt.Mma.gate_up_steps(accum, up_rows, b_words, gate_scale, up_scale)
        s.pages.release(slot)

    @cute.jit
    def exchange(self, p: MoeParams, s: MoeState, epoch, total_r):
        """Reduce the warps, write LocalO, tagged push to the 7 peers."""
        x = self.mailbox
        x.reduce_rows(
            s.scratch, s.local_o, p.LocalO, total_r, s.tid, s.warp, s.lane, s.bid
        )
        self.elect_add(p.Flags, s.warp, Flag.LOCAL_O + s.bid, scoped=False)
        x.push_peers(p, s.local_o_words, epoch, s.bid, s.tid, s.warp, s.lane)
        self.exchange_tail(p, s, epoch)

    @cute.jit
    def exchange_tail(self, p: MoeParams, s: MoeState, epoch):
        """Done flag, then warp 0 combines this CTA's slice."""
        self.elect_add(p.Flags, s.warp, Flag.DONE + s.bid, scoped=False)
        combining = s.warp == 0
        if combining:
            self.mailbox.combine(p, p.X, s.lane, s.bid, epoch, s.local_o_words)

    @cute.jit
    def routed_down(self, s: MoeState, total, count, first_down):
        """The routed down pages, two records per page."""
        route_lane = self.route_lane(s.lane)
        for page in range(cdiv(count + 1, 2)):
            self.routed_down_record_pair(s, total, count, first_down, page, route_lane)

    @cute.jit
    def routed_down_record_pair(
        self, s: MoeState, total, count, first_down, page, route_lane
    ):
        """Down page ``page``: rows, scales, admission, mma, route-weighted
        accumulation.
        """
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
        """Wait until the activation planes of down page ``page`` are in shared
        memory.
        """
        raise NotImplementedError

    @abstractmethod
    def down_plane_word(self, expert):
        """First act_words word of routed expert ``expert``'s plane (-1: shared)."""
        raise NotImplementedError

    @abstractmethod
    def down_lane_words(self, warp, lane):
        """The lane's mma B word offset within a plane (k tile 0)."""
        raise NotImplementedError

    @abstractmethod
    def route_lane(self, lane):
        """The lane's first C-fragment column in a route-weight row."""
        raise NotImplementedError

    @abstractmethod
    def route_pair(self, s: MoeState, expert, route_lane):
        """Route weights of routed expert ``expert`` for the lane's two C-fragment
        columns.
        """
        raise NotImplementedError

    @cute.jit
    def publish_logit(self, p: MoeParams, s: MoeState, logit, epoch):
        """Publish RouterLogits and the tagged logit word."""
        tid, bid, Flags = s.tid, s.bid, p.Flags
        p.RouterLogits[cdiv(tid, 2) * 256 + bid * 2 + cmod(tid, 2)] = logit
        bits = logit.bitcast(Uint32)
        if bits == Uint32(2147483648):
            bits = Uint32(0)
        word = cdiv(tid, 2) * 256 + bid * 2 + cmod(tid, 2)
        parity = Int32(cmod(epoch, 2))
        filled = Flag.PAD_FIRST + (1 + parity) * Flag.PAGE + word
        rt.Memory.st_relaxed_sys(rt.ga(Flags, filled), bits ^ Uint32(2147483648))
        rt.Memory.st_relaxed_sys(
            rt.ga(Flags, filled + Flag.MIRROR), bits ^ Uint32(2147483648)
        )

    @cute.jit
    def clear_stale_logits(self, Flags, epoch, bid, i, copy: cutlass.Constexpr):
        """Zero this CTA's logit word ``i`` of the next launch's page."""
        # Called at the loader branch ends.
        word = cdiv(i, 2) * 256 + bid * 2 + cmod(i, 2)
        parity = Int32(cmod(epoch, 2))
        rt.Memory.st_relaxed_sys(
            rt.ga(
                Flags,
                Flag.PAD_FIRST + (2 - parity) * Flag.PAGE + copy * Flag.MIRROR + word,
            ),
            0,
        )

    @cute.jit
    def count_router_slice(self, p: MoeParams, s: MoeState):
        """This CTA's router-slice counters."""
        Flags, bid = p.Flags, s.bid
        base = Flag.PAD_FIRST + bid
        cute.arch.atomic_add(Flags.iterator + base, Uint32(1))
        cute.arch.atomic_add(Flags.iterator + (base + Flag.MIRROR), Uint32(1))
        cute.arch.atomic_add(Flags.iterator + bid, Uint32(1))

    @cute.jit
    def pull_logits(self, p: MoeParams, s: MoeState, epoch_ld, lane_0):
        """Poll the router-slice counters, then the logit words; returns the polled
        page.
        """
        Flags = p.Flags
        # This SM's copy.
        near = Int32(
            rt.Memory.ld_relaxed_gpu(
                rt.ga(Flags, Flag.SM_DOMAIN + Int32(rt.Sync.smid()))
            )
        )
        polled = Flag.PAD_FIRST + near * Flag.MIRROR
        slot_owned = Int32(lane_0 < 128)
        slot_index = polled + cutlass.min(lane_0, 127)
        self.poll_all(Flags, slot_owned, slot_index, epoch_ld)
        rt.bar_sync(9, 128)
        tagged_base = (
            Flag.PAD_FIRST
            + (1 + Int32(cmod(epoch_ld, 2))) * Flag.PAGE
            + (1 - near) * Flag.MIRROR
        )
        self.poll_logit_words(Flags, s, tagged_base, lane_0, 2)
        return polled

    @cute.jit
    def poll_logit_words(
        self, Flags, s: MoeState, tagged_base, lane_0, halves: cutlass.Constexpr
    ):
        """Spin until the lane's logit words are all nonzero, then store them
        unflipped.
        """
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
    def up_page(self, p: MoeParams, s: MoeState, sequence, page, expert, row16):
        """One gate/up page of ``expert`` (two TMA tiles and their scales) into the
        ring.
        """
        slot_bytes = self.SLOT_BYTES
        slot = cmod(sequence, 4)
        with rt.warp_leader(s.warp, 8):
            s.pages.acquire(sequence, UpPage.TX, slot)
            base = Int64(expert) * LAYOUT.w13_scale_expert_bytes
            block = cdiv(row16, 8) * 48 + page * 8
            for h in cutlass.range_constexpr(2):
                box_dst = s.ring_addr + slot * slot_bytes + UpPage.BOX * h
                desc = self.up_map
                page_col, box_row = page * 4, 256 * h + row16 * 16
                full = s.ring_full + slot * 8
                rt.AsyncCopy.tma_4d(
                    box_dst, desc, 0, page_col, box_row, expert, full, _HINT
                )
                scale_dst = s.ring_addr + slot * slot_bytes + UpPage.SCALES + 32 * h
                scale_src = rt.ga(
                    p.RoutedDown,
                    base + (block + 96 * h) * 4 + LAYOUT.w13_scale_offset,
                )
                rt.bulk_g2s(scale_dst, scale_src, 32, s.ring_full + slot * 8)

    @cute.jit
    def routed_down_page(self, p: MoeParams, s: MoeState, sequence, page, count, ids):
        """One routed down page (two records) into ring slot sequence % 4."""
        slot_bytes = self.SLOT_BYTES
        slot = cmod(sequence, 4)
        with rt.warp_leader(s.warp, 8):
            rt.mbar_wait(s.ring_empty + slot * 8, cmod(cdiv(sequence, 4), 2) ^ 1)
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
                        desc, col = self.down_map, s.bid * COLUMNS
                        full = s.ring_full + slot * 8
                        rt.AsyncCopy.tma_3d(
                            box_dst, desc, 128 * h, col, record, full, _HINT
                        )
                    scale_dst = aligned + DownPlane.SCALES + within * 32
                    scales = DownPlane.scales(record)
                    scale_src = rt.ga(
                        p.RoutedDown,
                        scales + cdiv(cdiv(s.bid * COLUMNS, 128), 2) * 16,
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
    SLOTS = LAYOUT.slots
    Smem = MoeQ4Smem
    SLOT_BYTES = MoeQ4Smem.SLOT_BYTES
    SMEM_BYTES = MoeQ4Smem.SMEM_BYTES
    PARTIAL_STRIDE = 129
    SHARED_DOWN = MoeQ4Smem.SHARED_DOWN

    @cute.jit
    def shared_up_page(self, p: MoeParams, s: MoeState, page):
        """Shared-expert page through the Up descriptor (slot 256)."""
        self.up_page(p, s, page, page, LAYOUT.shared_slot, s.bid - ROUTED_CTAS)

    @cute.jit
    def compute_cta(self, p: MoeParams, s: MoeState):
        """Shared-expert slice, or router, union rounds, down GEMVs and the exchange."""
        tid, warp, bid = s.tid, s.warp, s.bid
        with rt.warp_leader(warp, 0):
            cute.arch.griddepcontrol_wait()
            rt.mbar_expect_tx(s.x_full, 4 * 2 * rt.HIDDEN)
            for token in cutlass.range_constexpr(4):
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
                padded = Flag.PAD_FIRST + bid
                rt.Memory.atomic_add_u32(p.Flags, padded, 1)
                rt.Memory.atomic_add_u32(p.Flags, padded + Flag.MIRROR, 1)
                rt.Memory.atomic_add_u32(p.Flags, Flag.ACT + bid, 1)
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
        """Shared CTAs after the activations: the done flag."""
        self.elect_add(p.Flags, s.warp, Flag.DONE + s.bid)

    @cute.jit
    def router(self, p: MoeParams, s: MoeState, gammas):
        """RMS fused with the router GEMV; logits published; X normalized in place."""
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
        """Scores, per-token top-8, then the expert union and its round count."""
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
        """The shared-expert down tile, then the union's routed pages."""
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
        """Round page / 4's planes before its first page."""
        if (page > 0) & (cmod(page, 4) == 0):
            rt.mbar_wait(s.planes_full + (cdiv(page, 4)) * 8, 0)

    def down_plane_word(self, expert):
        """Plane 1 + e: 4 token rows in the reused normed rows (wrapping at plane 9)."""
        return (1 + expert - 9 + 33 * Int32(1 + expert < 9)) * 4 * ActPlane.ROW_WORDS

    def down_lane_words(self, warp, lane):
        """Token row lane / 4 % 4 of the plane, then the k-tile word."""
        return cmod(cdiv(lane, 4), 4) * ActPlane.ROW_WORDS + DownPlane.k_words(
            warp, lane
        )

    def route_lane(self, lane):
        """Tokens 2 * (lane % 4) % 4, +1."""
        return cmod(cmod(lane, 4) * 2, 4)

    def route_pair(self, s: MoeState, expert, route_lane):
        """The union expert's route weights of the lane's two tokens."""
        route = expert * 4 + route_lane
        return s.union_route, (route, route + 1)

    @cute.jit
    def load_prologue(self, p: MoeParams, s: MoeState):
        """PDL wait, shared down plane, up pages 0-3."""
        warp, bid = s.warp, s.bid
        cute.arch.griddepcontrol_wait()
        if bid < ROUTED_CTAS:
            with rt.warp_leader(warp, 8):
                rt.mbar_expect_tx(s.shared_down_full, DownPlane.TX)
                dst = s.ring_addr + (
                    self.SHARED_DOWN + DownPlane.pad(s.ring_addr, self.SHARED_DOWN)
                )
                for h_ld in cutlass.range_constexpr(2):
                    box_dst = dst + DownPlane.BOX * h_ld
                    desc = self.down_map
                    col, record, full = (
                        bid * COLUMNS,
                        LAYOUT.shared_slot,
                        s.shared_down_full,
                    )
                    rt.AsyncCopy.tma_3d(
                        box_dst, desc, 128 * h_ld, col, record, full, _HINT
                    )
                scale_dst = dst + DownPlane.RECORD
                scales = DownPlane.scales(LAYOUT.shared_slot)
                scale_src = rt.ga(
                    p.RoutedDown, scales + cdiv(cdiv(bid * COLUMNS, 128), 2) * 16
                )
                rt.bulk_g2s(scale_dst, scale_src, 32, s.shared_down_full)
        else:
            self.shared_up_pages(p, s, 0, 4)

    @cute.jit
    def load_cta(self, p: MoeParams, s: MoeState, epoch_ld, lane_0):
        """Shared up pages 4-5, or the routed CTAs' loader."""
        bid = s.bid
        if bid >= ROUTED_CTAS:
            self.shared_up_pages(p, s, 4, 6)
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
        """Warp 8: down pages; warp 11: round publishes; warps 9-11: activation
        planes.
        """
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
                        base_ld = Flag.PAD_FIRST + (Flag.PAD_ROUND + bid)
                        cute.arch.atomic_add(Flags.iterator + base_ld, bump)
                        cute.arch.atomic_add(
                            Flags.iterator + (base_ld + Flag.MIRROR), bump
                        )
                        cute.arch.atomic_add(Flags.iterator + (Flag.ROUND + bid), bump)
                        if published == rounds_ld - 1:
                            cute.arch.atomic_add(
                                Flags.iterator + (Flag.ACT + bid), Uint32(1)
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
                            rt.ga(Flags, polled + Flag.PAD_ROUND + cta)
                        )
                        if count_ld > cdiv(cta, 16) + 8 * round_index_ld:
                            satisfied = satisfied & Int32(Int32(seen - target) >= 0)
                    shared_seen = Uint32(epoch_ld + 1)
                    if (round_index_ld == 0) & (lane_w < 16):
                        shared_seen = rt.Memory.ld_relaxed_gpu(
                            rt.ga(Flags, polled + Flag.PAD_SHARED + lane_w)
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
                        Flags.iterator + (Flag.PLANES + bid), Uint32(1)
                    )
            if (lane_0 >= 32) & (lane_0 < 40):
                self.clear_stale_logits(Flags, epoch_ld, bid, lane_0 - 32, 1)


class FusedMoeSeam(FusedMoeQ4):
    """q4 seam: column-slice combine instead of the per-CTA pull."""

    SEAM = True
    Params = MoeSeamParams

    @cute.jit
    def shared_tail(self, p: MoeParams, s: MoeState, epoch):
        """Done flag, then the shared CTAs combine slices 0-15, 48-63, 96-111."""
        self.elect_add(p.Flags, s.warp, Flag.DONE + s.bid)
        cute.arch.griddepcontrol_launch_dependents()
        seam_slice = s.bid - ROUTED_CTAS + 48 * s.warp
        if seam_slice < 128:
            self.mailbox.combine(p, p.X, s.lane, seam_slice, epoch, None, True)

    @cute.jit
    def exchange_tail(self, p: MoeParams, s: MoeState, epoch):
        """Own partial into slot 7, done flag, then CTAs bid >= 96 combine the rest."""
        if s.warp == 7:
            peer = rt.Memory.select_peer(p.peer_rank, p.peer_bases)
            self.mailbox.push(
                s.local_o_words, epoch, s.bid, s.lane, peer, self.mailbox.OWN
            )
        self.elect_add(p.Flags, s.warp, Flag.DONE + s.bid, scoped=False)
        if s.bid >= 96:
            cute.arch.griddepcontrol_launch_dependents()
            routed_slice = s.bid - 96 + 16 + 48 * s.warp
            if routed_slice < 128:
                self.mailbox.combine(p, p.X, s.lane, routed_slice, epoch)
