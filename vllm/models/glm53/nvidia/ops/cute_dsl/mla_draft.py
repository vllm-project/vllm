# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL worker of GLM-5.3's specialized MTP drafter (layer 78): ``DraftWorker``."""

import dataclasses

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint16, Uint32, Uint64
from cutlass.base_dsl.typing import GridConstant
from cutlass.experimental.cuda.tensor_map import TensorMap

from . import _rt as rt
from .mla_worker import (
    BAR_ALL,
    BAR_COMPUTE,
    BAR_LOADER,
    L2_HINT,
    NEG_FLT_MAX,
    SELECTION_BYTES,
    SPLIT_KEYS,
    SPLITS,
    TOPK,
    FlagWords,
    Handoff,
    KvRows,
    MlaWorker,
    Ring,
    TailArgs,
    WorkerParams,
    WorkerSmem,
    _Args,
    const_expr,
    stage,
)
from .mla_worker import INPUT_PAGES as INPUT_MAP
from .mla_worker import KVB_PAGES as KVB_MAP
from .mla_worker import O_PAGES as O_MAP

#: The drafter worker's role number (after the target roles kFull .. kSeam).
kDraft = 5
CONTEXT_WORDS = 2
HANDOFF_ROW_WORDS = 14848
PARTIAL_O_ROW_BYTES = 524288
PARTIAL_LSE_ROW_BYTES = 1024
QUERY_MAP = (2048, 2048, 128, 32, 1024, True)
ABSORB_MAP = (3584, 512, 64, 192, 64, False)


@rt.record
class DraftHeadArgs(_Args):
    """The pointers before the KvCache tensor map (W_UK is read from ``KVbWeight``)."""

    X: cute.Pointer = rt.ptr("bfloat16")
    GammaX: cute.Pointer = rt.ptr("float32")
    InputWeight: cute.Pointer = rt.ptr("float8_e4m3")
    InputScales: cute.Pointer = rt.ptr("uint8")
    GammaQ: cute.Pointer = rt.ptr("float32")
    QueryWeight: cute.Pointer = rt.ptr("float8_e4m3")
    QueryScales: cute.Pointer = rt.ptr("uint8")
    GammaKV: cute.Pointer = rt.ptr("float32")
    Freq: cute.Pointer = rt.ptr("float32")
    CurPos: cute.Pointer = rt.ptr("int32")


@rt.record
class DraftTailArgs(TailArgs):
    """kDraft's tail: Extra = SelectRow (the accepted row A), SelectContext (its key
    count p + A + 1, then the generation of the selection's records).
    """

    READ_ONLY = TailArgs.READ_ONLY | dict.fromkeys(
        ("SelectRow", "SelectContext"), Int32
    )
    SelectRow: cute.Pointer = rt.ptr("int32")
    SelectContext: cute.Pointer = rt.ptr("int32")


@rt.record
class DraftParams(WorkerParams):
    """``WorkerParams`` plus the drafter's two row words."""

    SelectRow: "rt.ReadOnlyView | None" = None
    SelectContext: "rt.ReadOnlyView | None" = None
    QueryScales: "cute.Tensor | None" = None
    QueryTmap: Int64 | None = None
    AbsorbTmap: Int64 | None = None

    @staticmethod
    def of(KvCache, *records, **maps) -> "DraftParams":
        addresses = {"KvCache": KvCache.get_ptr().toint()}
        addresses.update({name: m.get_ptr().toint() for name, m in maps.items()})
        views = [r.views() for r in records]
        values = {
            f.name: getattr(v, f.name) for v in views for f in dataclasses.fields(v)
        }
        bases = tuple(values.pop(f"peer_base{k}") for k in range(8))
        return DraftParams(**values, AbsorbWeight=None, peer_bases=bases, **addresses)  # type: ignore[arg-type]


class DraftWorker(MlaWorker):
    """kDraft: the drafter's layer-78 worker of the rank's eight heads
    (``mla_worker_draft_q{GROUP_ROWS}_paged`` for one request, ``..._r{ROWS}_paged``
    for the catch-up of a request tier and ``..._p{ROWS}_paged`` for its proposals).
    """

    ROLE = kDraft
    RECORDS = (DraftHeadArgs, DraftTailArgs)
    GROUP_ROWS = 4
    HEADS, LSE_STRIDE, Q_CTAS, O_PAGES, SMEM_BYTES = 8, 8, 64, 4, 227456
    X_FULL, NORM_FREE, ABSORB_FULL = 128, 136, 144
    MBAR_GROUPS = ((128, 1, 1), (136, 1, 256), (144, 1, 1))
    #: Bytes of one O-projection act row (the heads' V/O columns plus a 16-byte pad).
    O_ROW_BYTES = HEADS * 512 + 16
    INPUT_SCALE_WORD = 33792 // 4
    QUERY_SCALE_WORD = (2 * Ring.SLOT_BYTES + 33792) // 4

    def __init__(self, **constants: int | bool) -> None:
        super().__init__(**constants)
        rows, heads = self.GROUP_ROWS, self.HEADS
        self.KV_BASE = rows * Handoff.Q_DOWN_ROW
        q_base = self.KV_BASE + rows * Handoff.KV_LATENT_ROW
        self.Q = (q_base, 192 * heads)
        self.PE = (q_base + rows * 192 * heads, 64 * heads)
        self.OPERANDS = self.PE[0] + rows * 64 * heads
        self.MERGED = self.OPERANDS + rows * 512 * heads
        self.VO = (self.MERGED + rows * 512 * heads, 256 * heads)

    @property
    def GROUP_STRIDES(self) -> dict[str, int]:  # type: ignore[override]
        rows = self.GROUP_ROWS
        return {
            "X": rows * 6144,
            "Hidden": rows * 6144,
            "Final": 6144,
            "Freq": rows * 64,
            "CurPos": 2 * rows,
            "SelectRow": 1,
            "SelectContext": CONTEXT_WORDS,
            "SelectionMailbox": SELECTION_BYTES // 4,
            "Handoff": rows * HANDOFF_ROW_WORDS,
            "PartialO": rows * PARTIAL_O_ROW_BYTES // 4,
            "PartialLSE": rows * PARTIAL_LSE_ROW_BYTES // 4,
            "FullEpoch": 4,
            "Flags": 15804,
        }

    def kernel_extras(self, head, tail, extras: list) -> tuple:
        maps = (
            (head.InputWeight, INPUT_MAP),
            (head.QueryWeight, QUERY_MAP),
            (tail.KVbWeight, ABSORB_MAP),
            (tail.KVbWeight, KVB_MAP),
            (tail.OWeight, O_MAP),
        )
        return tuple(self.page_map(weight, *pages) for weight, pages in maps)

    @cute.kernel
    def kernel(
        self,
        head: DraftHeadArgs,
        KvCache: GridConstant[TensorMap],
        tail: TailArgs,
        InputTmap: GridConstant[TensorMap],
        QueryTmap: GridConstant[TensorMap],
        AbsorbTmap: GridConstant[TensorMap],
        KvbTmap: GridConstant[TensorMap],
        OTmap: GridConstant[TensorMap],
    ):
        maps = {
            "InputTmap": InputTmap,
            "QueryTmap": QueryTmap,
            "AbsorbTmap": AbsorbTmap,
            "KvbTmap": KvbTmap,
            "OTmap": OTmap,
        }
        if const_expr(self.GROUPS == 1):
            self.enter(DraftParams.of(KvCache, head, tail, **maps))
        else:
            for group in cutlass.range(self.GROUPS, unroll=1):
                if self.group_live(tail, group):
                    self.enter_group(group)
                    records = [self.move_to_group(r, group) for r in (head, tail)]
                    self.enter(DraftParams.of(KvCache, *records, **maps))
                    cute.arch.sync_threads()

    @cute.jit
    def prologue(self, p: WorkerParams):
        s = WorkerSmem.build(self, p)
        self.smem_record = s
        if (s.warp == 8) & (s.bid < 32 * self.GROUP_ROWS):
            rt.AsyncCopy.prefetch_tensormap(p.KvCache)
        with rt.warp_leader(s.warp, 9):
            for name in (
                "InputTmap",
                "QueryTmap",
                "AbsorbTmap",
                "KvbTmap",
                "OTmap",
            ):
                rt.AsyncCopy.prefetch_tensormap(getattr(p, name))
        with rt.warp_leader(s.warp, 0):
            rt.Prologue.init_mbarriers(
                s.smem,
                ((0, 4, 1), (32, 4, 256), (64, 1 + self.O_PAGES, 1)) + self.MBAR_GROUPS,
            )
        cute.arch.sync_threads()
        return rt.ThreadIds(s.tid, s.bid, s.warp, s.lane), s

    def row_select(self, p):
        return p.SelectRow[0]

    def context(self, p, row_select, query_):
        if const_expr(self.GROUP_ROWS == 1):
            return p.SelectContext[0]
        return 1 + (p.SelectContext[0] - 1) * Int32(row_select == query_)

    def merge_cta(self, bid):
        return (bid < 32 * self.GROUP_ROWS) & (rt.cmod(bid, 32) < 2 * self.HEADS)

    @stage
    def index_sequence(self, s):
        return cutlass.select_(
            s.bid < self.Q_CTAS,
            6 * Int32(s.bid < 82) + 2 - 1,
            cutlass.select_(s.bid < 82, Int32(6), Int32(0)),
        )

    @staticmethod
    def o_region(page):
        return 17408 + page * 24704

    @cute.jit
    def poll_row_flags(self, p, lane_0, epoch):
        owned = Int32(lane_0 < 2 + self.GROUP_ROWS)
        index_0 = 80 + rt.FastMath.min_typed(lane_0, 1)
        if lane_0 >= 2:
            index_0 = 276 + rt.FastMath.min_typed(lane_0 - 2, self.GROUP_ROWS - 1)
        self.warp_poll_ge(rt.ga(p.Flags, index_0), owned, epoch + 1)
        rt.Memory.fence_acquire_gpu()

    @stage
    @cute.jit
    def install_row_ids(self, p, s, index, fresh, record_index, generation, position):
        if s.tid < 320:
            if fresh == 0:
                record = rt.ga(p.SelectionMailbox, record_index * 2)
                index = Int32(rt.Poll.word(record, generation, False, True))
            appended = self.store_key(
                p, s, s.tid - 256, index, position, self.GROUP_ROWS
            )
            warp_hit = rt.Sync.vote_any(appended != 0)
            if s.lane == 0:
                s.warp_hits[s.warp - 8] = Int32(warp_hit != 0)

    @stage
    @cute.jit
    def exchange_rows(self, p, s, bases, epoch, slot):
        if (s.tid < 224) & (rt.cdiv(s.lane, 8) < self.GROUP_ROWS):
            self.tagged_push(
                p, s, bases, epoch, rt.cdiv(s.lane, 8), s.warp, rt.cmod(s.lane, 8), slot
            )
        rt.bar_sync(*BAR_ALL)
        query = rt.cdiv(s.warp, 2)
        rt.bar_sync(*BAR_ALL)
        if (rt.cmod(s.warp, 2) == 0) & (s.lane < 12) & (query < self.GROUP_ROWS):
            self.tagged_reduce(p, s, query)
        rt.bar_sync(*BAR_COMPUTE)
        if s.tid < 48:
            self.final_rows(p, s, slot)

    @stage
    def final_rows(self, p, s, slot):
        row = Int32(0)
        if const_expr(self.GROUP_ROWS > 1):
            row = self.row_select(p)
        value = BFloat16(
            rt.add(
                Float32(s.local_o[row * 48 + s.tid]),
                Float32(s.residual[row * 48 + s.tid]),
            )
        )
        for r in range(4):
            p.Final[r * 6144 + slot * 48 + s.tid] = value

    @stage
    @cute.jit
    def compute(self, p, s, ids):
        self.compute_prefetch(p, s)
        self.set_regs("compute")
        cute.arch.griddepcontrol_wait()
        if const_expr(self.GROUP_ROWS == 1):
            rt.Sync.launch_dependents_asm()
        cur_pos = p.CurPos[0]
        row_select = self.row_select(p)
        epoch = Uint32(0)
        if s.bid < 82:
            epoch = self.row_input(p, s, cur_pos)
        else:
            rt.bar_sync(*BAR_ALL)
            epoch = Uint32(s.epoch_slot[0])
        tag = (epoch + 1) & 32767 | 32768
        if s.bid < self.Q_CTAS:
            self.row_query(p, s, tag)
        if (s.bid >= 128) & (s.bid < 128 + self.GROUP_ROWS):
            self.kv_latent_store(
                p, s, self.KV_BASE + (s.bid - 128) * 512, s.bid - 128, cur_pos, tag
            )
            rt.bar_sync(*BAR_COMPUTE)
            self.release_flag(p, s, FlagWords.KV_LATENT_READY + s.bid - 128)
        if s.bid < self.Q_CTAS:
            absorb = (self.Q[0], self.Q[1], self.OPERANDS, 512 * self.HEADS)
            self.absorb_stage(
                p, s, tag, rt.cdiv(s.bid, 8), self.GROUP_ROWS, absorb, s.absorb_full
            )
        if s.bid < 32 * self.GROUP_ROWS:
            query_ = rt.cdiv(s.bid, 32)
            context = self.context(p, row_select, query_)
            if self.min_i32(context, TOPK) > rt.cmod(s.bid, SPLITS) * SPLIT_KEYS:
                backoff = Int32(s.bid >= self.Q_CTAS)
                self.operand_stage(
                    p, s, tag, query_, self.HEADS, self.PE[0], self.OPERANDS, backoff
                )
            rt.bar_sync(*BAR_ALL)
            first = self.index_sequence(s) + 1
            score_count = self.min_i32(context, TOPK)
            split = rt.cmod(s.bid, 32)
            self.split_attention(
                p, s, query_, split, score_count, first, self.HEADS, self.LSE_STRIDE
            )
        self.merge(p, s, tag, row_select)
        if s.bid < 128:
            self.row_output(p, s, tag, epoch)

    @stage
    @cute.jit
    def compute_prefetch(self, p, s):
        with rt.warp_leader(s.warp, 0):
            if s.bid < 82:
                policy = rt.AsyncCopy.createpolicy_evict_first(
                    rt.ga(p.InputWeight, 0), 16121856, 16121856
                )
                rt.mbar_expect_tx(s.ring_full, 32960)
                rt.bulk_g2s_hint(
                    s.ring_addr + 33792,
                    rt.baddr(p.InputScales, rt.cdiv(s.bid, 4) * 192),
                    192,
                    s.ring_full,
                    policy,
                )
                rt.AsyncCopy.tma_3d(
                    s.ring_addr, p.InputTmap, 0, s.bid * 32, 0, s.ring_full, L2_HINT
                )
            cute.arch.griddepcontrol_wait()
            if s.bid < 82:
                self.load_x(s, p.X)

    @stage
    @cute.jit
    def row_input(self, p, s, cur_pos):
        total = self.x_norm(p, s)
        rt.bar_sync(*BAR_ALL)
        late_inverse = rt.Reduce.rms_inverse(total)
        epoch = Uint32(s.epoch_slot[0])
        tag = (epoch + 1) & 32767 | 32768
        accum = rt.rmem_tensor(8, Float32)
        for i in cutlass.range_constexpr(8):
            accum[i] = Float32(0.0)
        for page in cutlass.range(0, 6, 1, unroll=2):
            self.tma_page_mma(s, accum, page, page, 3076, self.INPUT_SCALE_WORD)
        self.rows_to_scratch(s, accum, self.GROUP_ROWS, 2)
        if s.tid < 32 * self.GROUP_ROWS:
            value = rt.mul(self.warp_sum(s, s.tid, self.GROUP_ROWS, 2), late_inverse)
            row, col = rt.cdiv(s.tid, 32), rt.cmod(s.tid, 32)
            if s.bid < 64:
                self.put_tagged(
                    p,
                    Handoff.Q_DOWN + row * Handoff.Q_DOWN_ROW + s.bid * 32 + col,
                    value,
                    tag,
                )
            elif s.bid < 80:
                latent_addr = (
                    self.KV_BASE + row * Handoff.KV_LATENT_ROW + (s.bid - 64) * 32 + col
                )
                self.put_tagged(p, latent_addr, value, tag)
            else:
                dim = (s.bid - 80) * 32 + col
                rotated = self.rope(p, s, rt.Convert.bf16_round(value), dim)
                KvRows.store_rope_e4m3(
                    KvRows.rope_byte(p.PECache, p.BlockTable, cur_pos + row, dim),
                    rotated,
                )
        rt.bar_sync(*BAR_COMPUTE)
        if s.bid >= 80:
            self.release_flag(p, s, s.bid)
        # The Q CTAs free the norm buffer after their absorb; the other input CTAs here.
        if s.bid >= self.Q_CTAS:
            rt.mbar_arrive(s.norm_free)
        return epoch

    @stage
    @cute.jit
    def row_query(self, p, s, tag):
        gamma_q = rt.Memory.ld_global_v8_f32(rt.ga(p.GammaQ, s.tid * 8))
        firsts = [(s.tid + 256 * j) * 8 for j in range(self.GROUP_ROWS)]
        got = self.poll_handoff(p, tag << 16, firsts, Int32(1))
        for j in cutlass.range_constexpr(self.GROUP_ROWS):
            self.prmt_store(s.act_addr + j * 4112 + s.tid * 16, got, j)
        values = [
            rt.bf16x8_to_f32([s.act[q * 1028 + s.tid * 4 + i] for i in range(4)])
            for q in range(self.GROUP_ROWS)
        ]
        inverses = self.rms_inverses(s, values)
        for q in cutlass.range_constexpr(self.GROUP_ROWS):
            rt.Memory.sts_v4(
                s.act_addr + (q * 2056 + s.tid * 8) * 2,
                self.norm_bf16x8(values[q], gamma_q, inverses[q]),
            )
        rt.bar_sync(*BAR_COMPUTE)
        accum = rt.rmem_tensor(8, Float32)
        for i in cutlass.range_constexpr(8):
            accum[i] = Float32(0.0)
        for page in cutlass.range_constexpr(2):
            self.tma_page_mma(s, accum, 6 + page, page, 1028, self.QUERY_SCALE_WORD)
        self.rows_to_scratch(s, accum, self.GROUP_ROWS, 2)
        if s.tid < 32 * self.GROUP_ROWS:
            value = self.warp_sum(s, s.tid, self.GROUP_ROWS, 2)
            row, col = rt.cdiv(s.tid, 32), rt.cmod(s.tid, 32)
            head, tile = rt.cdiv(s.bid, 8), rt.cmod(s.bid, 8)
            q_base, q_stride = self.Q
            if tile < 6:
                self.put_tagged(
                    p,
                    q_base + row * q_stride + head * 192 + tile * 32 + col,
                    value,
                    tag,
                )
            else:
                index = head * 64 + (tile - 6) * 32 + col
                rotated = self.rope(p, s, rt.Convert.bf16_round(value), index)
                self.put_tagged(p, self.PE[0] + row * self.PE[1] + index, rotated, tag)
        rt.bar_sync(*BAR_COMPUTE)

    @stage
    @cute.jit
    def merge(self, p, s, tag, row_select):
        if self.merge_cta(s.bid):
            query_ = rt.cdiv(s.bid, 32)
            context = self.context(p, row_select, query_)
            rt.bar_sync(*BAR_ALL)
            head = rt.cmod(rt.cmod(s.bid, 32), self.HEADS)
            split_count = rt.cdiv(
                self.min_i32(context, TOPK) + SPLIT_KEYS - 1, SPLIT_KEYS
            )
            dims = (
                rt.cdiv(rt.cmod(s.bid, 32), self.HEADS) * 256 + rt.cmod(s.tid, 64) * 4
            )
            split_base = rt.cdiv(s.tid, 128) * 16
            loaded = rt.rmem_tensor(64, Float32)
            if rt.cmod(s.tid, 128) < 64:
                for si in cutlass.range_constexpr(16):
                    split = self.min_i32(split_base + si, split_count - 1)
                    part = rt.Memory.ld_global_v4_f32(
                        rt.ga(
                            p.PartialO,
                            ((query_ * 32 + split) * self.HEADS + head) * 512 + dims,
                        )
                    )
                    for j in cutlass.range_constexpr(4):
                        loaded[4 * si + j] = part[j]
            if s.warp == 0:
                lse_first = query_ * 32 * self.LSE_STRIDE + head
                self.merge_weights(p, s, split_count, lse_first, self.LSE_STRIDE)
            rt.bar_sync(*BAR_COMPUTE)
            merged = rt.rmem_tensor(4, Float32)
            for i in cutlass.range_constexpr(4):
                merged[i] = Float32(0.0)
            if rt.cmod(s.tid, 128) < 64:
                loaded_rows = [
                    [loaded[4 * si + j] for j in range(4)] for si in range(16)
                ]
                self.merge_sum(s, merged, loaded_rows, split_base)
            self.fold_halves(s, merged)
            if s.tid < 64:
                packet = [rt.Convert.bf16_low(merged[i]) | tag << 16 for i in range(4)]
                word = self.MERGED + (query_ * self.HEADS + head) * 512 + dims
                rt.Memory.st_volatile_v4(rt.ga(p.Handoff, word), packet)

    @stage
    @cute.jit
    def row_output(self, p, s, tag, epoch):
        if s.bid < self.Q_CTAS:
            head = rt.cdiv(s.bid, 8)
            if s.tid < 64 * self.GROUP_ROWS:
                first = self.MERGED + rt.cdiv(s.tid, 64) * 512 * self.HEADS + head * 512
                got = self.poll_handoff(p, tag << 16, [first + rt.cmod(s.tid, 64) * 8])
                act_row = (
                    s.act_addr + rt.cdiv(s.tid, 64) * 1040 + rt.cmod(s.tid, 64) * 16
                )
                self.prmt_store(act_row, got, 0)
            rt.bar_sync(*BAR_COMPUTE)
            rt.mbar_wait(s.tail_full, 0)
            cute.arch.fence_view_async_shared()
            vo_lane = rt.cdiv(s.lane, 4) * 64 + s.warp * 2048 + rt.cmod(s.lane, 4) * 16
            vo_k = (
                rt.cmod(rt.cdiv(s.lane, 4), 4) * 260
                + rt.cmod(s.lane, 4) * 8
                + s.warp * 32
            )
            vo_acc = self.vo_tile(s, 0, vo_lane, (0, 1024), vo_k)
            scale = s.ring_f32[4096 + rt.cdiv(s.warp, 2)]
            accum = [rt.mul(vo_acc[i], scale) for i in range(8)]
            self.rows_to_scratch(s, accum, self.GROUP_ROWS, 2)
            if s.tid < 32 * self.GROUP_ROWS:
                index = (
                    self.VO[0]
                    + rt.cdiv(s.tid, 32) * self.VO[1]
                    + head * 256
                    + rt.cmod(s.bid, 8) * 32
                    + rt.cmod(s.tid, 32)
                )
                self.put_tagged(
                    p, index, self.warp_sum(s, s.tid, self.GROUP_ROWS, 2), tag
                )
            rt.bar_sync(*BAR_COMPUTE)
        else:
            rt.bar_sync(*BAR_COMPUTE)
        self.o_projection(p, s, tag)
        self.exchange_rows(p, s, p.peer_bases, epoch, s.bid)

    @cute.jit
    def o_pending_load(self, p, o_pending, first, live):
        if live:
            self.load_pending(p, o_pending, first)

    @stage
    @cute.jit
    def o_projection(self, p, s, tag):
        o_token = rt.cdiv(s.lane, 8)
        o_first = (
            self.VO[0] + o_token * self.VO[1] + s.warp * 64 + rt.cmod(s.lane, 8) * 8
        )
        live = o_token < self.GROUP_ROWS
        o_pending = rt.rmem_tensor(8, Uint32)
        for i in cutlass.range_constexpr(8):
            o_pending[i] = tag << 16

        def o_load(page):
            self.o_pending_load(p, o_pending, o_first + page * 512, live)

        vo_tokens = (
            rt.cmod(rt.cdiv(s.lane, 4), 4) * (self.O_ROW_BYTES // 4)
            + rt.cmod(s.lane, 4) * 8
        )
        vo_lane = rt.cdiv(s.lane, 4) * 64 + s.warp * 3072 + rt.cmod(s.lane, 4) * 16

        def page_mma(page, accum):
            vo_k = vo_tokens + rt.cdiv(page * 512 + s.warp * 64, 2)
            vo_acc = self.vo_tile(
                s, self.o_region(page), vo_lane, (0, 1024, 2048), vo_k
            )
            for m in cutlass.range_constexpr(3):
                scale = s.ring_f32[4160 + m * 16 + page * 4 + rt.cdiv(s.warp, 2)]
                for item in cutlass.range_constexpr(4):
                    accum[m * 4 + item] = rt.fma(
                        vo_acc[m * 4 + item], scale, accum[m * 4 + item]
                    )

        accum = self.o_proj_pages(
            s,
            o_pending,
            o_token,
            tag << 16,
            o_load,
            o_load,
            page_mma,
            self.O_PAGES,
            self.O_ROW_BYTES,
            0,
        )
        self.rows_to_scratch(s, accum, self.GROUP_ROWS, 3)
        if s.tid < 48 * self.GROUP_ROWS:
            s.local_o[s.tid] = BFloat16(self.warp_sum(s, s.tid, self.GROUP_ROWS, 3))
        rt.bar_sync(*BAR_COMPUTE)

    @stage
    @cute.jit
    def load(self, p, s, ids):
        self.set_regs("load")
        policies = [
            rt.AsyncCopy.createpolicy_evict_first(rt.ga(t, 0), n, n)
            for t, n in (
                (p.InputWeight, 16121856),
                (p.QueryWeight, 4194304),
                (p.KVbWeight, 1835008),
                (p.OWeight, 12582912),
            )
        ]
        if s.bid < 82:
            for page in cutlass.range_constexpr(1, 4):
                self.issue_input_page(p, s, page, s.bid)
        cute.arch.griddepcontrol_wait()
        cur_pos = p.CurPos[0]
        row_select = self.row_select(p)
        with rt.warp_leader(s.warp, 9):
            s.epoch_slot[0] = rt.cdiv(
                rt.Memory.atom_add_u32(rt.ga(p.FullEpoch, 0), 1), 148
            )
        rt.bar_sync(*BAR_ALL)
        epoch = s.epoch_slot[0]
        if s.bid < 128:
            self.load_residual(p, s, s.bid, self.GROUP_ROWS)
        if s.bid < 82:
            for page in cutlass.range_constexpr(4, 6):
                self.issue_input_page(p, s, page, s.bid)
        if s.bid < self.Q_CTAS:
            with rt.warp_leader(s.warp, 8):
                for page in cutlass.range_constexpr(2):
                    slot = (6 + page) % 4
                    s.pages.acquire(6 + page, 32832 if page == 0 else 32768, slot)
                    slot_addr = s.ring_addr + slot * Ring.SLOT_BYTES
                    if const_expr(page == 0):
                        rt.bulk_g2s_hint(
                            slot_addr + 33792,
                            rt.baddr(p.QueryScales, rt.cdiv(s.bid, 4) * 64),
                            64,
                            s.ring_full + slot * 8,
                            policies[1],
                        )
                    rt.AsyncCopy.tma_3d(
                        slot_addr,
                        p.QueryTmap,
                        0,
                        s.bid * 32,
                        page * 8,
                        s.ring_full + slot * 8,
                        L2_HINT,
                    )
                rt.mbar_expect_tx(s.absorb_full, 12288)
                rt.AsyncCopy.tma_3d(
                    s.absorb_addr,
                    p.AbsorbTmap,
                    0,
                    rt.cdiv(s.bid, 8) * 448,
                    rt.cmod(s.bid, 8),
                    s.absorb_full,
                    L2_HINT,
                )
        if s.bid < 32 * self.GROUP_ROWS:
            self.kv_stage(p, s, cur_pos, row_select, epoch)
        if s.bid < 128:
            self.tail(p, s, epoch, policies)

    @stage
    @cute.jit
    def kv_stage(self, p, s, cur_pos, row_select, epoch):
        query_, split = rt.cdiv(s.bid, 32), rt.cmod(s.bid, 32)
        first = self.index_sequence(s) + 1
        context = self.context(p, row_select, query_)
        score_count = self.min_i32(context, TOPK)
        generation = Uint32(p.SelectContext[1])
        index, fresh, record_index = self.probe_ids(
            p, s, row_select, split, context, score_count, generation, None, True
        )
        rt.bar_sync(*BAR_LOADER)
        stale_ids = Int32((s.id_probe[0] == 0) | (s.id_probe[1] == 0))
        if stale_ids != 0:
            self.poll_flags(p, FlagWords.PE_READY, 2, s.tid - 256, epoch)
            rt.bar_sync(*BAR_LOADER)
            self.poll_flags(
                p, FlagWords.KV_LATENT_READY, self.GROUP_ROWS, s.tid - 256, epoch
            )
            rt.bar_sync(*BAR_LOADER)
            self.prefetch_context(p, s, cur_pos + self.GROUP_ROWS, s.bid)
        self.install_row_ids(p, s, index, fresh, record_index, generation, cur_pos)
        active = Int32(score_count > split * SPLIT_KEYS)
        self.arm_kv_ring(s, first, first + 1, active)
        rt.bar_sync(*BAR_LOADER)
        if stale_ids != 0:
            self.pe_operand(p, s, active, score_count, split)
        if active != 0:
            self.kv_gather(p, s, 0, first, first + 1, L2_HINT)
        overlap = (stale_ids == 0) & ((s.warp_hits[0] | s.warp_hits[1]) != 0)
        if overlap & (s.warp == 8):
            self.poll_row_flags(p, s.lane, epoch)
            rt.syncwarp()
        rt.bar_sync(*BAR_LOADER)
        if stale_ids == 0:
            self.pe_operand(p, s, active, score_count, split)
        else:
            cute.arch.cp_async_commit_group()
        if active != 0:
            self.kv_gather(p, s, 1, first, first + 1, L2_HINT)
        cute.arch.cp_async_wait_group(0)
        if active != 0:
            self.pe_widen(s)
        rt.bar_sync(*BAR_ALL)

    @stage
    @cute.jit
    def pe_operand(self, p, s, active, score_count, split):
        if s.bid < 82:
            rt.mbar_wait(s.norm_free, 0)
        if active != 0:
            self.pe_stage(p, s, score_count, split)
        cute.arch.cp_async_commit_group()

    @stage
    @cute.jit
    def tail(self, p, s, epoch, policies):
        with rt.warp_leader(s.warp, 8):
            if s.bid < 32 * self.GROUP_ROWS:
                self.drain_ring(s, self.index_sequence(s), Int32(s.bid < self.Q_CTAS))
            else:
                issued = rt.FastMath.select_i32(
                    s.bid < self.Q_CTAS, 8, rt.FastMath.select_i32(s.bid < 82, 6, 0)
                )
                for slot in cutlass.range_constexpr(4):
                    if issued > slot:
                        last_use = issued - 1 - rt.cmod(issued - 1 - slot, 4)
                        rt.mbar_wait(
                            s.ring_empty + slot * 8,
                            rt.cmod(rt.cdiv(last_use, 4), 2),
                        )
            if s.bid < self.Q_CTAS:
                rt.mbar_expect_tx(s.tail_full, 16400)
                kvb_row = rt.cdiv(s.bid, 8) * 448 + 192 + rt.cmod(s.bid, 8) * 32
                rt.AsyncCopy.tma_3d(
                    s.ring_addr, p.KvbTmap, 0, kvb_row, 0, s.tail_full, L2_HINT
                )
                rt.bulk_g2s_hint(
                    s.ring_addr + 16384,
                    rt.baddr(p.KVbScales, rt.cdiv(kvb_row, 128) * 16),
                    16,
                    s.tail_full,
                    policies[2],
                )
        if self.merge_cta(s.bid):
            split_done = FlagWords.SPLIT_DONE + rt.cdiv(s.bid, SPLITS) * SPLITS
            self.poll_flags(p, split_done, SPLITS, s.tid - 256, epoch)
            rt.bar_sync(*BAR_LOADER)
            rt.bar_sync(*BAR_ALL)
        with rt.warp_leader(s.warp, 8):
            for page in cutlass.range_constexpr(self.O_PAGES):
                self.issue_o_page(p, s, page, s.bid, policies[3])
        cute.arch.cp_async_wait_group(0)
        rt.bar_sync(*BAR_ALL)
        self.tagged_receive(p, s, p.peer_bases, epoch, self.GROUP_ROWS)
        rt.bar_sync(*BAR_ALL)

    @stage
    def tma_page_mma(self, s, accum, seq, page, token_stride: int, scale_word: int):
        s.pages.wait_full(seq)
        cute.arch.fence_view_async_shared()
        region = Ring.slot_offset(seq)
        group, quad = rt.cdiv(s.lane, 4), rt.cmod(s.lane, 4)
        plane = rt.cdiv(region + s.warp * 4096 + group * 128, 4)
        rows = [None] * 16
        for j in range(4):
            for h in range(2):
                w = rt.Memory.lds_b128(
                    s.ring_addr + (plane + ((2 * quad + h) ^ group) * 4 + 256 * j) * 4
                )
                rows[4 * j + 2 * h] = Uint64(w[0]) | Uint64(w[1]) << 32
                rows[4 * j + 2 * h + 1] = Uint64(w[2]) | Uint64(w[3]) << 32
        k_words = (
            rt.cmod(group, 4) * token_stride
            + quad * 16
            + rt.cdiv(page * 1024 + s.warp * 128, 2)
        )
        acc = [None] * 8
        for j in range(4):
            a = [None] * 16
            for h in range(2):
                even = rt.Convert.unpack_u16x4(rows[j + 8 * h])
                odd = rt.Convert.unpack_u16x4(rows[j + 4 + 8 * h])
                for i in range(4):
                    a[8 * h + 2 * i] = rt.Convert.e4m3x2_to_bf16x2(even[i])
                    a[8 * h + 2 * i + 1] = rt.Convert.e4m3x2_to_bf16x2(odd[i])
            for t in range(2):
                b = rt.Memory.lds_weak_b64(s.act_addr + (k_words + 4 * j + 2 * t) * 4)
                for h in range(2):
                    if j == 0 and t == 0:
                        acc[4 * h : 4 * h + 4] = rt.TensorCore.mma(
                            None, tuple(a[8 * h : 8 * h + 4]), tuple(b), True
                        )
                    else:
                        rt.TensorCore.mma_acc_at(
                            acc,
                            4 * h,
                            tuple(a[8 * h + 4 * t : 8 * h + 4 * t + 4]),
                            tuple(b),
                        )
        scale = s.ring_f32[scale_word + page * 8 + s.warp]
        for i in range(8):
            accum[i] = rt.fma(acc[i], scale, accum[i])
        s.pages.release(rt.cmod(seq, 4))

    def put_tagged(self, p, index, value, tag):
        rt.Memory.st_relaxed_sys(
            rt.ga(p.Handoff, index), rt.Convert.bf16_low(value) | tag << 16
        )

    def poll_handoff(self, p, tag_high, firsts, backoff=None, reconverge: bool = True):
        got = rt.rmem_tensor(8 * len(firsts), Uint32)

        def store(words):
            for i in range(8 * len(firsts)):
                got[i] = words[i // 4][i % 4]

        chunks = [rt.ga(p.Handoff, first) for first in firsts]
        self.poll_tagged16(chunks, tag_high, store, backoff, reconverge=reconverge)
        return got

    @stage
    def absorb_mma(self, s, scales):
        def b_frag(k32):
            column = rt.cdiv(k32 + rt.cdiv(s.lane, 8) * 8, 2)
            return rt.TensorCore.ldsm_x4(
                s.act_addr + (rt.cmod(s.lane, 4) * 100 + column) * 4
            )

        column = s.absorb_addr + s.warp * 16 + rt.cdiv(s.lane, 4) * 2
        row = rt.cmod(s.lane, 4) * 2

        def a_frag(k):
            first = column + (16 * k + row) * 64
            x = [Uint32(rt.Memory.lds(first + 64 * r, Uint16)) for r in (0, 1, 8, 9)]
            low, high = (
                rt.Convert.prmt(x[0], x[1], 0x5140),
                rt.Convert.prmt(x[2], x[3], 0x5140),
            )
            halves = (low, low >> 16, high, high >> 16)
            return [rt.Convert.e4m3x2_to_bf16x2(Uint16(h)) for h in halves]

        total = [Float32(0.0)] * 4
        absorb = [b_frag(0), b_frag(32)]
        a = [a_frag(0)]
        for k in range(12):
            if k + 1 < 12:
                a.append(a_frag(k + 1))
            if k in (2, 4, 6, 8):
                absorb[(k // 2 + 1) % 2] = b_frag(32 * (k // 2 + 1))
            b = absorb[(k // 2) % 2]
            atom = rt.TensorCore.mma(
                None, a[k], (b[(k % 2) * 2], b[(k % 2) * 2 + 1]), True
            )
            total = [rt.fma(atom[i], scales[k // 4], total[i]) for i in range(4)]
        return total

    @stage
    @cute.jit
    def absorb_stage(
        self,
        p,
        s,
        tag,
        head,
        rows: cutlass.Constexpr,
        layout: cutlass.Constexpr,
        full_addr,
    ):
        q_base, q_stride, out_base, out_stride = layout
        scales = [
            rt.Convert.bits_f32(
                rt.Memory.ldg_nc_u32(
                    rt.baddr(
                        p.KVbScales,
                        (
                            rt.cdiv(head * 448 + 64 * i, 128) * 4
                            + rt.cdiv(rt.cmod(s.bid, 8), 2)
                        )
                        * 4,
                    )
                )
            )
            for i in range(3)
        ]
        if s.tid < 24 * rows:
            c = s.tid
            first = q_base + rt.cdiv(c, 24) * q_stride + head * 192 + rt.cmod(c, 24) * 8
            got = self.poll_handoff(p, tag << 16, [first], None, (24 * rows) % 32 == 0)
            self.prmt_store(
                s.act_addr + rt.cdiv(c, 24) * 400 + rt.cmod(c, 24) * 16, got, 0
            )
        rt.bar_sync(*BAR_COMPUTE)
        rt.mbar_wait(full_addr, 0)
        cute.arch.fence_view_async_shared()
        if s.warp < 4:
            total = self.absorb_mma(s, scales)
            pair = rt.cmod(s.lane, 4) * 2
            row = out_base + s.bid * 64 + s.warp * 16 + rt.cdiv(s.lane, 4) * 2
            if pair < rows:
                self.put_tagged(p, row + pair * out_stride, total[0], tag)
                self.put_tagged(p, row + pair * out_stride + 1, total[2], tag)
                if pair + 1 < rows:
                    self.put_tagged(p, row + (pair + 1) * out_stride, total[1], tag)
                    self.put_tagged(p, row + (pair + 1) * out_stride + 1, total[3], tag)
        rt.bar_sync(*BAR_COMPUTE)
        rt.mbar_arrive(s.norm_free)

    @stage
    @cute.jit
    def operand_stage(
        self,
        p,
        s,
        tag,
        query_,
        n_heads: cutlass.Constexpr,
        pe_base: cutlass.Constexpr,
        q_base: cutlass.Constexpr,
        backoff,
    ):
        n_q = 64 * n_heads
        n_all = 72 * n_heads
        n_k = (n_all + 255) // 256
        firsts = []
        for k in cutlass.range_constexpr(n_k):
            c = s.tid + 256 * k
            if const_expr(k > 0):
                c = rt.FastMath.select_i32(c >= n_all, s.tid, c)
            pe_first = pe_base + query_ * n_q + (c - n_q) * 8
            firsts.append(
                rt.FastMath.select_i32(
                    c >= n_q, pe_first, q_base + query_ * n_q * 8 + c * 8
                )
            )
        got = self.poll_handoff(p, tag << 16, firsts, backoff)
        for k in cutlass.range_constexpr(n_k):
            c = s.tid + 256 * k
            head = rt.cdiv(c, 64)
            if c < n_q:
                group = rt.cdiv(rt.cmod(c, 64), 2) * 16
                half = rt.cmod(c, 2) * 8
                for j in cutlass.range_constexpr(2):
                    col = group + 8 * j
                    at = (
                        rt.cdiv(col, 64) * 2048
                        + head * 128
                        + ((rt.cmod(col, 64) * 2) ^ (rt.cmod(head, 8) * 16))
                        + half
                    )
                    rt.Memory.st_shared_v2(
                        s.act_addr + at,
                        self.prmt_word(got, k, j),
                        self.prmt_word(got, k, 2 + j),
                    )
            elif c < n_all:
                unit = c - n_q
                at = (
                    16384
                    + rt.cdiv(unit, 8) * 128
                    + (rt.cmod(unit, 8) ^ rt.cmod(rt.cdiv(unit, 8), 8)) * 16
                )
                self.prmt_store(s.act_addr + at, got, k)

    @stage
    @cute.jit
    def split_attention(
        self,
        p,
        s,
        query_,
        split,
        score_count,
        first,
        n_heads: cutlass.Constexpr,
        lse_stride: cutlass.Constexpr,
    ):
        q_addr = s.ring_addr - 16384
        split_count = rt.cdiv(score_count + 64 - 1, 64)
        last = first + 1
        key_base = s.warp * 8
        scores = rt.rmem_tensor(4, Float32)
        head_lo = rt.cdiv(s.lane, 4)
        quad = rt.cmod(s.lane, 4)
        if split < split_count:
            q_page = [None] * 16
            k_page = [None] * 8
            for f in cutlass.range_constexpr(4):
                row = rt.cmod(s.lane, 16)
                col = f * 16 + rt.cdiv(s.lane, 16) * 8
                q_page[4 * f : 4 * f + 4] = rt.TensorCore.ldsm_x4(
                    q_addr + 180224 + row * 128 + ((col * 2) ^ ((row & 7) << 4))
                )
                key_col = f * 16 + rt.cmod(rt.cdiv(s.lane, 8), 2) * 8
                key_row = key_base + rt.cmod(s.lane, 8)
                key_at = 182272 + key_row * 128 + ((key_col * 2) ^ ((key_row & 7) << 4))
                k_page[2 * f : 2 * f + 2] = rt.TensorCore.ldsm_x2(q_addr + key_at)
            for f in cutlass.range_constexpr(4):
                acc = None if f == 0 else rt.Memory.take(scores, 0, 4)
                q_frag, k_frag = q_page[4 * f : 4 * f + 4], k_page[2 * f : 2 * f + 2]
                rt.Memory.put(scores, 0, rt.TensorCore.mma(acc, q_frag, k_frag, f == 0))
            s.pages.wait_full(first)
            s.pages.wait_full(last)
            cute.arch.fence_view_async_shared()
            rt.bar_sync(*BAR_COMPUTE)
            for block in cutlass.range_constexpr(4):
                for half in cutlass.range_constexpr(2):
                    row = key_base + rt.cmod(s.lane, 8)
                    unit = half * 4 + rt.cdiv(s.lane, 8)
                    kv_at = (
                        rt.cmod(first, 4) * Ring.SLOT_BYTES
                        + block * Ring.KV_PLANE_BYTES
                        + row * 128
                        + ((unit * 16) ^ ((row & 7) << 4))
                    )
                    k_words = rt.TensorCore.ldsm_x4(s.ring_addr + kv_at)
                    for ch in cutlass.range_constexpr(4):
                        q_row = rt.cmod(s.lane, 16)
                        step = block * 8 + half * 4 + ch
                        col = rt.cmod(step, 4) * 16 + rt.cdiv(s.lane, 16) * 8
                        q_at = (
                            163840
                            + rt.cdiv(step, 4) * 2048
                            + q_row * 128
                            + ((col * 2) ^ ((q_row & 7) << 4))
                        )
                        q_page[4 * ch : 4 * ch + 4] = rt.TensorCore.ldsm_x4(
                            q_addr + q_at
                        )
                        k_page[2 * ch] = rt.Convert.e4m3x2_to_bf16x2(
                            Uint16(k_words[ch])
                        )
                        k_page[2 * ch + 1] = rt.Convert.e4m3x2_to_bf16x2(
                            Uint16(k_words[ch] >> 16)
                        )
                    for ch in cutlass.range_constexpr(4):
                        q_ch, k_ch = (
                            q_page[4 * ch : 4 * ch + 4],
                            k_page[2 * ch : 2 * ch + 2],
                        )
                        rt.TensorCore.mma_acc_at(scores, 0, q_ch, k_ch)
            if split == split_count - 1:
                for e in cutlass.range_constexpr(4):
                    if score_count <= split * 64 + key_base + 2 * quad + rt.cmod(e, 2):
                        scores[e] = Float32(NEG_FLT_MAX)
            if head_lo >= n_heads - 8:
                scores[2] = Float32(NEG_FLT_MAX)
                scores[3] = Float32(NEG_FLT_MAX)
            max_lo = rt.max_noftz(scores[0], scores[1])
            max_hi = rt.max_noftz(scores[2], scores[3])
            for step in cutlass.range_constexpr(2):
                max_lo = rt.max_noftz(
                    max_lo, cute.arch.shuffle_sync_bfly(max_lo, 1 << step)
                )
                max_hi = rt.max_noftz(
                    max_hi, cute.arch.shuffle_sync_bfly(max_hi, 1 << step)
                )
            pair_at = s.pairs_addr + (s.warp * 16 + 2 * head_lo) * 4
            if quad == 0:
                rt.Memory.st_shared_v2_f32(pair_at, max_lo, max_hi)
            rt.bar_sync(*BAR_COMPUTE)
            head_max = [Float32(NEG_FLT_MAX), Float32(NEG_FLT_MAX)]
            for k in cutlass.range_constexpr(8):
                p0 = s.pairs[k * 16 + 2 * head_lo]
                p1 = s.pairs[k * 16 + 2 * head_lo + 1]
                head_max = [
                    rt.max_noftz(head_max[0], p0),
                    rt.max_noftz(head_max[1], p1),
                ]
            rt.bar_sync(*BAR_COMPUTE)
            for e in cutlass.range_constexpr(4):
                scores[e] = cute.math.exp2(
                    rt.mul(
                        rt.FastMath.sub(scores[e], head_max[e // 2]),
                        0.09016843885183334,
                    ),
                    fastmath=False,
                )
            sum_lo = rt.add(scores[0], scores[1])
            sum_hi = rt.add(scores[2], scores[3])
            for step in cutlass.range_constexpr(2):
                sum_lo = rt.add(sum_lo, cute.arch.shuffle_sync_bfly(sum_lo, 1 + step))
                sum_hi = rt.add(sum_hi, cute.arch.shuffle_sync_bfly(sum_hi, 1 + step))
            if quad == 0:
                rt.Memory.st_shared_v2_f32(pair_at, sum_lo, sum_hi)
            for h in cutlass.range_constexpr(2):
                word = self.p_word(head_lo + 8 * h, key_base + 2 * quad)
                s.act[word] = rt.f32x2_to_bf16x2(scores[2 * h], scores[2 * h + 1])
            rt.bar_sync(*BAR_COMPUTE)
            head_sum = [Float32(0.0), Float32(0.0)]
            for k in cutlass.range_constexpr(8):
                p0 = s.pairs[k * 16 + 2 * head_lo]
                p1 = s.pairs[k * 16 + 2 * head_lo + 1]
                head_sum = [rt.add(head_sum[0], p0), rt.add(head_sum[1], p1)]
            output = [None] * 32
            for key_frag in cutlass.range_constexpr(4):
                pr = rt.cmod(s.lane, 16)
                pc = key_frag * 16 + rt.cdiv(s.lane, 16) * 8
                p_frag = rt.TensorCore.ldsm_x4(
                    q_addr + 190464 + pr * 128 + ((pc * 2) ^ ((pr & 7) << 4))
                )
                for block in cutlass.range_constexpr(4):
                    row = key_frag * 16 + rt.cmod(s.lane, 16)
                    v_at = (
                        rt.cmod(first, 4) * Ring.SLOT_BYTES
                        + block * Ring.KV_PLANE_BYTES
                        + row * 128
                        + ((s.warp * 16) ^ ((row & 7) << 4))
                    )
                    v_words = rt.TensorCore.ldsm_t_b8(s.ring_addr + v_at)
                    for t in cutlass.range_constexpr(2):
                        v_frag = (
                            rt.Convert.e4m3x2_to_bf16x2(Uint16(v_words[t])),
                            rt.Convert.e4m3x2_to_bf16x2(Uint16(v_words[t] >> 16)),
                        )
                        o = (block * 2 + t) * 4
                        acc = None if key_frag == 0 else output[o : o + 4]
                        output[o : o + 4] = rt.TensorCore.mma(
                            acc, p_frag, v_frag, key_frag == 0
                        )
            head_inverse, head_lse = self.lse_tail(head_sum, head_max)
            partial_base = (query_ * 32 + split) * n_heads * 512
            for plane in cutlass.range_constexpr(8):
                for h in cutlass.range_constexpr(2):
                    head = head_lo + 8 * h
                    dim = (
                        rt.cdiv(plane, 2) * 128
                        + s.warp * 16
                        + rt.cmod(plane, 2) * 8
                        + 2 * quad
                    )
                    if head < n_heads:
                        n0 = rt.mul(output[plane * 4 + 2 * h], head_inverse[h])
                        n1 = rt.mul(output[plane * 4 + 2 * h + 1], head_inverse[h])
                        rt.Memory.st_global_v2_f32(
                            rt.ga(p.PartialO, partial_base + head * 512 + dim), n0, n1
                        )
            if (s.warp == 0) & (quad == 0):
                for h in cutlass.range_constexpr(2):
                    if head_lo + 8 * h < n_heads:
                        p.PartialLSE[
                            (query_ * 32 + split) * lse_stride + head_lo + 8 * h
                        ] = head_lse[h]
        rt.bar_sync(*BAR_COMPUTE)
        rt.bar_sync(*BAR_COMPUTE)
        self.release_flag(p, s, 296 + s.bid)
        cute.arch.fence_view_async_shared()
        rt.mbar_arrive(s.ring_empty + rt.cmod(first, 4) * 8)
        rt.mbar_arrive(s.ring_empty + rt.cmod(last, 4) * 8)

    @stage
    @cute.jit
    def tagged_receive(self, p, s, bases, epoch, rows: cutlass.Constexpr):
        lane_0 = s.tid - 256
        n = 56 * rows
        n_k = (n + 127) // 128
        base = rt.TaggedMailbox.peer_base(p.peer_rank, bases)
        addresses = []
        for k in cutlass.range_constexpr(n_k):
            it = lane_0
            if const_expr(k > 0):
                it = rt.FastMath.select_i32(
                    lane_0 + 128 * k >= n, lane_0, lane_0 + 128 * k
                )
            mailbox = rt.OExchange.word(
                rt.cdiv(it, 56),
                rt.cmod(rt.cdiv(it, 8), 7),
                rt.cmod(epoch + 1, 2),
                s.bid,
                rt.cmod(it, 8),
            )
            addresses.append(base + Int64(mailbox) * 4)
        if lane_0 < n:
            packets = rt.rmem_tensor(4 * n_k, Uint32)
            rt.TaggedMailbox.poll(packets, addresses, rt.TaggedMailbox.tag8(epoch))
            if const_expr(n >= 128):
                rt.syncwarp()
            for k in cutlass.range_constexpr(n_k):
                it = lane_0 + 128 * k
                if it < n:
                    payload = rt.TaggedMailbox.unpack24(
                        [packets[4 * k + i] for i in range(4)]
                    )
                    for i in cutlass.range_constexpr(3):
                        s.received_words[
                            rt.cdiv(it, 8) * 24 + rt.cmod(it, 8) * 3 + i
                        ] = payload[i]

    def load_pending(self, p, o_pending, first):
        for h in range(2):
            v = rt.Memory.ld_volatile_v4(rt.ga(p.Handoff, first + 4 * h))
            for i in range(4):
                o_pending[4 * h + i] = v[i]

    @staticmethod
    def prmt_word(got, k, i):
        return rt.Convert.prmt(got[8 * k + 2 * i], got[8 * k + 2 * i + 1], 0x5410)

    @staticmethod
    def prmt_store(addr, got, k):
        rt.Memory.sts_v4(addr, [DraftWorker.prmt_word(got, k, i) for i in range(4)])
