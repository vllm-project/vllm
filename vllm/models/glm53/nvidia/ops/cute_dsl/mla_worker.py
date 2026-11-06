# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL GLM-5.3 MLA decode worker kernels (one class per role; 384
threads per CTA).
"""

# ruff: noqa: B023  (one lambda per traced loop iteration reads the loop variables)

import dataclasses
import functools
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint16, Uint32, Uint64
from cutlass.base_dsl.typing import GridConstant
from cutlass.experimental.cuda.tensor_map import TensorMap

from . import _rt as rt
from .fused_moe import LAYOUT
from .row_table import RowTableLayout

const_expr = cutlass.const_expr
NEG_FLT_MAX = -3.4028234663852886e38
L2_HINT = 0x12F0000000000000

BAR_COMPUTE, BAR_LOADER, BAR_ALL = (8, 256), (9, 128), (11, 384)
INPUT_CTAS = 82
X_ROW_BYTES = 12288
TOPK = 2048
SPLIT_KEYS, SPLITS = (64, 32)
NO_KEY = 4194304  # past every paged row
NO_ROW = 2147483647  # out of bounds: the TMA tile reads zeros
TAG_MASK = 0xFFFF0000


class Ring:
    """The four-slot shared-memory ring of weight pages and KV gathers (``SMEM.ring``,
    full/empty mbarriers).
    """

    SLOTS = 4
    SLOT_BYTES = 36864
    WEIGHT_PAGE_BYTES = 32896
    KV_PLANE_BYTES = 8192

    @staticmethod
    def slot_offset(seq):
        return rt.cmod(seq, Ring.SLOTS) * Ring.SLOT_BYTES


LOOP_KV_FIRST, LOOP_PE_STRIDE = 2, 10240


class KvRows:
    """The paged MLA latent cache (KVCachePtr and PECache view the same pool), 64 rows
    per block.
    """

    BLOCK_ROWS, BLOCK_SHIFT = 64, 6
    PITCH, ROPE_OFFSET = 576, 512
    BLOCK_BYTES = BLOCK_ROWS * PITCH

    @staticmethod
    def element(block, position, column) -> Int64:
        return block * (KvRows.BLOCK_BYTES // 2) + Int64(
            rt.cmod(position, KvRows.BLOCK_ROWS) * (KvRows.PITCH // 2) + column
        )

    @staticmethod
    def rope_byte(PECache, BlockTable, position, index):
        return rt.baddr(
            PECache,
            Int64(BlockTable[position >> KvRows.BLOCK_SHIFT]) * KvRows.BLOCK_BYTES
            + Int64(
                rt.cmod(position, KvRows.BLOCK_ROWS) * KvRows.PITCH
                + KvRows.ROPE_OFFSET
                + index
            ),
        )

    @staticmethod
    def store_rope_e4m3(address, value) -> None:
        addr = Int64(address)
        e4m3 = rt.Convert.f32x2_to_e4m3x2(value, value)
        rt.Core.ptx("st.global.b8 [$0], $1;", addr, e4m3, memory=False)

    @staticmethod
    def rope_pair_bf16(address) -> Uint32:
        return rt.Convert.e4m3x2_to_bf16x2(rt.Memory.gview(Uint16, address, 1, 2)[0])

    @staticmethod
    def store_rope_pair(address, word) -> None:
        high = rt.Convert.bits_f32(word & Uint32(0xFFFF0000))
        low = rt.Convert.bits_f32(word << Uint32(16))
        rt.Memory.gview(Uint16, address, 1, 2)[0] = rt.Convert.f32x2_to_e4m3x2(
            high, low
        )


class IndexerOperands:
    """The split attention's operand tile in ``act`` (bytes from ``SMEM.act``), filled
    from Handoff.
    """

    Q_CHUNKS = 512
    CHUNKS = 576
    PE = 16384

    @staticmethod
    def q_half(head, group, half):
        return (
            rt.cdiv(group, 64) * 2048
            + head * 128
            + ((rt.cmod(group, 64) * 2) ^ (rt.cmod(head, 8) * 16))
            + half
        )

    @staticmethod
    @cute.jit
    def store(act_addr, chunk, wa, wb):
        if chunk < IndexerOperands.CHUNKS:
            if chunk < IndexerOperands.Q_CHUNKS:
                head = rt.cdiv(chunk, 64)
                group = rt.cdiv(rt.cmod(chunk, 64), 2) * 16
                half = rt.cmod(chunk, 2) * 8
                rt.Memory.st_shared_v2(
                    act_addr + Uint32(IndexerOperands.q_half(head, group, half)),
                    rt.Convert.prmt(wa[0], wa[1], 0x5410),
                    rt.Convert.prmt(wb[0], wb[1], 0x5410),
                )
                rt.Memory.st_shared_v2(
                    act_addr + Uint32(IndexerOperands.q_half(head, group + 8, half)),
                    rt.Convert.prmt(wa[2], wa[3], 0x5410),
                    rt.Convert.prmt(wb[2], wb[3], 0x5410),
                )
            if chunk >= IndexerOperands.Q_CHUNKS:
                row = chunk - IndexerOperands.Q_CHUNKS
                offset = (
                    IndexerOperands.PE
                    + rt.cdiv(row, 8) * 128
                    + (rt.cmod(row, 8) ^ rt.cmod(rt.cdiv(row, 8), 8)) * 16
                )
                rt.Memory.sts_v4(
                    act_addr + Uint32(offset),
                    (
                        rt.Convert.prmt(wa[0], wa[1], 0x5410),
                        rt.Convert.prmt(wa[2], wa[3], 0x5410),
                        rt.Convert.prmt(wb[0], wb[1], 0x5410),
                        rt.Convert.prmt(wb[2], wb[3], 0x5410),
                    ),
                )


class Handoff:
    """The target roles' ``Handoff`` words (u32): rows one CTA hands to the others
    within a launch.
    """

    Q_DOWN, Q_DOWN_ROW = 0, 2048
    KV_LATENT, KV_LATENT_ROW = 8192, 512
    INDEXER_PE, INDEXER_PE_ROW = 16384, 512
    INDEXER_Q, INDEXER_Q_ROW = 18432, 4096
    VO, VO_ROW = 51200, 2048
    ROLE_TABLE, SPLIT_DONE = 59392, 59540
    WORDS = 59680
    ROLE_MASK, POOL_SHIFT = 65535, 16
    MIRROR = 1 << 23


SELECTION_BYTES = 3020800
PUSH_MAILBOX_BYTES = 1009792


class PushedRows:
    """The draft's pushed-latent mailbox in a selector rank's arena (u64 records from
    the peer base).
    """

    DRAFT = 917504
    TARGET = 917632
    PE_RECORDS = 256

    @staticmethod
    def record(base, parity, token, column):
        return base + ((parity * 4 + token) * 288 + column) * 8


class FlagWords:
    """Word indices of the per-rank ``Flags`` array the worker reads and writes."""

    PE_READY = 80
    KV_READY = 148
    KV_LATENT_READY = KV_READY + 128
    SPLIT_DONE = 296
    MOE_COUNT = 14652
    MOE_PACING = 14653
    MOE_SOURCE = 14658
    ROUTER_BIAS = 14686
    ROUTER_WEIGHT = 14690
    SHARED_INDEX = 15803
    STAMPS = 15756


kFull, kPush, kPushed, kShared, kSeam = range(5)

INPUT_PAGES = (2624, 6144, 128, 32, 1024, True)
O_PAGES = (6144, 2048, 64, 48, 512, False)
KVB_PAGES = (3584, 512, 64, 32, 512, False)


class _Args(rt.Params):
    """The worker's params records: views of up to 2^31 - 1 elements."""

    EXTENT = (1 << 31) - 1
    UNSIGNED = {"KVCachePtr": Uint16} | dict.fromkeys(
        ("SelectionMailbox", "Handoff", "FullEpoch", "Flags"), Uint32
    )
    UNSIGNED |= dict.fromkeys(
        ("InputWeight", "QueryWeight", "AbsorbWeight", "KVbWeight", "OWeight"), Uint8
    )
    READ_ONLY = {"Freq": Float32} | dict.fromkeys(
        ("CurPos", "BlockTable", "RowTable"), Int32
    )


@rt.record
class HeadArgs(_Args):
    """The pointers before the KvCache tensor map."""

    X: cute.Pointer = rt.ptr("bfloat16")
    GammaX: cute.Pointer = rt.ptr("float32")
    InputWeight: cute.Pointer = rt.ptr("float8_e4m3")
    InputScales: cute.Pointer = rt.ptr("uint8")
    GammaQ: cute.Pointer = rt.ptr("float32")
    QueryWeight: cute.Pointer = rt.ptr("float8_e4m3")
    AbsorbWeight: cute.Pointer = rt.ptr("float8_e4m3")
    GammaKV: cute.Pointer = rt.ptr("float32")
    Freq: cute.Pointer = rt.ptr("float32")
    CurPos: cute.Pointer = rt.ptr("int32")


@rt.record
class TailArgs(_Args):
    """The pointers and scalars after the KvCache tensor map, up to BlockTable."""

    KVCachePtr: cute.Pointer = rt.ptr("uint16")
    PECache: cute.Pointer = rt.ptr("bfloat16")
    SelectionMailbox: cute.Pointer = rt.ptr("uint32")
    Handoff: cute.Pointer = rt.ptr("uint32")
    PartialO: cute.Pointer = rt.ptr("float32")
    PartialLSE: cute.Pointer = rt.ptr("float32")
    KVbWeight: cute.Pointer = rt.ptr("float8_e4m3")
    KVbScales: cute.Pointer = rt.ptr("uint8")
    OWeight: cute.Pointer = rt.ptr("float8_e4m3")
    OScales: cute.Pointer = rt.ptr("uint8")
    Hidden: cute.Pointer = rt.ptr("bfloat16")
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
    BlockTable: cute.Pointer = rt.ptr("int32")
    RowTable: cute.Pointer = rt.ptr("int32", align=4)
    BlockStride: Int32


@rt.record
class SeamArgs(_Args):
    """kSeam's Extra pointer after the tail record: the seam input."""

    XG: cute.Pointer = rt.ptr("bfloat16")


@rt.record
class WorkerParams:
    """Every kernel parameter as a typed view, built once at the top of the kernel."""

    X: "cute.Tensor"
    GammaX: "cute.Tensor"
    InputWeight: "rt.TypedView"
    InputScales: "cute.Tensor"
    GammaQ: "cute.Tensor"
    QueryWeight: "rt.TypedView"
    AbsorbWeight: "rt.TypedView"
    GammaKV: "cute.Tensor"
    Freq: "rt.ReadOnlyView"
    CurPos: "rt.ReadOnlyView"
    KVCachePtr: "rt.TypedView"
    PECache: "cute.Tensor"
    SelectionMailbox: "rt.TypedView"
    Handoff: "rt.TypedView"
    PartialO: "cute.Tensor"
    PartialLSE: "cute.Tensor"
    KVbWeight: "rt.TypedView"
    KVbScales: "cute.Tensor"
    OWeight: "rt.TypedView"
    OScales: "cute.Tensor"
    Hidden: "cute.Tensor"
    Final: "cute.Tensor"
    FullEpoch: "rt.TypedView"
    Flags: "rt.TypedView"
    BlockTable: "rt.ReadOnlyView"
    RowTable: "rt.ReadOnlyView"
    BlockStride: Int32
    peer_rank: Int32
    peer_bases: tuple
    KvCache: Int64
    InputTmap: Int64 | None = None
    OTmap: Int64 | None = None
    KvbTmap: Int64 | None = None
    XG: "cute.Tensor | None" = None

    @staticmethod
    def of(KvCache, *records, **maps) -> "WorkerParams":
        addresses = {"KvCache": KvCache.get_ptr().toint()}
        addresses.update({name: m.get_ptr().toint() for name, m in maps.items()})
        views = [r.views() for r in records]
        values = {
            f.name: getattr(v, f.name) for v in views for f in dataclasses.fields(v)
        }
        bases = tuple(values.pop(f"peer_base{k}") for k in range(8))
        return WorkerParams(**values, peer_bases=bases, **addresses)


class SMEM(rt.SmemLayout):
    """The worker's shared-memory layout (byte offsets from the dynamic shared base)."""

    a = rt.SmemAlloc(227456)
    ring_full = a.mbarriers(4, arrivals=1)
    ring_empty = a.mbarriers(4, arrivals=256)
    tail_full = a.mbarriers(12, arrivals=1)
    a.pad(1024 - 160)
    scratch = a.array(Float32, 2304, align=16)
    with a.union():
        with a.arm():
            indices = a.array(Int32, 64, align=16)
            group_flags = a.array(Int32, 16, align=16)
            id_probe = a.array(Int32, 2, align=8)
            warp_hits = a.array(Int32, 46, align=8)
            pairs = a.array(Float32, 128, align=16)
            merge_weights = a.array(Float32, 32, align=16)
            maxes = a.array(Float32, 96, align=16)
            residual = a.array(BFloat16, 256, align=16)
            local_o = a.array(BFloat16, 256, align=16, also=(Uint32,))
            received_words = a.array(Uint32, 672, align=16)
        with a.arm():
            moe_keys = a.array(Uint32, 1024, align=16)
            moe_picks = a.array(Int32, 32, align=16)
            moe_listed = a.array(Int32, 32, align=16)
            moe_votes = a.array(Int32, 256, align=16)
    epoch_slot = a.array(Uint32, 64, align=16)
    row_words = a.array(Int32, 128, align=16)
    with a.union(align=1024):
        ring = a.array(
            Uint64, Ring.SLOTS * Ring.SLOT_BYTES // 8, align=16, also=(Float32,)
        )
        with a.arm():
            refine_control = a.array(Int32, 4, align=16)
            refine_warps = a.array(Int32, 8, align=16)
            refine_in = a.array(Uint32, 4, align=16)
            refine_tie_bits = a.array(Uint32, 240, align=16)
            refine_hist = a.array(Uint32, 1024, align=16)
            refine_cache = a.array(Uint32, 32480, align=16)
            refine_ties = a.array(Uint32, 3104, align=16)
    act = a.array(Uint32, 12544, align=16)
    absorb = a.array(Uint8, 12416, align=16)
    ORDER = (
        "scratch",
        "indices",
        "group_flags",
        "pairs",
        "merge_weights",
        "residual",
        "local_o",
        "received_words",
        "epoch_slot",
        "ring",
        "act",
        "moe_keys",
        "id_probe",
        "warp_hits",
        "maxes",
        "moe_picks",
        "moe_listed",
        "moe_votes",
    )


@dataclass(frozen=True)
class WorkerSmem:
    """Thread indices and the shared base; ``at_entry`` adds the ``SMEM`` record
    (mbarriers, views).
    """

    tid: Int32
    warp: Int32
    lane: Int32
    bid: Int32
    smem: Int32
    role_entry: Uint32 | None
    moe_list_full: Int32 | None

    @staticmethod
    def thread_context(explicit_smem_base: bool):
        tid, _, _ = cute.arch.thread_idx()
        tid = Int32(tid)
        warp = rt.Sync.warp_uniform(rt.cdiv(tid, 32))
        lane = rt.cmod(tid, 32)
        smem_raw = cute.arch.get_dyn_smem(Uint8, alignment=1024)
        if explicit_smem_base:
            smem = rt.Memory.cvta_shared_u32(smem_raw.toint())
            smem = rt.Sync.warp_uniform(smem)
        else:
            smem = Int32(smem_raw.toint())
        bid, _, _ = cute.arch.block_idx()
        return tid, warp, lane, smem, Int32(bid)

    @staticmethod
    def build(worker: "MlaWorker", p: WorkerParams) -> "WorkerSmem":
        tid, warp, lane, smem, bid = WorkerSmem.thread_context(
            worker.EXPLICIT_SMEM_BASE
        )
        moe_list_full = worker.MOE_LIST_FULL
        return WorkerSmem(
            tid=tid,
            warp=warp,
            lane=lane,
            bid=bid,
            smem=smem,
            role_entry=worker.role_entry(p, bid),
            moe_list_full=None if moe_list_full is None else smem + moe_list_full,
        )

    def at_entry(self, worker: "MlaWorker") -> "WorkerSmem":
        smem = self.smem
        mbars = {
            name: None if off is None else smem + off
            for name, off in worker.mbars().items()
        }
        context = {
            f.name: getattr(self, f.name) for f in dataclasses.fields(WorkerSmem)
        }
        return SMEM.record(
            smem,
            ring="pages",
            order=SMEM.ORDER,
            typed=True,
            bases=(WorkerSmem,),
            **context,
            **mbars,
        )


class stage:
    """Decorator of a worker stage method: its ``WorkerSmem`` argument is re-derived
    where it is called.
    """

    def __init__(self, fn):
        self.fn = fn
        functools.update_wrapper(self, fn)  # type: ignore[arg-type]

    def __get__(self, worker: "MlaWorker", owner: type | None = None):
        if worker is None:
            return self
        fresh = worker.smem_record.at_entry(worker)

        def call(*args, **kwargs):
            return self.fn(
                worker,
                *(fresh if isinstance(a, WorkerSmem) else a for a in args),
                **kwargs,
            )

        return call


class RowGroups:
    """What a kernel needs to serve a row tier: the build's row capacity and the row
    table (``row_table.py``) layout it reads, and the loop over row groups.
    """

    #: Row capacity (Rmax) and request capacity (Bmax) of the build.
    ROWS = 4
    REQUESTS = 1
    GROUP_ROWS = 4
    SELECTION_BYTES = SELECTION_BYTES
    GROUP_STRIDES: dict[str, int] = {}
    #: First row of the group being traced (a dynamic value inside the group loop).
    row0 = 0
    PASS_ROWS = 4
    WIDE = False
    WIDE_TAIL = 0

    @property
    def PROJ_ROWS(self) -> int:
        return ProjectionChunks.block_rows(self.PASS_ROWS)

    @property
    def PHASED(self) -> bool:
        return self.PASS_ROWS > 4

    @property
    def PASSES(self) -> int:
        return self.ROWS // self.PASS_ROWS

    @property
    def PASS_GROUPS(self) -> int:
        return self.PASS_ROWS // self.GROUP_ROWS

    @property
    def layout(self) -> RowTableLayout:
        return RowTableLayout(self.ROWS, self.REQUESTS)

    @property
    def GROUPS(self) -> int:
        return self.ROWS // self.GROUP_ROWS

    PREFETCH_LAST = False
    REFINE = True
    ATTN_LOOP = False

    def prefetch_slot(self, p, rid):
        if self.PREFETCH_LAST:
            return cutlass.select_(self.prefetch_group(p), rid, Int32(0))
        return rid

    def prefetch_group(self, p):
        if self.PREFETCH_LAST:
            return Int32(p.RowTable[0]) <= self.row0 + self.GROUP_ROWS
        return True

    def enter_group(self, group) -> None:
        self.row0 = group * self.GROUP_ROWS

    def table(self, record):
        return cute.make_tensor(record.RowTable, cute.make_layout(self.layout.words))

    def group_live(self, record, group):
        return self.table(record)[0] > group * self.GROUP_ROWS

    def position(self, p):
        layout = self.layout
        if cutlass.const_expr(self.GROUPS == 1):
            return p.RowTable[layout.context_word(0)] - 1
        request = p.RowTable[layout.row_request_word(0) + self.row0]
        first = p.RowTable[layout.first_row_word(0) + request]
        return p.RowTable[layout.context_word(0) + request] - 1 + self.row0 - first

    def move_to_group(self, record, group, block_row=None):
        def request(table):
            return table[self.layout.row_request_word(0) + group * self.GROUP_ROWS]

        return self.move_record(
            record, group, block_row, request, lambda name, stride: group * stride
        )

    def move_record(self, record, group, block_row, request, offset):
        """Return ``record`` moved to row group ``group`` (fields by ``offset``)."""
        names = {f.name for f in dataclasses.fields(record)}
        strides = self.GROUP_STRIDES | {"SelectionMailbox": self.SELECTION_BYTES // 4}
        moved = {
            name: getattr(record, name) + offset(name, stride)
            for name, stride in strides.items()
            if name in names
        }
        for name in names:
            if name.startswith("peer_base"):
                moved[name] = getattr(record, name) + Int64(group) * PUSH_MAILBOX_BYTES
            elif name.startswith("selection_base"):
                moved[name] = getattr(record, name) + Int64(group) * (
                    self.SELECTION_BYTES
                )
        if "BlockTable" in names:
            row = block_row
            if block_row is None:
                table = self.table(record)
                row = table[self.layout.block_row_word(0) + request(table)]
            moved["BlockTable"] = record.BlockTable + row * record.BlockStride
        return dataclasses.replace(record, **moved)


class PassPhases:
    """Phased passes: how a worker serves more than four rows per pass."""

    #: First ``epoch_slot`` word that keeps the epochs of the pass's groups.
    EPOCHS = 16


class ProjectionChunks:
    """The wide input projection of a phased pass (``WIDE`` builds), phase ``proj``."""

    PAGES = 6
    W_BYTES = 32768
    X_ROW_BYTES = 12288
    X_SLICE = 2048
    X_PITCH = 2064
    MAX_ROWS = 32
    #: First spare word of a Handoff block (after the split-done counters).
    RECORD_WORD = 59668
    #: ``epoch_slot`` words: the group's block-table row and position, by group in pass.
    BLOCK_ROW, POSITION = 32, 48

    @staticmethod
    def block_rows(rows: int) -> int:
        return rows // -(-rows // ProjectionChunks.MAX_ROWS)

    @staticmethod
    def offsets(block_rows: int) -> tuple[int, int, int, int]:
        x_bytes = block_rows * ProjectionChunks.X_PITCH
        x0 = ProjectionChunks.W_BYTES
        x1 = x0 + x_bytes
        w1 = -(-(x1 + x_bytes) // 1024) * 1024
        return 0, x0, w1, x1


class PackedRows:
    """Per-row request metadata of a packed build (``MlaWorker.PACKED``)."""

    BASE, BLOCK_ROW, TOKEN, ROWS = 0, 32, 64, 96

    @staticmethod
    def meta(layout, table, row, query_):
        rows = Int32(table[0])
        live = row < rows
        eff = cutlass.select_(live, row, rows - 1)
        request = table[layout.row_request_word(0) + eff]
        first = table[layout.first_row_word(0) + request]
        context = table[layout.context_word(0) + request]
        block_row = table[layout.block_row_word(0) + request]
        position = context - 1 + eff - first
        return position - query_, block_row, eff - (row - query_), Int32(live)


class PassContext:
    """Trace-time handle of one phased pass: the kernel's records and tensor maps, from
    which :meth:`params` builds the parameter view of any of the pass's row groups.
    """

    def __init__(self, worker, KvCache, records, maps):
        self.worker, self.KvCache, self.records, self.maps = (
            worker,
            KvCache,
            records,
            maps,
        )
        #: First row group of the pass being traced (a dynamic value).
        self.first = 0

    def begin(self, passed) -> None:
        self.first = passed * self.worker.PASS_GROUPS

    def live(self, group):
        return self.worker.group_live(self.records[1], group)

    def params(self, group, s=None, query=None) -> "WorkerParams":
        self.worker.enter_group(group)
        row = None
        if self.worker.PACKED:
            row = Int32(0)
            if s is not None and query is not None:
                cached = PackedRows.BLOCK_ROW + (group - self.first) * 4 + query
                row = Int32(s.row_words[cached])
        elif s is not None:
            row = Int32(s.epoch_slot[ProjectionChunks.BLOCK_ROW + group - self.first])
        moved = [self.worker.move_to_group(r, group, row) for r in self.records]
        return WorkerParams.of(self.KvCache, *moved, **self.maps)


class MlaWorker(RowGroups, rt.RoleKernel):
    """Shared base of the worker roles: launch, the per-kernel records and the stages
    every role uses.
    """

    ROLE: int | None = None
    GRID = 148
    BLOCK = 384
    SMEM_BYTES = 215040
    ROLES = (
        rt.Role("compute", 0, 7, regs=216),
        rt.Role("load", 8, 11, regs=72, increase=False),
    )
    EXPLICIT_SMEM_BASE = False
    RECORDS: tuple[type, ...] = (HeadArgs, TailArgs)
    HOST_ARGS = (("KvBytes", Int64),)
    X_FULL: int | None = None
    NORM_FREE: int | None = None
    MERGE_LOADED: int | None = None
    ABSORB_FULL: int | None = None
    MOE_LIST_FULL: int | None = None
    full_or_push = shared_or_seam = push = pushed = seam = False
    # Set while tracing (smem_record per kernel, ctx per pass).
    smem_record: "WorkerSmem" = None  # type: ignore[assignment]
    phase = "head"
    ctx: "PassContext" = None  # type: ignore[assignment]
    PACKED = False

    def set_phase(self, phase: str, ctx: "PassContext | None" = None) -> None:
        self.phase = phase
        if ctx is not None:
            self.ctx = ctx

    def live_rows(self, p0):
        rows = Int32(p0.RowTable[0])
        if self.PACKED:
            rows = (
                rt.cdiv(rows + self.GROUP_ROWS - 1, self.GROUP_ROWS) * self.GROUP_ROWS
            )
        return rows

    @staticmethod
    def own_row(rid):
        return cutlass.select_(
            rid < 128, rt.cdiv(rid, 32), cutlass.select_(rid < 132, rid - 128, Int32(0))
        )

    def step_generation(self, p) -> Uint32:
        return Uint32(p.RowTable[self.layout.step_word]) + 1

    def row_info(self, p, s, h, query_):
        if h is not None:
            at = h * self.GROUP_ROWS + query_
            return tuple(
                Int32(s.row_words[word + at])
                for word in (PackedRows.BASE, PackedRows.BLOCK_ROW, PackedRows.TOKEN)
            )
        base, block_row, token, _ = PackedRows.meta(
            self.layout, p.RowTable, self.row0 + query_, query_
        )
        return base, block_row, token

    def position(self, p):
        if self.PACKED:
            query_ = self.own_row(self.role_slot(self.smem_record))  # type: ignore[attr-defined]
            return self.row_info(p, self.smem_record, None, query_)[0]
        return RowGroups.position(self, p)

    GROUP_STRIDES = {
        "X": 4 * 6144,
        "XG": 4 * 6144,
        "Freq": 4 * 64,
        "Hidden": 4 * 6144,
        "Final": 4 * 6144,
        "SelectionMailbox": SELECTION_BYTES // 4,
        "Handoff": Handoff.WORDS,
        "PartialO": 2097152 // 4,
        "PartialLSE": 6144 // 4,
        "FullEpoch": 4,
        "Flags": 15804,
    }

    @staticmethod
    def kv_map(tail, kv_bytes):
        rows = (kv_bytes - KvRows.ROPE_OFFSET) // KvRows.PITCH + 1
        latent = [KvRows.ROPE_OFFSET, rows]
        return rt.byte_map(
            tail.KVCachePtr.toint(), latent, [KvRows.PITCH], [128, 1], True, True
        )

    @staticmethod
    def page_map(weight, rows, columns, chunk, page_rows, page_k, swizzle):
        dims, box = [chunk, rows, columns // chunk], [chunk, page_rows, page_k // chunk]
        return rt.byte_map(weight.toint(), dims, [columns, chunk], box, swizzle)

    def mbars(self) -> dict:
        return dict(
            x_full=self.X_FULL,
            norm_free=self.NORM_FREE,
            merge_loaded=self.MERGE_LOADED,
            absorb_full=self.ABSORB_FULL,
        )

    def role_entry(self, p: WorkerParams, bid):
        return None

    def role_table_word(self, p: WorkerParams, word):
        addr = rt.ga(p.Handoff, word)
        if const_expr(self.GROUPS > 1):
            addr = addr - Int64(rt.cdiv(self.row0, self.GROUP_ROWS)) * (
                Handoff.WORDS * 4
            )
        return addr

    def kernel_extras(self, head, tail, extras: list) -> tuple:
        return ()

    @cute.jit
    def enter(self, p: WorkerParams):
        ids, s = self.prologue(p)
        self.run(p, s, ids)

    @cute.jit
    def launch(self, *args):
        *buffers, kv_bytes, stream = args
        records = []
        for record in self.RECORDS:
            n = len(dataclasses.fields(record))
            records.append(record(*buffers[:n]))
            buffers = buffers[n:]
        head, tail, *extras = records
        self.kernel(
            head,
            self.kv_map(tail, kv_bytes),
            tail,
            *self.kernel_extras(head, tail, extras),
        ).launch(
            grid=(self.GRID, 1, 1),
            block=(self.BLOCK, 1, 1),
            stream=stream,
            use_pdl=self.USE_PDL,
            **self._launch_bounds(),
        )

    @stage
    @cute.jit
    def release_flag(self, p, s, index):
        with rt.warp_leader(s.warp, 0):
            rt.Memory.red_async_release_gpu_add(
                rt.ga(p.Flags, index), Uint32(1), memory=True
            )

    @cute.jit
    def poll_flags(self, p, first, n: cutlass.Constexpr, lane_0, epoch):
        self.warp_poll_ge(
            rt.ga(p.Flags, first + self.min_i32(lane_0, n - 1)),
            Int32(lane_0 < n),
            epoch + 1,
        )
        rt.syncwarp()
        rt.Memory.fence_acquire_gpu()

    @stage
    @cute.jit
    def rows_to_scratch(
        self, s, accum, rows: cutlass.Constexpr, n_m: cutlass.Constexpr
    ):
        row_stride = 16 * n_m
        group = rt.cdiv(s.lane, 4)
        pair = rt.cmod(s.lane, 4) * 2
        if pair < rows:
            for m in cutlass.range_constexpr(n_m):
                base = (
                    s.warp * (rows * row_stride + 1)
                    + pair * row_stride
                    + m * 16
                    + group
                )
                s.scratch[base] = accum[m * 4]
                s.scratch[base + 8] = accum[m * 4 + 2]
                if pair + 1 < rows:
                    s.scratch[base + row_stride] = accum[m * 4 + 1]
                    s.scratch[base + row_stride + 8] = accum[m * 4 + 3]
        rt.bar_sync(*BAR_COMPUTE)

    @stage
    def warp_sum(self, s, i, rows: int, n_m: int):
        return rt.Reduce.ordered_sum(
            lambda k: s.scratch[k * (rows * 16 * n_m + 1) + i], 8
        )

    @stage
    def rope(self, p, s, rounded, dim, row=None):
        partner = cute.arch.shuffle_sync_bfly(rounded, 1)
        row = rt.cdiv(s.tid, 32) if row is None else row
        angle = row * 64 + (rt.cmod(dim, 64) & -2)
        cosine, sine = p.Freq[angle], p.Freq[angle + 1]
        odd = rt.fma(rounded, cosine, rt.mul(partner, sine))
        even = rt.fma(rounded, cosine, -(rt.mul(partner, sine)))
        return cutlass.select_(rt.cmod(dim, 2) == 1, odd, even)

    def kv_row_off(self, p, position, column):
        return KvRows.element(
            Int64(p.BlockTable[position >> KvRows.BLOCK_SHIFT]), position, column
        )

    @cute.jit
    def poll_tagged16(
        self,
        chunks,
        tag_high,
        on_ready,
        backoff=None,
        converged: cutlass.Constexpr = False,
        reconverge: cutlass.Constexpr = True,
    ):
        done = Int32(0)
        if const_expr(converged):
            while rt.Sync.vote_any(done == 0) != 0:
                if done == 0:
                    done = self.tagged16_pass(chunks, tag_high, on_ready, backoff)
        else:
            while done == 0:
                done = self.tagged16_pass(chunks, tag_high, on_ready, backoff)
        if const_expr(reconverge):
            rt.syncwarp()

    @cute.jit
    def tagged16_pass(self, chunks, tag_high, on_ready, backoff):
        done = Int32(0)
        words = []
        for k in cutlass.range_constexpr(len(chunks)):
            words.append(rt.Memory.ld_volatile_v4(chunks[k]))
            words.append(rt.Memory.ld_volatile_v4(chunks[k] + 16))
        stale = Uint32(0)
        for k in cutlass.range_constexpr(len(chunks)):
            for r in cutlass.range_constexpr(4):
                stale = (
                    stale
                    | ((words[2 * k][r] & Uint32(TAG_MASK)) ^ tag_high)
                    | ((words[2 * k + 1][r] & Uint32(TAG_MASK)) ^ tag_high)
                )
        if stale == 0:
            on_ready(words)
            done = Int32(1)
        if const_expr(backoff is not None):  # noqa: SIM102
            if (done == 0) & (backoff != 0):
                self.spin_ns(128)
        return done

    @staticmethod
    @cute.jit
    def spin_ns(ns: cutlass.Constexpr):
        begin = Uint32(rt.globaltimer())
        waited = Int32(0)
        while waited == 0:
            if Uint32(rt.globaltimer()) - begin >= Uint32(ns):
                waited = Int32(1)

    def handoff_word(self, p, word, pool=0):
        if self.shared_or_seam:
            return rt.ga(p.Handoff, word + pool * Handoff.MIRROR)
        return rt.ga(p.Handoff, word)

    def mailbox(self, p, s, word, pool=0):
        return self.handoff_word(p, word, pool)

    def handoff(self, p, word):
        return self.handoff_word(p, word)

    @stage
    @cute.jit
    def kv_latent_store(self, p, s, hbase, query_, position, tag, live=None):
        column = s.tid * 2
        tag_high = tag << 16
        address = self.mailbox(p, s, hbase + column, 1)
        pair = Uint32(0)
        done = Int32(0)
        while done == 0:
            low = rt.Memory.ld_relaxed_gpu(address)
            high = rt.Memory.ld_relaxed_gpu(address + 4)
            pair = rt.Convert.prmt(low, high, 0x5410)
            if (((low ^ tag_high) | (high ^ tag_high)) & Uint32(TAG_MASK)) == 0:
                done = Int32(1)
        rt.syncwarp()
        x, y = rt.Convert.bf16x2_to_f32x2(pair)
        square = rt.fma(x, x, rt.mul(y, y))
        for step in cutlass.range_constexpr(5):
            square = rt.add(square, cute.arch.shuffle_sync_down(square, 16 >> step))
        if s.lane == 0:
            s.scratch[s.warp] = square
        rt.bar_sync(*BAR_COMPUTE)
        total = Float32(0.0)
        for pr in cutlass.range_constexpr(4):
            total = rt.add(total, rt.add(s.scratch[pr * 2], s.scratch[pr * 2 + 1]))
        inv = rt.FastMath.rsqrt(rt.fma(total, 0.001953125, 1e-05))
        low_f = rt.Convert.bf16_round(rt.mul(x, rt.mul(inv, p.GammaKV[column])))
        high_f = rt.Convert.bf16_round(rt.mul(y, rt.mul(inv, p.GammaKV[column + 1])))
        e4m3x2 = rt.Convert.f32x2_to_e4m3x2(high_f, low_f)
        row = position + query_
        if const_expr(live is None):
            at = self.kv_row_off(p, row, s.tid)
            rt.Memory.gview(Uint16, rt.ga(p.KVCachePtr, at), 1, 2)[0] = e4m3x2
        else:
            if live != 0:
                at = self.kv_row_off(p, row, s.tid)
                rt.Memory.gview(Uint16, rt.ga(p.KVCachePtr, at), 1, 2)[0] = e4m3x2

    @staticmethod
    def p_word(row, key):
        within = rt.cmod(key, 16)
        col = (
            rt.cdiv(key, 16) * 16
            + rt.cdiv(within, 4) * 2
            + rt.cmod(rt.cdiv(within, 2), 2) * 8
        )
        return rt.cdiv(26624 + row * 128 + ((col * 2) ^ ((row & 7) << 4)), 4)

    @staticmethod
    @cute.jit
    def lse_tail(head_sum, head_max):
        head_inverse = rt.rmem_tensor(2, Float32)
        head_lse = rt.rmem_tensor(2, Float32)
        for h in cutlass.range_constexpr(2):
            head_inverse[h] = Float32(0.0)
            head_lse[h] = Float32(float("-inf"))
            if rt.FastMath.fcmp(">", head_sum[h], Float32(0.0)):
                head_inverse[h] = rt.FastMath.rcp_rn(head_sum[h])
                log2_sum = rt.FastMath.lg2_approx(head_sum[h])
                head_lse[h] = rt.fma(
                    head_max[h], Float32(0.09016843885183334), log2_sum
                )
        return head_inverse, head_lse

    @stage
    @cute.jit
    def merge_weights(
        self, p, s, split_count, lse_first, lse_stride: cutlass.Constexpr
    ):
        lse = Float32(NEG_FLT_MAX)
        if split_count > s.lane:
            lse = p.PartialLSE[lse_first + s.lane * lse_stride]
        maximum = lse
        for step in cutlass.range_constexpr(5):
            maximum = rt.max_noftz(
                maximum, cute.arch.shuffle_sync_bfly(maximum, 16 >> step)
            )
        total = cute.math.exp2(rt.FastMath.sub(lse, maximum), fastmath=False)
        for step in cutlass.range_constexpr(5):
            total = rt.add(total, cute.arch.shuffle_sync_bfly(total, 16 >> step))
        global_lse = rt.add(maximum, rt.FastMath.lg2_approx(total))
        weight = Float32(0.0)
        if split_count > s.lane:
            weight = cute.math.exp2(rt.FastMath.sub(lse, global_lse), fastmath=False)
        s.merge_weights[s.lane] = weight

    @stage
    def merge_sum(self, s, value, parts, split_base):
        for i in range(4):
            value[i] = Float32(0.0)
        for si in range(16):
            weight = s.merge_weights[split_base + si]
            for i in range(4):
                value[i] = rt.fma(parts[si][i], weight, value[i])

    @stage
    @cute.jit
    def fold_halves(self, s, value):
        if s.tid >= 128:
            for i in cutlass.range_constexpr(4):
                s.scratch[(s.tid - 128) * 4 + i] = value[i]
        rt.bar_sync(*BAR_COMPUTE)
        if s.tid < 128:
            for i in cutlass.range_constexpr(4):
                value[i] = rt.add(value[i], s.scratch[s.tid * 4 + i])

    @stage
    def tagged_push(self, p, s, bases, epoch, token, packed_peer, chunk, slot=None):
        slot = s.bid if slot is None else slot
        rt.OExchange.push(
            s.local_o_words, bases, p.peer_rank, epoch, token, packed_peer, chunk, slot
        )

    @stage
    def tagged_reduce(self, p, s, query_):
        rt.OExchange.reduce(
            s.local_o_words, s.local_o_addr, p.peer_rank, query_, s.lane
        )

    @stage
    def final_residual(self, p, s, rows: int, slot=None):
        slot = s.bid if slot is None else slot
        for r in range(rows):
            o = Float32(s.local_o[r * 48 + s.tid])
            p.Final[r * 6144 + slot * 48 + s.tid] = BFloat16(
                rt.add(o, Float32(s.residual[r * 48 + s.tid]))
            )

    @stage
    @cute.jit
    def tagged_exchange_final(self, p, s, bases, epoch, slot):
        if s.tid < 224:
            self.tagged_push(
                p, s, bases, epoch, rt.cdiv(s.lane, 8), s.warp, rt.cmod(s.lane, 8), slot
            )
        rt.bar_sync(*BAR_ALL)
        query = rt.cdiv(s.warp, 2)
        rt.bar_sync(*BAR_ALL)
        if (rt.cmod(s.warp, 2) == 0) & (s.lane < 12) & (query < 4):
            self.tagged_reduce(p, s, query)
        rt.bar_sync(*BAR_COMPUTE)
        if s.tid < 48:
            self.final_residual(p, s, 4, slot)

    @stage
    @cute.jit
    def store_key(self, p, s, local_key, index, position, rows: cutlass.Constexpr):
        row = self.paged_kv_row(p, index)
        s.indices[local_key] = row
        appended = Int32((index >= position) & (index < position + rows))
        group_hit = rt.Sync.vote_any(appended != 0, Uint32(15) << Uint32(s.lane & 28))
        if rt.cmod(s.lane, 4) == 0:
            s.group_flags[rt.cdiv(local_key, 4)] = group_hit
        return appended

    @stage
    @cute.jit
    def probe_ids(
        self,
        p,
        s,
        query_,
        split,
        context,
        score_count,
        generation,
        early,
        memory: cutlass.Constexpr,
    ):
        local_key = s.tid - 256
        index = Int32(NO_KEY)
        fresh = Int32(1)
        record_index = Int32(0)
        if s.tid < 320:
            if score_count > split * SPLIT_KEYS + local_key:
                index = split * SPLIT_KEYS + local_key
                if context > TOPK:
                    record = (rt.cmod(generation, 2) * 4 + Uint32(query_)) * TOPK
                    record_index = Int32(
                        record + Uint32(split * SPLIT_KEYS) + Uint32(local_key)
                    )
                    if const_expr(early is not None):
                        word = early[1]
                        if rt.cmod(generation, 2) == 0:
                            word = early[0]
                    else:
                        word = rt.Memory.ld_volatile_u64(
                            rt.ga(p.SelectionMailbox, record_index * 2), memory=memory
                        )
                    index = Int32(word)
                    fresh = Int32(Uint32(word >> 32) == generation)
            probe = rt.Sync.vote_all(fresh != 0)
            if s.lane == 0:
                s.id_probe[s.warp - 8] = Int32(probe != 0)
        return index, fresh, record_index

    @stage
    @cute.jit
    def install_ids(
        self,
        p,
        s,
        index,
        fresh,
        record_index,
        generation,
        position,
        memory: cutlass.Constexpr,
    ):
        if s.tid < 320:
            if fresh == 0:
                record = rt.ga(p.SelectionMailbox, record_index * 2)
                index = Int32(rt.Poll.word(record, generation, False, memory))
            appended = self.store_key(p, s, s.tid - 256, index, position, 4)
            warp_hit = rt.Sync.vote_any(appended != 0)
            if s.lane == 0:
                s.warp_hits[s.warp - 8] = Int32(warp_hit != 0)

    @stage
    @cute.jit
    def arm_kv_ring(self, s, first, last, active):
        with rt.warp_leader(s.warp, 8):
            rt.mbar_wait(
                s.ring_empty + rt.cmod(first, 4) * 8, s.pages.parity(first) ^ 1
            )
            rt.mbar_arrive(s.ring_full + rt.cmod(first, 4) * 8)
            rt.mbar_wait(s.ring_empty + rt.cmod(last, 4) * 8, s.pages.parity(last) ^ 1)
            if active != 0:
                rt.mbar_expect_tx(s.ring_full + rt.cmod(last, 4) * 8, 32768)
            else:
                rt.mbar_arrive(s.ring_full + rt.cmod(last, 4) * 8)

    @stage
    @cute.jit
    def kv_gather(
        self, p, s, flag: cutlass.Constexpr, first, last, hint: cutlass.Constexpr
    ):
        if rt.elect_sync() != 0:
            for part in cutlass.range(0, 4, 1, unroll=1):
                group = s.warp - 8 + part * 4
                if s.group_flags[group] == flag:
                    if const_expr(flag == 1):
                        rt.Memory.fence_proxy_async_global()
                    rows = [s.indices[group * 4 + i] for i in range(4)]
                    for plane in cutlass.range(0, 4, 1, unroll=1):
                        slot = rt.cmod(first + rt.cdiv(plane, 4), 4)
                        dst = (
                            s.ring_addr
                            + slot * Ring.SLOT_BYTES
                            + rt.cmod(plane, 4) * Ring.KV_PLANE_BYTES
                            + group * 512
                        )
                        rt.AsyncCopy.gather4(
                            dst,
                            p.KvCache,
                            plane * 128,
                            rows[0],
                            rows[1],
                            rows[2],
                            rows[3],
                            s.ring_full + rt.cmod(last, 4) * 8,
                            hint,
                        )

    @stage
    @cute.jit
    def load_residual(self, p, s, slot, rows: cutlass.Constexpr):
        if s.tid < 256 + 6 * rows:
            lane_0 = s.tid - 256
            rt.AsyncCopy.cp_async_cg16(
                s.residual_addr + lane_0 * 16,
                rt.ga(
                    p.Hidden,
                    rt.cdiv(lane_0, 6) * 6144 + slot * 48 + rt.cmod(lane_0, 6) * 8,
                ),
            )
        cute.arch.cp_async_commit_group()

    @stage
    @cute.jit
    def prefetch_context(self, p, s, rows, slot):
        if (s.tid >= 320) & (rows <= 16384):
            slice_rows = rt.cdiv(rows + 128 - 1, 128)
            end = rt.FastMath.min_typed(rows, (slot + 1) * slice_rows)
            for row in cutlass.range(slot * slice_rows + s.tid - 320, end, 64):
                for plane in cutlass.range_constexpr(4):
                    rt.AsyncCopy.prefetch_tile_2d(
                        p.KvCache, plane * 128, self.paged_kv_row(p, row)
                    )

    @stage
    @cute.jit
    def drain_ring(self, s, index_sequence, middle):
        for d in cutlass.range_constexpr(1, 3):
            rt.mbar_wait(
                s.ring_empty + rt.cmod(index_sequence + d, 4) * 8,
                s.pages.parity(index_sequence + d),
            )
        if index_sequence > 0:
            rt.mbar_wait(
                s.ring_empty + rt.cmod(index_sequence - 1, 4) * 8,
                s.pages.parity(index_sequence - 1),
            )
        if middle != 0:
            rt.mbar_wait(
                s.ring_empty + rt.cmod(index_sequence, 4) * 8,
                s.pages.parity(index_sequence),
            )
        if (middle == 0) & (index_sequence > 3):
            rt.mbar_wait(
                s.ring_empty + rt.cmod(index_sequence - 4, 4) * 8,
                s.pages.parity(index_sequence - 4),
            )

    @stage
    @cute.jit
    def km_split_attention(
        self, p, s, first, last, split, split_count, score_count, query_0, pe_extra=None
    ):
        head_lo = rt.cdiv(s.lane, 4)
        quad = rt.cmod(s.lane, 4)
        warp_ = s.warp
        group = rt.cdiv(s.lane, 4)
        quad_0 = rt.cmod(s.lane, 4)
        row_base = warp_ * 16
        km_a = rt.rmem_tensor(16, Uint32)
        km_b = rt.rmem_tensor(8, Uint32)
        km_s = rt.rmem_tensor(4, Float32)
        km_t = rt.rmem_tensor(4, Float32)
        if warp_ < 4:
            for _f in cutlass.range_constexpr(4):
                row_q = row_base + rt.cmod(s.lane, 16)
                col_q = _f * 16 + rt.cdiv(s.lane, 16) * 8
                pe_a = s.ring_addr - 16384 + 182272
                if const_expr(pe_extra is not None):
                    pe_a = pe_a + pe_extra
                km_a[_f * 4], km_a[_f * 4 + 1], km_a[_f * 4 + 2], km_a[_f * 4 + 3] = (
                    rt.TensorCore.ldsm_x4(
                        pe_a
                        + Uint32(row_q * 128)
                        + Uint32((col_q * 2) ^ ((row_q & 7) << 4))
                    )
                )
                row_k = rt.cmod(s.lane, 8)
                col_k = _f * 16 + rt.cmod(rt.cdiv(s.lane, 8), 2) * 8
                km_b[_f * 2], km_b[_f * 2 + 1] = rt.TensorCore.ldsm_x2(
                    s.ring_addr
                    - 16384
                    + 163840
                    + 16384
                    + Uint32(row_k * 128)
                    + Uint32((col_k * 2) ^ ((row_k & 7) << 4))
                )
            a_lo = rt.Memory.take(km_a, 0, 4)
            rt.Memory.put(
                km_s, 0, rt.TensorCore.mma(None, a_lo, (km_b[0], km_b[1]), True)
            )
            a_hi = rt.Memory.take(km_a, 4, 4)
            rt.Memory.put(
                km_t, 0, rt.TensorCore.mma(None, a_hi, (km_b[2], km_b[3]), True)
            )
            rt.TensorCore.mma_acc_at(
                km_s, 0, (km_a[8], km_a[9], km_a[10], km_a[11]), (km_b[4], km_b[5])
            )
            rt.TensorCore.mma_acc_at(
                km_t, 0, (km_a[12], km_a[13], km_a[14], km_a[15]), (km_b[6], km_b[7])
            )
        rt.mbar_wait(s.ring_full + rt.cmod(first, 4) * 8, rt.cmod(rt.cdiv(first, 4), 2))
        rt.mbar_wait(s.ring_full + rt.cmod(last, 4) * 8, rt.cmod(rt.cdiv(last, 4), 2))
        cute.arch.fence_view_async_shared()
        rt.bar_sync(*BAR_COMPUTE)
        if warp_ < 4:
            for _blk in cutlass.range_constexpr(4):
                for _pr in cutlass.range_constexpr(4):
                    row_w = (
                        row_base
                        + rt.cmod(rt.cdiv(s.lane, 8), 2) * 8
                        + rt.cmod(s.lane, 8)
                    )
                    unit = _pr * 2 + rt.cdiv(s.lane, 16)
                    km_w = rt.TensorCore.ldsm_x4(
                        s.ring_addr
                        + Uint32(rt.cmod(first, 4) * Ring.SLOT_BYTES)
                        + Uint32(_blk * Ring.KV_PLANE_BYTES)
                        + Uint32(row_w * 128)
                        + Uint32((unit * 16) ^ ((row_w & 7) << 4))
                    )
                    for _o in cutlass.range_constexpr(2):
                        kb = _blk * 8 + _pr * 2 + _o
                        row_x = rt.cmod(s.lane, 8)
                        col_x = rt.cmod(kb, 4) * 16 + rt.cmod(rt.cdiv(s.lane, 8), 2) * 8
                        km_b[2 * _o], km_b[2 * _o + 1] = rt.TensorCore.ldsm_x2(
                            s.ring_addr
                            - 16384
                            + 163840
                            + Uint32(rt.cdiv(kb, 4) * 2048)
                            + Uint32(row_x * 128)
                            + Uint32((col_x * 2) ^ ((row_x & 7) << 4))
                        )
                        _a = rt.Convert.e4m3_pair_to_a(km_w[2 * _o], km_w[2 * _o + 1])
                        for _e in cutlass.range_constexpr(4):
                            km_a[4 * _o + _e] = _a[_e]
                        if const_expr(_o == 0):
                            rt.TensorCore.mma_acc_at(
                                km_s,
                                0,
                                (km_a[0], km_a[1], km_a[2], km_a[3]),
                                (km_b[0], km_b[1]),
                            )
                        else:
                            rt.TensorCore.mma_acc_at(
                                km_t,
                                0,
                                (km_a[4], km_a[5], km_a[6], km_a[7]),
                                (km_b[2], km_b[3]),
                            )
            for _e in cutlass.range_constexpr(4):
                km_s[_e] = rt.add(km_s[_e], km_t[_e])
        if warp_ < 4:
            if split == split_count - 1:
                for _e in cutlass.range_constexpr(4):
                    key = row_base + group + 8 * rt.cdiv(_e, 2)
                    if score_count <= split * 64 + key:
                        km_s[_e] = Float32(-3.4028234663852886e38)
            km_lo = rt.max_noftz(km_s[0], km_s[2])
            km_hi = rt.max_noftz(km_s[1], km_s[3])
            for _x in cutlass.range_constexpr(3):
                km_lo = rt.max_noftz(km_lo, cute.arch.shuffle_sync_bfly(km_lo, 4 << _x))
                km_hi = rt.max_noftz(km_hi, cute.arch.shuffle_sync_bfly(km_hi, 4 << _x))
            if group == 0:
                rt.Memory.st_shared_v2_f32(
                    s.maxes_addr + Uint32((warp_ * 8 + 2 * quad_0) * 4), km_lo, km_hi
                )
        rt.bar_sync(*BAR_COMPUTE)
        if warp_ < 4:
            km_max = [Float32(-3.4028234663852886e38), Float32(-3.4028234663852886e38)]
            for _pt in cutlass.range_constexpr(4):
                mx0 = s.maxes[(_pt * 8 + 2 * quad_0) + 0]
                mx1 = s.maxes[(_pt * 8 + 2 * quad_0) + 1]
                km_max[0] = rt.max_noftz(km_max[0], mx0)
                km_max[1] = rt.max_noftz(km_max[1], mx1)
            km_e = [None] * 4
            for _e in cutlass.range_constexpr(4):
                km_e[_e] = cute.math.exp2(
                    rt.mul(
                        rt.FastMath.sub(km_s[_e], km_max[rt.cmod(_e, 2)]),
                        Float32(0.09016843885183334),
                    ),
                    fastmath=False,
                )
            km_sum = [None] * 4
            for _e in cutlass.range_constexpr(4):
                km_sum[_e] = km_e[_e]
            for _rr in cutlass.range_constexpr(3):
                for _e in cutlass.range_constexpr(4):
                    km_sum[_e] = rt.add(
                        km_sum[_e], cute.arch.shuffle_sync_bfly(km_sum[_e], 4 << _rr)
                    )
            if group == 0:
                rt.Memory.st_shared_v2_f32(
                    s.pairs_addr + Uint32((2 * warp_ * 16 + 2 * quad_0) * 4),
                    km_sum[0],
                    km_sum[1],
                )
                rt.Memory.st_shared_v2_f32(
                    s.pairs_addr + Uint32(((2 * warp_ + 1) * 16 + 2 * quad_0) * 4),
                    km_sum[2],
                    km_sum[3],
                )
            odd = rt.cmod(group, 2)
            km_send_0 = km_e[0]
            km_send_1 = km_e[2]
            if odd == 0:
                km_send_0 = km_e[1]
                km_send_1 = km_e[3]
            km_recv0 = cute.arch.shuffle_sync_bfly(km_send_0, 4)
            km_recv1 = cute.arch.shuffle_sync_bfly(km_send_1, 4)
            km_pair_0 = km_e[0]
            km_pair_1 = km_recv0
            km_pair_2 = km_e[2]
            km_pair_3 = km_recv1
            if odd != 0:
                km_pair_0 = km_recv0
                km_pair_1 = km_e[1]
                km_pair_2 = km_recv1
                km_pair_3 = km_e[3]
            km_p0 = rt.f32x2_to_bf16x2(km_pair_0, km_pair_1)
            km_p1 = rt.f32x2_to_bf16x2(km_pair_2, km_pair_3)
            km_head = 2 * quad_0 + odd
            km_key = row_base + group - odd
            s.act[self.p_word(km_head, km_key)] = km_p0
            s.act[self.p_word(km_head, km_key + 8)] = km_p1
        if warp_ >= 4:
            p_zero = s.tid - 128
            rt.Memory.st_shared_v2(
                s.act_addr + Uint32(27648 + p_zero * 8), Uint32(0), Uint32(0)
            )
        rt.bar_sync(*BAR_COMPUTE)
        km_max_parts = [None] * 8
        km_sum_parts = [None] * 16
        for _h in cutlass.range_constexpr(2):
            for _pt in cutlass.range_constexpr(4):
                if const_expr(_h == 0):
                    km_max_parts[_h * 4 + _pt] = s.maxes[_pt * 8 + group]
                else:
                    km_max_parts[_h * 4 + _pt] = Float32(-3.4028234663852886e38)
            for _oc in cutlass.range_constexpr(8):
                if const_expr(_h == 0):
                    km_sum_parts[_h * 8 + _oc] = s.pairs[_oc * 16 + group]
                else:
                    km_sum_parts[_h * 8 + _oc] = Float32(0.0)
        out_t = [None] * 16
        for _kf in cutlass.range_constexpr(4):
            pt_row = rt.cmod(s.lane, 8)
            pt_col = _kf * 16 + rt.cmod(rt.cdiv(s.lane, 8), 2) * 8
            pt_b = rt.TensorCore.ldsm_x2(
                s.ring_addr
                - 16384
                + 190464
                + Uint32(pt_row * 128)
                + Uint32((pt_col * 2) ^ ((pt_row & 7) << 4))
            )
            for _t in cutlass.range_constexpr(4):
                row_t = _kf * 16 + rt.cmod(s.lane, 16)
                pt_w = rt.TensorCore.ldsm_t_b8(
                    s.ring_addr
                    + Uint32(rt.cmod(first, 4) * Ring.SLOT_BYTES)
                    + Uint32(_t * Ring.KV_PLANE_BYTES)
                    + Uint32(row_t * 128)
                    + Uint32((s.warp * 16) ^ ((row_t & 7) << 4))
                )
                pt_a = rt.Convert.e4m3_pair_to_a(pt_w[0], pt_w[1])
                if const_expr(_kf == 0):
                    rt.Memory.put(
                        out_t,
                        _t * 4,
                        rt.TensorCore.mma(None, pt_a, (pt_b[0], pt_b[1]), True),
                    )
                else:
                    rt.Memory.put(
                        out_t,
                        _t * 4,
                        rt.TensorCore.mma(
                            (
                                out_t[_t * 4],
                                out_t[_t * 4 + 1],
                                out_t[_t * 4 + 2],
                                out_t[_t * 4 + 3],
                            ),
                            pt_a,
                            (pt_b[0], pt_b[1]),
                            False,
                        ),
                    )
        head_max = [None] * 2
        head_sum = [None] * 2
        for _h in cutlass.range_constexpr(2):
            head_max[_h] = Float32(-3.4028234663852886e38)
            head_sum[_h] = Float32(0.0)
            for _pt in cutlass.range_constexpr(4):
                head_max[_h] = rt.max_noftz(head_max[_h], km_max_parts[_h * 4 + _pt])
            for _oc in cutlass.range_constexpr(8):
                head_sum[_h] = rt.add(head_sum[_h], km_sum_parts[_h * 8 + _oc])
        head_inverse, head_lse = self.lse_tail(head_sum, head_max)
        partial_base = (query_0 * 32 + split) * 8 * 512
        pt_inv0 = cute.arch.shuffle_sync(head_inverse[0], 8 * quad)
        pt_inv1 = cute.arch.shuffle_sync(head_inverse[0], 8 * quad + 4)
        for _t in cutlass.range_constexpr(4):
            for _rw in cutlass.range_constexpr(2):
                pt_dim = _t * 128 + s.warp * 16 + _rw * 8 + head_lo
                p.PartialO[partial_base + 2 * quad * 512 + pt_dim] = rt.mul(
                    out_t[_t * 4 + 2 * _rw], pt_inv0
                )
                p.PartialO[partial_base + (2 * quad + 1) * 512 + pt_dim] = rt.mul(
                    out_t[_t * 4 + (2 * _rw + 1)], pt_inv1
                )
        if (s.warp == 0) & (quad == 0):
            for _h1 in cutlass.range_constexpr(2):
                head_l = head_lo + 8 * _h1
                if head_l < 8:
                    p.PartialLSE[(query_0 * 32 + split) * 12 + head_l] = head_lse[_h1]

    @stage
    @cute.jit
    def km_merge(self, p, s, cur_pos_c, slot, arrive=None):
        rt.bar_sync(*BAR_ALL)
        query_ = rt.cdiv(slot, 32)
        head = rt.cdiv(rt.cmod(slot, 32), 4)
        split_count = rt.cdiv(cutlass.min(cur_pos_c + query_ + 1, 2048) + 64 - 1, 64)
        dims = rt.cmod(s.tid, 128) * 4
        split_base = rt.cdiv(s.tid, 128) * 16
        parts = [None] * 16
        for si in cutlass.range_constexpr(16):
            split = cutlass.min(split_base + si, split_count - 1)
            parts[si] = rt.Memory.ld_global_v4_f32(
                rt.ga(p.PartialO, ((query_ * 32 + split) * 8 + head) * 512 + dims)
            )
        if s.warp == 0:
            self.merge_weights(p, s, split_count, query_ * 384 + head, 12)
        rt.bar_sync(*BAR_COMPUTE)
        merged = rt.rmem_tensor(4, Float32)
        self.merge_sum(s, merged, parts, split_base)
        cute.arch.fence_view_async_shared()
        if const_expr(arrive is None):
            rt.mbar_arrive(s.merge_loaded)
        else:
            if arrive != 0:
                rt.mbar_arrive(s.merge_loaded)
        self.fold_halves(s, merged)
        if s.tid < 128:
            word = rt.cdiv(query_ * 1040 + 2 * dims, 4)
            s.act[word] = rt.f32x2_to_bf16x2(merged[0], merged[1])
            s.act[word + 1] = rt.f32x2_to_bf16x2(merged[2], merged[3])

    @stage
    @cute.jit
    def vo_token_partials(self, s, slot, o_vo: cutlass.Constexpr):
        pair_accum = [Float32(0.0)] * 16
        vo_tokens = rt.cmod(rt.cdiv(s.lane, 4), 4) * 260 + rt.cmod(s.lane, 4) * 8
        vo_lane = rt.cdiv(s.lane, 4) * 64 + s.warp * 2048 + rt.cmod(s.lane, 4) * 16
        rt.mbar_wait(s.tail_full, 0)
        cute.arch.fence_view_async_shared()
        vo_acc = o_vo(0, vo_lane, vo_tokens + rt.cdiv(s.warp * 64, 2))
        for m in cutlass.range_constexpr(4):
            vo_scale = s.ring_f32[
                rt.cdiv(rt.cdiv(m, 2) * 116224 + 16384, 4) + rt.cdiv(s.warp, 2)
            ]
            for item in cutlass.range_constexpr(4):
                pair_accum[m * 4 + item] = rt.fma(
                    vo_acc[m * 4 + item], vo_scale, pair_accum[m * 4 + item]
                )
        token_group = rt.cdiv(s.lane, 4)
        if rt.cmod(s.lane, 4) == rt.cdiv(rt.cdiv(slot, 32), 2):
            for m in cutlass.range_constexpr(4):
                token_low = pair_accum[m * 4]
                token_high = pair_accum[m * 4 + 2]
                if rt.cmod(rt.cdiv(slot, 32), 2) != 0:
                    token_low = pair_accum[m * 4 + 1]
                    token_high = pair_accum[m * 4 + 3]
                s.scratch[s.warp * 65 + m * 16 + token_group] = token_low
                s.scratch[s.warp * 65 + m * 16 + token_group + 8] = token_high
        rt.bar_sync(*BAR_COMPUTE)

    @stage
    @cute.jit
    def o_proj_pages(
        self,
        s,
        o_pending,
        o_token,
        tag_high,
        o_load: cutlass.Constexpr,
        o_reload: cutlass.Constexpr,
        page_mma: cutlass.Constexpr,
        pages: cutlass.Constexpr,
        act_row: cutlass.Constexpr,
        unroll: cutlass.Constexpr,
        after_page: cutlass.Constexpr = None,
    ):
        o_load(0)
        accum = rt.rmem_tensor(12, Float32)
        for item in cutlass.range_constexpr(12):
            accum[item] = Float32(0.0)
        for page in cutlass.range(0, pages, 1, unroll=unroll):
            rt.mbar_wait(s.tail_full + (1 + page) * 8, 0)
            done = Int32(0)
            while done == 0:
                stale = Uint32(0)
                for i in cutlass.range_constexpr(8):
                    stale = stale | ((o_pending[i] & Uint32(TAG_MASK)) ^ tag_high)
                if rt.Sync.vote_all(stale == 0) != 0:
                    done = Int32(1)
                if done == 0:
                    o_reload(page)
            rt.syncwarp()
            packed = [
                rt.Convert.prmt(o_pending[2 * k], o_pending[2 * k + 1], 0x5410)
                for k in range(4)
            ]
            column = page * 512 + s.warp * 64 + rt.cmod(s.lane, 8) * 8
            rt.Memory.sts_v4(
                Uint32(s.act_addr) + Uint32(o_token * act_row + 2 * column), packed
            )
            rt.syncwarp()
            if page + 1 < pages:
                o_load(page + 1)
            cute.arch.fence_view_async_shared()
            page_mma(page, accum)
            if const_expr(after_page is not None):
                after_page(page)
        return accum

    @stage
    def pe_stage(self, p, s, score_count, split, hint=L2_HINT, pe_extra=None):
        lane_0 = s.tid - 256
        pe_offset = (
            rt.cdiv(lane_0, 8) * 128
            + (rt.cmod(lane_0, 8) ^ rt.cmod(rt.cdiv(lane_0, 8), 8)) * 16
        )
        dst = s.act_addr + 18432 + pe_offset
        if const_expr(pe_extra is not None):
            dst = dst + pe_extra
        chunk = rt.cmod(lane_0, 8)
        even = Int32((chunk & 1) == 0)
        for part in range(4):
            local_key = rt.cdiv(lane_0, 8) + part * 16
            src = rt.baddr(
                p.PECache,
                Int64(s.indices[local_key]) * KvRows.PITCH
                + KvRows.ROPE_OFFSET
                + Int64(chunk * 8),
            )
            size = cutlass.select_(
                score_count > split * 64 + local_key, Int32(16), Int32(0)
            )
            rt.AsyncCopy.cp_async_cg16_zfill_if(
                even, dst + part * 2048, src, size, hint
            )

    def pe_widen(self, s, pe_extra=None):
        lane_0 = s.tid - 256
        row, chunk = rt.cdiv(lane_0, 8), rt.cmod(lane_0, 8)
        base = s.act_addr + 18432 + row * 128
        if const_expr(pe_extra is not None):
            base = base + pe_extra
        own = base + (chunk ^ rt.cmod(row, 8)) * 16
        partner = base + ((chunk + 1) ^ rt.cmod(row, 8)) * 16
        rt.Convert.widen_e4m3_pairs_shared(Int32((chunk & 1) == 0), own, partner)

    @stage
    def q_a_page_mma(self, s, accum, region, page):
        page_acc = self.fp8_page_mma(
            s, rt.cdiv(region, 8), s.act_addr, page * 1024 + s.warp * 128, 1028, True
        )
        scale_base = rt.cdiv(region, 4) + 8192
        for m in range(2):
            scale = s.ring_f32[scale_base + m * 8 + s.warp]
            for item in range(4):
                accum[m * 4 + item] = rt.fma(
                    page_acc[m * 4 + item], scale, accum[m * 4 + item]
                )

    @stage
    def vo_tile(self, s, vo_region, vo_lane, row_bases, vo_k):
        return self.vo_tile_wide(s, vo_region, vo_lane, row_bases, (vo_k,))[0]

    def vo_tile_wide(self, s, vo_region, vo_lane, row_bases, vo_ks):
        groups = len(row_bases)
        vo_rows = []
        for base in row_bases:
            for half in range(2):
                w = rt.Memory.lds_b128(
                    s.ring_addr
                    + rt.cdiv(vo_region + base + vo_lane + 512 * half, 4) * 4
                )
                vo_rows.append(Uint64(w[0]) | Uint64(w[1]) << 32)
                vo_rows.append(Uint64(w[2]) | Uint64(w[3]) << 32)
        vo_acc = [[None] * (4 * groups) for _ in vo_ks]
        vo_a = [None] * (8 * groups)
        for pj in range(2):
            for g in range(groups):
                ua = rt.Convert.unpack_u16x4(vo_rows[4 * g + pj])
                ub = rt.Convert.unpack_u16x4(vo_rows[4 * g + pj + 2])
                for i in range(4):
                    vo_a[8 * g + 2 * i] = rt.Convert.e4m3x2_to_bf16x2(ua[i])
                    vo_a[8 * g + 2 * i + 1] = rt.Convert.e4m3x2_to_bf16x2(ub[i])
            for h in range(2):
                for t, vo_k in enumerate(vo_ks):
                    vo_b = rt.Memory.lds_weak_b64(
                        s.act_addr + (vo_k + (4 * pj + 2 * h)) * 4
                    )
                    for g in range(groups):
                        a = vo_a[8 * g + 4 * h : 8 * g + 4 * h + 4]
                        if pj == 0 and h == 0:
                            vo_acc[t][4 * g : 4 * g + 4] = rt.TensorCore.mma(
                                None, a, vo_b, True
                            )
                        else:
                            rt.TensorCore.mma_acc_at(vo_acc[t], 4 * g, a, vo_b)
        return vo_acc

    def router_weight_addr(self, p) -> Int64:
        return Int64(
            Uint64(p.Flags[FlagWords.ROUTER_WEIGHT])
            | (Uint64(p.Flags[FlagWords.ROUTER_WEIGHT + 1]) << 32)
        )

    def router_enabled(self, p) -> Int32:
        return Int32(
            (p.Flags[FlagWords.ROUTER_WEIGHT] | p.Flags[FlagWords.ROUTER_WEIGHT + 1])
            != 0
        )

    @cute.jit
    def paged_kv_row(self, p, index) -> Int32:
        row = Int32(NO_ROW)
        if index < NO_KEY:
            row = p.BlockTable[
                index >> KvRows.BLOCK_SHIFT
            ] * KvRows.BLOCK_ROWS + rt.cmod(index, KvRows.BLOCK_ROWS)
        return row

    @cute.jit
    def poll_head_flags(self, p, lane_0, epoch):
        owned = Int32(lane_0 < 6)
        index_0 = 80 + rt.FastMath.min_typed(lane_0, 1)
        if lane_0 >= 2:
            index_0 = 276 + rt.FastMath.min_typed(lane_0 - 2, 3)
        self.warp_poll_ge(rt.ga(p.Flags, index_0), owned, epoch + 1)
        rt.Memory.fence_acquire_gpu()

    @cute.jit
    def ring_hit(self, p, rid, router: cutlass.Constexpr) -> Int32:
        hit = Int32(0)
        if const_expr(router):
            router_on_0_0 = Int32(0)
            if (rid >= 132) & (rid < 140):
                router_on_0_0 = self.router_enabled(p)
            hit = Int32((rid < 82) | (router_on_0_0 != 0))
        else:
            hit = Int32((rid < 82) | ((rid >= 132) & (rid < 140)))
        return hit

    @cute.jit
    def last_weight_sequence(self, p, rid, router: cutlass.Constexpr) -> Int32:
        hit = self.ring_hit(p, rid, router)
        weights_only = (rid < 128) | (rid >= 132)
        return cutlass.select_(
            weights_only, 6 * hit + 2 - 1, cutlass.select_(rid < 82, Int32(6), Int32(0))
        )

    @stage
    @cute.jit
    def issue_input_page(self, p, s, page: cutlass.Constexpr, rid):
        assert page > 0, "the scale words ride with page 0, which this never issues"
        input_slot = rt.cmod(page, 4)
        with rt.warp_leader(s.warp, 8):
            s.pages.acquire(page, 32768, input_slot)
            rt.AsyncCopy.tma_3d(
                s.ring_addr + input_slot * Ring.SLOT_BYTES,
                p.InputTmap,
                0,
                rid * 32,
                page * 8,
                s.ring_full + input_slot * 8,
                L2_HINT,
            )

    @stage
    @cute.jit
    def issue_ring_page(
        self, p, s, seq, wt, byte_off, policy, router: cutlass.Constexpr
    ):
        slot = rt.cmod(seq, 4)
        with rt.warp_leader(s.warp, 8):
            s.pages.acquire(seq, Ring.WEIGHT_PAGE_BYTES, slot)
            if const_expr(router):
                rt.bulk_g2s_hint(
                    s.ring_addr + slot * Ring.SLOT_BYTES,
                    self.router_weight_addr(p) + Int64(byte_off),
                    Ring.WEIGHT_PAGE_BYTES,
                    s.ring_full + slot * 8,
                    policy,
                )
            else:
                rt.bulk_g2s_hint(
                    s.ring_addr + slot * Ring.SLOT_BYTES,
                    rt.baddr(wt, byte_off),
                    Ring.WEIGHT_PAGE_BYTES,
                    s.ring_full + slot * 8,
                    policy,
                )

    @stage
    @cute.jit
    def issue_query_pages(
        self,
        p,
        s,
        p0: cutlass.Constexpr,
        p1: cutlass.Constexpr,
        router: cutlass.Constexpr,
        rid,
    ):
        if const_expr(router):
            policy = rt.AsyncCopy.createpolicy_evict_first(
                self.router_weight_addr(p), 1579008, 1579008
            )
        else:
            policy = rt.AsyncCopy.createpolicy_evict_first(
                p.QueryWeight.iterator.toint(), 1579008, 1579008
            )
        for page in cutlass.range_constexpr(p0, p1):
            self.issue_ring_page(
                p,
                s,
                page,
                p.QueryWeight,
                ((rid - 132) * 6 + page) * Ring.WEIGHT_PAGE_BYTES,
                policy,
                router,
            )

    @stage
    @cute.jit
    def issue_o_page(self, p, s, page: cutlass.Constexpr, rid, policy):
        mbar = s.tail_full + (1 + page) * 8
        if const_expr(page == 0):
            rt.mbar_expect_tx(mbar, 24768)
            rt.bulk_g2s_hint(
                s.ring_addr + 16640,
                rt.baddr(p.OScales, rt.cdiv(rid * 48, 128) * 16 * 4),
                64,
                mbar,
                policy,
            )
            for _rr in cutlass.range_constexpr(2):
                rt.bulk_g2s_hint(
                    s.ring_addr + (16704 + 64 * _rr),
                    rt.baddr(
                        p.OScales, rt.cdiv(rid * 48 + (16 + 16 * _rr), 128) * 16 * 4
                    ),
                    64,
                    mbar,
                    policy,
                )
        if const_expr(page != 0):
            rt.mbar_expect_tx(mbar, 24576)
        rt.AsyncCopy.tma_3d(
            s.ring_addr + (17408 + page * 24704),
            p.OTmap,
            0,
            rid * 48,
            page * 8,
            mbar,
            L2_HINT,
        )

    @stage
    def fp8_page_mma(
        self, s, slot_words, a_addr, k0, token_stride: int, warp_major: bool
    ):
        def ld(step):
            step = step if step < 8 else 7
            row = s.warp * 8 + step if warp_major else 8 * step + s.warp
            row_hi = 64 + s.warp * 8 + step if warp_major else (64 + 8 * step) + s.warp
            return [
                s.ring[slot_words + row * 32 + s.lane],
                s.ring[slot_words + row_hi * 32 + s.lane],
            ]

        def ldm(c):
            token = rt.cmod(s.lane, 4)
            column = k0 + c + rt.cdiv(s.lane, 8) * 8
            return rt.TensorCore.ldsm_x4(
                a_addr + (token * token_stride + rt.cdiv(column, 2)) * 4
            )

        regs = ld(0)
        pair_even = ldm(0)
        regs += ld(1)
        pair_odd = ldm(32)
        acc = [None] * 8
        for step in range(8):
            pair = pair_even if rt.cmod(rt.cdiv(step, 2), 2) == 0 else pair_odd
            b0, b1 = (pair[2 * rt.cmod(step, 2)], pair[2 * rt.cmod(step, 2) + 1])
            for t in range(2):
                u = rt.Convert.unpack_u16x4(regs[2 * step + t])
                a = [rt.Convert.e4m3x2_to_bf16x2(u[i]) for i in range(4)]
                if step == 0:
                    acc[4 * t : 4 * t + 4] = rt.TensorCore.mma(None, a, (b0, b1), True)
                else:
                    acc[4 * t : 4 * t + 4] = rt.TensorCore.mma(
                        acc[4 * t : 4 * t + 4], a, (b0, b1), False
                    )
            if step < 7:
                regs += ld(step + 2)
                if step == 1:
                    pair_even = ldm(64)
                if step == 3:
                    pair_odd = ldm(96)
        return acc

    @stage
    @cute.jit
    def x_norm(self, p, s):
        rows = self.GROUP_ROWS
        gamma = [
            rt.Memory.ld_global_v8_f32(rt.ga(p.GammaX, 2048 * j + s.tid * 8))
            for j in range(3)
        ]
        rt.mbar_wait(s.x_full, 0)
        cute.arch.fence_view_async_shared()
        words = [
            [
                s.act[rt.cdiv(c // 3 * 6152 + c % 3 * 2048 + s.tid * 8, 2) + i]
                for i in range(4)
            ]
            for c in range(3 * rows)
        ]
        values = [rt.bf16x8_to_f32(words[c]) for c in range(3 * rows)]
        late_squares = [
            self.pair_squares(values[3 * q] + values[3 * q + 1] + values[3 * q + 2])
            for q in range(rows)
        ]
        for c in cutlass.range_constexpr(3 * rows):
            scaled = [rt.mul(values[c][i], gamma[c % 3][i]) for i in range(8)]
            packed = [
                rt.f32x2_to_bf16x2(scaled[2 * i], scaled[2 * i + 1]) for i in range(4)
            ]
            rt.Memory.sts_v4(
                s.act_addr + (c // 3 * 6152 + c % 3 * 2048 + s.tid * 8) * 2, packed
            )
        self.stage_rows(s, late_squares, 2016)
        rt.bar_sync(*BAR_COMPUTE)
        row = self.min_i32(rt.cdiv(s.tid, 32), rows - 1)
        late_total = s.scratch[2016 + row * 8]
        for k in cutlass.range_constexpr(1, 8):
            late_total = rt.add(late_total, s.scratch[2016 + row * 8 + k])
        return late_total

    @stage
    @cute.jit
    def stage_rows(self, s, values, base: cutlass.Constexpr):
        for step in cutlass.range_constexpr(5):
            values = [
                rt.add(values[q], cute.arch.shuffle_sync_down(values[q], 16 >> step))
                for q in range(len(values))
            ]
        if s.lane == 0:
            for q in cutlass.range_constexpr(len(values)):
                s.scratch[base + q * 8 + s.warp] = values[q]

    @stage
    @cute.jit
    def rms_inverses(self, s, values):
        rows = len(values)
        self.stage_rows(s, [self.pair_squares(values[q]) for q in range(rows)], 0)
        rt.bar_sync(*BAR_COMPUTE)
        if s.tid < rows:
            row_total = s.scratch[s.tid * 8]
            for k in cutlass.range_constexpr(1, 8):
                row_total = rt.add(row_total, s.scratch[s.tid * 8 + k])
            s.scratch[32 + s.tid] = rt.FastMath.rsqrt(
                rt.fma(row_total, Float32(0.00048828125), Float32(1e-05))
            )
        rt.bar_sync(*BAR_COMPUTE)
        return [s.scratch[32 + q] for q in range(rows)]

    def load_x(self, s, x):
        rt.mbar_expect_tx(s.x_full, self.GROUP_ROWS * X_ROW_BYTES)
        for r in range(self.GROUP_ROWS):
            act_row = s.act_addr + (X_ROW_BYTES + 16) * r
            rt.bulk_g2s(act_row, rt.baddr(x, X_ROW_BYTES * r), X_ROW_BYTES, s.x_full)

    def is_router(self, slot):
        return (slot >= 132) & (slot < 140)

    def proj_router(self, p0, slot):
        router = Int32(self.is_router(slot))
        if const_expr(self.full_or_push):
            router = router & self.router_enabled(p0)
        return router

    def proj_router_source(self, p0, offset):
        if const_expr(self.full_or_push):
            return self.router_weight_addr(p0) + Int64(offset)
        return rt.baddr(p0.QueryWeight, offset)

    @staticmethod
    def min_i32(a, b):
        return rt.FastMath.select_i32(a < b, a, b)

    @staticmethod
    def pair_squares(values):
        squares = Float32(0.0)
        for p in range(len(values) // 2):
            squares = rt.add(
                squares,
                rt.fma(
                    values[2 * p],
                    values[2 * p],
                    rt.mul(values[2 * p + 1], values[2 * p + 1]),
                ),
            )
        return squares

    @staticmethod
    def norm_bf16x8(values, gamma, inverse):
        y = [rt.mul(values[i], rt.mul(gamma[i], inverse)) for i in range(8)]
        return [rt.f32x2_to_bf16x2(y[2 * i], y[2 * i + 1]) for i in range(4)]

    @staticmethod
    def peer_base_by_select(rank, bases) -> Int64:
        base = bases[0]
        for k in range(1, 8):
            base = cutlass.select_(rank == k, bases[k], base)
        return base

    @staticmethod
    def max8_u32(v):
        v = [Uint32(v[i]) for i in range(8)]
        m0 = rt.FastMath.max_typed(v[0], v[1])
        m1 = rt.FastMath.max_typed(v[2], v[3])
        m2 = rt.FastMath.max_typed(m0, m1)
        m3 = rt.FastMath.max_typed(v[4], v[5])
        m4 = rt.FastMath.max_typed(v[6], v[7])
        m5 = rt.FastMath.max_typed(m3, m4)
        return rt.FastMath.max_typed(m2, m5)

    @staticmethod
    @cute.jit
    def warp_poll_ge(addr, owned, target):
        done = Int32(0)
        while done == 0:
            satisfied = Int32(1)
            if owned != 0:
                observed = rt.Memory.ld_relaxed_gpu(addr)
                satisfied = Int32(observed >= target)
            done = rt.Sync.vote_all(satisfied != 0)


class TargetWorker(MlaWorker):
    """glm53_mla_worker<ROLE>: the target model's q4 worker roles (kFull .."""

    X_FULL, NORM_FREE, MERGE_LOADED, MOE_LIST_FULL = 104, 112, 120, 128

    def kernel_extras(self, head, tail, extras: list) -> tuple:
        o, kvb = (
            self.page_map(tail.OWeight, *O_PAGES),
            self.page_map(tail.KVbWeight, *KVB_PAGES),
        )
        return (*extras, self.page_map(head.InputWeight, *INPUT_PAGES), o, kvb)

    @cute.kernel
    def kernel(
        self,
        head: HeadArgs,
        KvCache: GridConstant[TensorMap],
        tail: TailArgs,
        InputTmap: GridConstant[TensorMap],
        OTmap: GridConstant[TensorMap],
        KvbTmap: GridConstant[TensorMap],
    ):
        maps = {"InputTmap": InputTmap, "OTmap": OTmap, "KvbTmap": KvbTmap}
        self.run_records(KvCache, (head, tail), maps)

    @cute.jit
    def run_records(self, KvCache, records, maps):
        if const_expr(self.PHASED):
            self.phased(KvCache, records, maps)
        else:
            assert self.GROUPS == 1, "more than four rows run phased passes"
            self.enter(WorkerParams.of(KvCache, *records, **maps))

    @cute.jit
    def phased(self, KvCache, records, maps):
        if const_expr(self.PACKED):
            assert not (
                self.push or self.pushed or (self.full_or_push and self.REFINE)
            ), "a packed build selects request-parallel: no push roles, no refine"
        assert self.WIDE and self.ATTN_LOOP, "a phased pass is the wide head and loop"
        assert not (self.ATTN_LOOP and self.REFINE and kFull == self.ROLE), (
            "the attention loop has no refine"
        )
        assert self.WIDE_TAIL in (0, 2), "the batched tail merges by quarters"
        self.set_phase("head", PassContext(self, KvCache, records, maps))
        for passed in cutlass.range(self.PASSES, unroll=1):
            self.ctx.begin(passed)
            if self.ctx.live(self.ctx.first):
                for phase in ("proj", "attn"):
                    phase_p = self.ctx.params(self.ctx.first)
                    self.set_phase(phase)
                    phase_ids, phase_s = self.prologue(phase_p)
                    self.run(phase_p, phase_s, phase_ids)
                    cute.arch.sync_threads()
                tail_p = self.ctx.params(self.ctx.first)
                self.set_phase("tail")
                tail_ids, tail_s = self.prologue(tail_p)
                self.run(tail_p, tail_s, tail_ids)
                cute.arch.sync_threads()

    @stage
    @cute.jit
    def compute_tail(self, s):
        self.set_regs("compute")
        own_o = rt.rmem_tensor(self.PASS_GROUPS, Uint32)  # word h: group h's own O sum
        slot = self.role_slot(s)
        if slot < 128:
            for h in cutlass.range(self.PASS_GROUPS, unroll=1):
                group = self.ctx.first + h
                if self.ctx.live(group):
                    p = self.ctx.params(group)
                    tag = self.group_tag(s, h)
                    self.km_merge(p, s, self.position(p), slot, Int32(h == 0))
                    self.vo_publish(p, s, slot, tag)
            for h in cutlass.range(self.PASS_GROUPS, unroll=1):
                group = self.ctx.first + h
                if self.ctx.live(group):
                    p = self.ctx.params(group)
                    self.vo_project(p, s, slot, self.group_tag(s, h))
                    if s.tid < 224:
                        self.tagged_push(
                            p,
                            s,
                            p.peer_bases,
                            self.exchange_epoch(p, s, h),
                            rt.cdiv(s.lane, 8),
                            s.warp,
                            rt.cmod(s.lane, 8),
                            slot,
                        )
                    if s.tid < 96:  # the group's own partial sums, kept for its reduce
                        own_o[h] = rt.Memory.lds(s.local_o_addr + s.tid * 4, Uint32)
                    rt.bar_sync(*BAR_COMPUTE)
            for h in cutlass.range(self.PASS_GROUPS, unroll=1):
                group = self.ctx.first + h
                if self.ctx.live(group):
                    p = self.ctx.params(group)
                    if s.tid < 96:
                        rt.Memory.sts(s.local_o_addr + s.tid * 4, own_o[h], Uint32)
                    rt.bar_sync(*BAR_ALL)
                    rt.bar_sync(*BAR_ALL)
                    query = rt.cdiv(s.warp, 2)
                    if (rt.cmod(s.warp, 2) == 0) & (s.lane < 12) & (query < 4):
                        self.tagged_reduce(p, s, query)
                    rt.bar_sync(*BAR_COMPUTE)
                    if s.tid < 48:
                        self.final_residual(p, s, 4, slot)
                    rt.bar_sync(*BAR_ALL)

    @stage
    @cute.jit
    def load_tail(self, s):
        p0 = self.ctx.params(self.ctx.first)
        self.set_regs("load")
        rid = self.role_slot(s)
        lane_0 = s.tid - 256
        if rid < 128:
            o_policy = self.tail_weights(p0, s, rid)
            for h in cutlass.range(self.PASS_GROUPS, unroll=1):
                group = self.ctx.first + h
                if self.ctx.live(group):
                    p = self.ctx.params(group)
                    poll_addr = self.split_done(
                        p,
                        s,
                        rt.cdiv(rid, 32) * 32 + rt.FastMath.min_typed(lane_0, 31),
                    )
                    self.warp_poll_ge(
                        poll_addr,
                        Int32(lane_0 < 32),
                        Uint32(s.epoch_slot[PassPhases.EPOCHS + h]) + 1,
                    )
                    rt.Memory.fence_acquire_gpu()
                    rt.bar_sync(*BAR_LOADER)
                    rt.bar_sync(*BAR_ALL)
            with rt.warp_leader(s.warp, 8):
                rt.mbar_wait(s.merge_loaded, 0)
                for page in cutlass.range_constexpr(2, 4):
                    self.issue_o_page(p0, s, page, rid, o_policy)
            for h in cutlass.range(self.PASS_GROUPS, unroll=1):
                group = self.ctx.first + h
                if self.ctx.live(group):
                    p = self.ctx.params(group)
                    self.load_residual(p, s, rid, 4)
                    cute.arch.cp_async_wait_group(0)
                    generation = self.exchange_epoch(p, s, h) + 1
                    parity = Int32(rt.cmod(generation, 2))
                    mail_tag = (generation & 127) | 128
                    base = p.peer_bases[0]
                    for owner in cutlass.range_constexpr(1, 8):
                        if p.peer_rank == owner:
                            base = p.peer_bases[owner]
                    item_0 = lane_0 + 128
                    if item_0 >= 224:
                        item_0 = lane_0
                    rt.bar_sync(*BAR_ALL)
                    rt.OExchange.receive(
                        s.received_words,
                        base,
                        parity,
                        mail_tag,
                        rid,
                        lane_0,
                        [lane_0, item_0],
                        224,
                    )
                    rt.bar_sync(*BAR_ALL)
                    rt.bar_sync(*BAR_ALL)
        self.seam_gates(s)

    @cute.jit
    def group_position(self, p0, s, h, query_=None):
        if const_expr(self.PACKED):
            return self.row_info(p0, s, h, query_)[0]
        assert self.WIDE, "the serial tail reads positions per group"
        return Int32(s.epoch_slot[ProjectionChunks.POSITION + h])

    def split_done_of(self, p0, s, slot, h):
        if const_expr(self.shared_or_seam):
            return self.handoff_word(
                p0, Handoff.SPLIT_DONE + slot + h * Handoff.WORDS, rt.cdiv(slot, 64)
            )
        return rt.ga(
            p0.Flags, FlagWords.SPLIT_DONE + slot + h * MlaWorker.GROUP_STRIDES["Flags"]
        )

    @cute.jit
    def split_count(self, p0, s, h, query_):
        position = self.group_position(p0, s, h, query_)
        return rt.cdiv(cutlass.min(position + query_ + 1, 2048) + 64 - 1, 64)

    @cute.jit
    def lse_issue(self, p0, s, rows_live, query_, head, lse):
        for i in cutlass.range_constexpr(-(-self.PASS_GROUPS // 8)):
            h = s.warp + 8 * i
            lse[i] = Float32(NEG_FLT_MAX)
            if (h < self.PASS_GROUPS) & (  # noqa: SIM102  nested bare-bool ifs: the DSL traces them
                rows_live > (self.ctx.first + h) * self.GROUP_ROWS
            ):
                if self.split_count(p0, s, h, query_) > s.lane:
                    lse[i] = p0.PartialLSE[h * 1536 + query_ * 384 + head + s.lane * 12]

    @cute.jit
    def merge_weights_wide(self, p0, s, rows_live, query_, lse):
        for i in cutlass.range_constexpr(-(-self.PASS_GROUPS // 8)):
            h = s.warp + 8 * i
            if (h < self.PASS_GROUPS) & (
                rows_live > (self.ctx.first + h) * self.GROUP_ROWS
            ):
                count = self.split_count(p0, s, h, query_)
                maximum = lse[i]
                for step in cutlass.range_constexpr(5):
                    maximum = rt.max_noftz(
                        maximum, cute.arch.shuffle_sync_bfly(maximum, 16 >> step)
                    )
                total = cute.math.exp2(rt.FastMath.sub(lse[i], maximum), fastmath=False)
                for step in cutlass.range_constexpr(5):
                    total = rt.add(
                        total, cute.arch.shuffle_sync_bfly(total, 16 >> step)
                    )
                global_lse = rt.add(maximum, rt.FastMath.lg2_approx(total))
                weight = Float32(0.0)
                if count > s.lane:
                    weight = cute.math.exp2(
                        rt.FastMath.sub(lse[i], global_lse), fastmath=False
                    )
                s.scratch[h * 32 + s.lane] = weight

    @cute.jit
    def merge_parts(self, p0, s, h, query_, head, dims, half, parts):
        count = self.split_count(p0, s, h, query_)
        for si in cutlass.range_constexpr(16):
            split = cutlass.min(half * 16 + si, count - 1)
            loaded = rt.Memory.ld_global_v4_f32(
                rt.ga(
                    p0.PartialO,
                    h * 524288 + ((query_ * 32 + split) * 8 + head) * 512 + dims,
                )
            )
            for i in cutlass.range_constexpr(4):
                parts[si * 4 + i] = loaded[i]

    @cute.jit
    def merge_sum_wide(self, s, h, parts, half):
        merged = rt.rmem_tensor(4, Float32)
        for i in cutlass.range_constexpr(4):
            merged[i] = Float32(0.0)
        for si in cutlass.range_constexpr(16):
            weight = s.scratch[h * 32 + half * 16 + si]
            for i in cutlass.range_constexpr(4):
                merged[i] = rt.fma(parts[si * 4 + i], weight, merged[i])
        for i in cutlass.range_constexpr(4):
            merged[i] = rt.add(merged[i], cute.arch.shuffle_sync_bfly(merged[i], 16))
        return merged

    @cute.jit
    def merge_publish(self, p0, s, g, query_, head, quarter, chunk, half, parts):
        merged = self.merge_sum_wide(s, g, parts, half)
        if s.lane < 16:
            tag = self.group_tag(s, g) ^ Uint32(0x4000)
            words = [
                Uint32(rt.f32x2_to_bf16x2(merged[i], Float32(0.0)) & 65535 | tag << 16)
                for i in range(4)
            ]
            at = (
                Handoff.INDEXER_Q
                + (query_ * 8 + head) * 512
                + quarter * 128
                + chunk * 4
                + g * Handoff.WORDS
            )
            rt.Memory.st_volatile_v4(self.handoff_word(p0, at), words)
            if const_expr(self.shared_or_seam):
                rt.Memory.st_volatile_v4(self.handoff_word(p0, at, 1), words)

    @cute.jit
    def merge_quarter(self, p0, s, rows_live, slot):
        query_ = rt.cdiv(slot, 32)
        head = rt.cdiv(rt.cmod(slot, 32), 4)
        quarter = rt.cmod(slot, 4)
        half = rt.cdiv(s.lane, 16)
        chunk = rt.cmod(s.lane, 16) + 16 * rt.cmod(s.warp, 2)
        dims = quarter * 128 + chunk * 4
        pair_warp = rt.cdiv(s.warp, 2)
        assert self.PASS_GROUPS == 4, "the quarter merge takes four groups (tier 16)"
        parts = rt.rmem_tensor(64, Float32)
        lse = rt.rmem_tensor(1, Float32)
        first_row = self.ctx.first * self.GROUP_ROWS
        live = rows_live > first_row + pair_warp * self.GROUP_ROWS
        self.lse_issue(p0, s, rows_live, query_, head, lse)
        if live:
            self.merge_parts(p0, s, pair_warp, query_, head, dims, half, parts)
        self.merge_weights_wide(p0, s, rows_live, query_, lse)
        rt.bar_sync(*BAR_COMPUTE)
        if live:
            self.merge_publish(
                p0, s, pair_warp, query_, head, quarter, chunk, half, parts
            )
        cute.arch.fence_view_async_shared()
        rt.mbar_arrive(s.merge_loaded)

    @cute.jit
    def gather_merged(self, p0, s, rows_live, slot):
        pairs = self.PASS_GROUPS // 2
        near = 0
        if const_expr(self.shared_or_seam):
            near = Int32(s.role_entry >> Handoff.POOL_SHIFT)
        chunk = rt.cmod(s.tid, 128)
        base = (
            Handoff.INDEXER_Q
            + (rt.cdiv(slot, 32) * 8 + rt.cdiv(rt.cmod(slot, 32), 4)) * 512
            + chunk * 4
        )
        pending = rt.rmem_tensor(4 * pairs, Uint32)
        tag_high = rt.rmem_tensor(pairs, Uint32)
        live_mask = rt.rmem_tensor(pairs, Uint32)
        for i in cutlass.range_constexpr(pairs):
            group = rt.cdiv(s.tid, 128) + 2 * i
            tag_high[i] = (self.group_tag(s, group) ^ Uint32(0x4000)) << 16
            live_mask[i] = Uint32(0)
            if rows_live > (self.ctx.first + group) * self.GROUP_ROWS:
                live_mask[i] = Uint32(0xFFFFFFFF)
        done = Int32(0)
        while done == 0:
            stale = Uint32(0)
            for i in cutlass.range_constexpr(pairs):
                group = rt.cdiv(s.tid, 128) + 2 * i
                v = rt.Memory.ld_volatile_v4(
                    self.handoff_word(p0, base + group * Handoff.WORDS, near)
                )
                bad = Uint32(0)
                for w in cutlass.range_constexpr(4):
                    pending[4 * i + w] = v[w]
                    bad = bad | ((v[w] & Uint32(TAG_MASK)) ^ tag_high[i])
                stale = stale | (bad & live_mask[i])
            if stale == 0:
                done = Int32(1)
        for i in cutlass.range_constexpr(pairs):
            word = (rt.cdiv(s.tid, 128) + 2 * i) * 260 + chunk * 2
            s.act[word] = rt.Convert.prmt(pending[4 * i], pending[4 * i + 1], 0x5410)
            s.act[word + 1] = rt.Convert.prmt(
                pending[4 * i + 2], pending[4 * i + 3], 0x5410
            )

    @cute.jit
    def vo_wide(self, p0, s, rows_live, slot):
        groups = self.PASS_GROUPS
        head = rt.cdiv(rt.cmod(slot, 32), 4)
        vo_lane = rt.cdiv(s.lane, 4) * 64 + s.warp * 2048 + rt.cmod(s.lane, 4) * 16
        part_base = 3120  # act word of the per-group, per-warp V/O partials
        rt.mbar_wait(s.tail_full, 0)
        cute.arch.fence_view_async_shared()
        for tile in cutlass.range_constexpr(-(-groups // 8)):
            vo_k = (
                rt.cdiv(s.lane, 4) * 260
                + rt.cmod(s.lane, 4) * 8
                + rt.cdiv(s.warp * 64, 2)
                + tile * 2080
            )
            vo_acc = self.vo_tiles(s, 0, vo_lane, vo_k, True)
            for m in cutlass.range_constexpr(4):
                scale = s.ring_f32[
                    rt.cdiv(rt.cdiv(m, 2) * 116224 + 16384, 4) + rt.cdiv(s.warp, 2)
                ]
                for j in cutlass.range_constexpr(2):
                    group = tile * 8 + 2 * rt.cmod(s.lane, 4) + j
                    if group < groups:
                        for half in cutlass.range_constexpr(2):
                            value = rt.fma(
                                vo_acc[m * 4 + 2 * half + j], scale, Float32(0.0)
                            )
                            rt.Memory.sts(
                                s.act_addr
                                + (
                                    part_base
                                    + group * 520
                                    + s.warp * 65
                                    + m * 16
                                    + half * 8
                                    + rt.cdiv(s.lane, 4)
                                )
                                * 4,
                                value,
                                Float32,
                            )
        rt.bar_sync(*BAR_COMPUTE)
        for i in cutlass.range_constexpr(-(-groups * 64 // 256)):
            index = s.tid + 256 * i
            group = rt.cdiv(index, 64)
            live = rows_live > (self.ctx.first + group) * self.GROUP_ROWS
            if (group < groups) & live:
                column = rt.cmod(index, 64)
                base = s.act_addr + (part_base + group * 520 + column) * 4
                token_value = rt.Memory.lds(base, Float32)
                for warp_ in cutlass.range_constexpr(1, 8):
                    token_value = rt.add(
                        token_value, rt.Memory.lds(base + warp_ * 260, Float32)
                    )
                word = (
                    rt.cdiv(slot, 32) * Handoff.VO_ROW
                    + head * 256
                    + rt.cmod(slot, 4) * 64
                    + column
                    + group * Handoff.WORDS
                )
                low = rt.f32x2_to_bf16x2(token_value, Float32(0.0)) & 65535
                tagged = Uint32(low | self.group_tag(s, group) << 16)
                rt.Memory.st_relaxed_sys(
                    self.handoff_word(p0, Handoff.VO + word), tagged
                )
                if const_expr(self.shared_or_seam):
                    rt.Memory.st_relaxed_sys(
                        self.handoff_word(p0, Handoff.VO + word, 1), tagged
                    )
        rt.bar_sync(*BAR_COMPUTE)

    @property
    def TAIL_BLOCK(self) -> int:
        assert self.PASS_GROUPS % 4 == 0, "the pass's groups come in blocks of four"
        return 4

    TILE_PITCH = 34

    @staticmethod
    def tail_keep(s):
        return s.ring_addr + 132736

    def tail_epochs(self, s):
        return self.tail_keep(s) + self.PASS_GROUPS * 768

    def rotated_group(self, block, rot, it):
        return block * self.TAIL_BLOCK + rt.cmod(it + rot, self.TAIL_BLOCK)

    @cute.jit
    def o_poll(self, p0, s, block, rot, near, page, pending):
        first = (
            Handoff.VO
            + rt.cdiv(s.lane, 8) * Handoff.VO_ROW
            + page * 512
            + s.warp * 64
            + rt.cmod(s.lane, 8) * 8
        )
        for it in cutlass.range_constexpr(self.TAIL_BLOCK):
            at = first + self.rotated_group(block, rot, it) * Handoff.WORDS
            w0 = rt.Memory.ld_volatile_v4(self.handoff_word(p0, at, near))
            w1 = rt.Memory.ld_volatile_v4(self.handoff_word(p0, at + 4, near))
            for k in cutlass.range_constexpr(4):
                pending[8 * it + k] = w0[k]
                pending[8 * it + 4 + k] = w1[k]

    @cute.jit
    def o_projection_wide(self, p0, s, rows_live, block, rot, accum):
        groups = self.TAIL_BLOCK
        rows = 4 * groups
        near = 0
        if const_expr(self.shared_or_seam):
            near = Int32(s.role_entry >> Handoff.POOL_SHIFT)
        tag_high = rt.rmem_tensor(groups, Uint32)
        live_mask = rt.rmem_tensor(groups, Uint32)
        for it in cutlass.range_constexpr(groups):
            group = self.rotated_group(block, rot, it)
            tag_high[it] = self.group_tag(s, group) << 16
            live_mask[it] = Uint32(0)
            if rows_live > (self.ctx.first + group) * self.GROUP_ROWS:
                live_mask[it] = Uint32(0xFFFFFFFF)
        pending = rt.rmem_tensor(8 * groups, Uint32)
        tile = s.act_addr + s.warp * rows * self.TILE_PITCH * 4
        for item in cutlass.range_constexpr(12 * (rows // 8)):
            accum[item] = Float32(0.0)
        vo_lane = rt.cdiv(s.lane, 4) * 64 + s.warp * 3072 + rt.cmod(s.lane, 4) * 16
        vo_ks = [
            s.warp * rows * self.TILE_PITCH
            + (8 * t + rt.cdiv(s.lane, 4)) * self.TILE_PITCH
            + rt.cmod(s.lane, 4) * 8
            for t in range(rows // 8)
        ]
        self.o_poll(p0, s, block, rot, near, 0, pending)
        for page in cutlass.range(0, 4, 1, unroll=1):
            rt.mbar_wait(s.tail_full + (1 + page) * 8, 0)
            done = Int32(0)
            while done == 0:
                stale = Uint32(0)
                for it in cutlass.range_constexpr(groups):
                    bad = Uint32(0)
                    for i in cutlass.range_constexpr(8):
                        bad = bad | (
                            (pending[8 * it + i] & Uint32(TAG_MASK)) ^ tag_high[it]
                        )
                    stale = stale | (bad & live_mask[it])
                if rt.Sync.vote_all(stale == 0) != 0:
                    done = Int32(1)
                if done == 0:
                    self.o_poll(p0, s, block, rot, near, page, pending)
            rt.syncwarp()
            for it in cutlass.range_constexpr(groups):
                at = (
                    tile
                    + ((it * 4 + rt.cdiv(s.lane, 8)) * self.TILE_PITCH) * 4
                    + rt.cmod(s.lane, 8) * 16
                )
                packed = [
                    rt.Convert.prmt(
                        pending[8 * it + 2 * k], pending[8 * it + 2 * k + 1], 0x5410
                    )
                    for k in range(4)
                ]
                rt.Memory.st_shared_v2(at, packed[0], packed[1])
                rt.Memory.st_shared_v2(at + 8, packed[2], packed[3])
            rt.syncwarp()
            if page + 1 < 4:
                self.o_poll(p0, s, block, rot, near, page + 1, pending)
            cute.arch.fence_view_async_shared()
            vo_acc = self.vo_tile_wide(
                s, 17408 + page * 24704, vo_lane, (0, 1024, 2048), vo_ks
            )
            for m in cutlass.range_constexpr(3):
                scale = s.ring_f32[4160 + m * 16 + page * 4 + rt.cdiv(s.warp, 2)]
                for t in cutlass.range_constexpr(rows // 8):
                    for item in cutlass.range_constexpr(4):
                        at = t * 12 + m * 4 + item
                        accum[at] = rt.fma(vo_acc[t][m * 4 + item], scale, accum[at])

    @cute.jit
    def own_o_wide(self, s, block, rot, accum):
        rows = 4 * self.TAIL_BLOCK
        stride = rows * 48 + 1
        rt.bar_sync(*BAR_COMPUTE)
        for t in cutlass.range_constexpr(rows // 8):
            for j in cutlass.range_constexpr(2):
                token = 8 * t + 2 * rt.cmod(s.lane, 4) + j
                for m in cutlass.range_constexpr(3):
                    for half in cutlass.range_constexpr(2):
                        rt.Memory.sts(
                            s.act_addr
                            + (
                                s.warp * stride
                                + token * 48
                                + m * 16
                                + half * 8
                                + rt.cdiv(s.lane, 4)
                            )
                            * 4,
                            accum[t * 12 + m * 4 + 2 * half + j],
                            Float32,
                        )
        rt.bar_sync(*BAR_COMPUTE)
        for i in cutlass.range_constexpr(-(-rows * 24 // 256)):
            index = s.tid + 256 * i
            if index < rows * 24:
                token, pair = rt.cdiv(index, 24), rt.cmod(index, 24)
                base = s.act_addr + (token * 48 + pair * 2) * 4
                group = self.rotated_group(block, rot, rt.cdiv(token, 4))
                sums = []
                for column in cutlass.range_constexpr(2):
                    sums.append(
                        rt.Reduce.ordered_sum(
                            lambda k: rt.Memory.lds(
                                base + (k * stride + column) * 4, Float32
                            ),
                            8,
                        )
                    )
                rt.Memory.sts(
                    self.tail_keep(s)
                    + ((group * 4 + rt.cmod(token, 4)) * 24 + pair) * 4,
                    rt.f32x2_to_bf16x2(sums[0], sums[1]),
                    Uint32,
                )
        rt.bar_sync(*BAR_COMPUTE)

    @cute.jit
    def exchange_epoch_of(self, p0, s, h):
        epoch = Uint32(s.epoch_slot[PassPhases.EPOCHS + h])
        if const_expr(self.shared_or_seam):
            epoch = (
                (epoch + 1) * 57
                + p0.Flags[
                    h * MlaWorker.GROUP_STRIDES["Flags"] + FlagWords.SHARED_INDEX
                ]
                - 1
            )
        return epoch

    @cute.jit
    def push_wide(self, p0, s, rows_live, block, slot):
        if s.tid < 224:
            token = rt.cdiv(s.lane, 8)
            chunk = rt.cmod(s.lane, 8)
            destination = rt.FastMath.select_i32(
                s.warp < p0.peer_rank, s.warp, s.warp + 1
            )
            source_slot = rt.FastMath.select_i32(
                destination > p0.peer_rank, p0.peer_rank, p0.peer_rank - 1
            )
            for in_block in cutlass.range_constexpr(self.TAIL_BLOCK):
                g = block * self.TAIL_BLOCK + in_block
                if rows_live > (self.ctx.first + g) * self.GROUP_ROWS:
                    epoch = rt.Memory.lds(self.tail_epochs(s) + g * 4, Uint32)
                    payload = [
                        rt.Memory.lds(
                            self.tail_keep(s)
                            + ((g * 4 + token) * 24 + chunk * 3 + i) * 4,
                            Uint32,
                        )
                        for i in range(3)
                    ]
                    words = rt.TaggedMailbox.pack24(
                        payload, rt.TaggedMailbox.tag8(epoch) << 24
                    )
                    first = rt.OExchange.word(
                        token, source_slot, rt.cmod(epoch + 1, 2), slot, chunk
                    )
                    bases = [b + Int64(g) * PUSH_MAILBOX_BYTES for b in p0.peer_bases]
                    rt.TaggedMailbox.push(
                        rt.TaggedMailbox.peer_base(destination, bases)
                        + Int64(first * 4),
                        words,
                    )

    @cute.jit
    def receive_wide(self, p0, s, rows_live, slot):
        if s.tid < 224:
            groups = self.PASS_GROUPS
            base = p0.peer_bases[0]
            for owner in cutlass.range_constexpr(1, 8):
                if p0.peer_rank == owner:
                    base = p0.peer_bases[owner]
            item = (
                rt.cdiv(s.tid, 56) * 57344
                + rt.cmod(rt.cdiv(s.tid, 8), 7) * 8192
                + slot * 32
                + rt.cmod(s.tid, 8) * 4
            )
            incoming = rt.rmem_tensor(4 * groups, Uint32)
            done = Int32(0)
            while done == 0:
                stale = Uint32(0)
                for g in cutlass.range_constexpr(groups):
                    if rows_live > (self.ctx.first + g) * self.GROUP_ROWS:
                        generation = (
                            rt.Memory.lds(self.tail_epochs(s) + g * 4, Uint32) + 1
                        )
                        v = rt.Memory.ld_volatile_v4(
                            base
                            + Int64(g * PUSH_MAILBOX_BYTES)
                            + Int64(item + rt.cmod(generation, 2) * 4096) * 4
                        )
                        for w in cutlass.range_constexpr(4):
                            incoming[4 * g + w] = v[w]
                            stale = stale | ((v[w] >> 24) ^ ((generation & 127) | 128))
                if stale == 0:
                    done = Int32(1)
            for g in cutlass.range_constexpr(groups):
                words = rt.TaggedMailbox.unpack24(
                    [incoming[4 * g + w] for w in range(4)]
                )
                for i in cutlass.range_constexpr(3):
                    rt.Memory.sts(
                        s.act_addr
                        + (
                            g * 672
                            + rt.cdiv(s.tid, 8) * 24
                            + rt.cmod(s.tid, 8) * 3
                            + i
                            + self.RECEIVED
                        )
                        * 4,
                        words[i],
                        Uint32,
                    )

    #: Word of ``act`` where a ``WIDE_TAIL`` pass keeps the groups' received O words.
    RECEIVED = 3200

    @cute.jit
    def reduce_wide(self, p0, s, rows_live):
        for i in cutlass.range_constexpr(-(-self.PASS_GROUPS * 48 // 256)):
            index = s.tid + 256 * i
            group = rt.cdiv(index, 48)
            if (group < self.PASS_GROUPS) & (
                rows_live > (self.ctx.first + group) * self.GROUP_ROWS
            ):
                query_ = rt.cdiv(rt.cmod(index, 48), 12)
                lane = rt.cmod(index, 12)
                own_at = self.tail_keep(s) + ((group * 4 + query_) * 24 + lane * 2) * 4
                loaded = []
                for r in cutlass.range_constexpr(8):
                    own = Int32(p0.peer_rank == r)
                    above = Int32(p0.peer_rank < r)
                    peer_at = (
                        s.act_addr
                        + (
                            group * 672
                            + query_ * 168
                            + (r - above) * 24
                            + lane * 2
                            + self.RECEIVED
                        )
                        * 4
                    )
                    at = own * own_at + (1 - own) * peer_at
                    loaded.append(
                        (rt.Memory.lds(at, Uint32), rt.Memory.lds(at + 4, Uint32))
                    )
                sums = [Float32(0.0)] * 4
                for r in cutlass.range_constexpr(8):
                    f = list(rt.Convert.bf16x2_to_f32x2(loaded[r][0])) + list(
                        rt.Convert.bf16x2_to_f32x2(loaded[r][1])
                    )
                    sums = [rt.add(sums[j], f[j]) for j in range(4)]
                rt.Memory.st_shared_v2(
                    own_at,
                    rt.f32x2_to_bf16x2(sums[0], sums[1]),
                    rt.f32x2_to_bf16x2(sums[2], sums[3]),
                )

    @cute.jit
    def final_wide(self, p0, s, rows_live, slot):
        residual = self.tail_keep(s) + self.PASS_GROUPS * 384
        for i in cutlass.range_constexpr(-(-self.PASS_GROUPS * 96 // 256)):
            index = s.tid + 256 * i
            row = rt.cdiv(index, 24)
            if (row < 4 * self.PASS_GROUPS) & (
                rows_live > (self.ctx.first + rt.cdiv(row, 4)) * self.GROUP_ROWS
            ):
                o = rt.Convert.bf16x2_to_f32x2(
                    rt.Memory.lds(self.tail_keep(s) + index * 4, Uint32)
                )
                r = rt.Convert.bf16x2_to_f32x2(
                    rt.Memory.lds(residual + index * 4, Uint32)
                )
                rt.Memory.st_global_b32(
                    rt.ga(p0.Final, row * 6144 + slot * 48 + rt.cmod(index, 24) * 2),
                    rt.f32x2_to_bf16x2(rt.add(o[0], r[0]), rt.add(o[1], r[1])),
                )

    @stage
    @cute.jit
    def compute_tail_wide(self, s):
        self.set_regs("compute")
        assert 4 <= self.PASS_GROUPS <= 12 and self.PASS_GROUPS % 4 == 0, (
            "the batched tail takes 4, 8 or 12 groups (received words fit act)"
        )
        slot = self.role_slot(s)
        if slot < 128:
            p0 = self.ctx.params(self.ctx.first)
            rows_live = Int32(p0.RowTable[0])
            if s.tid < self.PASS_GROUPS:  # the groups' O-exchange generations
                rt.Memory.sts(
                    self.tail_epochs(s) + s.tid * 4,
                    self.exchange_epoch_of(p0, s, s.tid),
                    Uint32,
                )
            rt.bar_sync(*BAR_ALL)
            self.merge_quarter(p0, s, rows_live, slot)
            self.gather_merged(p0, s, rows_live, slot)
            rt.bar_sync(*BAR_COMPUTE)
            self.vo_wide(p0, s, rows_live, slot)
            blocks = self.PASS_GROUPS // self.TAIL_BLOCK
            accum = rt.rmem_tensor(12 * (self.TAIL_BLOCK // 2), Float32)
            rot = rt.cmod(slot, self.TAIL_BLOCK)
            for step in cutlass.range(blocks, unroll=1):
                block = rt.cmod(step + rt.cdiv(slot, self.TAIL_BLOCK), blocks)
                self.o_projection_wide(p0, s, rows_live, block, rot, accum)
                self.own_o_wide(s, block, rot, accum)
                self.push_wide(p0, s, rows_live, block, slot)
            self.receive_wide(p0, s, rows_live, slot)
            rt.bar_sync(*BAR_COMPUTE)
            self.reduce_wide(p0, s, rows_live)
            rt.bar_sync(*BAR_ALL)
            self.final_wide(p0, s, rows_live, slot)

    @stage
    @cute.jit
    def load_tail_wide(self, s):
        p0 = self.ctx.params(self.ctx.first)
        self.set_regs("load")
        rid = self.role_slot(s)
        lane_0 = s.tid - 256
        rows_live = Int32(p0.RowTable[0])
        if rid < 128:
            o_policy = self.tail_weights(p0, s, rid)
            for i in cutlass.range_constexpr(-(-self.PASS_GROUPS // 4)):
                h = s.warp - 8 + 4 * i
                if (h < self.PASS_GROUPS) & (
                    rows_live > (self.ctx.first + h) * self.GROUP_ROWS
                ):
                    self.warp_poll_ge(
                        self.split_done_of(p0, s, rt.cdiv(rid, 32) * 32 + s.lane, h),
                        Int32(1),
                        Uint32(s.epoch_slot[PassPhases.EPOCHS + h]) + 1,
                    )
            rt.Memory.fence_acquire_gpu()
            rt.bar_sync(*BAR_LOADER)
            rt.bar_sync(*BAR_ALL)
            with rt.warp_leader(s.warp, 8):
                rt.mbar_wait(s.merge_loaded, 0)
                for page in cutlass.range_constexpr(2, 4):
                    self.issue_o_page(p0, s, page, rid, o_policy)
            for i in cutlass.range_constexpr(-(-self.PASS_GROUPS * 24 // 128)):
                chunk = lane_0 + 128 * i
                if chunk < self.PASS_GROUPS * 24:
                    rt.AsyncCopy.cp_async_cg16(
                        self.tail_keep(s) + self.PASS_GROUPS * 384 + chunk * 16,
                        rt.ga(
                            p0.Hidden,
                            rt.cdiv(chunk, 6) * 6144 + rid * 48 + rt.cmod(chunk, 6) * 8,
                        ),
                    )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            rt.bar_sync(*BAR_ALL)
        self.seam_gates(s)

    @stage
    @cute.jit
    def compute_attn(self, s):
        self.set_regs("compute")
        assert self.WIDE, "the attention loop reads the q-down words of phase proj"
        cute.arch.griddepcontrol_wait()
        first = self.ctx.first
        rid = self.role_slot(s)
        tile_cta = (rid < 128) | (rid >= 132)
        own = None
        if const_expr(self.PACKED):  # the row this CTA serves: its own block table
            own = self.own_row(rid)
        if tile_cta:
            self.query_norm_qa(
                self.ctx.params(first, s), s, rid, self.group_tag(s, 0), Int32(0), True
            )
        for h in cutlass.range(self.PASS_GROUPS, unroll=1):
            if self.ctx.live(first + h):
                start = rt.globaltimer()
                if tile_cta & (h + 1 < self.PASS_GROUPS):  # noqa: SIM102
                    if self.ctx.live(first + h + 1):
                        self.query_norm_qa(
                            self.ctx.params(first + h + 1, s),
                            s,
                            rid,
                            self.group_tag(s, h + 1),
                            Int32(0),
                            True,
                        )
                p = self.ctx.params(first + h, s, own)
                tag = self.group_tag(s, h)
                position = self.loop_position(s, h, own)
                if rid < 128:
                    self.loop_attention(p, s, position, tag, rid, h)
                if (rid >= 128) & (rid < 132):
                    query_ = rid - 128
                    self.kv_latent_store(
                        p,
                        s,
                        Handoff.KV_LATENT + query_ * Handoff.KV_LATENT_ROW,
                        query_,
                        position,
                        tag,
                        self.row_stores(p, s, (first + h) * self.GROUP_ROWS, query_),
                    )
                    rt.bar_sync(*BAR_COMPUTE)
                    self.release_flag(p, s, FlagWords.KV_READY + rid)
                if rid >= 132:  # noqa: SIM102
                    if self.prefetch_group(p):
                        self.moe_prefetch_tail(p, s, rid, start)

    @cute.jit
    def loop_attention(self, p, s, position, tag, slot, h):
        query_ = rt.cdiv(slot, 32)
        split = rt.cmod(slot, 32)
        score_count = cutlass.min(position + query_ + 1, 2048)
        if score_count > split * 64:
            self.poll_operands(p, s, tag, query_)
        buf = rt.cmod(h, 2)
        rt.mbar_wait(s.tail_full + buf * 8, rt.cmod(rt.cdiv(h, 2), 2))
        rt.bar_sync(*BAR_COMPUTE)  # every thread's operand words are in the tile
        split_count = rt.cdiv(score_count + 64 - 1, 64)
        seq = LOOP_KV_FIRST + buf + 4 * rt.cdiv(h, 2)
        if split < split_count:
            self.km_split_attention(
                p,
                s,
                seq,
                seq,
                split,
                split_count,
                score_count,
                query_,
                buf * LOOP_PE_STRIDE,
            )
        rt.bar_sync(*BAR_COMPUTE)
        rt.bar_sync(*BAR_COMPUTE)
        done = self.split_done(p, s, slot)
        with rt.warp_leader(s.warp, 0):
            rt.Memory.red_async_release_gpu_add(done, Uint32(1), memory=True)
        cute.arch.fence_view_async_shared()
        rt.mbar_arrive(s.ring_empty + rt.cmod(seq, 4) * 8)

    @stage
    @cute.jit
    def load_attn(self, s):
        self.set_regs("load")
        first = self.ctx.first
        rid = self.role_slot(s)
        p0 = self.ctx.params(first, s)
        cute.arch.griddepcontrol_wait()
        if (rid < 128) | (rid >= 132):
            policy = rt.AsyncCopy.createpolicy_evict_first(
                p0.AbsorbWeight.iterator.toint(), 9474048, 9474048
            )
            for page in cutlass.range_constexpr(2):
                self.issue_ring_page(
                    p0,
                    s,
                    page,
                    p0.AbsorbWeight,
                    ((rid - 4 * Int32(rid >= 132)) * 2 + page) * Ring.WEIGHT_PAGE_BYTES,
                    policy,
                    False,
                )
        own = None
        if const_expr(self.PACKED):  # the row this CTA serves: its own block table
            own = self.own_row(rid)
        for h in cutlass.range(self.PASS_GROUPS, unroll=1):
            if self.ctx.live(first + h):
                start = rt.globaltimer()
                p = self.ctx.params(first + h, s, own)
                epoch = Uint32(s.epoch_slot[PassPhases.EPOCHS + h])
                position = self.loop_position(s, h, own)
                if self.prefetch_slot(p, rid) >= 132:
                    self.moe_list_prefetch(p, s, rid, epoch, start)
                if rid < 128:
                    self.loop_kv(p, s, h, rid, epoch, position)

    @cute.jit
    def loop_kv(self, p, s, h, rid, epoch, position):
        query_ = rt.cdiv(rid, 32)
        split = rt.cmod(rid, 32)
        score_count = rt.FastMath.min_typed(position + query_ + 1, 2048)
        active = Int32(score_count > split * 64)
        buf = rt.cmod(h, 2)
        seq = LOOP_KV_FIRST + buf + 4 * rt.cdiv(h, 2)
        pe_extra = buf * LOOP_PE_STRIDE
        if s.tid >= 320:
            token = None
            if const_expr(self.PACKED):
                token = self.row_info(None, s, h, query_)[2]
            self.load_shared_ids(p, s, rid, epoch, position, token)
        rt.bar_sync(*BAR_LOADER)
        self.arm_kv_slot(s, seq, active)
        rt.bar_sync(*BAR_LOADER)
        if active != 0:
            self.kv_gather(p, s, 0, seq, seq, L2_HINT)
        if ((s.warp_hits[0] | s.warp_hits[1]) != 0) & (s.warp == 8):
            self.poll_head_flags(p, s.lane, epoch)
        rt.bar_sync(*BAR_LOADER)
        if active != 0:
            self.pe_stage(p, s, score_count, split, L2_HINT, pe_extra)
        cute.arch.cp_async_commit_group()
        if active != 0:
            self.kv_gather(p, s, 1, seq, seq, L2_HINT)
        cute.arch.cp_async_wait_group(0)
        if active != 0:
            self.pe_widen(s, pe_extra)
        rt.bar_sync(*BAR_LOADER)
        if s.tid == 256:
            rt.mbar_arrive(s.tail_full + buf * 8)

    @stage
    @cute.jit
    def arm_kv_slot(self, s, seq, active):
        with rt.warp_leader(s.warp, 8):
            rt.mbar_wait(s.ring_empty + rt.cmod(seq, 4) * 8, s.pages.parity(seq) ^ 1)
            if active != 0:
                rt.mbar_expect_tx(s.ring_full + rt.cmod(seq, 4) * 8, 32768)
            else:
                rt.mbar_arrive(s.ring_full + rt.cmod(seq, 4) * 8)

    def role_slot(self, s):
        slot = s.bid
        if const_expr(self.shared_or_seam):
            slot = Int32(s.role_entry & Handoff.ROLE_MASK)
        assert not self.pushed, "the pushed role runs no phased pass"
        return slot

    @cute.jit
    def group_tag(self, s, h):
        epoch = Uint32(s.epoch_slot[PassPhases.EPOCHS + h])
        return ((epoch + 1) & 32767) | 32768

    @cute.jit
    def exchange_epoch(self, p, s, h):
        epoch = Uint32(s.epoch_slot[PassPhases.EPOCHS + h])
        if const_expr(self.shared_or_seam):
            epoch = (epoch + 1) * 57 + p.Flags[FlagWords.SHARED_INDEX] - 1
        return epoch

    @cute.jit
    def tail_weights(self, p0, s, rid):
        with rt.warp_leader(s.warp, 8):
            rt.AsyncCopy.prefetch_tensormap(p0.OTmap)
            rt.AsyncCopy.prefetch_tensormap(p0.KvbTmap)
        kvb_policy = rt.AsyncCopy.createpolicy_evict_first(
            p0.KVbWeight.iterator.toint(), 1835008, 1835008
        )
        o_policy = rt.AsyncCopy.createpolicy_evict_first(
            p0.OWeight.iterator.toint(), 12582912, 12582912
        )
        self.tail_weight_pages(p0, s, rid, kvb_policy, o_policy)
        return o_policy

    @cute.jit
    def seam_gates(self, s):
        if const_expr(self.seam):
            for h in cutlass.range(self.PASS_GROUPS, unroll=1):
                group = self.ctx.first + h
                if self.ctx.live(group):
                    p = self.ctx.params(group)
                    self.seam_gate(p, s, Uint32(s.epoch_slot[PassPhases.EPOCHS + h]))

    @cute.jit
    def tail_weight_pages(self, p, s, rid, kvb_policy, o_policy):
        with rt.warp_leader(s.warp, 8):
            rt.mbar_expect_tx(s.tail_full, 32800)
            for half in cutlass.range_constexpr(2):
                kvb_row = (
                    rt.cdiv(2 * rt.cmod(rid, 32) + half, 8) * 448
                    + 192
                    + rt.cmod(2 * rt.cmod(rid, 32) + half, 8) * 32
                )
                rt.AsyncCopy.tma_3d(
                    s.ring_addr + 116224 * half,
                    p.KvbTmap,
                    0,
                    kvb_row,
                    0,
                    s.tail_full,
                    L2_HINT,
                )
                rt.bulk_g2s_hint(
                    s.ring_addr + (16384 + 116224 * half),
                    rt.baddr(p.KVbScales, rt.cdiv(kvb_row, 128) * 4 * 4),
                    16,
                    s.tail_full,
                    kvb_policy,
                )
            for page in cutlass.range_constexpr(2):
                self.issue_o_page(p, s, page, rid, o_policy)

    @cute.jit
    def seam_gate(self, p, s, epoch):
        with rt.warp_leader(s.warp, 9):
            done = Int32(0)
            while done == 0:
                seen = Uint32(rt.Memory.ld_relaxed_gpu(rt.ga(p.FullEpoch, 0)))
                if seen - epoch * 148 >= 148:
                    done = Int32(1)

    @cute.jit
    def prologue(self, p: WorkerParams):
        s = WorkerSmem.build(self, p)
        self.smem_record = s
        self.setup(p, s)
        return rt.ThreadIds(s.tid, s.bid, s.warp, s.lane), s

    @stage
    @cute.jit
    def setup(self, p, s):
        if (s.warp == 8) & (s.bid < 128):
            rt.AsyncCopy.prefetch_tensormap(p.KvCache)
        if s.warp == 0:
            leader = rt.elect_sync()
            if leader != 0:
                for _rr0 in cutlass.range_constexpr(2):
                    for _rr1 in cutlass.range_constexpr(4):
                        rt.AsyncCopy.mbar_init(
                            s.smem + ((8 * _rr1) + 32 * _rr0), (1 + 255 * _rr0)
                        )
                for _rr2 in cutlass.range_constexpr(5):
                    rt.AsyncCopy.mbar_init(s.smem + (64 + 8 * _rr2), 1)
                if const_expr(not self.pushed):
                    for _rr3_0 in cutlass.range_constexpr(2):
                        rt.AsyncCopy.mbar_init(
                            s.smem + (104 + 16 * _rr3_0), (1 + 255 * _rr3_0)
                        )
                        rt.AsyncCopy.mbar_init(
                            s.smem + (112 + 16 * _rr3_0), (256 - 255 * _rr3_0)
                        )
                if const_expr(self.pushed):
                    rt.AsyncCopy.mbar_init(s.smem + 104, 1)
                    rt.AsyncCopy.mbar_init(s.smem + 112, 256)
                    rt.AsyncCopy.mbar_init(s.smem + 120, 256)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

    @stage
    @cute.jit
    def compute(self, p, s, ids):
        if const_expr(self.phase == "tail"):
            if const_expr(self.WIDE_TAIL):
                self.compute_tail_wide(s)
            else:
                self.compute_tail(s)
        elif const_expr(self.phase == "proj"):
            self.compute_proj(s)
        elif const_expr(self.phase == "attn"):
            self.compute_attn(s)
        else:
            (
                compute_start,
                cur_pos_c,
                epoch,
                role_slot,
                router_on,
                shared_index_compute,
            ) = self.compute_prologue(p, s)
            (epoch, tag) = self.input_norm_q_down(
                p, s, cur_pos_c, epoch, role_slot, router_on
            )
            if const_expr(self.full_or_push):
                self.full_qa_kv(p, s, cur_pos_c, epoch, tag)
                self.full_scores_vo(p, s, compute_start, cur_pos_c, epoch, tag)
            if const_expr(self.full_or_push and self.REFINE):
                self.refine_selection(p, s, cur_pos_c, epoch)
            self.pushed_compute(p, s, cur_pos_c, epoch, tag)
            if const_expr(self.shared_or_seam):
                self.shared_query_attention(p, s, cur_pos_c, tag)
                self.shared_vo_exchange(
                    p, s, compute_start, epoch, shared_index_compute, tag
                )

    @stage
    @cute.jit
    def load(self, p, s, ids):
        if const_expr(self.phase == "tail"):
            if const_expr(self.WIDE_TAIL):
                self.load_tail_wide(s)
            else:
                self.load_tail(s)
        elif const_expr(self.phase == "proj"):
            self.load_proj(s)
        elif const_expr(self.phase == "attn"):
            self.load_attn(s)
        else:
            (
                absorb_policy,
                cur_pos_l,
                early_even,
                early_odd,
                early_position,
                epoch,
                gate_seen,
                input_policy,
                l2_range_policy_1,
                l2_range_policy_2,
                loader_start,
                rid,
                shared_index_load,
            ) = self.loader_setup(p, s)
            self.loader_pages(
                p,
                s,
                absorb_policy,
                cur_pos_l,
                early_even,
                early_odd,
                early_position,
                epoch,
                input_policy,
                loader_start,
                rid,
            )
            self.loader_kv_tail(
                p,
                s,
                cur_pos_l,
                epoch,
                gate_seen,
                l2_range_policy_1,
                l2_range_policy_2,
                rid,
                shared_index_load,
            )

    @stage
    @cute.jit
    def refine_selection(self, p, s, cur_pos_c, epoch):
        refine_bid0 = 128
        if const_expr(kFull == self.ROLE and not self.ATTN_LOOP):
            refine_bid0 = 144
        if (s.bid >= refine_bid0) & (s.bid < refine_bid0 + 4):
            refine_generation = epoch + 1
            refine_parity = Int32(rt.cmod(epoch, 2))
            refine_live = cur_pos_c + (s.bid - refine_bid0) + 1
            if refine_live > 2048:
                for refine_round in cutlass.range(0, 2, 1, unroll=1):
                    refine_dry = Int32(refine_round == 0)
                    if s.warp == 0:
                        header_word = Uint64(16779234)
                        _hdr_done = Int32(0)
                        while (1 - refine_dry > 0) & (_hdr_done == 0):
                            header_word = rt.Memory.ld_volatile_u64(
                                rt.ga(
                                    p.SelectionMailbox,
                                    552448
                                    + (refine_parity * 4 + (s.bid - refine_bid0)) * 2,
                                ),
                                memory=True,
                            )
                            _vote_0 = rt.Sync.vote_all(
                                Uint32(header_word >> 32) == refine_generation
                            )
                            if _vote_0 != 0:
                                _hdr_done = Int32(1)
                            if _hdr_done == 0:
                                gtimer_1 = rt.globaltimer()
                                backoff_start = gtimer_1
                                _backoff_done = Int32(0)
                                while _backoff_done == 0:
                                    gtimer_2 = rt.globaltimer()
                                    backoff_now = gtimer_2
                                    _vote_1 = rt.Sync.vote_all(
                                        backoff_now - backoff_start >= 256
                                    )
                                    if _vote_1 != 0:
                                        _backoff_done = Int32(1)
                        rt.syncwarp()
                        if s.lane == 0:
                            s.refine_in[0] = Uint32(header_word) & 65535
                            s.refine_in[1] = Uint32(header_word) >> 16
                            s.refine_in[2] = Uint32(0)
                            s.refine_in[3] = Uint32(0)
                    for part in cutlass.range_constexpr(4):
                        s.refine_hist[part * 256 + s.tid] = Uint32(0)
                    rt.bar_sync(*BAR_COMPUTE)
                    refine_above = Int32(s.refine_in[0])
                    refine_count = Int32(s.refine_in[1])
                    if refine_count <= 16240:
                        (
                            emit_begin,
                            emit_below,
                            emit_end,
                            emit_first,
                            emit_last,
                            emit_slot,
                            limit,
                            threshold,
                        ) = self.refine_emit_plan(
                            p,
                            s,
                            part,
                            refine_above,
                            refine_bid0,
                            refine_count,
                            refine_dry,
                            refine_generation,
                            refine_live,
                            refine_parity,
                        )
                        for emit_batch in cutlass.range(
                            emit_begin, emit_last, 256, unroll=1
                        ):
                            emit_keys = []
                            emit_ids = []
                            for item in cutlass.range_constexpr(8):
                                _min_2 = emit_end - 1
                                if (emit_batch + item * 32 + s.lane) < (emit_end - 1):
                                    _min_2 = emit_batch + item * 32 + s.lane
                                emit_ids.append(
                                    Uint32(s.refine_cache[(_min_2 * 2) + 0])
                                )
                                emit_keys.append(
                                    Uint32(s.refine_cache[(_min_2 * 2) + 1])
                                )
                            for item in cutlass.range_constexpr(8):
                                emit_take = Int32(
                                    emit_end > emit_batch + item * 32 + s.lane
                                ) & (
                                    Int32(threshold < emit_keys[item])
                                    | Int32(
                                        (emit_keys[item] == threshold)
                                        & (limit >= emit_ids[item])
                                    )
                                )
                                _ballot_0 = rt.Sync.vote_ballot(emit_take != 0)
                                if (emit_take != 0) & (refine_dry == 0):
                                    emit_record = Uint64(
                                        refine_generation
                                    ) << 32 | Uint64(emit_ids[item])
                                    _popc_0 = rt.Sync.popc(_ballot_0 & emit_below)
                                    self.store_selection(
                                        p, emit_first + emit_slot + _popc_0, emit_record
                                    )
                                _popc_0 = rt.Sync.popc(_ballot_0)
                                emit_slot = emit_slot + _popc_0

    @stage
    @cute.jit
    def compute_prologue(self, p, s):
        role_slot = s.bid
        if const_expr(self.shared_or_seam):
            role_slot = Int32(s.role_entry & Handoff.ROLE_MASK)
        gtimer = Uint64(0)
        compute_start = Uint64(0)
        shared_index_compute = Uint32(0)
        if const_expr(self.shared_or_seam):
            with rt.warp_leader(s.warp, 0):
                rt.AsyncCopy.prefetch_tensormap(p.InputTmap)
            shared_index_compute = p.Flags[FlagWords.SHARED_INDEX]
        if const_expr(self.full_or_push):
            with rt.warp_leader(s.warp, 0):
                rt.AsyncCopy.prefetch_tensormap(p.InputTmap)
        if const_expr(not self.pushed and not self.seam):
            gtimer = rt.globaltimer()
            compute_start = gtimer
        with rt.warp_leader(s.warp, 0):
            if const_expr(not self.pushed and not self.seam):
                cute.arch.griddepcontrol_wait()
            if const_expr(not self.pushed):
                if const_expr(self.full_or_push):
                    router_on_e = Int32(0)
                    if self.is_router(s.bid):
                        router_on_e = self.router_enabled(p)
                    router_go = router_on_e != 0
                else:
                    router_go = self.is_router(role_slot)
                if router_go:
                    if const_expr(not self.seam):
                        self.load_x(s, p.X)
                    if const_expr(self.full_or_push):
                        router_src = self.router_weight_addr(p)
                    else:
                        router_src = rt.ga(p.QueryWeight, 0)
                    l2_range_policy = rt.AsyncCopy.createpolicy_evict_first(
                        router_src, 1579008, 1579008
                    )
                    rt.mbar_expect_tx(s.ring_full, Ring.WEIGHT_PAGE_BYTES)
                    if const_expr(self.full_or_push):
                        router_src = self.router_weight_addr(p)
                    rt.bulk_g2s_hint(
                        s.ring_addr,
                        router_src
                        + Int64((role_slot - 132) * 6 * Ring.WEIGHT_PAGE_BYTES),
                        Ring.WEIGHT_PAGE_BYTES,
                        s.ring_full,
                        l2_range_policy,
                    )
                if role_slot < INPUT_CTAS:
                    if const_expr(not self.seam):
                        self.load_x(s, p.X)
                    l2_range_policy_0 = rt.AsyncCopy.createpolicy_evict_first(
                        rt.ga(p.InputWeight, 0), 16121856, 16121856
                    )
                    rt.mbar_expect_tx(s.ring_full, 32960)
                    rt.bulk_g2s_hint(
                        s.ring_addr + 33792,
                        rt.ga(p.InputScales, Int64(rt.cdiv(role_slot, 4) * 48 * 4)),
                        192,
                        s.ring_full,
                        l2_range_policy_0,
                    )
                    rt.AsyncCopy.tma_3d(
                        s.ring_addr,
                        p.InputTmap,
                        0,
                        role_slot * 32,
                        0,
                        s.ring_full,
                        L2_HINT,
                    )
            if const_expr(self.pushed):
                rt.AsyncCopy.prefetch_tensormap(p.InputTmap)
            if const_expr(self.seam):
                cute.arch.griddepcontrol_wait()
                if self.is_router(role_slot):
                    self.load_x(s, p.XG)
                if role_slot < INPUT_CTAS:
                    self.load_x(s, p.XG)
        self.set_regs("compute")
        if const_expr(not self.pushed):
            cute.arch.griddepcontrol_wait()
        cur_pos_c = self.position(p)
        if const_expr(self.seam):
            gtimer = rt.globaltimer()
            compute_start = gtimer
        epoch = Uint32(0)
        router_on = Int32(0)
        if const_expr(self.full_or_push):  # noqa: SIM102
            if self.is_router(s.bid):
                router_on = self.router_enabled(p)
        return (
            compute_start,
            cur_pos_c,
            epoch,
            role_slot,
            router_on,
            shared_index_compute,
        )

    @cute.jit
    def cache_rows(self, p0, s):
        if s.lane < self.PASS_ROWS:
            base, block_row, token, _ = PackedRows.meta(
                self.layout,
                p0.RowTable,
                self.ctx.first * self.GROUP_ROWS + s.lane,
                rt.cmod(s.lane, self.GROUP_ROWS),
            )
            s.row_words[PackedRows.BASE + s.lane] = base
            s.row_words[PackedRows.BLOCK_ROW + s.lane] = block_row
            s.row_words[PackedRows.TOKEN + s.lane] = token
        if s.lane == 0:
            s.row_words[PackedRows.ROWS] = Int32(p0.RowTable[0])

    @stage
    @cute.jit
    def compute_proj(self, s):
        self.set_regs("compute")
        assert self.PASS_ROWS % 8 == 0 and self.PASS_ROWS <= 56, (
            "the wide projection takes a multiple of eight rows, at most 56"
        )
        cute.arch.griddepcontrol_wait()
        p0 = self.ctx.params(self.ctx.first)
        layout = self.layout
        if s.warp == 0:
            if const_expr(self.PACKED):
                self.cache_rows(p0, s)
            if const_expr(not self.PACKED):  # noqa: SIM102
                if s.lane < self.PASS_GROUPS:
                    group = self.ctx.first + s.lane
                    first_row = group * self.GROUP_ROWS
                    if p0.RowTable[0] > first_row:
                        request = p0.RowTable[layout.row_request_word(0) + first_row]
                        context = p0.RowTable[layout.context_word(0) + request]
                        request_first = p0.RowTable[layout.first_row_word(0) + request]
                        block_row = p0.RowTable[layout.block_row_word(0) + request]
                        s.epoch_slot[ProjectionChunks.BLOCK_ROW + s.lane] = Uint32(
                            block_row
                        )
                        s.epoch_slot[ProjectionChunks.POSITION + s.lane] = Uint32(
                            context - 1 + first_row - request_first
                        )
        rt.bar_sync(*BAR_ALL)
        role_slot = self.role_slot(s)
        rows_live = self.live_rows(p0)
        if (role_slot < INPUT_CTAS) | (self.proj_router(p0, role_slot) != 0):
            self.proj_gemm(s, p0, role_slot, rows_live)
        else:
            self.proj_inverse(s, p0, role_slot, rows_live)

    @stage
    @cute.jit
    def proj_gemm(self, s, p0, role_slot, rows_live):
        nb = self.PROJ_ROWS
        tiles = nb // 8
        w0, x0, w1, x1 = ProjectionChunks.offsets(nb)
        ring = rt.Pipeline.Ring(s.ring_full, s.ring_empty, 2)
        router = self.proj_router(p0, role_slot)
        scale_stride = cutlass.select_(router != 0, Int32(32), Int32(8))
        blocks = rt.cdiv(rows_live + nb - 1, nb)
        quad = rt.cmod(s.lane, 4)
        row = rt.cdiv(s.lane, 4)
        w_thread = s.warp * 4096 + row * 128
        x_thread = row * ProjectionChunks.X_PITCH + (quad * 16 + 64 * s.warp) * 4
        rt.mbar_wait(s.tail_full, 0)  # the CTA's scale words
        for blk in cutlass.range(blocks, unroll=1):
            accum = rt.rmem_tensor(8 * tiles, Float32)
            for i in cutlass.range_constexpr(8 * tiles):
                accum[i] = Float32(0.0)
            for page in cutlass.range(ProjectionChunks.PAGES, unroll=1):
                seq = blk * ProjectionChunks.PAGES + page
                ring.wait_full(seq)
                cute.arch.fence_view_async_shared()
                chunk = rt.cmod(seq, 2)
                if const_expr(not self.seam):
                    self.proj_gamma(
                        s,
                        p0,
                        s.ring_addr + cutlass.select_(chunk == 0, Int32(x0), Int32(x1)),
                        nb,
                        page,
                    )
                w_base = s.ring_addr + cutlass.select_(chunk == 0, Int32(w0), Int32(w1))
                w_base = w_base + w_thread
                x_base = s.ring_addr + cutlass.select_(chunk == 0, Int32(x0), Int32(x1))
                x_base = x_base + x_thread
                vin_rows = [None] * 16
                for j in cutlass.range_constexpr(4):
                    for h in cutlass.range_constexpr(2):
                        w = rt.Memory.lds_b128(
                            w_base + (((2 * quad + h) ^ row) * 16) + 1024 * j
                        )
                        vin_rows[4 * j + 2 * h] = Uint64(w[0]) | Uint64(w[1]) << 32
                        vin_rows[4 * j + 2 * h + 1] = Uint64(w[2]) | Uint64(w[3]) << 32
                vin_acc = [None] * (8 * tiles)
                for j in cutlass.range_constexpr(4):
                    vin_a = [None] * 16
                    for h in cutlass.range_constexpr(2):
                        even = rt.Convert.unpack_u16x4(vin_rows[j + 8 * h])
                        odd = rt.Convert.unpack_u16x4(vin_rows[j + 4 + 8 * h])
                        for i in cutlass.range_constexpr(4):
                            vin_a[8 * h + 2 * i] = rt.Convert.e4m3x2_to_bf16x2(even[i])
                            vin_a[8 * h + 2 * i + 1] = rt.Convert.e4m3x2_to_bf16x2(
                                odd[i]
                            )
                    for t in cutlass.range_constexpr(2):
                        for nt in cutlass.range_constexpr(tiles):
                            vin_b = rt.Memory.lds_weak_b64(
                                x_base
                                + nt * 8 * ProjectionChunks.X_PITCH
                                + (4 * j + 2 * t) * 4
                            )
                            for h in cutlass.range_constexpr(2):
                                a = tuple(vin_a[8 * h + 4 * t : 8 * h + 4 * t + 4])
                                at = (nt * 2 + h) * 4
                                if const_expr(j == 0 and t == 0):
                                    vin_acc[at : at + 4] = rt.TensorCore.mma(
                                        None, a, tuple(vin_b), True
                                    )
                                else:
                                    rt.TensorCore.mma_acc_at(
                                        vin_acc, at, a, tuple(vin_b)
                                    )
                scale = rt.Memory.lds(
                    s.received_words_addr + (page * scale_stride + s.warp) * 4, Float32
                )
                for i in cutlass.range_constexpr(8 * tiles):
                    accum[i] = rt.fma(vin_acc[i], scale, accum[i])
                ring.release(chunk)
            self.proj_epilogue(s, p0, accum, blk, rows_live, role_slot)

    @cute.jit
    def proj_gamma(self, s, p0, x_chunk, nb, page):
        column = rt.cmod(s.tid, 128)
        gamma = rt.Memory.ld_global_v8_f32(rt.ga(p0.GammaX, page * 1024 + column * 8))
        for row in cutlass.range(rt.cdiv(s.tid, 128), nb, 2, unroll=1):
            address = x_chunk + row * ProjectionChunks.X_PITCH + column * 16
            w = rt.Memory.lds_b128(address)
            values = rt.bf16x8_to_f32([w[0], w[1], w[2], w[3]])
            rt.Memory.sts_v4(
                Uint32(address),
                [
                    rt.f32x2_to_bf16x2(
                        rt.mul(values[2 * i], gamma[2 * i]),
                        rt.mul(values[2 * i + 1], gamma[2 * i + 1]),
                    )
                    for i in range(4)
                ],
            )
        rt.Memory.fence_proxy_async_shared()
        rt.bar_sync(*BAR_COMPUTE)

    @stage
    @cute.jit
    def proj_epilogue(self, s, p0, accum, blk, rows_live, role_slot):
        nb = self.PROJ_ROWS
        if s.warp == 0:
            row = blk * nb + s.lane
            wanted = (s.lane < nb) & (row < rows_live)
            group = rt.cdiv(row, self.GROUP_ROWS)
            pg = self.ctx.params(self.ctx.first + group, s)
            address = self.handoff(
                pg, ProjectionChunks.RECORD_WORD + 2 * rt.cmod(row, self.GROUP_ROWS)
            )
            want = Uint32(s.epoch_slot[PassPhases.EPOCHS + group]) + 1
            inverse = Float32(0.0)
            done = Int32(0)
            while done == 0:
                ready = Int32(1)
                if wanted:
                    record = rt.Memory.ld_volatile_u64(address, memory=True)
                    ready = Int32(Uint32(record >> 32) == want)
                    inverse = rt.Convert.bits_f32(Uint32(record))
                done = rt.Sync.vote_all(ready != 0)
            if wanted:
                s.pairs[row] = inverse
        rt.bar_sync(*BAR_COMPUTE)
        for t in cutlass.range_constexpr(nb // 8):
            tile = [accum[t * 8 + i] for i in range(8)]
            self.rows_to_scratch(s, tile, 8, 2)
            value = self.warp_sum(s, s.tid, 8, 2)
            row = blk * nb + t * 8 + rt.cdiv(s.tid, 32)
            group = rt.cdiv(row, self.GROUP_ROWS)
            if row < rows_live:
                query = None
                if const_expr(self.PACKED):  # the row's own block table
                    query = rt.cmod(row, self.GROUP_ROWS)
                pg = self.ctx.params(self.ctx.first + group, s, query)
                epoch = Uint32(s.epoch_slot[PassPhases.EPOCHS + group])
                self.proj_store(
                    pg,
                    s,
                    rt.mul(value, s.pairs[row]),
                    rt.cmod(row, self.GROUP_ROWS),
                    rt.cmod(s.tid, 32),
                    role_slot,
                    ((epoch + 1) & 32767) | 32768,
                    self.proj_position(s, group, row),
                    self.proj_stores(s, row),
                )
            rt.bar_sync(*BAR_COMPUTE)
        if (role_slot >= 80) & (role_slot < 82):
            for g in cutlass.range_constexpr(nb // self.GROUP_ROWS):
                group = rt.cdiv(blk * nb, self.GROUP_ROWS) + g
                if group * self.GROUP_ROWS < rows_live:
                    self.release_flag(
                        self.ctx.params(self.ctx.first + group, s), s, role_slot
                    )

    def proj_position(self, s, group, row):
        if self.PACKED:
            return Int32(s.row_words[PackedRows.BASE + row])
        return Int32(s.epoch_slot[ProjectionChunks.POSITION + group])

    def proj_stores(self, s, row):
        if self.PACKED:
            return Int32(row < s.row_words[PackedRows.ROWS])
        return None

    def loop_position(self, s, h, own):
        if self.PACKED:
            return self.row_info(None, s, h, own)[0]
        return Int32(s.epoch_slot[ProjectionChunks.POSITION + h])

    def row_stores(self, p, s, row0, query_):
        if self.PACKED:
            return Int32(row0 + query_ < s.row_words[PackedRows.ROWS])
        return None

    @stage
    @cute.jit
    def proj_store(self, p, s, y, row, column, role_slot, tag_0, cur_pos_c, store=None):
        tagged_word = rt.Convert.bf16_low(y) | tag_0 << 16
        if self.is_router(role_slot):
            rt.Memory.st_relaxed_sys(
                rt.ga(p.Flags, 14716 + row * 256 + (role_slot - 132) * 32 + column),
                tagged_word,
            )
        elif role_slot < 80:
            q_word = Handoff.Q_DOWN + row * Handoff.Q_DOWN_ROW + role_slot * 32 + column
            kv_word = (
                Handoff.KV_LATENT
                + row * Handoff.KV_LATENT_ROW
                + (role_slot - 64) * 32
                + column
            )
            rt.Memory.st_relaxed_sys(
                self.mailbox(
                    p,
                    s,
                    cutlass.select_(role_slot < 64, q_word, kv_word),
                    Int32(role_slot >= 64),
                ),
                tagged_word,
            )
            if const_expr(self.shared_or_seam):  # noqa: SIM102
                if role_slot < 64:
                    rt.Memory.st_relaxed_sys(self.mailbox(p, s, q_word, 1), tagged_word)
        else:
            dim = (role_slot - 80) * 32 + column
            rotated = self.rope(p, s, rt.Convert.bf16_round(y), dim, row)
            if const_expr(store is None):
                KvRows.store_rope_e4m3(
                    KvRows.rope_byte(p.PECache, p.BlockTable, cur_pos_c + row, dim),
                    rotated,
                )
            else:
                if store != 0:
                    KvRows.store_rope_e4m3(
                        KvRows.rope_byte(p.PECache, p.BlockTable, cur_pos_c + row, dim),
                        rotated,
                    )

    @stage
    @cute.jit
    def proj_inverse(self, s, p0, role_slot, rows_live):
        row = role_slot - 82 - 8 * Int32(role_slot >= 140)
        if row < rows_live:
            raw = [
                rt.Memory.ld_cg_v4(rt.ga(p0.X, row * 6144 + 2048 * c + s.tid * 8))
                for c in range(3)
            ]
            values = (
                rt.bf16x8_to_f32(raw[0])
                + rt.bf16x8_to_f32(raw[1])
                + rt.bf16x8_to_f32(raw[2])
            )
            total = rt.Reduce.warp_sum_down(self.pair_squares(values))
            if s.lane == 0:
                s.scratch[s.warp] = total
            rt.bar_sync(*BAR_COMPUTE)
            inverse = rt.Reduce.rms_inverse(
                rt.Reduce.ordered_sum(lambda k: s.scratch[k], 8)
            )
            if s.tid == 0:
                group = rt.cdiv(row, self.GROUP_ROWS)
                pg = self.ctx.params(self.ctx.first + group, s)
                epoch = Uint32(s.epoch_slot[PassPhases.EPOCHS + group])
                rt.Memory.st_volatile_u64(
                    self.handoff(
                        pg,
                        ProjectionChunks.RECORD_WORD
                        + 2 * rt.cmod(row, self.GROUP_ROWS),
                    ),
                    (Uint64(epoch + 1) << 32) | Uint64(rt.Convert.f32_bits(inverse)),
                    memory=True,
                )

    @stage
    @cute.jit
    def load_proj(self, s):
        self.set_regs("load")
        rid = self.role_slot(s)
        p0 = self.ctx.params(self.ctx.first)
        nb = self.PROJ_ROWS
        w0, x0, w1, x1 = ProjectionChunks.offsets(nb)
        rows_live = self.live_rows(p0)
        router = self.proj_router(p0, rid)
        gemm = (rid < INPUT_CTAS) | (router != 0)
        total = rt.cdiv(rows_live + nb - 1, nb) * ProjectionChunks.PAGES
        tx = ProjectionChunks.W_BYTES + nb * ProjectionChunks.X_SLICE
        ring = rt.Pipeline.Ring(s.ring_full, s.ring_empty, 2)
        x_rows = p0.XG if self.seam else p0.X
        policy = rt.AsyncCopy.createpolicy_evict_first(
            self.proj_router_source(p0, 0), 1579008, 1579008
        )
        if gemm:  # noqa: SIM102  nested bare-bool ifs: the DSL traces them
            if s.warp == 8:  # noqa: SIM102  nested bare-bool ifs: the DSL traces them
                if s.lane == 0:
                    rt.AsyncCopy.prefetch_tensormap(p0.InputTmap)
                    self.proj_scales(p0, s, rid, router)
                    for seq in cutlass.range_constexpr(2):
                        rt.mbar_expect_tx(ring.full(seq), tx)
                        self.proj_weights(
                            p0, s, rid, router, seq, seq, w0 if seq == 0 else w1, policy
                        )
        cute.arch.griddepcontrol_wait()
        if s.warp == 9:  # noqa: SIM102  nested bare-bool ifs: the DSL traces them
            if s.lane < self.PASS_GROUPS:
                group = self.ctx.first + s.lane
                if rows_live > group * self.GROUP_ROWS:
                    ticket = rt.Memory.atom_add_u32(rt.ga(p0.FullEpoch, s.lane * 4), 1)
                    s.epoch_slot[PassPhases.EPOCHS + s.lane] = rt.cdiv(ticket, 148)
        rt.bar_sync(*BAR_ALL)
        if gemm:  # noqa: SIM102  nested bare-bool ifs: the DSL traces them
            if s.warp == 8:
                for seq in cutlass.range_constexpr(2):
                    self.proj_x_rows(
                        s, ring, x_rows, nb, seq, seq, x0 if seq == 0 else x1
                    )
                for seq in cutlass.range(2, total, 1, unroll=1):
                    chunk = rt.cmod(seq, 2)
                    rt.mbar_wait(ring.empty(chunk), ring.parity(seq) ^ 1)
                    if s.lane == 0:
                        rt.mbar_expect_tx(ring.full(chunk), tx)
                        self.proj_weights(
                            p0,
                            s,
                            rid,
                            router,
                            rt.cmod(seq, ProjectionChunks.PAGES),
                            chunk,
                            cutlass.select_(chunk == 0, Int32(w0), Int32(w1)),
                            policy,
                        )
                    self.proj_x_rows(
                        s,
                        ring,
                        x_rows,
                        nb,
                        seq,
                        chunk,
                        cutlass.select_(chunk == 0, Int32(x0), Int32(x1)),
                    )

    @cute.jit
    def proj_scales(self, p, s, rid, router):
        if router != 0:
            rt.mbar_expect_tx(s.tail_full, ProjectionChunks.PAGES * 128)
            for page in cutlass.range_constexpr(ProjectionChunks.PAGES):
                rt.bulk_g2s(
                    s.received_words_addr + page * 128,
                    self.proj_router_source(
                        p,
                        ((rid - 132) * 6 + page) * Ring.WEIGHT_PAGE_BYTES
                        + ProjectionChunks.W_BYTES,
                    ),
                    128,
                    s.tail_full,
                )
        else:
            rt.mbar_expect_tx(s.tail_full, 192)
            rt.bulk_g2s(
                s.received_words_addr,
                rt.baddr(p.InputScales, rt.cdiv(rid, 4) * 48 * 4),
                192,
                s.tail_full,
            )

    @cute.jit
    def proj_weights(self, p, s, rid, router, page, chunk, w_off, policy):
        ring = rt.Pipeline.Ring(s.ring_full, s.ring_empty, 2)
        dst = s.ring_addr + w_off
        if router != 0:
            rt.bulk_g2s_hint(
                dst,
                self.proj_router_source(
                    p, ((rid - 132) * 6 + page) * Ring.WEIGHT_PAGE_BYTES
                ),
                ProjectionChunks.W_BYTES,
                ring.full(chunk),
                policy,
            )
        else:
            rt.AsyncCopy.tma_3d(
                dst, p.InputTmap, 0, rid * 32, page * 8, ring.full(chunk), L2_HINT
            )

    @cute.jit
    def proj_x_rows(self, s, ring, x_rows, nb, seq, chunk, x_off):
        if s.lane < nb:
            block = rt.cdiv(seq, ProjectionChunks.PAGES)
            page = rt.cmod(seq, ProjectionChunks.PAGES)
            rt.bulk_g2s(
                s.ring_addr + x_off + s.lane * ProjectionChunks.X_PITCH,
                rt.baddr(
                    x_rows,
                    (block * nb + s.lane) * ProjectionChunks.X_ROW_BYTES
                    + page * ProjectionChunks.X_SLICE,
                ),
                ProjectionChunks.X_SLICE,
                ring.full(chunk),
            )

    @stage
    @cute.jit
    def input_norm_q_down(self, p, s, cur_pos_c, epoch, role_slot, router_on):
        if const_expr(not self.pushed):
            router = router_on
            if const_expr(self.shared_or_seam):
                router = Int32(self.is_router(role_slot))
            if (role_slot < 82) | (router != 0):
                if const_expr(self.seam):
                    raw = [
                        rt.Memory.ld_cg_v4(rt.ga(p.X, 2048 * c + s.tid * 8))
                        for c in range(12)
                    ]
                    rt.mbar_wait(s.x_full, 0)
                    cute.arch.fence_view_async_shared()
                    seam_pre = [
                        self.pair_squares(
                            rt.bf16x8_to_f32(raw[3 * r])
                            + rt.bf16x8_to_f32(raw[3 * r + 1])
                            + rt.bf16x8_to_f32(raw[3 * r + 2])
                        )
                        for r in range(4)
                    ]
                else:
                    late_inverse = rt.Reduce.rms_inverse(self.x_norm(p, s))
                rt.bar_sync(*BAR_ALL)
                epoch_0 = s.epoch_slot[0]
                tag_0 = ((epoch_0 + 1) & 32767) | 32768
                accum = rt.rmem_tensor(12, Float32)
                for m0 in cutlass.range_constexpr(8):
                    accum[m0] = Float32(0.0)
                for page in cutlass.range(0, 6, 1, unroll=1):
                    if const_expr(self.seam):  # noqa: SIM102
                        if page == 4:
                            self.stage_rows(s, seam_pre, 2016)
                    s.pages.wait_full(page)
                    cute.arch.fence_view_async_shared()
                    vin_region = Ring.slot_offset(page)
                    vin_g = rt.cdiv(s.lane, 4)
                    vin_q = rt.cmod(s.lane, 4)
                    vin_plane = rt.cdiv(vin_region + s.warp * 4096 + vin_g * 128, 4)
                    vin_rows = [None] * 16
                    for j in cutlass.range_constexpr(4):
                        for h in cutlass.range_constexpr(2):
                            w = rt.Memory.lds_b128(
                                s.ring_addr
                                + (vin_plane + ((2 * vin_q + h) ^ vin_g) * 4 + 256 * j)
                                * 4
                            )
                            vin_rows[4 * j + 2 * h] = Uint64(w[0]) | Uint64(w[1]) << 32
                            vin_rows[4 * j + 2 * h + 1] = (
                                Uint64(w[2]) | Uint64(w[3]) << 32
                            )
                    vin_k = (
                        rt.cmod(rt.cdiv(s.lane, 4), 4) * 3076
                        + rt.cmod(s.lane, 4) * 16
                        + rt.cdiv(page * 1024 + s.warp * 128, 2)
                    )
                    vin_acc = [None] * 8
                    for j in cutlass.range_constexpr(4):
                        vin_a = [None] * 16
                        for h in cutlass.range_constexpr(2):
                            even = rt.Convert.unpack_u16x4(vin_rows[j + 8 * h])
                            odd = rt.Convert.unpack_u16x4(vin_rows[j + 4 + 8 * h])
                            for i in cutlass.range_constexpr(4):
                                vin_a[8 * h + 2 * i] = rt.Convert.e4m3x2_to_bf16x2(
                                    even[i]
                                )
                                vin_a[8 * h + 2 * i + 1] = rt.Convert.e4m3x2_to_bf16x2(
                                    odd[i]
                                )
                        for t in cutlass.range_constexpr(2):
                            vin_b = rt.Memory.lds_weak_b64(
                                s.act_addr + (vin_k + 4 * j + 2 * t) * 4
                            )
                            for h in cutlass.range_constexpr(2):
                                if const_expr(j == 0 and t == 0):
                                    vin_acc[4 * h : 4 * h + 4] = rt.TensorCore.mma(
                                        None,
                                        tuple(vin_a[8 * h : 8 * h + 4]),
                                        tuple(vin_b),
                                        True,
                                    )
                                else:
                                    rt.TensorCore.mma_acc_at(
                                        vin_acc,
                                        4 * h,
                                        tuple(vin_a[8 * h + 4 * t : 8 * h + 4 * t + 4]),
                                        tuple(vin_b),
                                    )
                    vin_scale_word = 8448 + page * 8 + s.warp
                    if router != 0:
                        vin_scale_word = rt.cdiv(vin_region + 32768, 4) + s.warp
                    vin_scale = s.ring_f32[vin_scale_word]
                    for i in cutlass.range_constexpr(8):
                        accum[i] = rt.fma(vin_acc[i], vin_scale, accum[i])
                    s.pages.release(rt.cmod(page, 4))
                self.rows_to_scratch(s, accum, 4, 2)
                if s.tid < 128:
                    value = self.warp_sum(s, s.tid, 4, 2)
                    if const_expr(self.seam):
                        x_row = self.min_i32(rt.cdiv(s.tid, 32), 3)
                        row_inverse = rt.Reduce.rms_inverse(
                            rt.Reduce.ordered_sum(
                                lambda k: s.scratch[2016 + x_row * 8 + k], 8
                            )
                        )
                    else:
                        row_inverse = late_inverse
                    y = rt.mul(value, row_inverse)
                    row = rt.cdiv(s.tid, 32)
                    column = rt.cmod(s.tid, 32)
                    tagged_word = rt.Convert.bf16_low(y) | tag_0 << 16
                    if self.is_router(role_slot):
                        rt.Memory.st_relaxed_sys(
                            rt.ga(
                                p.Flags,
                                14716 + row * 256 + (role_slot - 132) * 32 + column,
                            ),
                            tagged_word,
                        )
                    elif role_slot < 80:
                        q_word = (
                            Handoff.Q_DOWN
                            + row * Handoff.Q_DOWN_ROW
                            + role_slot * 32
                            + column
                        )
                        kv_word = (
                            Handoff.KV_LATENT
                            + row * Handoff.KV_LATENT_ROW
                            + (role_slot - 64) * 32
                            + column
                        )
                        rt.Memory.st_relaxed_sys(
                            self.mailbox(
                                p,
                                s,
                                cutlass.select_(role_slot < 64, q_word, kv_word),
                                Int32(role_slot >= 64),
                            ),
                            tagged_word,
                        )
                        if const_expr(self.shared_or_seam):  # noqa: SIM102
                            if role_slot < 64:
                                rt.Memory.st_relaxed_sys(
                                    self.mailbox(p, s, q_word, 1), tagged_word
                                )
                    else:
                        dim = (role_slot - 80) * 32 + column
                        rotated = self.rope(p, s, rt.Convert.bf16_round(y), dim)
                        KvRows.store_rope_e4m3(
                            KvRows.rope_byte(
                                p.PECache, p.BlockTable, cur_pos_c + row, dim
                            ),
                            rotated,
                        )
                rt.bar_sync(*BAR_COMPUTE)
                if const_expr(self.push):
                    target = p.peer_rank - 3
                    if (s.bid >= 80) & (s.bid < 82) & (target >= 0) & (target < 3):
                        rt.bar_sync(*BAR_COMPUTE)
                        if s.tid < 64:
                            token = rt.cdiv(s.tid, 16)
                            word = (s.bid - 80) * 16 + rt.cmod(s.tid, 16)
                            pe_word = KvRows.rope_pair_bf16(
                                KvRows.rope_byte(
                                    p.PECache, p.BlockTable, cur_pos_c + token, word * 2
                                )
                            )
                            record = PushedRows.record(
                                PushedRows.TARGET,
                                Int32(rt.cmod(epoch_0 + 1, 2)),
                                token,
                                PushedRows.PE_RECORDS + word,
                            )
                            rt.Memory.st_volatile_u64(
                                self.peer_base_by_select(target, p.peer_bases)
                                + Int64(record),
                                Uint64(epoch_0 + 1) << 32 | Uint64(pe_word),
                            )
                epoch = epoch_0
                if (role_slot >= 80) & (role_slot < 82):
                    rt.bar_sync(*BAR_COMPUTE)
                    self.release_flag(p, s, role_slot)
                if const_expr(self.shared_or_seam):  # noqa: SIM102
                    if role_slot < 82:
                        rt.mbar_arrive(s.norm_free)
            else:
                rt.bar_sync(*BAR_ALL)
                epoch = s.epoch_slot[0]
        if const_expr(self.pushed):
            rt.bar_sync(*BAR_ALL)
            epoch = s.epoch_slot[0]
        tag = ((epoch + 1) & 32767) | 32768
        return epoch, tag

    @cute.jit
    def q_a_sequence(self, s, router_on):
        return 6 * Int32((s.bid < 82) | (router_on != 0))

    @cute.jit
    def scoring_sequence(self, s):
        return 6 * Int32(s.bid < 82) + 1

    @stage
    @cute.jit
    def full_qa_kv(self, p, s, cur_pos_c, epoch, tag):
        if (s.bid < 128) | (s.bid >= 132):
            router_on = Int32(0)
            if (s.bid >= 132) & (s.bid < 140):
                router_on = self.router_enabled(p)
            self.query_norm_qa(p, s, s.bid, tag, self.q_a_sequence(s, router_on))
            rt.mbar_arrive(s.norm_free)
            if const_expr(self.push):
                served = p.peer_rank - 3
                if (s.bid >= 140) & (s.bid < 148) & (served >= 0) & (served < 3):
                    operands = self.peer_base_by_select(served, p.peer_bases) + Int64(
                        self.PUSHED_OPERANDS
                    )
                    for extra in cutlass.range(18):
                        self.q_a_tile(
                            p,
                            s,
                            2 * (1 + extra),
                            (s.bid - 140) * 18 + extra,
                            tag,
                            0,
                            lambda word, pool: operands + Int64(word * 4),
                        )
        if (s.bid >= 128) & (s.bid < 132):
            query_ = s.bid - 128
            self.kv_latent_store(
                p,
                s,
                Handoff.KV_LATENT + query_ * Handoff.KV_LATENT_ROW,
                query_,
                cur_pos_c,
                tag,
                self.row_stores(p, s, self.row0, query_),
            )
            if const_expr(self.push):
                target = p.peer_rank - 3
                if (target >= 0) & (target < 3):
                    rt.bar_sync(*BAR_COMPUTE)
                    if s.tid < 128:
                        kv_word = rt.Memory.gview(
                            Uint32,
                            rt.ga(
                                p.KVCachePtr,
                                self.kv_row_off(p, cur_pos_c + query_, s.tid * 2),
                            ),
                        )[0]
                        record = PushedRows.record(
                            PushedRows.TARGET,
                            Int32(rt.cmod(Uint32(epoch + 1), 2)),
                            query_,
                            s.tid,
                        )
                        rt.Memory.st_volatile_u64(
                            self.peer_base_by_select(target, p.peer_bases)
                            + Int64(record),
                            (Uint64(Uint32(epoch + 1)) << 32) | Uint64(kv_word),
                            memory=True,
                        )
            rt.bar_sync(*BAR_COMPUTE)
            self.release_flag(p, s, FlagWords.KV_READY + s.bid)

    @stage
    @cute.jit
    def query_norm_qa(self, p, s, slot, tag, seq, resident: cutlass.Constexpr = False):
        gamma = rt.Convert.words_to_f32(
            rt.Memory.ld_global_v8(
                rt.ga(p.GammaQ, s.tid * 8), memory=self.shared_or_seam
            )
        )
        near = 0
        if const_expr(self.shared_or_seam):
            near = Int32(s.role_entry >> Handoff.POOL_SHIFT)
        chunks = [
            self.mailbox(p, s, Handoff.Q_DOWN + (s.tid + 256 * k) * 8, near)
            for k in range(4)
        ]

        def store(words):
            for k in cutlass.range_constexpr(4):
                rt.Memory.sts_v4(
                    s.act_addr + Uint32(k * 4112 + s.tid * 16),
                    [
                        rt.Convert.prmt(
                            words[2 * k + h][2 * i], words[2 * k + h][2 * i + 1], 0x5410
                        )
                        for h in range(2)
                        for i in range(2)
                    ],
                )

        self.poll_tagged16(
            chunks, tag << 16, store, Int32(slot >= 0), self.shared_or_seam
        )
        rows = [
            rt.bf16x8_to_f32(
                [s.act[rt.cdiv(q * 2056 + s.tid * 8, 2) + i] for i in range(4)]
            )
            for q in range(4)
        ]
        inverse = self.rms_inverses(s, rows)
        for q in cutlass.range_constexpr(4):
            rt.Memory.sts_v4(
                s.act_addr + Uint32((q * 2056 + s.tid * 8) * 2),
                self.norm_bf16x8(rows[q], gamma, inverse[q]),
            )
        rt.bar_sync(*BAR_COMPUTE)
        self.q_a_tile(
            p,
            s,
            seq,
            slot - 4 * Int32(slot >= 132),
            tag,
            Handoff.INDEXER_Q,
            lambda word, pool: self.mailbox(p, s, word, pool),
            resident,
        )

    @stage
    @cute.jit
    def q_a_tile(
        self,
        p,
        s,
        seq,
        tile,
        tag,
        q_base: cutlass.Constexpr,
        mailbox,
        resident: cutlass.Constexpr = False,
    ):
        accum = [Float32(0.0)] * 8
        for page in cutlass.range_constexpr(2):
            s.pages.wait_full(seq + page)
            rt.Memory.fence_proxy_async_shared()
            self.q_a_page_mma(s, accum, Ring.slot_offset(seq + page), page)
            if const_expr(not resident):
                s.pages.release(rt.cmod(seq + page, 4))
        self.rows_to_scratch(s, accum, 4, 2)
        if s.tid < 128:
            value = self.warp_sum(s, s.tid, 4, 2)
            row = rt.cdiv(s.tid, 32)
            if tile < 128:
                word = (
                    q_base
                    + row * Handoff.INDEXER_Q_ROW
                    + tile * 32
                    + rt.cmod(s.tid, 32)
                )
                rt.Memory.st_relaxed_sys(
                    mailbox(word, rt.cdiv(row, 2)),
                    rt.Convert.bf16_low(value) | tag << 16,
                )
            else:
                dim = (tile - 128) * 32 + rt.cmod(s.tid, 32)
                rotated = self.rope(p, s, rt.Convert.bf16_round(value), dim)
                rt.Memory.st_relaxed_sys(
                    mailbox(
                        Handoff.INDEXER_PE + row * Handoff.INDEXER_PE_ROW + dim,
                        rt.cdiv(row, 2),
                    ),
                    rt.Convert.bf16_low(rotated) | tag << 16,
                )
        rt.bar_sync(*BAR_COMPUTE)

    OPERAND_Q = Handoff.INDEXER_Q

    PUSHED_OPERANDS = 936064

    def split_done(self, p, s, slot):
        return rt.ga(p.Flags, FlagWords.SPLIT_DONE + slot)

    @cute.jit
    def poll_operands(self, p, s, tag, query_):
        firsts = []
        for k in cutlass.range_constexpr(3):
            chunk = s.tid + 256 * k
            if const_expr(k > 0):
                chunk = cutlass.select_(chunk >= IndexerOperands.CHUNKS, s.tid, chunk)
            pe_word = (
                Handoff.INDEXER_PE
                + query_ * Handoff.INDEXER_PE_ROW
                + (chunk - IndexerOperands.Q_CHUNKS) * 8
            )
            q_word = self.OPERAND_Q + query_ * Handoff.INDEXER_Q_ROW + chunk * 8
            firsts.append(
                self.mailbox(
                    p,
                    s,
                    cutlass.select_(chunk >= IndexerOperands.Q_CHUNKS, pe_word, q_word),
                    rt.cdiv(query_, 2),
                )
            )

        def store(words):
            for k in cutlass.range_constexpr(3):
                IndexerOperands.store(
                    s.act_addr, s.tid + 256 * k, words[2 * k], words[2 * k + 1]
                )

        self.poll_tagged16(firsts, tag << 16, store)

    @stage
    @cute.jit
    def scoring_split(
        self,
        p,
        s,
        cur_pos_c,
        tag,
        slot,
        index_sequence,
        merge: cutlass.Constexpr = True,
    ):
        query_ = rt.cdiv(slot, 32)
        split = rt.cmod(slot, 32)
        score_count = cutlass.min(cur_pos_c + query_ + 1, 2048)
        if score_count > split * 64:
            self.poll_operands(p, s, tag, query_)
        rt.bar_sync(*BAR_ALL)
        split_count = rt.cdiv(score_count + 64 - 1, 64)
        if split < split_count:
            first_seq = index_sequence + 1
            self.km_split_attention(
                p,
                s,
                first_seq,
                index_sequence + 2,
                split,
                split_count,
                score_count,
                query_,
            )
        rt.bar_sync(*BAR_COMPUTE)
        rt.bar_sync(*BAR_COMPUTE)
        done = self.split_done(p, s, slot)
        with rt.warp_leader(s.warp, 0):
            rt.Memory.red_async_release_gpu_add(done, Uint32(1), memory=True)
        cute.arch.fence_view_async_shared()
        for d in cutlass.range_constexpr(1, 3):
            rt.mbar_arrive(s.ring_empty + rt.cmod(index_sequence + d, 4) * 8)
        if const_expr(merge):
            self.km_merge(p, s, cur_pos_c, slot)

    @stage
    @cute.jit
    def full_scores_vo(self, p, s, compute_start, cur_pos_c, epoch, tag):
        if s.bid < 128:
            self.scoring_split(
                p,
                s,
                cur_pos_c,
                tag,
                s.bid,
                self.scoring_sequence(s),
                not self.PHASED,
            )
            if const_expr(not self.PHASED):
                self.vo_output(p, s, s.bid, tag, epoch)
        if s.bid >= 132:  # noqa: SIM102
            if self.prefetch_group(p):
                self.moe_prefetch_tail(p, s, s.bid, compute_start)

    @stage
    @cute.jit
    def prefetch_moe_weights(
        self,
        p,
        s,
        mine,
        lines,
        start,
        items: cutlass.Constexpr,
        listed_first: cutlass.Constexpr,
    ):
        delay = Uint64(p.Flags[FlagWords.MOE_PACING])
        rate = Uint64(p.Flags[FlagWords.MOE_PACING + 1])
        deadline = Uint64(p.Flags[FlagWords.MOE_PACING + 2])
        weights = Uint64(p.Flags[FlagWords.MOE_PACING + 3]) | (
            Uint64(p.Flags[FlagWords.MOE_PACING + 4]) << 32
        )
        interval = rt.cdiv(Uint64(2097152000), rate)
        steps = rt.cdiv(lines + 16384 - 1, 16384)
        begin = rt.FastMath.max_typed(start + delay, rt.globaltimer())
        step = Uint32(0)
        stop = Int32(0)
        while (stop == 0) & (step < steps):
            due = begin + Uint64(step) * interval
            waited = Int32(0)
            while waited == 0:
                if rt.globaltimer() >= due:
                    waited = Int32(1)
            rt.syncwarp()
            if (deadline != 0) & (rt.globaltimer() - start >= deadline):
                stop = Int32(1)
            if stop == 0:
                for item in cutlass.range_constexpr(items):
                    line = step * 16384 + mine + Uint32(item)
                    if line < lines:
                        entry = s.moe_listed[
                            listed_first + rt.cdiv(line, LAYOUT.w13_expert_lines)
                        ]
                        line_addr = (
                            weights
                            + Uint64(Int64(entry)) * LAYOUT.w13_expert_bytes
                            + Uint64(rt.cmod(line, LAYOUT.w13_expert_lines) * 128)
                        )
                        cute.arch.prefetch(
                            cute.make_ptr(
                                Uint8, Int64(line_addr), cute.AddressSpace.gmem
                            ).to_llvm_ptr(),
                            cache_level="L2",
                        )
            step = step + 1
        rt.syncwarp()

    @stage
    @cute.jit
    def vo_output(self, p, s, slot, tag, epoch, shared_index=None):
        self.vo_publish(p, s, slot, tag)
        self.vo_project(p, s, slot, tag)
        if const_expr(self.shared_or_seam):
            epoch = (epoch + 1) * 57 + shared_index - 1
        self.tagged_exchange_final(p, s, p.peer_bases, epoch, slot)

    @cute.jit
    def vo_publish(self, p, s, slot, tag):
        head = rt.cdiv(rt.cmod(slot, 32), 4)
        rt.bar_sync(*BAR_COMPUTE)

        def o_vo(region, vo_lane, vo_k):
            return self.vo_tiles(s, region, vo_lane, vo_k, True)

        self.vo_token_partials(s, slot, o_vo)
        if s.tid < 64:
            token_value = s.scratch[s.tid]
            for warp_ in cutlass.range_constexpr(1, 8):
                token_value = rt.add(token_value, s.scratch[warp_ * 65 + s.tid])
            word = (
                rt.cdiv(slot, 32) * Handoff.VO_ROW
                + head * 256
                + rt.cmod(slot, 4) * 64
                + s.tid
            )
            low = rt.f32x2_to_bf16x2(token_value, Float32(0.0)) & 65535
            rt.Memory.st_relaxed_sys(
                self.handoff_word(p, Handoff.VO + word), Uint32(low | tag << 16)
            )
            if const_expr(self.shared_or_seam):
                rt.Memory.st_relaxed_sys(
                    self.handoff_word(p, Handoff.VO + word, 1), Uint32(low | tag << 16)
                )
        rt.bar_sync(*BAR_COMPUTE)

    @cute.jit
    def vo_project(self, p, s, slot, tag):
        tag_high = tag << 16
        o_pending = rt.rmem_tensor(8, Uint32)
        o_token = rt.cdiv(s.lane, 8)
        near = 0
        if const_expr(self.shared_or_seam):
            near = Int32(s.role_entry >> Handoff.POOL_SHIFT)

        def o_load(page):
            first = (
                Handoff.VO
                + o_token * Handoff.VO_ROW
                + (page * 512 + s.warp * 64 + rt.cmod(s.lane, 8) * 8)
            )
            w0 = rt.Memory.ld_volatile_v4(self.handoff_word(p, first, near))
            w1 = rt.Memory.ld_volatile_v4(self.handoff_word(p, first + 4, near))
            for k in cutlass.range_constexpr(4):
                o_pending[k] = w0[k]
                o_pending[4 + k] = w1[k]

        o_reload = o_load

        vo_tokens = rt.cmod(rt.cdiv(s.lane, 4), 4) * 1028 + rt.cmod(s.lane, 4) * 8
        vo_lane = rt.cdiv(s.lane, 4) * 64 + s.warp * 3072 + rt.cmod(s.lane, 4) * 16

        def page_mma(page, accum):
            vo_acc = self.vo_tiles(
                s,
                17408 + page * 24704,
                vo_lane,
                vo_tokens + rt.cdiv(page * 512 + s.warp * 64, 2),
                False,
            )
            for m in cutlass.range_constexpr(3):
                scale = s.ring_f32[4160 + m * 16 + page * 4 + rt.cdiv(s.warp, 2)]
                for item in cutlass.range_constexpr(4):
                    accum[m * 4 + item] = rt.fma(
                        vo_acc[m * 4 + item], scale, accum[m * 4 + item]
                    )

        accum = self.o_proj_pages(
            s, o_pending, o_token, tag_high, o_load, o_reload, page_mma, 4, 4112, 1
        )
        self.rows_to_scratch(s, accum, 4, 3)
        if s.tid < 192:
            value = self.warp_sum(s, s.tid, 4, 3)
            s.local_o[rt.cdiv(s.tid, 48) * 48 + rt.cmod(s.tid, 48)] = BFloat16(value)
        rt.bar_sync(*BAR_COMPUTE)

    @stage
    @cute.jit
    def moe_prefetch_tail(self, p, s, slot, start):
        count = Uint32(p.Flags[FlagWords.MOE_COUNT])
        source = Uint32(p.Flags[FlagWords.MOE_SOURCE])
        if source != 0:
            count = Uint32(cutlass.select_(count < 12, count, Uint32(12)))
        if count > 8:
            rt.mbar_wait(s.moe_list_full, 0)
            if count > 8:
                mine = (Uint32((slot - 132) * 256) + Uint32(s.tid)) * 4
                self.prefetch_moe_weights(
                    p, s, mine, (count - 8) * LAYOUT.w13_expert_lines, start, 4, 8
                )

    def vo_tiles(self, s, region, vo_lane, vo_k, first: bool):
        bases = (0, 1024, 116224, 117248) if first else (0, 1024, 2048)
        return self.vo_tile(s, region, vo_lane, bases, vo_k)

    @cute.jit
    def store_selection(self, p, slot, record):
        rt.Memory.st_volatile_u64(
            rt.ga(p.SelectionMailbox, slot * 2), record, memory=True
        )
        if const_expr(self.push):
            base = p.peer_bases[0]
            for k in cutlass.range_constexpr(1, 8):
                if p.peer_rank - 3 == k:
                    base = p.peer_bases[k]
            own = p.peer_bases[0]
            for k in cutlass.range_constexpr(1, 8):
                if p.peer_rank == k:
                    own = p.peer_bases[k]
            served_words = rt.FastMath.cdiv64(Int64(base - own), 4)
            rt.Memory.st_volatile_u64(
                rt.ga(p.SelectionMailbox, served_words + Int64(slot * 2)),
                record,
                memory=True,
            )

    @stage
    @cute.jit
    def refine_emit_plan(
        self,
        p,
        s,
        part,
        refine_above,
        refine_bid0: cutlass.Constexpr,
        refine_count,
        refine_dry,
        refine_generation,
        refine_live,
        refine_parity,
    ):
        stream_largest = Uint32(0)
        stream_inverted = Uint32(0)
        for stream_base in cutlass.range(0, refine_count, 2048):
            stream_pending = Uint32(0)
            for item in cutlass.range_constexpr(8):
                if refine_count > stream_base + item * 256 + s.tid:
                    stream_pending = stream_pending | Uint32(1 << item)
            _stream_done = Int32(0)
            while _stream_done == 0:
                stream_ids = []
                stream_keys = []
                for item in cutlass.range_constexpr(8):
                    _sid = Uint64(0)
                    _skey = Uint64(0)
                    if ((stream_pending >> Uint32(item) & 1) != 0) & (refine_dry == 0):
                        stream_word = (
                            32768
                            + (
                                (refine_parity * 4 + (s.bid - refine_bid0)) * 16240
                                + stream_base
                                + item * 256
                                + s.tid
                            )
                            * 4
                        )
                        _sv = rt.Memory.ld_volatile_v4(
                            rt.ga(p.SelectionMailbox, stream_word)
                        )
                        _sid = Uint64(_sv[1]) << 32 | Uint64(_sv[0])
                        _skey = Uint64(_sv[3]) << 32 | Uint64(_sv[2])
                    stream_ids.append(_sid)
                    stream_keys.append(_skey)
                for item in cutlass.range_constexpr(8):
                    if (stream_pending >> Uint32(item) & 1) != 0:
                        stream_ready = refine_dry | Int32(
                            (Uint32(stream_ids[item] >> 32) == refine_generation)
                            & (Uint32(stream_keys[item] >> 32) == refine_generation)
                        )
                        if stream_ready != 0:
                            stream_slot = stream_base + item * 256 + s.tid
                            stream_key = Uint32(stream_keys[item])
                            stream_id = Uint32(stream_ids[item])
                            if refine_dry != 0:
                                stream_key = (
                                    Uint32(2147483648)
                                    | (Uint32(stream_slot) & 63) << 10
                                )
                                stream_id = Uint32(stream_slot)
                            rt.Memory.st_shared_v2(
                                Uint32(s.refine_cache_addr) + Uint32(stream_slot * 8),
                                stream_id,
                                stream_key,
                            )
                            rt.Memory.atom_add_shared(
                                s.refine_hist_addr + Int32(stream_key >> 10 & 1023) * 4,
                                1,
                            )
                            _smax_0 = stream_key
                            if stream_largest > stream_key:
                                _smax_0 = stream_largest
                            stream_largest = _smax_0
                            _smax_1 = stream_key ^ Uint32(4294967295)
                            if stream_inverted > (stream_key ^ Uint32(4294967295)):
                                _smax_1 = stream_inverted
                            stream_inverted = _smax_1
                            stream_pending = stream_pending & Uint32(
                                (4294967295 ^ 1 << item) & 0xFFFFFFFF
                            )
                _vote_0 = rt.Sync.vote_all(stream_pending == 0)
                if _vote_0 != 0:
                    _stream_done = Int32(1)
            rt.syncwarp()
        stream_largest = rt.Sync.redux_max_u32(stream_largest)
        stream_inverted = rt.Sync.redux_max_u32(stream_inverted)
        if s.lane == 0:
            rt.Memory.atom_max_shared(s.refine_in_addr + 2 * 4, stream_largest)
            rt.Memory.atom_max_shared(s.refine_in_addr + 3 * 4, stream_inverted)
        rt.bar_sync(*BAR_COMPUTE)
        refine_largest = Uint32(s.refine_in[2])
        refine_smallest = Uint32(s.refine_in[3]) ^ Uint32(4294967295)
        threshold = refine_largest
        remaining_need = 2048 - refine_above
        ties = refine_count
        if refine_largest != refine_smallest:
            threshold, remaining_need, ties = rt.Selection.threshold(
                s.refine_cache,
                s.refine_hist,
                s.refine_hist_addr,
                s.refine_control,
                s.refine_warps,
                refine_count,
                refine_largest,
                remaining_need,
                ties,
                s.tid,
                s.warp,
                s.lane,
            )
        limit = Uint32(4294967295)
        if remaining_need < ties:
            tie_blocks = rt.cdiv(refine_live + 128 - 1, 128)
            if refine_dry != 0:
                tie_blocks = Int32(2)
            limit = rt.Selection.tie_limit(
                s.refine_cache,
                s.refine_ties,
                s.refine_ties_addr,
                s.refine_tie_bits,
                s.refine_tie_bits_addr,
                s.refine_control,
                s.refine_warps,
                refine_count,
                tie_blocks,
                threshold,
                remaining_need,
                s.tid,
                s.warp,
                s.lane,
            )
        emit_span = rt.cdiv(refine_count + 256 - 1, 256) * 32
        emit_begin = refine_count
        if (s.warp * emit_span) < refine_count:
            emit_begin = s.warp * emit_span
        emit_end = refine_count
        if (emit_begin + emit_span) < refine_count:
            emit_end = emit_begin + emit_span
        emit_below = (Uint32(1) << Uint32(s.lane)) - 1
        emit_chosen = Int32(0)
        fused_ids = rt.rmem_tensor(8, Uint32)
        fused_ballots = rt.rmem_tensor(8, Uint32)
        fused_one = Int32(emit_span <= 256)
        for item in cutlass.range_constexpr(8):
            fused_ids[item] = Uint32(0)
            fused_ballots[item] = Uint32(0)
        for count_batch in cutlass.range(emit_begin, emit_end, 256, unroll=1):
            count_keys = []
            count_ids = []
            for item in cutlass.range_constexpr(8):
                _min_2 = emit_end - 1
                if (count_batch + item * 32 + s.lane) < (emit_end - 1):
                    _min_2 = count_batch + item * 32 + s.lane
                count_ids.append(Uint32(s.refine_cache[(_min_2 * 2) + 0]))
                count_keys.append(Uint32(s.refine_cache[(_min_2 * 2) + 1]))
            for item in cutlass.range_constexpr(8):
                count_take = Int32(emit_end > count_batch + item * 32 + s.lane) & (
                    Int32(threshold < count_keys[item])
                    | Int32(
                        (count_keys[item] == threshold) & (limit >= count_ids[item])
                    )
                )
                _ballot_0 = rt.Sync.vote_ballot(count_take != 0)
                _popc_0 = rt.Sync.popc(_ballot_0)
                emit_chosen = emit_chosen + _popc_0
                fused_ids[item] = count_ids[item]
                fused_ballots[item] = _ballot_0
        if s.lane == 0:
            s.refine_warps[s.warp] = emit_chosen
        rt.bar_sync(*BAR_COMPUTE)
        emit_slot = refine_above
        for warp_ in cutlass.range_constexpr(8):
            emit_count = s.refine_warps[warp_]
            if s.warp > warp_:
                emit_slot = emit_slot + emit_count
        emit_first = Int32(
            (rt.cmod(refine_generation, 2) * 4 + Uint32(s.bid - refine_bid0)) * 2048
        )
        emit_last = emit_end
        if fused_one != 0:
            emit_last = emit_begin
            for item in cutlass.range_constexpr(8):
                fused_take = fused_ballots[item] >> Uint32(s.lane) & 1
                if (fused_take != 0) & (refine_dry == 0):
                    fused_record = Uint64(refine_generation) << 32 | Uint64(
                        fused_ids[item]
                    )
                    _popc_0 = rt.Sync.popc(fused_ballots[item] & emit_below)
                    self.store_selection(
                        p, emit_first + emit_slot + _popc_0, fused_record
                    )
                _popc_0 = rt.Sync.popc(fused_ballots[item])
                emit_slot = emit_slot + _popc_0
        return (
            emit_begin,
            emit_below,
            emit_end,
            emit_first,
            emit_last,
            emit_slot,
            limit,
            threshold,
        )

    @stage
    @cute.jit
    def pushed_compute(self, p, s, cur_pos_c, epoch, tag):
        if const_expr(self.pushed):
            vb = (
                s.bid
                - 8
                + Int32(s.bid < 8) * (136 + 12 * Int32(s.bid >= 4))
                + Int32(s.bid >= 136) * 4
            )
            generation = Uint32(epoch + 1)
            if (vb >= 128) & (vb < 132):
                token = vb - 128
                parity = Int32(rt.cmod(generation, 2))
                if s.tid < 128:
                    kv_address = self.peer_base_by_select(
                        p.peer_rank, p.peer_bases
                    ) + Int64(
                        PushedRows.record(PushedRows.TARGET, parity, token, s.tid)
                    )
                    kv_word = self.poll_pushed_record(kv_address, generation)
                    kv_at = self.kv_row_off(p, cur_pos_c + token, s.tid * 2)
                    rt.Memory.gview(Uint32, rt.ga(p.KVCachePtr, kv_at))[0] = kv_word
                if token < 2:  # noqa: SIM102
                    if (s.tid >= 128) & (s.tid < 192):
                        pe_token = rt.cdiv(s.tid - 128, 16)
                        pe_column = token * 16 + rt.cmod(s.tid - 128, 16)
                        pe_record = PushedRows.record(
                            PushedRows.TARGET,
                            parity,
                            pe_token,
                            PushedRows.PE_RECORDS + pe_column,
                        )
                        pe_word = self.poll_pushed_record(
                            self.peer_base_by_select(p.peer_rank, p.peer_bases)
                            + Int64(pe_record),
                            generation,
                        )
                        KvRows.store_rope_pair(
                            KvRows.rope_byte(
                                p.PECache,
                                p.BlockTable,
                                cur_pos_c + pe_token,
                                pe_column * 2,
                            ),
                            pe_word,
                        )
                rt.bar_sync(*BAR_COMPUTE)
                self.release_flag(p, s, FlagWords.KV_READY + vb)
                if token < 2:
                    rt.bar_sync(*BAR_COMPUTE)
                    self.release_flag(p, s, FlagWords.PE_READY + token)
            if (vb >= 128) & (vb < 148) & (s.warp == 0):
                bulk_up = (
                    Uint64(p.Flags[FlagWords.MOE_PACING + 3])
                    | Uint64(p.Flags[FlagWords.MOE_PACING + 4]) << 32
                )
                if bulk_up != 0:
                    issuer = vb - 128
                    own_base = self.peer_base_by_select(p.peer_rank, p.peer_bases)
                    expected = Uint32(128) | (generation & 127)
                    expert_word = Uint32(0)
                    ready = Uint32(0)
                    wide_start = rt.globaltimer()
                    wide_done = Int32(0)
                    while wide_done == 0:
                        if s.lane < 12:
                            expert_word = rt.Memory.ld_relaxed_gpu(
                                own_base + Int64(PushedRows.DRAFT) + Int64(s.lane) * 4
                            )
                        ready = Uint32(
                            rt.Sync.vote_all(
                                (s.lane >= 12) | ((expert_word >> 24) == expected)
                            )
                        )
                        if ready != 0:
                            wide_done = Int32(1)
                        if wide_done == 0:  # noqa: SIM102
                            if rt.globaltimer() - wide_start > 50000:
                                wide_done = Int32(1)
                    rt.syncwarp()
                    if ready != 0:
                        wide_page = issuer + s.lane * 20
                        expert = cute.arch.shuffle_sync(
                            expert_word, rt.cmod(wide_page, 12)
                        )
                        if wide_page < 384:
                            chunk = rt.cdiv(wide_page, 12)
                            rt.AsyncCopy.bulk_prefetch_l2(
                                bulk_up
                                + Uint64(expert & 65535) * LAYOUT.w13_expert_bytes
                                + Uint64(chunk * 32768),
                                Uint32(32768),
                            )
            if vb < 128:
                self.scoring_split(p, s, cur_pos_c, tag, vb, 0, not self.PHASED)
                if const_expr(not self.PHASED):
                    self.vo_output(p, s, vb, tag, epoch)

    @cute.jit
    def poll_pushed_record(self, address, generation) -> Uint32:
        record = rt.Poll.word(address, generation, False, True)
        rt.syncwarp()
        return Uint32(record)

    @stage
    @cute.jit
    def shared_query_attention(self, p, s, cur_pos_c, tag):
        role_lo = Int32(s.role_entry & Handoff.ROLE_MASK)
        seq6 = 6 * Int32((role_lo < 82) | ((role_lo >= 132) & (role_lo < 140)))
        if (role_lo < 128) | (role_lo >= 132):
            self.query_norm_qa(p, s, role_lo, tag, seq6)
            if role_lo >= 82:
                rt.mbar_arrive(s.norm_free)
        if (role_lo >= 128) & (role_lo < 132):
            query_ = role_lo - 128
            self.kv_latent_store(
                p,
                s,
                Handoff.KV_LATENT + query_ * Handoff.KV_LATENT_ROW,
                query_,
                cur_pos_c,
                tag,
                self.row_stores(p, s, self.row0, query_),
            )
            rt.bar_sync(*BAR_COMPUTE)
            self.release_flag(p, s, FlagWords.KV_READY + role_lo)
        if role_lo < 128:
            self.scoring_split(p, s, cur_pos_c, tag, role_lo, seq6 + 1, not self.PHASED)

    @stage
    @cute.jit
    def shared_vo_exchange(self, p, s, compute_start, epoch, shared_index_compute, tag):
        slot = Int32(s.role_entry & Handoff.ROLE_MASK)
        if const_expr(not self.PHASED):  # noqa: SIM102
            if slot < 128:
                self.vo_output(p, s, slot, tag, epoch, shared_index_compute)
        if slot >= 132:  # noqa: SIM102
            if self.prefetch_group(p):
                self.moe_prefetch_tail(p, s, slot, compute_start)

    @cute.jit
    def load_shared_ids(self, p, s, rid, epoch, position, token=None):
        local_key = s.tid - 320
        query_ = rt.cdiv(rid, 32)
        split = rt.cmod(rid, 32)
        generation = epoch + 1
        ids_token = query_
        if const_expr(self.PACKED):
            generation = self.step_generation(p)
            ids_token = token
        record_index = Int32(
            (rt.cmod(generation, 2) * 4 + Uint32(ids_token)) * 2048
            + Uint32(split * 64)
            + Uint32(local_key)
        )
        word = rt.Memory.ld_volatile_u64(rt.ga(p.SelectionMailbox, record_index * 2))
        context = position + query_ + 1
        index = Int32(4194304)
        if rt.FastMath.min_typed(context, 2048) > split * 64 + local_key:
            index = split * 64 + local_key
            if context > 2048:
                index = Int32(word)
                if Uint32(word >> 32) != generation:
                    record = rt.ga(p.SelectionMailbox, record_index * 2)
                    index = Int32(rt.Poll.word(record, generation, False))
        appended = self.store_key(p, s, local_key, index, position, 4)
        warp_hit = rt.Sync.vote_any(appended != 0)
        if s.lane == 0:
            s.warp_hits[s.warp - 10] = Int32(warp_hit != 0)
        if s.lane == 0:
            s.id_probe[s.warp - 10] = Int32(1)

    @stage
    @cute.jit
    def loader_setup(self, p, s):
        absorb_policy = None
        loader_start = Uint64(0)
        self.set_regs("load")
        with rt.warp_leader(s.warp, 8):
            rt.AsyncCopy.prefetch_tensormap(p.InputTmap)
            rt.AsyncCopy.prefetch_tensormap(p.OTmap)
            rt.AsyncCopy.prefetch_tensormap(p.KvbTmap)
        shared_index_load = Uint32(0)
        if const_expr(self.shared_or_seam):
            shared_index_load = p.Flags[FlagWords.SHARED_INDEX]
        if const_expr(not self.pushed and not self.seam):
            gtimer_0 = rt.globaltimer()
            loader_start = gtimer_0
            cute.arch.griddepcontrol_wait()
        input_policy = rt.AsyncCopy.createpolicy_evict_first(
            p.InputWeight.iterator.toint(), 16121856, 16121856
        )
        if const_expr(self.push):
            absorb_policy = rt.AsyncCopy.createpolicy_evict_first(
                p.AbsorbWeight.iterator.toint(), 18948096, 18948096
            )
        else:
            absorb_policy = rt.AsyncCopy.createpolicy_evict_first(
                p.AbsorbWeight.iterator.toint(), 9474048, 9474048
            )
        l2_range_policy_1 = rt.AsyncCopy.createpolicy_evict_first(
            p.KVbWeight.iterator.toint(), 1835008, 1835008
        )
        l2_range_policy_2 = rt.AsyncCopy.createpolicy_evict_first(
            p.OWeight.iterator.toint(), 12582912, 12582912
        )
        rid = s.bid
        if const_expr(self.shared_or_seam):
            rid = Int32(s.role_entry & Handoff.ROLE_MASK)
        if const_expr(self.pushed):
            rid = (
                s.bid
                - 8
                + Int32(s.bid < 8) * (136 + 12 * Int32(s.bid >= 4))
                + Int32(s.bid >= 136) * 4
            )
        router_on = Int32(0)
        early_position = Int32(0)
        if const_expr(not self.pushed):
            if rid < 82:
                for page in cutlass.range_constexpr(1, 4):
                    self.issue_input_page(p, s, page, rid)
            if const_expr(self.full_or_push):
                if (rid >= 132) & (rid < 140):
                    router_on = self.router_enabled(p)
                if router_on != 0:
                    self.issue_query_pages(p, s, 1, 4, True, rid)
            if const_expr(self.shared_or_seam):  # noqa: SIM102
                if (rid >= 132) & (rid < 140):
                    self.issue_query_pages(p, s, 1, 4, False, rid)
        if const_expr(self.pushed):
            early_position = self.position(p)
        early_even = Uint64(0)
        early_odd = Uint64(0)
        if const_expr(self.pushed):  # noqa: SIM102
            if (rid < 128) & (s.tid < 320):
                early_slot = (
                    rt.cdiv(rid, 32) * 2048 + rt.cmod(rid, 32) * 64 + s.tid - 256
                )
                early_even = rt.Memory.ld_volatile_u64(
                    rt.ga(p.SelectionMailbox, early_slot * 2)
                )
                early_odd = rt.Memory.ld_volatile_u64(
                    rt.ga(p.SelectionMailbox, (8192 + early_slot) * 2)
                )
        if const_expr(self.seam):
            cute.arch.griddepcontrol_wait()
            gtimer = rt.globaltimer()
            loader_start = gtimer
        with rt.warp_leader(s.warp, 9):
            ticket = rt.Memory.atom_add_u32(rt.ga(p.FullEpoch, 0), 1)
            s.epoch_slot[0] = rt.cdiv(ticket, 148)
        rt.bar_sync(*BAR_ALL)
        epoch = Uint32(s.epoch_slot[0])
        cur_pos_l = self.position(p)
        gate_seen = Uint32(epoch * 148)
        if const_expr(self.shared_or_seam):  # noqa: SIM102
            if (rid < 128) & (s.tid >= 320):
                self.load_shared_ids(p, s, rid, epoch, cur_pos_l)
        return (
            absorb_policy,
            cur_pos_l,
            early_even,
            early_odd,
            early_position,
            epoch,
            gate_seen,
            input_policy,
            l2_range_policy_1,
            l2_range_policy_2,
            loader_start,
            rid,
            shared_index_load,
        )

    @cute.jit
    def moe_list_prefetch(self, p, s, rid, epoch, loader_start):
        count = p.Flags[FlagWords.MOE_COUNT]
        source = Uint32(0)
        source = p.Flags[FlagWords.MOE_SOURCE]
        if source != 0:
            count = rt.FastMath.min_typed(count, 12)
        if count > 0:
            if source != 0:
                tag = ((epoch + 1) & 32767) | 32768
                lane_0 = s.tid - 256
                for half in cutlass.range_constexpr(2):
                    s.moe_votes[lane_0 + 128 * half] = Int32(0)
                bias_address = Uint64(p.Flags[FlagWords.ROUTER_BIAS]) | (
                    Uint64(p.Flags[FlagWords.ROUTER_BIAS + 1]) << 32
                )
                biases = rt.rmem_tensor(2, Float32)
                for half in cutlass.range_constexpr(2):
                    biases[half] = rt.Memory.gview(
                        Float32,
                        Int64(bias_address + Uint64(lane_0 + 128 * half) * 4),
                    )[0]
                words = rt.rmem_tensor(8, Uint32)
                done = Int32(0)
                while done == 0:
                    stale = Uint32(0)
                    for item in cutlass.range_constexpr(8):
                        words[item] = rt.Memory.ld_relaxed_gpu(
                            rt.ga(
                                p.Flags,
                                14716
                                + rt.cdiv(item, 2) * 256
                                + lane_0
                                + 128 * rt.cmod(item, 2),
                            )
                        )
                        stale = stale | ((Uint32(words[item]) >> 16) ^ tag)
                    if stale == 0:
                        done = Int32(1)
                rt.syncwarp()
                for item in cutlass.range_constexpr(8):
                    expert = lane_0 + 128 * rt.cmod(item, 2)
                    logit, _ = rt.Convert.bf16x2_to_f32x2(Uint32(words[item]))
                    _exp2_0 = cute.math.exp2(
                        rt.mul(-logit, Float32(1.4426950408889634)),
                        fastmath=False,
                    )
                    _rcp_0 = cute.arch.rcp_approx(rt.add(Float32(1.0), _exp2_0))
                    score = rt.add(_rcp_0, biases[rt.cmod(item, 2)])
                    bits = rt.Convert.as_u32(score)
                    flip = ((bits >> 31) * 2147483647) | Uint32(2147483648)
                    s.moe_keys[rt.cdiv(item, 2) * 256 + expert] = (
                        (bits ^ flip) & Uint32(4294967040)
                    ) | Uint32(255 - expert)
                rt.bar_sync(*BAR_LOADER)
                pick_row = s.warp - 8
                local = rt.rmem_tensor(8, Uint32)
                for item in cutlass.range_constexpr(8):
                    local[item] = s.moe_keys[pick_row * 256 + s.lane + 32 * item]
                for slot in cutlass.range(0, 8, 1, unroll=1):
                    winning = rt.Sync.redux_max_u32(self.max8_u32(local))
                    for item in cutlass.range_constexpr(8):
                        if Uint32(local[item]) == winning:
                            local[item] = Uint32(0)
                    if s.lane == 0:
                        best_id = 255 - Int32(winning & 255)
                        s.moe_picks[pick_row * 8 + slot] = best_id
                        rt.Memory.atom_add_shared(
                            s.moe_votes_addr + best_id * 4,
                            1 + Int32(pick_row == 0) * 65536,
                        )
                rt.bar_sync(*BAR_LOADER)
                if s.warp == 8:
                    keys = rt.rmem_tensor(8, Uint32)
                    for item in cutlass.range_constexpr(8):
                        expert = s.lane + 32 * item
                        counted = s.moe_votes[expert]
                        _max_0 = rt.FastMath.max_typed(
                            s.moe_keys[expert], s.moe_keys[256 + expert]
                        )
                        _max_1 = rt.FastMath.max_typed(
                            s.moe_keys[512 + expert], s.moe_keys[768 + expert]
                        )
                        best_key = rt.FastMath.max_typed(_max_0, _max_1)
                        keys[item] = (
                            (Uint32(counted & 3) << 30)
                            | ((best_key >> 10) << 8)
                            | (best_key & 255)
                        )
                        if (counted >> 16) != 0:
                            keys[item] = Uint32(0)
                    for slot in cutlass.range(0, 4, 1, unroll=1):
                        winning_key = rt.Sync.redux_max_u32(self.max8_u32(keys))
                        for item in cutlass.range_constexpr(8):
                            if Uint32(keys[item]) == winning_key:
                                keys[item] = Uint32(0)
                        if s.lane == 0:
                            s.moe_listed[8 + slot] = 255 - Int32(winning_key & 255)
                    if s.lane < 8:
                        first_pick = s.moe_picks[s.lane]
                        s.moe_listed[s.lane] = first_pick
                    rt.syncwarp()
                    if (rid == 132) & (s.lane < 12):
                        entry = s.moe_listed[s.lane]
                        rt.Memory.st_relaxed_sys(
                            rt.ga(p.Flags, 14692 + s.lane),
                            Uint32(entry) | (tag << 16),
                        )
                        if const_expr(self.full_or_push):  # noqa: SIM102
                            if p.peer_rank == 3:
                                published = Uint32(entry) | (
                                    (((epoch + 1) & 127) | 128) << 24
                                )
                                for selector_rank in cutlass.range_constexpr(3):
                                    base = p.peer_bases[selector_rank]
                                    rt.Memory.st_relaxed_sys(
                                        base + 917504 + Int64(s.lane) * 4,
                                        published,
                                    )
                rt.bar_sync(*BAR_LOADER)
            if source == 0:
                lane_0 = s.tid - 256
                if lane_0 < 24:
                    host_entry = p.Flags[14660 + lane_0]
                    s.moe_listed[lane_0] = Int32(host_entry)
                rt.bar_sync(*BAR_LOADER)
        with rt.warp_leader(s.warp, 8):
            rt.mbar_arrive(s.moe_list_full)
        _min_0 = rt.FastMath.min_typed(count, 8)
        if _min_0 > 0:
            mine = (Uint32((rid - 132) * 128) + Uint32(s.tid - 256)) * 8
            self.prefetch_moe_weights(
                p, s, mine, _min_0 * LAYOUT.w13_expert_lines, loader_start, 8, 0
            )

    @stage
    @cute.jit
    def loader_pages(
        self,
        p,
        s,
        absorb_policy,
        cur_pos_l,
        early_even,
        early_odd,
        early_position,
        epoch,
        input_policy,
        loader_start,
        rid,
    ):
        if const_expr(not self.pushed):
            if const_expr(not self.PHASED):  # noqa: SIM102
                if rid < 128:
                    self.load_residual(p, s, rid, 4)
            if rid < 82:
                for page in cutlass.range_constexpr(4, 6):
                    self.issue_input_page(p, s, page, rid)
            router_on_0 = Int32(0)
            if const_expr(self.full_or_push):
                if (rid >= 132) & (rid < 140):
                    router_on_0 = self.router_enabled(p)
                if router_on_0 != 0:
                    self.issue_query_pages(p, s, 4, 6, True, rid)
            if const_expr(self.shared_or_seam):  # noqa: SIM102
                if (rid >= 132) & (rid < 140):
                    self.issue_query_pages(p, s, 4, 6, False, rid)
            if (rid < 128) | (rid >= 132):
                for page in cutlass.range_constexpr(2):
                    hit = self.ring_hit(p, rid, self.full_or_push)
                    self.issue_ring_page(
                        p,
                        s,
                        6 * hit + page,
                        p.AbsorbWeight,
                        ((rid - 4 * Int32(rid >= 132)) * 2 + page)
                        * Ring.WEIGHT_PAGE_BYTES,
                        absorb_policy,
                        False,
                    )
                served = p.peer_rank - 3
                if const_expr(self.push):  # noqa: SIM102
                    if (rid >= 140) & (rid < 148) & (served >= 0) & (served < 3):
                        pushed_page = (144 + (rid - 140) * 18) * 2
                        for extra in cutlass.range(0, 36, 1, unroll=1):
                            hit = self.ring_hit(p, rid, True)
                            self.issue_ring_page(
                                p,
                                s,
                                6 * hit + 2 + extra,
                                p.AbsorbWeight,
                                (pushed_page + extra) * Ring.WEIGHT_PAGE_BYTES,
                                absorb_policy,
                                False,
                            )
            if self.prefetch_slot(p, rid) >= 132:
                self.moe_list_prefetch(p, s, rid, epoch, loader_start)
        if rid < 128:
            query_ = rt.cdiv(rid, 32)
            split = rt.cmod(rid, 32)
            if const_expr(self.pushed):
                index_sequence = Int32(0)
                position = early_position
            else:
                index_sequence = self.last_weight_sequence(p, rid, self.full_or_push)
                position = cur_pos_l
            first = index_sequence + 1
            last = index_sequence + 2
            context = position + query_ + 1
            score_count = rt.FastMath.min_typed(context, 2048)
            if const_expr(not self.shared_or_seam):
                generation = epoch + 1
                token = query_
                early = (early_even, early_odd) if self.pushed else None
                index_, fresh, record_index = self.probe_ids(
                    p, s, token, split, context, score_count, generation, early, False
                )
            rt.bar_sync(*BAR_LOADER)
            if const_expr(not self.shared_or_seam):
                if (s.id_probe[0] == 0) | (s.id_probe[1] == 0):
                    if const_expr(self.full_or_push):
                        self.poll_flags(p, FlagWords.PE_READY, 2, s.tid - 256, epoch)
                        rt.bar_sync(*BAR_LOADER)
                        self.poll_flags(
                            p, FlagWords.KV_LATENT_READY, 4, s.tid - 256, epoch
                        )
                        rt.bar_sync(*BAR_LOADER)
                    if const_expr(self.pushed):  # noqa: SIM102
                        if s.warp == 10:
                            self.poll_head_flags(p, s.lane, epoch)
                    self.prefetch_context(p, s, position + 4, rid)
                self.install_ids(
                    p, s, index_, fresh, record_index, generation, position, False
                )
            active = Int32(score_count > split * 64)
            self.arm_kv_ring(s, first, last, active)
            rt.bar_sync(*BAR_LOADER)
            if (s.id_probe[0] == 0) | (s.id_probe[1] == 0):
                if const_expr(not self.pushed):  # noqa: SIM102
                    if rid < 82:
                        rt.mbar_wait(s.norm_free, 0)
                if active != 0:
                    self.pe_stage(p, s, score_count, split)
                cute.arch.cp_async_commit_group()
            if active != 0:
                self.kv_gather(p, s, 0, first, last, L2_HINT)
            overlap = Int32(
                (s.id_probe[0] != 0)
                & (s.id_probe[1] != 0)
                & ((s.warp_hits[0] | s.warp_hits[1]) != 0)
            )
            if (overlap != 0) & (s.warp == 8):
                self.poll_head_flags(p, s.lane, epoch)

    @stage
    @cute.jit
    def loader_kv_tail(
        self,
        p,
        s,
        cur_pos_l,
        epoch,
        gate_seen,
        l2_range_policy_1,
        l2_range_policy_2,
        rid,
        shared_index_load,
    ):
        self.loader_kv_gather(p, s, cur_pos_l, epoch, rid)
        if const_expr(not self.PHASED):
            if rid < 128:
                with rt.warp_leader(s.warp, 8):
                    if const_expr(self.pushed):
                        index_sequence = Int32(0)
                    else:
                        index_sequence = self.last_weight_sequence(
                            p, rid, self.full_or_push
                        )
                    self.drain_ring(s, index_sequence, Int32(0 if self.pushed else 1))
                    rt.mbar_expect_tx(s.tail_full, 32800)
                    for half in cutlass.range_constexpr(2):
                        kvb_row = (
                            rt.cdiv(2 * rt.cmod(rid, 32) + half, 8) * 448
                            + 192
                            + rt.cmod(2 * rt.cmod(rid, 32) + half, 8) * 32
                        )
                        kvb_dst = s.ring_addr + 116224 * half
                        rt.AsyncCopy.tma_3d(
                            kvb_dst, p.KvbTmap, 0, kvb_row, 0, s.tail_full, L2_HINT
                        )
                        rt.bulk_g2s_hint(
                            s.ring_addr + (16384 + 116224 * half),
                            rt.baddr(p.KVbScales, rt.cdiv(kvb_row, 128) * 4 * 4),
                            16,
                            s.tail_full,
                            l2_range_policy_1,
                        )
                    for page in cutlass.range_constexpr(2):
                        self.issue_o_page(p, s, page, rid, l2_range_policy_2)
                if rid < 128:
                    lane_0 = s.tid - 256
                    poll_addr = self.split_done(
                        p,
                        s,
                        rt.cdiv(rid, 32) * 32 + rt.FastMath.min_typed(lane_0, 31),
                    )
                    self.warp_poll_ge(poll_addr, Int32(lane_0 < 32), epoch + 1)
                    rt.Memory.fence_acquire_gpu()
                    rt.bar_sync(*BAR_LOADER)
                    rt.bar_sync(*BAR_ALL)
                with rt.warp_leader(s.warp, 8):
                    if rid < 128:
                        rt.mbar_wait(s.merge_loaded, 0)
                    for page in cutlass.range_constexpr(2, 4):
                        self.issue_o_page(p, s, page, rid, l2_range_policy_2)
                lane_0 = s.tid - 256
                if const_expr(self.pushed):
                    cute.arch.griddepcontrol_wait()
                    self.load_residual(p, s, rid, 4)
                cute.arch.cp_async_wait_group(0)
                if const_expr(self.seam):  # noqa: SIM102
                    if s.warp == 9:  # noqa: SIM102
                        if rt.elect_sync() != 0:
                            gate_seen = Uint32(
                                rt.Memory.ld_relaxed_gpu(rt.ga(p.FullEpoch, 0))
                            )
                if const_expr(self.shared_or_seam):
                    exchange_generation_load = (epoch + 1) * 57 + shared_index_load
                    parity = Int32(rt.cmod(exchange_generation_load, 2))
                    mail_tag = (exchange_generation_load & 127) | 128
                else:
                    parity = Int32(rt.cmod(epoch + 1, 2))
                    mail_tag = ((epoch + 1) & 127) | 128
                base = p.peer_bases[0]
                for owner in cutlass.range_constexpr(1, 8):
                    if p.peer_rank == owner:
                        base = p.peer_bases[owner]
                item = lane_0
                item_0 = lane_0 + 128
                if item_0 >= 224:
                    item_0 = lane_0
                rt.bar_sync(*BAR_ALL)
                rt.OExchange.receive(
                    s.received_words,
                    base,
                    parity,
                    mail_tag,
                    rid,
                    lane_0,
                    [item, item_0],
                    224,
                )
                rt.bar_sync(*BAR_ALL)
            if const_expr(self.seam):
                with rt.warp_leader(s.warp, 9):
                    done = Int32(0)
                    while done == 0:
                        if gate_seen - epoch * 148 >= 148:
                            done = Int32(1)
                        if done == 0:
                            gate_seen = Uint32(
                                rt.Memory.ld_relaxed_gpu(rt.ga(p.FullEpoch, 0))
                            )

    @cute.jit
    def loader_kv_gather(self, p, s, cur_pos_l, epoch, rid):
        if rid < 128:
            query_ = rt.cdiv(rid, 32)
            split = rt.cmod(rid, 32)
            if const_expr(self.pushed):
                index_sequence = Int32(0)
            else:
                index_sequence = self.last_weight_sequence(p, rid, self.full_or_push)
            first = index_sequence + 1
            last = index_sequence + 2
            score_count = rt.FastMath.min_typed(cur_pos_l + query_ + 1, 2048)
            active = Int32(score_count > split * 64)
            rt.bar_sync(*BAR_LOADER)
            pe_staged = Int32(0)
            pe_staged = Int32((s.id_probe[0] == 0) | (s.id_probe[1] == 0))
            if pe_staged == 0:
                if const_expr(not self.pushed):  # noqa: SIM102
                    if rid < 82:
                        rt.mbar_wait(s.norm_free, 0)
                if active != 0:
                    self.pe_stage(p, s, score_count, split)
            cute.arch.cp_async_commit_group()
            if active != 0:
                self.kv_gather(p, s, 1, first, last, L2_HINT)
            cute.arch.cp_async_wait_group(0)
            if active != 0:
                self.pe_widen(s)
            rt.bar_sync(*BAR_ALL)


class FullWorker(TargetWorker):
    """kFull: full-sequence worker (fused MoE emit, replicated selection refine)."""

    ROLE = kFull
    full_or_push = True


class PushWorker(FullWorker):
    """kPush: kFull that also pushes latent, heads and selection ids to its peers."""

    ROLE = kPush
    push = True


class PushedWorker(TargetWorker):
    """kPushed: consumes the pushed, colocated list (fast start, probe)."""

    ROLE = kPushed
    pushed = True
    OPERAND_Q = 0

    def mailbox(self, p, s, word, pool=0):
        return (
            self.peer_base_by_select(p.peer_rank, p.peer_bases)
            + Int64(self.PUSHED_OPERANDS)
            + Int64(word) * 4
        )


class SharedWorker(TargetWorker):
    """kShared: shared-mailbox worker, RoleTable placement; polls the tagged
    selections.
    """

    ROLE = kShared
    shared_or_seam = True

    def split_done(self, p, s, slot):
        return self.handoff_word(p, Handoff.SPLIT_DONE + slot, rt.cdiv(slot, 64))

    def role_entry(self, p: WorkerParams, bid):
        return rt.Memory.ldg_nc_u32(self.role_table_word(p, Handoff.ROLE_TABLE + bid))


class SeamWorker(SharedWorker):
    """kSeam: kShared that also owns the seam input XG."""

    ROLE = kSeam
    seam = True
    RECORDS = (HeadArgs, TailArgs, SeamArgs)

    def role_entry(self, p: WorkerParams, bid):
        return rt.Memory.ldg_nc_u32(
            self.role_table_word(p, Handoff.ROLE_TABLE + Int32(rt.Sync.smid()))
        )

    @cute.kernel
    def kernel(
        self,
        head: HeadArgs,
        KvCache: GridConstant[TensorMap],
        tail: TailArgs,
        seam: SeamArgs,
        InputTmap: GridConstant[TensorMap],
        OTmap: GridConstant[TensorMap],
        KvbTmap: GridConstant[TensorMap],
    ):
        maps = {"InputTmap": InputTmap, "OTmap": OTmap, "KvbTmap": KvbTmap}
        self.run_records(KvCache, (head, tail, seam), maps)
