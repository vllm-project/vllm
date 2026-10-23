# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL GLM-5.3 MLA decode worker kernels (one class per role; 384 threads per
CTA).
"""

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
        """Byte offset (from the ring base) of the slot of page ``seq``."""
        return rt.cmod(seq, Ring.SLOTS) * Ring.SLOT_BYTES


class KvRows:
    """The paged MLA latent cache (KVCachePtr and PECache view the same pool), 64 rows
    per block.
    """

    BLOCK_ROWS, BLOCK_SHIFT = 64, 6
    PITCH, ROPE_OFFSET = 576, 512
    BLOCK_BYTES = BLOCK_ROWS * PITCH

    @staticmethod
    def element(block, position, column) -> Int64:
        """u16 index of ``column`` in token ``position``'s row of block ``block`` (an
        Int64 block number).
        """
        return block * (KvRows.BLOCK_BYTES // 2) + Int64(
            rt.cmod(position, KvRows.BLOCK_ROWS) * (KvRows.PITCH // 2) + column
        )

    @staticmethod
    def rope_byte(PECache, BlockTable, position, index):
        """Address of RoPE key ``index`` (a byte) in the paged row of ``position``."""
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
        """One RoPE key, ``cvt.rn.satfinite.e4m3x2.f32`` of the FP32 ``value`` (low byte
        stored).
        """
        addr = Int64(address)
        e4m3 = rt.Convert.f32x2_to_e4m3x2(value, value)
        rt.Core.ptx("st.global.b8 [$0], $1;", addr, e4m3, memory=False)

    @staticmethod
    def rope_pair_bf16(address) -> Uint32:
        """The two E4M3 RoPE keys at ``address`` as a BF16 pair (exact)."""
        return rt.Convert.e4m3x2_to_bf16x2(rt.Memory.gview(Uint16, address, 1, 2)[0])

    @staticmethod
    def store_rope_pair(address, word) -> None:
        """A BF16 pair of E4M3 values (``rope_pair_bf16``) narrowed back to its two
        bytes (exact).
        """
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
        """Byte offset of the 8-byte half ``half`` (0 or 8) of column group ``group``
        (multiple of 8) of ``head``.
        """
        return (
            rt.cdiv(group, 64) * 2048
            + head * 128
            + ((rt.cmod(group, 64) * 2) ^ (rt.cmod(head, 8) * 16))
            + half
        )

    @staticmethod
    @cute.jit
    def store(act_addr, chunk, wa, wb):
        """Store polled chunk ``chunk`` (its two v4 halves ``wa``, ``wb``) into the tile
        (chunks past 575: none).
        """
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


class PushedRows:
    """The draft's pushed-latent mailbox in a selector rank's arena (u64 records from
    the peer base).
    """

    DRAFT = 917504
    TARGET = 917632
    PE_RECORDS = 256

    @staticmethod
    def record(base, parity, token, column):
        """Byte offset of record ``column`` of row ``token`` at ``parity`` in the
        mailbox at byte ``base``.
        """
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
    READ_ONLY = {"Freq": Float32} | dict.fromkeys(("CurPos", "BlockTable"), Int32)


@dataclass(frozen=True)
class HeadArgs(_Args):
    """The pointers before the KvCache tensor map."""

    X: cute.Pointer = rt.ptr("bfloat16")
    GammaX: cute.Pointer = rt.ptr("float32")
    InputWeight: cute.Pointer = rt.ptr("float8_e4m3")
    GammaQ: cute.Pointer = rt.ptr("float32")
    QueryWeight: cute.Pointer = rt.ptr("float8_e4m3")
    AbsorbWeight: cute.Pointer = rt.ptr("float8_e4m3")
    GammaKV: cute.Pointer = rt.ptr("float32")
    Freq: cute.Pointer = rt.ptr("float32")
    CurPos: cute.Pointer = rt.ptr("int32")


@dataclass(frozen=True)
class TailArgs(_Args):
    """The pointers and scalars after the KvCache tensor map, up to BlockTable."""

    KVCachePtr: cute.Pointer = rt.ptr("uint16")
    PECache: cute.Pointer = rt.ptr("bfloat16")
    SelectionMailbox: cute.Pointer = rt.ptr("uint32")
    Handoff: cute.Pointer = rt.ptr("uint32")
    PartialO: cute.Pointer = rt.ptr("float32")
    PartialLSE: cute.Pointer = rt.ptr("float32")
    KVbWeight: cute.Pointer = rt.ptr("float8_e4m3")
    OWeight: cute.Pointer = rt.ptr("float8_e4m3")
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


@dataclass(frozen=True)
class SeamArgs(_Args):
    """kSeam's Extra pointer after the tail record: the seam input."""

    XG: cute.Pointer = rt.ptr("bfloat16")


@dataclass(frozen=True)
class WorkerParams:
    """Every kernel parameter as a typed view, built once at the top of the kernel."""

    X: "cute.Tensor"
    GammaX: "cute.Tensor"
    InputWeight: "rt.TypedView"
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
    OWeight: "rt.TypedView"
    Hidden: "cute.Tensor"
    Final: "cute.Tensor"
    FullEpoch: "rt.TypedView"
    Flags: "rt.TypedView"
    BlockTable: "rt.ReadOnlyView"
    peer_rank: Int32
    peer_bases: tuple
    KvCache: Int64
    InputTmap: Int64 | None = None
    OTmap: Int64 | None = None
    KvbTmap: Int64 | None = None
    XG: "cute.Tensor | None" = None

    @staticmethod
    def of(KvCache, *records, **maps) -> "WorkerParams":
        """Join the kernel's records and tensor maps: typed views, descriptor addresses,
        the eight peer bases.
        """
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
        """Per-thread kernel context, in this order: ``(tid, warp, lane, smem,
        bid)``.
        """
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
        """The kernel's record: thread context, RoleTable word and MoE-list mbarrier
        (kernel top level).
        """
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
        """This record with every mbarrier address and view derived from ``smem`` here
        (a stage's entry).
        """
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
        functools.update_wrapper(self, fn)

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


class MlaWorker(rt.RoleKernel):
    """Shared base of the worker roles: launch, the per-kernel records and the stages
    every role uses.
    """

    ROLE = None
    GRID = 148
    BLOCK = 384
    SMEM_BYTES = 215040
    ROLES = (
        rt.Role("compute", 0, 7, regs=216),
        rt.Role("load", 8, 11, regs=72, increase=False),
    )
    EXPLICIT_SMEM_BASE = False
    RECORDS = (HeadArgs, TailArgs)
    HOST_ARGS = (("KvBytes", Int64),)
    X_FULL = NORM_FREE = MERGE_LOADED = ABSORB_FULL = MOE_LIST_FULL = None
    full_or_push = shared_or_seam = push = pushed = seam = False
    smem_record: "WorkerSmem | None" = None

    @staticmethod
    def kv_map(tail, kv_bytes):
        """Host wrapper: the gather descriptor of the paged pool (``kv_bytes``): one
        row's 512 latent bytes per box.
        """
        rows = (kv_bytes - KvRows.ROPE_OFFSET) // KvRows.PITCH + 1
        latent = [KvRows.ROPE_OFFSET, rows]
        return rt.byte_map(
            tail.KVCachePtr.toint(), latent, [KvRows.PITCH], [128, 1], True, True
        )

    @staticmethod
    def page_map(weight, rows, columns, chunk, page_rows, page_k, swizzle):
        """Host wrapper: a vLLM linear's page descriptor (``chunk``-byte columns,
        ``page_rows`` by ``page_k`` boxes).
        """
        dims, box = [chunk, rows, columns // chunk], [chunk, page_rows, page_k // chunk]
        return rt.byte_map(weight.toint(), dims, [columns, chunk], box, swizzle)

    def mbars(self) -> dict:
        """The role's per-stage mbarrier byte offsets by ``WorkerSmem`` field name
        (None: absent).
        """
        return dict(
            x_full=self.X_FULL,
            norm_free=self.NORM_FREE,
            merge_loaded=self.MERGE_LOADED,
            absorb_full=self.ABSORB_FULL,
        )

    def role_entry(self, p: WorkerParams, bid):
        """This CTA's RoleTable word (kShared / kSeam); None for the other roles."""
        return None

    def kernel_extras(self, head, tail, extras: list) -> tuple:
        """Host wrapper: the kernel parameters after the tail record (the role's Extra
        records, the page maps).
        """
        return ()

    @cute.jit
    def enter(self, p: WorkerParams):
        """The kernel after its parameter records are joined: prologue, then the role
        dispatch.
        """
        ids, s = self.prologue(p)
        self.run(p, s, ids)

    @cute.jit
    def launch(self, *args):
        """Host wrapper: ``args`` are the record fields in order, the paged pool's
        bytes, then the stream.
        """
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
        """One elected lane of warp 0 releases ``Flags[index] += 1``."""
        with rt.warp_leader(s.warp, 0):
            rt.Memory.red_async_release_gpu_add(
                rt.ga(p.Flags, index), Uint32(1), memory=True
            )

    @cute.jit
    def poll_flags(self, p, first, n: cutlass.Constexpr, lane_0, epoch):
        """Lanes ``lane_0 < n`` spin on ``Flags[first + lane_0] >= epoch + 1``; then
        reconverge and acquire.
        """
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
        """Stage the ``rows`` live columns of ``n_m`` m16 x n8 tiles per warp in
        scratch, then bar 8.
        """
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
        """Scratch row ``i`` summed over the eight warps' staged partials, in warp
        order.
        """
        return rt.Reduce.ordered_sum(
            lambda k: s.scratch[k * (rows * 16 * n_m + 1) + i], 8
        )

    @stage
    def rope(self, p, s, rounded, dim):
        """Rotary embedding of the bf16-rounded ``rounded`` at column ``dim`` (pair
        partner by shuffle).
        """
        partner = cute.arch.shuffle_sync_bfly(rounded, 1)
        angle = rt.cdiv(s.tid, 32) * 64 + (rt.cmod(dim, 64) & -2)
        cosine, sine = p.Freq[angle], p.Freq[angle + 1]
        odd = rt.fma(rounded, cosine, rt.mul(partner, sine))
        even = rt.fma(rounded, cosine, -(rt.mul(partner, sine)))
        return cutlass.select_(rt.cmod(dim, 2) == 1, odd, even)

    def kv_row_off(self, p, position, column):
        """u16 index of ``column`` in token ``position``'s paged latent row
        (``KvRows``).
        """
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
        """Spin until the 8-word Tagged16 chunks at global addresses ``chunks`` all
        carry ``tag_high``.
        """
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
        """One pass of ``poll_tagged16``: returns 1 once every chunk carries the tag
        (after ``on_ready``).
        """
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
        """Busy-wait ``ns`` nanoseconds on the low word of ``%globaltimer``."""
        begin = Uint32(rt.globaltimer())
        waited = Int32(0)
        while waited == 0:
            if Uint32(rt.globaltimer()) - begin >= Uint32(ns):
                waited = Int32(1)

    def handoff_word(self, p, word, pool=0):
        """Address of Handoff word ``word`` (pool ``pool`` for Shared/Seam)."""
        if self.shared_or_seam:
            return rt.ga(p.Handoff, word + pool * Handoff.MIRROR)
        return rt.ga(p.Handoff, word)

    def mailbox(self, p, s, word, pool=0):
        """Address of Handoff word ``word`` as this role reads it."""
        return self.handoff_word(p, word, pool)

    @stage
    @cute.jit
    def kv_latent_store(self, p, s, hbase, query_, position, tag):
        """KV latent of row ``query_``: poll its Handoff pair, RMS norm, E4M3 into the
        paged cache.
        """
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
        at = self.kv_row_off(p, row, s.tid)
        rt.Memory.gview(Uint16, rt.ga(p.KVCachePtr, at), 1, 2)[0] = e4m3x2

    @staticmethod
    def p_word(row, key):
        """Word index in ``act`` of the P tile's bf16 pair (key, key + 1) of head row
        ``row``.
        """
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
        """Per head pair: ``1 / sum`` and the log-sum-exp ``max * scale + log2(sum)`` (0
        and -inf when empty).
        """
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
        """Warp 0: the splits' merge weights ``exp2(lse - global_lse)`` into
        ``merge_weights[lane]``.
        """
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
        """``value = sum_s parts[si] * merge_weights[split_base + si]`` (16 splits,
        f32x4 each), from zero.
        """
        for i in range(4):
            value[i] = Float32(0.0)
        for si in range(16):
            weight = s.merge_weights[split_base + si]
            for i in range(4):
                value[i] = rt.fma(parts[si][i], weight, value[i])

    @stage
    @cute.jit
    def fold_halves(self, s, value):
        """Threads 128-255 stage their four partial sums; threads 0-127 add them."""
        if s.tid >= 128:
            for i in cutlass.range_constexpr(4):
                s.scratch[(s.tid - 128) * 4 + i] = value[i]
        rt.bar_sync(*BAR_COMPUTE)
        if s.tid < 128:
            for i in cutlass.range_constexpr(4):
                value[i] = rt.add(value[i], s.scratch[s.tid * 4 + i])

    @stage
    def tagged_push(self, p, s, bases, epoch, token, packed_peer, chunk, slot=None):
        """Send 12 bytes of O row ``token`` (mailbox ``slot``, default the CTA) to the
        ``packed_peer``-th other rank.
        """
        slot = s.bid if slot is None else slot
        rt.OExchange.push(
            s.local_o_words, bases, p.peer_rank, epoch, token, packed_peer, chunk, slot
        )

    @stage
    def tagged_reduce(self, p, s, query_):
        """Sum the eight ranks' bf16 O words (own row from local_o, peers' from
        received) into local_o.
        """
        rt.OExchange.reduce(
            s.local_o_words, s.local_o_addr, p.peer_rank, query_, s.lane
        )

    @stage
    def final_residual(self, p, s, rows: int, slot=None):
        """``Final = bf16(local_o + residual)`` for the 48 columns of ``slot`` (default
        the CTA), ``rows`` rows.
        """
        slot = s.bid if slot is None else slot
        for r in range(rows):
            o = Float32(s.local_o[r * 48 + s.tid])
            p.Final[r * 6144 + slot * 48 + s.tid] = BFloat16(
                rt.add(o, Float32(s.residual[r * 48 + s.tid]))
            )

    @stage
    @cute.jit
    def tagged_exchange_final(self, p, s, bases, epoch, slot):
        """Four-row O exchange of mailbox ``slot``: push to the 7 peers, reduce the 8
        ranks, add the residual.
        """
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
        """The key's paged row into ``indices``; a quad's flag: one of its keys is an
        appended row.
        """
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
        """Load warps 8-9: each key's selection id (tagged records past TOPK keys);
        id_probe: all fresh.
        """
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
        """Load warps 8-9: re-poll stale ids, install the keys' rows (``store_key``) and
        the warps' append hits.
        """
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
        """Warp 8: free page ``first`` (no copy) and arm ``last`` for the 32 KiB of
        gathered KV rows.
        """
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
        """gather4 of the KV rows of every 4-key group whose append flag equals
        ``flag``, L2 hint ``hint``.
        """
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
        """Load lanes: ``cp.async`` the ``rows`` Hidden rows' 48 columns of ``slot``
        into residual, committed.
        """
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
        """Warps 10-11: L2 prefetch of ``slot``'s 1/128 slice of the ``rows`` context
        rows (4 KV planes).
        """
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
        """Wait until the KV pages and the weight pages before them drained (the tail
        pages reuse the ring).
        """
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
        self, p, s, first, last, split, split_count, score_count, query_0
    ):
        """One 64-key split of row ``query_0`` for the 8 local heads (MlaWorker roles):
        scores, softmax, P x V.
        """
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
                km_a[_f * 4], km_a[_f * 4 + 1], km_a[_f * 4 + 2], km_a[_f * 4 + 3] = (
                    rt.TensorCore.ldsm_x4(
                        s.ring_addr
                        - 16384
                        + 182272
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
    def km_merge(self, p, s, cur_pos_c, slot):
        """Merge the 32 splits' partial O of head ``slot % 32 / 4`` of row ``slot / 32``
        into the activation tile.
        """
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
        rt.mbar_arrive(s.merge_loaded)
        self.fold_halves(s, merged)
        if s.tid < 128:
            word = rt.cdiv(query_ * 1040 + 2 * dims, 4)
            s.act[word] = rt.f32x2_to_bf16x2(merged[0], merged[1])
            s.act[word + 1] = rt.f32x2_to_bf16x2(merged[2], merged[3])

    @stage
    @cute.jit
    def vo_token_partials(self, s, slot, o_vo: cutlass.Constexpr):
        """Tail page 0: this warp's V/O partials of the 4 tokens for head pair ``slot /
        32 / 2``, staged in scratch.
        """
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
        """O projection over ``pages`` O-weight pages, each once its Handoff words carry
        the tag.
        """
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
    def pe_stage(self, p, s, score_count, split, hint=L2_HINT):
        """Load warps: the even lanes cp.async their PE chunk pair (zero-filled past
        ``score_count``).
        """
        lane_0 = s.tid - 256
        pe_offset = (
            rt.cdiv(lane_0, 8) * 128
            + (rt.cmod(lane_0, 8) ^ rt.cmod(rt.cdiv(lane_0, 8), 8)) * 16
        )
        dst = s.act_addr + 18432 + pe_offset
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

    def pe_widen(self, s):
        """Even load lanes widen their staged E4M3 PE chunk pairs (``pe_stage``) to
        BF16.
        """
        lane_0 = s.tid - 256
        row, chunk = rt.cdiv(lane_0, 8), rt.cmod(lane_0, 8)
        base = s.act_addr + 18432 + row * 128
        own = base + (chunk ^ rt.cmod(row, 8)) * 16
        partner = base + ((chunk + 1) ^ rt.cmod(row, 8)) * 16
        rt.Convert.widen_e4m3_pairs_shared(Int32((chunk & 1) == 0), own, partner)

    @stage
    def q_a_page_mma(self, s, accum, region, page):
        """One ring page of the q_a projection (warp-major ``fp8_page_mma``), scaled per
        m16 tile.
        """
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
        """One fp8 V/O tile: the ``ld.shared.b128`` row loads, then two k-passes of
        unpack + weight load + mma.
        """
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
        vo_acc = [None] * (4 * groups)
        vo_a = [None] * (8 * groups)
        for pj in range(2):
            for g in range(groups):
                ua = rt.Convert.unpack_u16x4(vo_rows[4 * g + pj])
                ub = rt.Convert.unpack_u16x4(vo_rows[4 * g + pj + 2])
                for i in range(4):
                    vo_a[8 * g + 2 * i] = rt.Convert.e4m3x2_to_bf16x2(ua[i])
                    vo_a[8 * g + 2 * i + 1] = rt.Convert.e4m3x2_to_bf16x2(ub[i])
            for h in range(2):
                vo_b = rt.Memory.lds_weak_b64(
                    s.act_addr + (vo_k + (4 * pj + 2 * h)) * 4
                )
                for g in range(groups):
                    a = vo_a[8 * g + 4 * h : 8 * g + 4 * h + 4]
                    if pj == 0 and h == 0:
                        vo_acc[4 * g : 4 * g + 4] = rt.TensorCore.mma(
                            None, a, vo_b, True
                        )
                    else:
                        rt.TensorCore.mma_acc_at(vo_acc, 4 * g, a, vo_b)
        return vo_acc

    def router_weight_addr(self, p) -> Int64:
        """The router weight address ``(u64)Flags[ROUTER_WEIGHT] |
        (u64)Flags[ROUTER_WEIGHT + 1] << 32``.
        """
        return Int64(
            Uint64(p.Flags[FlagWords.ROUTER_WEIGHT])
            | (Uint64(p.Flags[FlagWords.ROUTER_WEIGHT + 1]) << 32)
        )

    def router_enabled(self, p) -> Int32:
        """1 when this step has a router (its weight address in Flags is non-zero)."""
        return Int32(
            (p.Flags[FlagWords.ROUTER_WEIGHT] | p.Flags[FlagWords.ROUTER_WEIGHT + 1])
            != 0
        )

    @cute.jit
    def paged_kv_row(self, p, index) -> Int32:
        """The paged KV row of selection id ``index`` (``NO_ROW`` for ``NO_KEY``: the
        gather reads zeros).
        """
        row = Int32(NO_ROW)
        if index < NO_KEY:
            row = p.BlockTable[
                index >> KvRows.BLOCK_SHIFT
            ] * KvRows.BLOCK_ROWS + rt.cmod(index, KvRows.BLOCK_ROWS)
        return row

    @cute.jit
    def poll_head_flags(self, p, lane_0, epoch):
        """Six-lane head poll (lanes 0-1: Flags[80..81], lanes 2-5: Flags[276..279])
        then ``fence.acquire.gpu``.
        """
        owned = Int32(lane_0 < 6)
        index_0 = 80 + rt.FastMath.min_typed(lane_0, 1)
        if lane_0 >= 2:
            index_0 = 276 + rt.FastMath.min_typed(lane_0 - 2, 3)
        self.warp_poll_ge(rt.ga(p.Flags, index_0), owned, epoch + 1)
        rt.Memory.fence_acquire_gpu()

    @cute.jit
    def ring_hit(self, p, rid, router: cutlass.Constexpr) -> Int32:
        """``(int)(rid < 82 || <query pages ran>)``; kFull/kPush re-read the router flag
        (router_on_0_0).
        """
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
        """The ring sequence number of the last weight page before the KV pages."""
        hit = self.ring_hit(p, rid, router)
        weights_only = (rid < 128) | (rid >= 132)
        return cutlass.select_(
            weights_only, 6 * hit + 2 - 1, cutlass.select_(rid < 82, Int32(6), Int32(0))
        )

    @stage
    @cute.jit
    def issue_input_page(self, p, s, page: cutlass.Constexpr, rid, policy):
        """One input-projection page: 3-D TMA tile of InputTmap into ring slot page %
        4.
        """
        input_slot = rt.cmod(page, 4)
        with rt.warp_leader(s.warp, 8):
            s.pages.acquire(page, 32960 if page == 0 else 32768, input_slot)
            if const_expr(page == 0):
                rt.bulk_g2s_hint(
                    s.ring_addr + 33792,
                    rt.baddr(p.InputWeight, 16121856 + rt.cdiv(rid, 4) * 48 * 4),
                    192,
                    s.ring_full + input_slot * 8,
                    policy,
                )
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
        """One Ring.WEIGHT_PAGE_BYTES-byte weight page (query / router / absorb) into
        ring slot seq % 4 by cp.async.bulk.
        """
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
        """Query (kShared/kSeam: QueryWeight) or router (kFull/kPush: Flags-addressed)
        pages p0..p1-1.
        """
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
        """One output-projection page into tail_full[1 + page] (page 0 also brings the
        three scale blocks).
        """
        mbar = s.tail_full + (1 + page) * 8
        if const_expr(page == 0):
            rt.mbar_expect_tx(mbar, 24768)
            rt.bulk_g2s_hint(
                s.ring_addr + 16640,
                rt.baddr(p.OWeight, 12582912 + rt.cdiv(rid * 48, 128) * 16 * 4),
                64,
                mbar,
                policy,
            )
            for _rr in cutlass.range_constexpr(2):
                rt.bulk_g2s_hint(
                    s.ring_addr + (16704 + 64 * _rr),
                    rt.baddr(
                        p.OWeight,
                        12582912 + rt.cdiv(rid * 48 + (16 + 16 * _rr), 128) * 16 * 4,
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
        """One FP8 weight page of a row projection: 8 k16 steps x 2 m16 tiles (keep this
        order).
        """

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
    def x_norm4(self, p, s):
        """Stage ``x * GammaX`` of the four X rows in act; return this thread's row sum
        of squares.
        """
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
            for c in range(12)
        ]
        values = [rt.bf16x8_to_f32(words[c]) for c in range(12)]
        late_squares = [
            self.pair_squares(values[3 * q] + values[3 * q + 1] + values[3 * q + 2])
            for q in range(4)
        ]
        for c in cutlass.range_constexpr(12):
            scaled = [rt.mul(values[c][i], gamma[c % 3][i]) for i in range(8)]
            packed = [
                rt.f32x2_to_bf16x2(scaled[2 * i], scaled[2 * i + 1]) for i in range(4)
            ]
            rt.Memory.sts_v4(
                s.act_addr + (c // 3 * 6152 + c % 3 * 2048 + s.tid * 8) * 2, packed
            )
        self.stage_rows4(s, late_squares, 2016)
        rt.bar_sync(*BAR_COMPUTE)
        row = self.min_i32(rt.cdiv(s.tid, 32), 3)
        late_total = s.scratch[2016 + row * 8]
        for k in cutlass.range_constexpr(1, 8):
            late_total = rt.add(late_total, s.scratch[2016 + row * 8 + k])
        return late_total

    @stage
    @cute.jit
    def stage_rows4(self, s, values, base: cutlass.Constexpr):
        """Warp-sum four per-row partials (rows interleaved); lane 0 stages them at
        ``scratch[base + q * 8 + warp]``.
        """
        for step in cutlass.range_constexpr(5):
            values = [
                rt.add(values[q], cute.arch.shuffle_sync_down(values[q], 16 >> step))
                for q in range(4)
            ]
        if s.lane == 0:
            for q in cutlass.range_constexpr(4):
                s.scratch[base + q * 8 + s.warp] = values[q]

    @stage
    @cute.jit
    def rms_inverses4(self, s, values):
        """RMS inverses of four 512-wide rows (``values[q]``: this thread's 8 values of
        row q).
        """
        self.stage_rows4(s, [self.pair_squares(values[q]) for q in range(4)], 0)
        rt.bar_sync(*BAR_COMPUTE)
        if s.tid < 4:
            row_total = s.scratch[s.tid * 8]
            for k in cutlass.range_constexpr(1, 8):
                row_total = rt.add(row_total, s.scratch[s.tid * 8 + k])
            s.scratch[32 + s.tid] = rt.FastMath.rsqrt(
                rt.fma(row_total, Float32(0.00048828125), Float32(1e-05))
            )
        rt.bar_sync(*BAR_COMPUTE)
        inverses = [s.scratch[32 + q] for q in range(4)]
        return inverses

    def load_x_rows(self, s, x):
        """The four decode rows of ``x`` (X or XG, 6144 bf16 each) by bulk copy into act
        on ``x_full``.
        """
        rt.mbar_expect_tx(s.x_full, 4 * X_ROW_BYTES)
        for r in range(4):
            act_row = s.act_addr + (X_ROW_BYTES + 16) * r
            rt.bulk_g2s(act_row, rt.baddr(x, X_ROW_BYTES * r), X_ROW_BYTES, s.x_full)

    def is_router(self, slot):
        """Worker slot ``slot`` is one of the router slots [132, 140)."""
        return (slot >= 132) & (slot < 140)

    @staticmethod
    def min_i32(a, b):
        """``a < b ?"""
        return rt.FastMath.select_i32(a < b, a, b)

    @staticmethod
    def pair_squares(values):
        """``sum_p fma(v[2p], v[2p], v[2p + 1]^2)`` added left to right (the RMS square
        sum).
        """
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
        """Four bf16x2 words of ``v * (gamma * inverse)`` for eight values."""
        y = [rt.mul(values[i], rt.mul(gamma[i], inverse)) for i in range(8)]
        return [rt.f32x2_to_bf16x2(y[2 * i], y[2 * i + 1]) for i in range(4)]

    @staticmethod
    def peer_base_by_select(rank, bases) -> Int64:
        """``bases[rank]`` by an unrolled select chain over the eight peer bases."""
        base = bases[0]
        for k in range(1, 8):
            base = cutlass.select_(rank == k, bases[k], base)
        return base

    @staticmethod
    def max8_u32(v):
        """Max tree over eight u32 register values (rmem loads come back signless)."""
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
        """Warp spin: each owned lane re-reads ``ld.relaxed.gpu`` at ``addr`` until the
        whole warp sees >= target.
        """
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
        """The role's Extra records, then InputTmap, OTmap and KvbTmap."""
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
        """Entry kernel without Extra parameters (kFull, kPush, kPushed)."""
        self.enter(
            WorkerParams.of(
                KvCache, head, tail, InputTmap=InputTmap, OTmap=OTmap, KvbTmap=KvbTmap
            )
        )

    @cute.jit
    def prologue(self, p: WorkerParams):
        """Records, the KvCache descriptor prefetch and the mbarrier init; returns
        ``(ids, s)``.
        """
        s = WorkerSmem.build(self, p)
        self.smem_record = s
        self.setup(p, s)
        return rt.ThreadIds(s.tid, s.bid, s.warp, s.lane), s

    @stage
    @cute.jit
    def setup(self, p, s):
        """KvCache descriptor prefetch and the mbarrier init."""
        if const_expr(not self.shared_or_seam):
            if (s.warp == 8) & (s.bid < 128):
                rt.AsyncCopy.prefetch_tensormap(p.KvCache)
        else:
            if s.warp == 8:
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
        """Compute role (warps 0-7): prologue, input RMS norm and q-down pages, then the
        role's attention.
        """
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
        if const_expr(self.full_or_push):
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
        """Load role (warps 8-11): weight pages, selection ids, KV gathers and the tail
        pages.
        """
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
        """Selection refine (kFull: CTAs 144..147, kPush: CTAs 128..131), past 2048
        context keys.
        """
        refine_bid0 = 128
        if const_expr(kFull == self.ROLE):
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
        """Compute-role prologue: role slot, PDL wait, current position, router flag and
        the epoch seed.
        """
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
                        self.load_x_rows(s, p.X)
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
                        self.load_x_rows(s, p.X)
                    l2_range_policy_0 = rt.AsyncCopy.createpolicy_evict_first(
                        rt.ga(p.InputWeight, 0), 16126080, 16126080
                    )
                    rt.mbar_expect_tx(s.ring_full, 32960)
                    rt.bulk_g2s_hint(
                        s.ring_addr + 33792,
                        rt.ga(
                            p.InputWeight,
                            Int64(16121856 + rt.cdiv(role_slot, 4) * 48 * 4),
                        ),
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
                    self.load_x_rows(s, p.XG)
                if role_slot < INPUT_CTAS:
                    self.load_x_rows(s, p.XG)
        self.set_regs("compute")
        if const_expr(not self.pushed):
            cute.arch.griddepcontrol_wait()
        cur_pos_c = p.CurPos[0]
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

    @stage
    @cute.jit
    def input_norm_q_down(self, p, s, cur_pos_c, epoch, role_slot, router_on):
        """Input RMS norm and the six q-down pages (input CTAs < 82 and router slots),
        the epoch tag.
        """
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
                    late_inverse = rt.Reduce.rms_inverse(self.x_norm4(p, s))
                rt.bar_sync(*BAR_ALL)
                epoch_0 = s.epoch_slot[0]
                tag_0 = ((epoch_0 + 1) & 32767) | 32768
                accum = rt.rmem_tensor(12, Float32)
                for m0 in cutlass.range_constexpr(8):
                    accum[m0] = Float32(0.0)
                for page in cutlass.range(0, 6, 1, unroll=1):
                    if const_expr(self.seam):  # noqa: SIM102
                        if page == 4:
                            self.stage_rows4(s, seam_pre, 2016)
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

    @stage
    @cute.jit
    def full_qa_kv(self, p, s, cur_pos_c, epoch, tag):
        """Full/Push: query norm and q_a tile (kPush CTAs 140..147: 18 tiles of the
        served peer), KV latent row.
        """
        if (s.bid < 128) | (s.bid >= 132):
            router_on = Int32(0)
            if (s.bid >= 132) & (s.bid < 140):
                router_on = self.router_enabled(p)
            self.query_norm_qa(
                p, s, s.bid, tag, 6 * Int32((s.bid < 82) | (router_on != 0))
            )
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
    def query_norm_qa(self, p, s, slot, tag, seq):
        """Query RMS norm and q_a tile ``slot - 4 * (slot >= 132)`` of the four rows
        (Full/Push, Shared/Seam).
        """
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
        inverse = self.rms_inverses4(s, rows)
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
        )

    @stage
    @cute.jit
    def q_a_tile(self, p, s, seq, tile, tag, q_base: cutlass.Constexpr, mailbox):
        """q_a tile ``tile`` (32 columns) of the four normalized query rows, handed to
        the scorers as Tagged16.
        """
        accum = [Float32(0.0)] * 8
        for page in cutlass.range_constexpr(2):
            s.pages.wait_full(seq + page)
            rt.Memory.fence_proxy_async_shared()
            self.q_a_page_mma(s, accum, Ring.slot_offset(seq + page), page)
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
        """Address of worker slot ``slot``'s split-done counter (``Flags[SPLIT_DONE +
        slot]``).
        """
        return rt.ga(p.Flags, FlagWords.SPLIT_DONE + slot)

    @stage
    @cute.jit
    def scoring_split(self, p, s, cur_pos_c, tag, slot, index_sequence):
        """Worker slot ``slot``'s split of the sparse attention: row ``slot / 32``, keys
        ``slot % 32 * 64`` ...
        """
        query_ = rt.cdiv(slot, 32)
        split = rt.cmod(slot, 32)
        score_count = cutlass.min(cur_pos_c + query_ + 1, 2048)
        if score_count > split * 64:
            firsts = []
            for k in cutlass.range_constexpr(3):
                chunk = s.tid + 256 * k
                if const_expr(k > 0):
                    chunk = cutlass.select_(
                        chunk >= IndexerOperands.CHUNKS, s.tid, chunk
                    )
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
                        cutlass.select_(
                            chunk >= IndexerOperands.Q_CHUNKS, pe_word, q_word
                        ),
                        rt.cdiv(query_, 2),
                    )
                )

            def store(words):
                """The polled chunks into the operand tile."""
                for k in cutlass.range_constexpr(3):
                    IndexerOperands.store(
                        s.act_addr, s.tid + 256 * k, words[2 * k], words[2 * k + 1]
                    )

            self.poll_tagged16(firsts, tag << 16, store)
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
        self.km_merge(p, s, cur_pos_c, slot)

    @stage
    @cute.jit
    def full_scores_vo(self, p, s, compute_start, cur_pos_c, epoch, tag):
        """Full/Push: indexer operands, KM split attention, merge, V/O, O projection, O
        exchange.
        """
        if s.bid < 128:
            self.scoring_split(p, s, cur_pos_c, tag, s.bid, 6 * Int32(s.bid < 82) + 1)
            self.vo_output(p, s, s.bid, tag, epoch)
        if s.bid >= 132:
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
        """Paced L2 prefetch of the listed experts' w13 weights ahead of the fused MoE
        kernel.
        """
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
        """V/O, O projection and O exchange of worker slot ``slot``."""
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
        tag_high = tag << 16
        o_pending = rt.rmem_tensor(8, Uint32)
        o_token = rt.cdiv(s.lane, 8)
        near = 0
        if const_expr(self.shared_or_seam):
            near = Int32(s.role_entry >> Handoff.POOL_SHIFT)

        def o_load(page):
            """Load this lane's 8 pending O words of ``page`` from Handoff."""
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
        if const_expr(self.shared_or_seam):
            epoch = (epoch + 1) * 57 + shared_index - 1
        self.tagged_exchange_final(p, s, p.peer_bases, epoch, slot)

    @stage
    @cute.jit
    def moe_prefetch_tail(self, p, s, slot, start):
        """Worker slots 132..: paced L2 prefetch of the listed experts past the first
        eight.
        """
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
        """V/O tiles of a tail page: page 0 holds 4 tiles, later pages 3."""
        bases = (0, 1024, 116224, 117248) if first else (0, 1024, 2048)
        return self.vo_tile(s, region, vo_lane, bases, vo_k)

    @cute.jit
    def store_selection(self, p, slot, record):
        """One selection record into mailbox ``slot``; kPush also copies it into its
        served peer's mailbox.
        """
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
        """Selection refine: histogram threshold search and the emit window of this
        refine CTA.
        """
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
        """The kPushed compute: install the pushed KV/PE rows, prefetch the listed
        experts, then split, merge, V/O.
        """
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
                            chunk_row = (
                                rt.cdiv(rt.cmod(chunk, 6), 3) * 256
                                + rt.cdiv(chunk, 6) * 16
                            )
                            rt.AsyncCopy.bulk_prefetch_l2(
                                bulk_up
                                + Uint64(expert & 65535) * LAYOUT.w13_expert_bytes
                                + Uint64(
                                    chunk_row * 6144
                                    + rt.cmod(rt.cmod(chunk, 6), 3) * 32768
                                ),
                                Uint32(32768),
                            )
            if vb < 128:
                self.scoring_split(p, s, cur_pos_c, tag, vb, 0)
                self.vo_output(p, s, vb, tag, epoch)

    @cute.jit
    def poll_pushed_record(self, address, generation) -> Uint32:
        """Spin on the u64 pushed record at ``address`` until its high word is
        ``generation``; its payload word.
        """
        record = rt.Poll.word(address, generation, False, True)
        rt.syncwarp()
        return Uint32(record)

    @stage
    @cute.jit
    def shared_query_attention(self, p, s, cur_pos_c, tag):
        """Shared/Seam: query norm and q_a tile, KV latent row, then the CTA's attention
        split and merge.
        """
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
            )
            rt.bar_sync(*BAR_COMPUTE)
            self.release_flag(p, s, FlagWords.KV_READY + role_lo)
        if role_lo < 128:
            self.scoring_split(p, s, cur_pos_c, tag, role_lo, seq6 + 1)

    @stage
    @cute.jit
    def shared_vo_exchange(self, p, s, compute_start, epoch, shared_index_compute, tag):
        """Shared/Seam: V/O tiles, O projection pages, O exchange and the MoE weight
        prefetch.
        """
        slot = Int32(s.role_entry & Handoff.ROLE_MASK)
        if slot < 128:
            self.vo_output(p, s, slot, tag, epoch, shared_index_compute)
        if slot >= 132:
            self.moe_prefetch_tail(p, s, slot, compute_start)

    @stage
    @cute.jit
    def loader_setup(self, p, s):
        """Loader role (warps 8-11): policies, ring slot, epoch ticket and the early
        selection reads.
        """
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
            p.InputWeight.iterator.toint(), 16126080, 16126080
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
            p.KVbWeight.iterator.toint(), 1835648, 1835648
        )
        l2_range_policy_2 = rt.AsyncCopy.createpolicy_evict_first(
            p.OWeight.iterator.toint(), 12586112, 12586112
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
                    self.issue_input_page(p, s, page, rid, input_policy)
            if const_expr(self.full_or_push):
                if (rid >= 132) & (rid < 140):
                    router_on = self.router_enabled(p)
                if router_on != 0:
                    self.issue_query_pages(p, s, 1, 4, True, rid)
            if const_expr(self.shared_or_seam):  # noqa: SIM102
                if (rid >= 132) & (rid < 140):
                    self.issue_query_pages(p, s, 1, 4, False, rid)
        if const_expr(self.pushed):
            early_position = p.CurPos[0]
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
        cur_pos_l = p.CurPos[0]
        gate_seen = Uint32(epoch * 148)
        if const_expr(self.shared_or_seam):  # noqa: SIM102
            if (rid < 128) & (s.tid >= 320):
                local_key = s.tid - 320
                query_ = rt.cdiv(rid, 32)
                split = rt.cmod(rid, 32)
                generation = epoch + 1
                record_index = Int32(
                    (rt.cmod(generation, 2) * 4 + Uint32(query_)) * 2048
                    + Uint32(split * 64)
                    + Uint32(local_key)
                )
                word = rt.Memory.ld_volatile_u64(
                    rt.ga(p.SelectionMailbox, record_index * 2)
                )
                position = cur_pos_l
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
        """Loader role: input/query/absorb ring pages and KV stage 1 (selection ids, PE
        operand, gather4).
        """
        if const_expr(not self.pushed):
            if rid < 128:
                self.load_residual(p, s, rid, 4)
            if rid < 82:
                for page in cutlass.range_constexpr(4, 6):
                    self.issue_input_page(p, s, page, rid, input_policy)
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
            if rid >= 132:
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
                            local[item] = s.moe_keys[
                                pick_row * 256 + s.lane + 32 * item
                            ]
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
                                    s.moe_listed[8 + slot] = 255 - Int32(
                                        winning_key & 255
                                    )
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
                early = (early_even, early_odd) if self.pushed else None
                index_, fresh, record_index = self.probe_ids(
                    p, s, query_, split, context, score_count, generation, early, False
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
        """Loader role: KV stage 2 gathers, KVb/O weight tail pages, merge gate and peer
        mailbox receive.
        """
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
                        rt.baddr(p.KVbWeight, 1835008 + rt.cdiv(kvb_row, 128) * 4 * 4),
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
        """Address of operand word ``word`` in this rank's peer mailbox (kPush writes it
        there).
        """
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
        """Address of worker slot ``slot``'s split-done counter: the counters of the
        slots of tokens 0-1 and tokens 2-3 are in the two pools.
        """
        return self.handoff_word(p, Handoff.SPLIT_DONE + slot, rt.cdiv(slot, 64))

    def role_entry(self, p: WorkerParams, bid):
        """``RoleTable[bid]``: a role in the low 16 bits, the near pool above."""
        return rt.Memory.ldg_nc_u32(rt.ga(p.Handoff, Handoff.ROLE_TABLE + bid))


class SeamWorker(SharedWorker):
    """kSeam: kShared that also owns the seam input XG and picks its role by SM id."""

    ROLE = kSeam
    seam = True
    RECORDS = (HeadArgs, TailArgs, SeamArgs)

    def role_entry(self, p: WorkerParams, bid):
        """``RoleTable[%smid]``."""
        return rt.Memory.ldg_nc_u32(
            rt.ga(p.Handoff, Handoff.ROLE_TABLE + Int32(rt.Sync.smid()))
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
        """Entry kernel with the XG parameter."""
        self.enter(
            WorkerParams.of(
                KvCache,
                head,
                tail,
                seam,
                InputTmap=InputTmap,
                OTmap=OTmap,
                KvbTmap=KvbTmap,
            )
        )
