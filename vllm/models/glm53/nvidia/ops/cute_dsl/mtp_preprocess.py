# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP drafter input projection for ROWS (1 or 4) query rows on one TP8 rank."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass.base_dsl.typing import GridConstant
from cutlass.experimental.cuda.tensor_map import TensorMap

from . import _rt as rt

HIDDEN = rt.Core.HIDDEN
CTAS = 128
CTA_COLS = HIDDEN // CTAS
K_SLICE = 1536
PAGES = 6
PAGE_BYTES, SLOT_BYTES = 24576, 36864
#: One TMA box: 48 rows x 64 columns (128 bytes, swizzled), four per page.
BOX_BYTES, PAGE_BOXES = 6144, 4
EH_COLUMNS = 2 * HIDDEN
_HINT = 0x12F0000000000000
RING_SLOTS = 4
COMPUTE_THREADS, ALL_THREADS = 256, 384
PEERS = rt.Core.RANKS - 1
SMEM_BYTES = 214016


@rt.record
class MtpPreprocessParams(rt.Params):
    """Kernel parameters of ``mtp_preprocess_q{1,4}``."""

    Rows: cute.Pointer = rt.ptr("bfloat16")
    Gamma: cute.Pointer = rt.ptr("float32")
    Weight: cute.Pointer = rt.ptr("bfloat16")
    Position: cute.Pointer = rt.ptr("int32")
    LocalO: cute.Pointer = rt.ptr("bfloat16")
    Final: cute.Pointer = rt.ptr("bfloat16")
    FullEpoch: cute.Pointer = rt.ptr("uint32")
    peer_rank: Int32
    peer_base0: Int64
    peer_base1: Int64
    peer_base2: Int64
    peer_base3: Int64
    peer_base4: Int64
    peer_base5: Int64
    peer_base6: Int64
    peer_base7: Int64


class MtpLayout(rt.SmemLayout):
    """Dynamic shared memory, sized for 4 rows (the 1-row kernel uses the same map)."""

    ROWS = 4
    RING_GAP = 6144

    a = rt.SmemAlloc(SMEM_BYTES)
    ring_full = a.mbarriers(RING_SLOTS, arrivals=1)
    ring_empty = a.mbarriers(RING_SLOTS, arrivals=COMPUTE_THREADS)
    scratch = a.array(Float32, K_SLICE, align=1024)
    epoch = a.array(Uint32, 1)
    local_o = a.array(BFloat16, (ROWS, CTA_COLS), align=256)
    received = a.array(Uint32, (ROWS, PEERS, CTA_COLS // 2))
    norm = a.array(BFloat16, (ROWS, HIDDEN), align=1024)
    a.pad(RING_GAP)
    ring = a.array(Uint8, (RING_SLOTS, SLOT_BYTES), align=1024)
    assert (ring_full.offset, ring_empty.offset) == (0, 32)


L = MtpLayout


def _view(smem: cute.Pointer, region, dtype, shape, stride=None) -> cute.Tensor:
    layout = cute.make_layout(shape, stride=stride)
    return rt.Memory.smem_view(smem, region.offset, dtype, layout)


@rt.record
class MtpSmem:
    """The warp-uniform shared base, its byte pointer and the typed views."""

    addr: Int32
    smem: cute.Pointer
    scratch: cute.Tensor
    epoch_slot: cute.Tensor
    local_o: cute.Tensor
    local_o_words: cute.Tensor
    received: cute.Tensor
    norm_words: cute.Tensor
    c128: cute.CopyAtom

    @staticmethod
    def build(smem_addr, R: int) -> "MtpSmem":
        smem = cute.make_ptr(
            Uint8, smem_addr, cute.AddressSpace.smem, assumed_align=1024
        )
        return MtpSmem(
            addr=smem_addr,
            smem=smem,
            scratch=_view(smem, L.scratch, Float32, K_SLICE),
            epoch_slot=_view(smem, L.epoch, Uint32, 1),
            local_o=_view(smem, L.local_o, BFloat16, L.local_o.size),
            local_o_words=rt.Memory.typed(
                _view(smem, L.local_o, Uint32, L.local_o.size // 2), Uint32
            ),
            received=_view(smem, L.received, Uint32, L.received.size),
            norm_words=_view(smem, L.norm, Uint32, R * HIDDEN // 2),
            c128=cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Uint32, num_bits_per_copy=128
            ),
        )


class MtpPreprocess(rt.RoleKernel):
    """Compute warps 0-7, loader warps 8-11."""

    Params = MtpPreprocessParams
    GRID = CTAS
    SMEM_BYTES = SMEM_BYTES
    MAX_THREADS = True
    ROLES = (rt.Role("compute", 0, 7, 216, True), rt.Role("load", 8, 11, 72, False))
    ROWS = 1
    weight_map = None

    def tensor_maps(self, p) -> tuple:
        row_bytes = 2 * EH_COLUMNS
        dims, strides = [128, row_bytes // 128, HIDDEN], [128, row_bytes]
        return (rt.byte_map(p.Weight.toint(), dims, strides, [128, 1, 48], True),)

    @cute.kernel
    def kernel(self, p, WeightMap: GridConstant[TensorMap]):
        self.weight_map = WeightMap.get_ptr().toint()
        ids, s = self.prologue(p)
        self.run(p, s, ids)

    @cute.jit
    def prologue(self, p: MtpPreprocessParams):
        tid, bid, warp, lane = rt.Prologue.thread_ids()
        _, smem_addr = rt.Prologue.smem()
        s = MtpSmem.build(smem_addr, self.ROWS)
        if warp == 0:
            with cute.arch.elect_one():
                rt.Prologue.init_mbarriers(
                    smem_addr,
                    (
                        (L.ring_full.offset, RING_SLOTS, 1),
                        (L.ring_empty.offset, RING_SLOTS, COMPUTE_THREADS),
                    ),
                )
        cute.arch.sync_threads()
        return rt.ThreadIds(tid, bid, warp, lane), s

    @cute.jit
    def compute(self, p: MtpPreprocessParams, s: MtpSmem, ids: rt.ThreadIds):
        R = self.ROWS
        tid, bid, warp, lane = ids.tid, ids.bid, ids.warp, ids.lane
        peer_rank, c128, scratch, local_o = p.peer_rank, s.c128, s.scratch, s.local_o
        self.set_regs("compute")
        rt.Sync.pdl_prologue()
        rows = cute.make_tensor(
            cute.recast_ptr(p.Rows, dtype=Uint32),
            cute.make_layout((4, 3 * R, 256), stride=(1, 1024, 4)),
        )
        chunks = 3 * R
        raw = cute.make_rmem_tensor((4, chunks), Uint32)
        for c in cutlass.range_constexpr(chunks):
            cute.copy(c128, rows[None, c, tid], raw[None, c])
        gamma = cute.make_tensor(
            rt.Memory.gmem_ptr(Float32, p.Gamma.toint(), 32),
            cute.make_layout((8, 3, 256), stride=(1, 2048, 8)),
        )
        g = cute.make_rmem_tensor((8, 3), Float32)
        c256 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=256
        )
        for c in cutlass.range_constexpr(3):
            cute.copy(c256, gamma[None, c, tid], g[None, c])
        values = cute.make_rmem_tensor((8, 3, R), Float32)
        for q in cutlass.range_constexpr(R):
            for c in cutlass.range_constexpr(3):
                for pp in cutlass.range_constexpr(4):
                    values[2 * pp, c, q], values[2 * pp + 1, c, q] = (
                        rt.Convert.bf16x2_to_f32x2(raw[pp, 3 * q + c])
                    )
        squares = cute.make_rmem_tensor(R, Float32)
        squares.fill(0.0)
        for c in cutlass.range_constexpr(3):
            for pp in cutlass.range_constexpr(4):
                for q in cutlass.range_constexpr(R):
                    v0, v1 = values[2 * pp, c, q], values[2 * pp + 1, c, q]
                    pair = rt.fma(v0, v0, rt.mul(v1, v1))
                    squares[q] = rt.add(squares[q], pair)
        for step in cutlass.range_constexpr(5):
            for q in cutlass.range_constexpr(R):
                squares[q] = rt.add(
                    squares[q], cute.arch.shuffle_sync_down(squares[q], 16 >> step)
                )
        if lane == 0:
            for q in cutlass.range_constexpr(R):
                scratch[q * 8 + warp] = squares[q]
        rt.bar_sync(8, COMPUTE_THREADS)
        inverse = [
            rt.Reduce.rms_inverse(
                rt.Reduce.ordered_sum(lambda w, q=q: scratch[q * 8 + w], 8)
            )
            for q in range(R)
        ]
        norm_out = _view(s.smem, L.norm, Uint32, (4, 256, 3, R), (1, 4, 1024, 3072))
        for c in cutlass.range_constexpr(3):
            normalized = cute.make_rmem_tensor((8, R), Float32)
            for i in cutlass.range_constexpr(8):
                for q in cutlass.range_constexpr(R):
                    normalized[i, q] = rt.mul(
                        values[i, c, q], rt.mul(g[i, c], inverse[q])
                    )
            for q in cutlass.range_constexpr(R):
                packed = cute.make_rmem_tensor(4, Uint32)
                for pp in cutlass.range_constexpr(4):
                    packed[pp] = rt.f32x2_to_bf16x2(
                        normalized[2 * pp, q], normalized[2 * pp + 1, q]
                    )
                cute.copy(c128, packed, norm_out[None, tid, c, q])
        rt.bar_sync(8, COMPUTE_THREADS)
        column = peer_rank % 4 * K_SLICE
        if peer_rank < 4:  # noqa: SIM102
            if cute.make_tensor(p.Position, cute.make_layout(1))[0] == 0:  # noqa: SIM102
                if tid < 192:
                    zero = cute.make_rmem_tensor(4, Uint32)
                    zero.fill(0)
                    norm_v4 = _view(
                        s.smem, L.norm, Uint32, (4, R * HIDDEN // 8), (1, 4)
                    )
                    cute.copy(c128, zero, norm_v4[None, column // 8 + tid])
        rt.bar_sync(8, COMPUTE_THREADS)
        rt.bar_sync(11, ALL_THREADS)
        epoch = Uint32(s.epoch_slot[0])
        group = lane // 4
        lane_pair = lane % 4
        accum = cute.make_rmem_tensor((4, 3), Float32)
        pages = rt.Pipeline.Ring(s.addr, s.addr + L.ring_empty.offset, RING_SLOTS)
        ld_row = lane % 8 + lane // 8 % 2 * 8
        for page in cutlass.range_constexpr(PAGES):
            slot = page % RING_SLOTS
            pages.wait_full(page)
            cute.arch.fence_proxy("async.shared", space="cta")
            for kt in cutlass.range_constexpr(2):
                k_tile = warp * 2 + kt
                box = (
                    s.addr + L.ring.offset + slot * SLOT_BYTES + k_tile // 4 * BOX_BYTES
                )
                unit = k_tile % 4 * 2 + lane // 16
                b = cute.make_rmem_tensor(2, Uint32)
                b[0] = Uint32(0)
                b[1] = Uint32(0)
                if group < R:
                    kk = column + page * 256 + k_tile * 16 + lane_pair * 2
                    b[0] = s.norm_words[group * 3072 + kk // 2]
                    b[1] = s.norm_words[group * 3072 + kk // 2 + 4]
                for m in cutlass.range_constexpr(3):
                    row = m * 16 + ld_row
                    a = rt.TensorCore.ldsm_x4(box + row * 128 + (unit ^ row % 8) * 16)
                    rt.TensorCore.mma_acc(accum[None, m], a, b, page + kt == 0)
            pages.release(slot)
        if lane_pair * 2 < R:
            for m in cutlass.range_constexpr(3):
                at = (warp * R + lane_pair * 2) * 48 + m * 16 + group
                scratch[at] = accum[0, m]
                scratch[at + 8] = accum[2, m]
                if lane_pair * 2 + 1 < R:
                    scratch[at + 48] = accum[1, m]
                    scratch[at + 48 + 8] = accum[3, m]
        rt.bar_sync(8, COMPUTE_THREADS)
        if tid < 48 * R:
            total = rt.Reduce.ordered_sum(lambda w: scratch[w * R * 48 + tid], 8)
            local_o[tid] = total.to(BFloat16)
            # A subscript-store target must be a plain name (the DSL marks it written).
            local_o_out = cute.make_tensor(p.LocalO, cute.make_layout(R * HIDDEN))
            local_o_out[tid // 48 * HIDDEN + bid * 48 + tid % 48] = total.to(BFloat16)
        rt.bar_sync(8, COMPUTE_THREADS)
        if tid < 56 * R:
            token, packed_peer, chunk = tid // 56, tid // 8 % 7, tid % 8
            rt.OExchange.push(
                s.local_o_words,
                p.peer_bases,
                peer_rank,
                epoch,
                token,
                packed_peer,
                chunk,
                bid,
            )
        rt.bar_sync(11, ALL_THREADS)
        query_row = warp // 2
        if (warp % 2 == 0) & (lane < 12) & (query_row < R):
            received = (L.received.offset - L.local_o.offset) // 4
            local_o_addr = s.addr + L.local_o.offset
            rt.OExchange.reduce(
                s.local_o_words, local_o_addr, peer_rank, query_row, lane, received
            )
        rt.bar_sync(8, COMPUTE_THREADS)
        if tid < 48:
            final = cute.make_tensor(p.Final, cute.make_layout(R * HIDDEN))
            for t in cutlass.range_constexpr(R):
                final[t * HIDDEN + bid * 48 + tid] = local_o[t * 48 + tid]

    @cute.jit
    def load(self, p: MtpPreprocessParams, s: MtpSmem, ids: rt.ThreadIds):
        R = self.ROWS
        tid, bid, warp = ids.tid, ids.bid, ids.warp
        epoch_slot, received = s.epoch_slot, s.received
        self.set_regs("load")
        if warp == 9:
            with cute.arch.elect_one():
                epoch_slot[0] = cute.arch.atomic_add(p.FullEpoch, Uint32(1)) // CTAS
        first_chunk = p.peer_rank * (K_SLICE // 64)
        for page in cutlass.range_constexpr(PAGES):
            if cutlass.const_expr(page == 4):
                rt.bar_sync(11, ALL_THREADS)
                epoch = Uint32(epoch_slot[0])
            slot = page % RING_SLOTS
            if warp == 8:
                with cute.arch.elect_one():
                    rt.mbar_wait(
                        s.addr + L.ring_empty.offset + 8 * slot, page // 4 % 2 ^ 1
                    )
                    full = s.addr + 8 * slot
                    rt.mbar_expect_tx(full, PAGE_BYTES)
                    dst = s.addr + L.ring.offset + slot * SLOT_BYTES
                    for c in cutlass.range_constexpr(PAGE_BOXES):
                        chunk = first_chunk + page * PAGE_BOXES + c
                        rt.AsyncCopy.tma_3d(
                            dst + c * BOX_BYTES,
                            self.weight_map,
                            0,
                            chunk,
                            bid * 48,
                            full,
                            _HINT,
                        )
        lane_0 = tid - COMPUTE_THREADS
        parity = Int32((epoch + 1) % 2)
        mailbox = rt.TaggedMailbox.peer_base(p.peer_rank, p.peer_bases)
        tag8 = rt.TaggedMailbox.tag8(epoch)
        if cutlass.const_expr(R == 1):
            if lane_0 < 56:
                rt.OExchange.receive(
                    received, mailbox, parity, tag8, bid, lane_0, [lane_0], 56
                )
        else:
            item_0 = lane_0 + 128
            if item_0 >= 224:
                item_0 = lane_0
            rt.OExchange.receive(
                received, mailbox, parity, tag8, bid, lane_0, [lane_0, item_0], 224
            )
        rt.bar_sync(11, ALL_THREADS)
