# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP top-k candidate exchange over NVLink and its exact merge (K3.1 DSA decode).

The #63 one-round merge (``vllm/v1/attention/ops/dcp_topk_merge``) all-gathers
every rank's local top-N B' rows with NCCL (a 1-CTA ring all-gather of ~19 us
at DCP8 plus its launch gap), then selects (one kernel) and rescues saturated
rows (a second kernel). Here, instead:

  push   ``IndexerLocalTopk``'s epilogue (indexer.py, ``exchange``) stores its
         row's B' words into every rank's symmetric ``TopkExchange`` slot
         [parity, source rank, row] as NCCL-LL pairs: each 16-byte store
         carries two words and the call's epoch twice, ``(w0, e, w1, e)``, so
         a reader checks each 8-byte half's epoch instead of a flag behind a
         system fence.
  merge  ``ExchangeMerge`` (one 512-thread CTA per row, launched early by PDL)
         polls the W ranks' LL words of its row, then runs the #63 merge's
         own select (``_make_radix_select``: the same keys, threshold and tie
         rule, so the same ids and rows in the same order), its saturation
         word and saturated-rank push (the rescue workspace of
         ``init_dcp_topk_merge``, the same seq2 / flag protocol), and the
         rescue's second round in the same CTA.

The epoch of row t is ``epoch[t] + 1`` (``epoch``: per rank, bumped by the
merge after it read the row); every rank runs the same calls over the same
rows, so the epochs agree. The slot parity is the epoch's low bit: a rank can
store call n + 1's words while a peer still reads call n's.
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint16, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from vllm.cute_utils import recast_val
from vllm.v1.attention.ops.dcp_topk_merge import (
    LOCAL_INDEX_BITS,
    packed_row_words,
    rescue_slots,
)

from ..common.host import dynamic_fake, dynamic_view, stream
from ..common.primitives import load_words, store_words

TB = 512  # the merge's threads (the #63 select's, kernels._TB_SIZE)
_WARP = 32


def ll_pairs(local_topk: int) -> int:
    """16-byte LL stores per B' row (two words each)."""
    return -(-packed_row_words(local_topk) // 2)


@cute.jit
def ll_push(dst: Int64, w0, w1, epoch: Uint32):
    """One LL store: words ``w0``, ``w1`` tagged with ``epoch``."""
    store_words(dst, (Uint32(w0), epoch, Uint32(w1), epoch), scope="sys")


def ll_offset(j):
    """Byte offset of B' word ``j`` in a row's LL pairs."""
    return Int64(j >> 1) * 16 + Int64(j & 1) * 8


class TopkExchange:
    """The symmetric LL slots of the exchange over the DCP ``group``: int32
    [2, W, rows, 4 * ll_pairs(N)] (parity, source rank, row, (word, epoch)
    pairs), every rank's address in ``peers``, and this rank's per-row
    ``epoch``. One per device, shared by every DSA layer."""

    def __init__(self, group, device, world, rank, rows, local_topk):
        import torch.distributed._symmetric_memory as symm_mem

        from ....specialized_decode.symm import peer_rendezvous

        self.world, self.rank, self.rows, self.local_topk = (
            world,
            rank,
            rows,
            local_topk,
        )
        self.pairs = ll_pairs(local_topk)
        shape = (2, world, rows, 4 * self.pairs)
        self.recv = symm_mem.empty(shape, dtype=torch.int32, device=device)
        # Peer stores only: no multicast object.
        handle = peer_rendezvous(self.recv, group)
        i64 = dict(dtype=torch.int64, device=device)
        self.peers = torch.tensor(handle.buffer_ptrs, **i64)
        self.recv.zero_()  # epoch 0 never matches (the first call's is 1)
        self.epoch = torch.zeros(rows, dtype=torch.int32, device=device)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)


_EXCHANGES: dict = {}


def reserve(group, device, world, rank, rows, local_topk):
    """The device's ``TopkExchange`` of at least ``rows`` rows (collective on
    first use: every DCP rank builds its DSA layers in the same order)."""
    key = (torch.device(device), local_topk)
    ex = _EXCHANGES.get(key)
    if ex is None or ex.rows < rows:
        ex = TopkExchange(group, device, world, rank, rows, local_topk)
        _EXCHANGES[key] = ex
    return ex


@functools.cache
def build_merge(rows, topk, local_topk, world, block_size, emit_ids):
    """``ExchangeMerge`` of ``rows`` rows (see the module docstring)."""
    from vllm.v1.attention.ops.dcp_topk_merge import kernels as stock

    assert world <= 16 and local_topk < topk and block_size > 0
    assert stock._TB_SIZE == TB
    n, tb = local_topk, TB
    num_bins, num_warps = stock._NUM_BINS, TB // _WARP
    pairs = ll_pairs(n)
    num_r1 = world * n
    kpt = -(-num_r1 // tb)
    num_slots = rescue_slots(topk, n)
    recv_words = packed_row_words(topk)
    num_keys2 = num_r1 + num_slots * topk
    kpt2 = -(-num_keys2 // tb)
    gid_bits = stock._gid_bits(world)
    select_args = dict(topk=topk, world_size=world, emit_ids=emit_ids)
    select_args["block_size"] = block_size
    select = stock._make_radix_select(keys_per_thread=kpt, **select_args)
    rescue = stock._make_radix_select(keys_per_thread=kpt2, **select_args)

    @cute.jit
    def ll_word(address: Int64, epoch: Uint32) -> Uint32:
        """The word of the LL half at ``address`` once its tag is ``epoch``."""
        word, tag = load_words(address, count=2, volatile=True)
        if tag != epoch:
            start = stock._globaltimer()
            while tag != epoch:  # no back-off: it measured slower (GB300 DCP8)
                word, tag = load_words(address, count=2, volatile=True)
                if stock._globaltimer() - start > Int64(stock._SPIN_TIMEOUT_NS):
                    cute.printf("DCP top-k exchange timeout epoch=%d\n", epoch)
                    stock._trap()
        return word

    @cute.struct
    class SharedStorage:
        hist: cute.struct.MemRange[Int32, num_bins]
        warp_totals: cute.struct.MemRange[Int32, num_warps]
        scalars: cute.struct.MemRange[Int32, 8]
        notsat: cute.struct.MemRange[Int32, 16]
        owner: cute.struct.MemRange[Int32, 16]
        prefix: cute.struct.Align[cute.struct.MemRange[Uint64, 1], 8]
        push: cute.struct.Align[cute.struct.MemRange[Int32, recv_words], 16]

    @cute.jit
    def push_local_topk(
        row, tid, lane, warp, hist, warp_totals, scalars, buf, logits, row_ends,
        slot, dcp_rank, seq2, peer_recv, peer_flags, max_rows,
    ):  # fmt: skip
        """The #63 select's push of a saturated rank (DCPTopkSelectKernel,
        ``push_local_topk``, verbatim): this rank's local top-min(K, valid)
        of ``logits[row]`` in the B' layout -> every rank's
        ``recv[row, slot]``; then the flag."""
        end = Int32(logits.shape[1])
        if end > Int32(1 << LOCAL_INDEX_BITS):
            end = Int32(1 << LOCAL_INDEX_BITS)
        if row_ends[row] < end:
            end = row_ends[row]
        hist[tid] = Int32(0)
        if tid == Int32(0):
            scalars[4] = Int32(-1)
            scalars[5] = Int32(0)
        cute.arch.sync_threads()
        for loc_idx in cutlass.range(tid, end, tb, unroll=1):
            bits = Uint32(recast_val(logits[row, loc_idx], Uint16))
            if (bits & Uint32(0x7FFF)) <= Uint32(0x7F80):
                b = Int32(stock._ord16(bits) >> Uint32(7))
                cute.arch.atomic_add(
                    hist.iterator + b, Int32(1), sem="relaxed", scope="cta"
                )
        cute.arch.sync_threads()
        count = hist[Int32(num_bins - 1) - tid]
        incl = stock._block_scan_inclusive(count, lane, warp, warp_totals, num_warps)
        if count > Int32(0) and incl >= Int32(topk) and incl - count < Int32(topk):
            scalars[4] = Int32(num_bins - 1) - tid
            scalars[5] = incl - count
        cute.arch.sync_threads()
        b1 = scalars[4]
        above = scalars[5]
        v = Uint32(0)
        c_gt = Int32(0)
        if b1 >= Int32(0):
            hist[tid] = Int32(0)
            cute.arch.sync_threads()
            for loc_idx in cutlass.range(tid, end, tb, unroll=1):
                bits = Uint32(recast_val(logits[row, loc_idx], Uint16))
                if (bits & Uint32(0x7FFF)) <= Uint32(0x7F80):
                    o = stock._ord16(bits)
                    if Int32(o >> Uint32(7)) == b1:
                        cute.arch.atomic_add(
                            hist.iterator + Int32(o & Uint32(127)),
                            Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
            cute.arch.sync_threads()
            count2 = hist[Int32(num_bins - 1) - tid]
            incl2 = stock._block_scan_inclusive(
                count2, lane, warp, warp_totals, num_warps
            )
            need2 = Int32(topk) - above
            if count2 > Int32(0) and incl2 >= need2 and incl2 - count2 < need2:
                scalars[6] = Int32(num_bins - 1) - tid
                scalars[7] = above + incl2 - count2
            cute.arch.sync_threads()
            v = (Uint32(b1) << Uint32(7)) | Uint32(scalars[6])
            c_gt = scalars[7]
        for i in cutlass.range(tid, recv_words, tb, unroll=1):
            buf[i] = Int32(0)
        if tid == Int32(0):
            scalars[5] = Int32(0)
        cute.arch.sync_threads()
        eq_run = Int32(0)
        gt_run = Int32(0)
        for base in cutlass.range(0, end, tb, unroll=1):
            loc_idx = base + tid
            o = Uint32(0)
            if loc_idx < end:
                bits = Uint32(recast_val(logits[row, loc_idx], Uint16))
                if (bits & Uint32(0x7FFF)) <= Uint32(0x7F80):
                    o = stock._ord16(bits)
            is_eq = o == v and o != Uint32(0)
            flag = Int32(0)
            if is_eq:
                flag = Int32(1)
            if o > v:
                flag = Int32(1 << 16)
            incl = stock._block_scan_inclusive(flag, lane, warp, warp_totals, num_warps)
            if tid == Int32(tb - 1):
                scalars[6] = incl
            cute.arch.sync_threads()
            eq_incl = incl & Int32(0xFFFF)
            pos = Int32(-1)
            if o > v:
                pos = gt_run + (incl >> Int32(16)) - Int32(1)
            if is_eq:
                idx = eq_run + eq_incl - Int32(1)
                if idx < Int32(topk) - c_gt:
                    pos = c_gt + idx
            if pos >= Int32(0):
                l_u = recast_val(loc_idx, Uint32)
                buf[pos] = recast_val(
                    (o << Uint32(16)) | (Uint32(0xFFFF) - (l_u & Uint32(0xFFFF))),
                    Int32,
                )
                if (l_u >> Uint32(16)) != Uint32(0):
                    cute.arch.atomic_or(
                        buf.iterator + (Int32(topk) + (pos >> Int32(5))),
                        Int32(1) << (pos & Int32(31)),
                        sem="relaxed",
                        scope="cta",
                    )
            eq_run += scalars[6] & Int32(0xFFFF)
            gt_run += scalars[6] >> Int32(16)
            cute.arch.sync_threads()
        row_offset = (Int64(row) * Int64(num_slots) + Int64(slot)) * Int64(
            recv_words * 4
        )
        for p in cutlass.range_constexpr(world):
            dst = stock._ld_i64(peer_recv + Int64(p * 8)) + row_offset
            for vi in cutlass.range(tid, recv_words // 4, tb, unroll=1):
                stock._st_v4(
                    dst + Int64(vi) * Int64(16),
                    buf[vi * 4],
                    buf[vi * 4 + 1],
                    buf[vi * 4 + 2],
                    buf[vi * 4 + 3],
                )
        cute.arch.sync_threads()
        if tid == Int32(0):
            cute.arch.fence_acq_rel_sys()
            flag = seq2[row] + Int32(1)
            flag_offset = (Int64(dcp_rank) * Int64(max_rows) + Int64(row)) * Int64(4)
            for p in cutlass.range_constexpr(world):
                stock._st_release_sys(
                    stock._ld_i64(peer_flags + Int64(p * 8)) + flag_offset, flag
                )

    @cute.kernel
    def device_kernel(
        recv: Int64,
        epoch: cute.Tensor,
        xrows: Int32,
        out: cute.Tensor,
        req_id: cute.Tensor,
        block_table: cute.Tensor,
        rows_out: cute.Tensor,
        valid_out: cute.Tensor,
        dcp_rank: Int32,
        stride_rows: Int32,
        sat_word: cute.Tensor,
        seq2: cute.Tensor,
        logits: cute.Tensor,
        row_ends: cute.Tensor,
        peer_recv: Int64,
        peer_flags: Int64,
        max_rows: Int32,
        rescue_recv: Int64,
        flags: Int64,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage, 16)
        hist = storage.hist.get_tensor(cute.make_layout((num_bins,)))
        warp_totals = storage.warp_totals.get_tensor(cute.make_layout((num_warps,)))
        scalars = storage.scalars.get_tensor(cute.make_layout((8,)))
        notsat = storage.notsat.get_tensor(cute.make_layout((16,)))
        owner = storage.owner.get_tensor(cute.make_layout((16,)))
        prefix_ptr = storage.prefix.data_ptr()

        # Round 1: the W ranks' B' words of this row, once their epoch landed.
        ep = epoch[row] + Int32(1)
        tag = recast_val(ep, Uint32)
        part = Int64(ep & Int32(1))
        keys = cute.make_rmem_tensor((kpt,), Uint64)
        for i in cutlass.range_constexpr(kpt):
            flat = tid + Int32(i * tb)
            key = Uint64(0)
            if flat < Int32(num_r1):
                rank = flat // Int32(n)
                j = flat - rank * Int32(n)
                slot = (part * Int64(world) + Int64(rank)) * Int64(xrows) + Int64(row)
                base = recv + slot * Int64(pairs * 16)
                word = ll_word(base + ll_offset(j), tag)
                high = ll_word(base + ll_offset(Int32(n) + (j >> Int32(5))), tag)
                key = stock._decode_key(
                    word, high, j, recast_val(rank, Uint32), world, gid_bits
                )
            keys[i] = key
        if tid < Int32(16):
            notsat[tid] = Int32(0)

        sel_min = select(
            keys, row, tid, lane, warp, hist, warp_totals, scalars, prefix_ptr,
            out, req_id, block_table, rows_out, valid_out, dcp_rank, stride_rows,
        )  # fmt: skip

        # The #63 select's saturation word (verbatim).
        wmin = cute.arch.warp_redux_sync(sel_min, "min")
        hist[tid] = Int32(0x10000)
        cute.arch.sync_threads()
        if lane == Int32(0):
            hist[warp] = wmin
        cute.arch.sync_threads()
        tau = Int32(0x10000)
        for w in cutlass.range_constexpr(num_warps):
            if hist[w] < tau:
                tau = hist[w]
        if scalars[0] < Int32(topk):
            tau = Int32(0)
        for i in cutlass.range_constexpr(kpt):
            flat = tid + Int32(i * tb)
            if flat < Int32(num_r1):
                key = keys[i]
                above = key != Uint64(0) and Int32(key >> Uint64(gid_bits)) > tau
                ballot = recast_val(cute.arch.vote_ballot_sync(above), Uint32)
                if lane == Int32(0) and ballot != Uint32(0xFFFFFFFF):
                    notsat[flat // Int32(n)] = Int32(1)
        cute.arch.sync_threads()
        mask = Int32(0)
        for r in cutlass.range_constexpr(world):
            if notsat[r] == Int32(0):
                mask = mask | Int32(1 << r)
        num_sat = recast_val(cute.arch.popc(recast_val(mask, Uint32)), Int32)
        if num_sat > Int32(num_slots):
            mask = Int32(0)
        if tid == Int32(0):
            sat_word[row] = (mask << Int32(16)) | tau
        if ((mask >> dcp_rank) & Int32(1)) != Int32(0):
            # The logits (IndexerScore, before IndexerLocalTopk): wait for them.
            cute.arch.griddepcontrol_wait()
            slot = recast_val(
                cute.arch.popc(
                    recast_val(mask & ((Int32(1) << dcp_rank) - Int32(1)), Uint32)
                ),
                Int32,
            )
            push_buf = storage.push.get_tensor(cute.make_layout((recv_words,)))
            push_local_topk(
                row, tid, lane, warp, hist, warp_totals, scalars, push_buf, logits,
                row_ends, slot, dcp_rank, seq2, peer_recv, peer_flags, max_rows,
            )  # fmt: skip

        # Round 2 (the #63 rescue, DCPTopkRescueKernel, for this row): the
        # other ranks' round-1 keys at or above tau16 plus the saturated
        # ranks' pushed local top-K lists.
        if mask != Int32(0):
            expected = seq2[row] + Int32(1)
            if tid == Int32(0):
                k = Int32(0)
                for r in cutlass.range_constexpr(world):
                    if ((mask >> Int32(r)) & Int32(1)) != Int32(0):
                        owner[k] = Int32(r)
                        k += Int32(1)
            if (
                warp == Int32(0)
                and lane < Int32(world)
                and ((mask >> lane) & Int32(1)) != Int32(0)
            ):
                flag_addr = flags + (
                    Int64(lane) * Int64(max_rows) + Int64(row)
                ) * Int64(4)
                start = stock._globaltimer()
                while stock._ld_acquire_sys(flag_addr) != expected:
                    stock._nanosleep()
                    if stock._globaltimer() - start > Int64(stock._SPIN_TIMEOUT_NS):
                        cute.printf(
                            "DCP top-k exchange rescue timeout row=%d rank=%d\n",
                            row,
                            lane,
                        )
                        stock._trap()
            cute.arch.sync_threads()
            keys2 = cute.make_rmem_tensor((kpt2,), Uint64)
            row_base = rescue_recv + Int64(row) * Int64(num_slots * recv_words * 4)
            for i in cutlass.range_constexpr(kpt2):
                flat = tid + Int32(i * tb)
                key = Uint64(0)
                if flat < Int32(num_r1):
                    rank = flat // Int32(n)
                    if cutlass.const_expr(i < kpt):  # noqa: SIM102 (flat < num_r1 only below kpt)
                        if ((mask >> rank) & Int32(1)) == Int32(0):
                            key = keys[i]
                            if Int32(key >> Uint64(gid_bits)) < tau:
                                key = Uint64(0)
                elif flat < Int32(num_keys2):
                    s = (flat - Int32(num_r1)) // Int32(topk)
                    if s < num_sat:
                        j = flat - Int32(num_r1) - s * Int32(topk)
                        base = row_base + Int64(s) * Int64(recv_words * 4)
                        key = stock._decode_key(
                            stock._ld_u32(base + Int64(j) * Int64(4)),
                            stock._ld_u32(
                                base + Int64(Int32(topk) + (j >> Int32(5))) * Int64(4)
                            ),
                            j,
                            recast_val(owner[s], Uint32),
                            world,
                            gid_bits,
                        )
                keys2[i] = key
            rescue(
                keys2, row, tid, lane, warp, hist, warp_totals, scalars, prefix_ptr,
                out, req_id, block_table, rows_out, valid_out, dcp_rank, stride_rows,
            )  # fmt: skip
            cute.arch.sync_threads()
            if tid == Int32(0):
                seq2[row] = expected
        # Every thread has read this row's words: the next call's epoch.
        # Complete only after IndexerLocalTopk (PDL wait): its CTAs may still be
        # pushing to the other ranks (reading their B' rows) when this rank's
        # words have all landed, and the kernels after this one wait for this
        # grid alone; without the wait, the next layer's local select could
        # rewrite those B' rows under the tail of this one's push.
        cute.arch.griddepcontrol_wait()
        cute.arch.sync_threads()
        if tid == Int32(0):
            epoch[row] = ep

    @cute.jit
    def host_entrypoint(
        recv: Int64,
        epoch: cute.Tensor,
        xrows: Int32,
        out: cute.Tensor,
        req_id: cute.Tensor,
        block_table: cute.Tensor,
        rows_out: cute.Tensor,
        valid_out: cute.Tensor,
        dcp_rank: Int32,
        stride_rows: Int32,
        sat_word: cute.Tensor,
        seq2: cute.Tensor,
        logits: cute.Tensor,
        row_ends: cute.Tensor,
        peer_recv: Int64,
        peer_flags: Int64,
        max_rows: Int32,
        rescue_recv: Int64,
        flags: Int64,
        s,
    ):
        device_kernel(
            recv, epoch, xrows, out, req_id, block_table, rows_out, valid_out,
            dcp_rank, stride_rows, sat_word, seq2, logits, row_ends, peer_recv,
            peer_flags, max_rows, rescue_recv, flags,
        ).launch(
            grid=(rows, 1, 1), block=(tb, 1, 1), stream=s, use_pdl=True
        )  # fmt: skip

    return host_entrypoint


@functools.cache
def compile_merge(rows, topk, local_topk, world, block_size, emit_ids, device):
    """The compiled ``ExchangeMerge`` of ``rows`` rows (shared by every layer)."""
    entry = build_merge(rows, topk, local_topk, world, block_size, emit_ids)
    i32 = torch.int32
    vec, mat = dynamic_fake(1, i32), dynamic_fake(2, i32)
    with torch.accelerator.device_index(device):
        return cute.compile(
            entry,
            Int64(0),
            vec,
            Int32(1),
            mat,
            vec,
            mat,
            mat,
            vec,
            Int32(0),
            Int32(1),
            vec,
            vec,
            dynamic_fake(2, torch.bfloat16),
            vec,
            Int64(0),
            Int64(0),
            Int32(1),
            Int64(0),
            Int64(0),
            make_fake_stream(),
        )


class ExchangeMerge:
    """Host side of the exchange merge of one (rows, block size)."""

    def __init__(self, exchange, workspace, rows, topk, block_size, emit_ids=False):
        assert rows <= exchange.rows and rows <= workspace.max_rows
        assert (workspace.topk, workspace.local_topk) == (topk, exchange.local_topk)
        assert (workspace.world_size, workspace.rank) == (exchange.world, exchange.rank)
        self.exchange, self.ws, self.rows = exchange, workspace, rows
        args = rows, topk, exchange.local_topk, exchange.world, block_size, emit_ids
        self.kernel = compile_merge(*args, exchange.recv.device.index)
        self.emit_ids = emit_ids

    def __call__(
        self, logits, row_ends, req_id, block_table, stride_rows, rows_out,
        valid_out, ids_out=None,
    ):  # fmt: skip
        """Every rank's pushed B' rows of the ``rows`` rows -> this rank's
        compacted KV rows ``rows_out`` [rows, >= K], counts ``valid_out``
        and, compiled with ``emit_ids``, the global ids ``ids_out`` [rows, K].
        ``logits`` / ``row_ends``: this rank's (a saturated rank re-reads
        its row), bounded as the #63 merge's (``_bound_logits``)."""
        ex, ws, rows = self.exchange, self.ws, self.rows
        assert rows_out.shape[0] >= rows and valid_out.shape[0] >= rows
        assert (ids_out is not None) == self.emit_ids
        out = ids_out if ids_out is not None else rows_out
        self.kernel(
            Int64(ex.recv.data_ptr()),
            dynamic_view(ex.epoch),
            Int32(ex.rows),
            dynamic_view(out),
            dynamic_view(req_id),
            dynamic_view(block_table),
            dynamic_view(rows_out),
            dynamic_view(valid_out),
            Int32(ex.rank),
            Int32(stride_rows),
            dynamic_view(ws.sat_word),
            dynamic_view(ws.seq2),
            dynamic_view(logits),
            dynamic_view(row_ends),
            Int64(ws.peer_recv.data_ptr()),
            Int64(ws.peer_flags.data_ptr()),
            Int32(ws.max_rows),
            Int64(ws.recv.data_ptr()),
            Int64(ws.flags.data_ptr()),
            stream(row_ends.device),
        )
