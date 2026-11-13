# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP top-k candidate exchange over NVLink and its exact merge (K3.1 DSA decode).

The #63 one-round merge (`vllm/models/kimi_k3/nvidia/ops/dcp_topk_merge`) all-gathers
every rank's local top-N B' rows with NCCL (a 1-CTA ring all-gather of ~19 us
at DCP8 plus its launch gap), then selects (one kernel) and rescues saturated
rows (a second kernel). Here, instead:

  push   `IndexerLocalTopk`'s epilogue (indexer.py, `exchange`) stores its
         row's B' words into every rank's symmetric `TopkExchange` slot
         [parity, source rank, row] as NCCL-LL pairs: each 16-byte store
         carries two words and the call's epoch twice, `(w0, e, w1, e)`, so
         a reader checks each 8-byte half's epoch instead of a flag behind a
         system fence. With `LL64` each half is one 64-bit element (the word
         in the low 32 bits, the epoch in the high) of a `v2.b64` store, read
         with one 64-bit load, both relaxed at system scope: the PTX memory
         model treats a vector access as one access per element, and two
         morally strong accesses to the same 8 bytes are single-copy atomic,
         so a half whose epoch matches holds that call's word. (The 32-bit
         form stores and loads the word and the epoch as separate elements.)
         Both forms write the same bytes.
  merge  `ExchangeMerge` (one 512-thread CTA per row, launched early by PDL)
         polls the W ranks' LL words of its row, then runs the #63 merge's
         own select (`_make_radix_select`: the same keys, threshold and tie
         rule, so the same ids and rows in the same order), its saturation
         word and saturated-rank push (the rescue workspace of
         `init_dcp_topk_merge`, the same seq2 / flag protocol), and the
         rescue's second round in the same CTA.

`MERGE_SELECT` picks, per DCP world size and rows, the "digits" form of the
merge's first round instead (the same rows, counts, saturation word and rescue,
bit for bit; laid out for latency): warp w of the CTA owns source rank w. It
issues every LL load of that rank before it checks a tag, so each rank's words
are decoded as they land, and adds the keys to an 11-bit histogram of the
score; two digit scans (11 bits over all warps, then the low 5 bits) give the
K-th score. Ties at it are ranked by global id: by ballots against the listed
ties when there are few, else by a block-wide two-digit select of the tie ids.
Only the warp of this rank's own source then writes the rows: the #63 select
writes a thread's keys in thread order and this rank's N keys sit on N
consecutive flat slots (thread = slot mod 512, N <= 512), so their order is a
rotation of the slot order. A row with saturated ranks moves the round-1 keys
into the #63 thread order and runs the same rescue code.

The epoch of row t is `epoch[t] + 1` (`epoch`: per rank, bumped by the
merge after it read the row); every rank runs the same calls over the same
rows, so the epochs agree. The slot parity is the epoch's low bit: a rank can
store call n + 1's words while a peer still reads call n's.
"""

import functools
import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint16, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from vllm.cute_utils import recast_val
from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge import (
    LOCAL_INDEX_BITS,
    packed_row_words,
    rescue_slots,
)

from ..common.host import dynamic_fake, dynamic_view, stream
from ..common.policy import batch_table
from ..common.primitives import asm, load_words, smem_words, store_words

TB = 512  # the merge's threads (the #63 select's, kernels._TB_SIZE)
_WARP = 32
# Each LL half one 64-bit access (see the module docstring); "0": the 32-bit
# form. The same bytes either way.
LL64 = os.environ.get("VLLM_KIMI_K3_DSA_LL64", "1") != "0"
# The merge's first round per DCP world size and rows: "digits" (see the module
# docstring) or "radix" (the #63 select, every other shape). Set where "digits"
# shortened the merge's tail after the push in a one-GPU A/B of the LocalTopk
# push -> merge chain (DCP16, 8 rows, 32K-512K contexts, the first or the last
# source rank landing last).
MERGE_SELECT = {16: "1-8:digits"}
DIGIT_BINS = 2048  # "digits": the score's top 11 bits
DIGIT_LOW = 32  # then its low 5 bits
TIE_LOW_BITS = 10  # tie ids: a (gid bits - 10)-bit digit, then a 10-bit one
TIE_REGS = 8  # ties ranked by ballots: up to 32 per register
# This rank's ties ranked by ballots, up to this many; past 256 ties or this many
# of its own, the block-wide select (a fixed cost, where the ballots cost per tie).
# Tunable; set from a one-GPU sweep.
TIE_OWN = 8
TABLE_CAP = 1024  # block-table columns staged in shared memory


def merge_select(rows, topk, local_topk, world, emit_ids) -> str:
    """The first round of the merge of `rows` rows: "digits" where
    `MERGE_SELECT` names it and the shape fits it (one warp per source rank,
    an even number of 32-pair groups per rank, a rank's keys within one 512-key
    stride, the rows only), else "radix"."""
    kind = batch_table(MERGE_SELECT.get(world, "")).get(rows, "radix")
    fits = (
        world == TB // _WARP
        and local_topk % 64 == 0
        and local_topk <= TB
        and local_topk < topk < world * local_topk
        and packed_row_words(topk) >= TABLE_CAP + (1 << TIE_LOW_BITS)
    )
    return "digits" if kind == "digits" and fits and not emit_ids else "radix"


def ll_pairs(local_topk: int) -> int:
    """16-byte LL stores per B' row (two words each)."""
    return -(-packed_row_words(local_topk) // 2)


@asm()
def store_ll_pairs(pointer, w0, w1, epoch):
    """Halves (w0, epoch), (w1, epoch) as the two 64-bit elements of one 16-byte
    relaxed system-scope store (the word in each element's low 32 bits)."""
    code = (
        "{ .reg .b64 ll0, ll1; mov.b64 ll0, {$1, $3}; mov.b64 ll1, {$2, $3}; "
        "st.relaxed.sys.global.v2.b64 [$0], {ll0, ll1}; }"
    )
    operands = (pointer, Uint32(w0), Uint32(w1), Uint32(epoch))
    return code, "l,r,r,r,~{memory}", operands


@asm()
def load_ll_half(pointer):
    """(word, epoch) of one LL half: one 64-bit relaxed system-scope load."""
    code = "{ .reg .b64 ll; ld.relaxed.sys.global.b64 ll, [$2]; mov.b64 {$0, $1}, ll; }"
    return code, "=r,=r,l,~{memory}", (pointer,), (Uint32, Uint32)


def ll_load(address):
    """(word, epoch) of the LL half at `address`."""
    if LL64:
        return load_ll_half(address)
    return load_words(address, count=2, volatile=True)


@asm()
def load_ll_pair(pointer):
    """(w0, epoch, w1, epoch) of one LL store: one 16-byte relaxed system-scope
    load of its two 64-bit elements."""
    code = (
        "{ .reg .b64 ll0, ll1; ld.relaxed.sys.global.v2.b64 {ll0, ll1}, [$4]; "
        "mov.b64 {$0, $1}, ll0; mov.b64 {$2, $3}, ll1; }"
    )
    return code, "=r,=r,=r,=r,l,~{memory}", (pointer,), (Uint32,) * 4


def ll_load_pair(address):
    """(w0, epoch, w1, epoch) of the LL store at `address`."""
    if LL64:
        return load_ll_pair(address)
    return load_words(address, count=4, volatile=True)


@asm(Uint32)
def shared_fetch_add(pointer, value):
    code = "atom.shared.add.u32 $0, [$1], $2;"
    return code, "=r,r,r,~{memory}", (pointer, Uint32(value))


@asm(Uint32, pure=True)
def lanemask_lt():
    return "mov.u32 $0, %lanemask_lt;", "=r", ()


@cute.jit
def popc(x: Uint32) -> Uint32:
    return Uint32(cute.arch.popc(x))


@cute.jit
def ballot(pred) -> Uint32:
    return recast_val(cute.arch.vote_ballot_sync(pred), Uint32)


@cute.jit
def lanes_below(x: Int32) -> Uint32:
    """The lanes l with 2 l < `x`."""
    c = (x + Int32(1)) >> Int32(1)
    c = cutlass.min(cutlass.max(c, Int32(0)), Int32(32))
    m = Uint32(0xFFFFFFFF)
    if c < Int32(32):
        m = (Uint32(1) << Uint32(c)) - Uint32(1)
    return m


@cute.jit
def ll_push(dst: Int64, w0, w1, epoch: Uint32):
    """One LL store: words `w0`, `w1` tagged with `epoch`."""
    if cutlass.const_expr(LL64):
        store_ll_pairs(dst, w0, w1, epoch)
    else:
        store_words(dst, (Uint32(w0), epoch, Uint32(w1), epoch), scope="sys")


def ll_offset(j):
    """Byte offset of B' word `j` in a row's LL pairs."""
    return Int64(j >> 1) * 16 + Int64(j & 1) * 8


class TopkExchange:
    """The symmetric LL slots of the exchange over the DCP `group`: int32
    [2, W, rows, 4 * ll_pairs(N)] (parity, source rank, row, (word, epoch)
    pairs), every rank's address in `peers`, and this rank's per-row
    `epoch`. One per device, shared by every DSA layer."""

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
    """The device's `TopkExchange` of at least `rows` rows (collective on
    first use: every DCP rank builds its DSA layers in the same order)."""
    key = (torch.device(device), local_topk)
    ex = _EXCHANGES.get(key)
    if ex is None or ex.rows < rows:
        ex = TopkExchange(group, device, world, rank, rows, local_topk)
        _EXCHANGES[key] = ex
    return ex


@functools.cache
def build_merge(rows, topk, local_topk, world, block_size, emit_ids, digits=False):
    """`ExchangeMerge` of `rows` rows (see the module docstring); `digits`:
    its "digits" first round (`merge_select`)."""
    from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge import kernels as stock

    from .indexer import low_digit, shared_add, shared_or, warp_inclusive

    assert world <= 16 and local_topk < topk and block_size > 0
    assert stock._TB_SIZE == TB
    assert not digits or (
        world == TB // _WARP
        and local_topk % 64 == 0
        and local_topk <= TB
        and topk < world * local_topk
        and packed_row_words(topk) >= TABLE_CAP + (1 << TIE_LOW_BITS)
        and not emit_ids
    )
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
        """The word of the LL half at `address` once its tag is `epoch`."""
        word, tag = ll_load(address)
        if tag != epoch:
            start = stock._globaltimer()
            while tag != epoch:  # no back-off: it measured slower (GB300 DCP8)
                word, tag = ll_load(address)
                if stock._globaltimer() - start > Int64(stock._SPIN_TIMEOUT_NS):
                    cute.printf("DCP top-k exchange timeout epoch=%d\n", epoch)
                    stock._trap()
        return word

    @cute.jit
    def settle(address: Int64, w0, e0, w1, e1, epoch: Uint32):
        """The two words of the LL store at `address` once both halves carry
        `epoch` (`w0, e0, w1, e1`: a first load of it)."""
        bad = (e0 ^ epoch) | (e1 ^ epoch)
        if bad != Uint32(0):
            start = stock._globaltimer()
            while bad != Uint32(0):
                w0, e0, w1, e1 = ll_load_pair(address)
                bad = (e0 ^ epoch) | (e1 ^ epoch)
                if stock._globaltimer() - start > Int64(stock._SPIN_TIMEOUT_NS):
                    cute.printf("DCP top-k exchange timeout epoch=%d\n", epoch)
                    stock._trap()
        return w0, w1

    @cute.jit
    def top_digit(
        counts, per_lane: cutlass.Constexpr[int], need: Uint32, warp, lane, wtot,
        info, at: cutlass.Constexpr[int],
    ):  # fmt: skip
        """The CTA over `counts` [TB * per_lane] (warp w its 32 * per_lane bins
        from 32 * per_lane * w, lane l its `per_lane` from there): the bin
        where the counts summed from the top bin down reach `need` ->
        `info[at]`, the count above it -> `info[at + 1]` (when they do).
        Returns the total count."""
        first = (warp * Int32(_WARP) + lane) * Int32(per_lane)
        if cutlass.const_expr(per_lane == 4):
            c = smem_words(counts.iterator + first)
        else:
            c = load_words(
                counts.iterator + first, count=per_lane, shared=True, mutable=True
            )
        mine = Uint32(0)
        for q in cutlass.range_constexpr(per_lane):
            mine += c[q]
        incl = warp_inclusive(mine, lane)
        total_w = cute.arch.shuffle_sync(incl, 31)
        if lane == Int32(31):
            wtot[warp] = total_w
        cute.arch.sync_threads()
        above_w = Uint32(0)
        total = Uint32(0)
        for w in cutlass.range_constexpr(num_warps):
            t = wtot[w]
            total += t
            if Int32(w) > warp:
                above_w += t
        run = above_w + (total_w - incl)  # the counts above this lane's bins
        if total >= need and run < need and run + mine >= need:
            found = Int32(-1)
            above = Uint32(0)
            for q in cutlass.range_constexpr(per_lane - 1, -1, -1):
                if found < Int32(0):
                    if run + c[q] >= need:
                        found = first + Int32(q)
                        above = run
                    run += c[q]
            info[at] = found
            info[at + 1] = Int32(above)
        cute.arch.sync_threads()
        return total

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
        `push_local_topk`, verbatim): this rank's local top-min(K, valid)
        of `logits[row]` in the B' layout -> every rank's
        `recv[row, slot]`; then the flag."""
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

    @cute.jit
    def rescue_rounds(
        keys, mask, num_sat, tau, row, tid, lane, warp, hist, warp_totals,
        scalars, owner, prefix_ptr, push_buf, out, req_id, block_table, rows_out,
        valid_out, dcp_rank, stride_rows, seq2, logits, row_ends, peer_recv,
        peer_flags, max_rows, rescue_recv, flags,
    ):  # fmt: skip
        """The #63 rescue of a row whose saturated ranks are `mask` (`num_sat`
        of them; `tau`: the round-1 K-th score, `keys`: the round-1 keys in
        the #63 select's thread order): this rank's push when it saturated,
        then the second round over the pushed lists (both merge forms)."""
        if ((mask >> dcp_rank) & Int32(1)) != Int32(0):
            # The logits (IndexerScore, before IndexerLocalTopk): wait for them.
            cute.arch.griddepcontrol_wait()
            slot = recast_val(
                cute.arch.popc(
                    recast_val(mask & ((Int32(1) << dcp_rank) - Int32(1)), Uint32)
                ),
                Int32,
            )
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
        push_buf = storage.push.get_tensor(cute.make_layout((recv_words,)))
        rescue_rounds(
            keys, mask, num_sat, tau, row, tid, lane, warp, hist, warp_totals,
            scalars, owner, prefix_ptr, push_buf, out, req_id, block_table,
            rows_out, valid_out, dcp_rank, stride_rows, seq2, logits, row_ends,
            peer_recv, peer_flags, max_rows, rescue_recv, flags,
        )  # fmt: skip
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

    # "digits" (merge_select): warp w owns source rank w.
    groups = n // 64  # 32-lane groups of a source rank's candidate LL pairs
    kpl = 2 * groups  # a lane's keys
    gid_mask = (1 << gid_bits) - 1
    bin_shift = gid_bits + 5  # a key's 11-bit score digit
    tie_bins = 1 << TIE_LOW_BITS
    if digits:
        # Four score bins and two tie bins per thread; an 11-bit tie digit.
        assert 4 * tb == DIGIT_BINS and 2 * tb == tie_bins
        assert gid_bits - TIE_LOW_BITS <= 11

    @cute.kernel
    def exchange_merge_digits(
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
        owner = storage.owner.get_tensor(cute.make_layout((16,)))
        prefix_ptr = storage.prefix.data_ptr()
        push_buf = storage.push.get_tensor(cute.make_layout((recv_words,)))
        counts = smem.allocate_tensor(Uint32, cute.make_layout(DIGIT_BINS), 16)
        low = smem.allocate_tensor(Uint32, cute.make_layout(DIGIT_LOW), 16)
        # 0-1: the score's bin, 3-5: v, take, ties at v, 6: own ties, 7: listed
        # ties, 8-11: the tie select's two digits, 12: the saturated ranks.
        info = smem.allocate_tensor(Int32, cute.make_layout(16), 16)
        wtot = smem.allocate_tensor(Uint32, cute.make_layout(num_warps), 16)
        ties = smem.allocate_tensor(Uint32, cute.make_layout(num_r1), 16)
        # The rescue's push buffer (used after both) holds the block-table row
        # and the tie select's low-digit counts.
        table = cute.make_tensor(push_buf.iterator, cute.make_layout(TABLE_CAP))
        tie_low = cute.make_tensor(
            cute.recast_ptr(push_buf.iterator + TABLE_CAP, dtype=Uint32),
            cute.make_layout(tie_bins),
        )

        # Round 1: source rank ``warp``'s LL pairs, every load issued before
        # any tag check.
        ep = epoch[row] + Int32(1)
        tag = recast_val(ep, Uint32)
        part = Int64(ep & Int32(1))
        slot = (part * Int64(world) + Int64(warp)) * Int64(xrows) + Int64(row)
        base = recv + slot * Int64(pairs * 16)
        raw = []
        for m in cutlass.range_constexpr(groups):
            raw.append(ll_load_pair(base + Int64(lane + Int32(32 * m)) * 16))
        # The bit-plane pairs (lane m < groups holds pair m; the others reload
        # the last one).
        plane_pair = Int32(n // 2) + cutlass.min(lane, Int32(groups - 1))
        plane_address = base + Int64(plane_pair) * 16
        praw = ll_load_pair(plane_address)
        # The request's block-table row, in registers until the keys are in.
        rq = req_id[row]
        cols = Int32(block_table.shape[1])
        staged = cols <= Int32(TABLE_CAP)
        tv = []
        for c in cutlass.range_constexpr(TABLE_CAP // tb):
            e = Int32(c * tb) + tid
            value = Int32(0)
            if staged and e < cols:
                value = block_table[rq, e]
            tv.append(value)
        for b in cutlass.range(tid, DIGIT_BINS, tb, unroll=1):
            counts[b] = Uint32(0)
        if tid < Int32(DIGIT_LOW):
            low[tid] = Uint32(0)
        if tid == Int32(0):
            info[7] = Int32(0)
            info[12] = Int32(0)
        for i in cutlass.range(Int32(n) + tid, rows_out.shape[1], tb, unroll=1):
            rows_out[row, i] = Int32(-1)
        cute.arch.sync_threads()

        pw0, pw1 = settle(plane_address, praw[0], praw[1], praw[2], praw[3], tag)
        lt = lanemask_lt()
        me = recast_val(lane, Uint32)
        keys = cute.make_rmem_tensor((kpl,), Uint64)
        source = recast_val(warp, Uint32)
        for m in cutlass.range_constexpr(groups):
            w0, w1 = settle(
                base + Int64(lane + Int32(32 * m)) * 16, raw[m][0], raw[m][1],
                raw[m][2], raw[m][3], tag,
            )  # fmt: skip
            p0 = cute.arch.shuffle_sync(pw0, m)
            p1 = cute.arch.shuffle_sync(pw1, m)
            high = p0
            if lane >= Int32(16):
                high = p1
            j = Int32(64 * m) + Int32(2) * lane  # w0's B' slot (w1's: j + 1)
            keys[2 * m] = stock._decode_key(w0, high, j, source, world, gid_bits)
            keys[2 * m + 1] = stock._decode_key(
                w1, high, j + Int32(1), source, world, gid_bits
            )
        for k in cutlass.range_constexpr(kpl):
            key = keys[k]
            if key != Uint64(0):
                shared_add(counts.iterator + Uint32(key >> Uint64(bin_shift)), 1)
        for c in cutlass.range_constexpr(TABLE_CAP // tb):
            table[Int32(c * tb) + tid] = tv[c]
        cute.arch.sync_threads()

        # The K-th key's 11-bit score digit (when there are K keys).
        total = top_digit(counts, 4, Uint32(topk), warp, lane, wtot, info, 0)
        full = total >= Uint32(topk)
        # Its low 5 bits: the K-th score v, the ties at v to take, all of them.
        v = Uint32(0)
        take = Uint32(0)
        ties_at = Uint32(0)
        if full:
            b1 = Uint32(info[0])
            need1 = Uint32(topk) - Uint32(info[1])
            for k in cutlass.range_constexpr(kpl):
                key = keys[k]
                if key != Uint64(0) and Uint32(key >> Uint64(bin_shift)) == b1:
                    d = Uint32(key >> Uint64(gid_bits)) & Uint32(DIGIT_LOW - 1)
                    shared_add(low.iterator + d, 1)
            cute.arch.sync_threads()
            if warp == Int32(0):
                digit, above2 = low_digit(low[lane], need1, lane)
                if lane == Int32(0):
                    info[3] = Int32((b1 << Uint32(5)) | digit)
                    info[4] = Int32(need1 - above2)
                    info[5] = Int32(low[digit])
            cute.arch.sync_threads()
            v = Uint32(info[3])
            take = Uint32(info[4])
            ties_at = Uint32(info[5])

        # Rank ``warp``'s saturation (every key present and above v), and the
        # ties at v by inverted global id (when only some are taken).
        all_above = Uint32(1)
        for k in cutlass.range_constexpr(kpl):
            key = keys[k]
            if key == Uint64(0) or Uint32(key >> Uint64(gid_bits)) <= v:
                all_above = Uint32(0)
        sat = ballot(all_above != Uint32(0)) == Uint32(0xFFFFFFFF)
        if lane == Int32(0) and sat:
            shared_or(info.iterator + 12, Uint32(1) << recast_val(warp, Uint32))
        partial = full and take < ties_at
        if partial:
            own = Uint32(0)
            for k in cutlass.range_constexpr(kpl):
                key = keys[k]
                is_tie = key != Uint64(0) and Uint32(key >> Uint64(gid_bits)) == v
                m_tie = ballot(is_tie)
                at = Uint32(0)
                if lane == Int32(0) and m_tie != Uint32(0):
                    at = shared_fetch_add(info.iterator + 7, popc(m_tie))
                at = cute.arch.shuffle_sync(at, 0)
                if is_tie:
                    ties[Int32(at + popc(m_tie & lt))] = Uint32(key & Uint64(gid_mask))
                own += popc(m_tie)
            if warp == dcp_rank and lane == Int32(0):
                info[6] = Int32(own)
        cute.arch.sync_threads()

        # The saturated ranks and the K-th score: the #63 select's word.
        mask = info[12]
        num_sat = recast_val(cute.arch.popc(recast_val(mask, Uint32)), Int32)
        if num_sat > Int32(num_slots):
            mask = Int32(0)
        tau = recast_val(v, Int32)

        if tid == Int32(0):
            sat_word[row] = (mask << Int32(16)) | tau
        if mask == Int32(0):  # (else the rescue below writes the row)
            # Many ties (or many of this rank's): the take-th largest listed tie id
            # by a block-wide select of its two digits; this rank's ties at or
            # above it are taken.
            listed = info[7]
            many = partial and (
                listed > Int32(TIE_REGS * _WARP) or info[6] > Int32(TIE_OWN)
            )
            cut = Uint32(0)
            if many:
                for b in cutlass.range(tid, DIGIT_BINS, tb, unroll=1):
                    counts[b] = Uint32(0)
                for b in cutlass.range(tid, tie_bins, tb, unroll=1):
                    tie_low[b] = Uint32(0)
                cute.arch.sync_threads()
                for i in cutlass.range(tid, listed, tb, unroll=1):
                    shared_add(counts.iterator + (ties[i] >> Uint32(TIE_LOW_BITS)), 1)
                cute.arch.sync_threads()
                top_digit(counts, 4, take, warp, lane, wtot, info, 8)
                hi = Uint32(info[8])
                need2 = take - Uint32(info[9])
                for i in cutlass.range(tid, listed, tb, unroll=1):
                    x = ties[i]
                    if (x >> Uint32(TIE_LOW_BITS)) == hi:
                        shared_add(tie_low.iterator + (x & Uint32(tie_bins - 1)), 1)
                cute.arch.sync_threads()
                top_digit(tie_low, 2, need2, warp, lane, wtot, info, 10)
                cut = (hi << Uint32(TIE_LOW_BITS)) | Uint32(info[10])

            # This rank's selected keys -> its KV rows, in the #63 select's order:
            # its own source's warp alone.
            if warp == dcp_rank:
                taken = cute.make_rmem_tensor((kpl,), Uint32)
                for k in cutlass.range_constexpr(kpl):
                    taken[k] = Uint32(1)
                if partial:
                    if many:
                        for k in cutlass.range_constexpr(kpl):
                            key = keys[k]
                            if (
                                key != Uint64(0)
                                and Uint32(key >> Uint64(gid_bits)) == v
                            ):
                                taken[k] = Uint32(0)
                                if Uint32(key & Uint64(gid_mask)) >= cut:
                                    taken[k] = Uint32(1)
                    else:
                        # Each own tie's rank among the listed: a ballot per 32.
                        tl = []
                        for c in cutlass.range_constexpr(TIE_REGS):
                            x = Uint32(0)
                            if Int32(c * _WARP) + lane < listed:
                                x = ties[Int32(c * _WARP) + lane]
                            tl.append(x)
                        for k in cutlass.range_constexpr(kpl):
                            key = keys[k]
                            inv = Uint32(key & Uint64(gid_mask))
                            own_tie = (
                                key != Uint64(0)
                                and Uint32(key >> Uint64(gid_bits)) == v
                            )
                            pending = ballot(own_tie)
                            while pending != Uint32(0):
                                src = popc(
                                    (pending & (Uint32(0) - pending)) - Uint32(1)
                                )
                                y = cute.arch.shuffle_sync(inv, Int32(src))
                                r = Uint32(0)
                                for c in cutlass.range_constexpr(TIE_REGS):
                                    if Int32(c * _WARP) < listed:
                                        r += popc(ballot(tl[c] > y))
                                if me == src:
                                    taken[k] = Uint32(0)
                                    if r < take:
                                        taken[k] = Uint32(1)
                                pending = pending & (pending - Uint32(1))
                bits = cute.make_rmem_tensor((kpl,), Uint32)
                locs = cute.make_rmem_tensor((kpl,), Int32)
                for k in cutlass.range_constexpr(kpl):
                    key = keys[k]
                    chosen = key != Uint64(0)
                    if full:
                        chosen = (
                            chosen
                            and Uint32(key >> Uint64(gid_bits)) >= v
                            and taken[k] != Uint32(0)
                        )
                    gid = Uint32(gid_mask) - Uint32(key & Uint64(gid_mask))
                    local = recast_val(gid // Uint32(world), Int32)
                    locs[k] = local
                    bits[k] = ballot(chosen and local // Int32(block_size) < cols)
                total_own = Uint32(0)
                for k in cutlass.range_constexpr(kpl):
                    total_own += popc(bits[k])
                # The #63 order is thread order (thread = flat slot mod TB): when
                # this rank's slots wrap a multiple of TB, slots j >= jw come first.
                jw = Int32(tb) - (dcp_rank * Int32(n)) % Int32(tb)
                wrap = jw < Int32(n)
                before_jw = Uint32(0)
                for m in cutlass.range_constexpr(groups):
                    for e in cutlass.range_constexpr(2):
                        below = lanes_below(jw - Int32(64 * m + e))
                        before_jw += popc(bits[2 * m + e] & below)
                run = Uint32(0)
                for m in cutlass.range_constexpr(groups):
                    for e in cutlass.range_constexpr(2):
                        k = 2 * m + e
                        rank = run + popc(bits[2 * m] & lt) + popc(bits[2 * m + 1] & lt)
                        if cutlass.const_expr(e == 1):
                            rank += (bits[2 * m] >> me) & Uint32(1)
                        if ((bits[k] >> me) & Uint32(1)) != Uint32(0):
                            j = Int32(64 * m + e) + Int32(2) * lane
                            pos = rank
                            if wrap:
                                if j >= jw:
                                    pos = rank - before_jw
                                else:
                                    pos = rank + total_own - before_jw
                            local = locs[k]
                            block = local // Int32(block_size)
                            page = Int32(0)
                            if staged:  # noqa: SIM108 (a conditional global load)
                                page = table[block]
                            else:
                                page = block_table[rq, block]
                            rows_out[row, Int32(pos)] = page * stride_rows + (
                                local - block * Int32(block_size)
                            )
                    for e in cutlass.range_constexpr(2):
                        run += popc(bits[2 * m + e])
                for i in cutlass.range(
                    Int32(total_own) + lane, Int32(n), _WARP, unroll=1
                ):
                    rows_out[row, i] = Int32(-1)
                if lane == Int32(0):
                    valid_out[row] = Int32(total_own)

        # The next call's epoch, as in the radix form (the rescue below reads no
        # exchange slot).
        cute.arch.griddepcontrol_wait()
        cute.arch.sync_threads()
        if tid == Int32(0):
            epoch[row] = ep
        if mask != Int32(0):
            # Saturated ranks: the radix form's rescue writes the row (round 1
            # wrote none), after the rest so that the common path's code runs
            # straight through. It takes the round-1 keys in the #63 select's
            # thread order (thread t's i-th: flat slot t + TB i), moved through
            # ``ties`` (its tie list unread here) 32 bits at a time.
            skeys = cute.make_rmem_tensor((kpt,), Uint64)
            for h in cutlass.range_constexpr(2):
                for m in cutlass.range_constexpr(groups):
                    for e in cutlass.range_constexpr(2):
                        flat = warp * Int32(n) + Int32(64 * m + e) + Int32(2) * lane
                        ties[flat] = Uint32(keys[2 * m + e] >> Uint64(32 * h))
                cute.arch.sync_threads()
                for u in cutlass.range_constexpr(kpt):
                    flat = tid + Int32(u * tb)
                    word = Uint64(0)
                    if flat < Int32(num_r1):
                        word = Uint64(ties[flat])
                    if cutlass.const_expr(h == 0):
                        skeys[u] = word
                    else:
                        skeys[u] = skeys[u] | (word << Uint64(32))
                if cutlass.const_expr(h == 0):
                    cute.arch.sync_threads()
            rescue_rounds(
                skeys, mask, num_sat, tau, row, tid, lane, warp, hist, warp_totals,
                scalars, owner, prefix_ptr, push_buf, out, req_id, block_table,
                rows_out, valid_out, dcp_rank, stride_rows, seq2, logits, row_ends,
                peer_recv, peer_flags, max_rows, rescue_recv, flags,
            )  # fmt: skip

    merge_kernel = exchange_merge_digits if digits else device_kernel

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
        merge_kernel(
            recv, epoch, xrows, out, req_id, block_table, rows_out, valid_out,
            dcp_rank, stride_rows, sat_word, seq2, logits, row_ends, peer_recv,
            peer_flags, max_rows, rescue_recv, flags,
        ).launch(
            grid=(rows, 1, 1), block=(tb, 1, 1), stream=s, use_pdl=True
        )  # fmt: skip

    return host_entrypoint


@functools.cache
def compile_merge(
    rows, topk, local_topk, world, block_size, emit_ids, device, digits=False
):
    """The compiled `ExchangeMerge` of `rows` rows (shared by every layer)."""
    args = rows, topk, local_topk, world, block_size, emit_ids
    entry = build_merge(*args, digits)
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
        self.select = merge_select(*args[:4], emit_ids)
        digits = self.select == "digits"
        self.kernel = compile_merge(*args, exchange.recv.device.index, digits)
        self.emit_ids = emit_ids

    def __call__(
        self, logits, row_ends, req_id, block_table, stride_rows, rows_out,
        valid_out, ids_out=None,
    ):  # fmt: skip
        """Every rank's pushed B' rows of the `rows` rows -> this rank's
        compacted KV rows `rows_out` [rows, >= K], counts `valid_out`
        and, compiled with `emit_ids`, the global ids `ids_out` [rows, K].
        `logits` / `row_ends`: this rank's (a saturated rank re-reads
        its row), bounded as the #63 merge's (`_bound_logits`)."""
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
