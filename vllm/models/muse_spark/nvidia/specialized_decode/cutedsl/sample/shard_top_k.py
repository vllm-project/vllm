# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Muse Spark target head's soft cap and top-64 candidates of a vocab
shard in one CuTe kernel, posted to every TP rank.

From the BF16 logits of a vocab shard (the head GEMM's padded rows), one launch
soft-caps each logit and multicasts, per row, a packet of the shard's 64
largest (the smaller column first among ties), in column order, into this
rank's slot of every rank's `CandidateMailbox`: per candidate its order key
(`order_key`) and its vocab column, tagged with the row's sequence number.

The soft cap is `cap * tanh(multiplier * x / cap)` in FP32 with
`cute.math.tanh` (its non-fast-math `tanhf` sequence, not `tanh.approx`); the
tests check the greedy picks against the stock op's
(`ops.muse_spark_mul_tanh_soft_cap`). A logit ranks by its FP32 bits mapped
to uint32 in order (+0.0 above -0.0), ties to the smaller column.

Grid (chunks of 2048 columns, rows), 256 threads of 8 columns each. Each warp
sorts its 256 keys (bitonic) and the CTA merges the warps' 64 largest into the
chunk's: the 64th largest key `t` and how many keys exceed it. A CTA-wide scan
then takes, in column order, the keys above `t` and the first ties at `t`, and
stores them with the chunk's sorted keys. The row's last CTA to arrive (a
ticket) merges the chunks' sorted keys into the row's threshold the same way
and posts the row's candidates from the chunks' in column order. Where the GPU
has programmatic dependent launch, the kernel launches with it and reads and
writes nothing before the dependency wait.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from vllm.utils.math_utils import cdiv

from ..common.host import assert_not_capturing, pdl_supported
from ..common.primitives import (
    above_u32,
    global_ptr,
    load_cg_u32x4,
    load_global_bf16x8_as_f32_address,
    load_global_u32_address,
    load_shared_u32,
    load_shared_u32x2,
    order_key,
    select_max,
    store_global_u32_address,
    store_global_u32x2,
    store_global_u32x4,
    store_shared_u32,
    store_shared_u32x2_pred,
    store_shared_u32x4_pred,
    store_u32x2_pred,
    sync_warp,
)
from ..common.sequenced import generation, sequences, stmc_b64
from ..common.symmetric import NoMulticastError, symmetric_buffer

CHUNK = 2048
THREADS = 256
VEC = 8
WARPS = THREADS // 32
K = 64
SLOTS = 16
PACKET_WORDS = 2 * K
# A packet word's low TAG_BITS are its row's sequence number; the rest carry
# the candidate's key and its column of at most COLUMN_BITS bits.
TAG_BITS = 7
TAG_MASK = (1 << TAG_BITS) - 1
COLUMN_BITS = 32 - 2 * TAG_BITS
# Sequence numbers run 1 .. SEQUENCES, an even count (see `CandidateMailbox`).
SEQUENCES = sequences(TAG_BITS)


def tag_candidate(key: Uint32, column: Uint32, tag: Uint32) -> tuple:
    """The two packet words of a candidate: the key's high 25 bits, and its
    low 7 bits over the column, each above `tag`."""
    high = (key & Uint32(0xFFFFFFFF ^ TAG_MASK)) | tag
    low = (key << Uint32(32 - TAG_BITS)) | (column << Uint32(TAG_BITS)) | tag
    return high, low


def untag_candidate(high: Uint32, low: Uint32) -> tuple:
    """The (key, column) of a candidate's packet words (`tag_candidate`)."""
    key = (high & Uint32(0xFFFFFFFF ^ TAG_MASK)) | (low >> Uint32(32 - TAG_BITS))
    column = (low >> Uint32(TAG_BITS)) & Uint32((1 << COLUMN_BITS) - 1)
    return key, column


def soft_cap(x: Float32, multiplier: Float32, cap: Float32) -> Float32:
    """`cap * tanh(multiplier * x / cap)` in FP32 (`cute.math.tanh`, not
    fast-math)."""
    return cap * cute.math.tanh(multiplier * x / cap)


def ballot_count(predicate) -> Int32:
    """The number of lanes with a nonzero `predicate`."""
    return Int32(cute.arch.popc(cute.arch.vote_ballot_sync(Int32(predicate) != 0)))


def _bit(value, shift: int):
    return (value >> shift) & 1


def _log2(n: int) -> int:
    return n.bit_length() - 1


def warp_sort_desc(v: list, lane) -> list:
    """Bitonic sort, descending, of the 256 keys of a warp: element
    `lane * 8 + e` is `v[e]` of `lane`."""
    n = 32 * VEC
    k = 2
    while k <= n:
        j = k // 2
        while j >= 1:
            if j < VEC:
                for e in range(VEC):
                    p = e ^ j
                    if p < e:
                        continue
                    hi, lo = cute.math.max(v[e], v[p]), cute.math.min(v[e], v[p])
                    if k < VEC:
                        v[e], v[p] = (hi, lo) if (e & k) == 0 else (lo, hi)
                    elif k == n:
                        v[e], v[p] = hi, lo
                    else:
                        desc = Int32(1) ^ _bit(lane, _log2(k // VEC))
                        v[e] = select_max(v[e], v[p], desc)
                        v[p] = select_max(hi, lo, Int32(1) ^ desc)
            else:
                m = j // VEC
                lower = Int32(1) ^ _bit(lane, _log2(m))
                if k == n:
                    take = lower
                else:
                    desc = Int32(1) ^ _bit(lane, _log2(k // VEC))
                    take = Int32(1) ^ (lower ^ desc)
                for e in range(VEC):
                    v[e] = select_max(
                        v[e], cute.arch.shuffle_sync_bfly(v[e], offset=m), take
                    )
            j //= 2
        k *= 2
    return v


def warp_merge(base, a, b, lane) -> list:
    """The 64 largest keys of the descending lists in shared slots `a` and
    `b`, descending: element `2 * lane + e` is entry e."""
    c = []
    for e in range(2):
        i = 2 * lane + e
        x = load_shared_u32(base + (Int32(a) * K + i) * 4)
        y = load_shared_u32(base + (Int32(b) * K + (K - 1) - i) * 4)
        c.append(cute.math.max(x, y))
    j = K // 2
    while j >= 2:
        m = j // 2
        take = Int32(1) ^ _bit(lane, _log2(m))
        for e in range(2):
            c[e] = select_max(c[e], cute.arch.shuffle_sync_bfly(c[e], offset=m), take)
        j //= 2
    return [cute.math.max(c[0], c[1]), cute.math.min(c[0], c[1])]


def store_merged(base, a, lane, out) -> None:
    for e in range(2):
        store_shared_u32(base + (a * K + 2 * lane + e) * 4, out[e])


def merge_levels(count: int) -> list:
    """(stride, merges, last) of the pairwise merge of `count` lists."""
    levels = []
    stride = 1
    while stride < count:
        merges = (count - stride + 2 * stride - 1) // (2 * stride)
        levels.append((stride, merges, 2 * stride >= count))
        stride *= 2
    return levels


def threshold(top: list, lane):
    """(k-th largest key, the number of the top k above it) of a warp's merged
    list."""
    kth = cute.arch.shuffle_sync(top[1], 31)
    above = ballot_count(above_u32(top[0], kth)) + ballot_count(above_u32(top[1], kth))
    return kth, above


def warp_scan_inclusive(x):
    """The inclusive prefix sum of `x` over the warp's lanes."""
    lane = cute.arch.lane_idx()
    inclusive = x
    for delta in (1, 2, 4, 8, 16):
        up = cute.arch.shuffle_sync_up(inclusive, delta)
        inclusive = inclusive + cutlass.select_(lane >= delta, up, Int32(0))
    return inclusive


def count_words(keys, valid_flags, kth):
    """(per-key above flags, per-key tie flags, packed count above | ties << 16)."""
    above = [above_u32(keys[i], kth) for i in range(len(keys))]
    ties = [Int32(keys[i] == kth) & valid_flags[i] for i in range(len(keys))]
    packed = Int32(0)
    for i in range(len(keys)):
        packed = packed + above[i] + (ties[i] << 16)
    return above, ties, packed


def place(above, ties, prefix, need):
    """(selected, position) of each key in column order: the keys above the
    threshold and the first `need` ties."""
    taken_above = prefix & 0xFFFF
    taken_ties = prefix >> 16
    out = []
    for i in range(len(above)):
        tie_ok = ties[i] & Int32(taken_ties < need)
        selected = above[i] | tie_ok
        position = taken_above + cutlass.min(taken_ties, need)
        out.append((selected, position))
        taken_above = taken_above + above[i]
        taken_ties = taken_ties + ties[i]
    return out


class _ShardTopKKernel:
    """Grid (chunks, rows), 256 threads; see the module docstring. The packets
    go to `slot`, this rank's slot of generation 0 (with `multicast` its
    multicast address, else a local one), the generations `generation_bytes`
    apart."""

    def __init__(self, vocab: int, enable_pdl: bool, multicast: bool) -> None:
        self.vocab = vocab
        self.enable_pdl = enable_pdl
        self.multicast = multicast
        self.chunks = cdiv(vocab, CHUNK)
        self.cta_levels = merge_levels(WARPS)
        self.row_levels = merge_levels(self.chunks)

    @cute.jit
    def __call__(
        self,
        x: Int64,
        slot: Int64,
        generation_bytes: Int64,
        seqs: Int64,
        lists: Int64,
        tickets: Int64,
        rows: Int32,
        row_stride: Int32,
        vocab_offset: Int32,
        multiplier: Float32,
        cap: Float32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            x,
            slot,
            generation_bytes,
            seqs,
            lists,
            tickets,
            row_stride,
            vocab_offset,
            multiplier,
            cap,
        ).launch(  # fmt: skip
            grid=(self.chunks, rows, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        slot: Int64,
        generation_bytes: Int64,
        seqs: Int64,
        lists: Int64,
        tickets: Int64,
        row_stride: Int32,
        vocab_offset: Int32,
        multiplier: Float32,
        cap: Float32,
    ) -> None:
        vocab, chunks = self.vocab, self.chunks
        tidx, _, _ = cute.arch.thread_idx()
        chunk, row, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()

        smem = cutlass.memory.SmemAllocator()
        slots = smem.allocate_tensor(Uint32, cute.make_layout(SLOTS * K), 16)
        staged = smem.allocate_tensor(Uint32, cute.make_layout(2 * K), 16)
        scratch = smem.allocate_tensor(Uint32, cute.make_layout(WARPS + 4), 16)
        base = Int32(slots.iterator.toint())
        stage_at = Int32(staged.iterator.toint())
        totals = Int32(scratch.iterator.toint())
        shared_kth = totals + WARPS * 4
        shared_need = shared_kth + 4
        shared_ticket = shared_need + 4

        # A list of the row: [chunk][64] sorted keys, then [chunk][64]
        # (key, column) candidates in column order.
        list_row = lists + Int64(row) * Int64(chunks * K * 12)
        sorted_at = list_row + Int64(chunk) * Int64(K * 4)
        cands_row = list_row + Int64(chunks * K * 4)
        ticket_at = tickets + Int64(row) * Int64(4)

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        column = Int32(chunk * CHUNK + tidx * VEC)
        valid = Int32(column < vocab)
        x_at = x + (Int64(row) * Int64(row_stride) + Int64(column)) * Int64(2)
        raw = load_global_bf16x8_as_f32_address(x_at, valid, nc=True)
        values = [soft_cap(raw[i], multiplier, cap) for i in range(VEC)]
        keys = [order_key(values[i], valid) for i in range(VEC)]

        top = warp_sort_desc(list(keys), lane)
        keep = Int32(lane < K // VEC)
        for e in cutlass.range_constexpr(VEC // 4):
            store_shared_u32x4_pred(
                base + (warp * K + lane * VEC + 4 * e) * 4,
                top[4 * e : 4 * e + 4],
                keep,
            )
        cute.arch.sync_threads()
        final = self.merge_lists(self.cta_levels, base, warp, lane)
        if warp == 0:
            kth, above = threshold(final, lane)
            store_global_u32x2(sorted_at + Int64(lane * 8), (final[0], final[1]))
            if lane == 0:
                store_shared_u32(shared_kth, kth)
                store_shared_u32(shared_need, Uint32(K - above))
        cute.arch.sync_threads()

        kth = load_shared_u32(shared_kth)
        need = Int32(load_shared_u32(shared_need))
        above, ties, packed = count_words(keys, [valid] * VEC, kth)
        prefix = self.exclusive_prefix(packed, totals, warp, lane)
        for i, (selected, position) in enumerate(place(above, ties, prefix, need)):
            store_shared_u32x2_pred(
                stage_at + position * 8, keys[i], Uint32(column + i), selected
            )
        cute.arch.sync_threads()
        if warp == 0:
            entries = load_shared_u32x2(stage_at + lane * 16)
            more = load_shared_u32x2(stage_at + lane * 16 + 8)
            store_global_u32x4(
                cands_row + Int64(chunk) * Int64(K * 8) + Int64(lane * 16),
                (entries[0], entries[1], more[0], more[1]),
            )
            cute.arch.fence_acq_rel_gpu()
            sync_warp()
            if lane == 0:
                store_shared_u32(
                    shared_ticket,
                    cute.arch.atomic_add(
                        global_ptr(Uint32, ticket_at),
                        Uint32(1),
                        sem="acq_rel",
                        scope="gpu",
                    ),
                )
        cute.arch.sync_threads()
        arrived = load_shared_u32(shared_ticket)

        if arrived == Uint32(chunks - 1):
            # A thread per 4 sorted keys and per 4 candidates (in column
            # order, loaded early).
            loads = chunks * K // 4
            got = load_cg_u32x4(list_row + Int64(tidx * 16), Int32(tidx < loads))
            in_range = Int32(tidx < loads)
            first = load_cg_u32x4(cands_row + Int64(tidx * 32), in_range)
            second = load_cg_u32x4(cands_row + Int64(tidx * 32 + 16), in_range)
            store_shared_u32x4_pred(base + tidx * 16, got, in_range)
            cute.arch.sync_threads()
            best = self.merge_lists(self.row_levels, base, warp, lane)
            if warp == 0:
                row_kth, row_above = threshold(best, lane)
                if lane == 0:
                    store_shared_u32(shared_kth, row_kth)
                    store_shared_u32(shared_need, Uint32(K - row_above))
                    store_global_u32_address(ticket_at, Uint32(0))
            cute.arch.sync_threads()
            row_kth = load_shared_u32(shared_kth)
            row_need = Int32(load_shared_u32(shared_need))
            cand_keys = [first[0], first[2], second[0], second[2]]
            cand_cols = [first[1], first[3], second[1], second[3]]
            r_above, r_ties, r_packed = count_words(cand_keys, [in_range] * 4, row_kth)
            r_above = [r_above[i] & in_range for i in range(4)]
            r_packed = r_packed * in_range
            r_prefix = self.exclusive_prefix(r_packed, totals, warp, lane)
            # The row's sequence number tags the packet and picks its
            # generation; select advances it.
            seq = load_global_u32_address(seqs + Int64(row) * 4, ordered=True)
            packet_row = (
                slot
                + Int64(generation(seq)) * generation_bytes
                + Int64(row) * Int64(PACKET_WORDS * 4)
            )
            for i, (selected, position) in enumerate(
                place(r_above, r_ties, r_prefix, row_need)
            ):
                high, low = tag_candidate(
                    cand_keys[i], cand_cols[i] + Uint32(vocab_offset), seq
                )
                if cutlass.const_expr(self.multicast):
                    stmc_b64(packet_row + Int64(position * 8), high, low, selected)
                else:
                    store_u32x2_pred(
                        packet_row + Int64(position * 8), high, low, selected
                    )

    @cute.jit
    def merge_lists(self, levels: cutlass.Constexpr, base, warp, lane):
        """The 64 largest keys of the descending lists in the shared slots,
        merged pairwise (`merge_levels`), as `warp_merge` returns them."""
        merged = None
        for level in cutlass.range_constexpr(len(levels)):
            stride, merges, last = levels[level]
            if cutlass.const_expr(last):
                merged = warp_merge(base, 0, stride, lane)
            else:
                if warp < merges:
                    a = warp * (2 * stride)
                    store_merged(base, a, lane, warp_merge(base, a, a + stride, lane))
                cute.arch.sync_threads()
        return merged

    @cute.jit
    def exclusive_prefix(self, packed, totals, warp, lane):
        """The CTA-wide exclusive prefix sum of the threads' `packed` counts, in
        thread order (warp totals in the shared words at `totals`)."""
        inclusive = warp_scan_inclusive(packed)
        if lane == 31:
            store_shared_u32(totals + warp * 4, Uint32(inclusive))
        cute.arch.sync_threads()
        prefix = inclusive - packed
        for w in cutlass.range_constexpr(WARPS - 1):
            prefix = prefix + Int32(load_shared_u32(totals + w * 4)) * Int32(w < warp)
        return prefix


@functools.cache
def compile_shard_top_k(vocab: int, enable_pdl: bool, multicast: bool = True):
    """cute.compile the kernel of one shard width (before CUDA graph capture);
    the row count and row stride are runtime arguments."""
    assert_not_capturing("the shard top-k")
    return cute.compile(
        _ShardTopKKernel(vocab, enable_pdl, multicast),
        *([Int64(0)] * 6),
        Int32(1),
        Int32(vocab),
        Int32(0),
        Float32(1.0),
        Float32(1.0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


class CandidateMailbox:
    """The TP ranks' candidate packets: `[2, tp, rows, PACKET_WORDS]` int32
    slots (generation, rank, row) and a sequence number per row (`seqs`, kept
    by each rank).

    `ShardTopK` stores row r's packet into slot `[seq & 1, rank, r]` of every
    rank's copy, one multicast store per candidate, every word tagged with the
    row's sequence number `seq`. `_SelectKernel` polls the `tp` slots of the
    row in its own copy, a 64-bit load per candidate (the size of its store,
    so no accesses of mixed sizes race), until every word carries `seq`,
    reads the words in place and advances `seq`, wrapping within 1 ..
    `SEQUENCES` (slots start at tag 0): an even count, so the generation
    alternates across the wrap.

    Readiness is the tag alone, never the payload: a word is ready when its
    low `TAG_BITS` are this call's `seq`, whatever its other bits, so every
    key and column is safe. A candidate's two words may arrive apart (as
    `PacketMailbox` assumes, no store wider than a word is relied on to be
    seen whole); each is read once it has. A slot's stale words carry the
    tag of the row's call two before (`seq - 2`, never `seq` within the
    wrap) or the initial 0, so nothing needs re-arming. Two generations
    suffice: a rank stores row r's generation two calls on only after its
    select polled every rank's next packet of row r, which each rank stores
    after its select read this one, in stream order. The tagged words are
    the whole payload, so no data travels beside them for a fence to order.

    Every rank must run the same exchanges (a `ShardTopK` launch, then one
    select over the same rows) in the same order, so that the ranks' `seqs`
    agree. Collective over `group`, which raises `NoMulticastError` on every
    rank when the buffer has no multicast mapping; without a group the
    buffer is local and
    one process posts every rank's packets (`ShardTopK` with `multicast`
    false).
    """

    def __init__(
        self,
        tp: int,
        rows: int,
        device: torch.device,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        self.tp, self.rows = tp, rows
        shape = (2, tp, rows, PACKET_WORDS)
        if group is None:
            self.tensor = torch.zeros(shape, dtype=torch.int32, device=device)
            self.multicast = 0
        else:
            buffer = symmetric_buffer(group, shape, torch.int32, device)
            self.tensor, self.multicast = buffer.tensor, buffer.multicast
            if not self.multicast:
                raise NoMulticastError("the candidate mailbox has no NVLink multicast")
            if self.multicast % 16:
                raise RuntimeError("the candidate mailbox's multicast is misaligned")
            self.tensor.zero_()
        self.seqs = torch.ones(rows, dtype=torch.int32, device=device)
        self.rank_bytes = rows * PACKET_WORDS * torch.int32.itemsize
        self.generation_bytes = tp * self.rank_bytes
        self.local = self.tensor.data_ptr()
        if group is not None:
            torch.accelerator.synchronize(device)
            dist.barrier(group=group)

    def slot(self, rank: int) -> int:
        """Where `ShardTopK` stores rank `rank`'s packets of generation 0."""
        return (self.multicast or self.local) + rank * self.rank_bytes


class ShardTopK:
    """The soft cap and top-64 candidate packets of a vocab shard's BF16
    logits in one launch, for at most `max_rows` rows, posted to a
    `CandidateMailbox` (by multicast, or with `multicast` false to a local
    one). Compiled at construction (before CUDA graph capture)."""

    @staticmethod
    def supports(vocab: int) -> bool:
        chunks = cdiv(vocab, CHUNK)
        tail = vocab % CHUNK
        return vocab % VEC == 0 and 1 < chunks <= SLOTS and not 0 < tail < K

    def __init__(
        self, vocab: int, max_rows: int, device: torch.device, multicast: bool = True
    ) -> None:
        if not self.supports(vocab):
            raise ValueError(f"no shard top-k kernel for a {vocab}-column shard")
        self.vocab = vocab
        self.max_rows = max_rows
        self.multicast = multicast
        chunks = cdiv(vocab, CHUNK)
        self.lists = torch.empty(
            max_rows * chunks * K * 3, dtype=torch.int32, device=device
        )
        self.tickets = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self._lists, self._tickets = self.lists.data_ptr(), self.tickets.data_ptr()
        self.device_index = self.tickets.device.index
        self.kernel = compile_shard_top_k(vocab, pdl_supported(), multicast)

    def takes(self, x: torch.Tensor) -> bool:
        return (
            x.dim() == 2
            and x.dtype == torch.bfloat16
            and x.shape[1] == self.vocab
            and 1 <= x.shape[0] <= self.max_rows
            and x.stride(1) == 1
            and x.stride(0) % VEC == 0
            and x.data_ptr() % 16 == 0
        )

    def run(
        self,
        x: torch.Tensor,
        multiplier: float,
        cap: float,
        vocab_offset: int,
        mailbox: CandidateMailbox,
        rank: int,
    ) -> None:
        """Post rank `rank`'s packets of the soft-capped BF16 `x` (`[rows,
        vocab]`, row stride a multiple of 8) to `mailbox`; candidate columns
        are offset by `vocab_offset`."""
        assert self.takes(x), "x must be [rows, vocab] BF16 rows of an aligned stride"
        assert x.shape[0] <= mailbox.rows and 0 <= rank < mailbox.tp
        assert self.multicast == bool(mailbox.multicast)
        assert vocab_offset + self.vocab <= 1 << COLUMN_BITS
        stream = torch.cuda.current_stream(self.device_index).cuda_stream
        self.launch(x, multiplier, cap, vocab_offset, mailbox, rank, stream)

    def launch(
        self,
        x: torch.Tensor,
        multiplier: float,
        cap: float,
        vocab_offset: int,
        mailbox: CandidateMailbox,
        rank: int,
        stream: int,
    ) -> None:
        """`run` on the raw `stream`, without checking the arguments."""
        self.kernel(
            x.data_ptr(),
            mailbox.slot(rank),
            mailbox.generation_bytes,
            mailbox.seqs.data_ptr(),
            self._lists,
            self._tickets,
            x.shape[0],
            x.stride(0),
            vocab_offset,
            multiplier,
            cap,
            stream,
        )
