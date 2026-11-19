# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MTP draft passes' greedy token choice over vocab-parallel logits, in one
CuTe DSL kernel.

Each MTP draft pass ends by choosing its draft token: the argmax of the LM
head's logits over the vocabulary. Under tensor parallelism every rank holds
the logits of its vocabulary shard (`[rows, 62080]` BF16 at TP4), and the
stock path gathers all of them over NCCL (8 MB at 16 requests), copies them
into vocabulary order and then takes the argmax: 15-20 us per pass at one
request and about 65 us at sixteen on GB300. vLLM's
`use_local_argmax_reduction` speculative option lets the draft model choose
the token itself (`get_top_tokens`); `TokenChoice` is that choice
without gathering logits: each rank finds the best token of its shard and the
ranks exchange one 8-byte record per row.

Order. A token's *key* is `(ordered logit bits << 24) | (0xFFFFFF - index)`
for its vocabulary `index`: the order-preserving bits of the BF16 logit
(the sign bit flipped for non-negative values, all bits flipped for negative
ones), with NaN taken as -inf and -0 as +0. The largest key is therefore the
largest logit, ties going to the lowest index: the token `greedy_argmax`
(the stock draft choice: first index of the maximum, NaN as -inf) returns on
the gathered logits. Taking the maximum of keys is exact and order-free, so
every rank finds the same token whatever order the partial maxima arrive in.

Kernel. One CTA of 128 threads per (chunk of logits, row): each thread loads
two 16-byte packs (a 2048-logit chunk), or eight (8192) when the launch would
otherwise exceed `LARGE_FROM_CTAS` CTAs (many rows over a whole vocabulary
at TP1, where fewer, larger CTAs make fewer atomics and launch waves), keys
its values, and the CTA reduces them (`redux.sync`, then shared memory). A
chunk wholly inside the vocabulary (all of a row's but its last) keys two
values an instruction (`_ptx.pack_key_full`: `add.rn.bf16x2` with +0 turns
-0 into +0, `max.bf16x2` with -inf turns NaN into -inf, then integer ops on
both halves); the row's last chunk keys one value at a time and drops the
positions past the vocabulary (`_ptx.pack_key`). Both define the same keys.
Thread 0 folds the CTA's
key into the row's best key with a GPU-scope `atom.max` and counts the CTA
on the row's counter (acquire-release). The CTA that finds the count at
`chunks - 1` is the row's last: every other CTA's `max` happened before
its add, so the best key it swaps out (resetting it to 0) is the rank's
maximum; it zeroes the counter for the next launch. The next launch touches
both words only after this one completed (stream order; a programmatic
dependent launch waits in `griddepcontrol.wait`, which returns after this
grid completed, and this kernel never triggers its dependents early).

Exchange (TP > 1). The row's last CTA exchanges the rank's best key with its
peers through tagged 8-byte records in symmetric memory,
`records[row][parity][source]` u64 on every rank, zero at allocation. A
rank's own per-row epoch counter `e` (every rank runs the same launches, so
all hold the same `e`) gives the tag `1 + e mod (2^23 - 1)` (never 0) and
the parity `e & 1`. The thread stores `(key
<< 23) | tag` into slot `[row][parity][rank]` of every peer (one relaxed
8-byte store each: key and tag land together), polls its own slots
`[row][parity][source]` until each carries the tag, takes the maximum and
advances `e`.

Why the records need no reset. A slot is written once per epoch of its row, and
a launch runs one epoch per row. A peer writes parity `p` again only in epoch
`e + 2`, which it reaches only after collecting this rank's epoch `e + 1`
record, which this rank publishes in a later launch, after this launch read
its epoch `e` slots. So no record is overwritten before it is read, and a
stale record (epoch `e - 2` or older, the ranks being at most one epoch
apart) carries another tag. With one slot per row instead of two, a fast peer's
epoch `e + 1` record could overwrite its epoch `e` one before this rank
read it: the exchange would hang. The CPU model in
`tests/models/qwen4_exp/specialized_decode/test_logits_protocol.py` checks
both, over every interleaving of a small model.

Liveness (TP > 1). A row's last CTA polls for the peers' records of the row,
so a launch finishes only if every rank's row-last CTAs eventually run. CUDA
starts a grid's CTAs in no promised order, and a grid of many rows (31 CTAs a
row at TP4, 8 once a launch passes `LARGE_FROM_CTAS` CTAs) can exceed what
the GPU holds at once, so the polling CTAs must never hold every slot. Hence:

- Residency bound: only a row's last CTA polls, one per row, and a launch takes
  at most `capacity` rows, which the constructor requires to be below the
  device's SM count (the processor builds it for at most `MAX_DECODE_REQUESTS`
  rows, 64 at TP4). One CTA of this kernel fits on an SM (the launch fails
  otherwise), and nothing else holds an SM for good: the kernels before it on
  the stream complete without waiting on this one (it waits for them in
  `griddepcontrol.wait`), and it never triggers its dependents early. So at
  any time some SM can run a CTA that does not poll.
- Per-CTA order: every other CTA loads, reduces, folds its key and counts
  itself, and exits without waiting; the row's last CTA publishes its record
  to every peer before it polls.
- No circular wait: whatever order the CTAs start in, every non-polling CTA
  eventually runs on a free SM, so every row's last CTA on every rank is
  reached and publishes; a poll waits only for those publishes.

This assumes that no kernel on another stream that waits on these ranks holds
SMs beside it; Model Runner V2 runs the draft passes on one stream. Launches of
more rows than `capacity` take the stock path, decided from the row count, so
every rank agrees. The CPU model's residency mode
(`test_logits_protocol.py`) checks this over every start order on a GPU of a
few slots, a grid above the slots included, and that a launch with as many
rows as slots hangs. At TP1 nothing polls.

Stalls. A poll that waits longer than `stall_timeout` (a rank that skipped a
launch or ran another one) stores a diagnostic word in host memory
(`stall_report`) and traps, so the step raises a CUDA error rather than
hanging.

Scope: BF16 logits `[rows, shard]` with 16-byte aligned rows, 1 to
`capacity` rows a launch, vocabularies below `2^24`. Every rank of the
group must launch it for the same rows in the same order.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from ..boundary._ptx import (
    global_timer,
    load_shared_u32,
    report_stall,
    store_shared_u32,
    store_u32,
)
from ..hc._ptx import load_x4
from ._ptx import (
    INDEX_MASK,
    arrive_acq_rel_u32,
    atomic_max_u64,
    exchange_u64,
    load_record,
    load_u64,
    max_u32,
    max_u64,
    min_u32,
    pack_key,
    pack_key_full,
    store_record,
    store_u64,
    warp_max_u32,
)

THREADS = 128
WARPS = THREADS // 32
VALUES = 8  # BF16 values of a 16-byte pack
PACKS = 2  # packs a thread loads: 2048-logit chunks
LARGE_PACKS = 8  # 8192-logit chunks, for launches past LARGE_FROM_CTAS CTAs
LARGE_FROM_CTAS = 1024
TAG_BITS = 23
TAG_MASK = (1 << TAG_BITS) - 1
# Seconds a poll waits before it reports a stall and traps (default).
STALL_TIMEOUT_S = 30.0


class TokenChoiceKernel:
    """The token choice of one rank (module docstring).

    Args:
        world: Ranks of the group (1: no exchange).
        rank: This rank.
        shard: Logits a row holds on this rank (a multiple of 8).
        valid: The first `valid` of them are vocabulary entries; the rest
            (padding) never win.
        vocab_offset: The vocabulary index of the shard's first logit.
        stall_ns: Nanoseconds a poll waits before it reports a stall.
        packs: 16-byte packs a thread loads (`PACKS` or `LARGE_PACKS`).

    """

    def __init__(
        self,
        world: int,
        rank: int,
        shard: int,
        valid: int,
        vocab_offset: int,
        stall_ns: int,
        packs: int = PACKS,
    ) -> None:
        assert world >= 1 and 0 <= rank < world and stall_ns > 0
        assert shard % VALUES == 0 and 0 < valid <= shard
        assert vocab_offset + valid <= INDEX_MASK + 1
        # A chunk position must fit the CTA key's low 16 bits.
        assert packs > 0 and THREADS * VALUES * packs <= 1 << 16
        self.world = world
        self.rank = rank
        self.shard = shard
        self.valid = valid
        self.vocab_offset = vocab_offset
        self.stall_ns = stall_ns
        self.packs = packs
        self.chunk = THREADS * VALUES * packs
        self.chunks = (valid + self.chunk - 1) // self.chunk

    @cute.jit
    def __call__(
        self,
        logits: Int64,
        row_stride: Int32,
        tokens: Int64,
        best: Int64,
        counts: Int64,
        epochs: Int64,
        records: Int64,
        peers: Int64,
        stall: Int64,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """Launch on `stream` for `rows` rows: the BF16 logits (row
        stride `row_stride` values), the INT64 `tokens [rows]` output, the
        per-row best keys, counters and epochs, this rank's records and the
        device array of every rank's records address (`world > 1` only),
        and the host-memory stall word."""
        self.kernel(
            logits, row_stride, tokens, best, counts, epochs, records, peers, stall
        ).launch(
            grid=(self.chunks, rows, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        logits: Int64,
        row_stride: Int32,
        tokens: Int64,
        best: Int64,
        counts: Int64,
        epochs: Int64,
        records: Int64,
        peers: Int64,
        stall: Int64,
    ):
        """One CTA: one chunk of one row."""
        tid, _, _ = cute.arch.thread_idx()
        chunk, row, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        allocator = cutlass.utils.SmemAllocator()
        warp_keys = allocator.allocate_array(Uint32, WARPS, byte_alignment=16).toint()

        cute.arch.griddepcontrol_wait()

        # This thread's packs; a pack past the row's end is read at the row's
        # last pack and counts nothing (its positions pass the limit).
        row_base = logits + Int64(row) * Int64(row_stride) * 2
        start = Uint32(chunk * self.chunk)
        limit = Uint32(self.valid) - start
        offsets = []
        packs = []
        for j in cutlass.range_constexpr(self.packs):
            offset = Uint32(j * THREADS * VALUES) + Uint32(tid * VALUES)
            column = min_u32(start + offset, Uint32(self.shard - VALUES))
            offsets.append(offset)
            packs.append(load_x4(row_base + Int64(column) * 2))
        key = Uint32(0)
        # Every position of a chunk wholly inside the vocabulary counts (all
        # chunks but a row's last): the packed keys, uniform over the CTA.
        if limit >= Uint32(self.chunk):
            for j in cutlass.range_constexpr(self.packs):
                key = max_u32(key, pack_key_full(packs[j], offsets[j]))
        else:
            for j in cutlass.range_constexpr(self.packs):
                key = max_u32(key, pack_key(packs[j], offsets[j], limit))
        key = warp_max_u32(key)
        if lane == 0:
            store_shared_u32(warp_keys + warp * 4, key)
        cute.arch.sync_threads()

        if tid == 0:
            cta_key = load_shared_u32(warp_keys)
            for w in cutlass.range_constexpr(1, WARPS):
                cta_key = max_u32(cta_key, load_shared_u32(warp_keys + w * 4))
            # Chunk key -> token key: the logit bits, then the inverted
            # vocabulary index.
            position = Int64(Uint32(0xFFFF) - (cta_key & Uint32(0xFFFF)))
            index = Int64(self.vocab_offset) + Int64(start) + position
            token_key = (Int64(cta_key >> Uint32(16)) << 24) | (
                Int64(INDEX_MASK) - index
            )
            row_best = best + Int64(row) * 8
            row_count = counts + Int64(row) * 4
            atomic_max_u64(row_best, token_key)
            arrived = arrive_acq_rel_u32(row_count)
            if arrived == Uint32(self.chunks - 1):
                total = exchange_u64(row_best, Int64(0))
                store_u32(row_count, Uint32(0))
                if cutlass.const_expr(self.world > 1):
                    # The group's best key (module docstring, "Exchange").
                    epoch_word = epochs + Int64(row) * 8
                    epoch = load_u64(epoch_word)
                    tag = Int64(1) + epoch % Int64(TAG_MASK)
                    slot = (Int64(row) * 2 + (epoch & Int64(1))) * self.world
                    record = (total << TAG_BITS) | tag
                    for peer in cutlass.range_constexpr(self.world):
                        if cutlass.const_expr(peer != self.rank):
                            base = load_u64(peers + peer * 8)
                            store_record(base + (slot + self.rank) * 8, record)
                    begin = global_timer()
                    for source in cutlass.range_constexpr(self.world):
                        if cutlass.const_expr(source != self.rank):
                            address = records + (slot + source) * 8
                            got = load_record(address)
                            while (got & Int64(TAG_MASK)) != tag:
                                if global_timer() - begin > Int64(self.stall_ns):
                                    # The awaited rank, this rank, the row, the
                                    # epoch's low bits.
                                    report_stall(
                                        stall,
                                        (Int64(1) << 62)
                                        | (Int64(source) << 48)
                                        | (Int64(self.rank) << 40)
                                        | (Int64(row) << 24)
                                        | (epoch & Int64(0xFFFFFF)),
                                    )
                                got = load_record(address)
                            total = max_u64(total, got >> TAG_BITS)
                    store_u64(epoch_word, epoch + 1)
                store_u64(
                    tokens + Int64(row) * 8,
                    Int64(INDEX_MASK) - (total & Int64(INDEX_MASK)),
                )


@cache
def _compiled(
    device_index: int,
    world: int,
    rank: int,
    shard: int,
    valid: int,
    vocab_offset: int,
    stall_ns: int,
    packs: int,
):
    """`TokenChoiceKernel` compiled for a device, shard and chunk."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            TokenChoiceKernel(world, rank, shard, valid, vocab_offset, stall_ns, packs),
            Int64(0),
            Int32(shard),
            *(Int64(0),) * 7,
            Int32(1),
            make_fake_stream(),
        )


class TokenChoice:
    """The greedy token choice over this rank's vocabulary shard of the logits
    (module docstring).

    Construction is collective over `group` (every rank allocates its
    records and maps the others'); `group` None or of one rank builds the
    local variant. Launches may be captured into CUDA graphs; every rank must
    make the same launches in the same order.

    Args:
        group: The tensor-parallel process group, or None at TP 1.
        shard: Logits a row holds on this rank.
        valid: How many of them are vocabulary entries (the rest is padding).
        vocab_offset: The vocabulary index of the shard's first logit.
        capacity: Rows of the largest launch (below the SM count at TP > 1).
        device: This rank's GPU.
        stall_timeout: Seconds a poll waits for the other ranks before it
            reports a stall (`stall_report`) and traps.

    """

    def __init__(
        self,
        group: dist.ProcessGroup | None,
        shard: int,
        valid: int,
        vocab_offset: int,
        capacity: int,
        device: torch.device,
        stall_timeout: float = STALL_TIMEOUT_S,
    ) -> None:
        self.group = group
        self.world = 1 if group is None else dist.get_world_size(group)
        self.rank = 0 if group is None else dist.get_rank(group)
        self.shard = shard
        self.valid = valid
        self.vocab_offset = vocab_offset
        self.capacity = capacity
        self.device = torch.device(device)
        self.stall_ns = int(stall_timeout * 1e9)
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        if self.world > 1 and capacity >= sms:
            # The residency bound (module docstring, Liveness).
            raise ValueError(
                f"TokenChoice of {capacity} rows needs fewer rows than the "
                f"{sms} SMs: one CTA a row polls"
            )
        # Per row: the rank's best key and arrived CTAs of the current
        # launch (reset by the row's last CTA), and the exchange epoch.
        self._best = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        self._counts = torch.zeros(capacity, dtype=torch.int32, device=self.device)
        self._epochs = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        self._records = torch.zeros(0, dtype=torch.int64, device=self.device)
        self._peers = torch.zeros(self.world, dtype=torch.int64, device=self.device)
        self._stall_host = torch.zeros(1, dtype=torch.int64)
        self._stall = torch.zeros(1, dtype=torch.int64, device=self.device)
        # Compiled before any rank polls (the warm-up synchronizes): the
        # 2048-logit chunks, and the 8192-logit ones when a launch of
        # ``capacity`` rows would pass LARGE_FROM_CTAS CTAs.
        self._small_chunks = -(-valid // (THREADS * VALUES * PACKS))
        self._kernels = {
            packs: _compiled(
                self.device.index,
                self.world,
                self.rank,
                shard,
                valid,
                vocab_offset,
                self.stall_ns,
                packs,
            )
            for packs in (PACKS, LARGE_PACKS)
            if packs == PACKS or capacity * self._small_chunks > LARGE_FROM_CTAS
        }
        if self.world > 1:
            assert group is not None
            self._stall_host = self._stall_host.pin_memory()
            self._stall = get_accelerator_view_from_cpu_tensor(self._stall_host)
            # [row][parity][source]; zero is never a valid tag.
            self._records = symm_mem.empty(
                (capacity * 2 * self.world,), dtype=torch.int64, device=self.device
            )
            handle = symm_mem.rendezvous(self._records, group)
            self._records.zero_()
            self._peers = torch.tensor(
                [int(handle.buffer_ptrs[r]) for r in range(self.world)],
                dtype=torch.int64,
                device=self.device,
            )
            torch.accelerator.synchronize(self.device)
            dist.barrier(group=group)

    def takes(self, rows: int) -> bool:
        """Whether a launch of `rows` rows fits (host state only)."""
        return 0 < rows <= self.capacity

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """The greedy token of every row, INT64 `[rows]`, the same on every
        rank (`greedy_argmax` of the gathered logits).

        Args:
            logits: This rank's logits, BF16 `[rows, shard]` with unit
                column stride and 16-byte aligned rows.

        """
        rows = logits.shape[0]
        assert self.takes(rows), f"{rows} rows, at most {self.capacity}"
        assert logits.dtype == torch.bfloat16 and logits.dim() == 2
        assert logits.shape[1] == self.shard and logits.stride(1) == 1
        assert logits.stride(0) % VALUES == 0 and logits.data_ptr() % 16 == 0
        tokens = torch.empty(rows, dtype=torch.int64, device=logits.device)
        # The row count picks the chunk size, the same on every rank.
        large = rows * self._small_chunks > LARGE_FROM_CTAS
        self._kernels[LARGE_PACKS if large else PACKS](
            Int64(logits.data_ptr()),
            Int32(logits.stride(0)),
            Int64(tokens.data_ptr()),
            Int64(self._best.data_ptr()),
            Int64(self._counts.data_ptr()),
            Int64(self._epochs.data_ptr()),
            Int64(self._records.data_ptr()),
            Int64(self._peers.data_ptr()),
            Int64(self._stall.data_ptr()),
            Int32(rows),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return tokens

    def warm_up(self) -> None:
        """Launch every compiled kernel once on rows of zeros, on every rank
        (collective): one row, and `capacity` rows when those take the
        large chunks."""
        for rows in sorted({1, self.capacity}):
            self(
                torch.zeros(rows, self.shard, dtype=torch.bfloat16, device=self.device)
            )
        torch.accelerator.synchronize(self.device)

    def stall_report(self) -> str | None:
        """What a stalled launch reported before it trapped, or None. Reads
        host memory only, so it works after the CUDA error."""
        word = int(self._stall_host[0])
        if not word:
            return None
        return (
            f"rank {word >> 40 & 0xFF} timed out waiting for rank "
            f"{word >> 48 & 0xFF}'s token record (row {word >> 24 & 0xFFFF}, "
            f"epoch {word & 0xFFFFFF} mod 2^24)"
        )
