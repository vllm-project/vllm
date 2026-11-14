# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The target's vocab-parallel logits all-gather over NVLink, in one CuTe DSL
kernel.

After each decode step's target forward, the LM head leaves every TP rank the
logits of its vocabulary shard, `[rows, 62080]` BF16 at TP4, and the sampler
needs all of them on every rank. The stock `tensor_model_parallel_all_gather`
runs an NCCL ring all-gather into rank-major blocks, then a copy into
`[rows, vocabulary]` order: about 20 us at one request and 100 us at sixteen
(64 rows, 8 MB a rank) on GB300. `LogitsAllGather` writes the gathered
logits in their final order directly, one kernel, through a Lamport mailbox
in symmetric memory.

Protocol. That of the sublayer boundary kernel (`..boundary.all_reduce_hc`),
with peer stores instead of multicast ones. A rank's mailbox holds
`[generation][peer slot][pack]` 16-byte packs, a slot per other rank
(`(source - receiver - 1) mod world`). The rank's logits are `packs`
16-byte packs. The grid has `min(ceil(packs / 128), resident)` CTAs of 128
threads (`LogitsAllGather.ctas`), `resident` being how many the GPU holds
at once (Liveness), so thread `i` of the grid's `T` threads owns packs
`i, i + T, i + 2T, ...`: one pack while the call fits in `128 * resident`
packs, a few at 64 rows at TP4 and four times as many at 256. Every thread
runs two phases over its packs, each in increasing pack order:

1. publish: for every pack it owns, write the pack into its place of the
   output and store it into every peer's mailbox slot of this rank (over
   NVLink);
2. receive: for every pack it owns, poll its own mailbox's slots of that pack
   until no other rank's holds the empty-slot sentinel, write them into the
   output and reset them to the sentinel.

An empty slot holds BF16 negative zero in every value; every value of the
gathered logits that is a negative zero, this rank's own included, goes out as
positive zero, so data and flag are the same bytes and every rank holds the
same bits.

Liveness. A polling thread waits for threads of the other ranks, so the call
finishes only if every thread of every rank's grid eventually runs. CUDA
starts a grid's CTAs in no promised order, so a grid larger than the GPU holds
at once can hang: one rank's resident CTAs poll for packs whose owners on
another rank have not started, while that rank's resident CTAs poll for packs
whose owners wait for the slots the first rank's pollers hold. Hence:

- Residency bound: the grid has at most `resident` CTAs, the CTAs of this
  kernel an SM holds at once times the SM count. The CTAs an SM holds come
  from the CUDA occupancy calculator for the kernel's own cubin (registers,
  threads and shared memory), measured when the kernel is compiled; one CTA
  per SM if the cubin is not available. Nothing else holds an SM for good: the
  kernels before it on the stream complete without waiting on this one (it
  waits for them in `griddepcontrol.wait`, and a programmatic launch starts
  it only once all their CTAs have started), and this kernel never triggers
  its dependents early, so the next kernel takes no SM before all of this
  one's CTAs have exited. All CTAs of the grid are therefore resident together
  eventually, whatever order they start in.
- Per-thread order: a thread publishes all its packs before it polls for any,
  and polls them in increasing order, the same on every rank. Publishing
  never waits.
- No circular wait: a poll waits only for a peer thread's publish phase,
  which waits for nothing but that thread's residency, which the bound
  guarantees.

This assumes that no kernel on another stream that waits on these ranks holds
SMs beside it; in Model Runner V2's decode loop the target's forward, its
logits gather and the draft passes run on one stream. The CPU model's
residency mode (`test_logits_protocol.py`) checks the bound over every start
order on a GPU of a few slots, and that a grid of one CTA per pack beyond the
slots, polling as it publishes, hangs.

Generations. As in the boundary kernel: thread 0 of every CTA reads the
generation word for its CTA and counts the CTA on an arrival counter; the last
CTA to arrive stores the next generation and a zero counter at its end. Two
generations suffice: a rank starts call `n + 1` only after its call `n`
completed, which needs every rank's call `n` data, and each rank stores that
data only after its own call `n - 1` (including the resets of the slots it
read) completed; so a rank storing call `n + 1` into generation
`(n + 1) % 2` finds those slots reset. Every slot a call writes is read and
reset by its receiver within the same call, whatever the sizes of other calls.
The CPU model in `tests/models/qwen4_exp/specialized_decode/
test_logits_protocol.py` checks this exchange over every interleaving, with
one generation and a missing reset as the defects it must catch.

Why peer stores. At 64 rows a rank sends 8 MB: multicast would bring every rank
four copies (32 MB) over NVLink, its own included, peer stores three (24 MB);
at one row both are latency-bound.

Stalls. A thread that polls longer than `stall_timeout` stores a diagnostic
word in host memory (`stall_report`) and traps, so the step raises a CUDA
error instead of hanging.

Rows (`RowsAllGather`). The same kernel with `row_packs` 0 writes the
ranks' packs one after the other in rank order instead of interleaving them
by row: rank `q`'s `packs` packs go to packs `[q packs, q packs + packs)`
of the output, so every rank's contiguous `[rows, width]` arrive as rows
`[q rows, q rows + rows)` of `[world * rows, width]`. The row-split
hyper-connection reads (`..runtime.split`) gather the residual streams of
the ranks' rows with it. Only the output addresses differ; the exchange is the
one above.

The kernel launches with programmatic dependent launch and never triggers its
dependents early. Every rank must make the same calls in the same order, on one
stream.
"""

import tempfile
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
    SENTINEL_BF16X2,
    arrive_u32,
    empty_halves_x4,
    global_timer,
    load_shared_u32,
    load_volatile_u32,
    report_stall,
    sanitize_bf16x2,
    store_shared_u32,
    store_u32,
    store_x4,
)
from ..hc._ptx import load_x4
from ._ptx import load_u64, load_volatile_words, store_x4_peer

THREADS = 128
VALUES = 8  # BF16 values of a 16-byte pack
# Dynamic shared memory the occupancy query assumes a CTA takes: at least the
# kernel's (a 4-byte word), so the residency bound errs low.
SMEM_BOUND = 1024
GENERATIONS = 2
GENERATION_WORD = 0
ARRIVALS_WORD = 1
# Seconds a thread polls before it reports a stall and traps (default).
STALL_TIMEOUT_S = 30.0


class LogitsGatherKernel:
    """The logits all-gather of one rank (module docstring).

    Args:
        world: Ranks of the group (at least 2).
        rank: This rank.
        row_packs: 16-byte packs of a rank's logits row (shard / 8), or 0 to
            write the ranks' packs in rank order (module docstring, Rows).
        capacity: Packs of a mailbox slot (the largest call's rows times
            `row_packs`, or its packs).
        stall_ns: Nanoseconds a thread polls before it reports a stall.

    """

    def __init__(
        self, world: int, rank: int, row_packs: int, capacity: int, stall_ns: int
    ) -> None:
        assert world >= 2 and 0 <= rank < world and row_packs >= 0 and stall_ns > 0
        self.world = world
        self.rank = rank
        self.row_packs = row_packs
        self.capacity = capacity
        self.stall_ns = stall_ns

    @cute.jit
    def __call__(
        self,
        logits: Int64,
        out: Int64,
        mailbox: Int64,
        peers: Int64,
        state: Int64,
        stall: Int64,
        packs: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """Launch `ctas` CTAs (at most as many as the GPU holds) on `stream` for
        `packs` packs (rows times `row_packs`): the rank's contiguous BF16
        logits, the BF16 output `[rows, world * shard]`, this rank's
        mailbox, the device array of every rank's mailbox address, the
        protocol words and the host-memory stall word."""
        self.kernel(logits, out, mailbox, peers, state, stall, packs).launch(
            grid=(ctas, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        logits: Int64,
        out: Int64,
        mailbox: Int64,
        peers: Int64,
        state: Int64,
        stall: Int64,
        packs: Int32,
    ):
        """Publish every pack the thread owns, then receive them, each in
        increasing order (module docstring)."""
        world = self.world
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        first = block * THREADS + tid
        stride = grid * THREADS
        allocator = cutlass.utils.SmemAllocator()
        generation_shared = allocator.allocate_array(
            Uint32, 1, byte_alignment=16
        ).toint()

        cute.arch.griddepcontrol_wait()

        # Thread 0 reads the generation for the CTA, then arrives; the last
        # arriver advances it at its end (module docstring).
        arrival = Uint32(0)
        if tid == 0:
            word = load_volatile_u32(state + GENERATION_WORD * 4)
            store_shared_u32(generation_shared, word)
            arrival = arrive_u32(state + ARRIVALS_WORD * 4, word)
        cute.arch.sync_threads()
        generation = Int64(load_shared_u32(generation_shared))

        row_bytes = world * self.row_packs * 16
        # Publish: the thread's packs into the output and every peer's mailbox.
        pack = first
        while pack < packs:
            words = load_x4(logits + Int64(pack) * 16)
            published = []
            for i in cutlass.range_constexpr(4):
                published.append(sanitize_bf16x2(words[i]))
            if cutlass.const_expr(self.row_packs == 0):
                # Rank order (module docstring, Rows).
                own = out + Int64(self.rank * packs + pack) * 16
            else:
                row = pack // self.row_packs
                column = pack - row * self.row_packs
                own = (
                    out
                    + Int64(row) * row_bytes
                    + Int64(self.rank * self.row_packs + column) * 16
                )
            store_x4(own, tuple(published))
            for peer in cutlass.range_constexpr(world):
                if cutlass.const_expr(peer != self.rank):
                    # This rank's slot in the peer's mailbox.
                    index = (self.rank - peer - 1) % world
                    base = load_u64(peers + peer * 8)
                    slot = (generation * (world - 1) + index) * self.capacity + Int64(
                        pack
                    )
                    store_x4_peer(base + slot * 16, tuple(published))
            pack = pack + stride

        # Receive: poll the other ranks' slots of each pack until none is
        # empty, write them out and reset them.
        sources = [r for r in range(world) if r != self.rank]
        received = cute.make_rmem_tensor(cute.make_layout((world - 1, 4)), Uint32)
        pack = first
        while pack < packs:
            if cutlass.const_expr(self.row_packs == 0):
                column = pack
                row_packs = packs
                out_row = out
            else:
                row = pack // self.row_packs
                column = pack - row * self.row_packs
                row_packs = Int32(self.row_packs)
                out_row = out + Int64(row) * row_bytes
            begin = global_timer()
            pending = Int32(1)
            while pending != Int32(0):
                pending = Int32(0)
                for i in cutlass.range_constexpr(world - 1):
                    index = (sources[i] - self.rank - 1) % world
                    slot = (generation * (world - 1) + index) * self.capacity + Int64(
                        pack
                    )
                    got = load_volatile_words(mailbox + slot * 16)
                    for k in cutlass.range_constexpr(4):
                        received[i, k] = got[k]
                    pending = pending | empty_halves_x4(got)
                if pending != Int32(0):  # noqa: SIM102
                    if global_timer() - begin > Int64(self.stall_ns):
                        missing = Int64(0)
                        for i in cutlass.range_constexpr(world - 1):
                            index = (sources[i] - self.rank - 1) % world
                            slot = (
                                generation * (world - 1) + index
                            ) * self.capacity + Int64(pack)
                            empty = empty_halves_x4(
                                load_volatile_words(mailbox + slot * 16)
                            )
                            missing = missing | (Int64(empty) << sources[i])
                        # The ranks still missing, this rank, the generation,
                        # the pack.
                        report_stall(
                            stall,
                            (Int64(1) << 62)
                            | (missing << 48)
                            | (Int64(self.rank) << 40)
                            | (generation << 32)
                            | Int64(pack),
                        )
            sentinel = Uint32(SENTINEL_BF16X2)
            for i in cutlass.range_constexpr(world - 1):
                index = (sources[i] - self.rank - 1) % world
                store_x4(
                    out_row + Int64(sources[i] * row_packs + column) * 16,
                    (received[i, 0], received[i, 1], received[i, 2], received[i, 3]),
                )
                slot = (generation * (world - 1) + index) * self.capacity + Int64(pack)
                store_x4(mailbox + slot * 16, (sentinel, sentinel, sentinel, sentinel))
            pack = pack + stride

        # The last CTA to arrive advances the generation for the next call.
        if tid == 0:  # noqa: SIM102
            if arrival == Uint32(grid - 1):
                store_u32(
                    state + GENERATION_WORD * 4,
                    Uint32((generation + 1) % GENERATIONS),
                )
                store_u32(state + ARRIVALS_WORD * 4, Uint32(0))


def _checked(result: tuple):
    """The value of a CUDA driver call's `(error, value)` result (None for
    a call without one); raise on an error."""
    error, *values = result
    if error != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {error}")
    return values[0] if values else None


def _ctas_per_sm(cubin: bytes) -> int:
    """CTAs of `THREADS` threads of the one kernel in `cubin` that an SM of
    the current device holds at once (the CUDA occupancy calculator)."""
    library = _checked(cuda.cuLibraryLoadData(cubin, None, None, 0, None, None, 0))
    try:
        assert _checked(cuda.cuLibraryGetKernelCount(library)) == 1
        kernel = _checked(cuda.cuLibraryEnumerateKernels(1, library))[0]
        function = _checked(cuda.cuKernelGetFunction(kernel))
        return _checked(
            cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                function, THREADS, SMEM_BOUND
            )
        )
    finally:
        _checked(cuda.cuLibraryUnload(library))


@cache
def _compiled(
    device_index: int,
    world: int,
    rank: int,
    row_packs: int,
    capacity: int,
    stall_ns: int,
):
    """`LogitsGatherKernel` compiled for a device, rank and shard, and
    the CTAs of it an SM holds at once (module docstring, Liveness)."""
    with (
        torch.accelerator.device_index(device_index),
        tempfile.TemporaryDirectory() as dump,
    ):
        compiled = cute.compile(
            LogitsGatherKernel(world, rank, row_packs, capacity, stall_ns),
            *(Int64(0),) * 6,
            Int32(1),
            Int32(1),
            make_fake_stream(),
            options=f"--dump-dir={dump} --keep-cubin",
        )
        cubin = compiled.artifacts.CUBIN
        # One CTA fits on any SM (the launch fails otherwise).
        per_sm = _ctas_per_sm(cubin) if cubin else 1
    return compiled, per_sm


class LogitsAllGather:
    """All-gather of vocab-parallel logits rows along the vocabulary
    (module docstring), for 1 to `max_rows` rows of `shard` logits.

    Construction is collective over `group` (at least two ranks): every
    rank compiles the kernel, allocates its mailbox and maps the others'.
    `accepts` decides from the tensor's shape alone, so every rank takes the
    same path. Calls may be captured into CUDA graphs; every rank must make the
    same calls in the same order.

    Args:
        group: The tensor-parallel process group.
        shard: Logits a rank holds per row (a multiple of 8).
        max_rows: Rows of the largest call.
        device: This rank's GPU.
        stall_timeout: Seconds a thread polls for the other ranks' logits
            before it reports a stall (`stall_report`) and traps.

    """

    # Whether the kernel writes the ranks' packs in rank order (module
    # docstring, Rows) instead of interleaving them by row.
    rank_major = False
    # What the stall report says it waited for.
    payload = "logits"

    def __init__(
        self,
        group: dist.ProcessGroup,
        shard: int,
        max_rows: int,
        device: torch.device,
        stall_timeout: float = STALL_TIMEOUT_S,
    ) -> None:
        assert shard % VALUES == 0 and max_rows > 0
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        assert self.world >= 2
        self.shard = shard
        self.max_rows = max_rows
        self.device = torch.device(device)
        self.row_packs = shard // VALUES
        capacity = max_rows * self.row_packs
        self.stall_ns = int(stall_timeout * 1e9)
        self._kernel, per_sm = _compiled(
            self.device.index,
            self.world,
            self.rank,
            0 if self.rank_major else self.row_packs,
            capacity,
            self.stall_ns,
        )
        # The residency bound (module docstring, Liveness).
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.resident = per_sm * sms
        self._state = torch.zeros(2, dtype=torch.int32, device=self.device)
        self._stall_host = torch.zeros(1, dtype=torch.int64).pin_memory()
        self._stall = get_accelerator_view_from_cpu_tensor(self._stall_host)
        # [generation][peer slot][pack] of 8 BF16 values.
        self._mailbox = symm_mem.empty(
            (GENERATIONS, self.world - 1, capacity * VALUES),
            dtype=torch.bfloat16,
            device=self.device,
        )
        handle = symm_mem.rendezvous(self._mailbox, group)
        self._peers = torch.tensor(
            [int(handle.buffer_ptrs[r]) for r in range(self.world)],
            dtype=torch.int64,
            device=self.device,
        )
        # Every slot starts empty: BF16 negative zero.
        self._mailbox.view(torch.int16).fill_(-32768)
        torch.accelerator.synchronize(self.device)
        dist.barrier(group=group)

    @property
    def mailbox_bytes(self) -> int:
        """Bytes of this rank's mailbox."""
        return self._mailbox.numel() * self._mailbox.element_size()

    def ctas(self, rows: int) -> int:
        """CTAs of a call of `rows` rows: one per `THREADS` packs, at most
        `resident` (module docstring, Liveness)."""
        return min(-(-rows * self.row_packs // THREADS), self.resident)

    def accepts(self, logits: torch.Tensor) -> bool:
        """Whether `logits` fits: BF16 `[rows, shard]`, 1 to `max_rows`
        rows (decided from its shape, the same on every rank)."""
        return (
            logits.dim() == 2
            and logits.dtype == torch.bfloat16
            and logits.shape[1] == self.shard
            and 0 < logits.shape[0] <= self.max_rows
        )

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Every rank's logits concatenated along the vocabulary in rank
        order, BF16 `[rows, world * shard]`, as
        `tensor_model_parallel_all_gather(logits)` lays them out; negative
        zeros arrive as positive zeros, on every rank.

        Args:
            logits: This rank's logits, BF16 `[rows, shard]`.

        """
        assert self.accepts(logits), f"logits {tuple(logits.shape)} do not fit"
        if not logits.is_contiguous() or logits.data_ptr() % 16:
            logits = logits.contiguous()
        rows = logits.shape[0]
        out = torch.empty(
            (rows, self.world * self.shard), dtype=logits.dtype, device=logits.device
        )
        self._kernel(
            Int64(logits.data_ptr()),
            Int64(out.data_ptr()),
            Int64(self._mailbox.data_ptr()),
            Int64(self._peers.data_ptr()),
            Int64(self._state.data_ptr()),
            Int64(self._stall.data_ptr()),
            Int32(rows * self.row_packs),
            Int32(self.ctas(rows)),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return out

    def warm_up(self) -> None:
        """Run one call of one row of zeros on every rank (collective)."""
        self(torch.zeros(1, self.shard, dtype=torch.bfloat16, device=self.device))
        torch.accelerator.synchronize(self.device)

    def stall_report(self) -> str | None:
        """What a stalled call reported before it trapped, or None. Reads host
        memory only, so it works after the CUDA error."""
        word = int(self._stall_host[0])
        if not word:
            return None
        missing = [r for r in range(self.world) if word >> (48 + r) & 1]
        return (
            f"rank {word >> 40 & 0xFF} timed out in generation {word >> 32 & 0xFF} "
            f"waiting for the {self.payload} of rank(s) {missing} "
            f"(pack {word & 0xFFFFFFFF})"
        )


class RowsAllGather(LogitsAllGather):
    """All-gather of each rank's rows (module docstring, Rows): every rank's
    `[rows, width]` BF16 rows stacked in rank order, for 1 to `max_rows`
    rows a rank. Construction and calls as `LogitsAllGather`, with
    `width` (a multiple of 8) in place of `shard`."""

    rank_major = True
    payload = "rows"

    def __call__(
        self, rows: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Every rank's `rows` stacked in rank order, BF16
        `[world * rows, width]`, as `torch.cat` of the ranks' tensors lays
        them out; negative zeros arrive as positive zeros, on every rank.

        Args:
            rows: This rank's rows, BF16 `[rows, width]`.
            out: Where to write them: contiguous, 16-byte aligned BF16
                `[world * rows, width]`; None allocates it.

        """
        assert self.accepts(rows), f"rows {tuple(rows.shape)} do not fit"
        if not rows.is_contiguous() or rows.data_ptr() % 16:
            rows = rows.contiguous()
        count = rows.shape[0]
        if out is None:
            out = torch.empty(
                (self.world * count, self.shard), dtype=rows.dtype, device=rows.device
            )
        assert out.shape == (self.world * count, self.shard)
        assert out.dtype == rows.dtype and out.is_contiguous()
        assert out.data_ptr() % 16 == 0
        self._kernel(
            Int64(rows.data_ptr()),
            Int64(out.data_ptr()),
            Int64(self._mailbox.data_ptr()),
            Int64(self._peers.data_ptr()),
            Int64(self._state.data_ptr()),
            Int64(self._stall.data_ptr()),
            Int32(count * self.row_packs),
            Int32(self.ctas(count)),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return out
