# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TP4 PLE row gather of the specialized decode: the vocab-parallel PLE
lookup's exchange and dequantization in one CuTe DSL kernel.

The PLE layer of Qwen3.8-Flash-Next (`..ple_layer`, layer 1) embeds each
token as 16 rows of a large n-gram table (`ple_embed_dim` 2560 = 16 heads of
160 values), stored as FP8 E4M3 with one global scale. At TP > 1 the table is
split along its rows among the ranks (vocab-parallel): each rank holds a
contiguous row range in pinned host memory, looks up the rows of the step's
n-gram ids that fall in its range (zeros for the others, on a side stream one
layer ahead) and the stock path then sums the FP8 bytes of all ranks with an
NCCL all-reduce, copies the result into place and dequantizes it
(`rows.to(bf16) * scale.to(bf16)`): about 27 us at four requests on GB300,
the all-reduce 17 us of it.

Every row has exactly one owner, the rank whose range holds its id, so the sum
is a gather. `PleRowGather` gathers and dequantizes in one kernel,
through a Lamport mailbox in symmetric memory: the owner of a row dequantizes
it and stores it into its own output and into the same slot of every peer's
mailbox; every other rank polls that slot until the row is there. The output
is the BF16 `[tokens, 2560]` block the PLE projection reads, the stock
path's bits on every rank.

Units. A row is 160 FP8 bytes in the lookup buffer and 320 BF16 bytes in the
output; a pack is 8 values of a row (8 bytes in, 16 bytes out), 20 packs a
row. Pack `p` of the call is pack `p % 20` of row `p // 20`; row `r`
is head `r % 16` of token `r // 16`. The grid has `min(ceil(packs /
128), resident)` CTAs of 128 threads, so thread `i` of the grid's `T`
threads owns packs `i, i + T, ...`. Each thread runs two phases over its
packs, each in increasing pack order:

1. publish: for each pack of a row this rank owns (its id in
   `[vocab_start, vocab_end)`), dequantize the 8 values, store them into the
   output and into the pack's slot of every peer's mailbox;
2. receive: for each pack of a row another rank owns, poll this rank's
   mailbox slot of the pack until it holds data, store it into the output and
   reset the slot.

A mailbox holds `[generation][pack]` 16-byte slots, one per pack of the
largest call: each slot has one writer per call (the row's owner), whatever
rank that is. An empty slot holds BF16 negative zero in every value; a
dequantized negative zero goes out as positive zero (the same value), so data
and flag are the same bytes and every rank holds the same bits.

Liveness, generations and stalls are the logits gather's
(`..logits.gather`): at most `resident` CTAs, each thread publishes all
its packs before it polls any, the kernel waits for its predecessor before it
touches anything, two generations advanced by the last CTA to arrive, and a
poll longer than `stall_timeout` reports (`stall_report`) and traps. Every
rank must make the same calls in the same order on one stream, with the same
ids: the ids come from the step's tokens, which every rank holds alike.

Generations need one more condition here. Call `n + 2` reuses call `n`'s
slots, so no rank may start call `n + 2` before every rank has completed
call `n` (read and reset its slots). In the logits gather every rank
publishes in every call, so a rank completes call `n + 1` only after every
rank started it, that is completed call `n`. Here only owners publish: when
a rank owns no row of call `n + 1`, the others can complete call `n + 1`
and store call `n + 2`'s rows into its mailbox while it is still in call
`n`, which it would read as call `n`'s. So between calls `n` and
`n + 2` either every rank owns a row of call `n + 1`, or every rank runs a
collective that no rank completes before every rank has entered it (an
all-reduce, the boundary kernel) after its call `n + 1`. The executor's
steps call each gather once and run every decoder layer's all-reduce between
two calls (`runtime/ple.py`).

One difference: with `trigger_early` (the executor's setting) every thread
triggers the dependents after its publish phase (a CTA counts as triggered at
its first thread's trigger), so the next kernel launches while the gather
polls (0.5 us a call at TP4). That keeps the residency argument: a
programmatic launch waits until every CTA of the grid has triggered or
exited, so every CTA of the gather is resident by then, and the
next kernel's CTAs only take SMs beside them; nothing the gather still does
needs another SM. The next kernel reads the gather's output only after its
`griddepcontrol.wait`, which waits for the whole grid to complete and flush.

Every id must fall in some rank's range, or its row's slot is never written
and the receivers stall; the n-gram hashing maps every id into the table's
rows, which the ranges cover.
"""

import tempfile
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Float32, Int32, Int64, Uint32
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
from ..logits._ptx import load_u64, load_volatile_words, store_x4_peer
from ..logits.gather import _ctas_per_sm
from ._ptx import dequant_e4m3x4, load_x2

THREADS = 128
VALUES = 8  # values of a pack: 8 FP8 bytes in, 8 BF16 values out
GENERATIONS = 2
GENERATION_WORD = 0
ARRIVALS_WORD = 1
# Seconds a thread polls before it reports a stall and traps (default).
STALL_TIMEOUT_S = 30.0


class PleGatherKernel:
    """The PLE row gather of one rank (module docstring).

    Args:
        world: Ranks of the group (at least 2).
        rank: This rank.
        row_packs: Packs of a row (head dimension / 8).
        capacity: Packs of a mailbox generation (the largest call's).
        stall_ns: Nanoseconds a thread polls before it reports a stall.
        trigger_early: Trigger the dependents once the CTA has published
            (module docstring, Liveness), so the next kernel's prologue runs
            while this one polls.

    """

    def __init__(
        self,
        world: int,
        rank: int,
        row_packs: int,
        capacity: int,
        stall_ns: int,
        trigger_early: bool,
    ) -> None:
        assert world >= 2 and 0 <= rank < world and row_packs > 0 and stall_ns > 0
        self.world = world
        self.rank = rank
        self.row_packs = row_packs
        self.capacity = capacity
        self.stall_ns = stall_ns
        self.trigger_early = trigger_early

    @cute.jit
    def __call__(
        self,
        lookup: Int64,
        ids: Int64,
        out: Int64,
        mailbox: Int64,
        peers: Int64,
        state: Int64,
        stall: Int64,
        scale: Float32,
        vocab_start: Int64,
        vocab_end: Int64,
        packs: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """Launch `ctas` CTAs (at most as many as the GPU holds) on
        `stream` for `packs` packs: the FP8 lookup rows (this rank's rows
        valid), their int64 ids, the BF16 output, this rank's mailbox, the
        device array of every rank's mailbox address, the protocol words, the
        host-memory stall word, the BF16 scale (as FP32) and this rank's id
        range."""
        self.kernel(
            lookup,
            ids,
            out,
            mailbox,
            peers,
            state,
            stall,
            scale,
            vocab_start,
            vocab_end,
            packs,
        ).launch(
            grid=(ctas, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        lookup: Int64,
        ids: Int64,
        out: Int64,
        mailbox: Int64,
        peers: Int64,
        state: Int64,
        stall: Int64,
        scale: Float32,
        vocab_start: Int64,
        vocab_end: Int64,
        packs: Int32,
    ):
        """Publish every pack of an owned row the thread holds, then receive
        the others, each in increasing order (module docstring)."""
        world = self.world
        row_packs = self.row_packs
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

        # Publish: each pack of an owned row into the output and every peer's
        # mailbox.
        pack = first
        while pack < packs:
            row_id = load_u64(ids + Int64(pack // row_packs) * 8)
            mine = Int32(0)
            if row_id >= vocab_start:  # noqa: SIM102
                if row_id < vocab_end:
                    mine = Int32(1)
            if mine != Int32(0):
                low, high = load_x2(lookup + Int64(pack) * VALUES)
                w0, w1 = dequant_e4m3x4(low, scale)
                w2, w3 = dequant_e4m3x4(high, scale)
                words = (
                    sanitize_bf16x2(w0),
                    sanitize_bf16x2(w1),
                    sanitize_bf16x2(w2),
                    sanitize_bf16x2(w3),
                )
                store_x4(out + Int64(pack) * 16, words)
                slot = generation * self.capacity + Int64(pack)
                for peer in cutlass.range_constexpr(world):
                    if cutlass.const_expr(peer != self.rank):
                        base = load_u64(peers + peer * 8)
                        store_x4_peer(base + slot * 16, words)
            pack = pack + stride
        if cutlass.const_expr(self.trigger_early):
            cute.arch.griddepcontrol_launch_dependents()

        # Receive: poll the slot of each pack of a row another rank owns until
        # it holds data, write it out and reset it.
        sentinel = Uint32(SENTINEL_BF16X2)
        received = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
        pack = first
        while pack < packs:
            row_id = load_u64(ids + Int64(pack // row_packs) * 8)
            mine = Int32(0)
            if row_id >= vocab_start:  # noqa: SIM102
                if row_id < vocab_end:
                    mine = Int32(1)
            if mine == Int32(0):
                slot = mailbox + (generation * self.capacity + Int64(pack)) * 16
                begin = global_timer()
                pending = Int32(1)
                while pending != Int32(0):
                    got = load_volatile_words(slot)
                    for k in cutlass.range_constexpr(4):
                        received[k] = got[k]
                    pending = empty_halves_x4(got)
                    if pending != Int32(0):  # noqa: SIM102
                        if global_timer() - begin > Int64(self.stall_ns):
                            # This rank, the generation, the pack.
                            report_stall(
                                stall,
                                (Int64(1) << 62)
                                | (Int64(self.rank) << 40)
                                | (generation << 32)
                                | Int64(pack),
                            )
                store_x4(
                    out + Int64(pack) * 16,
                    (received[0], received[1], received[2], received[3]),
                )
                store_x4(slot, (sentinel, sentinel, sentinel, sentinel))
            pack = pack + stride

        # The last CTA to arrive advances the generation for the next call.
        if tid == 0:  # noqa: SIM102
            if arrival == Uint32(grid - 1):
                store_u32(
                    state + GENERATION_WORD * 4,
                    Uint32((generation + 1) % GENERATIONS),
                )
                store_u32(state + ARRIVALS_WORD * 4, Uint32(0))


@cache
def _compiled(
    device_index: int,
    world: int,
    rank: int,
    row_packs: int,
    capacity: int,
    stall_ns: int,
    trigger_early: bool,
):
    """`PleGatherKernel` compiled for a device and rank, and the CTAs
    of it an SM holds at once (module docstring, Liveness)."""
    with (
        torch.accelerator.device_index(device_index),
        tempfile.TemporaryDirectory() as dump,
    ):
        compiled = cute.compile(
            PleGatherKernel(world, rank, row_packs, capacity, stall_ns, trigger_early),
            *(Int64(0),) * 7,
            Float32(1.0),
            Int64(0),
            Int64(0),
            Int32(1),
            Int32(1),
            make_fake_stream(),
            options=f"--dump-dir={dump} --keep-cubin",
        )
        cubin = compiled.artifacts.CUBIN
        # One CTA fits on any SM (the launch fails otherwise).
        per_sm = _ctas_per_sm(cubin) if cubin else 1
    return compiled, per_sm


class PleRowGather:
    """Gather and dequantization of a vocab-parallel FP8 PLE lookup (module
    docstring), for 1 to `max_rows` rows of `head_dim` values.

    Construction is collective over `group` (at least two ranks): every
    rank compiles the kernel, allocates its mailbox and maps the others'.
    Calls may be captured into CUDA graphs; every rank must make the same
    calls in the same order, with the same ids, and after a call in which
    some rank owns no row, run a collective over the group before the next
    call (module docstring, Generations need one more condition).

    Args:
        group: The process group the table is split over.
        head_dim: Values of a row (a multiple of 8).
        max_rows: Rows (tokens times heads) of the largest call.
        vocab_start: First id of this rank's rows.
        vocab_end: One past the last id of this rank's rows.
        scale: The table's global scale, as the stock path rounds it (BF16).
        device: This rank's GPU.
        stall_timeout: Seconds a thread polls for another rank's row before it
            reports a stall (`stall_report`) and traps.
        trigger_early: Trigger the dependents once a CTA has published.

    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        head_dim: int,
        max_rows: int,
        vocab_start: int,
        vocab_end: int,
        scale: float,
        device: torch.device,
        stall_timeout: float = STALL_TIMEOUT_S,
        trigger_early: bool = False,
    ) -> None:
        assert head_dim % VALUES == 0 and max_rows > 0
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        assert self.world >= 2
        self.head_dim = head_dim
        self.max_rows = max_rows
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.scale = float(torch.tensor(scale).to(torch.bfloat16))
        self.device = torch.device(device)
        self.row_packs = head_dim // VALUES
        capacity = max_rows * self.row_packs
        self.stall_ns = int(stall_timeout * 1e9)
        self._kernel, per_sm = _compiled(
            self.device.index,
            self.world,
            self.rank,
            self.row_packs,
            capacity,
            self.stall_ns,
            trigger_early,
        )
        # The residency bound (module docstring, Liveness).
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.resident = per_sm * sms
        self._state = torch.zeros(2, dtype=torch.int32, device=self.device)
        self._stall_host = torch.zeros(1, dtype=torch.int64).pin_memory()
        self._stall = get_accelerator_view_from_cpu_tensor(self._stall_host)
        # [generation][pack] of 8 BF16 values.
        self._mailbox = symm_mem.empty(
            (GENERATIONS, capacity * VALUES), dtype=torch.bfloat16, device=self.device
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

    def __call__(
        self, lookup: torch.Tensor, ids: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """Fill `out` with every row of `ids`, dequantized: the stock
        all-reduced lookup's `rows.to(bf16) * scale.to(bf16)`, negative
        zeros as positive zeros, on every rank.

        Args:
            lookup: This rank's FP8 lookup, `[..., head_dim]` contiguous,
                one row per id (the rows of other ranks' ids are not read).
            ids: The rows' int64 ids, contiguous, in the lookup's row order.
            out: BF16 output of `ids.numel() * head_dim` contiguous values.

        """
        rows = ids.numel()
        assert 0 < rows <= self.max_rows, f"{rows} rows do not fit"
        assert lookup.dtype == torch.float8_e4m3fn and lookup.is_contiguous()
        assert lookup.numel() == rows * self.head_dim
        assert ids.dtype == torch.int64 and ids.is_contiguous()
        assert out.dtype == torch.bfloat16 and out.is_contiguous()
        assert out.numel() == rows * self.head_dim
        self._kernel(
            Int64(lookup.data_ptr()),
            Int64(ids.data_ptr()),
            Int64(out.data_ptr()),
            Int64(self._mailbox.data_ptr()),
            Int64(self._peers.data_ptr()),
            Int64(self._state.data_ptr()),
            Int64(self._stall.data_ptr()),
            Float32(self.scale),
            Int64(self.vocab_start),
            Int64(self.vocab_end),
            Int32(rows * self.row_packs),
            Int32(self.ctas(rows)),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return out

    def stall_report(self) -> str | None:
        """What a stalled call reported before it trapped, or None. Reads host
        memory only, so it works after the CUDA error."""
        word = int(self._stall_host[0])
        if not word:
            return None
        return (
            f"rank {word >> 40 & 0xFF} timed out in generation {word >> 32 & 0xFF} "
            f"waiting for PLE pack {word & 0xFFFFFFFF}"
        )
