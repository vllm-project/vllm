# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Up projection and stream mix of a decode hyper-connection read in CuTe DSL.

The stock read ends with an up projection `gate = low @ up.T` (a BF16 GEMM,
`up [10240, 320]`, 6.6 MB) followed by the gated mean over the four streams,
`h = mean_s(sigmoid(gate_s) * xn_s)` (`hc_gate_mix`). `hc_up_mix`
does both in one kernel, with the up rows on chip before the grid-dependency
wait:

* CTA `(b, y)` owns the 16 hidden columns `[16 b, 16 b + 16)` (160 column
  blocks); warp `s` owns stream `s`, i.e. up rows `2560 s + 16 b + (0..15)`,
  which it loads into registers (160 B per lane) before `griddepcontrol.wait`
  and so streams while the down projection still runs. Every CTA triggers its
  dependents once those loads are issued.
* After the wait, the CTA stages `low` of its token rows in shared memory
  (the MMA B operand) and loads its `xn` values meanwhile; each warp runs 20
  `m16n8k16` MMAs per eight-row group, applies the sigmoid to the FP32 sum and
  multiplies by `xn`. The four streams meet in shared memory and are averaged
  in stream order, `h = bf16((((m0 + m1) + m2) + m3) / 4)`, the order of
  `hc_gate_mix`.

Numerics: the gate stays in FP32 (the stock chain rounds it to BF16 before the
sigmoid), the deviation the stock fused `hc_up_gate_mix` makes as well.

Grid: `(160, ceil(groups / per))` with `per` eight-row groups per CTA, at
most four: one CTA per column block for one group, else two. When the rows do
not fill a CTA's `8 per` MMA columns (a short last group, or a second CTA
with fewer groups than the first), the empty columns are not staged and their
outputs are dropped.

All-gather (`HcUpMixGather`, TP > 1). In a row-split read (the
boundary kernel's `split` calls) each rank reads only its own `rows` rows,
while the next block needs the block input of all `world * rows` rows on
every rank. The gather variant ends with a Lamport all-gather of its output,
the protocol of the logits gather (`..logits.gather`) with this kernel's
work split: after the stream mean, the CTA's 16-byte pieces (one row, eight
columns) are written into the rank's rows of the output (rank `q`'s rows
are `[q rows, q rows + rows)`) and stored into every peer's mailbox
`[generation][source rank][row][2560]`; then the same thread polls its own
mailbox for the three peers' pieces at the same row and columns, writes them
into the output and resets the slots. Negative zeros go out as positive zeros
on every rank, the owner's own copy included (the empty-slot sentinel), so all
ranks hold the same bits. Generations and the arrival counter are the
boundary kernel's. Liveness: every thread publishes before it polls, a poll
waits only for the same thread of a peer's grid, and every CTA of the grid is
resident by the time any later kernel takes an SM (a programmatic launch of
the dependents waits until every CTA has triggered, and the grid has at most
320 CTAs at any row count up to `GATHER_MAX_ROWS`, two per column block of
at most four groups each; with its launch bound of four CTAs an SM and at most
22 KiB of shared memory a CTA, GB300 holds them all at once, beside the
finishing down projection, which waits on nothing of this kernel).
"""

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

from .. import HC_HIDDEN, HC_LOW_RANK, HC_STREAMS
from ..boundary._ptx import (
    SENTINEL_BF16X2,
    arrive_u32,
    empty_halves_x4,
    global_timer,
    load_shared_u32,
    load_volatile_u32,
    pack_bf16x2,
    report_stall,
    sanitize_bf16x2,
    store_shared_u32,
    store_u32,
    store_x4,
)
from ..logits._ptx import load_u64, load_volatile_words, store_x4_peer
from ._ptx import (
    copy_async_16,
    copy_async_wait_all,
    evict_first_policy,
    load_bf16,
    load_shared_f32,
    load_shared_x4,
    load_weight_x4,
    mma_chunk,
    store_bf16,
    store_shared_f32,
)

HIDDEN = HC_HIDDEN
STREAMS = HC_STREAMS
WIDTH = STREAMS * HIDDEN
LOW_RANK = HC_LOW_RANK
MAX_ROWS = 64

COLUMNS = 16  # hidden columns of a CTA (the MMA's M: weight rows)
THREADS = 32 * STREAMS  # one warp per stream
CHUNKS = LOW_RANK // 32  # 32-wide k chunks: 10
GROUP_ROWS = 8  # token rows of one MMA (its N)
MAX_GROUPS_PER_CTA = 4
# A staged low row: 320 values padded to 44 16-byte units, so the B-fragment
# reads of a warp hit distinct banks.
LOW_ROW_BYTES = (LOW_RANK + 32) * 2
# The all-gather variant (module docstring): rows a rank reads at most (eight
# groups, four per CTA: 320 CTAs), the mailbox generations and protocol words.
GATHER_MAX_ROWS = MAX_ROWS
GENERATIONS = 2
GENERATION_WORD = 0
ARRIVALS_WORD = 1
# Seconds a thread polls before it reports a stall and traps (default).
STALL_TIMEOUT_S = 30.0


def _sigmoid(x: Float32) -> Float32:
    """`1 / (1 + exp(-x))`."""
    return cute.math.rcp(1.0 + cute.math.exp(-x), approx=True)


def groups_per_cta(rows: int) -> int:
    """Eight-row groups per CTA for `rows` token rows: one CTA per column
    block for one group, else two CTAs, with at most four groups each."""
    groups = (rows + GROUP_ROWS - 1) // GROUP_ROWS
    return min(MAX_GROUPS_PER_CTA, (groups + 1) // 2)


class HcUpMix:
    """Up projection, gate and stream mean for one group count per CTA.

    Args:
        per: Eight-row token groups per CTA (1 to 4).
        world: Ranks of the all-gather of the output (1: none; module
            docstring, All-gather).
        rank: This rank (`world > 1`).
        capacity: Rows of a rank in a mailbox generation (`world > 1`).
        stall_ns: Nanoseconds a thread polls before it reports a stall and
            traps (`world > 1`).

    """

    def __init__(
        self,
        per: int,
        world: int = 1,
        rank: int = 0,
        capacity: int = 0,
        stall_ns: int = 0,
    ) -> None:
        assert 1 <= per <= MAX_GROUPS_PER_CTA
        assert world == 1 or (
            0 <= rank < world and 0 < capacity <= GATHER_MAX_ROWS and stall_ns > 0
        )
        self.per = per
        self.world = world
        self.rank = rank
        self.capacity = capacity
        self.stall_ns = stall_ns
        self.rows = GROUP_ROWS * per  # staged token rows of a CTA
        self.shared_bytes = max(
            self.rows * LOW_ROW_BYTES, STREAMS * self.rows * COLUMNS * 4
        )

    @cute.jit
    def __call__(
        self,
        low: Int64,
        low_stride: Int32,
        xn: Int64,
        up: Int64,
        out: Int64,
        mailbox: Int64,
        peers: Int64,
        state: Int64,
        stall: Int64,
        rows: Int32,
        ctas_y: Int32,
        stream: cuda.CUstream,
    ):
        """Launch on `stream`: `low [rows, low_stride]`, `xn [rows,
        10240]`, `up [10240, 320]`, `out [rows, 2560]` BF16 addresses
        (`world > 1`: `out [world * rows, 2560]`, and the rank's mailbox,
        the device array of every rank's mailbox address, the protocol words
        and the host-memory stall word)."""
        self.kernel(
            low, low_stride, xn, up, out, mailbox, peers, state, stall, rows
        ).launch(
            grid=(HIDDEN // COLUMNS, ctas_y, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=4,
        )

    @cute.kernel
    def kernel(
        self,
        low: Int64,
        low_stride: Int32,
        xn: Int64,
        up: Int64,
        out: Int64,
        mailbox: Int64,
        peers: Int64,
        state: Int64,
        stall: Int64,
        rows: Int32,
    ):
        """One CTA: 16 hidden columns of all four streams, for `per` groups
        (`world > 1`: then the all-gather of those outputs)."""
        per = self.per
        staged = self.rows
        tid, _, _ = cute.arch.thread_idx()
        block, part, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)  # the stream
        lane = tid % 32
        g = lane // 4
        t = lane % 4
        c0 = block * COLUMNS

        # The up rows of this warp's stream, before the grid-dependency wait.
        policy = evict_first_policy()
        address_lo = up + Int64((warp * HIDDEN + c0 + g) * LOW_RANK + 8 * t) * 2
        address_hi = address_lo + 8 * LOW_RANK * 2
        lo = []
        hi = []
        for j in cutlass.range_constexpr(CHUNKS):
            lo.append(load_weight_x4(address_lo + j * 64, policy))
            hi.append(load_weight_x4(address_hi + j * 64, policy))

        # The staged low rows, later reused for the stream products
        # [stream][token][column] FP32; then the generation word of the
        # all-gather (one word in a 16-byte block).
        allocator = cutlass.utils.SmemAllocator()
        shared = allocator.allocate_array(
            cutlass.Uint8, self.shared_bytes, byte_alignment=16
        ).toint()
        generation_shared = allocator.allocate_array(
            Uint32, 4, byte_alignment=16
        ).toint()

        # Dependents may launch now: they read this kernel's output only after
        # their own grid-dependency wait. Later kernels in the chain are covered
        # too: each kernel ends only after its own wait returned, so a
        # completed kernel implies completed predecessors. The trigger follows
        # the weight loads on purpose: placed before them, ptxas schedules the
        # wait right after the trigger and sinks the non-coherent weight loads
        # below it.
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()

        arrival = Uint32(0)
        if cutlass.const_expr(self.world > 1):  # noqa: SIM102
            # Thread 0 reads the generation for the CTA, then arrives; the
            # last arriver advances it at its end (the boundary kernel's
            # protocol). The barriers below publish the word to the CTA.
            if tid == 0:
                word = load_volatile_u32(state + GENERATION_WORD * 4)
                store_shared_u32(generation_shared, word)
                arrival = arrive_u32(state + ARRIVALS_WORD * 4, word)

        # Stage low of the CTA's rows: 40 16-byte pieces a row, all in flight.
        # Rows past `rows` are not staged; their MMA columns read whatever
        # the shared memory holds, possibly NaN. That cannot reach a real row:
        # an m16n8k16 output column depends only on its own B column, so a
        # padding token's column changes only its own gate and product, and
        # the mean below stores only rows below `rows`.
        base = staged * part
        pieces = staged * (LOW_RANK // 8)
        for step in cutlass.range_constexpr((pieces + THREADS - 1) // THREADS):
            i = step * THREADS + tid
            if (i < pieces) & (base + i // (LOW_RANK // 8) < rows):
                local = i // (LOW_RANK // 8)
                k8 = i % (LOW_RANK // 8)
                token = cute.min(base + local, rows - 1)
                copy_async_16(
                    shared + local * LOW_ROW_BYTES + k8 * 16,
                    low + Int64(token * low_stride + 8 * k8) * 2,
                )
        # This lane's xn values of the epilogue: hidden column c0 + g + 8 (i >> 1)
        # of stream `warp`, token rows base + 8 f + 2 t + (i & 1).
        scales = []
        for f in cutlass.range_constexpr(per):
            for i in cutlass.range_constexpr(4):
                token = cute.min(base + GROUP_ROWS * f + 2 * t + (i & 1), rows - 1)
                column = warp * HIDDEN + c0 + g + 8 * (i >> 1)
                scales.append(load_bf16(xn + Int64(token * WIDTH + column) * 2))
        copy_async_wait_all()
        cute.arch.sync_threads()

        # Chunk-outer order interleaves the groups' independent accumulators.
        accs = []
        for _f in cutlass.range_constexpr(per):
            accs.append((Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)))
        for j in cutlass.range_constexpr(CHUNKS):
            for f in cutlass.range_constexpr(per):
                operand = load_shared_x4(
                    shared + (GROUP_ROWS * f + g) * LOW_ROW_BYTES + j * 64 + t * 16
                )
                accs[f] = mma_chunk(accs[f], lo[j], hi[j], operand)
        cute.arch.sync_threads()  # the staged rows are dead

        # Lane (g, t) holds D[g][2t], D[g][2t+1], D[g+8][2t], D[g+8][2t+1]:
        # hidden column c0 + g (+ 8) by token 8 f + 2t (+ 1).
        for f in cutlass.range_constexpr(per):
            for i in cutlass.range_constexpr(4):
                local = GROUP_ROWS * f + 2 * t + (i & 1)
                column = g + 8 * (i >> 1)
                store_shared_f32(
                    shared + ((warp * staged + local) * COLUMNS + column) * 4,
                    _sigmoid(accs[f][i]) * scales[4 * f + i],
                )
        cute.arch.sync_threads()

        # The mean over the streams, in stream order.
        outputs = staged * COLUMNS
        stream_bytes = outputs * 4
        if cutlass.const_expr(self.world == 1):
            for step in cutlass.range_constexpr((outputs + THREADS - 1) // THREADS):
                i = step * THREADS + tid
                if i < outputs:
                    row = base + i // COLUMNS
                    if row < rows:
                        element = shared + i * 4
                        total = load_shared_f32(element)
                        for s in cutlass.range_constexpr(1, STREAMS):
                            total = total + load_shared_f32(element + s * stream_bytes)
                        store_bf16(
                            out + Int64(row * HIDDEN + c0 + i % COLUMNS) * 2,
                            total * (1.0 / STREAMS),
                        )
        else:
            self._gather(
                shared, generation_shared, out, mailbox, peers, stall, base, rows, c0
            )
            # The last CTA to arrive advances the generation for the next call.
            if tid == 0:  # noqa: SIM102
                grid_x, grid_y, _ = cute.arch.grid_dim()
                if arrival == Uint32(grid_x * grid_y - 1):
                    generation = load_shared_u32(generation_shared)
                    store_u32(
                        state + GENERATION_WORD * 4,
                        Uint32((generation + 1) % GENERATIONS),
                    )
                    store_u32(state + ARRIVALS_WORD * 4, Uint32(0))

    @cute.jit
    def _gather(
        self,
        shared: Int32,
        generation_shared: Int32,
        out: Int64,
        mailbox: Int64,
        peers: Int64,
        stall: Int64,
        base: Int32,
        rows: Int32,
        c0: Int32,
    ):
        """The stream mean and the all-gather (module docstring): thread
        `2 r + h` of the CTA takes the eight columns `c0 + 8 h + [0, 8)`
        of its staged row `r`."""
        world = self.world
        staged = self.rows
        tid, _, _ = cute.arch.thread_idx()
        stream_bytes = staged * COLUMNS * 4
        if tid < staged * 2:
            local = tid // 2
            half = tid % 2
            row = base + local
            if row < rows:
                generation = Int64(load_shared_u32(generation_shared))
                words = []
                for v in cutlass.range_constexpr(4):
                    pair = []
                    for e in cutlass.range_constexpr(2):
                        element = shared + (local * COLUMNS + half * 8 + 2 * v + e) * 4
                        total = load_shared_f32(element)
                        for s in cutlass.range_constexpr(1, STREAMS):
                            total = total + load_shared_f32(element + s * stream_bytes)
                        pair.append(total * (1.0 / STREAMS))
                    words.append(sanitize_bf16x2(pack_bf16x2(pair[0], pair[1])))
                column = c0 + half * 8
                store_x4(
                    out + (Int64(self.rank * rows + row) * HIDDEN + Int64(column)) * 2,
                    tuple(words),
                )
                slots = []
                for r in cutlass.range_constexpr(world):
                    slots.append(
                        (
                            ((generation * world + r) * self.capacity + Int64(row))
                            * HIDDEN
                            + Int64(column)
                        )
                        * 2
                    )
                for peer in cutlass.range_constexpr(world):
                    if cutlass.const_expr(peer != self.rank):
                        store_x4_peer(
                            load_u64(peers + peer * 8) + slots[self.rank], tuple(words)
                        )
                # Poll the peers' pieces of this row and columns.
                received = cute.make_rmem_tensor(cute.make_layout((world, 4)), Uint32)
                start = global_timer()
                pending = Int32(1)
                while pending != Int32(0):
                    pending = Int32(0)
                    for r in cutlass.range_constexpr(world):
                        if cutlass.const_expr(r != self.rank):
                            got = load_volatile_words(mailbox + slots[r])
                            for k in cutlass.range_constexpr(4):
                                received[r, k] = got[k]
                            pending = pending | empty_halves_x4(got)
                    if pending != Int32(0):  # noqa: SIM102
                        if global_timer() - start > Int64(self.stall_ns):
                            missing = Int64(0)
                            for r in cutlass.range_constexpr(world):
                                if cutlass.const_expr(r != self.rank):
                                    empty = empty_halves_x4(
                                        load_volatile_words(mailbox + slots[r])
                                    )
                                    missing = missing | (Int64(empty) << r)
                            # The ranks still missing, this rank, the
                            # generation, the row and the column.
                            report_stall(
                                stall,
                                (Int64(1) << 62)
                                | (missing << 48)
                                | (Int64(self.rank) << 40)
                                | (generation << 32)
                                | (Int64(row) << 16)
                                | Int64(column),
                            )
                sentinel = Uint32(SENTINEL_BF16X2)
                for r in cutlass.range_constexpr(world):
                    if cutlass.const_expr(r != self.rank):
                        store_x4(
                            out + (Int64(r * rows + row) * HIDDEN + Int64(column)) * 2,
                            (
                                received[r, 0],
                                received[r, 1],
                                received[r, 2],
                                received[r, 3],
                            ),
                        )
                        store_x4(
                            mailbox + slots[r], (sentinel, sentinel, sentinel, sentinel)
                        )


@cache
def _compiled(
    device_index: int,
    per: int,
    world: int = 1,
    rank: int = 0,
    capacity: int = 0,
    stall_ns: int = 0,
):
    """`HcUpMix` compiled for a device and a group count per CTA (and,
    for `world > 1`, the all-gather of a rank)."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            HcUpMix(per, world, rank, capacity, stall_ns),
            Int64(0),
            Int32(8),
            *(Int64(0),) * 7,
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


def _check_operands(low: torch.Tensor, xn: torch.Tensor, up: torch.Tensor) -> None:
    """The operand checks of `hc_up_mix`."""
    rows = low.shape[0]
    assert low.dtype == xn.dtype == up.dtype == torch.bfloat16
    assert low.shape[1] == LOW_RANK and low.stride(1) == 1
    assert low.stride(0) % 8 == 0 and low.data_ptr() % 16 == 0
    assert xn.shape == (rows, WIDTH) and xn.is_contiguous()
    assert up.shape == (WIDTH, LOW_RANK) and up.is_contiguous()


def hc_up_mix(low: torch.Tensor, xn: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """`mean_s(sigmoid(low @ up_s.T) * xn_s)` over the four streams, with
    the gate in FP32, for 1-64 rows.

    Args:
        low: Low-rank activations, BF16 `[rows, 320]` with unit column stride
            and a row stride that is a multiple of 8 (a column view of the down
            projection's output is accepted).
        xn: Normalized streams, contiguous BF16 `[rows, 10240]`.
        up: Up-projection weight, contiguous BF16 `[10240, 320]`.

    Returns:
        The block input, BF16 `[rows, 2560]`.

    """
    rows = low.shape[0]
    assert 1 <= rows <= MAX_ROWS, f"hc_up_mix serves 1-{MAX_ROWS} rows, got {rows}"
    _check_operands(low, xn, up)
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=low.device)
    per = groups_per_cta(rows)
    groups = (rows + GROUP_ROWS - 1) // GROUP_ROWS
    _compiled(low.device.index, per)(
        Int64(low.data_ptr()),
        Int32(low.stride(0)),
        Int64(xn.data_ptr()),
        Int64(up.data_ptr()),
        Int64(out.data_ptr()),
        *(Int64(0),) * 4,
        Int32(rows),
        Int32((groups + per - 1) // per),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return out


class HcUpMixGather:
    """`hc_up_mix` of a rank's rows, followed by the all-gather of the
    block input over the TP group (module docstring, All-gather), in one
    kernel.

    Construction is collective over `group` (every rank allocates its
    mailbox and maps the others'). Calls may be captured into CUDA graphs;
    every rank must make the same calls in the same order, on one stream.

    Args:
        group: The tensor-parallel process group (at least two ranks).
        max_rows: Rows of a rank in the largest call (at most
            `GATHER_MAX_ROWS`).
        device: This rank's GPU.
        stall_timeout: Seconds a thread polls for the other ranks' rows
            before it reports a stall (`stall_report`) and traps.

    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        max_rows: int,
        device: torch.device,
        stall_timeout: float = STALL_TIMEOUT_S,
    ) -> None:
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        assert self.world >= 2 and 1 <= max_rows <= GATHER_MAX_ROWS
        self.max_rows = max_rows
        self.device = torch.device(device)
        self.stall_ns = int(stall_timeout * 1e9)
        self._state = torch.zeros(2, dtype=torch.int32, device=self.device)
        self._stall_host = torch.zeros(1, dtype=torch.int64).pin_memory()
        self._stall = get_accelerator_view_from_cpu_tensor(self._stall_host)
        # [generation][source rank][row][2560] (this rank's source slots unused).
        self._mailbox = symm_mem.empty(
            (GENERATIONS, self.world, max_rows, HIDDEN),
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

    def _kernel(self, per: int):
        """The compiled kernel of `per` groups per CTA for this rank."""
        return _compiled(
            self.device.index,
            per,
            self.world,
            self.rank,
            self.max_rows,
            self.stall_ns,
        )

    def warm_up(self) -> None:
        """Compile every variant, wait for every rank, then run one call of one
        row of zeros (collective)."""
        for per in {groups_per_cta(rows) for rows in range(1, self.max_rows + 1)}:
            self._kernel(per)
        torch.accelerator.synchronize(self.device)
        dist.barrier(group=self.group)
        low = torch.zeros(1, LOW_RANK, dtype=torch.bfloat16, device=self.device)
        xn = torch.zeros(1, WIDTH, dtype=torch.bfloat16, device=self.device)
        up = torch.zeros(WIDTH, LOW_RANK, dtype=torch.bfloat16, device=self.device)
        self(low, xn, up)
        torch.accelerator.synchronize(self.device)

    def __call__(
        self, low: torch.Tensor, xn: torch.Tensor, up: torch.Tensor
    ) -> torch.Tensor:
        """The block input of every rank's rows: `hc_up_mix(low, xn, up)` of
        each rank's `rows` rows, concatenated in rank order, BF16 `[world *
        rows, 2560]`; negative zeros as positive zeros, on every rank.

        Args:
            low: This rank's low-rank activations, as in `hc_up_mix`.
            xn: This rank's normalized streams, as in `hc_up_mix`.
            up: The up-projection weight, as in `hc_up_mix`.

        """
        rows = low.shape[0]
        assert 1 <= rows <= self.max_rows, f"{rows} rows, at most {self.max_rows}"
        _check_operands(low, xn, up)
        out = torch.empty(
            self.world * rows, HIDDEN, dtype=torch.bfloat16, device=low.device
        )
        per = groups_per_cta(rows)
        groups = (rows + GROUP_ROWS - 1) // GROUP_ROWS
        self._kernel(per)(
            Int64(low.data_ptr()),
            Int32(low.stride(0)),
            Int64(xn.data_ptr()),
            Int64(up.data_ptr()),
            Int64(out.data_ptr()),
            Int64(self._mailbox.data_ptr()),
            Int64(self._peers.data_ptr()),
            Int64(self._state.data_ptr()),
            Int64(self._stall.data_ptr()),
            Int32(rows),
            Int32((groups + per - 1) // per),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return out

    def stall_report(self) -> str | None:
        """What a stalled call reported before it trapped, or None. Reads host
        memory only, so it works after the CUDA error."""
        word = int(self._stall_host[0])
        if not word:
            return None
        missing = [r for r in range(self.world) if word >> (48 + r) & 1]
        return (
            f"rank {word >> 40 & 0xFF} timed out in generation {word >> 32 & 0xFF} "
            f"waiting for the block input of rank(s) {missing} (row "
            f"{word >> 16 & 0xFFFF}, column {word & 0xFFFF})"
        )
