# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TP all-reduce of DeepSeek-V4.1's decode boundaries on SM80, in CuTe DSL.

Every decode layer of DeepSeek-V4.1 all-reduces two partial sums across the
tensor-parallel ranks: the attention output projection's and the MoE's, one
BF16 row of `hidden` values per token (24 rows at four requests with five
drafts each, 245 KB at TP8). vLLM's custom all-reduce does it by loading:
below 256 KB at TP8 each rank reads every peer's whole buffer over NVLink
(one-shot), above it a reduce-scatter and an all-gather separated by
barriers (two-shot). On A100 both are latency-bound at decode sizes, about
28 us at 24 rows and 25 us at 48, several times what the bytes need.

`TwoShotAllReduce` pushes instead of loading, and synchronizes through the
data rather than with barriers (a Lamport mailbox): every slot of a rank's
mailbox holds a sentinel, BF16 negative zero, until a peer's store replaces
it, so a reader polls a slot until none of its values is the sentinel.
Senders turn real negative zeros into positive zeros. One CTA per row, one
16-byte fragment (8 values) per thread, three phases:

1. **Scatter.** Each thread stores its fragment of the local partial into
   the mailbox of the rank that owns the fragment's slice of the row (rank
   `s` owns fragments `s * hidden / (8 * world)` onward).
2. **Reduce.** The row's fragments of this rank's slice, times the `world`
   sources, are one per thread: groups of `world` lanes poll one fragment
   from every source, add them in FP32 with a butterfly (every lane gets the
   same bits), round to BF16, and each lane stores the sum into one rank's
   mailbox.
3. **Gather.** Each thread polls its fragment of the reduced row in its own
   mailbox and writes it to the output.

Each rank sends `2 (world - 1) / world` of the row instead of
`world - 1` rows, and no phase waits for anything but its own data.

A reader resets the slots it consumed to the sentinel. The mailbox has three
generations, and each row (CTA) keeps the generation its next call uses,
advanced modulo three by every call: a peer writes the same generation of a
row again three calls later, which needs this rank's push of two calls later,
issued after this call completed. All ranks run the same sequence of calls
with the same row counts, so a row's generation agrees across ranks; it is
per row because a call of fewer rows leaves the other rows' generations
alone. The mailbox starts as sentinels and the generations at zero, so the
kernel needs no per-launch initialization and is CUDA-graph safe.

A row's CTA waits for the same row's CTA on every peer, so all of a call's
CTAs must be resident at once on every rank. A call therefore launches at
most one CTA per SM, whatever the kernel's registers, and above that many
rows each CTA takes rows `ctas` apart in increasing order: the lowest
unfinished row is then in progress on every rank, so every call completes.
This also needs every kernel that runs beside a call to finish without
waiting on it, which holds for the only one, the side-stream mHC
coefficients.

`TwoShotAllReducePostCollapse` is the same all-reduce followed, in the
same CTA, by the mHC boundary's post-mix, collapse and RMSNorm
(`mhc.PostCollapse`, whose CTAs have the same shape: a row per CTA, eight
values per thread). The loads that do not depend on the sum (the row's
streams, mixes and norm weight) are issued before the scatter, so they
overlap the waits, and the reduced row never goes back to memory: the
boundary saves a launch and a round trip of the sublayer output.

`TwoShotAllReduceFinalizePostCollapse` takes a MoE output whose top-k
reduction is still open (`UnfinalizedMoEOutput` with one unweighted GEMM2
row per route, routes of a token adjacent): before the scatter, each thread
forms its fragment of the rank's partial row from the token's `top_k`
route rows, their routing weights and the shared expert's row. The MoE's
top-k sum and shared-expert add then need no kernels of their own, as on
SM90, whose fused all-reduce does the same.

Numerics: each rank's slice sums the `world` partials in FP32 in a fixed
butterfly order and rounds once to BF16, so every rank gets the same bits;
the order differs from vLLM's kernels, so results differ from theirs by FP32
rounding of the sum. The fused variant's results are bitwise those of the
all-reduce followed by `PostCollapse`. The finalize variant forms the
partial row in FP32, a fused multiply-add per route in route order plus the
shared row, rounded once to BF16 (as SM90's kernel), so it differs from
vLLM's own top-k sum by rounding.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.common.ops.moe_tail_fusion.primitives import (
    LAMPORT_GENERATIONS,
    NEGATIVE_ZERO_BF16_PAIR,
    current_cu_stream,
    fragment_has_negative_zero,
    sanitize_negative_zero_u32,
    store_global_u32x4,
)
from vllm.platforms import current_platform

from .codecs import f32_from_bits, load_s32, load_u32, store_u32
from .mhc import (
    HC,
    PostCollapse,
    _select_first_stream,
    bf16_high,
    bf16_low,
    fma,
    pack,
    pack_bf16x2,
    unpack,
)

# BF16 values per fragment (one 16-byte store) and bytes per fragment.
VEC = 8
FRAGMENT_BYTES = 16
# The most rows a call takes: decode steps of up to 21 requests with five
# drafts each (B16 is 96 rows).
MAX_ROWS = 128


def _asm(result, operands, text, constraints, side_effects, loc, ip):
    """One inline PTX statement."""
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_u64(address: Int64, *, loc=None, ip=None) -> Int64:
    """One 64-bit word that the kernel only reads (a mailbox base)."""
    return Int64(
        _asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u64 $0, [$1];",
            "=l,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def load_fragment(address: Int64, *, volatile: bool = False, loc=None, ip=None):
    """A 16-byte fragment as four words; `volatile` for polling a slot that
    peers store into."""
    opcode = "ld.volatile.global.v4.u32" if volatile else "ld.global.v4.u32"
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        f"{opcode} {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        volatile,
        loc,
        ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), loaded, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


def store_fragment(address: Int64, words: list) -> None:
    """Store a 16-byte fragment, to this rank's memory or a peer's."""
    store_global_u32x4(Int64(address), [Uint32(word) for word in words], volatile=True)


@cute.jit
def poll(address: Int64, words: cute.Tensor):
    """Wait until the slot at `address` holds no sentinel, read it into
    `words` and reset it to the sentinel."""
    fragment = load_fragment(address, volatile=True)
    for i in cutlass.range_constexpr(4):
        words[i] = fragment[i]
    dirty = fragment_has_negative_zero(words)
    while dirty:
        fragment = load_fragment(address, volatile=True)
        for i in cutlass.range_constexpr(4):
            words[i] = fragment[i]
        dirty = fragment_has_negative_zero(words)
    sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR)
    store_fragment(address, [sentinel] * 4)


class TwoShotAllReduce:
    """The push-based two-shot Lamport all-reduce; see the module docstring.

    `world` ranks, rows of `hidden` BF16 values, at most `capacity`
    rows a call.
    """

    def __init__(self, hidden: int, world: int, capacity: int):
        assert hidden % (VEC * world) == 0
        self.hidden = hidden
        self.world = world
        self.capacity = capacity
        # Fragments of a row (threads per CTA) and of a rank's slice.
        self.fragments = hidden // VEC
        self.slice_fragments = self.fragments // world
        # Byte offset of the gather slots after the scatter slots.
        self.gather_base = LAMPORT_GENERATIONS * capacity * hidden * 2

    def mailbox_bytes(self) -> int:
        """Bytes of a rank's mailbox: scatter and gather slots, three
        generations each."""
        return 2 * self.gather_base

    @cute.jit
    def __call__(
        self,
        x: Int64,
        out: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(x, out, peers, generations, rank, rows).launch(
            grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        out: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        rows: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        # Rows in increasing order, ``ctas`` apart (see the module docstring).
        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            partial = load_fragment(
                x + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            self.reduce_row(partial, peers, rank, generation, row, tid, words)
            store_fragment(
                out + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES,
                [words[i] for i in range(4)],
            )
            # Every thread has read the generation; advance it for the next call.
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas

    @cute.jit
    def reduce_row(
        self,
        partial: list,
        peers: Int64,
        rank: Int32,
        generation: Int32,
        row: Int32,
        tid: Int32,
        words: cute.Tensor,
    ):
        """The three phases for row `row` (one CTA), from thread `tid`'s
        fragment of this rank's partial row (four packed words), leaving its
        fragment of the reduced row in `words`."""
        world = self.world
        slice_fragments = self.slice_fragments
        local = load_u64(peers + Int64(rank) * 8)

        # 1. Scatter: fragment tid of the row to the owner of its slice, at
        # scatter slot [generation][source rank][row][fragment in slice].
        owner = tid // slice_fragments
        store_fragment(
            load_u64(peers + Int64(owner) * 8)
            + self.scatter_slot(generation, rank, row, tid % slice_fragments),
            [sanitize_negative_zero_u32(word) for word in partial],
        )

        # 2. Reduce: lane group tid // world takes fragment tid // world of
        # this rank's slice, lane tid % world its copy from that source.
        source = tid % world
        poll(local + self.scatter_slot(generation, source, row, tid // world), words)
        sums = [
            bf16_low(words[i // 2]) if i % 2 == 0 else bf16_high(words[i // 2])
            for i in range(VEC)
        ]
        offset = 1
        for _ in cutlass.range_constexpr(world.bit_length() - 1):
            sums = [
                value + cute.arch.shuffle_sync_bfly(value, offset=offset)
                for value in sums
            ]
            offset *= 2
        reduced = [
            sanitize_negative_zero_u32(pack_bf16x2(sums[2 * i], sums[2 * i + 1]))
            for i in range(4)
        ]
        # The reduced fragment goes to rank `source` (each lane of the group
        # serves one rank), at gather slot [generation][row][fragment].
        store_fragment(
            load_u64(peers + Int64(source) * 8)
            + self.gather_slot(generation, row, rank * slice_fragments + tid // world),
            reduced,
        )

        # 3. Gather: this thread's fragment of the reduced row.
        poll(local + self.gather_slot(generation, row, tid), words)

    @cute.jit
    def scatter_slot(
        self, generation: Int32, source: Int32, row: Int32, fragment: Int32
    ) -> Int64:
        """Byte offset of a scatter slot in a mailbox."""
        return (
            ((Int64(generation) * self.world + source) * self.capacity + row)
            * self.slice_fragments
            + fragment
        ) * FRAGMENT_BYTES

    @cute.jit
    def gather_slot(self, generation: Int32, row: Int32, fragment: Int32) -> Int64:
        """Byte offset of a gather slot in a mailbox."""
        return (
            self.gather_base
            + ((Int64(generation) * self.capacity + row) * self.fragments + fragment)
            * FRAGMENT_BYTES
        )


class TwoShotAllReducePostCollapse(TwoShotAllReduce):
    """`TwoShotAllReduce` followed in the same CTA by `PostCollapse`
    (with the post-mix); see the module docstring."""

    def __init__(self, hidden: int, world: int, capacity: int, write_aux: bool):
        super().__init__(hidden, world, capacity)
        self.collapse = PostCollapse(hidden, has_post=True, write_aux=write_aux)

    @cute.jit
    def __call__(
        self,
        x: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        ctas: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            sums: cute.struct.MemRange[Float32, self.collapse.warps]

        self.kernel(
            x,
            peers,
            generations,
            rank,
            residual,
            post,
            comb,
            pre,
            pre_stride,
            weight,
            residual_out,
            layer_input,
            aux,
            rows,
            eps,
            Storage,
        ).launch(grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        eps: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sSums = storage.sums.get_tensor(cute.make_layout(self.collapse.warps))
        ctas, _, _ = cute.arch.grid_dim()
        # Rows in increasing order, ``ctas`` apart (see the module docstring).
        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            # The row's streams, mixes and norm weight do not depend on the sum.
            loaded = self.collapse.load_row(
                residual, post, comb, pre, pre_stride, weight, row, tid
            )
            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            partial = load_fragment(
                x + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            self.reduce_row(partial, peers, rank, generation, row, tid, words)
            xs = unpack([words[i] for i in range(4)])
            self.collapse.finish_row(
                xs, loaded, residual_out, layer_input, aux, eps, sSums, row, tid
            )
            # Every thread has read the generation; advance it for the next call.
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas


class TwoShotAllReduceFinalizePostCollapse(TwoShotAllReducePostCollapse):
    """`TwoShotAllReducePostCollapse` of a MoE output with its top-k
    reduction open, `top_k` routes a token; see the module docstring."""

    def __init__(
        self, hidden: int, world: int, capacity: int, write_aux: bool, top_k: int
    ):
        super().__init__(hidden, world, capacity, write_aux)
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        routes: Int64,
        route_weights: Int64,
        shared: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        ctas: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            sums: cute.struct.MemRange[Float32, self.collapse.warps]

        self.kernel(
            routes,
            route_weights,
            shared,
            peers,
            generations,
            rank,
            residual,
            post,
            comb,
            pre,
            pre_stride,
            weight,
            residual_out,
            layer_input,
            aux,
            rows,
            eps,
            Storage,
        ).launch(grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        routes: Int64,
        route_weights: Int64,
        shared: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        eps: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sSums = storage.sums.get_tensor(cute.make_layout(self.collapse.warps))
        ctas, _, _ = cute.arch.grid_dim()
        top_k = self.top_k
        # Rows in increasing order, ``ctas`` apart (see the module docstring).
        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            # The token's route rows (route t * top_k + k is row t's k-th),
            # their weights and the shared row, then everything else the row
            # reads, all before any of it is used.
            route_words = [
                load_fragment(
                    routes
                    + ((Int64(row) * top_k + k) * self.fragments + tid) * FRAGMENT_BYTES
                )
                for k in range(top_k)
            ]
            scales = [
                f32_from_bits(load_u32(route_weights + (Int64(row) * top_k + k) * 4))
                for k in range(top_k)
            ]
            shared_words = load_fragment(
                shared + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            loaded = self.collapse.load_row(
                residual, post, comb, pre, pre_stride, weight, row, tid
            )
            sums = unpack(route_words[0])
            sums = [value * scales[0] for value in sums]
            for k in cutlass.range_constexpr(1, top_k):
                values = unpack(route_words[k])
                sums = [fma(values[e], scales[k], sums[e]) for e in range(VEC)]
            shared_values = unpack(shared_words)
            partial = pack([sums[e] + shared_values[e] for e in range(VEC)])
            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            self.reduce_row(partial, peers, rank, generation, row, tid, words)
            xs = unpack([words[i] for i in range(4)])
            self.collapse.finish_row(
                xs, loaded, residual_out, layer_input, aux, eps, sSums, row, tid
            )
            # Every thread has read the generation; advance it for the next call.
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas


@cache
def _kernel(device_index: int, hidden: int, world: int, capacity: int):
    """`TwoShotAllReduce` compiled for a device and shape."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            TwoShotAllReduce(hidden, world, capacity),
            *(Int64(0),) * 4,
            Int32(0),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _collapse_kernel(
    device_index: int, hidden: int, world: int, capacity: int, write_aux: bool
):
    """`TwoShotAllReducePostCollapse` compiled for a device and variant."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            TwoShotAllReducePostCollapse(hidden, world, capacity, write_aux),
            *(Int64(0),) * 3,
            Int32(0),
            *(Int64(0),) * 4,
            Int32(4),
            *(Int64(0),) * 4,
            Int32(1),
            Int32(1),
            Float32(1e-6),
            make_fake_stream(),
        )


@cache
def _finalize_kernel(
    device_index: int,
    hidden: int,
    world: int,
    capacity: int,
    write_aux: bool,
    top_k: int,
):
    """`TwoShotAllReduceFinalizePostCollapse` compiled for a device and
    variant."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            TwoShotAllReduceFinalizePostCollapse(
                hidden, world, capacity, write_aux, top_k
            ),
            *(Int64(0),) * 5,
            Int32(0),
            *(Int64(0),) * 4,
            Int32(4),
            *(Int64(0),) * 4,
            Int32(1),
            Int32(1),
            Float32(1e-6),
            make_fake_stream(),
        )


class AmpereAllReduce:
    """The mailbox and per-row generations of `TwoShotAllReduce` for one TP
    group; construction is collective over the group.

    Args:
        group: The TP group's device process group.
        hidden: Values per row.
        device: This rank's device.
        capacity: The most rows a call takes (`MAX_ROWS` by default).
        top_k: Routes a token of the MoE outputs `finalize_post_collapse`
            takes; None compiles no finalize kernels.

    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        hidden: int,
        device: torch.device,
        capacity: int = MAX_ROWS,
        top_k: int | None = None,
    ):
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.hidden = hidden
        self.capacity = capacity
        self.top_k = top_k
        # The most CTAs a call launches, one per SM (see the module docstring).
        self.ctas = current_platform.num_compute_units(device.index)
        self.kernel = _kernel(device.index, hidden, self.world, capacity)
        self.collapse_kernels = [
            _collapse_kernel(device.index, hidden, self.world, capacity, write_aux)
            for write_aux in (False, True)
        ]
        # Compiled here: the first finalize may come while a CUDA graph is
        # being captured.
        self.finalize_kernels = (
            []
            if top_k is None
            else [
                _finalize_kernel(
                    device.index, hidden, self.world, capacity, write_aux, top_k
                )
                for write_aux in (False, True)
            ]
        )
        nbytes = TwoShotAllReduce(hidden, self.world, capacity).mailbox_bytes()
        self.mailbox = symm_mem.empty(nbytes // 2, dtype=torch.bfloat16, device=device)
        handle = symm_mem.rendezvous(self.mailbox, group)
        self.peers = torch.tensor(handle.buffer_ptrs, dtype=torch.int64, device=device)
        # Every slot starts as the sentinel, BF16 negative zero.
        self.mailbox.view(torch.int16).fill_(-32768)
        self.generations = torch.zeros(capacity, dtype=torch.int32, device=device)
        torch.accelerator.synchronize()
        dist.barrier(group=group)

    def supports(self, x: torch.Tensor) -> bool:
        """Whether a call takes `x`: up to `capacity` contiguous BF16
        rows of `hidden` values, 16-byte aligned."""
        return (
            x.dim() == 2
            and 0 < x.shape[0] <= self.capacity
            and x.shape[1] == self.hidden
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.data_ptr() % FRAGMENT_BYTES == 0
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """The sum of `x` over the TP group, as a new tensor."""
        assert self.supports(x)
        out = torch.empty_like(x)
        self.kernel(
            Int64(x.data_ptr()),
            Int64(out.data_ptr()),
            Int64(self.peers.data_ptr()),
            Int64(self.generations.data_ptr()),
            Int32(self.rank),
            Int32(x.shape[0]),
            Int32(min(x.shape[0], self.ctas)),
            current_cu_stream(),
        )
        return out

    def post_collapse(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        pre_mix: torch.Tensor | None,
        norm_weight: torch.Tensor,
        norm_eps: float,
        write_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The sum of `x` over the TP group, post-mixed into the streams,
        collapsed and normalized as `mhc.mhc_post_collapse` does with the
        sum: the new streams, the layer input and the stream mean (empty
        without `write_aux`). `post_mix` is `[rows, 4]` or
        `[rows, 4, 1]`, `comb_mix` `[rows, 4, 4]`; `pre_mix` None
        selects stream zero."""
        assert self.supports(x)
        streams, layer_input, aux, args, _inputs = self._collapse_args(
            x.shape[0], residual, post_mix, comb_mix, pre_mix, norm_weight, write_aux
        )
        self.collapse_kernels[int(write_aux)](
            Int64(x.data_ptr()),
            *args,
            Float32(norm_eps),
            current_cu_stream(),
        )
        return streams, layer_input, aux

    def finalize_post_collapse(
        self,
        routes: torch.Tensor,
        route_weights: torch.Tensor,
        shared: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        pre_mix: torch.Tensor | None,
        norm_weight: torch.Tensor,
        norm_eps: float,
        write_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`post_collapse` of a MoE output with its top-k reduction open:
        the sublayer output of token `t` is `shared[t]` plus the sum over
        `k` of `route_weights[t, k] * routes[t * top_k + k]`. `routes`
        holds the unweighted GEMM2 rows, `route_weights` the FP32 routing
        weights `[rows, top_k]`, `shared` the shared expert's rows."""
        top_k = self.top_k
        assert top_k is not None, "built without finalize kernels"
        rows = shared.shape[0]
        assert self.supports(shared)
        assert route_weights.shape == (rows, top_k)
        assert route_weights.dtype == torch.float32 and route_weights.is_contiguous()
        assert routes.dim() == 2 and routes.shape[0] >= rows * top_k
        assert routes.shape[1] == self.hidden and routes.dtype == torch.bfloat16
        assert routes.is_contiguous() and routes.data_ptr() % FRAGMENT_BYTES == 0
        streams, layer_input, aux, args, _inputs = self._collapse_args(
            rows, residual, post_mix, comb_mix, pre_mix, norm_weight, write_aux
        )
        self.finalize_kernels[int(write_aux)](
            Int64(routes.data_ptr()),
            Int64(route_weights.data_ptr()),
            Int64(shared.data_ptr()),
            *args,
            Float32(norm_eps),
            current_cu_stream(),
        )
        return streams, layer_input, aux

    def _collapse_args(
        self,
        rows: int,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        pre_mix: torch.Tensor | None,
        norm_weight: torch.Tensor,
        write_aux: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple, tuple]:
        """The outputs of a collapsing call of `rows` rows, its kernel
        arguments after the sublayer output up to the CTA count, and the
        mixes those point to, which the caller keeps until the launch (they
        may be contiguous copies)."""
        device = residual.device
        assert residual.shape == (rows, HC, self.hidden)
        assert residual.dtype == torch.bfloat16 and residual.is_contiguous()
        assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
        post_mix = post_mix.reshape(rows, HC).contiguous()
        comb_mix = comb_mix.contiguous()
        assert post_mix.dtype == comb_mix.dtype == torch.float32
        if pre_mix is None:
            pre, pre_stride = _select_first_stream(device.index), 0
        else:
            assert pre_mix.shape == (rows, HC) and pre_mix.dtype == torch.float32
            assert pre_mix.is_contiguous()
            pre, pre_stride = pre_mix, HC
        streams = torch.empty_like(residual)
        layer_input = torch.empty(
            (rows, self.hidden), dtype=torch.bfloat16, device=device
        )
        aux = torch.empty(
            (rows if write_aux else 0, self.hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        args = (
            Int64(self.peers.data_ptr()),
            Int64(self.generations.data_ptr()),
            Int32(self.rank),
            Int64(residual.data_ptr()),
            Int64(post_mix.data_ptr()),
            Int64(comb_mix.data_ptr()),
            Int64(pre.data_ptr()),
            Int32(pre_stride),
            Int64(norm_weight.data_ptr()),
            Int64(streams.data_ptr()),
            Int64(layer_input.data_ptr()),
            Int64(aux.data_ptr()),
            Int32(rows),
            Int32(min(rows, self.ctas)),
        )
        return streams, layer_input, aux, args, (post_mix, comb_mix, pre)
