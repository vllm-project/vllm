# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record and commit: the GDN layers' speculative-state protocol of the
specialized decode.

The problem. In vLLM's speculative-decode state protocol (`slots`) a GDN
layer's verify reads each request's FP32 recurrent state from slot
`accepted - 1` and writes the state after every verified row `t` to slot
`t`: one read and `q` writes (`q = num_speculative_tokens + 1`, 4 with
three draft tokens) of 64 KiB per value head, per layer and step. Once the
sampler has accepted `a` rows, the readers outside the verify look at one
of those slots, `a - 1`, and on the step that crosses a prefix-caching block
boundary at one more (below).

The record/commit protocol (`records`) moves the state three times instead
of five:

- the verify (`..core.GdnDecodeCore` with `protocol="records"`)
  reads the state, runs its rows from registers and writes no recurrent
  state. It writes per-row *records* instead: the normalized key of each key
  head, the decay of each value head and the delta-rule correction of each
  value row, the exact FP32 values its state update used, and a header with
  the request's rows and slots. Its outputs and the convolution state are
  the `slots` protocol's, bit for bit;
- after sampling, the commit (`GdnStateCommit`, one launch for all
  layers) reads the state again, folds the `a` accepted rows into it from
  the records (`h = h * decay; h = h + k * delta`: the verify's own
  instructions on the verify's own values) and writes it to slot `a - 1`,
  bit for bit what the `slots` protocol writes there.

Which slots a commit writes. Slot `a - 1`, with `a = clamp(sampled, 1,
rows)`: the next verify, a stock step of the request and the next step's
prefix-caching pre-copy read it (`num_accepted_tokens = max(sampled, 1)`).
In prefix-caching `align` mode, the stock postprocess after sampling
(`postprocess_mamba_fused_kernel`) also copies the state at the block
boundary that the accepted rows crossed, the state after row `bias`, to the
boundary's block. With `new` the request's computed tokens after the step
(`num_computed_tokens`, already advanced) and `B` the mamba block size::

    running = new - a + 1           # tokens in the state after row 0
    aligned = (new // B) * B        # the last block boundary at or before new
    copy when aligned >= running, from row bias = aligned - running in [0, a - 1]

so the commit also writes the state after row `bias` to slot `bias` when
the copy happens and `bias < a - 1` (`bias == a - 1` is slot `a - 1`
already). No other slot is read before the next verify overwrites it.
`runtime/gdn_records.py` states these choices as host functions
(`commit_rows`, `boundary_row`) that the protocol model and the tests
check the kernel against.

Where the records live (`RecordLayout`). In a records step the
request's `q` speculative slots hold one live state, the source `s =
slots[accepted - 1]`, which the verify and the commit read; the other
`q - 1` hold nothing any reader looks at until a verify, a stock step or a
pre-copy writes them (invariant 4 of `runtime/gdn_records.py`): the
`slots` protocol writes exactly those slots with states nobody reads. The
verify writes its records into two of them, a full copy into each, so they
take no memory of their own. The commit writes at most two slots, `a - 1`
(unknown until the sampler runs) and the boundary row's, and every copy it
might read must be in a slot it does not write. So the verify chooses the
two *copy slots* that the boundary row's slot cannot be: the first two
non-source slots (`slots[t]`, `t != accepted - 1`) except, in `align`
mode, the slot of a row whose state ends at a block boundary. With `n` the
request's computed tokens before the step (`seq_lens - rows`), the state
after row `t` covers `n + t + 1` tokens, and only a row `t <= rows - 2`
with `(n + t + 1) % block_size == 0` can be the boundary row
(`boundary_row` with `computed = n + a`). With `block_size >= q -
1` at most one row qualifies, so two copy slots remain when `q - 1 >= 2 +
align`. Slot `a - 1` is then at most one of them; the commit reads the
other (`copy`: the first copy slot it does not write). Within a slot the
records of key head `k` lie at the start of the state rows of its value
heads `3k .. 3k + 2` (its *band*), which only the verify CTAs of that key
head read or write (word offsets from the band's start):

- keys `FP32 [q, 128]` (the key head's normalized keys);
- decays `FP32 [q, 3]`, padded to 16 bytes;
- deltas `FP32 [q, 3, 128]` (each value row's correction).

That is 8 KiB of the band's 192 KiB. The header `int32 [capacity, 3 + q]`,
`(rows, source slot, slot 0, ..., slot q - 1, copies)` by request row
`b` (the batch row; `rows` is 0 for a request the verify skipped: a
padded request, or null caches; bit `t` of `copies` set when slot `t`
is a copy slot), stays in a small buffer of its own per layer, which the
commit reads first to find the slots.

The commit kernel. One CTA owns `128 / splits` value rows of one (layer,
request, value head); each warp `W` of them, each lane four keys of every
row. A CTA loads its state rows, then the records of the accepted rows only
(predicated loads: rows past `a` are neither loaded nor folded) from the
copy, folds them token by token in the verify's order, storing the state
after row `bias` on a boundary step, and stores the state after row `a -
1`. Every thread reads its source values before it writes any slot, so a
slot that is both source and destination (`a` equal to the previous
step's) is safe; no CTA writes the copy, so no CTA's records change under
another's reads. The first CTA of each key head prefetches the copy's band
into L2 before it loads its state, so that the records of later rows, which
the compiler loads during the fold, come from L2. A request without a copy
slot outside the written slots (a null speculative slot, or a boundary row
the verify did not foresee) traps rather than fold stale values. The kernel
reads the step's sampled counts, the header, the records and, in `align`
mode, the request indices and computed tokens; it runs eagerly after the
sampler and the runner's `post_update` on the same stream (no programmatic
dependent launch), so every input is complete when it starts.
"""

from dataclasses import dataclass
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass.cute.runtime import make_fake_stream

from ._ptx import (
    fma_ftz_x2,
    load_f32_if,
    load_s32,
    load_s64,
    load_state_x4,
    load_state_x4_if,
    mul_ftz_x2,
    prefetch_l2,
    select_f32,
    select_s32,
    store_state_x4,
    trap,
)

HEAD_DIM = 128
GQA = 3
WARP = 32


def _round4(words: int) -> int:
    """`words` rounded up to a multiple of 4 (16-byte alignment)."""
    return -(-words // 4) * 4


@dataclass(frozen=True)
class RecordLayout:
    """Word offsets of one layer's verify records: the header buffer and,
    within a speculative slot, a key head's band (module docstring).

    Attributes:
        capacity: Requests (step rows) the header buffer holds.
        tokens: Rows per request (`q`).
        key_heads: Key heads of the rank.

    """

    capacity: int
    tokens: int
    key_heads: int

    @property
    def value_heads(self) -> int:
        """Value heads of the rank."""
        return GQA * self.key_heads

    @property
    def header_stride(self) -> int:
        """Words of one request's header."""
        return 3 + self.tokens

    @property
    def copies_word(self) -> int:
        """Word of a request's header that marks its copy slots."""
        return 2 + self.tokens

    @property
    def words(self) -> int:
        """Words of a layer's header buffer."""
        return self.capacity * self.header_stride

    @property
    def band_stride(self) -> int:
        """Words between two key heads' bands in a slot: the state rows of
        three value heads."""
        return GQA * HEAD_DIM * HEAD_DIM

    @property
    def keys(self) -> int:
        """Word offset of the keys in a band."""
        return 0

    @property
    def decays(self) -> int:
        """Word offset of the decays in a band."""
        return self.keys + self.tokens * HEAD_DIM

    @property
    def deltas(self) -> int:
        """Word offset of the deltas in a band."""
        return self.decays + _round4(self.tokens * GQA)

    @property
    def band_words(self) -> int:
        """Words of a band's records (a multiple of 4: 16 bytes)."""
        return self.deltas + self.tokens * GQA * HEAD_DIM

    def allocate(self, device: torch.device, layers: int = 1) -> torch.Tensor:
        """Zeroed header buffers of `layers` layers, int32 `[layers,
        words]` (every header says: nothing recorded)."""
        assert self.band_words <= self.band_stride
        return torch.zeros(layers, self.words, dtype=torch.int32, device=device)


@dataclass(frozen=True)
class CommitSchedule:
    """How the commit splits one (layer, request, value head).

    Attributes:
        splits: CTAs per value head, each with `128 / splits` value rows.
        warps: Warps per CTA; each folds `128 / (splits * warps)` rows.
        ctas_per_sm: CTAs the register budget must fit on one SM (0 leaves
            the register count to the compiler).

    """

    splits: int
    warps: int
    ctas_per_sm: int = 0

    @property
    def rows(self) -> int:
        """Value rows of one CTA."""
        return HEAD_DIM // self.splits

    @property
    def warp_rows(self) -> int:
        """Value rows of one warp."""
        return self.rows // self.warps

    def validate(self) -> None:
        """Raise if the split is not one the kernel supports."""
        warp_rows = HEAD_DIM // (self.splits * self.warps)
        if (
            HEAD_DIM % (self.splits * self.warps)
            or warp_rows < 1
            or warp_rows & (warp_rows - 1)
        ):
            raise ValueError(f"rows per warp must be a power of two: {self}")


# Commit schedules by step size, from a GB300 sweep of nine splits over 36
# layers at 4 and 16 key heads (1 to 64 requests): small CTAs of four warps,
# four per SM, so that some CTAs' loads stream while others fold and store.
# Steps of fewer than 48 (request, value head) pairs split each head over
# eight CTAs to fill the SMs (10.3 us at TP4 S1); larger ones over two, which
# reach 6.8 to 7.1 TB/s (S4 to T64).
COMMIT_FEW = CommitSchedule(splits=8, warps=4, ctas_per_sm=4)
COMMIT_MANY = CommitSchedule(splits=2, warps=4, ctas_per_sm=4)


def default_commit_schedule(requests: int, value_heads: int) -> CommitSchedule:
    """The commit schedule for `requests` requests of `value_heads` heads."""
    return COMMIT_FEW if requests * value_heads < 48 else COMMIT_MANY


class GdnStateCommit:
    """The commit of every layer's accepted rows after sampling.

    Args:
        layout: The records layout (shared by the layers).
        schedule: The CTA split.
        align: Also write the boundary row of prefix-caching `align` mode.

    """

    def __init__(
        self, layout: RecordLayout, schedule: CommitSchedule, align: bool
    ) -> None:
        schedule.validate()
        self.layout = layout
        self.tokens = layout.tokens
        self.key_heads = layout.key_heads
        self.value_heads = layout.value_heads
        self.schedule = schedule
        self.splits = schedule.splits
        self.warps = schedule.warps
        self.rows = schedule.rows
        self.warp_rows = schedule.warp_rows
        self.align = align

    @cute.jit
    def __call__(
        self,
        sampled: cute.Tensor,
        tables: Int64,
        layers: Int32,
        requests: Int32,
        idx_mapping: Int64,
        computed: Int64,
        block_size: Int32,
        stream: cuda.CUstream,
    ):
        """Launch on `stream` (see `gdn_state_commit`)."""
        self.kernel(sampled, tables, layers, idx_mapping, computed, block_size).launch(
            grid=(self.value_heads * self.splits, requests, layers),
            block=(WARP * self.warps, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.schedule.ctas_per_sm,
        )

    @cute.kernel
    def kernel(
        self,
        sampled: cute.Tensor,
        tables: Int64,
        layers: Int32,
        idx_mapping: Int64,
        computed: Int64,
        block_size: Int32,
    ):
        """One CTA: `rows` value rows of one (layer, request, value head)."""
        Q = self.tokens
        W = self.warp_rows
        lay = self.layout
        tid, _, _ = cute.arch.thread_idx()
        x, request, layer = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // WARP)
        lane = tid % WARP
        head = x // self.splits
        row0 = (x % self.splits) * self.rows + warp * W

        # The layer's cache, its slot stride in bytes and its header buffer.
        state = load_s64(tables + Int64(layer) * 8)
        slot_bytes = load_s64(tables + (Int64(layers) + layer) * 8)
        headers = load_s64(tables + (Int64(layers) * 2 + layer) * 8)
        header = headers + Int64(request) * lay.header_stride * 4
        rows = load_s32(header)
        source = load_s32(header + 4)
        slot_ids = [load_s32(header + Int64(2 + t) * 4) for t in range(Q)]
        count = cute.min(
            cute.max(load_s32(sampled.iterator.toint() + Int64(request) * 4), 1),
            rows,
        )
        target = load_s32(header + Int64(1 + cute.max(count, 1)) * 4)
        boundary = Int32(-1)
        boundary_slot = Int32(0)
        if cutlass.const_expr(self.align):
            # gdn_records.boundary_row on the computed tokens after the step.
            # The runner's idx_mapping is int64 (-1 marks a filtered row).
            state_index = load_s64(idx_mapping + Int64(request) * 8)
            after = load_s32(computed + cute.max(state_index, Int64(0)) * 4)
            running = after - count + 1
            aligned = after // block_size * block_size
            bias = aligned - running
            hit = (state_index >= Int64(0)) & (aligned >= running) & (bias < count - 1)
            boundary = select_s32(Int32(hit), bias, Int32(-1))
            boundary_slot = load_s32(header + Int64(2 + cute.max(boundary, 0)) * 4)
        # The copy of the records to read: the first copy slot this commit
        # does not write (gdn_records.record_copy).
        copies = load_s32(header + Int64(lay.copies_word) * 4)
        copy = Int32(0)
        for i in cutlass.range_constexpr(Q):
            t = Q - 1 - i
            free = (
                (((copies >> t) & 1) != 0)
                & (slot_ids[t] > 0)
                & (slot_ids[t] != source)
                & (slot_ids[t] != target)
                & ((boundary < 0) | (slot_ids[t] != boundary_slot))
            )
            copy = select_s32(Int32(free), slot_ids[t], copy)

        live = (rows > 0) & (source > 0)
        if live & (copy <= 0):
            trap()
        # The copy's records into L2 while the CTAs of its key head load their
        # state rows: one prefetch per band, by the band's first CTA (the
        # key head's CTAs are adjacent in the grid).
        if live & (tid == 0) & (x % (GQA * self.splits) == 0):
            prefetch_l2(
                state
                + Int64(copy) * slot_bytes
                + Int64((head // GQA) * lay.band_stride) * 4,
                Int32(lay.band_words * 4),
            )
        if live:
            lane_offset = Int64((head * HEAD_DIM + row0) * HEAD_DIM + lane * 4) * 4
            source_rows = state + Int64(source) * slot_bytes + lane_offset
            h = [list(load_state_x4(source_rows + r * HEAD_DIM * 4)) for r in range(W)]
            # The accepted rows' records (row t < count only), from the copy's
            # band of this value head's key head.
            band = (
                state
                + Int64(copy) * slot_bytes
                + Int64((head // GQA) * lay.band_stride) * 4
            )
            group_head = head % GQA
            takes = [Int32(t < count) for t in range(Q)]
            keys = []
            decays = []
            deltas = []
            for t in cutlass.range_constexpr(Q):
                key_address = lay.keys + t * HEAD_DIM + lane * 4
                keys.append(load_state_x4_if(takes[t], band + Int64(key_address) * 4))
                decay_address = lay.decays + t * GQA + group_head
                decays.append(load_f32_if(takes[t], band + Int64(decay_address) * 4))
                delta_rows = (
                    band
                    + Int64(lay.deltas + (t * GQA + group_head) * HEAD_DIM + row0) * 4
                )
                if cutlass.const_expr(W >= 4):
                    deltas.append(
                        [
                            value
                            for i in range(W // 4)
                            for value in load_state_x4_if(takes[t], delta_rows + i * 16)
                        ]
                    )
                else:
                    deltas.append(
                        [load_f32_if(takes[t], delta_rows + r * 4) for r in range(W)]
                    )
            boundary_rows = state + Int64(boundary_slot) * slot_bytes + lane_offset
            for t in cutlass.range_constexpr(Q):
                k = keys[t]
                decay = decays[t]
                for r in cutlass.range_constexpr(W):
                    d = deltas[t][r]
                    h01 = mul_ftz_x2((h[r][0], h[r][1]), (decay, decay))
                    h23 = mul_ftz_x2((h[r][2], h[r][3]), (decay, decay))
                    h01 = fma_ftz_x2((k[0], k[1]), (d, d), h01)
                    h23 = fma_ftz_x2((k[2], k[3]), (d, d), h23)
                    h[r] = [
                        select_f32(takes[t], new, old)
                        for new, old in zip((*h01, *h23), h[r])
                    ]
                # A compile-time condition around a runtime one: not one `if`.
                if cutlass.const_expr(self.align and t < Q - 1):  # noqa: SIM102
                    if (boundary == t) & (boundary_slot > 0):
                        for r in cutlass.range_constexpr(W):
                            store_state_x4(
                                boundary_rows + r * HEAD_DIM * 4, tuple(h[r])
                            )
            if target > 0:
                target_rows = state + Int64(target) * slot_bytes + lane_offset
                for r in cutlass.range_constexpr(W):
                    store_state_x4(target_rows + r * HEAD_DIM * 4, tuple(h[r]))


@cache
def _compiled(
    device_index: int, layout: RecordLayout, schedule: CommitSchedule, align: bool
):
    """`GdnStateCommit` compiled for a device, layout, schedule and mode
    (TVM-FFI: the launch takes the current stream of the sampled counts'
    device)."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            GdnStateCommit(layout, schedule, align),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (cute.sym_int(),), assumed_align=4
            ),
            Int64(0),
            Int32(1),
            Int32(1),
            Int64(0),
            Int64(0),
            Int32(1),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def commit_tables(states: list[torch.Tensor], headers: torch.Tensor) -> torch.Tensor:
    """The commit's address table, int64 `[3, layers]`: each layer's
    recurrent cache address (FP32 `[slots, value_heads, 128, 128]`, whose
    speculative slots hold the records), its slot stride in bytes and its
    header buffer address (row of `headers`)."""
    assert len(states) == headers.shape[0] and headers.is_contiguous()
    for state in states:
        assert state.dtype == torch.float32 and state.stride()[1:] == (
            HEAD_DIM * HEAD_DIM,
            HEAD_DIM,
            1,
        )
    return torch.tensor(
        [
            [state.data_ptr() for state in states],
            [state.stride(0) * 4 for state in states],
            [row.data_ptr() for row in headers],
        ],
        dtype=torch.int64,
        device=headers.device,
    )


def compiled_commit(
    device: torch.device, layout: RecordLayout, schedule: CommitSchedule, align: bool
):
    """The compiled commit (`gdn_state_commit` validates and calls it):
    `fn(sampled, tables, layers, requests, idx_mapping, computed,
    block_size)` with the sampled counts a tensor and the rest integers
    (addresses as integers), on the current stream."""
    index = torch.device(device).index
    assert index is not None
    return _compiled(index, layout, schedule, align)


def gdn_state_commit(
    tables: torch.Tensor,
    requests: int,
    sampled: torch.Tensor,
    layout: RecordLayout,
    schedule: CommitSchedule | None = None,
    idx_mapping: torch.Tensor | None = None,
    computed: torch.Tensor | None = None,
    block_size: int = 0,
) -> None:
    """Fold each request's accepted rows into its state, for every layer.

    Args:
        tables: The address table (`commit_tables`).
        requests: Request rows of the step whose verify wrote the records.
        sampled: The step's sampled tokens per request row (the accepted
            drafts plus one; `max(sampled, 1)` rows are folded), int32.
        layout: The records layout.
        schedule: The CTA split; `default_commit_schedule` when None.
        idx_mapping: Prefix-caching `align` mode: each request row's request
            state index, int64 (the runner's); None without `align` mode.
        computed: `align` mode: the computed tokens after the step, by
            request state index, int32.
        block_size: `align` mode: the mamba block size.

    """
    assert tables.dtype == torch.int64 and tables.shape[0] == 3
    assert sampled.dtype == torch.int32 and sampled.is_contiguous()
    assert 0 < requests <= min(layout.capacity, sampled.numel())
    addresses = (0, 0)
    if idx_mapping is not None:
        assert computed is not None and block_size > 0
        assert idx_mapping.dtype == torch.int64 and computed.dtype == torch.int32
        assert idx_mapping.is_contiguous() and computed.is_contiguous()
        assert idx_mapping.numel() >= requests
        addresses = (idx_mapping.data_ptr(), computed.data_ptr())
    if schedule is None:
        schedule = default_commit_schedule(requests, layout.value_heads)
    compiled_commit(sampled.device, layout, schedule, idx_mapping is not None)(
        sampled, tables.data_ptr(), tables.shape[1], requests, *addresses, block_size
    )
