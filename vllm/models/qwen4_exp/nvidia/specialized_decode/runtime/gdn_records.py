# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN layers' record/commit state protocol in the decode steps.

The problem
-----------
vLLM's speculative-decode state protocol has each GDN layer's verify write
the recurrent state after every verified row to that row's slot: per layer,
request and step, one read and `q` writes (`q = num_speculative_tokens +
1`) of the FP32 state (3 MiB a request and layer at TP1). Only one of those
slots, the last accepted row's, is read again in the common case. The fused
core's `records` protocol (`..gdn.commit`) writes no state in the verify
but small per-row records, and a commit after sampling folds the accepted
rows into that one slot: three state transfers instead of five, and every
slot that a later reader looks at holds bit for bit what vLLM's protocol
writes there. The records take no memory of their own: they go into two of
the request's other speculative slots, which hold nothing anyone reads until
the next verify (invariants 4 and 6).

Who does what
-------------
- `GdnLayers` (`gdn.py`) runs the fused core with `protocol="records"`
  in every GDN layer of the executor's FULL CUDA graph captures, each layer
  writing its records into its recurrent cache (two of each request's
  non-source speculative slots) and their headers into its own header buffer
  (`GdnRecords.headers`), and reports each such launch
  (`GdnRecords.launched`). In prefix-caching `align` mode the core reads
  the runner's `seq_lens` buffer (`GdnRecords.seq_lens`), which the model
  state binds in `prepare_attn` (`bind_seq_lens`) before each capture.
- `SpecializedDecode` (`decode.py`) brackets every FULL capture of the
  target model: `start_capture` before its layers, `finish_capture` after
  them, or `finish_capture` alone when it runs the stock layers.
  `GdnRecords` remembers, per captured row count, whether the FULL graphs of
  that size hold the records core (`recorded_rows`).
- The model state (`model_state.py`) marks a step whose forward replays a
  FULL graph of a recorded size in `prepare_attn` (`records_step`) and
  commits it in `postprocess_state`, after the sampler and the runner's
  `post_update`, before the stock postprocess (the accepted-count scatter
  and the prefix-caching `align` copies), the MTP drafter and the next
  step: `GdnRecords.commit` launches the commit kernel once for every GDN
  layer, eagerly, on the current stream.

Invariants
----------
1. A step's forward runs the records core if and only if the step commits.
   The records core exists only inside FULL graphs captured by the executor,
   and the FULL graphs of one row count either all hold it in every GDN layer
   or none does: `finish_capture` raises otherwise (a FULL graph of another
   kind at a recorded size, or a layer that took the stock path). FULL graphs
   replay only on steps whose `prepare_attn` sees `CUDAGraphMode.FULL` and
   the graph's row count (`num_tokens_after_padding`), so `records_step`
   names exactly the steps that replayed a records graph. A step that is
   marked but not committed raises at the next `prepare_attn`.
2. The commit folds `a = max(sampled, 1)` rows of each request row (the
   accepted-token count the stock postprocess stores and the next step
   reads) and writes slot `a - 1`; in `align` mode it also writes the
   slot of the boundary row the `align` postprocess copies
   (`..gdn.commit`, `boundary_row`). It writes only slots the stock
   verify writes in the same step, with the values it writes there.
3. Every reader of those slots runs after the commit on the same stream: the
   stock postprocess (after it in `postprocess_state`), the next step's
   `preprocess_state` copies, metadata and forward, and any stock step. The
   commit's inputs (the records of this step's verify, the sampled counts,
   the computed tokens `post_update` advanced) are written before it on
   that stream. Nothing between the forward and the commit reads or writes a
   GDN state: the sampler does not, and the MTP drafter (a QSA attention
   layer, no GDN state) runs after the commit.
4. Every slot the verify reads holds what vLLM's protocol leaves there: the
   source slot `accepted - 1` was written by the previous step's commit, a
   stock step, a pre-copy or a prefill, all before this step on the stream.
   Slots the commit does not write are never read before a verify, stock step
   or pre-copy overwrites them. That includes every slot the verify writes
   its records to: two of the `q - 1` non-source speculative slots of a
   request with caches. Every request of a records step (a uniform decode step)
   verifies `q` rows, so vLLM's verify writes each of those slots in the
   same step, with a state that no one reads: the records go to a subset of
   the slots vLLM's protocol writes, all of them the request's own (the
   runner allocates all `q` speculative blocks of a decoding request; in
   `align` mode the speculative columns past the running state are never
   full blocks, so never cached or shared). Nothing writes them between the
   verify and the commit (invariant 3).
5. A padded request row (FULL-graph batch padding, null caches) records a
   header of no rows and its commit writes nothing, as the stock verify
   writes nothing for it; rows past the step's requests are not committed.
6. The commit reads the records from a copy that no CTA of the commit
   writes. The verify writes two copies, into the request's *copy slots*
   (`copy_slots`): the first two non-source slots that are not the slot of
   a row the commit may write as the boundary row (`boundary_rows`: in
   `align` mode a row `t <= rows - 2` whose state, `n + t + 1` tokens
   with `n = seq_lens - rows` the computed tokens before the step, ends at
   a block boundary). It marks them in the request's header. The commit
   writes slot `a - 1`, which is at most one of them, and the boundary
   row's, which is neither; it reads the first copy slot that is neither
   (`record_copy`). With `block_size >= q - 1` (`align` mode) at most
   one row can be the boundary row, so two copy slots exist when `q - 1 >=
   2 + align` (`align`: 1 in `align` mode, else 0). The model state
   keeps vLLM's protocol otherwise (`records_fit`), `bind_caches` checks
   the bound block size and that every layer's slot holds the records, the
   verify reads the computed tokens from the `seq_lens` buffer the graphs
   were captured with (`bind_seq_lens` raises when the runner's moves),
   and the kernel traps on a request without a copy it does not write (a
   null speculative slot, or a boundary row the verify did not foresee)
   instead of folding stale values.

`tests/models/qwen4_exp/specialized_decode/test_gdn_records_protocol.py`
checks invariants 1 to 6 on a model of the runner's steps (a CPU simulator of
both protocols over random step sequences, which catches each listed defect),
`test_gdn_records.py` the kernels and real decode steps on the GPU, and
`finish_capture` the captured graphs.
"""

from collections.abc import Callable, Sequence

import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)

# The compiled commit: (sampled, table address, layers, requests, idx_mapping
# address, computed address, block size) -> None, on the current stream.
CommitFn = Callable[[torch.Tensor, int, int, int, int, int, int], None]


def commit_rows(sampled: int, rows: int) -> int:
    """The rows the commit folds for a request row: its sampled tokens (the
    accepted drafts plus the bonus token), at least one as the stock
    postprocess counts them (`max(sampled, 1)`), at most the rows the verify
    recorded; 0 for a row the verify skipped (`rows == 0`). The commit
    writes the state after the last of them to slot `count - 1`."""
    return min(max(sampled, 1), rows) if rows > 0 else 0


def boundary_row(count: int, computed: int, block_size: int) -> int:
    """The verify row whose state the prefix-caching `align` postprocess
    copies to a block boundary and that slot `count - 1` does not hold, or
    -1: `count` folded rows, `computed` tokens after the step (already
    advanced by them), mamba blocks of `block_size` tokens. The state after
    row `t` covers `computed - count + 1 + t` tokens; the postprocess copies
    the one that ends at the last block boundary, if any row does."""
    running = computed - count + 1
    aligned = computed // block_size * block_size
    bias = aligned - running
    return bias if aligned >= running and bias < count - 1 else -1


def boundary_rows(before: int, rows: int, block_size: int) -> list[int]:
    """The verify rows of a request that its commit may write as the
    boundary row (`boundary_row`) in `align` mode, whatever the sampler
    accepts: `before` computed tokens before the step (`seq_lens -
    rows`), `rows` verified rows, mamba blocks of `block_size` tokens.
    The state after row `t` covers `before + t + 1` tokens; the boundary
    row ends at a block boundary and precedes the last folded row."""
    return [t for t in range(rows - 1) if (before + t + 1) % block_size == 0]


def copy_slots(slots: Sequence[int], source: int, avoid: Sequence[int]) -> list[int]:
    """The slots the verify writes its two copies of a request's records to:
    the first two of its speculative `slots` that are not null, not the
    `source` and not the slot of a row in `avoid` (`boundary_rows`)."""
    usable = [
        slot
        for t, slot in enumerate(slots)
        if slot > 0 and slot != source and t not in avoid
    ]
    return usable[:2]


def record_copy(
    slots: Sequence[int], copies: Sequence[int], count: int, boundary: int
) -> int:
    """The slot whose records the commit of a request reads: the first of its
    speculative `slots` that is one of its `copies` (`copy_slots`) and
    that the commit does not write (neither slot `count - 1` nor, when
    `boundary >= 0`, the boundary row's slot); 0 when there is none (the
    kernel traps)."""
    written = {slots[count - 1]}
    if boundary >= 0:
        written.add(slots[boundary])
    return next((slot for slot in slots if slot in copies and slot not in written), 0)


def records_fit(tokens_per_request: int, align: bool, block_size: int) -> bool:
    """Whether a copy of the records always survives the commit: the verify
    finds two copy slots among a request's `tokens_per_request - 1`
    non-source speculative slots, of which `align` mode may exclude one
    (one boundary row at most when `block_size >= tokens_per_request - 1`),
    and the commit writes at most one of them (invariant 6)."""
    if align and block_size < tokens_per_request - 1:
        return False
    return tokens_per_request - 1 >= 2 + align


class GdnRecords:
    """The record header buffers, the commit and the capture bookkeeping of
    the executor's GDN layers.

    Args:
        layers: The backbone's local GDN layers (the fused core's).
        tokens_per_request: Rows of one request (the MTP draft tokens plus one).
        capacity: Requests of the largest step the executor runs.
        key_heads: Key heads of the rank.
        align: Prefix caching in `align` mode (the commit also writes the
            block-boundary row).

    """

    def __init__(
        self,
        layers: Sequence[QwenGatedDeltaNetAttention],
        tokens_per_request: int,
        capacity: int,
        key_heads: int,
        align: bool,
    ) -> None:
        # Imported here: CuTe DSL kernels (CUDA builds only).
        from ..gdn import RecordLayout

        assert layers and capacity > 0
        # The mamba block size is known only once the KV cache is: here
        # tokens_per_request stands in for it, and bind_caches checks the real
        # one.
        if not records_fit(tokens_per_request, align, tokens_per_request):
            raise ValueError(
                f"GDN records: {tokens_per_request} rows per request leave no "
                "speculative slot for a copy of the records that the commit "
                "does not write"
            )
        self.layers = list(layers)
        self._index: dict[int, int] = {id(attn): i for i, attn in enumerate(layers)}
        self.layout: RecordLayout = RecordLayout(
            capacity, tokens_per_request, key_heads
        )
        self.device = layers[0].A_log.device
        # One header buffer per layer, int32 [layers, words]; memory profiling
        # counts it. The records themselves live in the recurrent caches.
        self.headers = self.layout.allocate(self.device, len(self.layers))
        self.align = align
        # Bound to the KV cache (bind_caches).
        self.tables: torch.Tensor | None = None
        self._table_address = 0
        self._block_size = 0
        # The runner's seq_lens buffer, which the align-mode verify reads
        # (bind_seq_lens).
        self._seq_lens: torch.Tensor | None = None
        # The compiled commit by request count (warm_up).
        self._commits: list[CommitFn] = []
        # Per row count of the FULL graphs captured so far: whether they hold
        # the records core.
        self.recorded_rows: dict[int, bool] = {}
        # The layers that launched the records core in the capture running now.
        self._capture: set[int] | None = None

    @property
    def nbytes(self) -> int:
        """Bytes of the header buffers."""
        return self.headers.numel() * self.headers.element_size()

    def header(self, attn: QwenGatedDeltaNetAttention) -> torch.Tensor:
        """The header buffer of the layer `attn`."""
        return self.headers[self._index[id(attn)]]

    def warm_up(self) -> None:
        """Compile the commit for every request count and run each variant
        once (on headers of no rows: nothing is written), so that no step
        compiles or loads it."""
        # Imported here, as in __init__.
        from ..gdn import commit_tables, compiled_commit, default_commit_schedule

        scratch = torch.zeros(1, self.layout.value_heads, 128, 128, device=self.device)
        tables = commit_tables([scratch] * len(self.layers), self.headers)
        capacity = self.layout.capacity
        ones = torch.ones(capacity, dtype=torch.int32, device=self.device)
        # Request state index 0 (int64) and its computed tokens (int32), in
        # align mode.
        index = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        zeros = torch.zeros(capacity, dtype=torch.int32, device=self.device)
        index_address = index.data_ptr() if self.align else 0
        computed_address = zeros.data_ptr() if self.align else 0
        self._commits = []
        warmed: set[int] = set()
        for requests in range(1, capacity + 1):
            schedule = default_commit_schedule(requests, self.layout.value_heads)
            fn = compiled_commit(self.device, self.layout, schedule, self.align)
            self._commits.append(fn)
            if id(fn) not in warmed:
                warmed.add(id(fn))
                fn(
                    ones,
                    tables.data_ptr(),
                    len(self.layers),
                    requests,
                    index_address,
                    computed_address,
                    1,
                )
        torch.accelerator.synchronize(self.device)

    @property
    def block_size(self) -> int:
        """The mamba block size of `align` mode (0 otherwise)."""
        return self._block_size

    @property
    def seq_lens(self) -> torch.Tensor | None:
        """The runner's `seq_lens` buffer that the `align`-mode verify
        reads (None before `bind_seq_lens` and outside `align` mode)."""
        return self._seq_lens

    def bind_seq_lens(self, seq_lens: torch.Tensor) -> None:
        """Point the `align`-mode verify at the runner's `seq_lens` (int32,
        by request row; the computed tokens before the step plus its rows),
        before each capture (the capture's own `prepare_attn`): every
        step's `seq_lens` is a view of the runner's buffer at its start, the
        longest one seen is kept. The FULL graphs read the buffer they were
        captured with, so raise if a later step's `seq_lens` lies elsewhere
        (invariant 6)."""
        if not self.align:
            return
        assert seq_lens.dtype == torch.int32 and seq_lens.is_contiguous()
        if self._seq_lens is None:
            self._seq_lens = seq_lens
        elif seq_lens.data_ptr() == self._seq_lens.data_ptr():
            if seq_lens.numel() > self._seq_lens.numel():
                self._seq_lens = seq_lens
        else:
            raise RuntimeError(
                "Qwen4Exp specialized decode: the runner's seq_lens buffer "
                "moved; the GDN records verify of the captured graphs reads "
                "the old one."
            )

    def bind_caches(self, block_size: int) -> None:
        """Point the commit at the layers' recurrent caches, after the KV cache
        is allocated and before any capture; `block_size` is the mamba block
        size of `align` mode. Raise if a layer's state slot cannot hold the
        records or the block size leaves no two copy slots (invariant 6)."""
        # Imported here, as in __init__.
        from ..gdn import commit_tables

        states = [attn.kv_cache[1] for attn in self.layers]
        shape = (self.layout.value_heads, 128, 128)
        for state in states:
            if (
                state.dtype != torch.float32
                or tuple(state.shape[1:]) != shape
                or state.stride()[1:] != (128 * 128, 128, 1)
                or self.layout.band_words > self.layout.band_stride
            ):
                raise RuntimeError(
                    "Qwen4Exp specialized decode: a GDN recurrent cache of "
                    f"{state.dtype} {tuple(state.shape)} cannot hold the "
                    f"records of {self.layout.key_heads} key heads"
                )
        if not records_fit(self.layout.tokens, self.align, block_size):
            raise RuntimeError(
                "Qwen4Exp specialized decode: GDN records of "
                f"{self.layout.tokens} rows per request need a mamba block "
                f"size of at least {self.layout.tokens - 1}, not {block_size}"
            )
        self.tables = commit_tables(states, self.headers)
        self._table_address = self.tables.data_ptr()
        self._block_size = block_size if self.align else 0

    def start_capture(self) -> None:
        """A FULL capture of the executor's layers begins."""
        self._capture = set()

    def launched(self, attn: QwenGatedDeltaNetAttention) -> None:
        """The layer `attn` launched the records core in the capture."""
        if self._capture is not None:
            self._capture.add(self._index[id(attn)])

    def finish_capture(self, rows: int) -> None:
        """A FULL capture of `rows` rows ended (or ran the stock layers
        without `start_capture`): remember whether it holds the records
        core; raise if only some GDN layers launched it or if another FULL
        graph of `rows` rows did not match (invariant 1)."""
        layers, self._capture = self._capture or set(), None
        if layers and len(layers) != len(self.layers):
            raise RuntimeError(
                "Qwen4Exp specialized decode: a FULL CUDA graph of "
                f"{rows} rows runs the GDN records core in {len(layers)} of "
                f"{len(self.layers)} GDN layers; the commit needs all or none."
            )
        recorded = bool(layers)
        if self.recorded_rows.setdefault(rows, recorded) != recorded:
            raise RuntimeError(
                f"Qwen4Exp specialized decode: FULL CUDA graphs of {rows} rows "
                "with and without the GDN records core; a replay could not "
                "tell whether to commit."
            )

    def records_step(self, rows: int) -> bool:
        """Whether a FULL graph replay of `rows` rows runs the records core."""
        return self.recorded_rows.get(rows, False)

    def commit(
        self,
        requests: int,
        sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        computed: torch.Tensor | None,
    ) -> None:
        """Commit a records step of `requests` request rows: launch the
        commit for every GDN layer on the current stream.

        Args:
            requests: The step's request rows (`num_reqs`).
            sampled: The sampled tokens of each request row, int32.
            idx_mapping: Each request row's request state index, int64 (the
                runner's `input_batch.idx_mapping`).
            computed: The computed tokens after the step by request state
                index, int32 (`align` mode).

        """
        assert self.tables is not None, "commit before bind_caches"
        assert sampled.dtype == torch.int32 and sampled.is_contiguous()
        fn = self._commits[requests - 1]
        if self.align:
            assert computed is not None and computed.dtype == torch.int32
            assert idx_mapping.dtype == torch.int64 and idx_mapping.is_contiguous()
            fn(
                sampled,
                self._table_address,
                len(self.layers),
                requests,
                idx_mapping.data_ptr(),
                computed.data_ptr(),
                self._block_size,
            )
        else:
            fn(sampled, self._table_address, len(self.layers), requests, 0, 0, 0)
