# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""page_topk stage: per (decode row, index head) the 16 pages the sparse
decode attends to, written as physical page ids into the sparse block table.

The device code is CuTe DSL (`page_topk_kernel.py`, compiled for the live
device at build). Exact selection: forced
pages (the first `init_blocks` and the last `local_blocks` valid pages)
always kept, the rest by (score desc, page asc) on the fp32 bit order; output
in ascending page order, `-1` tail when fewer than 16 pages are valid.

Split rows: a row of at most `single_max` valid (4096)
pages is one CTA of 256 threads (the unsplit select); a longer row is cut
into chunks of `chunk` pages, one CTA each, that write their exact local
top pages to a partial buffer, and the last chunk CTA to arrive (a per-row
counter it re-arms) merges them. The union of the per-chunk top-`need`
holds the row's top-`need`, so the table is the same, bit for bit. Grid =
(rows x heads, splits), splits from the static column count; no CTA waits for
another, so the grid needs no residency check. PDL: waits after its pre-wait
loads (n_valid, page rows), then triggers dependents.
"""

from __future__ import annotations

import threading

TOPK = 16
MAX_COLS = 8192
# Rows of the largest launch (grid = rows x heads CTAs, no residency bound).
MAX_ROWS = 64
VARIANT = 5
# Split-row defaults (pages): rows above SINGLE_MAX pages are cut into CHUNK-page CTAs.
# Up to 4096 pages the one-CTA select is faster; above it the one CTA takes 32 keys per
# thread and the split wins in the decode chain. The tests pass smaller ``chunk`` /
# ``single_max`` to exercise the split at short contexts.
CHUNK = 1024
SINGLE_MAX = 4096
# Launches of at most SPLIT_MIN_ROWS rows (one request at K = 7) never split: the
# one-CTA select (6 / 16 / 32 keys per thread up to 8192 pages, one CTA per row) is as
# fast in the decode chain there, and a split grid's extra CTAs slow the next kernel's
# launch.
SPLIT_MIN_ROWS = 8
# Partial entries one row merges: splits x 16 <= 256.
MAX_SPLITS = 16
# (rows x heads) the partial / counter buffers are sized for.
MAX_ROW_HEADS = MAX_ROWS * 16


_lock = threading.Lock()
_compiled: dict = {}


def _ext():
    """The CuTe DSL kernel module (imported on first use; raises if the CuTe DSL is
    unavailable)."""
    from . import page_topk_kernel

    return page_topk_kernel


def max_splits(num_cols: int, chunk: int, single_max: int) -> int:
    """Chunk CTAs per (row, head) of a launch over `num_cols` score columns (1 = the
    one-CTA kernel)."""
    cap = min(int(num_cols), MAX_COLS)
    return 1 if cap <= single_max else -(-cap // chunk)


def _compiled_fn(split_mode: bool, has_diag: bool, pdl: bool, device):
    """cute.compile the launcher of (split_mode, has_diag, pdl); cached. Every argument
    but those is runtime."""
    key = (bool(split_mode), bool(has_diag), bool(pdl))
    fn = _compiled.get(key)
    if fn is not None:
        return fn
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "page_topk: compile (build()) outside CUDA-graph capture first"
        )
    k = _ext()
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    with _lock:
        fn = _compiled.get(key)
        if fn is None:
            fn = _compiled[key] = cute.compile(
                k.select_launch,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                1,
                0,
                1024,
                4096,
                0,
                0,
                1,
                key[0],
                key[1],
                key[2],
                stream,
            )
    return fn


class PageTopkStage:
    """`__call__(scores, n_valid, page_rows, table)`

    scores     float32 [nd, H, C]      index page scores (C <= 8192, stride(2) == 1);
                                       upstream `unified_scores[:nd]`
    n_valid    int32   [>= nd]         valid index pages of each row's token
    page_rows  int32   [>= nd, >= C]   logical page -> physical KV page id
    table      int32   [H, nd, >= 16]  columns 0..15 written
    """

    stage = "page_topk"

    def __init__(
        self,
        *,
        init_blocks: int,
        local_blocks: int,
        max_rows: int = 8,
        pdl: bool = True,
        chunk: int = CHUNK,
        single_max: int = SINGLE_MAX,
    ):
        if init_blocks < 0 or local_blocks < 1:
            raise ValueError(
                "page_topk: local_blocks must be >= 1 (the sparse decode assumes the "
                "partial last page is selected)"
            )
        self.init_blocks, self.local_blocks = int(init_blocks), int(local_blocks)
        if not 1 <= max_rows <= MAX_ROWS:
            raise ValueError(f"page_topk: max_rows {max_rows} outside 1..{MAX_ROWS}")
        self.max_rows, self.pdl = int(max_rows), bool(pdl)
        if not 512 <= chunk <= 4096 or not TOPK <= single_max <= MAX_COLS:
            raise ValueError(
                f"page_topk: chunk {chunk} outside 512..4096 or single_max "
                f"{single_max} outside 16..{MAX_COLS}"
            )
        if -(-MAX_COLS // chunk) > MAX_SPLITS:
            raise ValueError(
                f"page_topk: chunk {chunk} gives more than {MAX_SPLITS} splits at "
                f"{MAX_COLS} columns"
            )
        self.chunk, self.single_max = int(chunk), int(single_max)
        self._m: bool | None = None
        self._part = None
        self._cnt = None

    def single_max_of(self, rows: int) -> int:
        """Pages up to which a launch of `rows` rows keeps one CTA per row."""
        return MAX_COLS if rows <= SPLIT_MIN_ROWS else self.single_max

    def splits(self, cols: int, rows: int = SPLIT_MIN_ROWS + 1) -> int:
        """CTAs per (row, head) of a launch of `rows` rows over `cols` score
        columns."""
        cap = min(int(cols), MAX_COLS)
        return 1 if cap <= self.single_max_of(rows) else -(-cap // self.chunk)

    def build(self):
        import torch

        dev = torch.device("cuda", torch.accelerator.current_device_index())
        # Both launch shapes (one CTA per row; split rows) for this stage's PDL setting,
        # before any capture.
        for split_mode in (False, True):
            _compiled_fn(split_mode, False, self.pdl, dev)
        self._m = True
        # The counter is zeroed once; the merging CTA of each row re-arms it.
        self._cnt = torch.zeros(MAX_ROW_HEADS, dtype=torch.int32, device=dev)
        self._part = torch.zeros(
            MAX_ROW_HEADS * MAX_SPLITS * TOPK, dtype=torch.int64, device=dev
        )
        return self

    def __call__(self, scores, n_valid, page_rows, table, *, diag=None):
        import torch

        if self._m is None:
            self.build()
        nd = table.shape[1]
        if not 1 <= nd <= self.max_rows:
            raise ValueError(f"page_topk: {nd} rows outside 1..{self.max_rows}")
        if scores.dtype != torch.float32 or n_valid.dtype != torch.int32:
            raise TypeError("page_topk: scores float32, n_valid int32")
        if page_rows.dtype != torch.int32 or table.dtype != torch.int32:
            raise TypeError("page_topk: page_rows / table int32")
        if scores.shape[2] > MAX_COLS:
            raise ValueError(f"page_topk: {scores.shape[2]} columns > {MAX_COLS}")
        if scores.shape[0] < nd or scores.shape[1] != table.shape[0]:
            raise ValueError("page_topk: scores [nd, H, C] vs table [H, nd, 16]")
        if nd * scores.shape[1] > MAX_ROW_HEADS:
            raise ValueError(
                f"page_topk: {nd} rows x {scores.shape[1]} heads > {MAX_ROW_HEADS}"
            )
        if self._part is None or self._part.device != scores.device:
            raise ValueError("page_topk: build() on the scores' device first")
        if scores.stride(2) != 1 or table.stride(2) != 1 or page_rows.stride(1) != 1:
            raise ValueError(
                "page_topk: scores / table / page_rows need a contiguous last dim"
            )
        cols, heads = int(scores.shape[2]), int(scores.shape[1])
        single_max = self.single_max_of(nd)
        splits = max_splits(cols, self.chunk, single_max)
        fn = _compiled_fn(splits > 1, diag is not None, self.pdl, scores.device)
        import cuda.bindings.driver as cuda_driver

        stream = cuda_driver.CUstream(
            torch.cuda.current_stream(scores.device).cuda_stream
        )
        fn(
            scores.data_ptr(),
            scores.stride(0),
            scores.stride(1),
            n_valid.data_ptr(),
            page_rows.data_ptr(),
            page_rows.stride(0),
            cols,
            table.data_ptr(),
            table.stride(0),
            table.stride(1),
            nd * heads,
            heads,
            self.init_blocks,
            self.local_blocks,
            diag.data_ptr() if diag is not None else 0,
            self.chunk,
            single_max,
            self._part.data_ptr(),
            self._cnt.data_ptr(),
            splits,
            stream,
        )


__all__ = [
    "PageTopkStage",
    "TOPK",
    "MAX_COLS",
    "MAX_ROWS",
    "CHUNK",
    "SINGLE_MAX",
    "max_splits",
]
