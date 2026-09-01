# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve the Qwen4Exp PLE n-gram table from NVMe via mmap.

Why: the n-gram (PLE) table is tens of GiB (FP8-quantized, or unquantized
BF16 for e.g. AutoRound W4A16 exports) and the stock path keeps it resident
(GPU, via ``VocabParallelEmbedding``). On a box where host and GPU share one
unified memory pool, that table cannot sit next to the rest of the model. A
token only ever touches a handful of rows out of the whole table, so it can
live on disk and be served through the page cache instead.

How, with ``VLLM_PLE_MMAP=1``:
  * ``Qwen4ExpNGramEmbedding.__init__`` swaps the GPU-resident embedding for
    :class:`MmapNgramEmbedding`, a placeholder whose ``forward`` gathers rows
    from :class:`MmapPleTable` (``np.memmap`` views over the checkpoint's
    safetensors shards, page-cache backed).
  * ``Qwen4ExpNGramEmbedding.load_weights`` drops the per-shard tensors on
    the floor (never materialized into a resident table); for a dtype whose
    :class:`_PleDtype` descriptor requires one, it also keeps the
    checkpoint's global ``weight_scale`` as a buffer on the placeholder,
    which ``Qwen4ExpPLELayer._dequantize_embeddings`` already knows how to
    consume. Unquantized dtypes (e.g. BF16) need no scale and pass through
    unscaled.
  * Model Runner V2 only. ``Qwen4ExpModelState`` moves the trigram hashing
    and row gather out of model forward entirely: ``prepare_inputs`` calls
    :func:`MmapNgramEmbedding.gather_into` (via
    ``Qwen4ExpNGramEmbedding.prepare_mmap_rows``) directly into a stable,
    module-owned buffer BEFORE the compiled/captured forward runs.
    Captured model code only ever reads that buffer (a plain, symbolically
    sliced tensor read), so there is no host work, D2H, or Python custom-op
    body left inside the traced/captured graph — the original
    ``.numel()``-derived slicing that caused Dynamo's
    ``ConstraintViolationError`` on ``query_start_loc.size()[0]`` simply
    never runs under compilation. V1 does not prepare these inputs after
    #53896 and is rejected at construction with an actionable error.

This module is imported unconditionally at ``nvidia/ple_layer.py`` module
scope; every behavior above is gated on :func:`enabled` at call time. With
``VLLM_PLE_MMAP`` unset, nothing in this module is ever invoked and the
stock classes are untouched.

Knobs (env, registered in ``vllm/envs.py``):
  VLLM_PLE_MMAP=1            enable
  VLLM_PLE_MMAP_WORKERS=32   gather threads (page faults overlap across them)
  VLLM_PLE_MMAP_CHUNK=2048   rows per gather task
  VLLM_PLE_MMAP_PREWARM=0    1 = stream the table once at load, bounded by
                             free memory, to warm the page cache
  VLLM_PLE_MMAP_READAHEAD=0  N > 0 = before each gather, hand the kernel up
                             to N coalesced file ranges via
                             posix_fadvise(WILLNEED) so the worker pool
                             faults against in-flight I/O
  VLLM_PLE_MMAP_PINNED=0     1 = stage each input-prep gather through a
                             per-call pinned host buffer before H2D,
                             instead of copying straight out of
                             gather()'s pageable numpy array
  VLLM_PLE_MMAP_SERIAL=0     N > 0 = a gather touching at most N distinct
                             rows runs its tasks inline on the calling
                             thread instead of dispatching through the
                             worker pool
"""

from __future__ import annotations

import functools
import glob
import json
import math
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import regex as re
import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import get_dtype_size

if TYPE_CHECKING:
    from vllm.config import CompilationConfig, ModelConfig, VllmConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class _PleDtype:
    """One PLE table dtype's storage type and whether it needs a scale."""

    torch_dtype: torch.dtype
    requires_scale: bool


_PLE_DTYPES: dict[str, _PleDtype] = {
    "F8_E4M3": _PleDtype(torch.float8_e4m3fn, requires_scale=True),
    "BF16": _PleDtype(torch.bfloat16, requires_scale=False),
    # F8_E5M2 deliberately excluded: is_fp8() (fp8_utils.py) only recognizes
    # float8_e4m3fn/float8_e4m3fnuz, so an e5m2 table would silently skip
    # Qwen4ExpPLELayer._dequantize_embeddings's dequant gate and fail late,
    # deep in a downstream matmul, instead of at load (invariant 4).
}
_SCALE_TORCH_DTYPES: dict[str, torch.dtype] = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
}
_MAX_HEADER_BYTES = 100 << 20  # 100 MB
_PREWARM_HEADROOM_BYTES = 8 << 30  # 8 GiB
_LOG_INTERVAL_S = 60.0
_HAS_POSIX_FADVISE = hasattr(os, "posix_fadvise")

_SHARD_RE = re.compile(
    r"layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
)
_SCALE_RE = re.compile(
    r"layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.weight_scale$"
)
_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def _itemsize(dtype_str: str) -> int:
    """Look up a safetensors dtype string's byte width.

    Raises:
        ValueError: named, in place of a bare KeyError, so a checkpoint with
            an unrecognized dtype fails with a clear message.
    """
    torch_dtype = _SAFETENSORS_TO_TORCH_DTYPE.get(dtype_str)
    if torch_dtype is None:
        raise ValueError(f"PLE mmap: unrecognized safetensors dtype {dtype_str!r}")
    return get_dtype_size(torch_dtype)


def enabled() -> bool:
    """Return True when the mmap-backed PLE path is enabled."""
    return envs.VLLM_PLE_MMAP


# --------------------------------------------------------------------------- #
# safetensors header parsing. No model.safetensors.index.json exists for this
# checkpoint, so raw file offsets come from the header directly.
# --------------------------------------------------------------------------- #
def parse_safetensors_header(path: str) -> tuple[dict, int]:
    """Parse one safetensors file's header.

    Args:
        path: path to a ``.safetensors`` file.

    Returns:
        (tensor_metadata, data_start_offset): the header dict (with
        ``__metadata__`` removed) and the byte offset where tensor data
        begins.

    Raises:
        ValueError: the header is truncated, exceeds the size cap, or any
            tensor's ``data_offsets`` fall outside the file.
    """
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: truncated safetensors header length")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len > _MAX_HEADER_BYTES:
            raise ValueError(
                f"{path}: safetensors header is {header_len} bytes, "
                f"exceeding the {_MAX_HEADER_BYTES}-byte cap"
            )
        raw_header = f.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError(f"{path}: truncated safetensors header body")
        header = json.loads(raw_header)
    header.pop("__metadata__", None)
    data_start = 8 + header_len
    for name, meta in header.items():
        try:
            start, end = meta["data_offsets"]
        except (KeyError, ValueError):
            raise ValueError(
                f"{path}: tensor {name!r} header entry has no valid "
                f"data_offsets: {meta.get('data_offsets')!r}"
            ) from None
        if start < 0 or end < start or data_start + end > file_size:
            raise ValueError(
                f"{path}: tensor {name!r} data_offsets [{start}, {end}) "
                f"fall outside the file (size {file_size})"
            )
    return header, data_start


@dataclass
class _LayerShards:
    """Discovered PLE shard/scale tensors for one PLE layer."""

    shards: dict[int, tuple[str, int, int]]  # shard_idx -> (path, offset, rows)
    cols: int
    dtype_str: str
    scale_entry: tuple[str, int, int, str] | None  # (path, offset, nbytes, dtype)


@functools.cache
def discover_shards(model_path: str) -> dict[int, _LayerShards]:
    """Parse every safetensors header under ``model_path`` for PLE tensors.

    Header-only reads (a few KB per file), never the multi-GiB tensor data;
    cheap enough to run once per load regardless of checkpoint size — and
    memoized by ``model_path`` since ``validate_shards_for`` calls this once
    per PLE layer (construction happens per-layer), which would otherwise
    re-glob and re-parse every checkpoint file's header once per layer.

    Args:
        model_path: local directory holding the checkpoint's safetensors
            shards.

    Returns:
        Mapping from (0-based) decoder layer index to its discovered shards.

    Raises:
        ValueError: a shard's on-disk size does not match its declared
            shape/dtype, or shards for one layer disagree on dtype/width.
    """
    per_layer: dict[int, dict[int, tuple[str, int, int]]] = {}
    cols_by_layer: dict[int, int] = {}
    dtype_by_layer: dict[int, str] = {}
    scale_by_layer: dict[int, tuple[str, int, int, str]] = {}

    for path in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        header, data_start = parse_safetensors_header(path)
        for name, meta in header.items():
            shard_match = _SHARD_RE.search(name)
            if shard_match:
                layer_idx = int(shard_match.group(1))
                shard_idx = int(shard_match.group(2))
                start, end = meta["data_offsets"]
                try:
                    rows, cols = meta["shape"]
                except (KeyError, ValueError):
                    raise ValueError(
                        f"{path}: PLE shard {name!r} has an unexpected "
                        f"shape {meta.get('shape')!r} (expected a "
                        "2-element [rows, cols])"
                    ) from None
                dtype_str = meta["dtype"]
                if end - start != rows * cols * _itemsize(dtype_str):
                    raise ValueError(
                        f"{path}: PLE shard {name!r} size does not match "
                        f"its declared shape/dtype"
                    )
                prev_dtype = dtype_by_layer.setdefault(layer_idx, dtype_str)
                if prev_dtype != dtype_str:
                    raise ValueError(
                        f"PLE layer {layer_idx}: mixed shard dtypes "
                        f"{prev_dtype!r} vs {dtype_str!r}"
                    )
                prev_cols = cols_by_layer.setdefault(layer_idx, cols)
                if prev_cols != cols:
                    raise ValueError(
                        f"PLE layer {layer_idx}: mixed shard widths "
                        f"{prev_cols} vs {cols}"
                    )
                per_layer.setdefault(layer_idx, {})[shard_idx] = (
                    path,
                    data_start + start,
                    rows,
                )
                continue
            scale_match = _SCALE_RE.search(name)
            if scale_match:
                layer_idx = int(scale_match.group(1))
                start, end = meta["data_offsets"]
                scale_by_layer[layer_idx] = (
                    path,
                    data_start + start,
                    end - start,
                    meta["dtype"],
                )

    return {
        layer_idx: _LayerShards(
            shards=shards,
            cols=cols_by_layer[layer_idx],
            dtype_str=dtype_by_layer[layer_idx],
            scale_entry=scale_by_layer.get(layer_idx),
        )
        for layer_idx, shards in per_layer.items()
    }


def _read_scale(entry: tuple[str, int, int, str]) -> torch.Tensor:
    """Read one small scalar tensor directly out of a safetensors file."""
    path, offset, nbytes, dtype_str = entry
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise ValueError(f"{path}: truncated weight_scale read")
    torch_dtype = _SCALE_TORCH_DTYPES.get(dtype_str)
    if torch_dtype is None:
        raise ValueError(f"unsupported weight_scale dtype {dtype_str!r}")
    # A manual (u16 << 16) reconstruction overflows int32 for negative
    # (sign-bit-set) values in either 16-bit format; frombuffer with the
    # real dtype avoids the bit-manipulation entirely.
    itemsize = get_dtype_size(torch_dtype)
    raw_bytes = bytearray(raw[:itemsize])
    return torch.frombuffer(raw_bytes, dtype=torch_dtype).clone().squeeze()


# --------------------------------------------------------------------------- #
# The mmap-backed table itself.
# --------------------------------------------------------------------------- #
def _coalesce_runs(offsets: np.ndarray, row_bytes: int) -> list[tuple[int, int]]:
    """Merge equal-length spans at ascending ``offsets`` into byte runs.

    Only spans that abut are merged. Bridging a gap would fetch pages no row
    in this gather needs, and on a hash-scattered table that amplification
    re-creates the cold-read cost the readahead exists to remove.

    Args:
        offsets: ascending int64 file offsets, one per row.
        row_bytes: length of the span each offset starts.

    Returns:
        ``(file_offset, length)`` pairs covering exactly the input spans.
    """
    if offsets.size == 0:
        return []
    breaks = np.flatnonzero(offsets[1:] != offsets[:-1] + row_bytes) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [offsets.size]))
    return [
        (int(offsets[a]), (b - a) * row_bytes)
        for a, b in zip(starts.tolist(), ends.tolist())
    ]


class MmapPleTable:
    """Row gather over a PLE table split into shard files, served via mmap.

    Shard ``i`` holds global rows ``[i * shard_size, i * shard_size + rows)``
    — the same layout ``Qwen4ExpNGramEmbedding.load_weights``'s
    ``checkpoint_start`` math assumes, so shard/row lookup here must stay in
    lockstep with that code (see the shard-mapping contract test).

    ``model_path`` is recorded so :func:`build_tables` can detect a
    reload_weights call that repoints ``model_config`` at a different
    checkpoint on an already-attached layer: silently keeping the old
    table would serve checkpoint A's mmap rows against checkpoint B's
    scale.
    """

    def __init__(
        self,
        shards: dict[int, tuple[str, int, int]],
        shard_size: int,
        row_bytes: int,
        torch_dtype: torch.dtype,
        workers: int,
        chunk: int,
        model_path: str,
        readahead: int = 0,
        serial: int = 0,
    ) -> None:
        # First, and before anything below that can raise (the memmap loop,
        # _open_readahead_fds, the thread pool): __del__ unconditionally
        # calls close() on whatever state exists, including on a
        # half-constructed table, so these must never be missing.
        self._closed = False
        self._fds: dict[str, int] = {}
        self._shard_fds: list[int | None] = []
        self.mm: list[np.memmap | None] = []
        if not shards:
            raise ValueError("PLE mmap: no shards to build a table from")
        self.shard_size = int(shard_size)
        self.row_bytes = int(row_bytes)
        self.torch_dtype = torch_dtype
        self.model_path = model_path
        self.itemsize = get_dtype_size(torch_dtype)
        self.chunk = max(1, int(chunk))
        self.workers = max(1, int(workers))
        # posix_fadvise is POSIX-only. Where it is missing there is no
        # readahead mechanism at all, so the knob simply reads as off rather
        # than growing a fallback ladder.
        self.readahead = max(0, int(readahead)) if _HAS_POSIX_FADVISE else 0
        self.serial = max(0, int(serial))
        n_slots = max(shards) + 1
        self.mm = [None] * n_slots
        self.rows_total = 0
        for idx, (path, offset, rows) in shards.items():
            self.mm[idx] = np.memmap(
                path, dtype=np.uint8, mode="r", offset=offset, shape=(rows, row_bytes)
            )
            self.rows_total += rows
        self._shard_fds = [None] * n_slots
        if self.readahead > 0:
            self._open_readahead_fds(shards)
            covered_files = len(self._fds)
            if covered_files > 0:
                total_files = len({path for path, _offset, _rows in shards.values()})
                # No layer_idx here: info_once dedups on (msg, *args), so a
                # per-layer argument would re-log for every PLE layer. The
                # covered/total counts are fixed per table, so a different
                # table logging a different pair is expected and
                # informative, not a dedup failure.
                logger.info_once(
                    "PLE mmap: readahead pre-pass active (posix_fadvise "
                    "WILLNEED), %d/%d shard files opened, bounded at %d "
                    "coalesced runs per gather",
                    covered_files,
                    total_files,
                    self.readahead,
                )
            else:
                # _open_readahead_fds already warned per-file; this is the
                # table-level state that matters for an on/off A/B — without
                # it, self.readahead > 0 would keep reading as active while
                # the pre-pass never issues a single fadvise call.
                logger.warning(
                    "PLE mmap: readahead requested (VLLM_PLE_MMAP_READAHEAD="
                    "%d) but no shard file could be opened; the pre-pass is "
                    "inert",
                    self.readahead,
                )
                self.readahead = 0
        self.pool = ThreadPoolExecutor(max_workers=self.workers)
        self._pending = 0
        self._errors = 0
        self._bound_warned = False
        self._rows_since_log = 0
        # (elapsed_ms, populate_ms, copy_ms, coalesced runs) per gather.
        self._latencies_ms: list[tuple[float, float, float, int]] = []
        # Per-window engaged count backing the serial= log field -- see
        # _record for why it's window-scoped rather than the p99 sample's
        # own flag.
        self._serial_engaged_since_log = 0
        self._last_log = time.monotonic()

    def _open_readahead_fds(self, shards: dict[int, tuple[str, int, int]]) -> None:
        """Open one shared read-only fd per distinct shard file.

        Shards commonly share a safetensors file, so keying on the path keeps
        this to a handful of descriptors instead of one per shard slot.
        ``posix_fadvise`` takes an explicit offset and never consults the file
        position, so sharing a descriptor across the gather pool is safe.
        Worker processes fork before the model loads and ``build_tables`` runs
        in-worker, so these descriptors never cross a fork; a later fork would
        only duplicate read-only descriptors anyway.

        Fail-soft: a file that will not reopen just loses its readahead — its
        ``_shard_fds`` slot stays None and the memmap continues to serve it.
        """
        for idx, (path, _offset, _rows) in shards.items():
            fd = self._fds.get(path)
            if fd is None:
                try:
                    fd = os.open(path, os.O_RDONLY)
                except OSError as exc:
                    logger.warning_once(
                        "PLE mmap: readahead unavailable for a shard file "
                        "(%s); gathers are unaffected",
                        exc.strerror,
                    )
                    continue
                self._fds[path] = fd
            self._shard_fds[idx] = fd

    def _readahead(
        self,
        seg_starts: list[int],
        seg_ends: list[int],
        shard: np.ndarray,
        local: np.ndarray,
    ) -> int:
        """Start the kernel reading the rows this gather is about to copy.

        ``posix_fadvise(WILLNEED)`` returns once the readahead is queued, so
        the worker pool below faults against I/O already in flight rather than
        issuing it one page fault at a time.

        Reuses the caller's segmentation instead of recomputing one: ids
        arrive sorted from ``np.unique`` and ``shard = uniq // shard_size`` is
        monotonic in ``uniq``, so every shard owns exactly one contiguous
        segment and this pre-pass covers precisely the rows the copy tasks do.

        One numpy pass over every segment computes ``offsets`` (the FILE byte
        offset each row's span starts at) exactly once and stashes it with
        its shard fd; the coalesced run count needed to check the bound
        (``self.readahead``) is accumulated in the same pass. Recomputing
        ``offsets`` a second time — once to count, once to coalesce — was
        measured costing the active (bound-clears) arm ~37% extra populate
        time. Only once the total clears the bound does a second, Python-level
        pass turn the stashed per-segment offsets into the (fd, offset,
        length) run list and issue the ``posix_fadvise`` calls; a bound-
        skipped gather never reaches it, so it pays only the numpy pass.

        Returns:
            The coalesced run count, reported even when it exceeds the bound
            and nothing was issued — a silent skip would be indistinguishable
            from readahead being off.
        """
        total_runs = 0
        segments: list[tuple[int, np.ndarray]] = []
        for s, e in zip(seg_starts, seg_ends):
            si = int(shard[s])
            mm = self.mm[si]
            fd = self._shard_fds[si]
            # A missing or closed shard is the copy loop's error to raise, so
            # that its named IndexError stays the single reported failure.
            if mm is None or fd is None:
                continue
            offsets = mm.offset + local[s:e] * self.row_bytes
            total_runs += (
                int(np.count_nonzero(offsets[1:] != offsets[:-1] + self.row_bytes)) + 1
            )
            segments.append((fd, offsets))

        if total_runs > self.readahead:
            # Latched per table, not warning_once's process-wide (msg, *args)
            # cache: the run count varies almost every gather, which would
            # otherwise thrash that cache into re-logging on nearly every
            # bound-exceeded call instead of deduping anything.
            if not self._bound_warned:
                self._bound_warned = True
                logger.warning(
                    "PLE mmap: readahead skipped, %d coalesced runs exceed "
                    "VLLM_PLE_MMAP_READAHEAD=%d",
                    total_runs,
                    self.readahead,
                )
            return total_runs

        for fd, offsets in segments:
            for offset, length in _coalesce_runs(offsets, self.row_bytes):
                try:
                    os.posix_fadvise(fd, offset, length, os.POSIX_FADV_WILLNEED)
                except OSError as exc:
                    logger.warning_once(
                        "PLE mmap: readahead call failed (%s); the gather still "
                        "serves every row, just colder",
                        exc.strerror,
                    )
                    # This fd is bad for every shard it backs (shards commonly
                    # share a file) — latch every matching slot so future
                    # gathers stop paying for a call that will only fail again.
                    for idx, shard_fd in enumerate(self._shard_fds):
                        if shard_fd == fd:
                            self._shard_fds[idx] = None
        return total_runs

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """Gather table rows for a batch of global row ids.

        Args:
            ids: int64 array of global row ids, any shape.

        Returns:
            A fresh, writable ``uint8`` array shaped ``[ids.size,
            row_bytes]``, one row per input id in input order.

        Raises:
            IndexError: an id falls outside the table's row range.
        """
        start_t = time.monotonic()
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        uniq, inverse = np.unique(ids, return_inverse=True)
        if uniq[0] < 0 or uniq[-1] >= self.rows_total:
            self._errors += 1
            raise IndexError(
                f"PLE mmap: row id out of range [{uniq[0]}, {uniq[-1]}] "
                f"for {self.rows_total} rows"
            )
        shard = uniq // self.shard_size
        local = uniq - shard * self.shard_size
        out = np.empty((uniq.size, self.row_bytes), dtype=np.uint8)

        bounds = np.flatnonzero(np.diff(shard)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [uniq.size]))
        seg_starts = starts.tolist()
        seg_ends = ends.tolist()

        runs = 0
        populate_ms = 0.0
        if self.readahead > 0:
            populate_t = time.monotonic()
            try:
                runs = self._readahead(seg_starts, seg_ends, shard, local)
            except (OSError, ValueError) as exc:
                # The pre-pass never touches gathered data, so anything it
                # raises costs only the readahead: the copy loop below stays
                # the sole correctness path.
                logger.warning_once(
                    "PLE mmap: readahead pre-pass raised %s; continuing without it",
                    type(exc).__name__,
                )
            populate_ms = (time.monotonic() - populate_t) * 1000.0

        tasks: list[tuple[int, int, int]] = []
        for s, e in zip(seg_starts, seg_ends):
            si = int(shard[s])
            for c in range(s, e, self.chunk):
                tasks.append((si, c, min(c + self.chunk, e)))

        def run(task: tuple[int, int, int]) -> None:
            si, a, b = task
            mm = self.mm[si]
            if mm is None:
                raise IndexError(f"PLE mmap: shard {si} missing")
            # Fancy indexing on a memmap: page faults perform the I/O, and
            # NumPy releases the GIL for the copy, so tasks overlap.
            out[a:b] = mm[local[a:b]]

        self._pending = len(tasks)
        copy_t = time.monotonic()
        try:
            # VLLM_PLE_MMAP_SERIAL: an opt-in bypass of the executor for
            # small gathers, keyed on uniq.size (distinct rows), not task
            # count -- a 96-row batch-1 gather over 128 hash-scattered
            # shards degrades to ~96 one-row tasks and still pays pool
            # dispatch even though the len(tasks) == 1 case below already
            # runs inline. Measured 2026-08-29, estrella159 sm120: 0.18 ms
            # direct fancy-index vs 3.7 ms via the 32-worker pool at 96 warm
            # rows. The readahead pre-pass above only runs when
            # VLLM_PLE_MMAP_READAHEAD is also set (it defaults off and is
            # otherwise independent of this knob) -- serial has no upper
            # bound of its own, so a high threshold with readahead left off
            # serializes page faults on the calling thread for a cold,
            # prefill-sized gather with nothing to mitigate it. Accepted for
            # an opt-in knob with no default-behavior change.
            serial = self.serial > 0 and uniq.size <= self.serial
            if serial or len(tasks) == 1:
                for task in tasks:
                    run(task)
            else:
                for _ in self.pool.map(run, tasks):
                    pass
        except Exception:
            self._errors += 1
            raise
        finally:
            # Snapshot before resetting: _record's log line (fired at most
            # once per _LOG_INTERVAL_S) needs THIS call's task count, not
            # the always-zero post-reset value. This is a task count, not a
            # concurrency depth: VLLM_PLE_MMAP_SERIAL can run ~96 pending
            # tasks at depth 1 on the calling thread.
            pending_snapshot = self._pending
            self._pending = 0
        copy_ms = (time.monotonic() - copy_t) * 1000.0
        gathered = out[inverse]
        self._record(
            int(ids.size),
            pending_snapshot,
            (time.monotonic() - start_t) * 1000.0,
            populate_ms,
            copy_ms,
            runs,
            serial,
        )
        return gathered

    def _record(
        self,
        rows: int,
        pending: int,
        elapsed_ms: float,
        populate_ms: float,
        copy_ms: float,
        runs: int,
        serial: bool,
    ) -> None:
        self._latencies_ms.append((elapsed_ms, populate_ms, copy_ms, runs))
        self._rows_since_log += rows
        # serial= is a WINDOW statistic (engaged/total gathers), not the p99
        # SAMPLE's own flag: the p99 sample is, by construction, the biggest
        # gather in the window, and a mixed window's biggest gather is
        # exactly the one most likely to have crossed the serial threshold
        # into the pool branch -- keying on it reported serial=0 for a
        # window in which 19 of 20 gathers had engaged, the shape this
        # commit's regression test drives. Counting every call instead of
        # sampling one reflects what actually happened. The window's total
        # is len(self._latencies_ms), just appended to above.
        if serial:
            self._serial_engaged_since_log += 1
        now = time.monotonic()
        if now - self._last_log < _LOG_INTERVAL_S:
            return
        # Sorting the samples orders them by elapsed_ms, so the readahead and
        # copy halves reported below decompose the p99 call itself rather than
        # averaging a slow gather together with fast ones.
        latencies = sorted(self._latencies_ms)
        p99_idx = max(0, math.ceil(len(latencies) * 0.99) - 1)
        p99, p99_populate, p99_copy, p99_runs = (
            latencies[p99_idx] if latencies else (0.0, 0.0, 0.0, 0)
        )
        logger.info(
            "rows=%d p99_ms=%.2f populate_ms=%.2f copy_ms=%.2f runs=%d "
            "pending=%d errors=%d serial=%d/%d",
            self._rows_since_log,
            p99,
            p99_populate,
            p99_copy,
            p99_runs,
            pending,
            self._errors,
            self._serial_engaged_since_log,
            len(self._latencies_ms),
        )
        self._latencies_ms.clear()
        self._rows_since_log = 0
        self._serial_engaged_since_log = 0
        self._last_log = now

    def prewarm(self, max_bytes: int) -> int:
        """Stream up to ``max_bytes`` of the table into the page cache.

        Args:
            max_bytes: byte budget; a non-positive value skips prewarm.

        Returns:
            Bytes actually read.
        """
        if max_bytes <= 0:
            return 0
        block = 64 << 20
        remaining = max_bytes
        read_total = 0
        for mm in self.mm:
            if mm is None or remaining <= 0:
                continue
            path = mm.filename
            if path is None:
                # Every memmap here was opened from a path string in
                # __init__; None only occurs for an anonymous mmap, which
                # this class never creates.
                raise RuntimeError("PLE mmap: memmap has no backing file")
            start = mm.offset
            end = start + mm.shape[0] * mm.shape[1]
            with open(path, "rb", buffering=0) as f:
                f.seek(start)
                pos = start
                while pos < end and remaining > 0:
                    chunk = f.read(min(block, end - pos, remaining))
                    if not chunk:
                        break
                    pos += len(chunk)
                    remaining -= len(chunk)
                    read_total += len(chunk)
        return read_total

    def close(self) -> None:
        """Release the gather thread pool, readahead fds, and memmaps.

        Idempotent: safe to call more than once, and safe on a table that
        was never gathered from. Guards against leaking the previous
        table's ThreadPool when a layer's table is rebuilt in place (e.g. a
        weight-reload re-entering _attach_table on an already-populated
        placeholder).

        Also safe on a table whose ``__init__`` raised before ``self.pool``
        was assigned (e.g. the ThreadPoolExecutor call itself failed): the fd
        and memmap containers are always set first, so whatever
        ``_open_readahead_fds`` already opened by that point still gets
        closed instead of leaking.
        """
        if self._closed:
            return
        self._closed = True
        pool = getattr(self, "pool", None)
        if pool is not None:
            pool.shutdown(wait=False)
        self.mm = [None] * len(self.mm)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._shard_fds = [None] * len(self._shard_fds)

    def __del__(self) -> None:
        self.close()


def _mem_available_bytes(path: str = "/proc/meminfo") -> int:
    """Read ``MemAvailable`` from a ``/proc/meminfo``-format file, in bytes."""
    with open(path) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError(f"PLE mmap: MemAvailable not found in {path}")


def compute_prewarm_bound(table_bytes: int, mem_available_bytes: int) -> int:
    """Bound a prewarm read so it never eats into headroom.

    A ``mem_available_bytes`` below the headroom clamps to 0 rather than
    going negative, which would otherwise slice-read nearly the whole table
    exactly when memory is scarcest.
    """
    return min(table_bytes, max(0, mem_available_bytes - _PREWARM_HEADROOM_BYTES))


def _pin_host_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """Allocate a pinned host tensor shaped and dtyped like ``cpu_tensor``.

    The sole call to ``pin_memory=True`` in this module, kept behind this
    one-line indirection (the same monkeypatch-seam style as
    ``_coalesce_runs``): ``forward``'s ``ids.device.type == "cuda"`` gate
    makes the pinned branch unreachable on CPU, so an un-monkeypatched call
    here would only ever run on a real CUDA device — but a naive CPU-only
    test calling it directly would otherwise init a CUDA context.
    """
    return torch.empty_like(cpu_tensor, pin_memory=True)


def _stage_pinned(cpu_tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Copy ``cpu_tensor`` (pageable) into a fresh pinned host buffer.

    Per-call allocation from torch's caching host allocator, not a
    persistent grow-only buffer: the allocator only returns a block to the
    pool once the CUDA events recorded against it by the H2D copy that
    follows complete (``CachingHostAllocator``'s ``recordEvent`` contract,
    the same one ``copy_kernel_cuda`` relies on), so a fresh per-call
    allocation is safe against reuse racing an in-flight async copy.

    Fail-soft: allocation or the copy into it can raise (e.g. cudaHostAlloc
    exhaustion), in which case the caller's normal pageable H2D runs
    unchanged — this never costs the request its correctness, only the
    staging.

    Benign growth: torch's caching host allocator buckets by power-of-2
    size class, so the pinned pool this repeatedly draws from is bounded
    to roughly 2x the largest payload observed per class, not unbounded —
    the answer if a tester reports host memory growing after enabling this.

    Returns:
        ``(tensor, engaged)``. On success, ``tensor`` is the new pinned
        buffer holding ``cpu_tensor``'s values and ``engaged`` is True. If
        either the allocation or the copy into it raises, ``tensor`` is
        ``cpu_tensor`` itself and ``engaged`` is False.
    """
    try:
        staged = _pin_host_tensor(cpu_tensor)
        staged.copy_(cpu_tensor)
    except Exception as exc:
        # Exception, not just RuntimeError: this guards the copy_ above as
        # well as the allocation, and neither the allocator nor the copy's
        # failure modes are contractually limited to RuntimeError -- the
        # fallback path costs only the staging, never the request's
        # correctness, so there is no failure mode worth letting through
        # uncaught.
        #
        # type(exc).__name__, not str(exc) or exc itself: warning_once's
        # lru_cache keys on (msg, *args), an exception instance hashes by
        # identity (would defeat the dedup and re-emit, traceback and all,
        # on every failing call), and str(exc) is not safe either -- some
        # torch allocator errors interpolate the requested/available byte
        # counts into the message, so a persistently exhausted allocator
        # would still re-emit on every call as those counts drift. The
        # exception's type name is stable across calls and still names the
        # failure mode. scope="process": rank-visible on multi-rank boxes,
        # unlike the attach-time info_once below (which stays local-scope).
        logger.warning_once(
            "PLE mmap: pinned H2D staging failed (%s); falling "
            "back to the pageable H2D path and disabling pinned staging "
            "for this layer for the rest of the run",
            type(exc).__name__,
            scope="process",
        )
        return cpu_tensor, False
    return staged, True


# --------------------------------------------------------------------------- #
# Placeholder that stands in for VocabParallelEmbedding.
# --------------------------------------------------------------------------- #
class MmapNgramEmbedding(nn.Module):
    """Duck-types the surface ``Qwen4ExpNGramEmbedding``/``Qwen4ExpPLELayer``
    read off ``self.ngram_embedding``: ``org_vocab_size``, ``embedding_dim``,
    ``weight_scale``, and ``__call__``. No ``.weight``/``.shard_indices`` —
    the env-gated ``load_weights`` branch intercepts shard tensors before the
    stock code would ever read those.

    ``table`` is ``None`` until the top-level model's ``load_weights``
    attaches one (never from this class's own ``load_weights`` — see
    :func:`build_tables`). While unset, ``forward``'s behavior depends on
    whether a real (non-dummy) load ever streamed weights through this
    module: ``--load-format dummy`` profiling never calls ``load_weights``
    at all, so ``weights_streamed`` stays False and zeros in the
    placeholder's still-default fp8 dtype are the correct, intentional
    stand-in — harmless regardless of which dtype eventually attaches, since
    every :class:`_PleDtype` either dequantizes a zero to zero or passes it
    through unscaled. A real load that streamed weights but never got a
    table attached (build_tables didn't run, or raised and was swallowed
    somewhere) is a bug, and must raise loudly rather than silently serve
    zeros as if they were real embeddings (invariant 4: fail closed, never
    serve garbage).
    """

    # Declared so static type-checkers resolve this to Tensor instead of
    # falling back to nn.Module.__getattr__'s Tensor | Module return type;
    # the buffer itself is registered dynamically in __init__.
    weight_scale: torch.Tensor

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.org_vocab_size = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.torch_dtype: torch.dtype = torch.float8_e4m3fn
        self.table: MmapPleTable | None = None
        self.weight_scale_loaded = False
        self.weights_streamed = False
        # Snapshotted from VLLM_PLE_MMAP_PINNED AND torch.cuda.is_available()
        # once a real table attaches (_attach_table); stays False here so a
        # table-unset forward (the zero-fill path) never engages it. Folding
        # is_available() in here too keeps forward's gate to the cheap
        # device-type check instead of re-querying CUDA every call.
        self.pinned = False
        # (total_ms, ids_d2h_wait_ms, gather_ms, h2d_ms) for each real
        # input-preparation call. The IDs wait includes queued accelerator
        # dependencies.
        self._prep_timings_ms: list[tuple[float, float, float, float]] = []
        self._prep_rows_since_log = 0
        self._prep_pinned_since_log = 0
        self._prep_last_log = time.monotonic()
        self.register_buffer(
            "weight_scale",
            torch.tensor(1.0, dtype=torch.bfloat16),
            persistent=False,
        )

    def gather_into(self, ids: torch.Tensor, destination: torch.Tensor) -> None:
        """Gather this table's rows for ``ids`` directly into ``destination``.

        ``destination`` is the caller's stable, module-owned buffer slice
        (``Qwen4ExpNGramEmbedding._mmap_staging`` under V2 staging): every
        gathered byte lands there directly, on the current stream — no
        transient GPU result is allocated and no ``output.copy_``
        indirection sits between the H2D copy and the real destination.

        Args:
            ids: row ids, any shape, on the same device as ``destination``.
            destination: contiguous tensor shaped ``(*ids.shape,
                embedding_dim)``, on ``ids.device``, dtyped either this
                table's attached dtype or (no table) this placeholder's
                fallback dtype.

        Raises:
            ValueError: ``destination`` fails device/dtype/shape/contiguity
                validation, or (table attached) the table's row_bytes
                disagree with ``embedding_dim * itemsize``.
            RuntimeError: no table is attached and a real (non-dummy) load
                already streamed weights without ``build_tables`` ever
                attaching one.
        """
        expected_shape = (*ids.shape, self.embedding_dim)
        if destination.device != ids.device:
            raise ValueError(
                f"PLE mmap: destination device {destination.device} != "
                f"ids device {ids.device}"
            )
        if tuple(destination.shape) != expected_shape:
            raise ValueError(
                f"PLE mmap: destination shape {tuple(destination.shape)} != "
                f"expected {expected_shape}"
            )
        if not destination.is_contiguous():
            raise ValueError("PLE mmap: destination must be contiguous")

        table = self.table
        if table is None:
            if self.weights_streamed:
                raise RuntimeError(
                    "PLE mmap table not initialized — load_weights ran but "
                    "build_tables did not"
                )
            if destination.dtype != self.torch_dtype:
                raise ValueError(
                    f"PLE mmap: destination dtype {destination.dtype} != "
                    f"placeholder dtype {self.torch_dtype}"
                )
            # No table and this is a dummy load: zero the destination in
            # place rather than gathering — there is nothing to gather from.
            destination.zero_()
            return
        if destination.dtype != table.torch_dtype:
            raise ValueError(
                f"PLE mmap: destination dtype {destination.dtype} != "
                f"table dtype {table.torch_dtype}"
            )

        ids_d2h_wait_t = time.monotonic()
        ids_np = ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)
        # The blocking transfer also waits for dependency-ordered accelerator
        # work queued before it. This interval is not pure copy latency.
        gather_t = time.monotonic()
        ids_d2h_wait_ms = (gather_t - ids_d2h_wait_t) * 1000.0
        rows = table.gather(ids_np)  # uint8 [N, row_bytes], fresh & writable
        gather_ms = (time.monotonic() - gather_t) * 1000.0
        itemsize = table.itemsize
        if table.row_bytes != self.embedding_dim * itemsize:
            raise ValueError(
                f"PLE mmap: table row_bytes={table.row_bytes} does not "
                f"match embedding_dim={self.embedding_dim} * "
                f"itemsize={itemsize}"
            )
        h2d_t = time.monotonic()
        out = torch.from_numpy(rows).view(table.torch_dtype)
        pinned_engaged = False
        # torch.cuda.is_available() is already folded into self.pinned at
        # attach time (_attach_table) -- only the per-call device-type check
        # runs inside this timed window.
        if self.pinned and ids.device.type == "cuda":
            out, pinned_engaged = _stage_pinned(out)
            # A persistently exhausted allocator (e.g. cudaHostAlloc) would
            # otherwise retry -- and re-raise, re-catch, and re-build a
            # fallback tensor around -- every single forward, a silent
            # per-step tax on the ITL path. Resyncing to pinned_engaged is a
            # no-op on success and latches off on failure, so a call that
            # will only fail again is paid for at most once (same precedent
            # as the readahead pre-pass's per-fd latch above).
            self.pinned = pinned_engaged
        # Two arms reach this line. Pinned-staged (above): `out` is already
        # pinned host memory, so non_blocking=True is a genuine async H2D.
        # Pageable (the default): `out` is still gather()'s pageable numpy
        # array, so non_blocking=True has no effect and the copy is
        # effectively synchronous regardless of the flag passed here.
        # copy_ writes directly into the caller's stable destination on the
        # current stream -- this IS the H2D copy, not a second indirection.
        destination.copy_(out.reshape(destination.shape), non_blocking=True)
        h2d_ms = (time.monotonic() - h2d_t) * 1000.0
        self._record_input_prep_timing(
            ids_np.size,
            ids_d2h_wait_ms,
            gather_ms,
            h2d_ms,
            pinned=pinned_engaged,
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Allocate a fresh destination and delegate to :meth:`gather_into`.

        Not the active V2 model path (which gathers straight into a stable,
        pre-allocated buffer via ``prepare_mmap_rows`` before forward runs);
        kept for direct unit tests and any non-V2 direct caller.
        """
        dtype = self.table.torch_dtype if self.table is not None else self.torch_dtype
        destination = torch.empty(
            (*ids.shape, self.embedding_dim), dtype=dtype, device=ids.device
        )
        self.gather_into(ids, destination)
        return destination

    def _record_input_prep_timing(
        self,
        rows: int,
        ids_d2h_wait_ms: float,
        gather_ms: float,
        h2d_ms: float,
        *,
        pinned: bool,
    ) -> None:
        """Log rate-limited host blocking times for mmap input preparation.

        ``ids_d2h_wait_ms`` includes the IDs copy and accelerator work queued
        before its blocking D2H boundary. It is not pure transfer latency.
        Component fields describe the samples selected by total p50/p99 time.
        """
        total_ms = ids_d2h_wait_ms + gather_ms + h2d_ms
        self._prep_timings_ms.append((total_ms, ids_d2h_wait_ms, gather_ms, h2d_ms))
        self._prep_rows_since_log += rows
        if pinned:
            self._prep_pinned_since_log += 1
        now = time.monotonic()
        if now - self._prep_last_log < _LOG_INTERVAL_S:
            return
        timings = sorted(self._prep_timings_ms)
        n = len(timings)
        p50_idx = max(0, math.ceil(n * 0.50) - 1)
        p99_idx = max(0, math.ceil(n * 0.99) - 1)
        p50_total, p50_ids_wait, p50_gather, p50_h2d = (
            timings[p50_idx] if timings else (0.0, 0.0, 0.0, 0.0)
        )
        p99_total, p99_ids_wait, p99_gather, p99_h2d = (
            timings[p99_idx] if timings else (0.0, 0.0, 0.0, 0.0)
        )
        logger.info(
            "PLE mmap input prep: prep_rows=%d n=%d p50_ms=%.2f "
            "p50_ids_d2h_wait_ms=%.2f p50_gather_ms=%.2f "
            "p50_h2d_call_ms=%.2f prep_p99_ms=%.2f "
            "p99_ids_d2h_wait_ms=%.2f p99_gather_ms=%.2f "
            "p99_h2d_call_ms=%.2f pinned=%d/%d",
            self._prep_rows_since_log,
            n,
            p50_total,
            p50_ids_wait,
            p50_gather,
            p50_h2d,
            p99_total,
            p99_ids_wait,
            p99_gather,
            p99_h2d,
            self._prep_pinned_since_log,
            n,
        )
        self._prep_timings_ms.clear()
        self._prep_rows_since_log = 0
        self._prep_pinned_since_log = 0
        self._prep_last_log = now


def set_weight_scale(
    embedding: MmapNgramEmbedding, weight: torch.Tensor, device: torch.device
) -> None:
    """Register the checkpoint's global scale on the placeholder.

    FP8 (and other requires_scale) dtypes only — an unquantized dtype like
    BF16 has no scale on disk and never routes through this function.

    Called from ``Qwen4ExpNGramEmbedding.load_weights`` as it intercepts
    ``ngram_embedding.weight_scale`` from the streamed weight iterator, so
    ``device`` should match wherever the rest of the module already lives
    (e.g. an existing buffer's device) rather than being hardcoded — this is
    what lets seam tests build everything on CPU with no GPU present.
    """
    embedding.register_buffer(
        "weight_scale", weight.detach().to(device=device), persistent=False
    )
    embedding.weight_scale_loaded = True


# --------------------------------------------------------------------------- #
# Startup guard: mmap external staging is Model Runner V2-only.
# --------------------------------------------------------------------------- #
def check_cudagraph_safety(vllm_config: VllmConfig) -> None:
    """Raise unless Model Runner V2 is active.

    V1 no longer prepares the PLE query/context inputs after #53896 removed
    its host-side hash-and-gather path, so mmap has no working V1 fallback:
    V1 must be refused at construction, before weights load.

    V2's ``Qwen4ExpModelState`` moves the hash-and-gather entirely out of
    model forward and into ``prepare_inputs``/``prepare_dummy_inputs``,
    before the compiled/captured forward runs; captured code only reads a
    stable, pre-populated buffer. This makes every V2 cudagraph mode safe,
    including the FULL-containing ones — V2 model state initializes every
    mmap module's staging buffer before capture/replay, and a mmap forward
    without initialized staging raises rather than falling back to host
    work.

    Raises:
        RuntimeError: Model Runner V2 is not active.
    """
    if not vllm_config.use_v2_model_runner:
        raise RuntimeError(
            "VLLM_PLE_MMAP=1 requires Model Runner V2: V1 no longer "
            "prepares the PLE query/context inputs after #53896 removed "
            "its host-side hash-and-gather path, so mmap has no working V1 "
            "fallback. Set VLLM_USE_V2_MODEL_RUNNER=1."
        )


# --------------------------------------------------------------------------- #
# Table directory resolution.
# --------------------------------------------------------------------------- #
def resolve_model_path(model_config: ModelConfig) -> str:
    """Resolve ``model_config`` to a local directory holding the checkpoint.

    Mirrors ``DefaultModelLoader._prepare_weights``, whose resolved local
    folder is a local variable never stored on any config object: verbatim
    if ``model_weights``/``model`` is already an existing directory, else an
    OFFLINE ``snapshot_download`` (never treat a repo id as a raw path).
    """
    path = model_config.model_weights or model_config.model
    if os.path.isdir(path):
        return path
    from vllm.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(
        repo_id=model_config.model,
        revision=model_config.revision,
        allow_patterns=["*.safetensors"],
        local_files_only=True,
    )


def _extract_layer_idx(layer_name: str) -> int:
    match = _LAYER_IDX_RE.search(layer_name)
    if not match:
        raise RuntimeError(
            f"PLE mmap: cannot find a decoder layer index in {layer_name!r}"
        )
    return int(match.group(1))


def _validate_layer_shards(
    layer_shards: _LayerShards, head_dim: int, layer_idx: int, model_path: str
) -> tuple[str, int, int, str] | None:
    """Shared fail-closed checks between :func:`validate_shards_for` (cheap,
    construction-time) and :func:`_attach_table` (authoritative, attach-time)
    — the same class of validation runs at both points, just at
    different times relative to the checkpoint's streamed load.

    Returns:
        The layer's validated ``scale_entry``, or ``None`` when the
        discovered dtype's descriptor has ``requires_scale=False``.
    """
    if layer_shards.cols != head_dim:
        raise RuntimeError(
            f"PLE mmap: layer {layer_idx} shard width {layer_shards.cols} "
            f"!= head_dim {head_dim}"
        )
    desc = _PLE_DTYPES.get(layer_shards.dtype_str)
    if desc is None:
        raise RuntimeError(
            f"PLE mmap: layer {layer_idx} shards have unsupported dtype "
            f"{layer_shards.dtype_str!r}; only {sorted(_PLE_DTYPES)} "
            "are supported (F8_E5M2 is refused: is_fp8() does not "
            "recognize it, so dequant would silently never fire)"
        )
    if not desc.requires_scale:
        if layer_shards.scale_entry is not None:
            raise RuntimeError(
                f"PLE mmap: layer {layer_idx} is {layer_shards.dtype_str} "
                "(unquantized, no scale required) but has an "
                f"ngram_embedding.weight_scale on disk under {model_path} — "
                "drop the stray scale or export shards in a dtype that "
                "requires one"
            )
        return None
    if layer_shards.scale_entry is None:
        raise RuntimeError(
            f"PLE mmap: layer {layer_idx} has FP8 shards but no "
            f"ngram_embedding.weight_scale under {model_path}"
        )
    _scale_path, _scale_offset, scale_nbytes, scale_dtype_str = layer_shards.scale_entry
    scale_torch_dtype = _SCALE_TORCH_DTYPES.get(scale_dtype_str)
    if scale_torch_dtype is not None:
        expected_nbytes = get_dtype_size(scale_torch_dtype)
        if scale_nbytes != expected_nbytes:
            raise RuntimeError(
                f"PLE mmap: layer {layer_idx} weight_scale is {scale_nbytes} "
                f"bytes, expected {expected_nbytes} for a single "
                f"{scale_dtype_str} scalar — per-channel PLE scales are "
                "unsupported; export a single global scale for this layer"
            )
    return layer_shards.scale_entry


def _validate_shard_placement(
    layer_shards: _LayerShards,
    org_vocab_size: int,
    split_ngram_parts: int,
    layer_idx: int,
    model_path: str,
) -> None:
    """Shared fail-closed shard cardinality/placement checks between
    :func:`preflight_reload_check` (before approval/mutation) and
    :func:`_attach_table` (authoritative, attach-time).

    Verbatim shard-placement math from
    ``Qwen4ExpNGramEmbedding.load_weights`` (nvidia/ple_layer.py) --
    discovery and gather must stay two self-consistent halves of the same
    mapping. Dtype/width/scale are `_validate_layer_shards`'s job, not
    this function's: a reload can match dtype and width exactly while
    still carrying the wrong number of shards, or shards holding the
    wrong row count for their index -- e.g. a checkpoint exported with a
    different ``split_ngram_parts`` than the one already staged.

    Raises:
        RuntimeError: an expected shard index is missing, a present shard
            index exceeds ``split_ngram_parts``, or a present shard's row
            count does not match what its index implies.
    """
    shard_size = (org_vocab_size + split_ngram_parts - 1) // split_ngram_parts
    num_expected_shards = min(
        split_ngram_parts,
        (org_vocab_size + shard_size - 1) // shard_size if shard_size else 0,
    )
    missing = [i for i in range(num_expected_shards) if i not in layer_shards.shards]
    if missing:
        raise RuntimeError(
            f"PLE mmap: layer {layer_idx} missing shard(s) {missing} of "
            f"{num_expected_shards} under {model_path}"
        )
    for shard_index, (_path, _offset, rows) in layer_shards.shards.items():
        if shard_index >= split_ngram_parts:
            raise RuntimeError(
                f"PLE mmap: layer {layer_idx} shard {shard_index} exceeds "
                f"split_ngram_parts={split_ngram_parts}"
            )
        checkpoint_start = shard_index * shard_size
        expected_rows = max(0, min(shard_size, org_vocab_size - checkpoint_start))
        if rows != expected_rows:
            raise RuntimeError(
                f"PLE mmap: layer {layer_idx} shard {shard_index} has "
                f"{rows} rows, expected {expected_rows}"
            )


# --------------------------------------------------------------------------- #
# Construction-time validation: cheap, header-only checks run from
# Qwen4ExpNGramEmbedding.__init__, before the ~78 GiB backbone streams.
# --------------------------------------------------------------------------- #
def validate_shards_for(
    model_config: ModelConfig, layer_name: str, head_dim: int
) -> torch.dtype | None:
    """Refuse a bad checkpoint at construction time, not after the load.

    Header-only checks (path resolution, shard presence, dtype, width,
    weight_scale existence — the same class of fail-closed validation
    :func:`_attach_table` performs, just runnable before any weight
    streams). Row-count-per-shard and the streamed-vs-header scale
    cross-check stay exclusively in :func:`_attach_table`: those need
    the checkpoint's declared vocab size and the
    weights that only arrive during the real streamed load.

    Tolerates an unresolvable model path (e.g. a bare repo id with no local
    snapshot yet — common for ``--load-format dummy``/test construction):
    logs and returns rather than raising, since :func:`build_tables` still
    fail-closes at the real load if the checkpoint is genuinely broken, so
    skipping here masks nothing.

    Returns:
        The layer's validated checkpoint row dtype when the model path
        resolves and every header check passes, or ``None`` when the model
        path could not be resolved (deferred to load time). The caller
        (``Qwen4ExpNGramEmbedding.__init__``) seeds the placeholder's
        ``torch_dtype`` from this so a dummy load — which never streams
        weights and so never reaches :func:`_attach_table` — allocates its
        V2 staging buffer (:meth:`Qwen4ExpNGramEmbedding.initialize_mmap_staging`)
        at the checkpoint's real dtype instead of the placeholder's stale
        FP8 default.

    Raises:
        RuntimeError: the model path resolves but shards are missing,
            wrong-dtype, wrong-width, or scale-less.
    """
    try:
        model_path = resolve_model_path(model_config)
    except Exception:
        logger.warning(
            "PLE mmap: %s: cannot resolve model path to pre-validate shards "
            "at construction time; deferring to load time",
            layer_name,
        )
        return None
    layer_idx = _extract_layer_idx(layer_name)
    layer_shards = discover_shards(model_path).get(layer_idx)
    if layer_shards is None:
        raise RuntimeError(
            f"PLE mmap: no shard tensors for layer {layer_idx} "
            f"({layer_name!r}) under {model_path}"
        )
    _validate_layer_shards(layer_shards, head_dim, layer_idx, model_path)
    # _validate_layer_shards already refused an unrecognized dtype_str, so
    # this lookup cannot miss.
    return _PLE_DTYPES[layer_shards.dtype_str].torch_dtype


# --------------------------------------------------------------------------- #
# Reload preflight: called from the top-level model's load_weights, before
# AutoWeightsLoader is even constructed.
# --------------------------------------------------------------------------- #
def _resolve_reload_shards_path(weights_path: str) -> str | None:
    """Best-effort local resolution of a proposed reload's checkpoint path.

    Unlike :func:`resolve_model_path`, this never contacts the network: a
    reload preflight must fail closed on an unresolvable path rather than
    block on (or trigger) a download. Returns ``None`` when ``weights_path``
    is not already a local directory and no offline-cached snapshot exists.
    """
    if os.path.isdir(weights_path):
        return weights_path
    from vllm.transformers_utils.repo_utils import hf_api

    try:
        return hf_api().snapshot_download(
            repo_id=weights_path,
            revision=None,
            allow_patterns=["*.safetensors"],
            local_files_only=True,
        )
    except Exception:
        return None


def preflight_reload_check(
    compilation_config: CompilationConfig,
    *,
    weights_path: str | None = None,
    is_checkpoint_format: bool = True,
    has_weights_iterator: bool = False,
) -> None:
    """Fail closed, before any weights stream, on an unsafe PLE mmap reload.

    Called from the top-level model's ``load_weights``, before
    ``AutoWeightsLoader`` is constructed or ``weights`` is advanced: waiting
    for ``Qwen4ExpNGramEmbedding.load_weights``'s own guard would let
    ``AutoWeightsLoader`` mutate earlier parameters first, since it walks
    the checkpoint in stream order (that child-level guard stays in place
    as defense in depth). Also the sole gate for the runtime "kernel
    format" reload (``is_checkpoint_format=False``), which copies straight
    into named parameters and never calls ``load_weights`` at all.

    Args:
        compilation_config: checked layer by layer via
            ``static_forward_context``.
        weights_path: the reload's proposed checkpoint path/repo id, or
            ``None`` for a caller-supplied weights iterator with no path to
            prove compatibility against.
        is_checkpoint_format: ``False`` for pre-repacked "kernel format"
            weights, which ``discover_shards``' safetensors parsing cannot
            speak to.
        has_weights_iterator: whether the caller ALSO supplied a weights
            iterator alongside ``weights_path``. This function can only
            validate the checkpoint AT ``weights_path``; if the actual load
            instead streams from the iterator, a "validated" path proves
            nothing about what will load. Rejected outright as ambiguous
            for a model with a staged-but-tableless PLE layer.

    Raises:
        RuntimeError: the ambiguous iterator+path combination above; a PLE
            layer already has a table attached (reloading onto a live
            module would mix this reload's rows with the attached
            checkpoint's scale); a layer has staging initialized but no
            table and this reload cannot prove row-layout compatibility
            with a resolvable checkpoint; the checkpoint's on-disk shards
            otherwise fail :func:`_validate_layer_shards`; or the shards
            fail :func:`_validate_shard_placement` (missing/extra shard
            indices, or a per-shard row count that disagrees with the
            already-staged layer's ``org_vocab_size``/``split_ngram_parts``
            even though dtype and width matched).
    """
    if weights_path is not None and has_weights_iterator:
        # A dedicated pre-pass, run before any other check below: this
        # ambiguity must reject before approval, before
        # `model_config.model` is repointed at `weights_path`, before the
        # iterator is advanced, before `named_parameters` is walked, and
        # before a model loader is even constructed for the (unused)
        # `weights_path` -- all of which the runner's `reload_weights`
        # would otherwise do first if this ran interleaved with the
        # ordinary per-layer loop below.
        for layer_name, layer in compilation_config.static_forward_context.items():
            ple_embedding_module = getattr(layer, "ple_embedding", None)
            if ple_embedding_module is None:
                continue
            embedding = getattr(ple_embedding_module, "ngram_embedding", None)
            if not isinstance(embedding, MmapNgramEmbedding):
                continue
            if embedding.table is not None:
                # A different, unrelated rejection ("already has a table
                # attached") owns this layer's state; the loop below raises
                # it.
                continue
            staging = getattr(ple_embedding_module, "_mmap_staging", None)
            if staging is None:
                continue
            raise RuntimeError(
                f"PLE mmap: {layer_name!r} has mmap staging initialized "
                "but no table attached, and this reload supplied BOTH a "
                f"weights_path ({weights_path!r}) and a weights iterator. "
                "This preflight can only validate the checkpoint sitting "
                "at weights_path -- the actual load streams from whichever "
                "the caller passes to model.load_weights, which is not "
                "provably the same checkpoint. Supply exactly one of "
                "weights_path or a weights iterator for a model with an "
                "unattached, staged PLE layer."
            )

    for layer_name, layer in compilation_config.static_forward_context.items():
        ple_embedding_module = getattr(layer, "ple_embedding", None)
        if ple_embedding_module is None:
            continue
        embedding = getattr(ple_embedding_module, "ngram_embedding", None)
        if not isinstance(embedding, MmapNgramEmbedding):
            continue
        if embedding.table is not None:
            raise RuntimeError(
                f"PLE mmap: {layer_name!r} already has a table "
                "attached from a previous load; calling load_weights again "
                "on the same live module is unsupported — it would mix "
                "this reload's rows with the already-attached checkpoint's "
                "scale. Restart the seat to load different weights."
            )
        staging = getattr(ple_embedding_module, "_mmap_staging", None)
        if staging is None:
            # V2 model state has not (yet) initialized staging for this
            # layer -- nothing attached, nothing staged, nothing to prove
            # compatible with. Not this function's problem: the initial
            # load path reaches here before model state exists at all.
            continue
        if weights_path is None or not is_checkpoint_format:
            raise RuntimeError(
                f"PLE mmap: {layer_name!r} has mmap staging initialized "
                "but no table attached, and this reload cannot prove "
                "row-layout compatibility without a resolvable "
                "checkpoint-format weights_path (got weights_path="
                f"{weights_path!r}, is_checkpoint_format="
                f"{is_checkpoint_format!r}). Only a checkpoint-format "
                "reload from a resolvable path can be proven safe to "
                "attach onto already-initialized staging."
            )
        layer_idx = getattr(layer, "layer_idx", None)
        if layer_idx is None:
            raise RuntimeError(
                f"PLE mmap: {layer_name!r} has no layer_idx to resolve its "
                "shards against"
            )
        resolved_path = _resolve_reload_shards_path(weights_path)
        if resolved_path is None:
            raise RuntimeError(
                f"PLE mmap: {layer_name!r} has mmap staging initialized "
                "but no table attached, and the proposed reload path "
                f"{weights_path!r} does not resolve to a local checkpoint "
                "directory (and no offline-cached snapshot exists) -- "
                "cannot prove row-layout compatibility before attaching"
            )
        layer_shards = discover_shards(resolved_path).get(layer_idx)
        if layer_shards is None:
            raise RuntimeError(
                f"PLE mmap: {layer_name!r} (layer {layer_idx}) has no "
                f"shard tensors under the proposed reload path "
                f"{resolved_path!r}"
            )
        desc = _PLE_DTYPES.get(layer_shards.dtype_str)
        incoming_dtype = desc.torch_dtype if desc is not None else None
        current_dtype = staging.dtype
        current_width = staging.shape[-1]
        if incoming_dtype != current_dtype or layer_shards.cols != current_width:
            raise RuntimeError(
                f"PLE mmap: {layer_name!r} reload rejected — incoming "
                f"checkpoint layout (dtype={layer_shards.dtype_str}, "
                f"row_width={layer_shards.cols}) under {resolved_path!r} "
                "does not match the already-staged layout "
                f"(dtype={current_dtype}, row_width={current_width})"
            )
        # The dtype/width match above only proves this reload's SHAPE is
        # compatible with what is already staged -- it says nothing about
        # whether the checkpoint itself is a valid PLE export (e.g. an FP8
        # layer missing its weight_scale entirely, or one carrying a stray
        # scale it has no business having). Run the same header-only
        # fail-closed contract _attach_table applies at real attach time
        # before granting a same-layout approval. head_dim is
        # layer_shards.cols itself here (already proven == current_width
        # above), so this call adds the scale-presence/shape/dtype and
        # dtype-support checks without re-deriving the layout match just
        # performed -- and without changing the "reload rejected" message
        # above for a plain layout mismatch.
        _validate_layer_shards(
            layer_shards, layer_shards.cols, layer_idx, resolved_path
        )
        # Validate placement against the current layer, not the incoming
        # checkpoint's declaration.
        _validate_shard_placement(
            layer_shards,
            embedding.org_vocab_size,
            ple_embedding_module.split_ngram_parts,
            layer_idx,
            resolved_path,
        )


# Attribute name for the transaction-scoped reload approval token stashed
# directly on a model instance. See `_ReloadApproval` for the mechanism.
# Private/mangled-looking on purpose: nothing outside this module should
# read, set, or rely on its presence.
_RELOAD_APPROVAL_ATTR = "_ple_mmap_reload_approval"

# `_ReloadApproval.role`: which model instance owns calling `build_tables`
# at its own `load_weights` boundary. `_ROLE_ROOT` is the outermost
# `load_weights` call for this transaction -- a standalone model, or the
# `Qwen4ExpForConditionalGeneration` wrapper -- and builds exactly once,
# after its own (possibly recursive) load fully returns. `_ROLE_DEFER` is a
# nested causal LM granted a copy of the outer wrapper's token: it may be
# invoked more than once per transaction (AutoWeightsLoader's grouping is
# not a full partition), and every one of those calls must defer building
# to the root, since only the root has seen every group.
_ROLE_ROOT = "root"
_ROLE_DEFER = "nested"


@dataclass(frozen=True)
class _ReloadApproval:
    """One reload transaction's token, stashed on a model instance under
    `_RELOAD_APPROVAL_ATTR`.

    Mechanism: `AutoWeightsLoader`'s `load_weights` entry point has no
    `weights_path` argument, so it cannot itself run the path-aware
    `preflight_reload_check` a caller who DOES have a path (the runner's
    `preflight_reload_weights`) already ran. A token bridges the two: the
    path-aware caller mints one via `approve_reload`; `load_weights` then
    validates against it via `validate_reload_approval` instead of
    re-deriving -- and failing -- the same proof pathlessly. A TRUE initial
    load has no path-aware caller ahead of it at all, so
    `validate_reload_approval` mints its own token the first time it finds
    none, once the ordinary pathless preflight proves this really is a
    fresh load; that call then OWNS the token (see its docstring).

    Not single-use: `AutoWeightsLoader._groupby_prefix` groups an incoming
    weight stream by CONSECUTIVE runs of a shared top-level prefix, not a
    full partition, so a checkpoint that interleaves a mapped module's
    weights with an unrelated group's calls that module's `load_weights`
    MORE THAN ONCE per transaction -- every call must find the same token
    still there. Only `clear_reload_approval` ever removes it, once, when
    the transaction that owns it ends.

    For a multimodal model, both the outer wrapper's
    (`Qwen4ExpForConditionalGeneration`) and the nested causal LM's
    (`Qwen4ExpForCausalLM`) copies carry the SAME `transaction` sentinel --
    proof they share one transaction rather than two coincidentally-present
    approvals -- but each carries its OWN `role`: `_ROLE_ROOT` on the
    wrapper, `_ROLE_DEFER` on the nested causal LM, so the two copies of
    one transaction disagree, by design, on who builds tables.
    `compilation_config_id` binds the token to the exact
    `CompilationConfig` validated when it was minted; a token bound to a
    different config is discarded rather than honored.
    """

    transaction: object
    compilation_config_id: int
    weights_path: str | None
    is_checkpoint_format: bool
    role: str


def _grant_reload_transaction(
    model: object,
    compilation_config: CompilationConfig,
    *,
    weights_path: str | None,
    is_checkpoint_format: bool,
    inner_model_attr: str | None,
) -> None:
    """Mint one transaction and stash a `_ROLE_ROOT` token on `model` (and,
    when `inner_model_attr` names a nested causal LM attribute, a
    `_ROLE_DEFER` token sharing the same `transaction` sentinel on that
    model too -- a no-op when the attribute is absent or aliases `model`).
    See `_ReloadApproval` for why the two copies carry different roles."""
    transaction = object()
    setattr(
        model,
        _RELOAD_APPROVAL_ATTR,
        _ReloadApproval(
            transaction=transaction,
            compilation_config_id=id(compilation_config),
            weights_path=weights_path,
            is_checkpoint_format=is_checkpoint_format,
            role=_ROLE_ROOT,
        ),
    )
    inner_model = getattr(model, inner_model_attr, None) if inner_model_attr else None
    if inner_model is not None and inner_model is not model:
        setattr(
            inner_model,
            _RELOAD_APPROVAL_ATTR,
            _ReloadApproval(
                transaction=transaction,
                compilation_config_id=id(compilation_config),
                weights_path=weights_path,
                is_checkpoint_format=is_checkpoint_format,
                role=_ROLE_DEFER,
            ),
        )


def approve_reload(
    model: object,
    compilation_config: CompilationConfig,
    *,
    weights_path: str | None = None,
    is_checkpoint_format: bool = True,
    has_weights_iterator: bool = False,
    inner_model_attr: str | None = None,
) -> None:
    """Path-aware preflight a reload; on success, grant it a transaction
    token (see `_ReloadApproval`) for the matching `load_weights` call(s)
    to validate against. Raises and grants nothing if the reload is unsafe.

    Args:
        inner_model_attr: name of `model`'s nested causal LM attribute
            (``"language_model"`` for `Qwen4ExpForConditionalGeneration`),
            granted a copy of the same token. `None` for a standalone
            `Qwen4ExpForCausalLM`, which has no nested model to share with.
    """
    preflight_reload_check(
        compilation_config,
        weights_path=weights_path,
        is_checkpoint_format=is_checkpoint_format,
        has_weights_iterator=has_weights_iterator,
    )
    _grant_reload_transaction(
        model,
        compilation_config,
        weights_path=weights_path,
        is_checkpoint_format=is_checkpoint_format,
        inner_model_attr=inner_model_attr,
    )


def validate_reload_approval(
    model: object,
    compilation_config: CompilationConfig,
    *,
    inner_model_attr: str | None = None,
) -> tuple[bool, bool]:
    """Enter or validate `model`'s reload transaction; fail closed if unsafe.

    A pre-existing token bound to this `compilation_config` -- runner-
    granted via `approve_reload`, or an outer wrapper's token already
    copied onto this nested model -- validates without being consumed (see
    `_ReloadApproval`). Otherwise this call has no path-aware preflight
    ahead of it (a true initial load): the ordinary pathless
    `preflight_reload_check` must pass, and only then does this call mint
    its own fresh transaction via `_grant_reload_transaction`, so a same-
    transaction repeat invocation (interleaved `AutoWeightsLoader` groups)
    finds it on the next call. A freshly minted transaction always grants
    `model` itself the `_ROLE_ROOT` token (see `_grant_reload_transaction`),
    so minting implies both return values are True.

    Args:
        inner_model_attr: forwarded to `_grant_reload_transaction` when
            this call mints a fresh transaction; see `approve_reload`.

    Returns:
        `(owns_transaction, should_build_tables)`.

        `owns_transaction` is True when this call minted a fresh
        transaction it now OWNS and must itself clear via
        `clear_reload_approval` once its own `load_weights` body finishes,
        success or failure. False when it validated a token minted
        elsewhere, which it must leave untouched -- that transaction is not
        this call's to end.

        `should_build_tables` is True when `model`'s own token carries
        `_ROLE_ROOT`: this call is the sole table-build owner and must call
        `build_tables` itself, once, at its own `load_weights` boundary
        (whether or not it also owns clearing the transaction -- a
        runner-approved standalone reload is root but does not own).
        False when `model`'s token carries `_ROLE_DEFER`: a nested causal
        LM invoked under an outer `Qwen4ExpForConditionalGeneration`
        transaction, which must defer every one of its (possibly repeated)
        table-build opportunities to the outer wrapper's own return.
    """
    token = getattr(model, _RELOAD_APPROVAL_ATTR, None)
    if token is not None:
        if isinstance(token, _ReloadApproval) and token.compilation_config_id == id(
            compilation_config
        ):
            return False, token.role == _ROLE_ROOT
        # Wrong shape, or bound to a different compilation_config: not
        # proof of anything about THIS call's reload.
        delattr(model, _RELOAD_APPROVAL_ATTR)
    preflight_reload_check(compilation_config)
    _grant_reload_transaction(
        model,
        compilation_config,
        weights_path=None,
        is_checkpoint_format=True,
        inner_model_attr=inner_model_attr,
    )
    return True, True


def clear_reload_approval(model: object, inner_model: object | None = None) -> None:
    """End a reload transaction: remove `model`'s token (and
    `inner_model`'s copy, if any). The only place a token is ever removed
    on the success path -- `validate_reload_approval` never deletes one it
    merely validated. Call once, from whichever call owns the transaction:
    the runner's `finally`, the worker's weight-transfer session cleanup,
    or a `load_weights` call that itself minted the token (see
    `validate_reload_approval`'s return value). Idempotent; `inner_model`
    of `None`/identical-to-`model` is a no-op (the standalone
    `Qwen4ExpForCausalLM` case, which never had a second token).
    """
    if hasattr(model, _RELOAD_APPROVAL_ATTR):
        delattr(model, _RELOAD_APPROVAL_ATTR)
    if (
        inner_model is not None
        and inner_model is not model
        and hasattr(inner_model, _RELOAD_APPROVAL_ATTR)
    ):
        delattr(inner_model, _RELOAD_APPROVAL_ATTR)


# --------------------------------------------------------------------------- #
# Table construction, invoked once from the top-level model's load_weights.
# --------------------------------------------------------------------------- #
def build_tables(
    model_config: ModelConfig, compilation_config: CompilationConfig
) -> None:
    """Build and bounded-prewarm the table for every enabled PLE layer.

    Called from ``Qwen4ExpForConditionalGeneration.load_weights`` and
    ``Qwen4ExpForCausalLM.load_weights``, but only by whichever call holds
    the ``_ROLE_ROOT`` token for its transaction (``should_build_tables``
    from ``validate_reload_approval``) — never from
    ``Qwen4ExpNGramEmbedding.load_weights``, which would stream the whole
    table mid-load during the tightest memory transient. A nested
    ``Qwen4ExpForCausalLM`` invoked under a ``Qwen4ExpForConditionalGeneration``
    transaction holds ``_ROLE_DEFER`` instead and never calls this function
    itself, however many times ``AutoWeightsLoader`` invokes its
    ``load_weights`` during one load. Only the outer wrapper builds tables,
    after every nested group has streamed.
    A standalone ``Qwen4ExpForCausalLM`` (no wrapper) always holds
    ``_ROLE_ROOT`` and calls this itself, at its own load boundary.

    Two costs are avoided on that redundant second call: ``model_path`` is
    resolved (cheap: a directory check or a local HF cache lookup) so every
    already-attached layer can still be verified against it, but the
    ``pending`` list (layers whose table is still ``None``) is computed
    BEFORE the expensive part — ``discover_shards``' header scan of every
    checkpoint file — and that scan, plus the whole attach loop, is skipped
    entirely once ``pending`` is empty. This also restores the "prewarm on
    the return path" guarantee by construction: ``_attach_table`` only ever
    runs for a layer that has never been attached.

    On the ConditionalGeneration path, the inner ``Qwen4ExpForCausalLM``
    call reaches this function (and its prewarm) BEFORE the outer
    ``AutoWeightsLoader.load_weights`` call has fully returned — i.e.
    mid-load, not after. This is safe because ``compute_prewarm_bound``
    re-reads ``MemAvailable`` at that instant rather than assuming any
    particular point in the load timeline.

    Raises:
        RuntimeError: a PLE layer has no matching shards on disk, a
            discovered shard fails validation (invariant 4: fail closed),
            or an already-attached layer's table was built from a
            DIFFERENT checkpoint than the one ``model_config`` resolves to
            now — reload_weights repointing ``model_config`` at a new
            checkpoint is unsupported; serving checkpoint A's mmap rows
            against checkpoint B's scale would silently corrupt output.
    """
    model_path = resolve_model_path(model_config)

    pending: list[tuple[str, Any, Any, MmapNgramEmbedding]] = []
    for layer_name, layer in compilation_config.static_forward_context.items():
        ple_embedding_module = getattr(layer, "ple_embedding", None)
        if ple_embedding_module is None:
            continue
        embedding = getattr(ple_embedding_module, "ngram_embedding", None)
        if not isinstance(embedding, MmapNgramEmbedding):
            continue
        if embedding.table is not None:
            if embedding.table.model_path != model_path:
                raise RuntimeError(
                    f"PLE mmap: layer {layer_name!r} already has a table "
                    f"built from {embedding.table.model_path!r}, but this "
                    f"load resolves to {model_path!r} — reloading weights "
                    "onto a different checkpoint is unsupported; restart "
                    "the seat"
                )
            continue
        pending.append((layer_name, layer, ple_embedding_module, embedding))
    if not pending:
        return

    shard_map = discover_shards(model_path)

    for layer_name, layer, ple_embedding_module, embedding in pending:
        layer_idx = layer.layer_idx
        layer_shards = shard_map.get(layer_idx)
        if layer_shards is None:
            raise RuntimeError(
                f"PLE mmap: no shard tensors for layer {layer_idx} "
                f"({layer_name!r}) under {model_path}"
            )
        _attach_table(
            embedding,
            layer_shards,
            split_ngram_parts=ple_embedding_module.split_ngram_parts,
            layer_idx=layer_idx,
            model_path=model_path,
        )


def _attach_table(
    embedding: MmapNgramEmbedding,
    layer_shards: _LayerShards,
    split_ngram_parts: int,
    layer_idx: int,
    model_path: str,
) -> None:
    scale_entry = _validate_layer_shards(
        layer_shards, embedding.embedding_dim, layer_idx, model_path
    )
    desc = _PLE_DTYPES[layer_shards.dtype_str]
    if desc.requires_scale:
        assert scale_entry is not None
        if not embedding.weight_scale_loaded:
            if embedding.weights_streamed:
                # Rows were streamed but the scale was not — a broken or
                # truncated weight iterator, not an unstreamed family; stay
                # fail-closed rather than guess at a header value.
                raise RuntimeError(
                    f"PLE mmap: layer {layer_idx} weight_scale was never loaded "
                    "from the checkpoint's streamed weights"
                )
            # This layer's ngram_embedding family was never streamed at all
            # (a loader topology that never routes PLE weights to this
            # worker) — fall back to a direct header read instead of failing
            # closed on a scale that had nothing to be lost from.
            header_scale = _read_scale(scale_entry)
            logger.warning(
                "PLE mmap: layer %d weight_scale falling back to a direct "
                "header read — this layer's ngram_embedding family was never "
                "streamed through the checkpoint loader",
                layer_idx,
            )
            # Same device the streamed path resolves to: the buffer being replaced.
            set_weight_scale(embedding, header_scale, embedding.weight_scale.device)
        else:
            # Cross-check the streamed-and-registered scale against an independent
            # direct read of the same tensor off disk: these are two self-consistent
            # halves, and a mismatch would mean the streamed weight iterator silently
            # renamed or skipped something.
            header_scale = _read_scale(scale_entry).float()
            streamed_scale = embedding.weight_scale.detach().to("cpu").float()
            if not torch.allclose(header_scale, streamed_scale, atol=1e-6):
                # .tolist()[:4], not .item(): a malformed (non-scalar) streamed
                # scale must not crash the diagnostic itself with an unrelated
                # "cannot be converted to Scalar" error.
                raise RuntimeError(
                    f"PLE mmap: layer {layer_idx} weight_scale mismatch between the "
                    f"streamed checkpoint ({streamed_scale.flatten().tolist()[:4]}) "
                    f"and the header-parsed value "
                    f"({header_scale.flatten().tolist()[:4]})"
                )
    # else: no-scale dtype (e.g. BF16). _validate_layer_shards already
    # refused a stray on-disk scale above, so validation is the sole
    # authority here — this branch deliberately ignores
    # weight_scale_loaded/weights_streamed. A real loader never calls
    # set_weight_scale for this table (there is nothing for it to stream),
    # so weight_scale_loaded=True on an unscaled embedding only ever
    # happens in a test double, and attaching must not chase that state.

    vocab = embedding.org_vocab_size
    _validate_shard_placement(
        layer_shards, vocab, split_ngram_parts, layer_idx, model_path
    )
    # Same formula _validate_shard_placement just checked every shard
    # against; MmapPleTable needs it too, to map a global row id back to
    # its owning shard and local offset (see the class docstring's
    # shard-mapping contract).
    shard_size = (vocab + split_ngram_parts - 1) // split_ngram_parts

    if embedding.table is not None:
        # Defensive: build_tables' own idempotency skip (table is not None
        # -> layer skipped) should make this unreachable in practice, but a
        # direct _attach_table re-entry must not leak the old ThreadPool.
        embedding.table.close()
        embedding.table = None

    row_bytes = layer_shards.cols * _itemsize(layer_shards.dtype_str)
    table: MmapPleTable | None = None
    try:
        table = MmapPleTable(
            layer_shards.shards,
            shard_size,
            row_bytes,
            desc.torch_dtype,
            workers=envs.VLLM_PLE_MMAP_WORKERS,
            chunk=envs.VLLM_PLE_MMAP_CHUNK,
            model_path=model_path,
            readahead=envs.VLLM_PLE_MMAP_READAHEAD,
            serial=envs.VLLM_PLE_MMAP_SERIAL,
        )
        embedding.torch_dtype = table.torch_dtype
        # Snapshotted once here, not read inside forward: forward currently
        # reads no envs at all, and vllm's envs.__getattr__ re-runs
        # os.getenv on every access. Combined with torch.cuda.is_available()
        # here too (also a per-call cost forward would otherwise re-pay):
        # on a CUDA-less box this is always False, so the gate stays dead
        # regardless of the env value -- the CPU-path guarantee forward's
        # device-type check alone no longer has to provide.
        embedding.pinned = envs.VLLM_PLE_MMAP_PINNED and torch.cuda.is_available()
        if embedding.pinned:
            # No layer_idx here: info_once dedups on (msg, *args), so a
            # per-layer argument would re-log for every PLE layer (same
            # rationale as the readahead pre-pass breadcrumb above).
            logger.info_once(
                "PLE mmap: input-prep H2D will use a pinned host "
                "buffer on CUDA (VLLM_PLE_MMAP_PINNED=1)"
            )
        table_bytes = table.rows_total * row_bytes

        if envs.VLLM_PLE_MMAP_PREWARM:
            bound = compute_prewarm_bound(table_bytes, _mem_available_bytes())
            read = table.prewarm(bound)
            logger.info(
                "PLE mmap: layer %d prewarm read %.2f GiB (budget %.2f GiB)",
                layer_idx,
                read / (1 << 30),
                bound / (1 << 30),
            )

        embedding.table = table
    except Exception:
        # Nothing between construction and the attach above may raise and
        # leak the table's memmaps, readahead fds, or thread pool — close
        # whatever construction already opened before this propagates.
        if table is not None:
            table.close()
        raise
    logger.info(
        "PLE mmap: layer %d attached, %d shards, %d rows x %d B "
        "(%.2f GiB on disk), %d workers",
        layer_idx,
        len(layer_shards.shards),
        table.rows_total,
        row_bytes,
        table_bytes / (1 << 30),
        table.workers,
    )


__all__ = [
    "MmapNgramEmbedding",
    "MmapPleTable",
    "build_tables",
    "check_cudagraph_safety",
    "compute_prewarm_bound",
    "discover_shards",
    "enabled",
    "parse_safetensors_header",
    "preflight_reload_check",
    "resolve_model_path",
    "set_weight_scale",
    "validate_shards_for",
]
