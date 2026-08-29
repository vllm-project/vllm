# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve the Qwen4Exp PLE n-gram table from NVMe via mmap.

Why: the n-gram (PLE) table is tens of GiB in FP8 and the stock path keeps it
resident (GPU, via ``VocabParallelEmbedding``). On a box where host and GPU
share one unified memory pool, that table cannot sit next to the rest of the
model. A token only ever touches a handful of rows out of the whole table, so
it can live on disk and be served through the page cache instead.

How, with ``VLLM_PLE_MMAP=1``:
  * ``Qwen4ExpNGramEmbedding.__init__`` swaps the GPU-resident embedding for
    :class:`MmapNgramEmbedding`, a placeholder whose ``forward`` gathers rows
    from :class:`MmapPleTable` (``np.memmap`` views over the checkpoint's
    safetensors shards, page-cache backed).
  * ``Qwen4ExpNGramEmbedding.load_weights`` drops the per-shard tensors on
    the floor (never materialized into a resident table) and keeps the
    checkpoint's global FP8 ``weight_scale`` as a buffer on the placeholder,
    which the untouched ``Qwen4ExpPLELayer._dequantize_embeddings`` already
    knows how to consume.
  * the WHOLE forward — trigram hashing plus the row gather — is wrapped in
    a custom op, ``vllm::qwen4_exp_ple_mmap_forward``, so it runs OUTSIDE
    CUDA graph capture and outside torch.compile tracing entirely
    (the stock hashing's ``.numel()``-derived slicing specializes
    vLLM's dynamic dims under Dynamo — ``ConstraintViolationError`` on
    ``query_start_loc.size()[0]`` — when only the gather was split out;
    widening the op boundary to cover the hashing too sidesteps tracing
    it at all). The op is
    listed in ``splitting_ops`` the same way the narrower gather-only op
    was.

This module is imported unconditionally at ``nvidia/ple_layer.py`` module
scope so the custom op registers at import time; every behavior above is
gated on :func:`enabled` at call time. With ``VLLM_PLE_MMAP`` unset, nothing
in this module is ever invoked and the stock classes are untouched.

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
from vllm.config.compilation import CompilationMode
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op, get_dtype_size, vllm_lib

if TYPE_CHECKING:
    from vllm.config import CompilationConfig, ModelConfig

logger = init_logger(__name__)

OP_NAME = "qwen4_exp_ple_mmap_forward"
QUALIFIED_OP_NAME = f"vllm::{OP_NAME}"

_FP8_DTYPES: dict[str, torch.dtype] = {
    "F8_E4M3": torch.float8_e4m3fn,
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

    Args:
        offsets: ascending int64 file offsets, one per row.
        row_bytes: length of the span each offset starts.

    Returns:
        ``(file_offset, length)`` pairs covering exactly the input spans.

    Only spans that abut are merged. Bridging a gap would fetch pages no row
    in this gather needs, and on a hash-scattered table that amplification
    re-creates the cold-read cost the readahead exists to remove.
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
    ) -> None:
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
        n_slots = max(shards) + 1
        self.mm: list[np.memmap | None] = [None] * n_slots
        self.rows_total = 0
        for idx, (path, offset, rows) in shards.items():
            self.mm[idx] = np.memmap(
                path, dtype=np.uint8, mode="r", offset=offset, shape=(rows, row_bytes)
            )
            self.rows_total += rows
        self._fds: dict[str, int] = {}
        self._shard_fds: list[int | None] = [None] * n_slots
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
        self.pool = ThreadPoolExecutor(max_workers=self.workers)
        self._pending = 0
        self._errors = 0
        self._bound_warned = False
        self._rows_since_log = 0
        # (elapsed_ms, populate_ms, copy_ms, coalesced runs) per gather.
        self._latencies_ms: list[tuple[float, float, float, int]] = []
        self._last_log = time.monotonic()
        self._closed = False

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

        Two passes: a numpy count-only pass decides whether the bound
        (``self.readahead``) is exceeded before anything is materialized, so
        a bound-skipped gather pays only that count, not the (fd, offset,
        length) Python-level run list the active path builds. Only once the
        count clears the bound does a second pass build that list and issue
        the ``posix_fadvise`` calls.

        Returns:
            The coalesced run count, reported even when it exceeds the bound
            and nothing was issued — a silent skip would be indistinguishable
            from readahead being off.
        """
        total_runs = 0
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

        runs: list[tuple[int, int, int]] = []
        for s, e in zip(seg_starts, seg_ends):
            si = int(shard[s])
            mm = self.mm[si]
            fd = self._shard_fds[si]
            if mm is None or fd is None:
                continue
            offsets = mm.offset + local[s:e] * self.row_bytes
            runs.extend(
                (fd, offset, length)
                for offset, length in _coalesce_runs(offsets, self.row_bytes)
            )

        for fd, offset, length in runs:
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
            if len(tasks) == 1:
                run(tasks[0])
            else:
                for _ in self.pool.map(run, tasks):
                    pass
        except Exception:
            self._errors += 1
            raise
        finally:
            # Snapshot before resetting: _record's log line (fired at most
            # once per _LOG_INTERVAL_S) needs the concurrency depth THIS
            # call actually ran at, not the always-zero post-reset value.
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
    ) -> None:
        self._latencies_ms.append((elapsed_ms, populate_ms, copy_ms, runs))
        self._rows_since_log += rows
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
            "pending=%d errors=%d",
            self._rows_since_log,
            p99,
            p99_populate,
            p99_copy,
            p99_runs,
            pending,
            self._errors,
        )
        self._latencies_ms.clear()
        self._rows_since_log = 0
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
        """
        if self._closed:
            return
        self._closed = True
        self.pool.shutdown(wait=False)
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
    at all, so ``weights_streamed`` stays False and zeros are the correct,
    intentional stand-in against the default unit ``weight_scale``. A real
    load that streamed weights but never got a table attached (build_tables
    didn't run, or raised and was swallowed somewhere) is a bug, and must
    raise loudly rather than silently serve zeros as if they were real
    embeddings (invariant 4: fail closed, never serve garbage).
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
        self.register_buffer(
            "weight_scale",
            torch.tensor(1.0, dtype=torch.bfloat16),
            persistent=False,
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        table = self.table
        if table is None:
            if self.weights_streamed:
                raise RuntimeError(
                    "PLE mmap table not initialized — load_weights ran but "
                    "build_tables did not"
                )
            return torch.zeros(
                (*ids.shape, self.embedding_dim),
                dtype=self.torch_dtype,
                device=ids.device,
            )
        ids_np = ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)
        rows = table.gather(ids_np)  # uint8 [N, row_bytes], fresh & writable
        itemsize = table.itemsize
        if table.row_bytes != self.embedding_dim * itemsize:
            raise ValueError(
                f"PLE mmap: table row_bytes={table.row_bytes} does not "
                f"match embedding_dim={self.embedding_dim} * "
                f"itemsize={itemsize}"
            )
        out = torch.from_numpy(rows).view(table.torch_dtype)
        # non_blocking=True has no effect here: `rows` (from table.gather)
        # is pageable host memory, not pinned, so this H2D copy is
        # effectively synchronous. Pinned staging (Phase-4 lever 5) is a
        # separate, not-yet-pulled lever, not something this line hides.
        out = out.to(ids.device, non_blocking=True)
        return out.reshape(*ids.shape, self.embedding_dim)


def set_weight_scale(
    embedding: MmapNgramEmbedding, weight: torch.Tensor, device: torch.device
) -> None:
    """Register the checkpoint's FP8 global scale on the placeholder.

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
# Startup guard: the whole-forward op (hashing + gather) must never run
# inside CUDA graph capture.
# --------------------------------------------------------------------------- #
def check_cudagraph_safety(compilation_config: CompilationConfig) -> None:
    """Raise if VLLM_PLE_MMAP=1 would run the hashing+gather forward inside
    a capture.

    Three independent checks, any of which alone would miss
    a real route into a capture:
      * FULL cudagraph modes capture decode outside the fx graph regardless
        of splitting_ops membership.
      * enforce-eager (mode != VLLM_COMPILE) does not fully suppress capture
        on this model and leaves splitting_ops empty.
      * an operator-supplied ``-cc.splitting_ops`` list, or an attn-fusion
        reset, can silently drop our op from the split set even under
        PIECEWISE + VLLM_COMPILE.

    Raises:
        RuntimeError: any of the above conditions holds.
    """
    if compilation_config.cudagraph_mode.has_full_cudagraphs():
        raise RuntimeError(
            "VLLM_PLE_MMAP=1 requires piecewise-only CUDA graphs (the "
            "hashing+gather forward cannot run inside a capture); "
            "got cudagraph_mode="
            f"{compilation_config.cudagraph_mode}. Pass "
            "-cc.cudagraph_mode=PIECEWISE."
        )
    if compilation_config.mode != CompilationMode.VLLM_COMPILE:
        raise RuntimeError(
            "VLLM_PLE_MMAP=1 requires compilation_config.mode="
            "CompilationMode.VLLM_COMPILE; enforce-eager does not fully "
            f"suppress CUDA graph capture on this model. Got mode="
            f"{compilation_config.mode}."
        )
    if QUALIFIED_OP_NAME not in (compilation_config.splitting_ops or []):
        raise RuntimeError(
            f"VLLM_PLE_MMAP=1 requires {QUALIFIED_OP_NAME!r} in "
            "compilation_config.splitting_ops (an operator-supplied "
            "-cc.splitting_ops list, or an attn-fusion reset, can drop it). "
            f"Got splitting_ops={compilation_config.splitting_ops!r}."
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
) -> tuple[str, int, int, str]:
    """Shared fail-closed checks between :func:`validate_shards_for` (cheap,
    construction-time) and :func:`_attach_table` (authoritative, attach-time)
    — the same class of validation runs at both points, just at
    different times relative to the checkpoint's streamed load.

    Returns:
        The layer's validated (non-``None``) ``scale_entry``.
    """
    if layer_shards.cols != head_dim:
        raise RuntimeError(
            f"PLE mmap: layer {layer_idx} shard width {layer_shards.cols} "
            f"!= head_dim {head_dim}"
        )
    if layer_shards.dtype_str not in _FP8_DTYPES:
        raise RuntimeError(
            f"PLE mmap: layer {layer_idx} shards have unsupported dtype "
            f"{layer_shards.dtype_str!r}; only {sorted(_FP8_DTYPES)} "
            "is supported (F8_E5M2 is refused: is_fp8() does not "
            "recognize it, so dequant would silently never fire)"
        )
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


# --------------------------------------------------------------------------- #
# Construction-time validation: cheap, header-only checks run from
# Qwen4ExpNGramEmbedding.__init__, before the ~78 GiB backbone streams.
# --------------------------------------------------------------------------- #
def validate_shards_for(
    model_config: ModelConfig, layer_name: str, head_dim: int
) -> None:
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
        return
    layer_idx = _extract_layer_idx(layer_name)
    layer_shards = discover_shards(model_path).get(layer_idx)
    if layer_shards is None:
        raise RuntimeError(
            f"PLE mmap: no shard tensors for layer {layer_idx} "
            f"({layer_name!r}) under {model_path}"
        )
    _validate_layer_shards(layer_shards, head_dim, layer_idx, model_path)


# --------------------------------------------------------------------------- #
# Table construction, invoked once from the top-level model's load_weights.
# --------------------------------------------------------------------------- #
def build_tables(
    model_config: ModelConfig, compilation_config: CompilationConfig
) -> None:
    """Build and bounded-prewarm the table for every enabled PLE layer.

    Called from both ``Qwen4ExpForConditionalGeneration.load_weights`` and
    ``Qwen4ExpForCausalLM.load_weights`` after their respective streamed
    weight passes complete — never from
    ``Qwen4ExpNGramEmbedding.load_weights``, which would stream the whole
    table mid-load during the tightest memory transient. Since the
    ConditionalGeneration wrapper composes CausalLM internally, a single
    real load can reach this function twice.

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

    vocab = embedding.org_vocab_size
    # Verbatim shard-placement math from Qwen4ExpNGramEmbedding.load_weights
    # (nvidia/ple_layer.py) — discovery and gather must stay two
    # self-consistent halves of the same mapping.
    shard_size = (vocab + split_ngram_parts - 1) // split_ngram_parts
    num_expected_shards = min(
        split_ngram_parts, (vocab + shard_size - 1) // shard_size if shard_size else 0
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
        expected_rows = max(0, min(shard_size, vocab - checkpoint_start))
        if rows != expected_rows:
            raise RuntimeError(
                f"PLE mmap: layer {layer_idx} shard {shard_index} has "
                f"{rows} rows, expected {expected_rows}"
            )

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
            _FP8_DTYPES[layer_shards.dtype_str],
            workers=envs.VLLM_PLE_MMAP_WORKERS,
            chunk=envs.VLLM_PLE_MMAP_CHUNK,
            model_path=model_path,
            readahead=envs.VLLM_PLE_MMAP_READAHEAD,
        )
        embedding.torch_dtype = table.torch_dtype
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


# --------------------------------------------------------------------------- #
# Custom op: the WHOLE forward (hashing + gather), split out of the
# compiled/captured graph (see the module docstring for why the boundary
# is the whole forward rather than only the gather).
# --------------------------------------------------------------------------- #
def _qwen4_exp_ple_mmap_forward(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    from vllm.forward_context import get_forward_context

    try:
        layer = get_forward_context().no_compile_layers[layer_name]
    except KeyError:
        raise RuntimeError(
            f"PLE mmap: {layer_name!r} is not registered in no_compile_layers"
        ) from None
    ple_embedding_module = getattr(layer, "ple_embedding", None)
    if ple_embedding_module is None or not hasattr(
        ple_embedding_module, "_hash_ngram_ids"
    ):
        raise RuntimeError(f"PLE mmap: {layer_name!r} does not resolve to a PLE layer")
    # The stock trigram hashing runs here, eagerly and untraced — ordinary
    # GPU tensor ops are fine inside a custom op body; they are simply never
    # seen by Dynamo, which is the whole point of the widened boundary.
    ngram_ids = ple_embedding_module._hash_ngram_ids(
        input_ids, query_start_loc, ngram_context
    )
    result = ple_embedding_module.ngram_embedding(ngram_ids).flatten(-2)
    output.copy_(result)


def _qwen4_exp_ple_mmap_forward_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name=OP_NAME,
    op_func=_qwen4_exp_ple_mmap_forward,
    mutates_args=["output"],
    fake_impl=_qwen4_exp_ple_mmap_forward_fake,
)
# The op above registers under the platform-default dispatch key (CUDA in
# production). Unit tests run without a GPU-resident model and need the same
# op reachable with plain CPU tensors, so also register it directly under
# the CPU key — a second direct_register_custom_op call would
# re-define the schema and raise at MODULE IMPORT, killing every serve,
# since this module is imported unconditionally.
if current_platform.dispatch_key != "CPU":
    vllm_lib.impl(OP_NAME, _qwen4_exp_ple_mmap_forward, dispatch_key="CPU")

__all__ = [
    "OP_NAME",
    "QUALIFIED_OP_NAME",
    "MmapNgramEmbedding",
    "MmapPleTable",
    "build_tables",
    "check_cudagraph_safety",
    "compute_prewarm_bound",
    "discover_shards",
    "enabled",
    "parse_safetensors_header",
    "resolve_model_path",
    "set_weight_scale",
    "validate_shards_for",
]
