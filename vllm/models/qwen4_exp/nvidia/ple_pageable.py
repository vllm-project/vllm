# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read the Qwen4Exp PLE n-gram table in place from the checkpoint files.

On GPUs that dereference pageable host memory through the host page tables
(``CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES``, e.g. the
unified-memory GB10), a kernel can read a read-only ``mmap`` of the checkpoint's
safetensors files directly. There is then no table-sized device or pinned
allocation, no resident duplicate of the table and no CPU-gather/host-to-device
staging path: its pages live in the page cache as clean, file-backed pages that
the kernel can drop and re-read, shared by every process mapping the same files.

GPU faults on non-resident file pages are serviced one page at a time, so the
rows a step needs are faulted in ahead of the lookup by CPU threads
(:class:`PagePrefetcher`). The prefetch is only a hint: the lookup is correct
whether or not it has finished.
"""

import ctypes
import json
import mmap
import os
import queue
import struct
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import regex as re
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_PAGE = 4096
_MADV_RANDOM = 1
_CU_PAGEABLE_MEMORY_ACCESS = 88
_CU_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
_SAFETENSORS_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}
_SHARD_RE = re.compile(
    r"layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
)


def require_pageable_access(device_index: int) -> None:
    """Raise unless the GPU reads pageable host memory via host page tables.

    Being an integrated GPU is not sufficient, and an attribute that cannot be
    queried is treated as unsupported.
    """
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        raise RuntimeError("Cannot load the CUDA driver to query the GPU") from e
    device = ctypes.c_int()
    if cuda.cuInit(0) != 0 or cuda.cuDeviceGet(ctypes.byref(device), device_index):
        raise RuntimeError("Cannot query the CUDA driver for the device")
    for name, attribute in (
        ("PAGEABLE_MEMORY_ACCESS", _CU_PAGEABLE_MEMORY_ACCESS),
        (
            "PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES",
            _CU_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
        ),
    ):
        value = ctypes.c_int(0)
        rc = cuda.cuDeviceGetAttribute(ctypes.byref(value), attribute, device)
        if rc != 0 or value.value != 1:
            raise RuntimeError(
                "Engram checkpoint_mapped requires a GPU that reads pageable host "
                f"memory through the host page tables; device {device_index} "
                f"reports CU_DEVICE_ATTRIBUTE_{name}={value.value} (rc={rc})"
            )


@dataclass(frozen=True)
class ShardLocation:
    path: str
    offset: int
    rows: int


@dataclass(frozen=True)
class TableLayout:
    """Where every checkpoint shard of one layer's table lives."""

    num_rows: int
    row_bytes: int
    rows_per_shard: int
    shards: tuple[ShardLocation, ...]


def discover_table_layout(
    files: list[str],
    layer_index: int,
    num_rows: int,
    embedding_dim: int,
    dtype: torch.dtype,
    split_parts: int,
) -> TableLayout:
    """Locate and validate one layer's PLE shards among the loader's files.

    Mirrors ``Qwen4ExpNGramEmbedding.load_weights``: shard ``i`` holds rows
    ``[i * S, min((i + 1) * S, num_rows))`` with ``S = ceil(num_rows /
    split_parts)``. Every shard that owns rows must exist exactly once, with
    exactly that shape and ``dtype``; nothing is truncated or padded.
    """
    found: dict[int, tuple[str, int, list[int], str]] = {}
    if not files:
        raise FileNotFoundError("No safetensors files to map the PLE table from")
    # ``files`` is the loader's list: already filtered by the safetensors index,
    # so files the index excludes are never considered.
    for path in sorted(files):
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for name, meta in header.items():
            match = _SHARD_RE.search(name)
            if match is None or int(match.group(1)) != layer_index:
                continue
            shard = int(match.group(2))
            if shard in found:
                raise ValueError(
                    f"PLE shard {shard} of layer {layer_index} appears twice "
                    f"({found[shard][0]} and {path})"
                )
            found[shard] = (
                path,
                8 + header_len + meta["data_offsets"][0],
                meta["shape"],
                meta["dtype"],
            )

    rows_per_shard = (num_rows + split_parts - 1) // split_parts
    needed = [i for i in range(split_parts) if i * rows_per_shard < num_rows]
    missing = [i for i in needed if i not in found]
    unexpected = sorted(set(found) - set(needed))
    if missing or unexpected:
        raise ValueError(
            f"PLE shards of layer {layer_index} do not cover "
            f"{num_rows} rows in {split_parts} parts: missing {missing[:8]}, "
            f"unexpected {unexpected[:8]}"
        )
    shards = []
    for i in needed:
        path, offset, shape, st_dtype = found[i]
        want = [min(rows_per_shard, num_rows - i * rows_per_shard), embedding_dim]
        if list(shape) != want:
            raise ValueError(f"PLE shard {i} has shape {shape}, expected {want}")
        if _SAFETENSORS_DTYPES.get(st_dtype) != dtype:
            raise ValueError(
                f"PLE shard {i} is stored as {st_dtype}, but the embedding uses "
                f"{dtype}; checkpoint_mapped cannot convert rows"
            )
        shards.append(ShardLocation(path, offset, want[0]))
    return TableLayout(
        num_rows=num_rows,
        row_bytes=embedding_dim * dtype.itemsize,
        rows_per_shard=rows_per_shard,
        shards=tuple(shards),
    )


@triton.jit
def _gather_mapped_rows_kernel(
    ids_ptr,
    shard_base_ptr,
    out_ptr,
    rows_per_shard,
    vocab_start,
    vocab_end,
    ROW_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy owned rows byte-wise from mapped shards; write zeros otherwise."""
    pid = tl.program_id(0)
    row = tl.load(ids_ptr + pid).to(tl.int64)
    owned = (row >= vocab_start) & (row < vocab_end)
    row = tl.where(owned, row, 0)
    shard = row // rows_per_shard
    local = row - shard * rows_per_shard
    base = tl.load(shard_base_ptr + shard).to(tl.pointer_type(tl.uint8))
    offsets = tl.arange(0, BLOCK)
    in_row = offsets < ROW_BYTES
    values = tl.load(base + local * ROW_BYTES + offsets, mask=in_row & owned, other=0)
    # The store is unmasked by ownership so rows this rank does not own are zero,
    # never allocator residue or a previous graph replay's bytes.
    tl.store(out_ptr + pid.to(tl.int64) * ROW_BYTES + offsets, values, mask=in_row)


def verify_shard_bytes(
    table: "MappedTable",
    checkpoint_start: int,
    shard: torch.Tensor,
    chunk_rows: int = 65536,
) -> None:
    """Raise unless an incoming checkpoint shard equals the mapped rows exactly.

    Compares every byte, chunk by chunk, without copying the mapping. The
    mapping serves rows from the checkpoint files; weights that differ from
    those files (e.g. delivered from memory by a weight sync) cannot be served
    and must not be silently ignored.
    """
    if shard.numel() == 0:
        return
    index, offset = divmod(checkpoint_start, table.rows_per_shard)
    view = table.views[index] if index < len(table.views) else None
    rows = shard.shape[0]
    if view is None or offset != 0 or view.shape[0] != rows:
        raise ValueError(
            f"Incoming PLE shard at row {checkpoint_start} ({rows} rows) does not "
            "line up with the mapped checkpoint shards"
        )
    raw = shard.detach().contiguous().view(torch.uint8).reshape(rows, -1)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        incoming = raw[start:end].cpu().numpy()
        if not np.array_equal(view[start:end], incoming):
            bad = start + int(np.flatnonzero((view[start:end] != incoming).any(1))[0])
            raise ValueError(
                f"PLE row {checkpoint_start + bad} received from the weights source "
                "differs from the mapped checkpoint files. checkpoint_mapped serves "
                "the table from disk and cannot load it from in-memory weights; "
                "reload from a checkpoint on disk (weights_path) or disable "
                "checkpoint_mapped."
            )


class MappedTable:
    """Read-only mappings of a layer's shards plus the GPU and CPU accessors."""

    def __init__(self, layout: TableLayout, device: torch.device) -> None:
        self.layout: TableLayout | None = layout
        self.row_bytes = layout.row_bytes
        self.rows_per_shard = layout.rows_per_shard
        self.num_rows = layout.num_rows
        self._maps: dict[str, tuple[mmap.mmap, np.ndarray]] = {}
        libc = ctypes.CDLL(None, use_errno=True)
        for path in sorted({s.path for s in layout.shards}):
            fd = os.open(path, os.O_RDONLY)
            try:
                size = os.fstat(fd).st_size
                mapping = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
            finally:
                os.close(fd)
            array = np.frombuffer(mapping, dtype=np.uint8)
            address = array.__array_interface__["data"][0]
            # Rows are read at random; readahead would only evict useful pages.
            libc.madvise(ctypes.c_void_p(address), ctypes.c_size_t(size), _MADV_RANDOM)
            self._maps[path] = (mapping, array)
        self.views = [
            self._maps[s.path][1][
                s.offset : s.offset + s.rows * self.row_bytes
            ].reshape(s.rows, self.row_bytes)
            for s in layout.shards
        ]
        bases = [v.__array_interface__["data"][0] for v in self.views]
        self.shard_base = torch.tensor(bases, dtype=torch.int64, device=device)
        self._block = triton.next_power_of_2(self.row_bytes)

    @classmethod
    def zeros(
        cls, num_rows: int, row_bytes: int, device: torch.device
    ) -> "MappedTable":
        """An anonymous, never-written mapping: every row reads as zero.

        Used when no checkpoint is loaded (``--load-format dummy``); the zero
        page backs every read, so no memory is committed.
        """
        self = cls.__new__(cls)
        self.row_bytes = row_bytes
        self.rows_per_shard = max(num_rows, 1)
        self.num_rows = num_rows
        # Private and anonymous: every read is served by the shared zero page, so
        # touching the table commits no memory (a MAP_SHARED anonymous mapping
        # would allocate shmem pages on read).
        mapping = mmap.mmap(
            -1,
            max(num_rows * row_bytes, _PAGE),
            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            prot=mmap.PROT_READ,
        )
        array = np.frombuffer(mapping, dtype=np.uint8)
        self._maps = {"<zeros>": (mapping, array)}
        self.views = [array[: num_rows * row_bytes].reshape(num_rows, row_bytes)]
        self.layout = None
        self.shard_base = torch.tensor(
            [array.__array_interface__["data"][0]], dtype=torch.int64, device=device
        )
        self._block = triton.next_power_of_2(row_bytes)
        return self

    def gather_into(
        self,
        ids: torch.Tensor,
        out: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
    ) -> None:
        """Write rows ``ids`` into ``out`` (``[len(ids), row_bytes]`` uint8)."""
        if ids.numel() == 0:
            return
        _gather_mapped_rows_kernel[(ids.numel(),)](
            ids,
            self.shard_base,
            out,
            self.rows_per_shard,
            vocab_start,
            min(vocab_end, self.num_rows),
            ROW_BYTES=self.row_bytes,
            BLOCK=self._block,
        )

    def touch(self, rows: np.ndarray, pool: ThreadPoolExecutor | None) -> None:
        """Fault in both ends of every row (a row may straddle a page)."""
        rows = np.sort(rows)
        shard = rows // self.rows_per_shard
        local = rows - shard * self.rows_per_shard
        last = self.row_bytes - 1

        def groups(index: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
            s = shard[index]
            cuts = np.flatnonzero(np.diff(s)) + 1
            return [
                (self.views[int(group[0])], rows_in_shard)
                for group, rows_in_shard in zip(
                    np.split(s, cuts), np.split(local[index], cuts)
                )
                if group.size
            ]

        # Index the per-shard 2-D views: fancy-indexing the flat byte arrays is
        # ~12x slower cold with identical tasks.
        def fault(task: tuple[np.ndarray, np.ndarray]) -> int:
            view, index = task
            return int(view[index, 0].sum()) + int(view[index, last].sum())

        if pool is None or rows.size <= 4096:
            for task in groups(np.arange(rows.size)):
                fault(task)
            return
        tasks = []
        for chunk in np.array_split(np.arange(rows.size), 64):
            if chunk.size:
                tasks.extend(groups(chunk))
        list(pool.map(fault, tasks))


class PagePrefetcher:
    """Fault in each step's PLE rows from CPU threads before the GPU reads them.

    ``prepare`` copies the step's inputs to pinned staging buffers on the current
    stream and returns; a worker thread waits for the copy, computes the n-gram
    ids on the CPU, and touches the rows this rank owns. Each source's table is
    re-read on every step, so a remap (weight reload) takes effect here too.
    """

    SLOTS = 4
    THREADS = 64

    def __init__(
        self,
        sources: list["PrefetchSource"],
        device: torch.device,
        max_num_tokens: int,
        max_num_reqs: int,
        ngram_context_len: int,
    ) -> None:
        self.sources = sources
        if device.index is None:  # e.g. torch.device("cuda"): pin to the current GPU
            device = torch.device(device.type, torch.accelerator.current_device_index())
        self.device = device
        self.capacity = (max_num_tokens, max_num_reqs, ngram_context_len)
        # Pinned staging slots at full capacity (allocated on the first step,
        # with the dtypes of the model state's buffers): the per-step input_ids
        # view changes size between steps.
        self.slots: list[tuple[torch.Tensor, ...]] = []
        self.free: queue.Queue[int] = queue.Queue()
        for i in range(self.SLOTS):
            self.free.put(i)
        self.work: queue.Queue = queue.Queue()
        self.pool = ThreadPoolExecutor(self.THREADS, thread_name_prefix="ple-touch")
        self.skipped = 0
        threading.Thread(target=self._loop, name="ple-prefetch", daemon=True).start()

    def prepare(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
    ) -> None:
        max_tokens, max_reqs, ctx_len = self.capacity
        if (
            num_tokens > max_tokens
            or num_reqs > max_reqs
            or ngram_context.shape[1] > ctx_len
        ):
            logger.warning_once(
                "PLE page prefetch skipped a step larger than its staging "
                "capacity; lookups are unaffected"
            )
            return
        if not self.slots:
            self.slots = [
                (
                    torch.empty(max_tokens, dtype=input_ids.dtype).pin_memory(),
                    torch.empty(max_reqs + 1, dtype=query_start_loc.dtype).pin_memory(),
                    torch.empty(
                        max_reqs, ctx_len, dtype=ngram_context.dtype
                    ).pin_memory(),
                )
                for _ in range(self.SLOTS)
            ]
        try:
            slot = self.free.get_nowait()
        except queue.Empty:
            self.skipped += 1  # a hint only: the lookup stays correct
            return
        ids, qsl, ctx = self.slots[slot]
        ids[:num_tokens].copy_(input_ids[:num_tokens], non_blocking=True)
        qsl[: num_reqs + 1].copy_(query_start_loc[: num_reqs + 1], non_blocking=True)
        ctx[:num_reqs, : ngram_context.shape[1]].copy_(
            ngram_context[:num_reqs], non_blocking=True
        )
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self.work.put((slot, event, num_reqs, num_tokens, ngram_context.shape[1]))

    def _loop(self) -> None:
        try:
            self._serve()
        except Exception:
            # Never die silently: without the prefetch, cold rows fault one page
            # at a time on the GPU (correct, but slow).
            logger.exception("PLE page prefetch thread stopped")

    def _serve(self) -> None:
        torch.accelerator.set_device_index(self.device.index)
        while True:
            slot, event, num_reqs, num_tokens, ctx_len = self.work.get()
            try:
                event.synchronize()
                ids, qsl, ctx = self.slots[slot]
                ids = ids[:num_tokens].clone()
                qsl = qsl[: num_reqs + 1].clone()
                ctx = ctx[:num_reqs, :ctx_len].clone()
            finally:
                self.free.put(slot)
            try:
                for source in self.sources:
                    source.touch(ids, qsl, ctx, self.pool)
            except Exception:
                logger.exception("PLE page prefetch failed; lookups are unaffected")


class PrefetchSource:
    """One mapped embedding and a CPU hash of its ids, refreshed on remap."""

    def __init__(
        self,
        get_table: Callable[[], "MappedTable | None"],
        vocab_range: tuple[int, int],
        make_cpu_ids: Callable[[], Callable[..., torch.Tensor]],
    ) -> None:
        self.get_table = get_table
        self.vocab_range = vocab_range
        self.make_cpu_ids = make_cpu_ids
        self._table: MappedTable | None = None
        self._cpu_ids: Callable[..., torch.Tensor] | None = None

    def touch(self, ids, qsl, ctx, pool: ThreadPoolExecutor) -> None:
        table = self.get_table()
        if table is None:
            return
        if table is not self._table or self._cpu_ids is None:
            # First step, or the mapping was rebuilt: refresh the CPU hash buffers.
            self._cpu_ids = self.make_cpu_ids()
            self._table = table
        rows = self._cpu_ids(ids, qsl, ctx).reshape(-1).numpy()
        start, end = self.vocab_range
        rows = rows[(rows >= start) & (rows < min(end, table.num_rows))]
        table.touch(rows, pool)
