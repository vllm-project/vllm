# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor

# Default values for packed tensor transfer.
DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB
DEFAULT_PACKED_NUM_BUFFERS = 2


# Streams are cached across calls: the caching allocator keys cached blocks by the
# stream they were allocated on, so fresh streams at every call would strand the
# packed buffers' reserved memory per call. See gh-52950.
@cache
def _get_streams(device_idx: int, num_buffers: int) -> tuple[torch.cuda.Stream, ...]:
    return tuple(torch.cuda.Stream(device=device_idx) for _ in range(num_buffers))


def unpack_tensor(
    packed_tensor: torch.Tensor,
    names: list[str],
    shapes: list[list[int]],
    dtypes: list[torch.dtype],
    tensor_sizes: list[int],
) -> list[tuple[str, torch.Tensor]]:
    """Unpack a packed uint8 tensor into a list of named tensors.

    The returned tensors are **views** of `packed_tensor` (the
    `.contiguous()` call is a no-op on already-contiguous row-slices).
    If `packed_tensor` lives in storage that may be reused — e.g. a
    reused CUDA IPC buffer — callers must clone the results before the
    underlying storage is overwritten.

    Args:
        packed_tensor: The packed torch.uint8 tensor to unpack
        names: List of tensor names
        shapes: List of tensor shapes
        dtypes: List of tensor dtypes
        tensor_sizes: List of tensor sizes in bytes

    """
    unpacked_tensors = packed_tensor.split(tensor_sizes)

    return [
        (name, tensor.contiguous().view(dtype).view(shape))
        for name, shape, dtype, tensor in zip(names, shapes, dtypes, unpacked_tensors)
    ]


@dataclass(slots=True)
class PackedChunk:
    """Result of packing tensors into a single contiguous uint8 buffer."""

    packed_tensor: torch.Tensor
    names: list[str]
    shapes: list[list[int]]
    dtypes: list[torch.dtype]
    tensor_sizes: list[int]


def pack_tensors(
    iterator: Iterator[tuple[str, torch.Tensor]],
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int,
    tensor_list: list[torch.Tensor] | None = None,
    current_size: int = 0,
) -> PackedChunk | None:
    """Pack tensors from an iterator into a single contiguous uint8 buffer.

    Consumes from the iterator until the accumulated size exceeds
    buffer_size_bytes or the iterator is exhausted, then returns a
    PackedChunk. Returns None if no tensors were consumed.

    Args:
        iterator: Iterator of (name, tensor) pairs
        post_iter_func: Applied to each item before linearizing to uint8
        buffer_size_bytes: Max bytes before flushing
        tensor_list: Pre-existing tensor list to append to (for NCCL
                    multi-buffer reuse). If None, a fresh list is created.
        current_size: Byte count already accumulated in tensor_list

    """
    if tensor_list is None:
        tensor_list = []

    names: list[str] = []
    shapes: list[list[int]] = []
    dtypes: list[torch.dtype] = []
    tensor_sizes: list[int] = []
    total_bytes = current_size

    while True:
        try:
            item = next(iterator)
        except StopIteration:
            break

        name, orig_tensor = item
        # Apply post processing and convert to linearized uint8 tensor
        tensor = post_iter_func(item).contiguous().view(-1).view(torch.uint8)

        if tensor.numel() > buffer_size_bytes:
            import warnings

            warnings.warn(
                f"Tensor '{name}' has size {tensor.numel()} bytes, which "
                f"exceeds buffer_size_bytes={buffer_size_bytes}.",
                stacklevel=2,
            )

        tensor_list.append(tensor)
        names.append(name)
        shapes.append(list(orig_tensor.shape))
        dtypes.append(orig_tensor.dtype)
        tensor_sizes.append(tensor.numel())
        total_bytes += tensor.numel()

        if total_bytes > buffer_size_bytes:
            break

    if not tensor_list:
        return None

    packed = torch.cat(tensor_list, dim=0)
    del tensor_list
    return PackedChunk(
        packed_tensor=packed,
        names=names,
        shapes=shapes,
        dtypes=dtypes,
        tensor_sizes=tensor_sizes,
    )


# ── NCCL packed broadcast ──────────────────────────────────────────────


def wire_segments(
    shape: Sequence[int], dtype: torch.dtype, limit: int
) -> list[tuple[int, int]] | None:
    """Dim-0 row ranges a tensor travels as, or `None` to send it whole.

    A tensor over twice the buffer (e.g. one layer's expert stack) is split
    along dim 0 so that no side needs a buffer of its size."""
    nbytes = math.prod(shape) * dtype.itemsize
    if nbytes <= 2 * limit or len(shape) == 0 or shape[0] < 2:
        return None
    row = nbytes // shape[0]
    if row > limit:
        return None
    rows = limit // row
    return [(r, min(r + rows, shape[0])) for r in range(0, shape[0], rows)]


# (index, rows, shape, dtype, offset, nbytes) of one tensor or segment in a chunk
Piece = tuple[int, tuple[int, int] | None, list[int], torch.dtype, int, int]


def _pieces(
    specs: Iterable[tuple[Sequence[int], torch.dtype]], limit: int
) -> Iterator[tuple[bool, Piece]]:
    """Each piece in order, and whether it closes its chunk: a chunk takes
    pieces until it holds more than `limit` bytes (`pack_tensors`'s rule)."""
    size = 0
    for index, (shape, dtype) in enumerate(specs):
        nbytes = math.prod(shape) * dtype.itemsize
        segs = wire_segments(shape, dtype, limit)
        row = nbytes // shape[0] if segs else 0
        split: list[tuple[int, int] | None] = list(segs) if segs else [None]
        for rows in split:
            if rows is None:
                piece: Piece = (index, None, list(shape), dtype, size, nbytes)
            else:
                n = rows[1] - rows[0]
                piece = (index, rows, [n, *shape[1:]], dtype, size, n * row)
            size += piece[5]
            closes = size > limit
            if closes:
                size = 0
            yield closes, piece


@dataclass
class ChunkLayout:
    """How a sequence of tensors travels as packed chunks: `chunks[k]` holds
    the pieces of chunk k, which is `sizes[k]` bytes. The producer and every
    consumer derive it from the metadata alone, so they always agree."""

    chunks: list[list[Piece]]
    sizes: list[int]

    def views(self, k: int, buf: torch.Tensor) -> list[torch.Tensor]:
        """Chunk k's pieces in `buf`: typed, or raw bytes for a piece at an
        offset its dtype cannot view (pieces are packed back to back)."""
        out = []
        for _, _, shape, dtype, off, n in self.chunks[k]:
            raw = buf[off : off + n]
            try:
                out.append(raw.view(dtype).view(shape))
            except RuntimeError:
                out.append(raw)
        return out


def chunk_layout(
    specs: Iterable[tuple[Sequence[int], torch.dtype]], limit: int
) -> ChunkLayout:
    chunks: list[list[Piece]] = [[]]
    sizes: list[int] = []
    for closes, piece in _pieces(specs, limit):
        chunks[-1].append(piece)
        if closes:
            sizes.append(piece[4] + piece[5])
            chunks.append([])
    if chunks[-1]:
        sizes.append(chunks[-1][-1][4] + chunks[-1][-1][5])
    else:
        chunks.pop()
    return ChunkLayout(chunks, sizes)


def _source(piece: Piece, tensor: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    """What a piece copies from `tensor` into its `view`."""
    rows = piece[1]
    part = tensor if rows is None else tensor[rows[0] : rows[1]]
    return (
        part
        if view.dtype == part.dtype
        else part.contiguous().view(-1).view(view.dtype)
    )


def packed_nccl_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Broadcast tensors in a packed manner from trainer to workers.

    Each chunk is one `_foreach_copy_` of its tensors into a buffer, then one
    broadcast.

    Args:
        iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
        group: Process group (PyNcclCommunicator)
        src: Source rank (0 in current implementation)
        post_iter_func: Function to apply to each (name, tensor) pair before
                       packing, should return a tensor
        buffer_size_bytes: Size in bytes for each packed tensor buffer.
                          Both producer and consumer must use the same value.
        num_buffers: Number of buffers for double/triple buffering.
                    Both producer and consumer must use the same value.

    """
    tensors = (post_iter_func(item) for item in iterator)
    streams = _get_streams(torch.accelerator.current_device_index(), num_buffers)
    done: list[torch.Event | None] = [None] * num_buffers
    held: list[Any] = [None] * num_buffers  # alive until their broadcast ends
    for k, (buf, views, srcs) in enumerate(_chunks(tensors, buffer_size_bytes)):
        slot = k % num_buffers
        prev = done[slot]
        if prev is not None:
            prev.synchronize()
        with torch.cuda.stream(streams[slot]):
            torch._foreach_copy_(views, srcs)
            group.broadcast(buf, src=src)
            event = torch.Event()
            event.record(streams[slot])
        done[slot], held[slot] = event, (buf, srcs)
    for stream in streams:
        stream.synchronize()


def _chunks(
    tensors: Iterator[torch.Tensor], limit: int
) -> Iterator[tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]]:
    """(buffer, piece views, piece sources) per chunk."""
    stash: dict[int, torch.Tensor] = {}

    def specs() -> Iterator[tuple[Sequence[int], torch.dtype]]:
        for i, t in enumerate(tensors):
            stash[i] = t
            yield t.shape, t.dtype

    cur: list[Piece] = []
    for closes, piece in _pieces(specs(), limit):
        cur.append(piece)
        if closes:
            yield _fresh_chunk(cur, stash)
            cur = []
    if cur:
        yield _fresh_chunk(cur, stash)


def _fresh_chunk(pieces: list[Piece], stash: dict[int, torch.Tensor]):
    one = ChunkLayout([pieces], [pieces[-1][4] + pieces[-1][5]])
    buf = torch.empty(one.sizes[0], dtype=torch.uint8, device="cuda")
    views = one.views(0, buf)
    srcs = [_source(p, stash[p[0]], v) for p, v in zip(pieces, views)]
    for i in {p[0] for p in pieces} - {pieces[-1][0]}:
        stash.pop(i, None)  # the last tensor may continue in the next chunk
    return buf, views, srcs


def packed_nccl_broadcast_consumer(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    device: torch.device | str = "cuda",
) -> None:
    """Consume packed tensors and unpack them into a list of tensors.

    Args:
        iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
        group: Process group (PyNcclCommunicator)
        src: Source rank (0 in current implementation)
        post_unpack_func: Function to apply to each list of (name, tensor) after
                         unpacking
        buffer_size_bytes: Size in bytes for each packed tensor buffer.
                          Both producer and consumer must use the same value.
        num_buffers: Number of buffers for double/triple buffering.
                    Both producer and consumer must use the same value.
        device: Device for the receive buffers. Must match the device the NCCL
                communicator was created on (the worker's assigned device).

    """
    names, specs = [], []
    for name, spec in iterator:
        names.append(name)
        specs.append(spec)
    layout = chunk_layout(specs, buffer_size_bytes)
    streams = _get_streams(torch.accelerator.current_device_index(), num_buffers)
    caller_stream = torch.accelerator.current_stream()
    for stream in streams:
        stream.wait_stream(caller_stream)
    prev_received: torch.Event | None = None
    slot_done: list[torch.Event | None] = [None] * num_buffers
    segments: dict[int, list[torch.Tensor]] = {}  # a split tensor, joined

    for k, pieces in enumerate(layout.chunks):
        slot = k % num_buffers
        # Wait for this slot's previous pack. Not streams[slot].synchronize():
        # on ROCm, streams can share a hardware queue, and synchronizing one
        # stream then also waits for work queued later on the others.
        done = slot_done[slot]
        if done is not None:
            done.synchronize()
        with torch.cuda.stream(streams[slot]):
            packed = torch.empty(layout.sizes[k], dtype=torch.uint8, device=device)
            group.broadcast(packed, src=src)
            # post_unpack_func may keep the unpacked views and read them from a
            # later pack's callback, on another slot stream. Chaining the ready
            # events orders every callback after all earlier receives, and
            # record_stream keeps the allocator from reissuing this buffer once
            # it is freed while the broadcast or such a read is still pending.
            if prev_received is not None:
                streams[slot].wait_event(prev_received)
            prev_received = torch.Event()
            prev_received.record(streams[slot])
            for stream in (*streams, caller_stream):
                packed.record_stream(stream)
            out = []
            for piece, view in zip(pieces, layout.views(k, packed)):
                index, rows, shape, dtype = piece[:4]
                if view.dtype != dtype:  # at an offset its dtype cannot view
                    view = view.clone().view(dtype).view(shape)
                if rows is None:
                    out.append((names[index], view))
                    continue
                parts = segments.setdefault(index, [])
                parts.append(view.clone())  # the receive buffer is reused
                if rows[1] == specs[index][0][0]:
                    out.append((names[index], torch.cat(segments.pop(index))))
            if out:
                post_unpack_func(out)
            event = torch.Event()
            event.record(streams[slot])
            slot_done[slot] = event

    # Drain every slot before returning.
    for stream in streams:
        stream.synchronize()


# ── IPC packed transfer ────────────────────────────────────────────────


@dataclass(slots=True)
class PackedIpcChunk:
    """Metadata and IPC handle for a single packed chunk."""

    names: list[str]
    shapes: list[list[int]]
    dtype_names: list[str]
    tensor_sizes: list[int]
    ipc_handle: dict[str, tuple]


def packed_ipc_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    gpu_uuid: str,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
) -> Iterator[PackedIpcChunk]:
    """Pack tensors into a reusable IPC buffer and yield handles.

    Allocates a single GPU buffer of `buffer_size_bytes` and registers
    it for IPC once via `reduce_tensor`.  Each chunk's packed data is
    copied into this buffer before yielding, so only one IPC-shared
    allocation exists for the lifetime of the transfer.

    Callers **must** ensure the consumer has finished reading the buffer
    (e.g. `ray.get` returned) before resuming the generator for the
    next chunk.

    Args:
        iterator: Iterator of (name, tensor) pairs.
        gpu_uuid: Physical GPU UUID string for this rank.
        post_iter_func: Applied to each (name, tensor) before packing.
        buffer_size_bytes: Exact capacity of the reusable IPC buffer.
            Every chunk is guaranteed to fit within this size.  A
            `ValueError` is raised if any single tensor exceeds it.

    """
    ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device="cuda")
    _, ipc_args = reduce_tensor(ipc_buffer)

    names: list[str] = []
    shapes: list[list[int]] = []
    dtypes: list[torch.dtype] = []
    tensor_sizes: list[int] = []
    total_bytes = 0

    for name, orig_tensor in iterator:
        flat = (
            post_iter_func((name, orig_tensor)).contiguous().view(-1).view(torch.uint8)
        )

        if flat.numel() > buffer_size_bytes:
            raise ValueError(
                f"Tensor '{name}' has size {flat.numel()} bytes, "
                f"which exceeds buffer_size_bytes={buffer_size_bytes}. "
                f"Increase buffer_size_bytes to at least {flat.numel()}."
            )

        if total_bytes and total_bytes + flat.numel() > buffer_size_bytes:
            # Drain queued copies so the consumer sees a fully-written buffer.
            torch.accelerator.current_stream().synchronize()
            yield PackedIpcChunk(
                names=names,
                shapes=shapes,
                dtype_names=[str(d).split(".")[-1] for d in dtypes],
                tensor_sizes=tensor_sizes,
                ipc_handle={gpu_uuid: ipc_args},
            )
            # Rebind to fresh lists so the yielded chunk's metadata is
            # not mutated while the consumer is still reading.
            names, shapes, dtypes, tensor_sizes = [], [], [], []
            total_bytes = 0

        ipc_buffer[total_bytes : total_bytes + flat.numel()].copy_(flat)
        names.append(name)
        shapes.append(list(orig_tensor.shape))
        dtypes.append(orig_tensor.dtype)
        tensor_sizes.append(flat.numel())
        total_bytes += flat.numel()

    if total_bytes:
        torch.accelerator.current_stream().synchronize()
        yield PackedIpcChunk(
            names=names,
            shapes=shapes,
            dtype_names=[str(d).split(".")[-1] for d in dtypes],
            tensor_sizes=tensor_sizes,
            ipc_handle={gpu_uuid: ipc_args},
        )


class PackedBufferImporter:
    """One-slot cache for the consumer-side mapping of a packed IPC buffer.

    torch's cross-process refcount pairs one `reduce_tensor` export with
    one consumer rebuild-release cycle. The producer exports its buffer once
    and ships the same args with every chunk, so rebuilding and releasing
    per chunk decrements the counter once per chunk, drives it negative,
    and the dropped buffer is then never reclaimable by `ipc_collect` on
    the producer side (it leaks one buffer per transfer). Caching the
    rebuilt buffer keyed by the rebuild args restores the pairing: chunks
    of one transfer reuse the mapping (it aliases producer memory, so it
    always reads the current chunk's bytes), and replacing the entry when
    the next export arrives — or `close()` — releases the previous
    mapping exactly once.

    The importer that consumes a multi-chunk transfer must be one object
    reused across chunks; a fresh importer per chunk reintroduces the
    per-chunk release. `IPCWeightTransferEngine` owns one instance per
    engine and closes it on shutdown.
    """

    def __init__(self) -> None:
        self._entry: tuple[tuple, torch.Tensor] | None = None

    def rebuild(self, list_args: list) -> torch.Tensor:
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        key = tuple(list_args)
        if self._entry is not None and self._entry[0] == key:
            return self._entry[1]
        packed = rebuild_cuda_tensor(*list_args)
        self._entry = (key, packed)
        return packed

    def close(self) -> None:
        """Release the held mapping (one producer-side refcount decrement)."""
        self._entry = None


def packed_ipc_consumer(
    ipc_handle: dict[str, tuple],
    names: list[str],
    shapes: list[list[int]],
    dtype_names: list[str],
    tensor_sizes: list[int],
    device_index: int,
    importer: "PackedBufferImporter | None" = None,
) -> list[tuple[str, torch.Tensor]]:
    """Unpack a single packed IPC chunk into named tensors.

    Reconstructs the packed buffer via rebuild_cuda_tensor, unpacks
    into individual tensors, and clones each into independent storage
    before returning.

    The clone is intentional: the producer reuses one IPC buffer across
    chunks, so any tensor view that aliases the buffer would observe the
    *next* chunk's bytes as soon as the producer's generator is resumed.
    Callers that retain references past their own update_weights call
    (notably vLLM's layerwise reload, which buffers `bound_args` for
    replay in `_layerwise_process`) would otherwise replay against
    stale data and silently corrupt multi-chunk weight transfers.

    Args:
        ipc_handle: Mapping of GPU UUID to rebuild_cuda_tensor args tuple
        names: Parameter names in the packed buffer
        shapes: Parameter shapes
        dtype_names: Parameter dtype name strings (e.g. "float16")
        tensor_sizes: Size in bytes of each parameter in the packed buffer
        device_index: Local CUDA device index
        importer: Import cache that must be shared across all chunks of one
            export so the buffer mapping is opened and released exactly once
            (see `PackedBufferImporter`). `None` builds an ephemeral one,
            which is only safe for single-chunk consumption.

    """
    props = torch.cuda.get_device_properties(device_index)
    physical_gpu_id = str(props.uuid)

    if physical_gpu_id not in ipc_handle:
        raise ValueError(
            f"IPC handle not found for GPU UUID {physical_gpu_id}. "
            f"Available UUIDs: {list(ipc_handle.keys())}"
        )

    args = ipc_handle[physical_gpu_id]
    list_args = list(args)
    list_args[6] = device_index
    if importer is None:
        importer = PackedBufferImporter()
    packed = importer.rebuild(list_args)

    content_size = sum(tensor_sizes)
    packed = packed[:content_size]

    dtypes = [getattr(torch, dn) for dn in dtype_names]
    return [
        (name, t.clone())
        for name, t in unpack_tensor(packed, names, shapes, dtypes, tensor_sizes)
    ]
