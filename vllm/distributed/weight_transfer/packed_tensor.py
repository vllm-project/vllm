# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, NamedTuple

import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm.logger import init_logger

logger = init_logger(__name__)

# Default values for packed tensor transfer.
DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB
DEFAULT_PACKED_NUM_BUFFERS = 2


# Streams are cached across calls: the caching allocator keys cached blocks by the
# stream they were allocated on, so fresh streams at every call would strand the
# packed buffers' reserved memory per call. See gh-52950.
@cache
def _get_streams(device_idx: int, num_buffers: int) -> tuple[torch.cuda.Stream, ...]:
    return tuple(torch.cuda.Stream(device=device_idx) for _ in range(num_buffers))


# ── NCCL packed broadcast ──────────────────────────────────────────────


def wire_segments(
    shape: Sequence[int], dtype: torch.dtype, limit: int
) -> list[tuple[int, int]] | None:
    """Dim-0 row ranges a tensor travels as, or `None` to send it whole.

    A tensor over the buffer (e.g. one layer's expert stack) is split along
    dim 0 into segments that fit it, so no chunk buffer needs its size (a
    chunk holds at most twice the buffer); the consumer joins the segments
    into one tensor of the full size. A tensor whose single row exceeds
    the buffer travels whole.
    """
    nbytes = math.prod(shape) * dtype.itemsize
    if nbytes <= limit or len(shape) == 0 or shape[0] < 2:
        return None
    row = nbytes // shape[0]
    if row > limit:
        logger.warning_once(
            "Packed transfer: a %s tensor of shape %s has rows of %d bytes, over "
            "the %d-byte buffer, so it is sent whole; raise the buffer size to "
            "split it.",
            dtype,
            tuple(shape),
            row,
            limit,
        )
        return None
    rows = limit // row
    return [(r, min(r + rows, shape[0])) for r in range(0, shape[0], rows)]


class Piece(NamedTuple):
    """One tensor, or one dim-0 segment of it, in a packed chunk."""

    position: int  # the tensor's position in the sent sequence
    rows: tuple[int, int] | None  # dim-0 range of a segment; None: the whole tensor
    shape: list[int]  # shape of this piece
    dtype: torch.dtype
    offset: int  # byte offset in the chunk
    nbytes: int


def _chunks(
    specs: Iterable[tuple[Sequence[int], torch.dtype]], limit: int, segments: bool
) -> Iterator[list[Piece]]:
    """Each chunk's pieces, lazily: a chunk takes pieces until it holds more
    than `limit` bytes. Without `segments` every tensor travels whole, back to
    back."""
    chunk: list[Piece] = []
    size = 0
    for index, (shape, dtype) in enumerate(specs):
        nbytes = math.prod(shape) * dtype.itemsize
        segs = wire_segments(shape, dtype, limit) if segments else None
        split: list[tuple[int, int] | None] = list(segs) if segs else [None]
        for rows in split:
            if rows is None:
                piece = Piece(index, None, list(shape), dtype, size, nbytes)
            else:
                n = rows[1] - rows[0]
                row = nbytes // shape[0]
                piece = Piece(index, rows, [n, *shape[1:]], dtype, size, n * row)
            chunk.append(piece)
            size += piece.nbytes
            if size > limit:
                yield chunk
                chunk, size = [], 0
    if chunk:
        yield chunk


def chunk_bytes(chunk: list[Piece]) -> int:
    return chunk[-1].offset + chunk[-1].nbytes


def _view(buf: torch.Tensor, piece: Piece) -> torch.Tensor:
    """`piece` in `buf`: typed, or raw bytes at an offset its dtype cannot
    view (pieces are packed back to back)."""
    raw = buf[piece.offset : piece.offset + piece.nbytes]
    if raw.storage_offset() % piece.dtype.itemsize:
        return raw
    return raw.view(piece.dtype).view(piece.shape)


def _unpacked(view: torch.Tensor, piece: Piece) -> torch.Tensor:
    """A received piece as its dtype and shape; one at an offset its dtype
    cannot view is copied into aligned storage first."""
    if view.dtype == piece.dtype:
        return view
    return view.clone().view(piece.dtype).view(piece.shape)


def chunk_layout(
    specs: Iterable[tuple[Sequence[int], torch.dtype]],
    limit: int,
    segments: bool = False,
) -> list[list[Piece]]:
    """The chunks `specs` travel as, each chunk's pieces. The producer and
    every consumer derive it from the metadata alone, so they always agree.
    `segments` (both sides must agree; see
    `NCCLWeightTransferInitInfo.packed_wire_segments`) splits tensors over
    the buffer along dim 0."""
    return list(_chunks(specs, limit, segments))


def _source(piece: Piece, tensor: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    """What a piece copies from `tensor` into its `view`."""
    rows = piece.rows
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
    segments: bool = False,
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
        segments: split tensors over the buffer along dim 0. Both sides
            must agree (`chunk_layout`).

    """
    tensors: dict[int, torch.Tensor] = {}  # the ones the next chunk may read

    def specs() -> Iterator[tuple[Sequence[int], torch.dtype]]:
        for i, item in enumerate(iterator):
            t = tensors[i] = post_iter_func(item)
            yield t.shape, t.dtype

    streams = _get_streams(torch.accelerator.current_device_index(), num_buffers)
    caller_stream = torch.accelerator.current_stream()
    done: list[torch.Event | None] = [None] * num_buffers
    held: list[Any] = [None] * num_buffers  # alive until their broadcast ends
    for k, pieces in enumerate(_chunks(specs(), buffer_size_bytes, segments)):
        device = tensors[pieces[0].position].device
        buf = torch.empty(chunk_bytes(pieces), dtype=torch.uint8, device=device)
        views = [_view(buf, p) for p in pieces]
        srcs = [_source(p, tensors[p.position], v) for p, v in zip(pieces, views)]
        last = pieces[-1]
        continues = last.rows is not None and last.rows[1] < len(tensors[last.position])
        for i in range(pieces[0].position, last.position + (not continues)):
            del tensors[i]
        slot = k % num_buffers
        prev = done[slot]
        if prev is not None:
            prev.synchronize()
        # the sources may come from work the caller queued while iterating
        streams[slot].wait_stream(caller_stream)
        with torch.cuda.stream(streams[slot]):
            torch._foreach_copy_(views, srcs)
            group.broadcast(buf, src=src)
            event = torch.Event()
            event.record(streams[slot])
        done[slot], held[slot] = event, (buf, srcs)
    for stream in streams:
        stream.synchronize()


def packed_nccl_broadcast_consumer(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    device: torch.device | str = "cuda",
    segments: bool = False,
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
        segments: tensors over the buffer arrive as dim-0 segments
                 (`chunk_layout`); each is joined into one tensor of its
                 full size before `post_unpack_func` sees it.

    """
    names, specs = [], []
    for name, spec in iterator:
        names.append(name)
        specs.append(spec)
    streams = _get_streams(torch.accelerator.current_device_index(), num_buffers)
    caller_stream = torch.accelerator.current_stream()
    for stream in streams:
        stream.wait_stream(caller_stream)
    prev_done: torch.Event | None = None  # the previous chunk's last work
    slot_done: list[torch.Event | None] = [None] * num_buffers
    joined: dict[int, torch.Tensor] = {}  # a split tensor, filled as it arrives

    for k, pieces in enumerate(_chunks(specs, buffer_size_bytes, segments)):
        slot = k % num_buffers
        # Wait for this slot's previous pack. Not streams[slot].synchronize():
        # on ROCm, streams can share a hardware queue, and synchronizing one
        # stream then also waits for work queued later on the others.
        done = slot_done[slot]
        if done is not None:
            done.synchronize()
        with torch.cuda.stream(streams[slot]):
            packed = torch.empty(chunk_bytes(pieces), dtype=torch.uint8, device=device)
            group.broadcast(packed, src=src)
            # post_unpack_func may keep the unpacked tensors (and a split
            # tensor's earlier rows are copied on another slot) and read them
            # from a later chunk's callback, on another slot stream. Waiting
            # for the previous chunk's work orders every unpack and callback
            # after all earlier chunks' copies, and record_stream keeps the
            # allocator from reissuing this buffer once it is freed while the
            # broadcast or such a read is still pending.
            if prev_done is not None:
                streams[slot].wait_event(prev_done)
            for stream in (*streams, caller_stream):
                packed.record_stream(stream)
            out = []
            for piece in pieces:
                view = _view(packed, piece)
                index, rows, dtype = piece.position, piece.rows, piece.dtype
                if rows is not None:
                    # one tensor of the full size, written segment by segment
                    whole = joined.get(index)
                    if whole is None:
                        whole = joined[index] = torch.empty(
                            specs[index][0], dtype=dtype, device=device
                        )
                    # raw bytes into raw bytes: a misaligned segment needs no copy
                    dst = whole[rows[0] : rows[1]]
                    if view.dtype != dtype:
                        dst = dst.view(-1).view(torch.uint8)
                    dst.copy_(view)
                    if rows[1] < specs[index][0][0]:
                        continue
                    view = joined.pop(index)
                else:
                    view = _unpacked(view, piece)
                if view.untyped_storage().data_ptr() != packed.data_ptr():
                    # a copy, read like the packed buffer: keep it alive on
                    # every stream that may still read it once it is freed
                    for stream in (*streams, caller_stream):
                        view.record_stream(stream)
                out.append((names[index], view))
            if out:
                post_unpack_func(out)
            prev_done = slot_done[slot] = torch.Event()
            prev_done.record(streams[slot])

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
    accelerator = torch.accelerator.current_accelerator()
    assert accelerator is not None
    device = torch.device(accelerator.type, torch.accelerator.current_device_index())
    ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device=device)
    _, ipc_args = reduce_tensor(ipc_buffer)

    names: list[str] = []
    shapes: list[list[int]] = []
    dtypes: list[torch.dtype] = []
    tensor_sizes: list[int] = []
    total_bytes = 0

    def chunk() -> PackedIpcChunk:
        # Drain queued copies so the consumer sees a fully-written buffer.
        torch.accelerator.current_stream().synchronize()
        return PackedIpcChunk(
            names=names,
            shapes=shapes,
            dtype_names=[str(d).split(".")[-1] for d in dtypes],
            tensor_sizes=tensor_sizes,
            ipc_handle={gpu_uuid: ipc_args},
        )

    for name, orig_tensor in iterator:
        flat = (
            post_iter_func((name, orig_tensor)).contiguous().view(-1).view(torch.uint8)
        )
        nbytes = flat.numel()
        if nbytes > buffer_size_bytes:
            raise ValueError(
                f"Tensor '{name}' has size {nbytes} bytes, "
                f"which exceeds buffer_size_bytes={buffer_size_bytes}. "
                f"Increase buffer_size_bytes to at least {nbytes}."
            )
        # A chunk must fit the fixed IPC buffer: flush before overflowing it.
        if total_bytes and total_bytes + nbytes > buffer_size_bytes:
            yield chunk()
            # Fresh lists, so the yielded chunk's metadata is not mutated
            # while the consumer is still reading it.
            names, shapes, dtypes, tensor_sizes = [], [], [], []
            total_bytes = 0
        ipc_buffer[total_bytes : total_bytes + nbytes].copy_(flat)
        names.append(name)
        shapes.append(list(orig_tensor.shape))
        dtypes.append(orig_tensor.dtype)
        tensor_sizes.append(nbytes)
        total_bytes += nbytes

    if total_bytes:
        yield chunk()


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

    # The chunk's tensors, back to back as the producer packed them. Each
    # tensor's bytes are cloned into its own storage (see above), which is
    # also aligned for its dtype wherever it sat in the buffer.
    out, offset = [], 0
    for name, shape, dtype_name, nbytes in zip(
        names, shapes, dtype_names, tensor_sizes, strict=True
    ):
        raw = packed[offset : offset + nbytes].clone()
        out.append((name, raw.view(getattr(torch, dtype_name)).view(shape)))
        offset += nbytes
    return out
