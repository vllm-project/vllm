# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
import weakref
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
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
    dim 0 into segments that fit it, so no side needs a buffer of its size and
    a chunk holds at most twice the buffer. A tensor whose single row exceeds
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

    index: int  # the tensor's position in the sent sequence
    rows: tuple[int, int] | None  # dim-0 range of a segment; None: the whole tensor
    shape: list[int]  # shape of this piece
    dtype: torch.dtype
    offset: int  # byte offset in the chunk
    nbytes: int


def _pieces(
    specs: Iterable[tuple[Sequence[int], torch.dtype]], limit: int, segments: bool
) -> Iterator[tuple[bool, Piece]]:
    """Each piece in order, and whether it closes its chunk: a chunk takes
    pieces until it holds more than `limit` bytes. Without `segments` every
    tensor travels whole, back to back."""
    size = 0
    for index, (shape, dtype) in enumerate(specs):
        nbytes = math.prod(shape) * dtype.itemsize
        segs = wire_segments(shape, dtype, limit) if segments else None
        row = nbytes // shape[0] if segs else 0
        split: list[tuple[int, int] | None] = list(segs) if segs else [None]
        for rows in split:
            if rows is None:
                piece = Piece(index, None, list(shape), dtype, size, nbytes)
            else:
                n = rows[1] - rows[0]
                piece = Piece(index, rows, [n, *shape[1:]], dtype, size, n * row)
            size += piece.nbytes
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
    # (shape, dtype) of every tensor, in order; empty for a layout built
    # from one chunk's pieces (IPC, a streamed chunk)
    specs: list[tuple[tuple[int, ...], torch.dtype]] = field(default_factory=list)

    def views(self, k: int, buf: torch.Tensor) -> list[torch.Tensor]:
        """Chunk k's pieces in `buf`: typed, or raw bytes for a piece at an
        offset its dtype cannot view (pieces are packed back to back)."""
        return [_view(buf, p) for p in self.chunks[k]]


def _view(buf: torch.Tensor, piece: Piece) -> torch.Tensor:
    """`piece` in `buf`: typed, or raw bytes at an offset its dtype cannot
    view."""
    raw = buf[piece.offset : piece.offset + piece.nbytes]
    try:
        return raw.view(piece.dtype).view(piece.shape)
    except RuntimeError:
        return raw


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
) -> ChunkLayout:
    """The chunks `specs` travel as. `segments` (both sides must agree; see
    `NCCLWeightTransferInitInfo.packed_wire_segments`) splits tensors over
    the buffer along dim 0; without it every tensor travels whole, back to
    back, and a chunk takes tensors until it holds more than `limit` bytes."""
    full = [(tuple(shape), dtype) for shape, dtype in specs]
    chunks: list[list[Piece]] = [[]]
    sizes: list[int] = []
    for closes, piece in _pieces(full, limit, segments):
        chunks[-1].append(piece)
        if closes:
            sizes.append(piece.offset + piece.nbytes)
            chunks.append([])
    if chunks[-1]:
        last = chunks[-1][-1]
        sizes.append(last.offset + last.nbytes)
    else:
        chunks.pop()
    return ChunkLayout(chunks, sizes, full)


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
    layout: ChunkLayout | None = None,
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
        layout: `chunk_layout` of what `iterator` yields, for a caller that
            sends the same parameters again: its buffers and views are built
            once, so a chunk then costs no per-tensor Python. Each tensor is
            checked against it.
        segments: split tensors over the buffer along dim 0. Both sides
            must agree (`chunk_layout`); ignored when `layout` is given.

    """
    tensors = (post_iter_func(item) for item in iterator)
    streams = _get_streams(torch.accelerator.current_device_index(), num_buffers)
    caller_stream = torch.accelerator.current_stream()
    done: list[torch.Event | None] = [None] * num_buffers
    held: list[Any] = [None] * num_buffers  # alive until their broadcast ends
    for k, (buf, views, srcs) in enumerate(
        _chunks(tensors, buffer_size_bytes, num_buffers, layout, segments, group)
    ):
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


def _check_spec(layout: ChunkLayout, index: int, tensor: torch.Tensor) -> None:
    shape, dtype = layout.specs[index]
    if tuple(tensor.shape) != shape or tensor.dtype != dtype:
        raise ValueError(
            f"packed transfer: tensor {index} is {list(tensor.shape)} {tensor.dtype}, "
            f"but the layout it is sent with expects {list(shape)} {dtype}"
        )


def _chunks(
    tensors: Iterator[torch.Tensor],
    limit: int,
    num_buffers: int,
    layout: ChunkLayout | None,
    segments: bool,
    group: Any,
) -> Iterator[tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]]:
    """(buffer, piece views, piece sources) per chunk."""
    if layout is not None:
        bufs = slot_buffers(num_buffers, max(layout.sizes, default=0), group)
        bound = layout.__dict__.get("_bound")
        if bound is None or bound[0] is not bufs:
            bound = layout.__dict__["_bound"] = (
                bufs,
                [
                    layout.views(k, bufs[k % num_buffers])
                    for k in range(len(layout.sizes))
                ],
            )
        index, tensor = -1, torch.empty(0)
        for k, pieces in enumerate(layout.chunks):
            srcs = []
            for piece, view in zip(pieces, bound[1][k]):
                if piece.index != index:
                    index = piece.index
                    tensor = next(tensors, None)
                    if tensor is None:
                        raise ValueError(
                            f"packed transfer: {index} tensors sent, but the "
                            f"layout has {len(layout.specs)}"
                        )
                    _check_spec(layout, index, tensor)
                srcs.append(_source(piece, tensor, view))
            yield bufs[k % num_buffers][: layout.sizes[k]], bound[1][k], srcs
        if next(tensors, None) is not None:
            raise ValueError(
                f"packed transfer: more tensors sent than the layout's "
                f"{len(layout.specs)}"
            )
        return
    stash: dict[int, torch.Tensor] = {}

    def specs() -> Iterator[tuple[Sequence[int], torch.dtype]]:
        for i, t in enumerate(tensors):
            stash[i] = t
            yield t.shape, t.dtype

    cur: list[Piece] = []
    for closes, piece in _pieces(specs(), limit, segments):
        cur.append(piece)
        if closes:
            yield _fresh_chunk(cur, stash)
            cur = []
    if cur:
        yield _fresh_chunk(cur, stash)


def _fresh_chunk(pieces: list[Piece], stash: dict[int, torch.Tensor]):
    one = ChunkLayout([pieces], [pieces[-1].offset + pieces[-1].nbytes])
    device = stash[pieces[0].index].device
    buf = torch.empty(one.sizes[0], dtype=torch.uint8, device=device)
    views = one.views(0, buf)
    srcs = [_source(p, stash[p.index], v) for p, v in zip(pieces, views)]
    for i in {p.index for p in pieces} - {pieces[-1].index}:
        stash.pop(i, None)  # the last tensor may continue in the next chunk
    return buf, views, srcs


# Per communicator: two transfers that may run at once in one process (a
# sender and a receiver sharing a process, or two groups) never share buffers.
_SLOTS: "weakref.WeakKeyDictionary[Any, dict[tuple[int, int], list[torch.Tensor]]]"
_SLOTS = weakref.WeakKeyDictionary()


def slot_buffers(num_buffers: int, nbytes: int, group: Any) -> list[torch.Tensor]:
    """`group`'s reused packed buffers on this device, at least `nbytes` each.
    Shared by every transfer on that communicator (one set of buffers however
    many senders take turns on it). They grow in place and
    `release_slot_buffers` frees their memory between updates: views of them
    stay valid either way."""
    index = torch.accelerator.current_device_index()
    per_group = _SLOTS.setdefault(group, {})
    key = (index, num_buffers)
    if key not in per_group:
        accelerator = torch.accelerator.current_accelerator()
        assert accelerator is not None
        device = torch.device(accelerator.type, index)
        per_group[key] = [
            torch.empty(0, dtype=torch.uint8, device=device) for _ in range(num_buffers)
        ]
    bufs = per_group[key]
    for buf in bufs:
        if buf.numel() < nbytes:
            buf.resize_(nbytes)
        if buf.untyped_storage().nbytes() != buf.numel():
            buf.untyped_storage().resize_(buf.numel())
    return bufs


def release_slot_buffers(group: Any = None) -> None:
    """Free the memory of `group`'s buffers (every group's if `None`)."""
    groups = list(_SLOTS.values()) if group is None else [_SLOTS.get(group, {})]
    for per_group in groups:
        for bufs in per_group.values():
            for buf in bufs:
                buf.untyped_storage().resize_(0)


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
    layout = chunk_layout(specs, buffer_size_bytes, segments)
    streams = _get_streams(torch.accelerator.current_device_index(), num_buffers)
    caller_stream = torch.accelerator.current_stream()
    for stream in streams:
        stream.wait_stream(caller_stream)
    prev_received: torch.Event | None = None
    slot_done: list[torch.Event | None] = [None] * num_buffers
    joined: dict[int, torch.Tensor] = {}  # a split tensor, filled as it arrives

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
                index, rows, dtype = piece.index, piece.rows, piece.dtype
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
    accelerator = torch.accelerator.current_accelerator()
    assert accelerator is not None
    device = torch.device(accelerator.type, torch.accelerator.current_device_index())
    ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device=device)
    _, ipc_args = reduce_tensor(ipc_buffer)

    pieces: list[Piece] = []
    names: list[str] = []
    total_bytes = 0

    def chunk() -> PackedIpcChunk:
        # Drain queued copies so the consumer sees a fully-written buffer.
        torch.accelerator.current_stream().synchronize()
        return PackedIpcChunk(
            names=names,
            shapes=[p.shape for p in pieces],
            dtype_names=[str(p.dtype).split(".")[-1] for p in pieces],
            tensor_sizes=[p.nbytes for p in pieces],
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
            pieces, names, total_bytes = [], [], 0
        # Shape and dtype as sent (the original tensor's), bytes back to back.
        piece = Piece(
            len(pieces),
            None,
            list(orig_tensor.shape),
            orig_tensor.dtype,
            total_bytes,
            nbytes,
        )
        ipc_buffer[total_bytes : total_bytes + nbytes].copy_(flat)
        pieces.append(piece)
        names.append(name)
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

    # The chunk's tensors, back to back as the producer packed them.
    pieces, offset = [], 0
    for i, (shape, dtype_name, nbytes) in enumerate(
        zip(shapes, dtype_names, tensor_sizes, strict=True)
    ):
        dtype = getattr(torch, dtype_name)
        pieces.append(Piece(i, None, list(shape), dtype, offset, nbytes))
        offset += nbytes
    layout = ChunkLayout([pieces], [offset])
    # Every tensor gets its own storage (see above). One at an offset its
    # dtype cannot view (#64936) is copied into aligned storage, its own.
    out = []
    buffer_ptr = packed.untyped_storage().data_ptr()
    for name, piece, view in zip(names, pieces, layout.views(0, packed)):
        t = _unpacked(view, piece)
        if t.untyped_storage().data_ptr() == buffer_ptr:
            t = t.clone()
        out.append((name, t))
    return out
