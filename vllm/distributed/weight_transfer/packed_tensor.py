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

from vllm.utils.math_utils import round_down, round_up

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

# Every tensor starts at a multiple of this in the stream, so a whole
# tensor in a chunk can be viewed as its dtype.
ALIGN = 16


class Piece(NamedTuple):
    """Bytes `[start, end)` of the tensor at `position` in the sent
    sequence, at `offset` in its chunk."""

    position: int
    start: int
    end: int
    offset: int


def _nbytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    return math.prod(shape) * dtype.itemsize


class Chunk(NamedTuple):
    nbytes: int
    pieces: list[Piece]


def chunks(
    specs: Iterable[tuple[Sequence[int], torch.dtype]], buffer_size: int
) -> Iterator[Chunk]:
    """The chunks `specs` travel as, lazily. The tensors form one byte
    stream, each starting `ALIGN`-aligned, cut every `buffer_size` bytes
    (rounded down to `ALIGN`). A tensor that crosses a cut is split there.
    The producer and every consumer derive it from the metadata alone, so
    they always agree. A chunk's `nbytes` ends at its last byte of data:
    alignment padding at its end is not sent.

    A chunk is yielded as soon as it is full (no room left past the
    alignment), before the next spec is pulled: the specs pulled by then are
    exactly those with bytes in it or before it. The last chunk can have no
    bytes, when the stream ends on a cut and only empty tensors follow."""
    limit = round_down(buffer_size, ALIGN)
    if limit <= 0:
        raise ValueError(
            f"packed buffer size must be at least {ALIGN} bytes, got {buffer_size}"
        )
    pieces: list[Piece] = []
    used = 0  # bytes of the current chunk taken
    for position, (shape, dtype) in enumerate(specs):
        nbytes, start = _nbytes(shape, dtype), 0
        used = round_up(used, ALIGN)
        while True:
            end = min(nbytes, start + limit - used)
            pieces.append(Piece(position, start, end, used))
            used += end - start
            start = end
            if round_up(used, ALIGN) == limit:  # full, or only padding fits
                yield Chunk(used, pieces)
                pieces, used = [], 0
            if start == nbytes:
                break
    if pieces:
        yield Chunk(used, pieces)


def _pack_copies(
    buf: torch.Tensor | None, pieces: list[Piece], tensors: dict[int, torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """The copies that pack a chunk's `pieces` into `buf`, as the
    destinations and sources of one `_foreach_copy_`. `tensors` holds the
    tensors by position; each is dropped at its last piece. A whole tensor
    is copied as is (strided) into a view of its dtype, which its aligned
    offset allows; a split one is replaced by its contiguous bytes at its
    first piece, and its pieces are copied from those."""
    dsts, srcs = [], []
    for p in pieces:
        t = tensors[p.position].detach()  # a parameter: copy without autograd
        if p.end > p.start:
            assert buf is not None
            dst = buf[p.offset : p.offset + p.end - p.start]
            if p.start == 0 and p.end == t.nbytes:
                dsts.append(dst.view(t.dtype).view(t.shape))
                srcs.append(t)
            else:
                t = tensors[p.position] = t.contiguous().view(-1).view(torch.uint8)
                dsts.append(dst)
                srcs.append(t[p.start : p.end])
        if p.end == t.nbytes:
            del tensors[p.position]
    return dsts, srcs


def packed_nccl_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Broadcast tensors in a packed manner from trainer to workers.

    Each chunk (`chunks`) is one `_foreach_copy_` of its pieces into one of
    `num_buffers` reused buffers, then one broadcast.

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
    tensors: dict[int, torch.Tensor] = {}  # each tensor, until all is copied

    def specs() -> Iterator[tuple[Sequence[int], torch.dtype]]:
        for i, item in enumerate(iterator):
            t = post_iter_func(item)
            tensors[i] = t
            yield t.shape, t.dtype

    streams = _get_streams(torch.accelerator.current_device_index(), num_buffers)
    caller_stream = torch.accelerator.current_stream()
    done: list[torch.Event | None] = [None] * num_buffers
    bufs: list[torch.Tensor] = []  # allocated on the first chunk with bytes
    for k, (nbytes, pieces) in enumerate(chunks(specs(), buffer_size_bytes)):
        slot = k % num_buffers
        if nbytes and not bufs:
            device = tensors[pieces[0].position].device
            limit = round_down(buffer_size_bytes, ALIGN)
            bufs = [
                torch.empty(limit, dtype=torch.uint8, device=device)
                for _ in range(num_buffers)
            ]
        dsts, srcs = _pack_copies(bufs[slot] if bufs else None, pieces, tensors)
        if not nbytes:  # only empty tensors: nothing to send
            continue
        prev = done[slot]
        if prev is not None:
            prev.synchronize()
        # the sources may come from work the caller queued while iterating
        streams[slot].wait_stream(caller_stream)
        with torch.cuda.stream(streams[slot]):
            torch._foreach_copy_(dsts, srcs)
            # the sources may be freed once dropped: not reissued until
            # the copies end
            for t in srcs:
                t.record_stream(streams[slot])
            del dsts, srcs
            group.broadcast(bufs[slot][:nbytes], src=src)
            event = torch.Event()
            event.record(streams[slot])
        done[slot] = event
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
) -> None:
    """Consume packed tensors and unpack them into a list of tensors.

    A tensor inside one chunk is handed out as a view of the received buffer;
    one split across chunks is copied into a tensor of its own, handed out
    once its last bytes arrive.

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
    streams = _get_streams(torch.accelerator.current_device_index(), num_buffers)
    caller_stream = torch.accelerator.current_stream()
    for stream in streams:
        stream.wait_stream(caller_stream)
    prev_unpacked: torch.Event | None = None  # the previous chunk's copies
    slot_done: list[torch.Event | None] = [None] * num_buffers
    joined: dict[int, torch.Tensor] = {}  # a split tensor, filled as it arrives

    for k, (chunk_nbytes, pieces) in enumerate(chunks(specs, buffer_size_bytes)):
        slot = k % num_buffers
        # Wait for this slot's previous pack. Not streams[slot].synchronize():
        # on ROCm, streams can share a hardware queue, and synchronizing one
        # stream then also waits for work queued later on the others.
        done = slot_done[slot]
        if done is not None:
            done.synchronize()
        with torch.cuda.stream(streams[slot]):
            packed = torch.empty(chunk_nbytes, dtype=torch.uint8, device=device)
            if chunk_nbytes:  # a chunk of only empty tensors is not sent
                group.broadcast(packed, src=src)
            # post_unpack_func may keep the unpacked tensors and read them
            # from a later chunk's callback, on another slot stream, and a
            # split tensor's earlier bytes are copied on another slot. Waiting
            # for the previous chunk's unpack (its receive and copies) orders
            # every later unpack and callback after them, and record_stream
            # keeps the allocator from reissuing this buffer once it is freed
            # while the broadcast or such a read is still pending.
            if prev_unpacked is not None:
                streams[slot].wait_event(prev_unpacked)
            for stream in (*streams, caller_stream):
                packed.record_stream(stream)
            out = []
            for p in pieces:
                shape, dtype = specs[p.position]
                nbytes = _nbytes(shape, dtype)
                raw = packed[p.offset : p.offset + p.end - p.start]
                if p.start == 0 and p.end == nbytes:
                    tensor = raw.view(dtype).view(shape)
                else:
                    if p.position not in joined:
                        joined[p.position] = torch.empty(
                            shape, dtype=dtype, device=device
                        )
                    whole = joined[p.position]
                    whole.view(-1).view(torch.uint8)[p.start : p.end].copy_(raw)
                    if p.end < nbytes:
                        continue
                    tensor = joined.pop(p.position)
                    # read like the packed buffer: keep it alive on every
                    # stream that may still read it once it is freed
                    for stream in (*streams, caller_stream):
                        tensor.record_stream(stream)
                out.append((names[p.position], tensor))
            prev_unpacked = torch.Event()
            prev_unpacked.record(streams[slot])
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
    """One chunk of a packed IPC transfer (layout: `chunks`, with the IPC
    buffer's size). It names the tensors that start in it, so each
    tensor's metadata is sent once, with the chunk holding its first bytes."""

    names: list[str]
    shapes: list[list[int]]
    dtype_names: list[str]
    ipc_handle: dict[str, tuple]


def packed_ipc_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    gpu_uuid: str,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
) -> Iterator[PackedIpcChunk]:
    """Pack tensors into a reusable IPC buffer and yield handles.

    Allocates a single GPU buffer of `buffer_size_bytes` (rounded down to
    `ALIGN`), the chunk size, and registers it for IPC once via
    `reduce_tensor`. Each chunk's pieces are copied into it before the chunk
    is yielded, so only one IPC-shared allocation exists for the lifetime of
    the transfer. A tensor larger than the buffer is split across chunks.

    Callers **must** ensure the consumer has finished reading the buffer
    (e.g. `ray.get` returned) before resuming the generator for the
    next chunk.

    Args:
        iterator: Iterator of (name, tensor) pairs.
        gpu_uuid: Physical GPU UUID string for this rank.
        post_iter_func: Applied to each (name, tensor) before packing.
        buffer_size_bytes: Size of the reusable IPC buffer, the chunk size.

    """
    accelerator = torch.accelerator.current_accelerator()
    assert accelerator is not None
    device = torch.device(accelerator.type, torch.accelerator.current_device_index())
    buf = torch.empty(
        round_down(buffer_size_bytes, ALIGN), dtype=torch.uint8, device=device
    )
    _, ipc_args = reduce_tensor(buf)

    tensors: dict[int, torch.Tensor] = {}  # each tensor, until all is copied
    started: list[tuple[str, torch.Tensor]] = []  # since the last chunk

    def specs() -> Iterator[tuple[Sequence[int], torch.dtype]]:
        for i, item in enumerate(iterator):
            t = post_iter_func(item)
            tensors[i] = t
            started.append((item[0], t))
            yield t.shape, t.dtype

    for _, pieces in chunks(specs(), buffer_size_bytes):
        dsts, srcs = _pack_copies(buf, pieces, tensors)
        if dsts:  # else only empty tensors
            torch._foreach_copy_(dsts, srcs)
        del dsts, srcs
        # Drain the copies so the consumer sees a fully-written buffer.
        torch.accelerator.current_stream().synchronize()
        yield PackedIpcChunk(
            names=[name for name, _ in started],
            shapes=[list(t.shape) for _, t in started],
            dtype_names=[str(t.dtype).removeprefix("torch.") for _, t in started],
            ipc_handle={gpu_uuid: ipc_args},
        )
        started.clear()


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


class PackedIpcConsumer:
    """Receives one packed IPC transfer, a chunk per `receive` call.

    Each chunk names the tensors that start in it (`PackedIpcChunk`), so
    the specs arrive in order and the layout (`chunks`) is derived as they
    come. Every piece is copied out of the shared buffer into its tensor's
    own storage, allocated at its first piece, and the tensor is handed out
    at its last. Copying is required: the producer reuses its buffer for the
    next chunk, while callers may keep the tensors (layerwise reload buffers
    them for replay). A tensor split across chunks persists across calls,
    so use one consumer per transfer.

    Args:
        importer: Import cache that must be shared across all chunks of one
            export so the buffer mapping is opened and released exactly once
            (see `PackedBufferImporter`). `None` builds one for this consumer.

    """

    def __init__(self, importer: PackedBufferImporter | None = None) -> None:
        self._importer = importer or PackedBufferImporter()
        self._names: list[str] = []
        self._specs: list[tuple[list[int], torch.dtype]] = []
        self._layout: Iterator[Chunk] | None = None
        self._tensors: dict[int, torch.Tensor] = {}  # filled as pieces arrive

    def _received_specs(self) -> Iterator[tuple[list[int], torch.dtype]]:
        # The producer yields a chunk as soon as it is full, before pulling
        # the next spec, so a full chunk's layout needs no spec past those
        # received; running out of specs means the producer's iterator
        # ended, and this is its last chunk.
        i = 0
        while i < len(self._specs):
            yield self._specs[i]
            i += 1

    def receive(
        self,
        ipc_handle: dict[str, tuple],
        names: list[str],
        shapes: list[list[int]],
        dtype_names: list[str],
        device_index: int,
    ) -> list[tuple[str, torch.Tensor]]:
        """Unpack the next chunk; returns the tensors it completes.

        Args:
            ipc_handle: Mapping of GPU UUID to rebuild_cuda_tensor args tuple
            names: Names of the tensors that start in this chunk
            shapes: Their shapes
            dtype_names: Their dtype name strings (e.g. "float16")
            device_index: Local CUDA device index

        """
        props = torch.cuda.get_device_properties(device_index)
        physical_gpu_id = str(props.uuid)
        if physical_gpu_id not in ipc_handle:
            raise ValueError(
                f"IPC handle not found for GPU UUID {physical_gpu_id}. "
                f"Available UUIDs: {list(ipc_handle.keys())}"
            )
        list_args = list(ipc_handle[physical_gpu_id])
        list_args[6] = device_index
        packed = self._importer.rebuild(list_args)

        self._names += names
        for shape, dtype_name in zip(shapes, dtype_names, strict=True):
            self._specs.append((shape, getattr(torch, dtype_name)))
        if self._layout is None:
            # the producer's buffer is exactly the chunk size
            self._layout = chunks(self._received_specs(), packed.numel())
        chunk = next(self._layout, None)
        if chunk is None:
            raise ValueError("received a chunk past the end of the packed layout")
        out = []
        for p in chunk.pieces:
            t = self._tensors.get(p.position)
            if t is None:
                shape, dtype = self._specs[p.position]
                t = torch.empty(shape, dtype=dtype, device=packed.device)
                self._tensors[p.position] = t
            raw = packed[p.offset : p.offset + p.end - p.start]
            t.view(-1).view(torch.uint8)[p.start : p.end].copy_(raw)
            if p.end == t.nbytes:
                out.append((self._names[p.position], self._tensors.pop(p.position)))
        # The producer overwrites the buffer once this call returns.
        torch.accelerator.current_stream(packed.device).synchronize()
        return out
