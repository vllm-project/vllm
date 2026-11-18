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

# Tensor alignment in the stream, so a whole tensor can be viewed as its dtype.
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
    """Lazily cut the `ALIGN`-aligned byte stream of `specs` into chunks of
    `buffer_size` bytes, splitting tensors that cross a cut.

    Both sides derive it from metadata alone. A full chunk is yielded before
    the next spec is pulled; trailing padding is not counted in `nbytes`."""
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
    """Destinations and sources of one `_foreach_copy_` packing `pieces` into
    `buf`. Each tensor is dropped from `tensors` at its last piece."""
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
            # keep dropped sources alive until the copies end
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

    Whole tensors are views of the received buffer; split ones are copied
    into their own storage.

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
    *recv_streams, unpack_stream = _get_streams(
        torch.accelerator.current_device_index(), num_buffers + 1
    )
    caller_stream = torch.accelerator.current_stream()
    # Unpacking and callbacks run in chunk order on one stream, so a callback
    # may keep tensors and read them from a later callback.
    unpack_stream.wait_stream(caller_stream)
    unpacked: list[torch.Event | None] = [None] * num_buffers  # per slot
    joined: dict[int, torch.Tensor] = {}  # a split tensor, filled as it arrives

    for k, (chunk_nbytes, pieces) in enumerate(chunks(specs, buffer_size_bytes)):
        slot = k % num_buffers
        # Wait for this slot's previous chunk. Not a stream synchronize(): on
        # ROCm, streams can share a hardware queue.
        done = unpacked[slot]
        if done is not None:
            done.synchronize()
        with torch.cuda.stream(recv_streams[slot]):
            packed = torch.empty(chunk_nbytes, dtype=torch.uint8, device=device)
            if chunk_nbytes:  # a chunk of only empty tensors is not sent
                group.broadcast(packed, src=src)
            received = torch.Event()
            received.record()
        # Read on the unpack stream and by the caller after return.
        packed.record_stream(unpack_stream)
        packed.record_stream(caller_stream)
        with torch.cuda.stream(unpack_stream):
            unpack_stream.wait_event(received)
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
                    tensor.record_stream(caller_stream)  # as `packed`
                out.append((names[p.position], tensor))
            if out:
                post_unpack_func(out)
            event = torch.Event()
            event.record()
            unpacked[slot] = event

    unpack_stream.synchronize()


# ── IPC packed transfer ────────────────────────────────────────────────


@dataclass(slots=True)
class PackedIpcChunk:
    """One chunk of a packed IPC transfer, naming the tensors that start in it."""

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

    One buffer of `buffer_size_bytes` is shared for IPC once and refilled per
    chunk; a tensor larger than it is split across chunks.

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

    Tensors are copied out of the shared buffer, since the producer reuses it
    and callers may keep them. Use one consumer per transfer.

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
        # A full chunk needs no spec past those received; running out means
        # this is the last chunk.
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
