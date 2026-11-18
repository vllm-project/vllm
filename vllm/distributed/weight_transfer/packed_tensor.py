# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, NamedTuple

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

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


# ── Packed layout (shared by NCCL and IPC) ─────────────────────────────

# Tensor alignment in the stream, so a whole tensor can be viewed as its dtype.
ALIGN = 16


def chunk_capacity(buffer_size: int) -> int:
    """The largest chunk, in bytes, of a `buffer_size` packed buffer."""
    capacity = round_down(buffer_size, ALIGN)
    if capacity < ALIGN:
        raise ValueError(
            f"packed buffer size must be at least {ALIGN} bytes, got {buffer_size}"
        )
    return capacity


def fit_buffer(
    buf: torch.Tensor | None, nbytes: int, capacity: int, device: torch.device
) -> torch.Tensor:
    """`buf`, or a new buffer if it holds fewer than `nbytes`; a full chunk gets
    `capacity` bytes, so it is reallocated at most once per transfer."""
    if buf is not None and buf.numel() >= nbytes:
        return buf
    size = capacity if nbytes > capacity - ALIGN else nbytes
    return torch.empty(size, dtype=torch.uint8, device=device)


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
    `chunk_capacity(buffer_size)` bytes, splitting tensors that cross a cut.

    A full chunk is yielded before the next spec is pulled; trailing padding is
    not counted in `nbytes`."""
    limit = chunk_capacity(buffer_size)
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


class _Unpacker:
    """Rebuilds the tensors of a packed transfer, one chunk at a time.

    Split tensors are copied into their own storage; whole ones too unless
    `views`, which returns them as views of the chunk buffer."""

    def __init__(self, buffer_size: int, views: bool) -> None:
        self.names: list[str] = []
        self.specs: list[tuple[Sequence[int], torch.dtype]] = []
        # A full chunk pulls no spec past it, so specs may arrive per chunk.
        self.layout = chunks(iter(self.specs), buffer_size)
        self._views = views
        self._partial: dict[int, torch.Tensor] = {}  # split, filled as it arrives

    @property
    def incomplete(self) -> bool:
        """A split tensor still misses bytes."""
        return bool(self._partial)

    def unpack(
        self, buf: torch.Tensor, pieces: list[Piece]
    ) -> list[tuple[str, torch.Tensor]]:
        """Unpack a chunk of `layout` from `buf`; returns the tensors it ends."""
        out, dsts, srcs = [], [], []
        for p in pieces:
            shape, dtype = self.specs[p.position]
            nbytes = _nbytes(shape, dtype)
            raw = buf[p.offset : p.offset + p.end - p.start]
            if self._views and p.start == 0 and p.end == nbytes:
                out.append((self.names[p.position], raw.view(dtype).view(shape)))
                continue
            t = self._partial.get(p.position)
            if t is None:
                t = torch.empty(shape, dtype=dtype, device=buf.device)
                self._partial[p.position] = t
            if p.end > p.start:
                dsts.append(t.view(-1).view(torch.uint8)[p.start : p.end])
                srcs.append(raw)
            if p.end == nbytes:
                out.append((self.names[p.position], self._partial.pop(p.position)))
        if dsts:
            torch._foreach_copy_(dsts, srcs)
        return out


# ── NCCL packed broadcast ──────────────────────────────────────────────


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
        buffer_size_bytes: Chunk size in bytes. Both sides must agree on it and
                          on `num_buffers`.
        num_buffers: Number of buffers for double/triple buffering.

    """
    tensors: dict[int, torch.Tensor] = {}  # each tensor, until all is copied

    def specs() -> Iterator[tuple[Sequence[int], torch.dtype]]:
        for i, item in enumerate(iterator):
            t = post_iter_func(item)
            tensors[i] = t
            yield t.shape, t.dtype

    capacity = chunk_capacity(buffer_size_bytes)
    streams = _get_streams(torch.accelerator.current_device_index(), num_buffers)
    caller_stream = torch.accelerator.current_stream()
    done: list[torch.Event | None] = [None] * num_buffers
    bufs: list[torch.Tensor | None] = [None] * num_buffers  # sized to the chunks
    for k, (nbytes, pieces) in enumerate(chunks(specs(), capacity)):
        slot = k % num_buffers
        if not nbytes:  # only empty tensors: nothing to send
            _pack_copies(None, pieces, tensors)
            continue
        prev = done[slot]
        if prev is not None:
            prev.synchronize()
        device = tensors[pieces[0].position].device
        buf = bufs[slot] = fit_buffer(bufs[slot], nbytes, capacity, device)
        dsts, srcs = _pack_copies(buf, pieces, tensors)
        # the sources may come from work the caller queued while iterating
        streams[slot].wait_stream(caller_stream)
        with torch.cuda.stream(streams[slot]):
            torch._foreach_copy_(dsts, srcs)
            # keep dropped sources alive until the copies end
            for t in srcs:
                t.record_stream(streams[slot])
            del dsts, srcs
            group.broadcast(buf[:nbytes], src=src)
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
        buffer_size_bytes: Chunk size in bytes, as on the producer.
        num_buffers: Number of buffers, as on the producer.
        device: Device of the NCCL communicator (the worker's device).

    """
    unpacker = _Unpacker(buffer_size_bytes, views=True)
    for name, spec in iterator:
        unpacker.names.append(name)
        unpacker.specs.append(spec)
    *recv_streams, unpack_stream = _get_streams(
        torch.accelerator.current_device_index(), num_buffers + 1
    )
    caller_stream = torch.accelerator.current_stream()
    # Unpacking and callbacks run in chunk order on one stream, so a callback
    # may keep tensors and read them from a later callback.
    unpack_stream.wait_stream(caller_stream)
    unpacked: list[torch.Event | None] = [None] * num_buffers  # per slot

    for k, (chunk_nbytes, pieces) in enumerate(unpacker.layout):
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
            out = unpacker.unpack(packed, pieces)
            for _, t in out:
                if t._base is None:  # joined from a split, allocated here
                    t.record_stream(caller_stream)
            if out:
                post_unpack_func(out)
            event = torch.Event()
            event.record()
            unpacked[slot] = event

    unpack_stream.synchronize()


# ── IPC packed transfer ────────────────────────────────────────────────


def _local_rebuild_args(
    ipc_handle: dict[str, tuple], gpu_uuid: str, device_index: int
) -> list:
    """`rebuild_cuda_tensor` args of this GPU's entry, on `device_index`."""
    if gpu_uuid not in ipc_handle:
        raise ValueError(
            f"IPC handle not found for GPU UUID {gpu_uuid}. "
            f"Available UUIDs: {list(ipc_handle.keys())}"
        )
    args = list(ipc_handle[gpu_uuid])
    args[6] = device_index  # the exporter's device index
    return args


@dataclass(slots=True)
class PackedIpcChunk:
    """One chunk of a packed IPC transfer, naming the tensors that start in it.
    Only the first chunk carries the buffer's `ipc_handle`."""

    names: list[str]
    shapes: list[list[int]]
    dtype_names: list[str]
    ipc_handle: dict[str, tuple] | None


def packed_ipc_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    gpu_uuid: str,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
) -> Iterator[PackedIpcChunk]:
    """Pack tensors into one IPC buffer, refilled and yielded per chunk.

    The consumer must have read a chunk before the generator is resumed.

    Args:
        iterator: Iterator of (name, tensor) pairs.
        gpu_uuid: Physical GPU UUID string for this rank.
        post_iter_func: Applied to each (name, tensor) before packing.
        buffer_size_bytes: Size of the IPC buffer, the chunk size.

    """
    accelerator = torch.accelerator.current_accelerator()
    assert accelerator is not None
    device = torch.device(accelerator.type, torch.accelerator.current_device_index())
    buf = torch.empty(
        chunk_capacity(buffer_size_bytes), dtype=torch.uint8, device=device
    )
    ipc_handle: dict[str, tuple] | None = {gpu_uuid: reduce_tensor(buf)[1]}

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
        torch.accelerator.current_stream().synchronize()  # before the consumer reads
        yield PackedIpcChunk(
            names=[name for name, _ in started],
            shapes=[list(t.shape) for _, t in started],
            dtype_names=[str(t.dtype).removeprefix("torch.") for _, t in started],
            ipc_handle=ipc_handle,
        )
        ipc_handle = None
        started.clear()


class PackedIpcConsumer:
    """Receives one packed IPC transfer, a chunk per `receive` call.

    The producer's buffer is mapped at the first chunk and released by
    `close()`. Tensors are copied out, since the producer reuses the buffer."""

    def __init__(self, device_index: int, gpu_uuid: str | None = None) -> None:
        self._device_index = device_index
        if gpu_uuid is None:
            gpu_uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        self._gpu_uuid = gpu_uuid
        self._buf: torch.Tensor | None = None
        self._unpacker: _Unpacker | None = None

    def receive(
        self,
        ipc_handle: dict[str, tuple] | None,
        names: list[str],
        shapes: list[list[int]],
        dtype_names: list[str],
    ) -> list[tuple[str, torch.Tensor]]:
        """Unpack the next chunk; returns the tensors it completes.

        Args:
            ipc_handle: GPU UUID to `rebuild_cuda_tensor` args; read on the
                first chunk only.
            names: Names of the tensors that start in this chunk
            shapes: Their shapes
            dtype_names: Their dtype name strings (e.g. "float16")

        """
        if self._buf is None:
            if not ipc_handle:
                raise ValueError("the first packed IPC chunk must carry a handle")
            args = _local_rebuild_args(ipc_handle, self._gpu_uuid, self._device_index)
            self._buf = rebuild_cuda_tensor(*args)
            # the producer's buffer is exactly the chunk size
            self._unpacker = _Unpacker(self._buf.numel(), views=False)
        assert self._unpacker is not None
        self._unpacker.names += names
        for shape, dtype_name in zip(shapes, dtype_names, strict=True):
            self._unpacker.specs.append((shape, getattr(torch, dtype_name)))
        chunk = next(self._unpacker.layout, None)
        if chunk is None:
            raise ValueError("received a chunk past the end of the packed layout")
        out = self._unpacker.unpack(self._buf, chunk.pieces)
        # The producer overwrites the buffer once this call returns.
        torch.accelerator.current_stream(self._buf.device).synchronize()
        return out

    def close(self) -> None:
        """Release the buffer; raises if the transfer stopped inside a tensor."""
        self._buf = None
        if self._unpacker is not None and self._unpacker.incomplete:
            raise RuntimeError("packed IPC transfer ended inside a split tensor")
