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
    full: bool
    """At capacity: at most padding fits after its bytes."""

    @property
    def sent(self) -> bool:
        """A chunk of only empty tensors has no bytes and is not sent."""
        return self.nbytes > 0


def fit_buffer(
    buf: torch.Tensor | None, chunk: Chunk, capacity: int, device: torch.device
) -> torch.Tensor:
    """`buf`, or a new buffer if it is too small for `chunk`; a full chunk
    allocates `capacity` bytes, which fits every later chunk."""
    if buf is not None and buf.numel() >= chunk.nbytes:
        return buf
    size = capacity if chunk.full else chunk.nbytes
    return torch.empty(size, dtype=torch.uint8, device=device)


def dtype_name(dtype: torch.dtype) -> str:
    """The wire name of `dtype`, e.g. "float16" (`getattr(torch, name)`)."""
    return str(dtype).removeprefix("torch.")


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
                yield Chunk(used, pieces, full=True)
                pieces, used = [], 0
            if start == nbytes:
                break
    if pieces:
        yield Chunk(used, pieces, full=False)


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
        # A full chunk pulls no spec past it, so specs may be fed per chunk.
        self.layout = chunks(self._fed_specs(), buffer_size)
        self._views = views
        self._partial: dict[int, torch.Tensor] = {}  # split, filled as it arrives

    def feed(
        self, names: list[str], specs: Iterable[tuple[Sequence[int], torch.dtype]]
    ) -> None:
        """Append the tensors that start in the next chunks."""
        self.names += names
        self.specs += specs

    def _fed_specs(self) -> Iterator[tuple[Sequence[int], torch.dtype]]:
        # Ends once every fed spec is pulled, which ends the layout.
        i = 0
        while i < len(self.specs):
            yield self.specs[i]
            i += 1

    def next_chunk(self) -> Chunk:
        chunk = next(self.layout, None)
        if chunk is None:
            raise ValueError("received a chunk past the end of the packed layout")
        return chunk

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
        buffer_size_bytes: Chunk size in bytes; the consumer must use the same.
        num_buffers: Number of chunks in flight; may differ from the
                     consumer's.

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
    try:
        for k, chunk in enumerate(chunks(specs(), buffer_size_bytes)):
            slot = k % num_buffers
            if not chunk.sent:
                _pack_copies(None, chunk.pieces, tensors)
                continue
            prev = done[slot]
            if prev is not None:
                prev.synchronize()
            device = tensors[chunk.pieces[0].position].device
            # the sources may come from work the caller queued while iterating
            streams[slot].wait_stream(caller_stream)
            with torch.cuda.stream(streams[slot]):
                buf = bufs[slot] = fit_buffer(bufs[slot], chunk, capacity, device)
                dsts, srcs = _pack_copies(buf, chunk.pieces, tensors)
                torch._foreach_copy_(dsts, srcs)
                # keep dropped sources alive until the copies end
                for t in srcs:
                    t.record_stream(streams[slot])
                del dsts, srcs
                group.broadcast(buf[: chunk.nbytes], src=src)
                event = torch.Event()
                event.record(streams[slot])
            done[slot] = event
    finally:
        # also on failure: the buffers are freed once the queued work ends
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

    Args:
        iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
        group: Process group (PyNcclCommunicator)
        src: Source rank (0 in current implementation)
        post_unpack_func: Function to apply to each list of (name, tensor) after
                         unpacking
        buffer_size_bytes: Chunk size in bytes, as on the producer.
        num_buffers: Number of chunks in flight; may differ from the
                     producer's.
        device: Device of the NCCL communicator (the worker's device).

    """
    unpacker = _Unpacker(buffer_size_bytes, views=True)
    for name, spec in iterator:
        unpacker.feed([name], [spec])
    *recv_streams, unpack_stream = _get_streams(
        torch.accelerator.current_device_index(), num_buffers + 1
    )
    caller_stream = torch.accelerator.current_stream()
    # Unpacking and callbacks run in chunk order on one stream, so a callback
    # may keep tensors and read them from a later callback.
    unpack_stream.wait_stream(caller_stream)
    unpacked: list[torch.Event | None] = [None] * num_buffers  # per slot

    for k, chunk in enumerate(unpacker.layout):
        slot = k % num_buffers
        # Wait for this slot's previous chunk. Not a stream synchronize(): on
        # ROCm, streams can share a hardware queue.
        done = unpacked[slot]
        if done is not None:
            done.synchronize()
        with torch.cuda.stream(recv_streams[slot]):
            packed = torch.empty(chunk.nbytes, dtype=torch.uint8, device=device)
            if chunk.sent:
                group.broadcast(packed, src=src)
            received = torch.Event()
            received.record()
        # Read on the unpack stream and by the caller after return.
        packed.record_stream(unpack_stream)
        packed.record_stream(caller_stream)
        with torch.cuda.stream(unpack_stream):
            unpack_stream.wait_event(received)
            out = unpacker.unpack(packed, chunk.pieces)
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


def local_rebuild_args(
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

    for chunk in chunks(specs(), buffer_size_bytes):
        dsts, srcs = _pack_copies(buf, chunk.pieces, tensors)
        if chunk.sent:
            torch._foreach_copy_(dsts, srcs)
        del dsts, srcs
        torch.accelerator.current_stream().synchronize()  # before the consumer reads
        yield PackedIpcChunk(
            names=[name for name, _ in started],
            shapes=[list(t.shape) for _, t in started],
            dtype_names=[dtype_name(t.dtype) for _, t in started],
            ipc_handle=ipc_handle,
        )
        ipc_handle = None
        started.clear()


class PackedIpcConsumer:
    """Receives packed IPC transfers, a chunk per `receive` call.

    A chunk carrying a handle starts a transfer and maps the producer's buffer,
    which `close()` releases."""

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
        """Unpack the next chunk; returns the tensors it completes, copied out
        on the current stream. The caller must synchronize it before the
        producer refills the buffer.

        Args:
            ipc_handle: GPU UUID to `rebuild_cuda_tensor` args; set on the
                first chunk of a transfer only.
            names: Names of the tensors that start in this chunk
            shapes: Their shapes
            dtype_names: Their dtype names (e.g. "float16")

        """
        if ipc_handle:
            if self._unpacker is not None and self._unpacker.incomplete:
                raise RuntimeError(
                    "a packed IPC transfer started inside a split tensor of "
                    "the previous one"
                )
            args = local_rebuild_args(ipc_handle, self._gpu_uuid, self._device_index)
            buf = rebuild_cuda_tensor(*args)
            if buf.dtype != torch.uint8 or buf.numel() % ALIGN:
                raise ValueError(
                    f"the packed IPC buffer must be uint8 with a multiple of "
                    f"{ALIGN} bytes, got {buf.dtype} with {buf.numel()} elements"
                )
            self._buf = buf
            # the producer's buffer is exactly the chunk size
            self._unpacker = _Unpacker(buf.numel(), views=False)
        elif self._buf is None:
            raise ValueError("the first packed IPC chunk must carry a handle")
        assert self._unpacker is not None and self._buf is not None
        self._unpacker.feed(
            names,
            [
                (shape, getattr(torch, name))
                for shape, name in zip(shapes, dtype_names, strict=True)
            ],
        )
        return self._unpacker.unpack(self._buf, self._unpacker.next_chunk().pieces)

    def close(self) -> None:
        """Release the buffer; raises if the transfer stopped inside a tensor."""
        self._buf = None
        if self._unpacker is not None and self._unpacker.incomplete:
            raise RuntimeError("packed IPC transfer ended inside a split tensor")
