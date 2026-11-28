# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import itertools
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, NamedTuple

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from vllm.device_allocator.alloc_conf import non_expandable_allocations
from vllm.utils.math_utils import round_down, round_up

# Default values for packed tensor transfer.
DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB
DEFAULT_PACKED_NUM_BUFFERS = 2
# Smallest buffer `grow_buffer` allocates (unless the capacity is smaller), so a
# large transfer grows each buffer only a few times.
MIN_PACKED_BUFFER_BYTES = 64 * 1024 * 1024


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


def _bytes_end(pieces: list[Piece]) -> int:
    """Where the bytes of `pieces` end in their chunk; 0 if they have none."""
    return max(
        (p.offset + p.end - p.start for p in pieces if p.end > p.start), default=0
    )


def grow_buffer(
    buf: torch.Tensor | None,
    nbytes: int,
    capacity: int,
    device: torch.device,
    keep: int = 0,
) -> torch.Tensor:
    """`buf` if it holds `nbytes`, else a new buffer with its first `keep` bytes:
    `buf`'s size (at first, `MIN_PACKED_BUFFER_BYTES`) doubled to hold `nbytes`,
    at most `capacity`."""
    if buf is not None and buf.numel() >= nbytes:
        return buf
    if buf is None:
        size = max(MIN_PACKED_BUFFER_BYTES, round_up(nbytes, ALIGN))
    else:
        size = buf.numel()
    while size < nbytes:
        size *= 2
    size = min(size, capacity)
    grown = torch.empty(size, dtype=torch.uint8, device=device)
    if keep:
        assert buf is not None
        grown[:keep].copy_(buf[:keep])
    return grown


def _record_sources(
    srcs: list[torch.Tensor], device: torch.device, stream: torch.cuda.Stream
) -> None:
    """Keep GPU `srcs` alive until their copies end: on `stream` on `device`, on
    the source device's current stream (where `copy_` runs) on another GPU."""
    for t in srcs:
        if t.device == device:
            t.record_stream(stream)
        elif t.device.type != "cpu":  # host copies are synchronous
            t.record_stream(torch.cuda.current_stream(t.device))


def dtype_name(dtype: torch.dtype) -> str:
    """The wire name of `dtype`, e.g. "float16" (`getattr(torch, name)`)."""
    return str(dtype).removeprefix("torch.")


def chunks(
    specs: Iterable[tuple[Sequence[int], torch.dtype]],
    buffer_size: int,
    before_pull: Callable[[list[Piece]], None] | None = None,
) -> Iterator[Chunk]:
    """Lazily cut the `ALIGN`-aligned byte stream of `specs` into chunks of
    `chunk_capacity(buffer_size)` bytes, splitting tensors that cross a cut.
    A full chunk is yielded before the next spec is pulled, and `before_pull`
    gets each piece once, before the next pull and before its chunk is yielded."""
    limit = chunk_capacity(buffer_size)
    pieces: list[Piece] = []
    seen = 0  # pieces passed to `before_pull`
    used = 0  # bytes of the current chunk taken
    specs = iter(specs)
    for position in itertools.count():
        if before_pull is not None and seen < len(pieces):
            before_pull(pieces[seen:])
        seen = len(pieces)
        spec = next(specs, None)
        if spec is None:
            break
        shape, dtype = spec
        nbytes, start = _nbytes(shape, dtype), 0
        if nbytes == 0:  # takes no bytes, so no padding either
            pieces.append(Piece(position, 0, 0, round_up(used, ALIGN)))
            continue
        used = round_up(used, ALIGN)
        while start < nbytes:
            end = min(nbytes, start + limit - used)
            pieces.append(Piece(position, start, end, used))
            used += end - start
            start = end
            if round_up(used, ALIGN) == limit:  # full, or only padding fits
                if before_pull is not None:
                    before_pull(pieces[seen:])
                yield Chunk(used, pieces, full=True)
                pieces, seen, used = [], 0, 0
    if pieces:
        yield Chunk(used, pieces, full=False)


def _pack_copies(
    buf: torch.Tensor | None, pieces: list[Piece], tensors: dict[int, torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Destinations and sources of one `_foreach_copy_` packing `pieces` into
    `buf`, as bytes: CUDA copies lack some dtypes (float8_e8m0fnu). Each tensor
    is dropped from `tensors` at its last piece."""
    dsts, srcs = [], []
    for p in pieces:
        t = tensors[p.position].detach()  # a parameter: copy without autograd
        if t.dtype.itemsize == 1:  # float8/float4 too, also strided
            t = t.view(torch.uint8)
        if p.end > p.start:
            assert buf is not None
            dst = buf[p.offset : p.offset + p.end - p.start]
            if p.start == 0 and p.end == t.nbytes and not t.is_contiguous():
                # typed, without a contiguous temporary; its strides keep it
                # off the `_foreach_copy_` fast path
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
        # A full chunk pulls no spec past it, so specs may be fed per chunk: a
        # list iterator sees what is appended before it is exhausted.
        self.layout = chunks(self.specs, buffer_size)
        self._views = views
        self._partial: torch.Tensor | None = None  # filled as it arrives

    def feed(
        self, names: list[str], specs: Iterable[tuple[Sequence[int], torch.dtype]]
    ) -> None:
        """Append the tensors that start in the next chunks."""
        self.names += names
        self.specs += specs

    def next_chunk(self) -> Chunk:
        chunk = next(self.layout, None)
        if chunk is None:
            raise ValueError("received a chunk past the end of the packed layout")
        return chunk

    @property
    def incomplete(self) -> bool:
        """A split tensor still misses bytes."""
        return self._partial is not None

    def unpack(
        self, buf: torch.Tensor, pieces: list[Piece]
    ) -> list[tuple[str, torch.Tensor]]:
        """Unpack a chunk of `layout` from `buf`; returns the tensors it ends."""
        out, dsts, srcs = [], [], []
        for p in pieces:
            shape, dtype = self.specs[p.position]
            nbytes = _nbytes(shape, dtype)
            raw = buf[p.offset : p.offset + p.end - p.start]
            if self._views and p.start == 0 and p.end == nbytes > 0:
                out.append((self.names[p.position], raw.view(dtype).view(shape)))
                continue
            if p.start == 0:  # the layout ends a tensor before the next
                self._partial = torch.empty(shape, dtype=dtype, device=buf.device)
            t = self._partial
            assert t is not None
            if p.end > p.start:
                dsts.append(t.view(-1).view(torch.uint8)[p.start : p.end])
                srcs.append(raw)
            if p.end == nbytes:
                out.append((self.names[p.position], t))
                self._partial = None
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
    """Broadcast tensors in a packed manner from trainer to workers. A tensor
    is copied before the next is pulled, so the iterator may reuse one tensor
    for consecutive parameters.

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
    device = group.device
    streams = _get_streams(device.index, num_buffers)
    caller_stream = torch.cuda.current_stream(device)
    done: list[torch.Event | None] = [None] * num_buffers
    bufs: list[torch.Tensor | None] = [None] * num_buffers
    k = 0  # the chunk being packed
    filled = 0  # the bytes of its copied pieces end at

    def pack(pieces: list[Piece]) -> None:
        nonlocal filled
        end = _bytes_end(pieces)
        if end == 0:
            _pack_copies(None, pieces, tensors)
            return
        slot = k % num_buffers
        prev = done[slot]
        if prev is not None:
            prev.synchronize()
            done[slot] = None
        # On the caller stream, so the source's next writes wait for the copies.
        with torch.cuda.stream(caller_stream):
            buf = bufs[slot] = grow_buffer(bufs[slot], end, capacity, device, filled)
            dsts, srcs = _pack_copies(buf, pieces, tensors)
            torch._foreach_copy_(dsts, srcs)
            _record_sources(srcs, buf.device, caller_stream)
        filled = end

    for chunk in chunks(specs(), buffer_size_bytes, before_pull=pack):
        slot = k % num_buffers
        buf = bufs[slot]
        if chunk.sent:
            assert buf is not None
            streams[slot].wait_stream(caller_stream)
            buf.record_stream(streams[slot])
            with torch.cuda.stream(streams[slot]):
                group.broadcast(buf[: chunk.nbytes], src=src)
                event = torch.Event()
                event.record(streams[slot])
            done[slot] = event
        k, filled = k + 1, 0
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

    def load(packed: torch.Tensor, pieces: list[Piece]) -> None:
        # A function: the chunk's tensors are released before the next one's.
        out = unpacker.unpack(packed, pieces)
        for _, t in out:
            if t._base is None:  # joined from a split, allocated here
                t.record_stream(caller_stream)
        if out:
            post_unpack_func(out)

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
            load(packed, chunk.pieces)
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
    Only the first chunk carries `chunk_capacity` and the buffer's `ipc_handle`
    (none if the transfer has no bytes); only the last one has `last` set."""

    names: list[str]
    shapes: list[list[int]]
    dtype_names: list[str]
    ipc_handle: dict[str, tuple] | None
    chunk_capacity: int | None
    last: bool


def packed_ipc_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    gpu_uuid: str,
    device_index: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
) -> Iterator[PackedIpcChunk]:
    """Pack tensors into one IPC buffer, refilled and yielded per chunk.

    The consumer must have read a chunk before the generator is resumed. A
    tensor is copied before the next is pulled, so the iterator may reuse one
    tensor for consecutive parameters.

    Args:
        iterator: Iterator of (name, tensor) pairs.
        gpu_uuid: Physical GPU UUID string of `device_index`.
        device_index: The CUDA device the IPC buffer is allocated on.
        post_iter_func: Applied to each (name, tensor) before packing.
        buffer_size_bytes: Chunk size; the IPC buffer grows up to it while the
            first chunk is packed.

    """
    capacity = chunk_capacity(buffer_size_bytes)
    device = torch.device("cuda", device_index)
    caller_stream = torch.cuda.current_stream(device)
    buf: torch.Tensor | None = None  # at the first byte: none for no bytes
    filled = 0  # the bytes of the current chunk's copied pieces end at
    items = iter(iterator)
    # pulled to see if a chunk is last; None once `items` is exhausted
    ahead: list[tuple[str, torch.Tensor] | None] = []

    tensors: dict[int, torch.Tensor] = {}  # each tensor, until all is copied
    started: list[tuple[str, list[int], str]] = []  # since the last chunk

    def specs() -> Iterator[tuple[Sequence[int], torch.dtype]]:
        for i in itertools.count():
            item = ahead.pop() if ahead else next(items, None)
            if item is None:
                return
            t = post_iter_func(item)
            tensors[i] = t
            started.append((item[0], list(t.shape), dtype_name(t.dtype)))
            yield t.shape, t.dtype

    def pack(pieces: list[Piece]) -> None:
        nonlocal buf, filled
        end = _bytes_end(pieces)
        with torch.cuda.stream(caller_stream):
            if end:
                # The consumer opens this buffer over CUDA IPC, which for an
                # expandable segment needs pidfd_getfd, i.e. ptrace permission.
                with non_expandable_allocations():
                    buf = grow_buffer(buf, end, capacity, device, filled)
                filled = end
            dsts, srcs = _pack_copies(buf, pieces, tensors)
            if dsts:
                assert buf is not None
                torch._foreach_copy_(dsts, srcs)
                _record_sources(srcs, buf.device, caller_stream)

    for k, chunk in enumerate(chunks(specs(), buffer_size_bytes, before_pull=pack)):
        caller_stream.synchronize()  # before the consumer reads
        # the last chunk ends a tensor, and no item follows
        last = chunk.pieces[-1].position not in tensors
        if last and chunk.full:
            ahead.append(next(items, None))
            last = ahead[0] is None
        handle = None
        if k == 0 and buf is not None:  # only the first chunk grows `buf`
            handle = {gpu_uuid: reduce_tensor(buf)[1]}
        yield PackedIpcChunk(
            names=[name for name, _, _ in started],
            shapes=[shape for _, shape, _ in started],
            dtype_names=[dtype for _, _, dtype in started],
            ipc_handle=handle,
            chunk_capacity=None if k else capacity,
            last=last,
        )
        filled = 0
        started.clear()


class PackedIpcConsumer:
    """Receives packed IPC transfers, a chunk per `receive` call.

    The first chunk of a transfer maps the producer's buffer, if it has bytes;
    `close()` releases it and raises if a transfer did not reach its last
    chunk."""

    def __init__(self, device_index: int, gpu_uuid: str) -> None:
        self._device_index = device_index
        self._gpu_uuid = gpu_uuid
        self._buf: torch.Tensor | None = None
        self._unpacker: _Unpacker | None = None  # of the transfer in progress

    def receive(
        self,
        ipc_handle: dict[str, tuple] | None,
        names: list[str],
        shapes: list[list[int]],
        dtype_names: list[str],
        chunk_capacity: int | None,
        last: bool,
    ) -> list[tuple[str, torch.Tensor]]:
        """Unpack the next chunk; returns the tensors it completes, copied out
        on the current stream. The caller must synchronize it before the
        producer refills the buffer.

        Args:
            ipc_handle: GPU UUID to `rebuild_cuda_tensor` args; set on the
                first chunk of a transfer with bytes only.
            names: Names of the tensors that start in this chunk
            shapes: Their shapes
            dtype_names: Their dtype names (e.g. "float16")
            chunk_capacity: The chunk size in bytes; set on the first chunk.
            last: Whether this chunk ends the transfer.

        """
        if self._unpacker is None:  # the first chunk of a transfer
            assert chunk_capacity is not None
            if ipc_handle:
                args = local_rebuild_args(
                    ipc_handle, self._gpu_uuid, self._device_index
                )
                self._buf = rebuild_cuda_tensor(*args)
            else:  # no bytes
                self._buf = torch.empty(
                    0, dtype=torch.uint8, device=f"cuda:{self._device_index}"
                )
            self._unpacker = _Unpacker(chunk_capacity, views=False)
        assert self._buf is not None
        unpacker = self._unpacker
        unpacker.feed(
            names,
            [
                (shape, getattr(torch, name))
                for shape, name in zip(shapes, dtype_names, strict=True)
            ],
        )
        chunk = unpacker.next_chunk()
        out = unpacker.unpack(self._buf, chunk.pieces)
        if last:
            if unpacker.incomplete or next(unpacker.layout, None) is not None:
                raise RuntimeError("packed IPC transfer ended inside its layout")
            self._unpacker = None
        return out

    def close(self) -> None:
        """Release the buffer; raises if a transfer did not reach its last
        chunk."""
        self._buf = None
        if self._unpacker is not None:
            self._unpacker = None
            raise RuntimeError("packed IPC transfer ended before its last chunk")
