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


def _chunks(
    specs: Iterable[tuple[Sequence[int], torch.dtype]], buffer_size: int
) -> Iterator[Chunk]:
    """The chunks, lazily. The tensors form one byte stream, each starting
    `ALIGN`-aligned, cut every `buffer_size` bytes (rounded down to `ALIGN`):
    every chunk but the last is that size. A tensor that crosses a cut is
    split there."""
    limit = buffer_size - buffer_size % ALIGN
    assert limit > 0, f"packed buffer size must be at least {ALIGN} bytes"
    pieces: list[Piece] = []
    used = 0  # bytes of the current chunk taken, padding included
    for position, (shape, dtype) in enumerate(specs):
        nbytes, start = _nbytes(shape, dtype), 0
        used += -used % ALIGN
        while True:
            if used == limit:
                yield Chunk(limit, pieces)
                pieces, used = [], 0
            end = min(nbytes, start + limit - used)
            pieces.append(Piece(position, start, end, used))
            used += end - start
            start = end
            if start == nbytes:
                break
    if pieces:
        yield Chunk(used, pieces)


def chunk_layout(
    specs: Iterable[tuple[Sequence[int], torch.dtype]], buffer_size: int
) -> list[Chunk]:
    """The chunks `specs` travel as. The producer and every consumer derive
    it from the metadata alone, so they always agree."""
    return list(_chunks(specs, buffer_size))


def packed_nccl_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Broadcast tensors in a packed manner from trainer to workers.

    Each chunk (`chunk_layout`) is one `_foreach_copy_` of its pieces into a
    buffer, then one broadcast.

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
    flat: dict[int, torch.Tensor] = {}  # each tensor's bytes, until all are sent

    def specs() -> Iterator[tuple[Sequence[int], torch.dtype]]:
        for i, item in enumerate(iterator):
            t = post_iter_func(item)
            flat[i] = t.contiguous().view(-1).view(torch.uint8)
            yield t.shape, t.dtype

    streams = _get_streams(torch.accelerator.current_device_index(), num_buffers)
    caller_stream = torch.accelerator.current_stream()
    done: list[torch.Event | None] = [None] * num_buffers
    held: list[Any] = [None] * num_buffers  # alive until their broadcast ends
    for k, (nbytes, pieces) in enumerate(_chunks(specs(), buffer_size_bytes)):
        device = flat[pieces[0].position].device
        buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
        views = [buf[p.offset : p.offset + p.end - p.start] for p in pieces]
        srcs = [flat[p.position][p.start : p.end] for p in pieces]
        for p in pieces:
            if p.end == flat[p.position].numel():
                del flat[p.position]
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

    for k, (chunk_nbytes, pieces) in enumerate(_chunks(specs, buffer_size_bytes)):
        slot = k % num_buffers
        # Wait for this slot's previous pack. Not streams[slot].synchronize():
        # on ROCm, streams can share a hardware queue, and synchronizing one
        # stream then also waits for work queued later on the others.
        done = slot_done[slot]
        if done is not None:
            done.synchronize()
        with torch.cuda.stream(streams[slot]):
            packed = torch.empty(chunk_nbytes, dtype=torch.uint8, device=device)
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
