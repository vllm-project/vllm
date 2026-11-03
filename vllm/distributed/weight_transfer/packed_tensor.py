# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor

# Default values for packed tensor transfer.
DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB
DEFAULT_PACKED_NUM_BUFFERS = 2
# Entry start alignment of the packed layout (`packed_alignment`, a must-agree
# wire param). vLLM's trainer engines default to 16 bytes, which covers every
# dtype's itemsize, so unpacking never copies. Everything else — the worker init
# info and the helpers in this module — defaults to 1, the original unpadded
# layout, so producers that do not send `packed_alignment` keep working.
DEFAULT_PACKED_ALIGNMENT = 16
LEGACY_PACKED_ALIGNMENT = 1


def check_packed_alignment(alignment: int) -> None:
    if not (isinstance(alignment, int) and alignment >= 1):
        raise ValueError(f"packed_alignment must be a positive int, got {alignment!r}")
    if alignment & (alignment - 1):
        raise ValueError(f"packed_alignment must be a power of two, got {alignment}")


def _round_up(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


def packed_offsets(tensor_sizes: list[int], alignment: int) -> tuple[list[int], int]:
    """Start offsets and total length of the packed layout.

    Each entry starts at the previous entry's end rounded up to ``alignment``
    (the first at 0); the total is the last entry's end, with no trailing pad.
    ``alignment=1`` is the unpadded layout. Every producer and consumer of a
    packed buffer derives its layout from this.
    """
    offsets: list[int] = []
    end = 0
    for size in tensor_sizes:
        start = _round_up(end, alignment)
        offsets.append(start)
        end = start + size
    return offsets, end


def _take_chunk(
    items: Iterator[Any],
    size_of: Callable[[Any], int],
    buffer_size_bytes: int,
    alignment: int,
) -> list[Any]:
    """Pull items for one NCCL pack: up to and including the first item whose
    end crosses ``buffer_size_bytes``, or until ``items`` is exhausted. Shared
    by producer and consumer so both cut the stream at the same boundaries."""
    chunk: list[Any] = []
    end = 0
    for item in items:
        chunk.append(item)
        end = _round_up(end, alignment) + size_of(item)
        if end > buffer_size_bytes:
            break
    return chunk


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
    alignment: int = LEGACY_PACKED_ALIGNMENT,
) -> list[tuple[str, torch.Tensor]]:
    """Unpack a packed uint8 tensor into a list of named tensors.

    Entries are located with ``packed_offsets(tensor_sizes, alignment)``. An
    entry whose offset is a multiple of its dtype's itemsize is returned as a
    **view** of ``packed_tensor``; otherwise (possible only when ``alignment``
    is smaller than the itemsize, e.g. the unpadded legacy layout) it is
    copied. If ``packed_tensor`` lives in storage that may be reused — e.g. a
    reused CUDA IPC buffer — callers must clone the results before the
    underlying storage is overwritten.

    Args:
        packed_tensor: The packed torch.uint8 tensor to unpack. May be longer
            than the packed content.
        names: List of tensor names
        shapes: List of tensor shapes
        dtypes: List of tensor dtypes
        tensor_sizes: List of tensor sizes in bytes (unpadded)
        alignment: Entry alignment the producer packed with

    """
    offsets, total = packed_offsets(tensor_sizes, alignment)
    if total > packed_tensor.numel():
        raise ValueError(
            f"Packed layout needs {total} bytes but the buffer has "
            f"{packed_tensor.numel()}; producer and consumer disagree on "
            "tensor sizes or packed_alignment."
        )
    unpacked = []
    for name, shape, dtype, offset, size in zip(
        names, shapes, dtypes, offsets, tensor_sizes
    ):
        raw = packed_tensor[offset : offset + size]
        if raw.storage_offset() % dtype.itemsize:
            raw = raw.clone()
        unpacked.append((name, raw.view(dtype).view(shape)))
    return unpacked


@dataclass
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
    alignment: int = LEGACY_PACKED_ALIGNMENT,
) -> PackedChunk | None:
    """Pack tensors from an iterator into a single contiguous uint8 buffer.

    Consumes from the iterator until the packed size exceeds
    buffer_size_bytes or the iterator is exhausted, then returns a
    PackedChunk laid out per ``packed_offsets(tensor_sizes, alignment)``
    (padding bytes are zero). Returns None if no tensors were consumed.

    Args:
        iterator: Iterator of (name, tensor) pairs
        post_iter_func: Applied to each item before linearizing to uint8
        buffer_size_bytes: Max bytes before flushing
        alignment: Entry start alignment in bytes

    """

    def linearize(item: tuple[str, torch.Tensor]):
        name, orig_tensor = item
        flat = post_iter_func(item).contiguous().view(-1).view(torch.uint8)
        if flat.numel() > buffer_size_bytes:
            import warnings

            warnings.warn(
                f"Tensor '{name}' has size {flat.numel()} bytes, which "
                f"exceeds buffer_size_bytes={buffer_size_bytes}.",
                stacklevel=4,
            )
        return name, orig_tensor, flat

    items = _take_chunk(
        map(linearize, iterator),
        lambda item: item[2].numel(),
        buffer_size_bytes,
        alignment,
    )
    if not items:
        return None

    names = [name for name, _, _ in items]
    shapes = [list(orig.shape) for _, orig, _ in items]
    dtypes = [orig.dtype for _, orig, _ in items]
    flats = [flat for _, _, flat in items]
    tensor_sizes = [flat.numel() for flat in flats]

    # One cat with zero pads between entries, rather than a copy per entry.
    offsets, _ = packed_offsets(tensor_sizes, alignment)
    zeros = flats[0].new_zeros(alignment - 1)
    pieces: list[torch.Tensor] = []
    end = 0
    for offset, flat in zip(offsets, flats):
        if offset > end:
            pieces.append(zeros[: offset - end])
        pieces.append(flat)
        end = offset + flat.numel()
    return PackedChunk(
        packed_tensor=torch.cat(pieces),
        names=names,
        shapes=shapes,
        dtypes=dtypes,
        tensor_sizes=tensor_sizes,
    )


# ── NCCL packed broadcast ──────────────────────────────────────────────


def packed_nccl_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    alignment: int = LEGACY_PACKED_ALIGNMENT,
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
        alignment: Entry start alignment of the packed layout.
                  Both producer and consumer must use the same value.

    """
    streams = _get_streams(torch.accelerator.current_device_index(), num_buffers)
    # Keep references to in-flight chunks so their packed_tensors
    # aren't freed while an async broadcast is still reading them.
    in_flight: list[PackedChunk | None] = [None] * num_buffers
    buffer_idx = 0

    while True:
        # Synchronize the current stream
        streams[buffer_idx].synchronize()
        # Previous chunk on this buffer slot is now safe to free
        in_flight[buffer_idx] = None
        # Start tasks for the new buffer in a new stream
        with torch.cuda.stream(streams[buffer_idx]):
            chunk = pack_tensors(iterator, post_iter_func, buffer_size_bytes, alignment)
            if chunk is None:
                break
            # Pack the tensors and call broadcast collective
            group.broadcast(chunk.packed_tensor, src=src)
            # Hold reference until this stream is synchronized
            in_flight[buffer_idx] = chunk
            # Move to the next buffer
            buffer_idx = (buffer_idx + 1) % num_buffers

    # Drain every slot before returning.
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
    alignment: int = LEGACY_PACKED_ALIGNMENT,
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
        alignment: Entry start alignment of the packed layout.
                  Both producer and consumer must use the same value.

    """
    streams = _get_streams(torch.accelerator.current_device_index(), num_buffers)
    caller_stream = torch.accelerator.current_stream()
    prev_received: torch.Event | None = None
    slot_done: list[torch.Event | None] = [None] * num_buffers

    metas = (
        (name, shape, dtype, math.prod(shape) * dtype.itemsize)
        for name, (shape, dtype) in iterator
    )

    def receive_and_load(buffer_idx: int, chunk: list) -> None:
        nonlocal prev_received
        names, shapes, dtypes, tensor_sizes = map(list, zip(*chunk))
        _, total = packed_offsets(tensor_sizes, alignment)
        packed = torch.empty(total, dtype=torch.uint8, device=device)
        group.broadcast(packed, src=src)
        # Misaligned entries (legacy layout only) are copied here, on this
        # slot's stream, so the ready event below covers them too.
        weights = unpack_tensor(packed, names, shapes, dtypes, tensor_sizes, alignment)

        # post_unpack_func may keep the unpacked tensors and read them from a
        # later pack's callback, on another slot stream. Chaining the ready events
        # orders every callback after all earlier receives, and record_stream
        # keeps the allocator from reissuing this buffer (or a copy) once it is
        # freed while the broadcast or such a read is still pending.
        if prev_received is not None:
            streams[buffer_idx].wait_event(prev_received)
        prev_received = torch.Event()
        prev_received.record(streams[buffer_idx])
        copies = (t for _, t in weights if t._base is not packed and t.numel())
        for tensor in (packed, *copies):
            for stream in (*streams, caller_stream):
                tensor.record_stream(stream)

        post_unpack_func(weights)
        event = torch.Event()
        event.record(streams[buffer_idx])
        slot_done[buffer_idx] = event

    buffer_idx = 0
    while chunk := _take_chunk(metas, lambda m: m[3], buffer_size_bytes, alignment):
        # Wait for this slot's previous pack. Not streams[buffer_idx].synchronize():
        # on ROCm, streams can share a hardware queue, and synchronizing one
        # stream then also waits for work queued later on the others.
        done = slot_done[buffer_idx]
        if done is not None:
            done.synchronize()
        with torch.cuda.stream(streams[buffer_idx]):
            receive_and_load(buffer_idx, chunk)
        buffer_idx = (buffer_idx + 1) % num_buffers

    # Drain every slot before returning.
    for stream in streams:
        stream.synchronize()


# ── IPC packed transfer ────────────────────────────────────────────────


@dataclass
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
    alignment: int = LEGACY_PACKED_ALIGNMENT,
) -> Iterator[PackedIpcChunk]:
    """Pack tensors into a reusable IPC buffer and yield handles.

    Allocates a single GPU buffer of ``buffer_size_bytes`` and registers
    it for IPC once via ``reduce_tensor``.  Each chunk's packed data is
    copied into this buffer before yielding, so only one IPC-shared
    allocation exists for the lifetime of the transfer.

    Callers **must** ensure the consumer has finished reading the buffer
    (e.g. ``ray.get`` returned) before resuming the generator for the
    next chunk.

    Args:
        iterator: Iterator of (name, tensor) pairs.
        gpu_uuid: Physical GPU UUID string for this rank.
        post_iter_func: Applied to each (name, tensor) before packing.
        buffer_size_bytes: Exact capacity of the reusable IPC buffer.
            Every chunk is guaranteed to fit within this size.  A
            ``ValueError`` is raised if any single tensor exceeds it.
        alignment: Entry start alignment of the packed layout; the consumer
            must unpack with the same value.

    """
    ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device="cuda")
    _, ipc_args = reduce_tensor(ipc_buffer)

    names: list[str] = []
    shapes: list[list[int]] = []
    dtypes: list[torch.dtype] = []
    tensor_sizes: list[int] = []
    end = 0

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

        start = _round_up(end, alignment)
        if end and start + flat.numel() > buffer_size_bytes:
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
            start = 0

        ipc_buffer[start : start + flat.numel()].copy_(flat)
        names.append(name)
        shapes.append(list(orig_tensor.shape))
        dtypes.append(orig_tensor.dtype)
        tensor_sizes.append(flat.numel())
        end = start + flat.numel()

    if names:
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

    torch's cross-process refcount pairs one ``reduce_tensor`` export with
    one consumer rebuild-release cycle. The producer exports its buffer once
    and ships the same args with every chunk, so rebuilding and releasing
    per chunk decrements the counter once per chunk, drives it negative,
    and the dropped buffer is then never reclaimable by ``ipc_collect`` on
    the producer side (it leaks one buffer per transfer). Caching the
    rebuilt buffer keyed by the rebuild args restores the pairing: chunks
    of one transfer reuse the mapping (it aliases producer memory, so it
    always reads the current chunk's bytes), and replacing the entry when
    the next export arrives — or ``close()`` — releases the previous
    mapping exactly once.

    The importer that consumes a multi-chunk transfer must be one object
    reused across chunks; a fresh importer per chunk reintroduces the
    per-chunk release. ``IPCWeightTransferEngine`` owns one instance per
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
    alignment: int = LEGACY_PACKED_ALIGNMENT,
) -> list[tuple[str, torch.Tensor]]:
    """Unpack a single packed IPC chunk into named tensors.

    Reconstructs the packed buffer via rebuild_cuda_tensor, unpacks
    into individual tensors, and clones each into independent storage
    before returning.

    The clone is intentional: the producer reuses one IPC buffer across
    chunks, so any tensor view that aliases the buffer would observe the
    *next* chunk's bytes as soon as the producer's generator is resumed.
    Callers that retain references past their own update_weights call
    (notably vLLM's layerwise reload, which buffers ``bound_args`` for
    replay in ``_layerwise_process``) would otherwise replay against
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
            (see ``PackedBufferImporter``). ``None`` builds an ephemeral one,
            which is only safe for single-chunk consumption.
        alignment: Entry start alignment the producer packed with.

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

    dtypes = [getattr(torch, dn) for dn in dtype_names]
    return [
        (name, t.clone())
        for name, t in unpack_tensor(
            packed, names, shapes, dtypes, tensor_sizes, alignment
        )
    ]
