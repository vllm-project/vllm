# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor

# Default values for packed tensor configuration.
# These are imported by NCCLWeightTransferUpdateInfo and trainer_send_weights.
DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB
DEFAULT_PACKED_NUM_BUFFERS = 2


def unpack_tensor(
    packed_tensor: torch.Tensor,
    names: list[str],
    shapes: list[list[int]],
    dtypes: list[torch.dtype],
    tensor_sizes: list[int],
) -> list[tuple[str, torch.Tensor]]:
    """Unpack a packed uint8 tensor into a list of named tensors.

    The .contiguous().view() calls on split slices create independent copies,
    so callers do not need to clone the results.

    Args:
        packed_tensor: The packed torch.uint8 tensor to unpack
        names: List of tensor names
        shapes: List of tensor shapes
        dtypes: List of tensor dtypes
        tensor_sizes: List of tensor sizes in bytes
    """
    unpacked_tensors = packed_tensor.split(tensor_sizes)

    return [
        (name, tensor.contiguous().view(dtype).view(*shape))
        for name, shape, dtype, tensor in zip(names, shapes, dtypes, unpacked_tensors)
    ]


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
        tensor = post_iter_func(item).contiguous().view(torch.uint8).view(-1)

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
    streams = [torch.cuda.Stream() for _ in range(num_buffers)]
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
            chunk = pack_tensors(iterator, post_iter_func, buffer_size_bytes)
            if chunk is None:
                break
            # Pack the tensors and call broadcast collective
            group.broadcast(chunk.packed_tensor, src=src)
            # Hold reference until this stream is synchronized
            in_flight[buffer_idx] = chunk
            # Move to the next buffer
            buffer_idx = (buffer_idx + 1) % num_buffers


def packed_nccl_broadcast_consumer(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
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

    """
    target_packed_tensor_size = buffer_size_bytes

    streams = [torch.cuda.Stream() for _ in range(num_buffers)]
    buffer_idx = 0

    packing_tensor_meta_data: list[list[tuple[str, list[int], torch.dtype, int]]] = [
        [] for _ in range(num_buffers)
    ]
    packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
    packed_tensors: list[torch.Tensor] = [
        torch.empty(0, dtype=torch.uint8, device="cuda") for _ in range(num_buffers)
    ]

    while True:
        # Synchronize the current stream
        streams[buffer_idx].synchronize()
        with torch.cuda.stream(streams[buffer_idx]):
            # Initialize the packing tensor meta data
            packing_tensor_meta_data[buffer_idx] = []
            packing_tensor_sizes[buffer_idx] = 0
            try:
                # Form a packed tensor
                while True:
                    name, (shape, dtype) = next(iterator)
                    tensor_size = math.prod(shape) * dtype.itemsize
                    packing_tensor_meta_data[buffer_idx].append(
                        (name, shape, dtype, tensor_size)
                    )
                    packing_tensor_sizes[buffer_idx] += tensor_size
                    if packing_tensor_sizes[buffer_idx] > target_packed_tensor_size:
                        break
                # Create a packed tensor and broadcast it
                packed_tensors[buffer_idx] = torch.empty(
                    packing_tensor_sizes[buffer_idx], dtype=torch.uint8, device="cuda"
                )
                group.broadcast(packed_tensors[buffer_idx], src=src)
                # Load the packed tensor into the model
                names, shapes, dtypes, tensor_sizes = zip(
                    *packing_tensor_meta_data[buffer_idx]
                )
                post_unpack_func(
                    unpack_tensor(
                        packed_tensors[buffer_idx],
                        list(names),
                        list(shapes),
                        list(dtypes),
                        list(tensor_sizes),
                    )
                )
                # Move to the next buffer
                buffer_idx = (buffer_idx + 1) % num_buffers
            except StopIteration:
                # Do the last broadcast if there are remaining tensors
                if len(packing_tensor_meta_data[buffer_idx]) > 0:
                    # Create a packed tensor and broadcast it
                    packed_tensors[buffer_idx] = torch.empty(
                        packing_tensor_sizes[buffer_idx],
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    group.broadcast(packed_tensors[buffer_idx], src=src)
                    # Load the packed tensor into the model
                    names, shapes, dtypes, tensor_sizes = zip(
                        *packing_tensor_meta_data[buffer_idx]
                    )
                    post_unpack_func(
                        unpack_tensor(
                            packed_tensors[buffer_idx],
                            list(names),
                            list(shapes),
                            list(dtypes),
                            list(tensor_sizes),
                        )
                    )
                break


# ── IPC packed transfer ────────────────────────────────────────────────


@dataclass
class PackedIpcChunk:
    """Metadata and IPC handle for a single packed chunk."""

    names: list[str]
    shapes: list[list[int]]
    dtype_names: list[str]
    tensor_sizes: list[int]
    ipc_handle: dict[str, tuple]
    is_first: bool
    is_last: bool


def packed_ipc_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    gpu_uuid: str,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
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
    """
    ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device="cuda")
    _, ipc_args = reduce_tensor(ipc_buffer)

    chunk_idx = 0
    pending: tuple[str, torch.Tensor, torch.Tensor] | None = None
    exhausted = False

    while not exhausted or pending is not None:
        names: list[str] = []
        shapes: list[list[int]] = []
        dtypes: list[torch.dtype] = []
        tensor_sizes: list[int] = []
        tensors: list[torch.Tensor] = []
        total_bytes = 0

        if pending is not None:
            p_name, p_orig, p_flat = pending
            tensors.append(p_flat)
            names.append(p_name)
            shapes.append(list(p_orig.shape))
            dtypes.append(p_orig.dtype)
            tensor_sizes.append(p_flat.numel())
            total_bytes += p_flat.numel()
            pending = None

        while not exhausted:
            item = next(iterator, None)
            if item is None:
                exhausted = True
                break

            name, orig_tensor = item
            flat = post_iter_func(item).contiguous().view(torch.uint8).view(-1)

            if flat.numel() > buffer_size_bytes:
                raise ValueError(
                    f"Tensor '{name}' has size {flat.numel()} bytes, "
                    f"which exceeds buffer_size_bytes={buffer_size_bytes}. "
                    f"Increase buffer_size_bytes to at least {flat.numel()}."
                )

            if total_bytes + flat.numel() > buffer_size_bytes and tensors:
                pending = (name, orig_tensor, flat)
                break

            tensors.append(flat)
            names.append(name)
            shapes.append(list(orig_tensor.shape))
            dtypes.append(orig_tensor.dtype)
            tensor_sizes.append(flat.numel())
            total_bytes += flat.numel()

        if not tensors:
            break

        packed = torch.cat(tensors, dim=0)
        del tensors
        ipc_buffer[: packed.numel()].copy_(packed)
        del packed

        is_last = exhausted and pending is None
        dtype_names = [str(d).split(".")[-1] for d in dtypes]

        yield PackedIpcChunk(
            names=names,
            shapes=shapes,
            dtype_names=dtype_names,
            tensor_sizes=tensor_sizes,
            ipc_handle={gpu_uuid: ipc_args},
            is_first=chunk_idx == 0,
            is_last=is_last,
        )
        chunk_idx += 1

    del ipc_buffer


def packed_ipc_consumer(
    ipc_handle: dict[str, tuple],
    names: list[str],
    shapes: list[list[int]],
    dtype_names: list[str],
    tensor_sizes: list[int],
    device_index: int,
) -> list[tuple[str, torch.Tensor]]:
    """Unpack a single packed IPC chunk into named tensors.

    Reconstructs the packed buffer via rebuild_cuda_tensor, then unpacks
    into individual tensors. The .contiguous().view() calls in
    unpack_tensor create independent copies, releasing the IPC
    reference naturally.

    Args:
        ipc_handle: Mapping of GPU UUID to rebuild_cuda_tensor args tuple
        names: Parameter names in the packed buffer
        shapes: Parameter shapes
        dtype_names: Parameter dtype name strings (e.g. "float16")
        tensor_sizes: Size in bytes of each parameter in the packed buffer
        device_index: Local CUDA device index
    """
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

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
    packed = rebuild_cuda_tensor(*list_args)

    content_size = sum(tensor_sizes)
    packed = packed[:content_size]

    dtypes = [getattr(torch, dn) for dn in dtype_names]
    return unpack_tensor(packed, names, shapes, dtypes, tensor_sizes)
