# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Any

import torch

# Configuration constants (can be overridden via environment variables later)
REFIT_BUFFER_MEMORY_RATIO = 0.02
REFIT_NUM_BUFFERS = 2
REFIT_MAX_BUFFER_SIZE = 5 * 1024**3  # 5GB max


@lru_cache(maxsize=1)
def get_target_packed_tensor_size() -> int:
    """Calculate target packed tensor size based on GPU memory."""
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total_memory_bytes = props.total_memory
    target_size = min(
        int(total_memory_bytes * REFIT_BUFFER_MEMORY_RATIO), REFIT_MAX_BUFFER_SIZE
    )
    return target_size


def packed_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
) -> None:
    """Broadcast tensors in a packed manner from trainer to workers.

    Args:
        iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
        group: Process group (PyNcclCommunicator)
        src: Source rank (0 in current implementation)
        post_iter_func: Function to apply to each (name, tensor) pair before
                       packing, should return a tensor

    """
    target_packed_tensor_size = get_target_packed_tensor_size()
    num_buffers = REFIT_NUM_BUFFERS

    streams = [torch.cuda.Stream() for _ in range(num_buffers)]
    buffer_idx = 0

    packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
    packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
    packed_tensors: list[torch.Tensor] = [
        torch.empty(0, dtype=torch.uint8, device="cuda") for _ in range(num_buffers)
    ]

    while True:
        # Move to the next buffer
        buffer_idx = (buffer_idx + 1) % num_buffers
        # Synchronize the current stream
        streams[buffer_idx].synchronize()
        # Start tasks for the new buffer in a new stream
        with torch.cuda.stream(streams[buffer_idx]):
            try:
                # Initialize the packing tensor list and sizes
                packing_tensor_list[buffer_idx] = []
                packing_tensor_sizes[buffer_idx] = 0
                # Pack the tensors
                while True:
                    # Apply post processing and convert to linearized uint8 tensor
                    tensor = post_iter_func(next(iterator)).view(torch.uint8).view(-1)
                    packing_tensor_list[buffer_idx].append(tensor)
                    packing_tensor_sizes[buffer_idx] += tensor.numel()
                    if packing_tensor_sizes[buffer_idx] > target_packed_tensor_size:
                        break
                # Pack the tensors and call broadcast collective
                packed_tensors[buffer_idx] = torch.cat(
                    packing_tensor_list[buffer_idx], dim=0
                )
                group.broadcast(packed_tensors[buffer_idx], src=src)
            except StopIteration:
                # Do the last broadcast if there are remaining tensors
                if len(packing_tensor_list[buffer_idx]) > 0:
                    packed_tensors[buffer_idx] = torch.cat(
                        packing_tensor_list[buffer_idx], dim=0
                    )
                    group.broadcast(packed_tensors[buffer_idx], src=src)
                break


def packed_broadcast_consumer(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
) -> None:
    """Consume packed tensors and unpack them into a list of tensors.

    Args:
        iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
        group: Process group (PyNcclCommunicator)
        src: Source rank (0 in current implementation)
        post_unpack_func: Function to apply to each list of (name, tensor) after
                         unpacking

    """

    def unpack_tensor(
        packed_tensor: torch.Tensor,
        meta_data_list: list[tuple[str, list[int], torch.dtype, int, int]],
    ) -> list[tuple[str, torch.Tensor]]:
        """Unpack a single tensor into a list of tensors.

        Args:
            packed_tensor: The packed torch.uint8 tensor to unpack
            meta_data_list: List[(name, shape, dtype, offset, tensor_size)]

        Returns:
            unpacked List[(name, tensor)]
        """
        # Perform batched split with torch.split_with_sizes
        packed_tensor_sizes = [meta[4] for meta in meta_data_list]
        unpacked_tensors = packed_tensor.split_with_sizes(packed_tensor_sizes)

        unpacked_list = [
            (
                meta_data_list[i][0],
                tensor.view(meta_data_list[i][2]).view(*meta_data_list[i][1]),
            )
            for i, tensor in enumerate(unpacked_tensors)
        ]

        return unpacked_list

    target_packed_tensor_size = get_target_packed_tensor_size()
    num_buffers = REFIT_NUM_BUFFERS

    streams = [torch.cuda.Stream() for _ in range(num_buffers)]
    buffer_idx = 0

    packing_tensor_meta_data: list[
        list[tuple[str, list[int], torch.dtype, int, int]]
    ] = [[] for _ in range(num_buffers)]
    packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
    offsets: list[int] = [0 for _ in range(num_buffers)]
    packed_tensors: list[torch.Tensor] = [
        torch.empty(0, dtype=torch.uint8, device="cuda") for _ in range(num_buffers)
    ]

    while True:
        # Move to the next buffer
        buffer_idx = (buffer_idx + 1) % num_buffers
        # Synchronize the current stream
        streams[buffer_idx].synchronize()
        with torch.cuda.stream(streams[buffer_idx]):
            # Initialize the packing tensor meta data
            packing_tensor_meta_data[buffer_idx] = []
            packing_tensor_sizes[buffer_idx] = 0
            offsets[buffer_idx] = 0
            try:
                # Form a packed tensor
                while True:
                    name, (shape, dtype) = next(iterator)
                    tensor_size = math.prod(shape) * dtype.itemsize
                    packing_tensor_meta_data[buffer_idx].append(
                        (name, shape, dtype, offsets[buffer_idx], tensor_size)
                    )
                    packing_tensor_sizes[buffer_idx] += tensor_size
                    offsets[buffer_idx] += tensor_size
                    if packing_tensor_sizes[buffer_idx] > target_packed_tensor_size:
                        break
                # Create a packed tensor and broadcast it
                packed_tensors[buffer_idx] = torch.empty(
                    packing_tensor_sizes[buffer_idx], dtype=torch.uint8, device="cuda"
                )
                group.broadcast(packed_tensors[buffer_idx], src=src)
                # Load the packed tensor into the model
                post_unpack_func(
                    unpack_tensor(
                        packed_tensors[buffer_idx],
                        packing_tensor_meta_data[buffer_idx],
                    )
                )
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
                    post_unpack_func(
                        unpack_tensor(
                            packed_tensors[buffer_idx],
                            packing_tensor_meta_data[buffer_idx],
                        )
                    )
                break
