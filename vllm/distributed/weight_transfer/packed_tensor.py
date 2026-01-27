# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
from collections.abc import Callable, Iterator
from typing import Any

import torch

from vllm import envs


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
    target_packed_tensor_size = envs.VLLM_PACKED_TENSOR_BUFFER_SIZE_BYTES
    num_buffers = envs.VLLM_PACKED_TENSOR_NUM_BUFFERS

    streams = [torch.cuda.Stream() for _ in range(num_buffers)]
    buffer_idx = 0

    packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
    packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
    packed_tensors: list[torch.Tensor] = [
        torch.empty(0, dtype=torch.uint8, device="cuda") for _ in range(num_buffers)
    ]

    while True:
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
                # Move to the next buffer
                buffer_idx = (buffer_idx + 1) % num_buffers
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
        names: list[str],
        shapes: list[list[int]],
        dtypes: list[torch.dtype],
        tensor_sizes: list[int],
    ) -> list[tuple[str, torch.Tensor]]:
        """Unpack a single tensor into a list of tensors.

        Args:
            packed_tensor: The packed torch.uint8 tensor to unpack
            names: List of tensor names
            shapes: List of tensor shapes
            dtypes: List of tensor dtypes
            tensor_sizes: List of tensor sizes in bytes

        Returns:
            unpacked List[(name, tensor)]
        """
        unpacked_tensors = packed_tensor.split_with_sizes(tensor_sizes)

        unpacked_list = [
            (name, tensor.view(dtype).view(*shape))
            for name, shape, dtype, tensor in zip(
                names, shapes, dtypes, unpacked_tensors
            )
        ]

        return unpacked_list

    target_packed_tensor_size = envs.VLLM_PACKED_TENSOR_BUFFER_SIZE_BYTES
    num_buffers = envs.VLLM_PACKED_TENSOR_NUM_BUFFERS

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
