# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-parameter weight digests and reset, for verifying weight updates."""

import hashlib
from collections.abc import Iterator

import torch
import torch.nn as nn
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from vllm.model_executor.parameter import SharedWeightParameter


def _tensors(name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    # Both keep their bytes in inner tensors (e.g. TorchAO's qdata and scale).
    if isinstance(tensor, SharedWeightParameter):
        for index, partition in tensor.partitions.items():
            yield from _tensors(f"{name}.{index}", partition)
    elif is_traceable_wrapper_subclass(tensor):
        for attr in tensor.__tensor_flatten__()[0]:
            yield from _tensors(f"{name}.{attr}", getattr(tensor, attr))
    else:
        yield name, tensor


def _weights(model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    for name, param in model.named_parameters():
        yield from _tensors(name, param)


def compute_tensor_digests(model: nn.Module) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            weight.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy()
        ).hexdigest()
        for name, weight in _weights(model)
    }


def zero_weights(model: nn.Module) -> None:
    for _, weight in _weights(model):
        weight.data.zero_()


def find_shared_weight_storage(
    model: nn.Module, other: nn.Module
) -> tuple[str, str] | None:
    """Find a weight whose storage overlaps a parameter or buffer of `other`."""

    def storage_ranges(tensors: Iterator[tuple[str, torch.Tensor]]):
        for name, tensor in tensors:
            storage = tensor.untyped_storage()
            start = storage.data_ptr()
            if start and storage.nbytes():
                yield str(tensor.device), start, start + storage.nbytes(), name

    def other_tensors():
        yield from _weights(other)
        for name, buffer in other.named_buffers():
            yield from _tensors(name, buffer)

    other_ranges = iter(sorted(storage_ranges(other_tensors())))
    current = next(other_ranges, None)
    for device, start, end, name in sorted(storage_ranges(_weights(model))):
        while current is not None and (current[0], current[2]) <= (device, start):
            current = next(other_ranges, None)
        if current is None:
            break
        other_device, other_start, _, other_name = current
        if other_device == device and other_start < end:
            return name, other_name
    return None
