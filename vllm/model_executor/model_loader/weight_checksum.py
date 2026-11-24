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


def reset_model_weights(
    target: nn.Module | None, draft: nn.Module | None
) -> str | None:
    """Validate the selected draft before clearing either model."""
    if draft is not None and not any(weight.numel() for _, weight in _weights(draft)):
        return "Weight Checker found no resettable draft weights."
    for model in (target, draft):
        if model is not None:
            zero_weights(model)
    return None
