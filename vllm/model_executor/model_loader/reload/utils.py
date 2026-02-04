# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .types import LayerTensors

__all__ = [
    "get_layer_tensors",
    "get_layer_params_buffers",
    "get_layer_size",
]


def get_layer_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Get all parameters and buffers from a module as a dict."""
    params, buffers = get_layer_params_buffers(layer)
    return params | buffers


def get_layer_params_buffers(layer: torch.nn.Module) -> LayerTensors:
    """Get all parameters and buffers of a module as a tuple of dicts."""
    return (
        {name: param for name, param in layer._parameters.items() if param is not None},
        {name: buffer for name, buffer in layer._buffers.items() if buffer is not None},
    )


# Buffers that are computed locally, not loaded from checkpoint
_BUFFER_BLOCKLIST = {"_expert_map"}


def get_layer_size(layer: torch.nn.Module) -> int:
    """Calculate total number of elements across all tensors in a layer."""
    params, buffers = get_layer_params_buffers(layer)
    param_size = sum(p.numel() for p in params.values())
    buffer_size = sum(
        b.numel() for name, b in buffers.items() if name not in _BUFFER_BLOCKLIST
    )
    return param_size + buffer_size


class CopyCounter(TorchDispatchMode):
    """
    Tracks total number of elements modified with `copy_`.

    Useful for keeping track of weight loading where underlying weights can be
    arbitrarily transformed (such as with `narrow`) before calling copy.

    Note: Assumes that copy kwargs are not used.
    """

    def __init__(self):
        super().__init__()
        self.copied_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func is torch.ops.aten.copy_.default:
            assert args[0].numel() == args[1].numel()
            self.copied_numel += args[0].numel()

        return func(*args, **kwargs)
