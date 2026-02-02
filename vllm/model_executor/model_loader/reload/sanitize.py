# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import MethodType

import torch

__all__ = ["sanitize_layer_refs", "restore_layer_refs"]


layer_ref_sentinel = object()


def sanitize_layer_refs(tensor: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
    """
    Removes references to layer held by tensor attributes. Specifically, removes the
    `__self__` attribute of weight loader methods attached to the tensor.

    Used by `capture_layer_to_meta` to avoid circular references to layers in
    `LAYERWISE_INFO`, leading to modules never being cleaned up. Without sanitation,
    tensors will reference layers, and the WeakKeyDictionary will never evict entries,
    even when the model is deleted.

    :param tensor: tensor to be sanitized
    :param layer: layer whose references should be removed
    :return: sanitized tensor
    """
    for key, value in tensor.__dict__.items():
        if isinstance(value, MethodType) and value.__self__ is layer:
            tensor.__dict__[key] = value.__func__.__get__(layer_ref_sentinel)

    return tensor


def restore_layer_refs(tensor: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
    """
    Restores references to layer held by tensor attributes.

    Used by `restore_layer_on_meta` to add back layer references, allowing for proper
    weight loading.

    :param tensor: tensor to be sanitized
    :param layer: layer whose references should be removed
    :return: sanitized tensor

    """
    for key, value in tensor.__dict__.items():
        if isinstance(value, MethodType) and value.__self__ is layer_ref_sentinel:
            tensor.__dict__[key] = value.__func__.__get__(layer)

    return tensor
