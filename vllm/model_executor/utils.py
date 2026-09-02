# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utils for model executor."""

import copy
from typing import Any

import torch

from vllm.utils.torch_utils import is_torch_equal_or_newer


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"

        # NOTE(woosuk): During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        from vllm.platforms import current_platform

        if current_platform.use_sync_weight_loader() and key == "weight_loader":
            value = current_platform.make_synced_weight_loader(value)
        setattr(weight, key, value)


def replace_parameter(
    layer: torch.nn.Module,
    param_name: str,
    new_tensor: torch.Tensor | None,
    prefer_copy: bool = False,
):
    """
    Replace a parameter of a layer while maintaining the ability to reload the weight.
    Called within implementations of the `process_weights_after_loading` method.

    Custom attributes set on ``new_tensor`` (e.g. kernel dispatch flags such as
    ``is_shuffled``) are carried over to the replacement parameter, except
    ``weight_loader``, which is always taken from the existing parameter.

    Attributes of the existing parameter are otherwise dropped when a new
    parameter is registered, but kept when ``prefer_copy`` reuses it in place.

    This function should not be called on weights which are tied/shared

    Args:
        layer: Layer containing parameter to replace
        param_name: Name of parameter to replace
        new_tensor: New data of the new parameter, or None to set the parameter
            to None
        prefer_copy: If True and the existing parameter is compatible with
            ``new_tensor`` (same shape, dtype, and device), copy ``new_tensor``
            into the existing parameter in place rather than re-registering
            a new parameter. This preserves the parameter's storage address
            (``data_ptr``), which is required for captured CUDA graphs to
            remain valid across weight updates (e.g. in RL training loops).
    """
    # should not be used on a tied/shared param

    # If new_tensor is None, set the parameter to None
    if new_tensor is None:
        setattr(layer, param_name, None)
        return

    old_param: torch.nn.Parameter | None = getattr(layer, param_name, None)

    new_tensor_attrs = dict(new_tensor.__dict__)

    # `weight_loader` is the only attribute not ported over from new_tensor,
    # old_param.weight_loader only is supported. `_weight_loader` is
    # `BasevLLMParameter`'s backing field for its `weight_loader` property, so
    # a loader riding along under that name has to be dropped as well.
    new_tensor_attrs.pop("weight_loader", None)
    new_tensor_attrs.pop("_weight_loader", None)

    if isinstance(new_tensor, torch.nn.Parameter):
        new_tensor = new_tensor.data

    if (
        prefer_copy
        and old_param is not None
        and old_param.shape == new_tensor.shape
        and old_param.dtype == new_tensor.dtype
        and old_param.device == new_tensor.device
    ):
        old_param.copy_(new_tensor)
        for attr_name, attr in new_tensor_attrs.items():
            setattr(old_param, attr_name, attr)
        return

    new_param = torch.nn.Parameter(new_tensor, requires_grad=False)

    for attr_name, attr in new_tensor_attrs.items():
        setattr(new_param, attr_name, attr)

    if old_param is not None and hasattr(old_param, "weight_loader"):
        weight_loader = old_param.weight_loader
        set_weight_attrs(new_param, {"weight_loader": weight_loader})

    setattr(layer, param_name, new_param)


def get_packed_modules_mapping(model: torch.nn.Module) -> dict[str, list[str]]:
    parent_map = getattr(model, "packed_modules_mapping", None)
    parent_map = copy.deepcopy(parent_map) if parent_map is not None else {}

    # don't infer mapping if the model has defined it explicitly.
    if parent_map:
        return parent_map

    # We only check main components instead of whole model submodules
    for child in model.children():
        child_map = getattr(child, "packed_modules_mapping", None)
        child_map = copy.deepcopy(child_map) if child_map is not None else {}

        if any((k in parent_map and parent_map[k] != v) for k, v in child_map.items()):
            raise ValueError(
                f"Can't update {type(model).__name__}'s packed_modules_mapping "
                f"safely because of conflicts from {type(child).__name__}."
            )
        else:
            parent_map.update(child_map)
    return parent_map


def get_moe_expert_mapping(
    model: torch.nn.Module,
) -> list[tuple[str, str, int, str]]:
    """Get the expert mapping from a model.

    It will be retrieved from the first module that has a `get_expert_mapping` method.
    If the model manually implements `get_expert_mapping`, it will be used.
    Otherwise, it will use the first RoutedExperts layer."""
    for _, module in model.named_modules():
        get_mapping = getattr(module, "get_expert_mapping", None)
        if get_mapping is not None:
            return get_mapping()
    raise ValueError("No module in the model has a `get_expert_mapping` method.")


def maybe_disable_graph_partition(current_backend: str) -> dict[str, bool]:
    if current_backend == "inductor" and is_torch_equal_or_newer("2.9.0.dev"):
        return {"graph_partition": False}
    else:
        return {}
