# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utils for model executor."""
import copy
from typing import Any, Optional

import torch


def set_random_seed(seed: int) -> None:
    from vllm.platforms import current_platform
    current_platform.seed_everything(seed)


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[dict[str, Any]],
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
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")

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
        if current_platform.is_tpu() and key == "weight_loader":
            value = _make_synced_weight_loader(value)
        setattr(weight, key, value)


def _make_synced_weight_loader(original_weight_loader):

    def _synced_weight_loader(param, *args, **kwargs):
        original_weight_loader(param, *args, **kwargs)
        # torch._sync doesn't support, is not needed for CPU tensors.
        if param.device != torch.device("cpu"):
            torch._sync(param)

    return _synced_weight_loader


def get_packed_modules_mapping(model: torch.nn.Module) -> dict[str, list[str]]:
    parent_map = copy.deepcopy(getattr(model, "packed_modules_mapping", {}))

    # don't infer mapping if the model has defined it explicitly.
    if parent_map:
        return parent_map

    # We only check main components instead of whole model submodules
    for child in model.children():
        child_map = getattr(child, "packed_modules_mapping", {})
        if any((k in parent_map and parent_map[k] != v)
               for k, v in child_map.items()):
            raise ValueError(
                f"Can't update {type(model).__name__}'s packed_modules_mapping "
                f"safely because of conflicts from {type(child).__name__}.")
        else:
            parent_map.update(child_map)
    return parent_map