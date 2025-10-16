# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utils for model executor."""

import copy
from typing import Any

import torch

from vllm.utils import is_torch_equal_or_newer


def set_random_seed(seed: int) -> None:
    from vllm.platforms import current_platform

    current_platform.seed_everything(seed)


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
    if parent_map := getattr(model, "get_expert_mapping", None):
        return parent_map()
    else:
        # We only check main components instead of whole model submodules
        for child in model.children():
            child_map = getattr(child, "get_expert_mapping", None)
            if child_map is not None:
                return child_map()
        return []


def disable_inductor_graph_partition(func):
    """Decorator to disable inductor graph partition.
    This is used to avoid nested cudagraph capture.

    Example:
    1. We apply torch.compile directly on some ops (e.g., grouped_topk) wrapped
    in custom ops. Inductor graph partition applies cudagraph within the custom op.
    2. At the same time, we compile the model which uses these custom ops. Inductor
    graph partition also wraps each graph partition with CUDAGraph. Some partitions
    may include custom ops, which has already been applied cudagraph. This leads to
    nested cudagraph which is not supported.

    This context manager should be wrapped around torch.compile calls within custom ops
    to avoid the nested cudagraph capture.

    Expected Usage:
    @disable_inductor_graph_partition
    @torch.compile()
    def op_eager_code(...):
        ...

    Note that `@disable_inductor_graph_partition` should be applied on top of
    `torch.compile()`
    """

    def wrapper(*args, **kwargs):
        old_val = torch._inductor.config.graph_partition
        torch._inductor.config.graph_partition = False
        out = func(*args, **kwargs)
        torch._inductor.config.graph_partition = old_val
        return out

    return wrapper if is_torch_equal_or_newer("2.9.0.dev") else func
