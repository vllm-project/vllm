# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utils for model executor."""

import copy
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch

from vllm.utils.torch_utils import is_torch_equal_or_newer

_weights_pre_processed: ContextVar[bool] = ContextVar(
    "weights_pre_processed", default=False
)


@contextmanager
def weights_already_processed():
    """Mark that weights are already in post-processed (runtime) format, so
    ``process_weights_after_loading`` skips tensor transforms (used by the
    weight cache IPC loader).
    """
    token = _weights_pre_processed.set(True)
    try:
        yield
    finally:
        _weights_pre_processed.reset(token)


def is_weights_pre_processed() -> bool:
    return _weights_pre_processed.get()


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


def _extent(t: torch.Tensor) -> int:
    """Number of elements spanned by ``t`` in its storage."""
    if t.numel() == 0:
        return 0
    return 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))


def _in_own_storage(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    """``new`` copied into the bytes ``old`` spans if it is smaller, so the loaded
    storage is kept (with its ``data_ptr``); else ``new`` itself."""
    storage = old.untyped_storage()
    offset, size = old.storage_offset() * old.element_size(), new.element_size()
    if (
        new.numel() == 0
        or old.device != new.device
        or new.untyped_storage().data_ptr() == storage.data_ptr()
        or _extent(new) != new.numel()  # overlapping or with gaps
        or offset % size
        or new.numel() * size >= _extent(old) * old.element_size()
    ):
        return new
    view = new.new_empty(0).set_(storage, offset // size, new.shape, new.stride())
    return view.copy_(new)


def replace_parameter(
    layer: torch.nn.Module,
    param_name: str,
    new_tensor: torch.Tensor | None,
):
    """Replace a parameter of a layer while maintaining the ability to reload the
    weight. Called within implementations of the `process_weights_after_loading` method.

    The existing parameter object is kept (and so its class and loader attributes
    for weight reloading): ``new_tensor`` is copied into it when shape, stride,
    dtype and device match, which preserves ``data_ptr`` for captured CUDA graphs;
    otherwise a smaller ``new_tensor`` is copied into the parameter's own storage
    (so the loaded storage never shrinks), else ``.data`` is rebound to it. A new
    parameter is registered only if none exists under ``param_name``, or if the
    existing one is an online requantization source (released after conversion).

    Custom attributes set on ``new_tensor`` (e.g. kernel dispatch flags such as
    ``is_shuffled``) are carried over to the parameter, except ``weight_loader``,
    which is always taken from the existing parameter.

    This function should not be called on weights which are tied/shared

    Args:
        layer: Layer containing parameter to replace
        param_name: Name of parameter to replace
        new_tensor: New data of the new parameter, or None to set the parameter
            to None

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

    from vllm.model_executor.model_loader.reload.direct import (
        place_in_runtime_layout,
    )

    # Keep the object, except online requantization sources (released later)
    # and meta params (no storage; `.data` cannot be rebound to a real tensor).
    if (
        isinstance(old_param, torch.nn.Parameter)
        and not getattr(old_param, "requantization_source", False)
        and not (old_param.is_meta and not new_tensor.is_meta)
    ):
        if (
            old_param.shape == new_tensor.shape
            and old_param.stride() == new_tensor.stride()
            and old_param.dtype == new_tensor.dtype
            and old_param.device == new_tensor.device
        ):
            old_param.data.copy_(new_tensor)
        elif getattr(getattr(layer, "quant_method", None), "uses_meta_device", False):
            old_param.data = new_tensor  # online quantization frees the source
        elif (placed := place_in_runtime_layout(old_param, new_tensor)) is not None:
            old_param.data = placed
        else:
            old_param.data = _in_own_storage(old_param, new_tensor)
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


def move_parameter(layer: torch.nn.Module, src_name: str, dst_name: str) -> None:
    """Register the ``src_name`` parameter object under ``dst_name`` as well.

    Both names then refer to one object and so to one storage: the loader writes
    through the checkpoint name, kernels read the runtime name, and a later
    `replace_parameter` on either name updates both. ``dst_name`` becomes the
    name `kernel_parameters` reports for the object."""
    param = getattr(layer, src_name)
    existing = layer._parameters.get(dst_name)
    if existing is not None and existing is not param:
        raise ValueError(f"{dst_name!r} is already registered as another parameter")
    layer.register_parameter(dst_name, param)
    param.runtime_name = dst_name


def retire_parameter(layer: torch.nn.Module, name: str) -> None:
    """Mark a loaded parameter that no kernel reads after post-processing.

    It stays registered with its loaded data (so it can be loaded again);
    `kernel_parameters` and `get_live_param` skip it."""
    getattr(layer, name).retired = True


def set_kernel_parameter(
    layer: torch.nn.Module, name: str, tensor: torch.Tensor
) -> None:
    """`replace_parameter`, for a tensor kernels read under ``name``: the parameter
    is live even if it was retired before, and ``name`` becomes its runtime name
    if it is registered under several names (see `move_parameter`)."""
    replace_parameter(layer, name, tensor)
    param = getattr(layer, name)
    param.__dict__.pop("retired", None)
    if hasattr(param, "runtime_name"):
        param.runtime_name = name


def _is_live(param: torch.Tensor | None) -> bool:
    return (
        param is not None and not getattr(param, "retired", False) and param.numel() > 0
    )


def get_live_param(layer: torch.nn.Module, name: str) -> torch.Tensor | None:
    """``getattr(layer, name, None)``, treating retired or empty params as absent."""
    tensor = getattr(layer, name, None)
    if isinstance(tensor, torch.nn.Parameter) and not _is_live(tensor):
        return None
    return tensor


def kernel_parameters(layer: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    """The layer's own parameters as kernels see them: live parameters only, each
    object once, under its runtime name (see `move_parameter`)."""
    names: dict[int, str] = {}
    params: dict[str, torch.nn.Parameter] = {}
    for name, param in layer._parameters.items():
        if not _is_live(param):
            continue
        first = names.setdefault(id(param), name)
        if first == name:
            params[name] = param
        elif getattr(param, "runtime_name", None) == name:
            del params[first]
            params[name] = param
            names[id(param)] = name
    return params


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
