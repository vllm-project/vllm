# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Layerwise weight reloading utilities for vLLM.

This module provides functionality to reload model weights layer-by-layer,
which is useful for weight updates without full model reconstruction.

Limitations:
    - Does not compose with CPU offloading. This is because `device_loading_context`
      doesn't work in all cases (e.g., when parameter is renamed).
    - Does not handle layers where only some weight elements are loaded, but some
      weights aren't. For example, only loading q_scale, but not k_scale or v_scale
    - Unties weights during loading, but not on cuda graph

TODO:
    - Decide on reloading interface, back-compat with reload_weights
    - Do Attention/MLA processing
    - Check composability with EPLB
"""

import inspect
from functools import wraps
from typing import Callable
from dataclasses import dataclass, field

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.attention.layer import Attention, MLAAttention
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)


LayerTensors = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]

@dataclass
class LayerReloadingInfo:
    # model format (meta), populated by `record_metadata_for_reloading`
    restore_metadata: LayerTensors

    # kernel format (device)
    kernel_tensors: LayerTensors = ({}, {})

    # track how many restored elements are ready for loading
    load_numel: int | float = 0
    load_numel_total: int | float = float("inf")

    # stores arguments and tensors ready for loading
    loaded_weights: list[tuple[str, inspect.BoundArguments]] = field(default_factory=list)

    def reset(self):
        self.kernel_tensors = ({}, {})
        self.load_numel = 0
        self.load_numel_total = float("inf")
        self.loaded_weights = list()

LAYER_RELOADING_INFO: dict[torch.nn.Module, LayerReloadingInfo] = dict()


# -----------------------------------------------------------------------------
# Materialize Utilities
# -----------------------------------------------------------------------------


def to_meta_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to a meta tensor while preserving class and attributes."""
    meta_tensor = tensor.data.to("meta")
    meta_tensor.__class__ = tensor.__class__
    meta_tensor.__dict__ = tensor.__dict__
    return meta_tensor


def materialize_meta_tensor(meta_tensor: torch.Tensor) -> torch.Tensor:
    """
    Materialize a meta tensor into an actual tensor on the current device.

    Note: Should be called within a torch device context.

    TODO: Need a way of reconstructing vLLMBaseParameters on the meta device.
    """
    tensor = torch.empty_strided(
        size=tuple(meta_tensor.size()),
        stride=tuple(meta_tensor.stride()),
        dtype=meta_tensor.dtype,
        requires_grad=False,
    )
    tensor.__class__ = meta_tensor.__class__
    tensor.__dict__ = meta_tensor.__dict__
    return tensor


def materialize_layer(layer: torch.nn.Module) -> None:
    """Materialize all meta tensors in a layer to actual tensors."""
    for name, tensor in get_layer_tensors(layer).items():
        setattr(layer, name, materialize_meta_tensor(tensor))


class MetaCopyCounter(TorchDispatchMode):
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

        if func is torch.ops.aten.copy_.default and args[0].device.type == "meta":
            assert args[0].numel() == args[1].numel()
            self.copied_numel += args[0].numel()

        return func(*args, **kwargs)


def get_numel_loaded(weight_loader: Callable, args: inspect.BoundArguments) -> int:
    """
    Determine how many elements would be loaded by a weight loader call.

    Runs the weight loader with a CopyNumelCounter to track copy operations.
    """
    assert args.arguments["param"].device.type == "meta"
    with MetaCopyCounter() as counter:
        weight_loader(*args.args, **args.kwargs)
    return counter.copied_numel


# -----------------------------------------------------------------------------
# Layer Utilities
# -----------------------------------------------------------------------------


def get_layer_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Get all parameters and buffers from a module as a dict."""
    params, buffers = get_layer_params_buffers(layer)
    return params | buffers


def get_layer_params_buffers(layer: torch.nn.Module) -> LayerTensors:
    return (
        {
            name: param
            for name, param in layer._parameters.items()
            if param is not None
        },
        {
            name: buffer
            for name, buffer in layer._buffers.items()
            if buffer is not None
        },
    )


def get_layer_size(layer: torch.nn.Module) -> int:
    """Calculate total number of elements across all tensors in a layer."""
    return sum(tensor.numel() for tensor in get_layer_tensors(layer).values())


# -----------------------------------------------------------------------------
# Main Reloading Functions
# -----------------------------------------------------------------------------


def record_metadata_for_reloading(layer: torch.nn.Module) -> None:
    """
    Record layer metadata needed for later reloading.

    Stores parameter and buffer metadata as meta tensors for restoration.
    Must be called before `layerwise_restore_and_process`.

    Note: Buffers will be restored as parameters.
    """
    params, buffers = get_layer_params_buffers(layer)
    params = {name: to_meta_tensor(param) for name, param in params.items()}
    buffers = {name: to_meta_tensor(buffer) for name, buffer in buffers.items()}
    LAYER_RELOADING_INFO[layer] = LayerReloadingInfo(restore_metadata=(params, buffers))


def restore_layer_on_meta(layer: torch.nn.Module):
    if layer not in LAYER_RELOADING_INFO:
        raise ValueError("Must call `record_metadata_for_reloading` reloading")
    
    info = LAYER_RELOADING_INFO[layer]
    
    for name in get_layer_tensors(layer).keys():
        delattr(layer, name)

    restore_params, restore_buffers = info.restore_metadata
    for name, param in restore_params.items():
        layer.register_parameter(name, param)
    for name, buffer in restore_buffers.items():
        layer.register_buffer(name, buffer)


@torch.no_grad()
def layerwise_restore_and_process(layer: torch.nn.Module) -> None:
    """
    Set up layerwise weight loading with deferred processing.

    Must be called after `record_metadata_for_reloading`. This function:
    1. Saves current kernel tensors for later copying
    2. Restores layer parameters/buffers from metadata (on meta device)
    3. Wraps weight loaders to defer processing until all weights are loaded

    When all weights for a layer are loaded, the wrapped loaders will:
    1. Materialize the layer onto the target device
    2. Load all cached weights
    3. Run quantization processing if applicable
    4. Copy processed values back to original tensor storage
    """
    if layer not in LAYER_RELOADING_INFO:
        raise ValueError("Must call `record_metadata_for_reloading` reloading")
    
    info = LAYER_RELOADING_INFO[layer]

    # Save current tensors for later copying
    info.kernel_tensors = get_layer_params_buffers(layer)

    # Restore layer parameters/buffers onto meta device
    restore_layer_on_meta(layer)

    # Track loading progress to determine when to process/copy
    info.load_numel = 0
    info.load_numel_total = get_layer_size(layer)

    # Wrap each parameter's weight loader
    # Note that nested wrapping will occur for shared tensors
    for name, param in get_layer_tensors(layer).items():
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader_signature = inspect.signature(weight_loader)

        def make_restore_loader(
            layer: torch.nn.Module,
            param_name: str,
            original_loader: Callable,
            loader_signature: inspect.Signature,
        ) -> Callable:
            """Create a wrapped weight loader that defers processing."""

            @wraps(original_loader, assigned=("__doc__", "__annotations__"))
            def restore_and_process_loader(*args, **kwargs):
                # Bind and normalize arguments
                bound_args = loader_signature.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Cache loaded weights, track loading progress
                # `get_numel_loaded` triggers inner wrapped function for shared tensors
                info = LAYER_RELOADING_INFO[layer]
                info.loaded_weights.append((param_name, bound_args))
                info.load_numel += get_numel_loaded(original_loader, bound_args)

                # Process and copy when all weights are loaded
                if (
                    info.load_numel >= info.load_numel_total
                    and not isinstance(layer, (Attention, MLAAttention))
                ):
                    _finalize_process_layer(layer, original_loader)

            return restore_and_process_loader

        param.weight_loader = make_restore_loader(
            layer, name, weight_loader, weight_loader_signature
        )


def _finalize_process_layer(
    layer: torch.nn.Module,
    weight_loader: Callable,
) -> None:
    """
    Finalize layer loading after all weights have been cached.

    This function:
    1. Materializes the layer onto the target device
    2. Loads all cached weights
    3. Runs quantization processing if applicable
    4. Copies processed values back to original tensor storage
    """
    # Get info related to reloading
    info = LAYER_RELOADING_INFO[layer]

    # Materialize layer onto device
    materialize_layer(layer)

    # Load all cached weights into materialized layer
    for name, args in info.loaded_weights:
        param = getattr(layer, name)
        args.arguments["param"] = param
        unwrap_loader(weight_loader)(*args.args, **args.kwargs)

    # Process weights (quantization, repacking, etc.)
    # Attention/MLA gets processed in `finalize_layerwise_restore_and_process`
    quant_method = getattr(layer, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        quant_method.process_weights_after_loading(layer)

    # Copy processed values into original tensor storage (preserves cudagraph refs)
    parameters, buffers = info.kernel_tensors
    for param in parameters.values():
        param.data.copy_(getattr(layer, name))
        layer.register_parameter(name, param)
    for buffer in buffers.values():
        buffer.data.copy_(getattr(layer, name))
        layer.register_buffer(name, buffer)

    info.reset()


def unwrap_loader(loader: Callable) -> Callable:
    while loader.__name__ == "restore_and_process_loader":
        loader = loader.__wrapped__

    return loader
    


def finalize_layerwise_restore_and_process(layer: torch.nn.Module) -> None:
    """
    Remove the outermost layer of weight loading wrappers.

    This function should be called after `layerwise_restore_and_process` to
    unwrap the layerwise weight loaders and restore original functionality.

    Also handles cleanup for modules (like Attention) that have kernel tensors
    but don't load module tensors (e.g., when model is not quantized).
    """
    for param in get_layer_tensors(layer).values():
        if hasattr(param, "weight_loader"):
            # TODO: limit unwrapping to only the layerwise weight loaders
            param.weight_loader = unwrap_loader(param.weight_loader)

    info = LAYER_RELOADING_INFO[layer]

    # Cannot process a layer if only some elements are loaded
    if info.load_numel > 0 and info.load_numel < info.load_numel_total:
        raise ValueError("Only some weights loaded")
    
    # Attention/MLA layers are processed after all other layers
    if isinstance(layer, (Attention, MLAAttention)):
        if info.load_numel > 0:
            raise NotImplementedError("Layerwise reloading of Q/K/V scale weights")
        
        # No processing: place kernel tensors back
        else:
            for name in get_layer_tensors(layer).keys():
                delattr(layer, name)

            parameters, buffers = LAYER_RELOADING_INFO[layer].kernel_tensors
            for name, param in parameters.items():
                layer.register_parameter(name, param)
            for name, buffer in buffers.items():
                layer.register_buffer(name, buffer)

    info.reset()


def supports_reloading():
    # TODO:
    # reload counter (don't wrap first time)
    # 
    func = ...
    model = ...
    def wrapper(*args, **kwargs):
        model.apply(layerwise_restore_and_process)
        ret = func(*args, **kwargs)
        model.apply(finalize_layerwise_restore_and_process)

        return ret

