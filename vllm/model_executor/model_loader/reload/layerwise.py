# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from functools import wraps

import torch

from vllm.attention.layer import Attention, MLAAttention
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .helpers import get_layer_params_buffers, get_layer_size, get_layer_tensors
from .meta import (
    get_numel_loaded,
    materialize_layer,
    restore_layer_on_meta,
    to_meta_tensor,
)
from .types import LayerReloadingInfo

logger = init_logger(__name__)

__all__ = [
    "record_metadata_for_reloading",
    "layerwise_restore_and_process",
    "finalize_layerwise_restore_and_process",
]


LAYER_RELOADING_INFO: dict[torch.nn.Module, LayerReloadingInfo] = dict()


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
    restore_layer_on_meta(layer, info)

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
                if info.load_numel >= info.load_numel_total and not isinstance(
                    layer, (Attention, MLAAttention)
                ):
                    _finalize_process_layer(layer, original_loader)

            return restore_and_process_loader

        param.weight_loader = make_restore_loader(
            layer, name, weight_loader, weight_loader_signature
        )


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
            param.weight_loader = _unwrap_loader(param.weight_loader)

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
            for name in get_layer_tensors(layer):
                delattr(layer, name)

            parameters, buffers = LAYER_RELOADING_INFO[layer].kernel_tensors
            for name, param in parameters.items():
                layer.register_parameter(name, param)
            for name, buffer in buffers.items():
                layer.register_buffer(name, buffer)

    info.reset()


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
        _unwrap_loader(weight_loader)(*args.args, **args.kwargs)

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


def _unwrap_loader(loader: Callable) -> Callable:
    while loader.__name__ == "restore_and_process_loader":
        loader = loader.__wrapped__  # type: ignore[attr-defined]

    return loader
