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
    "initialize_layerwise_reload",
    "finalize_layerwise_reload",
]


LAYER_RELOADING_INFO: dict[torch.nn.Module, LayerReloadingInfo] = dict()


def record_metadata_for_reloading(layer: torch.nn.Module) -> None:
    """
    Record layer metadata needed for later reloading.

    Stores parameter and buffer metadata as meta tensors for restoration.
    Must be called before `initialize_layerwise_reload`.

    Note: Buffers will be restored as parameters.
    """
    params, buffers = get_layer_params_buffers(layer)
    params = {name: to_meta_tensor(param) for name, param in params.items()}
    buffers = {name: to_meta_tensor(buffer) for name, buffer in buffers.items()}
    LAYER_RELOADING_INFO[layer] = LayerReloadingInfo(restore_metadata=(params, buffers))


@torch.no_grad()
def initialize_layerwise_reload(layer: torch.nn.Module) -> None:
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

    # Skip if the layer has already been initialized
    if info.is_initialized():
        return

    # Save current tensors for later copying
    info.kernel_tensors = get_layer_params_buffers(layer)

    # Restore layer parameters/buffers onto meta device
    restore_layer_on_meta(layer, info)

    # Track loading progress to determine when to process/copy
    info.load_numel = 0
    info.load_numel_total = get_layer_size(layer)

    print(f"Initialize {layer.__class__.__name__}")

    # Wrap each parameter's weight loader
    # Note that nested wrapping will occur for shared tensors
    for param_name, param in get_layer_tensors(layer).items():

        def make_restore_loader(
            layer: torch.nn.Module,
            param_name: str,
            param: torch.Tensor,
        ) -> Callable:
            """Create a wrapped weight loader that defers processing."""
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader_signature = inspect.signature(weight_loader)

            @wraps(weight_loader, assigned=("__doc__", "__annotations__"))
            def restore_and_process_loader(*args, **kwargs):
                # Bind and normalize arguments
                bound_args = weight_loader_signature.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Cache loaded weights, track loading progress
                # `get_numel_loaded` triggers inner wrapped function for shared tensors
                info = LAYER_RELOADING_INFO[layer]
                if info.is_initialized():
                    info.loaded_weights.append((param_name, bound_args))
                    info.load_numel += get_numel_loaded(weight_loader, bound_args)
                else:
                    # Unfortunately, some qconfigs are set up to load the same weight
                    # multiple times. For example, CT_WNA16 loads `weight_shape` for
                    # each of the qkv partitions. This results in layers loading extra
                    # weights (beyond load_numel_total) after it's already processed.
                    # 
                    # Best solution is to ensure that `load_numel_total` reflects the
                    # actual number of weights loaded, either by modifying qconfigs to
                    # create as many weights as loaded (see padding issue as well)
                    # or maybe capturing how many weights are loaded on first pass
                    # 
                    # For now, `load_numel_total` is still safe to use as long as
                    # there's no way to reach `load_numel_total` without loading all
                    # necessary weights. `weight_shape` is very small, so this is safe.
                    return

                print(
                    f"{layer.__class__.__name__}: {info.load_numel} / {info.load_numel_total}"
                )

                # Process and copy when all weights are loaded
                if info.load_numel >= info.load_numel_total and not isinstance(
                    layer, (Attention, MLAAttention)
                ):
                    _layerwise_process(layer, info)

            return restore_and_process_loader

        param.weight_loader = make_restore_loader(layer, param_name, param)


def finalize_layerwise_reload(layer: torch.nn.Module) -> None:
    """
    Remove the outermost layer of weight loading wrappers.

    This function should be applied after `initialize_layerwise_reload` is applied
    unwrap the layerwise weight loaders.

    Also processes Attention/MLA layers, which must be processed after all other layers
    """
    info = LAYER_RELOADING_INFO[layer]

    # Attention/MLA layers are processed after all other layers
    if isinstance(layer, (Attention, MLAAttention)) and info.load_numel > 0:
        # when implementing, remember to unwrap layerwise loaders
        raise NotImplementedError("Layerwise reloading of Q/K/V scale weights")

    # Process non-attention layers which did not load all elements due to (padding)
    # Having too many of these delayed layers can lead to execess memory usage
    if info.load_numel > 0 and info.load_numel < info.load_numel_total:
        print("SKIP THING")
        print(f"skip thing: {layer.__class__.__name__}")
        #_layerwise_process(layer, info)

    # No weights were loaded, place kernel tensors back
    elif info.is_initialized():
        for name in get_layer_tensors(layer):
            delattr(layer, name)

        parameters, buffers = LAYER_RELOADING_INFO[layer].kernel_tensors
        for name, param in parameters.items():
            layer.register_parameter(name, param)
        for name, buffer in buffers.items():
            layer.register_buffer(name, buffer)

    info.reset()


def _layerwise_process(layer: torch.nn.Module, info: LayerReloadingInfo):
    """
    Finalize layer loading after all weights have been cached.

    This function:
    1. Materializes the layer onto the target device
    2. Loads all cached weights
    3. Runs quantization processing if applicable
    4. Copies processed values back to original tensor storage
    """
    # Materialize layer onto device
    materialize_layer(layer)

    # Unwrap layerwise loading wrappers
    for param in get_layer_tensors(layer).values():
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        param.weight_loader = _unwrap_loader(weight_loader)

    # Load all cached weights into materialized layer (using original loaders)
    for name, args in info.loaded_weights:
        param = getattr(layer, name)
        args.arguments["param"] = param
        param.weight_loader(*args.args, **args.kwargs)

    # if "weight_packed" in info.kernel_tensors[0]:
    #     breakpoint()
    print(f"start: {layer.__class__.__name__}")

    # Process weights (quantization, repacking, etc.)
    # Attention/MLA are processed in `finalize_layerwise_reload`
    quant_method = getattr(layer, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        quant_method.process_weights_after_loading(layer)

    # Copy processed values into original tensor storage (preserves cudagraph refs)
    parameters, buffers = info.kernel_tensors
    for name, param in parameters.items():
        param.data.copy_(getattr(layer, name))
        layer.register_parameter(name, param)
    for name, buffer in buffers.items():
        buffer.data.copy_(getattr(layer, name))
        layer.register_buffer(name, buffer)

    info.reset()

    print(f"process: {layer.__class__.__name__}")


def _unwrap_loader(loader: Callable) -> Callable:
    """Return the weight loader with any layerwise wrappers removed"""
    while loader.__name__ == "restore_and_process_loader":
        loader = loader.__wrapped__  # type: ignore[attr-defined]

    return loader
