# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from functools import wraps
from weakref import WeakKeyDictionary

import torch

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .meta import (
    capture_layer_to_meta,
    get_numel_loaded,
    materialize_layer,
    materialize_layer_tensors_with_device_meta,
    restore_layer_on_meta,
)
from .types import LayerReloadingInfo
from .utils import (
    CopyCounter,
    get_layer_params_buffers,
    get_layer_size,
    get_layer_tensors,
)

logger = init_logger(__name__)

__all__ = [
    "get_layerwise_info",
    "record_metadata_for_reloading",
    "initialize_layerwise_reload",
    "finalize_layerwise_reload",
]


# Global dict storing information used for layerwise restoring, loading, and processing.
# For more information regarding what info is stored when, see `LayerReloadingInfo`
#
# Use a weak ref dictionary so that modules can be freed when the model is freed.
# Values are sanitized from references to the layer key in order to avoid circular refs
LAYERWISE_INFO: WeakKeyDictionary[torch.nn.Module, LayerReloadingInfo] = (
    WeakKeyDictionary()
)


def get_layerwise_info(layer: torch.nn.Module) -> LayerReloadingInfo:
    """
    Get information related to restoring and layerwise processing. If no previous
    information existed, a new entry is constructed
    """
    if layer not in LAYERWISE_INFO:
        LAYERWISE_INFO[layer] = LayerReloadingInfo()

    return LAYERWISE_INFO[layer]


def record_metadata_for_reloading(model: torch.nn.Module):
    """
    Record layer metadata needed for later reloading.

    Stores parameter and buffer metadata as meta tensors for restoration.
    Must be called before `initialize_layerwise_reload`.
    """
    for layer in model.modules():
        info = get_layerwise_info(layer)
        info.restore_metadata = capture_layer_to_meta(layer)


@torch.no_grad()
def initialize_layerwise_reload(
    model: torch.nn.Module,
    is_reload: bool = True,
    target_device: torch.device | None = None,
):
    """
    Set up layerwise weight reloading|initial loading with deferred processing.

    For weight reloading (is_reload = True):

        Must be called after `record_metadata_for_reloading`. This function:
        1. Saves current kernel tensors for later copying
        2. Restores layer parameters/buffers from metadata (on meta device)
        3. Wraps weight loaders to defer processing until all weights are loaded

        When all weights for a layer are loaded, the wrapped loaders will:
        1. Materialize the layer onto the target device
        2. Load all cached weights
        3. Run quantization processing if applicable
        4. Copy processed values back to original tensor storage

    For weight initial loading (is_reload = False):

        1. saves `target_device` to be used during weight loading
        2. wraps all weight loaders to enable deffered processing
    """
    # disable torchao reloading to avoid infinite recursion
    model._original_do_torchao_reload = getattr(model, "_do_torchao_reload", False)
    model._do_torchao_reload = False

    for layer in model.modules():
        info = get_layerwise_info(layer)

        # Skip if the layer has already been initialized
        if info.can_process():
            continue

        # Save current tensors for later copying
        if is_reload:
            # reload path, TODO document more
            assert target_device is None
            info.kernel_tensors = get_layer_params_buffers(layer)
            # Restore layer parameters/buffers onto meta device
            restore_layer_on_meta(layer, info)
        else:
            # initial load path
            assert target_device is not None
            info.initial_load_target_device = target_device

        # Track loading progress to determine when to process/copy
        info.load_numel = 0
        info.load_numel_total = get_layer_size(layer)

        # Wrap each parameter's weight loader
        # Note that nested wrapping will occur for shared tensors
        for name, tensor in get_layer_tensors(layer).items():
            if _get_weight_loader(tensor).__name__ != "online_process_loader":
                tensor.weight_loader = make_online_process_loader(
                    layer, name, is_reload
                )


def make_online_process_loader(
    layer: torch.nn.Module, param_name: str, is_reload: bool
) -> Callable:
    """
    Create a wrapped weight loader that defers processing.

    * If `is_reload` is True, weights are cached on CPU and only actually
      loaded when every chunk has been processed.
    * If `is_reload` is False, weights are loaded directly (no CPU cache)
    """
    info = get_layerwise_info(layer)
    param = getattr(layer, param_name)
    original_loader = _get_original_loader(param)
    loader_signature = inspect.signature(original_loader)

    @wraps(original_loader, assigned=("__doc__", "__annotations__"))
    def online_process_loader(*args, **kwargs):
        if not info.can_process():
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
            # see Limitations(4)
            logger.debug(
                "%s: Excessive loading of param '%s'",
                layer.__class__.__name__,
                param_name,
            )
            return

        # Bind and normalize arguments
        bound_args = loader_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        if is_reload:
            # Cache loaded weights, track loading progress. Weights will be
            # actually loaded later.
            info.loaded_weights.append((param_name, bound_args))
            num_loaded, ret = get_numel_loaded(original_loader, bound_args)

        else:
            if info.load_numel == 0:
                # If this is the first weight chunk being loaded, materialize any
                # non-materialized tensors in this layer
                assert info.initial_load_target_device is not None
                with info.initial_load_target_device:
                    materialize_layer_tensors_with_device_meta(layer)

            # Update param reference to point to the current (materialized)
            # tensor instead of the old meta tensor that was captured when
            # the loader was wrapped
            current_param = getattr(layer, param_name)
            bound_args.arguments["param"] = current_param

            # Load weights directly
            with CopyCounter() as counter:
                ret = original_loader(*bound_args.args, **bound_args.kwargs)
            num_loaded = counter.copied_numel

        info.load_numel += num_loaded

        logger.debug(
            "%s: %d / %d",
            layer.__class__.__name__,
            info.load_numel,
            info.load_numel_total,
        )

        # Process and copy when all weights are loaded
        if info.load_numel >= info.load_numel_total and not isinstance(  # type: ignore[operator]
            layer, (Attention, MLAAttention)
        ):
            _layerwise_process(layer, info, is_reload)

        return ret

    return online_process_loader


def finalize_layerwise_reload(
    model: torch.nn.Module, model_config: ModelConfig, is_reload: bool = True
):
    """
    Remove the outermost layer of weight loading wrappers.

    This function should be applied after `initialize_layerwise_reload` is applied
    unwrap the layerwise weight loaders.

    Also processes Attention/MLA layers, which must be processed after all other layers
    """
    model._do_torchao_reload = model._original_do_torchao_reload

    for layer in model.modules():
        info = get_layerwise_info(layer)

        # Attention/MLA layers are processed after all other layers
        if isinstance(layer, (Attention, MLAAttention)):
            if info.load_numel > 0:
                raise NotImplementedError(
                    "Layerwise reloading of Q/K/V scale weights is not implemented yet"
                )

            else:
                if is_reload:
                    _place_kernel_tensors(layer, info)
                layer.process_weights_after_loading(model_config.dtype)

        # No weights were loaded, place kernel tensors back
        elif info.can_process() and info.load_numel <= 0:
            if is_reload:
                _place_kernel_tensors(layer, info)

        # Process non-attention layers which did not load all elements. This can happen
        # if the created weight has extra padding elements which are not loaded
        # Having too many of these delayed layers can lead to execess memory usage
        # see Limitations(4)
        elif info.load_numel > 0 and info.load_numel < info.load_numel_total:  # type: ignore[operator]
            logger.debug("%s: Delayed processing", layer.__class__.__name__)
            _layerwise_process(layer, info, is_reload)

        info.reset()


def _layerwise_process(
    layer: torch.nn.Module, info: LayerReloadingInfo, is_reload: bool
):
    """
    Finalize layer loading after all weights have been cached.

    If `is_reload` is True, this function:
        1. Materializes the layer onto the target device
        2. Loads all cached weights
        3. Runs `process_weights_after_loading` if applicable
        4. Copies processed values back to original tensor storage

    If `is_reload` is False, this function:
        1. Runs `process_weights_after_loading` if applicable
    """
    if is_reload:
        # Materialize layer tensors onto device
        materialize_layer(layer)

    # Unwrap layerwise loading wrappers
    for param in get_layer_tensors(layer).values():
        param.weight_loader = _get_original_loader(param)

    if is_reload:
        # Load all cached weights into materialized layer (using original loaders)
        for name, args in info.loaded_weights:
            param = getattr(layer, name)
            args.arguments["param"] = param
            param.weight_loader(*args.args, **args.kwargs)

    # Process weights (quantization, repacking, etc.)
    # Attention/MLA are processed in `finalize_layerwise_reload`
    quant_method = getattr(layer, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        quant_method.process_weights_after_loading(layer)

    if is_reload:
        # Reloading path
        # Copy processed values into original tensor storage (preserves cudagraph refs)
        # this code is a no-op if not reloading (because kernel tensors is empty)
        parameters, buffers = info.kernel_tensors
        for name, param in parameters.items():
            param.data.copy_(getattr(layer, name))
        for name, buffer in buffers.items():
            buffer.data.copy_(getattr(layer, name))

        _place_kernel_tensors(layer, info)

    info.reset()
    logger.debug("%s: Processed", layer.__class__.__name__)


def _get_original_loader(tensor: torch.Tensor) -> Callable:
    """Return the weight loader with any layerwise wrappers removed"""
    loader = _get_weight_loader(tensor)
    while loader.__name__ == "online_process_loader":
        loader = loader.__wrapped__  # type: ignore[union-attr]

    return loader


def _get_weight_loader(tensor: torch.Tensor):
    return getattr(tensor, "weight_loader", default_weight_loader)


def _place_kernel_tensors(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Assign each parameter/buffer in `info.kernel_tensors` back onto `layer`"""
    for name in get_layer_tensors(layer):
        delattr(layer, name)

    parameters, buffers = info.kernel_tensors
    for name, param in parameters.items():
        layer.register_parameter(name, param)
    for name, buffer in buffers.items():
        layer.register_buffer(name, buffer)
