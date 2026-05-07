# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from functools import wraps
from weakref import WeakKeyDictionary, WeakSet

import torch

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .meta import (
    SKIP_TENSORS,
    capture_layer_to_meta,
    get_numel_loaded,
    materialize_layer,
    restore_layer_on_meta,
)
from .types import LayerReloadingInfo
from .utils import (
    get_info_size,
    get_layer_params_buffers,
    get_layer_size,
    get_layer_tensors,
    has_device_tensors,
)

logger = init_logger(__name__)

__all__ = [
    "get_layerwise_info",
    "record_metadata_for_reloading",
    "initialize_layerwise_reload",
    "finalize_layerwise_processing",
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

# Global set used to track loading for logging purposes only
LOADING_LAYERS: WeakSet[torch.nn.Module] = WeakSet()


def get_layerwise_info(layer: torch.nn.Module) -> LayerReloadingInfo:
    """
    Get information related to restoring and layerwise processing. If no previous
    information existed, a new entry is constructed
    """
    if layer not in LAYERWISE_INFO:
        LAYERWISE_INFO[layer] = LayerReloadingInfo(
            restore_metadata=({}, {}),
            restore_device=torch.get_default_device(),
        )

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
        info.restore_device = torch.get_default_device()


@torch.no_grad()
def initialize_layerwise_reload(model: torch.nn.Module):
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
    # disable torchao reloading to avoid infinite recursion
    model._original_do_torchao_reload = getattr(model, "_do_torchao_reload", False)
    model._do_torchao_reload = False

    for layer in model.modules():
        info = get_layerwise_info(layer)

        # Skip if the layer has already been initialized
        if info.can_load():
            continue

        # Save current tensors for later copying
        info.kernel_tensors = get_layer_params_buffers(layer)

        # Restore layer parameters/buffers onto meta device
        restore_layer_on_meta(layer, info)

        # Wrap weight loaders to buffer loading
        initialize_online_processing(layer)


def initialize_online_processing(layer: torch.nn.Module):
    """
    Wrap a layer's weight loaders with online processing loaders.
    Called by either `initialize_layerwise_reload` or an online quantization scheme,
    prevents double wrapping in the case of online quantization + reloading

    :param layer: layer whose parameter weight loaders will be wrapped
    """
    info = get_layerwise_info(layer)

    # Track loading progress to determine when to process/copy
    info.load_numel = 0
    info.load_numel_total = get_layer_size(layer)

    # Wrap each parameter's weight loader
    # Note that nested wrapping will occur for shared tensors
    for name, tensor in get_layer_tensors(layer).items():
        if name in SKIP_TENSORS:
            continue
        if _get_weight_loader(tensor).__name__ != "online_process_loader":
            tensor.weight_loader = make_online_process_loader(layer, name)


def make_online_process_loader(layer: torch.nn.Module, param_name: str) -> Callable:
    """Create a wrapped weight loader that defers processing."""
    info = get_layerwise_info(layer)
    param = getattr(layer, param_name)
    original_loader = _get_original_loader(param)
    loader_signature = inspect.signature(original_loader)

    @wraps(original_loader, assigned=("__doc__", "__annotations__"))
    def online_process_loader(*args, **kwargs):
        if not info.can_load():
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
            logger.debug("%s: Excessive loading", layer.__class__.__name__)
            return

        # Bind and normalize arguments
        bound_args = loader_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Buffer loaded weights, track loading progress
        info.loaded_weights.append((param_name, bound_args))
        num_loaded, ret = get_numel_loaded(original_loader, bound_args)
        info.load_numel += num_loaded

        logger.debug(
            "%s: %d / %d",
            layer.__class__.__name__,
            info.load_numel,
            info.load_numel_total,
        )

        # Do not online process attention layers, must wait until finalize
        if isinstance(layer, (Attention, MLAAttention)):
            return ret

        # Log warnings allocating excessive buffers on device
        if has_device_tensors(bound_args):
            LOADING_LAYERS.add(layer)
            if len(LOADING_LAYERS) >= 2:
                names = sorted([layer.__class__.__name__ for layer in LOADING_LAYERS])
                mem_used = sum(
                    get_info_size(LAYERWISE_INFO[layer]) for layer in LOADING_LAYERS
                )
                logger.warning_once(
                    "Allocating %.1f MB of device memory to buffers to load %s layers. "
                    "This extra memory usage can be avoided by ordering weights "
                    "by their parent layer when reloading.",
                    mem_used / 1e6,
                    str(list(names)),
                )

        # Process and copy when all weights are loaded
        if info.load_numel >= info.load_numel_total:  # type: ignore[operator]
            _layerwise_process(layer, info)
            LOADING_LAYERS.discard(layer)

        return ret

    return online_process_loader


def finalize_layerwise_processing(model: torch.nn.Module, model_config: ModelConfig):
    """
    Apply processing to any layers which were not layerwise processed during loading.
    This includes attention layers and layers which have weight elements which are not
    loaded (due to padding).

    This function should be applied after `initialize_layerwise_reload` is applied
    unwrap the layerwise weight loaders.

    :param model: model to finalize processing for
    :param model_config: config needed for applying processing to attention layers
    """
    if hasattr(model, "_original_do_torchao_reload"):
        model._do_torchao_reload = model._original_do_torchao_reload

    deferred_attn: list[tuple[torch.nn.Module, LayerReloadingInfo]] = []

    for layer in model.modules():
        info = get_layerwise_info(layer)
        if not info.can_load():
            info.reset()
            continue

        # Attention/MLA layers are processed after all other layers
        if isinstance(layer, (Attention, MLAAttention)):
            deferred_attn.append((layer, info))
            continue

        # No weights were loaded
        if info.load_numel <= 0:
            # first load: checkpoint did not contain weights for this layer
            if info.kernel_tensors is None:
                _layerwise_process(layer, info)
                continue

            # reloading: place kernel tensors back as a fallback
            elif info.load_numel_total > 0:  # type: ignore[operator]
                logger.warning("%s: Failed to load weights", layer.__class__.__name__)
                _place_kernel_tensors(layer, info)

        # Process non-attention layers which did not load all elements. This can happen
        # if the created weight has extra padding elements which are not loaded
        # Having too many of these delayed layers can lead to excess memory usage
        # see Limitations(4)
        elif info.load_numel > 0 and info.load_numel < info.load_numel_total:  # type: ignore[operator]
            logger.debug("%s: Delayed processing", layer.__class__.__name__)
            _layerwise_process(layer, info)

        info.reset()

    # Process attention layers after all other layers are done
    for layer, info in deferred_attn:
        _finalize_attention_layer(layer, info, model_config)
        info.reset()

    LOADING_LAYERS.clear()


def finalize_layerwise_reload(*args, **kwargs):
    finalize_layerwise_processing(*args, **kwargs)


def _finalize_attention_layer(
    layer: torch.nn.Module, info: LayerReloadingInfo, model_config: ModelConfig
) -> None:
    if info.load_numel > 0 and info.kernel_tensors is not None:
        # Reload with new scale weights from checkpoint
        _place_kernel_tensors(layer, info)
        _reload_attention_scales(layer, info)
    elif info.load_numel > 0 or info.kernel_tensors is None:
        raise ValueError(
            "Layerwise loading of attention layers is not supported. "
            "Attention must always process after linears."
        )
    else:
        _place_kernel_tensors(layer, info)
    layer.process_weights_after_loading(model_config.dtype)


def _reload_attention_scales(layer: torch.nn.Module, info: LayerReloadingInfo) -> None:
    """Load and process attention scale weights (k_scale, v_scale, etc.)
    during reload.

    Assumes dtype/shapes of attention tensors do not change during
    processing, since we use .data.copy_() to preserve kernel tensor
    references."""
    quant_method = getattr(layer, "quant_method", None)
    if quant_method is None:
        return

    # Re-create scale Parameters with sentinel values so unloaded scales
    # are correctly detected by process_weights_after_loading
    quant_method.create_weights(layer)

    for name, args in info.loaded_weights:
        param = getattr(layer, name)
        args.arguments["param"] = param
        _get_weight_loader(param)(*args.args, **args.kwargs)

    quant_method.process_weights_after_loading(layer)

    _copy_and_restore_kernel_tensors(layer, info)


def _layerwise_process(layer: torch.nn.Module, info: LayerReloadingInfo):
    """
    Finalize layer loading after all weights have been buffered.

    This function:
    1. Materializes the layer onto the target device
    2. Loads all buffered weights
    3. Runs quantization processing if applicable
    4. Copies processed values back to original tensor storage
    """
    # Materialize layer tensors onto device
    materialize_layer(layer, info)

    # Reset online quantization flag so process_weights_after_loading
    # will run again during reload
    if hasattr(layer, "_already_called_process_weights_after_loading"):
        delattr(layer, "_already_called_process_weights_after_loading")

    # Unwrap layerwise loading wrappers
    for param in get_layer_tensors(layer).values():
        param.weight_loader = _get_original_loader(param)

    # Load all buffered weights into materialized layer (using original loaders)
    for name, args in info.loaded_weights:
        param = getattr(layer, name)
        args.arguments["param"] = param
        param.weight_loader(*args.args, **args.kwargs)

    # Process weights (quantization, repacking, etc.)
    quant_method = getattr(layer, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        quant_method.process_weights_after_loading(layer)

    # Copy processed values into original tensor storage (preserves cudagraph refs)
    # this code is a no-op if not reloading (because kernel tensors is empty)
    if info.kernel_tensors is not None:
        _copy_and_restore_kernel_tensors(layer, info)

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


def _copy_and_restore_kernel_tensors(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Copy processed values into original kernel tensor storage and restore
    kernel tensor references on the layer. Preserves cudagraph references."""
    assert info.kernel_tensors is not None
    parameters, buffers = info.kernel_tensors
    for name, param in parameters.items():
        param.data.copy_(getattr(layer, name))
    for name, buffer in buffers.items():
        buffer.data.copy_(getattr(layer, name))

    _place_kernel_tensors(layer, info)


def _place_kernel_tensors(layer: torch.nn.Module, info: LayerReloadingInfo):
    for name in get_layer_tensors(layer):
        delattr(layer, name)

    assert info.kernel_tensors is not None
    parameters, buffers = info.kernel_tensors
    for name, param in parameters.items():
        layer.register_parameter(name, param)
    for name, buffer in buffers.items():
        layer.register_buffer(name, buffer)
