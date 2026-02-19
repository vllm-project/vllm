# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from functools import wraps
from weakref import WeakKeyDictionary

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload.layerwise import (
    _get_original_loader,
    _get_weight_loader,
)

from .meta import (
    materialize_layer_tensors_with_device_meta,
)
from .types import LayerReloadingInfo
from .utils import get_layer_size, get_layer_tensors

logger = init_logger(__name__)

# Global dict storing information used for layerwise loading
INITIAL_LOAD_LAYERWISE_INFO: WeakKeyDictionary[torch.nn.Module, LayerReloadingInfo] = (
    WeakKeyDictionary()
)


def get_initial_load_layerwise_info(layer: torch.nn.Module) -> LayerReloadingInfo:
    """
    Get information related to restoring and layerwise processing. If no previous
    information existed, a new entry is constructed
    """
    if layer not in INITIAL_LOAD_LAYERWISE_INFO:
        INITIAL_LOAD_LAYERWISE_INFO[layer] = LayerReloadingInfo()

    return INITIAL_LOAD_LAYERWISE_INFO[layer]


# TODO(before review): move fp8's one to a common place and use that
class CopyCounter(TorchDispatchMode):
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

        if func is torch.ops.aten.copy_.default:
            assert args[0].numel() == args[1].numel()
            self.copied_numel += args[0].numel()

        return func(*args, **kwargs)


@torch.no_grad()
def initialize_layerwise_initial_load(model: torch.nn.Module, target_device):
    """
    Initialize layerwise initial loading of model weights. In detail:

    1. set up global state to track how many elements have been loaded
       into each layer
    2. wrap original weight loaders to turn on layerwise post-processing.
       Specifically, when all of a weight's chunks are loaded, the
       `process_weights_after_loading` function will be called immediately.
       For online quantiation this minimizes peak memory usage compared
       to loading weights for the entire model first and then post-processing
       weights.
    """
    for layer in model.modules():
        info = get_initial_load_layerwise_info(layer)

        # Track loading progress to determine when to process/copy
        info.load_numel = 0
        info.load_numel_total = get_layer_size(layer)
        info.load_device = target_device

        # Wrap each parameter's weight loader
        # Note that nested wrapping will occur for shared tensors
        for name, tensor in get_layer_tensors(layer).items():
            if (
                _get_weight_loader(tensor).__name__
                != "online_initial_load_process_loader"
            ):
                tensor.weight_loader = make_online_initial_load_process_loader(
                    layer, name
                )


def make_online_initial_load_process_loader(
    layer: torch.nn.Module, param_name: str
) -> Callable:
    """Create a wrapped weight loader that defers processing."""
    info = get_initial_load_layerwise_info(layer)
    param = getattr(layer, param_name)
    original_loader = _get_original_loader(param)
    loader_signature = inspect.signature(original_loader)

    @wraps(original_loader, assigned=("__doc__", "__annotations__"))
    def online_initial_load_process_loader(*args, **kwargs):
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
            logger.debug("%s: Excessive loading", layer.__class__.__name__)
            return

        if info.load_numel == 0:
            # When the first weight chunk for a layer is seen,
            # Materialize any layer tensors on device meta onto device.
            # For most layers this is a no-op. For layers which initialize
            # weights on device meta during `create_weights`, this is where
            # the materialization happens.
            with info.load_device:
                materialize_layer_tensors_with_device_meta(layer)

        # Bind and normalize arguments
        bound_args = loader_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Update param reference to point to the current (materialized) tensor
        # instead of the old meta tensor that was captured when the loader was wrapped
        current_param = getattr(layer, param_name)
        bound_args.arguments["param"] = current_param

        with CopyCounter() as counter:
            ret = original_loader(*bound_args.args, **bound_args.kwargs)

        info.load_numel += counter.copied_numel

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
            _layerwise_initial_load_process(layer, info)

        return ret

    return online_initial_load_process_loader


def finalize_layerwise_initial_load(model: torch.nn.Module, model_config: ModelConfig):
    """
    Call `process_weights_after_loading` for any layers that did not participate
    in layerwise loading:
    1. Attention (hardcoded out for now due to data dependencies)
    2. layers where not all elements were loaded during `model.load_weights()`
    """

    for layer in model.modules():
        info = get_initial_load_layerwise_info(layer)

        # Attention/MLA layers are processed after all other layers
        if isinstance(layer, (Attention, MLAAttention)):
            if info.load_numel > 0:
                raise NotImplementedError(
                    "Layerwise loading of Q/K/V scale weights is not implemented yet"
                )

            else:
                layer.process_weights_after_loading(model_config.dtype)

        # No weights were loaded, nothing to do
        elif info.can_process() and info.load_numel <= 0:
            pass

        # Process non-attention layers which did not load all elements. This can happen
        # if the created weight has extra padding elements which are not loaded
        elif info.load_numel > 0 and info.load_numel < info.load_numel_total:  # type: ignore[operator]
            logger.debug("%s: Delayed processing", layer.__class__.__name__)
            _layerwise_initial_load_process(layer, info)


def _layerwise_initial_load_process(layer: torch.nn.Module, info: LayerReloadingInfo):
    # Process weights (quantization, repacking, etc.)
    # Attention/MLA are processed in `finalize_layerwise_initial_load`
    quant_method = getattr(layer, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        quant_method.process_weights_after_loading(layer)
    logger.debug("%s: Processed", layer.__class__.__name__)
