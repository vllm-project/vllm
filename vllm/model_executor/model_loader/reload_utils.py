# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import wraps
from itertools import chain

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)

RESTORE_ATTRS = ["weight_loader"]


"""
TODO:
* pass arguments to process_weights_after_loading
* check composability with MLA processing
* check composability with EPLB

Limitations:
* Does not compose with CPU offloading. This is because `device_loading_context`
doesn't work in all cases? For example, when parameter is renamed.
"""


def record_metadata_for_reloading(model: torch.nn.Module):
    """

    Note that buffers will be restored as parameters

    """
    for module in model.modules():
        module.restore_metadata = {}

        for name, param in get_module_tensors(module).items():
            meta = param.to(device=torch.device("meta"))

            for attr_name in RESTORE_ATTRS:
                if hasattr(module, attr_name):
                    setattr(meta, attr_name, getattr(module, attr_name))

            module.restore_metadata[name] = meta


def layerwise_restore_and_process(layer: torch.nn.Module):
    """
    Called after `record_metadata_for_reloading`.
    """
    # get parameters for later copying
    original_parameters = get_module_tensors(layer)

    # restore layer onto meta device
    if hasattr(layer, "restore_metadata"):
        for name in layer.original_parameters.keys():
            delattr(layer, name)

        for name, tensor in layer.restore_metadata.items():
            setattr(layer, name, tensor)

    # track loading progress to determine when to process/copy
    load_numel_remaining = get_layer_size(layer)

    for name, param in get_module_tensors(layer).items():
        original_weight_loader = getattr(param, "weight_loader", default_weight_loader)

        @wraps(original_weight_loader)
        def restore_and_process_loader(
            param: torch.Tensor, loaded_weight: torch.Tensor
        ):
            nonlocal load_numel_remaining

            # materialize meta tensor if necessary
            if param.device == torch.device("meta"):
                param = materialize_meta_tensor(param)
                setattr(layer, name, param)

            # load weight into target
            original_weight_loader(param, loaded_weight)

            # process and copy when weights are loaded
            load_numel_remaining -= loaded_weight.numel()
            if load_numel_remaining <= 0:
                # process weights (no-op for online quant)
                process_weights_after_loading(layer)

                # copy newly processed values into
                # original tensor data (and cudagraph)
                assert get_module_tensors(layer).keys() == original_parameters.keys()
                for name in original_parameters.keys():
                    original_parameters[name].copy_(layer[name])
                    layer[name] = original_parameters[name]

        param.weight_loader = restore_and_process_loader


def unwrap_weight_loaders(layer: torch.nn.Module):
    """
    Removes outer-most layer of weight loading wrappers. This function can be called
    after `layerwise_restore_and_process` to unwrap layerwise weight loaders.
    """
    for param in get_module_tensors(layer).values():
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        if hasattr(weight_loader, "__wrapped__"):
            param.weight_loader = weight_loader.__wrapped__


def get_layer_size(layer: torch.nn.Module) -> int:
    return sum(
        tensor.numel()
        for tensor in chain(layer._parameters.values(), layer._buffers.values())
    )


def materialize_meta_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # should be called within a torch device context
    return torch.empty_strided(
        size=tuple(tensor.size()),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        requires_grad=False,  # set below to match input
    )


def get_module_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return module._parameters.copy().update(module._buffers)
