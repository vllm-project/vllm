# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import wraps
from itertools import chain

import torch

from vllm.logger import init_logger
#from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)

RESTORE_ATTRS = ["weight_loader"]


"""
TODO:
* decide on reloading interface, back-compat with reload_weights
* pass arguments to process_weights_after_loading
* only onload once all weights are present
* check composability with MLA processing
* check composability with EPLB
* do attention/MLA processing after module weights are processed
    * probably means call process_after_weight_loading after weight loading

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
            meta = to_meta_tensor(param)

            module.restore_metadata[name] = meta


@torch.no_grad()
def layerwise_restore_and_process(layer: torch.nn.Module):
    """
    Called after `record_metadata_for_reloading`.
    """
    # get parameters for later copying
    original_parameters = get_module_tensors(layer)

    # restore layer onto meta device
    if hasattr(layer, "restore_metadata"):
        for name in original_parameters.keys():
            delattr(layer, name)

        for name, tensor in layer.restore_metadata.items():
            setattr(layer, name, tensor)

    # track loading progress to determine when to process/copy
    load_numel_remaining = get_layer_size(layer)

    import inspect

    for name, param in get_module_tensors(layer).items():
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        loaded_weight_kwargs: list[inspect.BoundArguments] = list()
        weight_loader_signature = inspect.signature(weight_loader)

        @wraps(weight_loader)
        def restore_and_process_loader(*args, **kwargs):
            nonlocal load_numel_remaining
            nonlocal loaded_weight_kwargs
            nonlocal name

            args = weight_loader_signature.bind(*args, **kwargs)
            args.apply_defaults()

            # cache loaded weights on load device until all are ready
            loaded_weight_kwargs.append(args)

            # process and copy when all weights are loaded
            load_numel_remaining -= get_numel_loaded(weight_loader, args)
            if load_numel_remaining <= 0:
                # materialize layer onto device
                materialize_layer(layer)

                # load weights
                for args in loaded_weight_kwargs:
                    param = getattr(layer, name)
                    args.arguments["param"] = param
                    weight_loader(*args.args, **args.kwargs)

                # TODO: process weights
                print("PROCESS WEIGHTS")
                #process_weights_after_loading(layer)

                # copy newly processed values into
                # original tensor data (and cudagraph)
                assert get_module_tensors(layer).keys() == original_parameters.keys()
                for name in original_parameters.keys():
                    original_parameters[name].data.copy_(getattr(layer, name))
                    setattr(layer, name, original_parameters[name])

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
    return sum(tensor.numel() for tensor in get_module_tensors(layer).values())


from torch.utils._python_dispatch import TorchDispatchMode

class CopyNumelCounter(TorchDispatchMode):
    """
    Tracks total number of elements modified with `copy_`. Useful for keeping
    track of weight loading where underlying weights can be arbitrarily
    transformed (such as with `narrow`) before calling copy.

    Assumes that copy kwargs are not used
    """

    def __init__(self, param: torch.Tensor):
        super().__init__()
        self.copied_numel = 0
        self.param = param

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.ops.aten.copy_.default:
            if args[0].device.type == "meta":
                assert args[0].numel() == args[1].numel()
                self.copied_numel += args[0].numel()
        out = func(*args, **kwargs)
        return out


import inspect
def get_numel_loaded(weight_loader: callable, args: inspect.BoundArguments):
    args.arguments["param"] = to_meta_tensor(args.arguments["param"])

    # TODO: use context exit
    context = CopyNumelCounter(args.arguments["param"])
    with context:
        breakpoint()
        weight_loader(*args.args, **args.kwargs)

    print(f"found {context.copied_numel} loaded")
    return context.copied_numel


def materialize_layer(layer: torch.nn.Module):
    for name, tensor in get_module_tensors(layer).items():
        setattr(layer, name, materialize_meta_tensor(tensor))


def materialize_meta_tensor(meta_tensor: torch.Tensor) -> torch.Tensor:
    # TODO: need a way of (reconstructing?) vLLMBaseParameters on the meta device

    # should be called within a torch device context
    tensor = meta_tensor.__class__(data=torch.empty_strided(
        size=tuple(meta_tensor.size()),
        stride=tuple(meta_tensor.stride()),
        dtype=meta_tensor.dtype,
        requires_grad=False,
    ))

    for attr_name in RESTORE_ATTRS:
        if hasattr(meta_tensor, attr_name):
            setattr(tensor, attr_name, getattr(meta_tensor, attr_name))

    return tensor


def to_meta_tensor(tensor: torch.Tensor) -> torch.Tensor:
    meta_tensor = tensor.__class__(data=tensor.data.to("meta"))

    for attr_name in RESTORE_ATTRS:
        if hasattr(tensor, attr_name):
            setattr(meta_tensor, attr_name, getattr(tensor, attr_name))

    return meta_tensor


def get_module_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in chain(module._parameters.items(), module._buffers.items())
        if value is not None
    }
