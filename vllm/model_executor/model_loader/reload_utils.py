# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import wraps
from itertools import chain

import torch

from vllm.logger import init_logger
#from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
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

import inspect

def record_metadata_for_reloading(layer: torch.nn.Module):
    """

    Note that buffers will be restored as parameters

    """
    layer.restore_metadata = (
        {
            name: to_meta_tensor(param)
            for name, param in layer._parameters.items()
            if param is not None
        },
        {
            name: to_meta_tensor(buffer)
            for name, buffer in layer._buffers.items()
            if buffer is not None
        },
    )

from functools import partial

@torch.no_grad()
def layerwise_restore_and_process(layer: torch.nn.Module):
    """
    Called after `record_metadata_for_reloading`.
    """
    if not hasattr(layer, "restore_metadata"):
        raise ValueError("Must call `initialize_model` before a model can be reloaded")
    
    # get parameters for later copying
    layer.kernel_tensors = get_module_tensors(layer)

    # restore layer onto meta device
    # for name in kernel_tensors.keys():
    #     delattr(layer, name)

    restore_params, restore_buffers = layer.restore_metadata
    for name, param in restore_params.items():
        layer.register_parameter(name, param)
    for name, buffer in restore_buffers.items():
        layer.register_buffer(name, buffer)

    # track loading progress to determine when to process/copy
    load_numel_remaining = get_layer_size(layer)
    total_debug = load_numel_remaining

    loaded_weight_kwargs: list[tuple[str, inspect.BoundArguments]] = list()
    # TODO: delete when done?

    for _name, param in get_module_tensors(layer).items():
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader_signature = inspect.signature(weight_loader)

        if isinstance(weight_loader, partial):
            # shared params? I think guarded by hasattr(layer, "kernel_tensors") now
            continue

        def restore_and_process_loader(*args, param_name=None, **kwargs):
            nonlocal load_numel_remaining
            nonlocal loaded_weight_kwargs

            args = weight_loader_signature.bind(*args, **kwargs)
            args.apply_defaults()

            # cache loaded weights on load device until all are ready
            loaded_weight_kwargs.append((param_name, args))

            # process and copy when all weights are loaded
            num_loaded = get_numel_loaded(weight_loader, args)
            load_numel_remaining -= num_loaded
            print(f"Loaded: {param_name}: {num_loaded}/{total_debug}; {load_numel_remaining}")

            if load_numel_remaining <= 0 and hasattr(layer, "kernel_tensors"):
                # materialize layer onto device
                materialize_layer(layer)

                # load weights
                for name, args in loaded_weight_kwargs:
                    param = getattr(layer, name)
                    args.arguments["param"] = param
                    weight_loader(*args.args, **args.kwargs)

                # process weights
                quant_method = getattr(layer, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    quant_method.process_weights_after_loading(layer)
                    print("PROCESS")

                # copy newly processed values into
                # original tensor data (and cudagraph)
                for name in layer.kernel_tensors.keys():
                    assert torch.equal(layer.kernel_tensors[name].data, getattr(layer, name))
                    layer.kernel_tensors[name].data.copy_(getattr(layer, name))
                    setattr(layer, name, layer.kernel_tensors[name])
                del layer.kernel_tensors

        param.weight_loader = wraps(weight_loader)(partial(restore_and_process_loader, param_name=_name))


def unwrap_weight_loaders(layer: torch.nn.Module):
    """
    Removes outer-most layer of weight loading wrappers. This function can be called
    after `layerwise_restore_and_process` to unwrap layerwise weight loaders.
    """
    for param in get_module_tensors(layer).values():
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        if hasattr(weight_loader, "__wrapped__") and isinstance(weight_loader, partial):
            param.weight_loader = weight_loader.__wrapped__

        # TODO: process attention

        # sometimes modules (like Attention) have kernel tensors (_q_scale, ect.)
        # but do not load module tensors (model is not quantized)
        # for these cases, we need finalize the layerwise loading by removing the
        # meta tensors and copying back the kernel tensors
        if hasattr(layer, "kernel_tensors"):
            for name in layer.kernel_tensors.keys():
                setattr(layer, name, layer.kernel_tensors[name])

            del layer.kernel_tensors


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
    # maybe unnecessary, since targeted param is already on meta
    #args.arguments["param"] = to_meta_tensor(args.arguments["param"])

    # TODO: use context exit
    context = CopyNumelCounter(args.arguments["param"])
    with context:
        weight_loader(*args.args, **args.kwargs)

    return context.copied_numel


def materialize_layer(layer: torch.nn.Module):
    for name, tensor in get_module_tensors(layer).items():
        setattr(layer, name, materialize_meta_tensor(tensor))


def materialize_meta_tensor(meta_tensor: torch.Tensor) -> torch.Tensor:
    # TODO: need a way of (reconstructing?) vLLMBaseParameters on the meta device

    # should be called within a torch device context
    tensor = torch.empty_strided(
        size=tuple(meta_tensor.size()),
        stride=tuple(meta_tensor.stride()),
        dtype=meta_tensor.dtype,
        requires_grad=False,
    )
    tensor.__class__ = meta_tensor.__class__
    tensor.__dict__ = meta_tensor.__dict__
    assert tensor.device != torch.device("meta")

    # for attr_name in RESTORE_ATTRS:
    #     if hasattr(meta_tensor, attr_name):
    #         setattr(tensor, attr_name, getattr(meta_tensor, attr_name))

    return tensor


def to_meta_tensor(tensor: torch.Tensor) -> torch.Tensor:
    meta_tensor = tensor.data.to("meta")
    meta_tensor.__class__ = tensor.__class__
    meta_tensor.__dict__ = tensor.__dict__

    return meta_tensor


def get_module_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in chain(module._parameters.items(), module._buffers.items())
        if value is not None
    }
