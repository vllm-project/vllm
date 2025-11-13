# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from copy import deepcopy
from types import MethodType

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

ONLINE_RELOAD_QUANT_CONFIGS = {
    "torchao",
    "fp8",
}

"""

First time loading lifecycle
1. Model checkpoint is loaded by `ModelLoader.get_all_weights` into `weights_iterator`
2. `weights_iterator` is loaded into model by `model.load_weights`
3. Model state is captured by `record_weights_for_reloading`
4. `process_weights_after_loading` converts model state into kernel format
5. Model can run now that weights are in kernel format


Subsequent reloading lifecycle
1. Model weights updates are packed into an async/chunked `weights_iterator`
or model checkpoint is loaded from disk into `weights_iterator`
2. Model state is restored to by `restore_weights_for_reloading`
3. 
"""


def record_weights_for_reloading(model: nn.Module, model_config: ModelConfig):
    # this function should be called before `process_weights_after_loading`
    # in practice, this happens at the very start of `process_weights_after_loading`
    if model_config.quantization not in ONLINE_RELOAD_QUANT_CONFIGS:
        return

    if not hasattr(model, "weight_loading_metadata"):
        model.weight_loading_metadata = {
            name: _copy_to_meta_tensor(param)
            for name, param in model.named_parameters(remove_duplicate=False)
        }


def restore_weights_for_reloading(model: nn.Module):
    assert hasattr(model, "weight_loading_metadata")
    metadata: dict[str, torch.Tensor] = model.weight_loading_metadata
    model_param_names = dict(model.named_parameters(remove_duplicate=False)).keys()

    # remove parameters which were not present at load time
    params_to_remove = model_param_names - metadata.keys()
    for param_fqn in params_to_remove:
        module_name, param_name = param_fqn.rsplit(".", 1)
        module = model.get_submodule(module_name)
        delattr(module, param_name)

    # restore parameters that were present at load time
    for param_fqn, meta_tensor in metadata.items():
        module_name, param_name = param_fqn.rsplit(".", 1)
        module = model.get_submodule(module_name)

        # for faster runtime, skip materialization if the tensors match
        original_tensor = getattr(module, param_name, None)
        if _tensors_alike(original_tensor, meta_tensor):
            continue

        delattr(module, param_name)  # delete before materialization to avoid oom
        param = _materialize_meta_tensor(meta_tensor)
        setattr(module, param_name, param)


def _copy_to_meta_tensor(tensor: torch.Tensor) -> torch.Tensor:
    meta_tensor = tensor.to("meta")
    meta_tensor.__class__ = tensor.__class__
    meta_tensor.__dict__ = deepcopy(tensor.__dict__)
    meta_tensor._original_device = tensor.device

    return meta_tensor


def _tensors_alike(tensor: torch.Tensor | None, meta_tensor: torch.Tensor) -> bool:
    if tensor is None:
        return False

    return (
        tensor.device == meta_tensor._original_device
        and tensor.dtype == meta_tensor.dtype
        and tensor.shape == meta_tensor.shape
        and tensor.__dict__ == meta_tensor.__dict__
    )


def _materialize_meta_tensor(meta_tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.empty_strided(
        size=tuple(meta_tensor.size()),
        stride=tuple(meta_tensor.stride()),
        dtype=meta_tensor.dtype,
        device=meta_tensor._original_device,
        requires_grad=meta_tensor.requires_grad,
    )
    tensor.__class__ = meta_tensor.__class__
    tensor.__dict__ = deepcopy(meta_tensor.__dict__)

    # rebind any references to the original tensor
    # assume that methods are bound to the original tensor
    for key, value in tensor.__dict__.items():
        if isinstance(value, MethodType):
            tensor[key] = MethodType(value.__func__, tensor)

    return tensor
