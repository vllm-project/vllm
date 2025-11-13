# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utilities for enabling weight reloading and online quantization
For more information and diagrams, see https://github.com/neuralmagic/vllm/pull/128

## Model Reloading Lifecycle ##
1. Model is loadeded for the first time
    a. Checkpoint is loaded by `ModelLoader.get_all_weights` into `weights_iterator`
    b. `weights_iterator` is loaded into model by `model.load_weights`
    c.  Model state is captured by `record_weights_for_reloading`
    d. `process_weights_after_loading` converts model state into kernel format.
       The model is no longer loadable while its weights are in kernel format

2. Model is reloaded via `reload_weights`
    a. A `weights_iterator` is provided, which may be async/ chunked/ sharded
    b. The original model state is restored by `restore_weights_for_reloading`
       using metadata information from `record_weights_for_reloading`
    c. `weights_iterator` is loaded into model by `model.load_weights`
    d. `process_weights_after_loading` converts model state into kernel format.
       The model is no longer loadable while its weights are in kernel format

Alternatively, if a user does not want to use `reload_weights`, they can call
steps 2b and 2d manually:

```python
record_weights_for_reloading(model)

for weights in weights_iterator:  # may be async/ chunked/ sharded
    model.load_weights(weights)

process_weights_after_loading(model, model_config, device)
```
"""

from types import MethodType

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "RELOADABLE_QUANT_CONFIGS",
    "record_weights_for_reloading",
    "restore_weights_for_reloading",
]

# in theory, this implementation of weight recording/restoring
# should support any quantization config
RELOADABLE_QUANT_CONFIGS = {
    None,
    "torchao",
    "fp8",
}


def record_weights_for_reloading(model: nn.Module):
    # this function should be called before `process_weights_after_loading`
    # in practice, this happens at the very start of `process_weights_after_loading`
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
    meta_tensor.__dict__ = tensor.__dict__
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
    tensor.__dict__ = meta_tensor.__dict__

    # rebind any references to the original tensor
    # assume that methods are bound to the original tensor
    for key, value in tensor.__dict__.items():
        if isinstance(value, MethodType):
            tensor[key] = MethodType(value.__func__, tensor)

    return tensor
