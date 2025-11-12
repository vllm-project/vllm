# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from copy import deepcopy

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

ONLINE_RELOAD_QUANT_CONFIGS = {
    "torchao",
    "fp8",
}

# Notes for Online Quantization
# In terms of state of checkpoints, quantization config and their
# correspondance to online quantization:
# | Use Case      | Checkpoints          |  model_config.quantization |
# | no quant      | high precision       |  None   |
# | offline quant | quantized |  fp8, torchao etc. |
# | online quant  | high precision | torchao etc. |
#
# The process for loading non-quantized checkpoint
# 1. load non-quantized weights (load_weights)
# 2. do any additional post processing (process_weights_after_loading)
#
# The process for loading offline quantized checkpoint
# 1. load offline-quantized weights (load_weights)
# 2. do any additional post processing (process_weights_after_loading)

# The process for unquantized model reloading
# (repeated run in RL training loop)
# first run
#   UI1. load_weights: load bfloat16 weights
#   UI2. process_weights_after_loading: any additional post processing
# subsequent run
#   UC1: load_weights: load bfloat16 weights
#      (shouldn't be any issues since we didn't change any attributes
#       of the weights)
#   UC2: process_weights_after_loading: any additional post processing

# The process for weight reloading with online quantization
# (repeated run in RL training loop)
# first run
#  I1. load_weights: load bfloat16 weights
#  I2. process_weights_after_loading:
#        record weight metadata and attributes for R1 and R2
#        quantize weights to fp8
# subsequent run
#  (beginning model weight is in fp8)
#  load_weights:
#    R1. restore bfloat16 model weight metadata
#    R2. restore the model weight attributes
#    R3. reload bfloat16 weights
#    R4. quantize weights (by calling process_weights_after_loading),
#    also set `process_weights_after_loading_already_called` to
#    True to stop it from running again
#  process_weights_after_loading (if called):
#    this will be skipped since it's already ran in
#    load_weights


def maybe_save_metadata_and_attributes_for_weight_reloading(
    model: nn.Module, model_config: ModelConfig
):
    # this function should be called at the start of `process_weights_after_loading`
    from vllm.model_executor.model_loader.weight_utils import get_quant_config

    quant_config = get_quant_config(model_config, None)
    if quant_config.get_name() not in ONLINE_RELOAD_QUANT_CONFIGS:
        return

    if not hasattr(model, "weight_loading_metadata"):
        model.weight_loading_metadata = {
            name: _copy_to_meta_tensor(param)
            for name, param in model.named_parameters()
        }

    return model.weight_loading_metadata


def restore_weights_for_loading(model: nn.Module):
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

        param = _materialize_meta_tensor(meta_tensor)
        setattr(module, param_name, param)


def _copy_to_meta_tensor(tensor: torch.Tensor) -> torch.Tensor:
    new_tensor = tensor.to("meta")
    new_tensor.__class__ = tensor.__class__
    new_tensor.__dict__ = deepcopy(tensor.__dict__)
    new_tensor._original_device = tensor.device
    return new_tensor


def _tensors_alike(tensor: torch.Tensor | None, meta: torch.Tensor) -> bool:
    if tensor is None:
        return False

    return (
        tensor.device == meta._original_device
        and tensor.dtype == meta.dtype
        and tensor.shape == meta.shape
        and tensor.__dict__ == meta.__dict__
    )


def _materialize_meta_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_strided(
        size=tuple(tensor.size()),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor._original_device,
        requires_grad=False,  # set below to match input
    )
