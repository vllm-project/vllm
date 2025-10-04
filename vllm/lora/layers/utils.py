# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LoRAMapping:
    index_mapping: tuple[int, ...]
    prompt_mapping: tuple[int, ...]
    is_prefill: bool = False

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)


def _get_layer_weight(layer: torch.nn.Module) -> torch.Tensor:
    # unquantizedLinear
    if hasattr(layer, "weight"):
        return layer.weight
    # Compressed Tensor
    elif hasattr(layer, "weight_packed"):
        return layer.weight_packed
    # GPTQ/AWQ
    elif hasattr(layer, "qweight"):
        return layer.qweight
    # marlin
    elif hasattr(layer, "B"):
        return layer.B
    # HQQ marlin
    elif hasattr(layer, "W_q"):
        return layer.W_q
    else:
        raise ValueError(f"Unsupported base layer: {layer}")


def _get_layer_device(base_layer: nn.Module) -> torch.device:
    weight = _get_layer_weight(base_layer)
    return weight.device


def _get_layer_dtype(base_layer: nn.Module) -> torch.dtype:
    weight = _get_layer_weight(base_layer)
    return weight.dtype


def _not_fully_sharded_can_replace(can_replace):
    """
    decorator which adds the condition of not using fully sharded loras
    intended to wrap can_replace_layer()
    """

    def dec(*args, **kwargs):
        decorate = kwargs.pop("decorate") if "decorate" in kwargs else True
        condition = (not kwargs["lora_config"].fully_sharded_loras
                     if decorate else True)
        return can_replace(*args, **kwargs) and condition

    return dec


def _fully_sharded_can_replace(can_replace):
    """
    decorator which adds the condition of fully sharded loras
    intended to wrap can_replace_layer()
    """

    def dec(*args, **kwargs):
        return (can_replace(*args, **kwargs)
                and kwargs["lora_config"].fully_sharded_loras)

    return dec
