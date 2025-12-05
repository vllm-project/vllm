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


def _get_lora_device(base_layer: nn.Module) -> torch.device:
    # code borrowed from https://github.com/fmmoret/vllm/blob/fm-support-lora-on-quantized-models/vllm/lora/layers.py#L34
    """Returns the device for where to place the LoRA tensors."""
    # unquantizedLinear
    if hasattr(base_layer, "weight"):
        return base_layer.weight.device
    # Compressed Tensor
    elif hasattr(base_layer, "weight_packed"):
        return base_layer.weight_packed.device
    # GPTQ/AWQ
    elif hasattr(base_layer, "qweight"):
        return base_layer.qweight.device
    # HQQ marlin
    elif hasattr(base_layer, "W_q"):
        return base_layer.W_q.device
    # MoE layer
    elif hasattr(base_layer, "w2_weight"):
        return base_layer.w2_weight.device
    # MoE Compressed Tensor
    elif hasattr(base_layer, "w2_weight_packed"):
        return base_layer.w2_weight_packed.device
    # MoE GPTQ/AWQ/GGUF
    elif hasattr(base_layer, "w2_qweight"):
        return base_layer.w2_qweight.device
    else:
        raise ValueError(f"Unsupported base layer: {base_layer}")


def _not_fully_sharded_can_replace(can_replace):
    """
    decorator which adds the condition of not using fully sharded loras
    intended to wrap can_replace_layer()
    """

    def dec(*args, **kwargs):
        decorate = kwargs.pop("decorate") if "decorate" in kwargs else True
        condition = not kwargs["lora_config"].fully_sharded_loras if decorate else True
        return can_replace(*args, **kwargs) and condition

    return dec


def _fully_sharded_can_replace(can_replace):
    """
    decorator which adds the condition of fully sharded loras
    intended to wrap can_replace_layer()
    """

    def dec(*args, **kwargs):
        return (
            can_replace(*args, **kwargs) and kwargs["lora_config"].fully_sharded_loras
        )

    return dec
