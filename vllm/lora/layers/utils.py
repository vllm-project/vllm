# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
from vllm.utils.math_utils import next_power_of_2


class LoRAMappingType(Enum):
    LANGUAGE = 1
    TOWER = 2
    CONNECTOR = 3


@dataclass
class LoRAMapping:
    index_mapping: tuple[int, ...]
    prompt_mapping: tuple[int, ...]
    is_prefill: bool = False
    type: LoRAMappingType = LoRAMappingType.LANGUAGE

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


def try_get_optimal_moe_lora_config(
    op_type: str,
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    rank: int,
    top_k: int,
    dtype: str | None,
    M: int,
    block_shape: list[int] | None = None,
) -> dict[str, int | None]:
    config = try_get_optimal_moe_config(
        w1_shape, w2_shape, top_k, dtype, M, block_shape
    ).copy()
    if op_type in [
        "fused_moe_lora_w13_shrink",
        "fused_moe_lora_w2_shrink",
    ]:
        config["BLOCK_SIZE_N"] = min(
            config.get("BLOCK_SIZE_N", 64), next_power_of_2(rank)
        )
    elif op_type in [
        "fused_moe_lora_w13_expand",
        "fused_moe_lora_w2_expand",
    ]:
        config["BLOCK_SIZE_K"] = max(
            16, min(config.get("BLOCK_SIZE_K", 32), next_power_of_2(rank))
        )
    return config
