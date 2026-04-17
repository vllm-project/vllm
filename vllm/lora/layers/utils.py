# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
from vllm.utils.math_utils import next_power_of_2

logger = init_logger(__name__)

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

    # In case some module wrap the Tensor in ParameterList
    def get_dev(obj):
        dev = None
        if obj is not None:
            if hasattr(obj, "device"):
                dev = obj.device
                logger.debug(f"get_dev type of obj = {type(obj)} dev = {dev}")
            elif isinstance(obj, (nn.ParameterList, list, tuple)) and len(obj) > 0:
                if hasattr(obj[0], "device"):
                    dev = obj[0].device
                    logger.debug(f"get_dev type of obj[0] = {type(obj[0])} dev = {dev}")
        logger.debug(f"get_dev final return dev = {dev}")
        return dev

    attr_names = ["weight", # unquantizedLinear
                  "weight_packed", # Compressed Tensor
                  "qweight", # GPTQ/AWQ
                  "w2_weight", # MoE layer
                  "w2_weight_packed", # MoE Compressed Tensor
                  "w2_qweight", # MoE GPTQ/AWQ/GGUF
                 ]
    for attr in attr_names:
        logger.debug(f"lora base_layer = {base_layer} attr_name = {attr}")
        target = getattr(base_layer, attr, None)
        dev = get_dev(target)
        if dev is not None:
            return dev

    try:
        return next(base_layer.parameters()).device
    except StopIteration:
        logger.debug("lora base_layer = {base_layer} in except StopInteration return cpu")
        return torch.device("cpu")


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
) -> dict[str, int | None]:
    # LoRA shrink/expand operates on bf16/fp16 adapters regardless of the
    # base MoE weight's block-wise quantization, so block_shape is omitted
    # from the config lookup — the non-quantized branch in get_default_config
    # ignores it anyway.
    config = try_get_optimal_moe_config(w1_shape, w2_shape, top_k, dtype, M).copy()
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
