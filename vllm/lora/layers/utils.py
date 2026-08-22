# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from vllm import envs
from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2

_lora_aux_cuda_stream: torch.cuda.Stream | None = None


def _get_lora_aux_cuda_stream() -> torch.cuda.Stream | None:
    if not envs.VLLM_LORA_ENABLE_DUAL_STREAM:
        return None
    global _lora_aux_cuda_stream
    if _lora_aux_cuda_stream is None and current_platform.is_cuda_alike():
        _lora_aux_cuda_stream = torch.cuda.Stream()
    return _lora_aux_cuda_stream


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

    # Per-request LoRA strength. `index_mapping` / `prompt_mapping` say which
    # adapter each token uses; these two say how strongly it is applied, so
    # requests sharing one adapter can ask for different scales without
    # consuming extra adapter slots.
    #   request_scales[i]  -> scale for the i-th request in the batch
    #   token_to_req[i]    -> request index of the i-th token of index_mapping
    #   sample_to_req[i]   -> request index of the i-th entry of prompt_mapping
    # All three are None when the feature is disabled, which is the default
    # and reproduces stock behavior.
    request_scales: tuple[float, ...] | None = None
    token_to_req: tuple[int, ...] | None = None
    sample_to_req: tuple[int, ...] | None = None

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)
        if self.request_scales is not None:
            self.request_scales = tuple(self.request_scales)
            assert self.token_to_req is not None, (
                "token_to_req is required alongside request_scales"
            )
            self.token_to_req = tuple(self.token_to_req)
            assert len(self.token_to_req) == len(self.index_mapping)
            if self.sample_to_req is not None:
                self.sample_to_req = tuple(self.sample_to_req)
                assert len(self.sample_to_req) == len(self.prompt_mapping)

    @property
    def has_request_scales(self) -> bool:
        """True when any request asks for a scale other than the adapter's own."""
        return self.request_scales is not None


def _get_lora_device(base_layer: nn.Module) -> torch.device:
    # code borrowed from https://github.com/fmmoret/vllm/blob/fm-support-lora-on-quantized-models/vllm/lora/layers.py#L34
    """Returns the device for where to place the LoRA tensors."""
    if hasattr(base_layer, "routed_experts"):
        base_layer = base_layer.routed_experts

    # unquantizedLinear
    if hasattr(base_layer, "weight"):
        return base_layer.weight.device
    # Compressed Tensor
    elif hasattr(base_layer, "weight_packed"):
        return base_layer.weight_packed.device
    # GPTQ/AWQ
    elif hasattr(base_layer, "qweight"):
        return base_layer.qweight.device
    # INC WNA16 (AutoRound)
    elif hasattr(base_layer, "ark_linear"):
        return base_layer.ark_linear.qweight.device
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
) -> dict[str, int | None]:
    # LoRA shrink/expand operates on bf16/fp16 adapters regardless of the
    # base MoE weight's block-wise quantization, so block_shape is omitted
    # from the config lookup — the non-quantized branch in get_default_config
    # ignores it anyway.
    raw_config = try_get_optimal_moe_config(w1_shape, w2_shape, top_k, dtype, M)
    config: dict[str, int | None] = dict(raw_config)
    if op_type in [
        "fused_moe_lora_w13_shrink",
        "fused_moe_lora_w2_shrink",
    ]:
        block_size_n = config.get("BLOCK_SIZE_N")
        config["BLOCK_SIZE_N"] = min(
            block_size_n if block_size_n is not None else 64,
            next_power_of_2(rank),
        )
    elif op_type in [
        "fused_moe_lora_w13_expand",
        "fused_moe_lora_w2_expand",
    ]:
        block_size_k = config.get("BLOCK_SIZE_K")
        config["BLOCK_SIZE_K"] = max(
            16,
            min(
                block_size_k if block_size_k is not None else 32,
                next_power_of_2(rank),
            ),
        )
    return config
