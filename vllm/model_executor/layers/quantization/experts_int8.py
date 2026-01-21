# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

import torch

from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs


class ExpertsInt8Config(QuantizationConfig):
    """Config class for Int8 experts quantization."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "experts_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExpertsInt8Config":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        elif isinstance(layer, FusedMoE):
            return ExpertsInt8MoEMethod(self, layer.moe_config)
        return None


class ExpertsInt8MoEMethod(FusedMoEMethodBase):
    def __init__(
        self,
        quant_config: ExpertsInt8Config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        int8_dtype = torch.int8

        assert "weight_loader" in extra_weight_attrs
        weight_loader = extra_weight_attrs["weight_loader"]
        wrapped_weight_loader = ExpertsInt8MoEMethod.quantizing_weight_loader(
            layer, weight_loader
        )
        extra_weight_attrs["weight_loader"] = wrapped_weight_loader

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=int8_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=int8_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scale", w13_scale)

        w2_scale = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_scale", w2_scale)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return int8_w8a16_moe_quant_config(
            w1_scale=layer.w13_scale, w2_scale=layer.w2_scale, w1_zp=None, w2_zp=None
        )

    def apply(
        self,
        layer: FusedMoE,
        router: FusedMoERouter,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = router.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )

    @staticmethod
    def quantizing_weight_loader(layer, weight_loader):
        def quantize_and_call_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: int,
            expert_id: int,
        ):
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = layer.intermediate_size_per_partition
            shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
            device = get_tp_group().device
            loaded_weight = loaded_weight.to(device)
            # w1, gate_proj case: Load into first shard of w13.
            if shard_id == "w1":
                scales = quantize_in_place_and_get_scales(loaded_weight[shard, :])
                layer.w13_scale.data[expert_id, 0:shard_size].copy_(scales[:, 0])
            # w3, up_proj case: Load into second shard of w13.
            elif shard_id == "w3":
                scales = quantize_in_place_and_get_scales(loaded_weight[shard, :])
                layer.w13_scale.data[expert_id, shard_size : 2 * shard_size].copy_(
                    scales[:, 0]
                )
            # w2, down_proj case: Load into only shard of w2.
            elif shard_id == "w2":
                scales = quantize_in_place_and_get_scales(loaded_weight[:, shard])
                layer.w2_scale.data[expert_id, :].copy_(scales[:, 0])
            else:
                raise ValueError(f"Shard id must be in [0,1,2] but got {shard_id}")
            weight_loader(param, loaded_weight, weight_name, shard_id, expert_id)

        return quantize_and_call_weight_loader


def quantize_in_place_and_get_scales(weight: torch.Tensor) -> torch.Tensor:
    vmax = torch.iinfo(torch.int8).max
    scales = torch.max(torch.abs(weight), dim=1, keepdim=True)[0] / vmax

    weight.div_(scales)
    weight.round_()
    weight.clamp_(-vmax, vmax)

    return scales
