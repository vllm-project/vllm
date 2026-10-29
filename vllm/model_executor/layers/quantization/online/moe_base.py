# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod

import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import triton_weight
from vllm.model_executor.layers.quantization.online.requantization import (
    OnlineRequantizationMixin,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.utils import set_weight_attrs


class OnlineMoEMethodBase(OnlineRequantizationMixin, FusedMoEMethodBase):
    """Base for MoE methods that load full-precision weights on meta device
    and quantize them after loading via the QeRL layerwise processing system.
    """

    uses_meta_device: bool = True

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Target conversion needs this metadata even when the source creates weights.
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.requantization_source is not None:
            self.create_requantization_source_weights(
                layer,
                num_experts,
                layer.moe_config.hidden_dim_unpadded,
                layer.moe_config.intermediate_size_per_partition_unpadded,
                params_dtype,
                **extra_weight_attrs,
            )
            return

        # Fused gate_up_proj (column parallel) — full precision on meta device
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                device="meta",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel) — full precision on meta device
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # BIASES (for models like GPT-OSS that have biased MoE)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    device="meta",
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    device="meta",
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

        initialize_online_processing(layer)

    def get_weights_for_quantization(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.requantization_source is None:
            return layer.w13_weight, layer.w2_weight

        weights = self.requantization_source.dequantize_weight(layer)
        if not isinstance(weights, tuple):
            raise TypeError("MoE requantization requires w13 and w2 weight tensors.")

        w13, w2 = weights

        config = layer.moe_config
        hidden_size = config.hidden_dim_unpadded
        intermediate_size = config.intermediate_size_per_partition_unpadded
        target_hidden_size = config.hidden_dim
        target_intermediate_size = config.intermediate_size_per_partition

        num_experts = layer.local_num_experts
        num_shards = self.moe.w13_num_shards

        if w13.shape != (num_experts, num_shards * intermediate_size, hidden_size):
            raise ValueError("Requantization requires unpadded source w13 weights.")
        if w2.shape != (num_experts, hidden_size, intermediate_size):
            raise ValueError("Requantization requires unpadded source w2 weights.")

        if (
            target_hidden_size < hidden_size
            or target_intermediate_size < intermediate_size
        ):
            raise ValueError("Requantization target dimensions cannot shrink weights.")

        # No padding required.
        if (hidden_size, intermediate_size) == (
            target_hidden_size,
            target_intermediate_size,
        ):
            return w13, w2

        if any(
            getattr(layer, name, None) is not None for name in ("w13_bias", "w2_bias")
        ):
            raise NotImplementedError(
                "Padded requantization with biases is unsupported."
            )

        def pad_moe_weights(
            w13: torch.Tensor, w2: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            padded_w13 = w13.new_zeros(
                (num_experts, num_shards * target_intermediate_size, target_hidden_size)
            )
            padded_w2 = w2.new_zeros(
                (num_experts, target_hidden_size, target_intermediate_size)
            )
            for shard in range(num_shards):
                source_start = shard * intermediate_size
                target_start = shard * target_intermediate_size
                padded_w13[
                    :, target_start : target_start + intermediate_size, :hidden_size
                ] = w13[:, source_start : source_start + intermediate_size, :]
            padded_w2[:, :hidden_size, :intermediate_size] = w2
            return padded_w13, padded_w2

        return pad_moe_weights(w13, w2)

    def _zero_padding(
        self,
        layer: torch.nn.Module,
        w13_weight: torch.Tensor | None = None,
        w2_weight: torch.Tensor | None = None,
    ) -> None:
        w13_weight = layer.w13_weight if w13_weight is None else w13_weight
        w2_weight = layer.w2_weight if w2_weight is None else w2_weight
        hidden_size = layer.moe_config.hidden_dim_unpadded
        intermediate_size = layer.moe_config.intermediate_size_per_partition_unpadded

        w13_shard = w13_weight.shape[1] // self.moe.w13_num_shards
        if w13_shard > intermediate_size:
            for shard in range(self.moe.w13_num_shards):
                start = shard * w13_shard + intermediate_size
                w13_weight[:, start : (shard + 1) * w13_shard, :] = 0
        if w13_weight.shape[2] > hidden_size:
            w13_weight[:, :, hidden_size:] = 0

        if w2_weight.shape[1] > hidden_size:
            w2_weight[:, hidden_size:, :] = 0
        if w2_weight.shape[2] > intermediate_size:
            w2_weight[:, :, intermediate_size:] = 0

        if getattr(layer, "w13_bias", None) is not None:
            w13_bias_shard = layer.w13_bias.shape[1] // self.moe.w13_num_shards
            if w13_bias_shard > intermediate_size:
                for shard in range(self.moe.w13_num_shards):
                    start = shard * w13_bias_shard + intermediate_size
                    layer.w13_bias[:, start : (shard + 1) * w13_bias_shard] = 0

        if (
            getattr(layer, "w2_bias", None) is not None
            and layer.w2_bias.shape[1] > hidden_size
        ):
            layer.w2_bias[:, hidden_size:] = 0

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    @property
    def supports_eplb(self) -> bool:
        return True

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            triton_weight(layer, "w13_weight"),
            triton_weight(layer, "w2_weight"),
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
            routing_sink=layer.routing_sink,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            triton_weight(layer, "w13_weight"),
            triton_weight(layer, "w2_weight"),
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
