# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase


class OnlineRequantizationMixin:
    """Shared lifecycle for converting serialized weights before quantization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requantization_source: QuantizeMethodBase | None = None
        self.requantization_source_parameters: dict[str, torch.nn.Parameter] = {}

    def set_requantization_source(self, source_method: QuantizeMethodBase) -> None:
        """Configure serialized-weight conversion before online quantization."""
        self.requantization_source = source_method
        self.uses_meta_device = False

    def create_requantization_source_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ) -> bool:
        """Delegate weight creation to the source and track its parameters."""
        if self.requantization_source is None:
            return False

        existing_parameter_names = set(layer._parameters)
        self.requantization_source.create_weights(
            layer, *weight_args, **extra_weight_attrs
        )
        self.requantization_source_parameters = {
            name: parameter
            for name, parameter in layer._parameters.items()
            if name not in existing_parameter_names and parameter is not None
        }
        return True

    def release_requantization_source_weights(self, layer: torch.nn.Module) -> None:
        """Release serialized parameters after successful requantization."""
        for name, source_parameter in self.requantization_source_parameters.items():
            if layer._parameters.get(name) is source_parameter:
                delattr(layer, name)
        self.requantization_source_parameters.clear()


class OnlineLinearRequantizationMixin(OnlineRequantizationMixin):
    """Requantization lifecycle specialized for a linear weight."""

    def get_weight_for_quantization(self, layer: torch.nn.Module) -> torch.Tensor:
        if self.requantization_source is None:
            return layer.weight
        weight = self.requantization_source.dequantize_weight(layer)
        if not isinstance(weight, torch.Tensor):
            raise TypeError("Linear requantization requires a single weight tensor.")
        return weight


class OnlineMoERequantizationMixin(OnlineRequantizationMixin):
    """Requantization lifecycle specialized for MoE w13 and w2 weights."""

    def get_weights_for_quantization(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
            FusedMoEMethodBase,
        )

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

        assert isinstance(self, FusedMoEMethodBase)
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
