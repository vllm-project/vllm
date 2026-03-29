# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod

import torch

from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.utils import set_weight_attrs


class OnlineMoEMethodBase(FusedMoEMethodBase):
    """Base for MoE methods that load full-precision weights and quantize
    them during model loading via the QeRL layerwise processing system."""

    uses_meta_device: bool = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts

        # Fused gate_up_proj (column parallel) — full precision on meta device
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
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

        # Hook for subclasses to add extra params (biases, etc.)
        # before initialize_online_processing counts total elements.
        self._create_extra_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

        initialize_online_processing(layer)

    def _create_extra_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Override to create additional parameters before online processing
        initialization. Called after w13/w2 weights are registered but before
        ``initialize_online_processing``."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        self._quantize_weights(layer)
        layer._already_called_process_weights_after_loading = True

    @abstractmethod
    def _quantize_weights(self, layer: torch.nn.Module) -> None:
        """Quantize full-precision weights after all experts are loaded."""
        ...
