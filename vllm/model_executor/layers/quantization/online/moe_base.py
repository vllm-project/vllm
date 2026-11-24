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
from vllm.model_executor.layers.quantization.online.moe_streaming import (
    ExpertWeightStream,
)
from vllm.model_executor.layers.quantization.online.requantization import (
    OnlineRequantizationMixin,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    get_layerwise_info,
    get_layerwise_info,
    initialize_online_processing,
)
from vllm.model_executor.utils import set_weight_attrs


def pad_expert_statistic(statistic: torch.Tensor, size: int) -> torch.Tensor:
    """Zero-pad the first dim of an unsharded expert statistic, e.g. a per
    channel amax over the hidden size, to the layer's padded `size`."""
    padding = size - statistic.shape[0]
    if padding == 0:
        return statistic
    return torch.cat([statistic, statistic.new_zeros(padding, *statistic.shape[1:])])


class OnlineMoEMethodBase(OnlineRequantizationMixin, FusedMoEMethodBase):
    """Base for MoE methods that quantize their experts online.

    The experts are created on the meta device, and quantized one at a time as
    they load by an `ExpertWeightStream`: subclasses implement
    `quantize_expert`, `expert_statistics` if their scales span a TP-sharded
    dimension, and `setup_quantized_weights` for the layer-wide conversion to
    the kernel format.
    """

    uses_meta_device: bool = True

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        # Checkpoint parameters of one routed expert, which the stream decodes
        # into full-precision weights, and checkpoint activation scales, unused.
        self.expert_weight_names: tuple[str, ...] = ("w13_weight", "w2_weight")
        self.expert_activation_scale_names: tuple[str, ...] = ()

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

        if self.requantization_source is not None:
            # In case FSE is used, shared experts slots are initialized on RoutedExperts
            # online quant method, the source only holds the routed experts.
            self.create_requantization_source_weights(
                layer,
                num_experts - layer.expert_map_manager.num_fused_shared_experts,
                layer.moe_config.hidden_dim_unpadded,
                layer.moe_config.intermediate_size_per_partition_unpadded,
                params_dtype,
                **extra_weight_attrs,
            )
            layer.num_experts = num_experts
            source_names = tuple(self.requantization_source_parameters)
            self.expert_weight_names = tuple(
                name
                for name in source_names
                if "input" not in name and "bias" not in name
            )
            self.expert_activation_scale_names = tuple(
                name for name in source_names if "input" in name
            )
            # The source describes its serialized weights (e.g. a block size) on the
            # layer: the target owns the layer's quantization attributes.
            layer.weight_block_size = None
            return

        layer.weight_block_size = None

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

    def create_weight_stream(self, device: torch.device) -> ExpertWeightStream | None:
        """Stream loading the layer's experts, for `initialize_online_processing`."""
        return ExpertWeightStream(self, device)

    def expert_statistics(
        self, w13: torch.Tensor, w2: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Statistics that the scales of one expert depend on beyond its TP shard.

        Args:
            w13: Unsharded `[num_shards * intermediate_size, hidden_size]` weight.
            w2: Unsharded `[hidden_size, intermediate_size]` weight.

        Returns:
            Statistics passed to `quantize_expert`. As they are computed on the
            unsharded expert, the scales do not depend on the TP configuration.

        """
        return {}

    @abstractmethod
    def quantize_expert(
        self,
        layer: torch.nn.Module,
        w13: torch.Tensor,
        w2: torch.Tensor,
        statistics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Quantize one expert.

        Args:
            layer: Layer of the expert.
            w13: This rank's `[num_shards * intermediate_size_per_partition,
                hidden_size]` weight, zero-padded to the layer's sizes.
            w2: This rank's `[hidden_size, intermediate_size_per_partition]`
                weight, zero-padded to the layer's sizes.
            statistics: The `expert_statistics` of the unsharded expert.

        Returns:
            The quantized tensors of the expert, e.g. `(w13, w2, w13_scale,
            w2_scale)`, stacked over the experts for `setup_quantized_weights`.

        """

    @abstractmethod
    def setup_quantized_weights(
        self, layer: torch.nn.Module, *weights: torch.Tensor
    ) -> None:
        """Convert the stacked `quantize_expert` outputs of all local experts to
        the kernel format, and set up the kernel."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        info = get_layerwise_info(layer)
        stream = info.weight_stream
        if not isinstance(stream, ExpertWeightStream) or stream.method is not self:
            # The weights were loaded outside of layerwise processing, into the
            # materialized layer: quantize them one expert at a time all the same.
            stream = ExpertWeightStream(self, info.restore_device)
        self.setup_quantized_weights(layer, *stream.quantized_weights(layer))
        self.release_requantization_source_weights(layer)
        if (loader := getattr(layer, "shared_expert_online_loader", None)) is not None:
            loader.release_staging(layer)

        # Prevent duplicate processing (e.g., during weight reload).
        layer._already_called_process_weights_after_loading = True

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
