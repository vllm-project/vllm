# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from typing import Any

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.model_executor.layers.fused_moe import RoutedExperts, SharedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.lut_b_moe import (
    LutBSm107Experts,
    make_lut_b_moe_kernel,
    select_lut_b_moe_experts,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    LUT_B_BLOCK_K,
    LUT_B_BLOCK_N,
    LUT_B_CODEBOOK_SIZE,
    LUT_B_PACKED_TILE_BYTES,
    dequantize_lut_b,
    lut_b_to_sm107_tiles,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs


class CompressedTensorsLutBMoEMethod(CompressedTensorsMoEMethod):
    """Load compressed-tensors LUT-B expert weights and use LUT-B MoE kernels."""

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        if (
            weight_quant.type != QuantizationType.CODEBOOK
            or weight_quant.num_bits != 3
            or weight_quant.strategy != QuantizationStrategy.BLOCK
            or weight_quant.block_structure != [LUT_B_BLOCK_N, LUT_B_BLOCK_K]
            or not weight_quant.symmetric
            or weight_quant.dynamic
            or weight_quant.scale_dtype not in (None, torch.uint8)
        ):
            raise ValueError(
                "LUT-B MoE requires 3-bit symmetric codebook weights with "
                f"block_structure={[LUT_B_BLOCK_N, LUT_B_BLOCK_K]} and "
                "E8M0 scales."
            )
        if input_quant is not None:
            raise ValueError("LUT-B MoE checkpoints do not quantize activations.")
        if moe.has_bias:
            raise ValueError("LUT-B MoE does not support expert biases.")

        self.experts_cls = select_lut_b_moe_experts(moe)

    @property
    def supports_eplb(self) -> bool:
        return True

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if hidden_size % LUT_B_BLOCK_K or intermediate_size_per_partition % (
            LUT_B_BLOCK_N
        ):
            raise ValueError(
                "LUT-B MoE requires hidden size divisible by 64 and local "
                "intermediate size divisible by 8."
            )
        if intermediate_size_per_partition % LUT_B_BLOCK_K:
            raise ValueError(
                "LUT-B down-projection weights require local intermediate "
                "size divisible by 64."
            )

        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        weight_attrs = {
            **extra_weight_attrs,
            "weight_loader": partial(self._weight_loader, layer=layer),
        }
        w13_shape = (
            num_experts,
            self.moe.w13_num_shards * intermediate_size_per_partition // LUT_B_BLOCK_N,
            hidden_size // LUT_B_BLOCK_K,
        )
        w2_shape = (
            num_experts,
            hidden_size // LUT_B_BLOCK_N,
            intermediate_size_per_partition // LUT_B_BLOCK_K,
        )
        self._register_lut_b_weights(layer, "w13", w13_shape, weight_attrs)
        self._register_lut_b_weights(layer, "w2", w2_shape, weight_attrs)

    @staticmethod
    def _register_lut_b_weights(
        layer: RoutedExperts,
        prefix: str,
        tile_shape: tuple[int, int, int],
        weight_attrs: dict[str, Any],
    ) -> None:
        for suffix, tail, dtype in (
            ("packed", (LUT_B_PACKED_TILE_BYTES,), torch.uint8),
            ("codebook", (LUT_B_CODEBOOK_SIZE,), torch.float8_e4m3fn),
            ("scale", (), torch.uint8),
        ):
            param = torch.nn.Parameter(
                torch.empty(*tile_shape, *tail, dtype=dtype), requires_grad=False
            )
            layer.register_parameter(f"{prefix}_weight_{suffix}", param)
            set_weight_attrs(param, weight_attrs)

    def _weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
        *,
        layer: RoutedExperts,
    ) -> bool | None:
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'], got {shard_id!r}.")

        expected_ndim = param.ndim - 1
        if loaded_weight.ndim != expected_ndim:
            raise ValueError(
                f"Expected a per-expert LUT-B tensor with {expected_ndim} dims, "
                f"got shape {tuple(loaded_weight.shape)} for {weight_name}."
            )

        local_expert_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
        if local_expert_id == -1:
            return False if return_success else None

        if shard_id in ("w1", "w3"):
            shard_dim = 0
            target = param.data[local_expert_id]
            shard_size = target.shape[0] // self.moe.w13_num_shards
            target = target.narrow(0, 0 if shard_id == "w1" else shard_size, shard_size)
        else:
            shard_dim = 1
            target = param.data[local_expert_id]

        tp_size = self.moe.tp_size
        if loaded_weight.shape[shard_dim] % tp_size:
            raise ValueError(
                "LUT-B MoE TP loading requires tile counts divisible by "
                f"tp_size={tp_size}; got {loaded_weight.shape[shard_dim]} tiles."
            )
        per_rank_tiles = loaded_weight.shape[shard_dim] // tp_size
        loaded_weight = loaded_weight.narrow(
            shard_dim, self.moe.tp_rank * per_rank_tiles, per_rank_tiles
        )
        target.zero_()
        for dim, (target_size, loaded_size) in enumerate(
            zip(target.shape, loaded_weight.shape)
        ):
            if loaded_size > target_size:
                raise ValueError(
                    "LUT-B checkpoint shard is larger than the local expert "
                    f"buffer: {loaded_weight.shape} vs {target.shape}."
                )
            if loaded_size < target_size:
                target = target.narrow(dim, 0, loaded_size)
        target.copy_(loaded_weight.to(device=target.device, dtype=target.dtype))
        return True if return_success else None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        for name, packed, codebook, scale in (
            (
                "w13_weight",
                layer.w13_weight_packed,
                layer.w13_weight_codebook,
                layer.w13_weight_scale,
            ),
            (
                "w2_weight",
                layer.w2_weight_packed,
                layer.w2_weight_codebook,
                layer.w2_weight_scale,
            ),
        ):
            if self.experts_cls is LutBSm107Experts:
                weight = lut_b_to_sm107_tiles(packed, codebook, scale)
            else:
                weight = dequantize_lut_b(
                    packed, codebook, scale, out_dtype=layer.params_dtype
                )
            replace_parameter(layer, name, weight)

        for name in (
            "w13_weight_packed",
            "w13_weight_codebook",
            "w13_weight_scale",
            "w2_weight_packed",
            "w2_weight_codebook",
            "w2_weight_scale",
        ):
            delattr(layer, name)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_lut_b_moe_kernel(
            self.experts_cls,
            self.moe,
            self.moe_quant_config,
            layer._expert_routing_tables(),
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
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
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
