# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Octave-W3 MoE experts (format ``octave-w3-planar``) for compressed-tensors.

The config group carries ``num_bits: 3``, ``group_size: 32``, symmetric; the
block Hadamard sizes live in ``quantization_config["octave_w3"]``
(``had_gate_up``, ``had_down``). Only the RDNA3 HIP kernel runs it.
"""

import torch
from compressed_tensors.quantization import QuantizationArgs

from vllm.model_executor.layers.fused_moe import RoutedExperts, SharedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.rdna3_octw3_moe import (
    Rdna3OctaveW3Experts,
    octw3_repack,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

OCTAVE_W3_FORMAT = "octave-w3-planar"


class CompressedTensorsOctaveW3MoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        octave_config: dict,
    ):
        super().__init__(moe)
        if (
            weight_quant.num_bits != 3
            or weight_quant.group_size != 32
            or not weight_quant.symmetric
        ):
            raise ValueError(
                f"{OCTAVE_W3_FORMAT} needs num_bits 3, group_size 32, symmetric; "
                f"got {weight_quant}"
            )
        if not Rdna3OctaveW3Experts._supports_current_device():
            raise ValueError(f"{OCTAVE_W3_FORMAT} MoE runs only on gfx1100")
        if not moe.is_act_and_mul:
            raise ValueError(f"{OCTAVE_W3_FORMAT} MoE needs gated experts")
        self.had_w13 = int(octave_config.get("had_gate_up", 128))
        self.had_w2 = int(octave_config.get("had_down", 32))

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.float16:
            raise ValueError(f"{OCTAVE_W3_FORMAT} MoE runs in float16")
        n = intermediate_size_per_partition
        for size, name in ((hidden_size, "hidden_size"), (n, "intermediate size")):
            if size % 32:
                raise ValueError(f"{OCTAVE_W3_FORMAT}: {name} {size} % 32 != 0")
        for size, had in ((hidden_size, self.had_w13), (n, self.had_w2)):
            if had and size % had:
                raise ValueError(
                    f"{OCTAVE_W3_FORMAT}: Hadamard block {had} must divide the "
                    f"per-rank input size {size}"
                )
        extra_weight_attrs.update({"is_transposed": False, "quant_method": "group"})

        def param(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
            p = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, extra_weight_attrs)

        param(
            "w13_weight_packed",
            (num_experts, 2 * n, hidden_size // 32 * 3),
            torch.int32,
        )
        param("w2_weight_packed", (num_experts, hidden_size, n // 32 * 3), torch.int32)
        param(
            "w13_weight_scale", (num_experts, 2 * n, hidden_size // 32), torch.float16
        )
        param("w2_weight_scale", (num_experts, hidden_size, n // 32), torch.float16)
        param("w13_weight_shape", (num_experts, 2), torch.float32)
        param("w2_weight_shape", (num_experts, 2), torch.float32)
        layer.a13_scale = None
        layer.a2_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        replace_parameter(
            layer, "w13_weight_packed", octw3_repack(layer.w13_weight_packed.data)
        )
        replace_parameter(
            layer, "w2_weight_packed", octw3_repack(layer.w2_weight_packed.data)
        )
        replace_parameter(
            layer,
            "w13_weight_scale",
            layer.w13_weight_scale.data.transpose(1, 2).contiguous(),
        )
        replace_parameter(
            layer,
            "w2_weight_scale",
            layer.w2_weight_scale.data.transpose(1, 2).contiguous(),
        )
        layer.w13_weight = layer.w13_weight_packed
        layer.w2_weight = layer.w2_weight_packed

        from vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )
        from vllm.model_executor.layers.fused_moe.modular_kernel import (
            FusedMoEKernel,
        )

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        prepare_finalize = maybe_make_prepare_finalize(
            moe=self.moe,
            quant_config=self.moe_quant_config,
            routing_tables=layer._expert_routing_tables(),
            allow_new_interface=True,
            use_monolithic=False,
        )
        assert prepare_finalize is not None
        self.moe_kernel = FusedMoEKernel(
            prepare_finalize,
            Rdna3OctaveW3Experts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                had_w13=self.had_w13,
                had_w2=self.had_w2,
            ),
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # 16-bit activations; the descriptor only carries the group scales.
        return int4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            block_shape=[0, 32],
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
