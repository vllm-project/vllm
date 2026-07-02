# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.hw_agnostic.custom_op import CustomOp
from vllm.model_executor.hw_agnostic.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.experts.triton_moe import (
    TritonExperts,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.fused_moe_method_base import (  # noqa: E501
    FusedMoEMethodBase,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.modular_kernel import (
    FusedMoEKernel,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.runner.shared_experts import (  # noqa: E501
    SharedExperts,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.hw_agnostic.layers.fused_moe.routed_experts import (  # noqa: E501
        RoutedExperts,
    )

logger = init_logger(__name__)


# --8<-- [start:unquantized_fused_moe]
@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization (BF16/FP16 inputs and weights)."""

    # --8<-- [end:unquantized_fused_moe]

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.experts_cls = TritonExperts

    @property
    def supports_eplb(self) -> bool:
        return True

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.moe.is_act_and_mul:
            w13_up_dim = 2 * intermediate_size_per_partition
        else:
            w13_up_dim = intermediate_size_per_partition
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts, w13_up_dim, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _setup_kernel(
        self,
        layer: "RoutedExperts",
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        # ``moe_kernel`` is initialized to None in FusedMoEMethodBase. On
        # the first call we replace the parameter normally; on subsequent
        # calls (e.g. RL weight updates that re-trigger
        # process_weights_after_loading) the kernel is already built and
        # CUDA graphs may have captured parameter addresses, so we copy
        # data into the existing storage instead of re-registering.
        is_weight_update = self.moe_kernel is not None  # type: ignore[has-type]
        replace_parameter(
            layer, "w13_weight", w13.contiguous(), prefer_copy=is_weight_update
        )
        replace_parameter(
            layer, "w2_weight", w2.contiguous(), prefer_copy=is_weight_update
        )

        if not is_weight_update:
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.moe_quant_config is not None
            assert self.experts_cls is not None
            prepare_finalize = maybe_make_prepare_finalize(self.moe)
            assert prepare_finalize is not None
            experts = self.experts_cls(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
            )
            self.moe_kernel = FusedMoEKernel(prepare_finalize, experts)

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        super().process_weights_after_loading(layer)
        self._setup_kernel(
            layer=layer,
            w13=layer.w13_weight,
            w2=layer.w2_weight,
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        if self.moe.has_bias:
            return biased_moe_quant_config(
                layer.w13_bias,
                layer.w2_bias,
            )
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.forward(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def forward_native(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
