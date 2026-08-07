# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.utils.math_utils import round_up


class DeepseekV4AmdMxfp4MoEMethod(Mxfp4MoEMethod):
    def __init__(
        self,
        moe: FusedMoEConfig,
        num_fused_shared_experts: int,
    ):
        super().__init__(moe)
        self.num_fused_shared_experts = num_fused_shared_experts

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        use_native_aiter_shape = (
            self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16
            and self.moe.activation == MoEActivation.SILU
            and self.moe.routing_method == RoutingMethodType.DeepseekV4
            and self.num_fused_shared_experts == 0
        )
        if not use_native_aiter_shape:
            return super().maybe_roundup_sizes(
                hidden_size,
                intermediate_size_per_partition,
                act_dtype,
                moe_parallel_config,
            )

        hidden_size, intermediate_size_per_partition = (
            FusedMoEMethodBase.maybe_roundup_sizes(
                self,
                hidden_size,
                intermediate_size_per_partition,
                act_dtype,
                moe_parallel_config,
            )
        )
        return (
            round_up(hidden_size, 128),
            round_up(intermediate_size_per_partition, 128),
        )


class DeepseekV4AmdRoutedExperts(RoutedExperts):
    def __init__(
        self,
        layer_name: str,
        params_dtype: torch.dtype,
        moe_config: FusedMoEConfig,
        quant_config: QuantizationConfig | None,
        expert_map_manager: ExpertMapManager,
        **kwargs: Any,
    ):
        self.num_fused_shared_experts = expert_map_manager.num_fused_shared_experts
        super().__init__(
            layer_name,
            params_dtype,
            moe_config,
            quant_config,
            expert_map_manager=expert_map_manager,
            **kwargs,
        )

    def _get_quant_method(
        self,
        prefix: str,
        quant_config: QuantizationConfig | None,
        moe_config: FusedMoEConfig,
    ) -> FusedMoEMethodBase:
        quant_method = super()._get_quant_method(prefix, quant_config, moe_config)
        if type(quant_method) is Mxfp4MoEMethod:
            return DeepseekV4AmdMxfp4MoEMethod(
                moe_config,
                num_fused_shared_experts=self.num_fused_shared_experts,
            )
        return quant_method
