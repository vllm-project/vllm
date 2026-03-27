# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.routed_experts import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.runner.chunking_moe_runner import (
    ChunkingMoERunner,
)
from vllm.model_executor.layers.fused_moe.runner.default_moe_runner import (
    DefaultMoERunner,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


def create_moe_runner(
    layer_name: str,
    moe_config: FusedMoEConfig,
    router: FusedMoERouter,
    routed_input_transform: torch.nn.Module | None,
    gate: torch.nn.Module | None,
    shared_experts: torch.nn.Module | None,
    routed_experts: RoutedExperts,
    enable_dbo: bool,
    routed_output_transform: torch.nn.Module | None = None,
    apply_scale_to_output: bool = False,
    routed_scaling_factor: float = 1.0,
) -> MoERunner:
    runner = DefaultMoERunner(
        layer_name,
        moe_config,
        router,
        routed_input_transform,
        gate,
        shared_experts,
        routed_experts,
        enable_dbo,
        routed_output_transform=routed_output_transform,
        apply_scale_to_output=apply_scale_to_output,
        routed_scaling_factor=routed_scaling_factor,
    )
    if moe_config.moe_parallel_config.use_dp_chunking:
        return ChunkingMoERunner(
            inner=runner,
            layer_name=layer_name,
            moe_config=moe_config,
            router=router,
            routed_input_transform=routed_input_transform,
            gate=gate,
            shared_experts=shared_experts,
            routed_experts=routed_experts,
            enable_dbo=enable_dbo,
            routed_output_transform=routed_output_transform,
            apply_scale_to_output=apply_scale_to_output,
            routed_scaling_factor=routed_scaling_factor,
        )
    return runner
