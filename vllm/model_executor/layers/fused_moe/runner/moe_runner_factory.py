# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
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
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)


def create_moe_runner(
    layer: torch.nn.Module,
    moe_config: FusedMoEConfig,
    router: FusedMoERouter,
    routed_input_transform: torch.nn.Module | None,
    gate: torch.nn.Module | None,
    shared_experts: SharedExperts | None,
    quant_method: FusedMoEMethodBase,
    reduce_results: bool,
    enable_dbo: bool,
) -> MoERunner:
    if moe_config.moe_parallel_config.use_dp_chunking:
        return ChunkingMoERunner(
            layer,
            moe_config,
            router,
            routed_input_transform,
            gate,
            shared_experts,
            quant_method,
            reduce_results,
            enable_dbo,
        )
    else:
        return DefaultMoERunner(
            layer,
            moe_config,
            router,
            routed_input_transform,
            gate,
            shared_experts,
            quant_method,
            reduce_results,
            enable_dbo,
        )
