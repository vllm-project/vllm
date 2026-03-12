import os
from dataclasses import dataclass

import pytest
import torch

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.v1.worker.workspace import init_workspace_manager


@dataclass(frozen=True)
class MoeLayerDimensions:
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int


deepseek_dims = MoeLayerDimensions(
    num_experts=256, hidden_size=7168, intermediate_size=2048, top_k=8
)
ml3_dims = MoeLayerDimensions(
    num_experts=128, hidden_size=7168, intermediate_size=4096, top_k=4
)


@pytest.mark.parametrize("layer_dimensions", [deepseek_dims, ml3_dims])
@pytest.mark.parametrize("num_tokens", [1, 64, 1024])
def test_something(layer_dimensions: MoeLayerDimensions, num_tokens: int) -> None:

    # First, we calculate what will be reference results on each rank.
    # TODO: only calculate on rank 0? But then we need a scatter.
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    init_workspace_manager(device)
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(parallel_config=parallel_config)
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context({}, vllm_config=vllm_config),
    ):
        init_distributed_environment()
        initialize_model_parallel(1, 1)
        num_experts = layer_dimensions.num_experts
        hidden_size = layer_dimensions.hidden_size
        intermediate_size = layer_dimensions.intermediate_size
        top_k = layer_dimensions.top_k
        layer = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        layer.to(device)
        layer.quant_method.process_weights_after_loading(layer)
        layer.maybe_init_modular_kernel()

        hidden_states = torch.randn(num_tokens, hidden_size, device=device)
        router_logits = torch.randn(num_tokens, num_experts, device=device)
        reference_results = layer.forward_native(
            hidden_states=hidden_states, router_logits=router_logits
        )

    # Then, do an EP4 run and compare with reference results.
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(parallel_config=parallel_config)
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context({}, vllm_config=vllm_config),
    ):
        init_distributed_environment()
        initialize_model_parallel(1, 1)
        num_experts = layer_dimensions.num_experts
        hidden_size = layer_dimensions.hidden_size
        intermediate_size = layer_dimensions.intermediate_size
        top_k = layer_dimensions.top_k
        layer = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        layer.to(device)
        layer.quant_method.process_weights_after_loading(layer)
        layer.maybe_init_modular_kernel()

        hidden_states = torch.randn(num_tokens, hidden_size, device=device)
        router_logits = torch.randn(num_tokens, num_experts, device=device)
        reference_results = layer.forward_native(
            hidden_states=hidden_states, router_logits=router_logits
        )
