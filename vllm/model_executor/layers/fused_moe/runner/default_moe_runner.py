# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.routed_experts import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner_base import MoERunnerBase

logger = init_logger(__name__)


class DefaultMoERunner(MoERunnerBase):
    """
    Standard MoE runner implementation for executing Mixture of Experts layers.

    This is the primary concrete implementation of MoE execution logic, providing
    comprehensive support for standard MoE operations. It handles:
    - Expert routing and token dispatching using various routing strategies
    - Shared experts computation with optional parallel execution using CUDA streams
    - Tensor model parallel and expert parallel operations
    - Multiple quantization methods and optimized kernel selection
    - Both monolithic and decomposed expert execution paths
    - Integration with various parallel execution modes (TP, EP, DP)

    The runner orchestrates the complete MoE forward pass including routing tokens
    to experts, executing expert computations in parallel, and combining results.
    It supports advanced features like overlapped execution of shared experts,
    optimized kernels for different parallel configurations, and seamless
    integration with vLLM's distributed execution framework.

    This implementation is suitable for most standard MoE use cases. For specialized
    scenarios like large batch chunking, alternative runners like ChunkingMoERunner
    may be more appropriate.

    Eventually, this class may be split into more specialized implementations
    for different configurations (e.g., with/without shared experts, gates, etc.).
    """

    def __init__(
        self,
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
    ):
        super().__init__(
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

    @property
    def do_naive_dispatch_combine(self) -> bool:
        return (
            self.moe_config.dp_size > 1
            and not self.routed_experts.quant_method.supports_internal_mk
        )

    def _maybe_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # For naive dispatch/combine Dp/Ep, dispatch the hidden states and
        # router logits to all experts.
        # NOTE: this will be removed once all kernels are migrated into the
        # MoEKernel framework.
        if self.do_naive_dispatch_combine:
            hidden_states, router_logits = get_ep_group().dispatch_router_logits(
                hidden_states,
                router_logits,
                self.moe_config.is_sequence_parallel,
            )

        # NOTE: Similar with DP, PCP also needs dispatch and combine. For
        # simplicity, AgRsAll2All was added separately for PCP here. Maybe
        # we should modify All2AllManager abstraction to better support PCP.
        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().all_gather(
                hidden_states,
                dim=0,
            )
            router_logits = get_pcp_group().all_gather(
                router_logits,
                dim=0,
            )

        return hidden_states, router_logits

    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if self.do_naive_dispatch_combine:
            hidden_states = get_ep_group().combine(
                hidden_states, self.moe_config.is_sequence_parallel
            )

        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().reduce_scatter(
                hidden_states,
                dim=0,
            )

        if self.shared_experts is not None:
            assert shared_output is not None
            return shared_output, hidden_states
        else:
            return hidden_states

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # TODO(bnell): parts of the dispatch/combine steps will go away once
        # #32567 lands and the remaining kernels are made MKs.  The PCP
        # code will probably remain
        hidden_states, router_logits = self._maybe_dispatch(
            hidden_states,
            router_logits,
        )

        shared_output, hidden_states = self._apply_quant_method(
            hidden_states=hidden_states,
            router_logits=router_logits,
            shared_experts_input=shared_experts_input,
        )

        return self._maybe_combine(
            shared_output,
            hidden_states,
        )
