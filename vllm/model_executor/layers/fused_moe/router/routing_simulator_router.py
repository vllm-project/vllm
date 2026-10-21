# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Any

import torch

import vllm.envs as envs
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

logger = init_logger(__name__)


class RoutingStrategy(ABC):
    """Base class for token-to-expert routing strategies."""

    @abstractmethod
    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids)

        """
        pass


class DistributionBasedRouting(RoutingStrategy):
    """Distribution-based random routing strategy with configurable distributions.

    This routing strategy randomly selects experts for each token based on
    different probability distributions. Currently supports uniform,
    uniform_subset, and normal distributions for testing different routing
    patterns.
    """

    def __init__(self, distribution: str = "uniform", **distribution_params: Any):
        """Initialize distribution-based routing.

        Args:
            distribution: Type of distribution to use for sampling
                - "uniform": Uniform distribution (default)
                - "uniform_subset": Uniform over a prefix of experts
                - "normal": Normal/Gaussian distribution
            **distribution_params: Parameters specific to the
                chosen distribution
                For "uniform": No additional parameters needed
                For "uniform_subset": subset_size (required)
                For "normal": mean (default: 0.0), std (default: 1.0)

        """
        self.distribution = distribution.lower()
        self.distribution_params = distribution_params

        # Validate distribution and parameters
        self._validate_distribution_params()

    def _validate_distribution_params(self):
        """Validate distribution type and parameters."""
        valid_distributions = ["uniform", "uniform_subset", "normal"]

        if self.distribution not in valid_distributions:
            raise ValueError(
                f"Unsupported distribution: {self.distribution}. "
                f"Supported distributions: {valid_distributions}"
            )

        if self.distribution == "uniform_subset":
            subset_size = self.distribution_params.get("subset_size")
            if subset_size is None:
                raise ValueError(
                    "uniform_subset routing requires subset_size to be set"
                )
            if subset_size < 1:
                raise ValueError(f"subset_size must be >= 1, got {subset_size}")

        # Set default parameters if not provided
        if self.distribution == "normal":
            self.distribution_params.setdefault("mean", 0.0)
            self.distribution_params.setdefault("std", 1.0)

    def _resolve_pool_size(self, num_experts: int, top_k: int) -> int:
        if self.distribution == "uniform_subset":
            subset_size = self.distribution_params["subset_size"]
            if subset_size > num_experts:
                raise ValueError(
                    f"subset_size ({subset_size}) exceeds num_experts ({num_experts})"
                )
            if top_k > subset_size:
                raise ValueError(f"top_k ({top_k}) exceeds subset_size ({subset_size})")
            return subset_size
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) exceeds num_experts ({num_experts})")
        return num_experts

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly select experts for each token using the specified distribution.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids) where:
            - topk_weights: Weights based on distribution sampling
            - topk_ids: Expert indices sampled from the distribution

        """
        num_tokens = hidden_states.shape[0]
        num_experts = router_logits.shape[-1]

        if indices_type is None:
            indices_type = torch.long

        pool_size = self._resolve_pool_size(num_experts, top_k)

        # Generate expert IDs based on the specified distribution
        topk_ids = self._sample_expert_ids(
            num_tokens,
            pool_size,
            top_k,
            hidden_states.device,
            indices_type,
        )

        # Generate weights based on the distribution
        topk_weights = self._generate_weights(num_tokens, top_k, hidden_states.device)

        return topk_weights, topk_ids

    def _sample_expert_ids(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        device: torch.device,
        indices_type: torch.dtype,
    ) -> torch.Tensor:
        """Sample expert IDs based on the specified distribution."""
        if self.distribution in ("uniform", "uniform_subset"):
            # Random scores + top-k avoids duplicate expert ids.
            scores = torch.rand(num_tokens, num_experts, device=device)
            _, topk_ids = torch.topk(scores, top_k, dim=-1)
            return topk_ids.to(indices_type)

        elif self.distribution == "normal":
            # For normal distribution, sample continuous values and map to
            # expert IDs
            continuous_samples = self._sample_continuous_distribution(
                num_tokens, top_k, device
            )

            # Map continuous samples to expert indices
            # Normalize to [0, 1] range and scale to [0, num_experts)
            normalized_samples = self._normalize_samples(continuous_samples)
            expert_ids = (normalized_samples * num_experts).long()
            expert_ids = torch.clamp(expert_ids, 0, num_experts - 1)

            return expert_ids.to(dtype=indices_type)

        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def _sample_continuous_distribution(
        self, num_tokens: int, top_k: int, device: torch.device
    ) -> torch.Tensor:
        """Sample from continuous distributions."""
        shape = (num_tokens, top_k)

        if self.distribution == "normal":
            mean = self.distribution_params["mean"]
            std = self.distribution_params["std"]
            return torch.normal(mean, std, size=shape, device=device)

        else:
            raise ValueError(
                f"Unsupported continuous distribution: {self.distribution}"
            )

    def _normalize_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Normalize samples to [0, 1] range."""
        if self.distribution == "normal":
            # Use sigmoid to map normal distribution to [0, 1]
            return torch.sigmoid(samples)

        else:
            raise ValueError(
                f"Unsupported distribution for normalization: {self.distribution}"
            )

    def _generate_weights(
        self, num_tokens: int, top_k: int, device: torch.device
    ) -> torch.Tensor:
        """Generate weights based on the distribution."""
        if self.distribution in ("uniform", "uniform_subset"):
            # All-ones weights for uniform distribution
            return torch.ones(
                (num_tokens, top_k),
                dtype=torch.float32,
                device=device,
            )

        elif self.distribution == "normal":
            # For normal distribution, generate weights from the same
            # distribution
            continuous_weights = self._sample_continuous_distribution(
                num_tokens, top_k, device
            )
            # Normalize to positive values and sum to 1
            weights = torch.abs(continuous_weights)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            return weights

        else:
            raise ValueError(
                f"Unsupported distribution for weight generation: {self.distribution}"
            )

    def get_distribution_info(self) -> dict:
        """Get information about the current distribution configuration."""
        return {
            "distribution": self.distribution,
            "parameters": self.distribution_params.copy(),
        }


class RoutingSimulator:
    """Token-to-Expert Routing Simulator.

    This class provides a framework for testing and comparing different
    routing strategies for MoE models. It can simulate routing behavior
    and collect statistics for analysis.
    """

    # Class-level registry of routing strategies
    _routing_strategies: dict[str, RoutingStrategy] = {
        # Basic routing strategies
        "uniform_random": DistributionBasedRouting(
            distribution="uniform", mean=0.0, std=1.0
        ),
        "normal_routing": DistributionBasedRouting(
            distribution="normal", mean=0.0, std=1.0
        ),
    }

    @classmethod
    def register_strategy(cls, name: str, strategy: RoutingStrategy):
        """Register a custom routing strategy.

        Args:
            name: Name of the strategy
            strategy: RoutingStrategy instance

        """
        cls._routing_strategies[name] = strategy

    @classmethod
    def _resolve_strategy(cls, strategy_name: str) -> RoutingStrategy:
        if strategy_name == "uniform_subset":
            subset_size = envs.VLLM_MOE_ROUTING_SIMULATION_SUBSET_SIZE
            if subset_size is None:
                raise ValueError(
                    "uniform_subset routing requires "
                    "VLLM_MOE_ROUTING_SIMULATION_SUBSET_SIZE to be set"
                )
            return DistributionBasedRouting(
                distribution="uniform_subset",
                subset_size=subset_size,
            )

        if strategy_name not in cls._routing_strategies:
            raise ValueError(
                f"Unknown routing strategy: {strategy_name}. "
                f"Available strategies: {cls.get_available_strategies()}"
            )
        return cls._routing_strategies[strategy_name]

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """Get list of available routing strategy names.

        Returns:
            List of available strategy names

        """
        strategies = list(cls._routing_strategies.keys())
        strategies.append("uniform_subset")
        return strategies

    @staticmethod
    def simulate_routing(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        strategy_name: str,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate token-to-expert routing using the specified strategy.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            strategy_name: Name of the routing strategy to use
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids)

        """
        if (
            strategy_name not in RoutingSimulator._routing_strategies
            and strategy_name != "uniform_subset"
        ):
            raise ValueError(
                f"Unknown routing strategy: {strategy_name}. "
                f"Available strategies: "
                f"{RoutingSimulator.get_available_strategies()}"
            )
        logger.warning_once(
            "Simulating MoE routing using a %s strategy. "
            "This should only be used for performance testing. "
            "Model outputs will not be valid.",
            strategy_name,
        )

        strategy = RoutingSimulator._resolve_strategy(strategy_name)
        return strategy.route_tokens(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            indices_type=indices_type,
        )


class RoutingSimulatorRouter(BaseRouter):
    """Router that uses routing simulation strategies for testing/debugging."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.Simulated

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use routing simulator to compute routing."""
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        topk_weights, topk_ids = RoutingSimulator.simulate_routing(
            hidden_states=hidden_states,
            router_logits=router_logits,
            strategy_name=routing_strategy,
            top_k=self.top_k,
            indices_type=indices_type,
        )
        return topk_weights, topk_ids
