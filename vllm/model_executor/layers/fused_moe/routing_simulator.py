# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Token-to-Expert Routing Simulator

This module provides a framework for simulating and testing different
token-to-expert routing strategies for Mixture of Experts (MoE) models.
It supports routing logic customization and includes example implementations
like uniform random routing.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F


class RoutingStrategy(ABC):
    """Base class for token-to-expert routing strategies."""

    @abstractmethod
    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids)
        """
        pass


class UniformRandomRouting(RoutingStrategy):
    """
    Uniform random routing strategy for perfect load balancing in expectation.

    This routing strategy randomly selects experts for each token, providing
    perfect load balancing in expectation. It's useful for performance analysis
    when using dummy weights, but will not produce correct model outputs.
    """

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly select experts for each token.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids) where:
            - topk_weights: All-ones weights [num_tokens, top_k]
            - topk_ids: Random expert indices [num_tokens, top_k]
        """
        num_tokens = hidden_states.shape[0]
        global_num_experts = router_logits.shape[-1]

        if indices_type is None:
            indices_type = torch.long

        # Random expert IDs, uniform in [0, global_num_experts)
        topk_ids = torch.randint(
            low=0,
            high=global_num_experts,
            size=(num_tokens, top_k),
            dtype=indices_type,
            device=hidden_states.device,
        )

        # All-ones weights
        topk_weights = torch.ones(
            (num_tokens, top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )

        return topk_weights, topk_ids


class SoftmaxRouting(RoutingStrategy):
    """
    Standard softmax-based routing strategy.

    This is the default routing strategy that uses softmax on router logits
    to select the top-k experts for each token.
    """

    def __init__(self, renormalize: bool = True):
        """
        Initialize softmax routing.

        Args:
            renormalize: Whether to renormalize the selected expert weights
        """
        self.renormalize = renormalize

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k experts using softmax routing.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids)
        """
        # Apply softmax to get probabilities
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        topk_weights, topk_ids = torch.topk(routing_weights, top_k, dim=-1)

        # Renormalize weights if requested
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1,
                                                           keepdim=True)

        # Convert indices to requested type
        if indices_type is not None:
            topk_ids = topk_ids.to(dtype=indices_type)

        return topk_weights, topk_ids


class WeightedRandomRouting(RoutingStrategy):
    """
    Weighted random routing strategy.

    This strategy samples experts randomly based on the router logits
    probabilities, providing a stochastic routing approach that still respects
    the learned routing preferences while introducing randomness for load
    balancing.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize weighted random routing.

        Args:
            temperature: Temperature for softmax sampling (lower = more
            deterministic)
        """
        self.temperature = temperature

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample top-k experts using weighted random selection.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids)
        """
        num_tokens, num_experts = router_logits.shape

        # Apply temperature scaling and softmax
        scaled_logits = router_logits / self.temperature
        routing_probs = F.softmax(scaled_logits, dim=-1)

        # Sample experts without replacement
        topk_ids = torch.multinomial(routing_probs, top_k, replacement=False)

        # Get corresponding weights
        topk_weights = routing_probs.gather(dim=-1, index=topk_ids)

        # Renormalize weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Convert indices to requested type
        if indices_type is not None:
            topk_ids = topk_ids.to(dtype=indices_type)

        return topk_weights, topk_ids


class RoutingSimulator:
    """
    Token-to-Expert Routing Simulator.

    This class provides a framework for testing and comparing different
    routing strategies for MoE models. It can simulate routing behavior
    and collect statistics for analysis.
    """

    # Class-level registry of routing strategies
    _routing_strategies = {
        "uniform_random": UniformRandomRouting(),
        "softmax": SoftmaxRouting(),
        "weighted_random": WeightedRandomRouting(),
    }

    @classmethod
    def register_strategy(cls, name: str, strategy: RoutingStrategy):
        """
        Register a custom routing strategy.

        Args:
            name: Name of the strategy
            strategy: RoutingStrategy instance
        """
        cls._routing_strategies[name] = strategy

    @classmethod
    def get_available_strategies(cls):
        """
        Get list of available routing strategy names.

        Returns:
            List of available strategy names
        """
        return list(cls._routing_strategies.keys())

    @staticmethod
    def simulate_routing(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        strategy_name: str,
        top_k: int,
        indices_type: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate token-to-expert routing using the specified strategy.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            strategy_name: Name of the routing strategy to use
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids)
        """
        if strategy_name not in RoutingSimulator._routing_strategies:
            raise ValueError(
                f"Unknown routing strategy: {strategy_name}. "
                f"Available strategies: "
                f"{list(RoutingSimulator._routing_strategies.keys())}")

        strategy = RoutingSimulator._routing_strategies[strategy_name]
        return strategy.route_tokens(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            indices_type=indices_type,
        )
