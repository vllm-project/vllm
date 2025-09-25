# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pick & Ban Routing Algorithms for MoE Models

This module implements the Pick and Ban routing strategies as described in the
paper:
- Pick: Key expert enhancement strategy (Performance-Oriented)
- Ban: Importance-based expert pruning (Efficiency-Oriented)

These algorithms modify the MoE routing logic during inference to achieve
more efficient and accurate expert selection without requiring training or
fine-tuning.
"""

import logging
from typing import Callable, Optional

import torch

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute softmax with numerical stability."""
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def normalize(weights: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize weights to sum to 1."""
    return weights / torch.sum(weights, dim=dim, keepdim=True)


class PickBanRouting:
    """
    Pick & Ban Routing Algorithms for MoE Models
    
    This class implements both Pick and Ban routing strategies that can be used
    as custom routing functions in vLLM's MoE layers.
    """
    
    def __init__(self,
                 key_experts_per_layer: Optional[dict[int, set[int]]] = None,
                 lambda_threshold: float = 0.7,
                 tau_threshold: float = 0.9,
                 strategy: str = "pick_and_ban"):
        """
        Initialize Pick & Ban routing.
        
        Args:
            key_experts_per_layer: Dict mapping layer indices to sets of key
                            expert IDs e.g., {10: {43}, 11: {12, 45}} means
                              layer 10 has key expert 43, layer 11 has key
                              experts 12 and 45
            lambda_threshold: Threshold λ for Pick strategy (0.7-0.9)
            tau_threshold: Threshold τ for Ban strategy (0.7-0.9)
            strategy: Routing strategy to use ("pick", "ban", "pick_and_ban")
        """
        self.key_experts_per_layer = key_experts_per_layer or {}
        self.lambda_threshold = lambda_threshold
        self.tau_threshold = tau_threshold
        self.strategy = strategy
        
        # Validate thresholds
        if not 0.0 <= lambda_threshold <= 1.0:
            raise ValueError(
                f"lambda_threshold must be in [0, 1], got {lambda_threshold}")
        if not 0.0 <= tau_threshold <= 1.0:
            raise ValueError(
                f"tau_threshold must be in [0, 1], got {tau_threshold}")
        if strategy not in ["pick", "ban", "pick_and_ban"]:
            raise ValueError(
                f"strategy must be one of ['pick', 'ban', 'pick_and_ban'], "
                f"got {strategy}"
            )
    
    def pick_routing(
        self,
        router_logits: torch.Tensor,
        top_k: int,
        key_experts: set[int],
        lambda_threshold: Optional[float] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pick routing algorithm: Key expert enhancement strategy.
        
        Args:
            router_logits: [num_tokens, num_experts] or [num_experts] router
                          logits
            top_k: Original top-k value (e.g., 6 for DeepSeek-V2)
            key_experts: Set of key expert IDs for current layer
            lambda_threshold: Threshold λ for non-key expert selection
            
        Returns:
            (selected_experts, weights): Expert IDs and normalized weights
        """
        if lambda_threshold is None:
            lambda_threshold = self.lambda_threshold
            
        # Handle both 1D and 2D router_logits
        if router_logits.dim() == 1:
            router_logits = router_logits.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        num_tokens, num_experts = router_logits.shape
        
        # Step 1: Compute original routing weights (softmax)
        router_weights = softmax(router_logits, dim=-1)
        
        # Initialize output tensors
        selected_experts = torch.zeros((num_tokens, top_k),
                                       dtype=torch.long,
                                       device=router_logits.device)
        weights = torch.zeros((num_tokens, top_k),
                              dtype=torch.float32,
                              device=router_logits.device)
        
        for token_idx in range(num_tokens):
            token_weights = router_weights[token_idx]
            
            # Step 2: Force include all key experts
            selected_expert_set = set(key_experts)
            
            # Step 3: Select non-key experts based on λ threshold
            non_key_experts = []
            for expert_id in range(num_experts):
                if expert_id not in key_experts:
                    non_key_experts.append(
                        (expert_id, token_weights[expert_id].item()))
            
            # Sort by weight (descending)
            non_key_experts.sort(key=lambda x: x[1], reverse=True)
            
            # Apply λ threshold: only keep experts with cumulative weight < λ
            cumsum = 0.0
            candidates = []
            for expert_id, weight in non_key_experts:
                cumsum += weight
                if cumsum < lambda_threshold:
                    candidates.append(expert_id)
                else:
                    break
            
            # Step 4: Fill remaining slots up to top_k
            remaining_slots = max(0, top_k - len(selected_expert_set))
            final_non_key = candidates[:remaining_slots]
            selected_expert_set.update(final_non_key)
            
            # Step 5: Get final expert list and weights
            final_experts = sorted(list(selected_expert_set))
            
            # If we have more experts than top_k, keep only the top_k with
            # highest weights
            if len(final_experts) > top_k:
                expert_weights_temp = [(expert_id,
                                        token_weights[expert_id].item())
                                       for expert_id in final_experts]
                expert_weights_temp.sort(key=lambda x: x[1], reverse=True)
                final_experts = [
                    expert_id for expert_id, _ in expert_weights_temp[:top_k]
                ]
            
            # Pad with -1 if we have fewer than top_k experts
            while len(final_experts) < top_k:
                final_experts.append(-1)
            
            # Get weights for selected experts
            expert_weights = []
            for expert_id in final_experts:
                if expert_id == -1:
                    expert_weights.append(0.0)
                else:
                    expert_weights.append(token_weights[expert_id].item())
            
            # Normalize weights
            expert_weights = torch.tensor(expert_weights,
                                          device=router_logits.device)
            if torch.sum(expert_weights) > 0:
                expert_weights = normalize(expert_weights)

            selected_experts[token_idx] = torch.tensor(
                final_experts, device=router_logits.device)
            weights[token_idx] = expert_weights
        
        if squeeze_output:
            selected_experts = selected_experts.squeeze(0)
            weights = weights.squeeze(0)
            
        return selected_experts, weights
    
    def ban_routing(
        self,
        router_logits: torch.Tensor,
        top_k: int,
        tau_threshold: Optional[float] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ban routing algorithm: Importance-based expert pruning.
        
        Args:
            router_logits: [num_tokens, num_experts] or [num_experts] router
                          logits
            top_k: Original top-k value
            tau_threshold: Cumulative weight threshold τ
            
        Returns:
            (selected_experts, weights): Expert IDs and normalized weights
        """
        if tau_threshold is None:
            tau_threshold = self.tau_threshold
            
        # Handle both 1D and 2D router_logits
        if router_logits.dim() == 1:
            router_logits = router_logits.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        num_tokens, num_experts = router_logits.shape
        
        # Step 1: Compute routing weights
        router_weights = softmax(router_logits, dim=-1)
        
        # Initialize output tensors
        selected_experts = torch.zeros((num_tokens, top_k),
                                       dtype=torch.long,
                                       device=router_logits.device)
        weights = torch.zeros((num_tokens, top_k),
                              dtype=torch.float32,
                              device=router_logits.device)
        
        for token_idx in range(num_tokens):
            token_weights = router_weights[token_idx]
            
            # Step 2: Sort experts by weight (descending)
            sorted_indices = torch.argsort(token_weights, descending=True)
            
            # Step 3: Select experts until cumulative weight >= τ
            cumsum = 0.0
            selected_expert_list = []
            for idx in sorted_indices:
                cumsum += token_weights[idx].item()
                selected_expert_list.append(idx.item())
                if cumsum >= tau_threshold:
                    break
            
            # Step 4: Limit to top_k experts
            if len(selected_expert_list) > top_k:
                selected_expert_list = selected_expert_list[:top_k]
            
            # Pad with -1 if we have fewer than top_k experts
            while len(selected_expert_list) < top_k:
                selected_expert_list.append(-1)
            
            # Step 5: Get weights and normalize
            expert_weights = []
            for expert_id in selected_expert_list:
                if expert_id == -1:
                    expert_weights.append(0.0)
                else:
                    expert_weights.append(token_weights[expert_id].item())
            
            expert_weights = torch.tensor(expert_weights,
                                          device=router_logits.device)
            if torch.sum(expert_weights) > 0:
                expert_weights = normalize(expert_weights)

            selected_experts[token_idx] = torch.tensor(
                selected_expert_list, device=router_logits.device)
            weights[token_idx] = expert_weights
        
        if squeeze_output:
            selected_experts = selected_experts.squeeze(0)
            weights = weights.squeeze(0)
            
        return selected_experts, weights
    
    def pick_and_ban_routing(
        self,
        router_logits: torch.Tensor,
        top_k: int,
        key_experts: set[int],
        lambda_threshold: Optional[float] = None,
        tau_threshold: Optional[float] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combined Pick & Ban routing strategy.
        
        First applies Pick to force include key experts, then applies Ban
        to prune non-key experts based on importance.
        
        Args:
            router_logits: [num_tokens, num_experts] or [num_experts] router
                          logits
            top_k: Original top-k value
            key_experts: Set of key expert IDs for current layer
            lambda_threshold: Threshold λ for Pick strategy
            tau_threshold: Threshold τ for Ban strategy
            
        Returns:
            (selected_experts, weights): Expert IDs and normalized weights
        """
        logger.info(
            "🚀 Pick & Ban routing called! Key experts: %s, top_k: %s",
            key_experts, top_k
        )
        logger.info(
            "📊 Router logits shape: %s, λ: %s, τ: %s",
            router_logits.shape, lambda_threshold, tau_threshold
        )
        if lambda_threshold is None:
            lambda_threshold = self.lambda_threshold
        if tau_threshold is None:
            tau_threshold = self.tau_threshold
            
        # Handle both 1D and 2D router_logits
        if router_logits.dim() == 1:
            router_logits = router_logits.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        num_tokens, num_experts = router_logits.shape
        
        # Step 1: Compute routing weights
        router_weights = softmax(router_logits, dim=-1)
        
        # Initialize output tensors
        selected_experts = torch.zeros((num_tokens, top_k),
                                       dtype=torch.long,
                                       device=router_logits.device)
        weights = torch.zeros((num_tokens, top_k),
                              dtype=torch.float32,
                              device=router_logits.device)
        
        for token_idx in range(num_tokens):
            token_weights = router_weights[token_idx]
            
            # Step 1: Force include key experts (Pick)
            selected_expert_set = set(key_experts)
            
            # Step 2: Apply Ban strategy to non-key experts
            non_key_experts = []
            for expert_id in range(num_experts):
                if expert_id not in key_experts:
                    non_key_experts.append(
                        (expert_id, token_weights[expert_id].item()))
            
            # Sort by weight (descending)
            non_key_experts.sort(key=lambda x: x[1], reverse=True)
            
            # Apply τ threshold to non-key experts
            cumsum = 0.0
            ban_selected = []
            for expert_id, weight in non_key_experts:
                cumsum += weight
                if cumsum < tau_threshold:
                    ban_selected.append(expert_id)
                else:
                    break
            
            # Step 3: Merge and limit to top_k
            remaining_slots = max(0, top_k - len(selected_expert_set))
            final_non_key = ban_selected[:remaining_slots]
            selected_expert_set.update(final_non_key)
            
            # Get final expert list
            final_experts = sorted(list(selected_expert_set))
            
            # If we have more experts than top_k, keep only the top_k with
            # highest weights
            if len(final_experts) > top_k:
                expert_weights_temp = [(expert_id,
                                        token_weights[expert_id].item())
                                       for expert_id in final_experts]
                expert_weights_temp.sort(key=lambda x: x[1], reverse=True)
                final_experts = [
                    expert_id for expert_id, _ in expert_weights_temp[:top_k]
                ]
            
            # Pad with -1 if we have fewer than top_k experts
            while len(final_experts) < top_k:
                final_experts.append(-1)
            
            # Get weights and normalize
            expert_weights = []
            for expert_id in final_experts:
                if expert_id == -1:
                    expert_weights.append(0.0)
                else:
                    expert_weights.append(token_weights[expert_id].item())
            
            expert_weights = torch.tensor(expert_weights,
                                          device=router_logits.device)
            if torch.sum(expert_weights) > 0:
                expert_weights = normalize(expert_weights)

            selected_experts[token_idx] = torch.tensor(
                final_experts, device=router_logits.device)
            weights[token_idx] = expert_weights
        
        if squeeze_output:
            selected_experts = selected_experts.squeeze(0)
            weights = weights.squeeze(0)
            
        return selected_experts, weights
    
    def route_tokens(
            self,
            router_logits: torch.Tensor,
            top_k: int,
            layer_idx: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Main routing function that dispatches to the appropriate strategy.
        
        Args:
            router_logits: [num_tokens, num_experts] or [num_experts] router
                          logits
            top_k: Number of experts to select
            layer_idx: Layer index to get key experts for (if using Pick
                      strategy)
            
        Returns:
            (selected_experts, weights): Expert IDs and normalized weights
        """
        if self.strategy == "pick":
            if layer_idx is None:
                raise ValueError(
                    "layer_idx must be provided for Pick strategy")
            key_experts = self.key_experts_per_layer.get(layer_idx, set())
            return self.pick_routing(router_logits, top_k, key_experts)

        elif self.strategy == "ban":
            return self.ban_routing(router_logits, top_k)

        elif self.strategy == "pick_and_ban":
            if layer_idx is None:
                raise ValueError(
                    "layer_idx must be provided for Pick & Ban strategy")
            key_experts = self.key_experts_per_layer.get(layer_idx, set())
            return self.pick_and_ban_routing(router_logits, top_k, key_experts)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


def create_pick_routing_function(key_experts_per_layer: dict[int, set[int]],
                                 lambda_threshold: float = 0.7) -> Callable:
    """
    Create a Pick routing function compatible with vLLM's
    custom_routing_function interface.
    
    Args:
        key_experts_per_layer: Dict mapping layer indices to sets of key expert
                              IDs
        lambda_threshold: Threshold λ for Pick strategy
        
    Returns:
        Routing function that can be used as custom_routing_function in FusedMoE
    """
    router = PickBanRouting(key_experts_per_layer=key_experts_per_layer,
                            lambda_threshold=lambda_threshold,
                            strategy="pick")

    def pick_routing_function(
            hidden_states: torch.Tensor, gating_output: torch.Tensor,
            topk: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pick routing function compatible with vLLM interface.
        
        Note: This is a simplified version that doesn't use layer_idx.
        For full functionality, you would need to modify the MoE layer to pass
        layer information.
        """
        # For now, we'll use the first layer's key experts as a fallback
        # In a real implementation, you'd want to pass layer information
        layer_idx = 0  # This should be passed from the MoE layer
        key_experts = router.key_experts_per_layer.get(layer_idx, set())

        selected_experts, weights = router.pick_routing(
            gating_output, topk, key_experts)

        # Convert to the format expected by vLLM
        # vLLM expects (topk_weights, topk_ids) where topk_ids are int32
        return weights.to(torch.float32), selected_experts.to(torch.int32)

    return pick_routing_function


def create_ban_routing_function(tau_threshold: float = 0.9) -> Callable:
    """
    Create a Ban routing function compatible with vLLM's
    custom_routing_function interface.
    
    Args:
        tau_threshold: Threshold τ for Ban strategy
        
    Returns:
        Routing function that can be used as custom_routing_function in FusedMoE
    """
    router = PickBanRouting(tau_threshold=tau_threshold, strategy="ban")

    def ban_routing_function(
            hidden_states: torch.Tensor, gating_output: torch.Tensor,
            topk: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ban routing function compatible with vLLM interface.
        """
        selected_experts, weights = router.ban_routing(gating_output, topk)

        # Convert to the format expected by vLLM
        return weights.to(torch.float32), selected_experts.to(torch.int32)

    return ban_routing_function


def create_pick_and_ban_routing_function(
        key_experts_per_layer: dict[int, set[int]],
        lambda_threshold: float = 0.7,
        tau_threshold: float = 0.9) -> Callable:
    """
    Create a combined Pick & Ban routing function compatible with vLLM's
    interface.
    
    Args:
        key_experts_per_layer: Dict mapping layer indices to sets of key expert
                              IDs
        lambda_threshold: Threshold λ for Pick strategy
        tau_threshold: Threshold τ for Ban strategy
        
    Returns:
        Routing function that can be used as custom_routing_function in FusedMoE
    """
    router = PickBanRouting(key_experts_per_layer=key_experts_per_layer,
                            lambda_threshold=lambda_threshold,
                            tau_threshold=tau_threshold,
                            strategy="pick_and_ban")

    def pick_and_ban_routing_function(
            hidden_states: torch.Tensor, gating_output: torch.Tensor,
            topk: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combined Pick & Ban routing function compatible with vLLM interface.
        """
        print("=" * 60)
        print("🚀🚀🚀 PICK & BAN ROUTING FUNCTION CALLED! 🚀🚀🚀")
        print(f"🎯 topk: {topk}, gating_output shape: {gating_output.shape}")
        print(f"🎯 hidden_states shape: {hidden_states.shape}")
        print("=" * 60)
        
        logger.info(
            "🎯 Pick & Ban routing function called! topk: %s, "
            "gating_output shape: %s",
            topk, gating_output.shape
        )

        # For now, we'll use the first layer's key experts as a fallback
        layer_idx = 0  # This should be passed from the MoE layer
        key_experts = router.key_experts_per_layer.get(layer_idx, set())

        print(f"🔑 Using key experts for layer {layer_idx}: {key_experts}")
        logger.info(
            "🔑 Using key experts for layer %s: %s", layer_idx, key_experts)

        selected_experts, weights = router.pick_and_ban_routing(
            gating_output,
            topk,
            key_experts,
            lambda_threshold=router.lambda_threshold,
            tau_threshold=router.tau_threshold)

        print(
            f"📤 Returning weights shape: {weights.shape}, "
            f"selected_experts shape: {selected_experts.shape}"
        )
        print("✅✅✅ PICK & BAN ROUTING COMPLETED! ✅✅✅")
        print("=" * 60)

        logger.info(
            "📤 Returning weights shape: %s, selected_experts shape: %s",
            weights.shape, selected_experts.shape
        )

        # Convert to the format expected by vLLM
        return weights.to(torch.float32), selected_experts.to(torch.int32)

    return pick_and_ban_routing_function


# Example usage and configuration
def get_example_key_experts() -> dict[int, set[int]]:
    """
    Get example key expert configuration.
    
    This is a placeholder - in practice, you would determine key experts
    through offline analysis of your specific model and dataset.
    """
    return {
        # Layer 10: Expert 43 is key for mathematical reasoning
        10: {43},
        # Layer 11: Experts 12 and 45 are key for code generation
        11: {12, 45},
        # Layer 12: Expert 7 is key for common language understanding
        12: {7},
        # Add more layers as needed...
    }


# Pre-configured routing functions for easy use
PICK_ROUTING_LAMBDA_07 = create_pick_routing_function(
    key_experts_per_layer=get_example_key_experts(), lambda_threshold=0.7)

PICK_ROUTING_LAMBDA_08 = create_pick_routing_function(
    key_experts_per_layer=get_example_key_experts(), lambda_threshold=0.8)

PICK_ROUTING_LAMBDA_09 = create_pick_routing_function(
    key_experts_per_layer=get_example_key_experts(), lambda_threshold=0.9)

BAN_ROUTING_TAU_07 = create_ban_routing_function(tau_threshold=0.7)
BAN_ROUTING_TAU_08 = create_ban_routing_function(tau_threshold=0.8)
BAN_ROUTING_TAU_09 = create_ban_routing_function(tau_threshold=0.9)

PICK_AND_BAN_ROUTING = create_pick_and_ban_routing_function(
    key_experts_per_layer=get_example_key_experts(),
    lambda_threshold=0.7,
    tau_threshold=0.9)