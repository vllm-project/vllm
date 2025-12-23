# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Helper class for ROCm AITER shared expert fusion in FusedMoE.

This module encapsulates the scattered logic for AITER shared expert fusion,
providing a cleaner interface for the FusedMoE layer. It handles:
- Capability checks for AITER fused MoE and shared expert fusion
- Computing and validating num_fused_shared_experts
- Initializing topK metadata buffers
- Providing expert map augmentation for determine_expert_map()
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE


@dataclass
class AiterSharedExpertFusion:
    """
    Encapsulates ROCm AITER shared expert fusion logic for FusedMoE.

    This helper class manages the state and operations related to AITER's
    shared expert fusion feature, reducing scattered if-branches in the
    FusedMoE layer.

    Attributes:
        rocm_aiter_fmoe_enabled: Whether ROCm AITER fused MoE is enabled.
        aiter_shared_expert_enabled: Whether AITER shared expert fusion
            is enabled.
        num_fused_shared_experts: Number of shared experts to fuse (0 if
            fusion is disabled).
    """

    rocm_aiter_fmoe_enabled: bool
    aiter_shared_expert_enabled: bool
    num_fused_shared_experts: int

    @classmethod
    def create(cls, n_shared_experts: int | None) -> "AiterSharedExpertFusion":
        """
        Factory method to create an AiterSharedExpertFusion instance.

        Args:
            n_shared_experts: Number of shared experts from the model config,
                or None if not specified.

        Returns:
            An AiterSharedExpertFusion instance with properly initialized
            state.

        Raises:
            ValueError: If n_shared_experts is provided but AITER shared
                expert fusion is not enabled.
        """
        rocm_aiter_fmoe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        aiter_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )

        # Compute num_fused_shared_experts
        num_fused_shared_experts = (
            n_shared_experts
            if n_shared_experts is not None and aiter_shared_expert_enabled
            else 0
        )

        # Validate configuration
        if (
            not aiter_shared_expert_enabled
            and n_shared_experts is not None
            and n_shared_experts > 0
        ):
            raise ValueError(
                "n_shared_experts is only supported on ROCm aiter when "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled"
            )

        return cls(
            rocm_aiter_fmoe_enabled=rocm_aiter_fmoe_enabled,
            aiter_shared_expert_enabled=aiter_shared_expert_enabled,
            num_fused_shared_experts=num_fused_shared_experts,
        )

    @property
    def is_enabled(self) -> bool:
        """Check if AITER fused MoE is enabled."""
        return self.rocm_aiter_fmoe_enabled

    @property
    def is_shared_expert_fusion_enabled(self) -> bool:
        """Check if shared expert fusion is enabled."""
        return self.aiter_shared_expert_enabled

    @property
    def has_fused_shared_experts(self) -> bool:
        """Check if there are shared experts to fuse."""
        return self.num_fused_shared_experts > 0

    def get_expert_map_kwargs(self) -> dict:
        """
        Get additional kwargs for determine_expert_map().

        Returns:
            Dictionary with num_fused_shared_experts and return_expert_mask
            arguments.
        """
        return {
            "num_fused_shared_experts": self.num_fused_shared_experts,
            "return_expert_mask": self.rocm_aiter_fmoe_enabled,
        }

    def validate_expert_mask(self, expert_mask: torch.Tensor | None) -> None:
        """
        Validate that expert_mask contains only 0s and 1s when AITER is enabled.

        Args:
            expert_mask: The expert mask tensor to validate.

        Raises:
            AssertionError: If expert_mask contains values other than 0 and 1.
        """
        if self.rocm_aiter_fmoe_enabled and expert_mask is not None:
            assert torch.all((expert_mask == 0) | (expert_mask == 1)), (
                "Aiter Fused MoE kernel only supports expert_map with 0 and 1s."
            )

    def init_topk_buffers(
        self,
        layer: "FusedMoE",
        vllm_config: VllmConfig,
        dp_size: int,
    ) -> None:
        """
        Initialize AITER topK metadata buffers if shared expert fusion is enabled.

        This method also updates layer.local_num_experts to include fused
        shared experts.

        Args:
            layer: The FusedMoE layer to initialize buffers for.
            vllm_config: The vLLM configuration.
            dp_size: Data parallel size.
        """
        if self.has_fused_shared_experts:
            from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
                init_aiter_topK_meta_data,
            )

            init_aiter_topK_meta_data(
                n_routed_experts=layer.global_num_experts,
                n_shared_experts=self.num_fused_shared_experts,
                top_k=layer.top_k,
                tp_rank=layer.ep_rank if layer.use_ep else layer.tp_rank,
                tp_size=layer.ep_size if layer.use_ep else layer.tp_size,
                shared_experts_score=1.0,
                max_num_tokens=(
                    vllm_config.scheduler_config.max_num_batched_tokens * dp_size
                ),
                is_EP=layer.use_ep,
            )
        layer.local_num_experts += self.num_fused_shared_experts

    def get_expert_map(
        self,
        expert_map: torch.Tensor | None,
        expert_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """
        Get the appropriate expert map based on AITER state.

        When AITER fused MoE is enabled, returns expert_mask instead of
        expert_map.

        Args:
            expert_map: The standard expert map tensor.
            expert_mask: The AITER-specific expert mask tensor.

        Returns:
            expert_mask if AITER is enabled, otherwise expert_map.
        """
        return expert_mask if self.rocm_aiter_fmoe_enabled else expert_map
