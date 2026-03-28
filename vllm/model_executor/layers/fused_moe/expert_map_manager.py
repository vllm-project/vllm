# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Expert Map Manager for MoE layers.

This module contains the ExpertMapManager class which manages expert ID
mappings and placement strategies for Expert Parallelism in MoE models.
"""

import torch

from vllm.config.parallel import ExpertPlacementStrategy
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

logger = init_logger(__name__)


def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: ExpertPlacementStrategy = "linear",
    num_fused_shared_experts: int = 0,
    return_expert_mask: bool = False,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """
    Calculates how many experts should be assigned to each rank for EP and
    creates a mapping from global to local expert index. Experts are
    distributed evenly across ranks. Any remaining are assigned to the
    last rank.

    Args:
        ep_size: The size of the expert parallel group
        ep_rank: The rank of the current process in the expert parallel
            group
        global_num_experts: The total number of experts in the model.
        expert_placement_strategy: The expert placement strategy.
        num_fused_shared_experts: Number of fused shared experts (for AITER)
        return_expert_mask: Whether to return expert mask for AITER

    Returns:
        tuple[int, Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple containing:
            - local_num_experts (int): The number of experts assigned
                to the current rank.
            - expert_map (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts,) mapping from global to local index.
                Contains -1 for experts not assigned to the current rank.
                Returns None if ep_size is 1.
            - expert_mask (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts + num_fused_shared_experts + 1,)
                containing 1 for experts assigned to the current rank
                and 0 for sentinel.
                Returns None if ep_size is 1.
                Used only when AITER MOE is enabled.
    """
    from typing import get_args

    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None, None)

    # Distribute experts as evenly as possible to each rank.
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    # Create an expert map for the local experts
    if expert_placement_strategy == "linear":
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx : start_idx + local_num_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    elif expert_placement_strategy == "round_robin":
        local_log_experts = torch.arange(
            ep_rank, global_num_experts, ep_size, dtype=torch.int32
        )

        expert_map[local_log_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    else:
        raise ValueError(
            "Unsupported expert placement strategy "
            f"'{expert_placement_strategy}', expected one of "
            f"{get_args(ExpertPlacementStrategy)}"
        )

    expert_mask = None
    if return_expert_mask:
        expert_mask = torch.ones(
            (global_num_experts + num_fused_shared_experts + 1,), dtype=torch.int32
        )
        expert_mask[-1] = 0
        expert_mask[:global_num_experts] = expert_map > -1
        expert_map = torch.cat(
            (
                expert_map,
                torch.tensor(
                    [local_num_experts + i for i in range(num_fused_shared_experts)],
                    dtype=torch.int32,
                ),
            ),
            dim=0,
        )

    return (local_num_experts, expert_map, expert_mask)


class ExpertMapManager:
    """
    Manages expert ID mappings and placement for Expert Parallelism.

    Responsibilities:
    - Calculate local vs global expert counts
    - Map between global, local, and physical expert IDs
    - Manage placement strategies (linear, round_robin)
    - Maintain routing tables for round-robin placement
    - Support dynamic reconfiguration of EP topology
    """

    def __init__(
        self,
        global_num_experts: int,
        logical_num_experts: int,
        moe_parallel_config: FusedMoEParallelConfig,
        placement_strategy: ExpertPlacementStrategy,
        num_fused_shared_experts: int = 0,
        rocm_aiter_enabled: bool = False,
        device: torch.device | None = None,
    ):
        """
        Initialize expert map manager.

        Args:
            global_num_experts: Total number of experts across all ranks
            logical_num_experts: Number of logical (non-redundant) experts
            moe_parallel_config: MoE parallel configuration (contains ep_size,
                                 ep_rank, backend flags)
            placement_strategy: Strategy for placing experts ('linear' or 'round_robin')
            num_fused_shared_experts: Number of fused shared experts (for AITER)
            rocm_aiter_enabled: Whether ROCm AITER fusion is enabled
            device: Device for tensor allocations
        """
        self.global_num_experts = global_num_experts
        self.logical_num_experts = logical_num_experts
        self.moe_parallel_config = moe_parallel_config
        self.num_fused_shared_experts = num_fused_shared_experts
        self.rocm_aiter_enabled = rocm_aiter_enabled
        self.device = device

        # Determine effective placement strategy
        self._placement_strategy = self._determine_placement_strategy(
            placement_strategy
        )

        # Calculate expert mappings
        self._calculate_expert_maps()

        # Initialize routing tables if needed
        self._maybe_init_routing_tables()

    @property
    def ep_size(self) -> int:
        """Expert parallelism world size."""
        return self.moe_parallel_config.ep_size

    @property
    def ep_rank(self) -> int:
        """Expert parallelism rank."""
        return self.moe_parallel_config.ep_rank

    @property
    def local_num_experts(self) -> int:
        """Number of experts assigned to this rank."""
        return self._local_num_experts

    @property
    def expert_map(self) -> torch.Tensor | None:
        """
        Mapping from global expert ID to local expert ID.

        Returns tensor of shape (global_num_experts,) where:
        - expert_map[global_id] = local_id if expert is on this rank
        - expert_map[global_id] = -1 if expert is not on this rank

        Returns None if EP is not enabled (ep_size == 1).
        """
        return self._expert_map

    @property
    def expert_mask(self) -> torch.Tensor | None:
        """
        Expert mask for AITER fusion (ROCm-specific).

        Returns tensor of shape (global_num_experts + num_fused_shared + 1,)
        where 1 indicates expert is on this rank, 0 otherwise.
        """
        return self._expert_mask

    @property
    def placement_strategy(self) -> ExpertPlacementStrategy:
        """Expert placement strategy ('linear' or 'round_robin')."""
        return self._placement_strategy

    @property
    def routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """
        Routing tables for round-robin placement.

        Returns (global_to_physical, physical_to_global, local_to_global)
        or None if not using round-robin or tables not needed.
        """
        if not hasattr(self, "_routing_tables"):
            return None
        return self._routing_tables

    def map_global_to_local(self, global_id: int) -> int:
        """
        Map global expert ID to local expert ID.

        Args:
            global_id: Global expert ID (0 to global_num_experts - 1)

        Returns:
            Local expert ID (0 to local_num_experts - 1)

        Raises:
            ValueError: If expert is not on this rank
        """
        if self._expert_map is None:
            return global_id

        return self._expert_map[global_id].item()

    def is_local_expert(self, global_id: int) -> bool:
        """Check if expert is assigned to this rank."""
        if self._expert_map is None:
            return True
        return self._expert_map[global_id] != -1

    def get_local_expert_ids(self) -> list[int]:
        """Get list of global IDs for experts on this rank."""
        if self._expert_map is None:
            return list(range(self.global_num_experts))

        return torch.where(self._expert_map != -1)[0].tolist()

    def update(
        self,
        new_ep_size: int | None = None,
        new_ep_rank: int | None = None,
    ) -> None:
        """
        Update expert mappings for new EP configuration.

        Used during dynamic reconfiguration (e.g., elastic scaling).

        Args:
            new_ep_size: New EP world size (if changed)
            new_ep_rank: New EP rank (if changed)
        """
        if new_ep_size is not None:
            self.moe_parallel_config.ep_size = new_ep_size
        if new_ep_rank is not None:
            self.moe_parallel_config.ep_rank = new_ep_rank

        # Recalculate everything
        self._placement_strategy = self._determine_placement_strategy(
            self._placement_strategy
        )
        self._calculate_expert_maps()
        self._maybe_init_routing_tables()

    def get_compressed_map_string(self) -> str:
        """
        Get compressed string representation of expert map for logging.

        Returns string mapping local to global expert IDs.
        """
        if self._expert_map is None:
            return f"[0..{self.global_num_experts - 1}]"

        global_indices = torch.where(self._expert_map != -1)[0]
        local_indices = self._expert_map[global_indices]
        return ", ".join(
            f"{local_index.item()}->{global_index.item()}"
            for local_index, global_index in zip(local_indices, global_indices)
        )

    # Private methods

    def _determine_placement_strategy(
        self, requested_strategy: ExpertPlacementStrategy
    ) -> ExpertPlacementStrategy:
        """Determine effective placement strategy based on config."""
        if requested_strategy != "round_robin":
            return requested_strategy

        # Round-robin requires specific conditions
        if self.ep_size == 1:
            return "linear"

        if (
            self.moe_parallel_config.use_all2all_kernels
            and not self.moe_parallel_config.use_deepep_ll_kernels
            and not self.moe_parallel_config.use_nixl_ep_kernels
        ):
            logger.warning(
                "Round-robin placement requires DeepEP-ll or NIXL backend. "
                "Falling back to linear."
            )
            return "linear"

        return "round_robin"

    def _calculate_expert_maps(self) -> None:
        """Calculate expert mappings based on placement strategy."""
        if self.ep_size == 1:
            # No EP, all experts are local
            self._local_num_experts = self.global_num_experts
            self._expert_map = None
            self._expert_mask = None
            return

        # Call determine_expert_map with current config
        (
            self._local_num_experts,
            self._expert_map,
            self._expert_mask,
        ) = determine_expert_map(
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            global_num_experts=self.global_num_experts,
            expert_placement_strategy=self._placement_strategy,
            num_fused_shared_experts=self.num_fused_shared_experts,
            return_expert_mask=self.rocm_aiter_enabled,
        )

        # Move to device if specified
        if self.device is not None:
            if self._expert_map is not None:
                self._expert_map = self._expert_map.to(self.device)
            if self._expert_mask is not None:
                self._expert_mask = self._expert_mask.to(self.device)

    def _maybe_init_routing_tables(self) -> None:
        """Initialize routing tables if needed for round-robin."""
        if self._placement_strategy != "round_robin":
            return

        if (
            not self.moe_parallel_config.use_deepep_ll_kernels
            and not self.moe_parallel_config.use_nixl_ep_kernels
        ):
            return

        if self._expert_map is None:
            return

        self._routing_tables = self._ensure_round_robin_expert_routing_tables()

    def _ensure_round_robin_expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build routing tables for round-robin placement."""
        device_kwargs = {"device": self.device} if self.device is not None else {}

        global_indices = torch.arange(
            self.global_num_experts, dtype=torch.long, **device_kwargs
        )
        owner = torch.remainder(global_indices, self.ep_size)
        local_index = torch.div(global_indices, self.ep_size, rounding_mode="floor")

        base = self.global_num_experts // self.ep_size
        remainder = self.global_num_experts % self.ep_size
        physical_offset = owner * base

        if remainder > 0:
            remainder_tensor = torch.tensor(
                remainder, dtype=torch.long, **device_kwargs
            )
            physical_offset = physical_offset + torch.minimum(owner, remainder_tensor)

        global_to_physical = physical_offset + local_index
        physical_to_global = torch.empty_like(global_to_physical)
        physical_to_global[global_to_physical] = global_indices

        local_global = torch.arange(
            self.ep_rank,
            self.global_num_experts,
            self.ep_size,
            dtype=torch.long,
            **device_kwargs,
        )
        if local_global.numel() != self._local_num_experts:
            local_global = local_global[: self._local_num_experts]

        return (global_to_physical, physical_to_global, local_global)
