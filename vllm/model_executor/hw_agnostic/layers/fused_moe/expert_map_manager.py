# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Expert Map Manager for MoE layers.

This module contains the ExpertMapManager class which manages expert ID
mappings and placement strategies for Expert Parallelism in MoE models.
"""

import torch

from vllm.config.parallel import ExpertPlacementStrategy
from vllm.logger import init_logger
from vllm.model_executor.hw_agnostic.layers.fused_moe.config import (
    FusedMoEParallelConfig,
)

logger = init_logger(__name__)


def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
) -> tuple[int, torch.Tensor | None]:
    """
    Calculates how many experts should be assigned to each rank for EP and
    creates a mapping from global to local expert index. Experts are
    distributed evenly across ranks; any remainder is assigned to the
    earliest ranks.

    Returns:
        - local_num_experts (int): The number of experts assigned
          to the current rank.
        - expert_map (torch.Tensor | None): A tensor of shape
          (global_num_experts,) mapping from global to local index.
          Contains -1 for experts not assigned to the current rank.
          Returns None if ep_size is 1.
    """
    assert ep_size > 0
    if ep_size == 1:
        return global_num_experts, None

    # Distribute experts as evenly as possible to each rank.
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    start_idx = ep_rank * base_experts + min(ep_rank, remainder)
    expert_map[start_idx : start_idx + local_num_experts] = torch.arange(
        0, local_num_experts, dtype=torch.int32
    )
    return local_num_experts, expert_map


class ExpertMapManager:
    """
    Manages expert ID mappings and placement for Expert Parallelism.

    Responsibilities:
    - Calculate local vs global expert counts
    - Map between global expert IDs and per-rank local IDs
    - Support dynamic reconfiguration of EP topology

    expert_map is None when ep_size == 1 (every expert is local). When
    ep_size > 1, ``expert_map[global_id]`` is the local index for experts
    owned by this rank and -1 otherwise; the experts kernel reads it to
    skip remote experts.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        top_k: int,
        global_num_experts: int,
        num_redundant_experts: int,
        num_expert_group: int | None,
        moe_parallel_config: FusedMoEParallelConfig,
        placement_strategy: ExpertPlacementStrategy,
        enable_eplb: bool,
    ):
        """
        Initialize expert map manager.

        Args:
            global_num_experts: Total number of experts across all ranks.
            moe_parallel_config: MoE parallel configuration (contains
                ep_size, ep_rank, backend flags).
            placement_strategy: Only ``"linear"`` is supported on the
                hw-agnostic path; ``"round_robin"`` is rejected here
                because it requires a HW-specific (DeepEP-LL/NIXL) transport.
        """
        self.global_num_experts = global_num_experts
        self.moe_parallel_config = moe_parallel_config
        self.top_k = top_k
        self.max_num_batched_tokens = max_num_batched_tokens

        # Round-robin needs a DeepEP-LL/NIXL transport, neither of which
        # is supported on the hw-agnostic path (HW-specific only).
        if placement_strategy != "linear":
            raise NotImplementedError(
                f"hw-agnostic FusedMoE supports placement_strategy='linear' "
                f"only; got {placement_strategy!r}."
            )
        self._placement_strategy: ExpertPlacementStrategy = "linear"
        del num_expert_group, num_redundant_experts, enable_eplb

        self._calculate_expert_maps()

        if self.use_ep:
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. "
                "Local/global number of experts: %s/%s. "
                "Experts local to global index map: %s.",
                self.ep_rank,
                self.ep_size,
                self.local_num_experts,
                self.global_num_experts,
                self.get_compressed_map_string(),
            )

    @property
    def use_ep(self) -> int:
        return self.moe_parallel_config.use_ep

    @property
    def ep_size(self) -> int:
        return self.moe_parallel_config.ep_size

    @property
    def ep_rank(self) -> int:
        return self.moe_parallel_config.ep_rank

    @property
    def tp_size(self) -> int:
        return self.moe_parallel_config.tp_size

    @property
    def tp_rank(self) -> int:
        return self.moe_parallel_config.tp_rank

    @property
    def local_num_experts(self) -> int:
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
    def placement_strategy(self) -> ExpertPlacementStrategy:
        return self._placement_strategy

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
        moe_parallel_config: FusedMoEParallelConfig,
        global_num_experts: int,
    ) -> None:
        """
        Update expert mappings for new EP configuration.

        Used during dynamic reconfiguration (e.g., elastic scaling).

        Args:
            global_num_experts: New total number of experts across all ranks
            moe_parallel_config: New MoE parallel configuration (contains ep_size,
                                 ep_rank, backend flags)
        """
        self.moe_parallel_config = moe_parallel_config
        self.global_num_experts = global_num_experts

        assert self._expert_map is not None, "_expert_map must be present."
        with self._expert_map.device:
            self._calculate_expert_maps()

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

    def _calculate_expert_maps(self) -> None:
        self._local_num_experts, self._expert_map = determine_expert_map(
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            global_num_experts=self.global_num_experts,
        )
