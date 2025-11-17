# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Expert parallelism load balancer (EPLB) metrics and states.

# Glossary

- **Logical Expert**: An expert that is part of the model's logical structure.
  It holds a set of weights and is replicated across multiple physical
  experts.
- **Redundant Expert**: To achieve load balancing, for some popular logical
  experts, we create additional copies of the expert weights. During inference,
  each of these copies can be routed to by the same set of tokens.
- **Physical Expert**: An expert that is instantiated on a specific device.
  It is a replica of a logical expert and can be rearranged across devices.
  I.e., one logical expert may have multiple sets of weights initialized on
  different devices, and each of these sets is a physical expert.
- **Local Physical Expert**: A physical expert that is instantiated on the
  current device.
- **Health Monitoring**: A timeout-based system that tracks per-expert latency
  and immediately masks out experts that exceed the timeout threshold, without
  waiting for the next rearrangement interval. Once masked, experts stay inactive
  and their health status is not updated (we don't track health of inactive experts).

For example: DeepSeek-R1 has 256 logical experts, so each MoE layer
has 256 sets of linear layer weights in the model parameters. If we add 32
redundant experts, DeepSeek-R1 will have 256 + 32 = 288 physical experts in
total. And when deploying, we'll have 288 sets of linear layer weights for each
MoE layer. If we have 32 EP ranks, then each GPU will hold 288 / 32 = 9 local
physical experts.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from logging import DEBUG

import torch
from torch.distributed import ProcessGroup, all_reduce

from vllm.config import ModelConfig, ParallelConfig
from vllm.distributed.parallel_state import (
    get_ep_group,
    get_node_count,
    in_the_same_node_as,
)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MixtureOfExperts

from .rebalance_algo import rebalance_experts, rebalance_masked_experts
from .rebalance_execute import rearrange_expert_weights_inplace

logger = init_logger(__name__)


def safe_scatter_add_with_masked_indices(
    target: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
) -> None:
    """
    Perform scatter_add while filtering out -1 indices (masked values).
    
    During health-based masking, index tensors may contain -1 for masked experts.
    This function filters these out to prevent CUDA index out of bounds errors.
    
    Strategy: Replace -1 indices with 0 AND zero out corresponding source values.
    This means masked positions perform: target[0] += 0 (harmless no-op).
    Expert 0's actual load is unaffected since we're adding zero, not real load values.
    
    Args:
        target: Target tensor to scatter into
        dim: Dimension along which to scatter
        index: Index tensor (may contain -1 for masked values)
        src: Source tensor with values to add
    """
    # Create mask for valid (non--1) indices
    valid_mask = index >= 0
    
    # Replace -1 indices with 0 to avoid CUDA index out of bounds
    # This is safe because we zero out the corresponding source values below
    safe_indices = torch.where(valid_mask, index, torch.zeros_like(index))
    
    # Zero out source values for -1 positions
    # This ensures masked positions contribute 0 to the scatter_add
    safe_src = torch.where(valid_mask, src, torch.zeros_like(src))
    
    # Perform scatter_add: valid indices add their load, masked indices add 0
    target.scatter_add_(dim=dim, index=safe_indices, src=safe_src)


@dataclass
class EplbModelState:
    """EPLB metrics."""

    physical_to_logical_map: torch.Tensor
    """
    Mapping from physical experts to logical experts.

    Shape: (num_moe_layers, num_physical_experts)

    # Example

    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the mapping could look like this:

    ```
    [[0, 1, 2, 3, 0, 1],
     [0, 2, 0, 1, 0, 3]]
    ```
    """
    logical_to_physical_map: torch.Tensor
    """
    Mapping from logical experts to physical experts.

    This is a sparse matrix, where -1 indicates no mapping.

    Shape: (num_moe_layers, num_logical_experts, num_redundant_experts + 1)
    where num_redundant_experts + 1 is the maximum possible replicas per logical expert

    # Example

    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the mapping could look like this:

    ```
    [[[0, 4, -1],
      [1, 5, -1],
      [2, -1, -1],
      [3, -1, -1]],
     [[0, 2, 4],
      [3, -1, -1],
      [1, -1, -1],
      [5, -1, -1]]]
    ```
    """
    logical_replica_count: torch.Tensor
    """
    Number of replicas for each logical expert.
    This is exactly the non-`-1` count in the `logical_to_physical_map`.

    Shape: (num_moe_layers, num_logical_experts)

    # Example
    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the count could look like this:

    ```
    [[2, 2, 1, 1],
     [3, 1, 1, 1]]
    """

    expert_load_pass: torch.Tensor
    """
    Expert load during this forward pass. 
    We use the token count each expert processes as the load.

    Shape: (num_moe_layers, num_physical_experts)
    """
    expert_load_window: torch.Tensor
    """
    A sliding window of expert load.

    Shape: (window_size, num_moe_layers, num_physical_experts)

    NOTE: The expert_load_view now records load for all physical experts
    rather than just local experts. This ensures consistent load statistics
    across different dispatch methods (naive all-to-all, DeepEP, pplx-kernels).
    The recorded load will be multiplied by dp_size when using naive all-to-all
    due to each DP rank contributing the same token set to the calculation.
    See:
    https://github.com/vllm-project/vllm/pull/22167#pullrequestreview-3086143856
    """
    model_name: str
    model: MixtureOfExperts
    
    expert_latency_window: torch.Tensor | None = None
    """
    Sliding window of per-expert latency (in milliseconds) for this model.
    Shape: (window_size, num_moe_layers, num_physical_experts)
    Value: latency if expert was active, 0.0 if inactive.
    Only allocated if health_check_enabled=True.
    """
    
    expert_health_mask: torch.Tensor | None = None
    """
    Boolean mask indicating which experts are healthy for this model.
    Shape: (num_moe_layers, num_physical_experts)
    True = healthy, False = unhealthy.
    Only allocated if health_check_enabled=True.
    """
    
    moe_layer_instances: list = field(default_factory=list)
    """
    Direct references to this model's MoE layer instances.
    Used for calling measure_and_update_latency().
    """


class EplbState:
    """
    EplbState of each expert parallel model. Key is the model config hash.
    """

    def __init__(self, parallel_config: ParallelConfig, device: torch.device):
        self.parallel_config = parallel_config
        self.device = device
        self.model_states: dict[str, EplbModelState] = {}
        """
        Current step in the sliding window.

        Different from `expert_rearrangement_step`, 
        each EP rank may have its own `expert_load_window_step`.
        """
        self.expert_load_window_step: int = 0
        """
        Size of the expert load sliding window.
        This is a constant and is taken from the config.
        """
        self.expert_load_window_size: int = 0
        """
        Steps after last rearrangement.
        Will trigger a rearrangement if it exceeds the threshold.

        NOTE: Keep in mind that all EP ranks need to have the same
        `expert_rearrangement_step` value to ensure synchronization.
        Otherwise, the rearrangement will hang at collective
        communication calls.
        """
        self.expert_rearrangement_step: int = 0
        """
        Interval for expert rearrangement steps.
        This is a constant and is taken from the config.
        """
        self.expert_rearrangement_step_interval: int = 0
        
        # Health monitoring and timeout-based masking (global across all models)
        """
        Global step counter that never resets. Used for mask_out_gpu_after feature.
        Incremented on every step() call.
        """
        self.global_step: int = 0
        
        """Absolute timeout threshold in milliseconds from EPLBConfig."""
        self.health_timeout_threshold: float = 100.0
        
        """
        List of step counts after which to mask out each GPU rank.
        For example, [100, 200] will mask out rank 0 after 100 steps
        and rank 1 after 200 steps. Empty list means no GPUs will be masked out.
        From EPLBConfig, used for testing fault tolerance.
        """
        self.mask_out_gpu_after: list[int] = []
        
        """
        Set of ranks that have been masked out via mask_out_gpu_after.
        Used to track which ranks are already masked to avoid re-masking.
        """
        self.masked_ranks: set[int] = set()

    @staticmethod
    def build_initial_global_physical_to_logical_map(
        num_routed_experts: int,
        num_redundant_experts: int,
    ) -> Sequence[int]:
        """
        Build an initial expert arrangement using the following structure:
        [original routed experts, redundant experts]

        Returns:
            physical_to_logical_map (Sequence[int]): A list of integers,
                where each integer is the index of the logical expert
                that the corresponding physical expert maps to.
        """
        global_physical_to_logical_map = list(range(num_routed_experts))
        global_physical_to_logical_map += [
            i % num_routed_experts for i in range(num_redundant_experts)
        ]
        return global_physical_to_logical_map

    def validate_ep_configuration(self, new_model: MixtureOfExperts):
        """
        Validate that the expert parallel configuration of
        the new model is the same as the existing models.
        """
        if len(self.model_states) > 0:
            model = next(iter(self.model_states.values())).model
            if (
                model.num_routed_experts != new_model.num_routed_experts
                or model.num_redundant_experts != new_model.num_redundant_experts
                or model.num_physical_experts != new_model.num_physical_experts
                or model.num_logical_experts != new_model.num_logical_experts
                or model.num_expert_groups != new_model.num_expert_groups
            ):
                raise RuntimeError(
                    "Model: {} "
                    "with config {} "
                    "{} {} {} {} "
                    "mismatch with new model {} "
                    "with config {} "
                    "{} {} {} {}".format(
                        type(model),
                        model.num_routed_experts,
                        model.num_redundant_experts,
                        model.num_physical_experts,
                        model.num_logical_experts,
                        model.num_expert_groups,
                        type(new_model),
                        new_model.num_routed_experts,
                        new_model.num_redundant_experts,
                        new_model.num_physical_experts,
                        new_model.num_logical_experts,
                        new_model.num_expert_groups,
                    )
                )

    def add_model(
        self,
        model: MixtureOfExperts,
        model_config: ModelConfig,
        global_expert_load: torch.Tensor | None = None,
        old_global_expert_indices: torch.Tensor | None = None,
        rank_mapping: dict[int, int] | None = None,
    ):
        """
        Build the initial EPLB state.
        """
        self.validate_ep_configuration(model)
        physical_to_logical_map_list = (
            EplbState.build_initial_global_physical_to_logical_map(
            model.num_routed_experts,
            model.num_redundant_experts,
            )
        )
        physical_to_logical_map = torch.tensor(
            physical_to_logical_map_list,
            device=self.device,
        )
        # Assuming 8 GPUs per node, this supports up to
        # (1023 + 1) / 8 = 128 nodes for now.
        # TODO(rui): make this configurable
        MAX_EXPERT_REDUNDANCY = 1023
        assert model.num_redundant_experts <= MAX_EXPERT_REDUNDANCY, (
            f"num_redundant_experts {model.num_redundant_experts} "
            f"must be less than or equal to {MAX_EXPERT_REDUNDANCY}"
        )
        max_slots_per_logical_expert = MAX_EXPERT_REDUNDANCY + 1
        logical_to_physical_map = torch.full(
            (model.num_logical_experts, max_slots_per_logical_expert),
            -1,
            device=self.device,
        )
        logical_replica_count = torch.zeros(
            (model.num_logical_experts,),
            device=self.device,
            dtype=torch.long,
        )

        for i in range(model.num_physical_experts):
            logical_idx = physical_to_logical_map[i]
            logical_to_physical_map[logical_idx, logical_replica_count[logical_idx]] = i
            logical_replica_count[logical_idx] += 1

        # Duplicate initial mapping for all layers
        physical_to_logical_map = (
            physical_to_logical_map.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
            )
            .contiguous()
        )
        logical_to_physical_map = (
            logical_to_physical_map.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
                -1,
            )
            .contiguous()
        )
        logical_replica_count = (
            logical_replica_count.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
            )
            .contiguous()
        )

        expert_load_pass = torch.zeros(
            (model.num_moe_layers, model.num_physical_experts),
            dtype=torch.int32,
            device=self.device,
        )
        self.expert_load_window_size = self.parallel_config.eplb_config.window_size
        expert_load_window = torch.zeros(
            (
                self.expert_load_window_size,
                model.num_moe_layers,
                model.num_physical_experts,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        # Initialize health monitoring if enabled (per-model)
        eplb_config = self.parallel_config.eplb_config
        self.health_timeout_threshold = eplb_config.health_timeout_threshold
        self.mask_out_gpu_after = eplb_config.mask_out_gpu_after
        
        expert_latency_window = None
        expert_health_mask = None
        moe_layer_instances = []

        # Create shared latency view for layers to write to (like expert_load_pass)
        expert_latency_pass = None
        if eplb_config.health_check_enabled:
            expert_latency_window = torch.zeros(
                (
                    self.expert_load_window_size,
                    model.num_moe_layers,
                    model.num_physical_experts,
                ),
                dtype=torch.float32,  # Latency in milliseconds
                device=self.device,
            )
            
            expert_health_mask = torch.ones(
                (model.num_moe_layers, model.num_physical_experts),
                dtype=torch.bool,
                device=self.device,
            )
            
            expert_latency_pass = torch.zeros(
                (model.num_moe_layers, model.num_physical_experts),
                dtype=torch.float32,
                device=self.device,
            )
            
            # Collect MoE layer references for this model
            if hasattr(model, "moe_layers"):
                moe_layer_instances = list(model.moe_layers)
            
            logger.info(
                "EPLB health monitoring enabled for model %s: timeout_threshold=%.1fms",
                model_config.model,
                eplb_config.health_timeout_threshold,
            )

        # Set the initial progress of rearrangement to 3/4
        eplb_step_interval = eplb_config.step_interval
        self.expert_rearrangement_step = max(
            0, eplb_step_interval - eplb_step_interval // 4
        )
        self.expert_rearrangement_step_interval = eplb_step_interval

        if global_expert_load is not None:
            ep_group = get_ep_group().device_group
            assert global_expert_load.shape == (
                model.num_moe_layers,
                model.num_logical_experts,
            )
            assert global_expert_load.dtype == torch.int64

            num_replicas = model.num_physical_experts
            num_groups = model.num_expert_groups
            num_nodes = get_node_count()
            num_gpus = ep_group.size()

            if num_gpus % num_nodes != 0:
                num_nodes = 1
                logger.warning_once(
                    f"num_gpus % num_nodes != 0, "
                    "not using hierarchical rearrangement algorithm.\n"
                    f"{num_gpus=}, {num_nodes=}"
                )

            # Get new expert mappings
            (
                new_physical_to_logical_map,
                new_logical_to_physical_map,
                new_logical_replica_count,
            ) = rebalance_experts(
                global_expert_load,
                num_replicas,
                num_groups,
                num_nodes,
                num_gpus,
            )

            max_physical_slots = new_logical_to_physical_map.shape[-1]
            assert max_physical_slots <= logical_to_physical_map.shape[-1]
            new_logical_to_physical_map = torch.nn.functional.pad(
                new_logical_to_physical_map,
                (0, logical_to_physical_map.shape[-1] - max_physical_slots),
                value=-1,
            )
            physical_to_logical_map = new_physical_to_logical_map.to(self.device)
            logical_to_physical_map.copy_(new_logical_to_physical_map)
            logical_replica_count.copy_(new_logical_replica_count)

        model.set_eplb_state(
            expert_load_pass,
            logical_to_physical_map,
            logical_replica_count,
            expert_latency_pass,
        )
        if global_expert_load is not None:
            rearrange_expert_weights_inplace(
                old_global_expert_indices,
                new_physical_to_logical_map,
                model.expert_weights,
                ep_group,
                False,
                rank_mapping,
            )
            self.expert_rearrangement_step = 0

        # Create and store the model state with per-model health tracking
        self.model_states[model_config.compute_hash()] = EplbModelState(
            physical_to_logical_map,
            logical_to_physical_map,
            logical_replica_count,
            expert_load_pass,
            expert_load_window,
            model_config.model,
            model,
            expert_latency_window,
            expert_health_mask,
            moe_layer_instances,
        )

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        """
        Step the EPLB state.

        Args:
            is_dummy (bool): If `True`, this is a dummy step and the load
                metrics recorded in this forward pass will not count.
                Defaults to `False`.
            is_profile (bool): If `True`, perform a dummy rearrangement
                with maximum communication cost. This is used in
                `profile_run` to reserve enough memory
                for the communication buffer.
            log_stats (bool): If `True`, log the expert load metrics.

        # Stats
            The metrics are all summed up across layers.
            - `avg_tokens`: The average load across ranks.
            - `max_tokens`: The maximum load across ranks.
            - `balancedness`: The ratio of average load to maximum load.
        """

        if is_profile:
            self.rearrange(is_profile=True)
            return

        if is_dummy:
            # Do not record load metrics for dummy steps
            for eplb_model_state in self.model_states.values():
                eplb_model_state.expert_load_pass.zero_()

        if log_stats:
            # Sync the expert load pass for each model (main and drafter).
            # expert_load_pass: (num_moe_layers, num_physical_experts)
            expert_load_pass_list = self._sync_load_pass()
            ep_group = get_ep_group().device_group
            for expert_load_pass, eplb_model_state in zip(
                expert_load_pass_list, self.model_states.values()
            ):
                # num_tokens_per_rank: (num_moe_layers, num_ranks)
                num_tokens_per_rank = (
                    expert_load_pass.reshape(
                        expert_load_pass.shape[0], ep_group.size(), -1
                    )
                    .sum(dim=-1)
                    .float()
                )
                
                # Log: Per-rank expert token counts (verify masked ranks get 0)
                if self.masked_ranks and self.global_step % 10 == 0:
                    # Sum across all layers for total per-rank counts
                    rank_loads = num_tokens_per_rank.sum(dim=0)  # Shape: (num_ranks,)
                    
                    logger.info(
                        "[MASKING] Rank=%d Model=%s Step=%d: Per-rank expert token counts: %s. "
                        "Masked ranks: %s (should have 0 tokens)",
                        ep_group.rank,
                        eplb_model_state.model_name,
                        self.global_step,
                        rank_loads.tolist(),
                        sorted(self.masked_ranks),
                    )
                    
                    # Verify masked ranks have zero load
                    for masked_rank in self.masked_ranks:
                        if masked_rank < ep_group.size() and rank_loads[masked_rank] > 0:
                            logger.error(
                                "[MASKING] ERROR: Masked rank %d received %.0f tokens! "
                                "Masking is not working correctly.",
                                masked_rank,
                                rank_loads[masked_rank].item(),
                            )

                # Compute balancedness ratio:
                # for each layer:
                #   (mean load across ranks) / (max load across ranks)
                avg_tokens_tensor = num_tokens_per_rank.mean(dim=0).sum(dim=0)
                max_tokens_tensor = num_tokens_per_rank.max(dim=0).values.sum(dim=0)

                # Just to make type checker happy
                tokens_tensors: list[float] = torch.stack(
                    [avg_tokens_tensor, max_tokens_tensor]
                ).tolist()
                avg_tokens, max_tokens = tokens_tensors
                balancedness = avg_tokens / max_tokens if max_tokens > 0 else 0.0

                if ep_group.rank() == 0:
                    logger.info(
                        "EPLB step: %d for model %s: avg_tokens=%.2f, "
                        "max_tokens=%d, balancedness=%.4f",
                        self.expert_rearrangement_step,
                        eplb_model_state.model_name,
                        avg_tokens,
                        max_tokens,
                        balancedness,
                    )


        # Record expert latencies for each model
        if not is_dummy:
            for eplb_model_state in self.model_states.values():
                if eplb_model_state.expert_latency_window is not None:
                    # FIRST: Trigger deferred measurement for this model
                    # (synchronizes and updates latency views)
                    for layer in eplb_model_state.moe_layer_instances:
                        if hasattr(layer, "measure_and_update_latency"):
                            layer.measure_and_update_latency()

                    # THEN: Read the measured values for this model
                    expert_latency_pass = eplb_model_state.model.get_expert_latencies()
                    if expert_latency_pass is not None:
                        # Update health mask BEFORE overwriting window
                        self._update_health_mask(
                            eplb_model_state, expert_latency_pass, log_stats
                        )
                        eplb_model_state.expert_latency_window[
                            self.expert_load_window_step
                        ] = expert_latency_pass.clone()
                        # Reset for next pass
                        expert_latency_pass.zero_()

            # Update the expert load sliding window
            for eplb_model_state in self.model_states.values():
                eplb_model_state.expert_load_window[self.expert_load_window_step] = (
                    eplb_model_state.expert_load_pass.clone()
                )
                eplb_model_state.expert_load_pass.zero_()

            self.expert_load_window_step += 1
            if self.expert_load_window_step >= self.expert_load_window_size:
                self.expert_load_window_step = 0

        # Step the expert rearrangement step
        # Note that even if this is a dummy step, we still increment the
        # rearrangement step and perform rearrangement to ensure all ranks are
        # performing collective communication.
        self.expert_rearrangement_step += 1
        self.global_step += 1
        
        # Check if any ranks should be masked out (for fault tolerance testing)
        rank_mapping_to_apply, has_new_mask = self._check_mask_out_gpu()
        
        if self.expert_rearrangement_step >= self.expert_rearrangement_step_interval:
            self.expert_rearrangement_step = 0
            # Use health masking if we have masked ranks
            self.rearrange(
                rank_mapping=rank_mapping_to_apply, 
                is_health_masking=rank_mapping_to_apply is not None
            )
        elif has_new_mask:
            # Only trigger immediate rearrangement when NEW ranks are masked
            self.rearrange(
                rank_mapping=rank_mapping_to_apply, 
                is_health_masking=True
            )
        
        # Log: Detailed physical_to_logical_map state per rank (controlled by log_stats)
        if log_stats and self.masked_ranks and self.global_step % 100 == 0:
            if logger.isEnabledFor(DEBUG):
                import logging
                ep_group = get_ep_group().device_group
                for model_name, eplb_model_state in self.model_states.items():
                    # Log physical_to_logical_map to show -1 slots
                    num_layers = eplb_model_state.physical_to_logical_map.shape[0]
                    for layer_idx in range(num_layers):
                        p2l_map = eplb_model_state.physical_to_logical_map[layer_idx]
                        
                        # Show which physical expert slots are masked
                        masked_expert_ids = (p2l_map == -1).nonzero(as_tuple=True)[0].tolist()
                        
                        # Calculate which ranks these belong to
                        num_local_experts = len(p2l_map) // ep_group.size()
                        masked_ranks_for_layer = set()
                        for expert_id in masked_expert_ids:
                            rank = expert_id // num_local_experts
                            masked_ranks_for_layer.add(rank)
                        
                        logger.debug(
                            "[MASKING] Rank=%d Model=%s Layer=%d: "
                            "Physical experts masked: %s (belong to ranks: %s). "
                            "physical_to_logical_map shape=%s, -1 count=%d",
                            ep_group.rank,
                            model_name, layer_idx,
                            masked_expert_ids,
                            sorted(masked_ranks_for_layer),
                            p2l_map.shape,
                            len(masked_expert_ids),
                        )

    def _check_mask_out_gpu(self) -> tuple[dict[int, int] | None, bool]:
        """
        Check if any GPU ranks should be masked out based on mask_out_gpu_after config.
        
        This is used for testing fault tolerance by simulating GPU failures/timeouts at specific steps.
        
        The mask_out_gpu_after list uses indices as rank IDs and values as global step counts:
        - mask_out_gpu_after[0] = 100 means rank 0 will be masked at global step 100
        - mask_out_gpu_after[1] = 200 means rank 1 will be masked at global step 200
        
        Returns:
            (rank_mapping, has_new_mask):
            - rank_mapping: dict if any ranks are currently masked, None otherwise
              Example: {0: 0, 1: -1, 2: 1, 3: 2} means rank 1 is masked
            - has_new_mask: True if new ranks were masked in this step
        """
        if not self.mask_out_gpu_after:
            return None, False
        
        ep_group = get_ep_group().device_group
        ep_size = ep_group.size()
        
        # Check if any ranks should be masked at this global step
        newly_masked_ranks = set()
        for rank_idx, mask_step in enumerate(self.mask_out_gpu_after):
            if rank_idx >= ep_size:
                break  # Invalid rank index
            if (self.global_step >= mask_step and 
                rank_idx not in self.masked_ranks):
                # Mask out this rank starting from this step
                newly_masked_ranks.add(rank_idx)
                self.masked_ranks.add(rank_idx)
                logger.warning(
                    "Simulating timeout: Masking out GPU rank %d at global step %d "
                    "(configured via mask_out_gpu_after[%d]=%d)",
                    rank_idx, self.global_step, rank_idx, mask_step
                )
        
        if not self.masked_ranks:
            return None, False
        
        # Build rank_mapping with all currently masked ranks
        rank_mapping = {}
        active_rank = 0
        for rank_idx in range(ep_size):
            if rank_idx in self.masked_ranks:
                rank_mapping[rank_idx] = -1  # Masked (simulating timeout/failure)
            else:
                rank_mapping[rank_idx] = active_rank
                active_rank += 1
        
        return rank_mapping, len(newly_masked_ranks) > 0

    def _update_health_mask(
        self,
        eplb_model_state: EplbModelState,
        current_latency: torch.Tensor,
        log_stats: bool = False,
    ) -> None:
        """
        Update expert health mask based on absolute timeout threshold for a specific model.

        Strategy: Check if current latency exceeds the absolute timeout threshold.
        If an expert exceeds the timeout, it is immediately masked out.
        0 latency means expert was inactive.

        Args:
            eplb_model_state: The model state containing health mask to update.
            current_latency: Current per-expert latency (before adding to window).
                Shape: (num_moe_layers, num_physical_experts)
            log_stats: Whether to log health status changes.
        """
        if eplb_model_state.expert_latency_window is None:
            return
        if eplb_model_state.expert_health_mask is None:
            return

        # Expert is unhealthy if:
        # 1. Currently active (current_latency > 0)
        # 2. Exceeds absolute timeout threshold
        is_unhealthy = (current_latency > 0) & (
            current_latency > self.health_timeout_threshold
        )

        new_health_mask = ~is_unhealthy

        if log_stats:
            newly_unhealthy = is_unhealthy & eplb_model_state.expert_health_mask
            if newly_unhealthy.any():
                unhealthy_coords = newly_unhealthy.nonzero(as_tuple=False)
                for layer_idx, expert_idx in unhealthy_coords:
                    current_lat = current_latency[layer_idx, expert_idx].item()
                    logger.warning(
                        "Expert [layer=%d, expert=%d] unhealthy: "
                        "current=%.3fms > timeout_threshold=%.1fms",
                        layer_idx,
                        expert_idx,
                        current_lat,
                        self.health_timeout_threshold,
                    )

            total_unhealthy = (~new_health_mask).sum().item()
            total_experts = new_health_mask.numel()
            if total_unhealthy > 0:
                logger.info(
                    "EPLB health summary: %d/%d experts unhealthy (%.1f%%)",
                    total_unhealthy,
                    total_experts,
                    100.0 * total_unhealthy / total_experts,
                )

        # Immediately mask out unhealthy experts for this model
        self._mask_unhealthy_experts(eplb_model_state, new_health_mask)

        eplb_model_state.expert_health_mask = new_health_mask

    def _mask_unhealthy_experts(
        self, eplb_model_state: EplbModelState, health_mask: torch.Tensor
    ) -> None:
        """
        Immediately mask out unhealthy experts from the logical_to_physical_map
        and update logical_replica_count for a specific model.

        This ensures that unhealthy experts are not routed to, without waiting
        for the next rearrangement interval.

        Args:
            eplb_model_state: The model state containing mappings to update.
            health_mask: Boolean tensor indicating which experts are healthy.
                Shape: (num_moe_layers, num_physical_experts)
        """
        # For each logical expert, remove unhealthy physical experts from the mapping
        num_layers, num_physical_experts = health_mask.shape
        num_logical_experts = eplb_model_state.logical_to_physical_map.shape[1]

        for layer_idx in range(num_layers):
            for logical_idx in range(num_logical_experts):
                # Get all physical experts mapped to this logical expert
                physical_experts = eplb_model_state.logical_to_physical_map[
                    layer_idx, logical_idx
                ]
                max_slots = eplb_model_state.logical_to_physical_map.shape[2]

                # Filter healthy experts and pad with -1
                healthy_experts = [
                    p
                    for p in physical_experts.tolist()
                    if p >= 0 and health_mask[layer_idx, p]
                ]
                new_mapping = healthy_experts + [-1] * (
                    max_slots - len(healthy_experts)
                )

                eplb_model_state.logical_to_physical_map[
                    layer_idx, logical_idx
                ] = torch.tensor(
                    new_mapping,
                    dtype=eplb_model_state.logical_to_physical_map.dtype,
                    device=eplb_model_state.logical_to_physical_map.device,
                )
                eplb_model_state.logical_replica_count[layer_idx, logical_idx] = len(
                    healthy_experts
                )

    def rearrange(
        self,
        is_profile: bool = False,
        execute_shuffle: bool = True,
        global_expert_loads: list[torch.Tensor] | None = None,
        rank_mapping: dict[int, int] | None = None,
        is_health_masking: bool = False,
    ) -> torch.Tensor | None:
        """
        Rearrange the experts according to the current load for all models.
        
        Iterates over all models in self.model_states and rearranges their expert
        weights based on load statistics.
        
        Args:
            is_profile: If True, perform dummy communication to reserve buffers without
                       actual weight movement. Used during profiling/warmup.
            execute_shuffle: If True, actually perform the weight shuffle. If False,
                           only compute and broadcast the new expert mappings.
            global_expert_loads: List of pre-computed global expert load statistics, one per model.
                               Each tensor has shape: [num_layers, num_logical_experts].
                               If None, will be computed from local statistics for each model.
            rank_mapping: Mapping from physical rank to new rank for elastic scaling/masking.
                        - None: Normal periodic rearrangement (no scaling or masking)
                        - len == ep_size: Scale-down or health-based masking
                          {0:0, 1:1, 2:-1, 3:2} means rank 2 is masked/removed
                        - len < ep_size: Scale-up (only maps old ranks)
            is_health_masking: If True, this is health-based masking (not elastic scaling).
                             For masking: maintain full ep_size tensors with -1 for masked ranks.
                             For scale-down: shrink tensors to active rank count.
                             Only meaningful when rank_mapping is provided.
                             
        Returns:
            Global expert load window if execute_shuffle=False, otherwise None.
        """

        ep_group = get_ep_group().device_group
        ep_rank = ep_group.rank()

        time_start = None
        is_main_rank = ep_rank == 0
        if is_main_rank:
            torch.cuda.synchronize()
            time_start = time.perf_counter()
            logger.info("Rearranging experts %s...", "(profile)" if is_profile else "")

        if global_expert_loads is None:
            # Map the physical expert load to global logical experts
            global_expert_load_windows = []
            if not execute_shuffle:
                num_models = torch.tensor(
                    [len(self.model_states)], dtype=torch.int32, device="cpu"
                )
                torch.distributed.broadcast(
                    num_models, group=get_ep_group().cpu_group, group_src=0
                )

            for eplb_model_state in self.model_states.values():
                logical_expert_load_window = torch.zeros(
                    self.expert_load_window_size,
                    eplb_model_state.model.num_moe_layers,
                    eplb_model_state.model.num_logical_experts,
                    dtype=eplb_model_state.expert_load_window.dtype,
                    device=eplb_model_state.expert_load_window.device,
                )
                
                # Aggregate physical expert loads to logical expert loads
                # Filter out -1 indices (masked experts) to prevent CUDA errors
                safe_scatter_add_with_masked_indices(
                    target=logical_expert_load_window,
                    dim=-1,
                    index=eplb_model_state.physical_to_logical_map.unsqueeze(0)
                    .expand_as(eplb_model_state.expert_load_window)
                    .long(),
                    src=eplb_model_state.expert_load_window,
                )

                if not execute_shuffle:
                    metadata = torch.tensor(
                        [
                            eplb_model_state.model.num_moe_layers,
                            eplb_model_state.model.num_logical_experts,
                            eplb_model_state.physical_to_logical_map.shape[1],
                        ],
                        dtype=torch.int32,
                        device="cpu",
                    )
                    torch.distributed.broadcast(
                        metadata, group=get_ep_group().cpu_group, group_src=0
                    )

                global_expert_load_window = logical_expert_load_window.sum(dim=0)
                global_expert_load_windows.append(global_expert_load_window)
            # Perform all-reduce to get the expert load across all ranks for each model
            global_expert_load_windows = self._allreduce_list(
                global_expert_load_windows
            )
            if not execute_shuffle:
                for eplb_model_state, global_expert_load_window in zip(
                    self.model_states.values(), global_expert_load_windows
                ):
                    # (num_moe_layers, old_num_physical_experts)
                    old_global_expert_indices = eplb_model_state.physical_to_logical_map
                    torch.distributed.broadcast(
                        old_global_expert_indices, group=ep_group, group_src=0
                    )
            if not execute_shuffle:
                return global_expert_load_windows
        else:
            assert execute_shuffle
            global_expert_load_windows = global_expert_loads

        # Rearrange experts for each model
        for eplb_model_state, global_expert_load_window in zip(
            self.model_states.values(), global_expert_load_windows
        ):
            # Get new expert mappings for the model
            num_replicas = eplb_model_state.model.num_physical_experts
            num_groups = eplb_model_state.model.num_expert_groups
            
            # Handle rank_mapping scenarios
            if rank_mapping is not None and len(rank_mapping) == ep_group.size():
                # rank_mapping provided with full ep_size
                # This could be:
                # 1. Health-based masking (is_health_masking=True): Keep full ep_size tensors
                # 2. Actual scale-down (is_health_masking=False): Shrink tensors
                
                # NOTE(yongji): scale down, we need to rebalance the experts on
                # remaining GPUs, transfer the experts while we haven't shutdown
                # the GPUs to be released.
                cpu_group = get_ep_group().cpu_group
                num_nodes = _node_count_with_rank_mapping(cpu_group, rank_mapping)
                num_gpus = sum(new_rank != -1 for new_rank in rank_mapping.values())
                
                if not is_health_masking:
                    # Actual elastic scale-down: shrink replicas
                    num_replicas = (
                        num_replicas // ep_group.size() * num_gpus
                    )  # handle num replicas change
                # else: Health-based masking - keep full num_replicas (ep_size * experts_per_rank)
            else:
                num_nodes = get_node_count()
                num_gpus = ep_group.size()

            if num_gpus % num_nodes != 0:
                self.num_nodes = 1
                logger.warning_once(
                    f"num_gpus % num_nodes != 0, "
                    "not using hierarchical rearrangement algorithm.\n"
                    f"{num_gpus=}, {num_nodes=}"
                )

            # Get new expert mappings
            if is_health_masking:
                # Health-based masking: use wrapper that maintains full ep_size
                (
                    new_physical_to_logical_map,
                    new_logical_to_physical_map,
                    new_logical_replica_count,
                ) = rebalance_masked_experts(
                    global_expert_load_window,
                    num_replicas,
                    num_groups,
                    num_nodes,
                    ep_group.size(),
                    rank_mapping,
                )
            else:
                # Normal or scale-down: standard rebalance
                (
                    new_physical_to_logical_map,
                    new_logical_to_physical_map,
                    new_logical_replica_count,
                ) = rebalance_experts(
                    global_expert_load_window,
                    num_replicas,
                    num_groups,
                    num_nodes,
                    num_gpus,
                )

            # Update expert weights
            rearrange_expert_weights_inplace(
                eplb_model_state.physical_to_logical_map,
                new_physical_to_logical_map,
                eplb_model_state.model.expert_weights,
                ep_group,
                is_profile,
                rank_mapping,
            )

            if not is_profile:
                if (
                    eplb_model_state.physical_to_logical_map.shape[1]
                    != new_physical_to_logical_map.shape[1]
                ):
                    # Only resize for actual scale events, not for health-based masking
                    if not is_health_masking:
                        eplb_model_state.physical_to_logical_map = new_physical_to_logical_map.to(
                            eplb_model_state.physical_to_logical_map.device
                        )
                    # For health-based masking, just update in place (same size)
                    else:
                        eplb_model_state.physical_to_logical_map.copy_(new_physical_to_logical_map)
                else:
                    eplb_model_state.physical_to_logical_map.copy_(new_physical_to_logical_map)
                max_physical_slots = new_logical_to_physical_map.shape[-1]
                assert max_physical_slots <= eplb_model_state.logical_to_physical_map.shape[-1]
                new_logical_to_physical_map = torch.nn.functional.pad(
                    new_logical_to_physical_map,
                    (0, eplb_model_state.logical_to_physical_map.shape[-1] - max_physical_slots),
                    value=-1,
                )
                eplb_model_state.logical_to_physical_map.copy_(new_logical_to_physical_map)
                eplb_model_state.logical_replica_count.copy_(new_logical_replica_count)

        if is_main_rank:
            assert time_start is not None
            torch.cuda.synchronize()
            time_end = time.perf_counter()
            
            # Log additional info for health-based masking
            if is_health_masking and rank_mapping is not None:
                masked_ranks = [
                    rank for rank, new_rank in rank_mapping.items() 
                    if new_rank == -1
                ]
                active_ranks = [
                    rank for rank, new_rank in rank_mapping.items() 
                    if new_rank != -1
                ]
                
                # Count masked experts across all models
                total_masked_experts = 0
                for eplb_model_state in self.model_states.values():
                    # Count -1 values in physical_to_logical_map
                    masked_count = (eplb_model_state.physical_to_logical_map == -1).sum().item()
                    total_masked_experts += masked_count
                
                logger.info(
                    "Successfully masked GPU rank(s) %s. "
                    "Active ranks: %s. "
                    "Masked %d expert slots. "
                    "Rearranged experts in %.2f seconds.",
                    masked_ranks,
                    active_ranks,
                    total_masked_experts,
                    time_end - time_start,
                )
            else:
                logger.info(
                    "Rearranged experts%sin %.2f seconds.",
                    " (profile) " if is_profile else " ",
                    time_end - time_start,
                )
        return None

    @staticmethod
    def recv_state() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Receive the expert load and old placement from the master rank.
        """
        ep_group = get_ep_group()
        num_models = torch.empty(1, dtype=torch.int32, device="cpu")
        torch.distributed.broadcast(num_models, group=ep_group.cpu_group, group_src=0)
        num_models = num_models.item()
        global_expert_loads = []
        old_global_expert_indices_per_model = []
        for _ in range(num_models):
            metadata = torch.empty(3, dtype=torch.int32, device="cpu")
            torch.distributed.broadcast(metadata, group=ep_group.cpu_group, group_src=0)
            num_moe_layers, num_logical_experts, num_old_physical_experts = (
                metadata.tolist()
            )
            global_expert_load = torch.zeros(
                (num_moe_layers, num_logical_experts),
                dtype=torch.int64,
                device=ep_group.device,
            )
            all_reduce(global_expert_load, group=ep_group.device_group)
            old_global_expert_indices = torch.empty(
                (num_moe_layers, num_old_physical_experts),
                dtype=torch.int64,
                device=ep_group.device,
            )
            torch.distributed.broadcast(
                old_global_expert_indices,
                group=ep_group.device_group,
                group_src=0,
            )
            global_expert_loads.append(global_expert_load)
            old_global_expert_indices_per_model.append(old_global_expert_indices)
        return global_expert_loads, old_global_expert_indices_per_model

    @classmethod
    def get_eep_state(
        cls, parallel_config: ParallelConfig
    ) -> tuple[
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
        dict[int, int] | None,
    ]:
        num_local_physical_experts = torch.empty(1, dtype=torch.int32, device="cpu")
        torch.distributed.broadcast(
            num_local_physical_experts,
            group=get_ep_group().cpu_group,
            group_src=0,
        )
        num_local_physical_experts = int(num_local_physical_experts.item())
        new_ep_size = get_ep_group().world_size
        global_expert_loads, old_global_expert_indices_per_model = (
            EplbState.recv_state()
        )

        # EP configuration for all models has to be the same so as eplb config
        num_logical_experts = global_expert_loads[0].shape[1]
        parallel_config.eplb_config.num_redundant_experts = (
            num_local_physical_experts * new_ep_size - num_logical_experts
        )
        assert (
            old_global_expert_indices_per_model[0].shape[1] % num_local_physical_experts
            == 0
        )
        old_ep_size = (
            old_global_expert_indices_per_model[0].shape[1]
            // num_local_physical_experts
        )
        rank_mapping = {old_ep_rank: old_ep_rank for old_ep_rank in range(old_ep_size)}
        return (
            global_expert_loads,
            old_global_expert_indices_per_model,
            rank_mapping,
        )

    def _allreduce_list(self, tensor_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        All-reduce a list of tensors.
        """
        if len(tensor_list) == 1:
            all_reduce(tensor_list[0], group=get_ep_group().device_group)
            return tensor_list
        assert all(t.dim() == 2 for t in tensor_list), "All tensors must be 2D."
        assert all(t.shape[1] == tensor_list[0].shape[1] for t in tensor_list), (
            "All tensors must have the same shape[1]."
        )
        # Concatenate, all_reduce, then unpack to original shapes.
        # We assume all tensors are 2D and shape[1] (num_physical_experts)
        # is the same across all models.
        shapes = [t.shape for t in tensor_list]
        concat_tensor = torch.cat(tensor_list, dim=0)

        ep_group = get_ep_group().device_group
        all_reduce(concat_tensor, group=ep_group)

        all_reduce_list = []
        offset = 0
        for shape in shapes:
            all_reduce_list.append(concat_tensor[offset : offset + shape[0], :])
            offset += shape[0]
        return all_reduce_list

    def _sync_load_pass(self) -> list[torch.Tensor]:
        """
        Sync the expert load pass across all ranks for log stats.
        Doesn't update the expert load pass in eplb_model_state.
        """
        load_pass_list = []
        for eplb_model_state in self.model_states.values():
            load_pass_list.append(eplb_model_state.expert_load_pass.clone())
        return self._allreduce_list(load_pass_list)


def _node_count_with_rank_mapping(
    pg: ProcessGroup | StatelessProcessGroup,
    rank_mapping: dict[int, int],
) -> int:
    if isinstance(pg, ProcessGroup):
        world_size = torch.distributed.get_world_size(group=pg)
    else:
        world_size = pg.world_size

    if world_size == 1:
        return 1

    # Build node assignment map
    node_assignment = [0] * world_size  # rank -> node_id
    next_node_id = 0

    for current_rank in range(world_size):
        if node_assignment[current_rank] != 0:
            continue  # Already assigned to a node

        assert current_rank in rank_mapping
        if rank_mapping[current_rank] == -1:
            continue  # Pending shutdown

        # Assign current rank to a new node
        next_node_id += 1
        node_assignment[current_rank] = next_node_id

        # Find all ranks on the same node as current_rank
        same_node_flags = in_the_same_node_as(pg, current_rank)
        for other_rank, is_same_node in enumerate(same_node_flags):
            if is_same_node and node_assignment[other_rank] == 0:
                node_assignment[other_rank] = next_node_id

    return next_node_id
