# SPDX-License-Identifier: Apache-2.0
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

For example: DeepSeek-R1 has 256 logical experts, so each MoE layer
has 256 sets of linear layer weights in the model parameters. If we add 32
redundant experts, DeepSeek-R1 will have 256 + 32 = 288 physical experts in
total. And when deploying, we'll have 288 sets of linear layer weights for each
MoE layer. If we have 32 EP ranks, then each GPU will hold 288 / 32 = 9 local
physical experts.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.distributed import all_gather, all_reduce

from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_ep_group, get_node_count
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MixtureOfExperts

from .rebalance_algo import rebalance_experts
from .rebalance_execute import rearrange_expert_weights_inplace

logger = init_logger(__name__)


@dataclass
class EplbState:
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

    Shape: (num_moe_layers, num_local_physical_experts)
    """
    expert_load_window: torch.Tensor
    """
    A sliding window of expert load.

    Shape: (window_size, num_moe_layers, num_local_physical_experts)
    """
    expert_load_window_step: int = 0
    """
    Current step in the sliding window.

    Different from `expert_rearrangement_step`, each EP rank may have its own
    `expert_load_window_step`.
    """
    expert_load_window_size: int = 0
    """
    Size of the expert load sliding window.
    This is a constant and is taken from the config.
    """

    expert_rearrangement_step: int = 0
    """
    Steps after last rearrangement.
    Will trigger a rearrangement if it exceeds the threshold.

    NOTE: Keep in mind that all EP ranks need to have the same
    `expert_rearrangement_step` value to ensure synchronization.
    Otherwise, the rearrangement will hang at collective
    communication calls.
    """
    expert_rearrangement_step_interval: int = 0
    """
    Interval for expert rearrangement steps.
    This is a constant and is taken from the config.
    """

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

    @classmethod
    def build(
        cls,
        model: MixtureOfExperts,
        device: torch.device,
        parallel_config: ParallelConfig,
    ) -> "EplbState":
        """
        Build the initial EPLB state.
        """
        physical_to_logical_map_list = (
            cls.build_initial_global_physical_to_logical_map(
                model.num_routed_experts,
                model.num_redundant_experts,
            ))
        physical_to_logical_map = torch.tensor(
            physical_to_logical_map_list,
            device=device,
        )
        logical_to_physical_map = torch.full(
            (model.num_logical_experts, model.num_redundant_experts + 1),
            -1,
            device=device,
        )
        logical_replica_count = torch.zeros(
            (model.num_logical_experts, ),
            device=device,
            dtype=torch.long,
        )

        for i in range(model.num_physical_experts):
            logical_idx = physical_to_logical_map[i]
            logical_to_physical_map[logical_idx,
                                    logical_replica_count[logical_idx]] = i
            logical_replica_count[logical_idx] += 1

        # Duplicate initial mapping for all layers
        physical_to_logical_map = physical_to_logical_map.unsqueeze(0).expand(
            model.num_moe_layers,
            -1,
        ).contiguous()
        logical_to_physical_map = logical_to_physical_map.unsqueeze(0).expand(
            model.num_moe_layers,
            -1,
            -1,
        ).contiguous()
        logical_replica_count = logical_replica_count.unsqueeze(0).expand(
            model.num_moe_layers,
            -1,
        ).contiguous()

        expert_load_pass = torch.zeros(
            (model.num_moe_layers, model.num_local_physical_experts),
            dtype=torch.int32,
            device=device,
        )
        expert_load_window_size = parallel_config.eplb_window_size
        expert_load_window = torch.zeros(
            (expert_load_window_size, model.num_moe_layers,
             model.num_local_physical_experts),
            dtype=torch.int32,
            device=device,
        )

        # Set the initial progress of rearrangement to 3/4
        eplb_step_interval = parallel_config.eplb_step_interval
        expert_rearrangement_step = max(
            0, eplb_step_interval - eplb_step_interval // 4)

        model.set_eplb_state(
            expert_load_pass,
            logical_to_physical_map,
            logical_replica_count,
        )

        return cls(
            physical_to_logical_map,
            logical_to_physical_map,
            logical_replica_count,
            expert_load_pass,
            expert_load_window,
            expert_load_window_size=expert_load_window_size,
            expert_rearrangement_step=expert_rearrangement_step,
            expert_rearrangement_step_interval=eplb_step_interval,
        )

    def step(self,
             model: MixtureOfExperts,
             is_dummy: bool = False,
             is_profile: bool = False,
             log_stats: bool = False) -> None:
        """
        Step the EPLB state.

        Args:
            model (MixtureOfExperts): The MoE model.
            is_dummy (bool): If `True`, this is a dummy step and the load
              metrics recorded in this forward pass will not count. Defaults
              to `False`.
            is_profile (bool): If `True`, perform a dummy rearrangement
              with maximum communication cost. This is used in `profile_run`
              to reserve enough memory for the communication buffer.
            log_stats (bool): If `True`, log the expert load metrics.

        # Stats
            The metrics are all summed up across layers.
            - `avg_tokens`: The average load across ranks.
            - `max_tokens`: The maximum load across ranks.
            - `balancedness`: The ratio of average load to maximum load.
        """

        if is_profile:
            self.rearrange(model, is_profile=True)
            return

        if is_dummy:
            # Do not record load metrics for dummy steps
            self.expert_load_pass.zero_()

        if log_stats:
            # `num_tokens`: (num_moe_layers,)
            num_tokens = self.expert_load_pass.sum(dim=-1)

            # Collect load metrics from all ranks
            ep_group = get_ep_group().device_group
            num_tokens_list = [
                torch.empty_like(num_tokens) for _ in range(ep_group.size())
            ]
            all_gather(num_tokens_list, num_tokens, group=ep_group)
            # Stack to get (num_ranks, num_moe_layers)
            num_tokens_per_rank = torch.stack(num_tokens_list).float()

            # Compute balancedness ratio:
            # for each layer:
            #   (mean load across ranks) / (max load across ranks)
            avg_tokens_tensor = num_tokens_per_rank.mean(dim=0).sum(dim=0)
            max_tokens_tensor = num_tokens_per_rank.max(dim=0).values.sum(
                dim=0)

            # Just to make type checker happy
            tokens_tensors: list[float] = torch.stack(
                [avg_tokens_tensor, max_tokens_tensor]).tolist()
            avg_tokens, max_tokens = tokens_tensors
            balancedness = avg_tokens / max_tokens if max_tokens > 0 else 0.0

            if ep_group.rank() == 0:
                logger.info(
                    "EPLB step: avg_tokens=%.2f, max_tokens=%d, "
                    "balancedness=%.4f", avg_tokens, max_tokens, balancedness)

        # Update the expert load sliding window
        if not is_dummy:
            self.expert_load_window[self.expert_load_window_step] = (
                self.expert_load_pass.clone())
            self.expert_load_window_step += 1
            if self.expert_load_window_step >= self.expert_load_window_size:
                self.expert_load_window_step = 0
            self.expert_load_pass.zero_()

        # Step the expert rearrangement step
        # Note that even if this is a dummy step, we still increment the
        # rearrangement step and perform rearrangement to ensure all ranks are
        # performing collective communication.
        self.expert_rearrangement_step += 1
        if (self.expert_rearrangement_step
                >= self.expert_rearrangement_step_interval):
            self.expert_rearrangement_step = 0
            self.rearrange(model)

    def rearrange(self,
                  model: MixtureOfExperts,
                  is_profile: bool = False) -> None:
        """
        Rearrange the experts according to the current load.
        """

        ep_group = get_ep_group().device_group
        ep_rank = ep_group.rank()

        time_start = None
        is_main_rank = ep_rank == 0
        if is_main_rank:
            torch.cuda.synchronize()
            time_start = time.perf_counter()
            logger.info("Rearranging experts %s...",
                        "(profile)" if is_profile else "")

        # This mapping is only used here, so we do not store it in the state
        physical_expert_start = ep_rank * model.num_local_physical_experts
        physical_expert_end = (physical_expert_start +
                               model.num_local_physical_experts)
        # (num_moe_layers, num_local_physical_experts)
        local_physical_to_logical_map = self.physical_to_logical_map[
            :,
            physical_expert_start:physical_expert_end,
        ]

        # Map the local physical expert load to global logical experts
        logical_expert_load_window = torch.zeros(
            self.expert_load_window_size,
            model.num_moe_layers,
            model.num_logical_experts,
            dtype=self.expert_load_window.dtype,
            device=self.expert_load_window.device,
        )
        logical_expert_load_window.scatter_add_(
            dim=-1,
            index=local_physical_to_logical_map.unsqueeze(0).expand_as(
                self.expert_load_window).long(),
            src=self.expert_load_window,
        )

        # Perform all-reduce to get the expert load across all ranks
        global_expert_load_window = logical_expert_load_window.sum(dim=0)
        all_reduce(global_expert_load_window, group=ep_group)

        # TODO(bowen): Treat differently for prefill and decode nodes
        num_replicas = model.num_physical_experts
        num_groups = model.num_expert_groups
        num_nodes = get_node_count()
        num_gpus = ep_group.size()

        if num_gpus % num_nodes != 0:
            logger.warning_once(
                f"num_gpus % num_nodes != 0, "
                "not using hierarchical rearrangement algorithm.\n"
                f"{num_gpus=}, {num_nodes=}")

        # Get new expert mappings
        (
            new_physical_to_logical_map,
            new_logical_to_physical_map,
            new_logical_replica_count,
        ) = (rebalance_experts(
            global_expert_load_window,
            num_replicas,
            num_groups,
            num_nodes,
            num_gpus,
        ))

        # Update expert weights
        rearrange_expert_weights_inplace(
            self.physical_to_logical_map,
            new_physical_to_logical_map,
            model.expert_weights,
            ep_group,
            is_profile,
        )

        if not is_profile:
            self.physical_to_logical_map.copy_(new_physical_to_logical_map)
            self.logical_to_physical_map.copy_(new_logical_to_physical_map)
            self.logical_replica_count.copy_(new_logical_replica_count)

        if is_main_rank:
            assert time_start is not None
            torch.cuda.synchronize()
            time_end = time.perf_counter()
            logger.info(
                "Rearranged experts%sin %.2f seconds.",
                " (profile) " if is_profile else " ",
                time_end - time_start,
            )
