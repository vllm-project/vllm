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

For example: DeepSeek-R1 has 256 logical experts, so each MoE layer
has 256 sets of linear layer weights in the model parameters. If we add 32
redundant experts, DeepSeek-R1 will have 256 + 32 = 288 physical experts in
total. And when deploying, we'll have 288 sets of linear layer weights for each
MoE layer. If we have 32 EP ranks, then each GPU will hold 288 / 32 = 9 local
physical experts.
"""
from collections.abc import Sequence
from typing import Optional

import torch

from vllm.config import ParallelConfig
from vllm.distributed.eplb.eplb_adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm.distributed.eplb.eplb_policy.abstract_policy import EplbPolicy
from vllm.distributed.parallel_state import get_ep_group, get_node_count

from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.distributed.eplb.eplb_data.eplb_data import EplbData
from vllm.distributed.eplb.eplb_loader.eplb_weight_loader import EplbWeightLoader
from vllm.distributed.eplb.eplb_updator.eplb_updator import EplbUpdator
from vllm.distributed.eplb.eplb_process.eplb_process import EplbProcess



logger = init_logger(__name__)


class EplbState:
    def __init__(self, model):
        self.eplb_data = Optional[EplbData]
        self.eplb_updator = Optional[EplbUpdator]
        self._async_processor = Optional[EplbProcess]
        self.eplb_policy = Optional[EplbPolicy]
        self.model = model
        self.eplb_adaptor = VllmEplbAdaptor(self.model)
        self.eplb_loader = EplbWeightLoader(self.eplb_adaptor)

    def __post_init__(self):
        # Initialize asynchronous process manager
        self._async_processor = EplbProcess(
            target_func=self.eplb_policy.rebalance_experts,
            num_wait_worker_iterations=self.eplb_data.num_wait_worker_iterations)

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

    def build(
            self,
            model: MixtureOfExperts,
            device: torch.device,
            parallel_config: ParallelConfig,
            global_expert_load: Optional[torch.Tensor] = None,
            old_global_expert_indices: Optional[torch.Tensor] = None,
            rank_mapping: Optional[dict[int, int]] = None,
    ) -> "EplbState":
        """
        Build the initial EPLB state.
        """
        physical_to_logical_map_list = (
            self.build_initial_global_physical_to_logical_map(
                model.num_routed_experts,
                model.num_redundant_experts,
            ))
        physical_to_logical_map = torch.tensor(
            physical_to_logical_map_list,
            device=device,
        )
        # Assuming 8 GPUs per node, this supports up to
        # (1023 + 1) / 8 = 128 nodes for now.
        # TODO(rui): make this configurable
        MAX_EXPERT_REDUNDANCY = 1023
        assert model.num_redundant_experts <= MAX_EXPERT_REDUNDANCY, (
            f"num_redundant_experts {model.num_redundant_experts} "
            f"must be less than or equal to {MAX_EXPERT_REDUNDANCY}")
        max_slots_per_logical_expert = MAX_EXPERT_REDUNDANCY + 1
        logical_to_physical_map = torch.full(
            (model.num_logical_experts, max_slots_per_logical_expert),
            -1,
            device=device,
        )
        logical_replica_count = torch.zeros(
            (model.num_logical_experts,),
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
            (model.num_moe_layers, model.num_physical_experts),
            dtype=torch.int32,
            device=device,
        )
        expert_load_window_size = parallel_config.eplb_config.window_size
        expert_load_window = torch.zeros(
            (expert_load_window_size, model.num_moe_layers,
             model.num_physical_experts),
            dtype=torch.int32,
            device=device,
        )

        # Set the initial progress of rearrangement to 3/4
        eplb_step_interval = parallel_config.eplb_config.step_interval
        expert_rearrangement_step = max(
            0, eplb_step_interval - eplb_step_interval // 4)

        if global_expert_load is not None:
            ep_group = get_ep_group().device_group
            assert global_expert_load.shape == (model.num_moe_layers,
                                                model.num_logical_experts)
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
                    f"{num_gpus=}, {num_nodes=}")

            # Get new expert mappings
            (
                new_physical_to_logical_map,
                new_logical_to_physical_map,
                new_logical_replica_count,
            ) = (self.eplb_policy.rebalance_experts(
                global_expert_load,
                num_replicas,
                num_groups,
                num_nodes,
                num_gpus,
            ))

            max_physical_slots = new_logical_to_physical_map.shape[-1]
            assert max_physical_slots <= logical_to_physical_map.shape[-1]
            new_logical_to_physical_map = torch.nn.functional.pad(
                new_logical_to_physical_map,
                (0, logical_to_physical_map.shape[-1] - max_physical_slots),
                value=-1,
            )
            physical_to_logical_map = new_physical_to_logical_map.to(device)
            logical_to_physical_map.copy_(new_logical_to_physical_map)
            logical_replica_count.copy_(new_logical_replica_count)

        model.set_eplb_state(
            expert_load_pass,
            logical_to_physical_map,
            logical_replica_count,
        )
        if global_expert_load is not None:
            self.eplb_updator.rearrange_expert_weights_inplace(
                old_global_expert_indices,
                new_physical_to_logical_map,
                model.expert_weights,
                ep_group,
                False,
                rank_mapping,
            )
            expert_rearrangement_step = 0
        self.eplb_data = EplbData(
            physical_to_logical_map=physical_to_logical_map,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            expert_load_pass=expert_load_pass,
            expert_load_window=expert_load_window,
            expert_load_window_size=expert_load_window_size,
            expert_rearrangement_step=expert_rearrangement_step,
            expert_rearrangement_step_interval=eplb_step_interval,
        )
        self.__post_init__()
        self.eplb_updator = EplbUpdator(eplb_data=self.eplb_data, eplb_loader=self.eplb_loader, eplb_process=self._async_processor, adaptor=self.eplb_adaptor)

        return self

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

        self.eplb_updator.step(model, is_dummy, is_profile, log_stats)

    def rearrange(self,
                  model: MixtureOfExperts,
                  is_profile: bool = False,
                  execute_shuffle: bool = True,
                  global_expert_load: Optional[torch.Tensor] = None,
                  rank_mapping: Optional[dict[int, int]] = None) -> None:
        self.eplb_updator.rearrange(model, is_profile, execute_shuffle, global_expert_load, rank_mapping)

    def recv_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Receives and returns the current state (e.g., expert load and capacity) from the EPLB updator.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors representing the received state.
        """
        return self.eplb_updator.recv_state()

    def forward_before(self):
        """
        Executes necessary operations or updates before the model's forward pass.
        This typically involves preparing the EPLB updator for the upcoming forward computation.
        """
        self.eplb_updator.step_before_forward()

    def forward_end(self):
        """
        Executes necessary operations or updates after the model's forward pass.
        This typically involves processing results or updating states based on the completed forward computation.
        """
        self.eplb_updator.step_after_forward()
