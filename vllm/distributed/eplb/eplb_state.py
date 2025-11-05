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

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy
import torch
from torch.distributed import ProcessGroup, all_reduce

from vllm.config import ParallelConfig
from vllm.config.parallel import ExpertMapperPolicy
from vllm.distributed.parallel_state import (
    get_ep_group,
    get_node_count,
    in_the_same_node_as,
)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MixtureOfExperts

from .eplb_process import EPLBProcess
from .rebalance_algo import rebalance_experts
from .rebalance_execute import idx_global_to_local, rearrange_expert_weights_inplace

logger = init_logger(__name__)


@dataclass
class RebalanceTaskArgs:
    global_expert_load_window: torch.Tensor
    num_replicas: int
    num_groups: int
    num_nodes: int
    num_gpus: int


@dataclass
class ExpertMapperArgs:
    num_moe_layers: int
    policy_type: ExpertMapperPolicy
    phyhsical_to_logical_map: torch.Tensor


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

    num_wait_worker_iterations: int = 0
    """
    Number of iterations to wait before applying a redistribution plan.
    """

    _async_processor: EPLBProcess | None = None
    """
    Asynchronous process manager.
    """

    enable_async: bool = False
    """
    A gate to trigger asynchronous expert load rebalancing and weight transfer.
    """

    expert_mapper_args: ExpertMapperArgs | None = None
    """
    Arguments passed to expert mapper strategy
    """

    rebalance_task_args: RebalanceTaskArgs | None = None
    """"
    Arguments passed to expert rebalancing task
    """

    _gather_buffer: torch.Tensor | None = None
    """
    Tensor used as apreallocated workspace for torch.distributed.all_gather_into_tensor.
    It stores each ranks's local expert load to avoid reallocation and to enable
    efficient global aggregation.
    """

    cur_layer_id: int = -1
    """
    Records the current moe layer being precessed for expert weight transfer.
    """

    num_moe_layers: int = 0
    """
    Number of Mixture-of-Experts (MoE) layers in the model.
    """

    buffer_tensor_list: list[list[torch.Tensor]] = field(default_factory=list)
    """
    A two-dimensional list storing temporary tensor buffers for expert weight
    communication. Each inner list corresponds to a layer, and each element represents
    a buffer tensor used during transfer or update operations.
    """

    recv_expert_list: list[tuple[int, int]] = field(default_factory=list)
    """
    A list of tuples representing the experts to be received. Each tuple contains
    (local_expert_id, buffer_tensor_id) indicating where the received weights should
    be placed.
    """

    comm_op_list: list[Any] = field(default_factory=list)
    """
    A list storing pending communication operations (send/receive tasks)
    for expert weight transfer.
    """

    reqs: list[Any] = field(default_factory=list)
    """
    A list to which the communication requests will be appended.
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
        global_expert_load: torch.Tensor | None = None,
        old_global_expert_indices: torch.Tensor | None = None,
        rank_mapping: dict[int, int] | None = None,
    ) -> "EplbState":
        """
        Build the initial EPLB state.
        """
        physical_to_logical_map_list = cls.build_initial_global_physical_to_logical_map(
            model.num_routed_experts,
            model.num_redundant_experts,
        )
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
            f"must be less than or equal to {MAX_EXPERT_REDUNDANCY}"
        )
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
            device=device,
        )
        expert_load_window_size = parallel_config.eplb_config.window_size
        expert_load_window = torch.zeros(
            (expert_load_window_size, model.num_moe_layers, model.num_physical_experts),
            dtype=torch.int32,
            device=device,
        )

        # Set the initial progress of rearrangement to 3/4
        eplb_step_interval = parallel_config.eplb_config.step_interval
        expert_rearrangement_step = max(0, eplb_step_interval - eplb_step_interval // 4)

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
            physical_to_logical_map = new_physical_to_logical_map.to(device)
            logical_to_physical_map.copy_(new_logical_to_physical_map)
            logical_replica_count.copy_(new_logical_replica_count)

        model.set_eplb_state(
            expert_load_pass,
            logical_to_physical_map,
            logical_replica_count,
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
            expert_rearrangement_step = 0
        expert_mapper_args = ExpertMapperArgs(
            model.num_moe_layers,
            parallel_config.eplb_config.expert_mapper_policy,
            None,
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
            num_wait_worker_iterations=(
                parallel_config.eplb_config.num_wait_worker_iterations
            ),
            enable_async=parallel_config.eplb_config.enable_async,
            expert_mapper_args=expert_mapper_args,
            num_moe_layers=model.num_moe_layers,
        )

    def __post_init__(self) -> None:
        # Initialize asynchronous process manager
        if self.enable_async:
            self._async_processor = EPLBProcess(target_func=rebalance_experts)

    def step(
        self,
        model: MixtureOfExperts,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        """
        Step the EPLB state.

        Args:
            model (MixtureOfExperts): The MoE model.
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
            self.rearrange(model, is_profile=True)
            return

        if is_dummy:
            # Do not record load metrics for dummy steps
            self.expert_load_pass.zero_()

        if log_stats:
            # total_expert_load_pass: (num_moe_layers, num_physical_experts)
            total_expert_load_pass = self.expert_load_pass.clone()

            # Collect load metrics from all ranks
            ep_group = get_ep_group().device_group
            all_reduce(total_expert_load_pass, group=ep_group)

            # num_tokens_per_rank: (num_moe_layers, num_ranks)
            num_tokens_per_rank = (
                total_expert_load_pass.reshape(
                    total_expert_load_pass.shape[0], ep_group.size(), -1
                )
                .sum(dim=-1)
                .float()
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
                    "EPLB step: avg_tokens=%.2f, max_tokens=%d, balancedness=%.4f",
                    avg_tokens,
                    max_tokens,
                    balancedness,
                )

        # Update the expert load sliding window
        if not is_dummy:
            self.expert_load_window[self.expert_load_window_step] = (
                self.expert_load_pass.clone()
            )
            self.expert_load_window_step += 1
            if self.expert_load_window_step >= self.expert_load_window_size:
                self.expert_load_window_step = 0
            self.expert_load_pass.zero_()

        if not self.enable_async:
            # Step the expert rearrangement step
            # Note that even if this is a dummy step, we still increment the
            # rearrangement step and perform rearrangement to ensure all ranks are
            # performing collective communication.
            self.expert_rearrangement_step += 1
            if (
                self.expert_rearrangement_step
                >= self.expert_rearrangement_step_interval
            ):
                self.expert_rearrangement_step = 0
                self.rearrange(model)
        else:
            if self.rebalance_expert_weight_flag():
                global_expert_load = self.compute_and_set_moe_load()
                ep_group = get_ep_group().device_group
                num_replicas, num_groups, num_nodes, num_gpus = (
                    self.prepare_rebalance_env(model=model, ep_group=ep_group)
                )
                self.rebalance_task_args = RebalanceTaskArgs(
                    global_expert_load.cpu(),
                    num_replicas,
                    num_groups,
                    num_nodes,
                    num_gpus,
                )
                input_args = self.rebalance_task_args

                assert self.expert_mapper_args is not None, (
                    "expert_mapper_args is not initialized"
                )
                self.expert_mapper_args.phyhsical_to_logical_map = (
                    self.physical_to_logical_map.cpu()
                )
                expert_mapper_args = self.expert_mapper_args

                self.rebalance_task(input_args, expert_mapper_args)

            if self.update_expert_map_and_weight_flag():
                for req in self.reqs:
                    req.wait()
                if self.comm_op_list is not None:
                    self.comm_op_list.clear()
                phy2log_map, log2phy_map = self._require_maps()

                if self.physical_to_logical_map.shape[1] != phy2log_map.shape[1]:
                    self.physical_to_logical_map[self.cur_layer_id] = phy2log_map[
                        self.cur_layer_id
                    ].to(self.physical_to_logical_map.device)
                else:
                    self.physical_to_logical_map[self.cur_layer_id].copy_(
                        phy2log_map[self.cur_layer_id]
                    )
                max_physical_slots = log2phy_map.shape[-1]
                assert max_physical_slots <= self.logical_to_physical_map.shape[-1]
                padded = torch.nn.functional.pad(
                    log2phy_map,
                    (0, self.logical_to_physical_map.shape[-1] - max_physical_slots),
                    value=-1,
                )
                self.logical_to_physical_map[self.cur_layer_id].copy_(
                    padded[self.cur_layer_id]
                )
                expert_weights = cast(
                    Sequence[Sequence[torch.Tensor]], model.expert_weights
                )
                for recv_expert_info in self.recv_expert_list:
                    local_expert_to_replace, buffer_tensor_id = recv_expert_info
                    if buffer_tensor_id >= len(self.buffer_tensor_list):
                        raise IndexError(
                            f"buffer_tensor_id {buffer_tensor_id} out of range"
                        )
                    for expert_tensor, buffer_tensor in zip(
                        expert_weights[self.cur_layer_id][local_expert_to_replace],
                        self.buffer_tensor_list[buffer_tensor_id],
                    ):
                        expert_tensor.copy_(buffer_tensor)
            logger.debug(
                "[EPLB] finished update expert weight for layer: %s", self.cur_layer_id
            )
            self.expert_rearrangement_step += 1
            if self.expert_rearrangement_step >= (
                self.expert_rearrangement_step_interval
                + self.num_wait_worker_iterations
                + self.num_moe_layers
            ):
                self.expert_rearrangement_step = 0

    def rearrange(
        self,
        model: MixtureOfExperts,
        is_profile: bool = False,
        execute_shuffle: bool = True,
        global_expert_load: torch.Tensor | None = None,
        rank_mapping: dict[int, int] | None = None,
    ) -> torch.Tensor | None:
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
            logger.info("Rearranging experts %s...", "(profile)" if is_profile else "")

        if global_expert_load is None:
            # Map the physical expert load to global logical experts
            logical_expert_load_window = torch.zeros(
                self.expert_load_window_size,
                model.num_moe_layers,
                model.num_logical_experts,
                dtype=self.expert_load_window.dtype,
                device=self.expert_load_window.device,
            )
            logical_expert_load_window.scatter_add_(
                dim=-1,
                index=self.physical_to_logical_map.unsqueeze(0)
                .expand_as(self.expert_load_window)
                .long(),
                src=self.expert_load_window,
            )

            if not execute_shuffle:
                metadata = torch.tensor(
                    [
                        model.num_moe_layers,
                        model.num_logical_experts,
                        self.physical_to_logical_map.shape[1],
                    ],
                    dtype=torch.int32,
                    device="cpu",
                )
                torch.distributed.broadcast(
                    metadata, group=get_ep_group().cpu_group, group_src=0
                )

            # Perform all-reduce to get the expert load across all ranks
            global_expert_load_window = logical_expert_load_window.sum(dim=0)
            all_reduce(global_expert_load_window, group=ep_group)

            if not execute_shuffle:
                # (num_moe_layers, old_num_physical_experts)
                old_global_expert_indices = self.physical_to_logical_map
                torch.distributed.broadcast(
                    old_global_expert_indices, group=ep_group, group_src=0
                )
                return global_expert_load_window
        else:
            assert execute_shuffle
            global_expert_load_window = global_expert_load

        num_replicas, num_groups, num_nodes, num_gpus = self.prepare_rebalance_env(
            model=model, ep_group=ep_group
        )

        # Get new expert mappings
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
            self.physical_to_logical_map,
            new_physical_to_logical_map,
            model.expert_weights,
            ep_group,
            is_profile,
            rank_mapping,
        )

        if not is_profile:
            if (
                self.physical_to_logical_map.shape[1]
                != new_physical_to_logical_map.shape[1]
            ):
                self.physical_to_logical_map = new_physical_to_logical_map.to(
                    self.physical_to_logical_map.device
                )
            else:
                self.physical_to_logical_map.copy_(new_physical_to_logical_map)
            max_physical_slots = new_logical_to_physical_map.shape[-1]
            assert max_physical_slots <= self.logical_to_physical_map.shape[-1]
            new_logical_to_physical_map = torch.nn.functional.pad(
                new_logical_to_physical_map,
                (0, self.logical_to_physical_map.shape[-1] - max_physical_slots),
                value=-1,
            )
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
        return None

    @staticmethod
    def recv_state() -> tuple[torch.Tensor, torch.Tensor]:
        """
        Receive the expert load and old placement from the master rank.
        """
        ep_group = get_ep_group()
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
            old_global_expert_indices, group=ep_group.device_group, group_src=0
        )

        return global_expert_load, old_global_expert_indices

    def prepare_rebalance_env(
        self,
        model: MixtureOfExperts,
        ep_group,
        rank_mapping: dict[int, int] | None = None,
    ) -> tuple[int, int, int, int]:
        """
        Compute effective (num_replicas, num_groups, num_nodes, num_gpus) for
        expert rebalancing under the current EP topology and optional rank_mapping.

        Returns:
            (num_replicas, num_groups, num_nodes, num_gpus)
        """
        # TODO(bowen): Treat differently for prefill and decode nodes
        num_replicas = model.num_physical_experts
        num_groups = model.num_expert_groups
        if rank_mapping is not None and len(rank_mapping) == ep_group.size():
            # NOTE(yongji): scale down, we need to rebalance the experts on
            # remaining GPUs, transfer the experts while we haven't shutdown
            # the GPUs to be released.
            cpu_group = get_ep_group().cpu_group
            num_nodes = _node_count_with_rank_mapping(cpu_group, rank_mapping)
            num_gpus = sum(new_rank != -1 for new_rank in rank_mapping.values())
            num_replicas = (
                num_replicas // ep_group.size() * num_gpus
            )  # handle num replicas change
        else:
            num_nodes = get_node_count()
            num_gpus = ep_group.size()

        if num_gpus % num_nodes != 0:
            self.num_nodes = 1
            logger.warning_once(
                "num_gpus % num_nodes != 0, "
                "not using hierarchical rearrangement algorithm.\n"
                "num_gpus=%s, num_nodes=%s",
                num_gpus,
                num_nodes,
            )

        return num_replicas, num_groups, num_nodes, num_gpus

    def step_before_forward(self, model) -> None:
        """
        Executes operations before the model's forward pass.
        If the EPLB process indicates it should process (e.g., a rearrangement
        is pending), it initiates asynchronous shuffling for each MoE layer.
        """
        if not self._async_processor:
            logger.error("Async processor is not initialized")
            return None
        if self.update_expert_map_and_weight_flag():
           self._async_processor.step()
        process_result = self._async_processor.result
        if process_result is None:
            logger.warning("Async processor result is None, skipping layer.")
            return None

        if isinstance(process_result, list) and len(process_result) > 0:
            result = process_result.pop(0)
            if not result:
                raise ValueError("Results is empty, cannot retrieve element")
            (
                expert_send_info,
                expert_recv_info,
                updated_phy2log_map,
                _,
                layer_id,
            ) = result
            if layer_id < 0 or layer_id >= self.num_moe_layers:
                raise ValueError(f"Index {layer_id} out of range for moe layers")
            updated_phy2log_map_this_rank = torch.from_numpy(
                numpy.array(updated_phy2log_map)
            )
            num_dense_layers = len(model.model.layers) - self.num_moe_layers
            self.generate_expert_d2d_transfer_task(
                model.expert_weights,
                expert_send_info,
                expert_recv_info,
                updated_phy2log_map_this_rank,
                layer_id + num_dense_layers,
            )
            self.reqs.clear()
            self.async_expert_weight_transfer()

    def generate_expert_d2d_transfer_task(
        self,
        expert_weights: Sequence[Sequence[torch.Tensor]],
        expert_send_info: Sequence[tuple[int, int]],
        expert_recv_info: Sequence[tuple[int, int]],
        updated_expert_map: torch.Tensor,
        layer_id: int,
    ) -> None:
        """
        Generates the expert data-to-data transfer tasks (send and receive operations)
        for a given layer based on the provided send/receive information and the new
        expert map.

        Args:
            expert_send_info: List of (destination_rank, global_expert_id) for experts
                              to send.
            expert_recv_info: List of (source_rank, global_expert_id) for experts to
                              receive.
            updated_expert_map: New expert map for the layer (global->local).
            layer_id: The ID of the MoE layer.
        """
        num_local_physical_experts = next(iter(expert_weights[0])).shape[0]
        if not (expert_send_info or expert_recv_info):
            return
        self.cur_layer_id = layer_id
        self.comm_op_list = []
        self.prepare_send(
            expert_weights, expert_send_info, num_local_physical_experts, layer_id
        )
        self.prepare_recv(expert_recv_info, updated_expert_map)

    def async_expert_weight_transfer(self) -> None:
        """
        Initiates the asynchronous expert weight transfer by executing the
        prepared P2P communication operations.
        """
        # set asynchronous stream for d2d expert weight transfer
        if self.comm_op_list:
            ret_list = torch.distributed.batch_isend_irecv(self.comm_op_list)
            self.reqs.extend(ret_list)
        logger.info("Expert weights start to transfer")

    def prepare_send(
        self,
        expert_weights: Sequence[Sequence[torch.Tensor]],
        expert_send_info: Sequence[tuple[int, int]],
        num_local_experts: int,
        layer_id: int,
    ) -> None:
        for dst_rank, global_expert_id_to_send in expert_send_info:
            local_expert_id_to_send = idx_global_to_local(
                global_expert_id_to_send,
                num_local_experts,
                get_ep_group().device_group.rank(),
            )
            for src_tensor in expert_weights[layer_id][local_expert_id_to_send]:
                self.comm_op_list.append(
                    torch.distributed.P2POp(
                        torch.distributed.isend, src_tensor, dst_rank
                    )
                )

    def prepare_recv(
        self,
        expert_recv_info: Sequence[Sequence[torch.Tensor]],
        updated_expert_map: torch.Tensor,
    ) -> None:
        for buffer_tensor_id, (recv_rank, global_expert_id_to_recv) in enumerate(
            expert_recv_info
        ):
            for buffer_tensor in self.buffer_tensor_list[buffer_tensor_id]:
                self.comm_op_list.append(
                    torch.distributed.P2POp(
                        torch.distributed.irecv, buffer_tensor, recv_rank
                    )
                )
            local_expert_id_to_replace = updated_expert_map[
                global_expert_id_to_recv
            ].item()
            self.recv_expert_list.append((local_expert_id_to_replace, buffer_tensor_id))

    def rebalance_expert_weight_flag(self) -> bool:
        """
        Determines whether expert rebalancing should be triggered in the current
        iteration. This typically occurs right before the expert weight update phase.

        Returns:
            True if it's time to trigger rebalancing, False otherwise.
        """
        return self.expert_rearrangement_step == (
            self.expert_rearrangement_step_interval - 1
        )

    def update_expert_map_and_weight_flag(self) -> bool:
        """
        Determines if expert maps should be updated in the current iteration. This
        is typically true for a short window after the EPLB update and worker wait.

        Returns:
            True if expert maps should be updated, False otherwise
        """
        map_update_counter = self.expert_rearrangement_step - (
            self.expert_rearrangement_step_interval + self.num_wait_worker_iterations
        )
        return 0 <= map_update_counter < self.num_moe_layers

    def compute_and_set_moe_load(self) -> torch.Tensor:
        """
        Computes the MoE load across all ranks and sets it in the shared dictionary.
        It gathers local expert load from all ranks and combines them.

        Returns:
            The gathered MoE load tenso with shape (num_moe_layers, num_logical_expert).
        """
        local_load = self.expert_load_pass.clone()

        if torch.distributed.is_initialized():
            ep_group = get_ep_group().device_group
            self.device = local_load.device
            ep_world_size = ep_group.size()

            if (
                self._gather_buffer is None
                or self._gather_buffer.shape[0] != ep_world_size
                or tuple(self._gather_buffer.shape[1:]) != tuple(local_load.shape)
            ):
                shape = (ep_world_size, *local_load.shape)
                self._gather_buffer = torch.empty(
                    shape, dtype=local_load.dtype, device=self.device
                )

            torch.distributed.all_gather_into_tensor(
                self._gather_buffer, local_load, group=ep_group
            )
            moe_load = self._gather_buffer.permute(1, 0, 2)
            L, W, E_local = moe_load.shape
            physical_view = moe_load.permute(0, 2, 1).reshape(L, W * E_local)
            physical_to_logical_map = self.physical_to_logical_map.to(
                device=self.device, dtype=torch.long
            )
            if hasattr(self, "logical_to_physical_map"):
                num_logical_experts = self.logical_to_physical_map.shape[1]
            else:
                num_logical_experts = int(physical_to_logical_map.max().item()) + 1

            global_expert_load = torch.zeros(
                L, num_logical_experts, dtype=physical_view.dtype, device=self.device
            )
            global_expert_load.scatter_add_(
                dim=1, index=physical_to_logical_map, src=physical_view
            )

            return global_expert_load
        else:
            moe_load = local_load.unsqueeze(1)  # (L, 1, E_local)

            L, _, E_local = moe_load.shape
            physical_view = moe_load.permute(0, 2, 1).reshape(L, E_local)

            physical_to_logical_map = self.physical_to_logical_map.to(
                device=moe_load.device, dtype=torch.long
            )
            if hasattr(self, "logical_to_physical_map"):
                num_logical_experts = self.logical_to_physical_map.shape[1]
            else:
                num_logical_experts = int(physical_to_logical_map.max().item()) + 1

            global_expert_load = torch.zeros(
                L,
                num_logical_experts,
                dtype=physical_view.dtype,
                device=moe_load.device,
            )
            global_expert_load.scatter_add_(
                dim=1, index=physical_to_logical_map, src=physical_view
            )

        return global_expert_load

    def rebalance_task(self, input_args, expert_mapper_args) -> None:
        # Submit task to asynchronous process
        if self._async_processor is None:
            logger.error("Async processor is not initialized, cannot submit task")
            return None

        if self._async_processor.has_pending_task:
            logger.info(
                "EPLB async process already has a pending task, skipping new submission"
            )
            return None

        if self._async_processor.is_post_processing:
            logger.info(
                "EPLB async process is pending post processing task, skipping"
                "new submission"
            )
            return None

        try:
            success = self._async_processor.submit_task(
                args=input_args,
                expert_mapper_args=expert_mapper_args,
            )
        except Exception as e:
            logger.error("Error submitting task to async process: %s", str(e))
            success = False

        if success:
            logger.info(
                "rebalance_experts task has been submitted to async process, "
                "will check results after maximum %s steps",
                str(self.num_wait_worker_iterations),
            )
        else:
            logger.error("Failed to submit rebalance task to async process")
        return None

    def __del__(self):
        """Clean up async process resources"""
        if self._async_processor:
            self._async_processor.cleanup()

    def _require_process(self) -> "EPLBProcess":
        if self._async_processor is None:
            raise RuntimeError("EPLB process not initialized")
        return self._async_processor

    def _require_maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        proc = self._require_process()
        phy2log = proc.phy2log_map
        if phy2log is None:
            raise RuntimeError("phy2log_map is None")
        log2phy = proc.log2phy_map
        if log2phy is None:
            raise RuntimeError("log2phy_map is None")
        return phy2log, log2phy


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
