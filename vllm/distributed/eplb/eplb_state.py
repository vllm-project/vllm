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

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup, all_reduce

from vllm.config import ModelConfig, ParallelConfig
from vllm.distributed.parallel_state import (
    get_ep_group,
    get_node_count,
    in_the_same_node_as,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MixtureOfExperts

from .async_worker import start_async_worker
from .eplb_utils import EplbEvent
from .policy import EPLB_POLICIES, AbstractEplbPolicy, DefaultEplbPolicy
from .rebalance_execute import (
    AsyncEplbLayerResult,
    move_from_buffer,
    rearrange_expert_weights_inplace,
)

logger = init_logger(__name__)


@dataclass
class EplbStats:
    """
    Model stats used in EPLB rebalancing algorithm.
    """

    global_expert_load_window: torch.Tensor
    """
    Experts load window.
    Shape: (window_size, num_moe_layers, num_physical_experts)
    """
    num_replicas: int
    """
    Number of physical experts.
    """
    num_groups: int
    """
    Number of expert groups.
    """
    num_nodes: int
    """
    Number of nodes.
    """
    num_gpus: int
    """
    Number of GPUs.
    """


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
    across different dispatch methods (naive all-to-all, DeepEP).
    The recorded load will be multiplied by dp_size when using naive all-to-all
    due to each DP rank contributing the same token set to the calculation.
    See:
    https://github.com/vllm-project/vllm/pull/22167#pullrequestreview-3086143856
    """
    model_name: str
    model: MixtureOfExperts
    expert_buffer: list[torch.Tensor]
    """
    The buffer to store the expert weights during transfer.
    """
    rebalanced: bool
    """
    The flag indicates whether the experts rebalance have been computed.
    """
    eplb_stats: EplbStats | None
    """
    EPLB stats for the model.
    """
    cuda_device_index: int | None
    """
    CUDA device index for the async EPLB worker thread.
    """
    pending_result: AsyncEplbLayerResult | None = None
    """
    Set by the async worker after all writes to expert_buffer are done. Consumed
    and reset to None by the main thread in move_to_workspace() after the contents of
    expert_buffer have been transferred out. At most one result is pending at a time.
    """


class EplbState:
    """
    EplbState of each expert parallel model. Key is the model config hash.
    """

    def __init__(self, parallel_config: ParallelConfig, device: torch.device):
        self.parallel_config = parallel_config
        self.device = device
        self.model_states: dict[str, EplbModelState] = {}
        self.policy: type[AbstractEplbPolicy] = DefaultEplbPolicy
        """
        Selected EPLB algorithm class
        """
        self.expert_load_window_step: int = 0
        """
        Current step in the sliding window.

        Different from `expert_rearrangement_step`, 
        each EP rank may have its own `expert_load_window_step`.
        """
        self.expert_load_window_size: int = 0
        """
        Size of the expert load sliding window.
        This is a constant and is taken from the config.
        """
        self.expert_rearrangement_step: int = 0
        """
        Steps after last rearrangement.
        Will trigger a rearrangement if it exceeds the threshold.

        NOTE: Keep in mind that all EP ranks need to have the same
        `expert_rearrangement_step` value to ensure synchronization.
        Otherwise, the rearrangement will hang at collective
        communication calls.
        """
        self.expert_rearrangement_step_interval: int = 0
        """
        Interval for expert rearrangement steps.
        This is a constant and is taken from the config.
        """
        self.is_async: bool = False
        """
        The flag indicates whether the EPLB is running in async mode.
        """
        self.rearrange_event: EplbEvent = EplbEvent()
        """
        Event to signal when a new rearrangement is needed for the async thread.
        """
        self.async_worker: threading.Thread | None = None
        """
        Background thread handling async transfers.
        """
        self.cuda_device_index: int | None = None
        """
        CUDA device index for the async EPLB worker thread.
        """
        self.num_valid_physical_experts: int = 0
        """
        Number of valid physical experts.
        This is the number of physical experts that are
        actually mapped to logical experts. In elastic EP,
        newly started EP ranks may not have physical experts
        mapped yet.
        """
        if self.device.type == "cuda":
            self.cuda_device_index = self.device.index
            if self.cuda_device_index is None and torch.cuda.is_available():
                self.cuda_device_index = torch.accelerator.current_device_index()

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
    ):
        """
        Build the initial EPLB state.
        """
        self.validate_ep_configuration(model)
        self.is_async = self.parallel_config.eplb_config.use_async

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

        # Set the initial progress of rearrangement to 3/4
        eplb_step_interval = self.parallel_config.eplb_config.step_interval
        self.expert_rearrangement_step = max(
            0, eplb_step_interval - eplb_step_interval // 4
        )
        self.expert_rearrangement_step_interval = eplb_step_interval

        policy_type = self.parallel_config.eplb_config.policy
        self.policy = EPLB_POLICIES[policy_type]
        logger.debug("Selected EPLB policy: %s", policy_type)

        model.set_eplb_state(
            expert_load_pass,
            logical_to_physical_map,
            logical_replica_count,
        )

        expert_buffer = [torch.empty_like(w) for w in model.expert_weights[0]]

        model_state = EplbModelState(
            physical_to_logical_map=physical_to_logical_map,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            expert_load_pass=expert_load_pass,
            expert_load_window=expert_load_window,
            model_name=model_config.model,
            model=model,
            expert_buffer=expert_buffer,
            rebalanced=False,
            eplb_stats=None,
            cuda_device_index=self.cuda_device_index,
        )
        self.model_states[model_config.compute_hash()] = model_state
        self.num_valid_physical_experts = model.num_physical_experts

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
        ep_group = get_ep_group().device_group
        if is_profile:
            self.rearrange(is_profile=True)
            return

        if is_dummy:
            # Do not record load metrics for dummy steps
            for eplb_model_state in self.model_states.values():
                eplb_model_state.expert_load_pass.zero_()

        if (
            log_stats
            and self.expert_rearrangement_step
            % self.parallel_config.eplb_config.log_balancedness_interval
            == 0
        ):
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
                        "max_tokens=%d, balancedness=%.4f, "
                        "steps until the next rearrangement: %d",
                        self.expert_rearrangement_step,
                        eplb_model_state.model_name,
                        avg_tokens,
                        max_tokens,
                        balancedness,
                        self.expert_rearrangement_step_interval
                        - self.expert_rearrangement_step,
                    )

        # Update the expert load sliding window
        if not is_dummy:
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

        if self.is_async:
            for eplb_model_state in self.model_states.values():
                if (
                    eplb_model_state.pending_result is not None
                    and self._all_ranks_result_ready(eplb_model_state)
                ):
                    self.move_to_workspace(
                        model_state=eplb_model_state,
                        ep_group=ep_group,
                    )

        if self.expert_rearrangement_step >= self.expert_rearrangement_step_interval:
            if self.is_async and any(
                eplb_model_state.rebalanced
                for eplb_model_state in self.model_states.values()
            ):
                # Still performing asynchronous rearrangement
                return
            self.expert_rearrangement_step = 0
            self.rearrange()

    def rearrange(
        self,
        is_profile: bool = False,
        rank_mapping: dict[int, int] | None = None,
    ) -> torch.Tensor | None:
        """
        Rearrange the experts according to the current load.

        Args:
            is_profile (bool): If `True`, perform a dummy rearrangement.
                This is used in `profile_run` to reserve enough memory,
                no memory movement will be performed. Default is False.
            rank_mapping (dict[int, int] | None): The rank mapping
                when scaling is done in EEP.
        """

        ep_group = get_ep_group().device_group
        ep_rank = ep_group.rank()

        start_event = None
        end_event = None
        is_main_rank = ep_rank == 0
        if is_main_rank:
            if not self.is_async or is_profile:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            logger.info(
                "Rearranging experts %s %s...",
                "(async mode)" if self.is_async else "sync mode",
                "(profile)" if is_profile else "",
            )

        # Map the physical expert load to global logical experts
        global_expert_load_windows = []
        for eplb_model_state in self.model_states.values():
            expert_load_window = eplb_model_state.expert_load_window[
                :, :, : self.num_valid_physical_experts
            ]
            logical_expert_load_window = torch.zeros(
                self.expert_load_window_size,
                eplb_model_state.model.num_moe_layers,
                eplb_model_state.model.num_logical_experts,
                dtype=eplb_model_state.expert_load_window.dtype,
                device=eplb_model_state.expert_load_window.device,
            )
            logical_expert_load_window.scatter_add_(
                dim=-1,
                index=eplb_model_state.physical_to_logical_map[
                    :, : self.num_valid_physical_experts
                ]
                .unsqueeze(0)
                .expand_as(expert_load_window)
                .long(),
                src=expert_load_window,
            )

            global_expert_load_window = logical_expert_load_window.sum(dim=0)
            global_expert_load_windows.append(global_expert_load_window)
        # Perform all-reduce to get the expert load across all ranks for each model
        global_expert_load_windows = self._allreduce_list(global_expert_load_windows)

        # TODO(bowen): Treat differently for prefill and decode nodes
        eplb_model_state = next(iter(self.model_states.values()))
        model = eplb_model_state.model
        num_replicas = model.num_physical_experts
        num_groups = model.num_expert_groups

        if rank_mapping is not None and len(rank_mapping) == ep_group.size():
            # NOTE(yongji): scale down, we need to rebalance the experts on
            # remaining GPUs, transfer the experts while we haven't shutdown
            # the GPUs to be released.
            coordinator = get_ep_group()
            assert isinstance(coordinator, StatelessGroupCoordinator)
            tcp_store_group = coordinator.tcp_store_group
            num_nodes = _node_count_with_rank_mapping(tcp_store_group, rank_mapping)
            num_gpus = sum(new_rank != -1 for new_rank in rank_mapping.values())
            num_replicas = (
                num_replicas // ep_group.size() * num_gpus
            )  # handle num replicas change
        else:
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
        for eplb_model_state, global_expert_load_window in zip(
            self.model_states.values(), global_expert_load_windows
        ):
            if not self.is_async or is_profile:
                # Get new expert mappings for the model
                new_physical_to_logical_map = self.policy.rebalance_experts(
                    global_expert_load_window.cpu(),
                    num_replicas,
                    num_groups,
                    num_nodes,
                    num_gpus,
                    eplb_model_state.physical_to_logical_map.cpu(),
                )

                num_logical_experts = global_expert_load_window.shape[-1]
                (new_logical_to_physical_map, new_logical_replica_count) = (
                    compute_logical_maps(
                        new_physical_to_logical_map, num_logical_experts
                    )
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
                        eplb_model_state.physical_to_logical_map = (
                            new_physical_to_logical_map.to(
                                eplb_model_state.physical_to_logical_map.device
                            )
                        )
                    else:
                        eplb_model_state.physical_to_logical_map.copy_(
                            new_physical_to_logical_map
                        )
                    max_physical_slots = new_logical_to_physical_map.shape[-1]
                    assert (
                        max_physical_slots
                        <= eplb_model_state.logical_to_physical_map.shape[-1]
                    )
                    new_logical_to_physical_map = torch.nn.functional.pad(
                        new_logical_to_physical_map,
                        (
                            0,
                            eplb_model_state.logical_to_physical_map.shape[-1]
                            - max_physical_slots,
                        ),
                        value=-1,
                    )
                    eplb_model_state.logical_to_physical_map.copy_(
                        new_logical_to_physical_map
                    )
                    eplb_model_state.logical_replica_count.copy_(
                        new_logical_replica_count
                    )
                if is_main_rank:
                    assert start_event is not None
                    assert end_event is not None
                    end_event.record()
                    end_event.synchronize()
                    gpu_elapsed = start_event.elapsed_time(end_event) / 1000.0
                    logger.info(
                        "Rearranged experts %s in %.2f s.",
                        " (profile) " if is_profile else " ",
                        gpu_elapsed,
                    )
            else:
                eplb_model_state.eplb_stats = EplbStats(
                    # We copy the tensor to snapshot the global_expert_load_window
                    # on the main thread so that async worker can access it safely
                    # while the main thread is running.
                    global_expert_load_window=global_expert_load_window.clone(),
                    num_replicas=num_replicas,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                    num_gpus=num_gpus,
                )
                eplb_model_state.rebalanced = True
        # Signal async thread to start transferring layers
        if self.is_async and (not is_profile):
            self.rearrange_event.record()
        return None

    def start_async_loop(
        self,
        rank_mapping: dict[int, int] | None = None,
        is_profile: bool = False,
    ):
        if not self.is_async:
            return
        if self.async_worker is None:
            self.async_worker = start_async_worker(
                self,
                is_profile=is_profile,
            )

    def _update_layer_mapping_from_new(
        self, model_state: EplbModelState, result: AsyncEplbLayerResult
    ) -> None:
        layer = result.layer_idx

        new_physical = result.new_physical_to_logical_map
        target_device = model_state.physical_to_logical_map.device
        # If the number of physical experts has changed, then the new map needs to
        # be copied synchronously to avoid a race condition with the async worker
        if model_state.physical_to_logical_map.shape[1] != new_physical.shape[1]:
            model_state.physical_to_logical_map = new_physical.to(target_device)
        else:
            model_state.physical_to_logical_map[layer].copy_(
                new_physical[layer].to(target_device, non_blocking=True)
            )

        num_logical_experts = model_state.logical_to_physical_map.shape[1]
        new_logical, new_replica_count = compute_logical_maps(
            new_physical[layer], num_logical_experts
        )

        logical_device = model_state.logical_to_physical_map.device
        max_slots = model_state.logical_to_physical_map.shape[-1]
        slot_delta = max_slots - new_logical.shape[-1]
        if slot_delta > 0:
            new_logical = torch.nn.functional.pad(
                new_logical, (0, slot_delta), value=-1
            )
        model_state.logical_to_physical_map[layer].copy_(new_logical.to(logical_device))

        replica_device = model_state.logical_replica_count.device
        model_state.logical_replica_count[layer].copy_(
            new_replica_count.to(replica_device)
        )

    def _all_ranks_result_ready(self, model_state: EplbModelState) -> bool:
        parallel_state = get_ep_group()
        has_result = int(model_state.pending_result is not None)

        cpu_group = getattr(parallel_state, "cpu_group", None)
        if cpu_group is not None and cpu_group.size() > 1:
            flag = torch.tensor((has_result,), dtype=torch.int32, device="cpu")
            all_reduce(flag, group=cpu_group)
            return int(flag.item()) == cpu_group.size()

        device_group = parallel_state.device_group
        if device_group.size() <= 1:
            return bool(has_result)

        device = getattr(
            parallel_state, "device", model_state.physical_to_logical_map.device
        )
        flag = torch.tensor((has_result,), dtype=torch.int32, device=device)
        all_reduce(flag, group=device_group)
        return int(flag.item()) == device_group.size()

    def move_to_workspace(
        self,
        model_state: EplbModelState,
        ep_group: ProcessGroup,
    ) -> None:
        result = model_state.pending_result
        assert result is not None
        move_from_buffer(
            expert_weights=model_state.model.expert_weights[result.layer_idx],
            expert_weights_buffers=model_state.expert_buffer,
            is_unchanged=result.is_unchanged,
            is_received_locally=result.is_received_locally,
            recv_metadata=result.recv_metadata,
            new_indices=result.new_physical_to_logical_map.numpy(),
            ep_rank=ep_group.rank(),
        )

        self._update_layer_mapping_from_new(model_state, result)
        logger.debug(
            "model %s successfully move_to_workspace layer %d",
            model_state.model_name,
            result.layer_idx,
        )
        if result.layer_idx == model_state.model.num_moe_layers - 1:
            model_state.rebalanced = False
            logger.info(
                "finish async transfer for model %s rank %d",
                model_state.model_name,
                ep_group.rank(),
            )

        # Reset pending_result before ublocking the async worker
        model_state.pending_result = None
        result.consumed_event.record()

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

    @classmethod
    def from_mapping(
        cls,
        model: MixtureOfExperts,
        model_config: ModelConfig,
        device: torch.device,
        parallel_config: ParallelConfig,
        expanded_physical_to_logical: torch.Tensor,
        num_valid_physical_experts: int,
    ) -> "EplbState":
        eplb_state = cls(
            parallel_config=parallel_config,
            device=device,
        )
        eplb_state.add_model(
            model=model,
            model_config=model_config,
        )
        eplb_state.num_valid_physical_experts = num_valid_physical_experts
        eplb_model_state = eplb_state.model_states[model_config.compute_hash()]
        eplb_model_state.physical_to_logical_map.copy_(expanded_physical_to_logical)

        (logical_to_physical_map_cpu, logical_replica_count_cpu) = compute_logical_maps(
            expanded_physical_to_logical.cpu(), model.num_logical_experts
        )

        max_num_replicas = eplb_model_state.logical_to_physical_map.shape[-1]
        num_replicas = logical_to_physical_map_cpu.shape[-1]
        logical_to_physical_map = torch.nn.functional.pad(
            logical_to_physical_map_cpu,
            (
                0,
                max_num_replicas - num_replicas,
            ),
            value=-1,
        ).to(device)
        logical_replica_count = logical_replica_count_cpu.to(device)

        eplb_model_state.logical_to_physical_map.copy_(logical_to_physical_map)
        eplb_model_state.logical_replica_count.copy_(logical_replica_count)

        return eplb_state


@dataclass
class EplbLayerState:
    """Runtime EPLB data stored in the MoE layer."""

    expert_load_view: torch.Tensor | None = None
    logical_to_physical_map: torch.Tensor | None = None
    logical_replica_count: torch.Tensor | None = None


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


def compute_logical_maps(
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Derive logical_to_physical_map and logical_replica_count from
    physical_to_logical_map.

    Args:
        physical_to_logical_map: [num_layers, num_physical_experts], logical
            expert index for each physical expert slot
        num_logical_experts: total number of logical experts

    Returns:
        logical_to_physical_map: [num_layers, num_logical_experts, max_replicas],
            physical slots per logical expert; -1 where unused
        logical_replica_count: [num_layers, num_logical_experts], number of
            physical replicas per logical expert
    """
    device = physical_to_logical_map.device
    assert physical_to_logical_map.device.type == "cpu"

    dtype = physical_to_logical_map.dtype

    # If computing maps for a single layer, unsqueeze a single element layer dimension
    per_layer = physical_to_logical_map.dim() == 1
    physical_to_logical_map_view = physical_to_logical_map
    if per_layer:
        physical_to_logical_map_view = physical_to_logical_map.unsqueeze(0)
    assert len(physical_to_logical_map_view.shape) == 2
    num_layers, num_physical = physical_to_logical_map_view.shape

    valid_mask = physical_to_logical_map_view >= 0
    logical_replica_count = torch.zeros(
        num_layers,
        num_logical_experts,
        dtype=dtype,
        device=device,
    )
    logical_replica_count.scatter_add_(
        1,
        physical_to_logical_map_view.clamp(min=0),
        valid_mask.to(dtype),
    )

    max_replicas = int(logical_replica_count.max().item())
    logical_to_physical_map_out = torch.full(
        (num_layers, num_logical_experts, max_replicas),
        -1,
        dtype=dtype,
        device=device,
    )

    running_count = torch.zeros_like(logical_replica_count)
    layer_indices = torch.arange(num_layers, device=device)
    for phys_idx in range(num_physical):
        # Logical expert at physical slot phys_idx for each layer
        logical_expert_ids = physical_to_logical_map_view[:, phys_idx]  # [num_layers]

        # Scale up will set the logical expert ids to -1 for all new physical experts.
        # Only consider "valid" experts when setting up the logical_to_physical map.
        valid_expert_mask = logical_expert_ids >= 0
        if not valid_expert_mask.any():
            continue
        valid_layers = layer_indices[valid_expert_mask]
        valid_experts = logical_expert_ids[valid_expert_mask]

        # Use the current running count as the replica index, then increment it.
        replica_idx = running_count[valid_layers, valid_experts]
        logical_to_physical_map_out[valid_layers, valid_experts, replica_idx] = phys_idx
        running_count[valid_layers, valid_experts] += 1

    # If computing maps for a single layer, squeeze out the extra layer dimension
    # before returning
    if per_layer:
        return logical_to_physical_map_out.squeeze(0), logical_replica_count.squeeze(0)
    return logical_to_physical_map_out, logical_replica_count
