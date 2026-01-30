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

import numpy as np
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

from .async_worker import start_async_worker
from .policy import EPLB_POLICIES, AbstractEplbPolicy, DefaultEplbPolicy
from .rebalance_execute import (
    RecvMetadata,
    move_from_buffer,
    rearrange_expert_weights_inplace,
)

logger = init_logger(__name__)


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
    across different dispatch methods (naive all-to-all, DeepEP, pplx-kernels).
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
    buffer_lock: threading.Lock
    """
    The lock to protect the expert buffer.
    """
    buffer_ready_event: torch.cuda.Event | None
    """
    CUDA event recorded when the async worker finishes filling the buffer.
    The main thread waits on this before consuming the buffer.
    """
    buffer_consumed_event: torch.cuda.Event | None
    """
    CUDA event recorded after the main thread finishes consuming the buffer.
    The async worker waits on this before writing to the buffer again.
    """
    ep_buffer_ready: int
    """
    The flag indicates whether the expert buffer is ready for transfer.
    0 or 1.
    """
    layer_to_transfer: int
    """
    The layer index to transfer in async mode.
    """
    rebalanced: bool
    """
    The flag indicates whether the experts rebalance have been computed.
    """
    pending_global_ready_check: bool
    """
    Whether the async EPLB needs to poll peers for buffer readiness.
    """
    is_unchanged: np.ndarray
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    The size is same as the num of physical experts in the current layer.
    """
    is_received_locally: np.ndarray
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    The size is same as the num of physical experts in the current layer.
    """
    recv_metadata: RecvMetadata
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    """
    cuda_device_index: int | None
    """
    CUDA device index for the async EPLB worker thread.
    """
    new_physical_to_logical_map: torch.Tensor | None = None
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    the size is same as physical_to_logical_map
    """
    new_logical_to_physical_map: torch.Tensor | None = None
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    the size is same as logical_to_physical_map
    """
    new_logical_replica_count: torch.Tensor | None = None
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    the size is same as logical_replica_count
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
        self.rearrange_event = threading.Event()
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
        if self.device.type == "cuda":
            self.cuda_device_index = self.device.index
            if self.cuda_device_index is None and torch.cuda.is_available():
                self.cuda_device_index = torch.cuda.current_device()

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

        # Set the policy based on the selected eplb algorithm type.
        policy_type = self.parallel_config.eplb_config.policy
        self.policy = EPLB_POLICIES[policy_type]
        logger.debug("Selected EPLB policy: %s", policy_type)
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
            ) = self.policy.rebalance_experts(
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
        else:
            new_physical_to_logical_map = None

            new_logical_to_physical_map = None

            new_logical_replica_count = None
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
            self.expert_rearrangement_step = 0

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
            buffer_lock=threading.Lock(),
            buffer_ready_event=None,
            buffer_consumed_event=None,
            ep_buffer_ready=0,
            layer_to_transfer=0,
            rebalanced=False,
            pending_global_ready_check=False,
            is_unchanged=np.array([]),
            is_received_locally=np.array([]),
            recv_metadata=RecvMetadata(
                recv_primary_mask=np.array([]),
                recv_count=0,
                recv_expert_ids=np.array([]),
                recv_dst_rows=np.array([]),
            ),
            cuda_device_index=self.cuda_device_index,
            new_physical_to_logical_map=new_physical_to_logical_map,
            new_logical_to_physical_map=new_logical_to_physical_map,
            new_logical_replica_count=new_logical_replica_count,
        )
        self.model_states[model_config.compute_hash()] = model_state

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
                all_ranks_buffer_ready = False
                if eplb_model_state.pending_global_ready_check:
                    all_ranks_buffer_ready = self._all_ranks_buffer_ready(
                        eplb_model_state
                    )
                if eplb_model_state.ep_buffer_ready and all_ranks_buffer_ready:
                    self.move_to_workspace(
                        model_state=eplb_model_state,
                        ep_group=ep_group,
                        is_profile=is_profile,
                    )
                    if (
                        eplb_model_state.layer_to_transfer
                        >= eplb_model_state.model.num_moe_layers
                    ):
                        self.post_eplb(eplb_model_state, is_profile)
                        eplb_model_state.rebalanced = False
                        eplb_model_state.layer_to_transfer = 0
                        eplb_model_state.pending_global_ready_check = False
                        logger.info(
                            "finish async transfer for model %s rank %d layer %d",
                            eplb_model_state.model_name,
                            ep_group.rank(),
                            eplb_model_state.model.num_moe_layers,
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
        execute_shuffle: bool = True,
        global_expert_loads: list[torch.Tensor] | None = None,
        rank_mapping: dict[int, int] | None = None,
    ) -> torch.Tensor | None:
        """
        Rearrange the experts according to the current load.

        Args:
            is_profile (bool): If `True`, perform a dummy rearrangement.
                This is used in `profile_run` to reserve enough memory,
                no memory movement will be performed. Default is False.
            execute_shuffle (bool): If `True`, execute the shuffle
                in elastic expert parallel (EEP). Default is True.
            global_expert_loads (list[torch.Tensor] | None): The global expert
                loads when scaling is done in EEP.
                List of expert loads for the main and drafter
                (when spec decode is used) models.
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
                logical_expert_load_window.scatter_add_(
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

        # TODO(bowen): Treat differently for prefill and decode nodes
        eplb_model_state = next(iter(self.model_states.values()))
        model = eplb_model_state.model
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
            # Get new expert mappings for the model
            (
                new_physical_to_logical_map,
                new_logical_to_physical_map,
                new_logical_replica_count,
            ) = self.policy.rebalance_experts(
                global_expert_load_window,
                num_replicas,
                num_groups,
                num_nodes,
                num_gpus,
                eplb_model_state.physical_to_logical_map,
            )

            if not self.is_async or is_profile:
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
                max_slots = eplb_model_state.logical_to_physical_map.shape[-1]
                padded_logical = torch.nn.functional.pad(
                    new_logical_to_physical_map,
                    (0, max(0, max_slots - new_logical_to_physical_map.shape[-1])),
                    value=-1,
                ).to(eplb_model_state.logical_to_physical_map.device)
                new_replica = new_logical_replica_count.to(
                    eplb_model_state.logical_replica_count.device
                )

                # Move map to cpu in advance
                eplb_model_state.new_physical_to_logical_map = (
                    new_physical_to_logical_map.cpu()
                )
                eplb_model_state.new_logical_to_physical_map = padded_logical
                eplb_model_state.new_logical_replica_count = new_replica

                eplb_model_state.rebalanced = True
                eplb_model_state.layer_to_transfer = 0
                eplb_model_state.pending_global_ready_check = True

        # Signal async thread to start transferring layers
        if self.is_async and (not is_profile):
            self.rearrange_event.set()
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
                rank_mapping=rank_mapping,
                is_profile=is_profile,
            )

    def _update_layer_mapping_from_new(
        self, model_state: EplbModelState, layer: int
    ) -> None:
        if (
            model_state.new_physical_to_logical_map is None
            or model_state.new_logical_to_physical_map is None
            or model_state.new_logical_replica_count is None
        ):
            return

        target_device = model_state.physical_to_logical_map.device
        new_physical = model_state.new_physical_to_logical_map
        if model_state.physical_to_logical_map.shape[1] != new_physical.shape[1]:
            model_state.physical_to_logical_map = new_physical.to(target_device)
        else:
            model_state.physical_to_logical_map[layer].copy_(
                new_physical[layer].to(target_device)
            )

        logical_device = model_state.logical_to_physical_map.device
        new_logical = model_state.new_logical_to_physical_map[layer].to(logical_device)
        max_slots = model_state.logical_to_physical_map.shape[-1]
        slot_delta = max_slots - new_logical.shape[-1]
        if slot_delta > 0:
            new_logical = torch.nn.functional.pad(
                new_logical, (0, slot_delta), value=-1
            )
        model_state.logical_to_physical_map[layer].copy_(new_logical)

        replica_device = model_state.logical_replica_count.device
        model_state.logical_replica_count[layer].copy_(
            model_state.new_logical_replica_count[layer].to(replica_device)
        )

    def _all_ranks_buffer_ready(self, model_state: EplbModelState) -> bool:
        parallel_state = get_ep_group()
        cpu_group = getattr(parallel_state, "cpu_group", None)
        if cpu_group is not None and cpu_group.size() > 1:
            flag = torch.tensor(
                (int(model_state.ep_buffer_ready),), dtype=torch.int32, device="cpu"
            )
            all_reduce(flag, group=cpu_group)
            return int(flag.item()) == cpu_group.size()

        device_group = parallel_state.device_group
        if device_group.size() <= 1:
            return bool(model_state.ep_buffer_ready)

        device = getattr(
            parallel_state, "device", model_state.physical_to_logical_map.device
        )
        flag = torch.tensor(
            (int(model_state.ep_buffer_ready),), dtype=torch.int32, device=device
        )
        all_reduce(flag, group=device_group)
        return int(flag.item()) == device_group.size()

    def move_to_workspace(
        self,
        model_state: EplbModelState,
        ep_group: ProcessGroup,
        is_profile: bool = False,
    ):
        # We call move_to_workspace only when ep_buffer_ready is 1.
        # It means we only need to wait for the lock for a short time.
        max_retries = 6  # 1 minute max
        retries = 0
        while not model_state.buffer_lock.acquire(blocking=True, timeout=10.0):
            retries += 1
            if retries >= max_retries:
                raise RuntimeError(
                    f"Rank {ep_group.rank()}: buffer_lock timeout after "
                    "{max_retries * 10}s"
                )
            logger.warning(
                "Rank %d: EPLB buffer_lock acquire failed, retrying (%d/%d)",
                ep_group.rank(),
                retries,
                max_retries,
            )
        try:
            assert model_state.new_physical_to_logical_map is not None
            device_index = model_state.cuda_device_index or self.cuda_device_index
            if model_state.buffer_ready_event is not None and device_index is not None:
                stream = torch.cuda.current_stream(device=device_index)
                stream.wait_event(model_state.buffer_ready_event)
                model_state.buffer_ready_event = None
            expert_weights = model_state.model.expert_weights[
                model_state.layer_to_transfer
            ]
            expert_weights_buffer = model_state.expert_buffer
            new_indices = (
                model_state.new_physical_to_logical_map[model_state.layer_to_transfer]
                .cpu()
                .numpy()
            )
            move_from_buffer(
                expert_weights=expert_weights,
                expert_weights_buffers=expert_weights_buffer,
                is_unchanged=model_state.is_unchanged,
                is_received_locally=model_state.is_received_locally,
                recv_metadata=model_state.recv_metadata,
                new_indices=new_indices,
                ep_rank=ep_group.rank(),
            )
            # Record event after consuming buffer to signal async thread
            # that it's safe to overwrite the buffer
            consumed_event = torch.cuda.Event()
            consumed_event.record()
            model_state.buffer_consumed_event = consumed_event

            transferred_layer = model_state.layer_to_transfer
            self._update_layer_mapping_from_new(model_state, transferred_layer)
            # After the main thread consumes, advance layer_to_transfer
            model_state.layer_to_transfer += 1
            model_state.ep_buffer_ready = 0
            logger.debug(
                "model %s successfully move_to_workspace layer %d",
                model_state.model_name,
                transferred_layer,
            )
        finally:
            try:
                model_state.buffer_lock.release()
            except Exception as e:
                logger.error(
                    "Rank %d: buffer_lock release failed in move_to_workspace: %s",
                    ep_group.rank(),
                    str(e),
                )

    def post_eplb(self, model_state: EplbModelState, is_profile: bool = False) -> None:
        assert model_state.new_physical_to_logical_map is not None
        assert model_state.new_logical_to_physical_map is not None
        assert model_state.new_logical_replica_count is not None
        if not is_profile:
            for layer_idx in range(model_state.physical_to_logical_map.shape[0]):
                self._update_layer_mapping_from_new(model_state, layer_idx)
        model_state.new_physical_to_logical_map = None
        model_state.new_logical_to_physical_map = None
        model_state.new_logical_replica_count = None

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
