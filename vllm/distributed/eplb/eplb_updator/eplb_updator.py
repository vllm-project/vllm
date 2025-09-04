import time
from multiprocessing import Manager

from typing import Optional, Sequence, Iterable

import torch
from torch._C._distributed_c10d import ProcessGroup
from torch.distributed import all_reduce, all_gather

from vllm.distributed import get_ep_group, get_node_count
from vllm.distributed.eplb import EplbWeightLoader
from vllm.distributed.eplb.eplb_data.eplb_data import EplbData
from vllm.distributed.eplb.eplb_updator.abstract_updator import BaseUpdator
from vllm.distributed.eplb.eplb_process.eplb_process import EplbProcess
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.utils import logger
import numpy
import torch.distrubuted as dist

from overrides import override


class EplbUpdator(BaseUpdator):
    """
    Manages the Expert Parallel Load Balancing (EPLB) update process,
    orchestrating the collection of expert load metrics, the re-balancing
    of experts across ranks, and the transfer of expert weights.

    It interacts with `EplbData` for shared state, `EplbWeightLoader` for
    weight transfer, and `EplbProcess` for expert mapping calculations.
    """
    def __init__(self, eplb_data: EplbData, eplb_loader: EplbWeightLoader,adaptor, eplb_process: EplbProcess):
        """
        Initializes the EplbUpdator.

        Args:
            eplb_data: An instance of EplbData holding shared EPLB configuration and metrics.
            eplb_loader: An instance of EplbWeightLoader responsible for expert weight transfer.
            adaptor: An adaptor to interact with the vLLM model's expert parameters.
            eplb_process: An instance of EplbProcess for handling expert mapping logic.
        """
        self.cur_iterations = None
        self.device = None
        self.world_size = None
        self._gather_buffer = None
        self.eplb_policy = None
        self.reqs = None
        self.eplb_process = eplb_process
        self.eplb_data = eplb_data
        self.eplb_loader = eplb_loader
        #新加init看是否需要
        self.eplb_adaptor = adaptor
        self.manager = Manager()
        self.shared_dict = self.manager.dict({
            # 当前rank_id的专家表[num_layers,num_experts]
            "expert_map": None,
            # 热度负载信息 [num_layers, world_size, num_experts]
            "moe_load": None,
            # 所有的专家表[num_layers, world_size, num_experts]
            "expert_maps": None,
        })


    def profile(self, model: MixtureOfExperts, is_profile):
        """
        Profiles the expert rearrangement process.
        During profiling, it performs a dummy rearrangement to reserve memory.

        Args:
            model: The MoE model.
            is_profile: If True, indicates a profiling run.
        """
        self.rearrange(model, is_profile)
        return

    def dummy(self, model: MixtureOfExperts):
        """
        Performs a dummy step for expert load balancing.
        It clears the expert load and increments the rearrangement step counter.
        If the counter reaches the interval, it triggers a rearrangement.

        Args:
            model: The MoE model.
        """
        self.eplb_data.expert_load_pass.zero_()
        self.eplb_data.expert_rearrangement_step += 1
        if (self.eplb_data.expert_rearrangement_step
                >= self.eplb_data.expert_rearrangement_step_interval):
            self.eplb_data.expert_rearrangement_step = 0
            self.rearrange(model)

    @override
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

        if log_stats:
            # total_expert_load_pass: (num_moe_layers, num_physical_experts)
            total_expert_load_pass = self.eplb_data.expert_load_pass.clone()

            # Collect load metrics from all ranks
            ep_group = get_ep_group().device_group
            all_reduce(total_expert_load_pass, group=ep_group)

            # num_tokens_per_rank: (num_moe_layers, num_ranks)
            num_tokens_per_rank = total_expert_load_pass.reshape(
                total_expert_load_pass.shape[0], ep_group.size(),
                -1).sum(dim=-1).float()

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
        # if not is_dummy:
        self.eplb_data.expert_load_window[self.eplb_data.expert_load_window_step] = (
            self.eplb_data.expert_load_pass.clone())
        self.eplb_data.expert_load_window_step += 1
        if self.eplb_data.expert_load_window_step >= self.eplb_data.expert_load_window_size:
            self.eplb_data.expert_load_window_step = 0
        self.eplb_data.expert_load_pass.zero_()

        # Step the expert rearrangement step
        # Note that even if this is a dummy step, we still increment the
        # rearrangement step and perform rearrangement to ensure all ranks are
        # performing collective communication.
        self.eplb_data.expert_rearrangement_step += 1
        if (self.eplb_data.expert_rearrangement_step
                >= self.eplb_data.expert_rearrangement_step_interval):
            self.eplb_data.expert_rearrangement_step = 0
            self.rearrange(model)

    @override
    def step_before_forward(self):
        """
        Executes operations before the model's forward pass.
        If the EPLB process indicates it should process (e.g., a rearrangement
        is pending), it initiates asynchronous shuffling for each MoE layer.
        """
        if self.eplb_process._should_process():
            # adaptor与updator解耦，数据相关的类型放到数据侧
            for layer_id in range(self.eplb_adaptor.num_moe_layers):
                self.shuffer_layer_async(layer_id)

    def shuffer_layer_async(self,layer):
        """
        Initiates asynchronous shuffling of experts for a specific MoE layer.
        This method retrieves the necessary information from `eplb_process`,
        prepares the weight loader for transfer tasks, and starts the
        asynchronous communication.

        Args:
            layer: The ID of the MoE layer to shuffle.
        """
        if self.eplb_process._should_process():
            (expert_send_info, expert_recv_info, updated_expert_map, 
            log2phy_map, layer_id) = self.eplb_process.get_at_index(layer)

            log2phy_map_this_rank = torch.from_numpy(numpy.array(log2phy_map))
            self.eplb_loader.set_log2phy_map(log2phy_map_this_rank)
            updated_expert_map_this_rank = torch.from_numpy(
                numpy.array(updated_expert_map))

            self.eplb_loader.generate_expert_d2d_transfer_task(  
                    expert_send_info, expert_recv_info, 
                    updated_expert_map_this_rank, 
                    layer_id + self.eplb_adaptor.num_dense_layers)
            self.reqs = []
            self.eplb_loader.async_expert_weight_transfer()
            
    @override
    def step_after_forward(self):
        """
        Executes operations after the model's forward pass.
        This includes waking up the EPLB worker, computing and setting MoE load,
        and updating expert weights if conditions are met.
        """
        if self.wakeup_eplb_worker_flag():
            self.compute_and_set_moe_load(is_clear=True)
 
        if self.update_expert_weight_flag():
            self.eplb_loader.update_expert_map_and_weight()
 
        self.update_iteration()

    def update_expert_weight_flag(self):
        """
        Determines if expert weights should be updated in the current iteration.
        This is typically true for a short window after the EPLB update and worker wait.

        Returns:
            True if expert weights should be updated, False otherwise.
        """
        weight_update_counter = self.cur_iterations - (
            self.eplb_data.num_iterations_eplb_update + self.eplb_data.num_wait_worker_iterations)
        return 0 <= weight_update_counter < self.eplb_data.num_moe_layers

    def wakeup_eplb_worker_flag(self):
        """
        Determines if the EPLB worker process should be woken up in the current iteration.
        This typically happens just before the expert weight update phase.

        Returns:
            True if the EPLB worker should be woken up, False otherwise.
        """
        return self.cur_iterations == (self.eplb_data.num_iterations_eplb_update - 1)

    def compute_and_set_moe_load(self, is_clear=False):
        """
        Computes the MoE load across all ranks and sets it in the shared dictionary.
        It gathers local expert load from all ranks and combines them.

        Args:
            is_clear: If True, indicates a clear operation (though not explicitly used here).

        Returns:
            The gathered MoE load tensor.
        """
        local_load = self.eplb_adaptor.get_rank_expert_workload() #取local load逻辑

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.device = local_load.device #local load换成self.expert_load_view
            if self._gather_buffer is None:
                shape = (self.world_size, *local_load.shape)
                self._gather_buffer = torch.empty(shape,
                                                  dtype=local_load.dtype,
                                                  device=self.device)

            dist.all_gather_into_tensor(self._gather_buffer, local_load)

            moe_load = self._gather_buffer.permute(1, 0, 2)
            self.shared_dict["moe_load"] = moe_load.cpu()
            logger.debug(
                f"[ModelRunner] Updated shared_dict['moe_load'] shape={moe_load.shape}"
            )
        else:
            moe_load = local_load.unsqueeze(1)
            self.shared_dict["moe_load"] = moe_load.cpu()
            logger.debug(
                f"[ModelRunner] Updated shared_dict['moe_load'] shape={moe_load.shape}"
            )
        return moe_load

    def rearrange(self,
                  model: MixtureOfExperts,
                  is_profile: bool = False,
                  execute_shuffle: bool = True,
                  global_expert_load: Optional[torch.Tensor] = None,
                  rank_mapping: Optional[dict[int, int]] = None) -> None:
        """
        Rearranges the experts according to the current load and the chosen EPLB policy.
        This involves aggregating global expert load, applying the rebalancing policy,
        and then physically moving expert weights.

        Args:
            model: The MoE model.
            is_profile: If `True`, performs a dummy rearrangement for profiling.
            execute_shuffle: If `True`, actually shuffles the weights; otherwise,
                             only calculates new mappings.
            global_expert_load: Pre-computed global expert load (optional).
            rank_mapping: A dictionary mapping old rank to new rank, used for scaling.
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

        if global_expert_load is None:
            # Map the physical expert load to global logical experts
            logical_expert_load_window = torch.zeros(
                self.eplb_data.expert_load_window_size,
                model.num_moe_layers,
                model.num_logical_experts,
                dtype=self.eplb_data.expert_load_window.dtype,
                device=self.eplb_data.expert_load_window.device,
            )
            logical_expert_load_window.scatter_add_(
                dim=-1,
                index=self.eplb_data.physical_to_logical_map.unsqueeze(0).expand_as(
                    self.eplb_data.expert_load_window).long(),
                src=self.eplb_data.expert_load_window,
            )

            if not execute_shuffle:
                metadata = torch.tensor(
                    [
                        model.num_moe_layers, model.num_logical_experts,
                        self.eplb_data.physical_to_logical_map.shape[1]
                    ],
                    dtype=torch.int32,
                    device="cpu",
                )
                torch.distributed.broadcast(metadata,
                                            group=get_ep_group().cpu_group,
                                            group_src=0)

            # Perform all-reduce to get the expert load across all ranks
            global_expert_load_window = logical_expert_load_window.sum(dim=0)
            all_reduce(global_expert_load_window, group=ep_group)

            if not execute_shuffle:
                # (num_moe_layers, old_num_physical_experts)
                old_global_expert_indices = self.eplb_data.physical_to_logical_map
                torch.distributed.broadcast(old_global_expert_indices,
                                            group=ep_group,
                                            group_src=0)
                return global_expert_load_window
        else:
            assert execute_shuffle
            global_expert_load_window = global_expert_load

        # TODO(bowen): Treat differently for prefill and decode nodes
        num_replicas = model.num_physical_experts
        num_groups = model.num_expert_groups
        if rank_mapping is not None and len(rank_mapping) == ep_group.size():
            # NOTE(yongji): scale down, we need to rebalance the experts on
            # remaining GPUs, transfer the experts while we haven't shutdown
            # the GPUs to be released.
            cpu_group = get_ep_group().cpu_group
            num_nodes = BaseUpdator._node_count_with_rank_mapping(cpu_group, rank_mapping)
            num_gpus = sum(new_rank != -1
                           for new_rank in rank_mapping.values())
            num_replicas = num_replicas // ep_group.size(
            ) * num_gpus  # handle num replicas change
        else:
            num_nodes = get_node_count()
            num_gpus = ep_group.size()

        if num_gpus % num_nodes != 0:
            self.num_nodes = 1
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
            global_expert_load_window,
            num_replicas,
            num_groups,
            num_nodes,
            num_gpus,
        ))

        # Update expert weights
        self.rearrange_expert_weights_inplace(
            self.eplb_data.physical_to_logical_map,
            new_physical_to_logical_map,
            model.expert_weights,
            ep_group,
            is_profile,
            rank_mapping,
        )

        if not is_profile:
            if self.eplb_data.physical_to_logical_map.shape[
                1] != new_physical_to_logical_map.shape[1]:
                self.physical_to_logical_map = new_physical_to_logical_map.to(
                    self.eplb_data.physical_to_logical_map.device)
            else:
                self.physical_to_logical_map.copy_(new_physical_to_logical_map)
            max_physical_slots = new_logical_to_physical_map.shape[-1]
            assert max_physical_slots <= self.eplb_data.logical_to_physical_map.shape[-1]
            new_logical_to_physical_map = torch.nn.functional.pad(
                new_logical_to_physical_map,
                (0,
                 self.eplb_data.logical_to_physical_map.shape[-1] - max_physical_slots),
                value=-1,
            )
            self.eplb_data.logical_to_physical_map.copy_(new_logical_to_physical_map)
            self.eplb_data.logical_replica_count.copy_(new_logical_replica_count)

        if is_main_rank:
            assert time_start is not None
            torch.cuda.synchronize()
            time_end = time.perf_counter()
            logger.info(
                "Rearranged experts%sin %.2f seconds.",
                " (profile) " if is_profile else " ",
                time_end - time_start,
            )

    def recv_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Receives the global expert load and old expert placement information
        from the master rank (rank 0). This is used when a rank is not the
        master and needs to synchronize state for rearrangement.

        Returns:
            A tuple containing:
            - global_expert_load: The aggregated expert load across all ranks.
            - old_global_expert_indices: The previous mapping of physical to logical experts.
        """
        ep_group = get_ep_group()
        metadata = torch.empty(3, dtype=torch.int32, device="cpu")
        torch.distributed.broadcast(metadata,
                                    group=ep_group.cpu_group,
                                    group_src=0)
        num_moe_layers, num_logical_experts, num_old_physical_experts = (
            metadata.tolist())
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
        torch.distributed.broadcast(old_global_expert_indices,
                                    group=ep_group.device_group,
                                    group_src=0)

        return global_expert_load, old_global_expert_indices



    def rearrange_expert_weights_inplace(self,
        old_global_expert_indices: torch.Tensor,
        new_global_expert_indices: torch.Tensor,
        expert_weights: Sequence[Iterable[torch.Tensor]],
        ep_group: ProcessGroup,
        is_profile: bool = False,
        rank_mapping: Optional[dict[int, int]] = None,
    ) -> None:
        """
        Rearranges the expert weights in place according to the new expert indices.

        The value of the indices arguments are logical indices of the experts,
        while keys are physical.

        Args:
            old_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
            new_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
            expert_weights: A sequence of shape (num_moe_layers)(weight_count)
                of tensors of shape (num_local_physical_experts, hidden_size_i).
                For example, a linear layer may have up and down projection,
                so weight_count = 2. Each weight's hidden size can be different.
            ep_group: The device process group for expert parallelism.
            is_profile (bool): If `True`, do not perform any actual weight copy.
                This is used during profile run, where we only perform dummy
                communications to reserve enough memory for the buffers.
            rank_mapping: A dictionary mapping old rank to new rank.
        """
        if rank_mapping is not None:
            if len(rank_mapping) == ep_group.size():
                # scale down
                new_global_expert_indices = \
                    self._map_new_expert_indices_with_rank_mapping(
                    new_global_expert_indices,
                    rank_mapping,
                )
            else:
                # scale up
                old_global_expert_indices = \
                    self._map_old_expert_indices_with_rank_mapping(
                    old_global_expert_indices,
                    rank_mapping,
                    ep_group.size(),
                )

        assert old_global_expert_indices.shape[
            1] == new_global_expert_indices.shape[1]

        num_moe_layers, num_physical_experts = old_global_expert_indices.shape
        assert len(expert_weights) == num_moe_layers

        num_local_physical_experts = next(iter(expert_weights[0])).shape[0]
        assert new_global_expert_indices.shape == (num_moe_layers,
                                                   num_physical_experts)

        ep_rank = ep_group.rank()
        ep_size = ep_group.size()
        assert num_physical_experts == ep_size * num_local_physical_experts

        # A buffer to hold the expert weights in one layer during the exchange.
        # NOTE: Currently we assume the same weights across different layers
        # have the same shape.
        expert_weights_buffer = [torch.empty_like(w) for w in expert_weights[0]]

        if is_profile:
            # Maximum send size is to send all local experts to all ranks,
            # So we use a dummy `all_gather` to reserve enough communication buffer
            for weight, buffer in zip(expert_weights[0], expert_weights_buffer):
                # A `/dev/null`-like buffer to avoid real memory allocation
                dummy_recv_buffer = [buffer for _ in range(ep_size)]
                # NOTE(bowen): Needed this barrier to avoid OOM during actual
                # execution. I'm not very sure why this is needed
                torch.distributed.barrier()
                all_gather(
                    dummy_recv_buffer,
                    weight,
                    group=ep_group,
                )
            return

        for layer in range(num_moe_layers):
            # NOTE(bowen): We need this synchronize to run, but I don't know why.
            # If you figure out the reason, please let me know -- thank you!
            torch.cuda.synchronize()
            self.eplb_loader.shuffle_layer(
                num_local_physical_experts,
                ep_rank,
                old_global_expert_indices[layer].tolist(),
                new_global_expert_indices[layer].tolist(),
                expert_weights[layer],
                expert_weights_buffer,
                ep_group,
            )



    def _map_old_expert_indices_with_rank_mapping(self,
        old_global_expert_indices: torch.Tensor,
        rank_mapping: dict[int, int],
        new_ep_size: int,
    ) -> torch.Tensor:
        """
        Map the old global expert indices to the new global expert indices.

        Args:
            old_global_expert_indices:
                Shape (num_layers, old_ep_size * num_local_physical_experts).
            rank_mapping: Mapping from old rank to new rank.
            new_ep_size: New expert parallelism size.

        Returns:
            Mapped expert indices with shape
            (num_layers, new_ep_size * num_local_physical_experts).
        """
        num_layers, old_num_physical_experts = old_global_expert_indices.shape
        assert rank_mapping, "Rank mapping is required"

        # Get sizes from parameters and rank_mapping
        old_ep_size = len(rank_mapping)
        num_local_physical_experts = old_num_physical_experts // old_ep_size
        new_num_physical_experts = new_ep_size * num_local_physical_experts

        # Create mapped tensor with new shape, initialized to -1
        mapped_expert_indices = torch.full(
            (num_layers, new_num_physical_experts),
            fill_value=-1,
            dtype=old_global_expert_indices.dtype,
            device=old_global_expert_indices.device,
        )

        # Handle rank mapping (scale up/down with rank changes)
        for old_rank in range(old_ep_size):
            new_rank = rank_mapping.get(old_rank)
            if new_rank is not None and new_rank >= 0 and new_rank < new_ep_size:
                # This old rank exists in the new configuration
                old_start_idx = old_rank * num_local_physical_experts
                old_end_idx = (old_rank + 1) * num_local_physical_experts
                new_start_idx = new_rank * num_local_physical_experts
                new_end_idx = (new_rank + 1) * num_local_physical_experts

                mapped_expert_indices[:, new_start_idx:new_end_idx] = \
                    old_global_expert_indices[:, old_start_idx:old_end_idx]
            # If new_rank is None or >= new_ep_size, the experts remain -1
            # (scale down case)

        return mapped_expert_indices



    def _map_new_expert_indices_with_rank_mapping(self,
        new_global_expert_indices: torch.Tensor,
        rank_mapping: dict[int, int],
    ) -> torch.Tensor:
        num_layers, new_num_physical_experts = new_global_expert_indices.shape
        assert rank_mapping, "Rank mapping is required"

        # Get sizes from parameters and rank_mapping
        old_ep_size = len(rank_mapping)
        new_ep_size = sum(new_rank != -1 for new_rank in rank_mapping.values())
        num_local_physical_experts = new_num_physical_experts // new_ep_size
        old_num_physical_experts = old_ep_size * num_local_physical_experts

        mapped_expert_indices = torch.full(
            (num_layers, old_num_physical_experts),
            fill_value=-1,
            dtype=new_global_expert_indices.dtype,
            device=new_global_expert_indices.device,
        )

        for old_rank in range(old_ep_size):
            new_rank = rank_mapping[old_rank]
            if 0 <= new_rank < new_ep_size:
                old_start_idx = old_rank * num_local_physical_experts
                old_end_idx = (old_rank + 1) * num_local_physical_experts
                new_start_idx = new_rank * num_local_physical_experts
                new_end_idx = (new_rank + 1) * num_local_physical_experts

                mapped_expert_indices[:, old_start_idx:old_end_idx] = \
                    new_global_expert_indices[:, new_start_idx:new_end_idx]

        return mapped_expert_indices

    def update_iteration(self):
        self.cur_iterations += 1
        if self.cur_iterations == (self.eplb_data.num_iterations_eplb_update +
                                   self.eplb_data.num_wait_worker_iterations + self.eplb_data.num_moe_layers):
            self.eplb_adaptor.model.clear_all_moe_loads()
            if not self.eplb_data.gate_eplb:
                self.cur_iterations = 0