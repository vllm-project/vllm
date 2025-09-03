import time
from typing import Optional, Sequence, Iterable

import torch
from torch._C._distributed_c10d import ProcessGroup
from torch.distributed import all_reduce, all_gather

from vllm.distributed import get_ep_group, get_node_count
from vllm.distributed.eplb import rebalance_experts
from vllm.distributed.eplb.eplb_data.eplb_data import EplbData
from vllm.distributed.eplb.eplb_updator.abstract_updator import BaseUpdator

from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.utils import logger

from overrides import override


class EplbUpdator(BaseUpdator):
    def __init__(self, eplb_data: EplbData, eplb_loader):
        self.eplb_data = eplb_data
        self.eplb_loader = eplb_loader

    @override
    def profile(self, model: MixtureOfExperts, is_profile):
        self.rearrange(model, is_profile)
        return

    @override
    def dummy(self, model: MixtureOfExperts):
        self.eplb_data.expert_load_pass.zero_()
        self.eplb_data.expert_rearrangement_step += 1
        if (self.eplb_data.expert_rearrangement_step
                >= self.eplb_data.expert_rearrangement_step_interval):
            self.eplb_data.expert_rearrangement_step = 0
            self.rearrange(model)

    def log_stats(self):
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

        if is_profile:
            self.profile(model)
        if is_dummy:
            # Do not record load metrics for dummy steps
            self.dummy(model)
        if log_stats:
            self.log_stats()

        # Update the expert load sliding window
        # if not is_dummy:
        self.eplb_data.expert_load_window[self.eplb_data.expert_load_window_step] = (
            self.eplb_data.expert_load_pass.clone())
        self.eplb_data.expert_load_window_step += 1
        if self.eplb_data.expert_load_window_step >= self.eplb_data.expert_load_window_size:
            self.expert_load_window_step = 0
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
    def step_after_forward(self):
        # 1 根据热度异步调用算法计算专家热度
        # 2 根据计算结果异步编排任务，返回任务参数
        # 3 根据编排好的参数调用generate_task(封装了prepare_send 以及 prepare_recv)
        # 4 调用传输方法

        pass

    @override
    def rearrange(self,
                  model: MixtureOfExperts,
                  is_profile: bool = False,
                  execute_shuffle: bool = True,
                  global_expert_load: Optional[torch.Tensor] = None,
                  rank_mapping: Optional[dict[int, int]] = None) -> None:
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
        ) = (rebalance_experts(
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
                self.eplb_data.physical_to_logical_map = new_physical_to_logical_map.to(
                    self.eplb_data.physical_to_logical_map.device)
            else:
                self.eplb_data.physical_to_logical_map.copy_(new_physical_to_logical_map)
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

    @override
    def recv_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Receive the expert load and old placement from the master rank.
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
            if new_rank >= 0 and new_rank < new_ep_size:
                old_start_idx = old_rank * num_local_physical_experts
                old_end_idx = (old_rank + 1) * num_local_physical_experts
                new_start_idx = new_rank * num_local_physical_experts
                new_end_idx = (new_rank + 1) * num_local_physical_experts

                mapped_expert_indices[:, old_start_idx:old_end_idx] = \
                    new_global_expert_indices[:, new_start_idx:new_end_idx]

        return mapped_expert_indices
