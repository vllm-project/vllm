# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import weakref
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import P2POp

from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.wrapper import reset_compile_wrapper
from vllm.config import (
    CompilationMode,
    set_current_vllm_config,
)
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_standby_dp_group,
    get_standby_ep_group,
    get_tp_group,
)
from vllm.distributed.parallel_state import (
    create_standby_groups,
    prepare_communication_buffer_for_model,
    switch_to_standby_groups,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoEParallelConfig
from vllm.utils.torch_utils import supports_dynamo
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

logger = init_logger(__name__)


def batch_transfer_weights(
    model: nn.Module,
    is_sender: bool,
    peer_rank: int,
    dp_group: StatelessGroupCoordinator,
    expert_weights: Sequence[Iterable[torch.Tensor]],
) -> None:
    device_comm = dp_group.device_communicator
    if device_comm is None:
        raise ValueError("No device communicator found")

    expert_weights_set = set()
    for weight_group in expert_weights:
        for weight in weight_group:
            expert_weights_set.add(weight.data_ptr())

    state_dict = model.state_dict()
    all_params = []

    for name, param in state_dict.items():
        if name.endswith("expert_map"):
            continue
        if param.data_ptr() not in expert_weights_set:
            all_params.append(param.data)

    assert len(all_params) > 0
    p2p_ops = []
    for param in all_params:
        op = object.__new__(P2POp)
        if is_sender:
            op.op = torch.distributed.isend
            op.tensor = param
        else:
            op.op = torch.distributed.irecv
            op.tensor = param
        op.group_peer = peer_rank
        p2p_ops.append(op)
    device_comm.batch_isend_irecv(p2p_ops)


def broadcast_expert_mapping(
    physical_to_logical: torch.Tensor | None,
    num_local_physical_experts: int | None,
    num_logical_experts: int | None,
    dp_group: StatelessGroupCoordinator,
    device: torch.device,
    src_rank: int = 0,
) -> tuple[torch.Tensor, int, int]:
    if dp_group.rank_in_group == src_rank:
        assert physical_to_logical is not None
        assert num_local_physical_experts is not None
        assert num_logical_experts is not None
        assert physical_to_logical.dtype == torch.int64
        shape_tensor = torch.tensor(
            list(physical_to_logical.shape), dtype=torch.int64, device="cpu"
        )
        metadata_tensor = torch.tensor(
            [num_local_physical_experts, num_logical_experts],
            dtype=torch.int64,
            device="cpu",
        )
    else:
        shape_tensor = torch.empty(2, dtype=torch.int64, device="cpu")
        metadata_tensor = torch.empty(2, dtype=torch.int64, device="cpu")

    shape_tensor = dp_group.tcp_store_group.broadcast(shape_tensor, src_rank)
    metadata_tensor = dp_group.tcp_store_group.broadcast(metadata_tensor, src_rank)

    if dp_group.rank_in_group != src_rank:
        assert device is not None
        physical_to_logical = torch.empty(
            tuple(shape_tensor.tolist()),
            dtype=torch.int64,
            device=device,
        )

    assert physical_to_logical is not None
    physical_to_logical = dp_group.broadcast(physical_to_logical, src_rank)
    num_local_physical_experts = int(metadata_tensor[0].item())
    num_logical_experts = int(metadata_tensor[1].item())

    return physical_to_logical, num_local_physical_experts, num_logical_experts


class ElasticEPScalingExecutor:
    def __init__(self, worker):
        self.worker_ref = weakref.ref(worker)
        self.reconfig_request = None

    @property
    def worker(self):
        worker = self.worker_ref()
        if worker is None:
            raise RuntimeError("Worker has been garbage collected")
        return worker

    def execute(self, execute_method: str, *args, **kwargs):
        method = getattr(self, execute_method, None)
        if method is None:
            raise ValueError(f"Unknown execute method: {execute_method}")
        return method(*args, **kwargs)

    def create_standby_groups(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        self.reconfig_request = reconfig_request
        new_dp_size = reconfig_request.new_data_parallel_size
        world_size = self.worker.vllm_config.parallel_config.world_size
        new_world_size_across_dp = world_size * new_dp_size
        # TODO(yongji): check whether we need to use updated vllm_config here
        with set_current_vllm_config(self.worker.vllm_config):
            create_standby_groups(
                new_dp_size=new_dp_size,
                new_world_size_across_dp=new_world_size_across_dp,
                master_ip=reconfig_request.new_data_parallel_master_ip,
                world_group_ports=reconfig_request.new_stateless_world_group_port_list,
                dp_group_ports=reconfig_request.new_stateless_dp_group_port_list,
                ep_group_ports=reconfig_request.new_stateless_ep_group_port_list,
            )
        self.worker.model_runner.eplb_disabled = True
        standby_ep_group = get_standby_ep_group()
        assert standby_ep_group is not None
        if standby_ep_group.rank == 0:
            logger.info("[Elastic EP] EPLB disabled during elastic scaling transition")

    def transfer_weights(self, old_dp_size: int, new_dp_size: int) -> None:
        standby_dp_group = get_standby_dp_group()
        assert standby_dp_group is not None
        # Broadcast old_dp_size to all workers in standby group
        if standby_dp_group.rank_in_group < old_dp_size:
            old_dp_size_tensor = torch.tensor(
                [old_dp_size], dtype=torch.int64, device="cpu"
            )
        else:
            old_dp_size_tensor = torch.empty(1, dtype=torch.int64, device="cpu")
        old_dp_size_tensor = standby_dp_group.tcp_store_group.broadcast(
            old_dp_size_tensor, 0
        )

        num_new_workers = new_dp_size - old_dp_size
        dp_rank = self.worker.vllm_config.parallel_config.data_parallel_rank

        ranks_to_send = []
        # NOTE(yongji): determine sender-receiver pairing in weight transfer.
        # Mapping rule:
        # Base: each existing worker i gets (num_new_workers // old_dp_size) new workers
        #   to send weights to. Worker i sends weights to new workers with global ranks
        #   in [old_dp_size + i * num_dst_per_sender,
        #   old_dp_size + (i + 1) * num_dst_per_sender].
        # Remainder: Each of the first (num_new_workers % old_dp_size) existing workers
        #   gets an additional new worker to send weights to, whose global rank is
        #   old_dp_size * (num_dst_per_sender + 1) + i.
        num_dst_per_sender = num_new_workers // old_dp_size
        sender_pos = dp_rank
        recv_begin = sender_pos * num_dst_per_sender
        recv_end = recv_begin + num_dst_per_sender
        ranks_to_send = list(range(old_dp_size + recv_begin, old_dp_size + recv_end))
        remainder_start = old_dp_size * num_dst_per_sender
        recver_pos = remainder_start + sender_pos
        if recver_pos < num_new_workers:
            ranks_to_send.append(old_dp_size + recver_pos)

        model = self.worker.model_runner.get_model()
        for new_worker_rank in sorted(ranks_to_send):
            batch_transfer_weights(
                model=model,
                is_sender=True,
                peer_rank=new_worker_rank,
                dp_group=standby_dp_group,
                expert_weights=model.expert_weights,
            )
        torch.cuda.synchronize()

    def broadcast_expert_mapping(self) -> None:
        standby_dp_group = get_standby_dp_group()
        assert standby_dp_group is not None
        model_config = self.worker.model_runner.model_config
        eplb_state = self.worker.model_runner.eplb_state
        assert eplb_state is not None
        eplb_model_state = eplb_state.model_states[model_config.compute_hash()]
        physical_to_logical = eplb_model_state.physical_to_logical_map
        num_physical_experts = physical_to_logical.shape[1]
        num_local_physical_experts = num_physical_experts // get_ep_group().world_size
        num_logical_experts = eplb_model_state.logical_replica_count.shape[1]
        broadcast_expert_mapping(
            physical_to_logical=physical_to_logical,
            num_local_physical_experts=num_local_physical_experts,
            num_logical_experts=num_logical_experts,
            dp_group=standby_dp_group,
            src_rank=0,
            device=self.worker.device,
        )

    def switch_and_prepare(self) -> None:
        old_dp_size = get_dp_group().world_size
        old_ep_size = get_ep_group().world_size

        switch_to_standby_groups()

        parallel_config = self.worker.vllm_config.parallel_config
        reconfig_request = self.reconfig_request
        assert reconfig_request is not None
        new_dp_size = reconfig_request.new_data_parallel_size
        new_ep_size = get_ep_group().world_size

        parallel_config.data_parallel_size = new_dp_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if (
            reconfig_request.new_data_parallel_rank_local
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank_local = (
                reconfig_request.new_data_parallel_rank_local
            )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )

        # Reconfigure MoE modules with new EP size
        moe_modules = [
            module
            for module in self.worker.model_runner.model.modules()
            if (
                module.__class__.__name__ == "FusedMoE"
                or module.__class__.__name__ == "SharedFusedMoE"
            )
        ]
        num_local_experts = moe_modules[0].moe_config.num_local_experts
        assert all(
            module.moe_config.num_local_experts == num_local_experts
            for module in moe_modules
        ), "All MoE modules must have the same number of experts"
        for module in moe_modules:
            module.moe_config.num_experts = num_local_experts * new_ep_size
            module.global_num_experts = module.moe_config.num_experts
            module.moe_parallel_config = FusedMoEParallelConfig.make(
                tp_size_=get_tp_group().world_size,
                pcp_size_=get_pcp_group().world_size,
                dp_size_=get_dp_group().world_size,
                vllm_parallel_config=parallel_config,
            )
            module.moe_config.moe_parallel_config = module.moe_parallel_config

        # Update EPLB state
        eplb_state = self.worker.model_runner.eplb_state
        assert eplb_state is not None
        model_config = self.worker.model_runner.model_config
        eplb_model_state = eplb_state.model_states[model_config.compute_hash()]

        num_physical_experts = num_local_experts * new_ep_size
        num_logical_experts = eplb_model_state.logical_replica_count.shape[1]
        parallel_config.eplb_config.num_redundant_experts = (
            num_physical_experts - num_logical_experts
        )
        old_physical_to_logical = eplb_model_state.physical_to_logical_map
        num_moe_layers = old_physical_to_logical.shape[0]
        num_local_experts = eplb_model_state.expert_load_pass.shape[1] // old_ep_size
        if new_dp_size > old_dp_size:
            expanded_physical_to_logical = torch.full(
                (num_moe_layers, num_local_experts * new_ep_size),
                -1,
                dtype=old_physical_to_logical.dtype,
                device=old_physical_to_logical.device,
            )
            expanded_physical_to_logical[:, : num_local_experts * old_ep_size] = (
                old_physical_to_logical
            )
            eplb_model_state.physical_to_logical_map = expanded_physical_to_logical

        old_num_physical_experts = eplb_model_state.expert_load_pass.shape[1]
        pad_size = num_physical_experts - old_num_physical_experts
        if new_dp_size > old_dp_size:
            assert pad_size > 0
            expanded_expert_load_pass = F.pad(
                eplb_model_state.expert_load_pass, (0, pad_size), value=0
            )
            expanded_expert_load_window = F.pad(
                eplb_model_state.expert_load_window, (0, pad_size), value=0
            )
            eplb_model_state.expert_load_pass = expanded_expert_load_pass
            eplb_model_state.expert_load_window = expanded_expert_load_window
            eplb_state.num_valid_physical_experts = old_num_physical_experts
        else:
            assert pad_size < 0
            eplb_model_state.expert_load_pass = eplb_model_state.expert_load_pass[
                :, :num_physical_experts
            ]
            eplb_model_state.expert_load_window = eplb_model_state.expert_load_window[
                :, :, :num_physical_experts
            ]
            eplb_state.num_valid_physical_experts = num_physical_experts

        model = self.worker.model_runner.get_model()
        model.expert_weights = []
        with set_current_vllm_config(self.worker.vllm_config):
            model.set_eplb_state(
                eplb_model_state.expert_load_pass,
                eplb_model_state.logical_to_physical_map,
                eplb_model_state.logical_replica_count,
            )
            model.update_physical_experts_metadata(
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_local_experts,
            )
            prepare_communication_buffer_for_model(self.worker.model_runner.model)
        if (
            self.worker.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
            and supports_dynamo()
        ):
            # NOTE(yongji): when using stock torch.compile,
            # torch.compile is triggered during GPUModelRunner's load_model()
            # TODO(yongji):check do we need to re-trigger torch.compile here?
            # any changes to the tensor shapes in execution should already
            # be handled internally by torch.compile.
            backend = self.worker.vllm_config.compilation_config.init_backend(
                self.worker.vllm_config
            )
            compilation_counter.stock_torch_compile_count += 1
            self.worker.model_runner.model.compile(fullgraph=True, backend=backend)

        # release all previously captured CUDA graphs
        if isinstance(self.worker.model_runner.model, CUDAGraphWrapper):
            # TODO(yongji): do we need to reset graph pool here?
            wrapper = self.worker.model_runner.model
            wrapper.concrete_cudagraph_entries = {}
        elif isinstance(self.worker.model_runner.model, UBatchWrapper):
            raise RuntimeError("DBO is not yet supported in elastic EP")

        # reset the compile wrapper
        with set_current_vllm_config(self.worker.vllm_config):
            reset_compile_wrapper(self.worker.model_runner.get_model())

        gc.collect()
        torch.cuda.empty_cache()
        self.worker.compile_or_warm_up_model()

    def perform_eplb_reshuffle(self, new_dp_size: int | None = None) -> None:
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding...")

        eplb_state = self.worker.model_runner.eplb_state
        assert eplb_state is not None

        model_config = self.worker.model_runner.model_config
        eplb_model_state = eplb_state.model_states[model_config.compute_hash()]
        is_async_enabled = eplb_model_state.is_async_enabled
        eplb_model_state.is_async_enabled = False
        if new_dp_size is None:
            eplb_state.rearrange()
        else:
            # scale down
            parallel_config = self.worker.vllm_config.parallel_config
            tp_size = parallel_config.tensor_parallel_size
            old_ep_size = parallel_config.data_parallel_size * tp_size
            new_ep_size = new_dp_size * tp_size

            rank_mapping = {
                old_ep_rank: old_ep_rank if old_ep_rank < new_ep_size else -1
                for old_ep_rank in range(old_ep_size)
            }

            eplb_state.rearrange(rank_mapping=rank_mapping)
        # NOTE(yongji): check whether we need to synchronize here
        torch.cuda.synchronize()
        # reset expert_rearrangement_step to ensure all ranks are synchronized
        eplb_state.expert_rearrangement_step = 0
        eplb_model_state.is_async_enabled = is_async_enabled
        self.worker.model_runner.eplb_disabled = False
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed")

    def receive_weights(self) -> None:
        dp_group = get_dp_group()
        assert isinstance(dp_group, StatelessGroupCoordinator)
        new_dp_size = dp_group.world_size
        dp_rank = self.worker.vllm_config.parallel_config.data_parallel_rank

        # Receive old_dp_size broadcasted during transfer_weights
        old_dp_size_tensor = torch.empty(1, dtype=torch.int64, device="cpu")
        old_dp_size_tensor = dp_group.tcp_store_group.broadcast(old_dp_size_tensor, 0)
        old_dp_size = int(old_dp_size_tensor[0].item())

        # Calculate which existing worker will send to this new worker
        num_new_workers = new_dp_size - old_dp_size
        new_worker_idx = dp_rank - old_dp_size
        num_dst_per_sender = num_new_workers // old_dp_size
        remainder = num_new_workers % old_dp_size

        if new_worker_idx < remainder * (num_dst_per_sender + 1):
            sender_rank = new_worker_idx // (num_dst_per_sender + 1)
        else:
            sender_rank = (
                remainder
                + (new_worker_idx - remainder * (num_dst_per_sender + 1))
                // num_dst_per_sender
            )

        model = self.worker.model_runner.get_model()
        batch_transfer_weights(
            model=model,
            is_sender=False,
            peer_rank=sender_rank,
            dp_group=dp_group,
            expert_weights=model.expert_weights,
        )
        torch.cuda.synchronize()

    def receive_expert_mapping(self) -> tuple[torch.Tensor, int, int]:
        dp_group = get_dp_group()
        assert isinstance(dp_group, StatelessGroupCoordinator)
        physical_to_logical, num_local_physical_experts, num_logical_experts = (
            broadcast_expert_mapping(
                physical_to_logical=None,
                num_local_physical_experts=None,
                num_logical_experts=None,
                dp_group=dp_group,
                src_rank=0,
                device=self.worker.device,
            )
        )
        num_moe_layers = physical_to_logical.shape[0]
        new_dp_size = get_dp_group().world_size
        tp_size = self.worker.vllm_config.parallel_config.tensor_parallel_size
        new_ep_size = new_dp_size * tp_size
        expanded_physical_to_logical = torch.full(
            (num_moe_layers, num_local_physical_experts * new_ep_size),
            -1,
            dtype=physical_to_logical.dtype,
            device=physical_to_logical.device,
        )
        old_num_physical_experts = physical_to_logical.shape[1]
        expanded_physical_to_logical[:, :old_num_physical_experts] = physical_to_logical
        return (
            expanded_physical_to_logical,
            num_logical_experts,
            old_num_physical_experts,
        )

    def prepare_new_worker(self) -> None:
        with set_current_vllm_config(self.worker.vllm_config):
            prepare_communication_buffer_for_model(self.worker.model_runner.get_model())
