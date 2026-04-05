# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import gc
import weakref
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

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
    get_tp_group,
)
from vllm.distributed.elastic_ep.standby_state import (
    create_standby_groups,
    get_standby_dp_group,
    get_standby_ep_group,
    pop_standby_groups,
)
from vllm.distributed.eplb.eplb_communicator import create_eplb_communicator
from vllm.distributed.parallel_state import (
    _abort_and_replace_active_groups,
    _replace_active_groups,
    get_eplb_group,
    prepare_communication_buffer_for_model,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoEParallelConfig
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

if TYPE_CHECKING:
    from vllm.distributed.eplb.eplb_state import EplbModelState

logger = init_logger(__name__)


def dead_dp_to_ep_ranks(
    dead_dp_ranks: set[int] | list[int],
    tp_size: int,
) -> set[int]:
    """Expand dead DP ranks to the corresponding dead EP ranks."""
    dead_ep: set[int] = set()
    for dp_rank in dead_dp_ranks:
        for tp_offset in range(tp_size):
            dead_ep.add(dp_rank * tp_size + tp_offset)
    return dead_ep


def strip_dead_columns(
    p2l: torch.Tensor,
    dead_ep_ranks: set[int],
    num_local: int,
) -> torch.Tensor:
    """Remove dead EP rank columns from physical_to_logical_map.

    Returns a new contiguous tensor with only the surviving ranks'
    columns, preserving order.
    """
    old_ep_size = p2l.shape[1] // num_local
    surviving_cols: list[int] = []
    for ep_r in range(old_ep_size):
        if ep_r not in dead_ep_ranks:
            start = ep_r * num_local
            surviving_cols.extend(range(start, start + num_local))
    col_idx = torch.tensor(surviving_cols, device=p2l.device)
    return p2l[:, col_idx].contiguous()


def rebuild_eplb_derived_maps(
    eplb_model_state: "EplbModelState",
) -> None:
    """Rebuild logical_to_physical_map and logical_replica_count
    from physical_to_logical_map.

    Call after any modification to physical_to_logical_map (compaction,
    reassignment) to keep the derived maps consistent. Modifies the
    tensors in-place so existing views (held by FusedMoE layers) see
    the updates.
    """
    p2l = eplb_model_state.physical_to_logical_map
    num_moe_layers, num_physical = p2l.shape
    eplb_model_state.logical_replica_count.zero_()
    eplb_model_state.logical_to_physical_map.fill_(-1)
    for layer_idx in range(num_moe_layers):
        for phys_idx in range(num_physical):
            lid = p2l[layer_idx, phys_idx].item()
            if lid >= 0:
                c = eplb_model_state.logical_replica_count[
                    layer_idx, lid
                ].item()
                eplb_model_state.logical_to_physical_map[
                    layer_idx, lid, c
                ] = phys_idx
                eplb_model_state.logical_replica_count[
                    layer_idx, lid
                ] += 1


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

    def _set_eplb_suppressed(self, suppressed: bool) -> None:
        self.worker.model_runner.eep_eplb_suppressed = suppressed
        ep_group = get_standby_ep_group() or get_ep_group()
        if ep_group.rank == 0:
            logger.info(
                "[Elastic EP] EPLB %s elastic scaling transition",
                "disabled during" if suppressed else "re-enabled after",
            )

    def load_model(self) -> None:
        (
            expanded_physical_to_logical,
            num_logical_experts,
            old_num_physical_experts,
        ) = self.receive_expert_mapping()
        num_physical_experts = expanded_physical_to_logical.shape[1]
        self.worker.parallel_config.eplb_config.num_redundant_experts = (
            num_physical_experts - num_logical_experts
        )
        self.worker.load_model(load_dummy_weights=True)
        self.worker.model_runner.setup_eplb_from_mapping(
            expanded_physical_to_logical, old_num_physical_experts
        )
        self._set_eplb_suppressed(True)

    def create_standby_groups(
        self,
        reconfig_request: ReconfigureDistributedRequest,
        dead_dp_ranks: set[int] | None = None,
    ) -> None:
        self.reconfig_request = reconfig_request
        new_dp_size = reconfig_request.new_data_parallel_size
        old_dp_size = get_dp_group().world_size
        world_size = self.worker.vllm_config.parallel_config.world_size
        new_world_size_across_dp = world_size * new_dp_size
        updated_config = copy.copy(self.worker.vllm_config)
        updated_config.parallel_config = copy.deepcopy(
            self.worker.vllm_config.parallel_config
        )
        updated_config.parallel_config.data_parallel_size = new_dp_size
        with set_current_vllm_config(updated_config):
            create_standby_groups(
                new_dp_size=new_dp_size,
                new_world_size_across_dp=new_world_size_across_dp,
                master_ip=reconfig_request.new_data_parallel_master_ip,
                coord_store_port=reconfig_request.coord_store_port,
                enable_eplb=updated_config.parallel_config.enable_eplb,
                dead_dp_ranks=dead_dp_ranks,
            )
        if new_dp_size > old_dp_size:
            self._set_eplb_suppressed(True)

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

        # Sender-receiver pairing: the first new_workers % old_dp_size
        # senders get (k+1) contiguous receivers, the rest get k
        # receivers.
        num_dst_per_sender = num_new_workers // old_dp_size
        remainder = num_new_workers % old_dp_size

        if dp_rank < remainder:
            recv_begin = dp_rank * (num_dst_per_sender + 1)
            recv_end = recv_begin + num_dst_per_sender + 1
        else:
            recv_begin = (
                remainder * (num_dst_per_sender + 1)
                + (dp_rank - remainder) * num_dst_per_sender
            )
            recv_end = recv_begin + num_dst_per_sender

        ranks_to_send = list(range(old_dp_size + recv_begin, old_dp_size + recv_end))

        model = self.worker.model_runner.get_model()
        for new_worker_rank in sorted(ranks_to_send):
            batch_transfer_weights(
                model=model,
                is_sender=True,
                peer_rank=new_worker_rank,
                dp_group=standby_dp_group,
                expert_weights=model.expert_weights,
            )
        torch.accelerator.synchronize()

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

    def _release_cuda_graphs(self) -> None:
        if isinstance(self.worker.model_runner.model, CUDAGraphWrapper):
            wrapper = self.worker.model_runner.model
            wrapper.concrete_cudagraph_entries = {}

        elif isinstance(self.worker.model_runner.model, UBatchWrapper):
            raise RuntimeError("DBO is not yet supported in elastic EP")

        torch.compiler.reset()
        with set_current_vllm_config(self.worker.vllm_config):
            reset_compile_wrapper(self.worker.model_runner.get_model())

        gc.collect()
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()

    def switch_and_remove(self) -> None:
        self._release_cuda_graphs()
        _replace_active_groups(world=None, dp=None, ep=None, eplb=None, node_count=None)

    def abort_and_switch(self) -> None:
        """Abort old NCCL groups and switch to standby groups.

        Used for fault-triggered scale-down where a peer rank has died
        and normal collective teardown would hang.  Identical to
        switch_and_prepare but uses abort instead of destroy.
        """
        old_dp_size = get_dp_group().world_size
        old_ep_size = get_ep_group().world_size
        logger.debug(
            "[Elastic EP] abort_and_switch: old EP %d → new EP %d",
            old_ep_size, old_ep_size - 1,
        )
        self._release_cuda_graphs()
        _abort_and_replace_active_groups(**pop_standby_groups())
        self._apply_new_config(old_dp_size, old_ep_size)

    def switch_and_prepare(self) -> None:
        old_dp_size = get_dp_group().world_size
        old_ep_size = get_ep_group().world_size
        self._release_cuda_graphs()
        _replace_active_groups(**pop_standby_groups())
        self._apply_new_config(old_dp_size, old_ep_size)

    def _apply_new_config(
        self, old_dp_size: int, old_ep_size: int
    ) -> None:
        """Apply reconfigure request to parallel config, MoE modules,
        EPLB state, and communication buffers after group replacement."""
        parallel_config = self.worker.vllm_config.parallel_config
        reconfig_request = self.reconfig_request
        assert reconfig_request is not None
        new_dp_size = reconfig_request.new_data_parallel_size
        new_ep_size = get_ep_group().world_size

        logger.debug(
            "[Elastic EP] _apply_new_config: DP %d→%d, EP %d→%d",
            old_dp_size, new_dp_size, old_ep_size, new_ep_size,
        )

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
            tp_size = get_tp_group().world_size
            is_sequence_parallel = parallel_config.use_sequence_parallel_moe
            sp_size = tp_size if is_sequence_parallel else 1
            module.moe_parallel_config = FusedMoEParallelConfig.make(
                tp_size_=tp_size,
                pcp_size_=get_pcp_group().world_size,
                dp_size_=get_dp_group().world_size,
                sp_size_=sp_size,
                vllm_parallel_config=parallel_config,
            )
            module.moe_config.moe_parallel_config = module.moe_parallel_config
            # Rebuild _expert_map for the new EP size/rank. Without this,
            # the map has the old size (e.g. 96 entries for 3 ranks) and
            # indexing with new expert IDs causes out-of-bounds access.
            if hasattr(module, 'update_expert_map') and module._expert_map is not None:
                module.update_expert_map()

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

            # Trim physical_to_logical_map to match the new EP size.
            # For fault-triggered scale-down, dead rank columns are in
            # the middle — strip them. For graceful, the highest columns
            # are removed — simple slice works too.
            dead_dp = set(
                reconfig_request.dead_dp_ranks
            ) if reconfig_request.dead_dp_ranks else set()
            if dead_dp:
                dead_ep = dead_dp_to_ep_ranks(
                    dead_dp, get_tp_group().world_size
                )
                eplb_model_state.physical_to_logical_map = (
                    strip_dead_columns(
                        old_physical_to_logical, dead_ep, num_local_experts
                    )
                )
            else:
                eplb_model_state.physical_to_logical_map = (
                    old_physical_to_logical[:, :num_physical_experts]
                )

            rebuild_eplb_derived_maps(eplb_model_state)

        model = self.worker.model_runner.get_model()
        model.expert_weights = []
        with set_current_vllm_config(self.worker.vllm_config):
            model.set_eplb_state(
                eplb_model_state.expert_load_pass,
                eplb_model_state.logical_to_physical_map,
                eplb_model_state.logical_replica_count,
            )
            eplb_state._init_should_record_tensor(model)
            model.update_physical_experts_metadata(
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_local_experts,
            )
            # Force re-creation of the modular kernel (and all2all manager)
            # for the new EP size by resetting quant_method to base
            for module in moe_modules:
                if hasattr(module.quant_method, "old_quant_method"):
                    module._replace_quant_method(module.quant_method.old_quant_method)
            prepare_communication_buffer_for_model(self.worker.model_runner.model)

        logger.debug(
            "[Elastic EP] _apply_new_config: p2l=%s, "
            "num_physical=%d, num_redundant=%d",
            list(eplb_model_state.physical_to_logical_map.shape),
            num_physical_experts,
            parallel_config.eplb_config.num_redundant_experts,
        )

        eplb_model_state.communicator = create_eplb_communicator(
            group_coordinator=get_eplb_group(),
            backend=parallel_config.eplb_config.communicator,
            expert_weights=model.expert_weights[0],
        )

        if (
            self.worker.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
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

        multi_block_table = self.worker.model_runner.input_batch.block_table
        saved_block_tables: list[tuple[torch.Tensor, torch.Tensor]] = []
        for bt in multi_block_table.block_tables:
            saved_block_tables.append(
                (bt.block_table.gpu.clone(), bt.block_table.cpu.clone())
            )
        multi_block_table.clear()

        # For fault-triggered scale-down, reassign missing experts
        # BEFORE warmup. The warmup does a real forward pass (for CUDA
        # graph capture), which will crash if logical experts are missing
        # from the p2l map (router produces expert IDs that map to -1).
        reconfig_request = self.reconfig_request
        if (reconfig_request and reconfig_request.dead_dp_ranks
                and new_dp_size < old_dp_size):
            self.reassign_missing_experts()

        unlock_workspace()
        self.worker.compile_or_warm_up_model()
        lock_workspace()

        for bt, (saved_gpu, saved_cpu) in zip(
            multi_block_table.block_tables, saved_block_tables
        ):
            bt.block_table.gpu.copy_(saved_gpu)
            bt.block_table.cpu.copy_(saved_cpu)
        if new_dp_size < old_dp_size:
            self._set_eplb_suppressed(False)

    def _perform_eplb_reshuffle(
        self, rank_mapping: dict[int, int] | None = None
    ) -> None:
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding...")

        eplb_state = self.worker.model_runner.eplb_state
        assert eplb_state is not None

        model_config = self.worker.model_runner.model_config
        eplb_model_state = eplb_state.model_states[model_config.compute_hash()]
        is_async_enabled = eplb_state.is_async
        eplb_state.is_async = False
        if rank_mapping is None:
            eplb_state.rearrange()
        else:
            eplb_state.rearrange(rank_mapping=rank_mapping)
        # NOTE(yongji): check whether we need to synchronize here
        torch.accelerator.synchronize()
        # reset expert_rearrangement_step to ensure all ranks are synchronized
        eplb_state.expert_rearrangement_step = 0
        eplb_state.num_valid_physical_experts = (
            eplb_model_state.physical_to_logical_map.shape[1]
        )
        eplb_state.is_async = is_async_enabled
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed")

    def perform_eplb_reshuffle(self) -> None:
        self._perform_eplb_reshuffle()
        self._set_eplb_suppressed(False)

    def reassign_missing_experts(self) -> None:
        """Reassign logical experts that lost all replicas after a fault.

        After a fault-triggered scale-down, _apply_new_config has already
        compacted physical_to_logical_map (stripped dead rank columns).
        This method:
        1. Identifies which logical experts have zero replicas.
        2. Replaces the most-redundant replicas with missing experts.
        3. Rebuilds the derived EPLB maps in-place.
        4. Transfers actual expert weights via NCCL P2P.

        All tensors are already sized for the new EP topology when this
        method runs (p2l, expert_load_pass, expert_load_window all have
        new_ep_size * num_local columns).
        """
        eplb_state = self.worker.model_runner.eplb_state
        if eplb_state is None:
            return

        model_config = self.worker.model_runner.model_config
        eplb_model_state = eplb_state.model_states[model_config.compute_hash()]
        p2l = eplb_model_state.physical_to_logical_map
        num_moe_layers = p2l.shape[0]
        num_physical = p2l.shape[1]
        num_logical = eplb_model_state.logical_replica_count.shape[1]

        ep_group = get_ep_group()
        ep_rank = ep_group.rank
        new_ep_size = ep_group.world_size

        # Save the current map — this reflects the true state of expert
        # weights in GPU memory (before reassignment modifies it).
        old_p2l = p2l.clone()

        logger.debug(
            "[Elastic EP] reassign_missing_experts: p2l shape=%s, "
            "%d logical experts, %d physical slots",
            list(p2l.shape), num_logical, num_physical,
        )

        # --- 1 & 2. Identify missing experts and reassign -----------------
        all_logical = set(range(num_logical))
        any_reassigned = False

        for layer_idx in range(num_moe_layers):
            layer_map = p2l[layer_idx]

            replica_count: dict[int, int] = {}
            global_redundant: list[tuple[int, int]] = []
            for phys_idx in range(num_physical):
                lid = layer_map[phys_idx].item()
                if lid >= 0:
                    replica_count[lid] = replica_count.get(lid, 0) + 1

            missing = sorted(all_logical - set(replica_count.keys()))
            if not missing:
                continue

            any_reassigned = True

            for phys_idx in range(num_physical):
                lid = layer_map[phys_idx].item()
                if lid >= 0 and replica_count.get(lid, 0) > 1:
                    global_redundant.append((replica_count[lid], phys_idx))
            global_redundant.sort(reverse=True)

            slot_iter = iter(global_redundant)
            for logical_id in missing:
                # Find a redundant slot whose donor still has >1 replica.
                placed = False
                while True:
                    candidate = next(slot_iter, None)
                    if candidate is None:
                        raise RuntimeError(
                            f"[Fault Tolerance] Layer {layer_idx}: no "
                            f"redundant slot available to place missing "
                            f"logical expert {logical_id}. This should "
                            f"not happen — the capacity check in "
                            f"_on_engine_process_died should have "
                            f"prevented scale-down."
                        )
                    _, global_slot = candidate
                    old_lid = layer_map[global_slot].item()
                    if replica_count.get(old_lid, 0) > 1:
                        p2l[layer_idx, global_slot] = logical_id
                        replica_count[old_lid] -= 1
                        placed = True
                        break
                if not placed:
                    raise RuntimeError(
                        f"[Fault Tolerance] Layer {layer_idx}: failed "
                        f"to place missing logical expert {logical_id}."
                    )

        if not any_reassigned:
            if ep_rank == 0:
                logger.info(
                    "[Fault Tolerance] All %d logical experts have at "
                    "least one replica in every layer — no "
                    "reassignment needed.",
                    num_logical,
                )
            return

        if ep_rank == 0:
            logger.info(
                "[Fault Tolerance] Replaced redundant expert slots "
                "with missing experts across %d layers.",
                num_moe_layers,
            )

        # --- 3. Rebuild derived maps in-place --------------------------------
        rebuild_eplb_derived_maps(eplb_model_state)

        # --- 4. Transfer actual expert weights via NCCL P2P ---------------
        # old_p2l reflects what's actually in GPU memory (pre-reassignment).
        # p2l reflects the desired state (post-reassignment).
        # rearrange_expert_weights_inplace sees the diff and transfers
        # weights from ranks that hold the needed expert.
        #
        # Split experts into two groups:
        # - reachable: have a surviving replica in old_p2l → NCCL P2P
        # - unreachable: no surviving replica → reload from disk
        old_present = set(old_p2l[old_p2l >= 0].tolist())
        new_required = set(p2l[p2l >= 0].tolist())
        unreachable = new_required - old_present

        from vllm.distributed.eplb.rebalance_execute import (
            rearrange_expert_weights_inplace,
        )

        model = self.worker.model_runner.get_model()
        ep_group_pg = get_ep_group().device_group
        rearrange_expert_weights_inplace(
            old_global_expert_indices=old_p2l,
            new_global_expert_indices=p2l,
            expert_weights=model.expert_weights,
            ep_group=ep_group_pg,
            communicator=eplb_model_state.communicator,
        )

        # Reload unreachable experts from disk.
        if unreachable:
            if ep_rank == 0:
                logger.info(
                    "[Fault Tolerance] Reloading %d experts from disk: %s",
                    len(unreachable),
                    sorted(unreachable),
                )
            self._reload_expert_weights_from_disk(unreachable)

        if ep_rank == 0:
            logger.info(
                "[Fault Tolerance] Expert weight transfer complete. "
                "%d physical experts across %d EP ranks.",
                num_physical, new_ep_size,
            )

    def _reload_expert_weights_from_disk(
        self, unreachable_experts: set[int]
    ) -> None:
        """Reload expert weights from the model checkpoint for experts
        that have no surviving replica in GPU memory.

        Uses the existing model.load_weights() pipeline. The FusedMoE
        weight_loader consults logical_to_physical_map (already updated
        by reassign_missing_experts) to route each expert's weights to
        the correct reassigned physical slot.
        """
        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )
        from vllm.model_executor.model_loader.ep_weight_filter import (
            parse_expert_id,
        )

        model = self.worker.model_runner.get_model()
        model_config = self.worker.model_runner.model_config
        load_config = self.worker.vllm_config.load_config

        loader = DefaultModelLoader(load_config)
        all_weights = loader.get_all_weights(model_config, model)
        filtered = (
            (name, tensor)
            for name, tensor in all_weights
            if parse_expert_id(name) in unreachable_experts
        )
        model.load_weights(filtered)

    def abort_eplb_group_for_dead_ranks(self) -> None:
        """Abort the EPLB NCCL process group so that pending/future NCCL
        ops involving dead ranks don't hang indefinitely.  After this call
        the old EPLB communicator is unusable — a new standby group must be
        created before any further EPLB operations."""
        from vllm.distributed.parallel_state import get_eplb_group
        eplb_group = get_eplb_group()
        logger.info(
            "[Elastic EP] Aborting old EPLB process group (rank %d) "
            "to unblock operations involving dead ranks.",
            eplb_group.rank,
        )
        eplb_group.device_group.abort()

    def perform_scale_down_eplb_reshuffle(
        self,
        new_dp_size: int,
        dead_dp_ranks: list[int] | None = None,
    ) -> None:
        self._set_eplb_suppressed(True)
        parallel_config = self.worker.vllm_config.parallel_config
        tp_size = parallel_config.tensor_parallel_size
        old_ep_size = parallel_config.data_parallel_size * tp_size
        new_ep_size = new_dp_size * tp_size

        if dead_dp_ranks:
            # Fault-triggered: map dead ranks' experts to -1, compact
            # surviving ranks to contiguous 0..new_ep_size-1.
            dead_ep = dead_dp_to_ep_ranks(dead_dp_ranks, tp_size)
            rank_mapping = {}
            new_rank = 0
            for old_ep_rank in range(old_ep_size):
                if old_ep_rank in dead_ep:
                    rank_mapping[old_ep_rank] = -1
                else:
                    rank_mapping[old_ep_rank] = new_rank
                    new_rank += 1
        else:
            # Graceful: remove the highest-ranked engines.
            rank_mapping = {
                old_ep_rank: old_ep_rank if old_ep_rank < new_ep_size else -1
                for old_ep_rank in range(old_ep_size)
            }
        self._perform_eplb_reshuffle(rank_mapping=rank_mapping)

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
        torch.accelerator.synchronize()

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
