# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.config import set_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.distributed.elastic_ep.ft_eplb_redistribute import (
    compute_dead_ep_ranks,
    mark_dead_columns_inplace,
    rebuild_logical_expert_maps,
    redistribute_expert_placement,
    reload_experts_from_disk,
)
from vllm.logger import init_logger
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.serial_utils import run_method

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


class WorkerSentinel:
    """Holds FT state for a single worker (mask tensors, DP config).

    Methods are called via collective_rpc from EngineCoreSentinel.
    """

    def __init__(self, worker: "Worker", device: torch.device):
        self.worker = worker
        self.device = device
        self.dp_rank = worker.parallel_config.data_parallel_rank
        self.dp_size = worker.parallel_config.data_parallel_size
        self.data_parallel_master_ip = worker.parallel_config.data_parallel_master_ip

    def handle_command(self, ft_request: FaultToleranceRequest):
        """Dispatch an FT command by instruction name."""
        with set_current_vllm_config(self.worker.vllm_config):
            return run_method(self, ft_request.instruction, (ft_request,), {})

    def retry(self, ft_request: FaultToleranceRequest):
        torch.accelerator.synchronize()
        params = ft_request.params
        self._clean_worker_state()
        self._reset_eplb_async_state()
        if self.dp_size > 1:
            self._reinit_gloo_group(
                get_dp_group(),
                params["new_stateless_dp_group_port"],
                self.dp_rank,
                self.dp_size,
            )
            self._get_all2all_manager().clean_buffers()

    def _get_all2all_manager(self):
        comm = get_ep_group().device_communicator
        assert comm and comm.all2all_manager
        return comm.all2all_manager

    def scale_down(self, ft_request: FaultToleranceRequest):
        params = ft_request.params
        port = params["new_stateless_dp_group_port"]
        removed_dp_ranks = params["removed_dp_ranks"]
        new_dp_size = params["new_dp_size"]
        new_dp_rank = params["new_dp_rank"]
        tp_size = self.worker.parallel_config.tensor_parallel_size

        torch.accelerator.synchronize()
        self._clean_worker_state()
        comm = get_ep_group().device_communicator
        assert comm and comm.all2all_manager
        mgr = comm.all2all_manager
        mgr.clean_buffers()

        dead_ep_ranks = compute_dead_ep_ranks(removed_dp_ranks, tp_size)
        for ep_rank in sorted(dead_ep_ranks):
            mgr.update_mask(ep_rank, masked=True)

        self._redistribute_experts(dead_ep_ranks)

        self._reinit_gloo_group(get_dp_group(), port, new_dp_rank, new_dp_size)
        self.worker.parallel_config.data_parallel_size = new_dp_size
        self.worker.parallel_config.data_parallel_rank = new_dp_rank
        self.dp_rank = new_dp_rank
        self.dp_size = new_dp_size

        self.worker.model_runner.eep_eplb_suppressed = True
        self._reset_eplb_async_state()

        logger.info(
            "[FT] Worker scale_down complete: dp_size=%d, dp_rank=%d, "
            "dead_ep_ranks=%s, eplb_suppressed=True",
            new_dp_size,
            new_dp_rank,
            sorted(dead_ep_ranks),
        )

    def _redistribute_experts(self, dead_ep_ranks: set[int]) -> None:
        """One-shot expert redistribution after scale-down."""
        model_runner = self.worker.model_runner
        assert model_runner.eplb_state is not None
        eplb_model_state = model_runner.eplb_state.model_states[
            model_runner.model_config.compute_hash()
        ]

        p2l = eplb_model_state.physical_to_logical_map
        l2p = eplb_model_state.logical_to_physical_map
        lrc = eplb_model_state.logical_replica_count
        num_logical = lrc.shape[1]
        ep_world_size = get_ep_group().world_size
        num_local_experts = p2l.shape[1] // ep_world_size

        surviving_slots = (ep_world_size - len(dead_ep_ranks)) * num_local_experts
        if surviving_slots < num_logical:
            raise RuntimeError(
                f"[FT] Cannot redistribute: {surviving_slots} surviving slots "
                f"< {num_logical} logical experts. "
            )

        mark_dead_columns_inplace(p2l, dead_ep_ranks, num_local_experts)
        reassignments = redistribute_expert_placement(
            p2l, num_logical, num_local_experts
        )
        rebuild_logical_expert_maps(p2l, l2p, lrc)
        self._rebuild_expert_maps(p2l)

        if reassignments:
            reload_experts_from_disk(
                model_runner.model,
                self.worker.vllm_config,
                reassignments,
            )

        logger.info(
            "[FT] Expert redistribution: num_logical=%d, "
            "ep_world_size=%d, reassignments=%d",
            num_logical,
            ep_world_size,
            len(reassignments),
        )

    def _rebuild_expert_maps(self, p2l: torch.Tensor) -> None:
        """Rebuild each FusedMoE layer's _expert_map from p2l table."""
        model = self.worker.model_runner.model
        moe_layers = getattr(model, "moe_layers", None)
        if moe_layers is None:
            return

        ep_rank = get_ep_group().rank_in_group
        for layer_idx, layer in enumerate(moe_layers):
            expert_map = getattr(layer, "_expert_map", None)
            if expert_map is None:
                routed = getattr(layer, "routed_experts", None)
                if routed is not None:
                    expert_map = getattr(routed, "_expert_map", None)
            if expert_map is None:
                continue
            num_local = p2l.shape[1] // layer.moe_config.moe_parallel_config.ep_size
            local_start = ep_rank * num_local
            p2l_row = p2l[layer_idx].cpu()

            new_map = torch.full_like(expert_map, -1)
            for local_idx in range(num_local):
                lid = int(p2l_row[local_start + local_idx].item())
                if 0 <= lid < new_map.shape[0]:
                    new_map[lid] = local_idx
            expert_map.copy_(new_map)

    def _reset_eplb_async_state(self) -> None:
        """Clear stale EPLB async state after fault or scale-down."""
        eplb_state = getattr(self.worker.model_runner, "eplb_state", None)
        if eplb_state is None:
            return

        for ms in eplb_state.model_states.values():
            ms.rebalanced = False
            ms.pending_result = None
            ms.expert_load_pass.zero_()
            ms.expert_load_window.zero_()

        eplb_state.expert_rearrangement_step = 0
        eplb_state.expert_load_window_step = 0

    def _reinit_gloo_group(
        self,
        group_coordinator,
        port: int,
        rank: int,
        size: int,
    ) -> None:
        """Destroy old cpu_group and create a new gloo group."""
        old = group_coordinator.cpu_group
        if old is not None:
            stateless_destroy_torch_distributed_process_group(old)
        group_coordinator.cpu_group = stateless_init_torch_distributed_process_group(
            self.data_parallel_master_ip,
            port,
            rank,
            size,
            backend="gloo",
        )

    def _clean_worker_state(self):
        self.worker.model_runner.execute_model_state = None
        self.worker.model_runner.kv_connector_output = None
        input_batch = self.worker.model_runner.input_batch
        cached_req_ids = input_batch.req_id_to_index.keys()
        for req_id in list(cached_req_ids):
            input_batch.remove_request(req_id)
        input_batch.condense()
        input_batch.refresh_metadata()
        input_batch.req_prompt_embeds.clear()
