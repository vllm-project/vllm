# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, cast

import torch

from vllm.config import set_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_eplb_group,
    get_tp_group,
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.sentinel.eplb_redistribute import (
    check_redundancy_sufficient,
    compute_dead_ep_ranks,
    mark_dead_expert_slots_inplace,
    rebuild_logical_expert_maps,
    rebuild_model_expert_maps,
    redistribute_expert_placement,
    reload_experts_from_disk,
    reset_eplb_async_state,
    sync_num_dispatchers_for_nixl_ep,
)

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

# All2all backends that support fault-tolerant timeout + rank masking,
# required for FT under DP+EP MoE deployments.
FT_BACKEND_SET = frozenset({"deepep_low_latency", "nixl_ep"})


def _reinit_cpu_group(
    group: "GroupCoordinator", master_ip: str, port: int, rank: int, size: int
) -> None:
    """Destroy and rebuild a group's Gloo cpu_group in place."""
    stateless_destroy_torch_distributed_process_group(group.cpu_group)
    group.cpu_group = stateless_init_torch_distributed_process_group(
        master_ip, port, rank, size, backend="gloo"
    )


class WorkerSentinel:
    """Holds FT state for a single worker (mask tensors, DP config).

    Methods are called via collective_rpc from EngineCoreSentinel.
    """

    def __init__(self, worker: "Worker"):
        self.worker = worker
        all2all_backend = worker.parallel_config.all2all_backend
        if all2all_backend not in FT_BACKEND_SET:
            raise ValueError(
                f"Fault tolerance requires an FT-capable all2all backend "
                f"(one of {sorted(FT_BACKEND_SET)}), but got '{all2all_backend}'."
            )

    def handle_command(self, ft_request: FaultToleranceRequest):
        """Dispatch an FT command by instruction name."""
        with set_current_vllm_config(self.worker.vllm_config):
            return run_method(self, ft_request.instruction, (ft_request,), {})

    def retry(self, ft_request: FaultToleranceRequest):
        torch.accelerator.synchronize()
        params = ft_request.params
        # After a scale_down the DP master may be re-elected and the rebuilt
        # gloo group is dense over the alive slots.
        master_ip = params["dp_master_ip"]
        dp_group_rank = params["dp_group_rank"]
        dp_group_size = params["dp_group_size"]
        self.worker.parallel_config.data_parallel_master_ip = master_ip
        self._clean_worker_state()
        reset_eplb_async_state(self.worker.model_runner)
        if self.worker.parallel_config.data_parallel_size > 1:
            mgr = get_ep_all2all_manager()
            mgr.clean_buffers()
            # clean_buffers wiped the mask; replay masks for the cumulative dead set.
            tp_size = self.worker.parallel_config.tensor_parallel_size
            for ep_rank in compute_dead_ep_ranks(params["dead_dp_ranks"], tp_size):
                mgr.update_mask(ep_rank, masked=True)
            world_size = self.worker.parallel_config.world_size
            port = params["new_stateless_dp_group_ports"][self.worker.rank % world_size]
            _reinit_cpu_group(
                get_dp_group(), master_ip, port, dp_group_rank, dp_group_size
            )
            get_dp_group().dead_dp_ranks = set(params["dead_dp_ranks"])
            if (
                self.worker.parallel_config.enable_eplb
                and not self.worker.model_runner.eep_eplb_suppressed
            ):
                self._reinit_eplb_groups(params, master_ip)
        if self.worker.parallel_config.tensor_parallel_size > 1:
            # The per-step TP barrier (dp_utils) leaves the group in a
            # timed-out state on fault; rebuild it for a clean slate.
            tp_group = get_tp_group()
            _reinit_cpu_group(
                tp_group,
                "127.0.0.1",
                params["new_tp_group_port"],
                tp_group.rank_in_group,
                tp_group.world_size,
            )

    def scale_down(self, ft_request: FaultToleranceRequest):
        model_runner = self.worker.model_runner
        eplb_config = self.worker.parallel_config.eplb_config
        if model_runner.eplb_state is None or eplb_config.num_redundant_experts <= 0:
            raise ValueError(
                "[FT] scale_down requires EPLB with num_redundant_experts > 0 "
                "to re-host the dead rank's experts."
            )
        tp_size = self.worker.parallel_config.tensor_parallel_size
        dead_dp_ranks = ft_request.params["dead_dp_ranks"]
        dead_ep_ranks = compute_dead_ep_ranks(dead_dp_ranks, tp_size)
        eplb_model_state = self._eplb_model_state()
        ep_world_size = get_ep_group().world_size

        check_redundancy_sufficient(
            eplb_model_state.logical_replica_count.shape[1],
            eplb_model_state.physical_to_logical_map.shape[1] // ep_world_size,
            ep_world_size,
            dead_ep_ranks,
        )
        # Suppress EPLB async rebalancing before retry so it skips the EPLB
        # group reinit; placement is fixed by the redistribution below.
        model_runner.eep_eplb_suppressed = True
        self.retry(ft_request)

        self._redistribute_experts(dead_ep_ranks)
        sync_num_dispatchers_for_nixl_ep(
            model_runner.model,
            self.worker.parallel_config.all2all_backend,
            dead_ep_ranks,
        )

        logger.info(
            "[FT] Worker scale_down complete: dp_group_size=%d, "
            "dp_group_rank=%d, dead_ep_ranks=%s, eplb_suppressed=True",
            ft_request.params["dp_group_size"],
            ft_request.params["dp_group_rank"],
            sorted(dead_ep_ranks),
        )

    def query_mask(self, ft_request: FaultToleranceRequest) -> dict:
        """Query the mask on a side stream."""
        with torch.cuda.stream(torch.cuda.Stream()):
            mask = get_ep_all2all_manager().query_active_mask()
            return {"mask": mask.tolist()}

    def _reinit_eplb_groups(self, params: dict, master_ip: str) -> None:
        """Reinit the EP/EPLB Gloo groups and refresh the EPLB
        communicator's cpu_group reference."""
        for port_key, get_group in [
            ("new_ep_group_port", get_ep_group),
            ("new_eplb_group_port", get_eplb_group),
        ]:
            port = params[port_key]
            group = get_group()
            _reinit_cpu_group(
                group, master_ip, port, group.rank_in_group, group.world_size
            )
            logger.info("[FT] Reinited %s Gloo group on port %d", port_key, port)

        eplb_state = self.worker.model_runner.eplb_state
        assert eplb_state is not None
        eplb_group = get_eplb_group()
        for ms in eplb_state.model_states.values():
            if hasattr(ms.communicator, "_cpu_group"):
                ms.communicator._cpu_group = eplb_group.cpu_group

    def _eplb_model_state(self):
        model_runner = self.worker.model_runner
        assert model_runner.eplb_state is not None
        return model_runner.eplb_state.model_states[
            model_runner.model_config.compute_hash()
        ]

    def _redistribute_experts(self, dead_ep_ranks: set[int]) -> None:
        """One-shot expert redistribution after scale-down."""
        model_runner = self.worker.model_runner
        eplb_model_state = self._eplb_model_state()

        p2l = eplb_model_state.physical_to_logical_map
        l2p = eplb_model_state.logical_to_physical_map
        lrc = eplb_model_state.logical_replica_count
        num_logical = lrc.shape[1]
        ep_world_size = get_ep_group().world_size
        num_local_experts = p2l.shape[1] // ep_world_size

        mark_dead_expert_slots_inplace(p2l, dead_ep_ranks, num_local_experts)
        reassignments = redistribute_expert_placement(
            p2l, num_logical, num_local_experts
        )
        rebuild_logical_expert_maps(p2l, l2p, lrc)
        rebuild_model_expert_maps(model_runner.model, p2l)

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

    def _clean_worker_state(self):
        model_runner = self.worker.model_runner
        model_runner.execute_model_state = None
        if self.worker.use_v2_model_runner:
            runner = cast("GPUModelRunnerV2", model_runner)
            for req_id in list(runner.req_states.req_id_to_index):
                runner._remove_request(req_id)
        else:
            model_runner.kv_connector_output = None

            input_batch = model_runner.input_batch
            cached_req_ids = list(input_batch.req_id_to_index)
            for req_id in cached_req_ids:
                model_runner.requests.pop(req_id, None)
                model_runner.num_prompt_logprobs.pop(req_id, None)
                input_batch.remove_request(req_id)

            input_batch.condense()
            input_batch.refresh_metadata()
            input_batch.req_prompt_embeds.clear()
