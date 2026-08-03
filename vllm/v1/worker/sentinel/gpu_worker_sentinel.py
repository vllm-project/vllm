# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, cast

import torch

from vllm.config import set_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    reinit_gloo_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.sentinel.eplb_redistribute import (
    compute_dead_ep_ranks,
    mark_dead_expert_slots_inplace,
    rebuild_logical_expert_maps,
    redistribute_expert_placement,
    refresh_eplb_communicator_group,
    reinit_eplb_gloo_groups,
    reload_experts_from_disk,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

# All2all backends that support fault-tolerant timeout + rank masking,
# required for FT under DP+EP MoE deployments.
FT_BACKEND_SET = frozenset({"deepep_low_latency", "nixl_ep"})


class WorkerSentinel:
    """Holds FT state for a single worker (mask tensors, DP config).

    Methods are called via collective_rpc from EngineCoreSentinel.
    """

    def __init__(self, worker: "Worker"):
        self.worker = worker
        self.dp_rank = worker.parallel_config.data_parallel_rank
        self.dp_size = worker.parallel_config.data_parallel_size
        self.data_parallel_master_ip = worker.parallel_config.data_parallel_master_ip
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
        self.data_parallel_master_ip = params["dp_master_ip"]
        self.worker.parallel_config.data_parallel_master_ip = (
            self.data_parallel_master_ip
        )
        self._clean_worker_state()
        self._reset_eplb_async_state()
        if self.dp_size > 1:
            get_ep_all2all_manager().clean_buffers()
            world_size = self.worker.parallel_config.world_size
            port = params["new_stateless_dp_group_ports"][self.worker.rank % world_size]
            reinit_gloo_group(
                get_dp_group(),
                self.data_parallel_master_ip,
                port,
                self.dp_rank,
                self.dp_size,
            )
            reinit_eplb_gloo_groups(params, self.data_parallel_master_ip)
            refresh_eplb_communicator_group(self.worker.model_runner)

    def scale_down(self, ft_request: FaultToleranceRequest):
        torch.accelerator.synchronize()
        params = ft_request.params
        self.data_parallel_master_ip = params["dp_master_ip"]
        self.worker.parallel_config.data_parallel_master_ip = (
            self.data_parallel_master_ip
        )
        removed_dp_ranks = params["removed_dp_ranks"]
        new_dp_size = params["new_dp_size"]
        new_dp_rank = params["new_dp_rank"]
        tp_size = self.worker.parallel_config.tensor_parallel_size

        self._clean_worker_state()
        mgr = get_ep_all2all_manager()
        mgr.clean_buffers()

        dead_ep_ranks = compute_dead_ep_ranks(removed_dp_ranks, tp_size)
        for ep_rank in sorted(dead_ep_ranks):
            mgr.update_mask(ep_rank, masked=True)

        self._redistribute_experts(dead_ep_ranks)
        self._sync_num_dispatchers_for_nixl_ep(dead_ep_ranks)

        world_size = self.worker.parallel_config.world_size
        port = params["new_stateless_dp_group_ports"][self.worker.rank % world_size]
        reinit_gloo_group(
            get_dp_group(),
            self.data_parallel_master_ip,
            port,
            new_dp_rank,
            new_dp_size,
        )
        self.worker.parallel_config.data_parallel_size = new_dp_size
        self.worker.parallel_config.data_parallel_rank = new_dp_rank
        self.dp_rank = new_dp_rank
        self.dp_size = new_dp_size

        if self.worker.use_v2_model_runner:
            runner = cast("GPUModelRunnerV2", self.worker.model_runner)
            runner.dp_size = new_dp_size
            runner.dp_rank = new_dp_rank

        self.worker.model_runner.eep_eplb_suppressed = True
        self._reset_eplb_async_state()

        logger.info(
            "[FT] Worker scale_down complete: dp_size=%d, dp_rank=%d, "
            "dead_ep_ranks=%s, eplb_suppressed=True",
            new_dp_size,
            new_dp_rank,
            sorted(dead_ep_ranks),
        )

    def query_mask(self, ft_request: FaultToleranceRequest) -> dict:
        """Return the current all2all active mask from the FT backend."""
        mask = get_ep_all2all_manager().query_active_mask()
        return {"mask": mask.tolist()}

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

        mark_dead_expert_slots_inplace(p2l, dead_ep_ranks, num_local_experts)
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
            # v2 runner wraps FusedMoE in MoERunner; the expert map lives on
            # routed_experts there, on the layer itself otherwise.
            routed = getattr(layer, "routed_experts", layer)
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

    def _sync_num_dispatchers_for_nixl_ep(self, dead_ep_ranks: set[int]) -> None:
        """Rewrite each MoE layer's num_dispatchers to the nixl_ep kernel's
        active_rank_bound (= highest surviving EP rank + 1) after masking.

        The kernel sizes combine output as active_rank_bound * max_tokens and
        asserts the width matches; vLLM cache num_dispatchers as the
        original EP world size, so masking the highest rank desyncs the two.
        DeepEP-LL keeps a fixed num_ranks-wide layout and needs no sync.
        """
        if self.worker.parallel_config.all2all_backend != "nixl_ep":
            return

        ep_world_size = get_ep_group().world_size
        surviving = sorted(set(range(ep_world_size)) - dead_ep_ranks)
        if not surviving:
            return
        active_rank_bound = surviving[-1] + 1

        moe_layers = getattr(self.worker.model_runner.model, "moe_layers", None)
        if moe_layers is None:
            return

        for layer in moe_layers:
            routed = getattr(layer, "routed_experts", layer)
            quant_method = getattr(routed, "quant_method", None)
            moe_kernel = getattr(quant_method, "moe_kernel", None)
            if moe_kernel is None or moe_kernel.is_monolithic:
                continue
            pf = moe_kernel.prepare_finalize
            experts = moe_kernel.fused_experts
            if hasattr(pf, "num_dispatchers_"):
                pf.num_dispatchers_ = active_rank_bound
            if getattr(experts, "num_dispatchers", None) is not None:
                experts.num_dispatchers = active_rank_bound

        logger.info(
            "[FT] Synced num_dispatchers to active_rank_bound=%d, dead_ep_ranks=%s",
            active_rank_bound,
            sorted(dead_ep_ranks),
        )

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
