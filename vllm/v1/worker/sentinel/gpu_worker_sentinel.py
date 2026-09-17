# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, cast

import torch

from vllm.config import set_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.serial_utils import run_method

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
        self._clean_worker_state()
        if self.dp_size > 1:
            get_ep_all2all_manager().clean_buffers()
            old_cpu_group = get_dp_group().cpu_group
            stateless_destroy_torch_distributed_process_group(old_cpu_group)
            world_size = self.worker.parallel_config.world_size
            port = params["new_stateless_dp_group_ports"][self.worker.rank % world_size]
            get_dp_group().cpu_group = stateless_init_torch_distributed_process_group(
                self.data_parallel_master_ip,
                port,
                self.dp_rank,
                self.dp_size,
                backend="gloo",
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
