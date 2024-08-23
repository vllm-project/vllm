import asyncio
from typing import List, Optional

import vllm.envs as envs
from vllm.executor.ray_gpu_executor import RayGPUExecutor, RayGPUExecutorAsync
from vllm.executor.xpu_executor import XPUExecutor
from vllm.logger import init_logger
from vllm.utils import get_vllm_instance_id, make_async

logger = init_logger(__name__)


class RayXPUExecutor(RayGPUExecutor, XPUExecutor):

    def _get_env_vars_to_be_updated(self):
        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids",
                                                    use_dummy_driver=True)

        VLLM_INSTANCE_ID = get_vllm_instance_id()

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [({
            "VLLM_INSTANCE_ID":
            VLLM_INSTANCE_ID,
            "VLLM_TRACE_FUNCTION":
            str(envs.VLLM_TRACE_FUNCTION),
        }, ) for (_, _) in worker_node_and_gpu_ids]
        return all_args_to_update_environment_variables


class RayXPUExecutorAsync(RayXPUExecutor, RayGPUExecutorAsync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_exec_method = make_async(self.driver_worker.execute_method)
        self.pp_locks: Optional[List[asyncio.Lock]] = None
