from typing import List, Optional, Tuple, Union

import torch

import vllm.envs as envs
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         SpeculativeConfig, VisionLanguageConfig)
from vllm.distributed import (broadcast_object_list,
                              tensor_model_parallel_all_gather)
from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.gpu_executor import GPUExecutor
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, PoolerOutput, SamplerOutput
from vllm.utils import make_async

logger = init_logger(__name__)

# A map between the device type (in device config) to its worker module.
DEVICE_TO_WORKER_MODULE_MAP = {
    "cuda": "vllm.worker.worker",
    "neuron": "vllm.worker.neuron_worker",
}


class TorchrunGPUExecutor(GPUExecutor):

    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig,
                 parallel_config: ParallelConfig,
                 scheduler_config: SchedulerConfig,
                 device_config: DeviceConfig, load_config: LoadConfig,
                 lora_config: Optional[LoRAConfig],
                 vision_language_config: Optional[VisionLanguageConfig],
                 speculative_config: Optional[SpeculativeConfig]) -> None:
        self.local_rank = envs.LOCAL_RANK
        self.rank = envs.RANK
        self.is_driver_worker = self.rank == 0
        super().__init__(model_config, cache_config, parallel_config,
                         scheduler_config, device_config, load_config,
                         lora_config, vision_language_config,
                         speculative_config)

    def _init_executor(self):
        self.driver_worker = self._create_worker(local_rank=self.local_rank,
                                                 rank=self.rank)
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_gpu_blocks, num_cpu_blocks = (
            self.driver_worker.determine_num_available_blocks())
        t = torch.tensor(
            [[num_gpu_blocks], [num_cpu_blocks]],
            device="cuda",
            dtype=torch.int32,
        )
        output = tensor_model_parallel_all_gather(t)
        return (torch.min(output[0]).item(), torch.min(output[1]).item())

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        output = self.driver_worker.execute_model(execute_model_req)
        if self.is_driver_worker:
            broadcast_object_list([output], src=0)
        else:
            res = [None]
            broadcast_object_list(res, src=0)
            output = res[0]
        return output


class TorchrunGPUExecutorAsync(TorchrunGPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req)
        if self.is_driver_worker:
            broadcast_object_list([output], src=0)
        else:
            res = [None]
            broadcast_object_list(res, src=0)
            output = res[0]
        return output

    async def check_health_async(self) -> None:
        # TorchrunGPUExecutor will always be healthy as long as
        # it's running.
        return
