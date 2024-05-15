import os
from typing import Dict, List, Optional

import torch

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, SpeculativeConfig,
                         VisionLanguageConfig)
from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.gpu_executor import GPUExecutor
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_object_list, tensor_model_parallel_all_gather)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)

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
                 device_config: DeviceConfig,
                 lora_config: Optional[LoRAConfig],
                 vision_language_config: Optional[VisionLanguageConfig],
                 speculative_config: Optional[SpeculativeConfig]) -> None:
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.is_driver_worker = self.local_rank == 0
        super().__init__(model_config, cache_config, parallel_config,
                         scheduler_config, device_config, lora_config,
                         vision_language_config, speculative_config)

    def _init_worker(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker

        assert self.parallel_config.world_size > 1, (
            "TorchrunGPUExecutor only supports multiple GPUs.")

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        self.driver_worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            self.cache_config,
            local_rank=self.local_rank,
            rank=self.local_rank,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            is_driver_worker=self.is_driver_worker,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def determine_num_available_blocks(self) -> tuple[int, int]:
        num_gpu_blocks, num_cpu_blocks = (
            self.driver_worker.determine_num_available_blocks())
        t = torch.tensor(
            [[num_gpu_blocks], [num_cpu_blocks]],
            device="cuda",
            dtype=torch.int32,
        )
        output = tensor_model_parallel_all_gather(t)
        return (torch.min(output[0]).item(), torch.min(output[1]).item())

    def execute_model(self,
                      seq_group_metadata_list: List[SequenceGroupMetadata],
                      blocks_to_swap_in: Dict[int, int],
                      blocks_to_swap_out: Dict[int, int],
                      blocks_to_copy: Dict[int, List[int]]) -> SamplerOutput:
        output = self.driver_worker.execute_model(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )
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
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        output = await make_async(self.driver_worker.execute_model)(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy)
        return output

    async def check_health_async(self) -> None:
        # TorchrunGPUExecutor will always be healthy as long as
        # it's running.
        return
