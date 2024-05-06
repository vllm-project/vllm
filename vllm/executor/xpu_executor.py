from typing import List, Optional

import torch

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         SpeculativeConfig, VisionLanguageConfig)
from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.gpu_executor import GPUExecutor
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)

logger = init_logger(__name__)


class XPUExecutor(GPUExecutor):

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig],
        speculative_config: Optional[SpeculativeConfig],
    ) -> None:
        assert device_config.device_type == "xpu"
        assert (not speculative_config
                ), "Speculative decoding not yet supported for XPU backend"

        model_config = _verify_and_get_model_config(model_config)

        self.model_config = model_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.lora_config = lora_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.vision_language_config = vision_language_config
        self.speculative_config = None

        # Instantiate the worker and load the model to GPU.
        self._init_executor()

    def _init_spec_worker(self):
        logger.error("not support speculative for XPU executor!")

    def _init_non_spec_worker(self):
        from vllm.worker.xpu_worker import XPUWorker

        assert self.parallel_config.world_size == 1, (
            "XPUExecutor only supports single GPU.")

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        self.driver_worker = XPUWorker(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            is_driver_worker=True,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        output = self.driver_worker.execute_model(execute_model_req)
        return output


class XPUExecutorAsync(XPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output


def _verify_and_get_model_config(config: ModelConfig) -> ModelConfig:
    if config.dtype == torch.bfloat16:
        logger.warning(
            "bfloat16 is not fully supported on XPU, casting to float16.")
        config.dtype = torch.float16
    if not config.enforce_eager:
        logger.warning(
            "CUDA graph is not supported on XPU, fallback to the eager "
            "mode.")
        config.enforce_eager = True
    return config
