from typing import List, Optional, Tuple

import torch

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, PromptAdapterConfig,
                         SchedulerConfig, SpeculativeConfig)
from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.gpu_executor import GPUExecutor
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import make_async

logger = init_logger(__name__)


class XPUExecutor(GPUExecutor):

    uses_ray: bool = False

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        prompt_adapter_config: Optional[PromptAdapterConfig],
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
        self.prompt_adapter_config = prompt_adapter_config
        self.speculative_config = None

        # Instantiate the worker and load the model to GPU.
        self._init_executor()

    def _get_worker_module_and_class(self) -> Tuple[str, str]:
        if self.speculative_config is not None:
            raise NotImplementedError(
                "XPU does not support speculative decoding")
        else:
            worker_module_name = "vllm.worker.xpu_worker"
            worker_class_name = "XPUWorker"
        return (worker_module_name, worker_class_name)

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
                                  )(execute_model_req=execute_model_req)
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
