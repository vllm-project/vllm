from typing import Callable, List, Optional, Tuple, Type, Union

import torch

from vllm.config import ModelConfig, ParallelConfig
from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.gpu_executor import GPUExecutor
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.utils import make_async
from vllm.worker.worker_base import WorkerBase

logger = init_logger(__name__)


class XPUExecutor(GPUExecutor):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.device_config.device_type == "xpu"
        assert self.speculative_config is None, (
            "Speculative decoding not yet supported for XPU backend")

        self.model_config = _verify_and_get_model_config(self.model_config)
        GPUExecutor._init_executor(self)

    def _get_worker_module_and_class(
            self) -> Tuple[str, str, Optional[Callable[[], Type[WorkerBase]]]]:
        worker_class_fn = None
        if self.speculative_config is not None:
            raise NotImplementedError(
                "XPU does not support speculative decoding")
        else:
            worker_module_name = "vllm.worker.xpu_worker"
            worker_class_name = "XPUWorker"
        return (worker_module_name, worker_class_name, worker_class_fn)

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[Union[SamplerOutput, PoolerOutput]]]:
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


def _verify_and_get_parallel_config(config: ParallelConfig) -> ParallelConfig:
    if (config.distributed_executor_backend is not None
            and config.distributed_executor_backend != "ray"):
        logger.warning(
            "%s is not supported on XPU, fallback to ray distributed executor "
            "backend.", config.distributed_executor_backend)
        config.distributed_executor_backend = "ray"
    return config
