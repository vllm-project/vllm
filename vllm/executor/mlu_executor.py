from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Type

from vllm.config import VllmConfig
from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.utils import make_async
from vllm.executor.gpu_executor import GPUExecutor, create_worker
from vllm.worker.worker_base import WorkerBase

logger = init_logger(__name__)


class MLUExecutor(GPUExecutor):

    uses_ray: bool = False

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super(GPUExecutor, self).__init__(vllm_config)

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        assert self.parallel_config.world_size == 1, (
            "MLUExecutor only supports single MLU.")

        self.driver_worker = self._create_worker()
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def _get_worker_module_and_class(
            self) -> Tuple[str, str, Optional[Callable[[], Type[WorkerBase]]]]:
        worker_class_fn = None
        if self.scheduler_config.is_multi_step:
            worker_module_name = "vllm.worker.multi_step_mlu_worker"            
            worker_class_name = "MLUMultiStepWorker"
        elif self.speculative_config:
            worker_module_name = "vllm.spec_decode.mlu_spec_decode_worker"
            worker_class_name = "create_mlu_spec_worker"
        else:
            worker_module_name = "vllm.worker.mlu_worker"
            worker_class_name = "MLUWorker"
        return (worker_module_name, worker_class_name, worker_class_fn)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-MLU configurations.
        logger.info("# MLU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)
        max_concurrency = (num_gpu_blocks * self.cache_config.block_size /
                           self.model_config.max_model_len)
        logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                    self.model_config.max_model_len, max_concurrency)

        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[Union[SamplerOutput, PoolerOutput]]]:
        output = self.driver_worker.execute_model(execute_model_req)
        return output


class MLUExecutorAsync(MLUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req)
        return output
