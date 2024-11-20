from typing import Callable, List, Optional, Tuple, Type, Union

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
