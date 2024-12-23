from typing import List, Optional, Union

from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.gpu_executor import GPUExecutor
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.utils import make_async

logger = init_logger(__name__)


class XPUExecutor(GPUExecutor):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.device_config.device_type == "xpu"
        assert self.speculative_config is None, (
            "Speculative decoding not yet supported for XPU backend")

        GPUExecutor._init_executor(self)

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
