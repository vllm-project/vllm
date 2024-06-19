from vllm.executor.multiproc_gpu_executor import (
    MultiprocessingGPUExecutor, MultiprocessingGPUExecutorAsync)
from vllm.executor.xpu_executor import XPUExecutor
from vllm.logger import init_logger
from vllm.utils import make_async

logger = init_logger(__name__)


class MultiprocessingXPUExecutor(MultiprocessingGPUExecutor, XPUExecutor):
    """Python multiprocessing-based multi-XPU executor"""

    def _check_executor_parameters(self):
        pass


class MultiprocessingXPUExecutorAsync(MultiprocessingXPUExecutor,
                                      MultiprocessingGPUExecutorAsync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_exec_model = make_async(self.driver_worker.execute_model)
