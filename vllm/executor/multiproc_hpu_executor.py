from typing import Callable, Optional, Tuple, Type

import habana_frameworks.torch  # noqa: F401
import torch

from vllm.executor.multiproc_gpu_executor import (
    MultiprocessingGPUExecutor, MultiprocessingGPUExecutorAsync)
from vllm.logger import init_logger
from vllm.utils import make_async
from vllm.worker.worker_base import WorkerBase

logger = init_logger(__name__)


class MultiprocessingHPUExecutor(MultiprocessingGPUExecutor):
    """Python multiprocessing-based multi-HPU executor"""

    def _get_worker_module_and_class(
            self) -> Tuple[str, str, Optional[Callable[[], Type[WorkerBase]]]]:
        worker_class_fn = None
        if self.scheduler_config.is_multi_step:
            module_name = "vllm.worker.multi_step_hpu_worker"
            class_name = "MultiStepHPUWorker"
        elif self.speculative_config is not None:
            module_name = "vllm.spec_decode.spec_decode_worker"
            class_name = "create_spec_worker"
        else:
            module_name = "vllm.worker.hpu_worker"
            class_name = "HPUWorker"
        return (module_name, class_name, worker_class_fn)

    def _check_executor_parameters(self):
        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        hpu_device_count = torch.hpu.device_count()
        assert tensor_parallel_size <= hpu_device_count, (
            f"please set tensor_parallel_size ({tensor_parallel_size}) "
            f"to less than max local hpu count ({hpu_device_count})")

        assert world_size <= hpu_device_count, (
            f"please ensure that world_size ({world_size}) "
            f"is less than than max local hpu count ({hpu_device_count})")

    def shutdown_inc(self):
        self._run_workers("shutdown_inc")

    def __del__(self):
        self.shutdown()


class MultiprocessingHPUExecutorAsync(MultiprocessingHPUExecutor,
                                      MultiprocessingGPUExecutorAsync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_exec_model = make_async(self.driver_worker.execute_model)
