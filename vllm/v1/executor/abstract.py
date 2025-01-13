from abc import ABC, abstractmethod
from typing import Tuple, Type

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.outputs import ModelRunnerOutput


class Executor(ABC):
    """Abstract class for executors."""

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> Type["Executor"]:
        executor_class: Type[Executor]
        distributed_executor_backend = (
            vllm_config.parallel_config.distributed_executor_backend)
        if distributed_executor_backend == "ray":
            if current_platform.is_cuda():
                from vllm.v1.executor.ray_executor import RayExecutor
                executor_class = RayExecutor
            elif current_platform.is_xpu():
                from vllm.v1.executor.xpu_ray_executor import RayXPUExecutor
                executor_class = RayXPUExecutor
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor
            executor_class = MultiprocExecutor
        else:
            assert (distributed_executor_backend is None)
            if current_platform.is_cuda():
                from vllm.v1.executor.uniproc_executor import UniprocExecutor
                executor_class = UniprocExecutor
            elif current_platform.is_xpu():
                from vllm.v1.executor.xpu_uniproc_executor import (  # noqa: E501
                    XPUUniprocExecutor)
                executor_class = XPUUniprocExecutor
        return executor_class

    @abstractmethod
    def __init__(self, vllm_config: VllmConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def initialize(self, num_gpu_blocks: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def execute_model(
        self,
        scheduler_output,
    ) -> ModelRunnerOutput:
        raise NotImplementedError

    @abstractmethod
    def profile(self, is_start: bool = True):
        raise NotImplementedError

    @abstractmethod
    def shutdown(self):
        pass

    @abstractmethod
    def check_health(self) -> None:
        raise NotImplementedError
