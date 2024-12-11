from abc import ABC, abstractmethod
from typing import Tuple

from vllm.config import VllmConfig
from vllm.v1.outputs import ModelRunnerOutput


class Executor(ABC):
    """Abstract class for executors."""

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
