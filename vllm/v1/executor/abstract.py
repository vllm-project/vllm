from abc import ABC, abstractmethod
from typing import Tuple

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput


class Executor(ABC):
    """Abstract class for executors."""

    @abstractmethod
    def __init__(self, vllm_config: VllmConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def initialize(self, kv_cache_config: KVCacheConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_available_memory(self) -> int:
        # in bytes
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_spec(self) -> KVCacheSpec:
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
