from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Set, Tuple

from vllm.config import VllmConfig
from vllm.v1.outputs import ModelRunnerOutput


@dataclass
class RPCParams:
    """Arguments for a collective RPC to run on workers"""
    method: str  # Which Worker method should be executed?
    output_ranks: Set[int]  # Which output ranks should return to the executor?
    timeout: Optional[float] = None


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
    def profile(self, is_start=True):
        raise NotImplementedError

    @abstractmethod
    def shutdown(self):
        pass

    @abstractmethod
    def check_health(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def collective_rpc(self, rpc: RPCParams, *args, **kwargs) -> [Optional]:
        raise NotImplementedError
