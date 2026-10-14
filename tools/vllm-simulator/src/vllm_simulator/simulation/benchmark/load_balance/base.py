from abc import ABC, abstractmethod
from typing import Optional

from vllm_simulator.simulation.benchmark.request import GenericRequest
from vllm_simulator.simulation.benchmark.base_runner import BaseWorker


class LoadBalancingPolicy(ABC):
    @abstractmethod
    def select_worker(
        self, workers: list[BaseWorker], req: GenericRequest
    ) -> Optional[BaseWorker]: ...

    @abstractmethod
    def name(self) -> str: ...

    def on_request_complete(self, worker_name: str, success: bool, req: GenericRequest | None  = None):
        pass

    def init_workers(self, workers: list[BaseWorker]):
        pass
