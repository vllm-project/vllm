from typing import Optional

from vllm_simulator.simulation.benchmark.request import GenericRequest
from vllm_simulator.simulation.benchmark.base_runner import BaseWorker
from vllm_simulator.simulation.benchmark.load_balance.base import LoadBalancingPolicy


class RoundRobinPolicy(LoadBalancingPolicy):
    """Sequential rotation through workers."""

    def __init__(self):
        self._counter = 0

    def select_worker(
        self, workers: list[BaseWorker], req: GenericRequest
    ) -> Optional[BaseWorker]:
        if not workers:
            return None
        idx = self._counter % len(workers)
        self._counter += 1
        return workers[idx]

    def name(self) -> str:
        return "round_robin"
