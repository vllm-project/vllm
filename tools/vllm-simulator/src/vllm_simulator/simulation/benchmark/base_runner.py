import asyncio
from abc import ABC, abstractmethod

import numpy as np

from vllm_simulator.simulation.benchmark.bench_config import BenchmarkConfig
from vllm_simulator.simulation.benchmark.request import GenericRequest


class BaseWorker(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def async_generate(self, req: GenericRequest):
        pass

    @abstractmethod
    async def trigger_simulation(self):
        """Round boundary: export this round's stats and reset them."""
        pass

    @abstractmethod
    def get_request_stats(self) -> list[dict]:
        pass

    @abstractmethod
    def get_iteration_stats(self) -> list[dict]:
        pass

    async def flush_cache(self):
        pass

    def shutdown(self):
        pass


class BaseBenchmarkRunner(ABC):
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        # Worker's name -> requests
        self.lb_routing_records: dict[str, list[GenericRequest]] = {}

    @property
    @abstractmethod
    def all_workers(self) -> list[BaseWorker]:
        pass

    @abstractmethod
    async def async_benchmark(
        self, benchmark_config: BenchmarkConfig, requests: list[GenericRequest]
    ) -> dict:
        pass

    def benchmark(
        self, benchmark_config: BenchmarkConfig, requests: list[GenericRequest]
    ) -> dict:
        return self.loop.run_until_complete(
            self.async_benchmark(benchmark_config, requests)
        )

    @staticmethod
    def _assign_created_time(
        requests: list[GenericRequest], benchmark_config: BenchmarkConfig
    ) -> None:
        """Set custom_params["created_time"]: Poisson arrivals at
        request_rate when ignoring the request timestamps."""
        arrival = 0
        for req in requests:
            if benchmark_config.ignore_request_timestamp:
                created_time = arrival
                arrival += np.random.exponential(
                    1.0 / benchmark_config.request_rate
                )
            else:
                created_time = req.custom_params.get("created_time", 0)
            req.custom_params["created_time"] = created_time

    def _record_routing(self, worker: BaseWorker, req: GenericRequest) -> None:
        self.lb_routing_records.setdefault(worker.name, []).append(req)

    def get_iteration_stats(self) -> list[dict]:
        result = []
        for worker in self.all_workers:
            iteration_stats = worker.get_iteration_stats()
            for item in iteration_stats:
                item["worker"] = worker.name
            result.extend(iteration_stats)
        return result

    def get_lb_routing_records(self) -> dict[str, list[GenericRequest]]:
        return self.lb_routing_records

    def flush_cache(self):
        async def _flush():
            for worker in self.all_workers:
                await worker.flush_cache()

        self.loop.run_until_complete(_flush())

    def shutdown(self):
        for worker in self.all_workers:
            worker.shutdown()
        self._close_loop()

    def _close_loop(self):
        """Drain tasks left on the runner loop (e.g. AsyncLLM output handlers)
        and close it, so they are not destroyed pending on a closed loop."""
        loop: asyncio.AbstractEventLoop = self.loop
        if loop.is_closed():
            return
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
