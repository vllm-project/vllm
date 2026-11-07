import asyncio
import math
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

from vllm_simulator.simulation.benchmark.bench_config import BenchmarkConfig
from vllm_simulator.simulation.benchmark.request import GenericRequest
from vllm_simulator.simulation.manager import Envs


class BaseWorker(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def async_generate(self, req: GenericRequest) -> dict:
        """Run one request to completion and return its RequestStats."""
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
        # Per-request stats of the last benchmark() call.
        self.request_stats: list[dict] = []

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
    def _validate_workers(workers: list[BaseWorker]) -> None:
        if not workers:
            raise ValueError("At least one worker is required")
        duplicates = [
            name
            for name, count in Counter(w.name for w in workers).items()
            if count > 1
        ]
        if duplicates:
            raise ValueError(f"Worker names must be unique: {duplicates}")

    @staticmethod
    def _validate_request_stats(
        stats: dict | None, *, expected_rid: str | None = None
    ) -> dict:
        if not isinstance(stats, dict):
            raise RuntimeError("Worker returned no request stats")
        rid = stats.get("rid")
        if not isinstance(rid, str) or not rid:
            raise RuntimeError("Request stats have no request id")
        if expected_rid is not None and rid != expected_rid:
            raise RuntimeError(
                f"Request stats id mismatch: expected {expected_rid}, got {rid}"
            )
        latencies = stats.get("gen_token_latencies")
        if not isinstance(latencies, list) or not latencies:
            raise RuntimeError(f"Request {rid} has no generated-token latencies")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in latencies
        ):
            raise RuntimeError(f"Request {rid} has invalid generated-token latencies")
        output_length = stats.get("output_length")
        if output_length != len(latencies):
            raise RuntimeError(
                f"Request {rid} expected {output_length} output tokens, "
                f"but has {len(latencies)} latency records"
            )
        return stats

    @staticmethod
    def _duration_start_time(stats: list[dict]) -> float:
        if Envs.simulation_mode() != "BLOCKING" or not stats:
            return 0.0
        return min(float(item["created_time"]) for item in stats)

    @staticmethod
    def _assign_created_time(
        requests: list[GenericRequest], benchmark_config: BenchmarkConfig
    ) -> None:
        """Set each request's logical arrival time."""
        arrival = 0
        for req in requests:
            if benchmark_config.ignore_request_timestamp:
                created_time = arrival
                arrival += np.random.exponential(1.0 / benchmark_config.request_rate)
            else:
                created_time = req.custom_params.get("created_time", 0)
            req.custom_params["created_time"] = created_time

    def _record_routing(self, worker: BaseWorker, req: GenericRequest) -> None:
        self.lb_routing_records.setdefault(worker.name, []).append(req)

    def get_request_stats(self) -> list[dict]:
        return self.request_stats

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
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
