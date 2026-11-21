import asyncio
from collections import Counter

from vllm_simulator.simulation.benchmark.base_runner import (
    BaseBenchmarkRunner,
    BaseWorker,
)
from vllm_simulator.simulation.benchmark.bench_config import BenchmarkConfig
from vllm_simulator.simulation.benchmark.load_balance import (
    LoadBalancingPolicy,
    RoundRobinPolicy,
)
from vllm_simulator.simulation.benchmark.request import GenericRequest
from vllm_simulator.simulation.utils import calc_metrics
from vllm_simulator.utils.logger import get_logger

logger = get_logger()


class MultiInstanceBenchmarkRunner(BaseBenchmarkRunner):
    def __init__(
        self, workers: list[BaseWorker], lb_proxy: LoadBalancingPolicy | None = None
    ):
        BaseBenchmarkRunner._validate_workers(workers)
        super().__init__()
        self.workers = workers
        self.lb_proxy = lb_proxy if lb_proxy is not None else RoundRobinPolicy()
        self.lb_proxy.init_workers(workers)

    @property
    def all_workers(self) -> list[BaseWorker]:
        return self.workers

    async def async_benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        requests: list[GenericRequest],
    ):
        self._assign_created_time(requests, benchmark_config)
        logger.info(f"Created {len(requests)} request tasks.")

        async def _generate_with_callback(w: BaseWorker, r: GenericRequest):
            try:
                stats = await w.async_generate(r)
                stats = self._validate_request_stats(
                    stats, expected_rid=r.custom_params.get("request_id")
                )
                self.lb_proxy.on_request_complete(w.name, True, r)
            except Exception:
                self.lb_proxy.on_request_complete(w.name, False, r)
                raise
            stats["worker"] = w.name
            return stats

        assignments = []
        for req in requests:
            worker = self.lb_proxy.select_worker(self.workers, req)
            if worker not in self.workers:
                raise RuntimeError("Load-balancing policy returned an unknown worker")
            self._record_routing(worker, req)
            assignments.append((worker, req))

        counts = Counter(worker for worker, _ in assignments)
        tasks = []
        for worker, req in assignments:
            # NON_BLOCKING engines dispatch after all requests routed to that
            # particular worker have arrived.
            req.custom_params["total_request"] = counts[worker]
            tasks.append(asyncio.create_task(_generate_with_callback(worker, req)))

        self.request_stats = await asyncio.gather(*tasks)
        return calc_metrics(
            self.request_stats, self._duration_start_time(self.request_stats)
        )
