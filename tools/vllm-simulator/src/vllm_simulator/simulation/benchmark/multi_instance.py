import asyncio

from vllm_simulator.simulation.benchmark.request import GenericRequest
from vllm_simulator.simulation.benchmark.base_runner import (
    BaseBenchmarkRunner,
    BaseWorker,
)
from vllm_simulator.simulation.benchmark.bench_config import BenchmarkConfig
from vllm_simulator.simulation.benchmark.load_balance import (
    LoadBalancingPolicy,
    RoundRobinPolicy,
)
from vllm_simulator.simulation.utils import calc_metrics
from vllm_simulator.utils.logger import get_logger

logger = get_logger()


class MultiInstanceBenchmarkRunner(BaseBenchmarkRunner):
    def __init__(
        self, workers: list[BaseWorker], lb_proxy: LoadBalancingPolicy | None = None
    ):
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
        for worker in self.workers:
            await worker.trigger_simulation()

        self._assign_created_time(requests, benchmark_config)
        logger.info(f"Created {len(requests)} request tasks.")

        async def _generate_with_callback(w: BaseWorker, r: GenericRequest):
            try:
                result = await w.async_generate(r)
                self.lb_proxy.on_request_complete(w.name, True, r)
                return result
            except Exception:
                self.lb_proxy.on_request_complete(w.name, False, r)
                raise

        tasks = []
        for req in requests:
            # NON_BLOCKING engines start dispatching after this many requests.
            req.custom_params["total_request"] = len(requests)
            worker = self.lb_proxy.select_worker(self.workers, req)
            self._record_routing(worker, req)
            tasks.append(asyncio.create_task(_generate_with_callback(worker, req)))

        await asyncio.gather(*tasks)

        # dump result
        for worker in self.workers:
            await worker.trigger_simulation()

        return calc_metrics(self.get_request_stats())

    def get_request_stats(self) -> list[dict]:
        result = []
        for worker in self.workers:
            request_stats = worker.get_request_stats()
            for item in request_stats:
                item["worker"] = worker.name
            result.extend(request_stats)
        return result
