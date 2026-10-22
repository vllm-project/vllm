"""PD-disaggregated benchmark runner for vLLM simulation workers.

Mirrors vLLM's PD proxy flow: each request first runs on a prefill worker
with ``max_tokens=1``, then on a decode worker with the original output
length.  Decode workers must be configured with the DecodeBenchConnector
(``kv_transfer_config={"kv_connector": "DecodeBenchConnector",
"kv_role": "kv_both"}``) so the prompt KV is treated as already transferred
and only the last prompt token is recomputed.  The transfer itself is
modelled as instantaneous.

Per-request stats only exist in each engine's ``request.jsonl`` dump, so the
two phases are separated by a round boundary on the prefill workers: the
prefill finish time read back from the dump becomes the decode request's
``created_time``.
"""

import asyncio
import dataclasses
import uuid
from collections import Counter

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


class PDDisaggBenchmarkRunner(BaseBenchmarkRunner):
    def __init__(
        self,
        prefill_workers: list[BaseWorker],
        decode_workers: list[BaseWorker],
        prefill_lb_proxy: LoadBalancingPolicy | None = None,
        decode_lb_proxy: LoadBalancingPolicy | None = None,
    ):
        super().__init__()
        self.prefill_workers = prefill_workers
        self.decode_workers = decode_workers
        self.prefill_lb_proxy = prefill_lb_proxy or RoundRobinPolicy()
        self.decode_lb_proxy = decode_lb_proxy or RoundRobinPolicy()
        self.prefill_lb_proxy.init_workers(prefill_workers)
        self.decode_lb_proxy.init_workers(decode_workers)

        self.request_stats: list[dict] = []

    @property
    def all_workers(self) -> list[BaseWorker]:
        return self.prefill_workers + self.decode_workers

    def _pair_request(self, req: GenericRequest):
        """Pair a request with its prefill and decode workers."""
        prefill_worker = self.prefill_lb_proxy.select_worker(
            self.prefill_workers, req
        )
        decode_worker = self.decode_lb_proxy.select_worker(
            self.decode_workers, req
        )
        self._record_routing(prefill_worker, req)
        self._record_routing(decode_worker, req)

        # Shared id joins the two engines' request.jsonl records.
        req.custom_params["request_id"] = uuid.uuid4().hex
        req.custom_params["prefill_worker"] = prefill_worker
        req.custom_params["decode_worker"] = decode_worker

    @staticmethod
    def _phase_request(
        req: GenericRequest, total_request: int, **overrides
    ) -> GenericRequest:
        """Per-phase copy of a request.

        ``total_request`` is the number of requests routed to the target
        worker: NON_BLOCKING engines start dispatching only after receiving
        that many requests.
        """
        custom_params = {
            **req.custom_params,
            "total_request": total_request,
            **overrides.pop("custom_params", {}),
        }
        return dataclasses.replace(req, custom_params=custom_params, **overrides)

    async def _run_phase(
        self,
        workers: list[BaseWorker],
        role: str,
        phase_requests: list[GenericRequest],
    ) -> dict[str, dict]:
        """Run one phase to completion and return its stats keyed by rid."""
        await asyncio.gather(
            *(req.custom_params[role].async_generate(req) for req in phase_requests)
        )

        # Round boundary: dump this phase's per-request stats.
        for worker in workers:
            await worker.trigger_simulation()
        return {
            item["rid"]: item
            for worker in workers
            for item in worker.get_request_stats()
        }

    @staticmethod
    def _merge_request_stats(
        req: GenericRequest, prefill_stat: dict, decode_stat: dict
    ) -> dict:
        """Merge P/D stats as seen by a client of vLLM's PD proxy.

        The client only receives the decode stream, so the first token
        latency spans the whole prefill plus the decode instance's first
        step; later token latencies are the decode instance's.
        """
        prefill_latencies = prefill_stat.get("gen_token_latencies") or []
        decode_latencies = decode_stat.get("gen_token_latencies") or []
        if decode_latencies:
            gen_token_latencies = [
                sum(prefill_latencies) + decode_latencies[0]
            ] + decode_latencies[1:]
        else:
            gen_token_latencies = prefill_latencies

        return {
            "rid": prefill_stat["rid"],
            "prefill_worker": req.custom_params["prefill_worker"].name,
            "decode_worker": req.custom_params["decode_worker"].name,
            "created_time": prefill_stat["created_time"],
            "queue_start": prefill_stat["queue_start"],
            "queue_end": prefill_stat["queue_end"],
            "last_event_time": decode_stat.get(
                "last_event_time", prefill_stat["last_event_time"]
            ),
            "input_length": prefill_stat["input_length"],
            "output_length": decode_stat.get("output_length", 1),
            # The prefix cache only matters on the prefill worker; the
            # decode worker's external hit is the simulated transfer.
            "local_kv_hit_len": prefill_stat["local_kv_hit_len"],
            "ext_kv_hit_len": prefill_stat["ext_kv_hit_len"],
            "gen_token_latencies": gen_token_latencies,
        }

    async def async_benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        requests: list[GenericRequest],
    ):
        self.request_stats = []
        for worker in self.all_workers:
            await worker.trigger_simulation()

        self._assign_created_time(requests, benchmark_config)
        logger.info(f"Created {len(requests)} PD disagg request tasks.")

        for req in requests:
            self._pair_request(req)

        # Phase 1: prefill only (first token), as vLLM's PD proxy does.
        prefill_counts = Counter(
            req.custom_params["prefill_worker"].name for req in requests
        )
        prefill_stats = await self._run_phase(
            self.prefill_workers,
            "prefill_worker",
            [
                self._phase_request(
                    req,
                    prefill_counts[req.custom_params["prefill_worker"].name],
                    output_length=1,
                )
                for req in requests
            ],
        )

        # Phase 2: decode, arriving when the prefill finishes.
        paired = []
        for req in requests:
            prefill_stat = prefill_stats.get(req.custom_params["request_id"])
            if prefill_stat is None:
                logger.error(
                    "No prefill stats for request %s; skipping its decode.",
                    req.custom_params["request_id"],
                )
                continue
            paired.append((req, prefill_stat))

        decode_counts = Counter(
            req.custom_params["decode_worker"].name for req, _ in paired
        )
        decode_stats = await self._run_phase(
            self.decode_workers,
            "decode_worker",
            [
                self._phase_request(
                    req,
                    decode_counts[req.custom_params["decode_worker"].name],
                    custom_params={
                        "created_time": prefill_stat["last_event_time"],
                        "queue_start": prefill_stat["last_event_time"],
                    },
                )
                for req, prefill_stat in paired
            ],
        )

        for req, prefill_stat in paired:
            decode_stat = decode_stats.get(req.custom_params["request_id"], {})
            self.request_stats.append(
                self._merge_request_stats(req, prefill_stat, decode_stat)
            )

        return calc_metrics(self.request_stats)

    def get_request_stats(self) -> list[dict]:
        return self.request_stats
