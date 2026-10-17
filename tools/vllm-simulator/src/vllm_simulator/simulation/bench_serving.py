"""Online serving benchmark client for the vLLM simulator.

Hijacks ``vllm.benchmarks.serve`` so that:

1. The client never sleeps between requests; each request carries
   ``simulation.created_time`` and the server replays arrivals from it
   (see ``C_VLLMEngineCoreHook``).
2. Reported metrics are aggregated by ``calc_metrics`` from the
   ``request.jsonl`` dumped by the server's profile hook.

Usage:
    python -m vllm_simulator.simulation.bench_serving [vllm bench serve args...]

Requires the simulator server (``vllm_simulator.simulation.vllm.launch_server``),
which uses /start_profile and /stop_profile as round markers.
"""

import argparse
import json
import os
from typing import AsyncGenerator, Literal

import numpy as np
from vllm.benchmarks import serve
from vllm.benchmarks.datasets import SampleRequest

from vllm_simulator.simulation.manager import Envs
from vllm_simulator.simulation.utils import calc_metrics


# Same arrival model as ``serve.get_request``, but the cumulative delay becomes
# ``simulation.created_time`` instead of a sleep.  Self-timed traces are shifted
# so the first request starts at zero.
async def override_get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
    self_timed: bool = False,
) -> AsyncGenerator[tuple[SampleRequest, float], None]:
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    if not isinstance(input_requests, list):
        input_requests = list(input_requests)

    total_requests = len(input_requests)
    assert total_requests > 0, "No requests provided."

    request_rates: list[float] = []
    delay_ts: list[float] = []

    if self_timed:
        input_requests.sort(key=lambda r: r.timestamp or 0.0)
        first_ts = input_requests[0].timestamp or 0.0
        for request in input_requests:
            delay_ts.append((request.timestamp or 0.0) - first_ts)
            # No notion of RPS for self-timed traces, same as native.
            request_rates.append(0.0)
    else:
        for request_index in range(total_requests):
            # Reuse the native helper so ramp-up semantics stay in sync.
            current_request_rate = serve._get_current_request_rate(
                ramp_up_strategy,
                ramp_up_start_rps,
                ramp_up_end_rps,
                request_index,
                total_requests,
                request_rate,
            )
            assert current_request_rate > 0.0, (
                f"Obtained non-positive request rate {current_request_rate}."
            )
            request_rates.append(current_request_rate)
            if current_request_rate == float("inf"):
                delay_ts.append(0)
            elif burstiness == float("inf"):
                # Constant inter-arrival time in the infinite-burstiness limit.
                delay_ts.append(1.0 / current_request_rate)
            else:
                theta = 1.0 / (current_request_rate * burstiness)
                delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))

        # Cumulative arrival offsets from the first request.
        for i in range(1, len(delay_ts)):
            delay_ts[i] += delay_ts[i - 1]
        if ramp_up_strategy is None and delay_ts[-1] != 0:
            # Same total-duration normalization as the native generator.
            target_total_delay_s = total_requests / request_rate
            normalize_factor = target_total_delay_s / delay_ts[-1]
            delay_ts = [delay * normalize_factor for delay in delay_ts]

    for request_index, request in enumerate(input_requests):
        # vllm_xargs is the only path into SamplingParams.extra_args; values
        # must be scalars/lists, hence the JSON string.
        overrides = dict(request.request_overrides or {})
        overrides["vllm_xargs"] = {
            **overrides.get("vllm_xargs", {}),
            "simulation": json.dumps({
                "created_time": delay_ts[request_index],
                "total_request": total_requests,
            }),
        }
        request.request_overrides = overrides
        yield request, request_rates[request_index]


# Replace client-side metrics with the simulated ones from ``request.jsonl``,
# dumped on the /stop_profile that ``serve.benchmark`` issues before returning.
# The client-side result is kept if the file is missing.
_BASE_RESULT_KEYS = {
    "duration": "duration",
    "completed": "completed",
    "total_input": "total_input_tokens",
    "total_output": "total_output_tokens",
    "request_throughput": "request_throughput",
    "output_throughput": "output_throughput",
    "total_throughput": "total_token_throughput",
}

# vLLM serve result name -> calc_metrics name.
_LATENCY_METRICS = {
    "ttft": "ttft",
    "tpot": "tpot",
    "itl": "itl",
    "e2el": "e2e_latency",
}

# Simulation-only stats worth keeping in the result json.
_EXTRA_RESULT_KEYS = (
    "num_requests",
    "input_throughput",
    "mean_queue_ms",
    "max_itl_ms",
    "prompt_tokens_cached_ratio",
    "prompt_tokens_local_cache_hit_ratio",
    "prompt_tokens_external_kv_transfer_ratio",
)


def _load_simulation_metrics() -> dict | None:
    """Aggregate per-request stats dumped by the server-side profile hook."""
    stats_path = os.path.join(Envs.output_dir(), "request.jsonl")
    if not os.path.exists(stats_path):
        return None
    with open(stats_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return None
    return calc_metrics(rows)


def _apply_simulation_metrics(result: dict, sim: dict) -> None:
    for sim_key, result_key in _BASE_RESULT_KEYS.items():
        result[result_key] = sim[sim_key]
    for vllm_name, sim_name in _LATENCY_METRICS.items():
        # Percentile keys exist only when selected via --percentile-metrics /
        # --metric-percentiles; override the ones present in the result.
        for stat in ("mean", "median", "std", "p90", "p95", "p99"):
            result_key = f"{stat}_{vllm_name}_ms"
            sim_key = f"{stat}_{sim_name}_ms"
            if result_key in result and sim_key in sim:
                result[result_key] = sim[sim_key]
        # calc_metrics reports the median instead of p50.
        p50_key = f"p50_{vllm_name}_ms"
        if p50_key in result:
            result[p50_key] = sim[f"median_{sim_name}_ms"]
    for key in _EXTRA_RESULT_KEYS:
        if key in sim:
            result[key] = sim[key]


original_benchmark = serve.benchmark


async def wrapped_benchmark(*args, **kwargs):
    result = await original_benchmark(*args, **kwargs)

    sim_metrics = _load_simulation_metrics()
    if sim_metrics is None:
        print(
            "[simulation] No backend request stats found under "
            f"{Envs.output_dir()}, keeping client-side metrics. Launch the "
            "server with a profiler configured so that /start_profile and "
            "/stop_profile are registered."
        )
        return result

    _apply_simulation_metrics(result, sim_metrics)

    print("{s:{c}^{n}}".format(s=" Simulation Metrics ", n=50, c="="))
    print(f"{'Completed requests:':<40} {sim_metrics['completed']:<10}")
    for label, key in (
        ("Simulated duration (s):", "duration"),
        ("Output token throughput (tok/s):", "output_throughput"),
        ("Mean TTFT (ms):", "mean_ttft_ms"),
        ("Mean TPOT (ms):", "mean_tpot_ms"),
        ("Mean E2EL (ms):", "mean_e2e_latency_ms"),
    ):
        print(f"{label:<40} {sim_metrics[key]:<10.2f}")
    print("=" * 50)
    return result


original_main_async = serve.main_async


async def wrapped_main_async(args: argparse.Namespace):
    # Profile routes are the round markers, so profiling is always on.
    args.profile = True
    return await original_main_async(args)


serve.get_request = override_get_request
serve.benchmark = wrapped_benchmark
serve.main_async = wrapped_main_async


if __name__ == "__main__":
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="vLLM simulator online serving benchmark "
        "(hijacked `vllm bench serve`)"
    )
    serve.add_cli_args(parser)
    args = parser.parse_args()
    serve.main(args)
