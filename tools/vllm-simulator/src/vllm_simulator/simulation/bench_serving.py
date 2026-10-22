"""Online serving benchmark client for the vLLM simulator.

Hijacks ``vllm.benchmarks.serve`` so that:

1. The client never sleeps between requests; each request carries
   ``simulation.created_time`` and the server replays arrivals from it
   (see ``C_VLLMEngineCoreHook``).
2. Non-streaming completions/chat responses return per-request stats in
   ``kv_transfer_params["simulation"]``. Native benchmark latency metrics use
   the simulated token latencies; ``calc_metrics`` supplies duration,
   throughput, and cache-hit metrics.

Usage:
    python -m vllm_simulator.simulation.bench_serving [vllm bench serve args...]

Requires the simulator server (``vllm_simulator.simulation.vllm.launch_server``).
"""

import inspect
import json
import time
import traceback
from collections.abc import AsyncGenerator
from contextvars import ContextVar
from typing import Literal

import aiohttp
import numpy as np

from vllm.benchmarks import serve
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib import endpoint_request_func as request_funcs
from vllm_simulator.simulation.utils import calc_metrics

# Enabled by get_request only after readiness checks and warmups finish.
_REQUEST_STATS: ContextVar[list[dict] | None] = ContextVar(
    "simulation_request_stats", default=None
)


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
    _REQUEST_STATS.set([])

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
            "simulation": json.dumps(
                {
                    "created_time": delay_ts[request_index],
                    "total_request": total_requests,
                }
            ),
        }
        request.request_overrides = overrides
        yield request, request_rates[request_index]


async def async_request_simulation(
    request_func_input: request_funcs.RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar=None,
) -> request_funcs.RequestFuncOutput:
    """Return simulated token timings from a non-streaming response."""
    output = request_funcs.RequestFuncOutput(
        prompt_len=request_func_input.prompt_len, start_time=time.perf_counter()
    )
    rows = _REQUEST_STATS.get()
    try:
        api_url = request_func_input.api_url
        is_chat = api_url.endswith("/chat/completions")
        request_funcs._validate_api_url(api_url, "Simulation API", "completions")
        payload = {
            "model": request_func_input.model_name or request_func_input.model,
        }
        if is_chat:
            payload["messages"] = (
                request_func_input.chat_messages
                if request_func_input.chat_messages is not None
                else request_funcs._get_chat_messages(request_func_input)
            )
            payload["max_completion_tokens"] = request_func_input.output_len
        else:
            payload.update(
                prompt=request_func_input.prompt,
                repetition_penalty=1.0,
                max_tokens=request_func_input.output_len,
                logprobs=request_func_input.logprobs,
            )
        request_funcs._update_payload_common(payload, request_func_input)
        payload["stream"] = False
        payload.pop("stream_options", None)
        if payload.get("n", 1) != 1:
            raise ValueError("Simulation stats require n=1.")

        # Readiness checks and warmups are independent one-request rounds.
        # Override any metadata left on a reused dataset by an earlier run.
        if rows is None:
            payload["vllm_xargs"] = {
                **payload.get("vllm_xargs", {}),
                "simulation": json.dumps({"created_time": 0, "total_request": 1}),
            }
        headers = request_funcs._get_headers("application/json")
        request_funcs._update_headers_common(headers, request_func_input)
        async with session.post(api_url, json=payload, headers=headers) as response:
            if response.status != 200:
                output.error = f"HTTP {response.status}: {await response.text()}"
                return output
            data = await response.json()

        stats = (data.get("kv_transfer_params") or {}).get("simulation")
        if not isinstance(stats, dict) or not stats.get("gen_token_latencies"):
            raise ValueError(
                "Response has no simulation request stats; use the simulator server."
            )
        if len(data["choices"]) != 1:
            raise ValueError("Simulation stats require a single prompt and choice.")
        choice = data["choices"][0]
        output.generated_text = (
            choice["message"].get("content") if is_chat else choice.get("text")
        ) or ""
        usage = data["usage"]
        output.prompt_len = usage["prompt_tokens"]
        output.output_tokens = usage["completion_tokens"]
        latencies = stats["gen_token_latencies"]
        output.ttft = latencies[0]
        output.itl = latencies[1:]
        output.latency = sum(latencies)
        output.tpot = float(np.mean(output.itl)) if output.itl else 0.0
        output.success = True
        if rows is not None:
            rows.append(
                {
                    **stats,
                    "input_length": output.prompt_len,
                    "output_length": output.output_tokens,
                }
            )
    except Exception:
        output.error = traceback.format_exc()
    finally:
        if pbar is not None:
            pbar.update(1)
    return output


# Native latency calculations already consume the simulated RequestFuncOutput.
# Only wall-clock duration/throughputs and simulation-only metrics need replacing.
_BASE_RESULT_KEYS = {
    "duration": "duration",
    "completed": "completed",
    "total_input": "total_input_tokens",
    "total_output": "total_output_tokens",
    "request_throughput": "request_throughput",
    "output_throughput": "output_throughput",
    "total_throughput": "total_token_throughput",
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


def _apply_simulation_metrics(result: dict, sim: dict) -> None:
    if result.get("request_goodput") is not None:
        result["request_goodput"] *= result["duration"] / sim["duration"]
    for sim_key, result_key in _BASE_RESULT_KEYS.items():
        result[result_key] = sim[sim_key]
    for key in _EXTRA_RESULT_KEYS:
        if key in sim:
            result[key] = sim[key]


original_benchmark = serve.benchmark


async def wrapped_benchmark(*args, **kwargs):
    options = inspect.signature(original_benchmark).bind(*args, **kwargs)
    options.apply_defaults()
    backend = options.arguments["endpoint_type"]
    if backend not in ("openai", "vllm", "openai-chat"):
        raise ValueError("Simulation supports openai, vllm, and openai-chat backends.")
    if (backend == "openai-chat") != options.arguments["api_url"].endswith(
        "/chat/completions"
    ):
        raise ValueError(
            "Use --endpoint /v1/chat/completions for openai-chat, "
            "or /v1/completions for openai/vllm."
        )
    concurrency = options.arguments["max_concurrency"]
    if concurrency and concurrency < len(options.arguments["input_requests"]):
        raise ValueError(
            "Simulation replay needs all requests in flight; omit --max-concurrency "
            "or set it to at least the number of requests."
        )
    if options.arguments["probe_request_rate"]:
        raise ValueError("Simulation replay does not support probe requests.")

    token = _REQUEST_STATS.set(None)
    try:
        result = await original_benchmark(*args, **kwargs)
        rows = _REQUEST_STATS.get() or []
    finally:
        _REQUEST_STATS.reset(token)
    if not rows:
        raise RuntimeError("No simulation request stats returned by the benchmark.")

    sim_metrics = calc_metrics(rows)
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


for backend in ("openai", "vllm", "openai-chat"):
    serve.ASYNC_REQUEST_FUNCS[backend] = async_request_simulation
serve.get_request = override_get_request
serve.benchmark = wrapped_benchmark


if __name__ == "__main__":
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="vLLM simulator online serving benchmark "
        "(hijacked `vllm bench serve`)"
    )
    serve.add_cli_args(parser)
    args = parser.parse_args()
    serve.main(args)
