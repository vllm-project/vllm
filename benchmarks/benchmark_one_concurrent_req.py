# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp  # Import aiohttp
import numpy as np
from tqdm import tqdm

from backend_request_func import RequestFuncInput, RequestFuncOutput
from benchmark_dataset import RandomDataset, SampleRequest

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


async def reset_cache(reset_url: str):
    """Sends a POST request to reset the prefix cache."""
    logger.debug("Resetting prefix cache at %s", reset_url)
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(reset_url) as response,
        ):
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            logger.debug("Prefix cache reset successful: %s", response.status)
    except aiohttp.ClientConnectorError as e:
        logger.error("Failed to connect to cache reset endpoint %s: %s}", reset_url, e)
    except aiohttp.ClientResponseError as e:
        logger.error(
            "Cache reset request failed with status %s: %s", e.status, e.message
        )
    except Exception as e:
        logger.error("An unexpected error occurred during cache reset: %s", e)


async def sequential_benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer,
    input_requests: list[SampleRequest],
    request_func,
    selected_percentiles: list[float],
    cache_reset_url: Optional[str] = None,
):
    """
    Benchmark that processes requests sequentially, waiting for each to complete
    before starting the next one. Resets prefix cache between requests.
    """
    outputs = []

    pbar = tqdm(total=len(input_requests))

    # Small request to force a forward pass.
    # Used for resetting the prefix cache.
    # dummy_req_input = RequestFuncInput(
    #     model=model_id,
    #     prompt="0",
    #     api_url=api_url,
    #     prompt_len=1,
    #     output_len=2,
    # )

    # print("Starting initial single prompt test run...")
    # test_output = await request_func(request_func_input=dummy_req_input)
    # if not test_output.success:
    #     raise ValueError(
    #         "Initial test run failed - Please check your configuration. "
    #         "Error: %s", test_output.error)
    # else:
    #     print("Initial test run completed. Starting sequential benchmark...")

    benchmark_start_time = time.perf_counter()

    # Process requests sequentially
    for request in input_requests:
        prompt, prompt_len, output_len = (
            request.prompt,
            request.prompt_len,
            request.expected_output_len,
        )

        logger.info("Sending request with len %s", request.prompt_len)
        logger.debug('Request str: "%s"', request.prompt[:50])
        request_start_time = time.perf_counter()

        # print(f"{prompt=}")
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
        )

        output = await request_func(request_func_input=request_func_input)

        request_end_time = time.perf_counter()
        # Add timing information
        if output.success and not hasattr(output, "latency"):
            output.latency = request_end_time - request_start_time
        logger.info("Finished request with latency %.4f s", output.latency)

        outputs.append(output)
        pbar.update(1)

        # Reset prefix cache if configured, except after the very last request
        if cache_reset_url and False:
            await request_func(request_func_input=dummy_req_input)
            await reset_cache(cache_reset_url)

    pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    # Calculate metrics
    metrics = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
    )

    print_results(metrics, benchmark_duration)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "input_lens": [request.prompt_len for request in input_requests],
        "output_lens": [
            output.output_tokens if output.success else 0 for output in outputs
        ],
        "ttfts": [output.ttft for output in outputs if output.success],
        "itls": [output.itl for output in outputs if output.success],
        "generated_texts": [
            output.generated_text for output in outputs if output.success
        ],
        "errors": [output.error for output in outputs if not output.success],
    }

    # Add summary statistics
    for stat_name in ["ttft", "itl", "e2el"]:
        for metric_name in ["mean", "median", "std"]:
            result[f"{metric_name}_{stat_name}_ms"] = getattr(
                metrics, f"{metric_name}_{stat_name}_ms"
            )

        for p, value in getattr(metrics, f"percentiles_{stat_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            result[f"p{p_word}_{stat_name}_ms"] = value

    return result


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer,
    selected_percentiles: list[float],
) -> BenchmarkMetrics:
    """Calculate benchmark metrics from results."""
    total_input = 0
    completed = 0
    total_output = 0
    ttfts = []
    itls = []
    e2els = []

    for i, output in enumerate(outputs):
        if output.success:
            output_len = output.output_tokens

            if not output_len:
                # Use tokenizer to count output tokens if not provided
                output_len = len(
                    tokenizer(output.generated_text, add_special_tokens=False).input_ids
                )

            total_output += output_len
            total_input += input_requests[i].prompt_len

            if hasattr(output, "ttft") and output.ttft is not None:
                ttfts.append(output.ttft)

            if hasattr(output, "itl") and output.itl:
                # Ensure itl is a list of floats
                if isinstance(output.itl, list):
                    itls.extend(output.itl)
                else:
                    logger.warning(
                        "Expected list for ITL but got %s. Appending as is.",
                        type(output.itl),
                    )
                    itls.append(output.itl)

            if hasattr(output, "latency") and output.latency is not None:
                e2els.append(output.latency)

            completed += 1

    return BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=total_output,
        mean_ttft_ms=np.mean(ttfts or [0]) * 1000,
        median_ttft_ms=np.median(ttfts or [0]) * 1000,
        std_ttft_ms=np.std(ttfts or [0]) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or [0], p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or [0]) * 1000,
        median_itl_ms=np.median(itls or [0]) * 1000,
        std_itl_ms=np.std(itls or [0]) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or [0], p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or [0]) * 1000,
        median_e2el_ms=np.median(e2els or [0]) * 1000,
        std_e2el_ms=np.std(e2els or [0]) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or [0], p) * 1000) for p in selected_percentiles
        ],
    )


def print_results(metrics: BenchmarkMetrics, benchmark_duration: float):
    """Print benchmark results in a formatted way."""
    print("{s:{c}^{n}}".format(s=" Sequential Benchmark Result ", n=60, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))

    def print_metric_stats(metric_name, header):
        print("{s:{c}^{n}}".format(s=header, n=60, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_name.lower()}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_name.lower()}_ms"),
            )
        )

        for p, value in getattr(metrics, f"percentiles_{metric_name.lower()}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))

    print_metric_stats("TTFT", "Time to First Token")
    print_metric_stats("ITL", "Inter-token Latency")
    print_metric_stats("E2EL", "End-to-end Latency")
    print("=" * 60)


async def main_async(args):
    # Import needed functions based on your setup
    from backend_request_func import ASYNC_REQUEST_FUNCS

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    # Set up API URL
    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    # Set up Cache Reset URL
    cache_reset_url = f"http://{args.host}:{args.port}/reset_prefix_cache"
    logger.info("Prefix cache reset configured at: %s", cache_reset_url)

    # Get tokenizer
    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)

    # Get request function
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    input_requests = RandomDataset().sample(
        tokenizer=tokenizer,
        num_requests=args.num_requests,
        prefix_len=0,
        input_len=args.input_len,
        output_len=args.output_len,
        range_ratio=0.0,
    )

    # Run benchmark
    result = await sequential_benchmark(
        backend=backend,
        api_url=api_url,
        model_id=model_id,
        tokenizer=tokenizer,
        input_requests=input_requests,
        request_func=request_func,
        selected_percentiles=[50, 90, 95, 99],
        cache_reset_url=cache_reset_url,
    )

    return result


def main(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential benchmark for LLM serving")
    parser.add_argument(
        "--backend", type=str, default="vllm", help="Backend to use for requests"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server base URL (overrides --host and --port)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint", type=str, default="/v1/completions", help="API endpoint"
    )
    parser.add_argument("--model", type=str, required=True, help="Name of the model")
    parser.add_argument(
        "--tokenizer", type=str, help="Name of the tokenizer (defaults to model name)"
    )
    parser.add_argument(
        "--num-requests", type=int, default=100, help="Number of requests to process"
    )
    parser.add_argument(
        "--input-len", type=int, default=128, help="Input len for generated prompts"
    )
    parser.add_argument(
        "--output-len", type=int, default=None, help="Override output len for requests"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )

    args = parser.parse_args()
    main(args)
