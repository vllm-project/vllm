"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import argparse
import asyncio
import json
import random
import time
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Tuple, Union

import numpy as np
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

from backend_request_func import ASYNC_REQUEST_FUNCS


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: Dict[str, Union[str, bool, float]],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[int, int, int, float, float, float, float, float]:
    total_output = 0
    total_input = 0
    completed = 0
    per_token_latencies = []
    for i in range(len(outputs)):
        if outputs[i]["success"]:
            output_len = len(tokenizer.encode(outputs[i]["generated_text"]))
            total_output += output_len
            total_input += input_requests[i][1]
            per_token_latencies.append(outputs[i]["latency"] / output_len)
            completed += 1

    request_throughput = completed / dur_s
    input_throughput = total_input / dur_s
    output_throughput = total_output / dur_s
    mean_tpot_ms = np.mean(per_token_latencies) * 1000
    median_tpot_ms = np.median(per_token_latencies) * 1000
    p99_tpot_ms = np.percentile(per_token_latencies, 99) * 1000

    return (
        completed,
        total_input,
        total_output,
        request_throughput,
        input_throughput,
        output_throughput,
        mean_tpot_ms,
        median_tpot_ms,
        p99_tpot_ms,
    )


async def throughput_benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print(f"Traffic request rate: {request_rate}")

    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_kwargs = {
            "model": model_id,
            "prompt": prompt,
            "api_url": api_url,
            "prompt_len": prompt_len,
            "output_len": output_len,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
        }
        tasks.append(asyncio.create_task(request_func(**request_func_kwargs)))
    outputs = await asyncio.gather(*tasks)
    benchmark_duration = time.perf_counter() - benchmark_start_time

    (
        completed,
        total_input,
        total_output,
        request_throughput,
        input_throughput,
        output_throughput,
        mean_tpot_ms,
        median_tpot_ms,
        p99_tpot_ms,
    ) = calculate_metrics(
        input_requests, outputs, benchmark_duration, tokenizer
    )

    print(f"Successful requests: {completed}")
    print(f"Benchmark duration: {benchmark_duration:2f} s")
    print(f"Total input tokens: {total_input}")
    print(f"Total generated tokens: {total_output}")
    print(f"Reuqest throughput: {request_throughput:.2f} requests/s")
    print(f"Input token throughput: {input_throughput:.2f} tokens/s")
    print(f"Output token throughput: {output_throughput:.2f} tokens/s")
    print(f"Mean latency per output token: {mean_tpot_ms:.2f} ms")
    print(f"Median latency per output token: {median_tpot_ms:.2f} ms")
    print(f"P99 latency per output token: {p99_tpot_ms:.2f} ms")

    result = {}
    result["completed"] = completed
    result["total_input"] = total_input
    result["total_output"] = total_output
    result["request_throughput"] = request_throughput
    result["input_throughput"] = input_throughput
    result["output_throughput"] = output_throughput
    result["duration"] = benchmark_duration
    result["mean_tpot_ms"] = mean_tpot_ms
    result["median_tpot_ms"] = median_tpot_ms
    result["p99_tpot_ms"] = p99_tpot_ms

    return result


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.api_url is not None:
        api_url = f"{args.api_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = get_tokenizer(
        tokenizer_id, trust_remote_code=args.trust_remote_code
    )
    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    benchmark_result = asyncio.run(
        throughput_benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            request_rate=args.request_rate,
        )
    )

    # Save config and results to json
    if args.save_result:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["version"] = args.version
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "tgi", "openai", "deepspeed-mii", "tensorrt-llm"],
    )
    parser.add_argument("--version", type=str, default="N/A")
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Server url or api base if not using host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        default="/generate",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default model tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="save benchmark results to a json file",
    )

    args = parser.parse_args()
    main(args)
