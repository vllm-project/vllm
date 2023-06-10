"""Benchmark the online serving throughput.

On the server side, run:
    python -m cacheflow.entrypoints.simple_fastapi_frontend \
        --disable-log-requests --model <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import aiohttp
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple

from cacheflow.server.tokenizer_utils import get_tokenizer  # FIXME(woosuk)
import numpy as np
from transformers import PreTrainedTokenizerBase


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
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

    # Filter out if the prompt length + output length is greater than 2048.
    tokenized_dataset = [
        (prompt, output_len)
        for prompt, prompt_token_ids, output_len in tokenized_dataset
        if len(prompt_token_ids) + output_len <= 2048
    ]

    # Sample the requests.
    sampled_requests = random.sample(tokenized_dataset, num_requests)
    return sampled_requests


async def send_request(
    api_url: str,
    prompt: str,
    output_len: int,
    n: int,
    use_beam_search: bool,
) -> None:
    headers = {"User-Agent": "Benchmark Client"}
    pload = {
        "prompt": prompt,
        "n": n,
        "use_beam_search": use_beam_search,
        "temperature": 0.0 if use_beam_search else 1.0,
        "top_p": 1.0,
        "max_tokens": output_len,
        "ignore_eos": True,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=pload) as response:
            chunks = []
            async for chunk, _ in response.content.iter_chunks():
                chunks.append(chunk)
        output = b"".join(chunks).decode("utf-8")


async def get_request(
    input_requests: List[Tuple[str, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int], None]:
    input_requests = iter(input_requests)
    while True:
        try:
            yield next(input_requests)
        except StopIteration:
            return

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to sleep.
            continue
        # Sample the interval between requests from an exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # Make sure the next request is sent after the interval.
        time.sleep(interval)


async def benchmark(
    api_url: str,
    input_requests: List[Tuple[str, int]],
    n: int,
    use_beam_search: bool,
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    async for prompt, output_len in get_request(input_requests, request_rate):
        task = asyncio.create_task(
            send_request(api_url, prompt, output_len, n, use_beam_search))
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    tokenizer = get_tokenizer(args.tokenizer)
    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)
    asyncio.run(benchmark(api_url, input_requests, args.n, args.use_beam_search,
                          args.request_rate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process for request "
                             "arrival times.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
