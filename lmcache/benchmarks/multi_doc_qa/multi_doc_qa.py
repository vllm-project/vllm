# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_long_document_qa_throughput.py

"""
Commandline arguments:
    --num-total-documents: The number of documents to sample prompts from.

    --document-length: The length of each document in tokens.
                       (Optional, default: 20000)

    --output-len: The number of tokens to generate for each prompt.
                  (Optional, default: 100)

    --num-requests: The number of requests to send.

    --num-docs-per-request: The number of documents to use in each prompt.

    --sampling-strategy: The sampling strategy to use. Currently only supports
                         "random".

    --random-seed: Random seed when the repeat mode is "random".
                    (Optional, default: 0)

    --blend-special-str: The special string to use for blending documents.
                         (Optional, default: " # # ")

    --port: Port to query the vLLM server

    --model: Model name

    --max-inflight-requests: Maximum number of in-flight requests. Default is 2

    --sleep-time-after-warmup: Sleep time after warm up iteration.
                              (Optional, default: 0.0 seconds)

    --output: Filename to write all responses to. If omitted, writes to stdout.

    --expected-ttft-gain: Expected minimum speed-up in time-to-first-token
                         (warmup/query) as a factor, e.g. 4.3 for 4.3×. If
                         actual gain is below this, exits.

    --expected-latency-gain: Expected minimum speed-up in total round time
                            (warmup/query) as a factor, e.g. 4.5 for 4.5×.
                            If actual gain is below this, exits.
"""

# Standard
import argparse
import asyncio
import random
import sys
import time

# Third Party
from openai import AsyncOpenAI
from transformers import AutoTokenizer

# Global output filename (set in __main__)
OUTPUT_FILE = None


def has_content(chunk):
    """
    Check if the chunk has content in the choices.
    Args:
        chunk: The response chunk from OpenAI API.

    Returns:
        bool: True if content exists, False otherwise.
    """
    return chunk.choices and chunk.choices[0].text


def extract_content(chunk):
    """
    Extract content from the response chunk.
    Args:
        chunk: The response chunk from OpenAI API.
    Returns:
        str: The content extracted from the chunk.
    """
    if chunk.choices[0].text is not None:
        return chunk.choices[0].text
    else:
        return ""


def write_resp(text: str):
    """
    Write text to the specified output file (if any), otherwise to stdout.
    """
    if OUTPUT_FILE:
        with open(OUTPUT_FILE, "a") as resp_file:
            resp_file.write(text)
    else:
        sys.stdout.write(text)


async def process_single_prompt(
    client, model, prompt, prompt_index, total_prompts, output_len, semaphore
):
    """
    Process a single prompt with the given client and model.

    Args:
        client: The OpenAI client for making API calls.
        model: The model name to use for generation.
        prompt: The prompt string to be processed.
        prompt_index: Index of the current prompt (0-based).
        total_prompts: Total number of prompts being processed.
        output_len: The maximum number of tokens to generate.
        semaphore: Asyncio semaphore to limit concurrent requests.

    Returns:
        float: Time-to-first-token measurement
    """
    async with semaphore:  # Acquire semaphore to limit concurrent requests
        write_resp(f"\n--- Sending prompt {prompt_index + 1}/{total_prompts} ---\n")
        start_time = time.time()
        first_token_time = None
        words = ""

        response = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=output_len,
            temperature=0.0,
            stream=True,
            extra_body={"ignore_eos": True},
        )

        responses = []
        # Collect the response chunks
        async for chunk in response:
            if not chunk.choices:
                continue

            # Handle content for chat completions
            if has_content(chunk):
                content = extract_content(chunk)
                if first_token_time is None and content != "":
                    first_token_time = time.time()
                responses.append(content)
                words += content

        final_response = "".join(responses)
        write_resp(f"\nResponse of request {prompt_index}: {final_response}\n")

        if first_token_time is not None:
            return first_token_time - start_time
        else:
            # If no content was generated, return a default value
            return 0.0


async def test_long_document_qa(
    client, model, prompts=None, output_len=100, max_inflight_requests=10
):
    """
    Test long document QA with the given prompts and sampling parameters.
    Process prompts concurrently with a limit on inflight requests.

    Args:
        client: The OpenAI client for making API calls.
        model: The model name to use for generation.
        prompts: A list of prompt strings to be processed by the LLM.
        output_len: The maximum number of tokens to generate.
        max_inflight_requests: Maximum number of concurrent requests.

    Returns:
        list: ttfts - a list of time-to-first-token measurements
    """
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_inflight_requests)

    # Create tasks for all prompts
    tasks = []
    for i, prompt in enumerate(prompts):
        task = process_single_prompt(
            client=client,
            model=model,
            prompt=prompt,
            prompt_index=i,
            total_prompts=len(prompts),
            output_len=output_len,
            semaphore=semaphore,
        )
        tasks.append(task)

    # Execute all tasks concurrently and collect results
    ttfts = await asyncio.gather(*tasks)

    return ttfts


def generate_warmup_prompt_ids(
    doc_prompts, sys_prompts, query_prompts, blend_special_str, tokenizer, offset=1
):
    blend_special_ids = tokenizer.encode(blend_special_str)[offset:]
    warmup_prompt_ids = []
    for doc_prompt, sys_prompt, query_prompt in zip(
        doc_prompts, sys_prompts, query_prompts, strict=False
    ):
        sys_prompt_ids = tokenizer.encode(sys_prompt)
        doc_prompt_ids = tokenizer.encode(doc_prompt)[offset:]
        query_prompt_ids = tokenizer.encode(query_prompt)[offset:]
        warmup_prompt_ids.append(
            sys_prompt_ids
            + blend_special_ids
            + doc_prompt_ids
            + blend_special_ids
            + query_prompt_ids
        )
    return warmup_prompt_ids


def generate_prompt_ids(
    doc_prompts: list[str],
    sys_prompts: list[str],
    query_prompts: list[str],
    num_requests: int,
    num_docs_per_request: int,
    blend_special_str: str,
    tokenizer,
    offset: int = 1,
):
    blend_special_ids = tokenizer.encode(blend_special_str)[offset:]

    prompt_ids = []

    for i in range(num_requests):
        temp_prompt_ids = []
        sample_docs = random.sample(doc_prompts, num_docs_per_request)
        sample_docs_ids = [tokenizer.encode(doc)[offset:] for doc in sample_docs]
        sys_prompt_ids = tokenizer.encode(sys_prompts[i])
        query_prompt_ids = tokenizer.encode(query_prompts[i])[offset:]
        temp_prompt_ids += sys_prompt_ids
        for doc_ids in sample_docs_ids:
            temp_prompt_ids += blend_special_ids + doc_ids
        temp_prompt_ids += blend_special_ids + query_prompt_ids

        prompt_ids.append(temp_prompt_ids)

    return prompt_ids


async def main(args):
    random.seed(args.random_seed)

    # Create the OpenAI client
    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1", api_key="sk-dummy"
    )
    model = args.model
    blend_special_str = args.blend_special_str
    num_requests = args.num_requests
    num_docs_per_request = args.num_docs_per_request
    document_length = args.document_length
    num_total_documents = args.num_total_documents

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    doc_prompts = [
        str(i) + " " + " ".join(["hi"] * document_length)
        for i in range(num_total_documents)
    ]
    warmup_sys_prompts = ["You are a helpful assistant."] * num_total_documents
    warmup_query_prompts = ["What's up? how are you recently?"] * num_total_documents

    warmup_prompt_ids = generate_warmup_prompt_ids(
        doc_prompts,
        warmup_sys_prompts,
        warmup_query_prompts,
        blend_special_str,
        tokenizer,
        offset=1,
    )

    sys_prompts = ["You are a helpful assistant."] * num_requests
    query_prompts = ["What's up? how are you recently?"] * num_requests

    prompt_ids = generate_prompt_ids(
        doc_prompts,
        sys_prompts,
        query_prompts,
        num_requests,
        num_docs_per_request,
        blend_special_str,
        tokenizer,
        offset=1,
    )

    write_resp("------warm up round------\n")
    warmup_start_time = time.time()
    warmup_ttfts = await test_long_document_qa(
        client=client,
        model=model,
        prompts=warmup_prompt_ids,
        output_len=args.output_len,
        max_inflight_requests=args.max_inflight_requests,
    )
    warmup_end_time = time.time()
    write_resp("------query round------\n")

    sleep_time_after_warmup = args.sleep_time_after_warmup
    if sleep_time_after_warmup > 0:
        write_resp(f"Sleeping for {sleep_time_after_warmup} seconds after warmup...\n")
        time.sleep(sleep_time_after_warmup)

    benchmark_start_time = time.time()
    benchmark_ttfts = await test_long_document_qa(
        client=client,
        model=model,
        prompts=prompt_ids,
        output_len=args.output_len,
        max_inflight_requests=args.max_inflight_requests,
    )
    benchmark_end_time = time.time()

    # Print results
    warmup_mean_ttft = sum(warmup_ttfts) / len(warmup_ttfts)
    query_mean_ttft = sum(benchmark_ttfts) / len(benchmark_ttfts)
    CSI = "\x1b["
    RESET = CSI + "0m"
    print(f"{CSI}36;1m\n=== BENCHMARK RESULTS ==={RESET}")
    print(f"{CSI}32mWarmup round mean TTFT: {warmup_mean_ttft:.3f}s{RESET}")
    print(
        f"{CSI}33mWarmup round time: {warmup_end_time - warmup_start_time:.3f}s{RESET}"
    )
    print(f"{CSI}35mWarmup round prompt count: {len(warmup_ttfts)}{RESET}")
    print(f"{CSI}32mQuery round mean TTFT: {query_mean_ttft:.3f}s{RESET}")
    print(
        f"{CSI}33mQuery round time: "
        f"{benchmark_end_time - benchmark_start_time:.3f}s{RESET}"
    )
    print(f"{CSI}35mQuery round prompt count: {len(benchmark_ttfts)}{RESET}")

    # Validate expected gains as multiplicative speed-ups
    if args.expected_ttft_gain is not None:
        actual_ttft_gain = (
            warmup_mean_ttft / query_mean_ttft if query_mean_ttft > 0 else float("inf")
        )
        print(f"{CSI}34mActual TTFT gain: {actual_ttft_gain:.2f}×{RESET}")
        if actual_ttft_gain < args.expected_ttft_gain:
            sys.exit(
                f"ERROR: TTFT gain {actual_ttft_gain:.2f}× < expected "
                f"{args.expected_ttft_gain:.2f}×"
            )

    if args.expected_latency_gain is not None:
        warmup_duration = warmup_end_time - warmup_start_time
        query_duration = benchmark_end_time - benchmark_start_time

        # compute per-prompt latency before comparing
        warmup_per_prompt = warmup_duration / len(warmup_ttfts)
        query_per_prompt = query_duration / len(benchmark_ttfts)
        actual_latency_gain = (
            warmup_per_prompt / query_per_prompt
            if query_per_prompt > 0
            else float("inf")
        )
        print(f"{CSI}34mActual latency gain: {actual_latency_gain:.2f}×{RESET}")
        if actual_latency_gain < args.expected_latency_gain:
            sys.exit(
                f"ERROR: latency gain {actual_latency_gain:.2f}× < expected "
                f"{args.expected_latency_gain:.2f}×"
            )


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark the performance forMulti-Doc QA."
    )

    parser.add_argument(
        "--document-length",
        type=int,
        # Roughly the number of tokens for a system paper,
        # excluding images
        default=3000,
        help="Length of each document in tokens.",
    )

    parser.add_argument(
        "--num-total-documents",
        type=int,
        default=100,
        help="Number of documents to generate for testing.",
    )

    parser.add_argument(
        "--output-len",
        type=int,
        default=10,
        help="Maximum number of tokens to generate for each prompt.",
    )

    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to send.",
    )

    parser.add_argument(
        "--num-docs-per-request",
        type=int,
        default=5,
        help="Number of requests to send.",
    )

    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="random",
        help="Random seed for sampling",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help='Random seed when the repeat mode is "random"',
    )

    parser.add_argument(
        "--blend-special-str",
        type=str,
        default=" # # ",
        help="Special string to separate different documents.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to query the vLLM server",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name",
    )

    parser.add_argument(
        "--max-inflight-requests",
        type=int,
        default=20,
        help="Maximum number of concurrent inflight requests",
    )

    parser.add_argument(
        "--sleep-time-after-warmup",
        type=float,
        default=0.0,
        help="Sleep time after warm up iteration",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Filename to write all responses to; if omitted, writes to stdout.",
    )
    parser.add_argument(
        "--expected-ttft-gain",
        type=float,
        default=None,
        help=(
            "Expected minimum speed-up in time-to-first-token (warmup/query) "
            "as a factor, e.g. 4.3 for 4.3×. If actual gain is below this, exits."
        ),
    )
    parser.add_argument(
        "--expected-latency-gain",
        type=float,
        default=None,
        help=(
            "Expected minimum speed-up in total round time (warmup/query) "
            "as a factor, e.g. 4.5 for 4.5×. If actual gain is below this, exits."
        ),
    )

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    OUTPUT_FILE = args.output
    asyncio.run(main(args))
