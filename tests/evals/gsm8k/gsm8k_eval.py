#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated GSM8K evaluation script for vLLM serve endpoint."""

import argparse
import ast
import asyncio
import json
import os
import tempfile
import time
from collections.abc import Generator

import aiohttp
import numpy as np
import regex as re
import requests
from requests.adapters import HTTPAdapter
from tqdm.asyncio import tqdm
from urllib3.util.retry import Retry

from vllm.assets.base import VLLM_S3_BUCKET_URL

INVALID = -9999999


def download_and_cache_file(url: str, filename: str | None = None) -> str:
    """Download and cache a file from a URL."""
    if filename is None:
        filename = os.path.join(tempfile.gettempdir(), url.split("/")[-1])

    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    with requests.Session() as session:
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

    return filename


def load_gsm8k_data() -> tuple[list[dict], list[dict]]:
    """Load GSM8K train and test data."""
    train_url = f"{VLLM_S3_BUCKET_URL}/ci-datasets/gsm8k/train.jsonl"
    test_url = f"{VLLM_S3_BUCKET_URL}/ci-datasets/gsm8k/test.jsonl"

    train_file = download_and_cache_file(train_url)
    test_file = download_and_cache_file(test_url)

    train_data = list(read_jsonl(train_file))
    test_data = list(read_jsonl(test_file))

    return train_data, test_data


def read_jsonl(filename: str) -> Generator[dict, None, None]:
    """Read a JSONL file."""
    with open(filename) as fin:
        for line in fin:
            if not line.startswith("#"):
                yield json.loads(line)


def get_answer_value(answer_str: str) -> int:
    """Extract the numerical answer from the response."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def _optional_params(**params: object) -> dict[str, object]:
    return {k: v for k, v in params.items() if v is not None}


async def call_vllm_api(
    session: aiohttp.ClientSession,
    prompt: str,
    temperature: float | None,
    max_tokens: int,
    stop: list[str] | None = None,
    url: str | None = None,
    seed: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> tuple[str, int]:
    """Call vLLM's OpenAI-compatible completions endpoint.

    Returns:
        Tuple of (response_text, completion_tokens)

    """
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stop": stop,
    }
    data.update(
        _optional_params(temperature=temperature, seed=seed, top_p=top_p, top_k=top_k)
    )

    try:
        async with session.post(f"{url}/v1/completions", json=data) as response:
            response.raise_for_status()
            result = await response.json()
            text = result["choices"][0]["text"]
            completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
            return text, completion_tokens
    except Exception as e:
        print(f"Error calling vLLM API ({type(e).__name__}): {e}")
        return "", 0


async def call_vllm_chat_api(
    session: aiohttp.ClientSession,
    model: str | None,
    prompt: str,
    temperature: float | None,
    max_tokens: int,
    stop: list[str] | None = None,
    url: str | None = None,
    seed: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    reasoning_effort: str | None = None,
    chat_template_kwargs: dict[str, object] | None = None,
) -> tuple[str, int]:
    """Call vLLM's OpenAI-compatible chat completions endpoint.

    ``model``, ``temperature``, ``stop`` and the other optional fields are
    omitted from the request when ``None``, so the server defaults apply (vLLM
    uses its served model when ``model`` is omitted).

    Returns:
        Tuple of (final answer content, completion_tokens). Reasoning returned
        separately by a reasoning parser is not part of the answer.

    """
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    data.update(
        _optional_params(
            stop=stop,
            model=model,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            top_k=top_k,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs,
        )
    )

    try:
        async with session.post(f"{url}/v1/chat/completions", json=data) as response:
            response.raise_for_status()
            result = await response.json()
            text = result["choices"][0]["message"]["content"] or ""
            completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
            return text, completion_tokens
    except Exception as e:
        print(f"Error calling vLLM chat API ({type(e).__name__}): {e}")
        return "", 0


def _build_gsm8k_prompts(
    num_questions: int = 1319,
    num_shots: int = 5,
    gen_prefix: str = "",
) -> tuple[list[str], list[int]]:
    """Build few-shot GSM8K completion prompts and ground-truth labels."""
    if num_questions == 0:
        return [], []
    train_data, test_data = load_gsm8k_data()
    num_questions = min(num_questions, len(test_data))

    few_shot_examples = ""
    for i in range(num_shots):
        few_shot_examples += (
            f"Question: {train_data[i]['question']}\n"
            f"Answer:{gen_prefix} {train_data[i]['answer']}\n\n"
        )

    prompts = []
    labels = []
    for i in range(num_questions):
        prompts.append(
            few_shot_examples
            + f"Question: {test_data[i]['question']}\nAnswer:{gen_prefix}"
        )
        labels.append(get_answer_value(test_data[i]["answer"]))

    assert all(label != INVALID for label in labels), "Some labels are invalid"
    return prompts, labels


def _score_gsm8k(
    states: list[str],
    output_tokens: list[int],
    labels: list[int],
    num_shots: int,
    max_tokens: int,
    latency: float,
) -> dict[str, float | int]:
    """Score GSM8K responses and return a results dict."""
    num_questions = len(labels)
    preds = [get_answer_value(state) for state in states]
    accuracy = np.mean(np.array(preds) == np.array(labels))
    invalid_rate = np.mean(np.array(preds) == INVALID)
    total_output_tokens = sum(output_tokens)
    tokens_per_second = total_output_tokens / latency if latency > 0 else 0.0

    return {
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
        "latency": latency,
        "questions_per_second": num_questions / latency if latency > 0 else 0.0,
        "total_output_tokens": total_output_tokens,
        "tokens_per_second": tokens_per_second,
        "num_questions": num_questions,
        "num_shots": num_shots,
        "max_tokens": max_tokens,
        "timestamp": time.time(),
    }


def evaluate_gsm8k(
    num_questions: int = 1319,
    num_shots: int = 5,
    max_tokens: int = 256,
    model: str | None = None,
    use_chat_completions: bool = False,
    host: str = "http://127.0.0.1",
    port: int = 8000,
    temperature: float | None = 0.0,
    seed: int | None = 42,
    request_timeout_seconds: float = 600,
    gen_prefix: str = "",
    max_concurrency: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    reasoning_effort: str | None = None,
    chat_template_kwargs: dict[str, object] | None = None,
) -> dict[str, float | int]:
    """Evaluate GSM8K accuracy using vLLM serve endpoint.

    ``temperature``, ``top_p`` and ``top_k`` are sent only when not ``None``;
    otherwise the server default applies. ``model`` is used only in chat mode,
    where it may be ``None`` (the server uses its served model).
    ``reasoning_effort`` and ``chat_template_kwargs`` require
    ``use_chat_completions=True``. Stop strings are sent only in completions
    mode.

    Returns dict with accuracy, invalid_rate, latency, etc.
    """
    if not use_chat_completions and (
        reasoning_effort is not None or chat_template_kwargs is not None
    ):
        raise ValueError(
            "reasoning_effort and chat_template_kwargs require "
            "use_chat_completions=True"
        )
    base_url = f"{host}:{port}"
    prompts, labels = _build_gsm8k_prompts(num_questions, num_shots, gen_prefix)
    num_questions = len(prompts)

    async def run_async_evaluation():
        states: list[str] = [""] * num_questions
        output_tokens: list[int] = [0] * num_questions

        async def get_answer(session: aiohttp.ClientSession, i: int) -> tuple[str, int]:
            if use_chat_completions:
                # No stop strings: the chat template delimits the turn, and they
                # would also cut off reasoning.
                answer, tokens = await call_vllm_chat_api(
                    session=session,
                    model=model,
                    prompt=prompts[i],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    url=base_url,
                    seed=seed,
                    top_p=top_p,
                    top_k=top_k,
                    reasoning_effort=reasoning_effort,
                    chat_template_kwargs=chat_template_kwargs,
                )
            else:
                answer, tokens = await call_vllm_api(
                    session=session,
                    prompt=prompts[i],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=["Question", "Assistant:", "<|separator|>"],
                    url=base_url,
                    seed=seed,
                    top_p=top_p,
                    top_k=top_k,
                )
            states[i] = answer
            output_tokens[i] = tokens
            return answer, tokens

        timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
        connector = (
            aiohttp.TCPConnector(limit=max_concurrency)
            if max_concurrency is not None
            else None
        )
        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            tasks = [get_answer(session, i) for i in range(num_questions)]
            await tqdm.gather(*tasks, desc="Evaluating")

        return states, output_tokens

    print(f"Running GSM8K evaluation: {num_questions} questions, {num_shots}-shot")

    tic = time.perf_counter()
    states, output_tokens = asyncio.run(run_async_evaluation())
    latency = time.perf_counter() - tic

    return _score_gsm8k(states, output_tokens, labels, num_shots, max_tokens, latency)


def evaluate_gsm8k_offline(
    llm,
    num_questions: int = 1319,
    num_shots: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.0,
    gen_prefix: str = "",
    use_chat_completions: bool = False,
    chat_template_kwargs: dict[str, object] | None = None,
) -> dict[str, float | int]:
    """Evaluate GSM8K accuracy using an offline vllm.LLM object.

    Same prompts and scoring as evaluate_gsm8k(), but runs generation
    directly via llm.generate() instead of calling a server over HTTP.

    When ``use_chat_completions=True``, prompts go through the chat template via
    ``llm.chat()`` instead of raw completion (for instruction-tuned models).
    ``chat_template_kwargs`` are forwarded to ``llm.chat()`` when provided.
    """
    from vllm import SamplingParams

    prompts, labels = _build_gsm8k_prompts(num_questions, num_shots, gen_prefix)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["Question", "Assistant:", "<|separator|>"],
    )
    mode = "chat" if use_chat_completions else "completion"
    print(
        f"Running offline GSM8K evaluation: {len(prompts)} questions, "
        f"{num_shots}-shot, {mode}"
    )

    tic = time.perf_counter()
    if use_chat_completions:
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = llm.chat(
            conversations,
            sampling_params,
            chat_template_kwargs=chat_template_kwargs,
        )
    else:
        outputs = llm.generate(prompts, sampling_params)
    latency = time.perf_counter() - tic

    states = [o.outputs[0].text for o in outputs]
    output_tokens = [len(o.outputs[0].token_ids) for o in outputs]

    return _score_gsm8k(states, output_tokens, labels, num_shots, max_tokens, latency)


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K evaluation for vLLM serve")
    parser.add_argument(
        "--num-shots", type=int, default=5, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=1319,
        help="Number of questions to evaluate",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens for generation"
    )
    parser.add_argument("--host", type=str, default="http://127.0.0.1", help="Host URL")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for generation (default: 0 for completions; "
        "server default with --use-chat-completions)",
    )
    parser.add_argument(
        "--top-p", type=float, help="Top-p for generation (server default if unset)"
    )
    parser.add_argument(
        "--top-k", type=int, help="Top-k for generation (server default if unset)"
    )
    parser.add_argument(
        "--use-chat-completions",
        action="store_true",
        help="Send each few-shot prompt as a user message to /v1/chat/completions "
        "instead of /v1/completions (for chat-only models)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to send with chat requests (default: omitted, so the "
        "server uses its served model)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        help="reasoning_effort for chat completions, e.g. none, low, high",
    )
    parser.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        help="JSON chat_template_kwargs, e.g. '{\"enable_thinking\": false}'",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=600,
        help="Timeout for each request, including time waiting for a connection",
    )
    parser.add_argument("--save-results", type=str, help="Save results to JSON file")

    args = parser.parse_args()
    temperature = args.temperature
    if temperature is None and not args.use_chat_completions:
        temperature = 0.0

    result = evaluate_gsm8k(
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        max_tokens=args.max_tokens,
        host=args.host,
        port=args.port,
        temperature=temperature,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
        request_timeout_seconds=args.request_timeout_seconds,
        model=args.model,
        use_chat_completions=args.use_chat_completions,
        top_p=args.top_p,
        top_k=args.top_k,
        reasoning_effort=args.reasoning_effort,
        chat_template_kwargs=args.chat_template_kwargs,
    )

    # Print results to terminal
    print("\nResults:")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"Invalid responses: {result['invalid_rate']:.3f}")
    print(f"Total latency: {result['latency']:.3f} s")
    print(f"Questions per second: {result['questions_per_second']:.3f}")
    print(f"Total output tokens: {result['total_output_tokens']}")
    print(f"Output tokens per second: {result['tokens_per_second']:.3f}")

    # Optional file saving
    if args.save_results:
        # None means the field was not sent and the server default applied.
        result["request_params"] = {
            "endpoint": "chat" if args.use_chat_completions else "completions",
            "model": args.model,
            "temperature": temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "stop": None
            if args.use_chat_completions
            else ["Question", "Assistant:", "<|separator|>"],
            "reasoning_effort": args.reasoning_effort,
            "chat_template_kwargs": args.chat_template_kwargs,
        }
        with open(args.save_results, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.save_results}")


if __name__ == "__main__":
    main()
