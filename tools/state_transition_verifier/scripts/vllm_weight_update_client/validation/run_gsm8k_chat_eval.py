#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import json
import sys
import time

import aiohttp
import numpy as np

sys.path.insert(0, "/path/to/vllm")
from tests.evals.gsm8k.gsm8k_eval import (
    INVALID,
    _build_gsm8k_prompts,
    get_answer_value,
)


CHAT_SUFFIX = "\nSolve step by step. End with the final answer as a number."


async def evaluate(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    prompts, labels = _build_gsm8k_prompts(args.num_questions, args.num_shots)
    records: list[dict] = [{} for _ in prompts]
    timeout = aiohttp.ClientTimeout(total=1800)
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def request_one(index: int) -> None:
            payload = {
                "model": args.model,
                "prompt": prompts[index],
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "seed": 42,
                "stop": ["Question", "Assistant:", "<|separator|>"],
            }
            endpoint = "completions"
            if args.api == "chat":
                question = prompts[index].rsplit("Question: ", 1)[-1]
                question = question.removesuffix("\nAnswer:").strip()
                payload = {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": question + CHAT_SUFFIX,
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": args.max_tokens,
                    "seed": 42,
                }
                endpoint = "chat/completions"
            try:
                async with (
                    semaphore,
                    session.post(
                        f"{args.host}:{args.port}/v1/{endpoint}", json=payload
                    ) as response,
                ):
                    body = await response.json()
                    response.raise_for_status()
                choice = body["choices"][0]
                if args.api == "chat":
                    message = choice.get("message") or {}
                    content = message.get("content") or ""
                    reasoning = (
                        message.get("reasoning_content")
                        or message.get("reasoning")
                        or ""
                    )
                    score_text = content.rsplit("</think>", 1)[-1]
                else:
                    content = choice.get("text") or ""
                    reasoning = ""
                    score_text = content
                records[index] = {
                    "index": index,
                    "response_id": body.get("id"),
                    "label": labels[index],
                    "prediction": get_answer_value(score_text),
                    "content": content,
                    "reasoning": reasoning,
                    "reasoning_content": reasoning,
                    "text": content,
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": body.get("usage", {}).get(
                        "completion_tokens", 0
                    ),
                }
            except Exception as error:
                records[index] = {
                    "index": index,
                    "label": labels[index],
                    "prediction": INVALID,
                    "error": f"{type(error).__name__}: {error}",
                    "completion_tokens": 0,
                }

        started = time.perf_counter()
        await asyncio.gather(*(request_one(i) for i in range(len(prompts))))
        latency = time.perf_counter() - started

    predictions = np.array([record["prediction"] for record in records])
    labels_array = np.array(labels)
    correct_count = int(np.sum(predictions == labels_array))
    total_tokens = sum(record["completion_tokens"] for record in records)
    request_error_count = sum("error" in record for record in records)
    summary = {
        "accuracy": float(correct_count / len(records)),
        "correct_count": correct_count,
        "invalid_rate": float(np.mean(predictions == INVALID)),
        "latency": latency,
        "questions_per_second": len(records) / latency,
        "total_output_tokens": total_tokens,
        "request_error_count": request_error_count,
        "tokens_per_second": total_tokens / latency,
        "num_questions": len(records),
        "num_shots": args.num_shots,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "concurrency": args.concurrency,
        "connection_limit": args.concurrency,
        "scored_text": "completion text"
        if args.api == "completions"
        else "chat content after </think>",
        "api": args.api,
    }
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-questions", type=int, default=1319)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=1319)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--api", choices=["completions", "chat"], default="completions")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.num_questions < 1:
        parser.error("--num-questions must be at least 1")
    if args.num_shots < 0:
        parser.error("--num-shots must be non-negative")
    summary, records = asyncio.run(evaluate(args))
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
    with open(args.details, "w", encoding="utf-8") as details:
        for record in records:
            details.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["request_error_count"]:
        raise SystemExit(f"GSM8K request failures: {summary['request_error_count']}")


if __name__ == "__main__":
    main()
