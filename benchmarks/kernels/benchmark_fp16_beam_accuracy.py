# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate deterministic long-context BEAM answers for paired comparisons."""

import argparse
import asyncio
import gzip
import json
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

PROBING_TEMPLATE = (
    "NOTE: Only provide the answer without any explanations. \nQuestion: {question}"
)


def load_chat(path: Path, chat_id: str) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row["chat_id"] == chat_id:
                return row["messages"]
    raise ValueError(f"chat {chat_id!r} not found in {path}")


def load_questions(path: Path, chat_id: str) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [row for line in file if (row := json.loads(line))["chat_id"] == chat_id]


def truncate_messages(
    messages: list[dict[str, str]], max_tokens: int, tokenizer
) -> list[dict[str, str]]:
    token_counts = [len(tokenizer.encode(message["content"])) for message in messages]
    total = sum(token_counts)
    start = 0
    while start < len(messages) and total > max_tokens:
        total -= token_counts[start]
        start += 1
    while start < len(messages) and messages[start]["role"] != "user":
        start += 1
    return messages[start:]


async def run(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    messages = load_chat(args.chats, args.chat_id)
    content_budget = args.max_context_tokens - args.max_tokens - 5000
    messages = truncate_messages(messages, content_budget, tokenizer)
    questions = load_questions(args.questions, args.chat_id)
    if args.limit:
        questions = questions[: args.limit]
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    headers = {"Content-Type": "application/json"}

    with args.output.open("w", encoding="utf-8") as output:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for question in questions:
                request_messages = messages + [
                    {
                        "role": "user",
                        "content": PROBING_TEMPLATE.format(
                            question=question["question"]
                        ),
                    }
                ]
                payload = {
                    "model": args.model,
                    "messages": request_messages,
                    "temperature": 0,
                    "seed": 0,
                    "max_tokens": args.max_tokens,
                }
                async with session.post(
                    args.url, data=json.dumps(payload), headers=headers
                ) as response:
                    body = await response.json()
                    if response.status != 200:
                        raise RuntimeError(body)
                choice = body["choices"][0]
                usage = body.get("usage", {})
                row = {
                    key: question[key]
                    for key in (
                        "chat_id",
                        "question_type",
                        "question_index",
                        "question",
                        "gold_answer",
                        "rubric",
                    )
                }
                row.update(
                    response=choice["message"]["content"] or "",
                    finish_reason=choice["finish_reason"],
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"{question['question_type']}[{question['question_index']}]: "
                    f"prompt={row['prompt_tokens']} output={row['completion_tokens']}",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--chats", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chat-id", default="1")
    parser.add_argument("--url", default="http://127.0.0.1:8005/v1/chat/completions")
    parser.add_argument("--max-context-tokens", type=int, default=200000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
