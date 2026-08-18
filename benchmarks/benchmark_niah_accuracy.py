# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run a deterministic needle-in-a-haystack serving accuracy benchmark."""

import argparse
import asyncio
import gzip
import json
import re
from collections.abc import Mapping
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

INSTRUCTION = (
    "Read the document below carefully. A seven-digit pass key is hidden in it. "
    "Remember the key and answer the question after the document.\n\n"
    "<document>\n"
)
QUESTION = (
    "\n</document>\n\nWhat is the seven-digit pass key hidden in the document? "
    "Return only the seven digits.\nAnswer:"
)
NEEDLE = (
    "\n\nImportant record: The seven-digit pass key is {key}. "
    "Remember that {key} is the seven-digit pass key.\n\n"
)


def load_haystack(path: Path, chat_id: str, tokenizer) -> list[int]:
    if path.is_dir():
        text = "\n\n".join(
            file.read_text(errors="ignore") for file in sorted(path.glob("*.txt"))
        )
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                if row["chat_id"] == chat_id:
                    text = "\n\n".join(
                        message["content"] for message in row["messages"]
                    )
                    break
            else:
                raise ValueError(f"chat {chat_id!r} not found in {path}")
    else:
        text = path.read_text(errors="ignore")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"haystack {path} has no text")
    return token_ids


def chat_prompt_length(tokenizer, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    return len(encoded)


def repeated_prefix(token_ids: list[int], length: int) -> list[int]:
    copies, remainder = divmod(length, len(token_ids))
    return token_ids * copies + token_ids[:remainder]


def make_case(
    tokenizer,
    haystack_ids: list[int],
    target_tokens: int,
    depth: float,
    key: str,
) -> tuple[list[dict[str, str]], int]:
    needle = NEEDLE.format(key=key)
    fixed_text = INSTRUCTION + needle + QUESTION
    fixed_tokens = chat_prompt_length(
        tokenizer, [{"role": "user", "content": fixed_text}]
    )
    document_tokens = target_tokens - fixed_tokens
    if document_tokens <= 0:
        raise ValueError(f"target {target_tokens} is too short for the prompt")

    document_ids = repeated_prefix(haystack_ids, document_tokens)
    split = round(len(document_ids) * depth)
    before = tokenizer.decode(document_ids[:split], skip_special_tokens=True)
    after = tokenizer.decode(document_ids[split:], skip_special_tokens=True)
    messages = [
        {
            "role": "user",
            "content": INSTRUCTION + before + needle + after + QUESTION,
        }
    ]
    actual_tokens = chat_prompt_length(tokenizer, messages)
    return messages, actual_tokens


async def run(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    haystack_ids = load_haystack(args.haystack, args.chat_id, tokenizer)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    headers = {"Content-Type": "application/json"}

    case_index = 0
    with args.output.open("w", encoding="utf-8") as output:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for target_tokens in args.context_lengths:
                for depth in args.depths:
                    for seed in range(args.seeds):
                        key = str(1_000_003 + case_index * 7919)
                        messages, actual_tokens = make_case(
                            tokenizer, haystack_ids, target_tokens, depth, key
                        )
                        payload = {
                            "model": args.model,
                            "messages": messages,
                            "temperature": 0,
                            "seed": 0,
                            "max_tokens": args.max_tokens,
                            "chat_template_kwargs": {"enable_thinking": False},
                        }
                        async with session.post(
                            args.url,
                            data=json.dumps(payload),
                            headers=headers,
                        ) as response:
                            body = await response.json()
                            if response.status != 200:
                                raise RuntimeError(body)
                        choice = body["choices"][0]
                        response_text = choice["message"]["content"] or ""
                        numbers = re.findall(r"(?<!\d)\d{7}(?!\d)", response_text)
                        row = {
                            "target_tokens": target_tokens,
                            "prompt_tokens_local": actual_tokens,
                            "prompt_tokens_server": body.get("usage", {}).get(
                                "prompt_tokens"
                            ),
                            "depth": depth,
                            "seed": seed,
                            "key": key,
                            "response": response_text,
                            "retrieved_numbers": numbers,
                            "exact_match": numbers == [key],
                            "finish_reason": choice["finish_reason"],
                            "completion_tokens": body.get("usage", {}).get(
                                "completion_tokens"
                            ),
                        }
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        print(
                            f"context={target_tokens} depth={depth:.2f} "
                            f"seed={seed} prompt={actual_tokens} "
                            f"exact={row['exact_match']} response={response_text!r}",
                            flush=True,
                        )
                        case_index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--haystack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chat-id", default="1")
    parser.add_argument("--url", default="http://127.0.0.1:8005/v1/chat/completions")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[10000, 50000, 100000, 190000],
    )
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
