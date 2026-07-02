#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Needle-in-a-Haystack (NIAH) evaluation for long-context KV cache quality.

Inserts a fact ("needle") at a specific depth within a long filler text
("haystack"), then asks the model to recall it. Measures retrieval accuracy
across context lengths and needle depths.
"""

import asyncio
import random
import time

import aiohttp

HAYSTACK_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Filler text for context padding. Here is some more text. "
)

NEEDLE_TEMPLATE = "The special magic number is {number}."

QUERY = "What is the special magic number mentioned in the text above?"


def build_niah_prompt(
    context_tokens: int,
    depth: float,
    tokenizer,
    needle_number: int,
) -> tuple[str, str]:
    """Build a NIAH prompt with needle at given depth.

    Returns (prompt, expected_answer).
    """
    needle = NEEDLE_TEMPLATE.format(number=needle_number)
    query_tokens = len(tokenizer.encode(QUERY))
    needle_tokens = len(tokenizer.encode(needle))
    available = context_tokens - query_tokens - needle_tokens - 20

    haystack_sentence_tokens = len(tokenizer.encode(HAYSTACK_SENTENCE))
    num_sentences = max(1, available // haystack_sentence_tokens)
    insert_pos = max(0, int(num_sentences * depth))

    sentences = [HAYSTACK_SENTENCE] * num_sentences
    sentences.insert(insert_pos, needle + " ")
    haystack = "".join(sentences)

    prompt = f"{haystack}\n\n{QUERY}"
    return prompt, str(needle_number)


async def _send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int = 32,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    async with session.post(f"{url}/v1/completions", json=payload) as resp:
        data = await resp.json()
        return data["choices"][0]["text"].strip()


async def evaluate_niah(
    host: str,
    port: int,
    model: str,
    context_lengths: list[int],
    depths: list[float],
    tokenizer,
    num_trials: int = 1,
) -> dict:
    """Run NIAH evaluation grid.

    Returns dict with results per (context_length, depth).
    """
    url = f"{host}:{port}"
    results = {}
    total = 0
    correct = 0

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        for ctx_len in context_lengths:
            for depth in depths:
                hits = 0
                for trial in range(num_trials):
                    number = random.randint(100000, 999999)
                    prompt, expected = build_niah_prompt(
                        ctx_len, depth, tokenizer, number
                    )
                    output = await _send_request(
                        session, url, model, prompt
                    )
                    if expected in output:
                        hits += 1
                        correct += 1
                    total += 1

                accuracy = hits / num_trials
                results[(ctx_len, depth)] = accuracy
                tag = "✓" if accuracy >= 0.5 else "✗"
                print(
                    f"  {tag} ctx={ctx_len:>7}, depth={depth:.1f}: "
                    f"{hits}/{num_trials} ({accuracy:.0%})"
                )

    overall = correct / total if total > 0 else 0
    print(f"\nOverall: {correct}/{total} ({overall:.1%})")
    return {
        "grid": results,
        "overall_accuracy": overall,
        "total": total,
        "correct": correct,
    }
