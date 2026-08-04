# SPDX-License-Identifier: Apache-2.0
"""
Generate (num_documents - 1) chat requests: each request permutes all documents
so no document appears in the same ordinal position as in the baseline order
(identity). Uses the same message shape as two-request-demo.py.

Inputs (CLI):
  --num-documents: how many distinct documents
  --document-length: approximate length of each document
    (same construction as multi_doc_qa)
  --output-len: max_tokens for generation
"""

# Future
from __future__ import annotations

# Standard
import argparse
import itertools
import os
import random
import sys
import time

# Third Party
from openai import OpenAI
from transformers import AutoTokenizer

SERVICE_PORT = os.environ.get("SERVICE_PORT", "10001")


def ordinal_word(i: int) -> str:
    """0 -> 'first', 1 -> 'second', ..."""
    names = (
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
    )
    if i < len(names):
        return names[i]
    return f"{i + 1}th"


def all_derangements(n: int) -> list[tuple[int, ...]]:
    """Permutations p of range(n) with p[i] != i for all i (only for small n)."""
    out: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(n)):
        if all(perm[i] != i for i in range(n)):
            out.append(perm)
    return out


def random_derangement(n: int, rng: random.Random) -> tuple[int, ...]:
    """Uniform random derangement via shuffle-and-reject (small n only)."""
    for _ in range(500_000):
        p = list(range(n))
        rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return tuple(p)
    raise RuntimeError(f"failed to sample a derangement for n={n}")


def pick_derangements(n: int, count: int, rng: random.Random) -> list[tuple[int, ...]]:
    """Return `count` distinct derangements; raises if mathematically impossible."""
    if count == 0:
        return []
    # For modest n, enumerate (correct feasibility check + reproducible shuffle).
    if n <= 8:
        all_d = all_derangements(n)
        if len(all_d) < count:
            raise ValueError(
                f"need {count} distinct derangements for n={n}, "
                f"but only {len(all_d)} exist"
            )
        rng.shuffle(all_d)
        return all_d[:count]

    # Large n: sample random derangements (full enumeration is intractable).
    seen: set[tuple[int, ...]] = set()
    for _ in range(count * 200_000):
        if len(seen) >= count:
            break
        seen.add(random_derangement(n, rng))
    if len(seen) < count:
        raise RuntimeError(
            f"could not collect {count} distinct derangements for n={n} "
            f"(got {len(seen)}); try a different --random-seed"
        )
    return list(seen)[:count]


def build_documents(num_documents: int, document_length: int) -> list[str]:
    """Synthetic docs aligned with multi_doc_qa.py."""
    return [
        str(i) + " " + " ".join(["hi"] * document_length) for i in range(num_documents)
    ]


def build_messages(
    documents: list[str],
    perm: tuple[int, ...],
) -> list[dict]:
    """
    Same structure as two-request-demo.py: system, then for each slot
    (user label, user body), then final summarize user message.
    `perm[k]` is which document index appears at ordinal position k.
    """
    n = len(documents)
    if len(perm) != n:
        raise ValueError("perm length must match number of documents")

    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    for slot in range(n):
        messages.append(
            {
                "role": "user",
                "content": f"Here is the {ordinal_word(slot)} document",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": documents[perm[slot]],
            }
        )

    if n == 1:
        summary = "Please summarize the above document."
    elif n == 2:
        summary = "Please summarize the above two documents"
    else:
        summary = f"Please summarize the above {n} documents"
    messages.append({"role": "user", "content": summary})
    return messages


def parse_chunk_output(choice) -> str | None:
    if not hasattr(choice, "delta"):
        return choice.text

    choice_delta = choice.delta

    if choice_delta is None:
        return None

    fields_to_scan = [
        "content",
        "function_call",
        "refusal",
        "role",
        "tool_calls",
        "reasoning",
    ]
    for field in fields_to_scan:
        if hasattr(choice_delta, field) and getattr(choice_delta, field) is not None:
            return getattr(choice_delta, field)
    return None


def query_and_measure_ttft(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> float:
    start = time.perf_counter()
    ttft = None

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0,
        stream=True,
        max_tokens=max_tokens,
    )

    for chunk in chat_completion:
        chunk_message = parse_chunk_output(chunk.choices[0])
        if chunk_message is not None:
            if ttft is None:
                ttft = time.perf_counter()
            print(chunk_message, end="", flush=True)

    print("\n")
    return ttft - start if ttft is not None else 0.0


def build_warmup_messages() -> list:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi how are you"},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-doc QA with (n-1) deranged shuffles; "
            "matches two-request-demo request shape."
        ),
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        required=True,
        help="Number of documents; emits (num_documents - 1) requests.",
    )
    parser.add_argument(
        "--document-length",
        type=int,
        required=True,
        help="Length field for each synthetic document (see multi_doc_qa).",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        required=True,
        help="max_tokens for each completion.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(SERVICE_PORT),
        help="vLLM/OpenAI-compatible server port (default: SERVICE_PORT or 10001).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for choosing which derangements when multiple exist.",
    )
    args = parser.parse_args()

    n = args.num_documents
    if n < 1:
        print("num-documents must be >= 1", file=sys.stderr)
        sys.exit(2)

    num_requests = n - 1
    if num_requests == 0:
        print("num_documents is 1: nothing to send (n-1 == 0). Exiting.")
        return

    documents = build_documents(n, args.document_length)
    rng = random.Random(args.random_seed)

    try:
        perms = pick_derangements(n, num_requests, rng)
    except (ValueError, RuntimeError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key="dummy-key",
        base_url=f"http://localhost:{args.port}/v1",
    )

    models = client.models.list()
    model = models.data[0].id
    AutoTokenizer.from_pretrained(model)  # validate model id like two-request-demo

    print("Warming up server")
    warmup_messages = build_warmup_messages()
    query_and_measure_ttft(
        client, model, warmup_messages, max_tokens=min(200, args.output_len)
    )
    print()
    print("-------------------------------")

    for i, perm in enumerate(perms):
        print(f"\n--- Request {i + 1}/{num_requests} derangement {perm} ---\n")
        messages = build_messages(documents, perm)
        ttft = query_and_measure_ttft(
            client, model, messages, max_tokens=args.output_len
        )
        print(f"\nTTFT: {ttft:.3f} seconds\n")


if __name__ == "__main__":
    main()
