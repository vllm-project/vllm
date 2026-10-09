# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU completion-logprob formatting/JSON and client decode microbenchmark.

Excludes inference, engine processing, HTTP, and SSE framing. Inputs are
prebuilt in the corresponding engine storage formats; sampled tokens are
outside the top-k. Reports medians and min/max over interleaved repetitions.
"""

import argparse
import json
import statistics
import time

import numpy as np
import pybase64

from vllm.entrypoints.openai.completion.packed_logprobs import (
    create_packed_completion_logprobs,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.logprobs import FlatLogprobs, append_logprobs_for_next_position


def benchmark(tokens: int, k: int, repeats: int) -> dict:
    flat = FlatLogprobs()
    ids = list(range(k + 1))
    values = [-20.0] + np.linspace(-0.5, -10.0, k, dtype=np.float32).tolist()
    for _ in range(tokens):
        append_logprobs_for_next_position(flat, ids, values, [None] * (k + 1), k + 1, k)
    standard = list(flat)
    sampled_ids = [0] * tokens
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    timings: dict[str, list[float]] = {
        name: []
        for name in (
            "standard_encode",
            "packed_encode",
            "standard_decode",
            "packed_decode",
        )
    }
    sizes = {}
    for repetition in range(repeats + 1):
        # Alternate order; the first pair warms both paths.
        for packed in (False, True) if repetition % 2 == 0 else (True, False):
            label = "packed" if packed else "standard"
            start = time.perf_counter()
            if packed:
                result = create_packed_completion_logprobs(sampled_ids, flat, k)
            else:
                result = serving._create_completion_logprobs(
                    sampled_ids, standard, k, None, return_as_token_id=True
                )
            body = result.model_dump_json()
            encoded = time.perf_counter()
            obj = json.loads(body)
            if packed:
                block = obj["packed_top_logprobs"]
                candidate_ids = np.frombuffer(
                    pybase64.b64decode(block["token_ids_b64"]), "<i4"
                ).reshape(tokens, k)
                candidate_values = np.frombuffer(
                    pybase64.b64decode(block["logprobs_b64"]), "<f4"
                ).reshape(tokens, k)
            else:
                # A numeric consumer must turn JSON token keys into arrays.
                candidate_ids = np.array(
                    [
                        [
                            int(key.removeprefix("token_id:"))
                            for key in row
                            if key != "token_id:0"
                        ]
                        for row in obj["top_logprobs"]
                    ],
                    dtype="<i4",
                )
                candidate_values = np.array(
                    [
                        [value for key, value in row.items() if key != "token_id:0"]
                        for row in obj["top_logprobs"]
                    ],
                    dtype="<f4",
                )
            decoded = time.perf_counter()
            np.testing.assert_array_equal(candidate_ids, np.tile(ids[1:], (tokens, 1)))
            np.testing.assert_array_equal(
                candidate_values, np.tile(values[1:], (tokens, 1))
            )
            assert obj["token_logprobs"] == [-20.0] * tokens
            sizes[label] = len(body.encode())
            if repetition:
                timings[f"{label}_encode"].append((encoded - start) * 1000)
                timings[f"{label}_decode"].append((decoded - encoded) * 1000)
    return {
        "tokens": tokens,
        "k": k,
        "repeats": repeats,
        "bytes": sizes,
        "milliseconds": {
            name: {
                "median": statistics.median(samples),
                "min": min(samples),
                "max": max(samples),
            }
            for name, samples in timings.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1000)
    parser.add_argument("--top-k", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if min(args.tokens, args.repeats, *args.top_k) < 1:
        parser.error("tokens, top-k and repeats must be positive")
    for k in args.top_k:
        print(json.dumps(benchmark(args.tokens, k, args.repeats)))
