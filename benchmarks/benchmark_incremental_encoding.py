# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the exact incremental prompt-encoding cache
(``vllm/tokenizers/incremental_encode.py``) against full re-encoding.

Simulates multi-turn conversations: for each configured history size, a
synthetic conversation is grown turn by turn and the per-turn encode
latency is measured for both the cold path (full re-encode, what vLLM does
today) and the incremental path. Only ``transformers`` is required.

Example usage:
    python benchmark_incremental_encoding.py \
        --tokenizer Qwen/Qwen2.5-0.5B \
        --history-tokens 5000 32000 128000 512000 \
        --num-turns 8
"""

import argparse
import random
import statistics
import time

from transformers import AutoTokenizer

try:
    from vllm.tokenizers.incremental_encode import IncrementalEncodeCache
except ImportError:  # allow running without a full vLLM installation
    import importlib.util
    from pathlib import Path

    _MODULE_PATH = (
        Path(__file__).resolve().parents[1]
        / "vllm"
        / "tokenizers"
        / "incremental_encode.py"
    )
    _spec = importlib.util.spec_from_file_location("incremental_encode", _MODULE_PATH)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    IncrementalEncodeCache = _module.IncrementalEncodeCache

FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "def process(batch):\n    return [x * 2 for x in batch]\n"
    "多轮对话的前缀在每一轮都保持不变。 "
    "Latency is dominated by re-tokenizing the unchanged prefix. "
)


def build_history(tokenizer, target_tokens: int, rng: random.Random) -> str:
    """Grow a synthetic conversation history to ~target_tokens tokens."""
    chunk = (
        f"<|im_start|>user\nRequest {{i}}: {FILLER}{{salt}}<|im_end|>\n"
        f"<|im_start|>assistant\nResponse {{i}}: {FILLER}<|im_end|>\n"
    )
    chunk_tokens = len(tokenizer(chunk, add_special_tokens=False)["input_ids"])
    parts = [
        chunk.format(i=i, salt=f"{rng.getrandbits(64):x}")
        for i in range(max(1, target_tokens // chunk_tokens))
    ]
    return "".join(parts)


def new_turn(i: int, rng: random.Random) -> str:
    return (
        f"<|im_start|>user\nTurn {i}: {FILLER} {rng.getrandbits(64):x}"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )


def percentile(values: list[float], p: float) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[
        min(98, max(0, round(p) - 1))
    ]


def bench_history_size(tokenizer, target_tokens: int, num_turns: int, seed: int):
    rng = random.Random(seed)
    history = build_history(tokenizer, target_tokens, rng)

    cache = IncrementalEncodeCache()
    cold_times: list[float] = []
    incr_times: list[float] = []

    # Seed the cache with the initial history (a miss, like the first
    # request of a conversation after server start).
    cache.encode(tokenizer, history)

    text = history
    for i in range(num_turns):
        text += new_turn(i, rng)

        start = time.perf_counter()
        expected = tokenizer(text)["input_ids"]
        cold_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        actual = cache.encode(tokenizer, text)
        incr_times.append(time.perf_counter() - start)

        assert actual == expected, "incremental encode is not token-exact!"

    return {
        "prompt_tokens": len(expected),
        "cold_p50_ms": statistics.median(cold_times) * 1e3,
        "cold_p99_ms": percentile(cold_times, 99) * 1e3,
        "incr_p50_ms": statistics.median(incr_times) * 1e3,
        "incr_p99_ms": percentile(incr_times, 99) * 1e3,
        "hits": cache.stats.hits,
        "misses": cache.stats.misses,
        "fallbacks": cache.stats.fallbacks,
    }


def main(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer: {args.tokenizer}, turns per size: {args.num_turns}")
    header = (
        f"{'history':>9} {'prompt_tok':>10} {'cold p50':>10} {'cold p99':>10} "
        f"{'incr p50':>10} {'incr p99':>10} {'speedup':>8} {'hit/miss/fb':>12}"
    )
    print(header)
    print("-" * len(header))
    for target in args.history_tokens:
        result = bench_history_size(tokenizer, target, args.num_turns, args.seed)
        speedup = result["cold_p50_ms"] / max(result["incr_p50_ms"], 1e-9)
        print(
            f"{target:>9} {result['prompt_tokens']:>10} "
            f"{result['cold_p50_ms']:>8.2f}ms {result['cold_p99_ms']:>8.2f}ms "
            f"{result['incr_p50_ms']:>8.2f}ms {result['incr_p99_ms']:>8.2f}ms "
            f"{speedup:>7.1f}x "
            f"{result['hits']:>4}/{result['misses']}/{result['fallbacks']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark incremental prompt encoding vs full re-encode."
    )
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument(
        "--history-tokens",
        type=int,
        nargs="+",
        default=[5000, 32000, 128000, 512000],
        help="Conversation history sizes (in tokens) to benchmark.",
    )
    parser.add_argument(
        "--num-turns",
        type=int,
        default=8,
        help="Number of appended turns measured per history size.",
    )
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
