# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test data generation utilities for hybrid attention benchmarks.

This module provides functions to generate consistent test prompts
for benchmarking hybrid attention configurations.

Usage:
    python benchmarks/hybrid_benchmark_data.py \
        --tokenizer meta-llama/Llama-3.2-1B \
        --input-lengths 512,1024,2048,4096 \
        --num-prompts 100 \
        --output-file benchmark_prompts.json
"""

import argparse
import json
import os
import random
from typing import Any

import numpy as np


def generate_random_prompts(
    vocab_size: int,
    input_lengths: list[int],
    num_prompts_per_length: int,
    output_len: int = 128,
    seed: int = 42,
    min_token_id: int = 100,
    max_token_id_offset: int = 100,
) -> list[dict]:
    """Generate random token sequences for benchmarking.

    Args:
        vocab_size: Size of the tokenizer vocabulary.
        input_lengths: List of target input lengths to generate.
        num_prompts_per_length: Number of prompts to generate per length.
        output_len: Target output length for generation.
        seed: Random seed for reproducibility.
        min_token_id: Minimum token ID to use (avoids special tokens).
        max_token_id_offset: Offset from vocab_size for max token ID.

    Returns:
        List of prompt dictionaries with token IDs and metadata.
    """
    random.seed(seed)
    np.random.seed(seed)

    max_token_id = vocab_size - max_token_id_offset
    prompts = []

    for target_length in input_lengths:
        for i in range(num_prompts_per_length):
            # Generate random token IDs
            token_ids = [
                random.randint(min_token_id, max_token_id)
                for _ in range(target_length)
            ]
            prompts.append(
                {
                    "prompt_token_ids": token_ids,
                    "target_length": target_length,
                    "output_len": output_len,
                    "prompt_idx": i,
                }
            )

    return prompts


def generate_repetitive_prompts(
    vocab_size: int,
    input_lengths: list[int],
    num_prompts_per_length: int,
    output_len: int = 128,
    seed: int = 42,
    pattern_length: int = 32,
) -> list[dict]:
    """Generate repetitive token sequences for prefix caching tests.

    These prompts have repeating patterns which can test prefix
    caching effectiveness.

    Args:
        vocab_size: Size of the tokenizer vocabulary.
        input_lengths: List of target input lengths to generate.
        num_prompts_per_length: Number of prompts to generate per length.
        output_len: Target output length for generation.
        seed: Random seed for reproducibility.
        pattern_length: Length of the repeating pattern.

    Returns:
        List of prompt dictionaries with token IDs and metadata.
    """
    random.seed(seed)
    np.random.seed(seed)

    prompts = []
    min_token_id = 100
    max_token_id = vocab_size - 100

    for target_length in input_lengths:
        for i in range(num_prompts_per_length):
            # Generate a base pattern
            pattern = [
                random.randint(min_token_id, max_token_id)
                for _ in range(pattern_length)
            ]
            # Repeat the pattern to fill the target length
            num_repeats = (target_length + pattern_length - 1) // pattern_length
            token_ids = (pattern * num_repeats)[:target_length]

            prompts.append(
                {
                    "prompt_token_ids": token_ids,
                    "target_length": target_length,
                    "output_len": output_len,
                    "prompt_idx": i,
                    "pattern_type": "repetitive",
                    "pattern_length": pattern_length,
                }
            )

    return prompts


def generate_shared_prefix_prompts(
    vocab_size: int,
    input_lengths: list[int],
    num_prompts_per_length: int,
    output_len: int = 128,
    seed: int = 42,
    shared_prefix_ratio: float = 0.5,
) -> list[dict]:
    """Generate prompts with shared prefixes for prefix caching tests.

    Args:
        vocab_size: Size of the tokenizer vocabulary.
        input_lengths: List of target input lengths to generate.
        num_prompts_per_length: Number of prompts to generate per length.
        output_len: Target output length for generation.
        seed: Random seed for reproducibility.
        shared_prefix_ratio: Ratio of tokens that are shared across prompts.

    Returns:
        List of prompt dictionaries with token IDs and metadata.
    """
    random.seed(seed)
    np.random.seed(seed)

    prompts = []
    min_token_id = 100
    max_token_id = vocab_size - 100

    for target_length in input_lengths:
        # Generate a shared prefix for this length group
        prefix_length = int(target_length * shared_prefix_ratio)
        shared_prefix = [
            random.randint(min_token_id, max_token_id) for _ in range(prefix_length)
        ]

        for i in range(num_prompts_per_length):
            # Generate unique suffix
            suffix_length = target_length - prefix_length
            suffix = [
                random.randint(min_token_id, max_token_id) for _ in range(suffix_length)
            ]
            token_ids = shared_prefix + suffix

            prompts.append(
                {
                    "prompt_token_ids": token_ids,
                    "target_length": target_length,
                    "output_len": output_len,
                    "prompt_idx": i,
                    "pattern_type": "shared_prefix",
                    "prefix_length": prefix_length,
                }
            )

    return prompts


def load_prompts_from_file(file_path: str) -> list[dict]:
    """Load prompts from a JSON file."""
    with open(file_path) as f:
        return json.load(f)


def save_prompts_to_file(prompts: list[dict], file_path: str) -> None:
    """Save prompts to a JSON file."""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Saved {len(prompts)} prompts to: {file_path}")


def get_prompt_statistics(prompts: list[dict]) -> dict[str, Any]:
    """Calculate statistics about the prompt dataset."""
    lengths = [p["target_length"] for p in prompts]
    output_lens = [p["output_len"] for p in prompts]

    return {
        "total_prompts": len(prompts),
        "unique_lengths": sorted(set(lengths)),
        "prompts_per_length": len(prompts) // len(set(lengths)) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "output_len": output_lens[0] if output_lens else 0,
        "total_input_tokens": sum(lengths),
        "total_output_tokens": sum(output_lens),
    }


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the data generation script."""
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Tokenizer to use for vocabulary size",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocabulary size (overrides tokenizer-based detection)",
    )
    parser.add_argument(
        "--input-lengths",
        type=str,
        default="512,1024,2048,4096",
        help="Comma-separated list of input lengths",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts per input length",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Target output length for each prompt",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="random",
        choices=["random", "repetitive", "shared_prefix"],
        help="Pattern type for prompt generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="benchmark_prompts.json",
        help="Output file path for generated prompts",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading tokenizer",
    )


def main(args: argparse.Namespace) -> None:
    """Main entry point for data generation."""
    # Get vocabulary size
    if args.vocab_size is not None:
        vocab_size = args.vocab_size
    else:
        print(f"Loading tokenizer: {args.tokenizer}")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code
        )
        vocab_size = tokenizer.vocab_size
        print(f"Vocabulary size: {vocab_size}")

    # Parse input lengths
    input_lengths = [int(x.strip()) for x in args.input_lengths.split(",")]

    print(f"\nGenerating prompts:")
    print(f"  Input lengths: {input_lengths}")
    print(f"  Prompts per length: {args.num_prompts}")
    print(f"  Output length: {args.output_len}")
    print(f"  Pattern: {args.pattern}")
    print(f"  Seed: {args.seed}")

    # Generate prompts based on pattern type
    if args.pattern == "random":
        prompts = generate_random_prompts(
            vocab_size=vocab_size,
            input_lengths=input_lengths,
            num_prompts_per_length=args.num_prompts,
            output_len=args.output_len,
            seed=args.seed,
        )
    elif args.pattern == "repetitive":
        prompts = generate_repetitive_prompts(
            vocab_size=vocab_size,
            input_lengths=input_lengths,
            num_prompts_per_length=args.num_prompts,
            output_len=args.output_len,
            seed=args.seed,
        )
    elif args.pattern == "shared_prefix":
        prompts = generate_shared_prefix_prompts(
            vocab_size=vocab_size,
            input_lengths=input_lengths,
            num_prompts_per_length=args.num_prompts,
            output_len=args.output_len,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown pattern: {args.pattern}")

    # Save prompts
    save_prompts_to_file(prompts, args.output_file)

    # Print statistics
    stats = get_prompt_statistics(prompts)
    print("\nDataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate test prompts for hybrid attention benchmarks"
    )
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)

