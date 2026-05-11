#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Perplexity calculation script using vLLM's score mode.

Implements sliding window perplexity calculation compatible with EXL3's
approach: concatenate text, tokenize, then evaluate fixed-size windows
with a configurable stride. Uses vLLM's score mode for efficient
GPU-side logprob extraction.

Usage:
    python examples/score_mode_perplexity.py \
        --model /path/to/model \
        --dataset wikitext \
        --dataset-config wikitext-2-raw-v1 \
        --num-samples 100 \
        --context-length 2048 \
        --stride 512
"""

import argparse
import logging
import math
import time
from typing import Any

from datasets import load_dataset

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

logger = logging.getLogger(__name__)


def _extract_logprobs_from_window(
    output_prompt_logprobs: list,
    window_tokens: list[int],
) -> tuple[float, int]:
    """Extract log-probabilities for each target position in a window.

    Returns (logprob_sum, count) for the window.
    """
    if not output_prompt_logprobs:
        raise ValueError("prompt_logprobs is None or empty")

    if len(output_prompt_logprobs) != len(window_tokens):
        raise ValueError(
            f"prompt_logprobs length ({len(output_prompt_logprobs)}) "
            f"does not match window length ({len(window_tokens)})"
        )

    window_sum = 0.0
    window_count = 0
    for i in range(1, len(output_prompt_logprobs)):
        logprobs_dict = output_prompt_logprobs[i]
        if logprobs_dict:
            actual_token = window_tokens[i]
            if actual_token in logprobs_dict:
                window_sum += logprobs_dict[actual_token].logprob
                window_count += 1

    return window_sum, window_count


def calculate_perplexity(
    llm: LLM,
    texts: list[str],
    context_length: int,
    stride: int,
    num_samples: int | None = None,
) -> tuple[float, int]:
    """
    Calculate perplexity using sliding window approach.

    Concatenates all texts, tokenizes as one sequence, then evaluates
    fixed-size windows with the given stride. Compatible with EXL3's
    evaluation methodology.

    Args:
        llm: Initialized vLLM LLM instance
        texts: List of text samples to evaluate
        context_length: Maximum context length for each window
        stride: Stride between windows (overlap = context_length - stride)
        num_samples: Maximum number of samples to process (None = all)

    Returns:
        Tuple of (perplexity, total_tokens_evaluated)
    """
    logprob_sum = 0.0
    logprob_count = 0

    samples_to_process = texts[:num_samples] if num_samples else texts
    concatenated_text = "\n\n".join(samples_to_process)

    tokens = llm.llm_engine.tokenizer.encode(
        concatenated_text, add_special_tokens=False
    )

    if len(tokens) < 2:
        raise ValueError("Not enough tokens after concatenation")

    # Limit to ~100 windows worth of tokens
    max_tokens_for_eval = context_length + 99 * stride
    if len(tokens) > max_tokens_for_eval:
        tokens = tokens[:max_tokens_for_eval]

    num_tokens = len(tokens)
    logger.debug("Total tokens for evaluation: %d", num_tokens)
    windows_processed = 0

    sampling_params = SamplingParams(
        prompt_logprobs=1,
        max_tokens=1,
        score_mode=True,
    )

    if num_tokens < context_length:
        if num_tokens >= 2:
            window_tokens = tokens
            target_token_ids = window_tokens[1:]

            prompt: TokensPrompt = {
                "prompt_token_ids": window_tokens,
                "target_token_ids": target_token_ids,
            }

            outputs = llm.generate([prompt], sampling_params=sampling_params)
            window_sum, window_count = _extract_logprobs_from_window(
                outputs[0].prompt_logprobs, window_tokens
            )
            logprob_sum += window_sum
            logprob_count += window_count
    else:
        for start_idx in range(0, num_tokens - context_length + stride, stride):
            end_idx = start_idx + context_length
            if end_idx > num_tokens:
                break

            window_tokens = tokens[start_idx:end_idx]
            if len(window_tokens) < 2:
                continue

            windows_processed += 1
            target_token_ids = window_tokens[1:]

            prompt: TokensPrompt = {
                "prompt_token_ids": window_tokens,
                "target_token_ids": target_token_ids,
            }

            outputs = llm.generate([prompt], sampling_params=sampling_params)
            window_sum, window_count = _extract_logprobs_from_window(
                outputs[0].prompt_logprobs, window_tokens
            )
            logprob_sum += window_sum
            logprob_count += window_count

            if windows_processed % 100 == 0:
                print(
                    f"Processed {windows_processed} windows, "
                    f"{logprob_count} tokens evaluated"
                )

    if logprob_count == 0:
        raise ValueError("No valid tokens found for perplexity calculation")

    logger.debug(
        "Evaluation complete: %d windows, %d tokens",
        windows_processed,
        logprob_count,
    )

    mean_log_prob = logprob_sum / logprob_count
    perplexity = math.exp(-mean_log_prob)
    return perplexity, logprob_count


def load_dataset_texts(
    dataset_name: str,
    dataset_config: str | None = None,
    split: str | None = None,
) -> list[str]:
    """Load and extract text from a HuggingFace dataset.

    Supports datasets with "text" fields, chat-format "messages" fields,
    or falls back to the first string field found.
    """
    if split is None:
        for candidate_split in ["test", "train", "validation"]:
            try:
                if dataset_config:
                    dataset = load_dataset(
                        dataset_name, dataset_config, split=candidate_split
                    )
                else:
                    dataset = load_dataset(dataset_name, split=candidate_split)
                split = candidate_split
                break
            except Exception:
                continue

        if split is None:
            raise ValueError(
                f"Could not load dataset {dataset_name} with any split "
                "(test/train/validation)"
            )

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    texts = []
    for example in dataset:
        if "text" in example:
            text = example["text"]
            if text and text.strip():
                texts.append(text)
        elif "messages" in example:
            messages = example["messages"]
            if isinstance(messages, list):
                text = "\n".join(
                    msg.get("content", "") for msg in messages if isinstance(msg, dict)
                )
                if text and text.strip():
                    texts.append(text)
        else:
            for key, value in example.items():
                if isinstance(value, str) and value.strip():
                    texts.append(value)
                    break

    if not texts:
        raise ValueError(f"No valid text found in dataset {dataset_name}")

    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Calculate perplexity using vLLM's score mode"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (e.g., 'awq', 'gptq', 'compressed-tensors')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'wikitext')",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset configuration (e.g., 'wikitext-2-raw-v1')",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help="Context length for each window (default: 2048)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride between windows (default: 512)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.30,
        help="GPU memory utilization (default: 0.30)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    print(f"Loading dataset: {args.dataset}")
    texts = load_dataset_texts(args.dataset, args.dataset_config)
    print(f"Loaded {len(texts)} text samples")

    llm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enable_prefix_caching": False,
        "max_model_len": args.context_length * 2,
    }

    if args.quantization:
        llm_kwargs["quantization"] = args.quantization

    print(f"Initializing LLM with model: {args.model}")
    llm = LLM(model=args.model, **llm_kwargs)

    print("\nCalculating perplexity...")
    print(f"  Context length: {args.context_length}")
    print(f"  Stride: {args.stride}")
    print(f"  Samples: {args.num_samples or len(texts)}")

    start_time = time.time()
    perplexity, total_tokens = calculate_perplexity(
        llm,
        texts,
        args.context_length,
        args.stride,
        args.num_samples,
    )
    elapsed_time = time.time() - start_time

    print("\nResults:")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"  Tokens/second: {total_tokens / elapsed_time:.2f}")


if __name__ == "__main__":
    main()
