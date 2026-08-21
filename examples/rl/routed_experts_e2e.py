# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end example for routed experts capture with hybrid models.

Validates that:
1. routed_experts is returned in CompletionOutput for MoE models.
2. Expert IDs are within valid range.
3. Results are deterministic across runs (baseline vs reference).

Usage:
    python examples/rl/routed_experts_e2e.py \
        --model Qwen/Qwen3-30B-A3B \
        --tp 4 \
        --max-model-len 4096 \
        --num-prompts 20 \
        --max-new-tokens 50
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field

import numpy as np

from vllm.engine.arg_utils import AsyncEngineArgs

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B"

TEST_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Explain quantum computing in simple terms:",
    "Write a Python function that sorts a list:",
    "The meaning of life is",
    "In a distant galaxy, there was a",
    "The best way to learn programming is",
    "Once upon a time in a land far away,",
    "The theory of relativity states that",
    "How does photosynthesis work?",
    "Describe the process of machine learning:",
    "What are the benefits of exercise?",
    "The history of artificial intelligence began",
    "Translate the following to French: Hello world",
    "Summarize the plot of Romeo and Juliet:",
    "What is the difference between TCP and UDP?",
    "The water cycle consists of",
    "Explain how a neural network learns:",
    "The periodic table organizes elements by",
    "Write a haiku about the ocean:",
]


@dataclass
class InferenceResult:
    """Result from a single inference run."""

    experts_list: list[np.ndarray] = field(default_factory=list)
    token_ids_list: list[list[int]] = field(default_factory=list)
    num_experts: int = 0


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


async def _run_async_inference(
    engine_args: AsyncEngineArgs,
    prompts: list[str],
    max_new_tokens: int,
) -> InferenceResult:
    """Run inference using AsyncLLM."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(engine_args)

    hf_config = engine.model_config.hf_text_config
    num_experts: int = getattr(hf_config, "num_experts", 0) or getattr(
        hf_config, "num_local_experts", 0
    )
    assert num_experts > 0, "Could not determine num_experts from model config"

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
    )

    async def _generate_one(prompt: str, idx: int):
        request_id = str(uuid.uuid4())
        final_output = None
        async for output in engine.generate(prompt, sampling_params, request_id):
            final_output = output
        assert final_output is not None

        completion = final_output.outputs[0]
        routed = completion.routed_experts
        num_prompt_tokens = len(final_output.prompt_token_ids)
        num_generated_tokens = len(completion.token_ids)
        expected_len = num_prompt_tokens + num_generated_tokens - 1
        assert routed is not None, f"Prompt {idx}: routed_experts is None"
        assert routed.shape[0] == expected_len, (
            f"Prompt {idx}: routed_experts length {routed.shape[0]} != "
            f"prompt ({num_prompt_tokens}) + generated ({num_generated_tokens})"
            f" - 1 = {expected_len}"
        )
        return idx, routed, list(completion.token_ids)

    tasks = [_generate_one(p, i) for i, p in enumerate(prompts)]
    outputs = await asyncio.gather(*tasks)

    # Sort by original index to maintain prompt order
    outputs.sort(key=lambda x: x[0])

    result = InferenceResult(num_experts=num_experts)
    for _, routed, token_ids in outputs:
        result.experts_list.append(routed)
        result.token_ids_list.append(token_ids)

    engine.shutdown()
    return result


def run_inference(
    model: str,
    prompts: list[str],
    max_new_tokens: int = 50,
    tp: int = 1,
    max_model_len: int = 4096,
) -> InferenceResult:
    """Run inference with routed experts capture enabled via AsyncLLM."""
    engine_args = AsyncEngineArgs(
        model=model,
        enable_return_routed_experts=True,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        disable_log_stats=True,
        attention_backend="FLASH_ATTN",
    )

    result = asyncio.run(_run_async_inference(engine_args, prompts, max_new_tokens))

    from vllm.platforms import current_platform

    if current_platform.is_cuda_alike():
        current_platform.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_expert_ids(
    experts_list: list[np.ndarray],
    num_experts: int,
) -> None:
    """Check that all expert IDs are within valid range [0, num_experts)."""
    for i, experts in enumerate(experts_list):
        assert np.all(experts >= 0), (
            f"Prompt {i}: negative expert IDs found, min={experts.min()}"
        )
        assert np.all(experts < num_experts), (
            f"Prompt {i}: expert ID out of range [0, {num_experts}), "
            f"max={experts.max()}"
        )


def validate_shapes(experts_list: list[np.ndarray]) -> None:
    """Check that all routed_experts arrays have at least 2 dimensions."""
    for i, experts in enumerate(experts_list):
        assert experts.ndim >= 2, (
            f"Prompt {i}: expected at least 2D array, got shape {experts.shape}"
        )
        logger.info("Prompt %d: routed_experts shape = %s", i, experts.shape)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def compare_token_ids(
    baseline: list[list[int]],
    reference: list[list[int]],
) -> float:
    """Compare token IDs from two runs. Returns mismatch ratio."""
    assert len(baseline) == len(reference), (
        f"Length mismatch: {len(baseline)} vs {len(reference)}"
    )

    total_tokens = 0
    total_mismatches = 0

    for i, (base, ref) in enumerate(zip(baseline, reference)):
        min_len = min(len(base), len(ref))
        max_len = max(len(base), len(ref))
        matches = 0
        for a, b in zip(base[:min_len], ref[:min_len]):
            if a != b:
                break
            matches += 1

        total_mismatches += max_len - matches
        total_tokens += max_len

        if matches < min_len or len(base) != len(ref):
            print(
                f"  Prompt {i}: token_ids len={len(base)} vs {len(ref)}, "
                f"mismatches={max_len - matches}/{max_len}"
            )

    if total_tokens == 0:
        raise ValueError("No tokens to compare")

    mismatch_ratio = total_mismatches / total_tokens
    print(
        f"Token ID mismatches: {total_mismatches}/{total_tokens} ({mismatch_ratio:.4%})"
    )
    return mismatch_ratio


def compare_routed_experts(
    baseline: list[np.ndarray],
    reference: list[np.ndarray],
    threshold: float = 0.05,
) -> float:
    """Compare two runs of routed experts. Returns mismatch ratio.

    Raises AssertionError if ratio exceeds threshold.
    """
    assert len(baseline) == len(reference), (
        f"Length mismatch: {len(baseline)} vs {len(reference)}"
    )

    total_elements = 0
    total_mismatches = 0

    for i, (base, ref) in enumerate(zip(baseline, reference)):
        min_len = min(len(base), len(ref))
        max_len = max(len(base), len(ref))
        if min_len == 0:
            continue

        base_trimmed = base[:min_len]
        ref_trimmed = ref[:min_len]

        matches = 0
        for a, b in zip(base_trimmed, ref_trimmed):
            if a.sum() != b.sum():
                break
            matches += 1

        total_mismatches += max_len - matches
        total_elements += max_len

        if matches < min_len or len(base) != len(ref):
            print(
                f"  Prompt {i}: routed_experts len={len(base)} vs {len(ref)}, "
                f"mismatches={max_len - matches}/{max_len}"
            )

    if total_elements == 0:
        raise ValueError("No elements to compare")

    mismatch_ratio = total_mismatches / total_elements
    print(
        f"Routed experts mismatches: {total_mismatches}/{total_elements} "
        f"({mismatch_ratio:.4%})"
    )

    assert mismatch_ratio < threshold, (
        f"Too many mismatches: {total_mismatches}/{total_elements} "
        f"({mismatch_ratio:.4%}) exceeds threshold {threshold:.4%}"
    )

    return mismatch_ratio


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

    parser = argparse.ArgumentParser(
        description="Test routed experts capture for MoE models"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Run twice and compare results for determinism check",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Maximum allowed mismatch ratio for determinism check",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    prompts = TEST_PROMPTS[: args.num_prompts]

    print(f"Model: {args.model}")
    print(f"TP: {args.tp}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print()

    print("=== Run 1 (baseline) ===")
    baseline = run_inference(
        model=args.model,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        tp=args.tp,
        max_model_len=args.max_model_len,
    )
    print(f"num_experts (from model config): {baseline.num_experts}")

    print("\n=== Validation ===")
    validate_shapes(baseline.experts_list)
    validate_expert_ids(baseline.experts_list, num_experts=baseline.num_experts)
    print(f"All {len(baseline.experts_list)} results passed validation.")

    for i, experts in enumerate(baseline.experts_list):
        print(
            f"  Prompt {i}: shape={experts.shape}, "
            f"min={experts.min()}, max={experts.max()}"
        )

    if args.deterministic:
        print("\n=== Run 2 (reference) ===")
        reference = run_inference(
            model=args.model,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            tp=args.tp,
            max_model_len=args.max_model_len,
        )

        print("\n=== Determinism Check ===")
        validate_expert_ids(reference.experts_list, num_experts=baseline.num_experts)

        print("\n--- Token IDs ---")
        token_mismatch = compare_token_ids(
            baseline.token_ids_list, reference.token_ids_list
        )

        print("\n--- Routed Experts ---")
        expert_mismatch = compare_routed_experts(
            baseline.experts_list,
            reference.experts_list,
            threshold=args.threshold,
        )

        print(
            f"\nDeterminism check passed. "
            f"Token mismatch: {token_mismatch:.4%}, "
            f"Expert mismatch: {expert_mismatch:.4%}"
        )

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
