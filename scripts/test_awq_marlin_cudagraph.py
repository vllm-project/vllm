#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproducer script for AWQ-Marlin CUDA Graph issue #32834

This script helps verify the fix for illegal memory access when using
AWQ-Marlin quantization with CUDA graphs.
"""

import argparse
import sys

from vllm import LLM, SamplingParams


def test_awq_marlin_cudagraph(
    model: str = "robertgshaw2/tinyllama-awq-marlin",
    num_iterations: int = 10,
    enforce_eager: bool = False,
):
    """Test AWQ-Marlin with CUDA graphs

    Args:
        model: Model name or path (should be AWQ-quantized)
        num_iterations: Number of generation iterations
        enforce_eager: If True, disable CUDA graphs (safer but slower)
    """
    print(f"\n{'=' * 70}")
    print("Testing AWQ-Marlin Quantization with CUDA Graphs")
    print(f"{'=' * 70}")
    print(f"Model: {model}")
    print(f"Iterations: {num_iterations}")
    print(f"Enforce Eager: {enforce_eager}")
    print(f"{'=' * 70}\n")

    try:
        # Initialize LLM with AWQ-Marlin
        llm = LLM(
            model=model,
            quantization="awq_marlin",
            max_model_len=512,
            gpu_memory_utilization=0.8,
            enforce_eager=enforce_eager,
        )

        print("[OK] Model loaded successfully")

        # Test with multiple iterations to trigger graph capture and replay
        prompts = [
            "Write a short sentence about artificial intelligence.",
            "Explain what CUDA graphs are in one sentence.",
            "What is quantization in machine learning?",
        ] * (num_iterations // 3 + 1)
        prompts = prompts[:num_iterations]

        sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

        print(f"\nRunning {num_iterations} generation iterations...")

        outputs = llm.generate(prompts, sampling_params)

        print(f"[OK] Successfully generated {len(outputs)} outputs")

        # Display first few outputs as sanity check
        print("\nSample outputs:")
        for i, output in enumerate(outputs[:3]):
            text = output.outputs[0].text
            print(f"  [{i + 1}] {text[:80]}...")

        print(f"\n{'=' * 70}")
        print("TEST PASSED: No crashes detected!")
        print(f"{'=' * 70}\n")

        return True

    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"TEST FAILED: {type(e).__name__}")
        print(f"{'=' * 70}")
        print(f"Error: {str(e)}")
        print(
            "\nIf you see 'illegal memory access' errors, this indicates "
            "the AWQ-Marlin CUDA graph safety fix may not be working properly."
        )
        print(f"{'=' * 70}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test AWQ-Marlin quantization with CUDA graphs (Issue #32834)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="robertgshaw2/tinyllama-awq-marlin",
        help="Model name or path (default: robertgshaw2/tinyllama-awq-marlin)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of generation iterations (default: 10)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs (use eager mode)",
    )

    args = parser.parse_args()

    success = test_awq_marlin_cudagraph(
        model=args.model,
        num_iterations=args.iterations,
        enforce_eager=args.enforce_eager,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
