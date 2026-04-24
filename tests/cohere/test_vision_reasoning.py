#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Vision model test for Command-A Vision model.
Tests that the model can correctly identify images of a duck and a lion.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

from vllm import LLM, SamplingParams

# Get the directory where this test file is located
TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"

# Local image paths
IMAGE_PATHS = [
    FIXTURES_DIR / "duck.jpg",  # duck
    FIXTURES_DIR / "lion.jpg",  # lion
]

QUESTION = "What is the content of each image?"


def run_vision_test(model_path: str, tensor_parallel_size: int = 1):
    """
    Run vision test similar to load_command_a_vision from examples.

    Args:
        model_path: Path to the model checkpoint
        tensor_parallel_size: Number of GPUs for tensor parallelism
    """
    print(f"Loading model from: {model_path}")
    print(f"Using tensor_parallel_size: {tensor_parallel_size}")

    # Load images from local files
    print("Loading images from local files...")
    images = [Image.open(path) for path in IMAGE_PATHS]

    # Set up engine args similar to load_command_a_vision
    llm = LLM(
        model=model_path,
        cudagraph_capture_sizes=[64],
        max_model_len=64000,
        tensor_parallel_size=tensor_parallel_size,
        limit_mm_per_prompt={"image": len(images)},
    )

    # Create message with image placeholders
    placeholders = [{"type": "image"} for _ in images]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": QUESTION},
            ],
        }
    ]

    # Load processor and apply chat template
    print("Loading processor and preparing prompt...")
    processor = AutoProcessor.from_pretrained(model_path)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Run inference
    print("Running inference...")
    sampling_params = SamplingParams(
        top_p=0.75, temperature=0.0, max_tokens=256, thinking_token_budget=31768
    )

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        },
        sampling_params=sampling_params,
    )

    # Get generated text
    generated_text = outputs[0].outputs[0].text
    print("-" * 50)
    print("Generated output:")
    print(generated_text)
    print("-" * 50)

    # Check if duck and lion are in the output
    generated_lower = generated_text.lower()
    has_duck = "duck" in generated_lower
    has_lion = "lion" in generated_lower
    has_start_thinking = "<|start_thinking|>" in generated_lower
    has_end_thinking = "<|end_thinking|>" in generated_lower

    print("\nTest results:")
    print(f"  Contains 'duck': {has_duck}")
    print(f"  Contains 'lion': {has_lion}")
    print(f"  Contains '<|start_thinking|>': {has_start_thinking}")
    print(f"  Contains '<|end_thinking|>': {has_end_thinking}")

    if has_duck and has_lion:
        if has_start_thinking and has_end_thinking:
            print("\n✓ Vision test PASSED: All required tokens found in output")
        elif not has_start_thinking or not has_end_thinking:
            print(
                "\n Vision test Passes Partially: "
                "Missing '<|start_thinking|>' or '<|end_thinking|>' token"
            )
        return 0
    else:
        print("\n✗ Vision test FAILED: Missing words in output")
        if not has_duck:
            print("  - 'duck' not found")
        if not has_lion:
            print("  - 'lion' not found")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Test vision model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism (default: 2)",
    )
    args = parser.parse_args()

    return run_vision_test(args.model, args.tensor_parallel_size)


if __name__ == "__main__":
    sys.exit(main())
