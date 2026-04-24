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
from test_utils_engine_args import get_engine_kwargs_with_overrides
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


def run_vision_test(
    model_path: str,
    tensor_parallel_size: int = 1,
    engine_args: str | None = None,
):
    """
    Run vision test similar to load_command_a_vision from examples.

    Args:
        model_path: Path to the model checkpoint
        tensor_parallel_size: Number of GPUs for tensor parallelism
        engine_args: CLI-style engine args (overrides
            VLLM_HARDWARE_PROFILE_ARGS env var)
    """
    print(f"Loading model from: {model_path}")
    print(f"Using tensor_parallel_size: {tensor_parallel_size}")

    # Load images from local files
    print("Loading images from local files...")
    images = [Image.open(path) for path in IMAGE_PATHS]

    # Get effective engine kwargs with hardware profile args + test-specific overrides
    effective_kwargs = get_engine_kwargs_with_overrides(
        test_kwargs={
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "limit_mm_per_prompt": {"image": len(images)},
        },
        engine_args_override=engine_args,
    )

    # Create LLM instance with merged kwargs
    llm = LLM(**effective_kwargs)

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
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

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

    print("\nTest results:")
    print(f"  Contains 'duck': {has_duck}")
    print(f"  Contains 'lion': {has_lion}")

    if has_duck and has_lion:
        print("\n✓ Vision test PASSED: Both 'duck' and 'lion' found in output")
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
    parser.add_argument(
        "--engine-args",
        type=str,
        default=None,
        help=(
            "CLI-style engine args to pass to LLM (e.g., '--max-model-len 32768 "
            "--enable-chunked-prefill'). "
            "If not provided, uses VLLM_HARDWARE_PROFILE_ARGS "
            "environment variable."
        ),
    )
    args = parser.parse_args()

    return run_vision_test(args.model, args.tensor_parallel_size, args.engine_args)


if __name__ == "__main__":
    sys.exit(main())
