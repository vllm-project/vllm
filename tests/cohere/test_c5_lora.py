#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA serving sanity check for the c5 3a30t (cohere2moe) model.

Loads the model with LoRA enabled, runs the standard c5 sanity prompts both
without a LoRA adapter and with one, and checks that expected keywords appear
in every response.

Required environment variables:
  C5_MODEL_DIR   -- path to the c5 3a30t (fp8) checkpoint directory
  C5_LORA_DIR    -- path to the LoRA adapter checkpoint directory
"""

import argparse
import os
import sys

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .test_utils_c5 import (
    C5_SANITY_EXPECTED,
    C5_SANITY_PROMPTS,
    shutdown_llm,
    validate_model_path,
)

C5_MODEL_DIR_ENV_KEY = "C5_MODEL_DIR"
C5_LORA_DIR_ENV_KEY = "C5_LORA_DIR"


def _build_lora_llm(model_path: str, tensor_parallel_size: int) -> LLM:
    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
    )


def _generate(
    llm: LLM,
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None = None,
):
    return llm.generate(
        C5_SANITY_PROMPTS,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )


def _check_outputs(
    outputs,
    label: str,
) -> int:
    """Return the number of failed cases."""
    errors = 0
    for i, (prompt, output, expected) in enumerate(
        zip(C5_SANITY_PROMPTS, outputs, C5_SANITY_EXPECTED)
    ):
        text = output.outputs[0].text.lower()
        if not any(phrase in text for phrase in expected):
            errors += 1
            print("\n" + "=" * 80)
            print(f"✗ [{label}] Sanity check FAILED (case {i})")
            print("-" * 80)
            print("Prompt:")
            print(prompt)
            print("\nGeneration:")
            print(text.rstrip())
            print("\nExpected at least one of:")
            for p in expected:
                print(f"  • {p}")
            print("=" * 80)
    return errors


def run_c5_lora_sanity_check_test(
    model_path: str,
    lora_path: str,
    tensor_parallel_size: int = 1,
) -> bool:
    """
    Run c5 LoRA serving sanity check test.

    Args:
        model_path: Path to the c5 3a30t model checkpoint directory
        lora_path: Path to the LoRA adapter checkpoint directory
        tensor_parallel_size: Number of GPUs for tensor parallelism

    Returns:
        True if there were any failures, False if all checks passed.
    """
    print(f"Loading model from: {model_path}")
    print(f"Using LoRA adapter from: {lora_path}")
    print(f"Using tensor_parallel_size: {tensor_parallel_size}")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    lora_request = LoRARequest(
        lora_name="c5-adapter",
        lora_int_id=1,
        lora_path=lora_path,
    )

    llm = _build_lora_llm(model_path, tensor_parallel_size)
    errors = 0
    try:
        # Run without LoRA
        print("\n--- Inference WITHOUT LoRA ---")
        outputs_no_lora = _generate(llm, sampling_params, lora_request=None)
        errors += _check_outputs(outputs_no_lora, "no-lora")

        # Run with LoRA
        print("\n--- Inference WITH LoRA ---")
        outputs_lora = _generate(llm, sampling_params, lora_request=lora_request)
        errors += _check_outputs(outputs_lora, "lora")
    finally:
        shutdown_llm(llm)

    if not errors:
        print("\n✓ c5 LoRA sanity check tests PASSED")

    return errors != 0


def test_c5_lora_sanity_check():
    """Pytest entry point -- reads config from environment variables."""
    raw_model = os.environ.get(C5_MODEL_DIR_ENV_KEY)
    if not raw_model:
        raise ValueError(
            f"{C5_MODEL_DIR_ENV_KEY} is required for tests/cohere/test_c5_lora.py."
        )
    raw_lora = os.environ.get(C5_LORA_DIR_ENV_KEY)
    if not raw_lora:
        raise ValueError(
            f"{C5_LORA_DIR_ENV_KEY} is required for tests/cohere/test_c5_lora.py."
        )
    model_path = validate_model_path(raw_model)
    lora_path = validate_model_path(raw_lora)
    had_errors = run_c5_lora_sanity_check_test(model_path, lora_path)
    assert not had_errors, "c5 LoRA sanity check had failures"


def main():
    parser = argparse.ArgumentParser(
        description="LoRA serving sanity check for the c5 3a30t model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the c5 3a30t model checkpoint directory",
    )
    parser.add_argument(
        "--lora",
        type=str,
        required=True,
        help="Path to the LoRA adapter checkpoint directory",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    args = parser.parse_args()
    model_path = validate_model_path(args.model)
    lora_path = validate_model_path(args.lora)

    return run_c5_lora_sanity_check_test(
        model_path, lora_path, args.tensor_parallel_size
    )


if __name__ == "__main__":
    sys.exit(main())
