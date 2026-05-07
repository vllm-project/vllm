#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Reward model test for Command-A Reward model.
Tests reward scores for certain inputs.
"""

import argparse
import sys

import torch
from test_utils_engine_args import get_engine_kwargs_with_overrides

from vllm import LLM, PoolingParams

PROMPTS = [
    (
        "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>What's the capital of "
        "Canada?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        "Ottawa<|END_OF_TURN_TOKEN|><EOS_TOKEN>"
    ),
    (
        "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>What's the capital of "
        "Canada?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        "Toronto<|END_OF_TURN_TOKEN|><EOS_TOKEN>"
    ),
]


def run_reward_test(
    model_path: str,
    tensor_parallel_size: int = 1,
    engine_args: str | None = None,
):
    """
    Run reward test

    Args:
        model_path: Path to the model checkpoint
        tensor_parallel_size: Number of GPUs for tensor parallelism
        engine_args: CLI-style engine args (overrides
            VLLM_HARDWARE_PROFILE_ARGS env var)
    """
    print(f"Loading model from: {model_path}")
    print(f"Using tensor_parallel_size: {tensor_parallel_size}")

    # Get effective engine kwargs with hardware profile args + test-specific overrides
    effective_kwargs = get_engine_kwargs_with_overrides(
        test_kwargs={
            "runner": "pooling",
            "model": model_path,
            "max_num_batched_tokens": 32768,
            "tensor_parallel_size": tensor_parallel_size,
            "seed": 0,
        },
        engine_args_override=engine_args,
    )

    # Create LLM instance with merged kwargs
    llm = LLM(**effective_kwargs)
    pooling_params = [
        PoolingParams(use_activation=False),
        PoolingParams(use_activation=False),
    ]
    outputs = llm.encode(
        PROMPTS,
        pooling_params=pooling_params,
        pooling_task="token_classify",
        tokenization_kwargs={"add_special_tokens": False},
    )
    scores = [x.outputs.data for x in outputs]

    score0 = torch.as_tensor(scores[0], dtype=torch.float32).flatten()[-1]
    score1 = torch.as_tensor(scores[1], dtype=torch.float32).flatten()[-1]

    assert torch.allclose(
        score0, torch.tensor(3.5156, dtype=torch.float32), atol=2e-1
    ), f"Got wrong score with Prompt 0: {score0}, Ground Truth: 3.5156"
    assert torch.allclose(
        score1, torch.tensor(0.1680, dtype=torch.float32), atol=2e-1
    ), f"Got wrong score with Prompt 1: {score1}, Ground Truth: 0.1680"


def main():
    parser = argparse.ArgumentParser(description="Test Reward model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Number of GPUs for tensor parallelism (default: 4)",
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

    return run_reward_test(args.model, args.tensor_parallel_size, args.engine_args)


if __name__ == "__main__":
    sys.exit(main())
