#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple example showing how to use the all_reduce registry with custom backends.

Usage:
    # With custom backend (sets VLLM_ALLREDUCE_BACKEND)
    python examples/offline_inference/all_reduce_registry_example.py \
        --backend=my_custom_backend

    # Without setting environment variable (uses system default)
    python examples/offline_inference/all_reduce_registry_example.py --no-env-backend
"""

import torch
from torch.distributed import ProcessGroup

from vllm import LLM, SamplingParams
from vllm.distributed.device_communicators.all_reduce_registry import (
    get_allreduce_info,
    register_allreduce_backend,
)


def my_custom_all_reduce(tensor: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    """
    Example custom all_reduce backend.

    This shows how to write your own all_reduce function.
    For safety, it uses torch.distributed.all_reduce internally.
    """
    print(f"Using my custom all_reduce! Tensor shape: {tensor.shape}")

    # Your custom logic here (preprocessing, logging, etc.)
    result = tensor.clone()
    torch.distributed.all_reduce(result, group=group)

    print("Custom all_reduce completed!")
    return result


def main():
    # Parse simple arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.2-3B-Instruct", help="Model to use"
    )
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument(
        "--backend", default="torch_distributed", help="All-reduce backend"
    )
    parser.add_argument(
        "--no-env-backend",
        action="store_true",
        help="Don't set VLLM_ALLREDUCE_BACKEND environment variable",
    )
    args = parser.parse_args()

    print("=== All-Reduce Registry Example ===")

    # Step 1: Register your custom backend
    print("1. Registering custom all_reduce backend...")
    register_allreduce_backend(name="my_custom_backend", backend=my_custom_all_reduce)

    # Step 2: Show what backends are available
    print("2. Available backends:")
    info = get_allreduce_info()
    print(f"   Backends: {info['backends']}")
    print(f"   Default: {info['default']}")

    # Step 3: Set which backend to use
    import os

    if args.no_env_backend:
        print("3. Skipping VLLM_ALLREDUCE_BACKEND (using system default)")
    else:
        print(f"3. Setting VLLM_ALLREDUCE_BACKEND to: {args.backend}")
        os.environ["VLLM_ALLREDUCE_BACKEND"] = args.backend

    # Step 4: Create LLM and run inference
    print("4. Creating LLM...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
    )

    print("5. Running inference...")
    prompts = ["Hello, my name is", "The capital of France is"]
    sampling_params = SamplingParams(temperature=0, max_tokens=20, top_k=1)
    outputs = llm.generate(prompts, sampling_params)

    # Step 5: Show results
    print("6. Results:")
    for output in outputs:
        print(f"   Prompt: {output.prompt}")
        print(f"   Output: {output.outputs[0].text}")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
