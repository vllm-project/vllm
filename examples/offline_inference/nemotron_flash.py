# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Nemotron-Flash-3B Example

This script demonstrates offline inference with NVIDIA's Nemotron-Flash-3B,
a hybrid architecture model that combines:
- Mamba2 (selective state space) layers
- DeltaNet (gated linear attention) layers
- Standard Transformer attention layers

Requirements:
    pip install fla  # Flash Linear Attention library for DeltaNet kernels

Usage:
    python nemotron_flash.py
    python nemotron_flash.py --mode chat
    python nemotron_flash.py --mode batch
    python nemotron_flash.py --enforce-eager  # Disable CUDA graphs

Known Limitations:
    - Memory: Requires low gpu_memory_utilization (~0.2) due to large state
      tensors for Mamba2 and DeltaNet layers
    - CUDA Graphs: Currently limited to 1-2 concurrent prompts with CUDA graphs
      enabled. Use --enforce-eager for >2 prompts until CUDA graph support for
      larger batch sizes is fully implemented.
    - Context Length: Best to limit max_model_len for memory efficiency
"""

import argparse

from vllm import LLM
from vllm.sampling_params import SamplingParams


def run_generate(args: argparse.Namespace):
    """Run simple text generation with Nemotron-Flash."""
    print("\n=== Nemotron-Flash-3B: Text Generation ===\n")

    llm = LLM(
        model="nvidia/Nemotron-Flash-3B",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
    )

    # Note: Limited to 2 prompts for CUDA graph compatibility.
    # Use --enforce-eager for more prompts, or --mode batch.
    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Generated: {generated!r}")
        print("-" * 60)


def run_chat(args: argparse.Namespace):
    """Run chat-style inference with Nemotron-Flash."""
    print("\n=== Nemotron-Flash-3B: Chat Mode ===\n")

    llm = LLM(
        model="nvidia/Nemotron-Flash-3B",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
    )

    # Simple conversation
    messages = [
        {
            "role": "user",
            "content": "What are the main differences between Mamba and "
            "Transformer architectures?",
        }
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )

    outputs = llm.chat(messages, sampling_params=sampling_params)
    print("User:", messages[0]["content"])
    print("\nAssistant:", outputs[0].outputs[0].text)
    print("-" * 60)


def run_batch(args: argparse.Namespace):
    """Run batch inference demonstrating multi-prompt handling."""
    print("\n=== Nemotron-Flash-3B: Batch Inference ===\n")

    # For >2 prompts with CUDA graphs, enforce_eager is recommended
    use_eager = args.enforce_eager or args.max_num_seqs > 2
    if use_eager and not args.enforce_eager:
        print("Note: Using enforce_eager=True for batch size > 2\n")

    llm = LLM(
        model="nvidia/Nemotron-Flash-3B",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=use_eager,
    )

    # Batch of diverse prompts
    prompts = [
        "Write a haiku about programming:",
        "Explain quantum computing in simple terms:",
        "What is the capital of Japan?",
        "Translate 'Hello, world!' to French:",
        "Give me a recipe for chocolate cake:",
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    print(f"Processed {len(outputs)} prompts:\n")
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"[{i + 1}] Prompt: {prompt}")
        print(f"    Output: {generated[:100]}...")
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nemotron-Flash-3B offline inference example"
    )

    parser.add_argument(
        "--mode",
        choices=["generate", "chat", "batch"],
        default="generate",
        help="Inference mode: generate (default), chat, or batch",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Maximum context length (default: 512)",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.2,
        help="GPU memory utilization (default: 0.2, limited by state size)",
    )

    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="Maximum number of concurrent sequences (default: 4)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate per prompt (default: 50)",
    )

    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs (recommended for >2 prompts)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "generate":
        run_generate(args)
    elif args.mode == "chat":
        run_chat(args)
    elif args.mode == "batch":
        run_batch(args)


if __name__ == "__main__":
    main()
