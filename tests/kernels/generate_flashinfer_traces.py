#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generate FlashInfer Bench traces from vLLM inference.

This script runs inference on various models with FlashInfer enabled and
captures workload traces using FlashInfer-Bench. These traces can then be
used to optimize FlashInfer operations with custom kernels.

Reference: https://bench.flashinfer.ai/docs/start/quickstart

Installation:
    pip install flashinfer-bench --no-deps

Usage:
    python tests/kernels/generate_flashinfer_traces.py --model qwen
    python tests/kernels/generate_flashinfer_traces.py --model llama
    python tests/kernels/generate_flashinfer_traces.py --model all --output-dir /path/to/traces
"""

import argparse
import os
import sys
from pathlib import Path


def check_compatibility():
    """Check if the environment is compatible for trace generation.
    
    Returns:
        tuple: (is_compatible, error_message)
    """
    # Check torch version
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        return False, "torch is not installed"
    
    # Check if flashinfer-bench is installed
    try:
        import importlib.util
        spec = importlib.util.find_spec("flashinfer_bench")
        if spec is None:
            return False, (
                "flashinfer-bench is not installed.\n"
                "Install with: pip install flashinfer-bench --no-deps"
            )
    except Exception as e:
        return False, f"Error checking flashinfer-bench: {e}"
    
    # Check if vLLM can be imported
    try:
        import vllm  # noqa: F401
    except Exception as e:
        return False, f"vLLM import failed: {e}"
    
    return True, f"Compatible (torch {torch_version})"


def generate_traces(
    model_id: str,
    model_name: str,
    tensor_parallel_size: int,
    num_prompts: int,
    max_tokens: int,
    output_dir: str,
    quantization: str | None = None,
    trust_remote_code: bool = True,
    extra_env_vars: dict | None = None,
):
    """Generate FlashInfer traces for a specific model."""
    
    # Set any extra environment variables
    if extra_env_vars:
        for key, value in extra_env_vars.items():
            os.environ[key] = value
    
    # Set up tracing
    os.environ["FIB_ENABLE_TRACING"] = "1"
    os.environ["FIB_DATASET_PATH"] = output_dir
    os.environ.setdefault("VLLM_USE_FLASHINFER", "1")
    
    # Import flashinfer_bench FIRST to install tracing adapters
    import flashinfer_bench  # noqa: F401
    print(f"✓ flashinfer_bench tracing enabled")
    
    # Now import vLLM
    from vllm import LLM, SamplingParams
    
    print(f"\n{'='*70}")
    print(f"Generating traces for: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"TP: {tensor_parallel_size}, Quantization: {quantization or 'none'}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Initialize the model
    llm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": 2048,
        "trust_remote_code": trust_remote_code,
        "gpu_memory_utilization": 0.7,
    }
    
    if quantization:
        llm_kwargs["quantization"] = quantization
    
    llm = LLM(**llm_kwargs)
    
    # Generate diverse prompts
    prompts = [
        "What is 2+2?",
        "Hello!",
        "What is the capital of France? Please answer briefly.",
        "Write a haiku about programming.",
        "Explain machine learning in simple terms.",
        "Write a short story about a robot.",
        "Compare the Renaissance and Enlightenment periods.",
    ]
    
    test_prompts = [prompts[i % len(prompts)] for i in range(num_prompts)]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    
    print(f"Running inference with {len(test_prompts)} prompts...")
    outputs = llm.generate(test_prompts, sampling_params)
    
    print(f"\n✓ Generated {len(outputs)} outputs")
    if outputs:
        print(f"Sample: {outputs[0].outputs[0].text[:100]}...")
    
    del llm
    print(f"\n✓ Traces saved to {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate FlashInfer Bench traces from vLLM inference"
    )
    parser.add_argument(
        "--model", type=str, default="qwen",
        choices=["qwen", "llama", "gpt-oss", "all"],
        help="Model to generate traces for",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to store traces",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=5,
        help="Number of prompts to run",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128,
        help="Maximum tokens per prompt",
    )
    parser.add_argument(
        "--tp", type=int, default=2,
        help="Tensor parallel size",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FLASHINFER BENCH TRACE GENERATION")
    print("=" * 70)
    
    # Check compatibility first
    is_compatible, message = check_compatibility()
    if not is_compatible:
        print(f"\n✗ ERROR: {message}")
        return 1
    
    print(f"\n✓ {message}")
    
    output_dir = args.output_dir or str(Path.home() / ".cache" / "flashinfer_bench" / "vllm_traces")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    models = {
        "qwen": {
            "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "model_name": "Qwen3-30B-A3B MoE",
            "tensor_parallel_size": args.tp,
        },
        "llama": {
            "model_id": "meta-llama/Llama-3.1-70B-Instruct",
            "model_name": "Llama-3.1-70B",
            "tensor_parallel_size": max(args.tp, 4),
        },
        "gpt-oss": {
            "model_id": "openai/gpt-oss-120b",
            "model_name": "GPT-OSS-120B",
            "tensor_parallel_size": max(args.tp, 4),
            "extra_env_vars": {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"},
        },
    }
    
    models_to_run = list(models.keys()) if args.model == "all" else [args.model]
    results = {}
    
    for model_key in models_to_run:
        config = models[model_key]
        model_output_dir = os.path.join(output_dir, model_key)
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            success = generate_traces(
                model_id=config["model_id"],
                model_name=config["model_name"],
                tensor_parallel_size=config["tensor_parallel_size"],
                num_prompts=args.num_prompts,
                max_tokens=args.max_tokens,
                output_dir=model_output_dir,
                extra_env_vars=config.get("extra_env_vars"),
            )
            results[model_key] = "✓ SUCCESS"
        except Exception as e:
            print(f"\n✗ Error: {e}")
            results[model_key] = f"✗ ERROR: {e}"
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for model_key, result in results.items():
        print(f"  {model_key}: {result}")
    
    print(f"\nTraces saved to: {output_dir}")
    print("\nTo analyze traces:")
    print(f"  flashinfer-bench run --local {output_dir}")
    
    return 0 if all("SUCCESS" in r for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
