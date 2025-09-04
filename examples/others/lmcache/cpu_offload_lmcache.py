# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file demonstrates the example usage of cpu offloading
with LMCache in vLLM v1.

Note that lmcache needs to be installed to run this example.
Learn more about LMCache in https://github.com/LMCache/LMCache.

For more details about CPU offloading with LMCache, please refer to
https://docs.lmcache.ai/getting_started/quickstart/offload_kv_cache.html
"""

import argparse
import os
import time

import torch
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CPU offloading example with LMCache")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts to generate (default: 10)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=10000,
        help="Number of tokens per prompt (default: 10000)",
    )
    parser.add_argument(
        "--enable-lmcache",
        action="store_true",
        help="Enable LMCache for CPU offloading (default: True)",
    )
    return parser.parse_args()


def setup_lmcache_environment(num_prompts, num_tokens):
    """
    Configure LMCache environment variables.
    Args:
        num_prompts: Number of prompts to process
        num_tokens: Number of tokens per prompt
    """
    cpu_size = num_prompts * num_tokens * 1.5 / 10000  # 1.5GB per 10000 tokens

    env_vars = {
        "LMCACHE_CHUNK_SIZE": "256",  # Set tokens per chunk
        "LMCACHE_LOCAL_CPU": "True",  # Enable local CPU backend
        "LMCACHE_MAX_LOCAL_CPU_SIZE": str(cpu_size),  # Dynamic CPU memory limit (GB)
    }
    for key, value in env_vars.items():
        os.environ[key] = value


def calculate_gpu_utilization(target_memory_gb=24):
    """
    Calculate GPU memory utilization to use exactly target_memory_gb of GPU memory.
    Args:
        target_memory_gb: Target GPU memory usage in gigabytes
    Returns:
        float: GPU memory utilization ratio (0.0 to 1.0)
    Raises:
        RuntimeError: If GPU memory is less than target_memory_gb
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")

    total_memory = torch.cuda.get_device_properties(0).total_memory / (
        1024**3
    )  # Convert to GB
    if total_memory < target_memory_gb:
        raise RuntimeError(
            f"GPU memory ({total_memory:.1f}GB) is less than \
                required memory ({target_memory_gb}GB)"
        )

    return target_memory_gb / total_memory


def create_test_prompts(num_prompts=10, num_tokens=1000):
    """
    Create test prompts with index prefix and dummy body.
    Args:
        num_prompts: Number of prompts to generate
        num_tokens: Approximate number of tokens per prompt (using 'Hi ' as token unit)
    Returns:
        list: List of prompts with format '[index] Hi Hi Hi...'
    """
    prompts = []
    dummy_text = "Hi " * num_tokens

    for i in range(num_prompts):
        prompt = f"[Prompt {i}] {dummy_text} how are you?"
        prompts.append(prompt)

    return prompts


def initialize_llm(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_len=16384,
    enable_lmcache=True,
):
    """
    Initialize the LLM with appropriate configurations.
    Args:
        model_name: Name of the model to load
        max_len: Maximum sequence length
    Returns:
        LLM: Configured LLM instance
    """
    ktc = (
        KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
        )
        if enable_lmcache
        else None
    )

    return LLM(
        model=model_name,
        kv_transfer_config=ktc,
        max_model_len=max_len,
        enable_prefix_caching=False,
        gpu_memory_utilization=calculate_gpu_utilization(),
    )


def generate_and_print_output(llm, prompts, sampling_params):
    """
    Generate text and print the results.
    Args:
        llm: LLM instance
        prompts: List of input prompts
        sampling_params: Configured sampling parameters
    Returns:
        float: Time taken for generation in seconds
    """
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")

    return end_time - start_time


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()

    # Setup environment if LMCache is enabled
    if args.enable_lmcache:
        setup_lmcache_environment(args.num_prompts, args.num_tokens)

    # Create prompts and sampling parameters
    prompts = create_test_prompts(
        num_prompts=args.num_prompts, num_tokens=args.num_tokens
    )
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Initialize model
    llm = initialize_llm(enable_lmcache=args.enable_lmcache)

    # First run
    print("\nFirst run:")
    first_run_time = generate_and_print_output(llm, prompts, sampling_params)
    print(f"First run time: {first_run_time:.2f} seconds")

    # Second run
    print("\nSecond run:")
    second_run_time = generate_and_print_output(llm, prompts, sampling_params)
    print(f"Second run time: {second_run_time:.2f} seconds")

    # Print speedup
    if second_run_time > 0:
        speedup = first_run_time / second_run_time
        print(f"\nSpeedup (first run / second run): {speedup:.2f}x")
    else:
        print("\nSecond run was too fast to measure, cannot calculate speedup.")

    # Cleanup if LMCache was enabled
    if args.enable_lmcache:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


if __name__ == "__main__":
    main()
