#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstration of KV Cache Lifetime Tracking Feature

This script shows how to use the new KV cache lifetime tracking feature
that was added to vLLM. It demonstrates:

1. Basic setup of vLLM with lifetime tracking enabled
2. Running inference to generate KV cache activity
3. Accessing lifetime statistics
4. Viewing the Prometheus metric

Usage:
    python examples/kv_cache_lifetime_tracking_demo.py
"""

import time

# Import vLLM components
from vllm import LLM, SamplingParams


def create_llm_with_lifetime_tracking():
    """Create an LLM instance with KV cache lifetime tracking enabled."""
    print("Creating LLM with KV cache lifetime tracking...")

    # Initialize with a small model for demo purposes
    llm = LLM(
        model="facebook/opt-125m",  # Small model for quick demo
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enable_prefix_caching=True,  # Enable caching to see lifetime tracking
        disable_log_stats=False,  # Enable statistics logging
    )

    return llm


def run_inference_workload(llm: LLM, prompts: list[str]) -> None:
    """Run inference to generate KV cache activity."""
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=50)

    # Generate responses - this will create and free KV cache blocks
    outputs = llm.generate(prompts, sampling_params)
    return outputs


def demonstrate_lifetime_stats():
    """Demonstrate the KV cache lifetime tracking feature."""
    print("KV Cache Lifetime Tracking Demonstration")
    print("-" * 50)

    # Create LLM instance
    llm = create_llm_with_lifetime_tracking()

    # Prepare some test prompts that will exercise the KV cache
    prompts = [
        "The future of artificial intelligence is",
        "Machine learning algorithms can help us",
        "The benefits of renewable energy include",
        "Space exploration has led to many discoveries such as",
        "The importance of education in society is",
    ]

    # Run inference
    print("Running inference with prefix caching enabled...")
    run_inference_workload(llm, prompts[:3])
    time.sleep(0.5)
    run_inference_workload(llm, prompts[3:])

    # Get lifetime stats via public API
    print("\nKV Cache Lifetime Statistics:")
    print("-" * 50)

    from vllm.v1.metrics.reader import Histogram

    for metric in llm.get_metrics():
        if isinstance(metric, Histogram) and "lifetime" in metric.name:
            print(f"{metric.name}:")
            print(f"  Total lifetime (sum): {metric.sum:.4f} seconds")
            print(f"  Total blocks freed (count): {metric.count}")
            if metric.count > 0:
                avg = metric.sum / metric.count
                print(f"  Average lifetime: {avg:.4f} seconds")
            print("\n  Buckets:")
            for bucket_le, value in metric.buckets.items():
                print(f"    le {bucket_le}: {value}")
            break

    print("\n" + "-" * 50)
    print("Demonstration completed!")
    print("-" * 50)


def show_promql_examples():
    """Show PromQL query examples."""
    print("\nPromQL Query Examples:")
    print("-" * 50)
    print("Average lifetime over 5m:")
    print("  rate(vllm:kv_cache_block_lifetime_seconds_sum[5m]) /")
    print("  rate(vllm:kv_cache_block_lifetime_seconds_count[5m])")
    print("\n90th percentile lifetime:")
    print("  histogram_quantile(0.9,")
    print("    rate(vllm:kv_cache_block_lifetime_seconds_bucket[5m]))")
    print("-" * 50)


if __name__ == "__main__":
    demonstrate_lifetime_stats()
    show_promql_examples()
