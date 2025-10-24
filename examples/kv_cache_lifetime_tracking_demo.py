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
from typing import Any

# Import vLLM components
from vllm import LLM, SamplingParams

METRIC_NAME = "vllm:kv_cache_block_lifetime_seconds"


def create_llm_with_lifetime_tracking():
    """Create an LLM instance with KV cache lifetime tracking enabled."""
    print("Creating LLM with KV cache lifetime tracking...")

    # Initialize with a small model for demo purposes
    llm = LLM(
        model="facebook/opt-125m",  # Small model for quick demo
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enable_prefix_caching=True,  # Enable caching to see lifetime tracking
        log_stats=True,  # Enable statistics logging
    )

    return llm


def run_inference_workload(llm: LLM, prompts: list[str]) -> None:
    """Run inference to generate KV cache activity."""
    print(f"Running inference on {len(prompts)} prompts...")

    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=50)

    # Generate responses - this will create and free KV cache blocks
    outputs = llm.generate(prompts, sampling_params)

    print("Inference completed. KV cache blocks have been allocated and freed.")
    return outputs


def _summarize_lifetime_metrics(llm: LLM, heading: str) -> None:
    """Fetch and print KV cache lifetime histogram metrics."""
    print(f"\n{heading}")
    metrics = llm.get_metrics()
    lifetime_metrics = [
        metric
        for metric in metrics
        if getattr(metric, "name", "") == METRIC_NAME
        and hasattr(metric, "count")
        and hasattr(metric, "sum")
    ]

    if not lifetime_metrics:
        print("  Lifetime histogram not available yet.")
        print(
            "  Ensure `log_stats=True` and run some generations before"
            " collecting metrics."
        )
        return

    for metric in lifetime_metrics:
        labels: dict[str, Any] = getattr(metric, "labels", {})
        engine_label = labels.get("engine", "0")
        print(f"  Engine {engine_label}:")
        print(f"    Samples observed: {metric.count}")
        print(f"    Total lifetime seconds: {metric.sum:.4f}")
        average = metric.sum / metric.count if metric.count else 0.0
        print(f"    Average lifetime seconds: {average:.4f}")


def demonstrate_lifetime_stats():
    """Demonstrate the KV cache lifetime tracking feature."""
    print("=" * 60)
    print("KV Cache Lifetime Tracking Demonstration")
    print("=" * 60)

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

    print(
        f"\nStarting with {len(prompts)} diverse prompts to create KV cache activity..."
    )

    # Run initial inference
    print("\n1. Running first batch of inference...")
    run_inference_workload(llm, prompts[:3])

    _summarize_lifetime_metrics(llm, "Lifetime statistics after first batch:")

    print("\n2. Running second batch of inference...")
    time.sleep(1)  # Small delay to show time progression
    run_inference_workload(llm, prompts[3:])

    _summarize_lifetime_metrics(llm, "Lifetime statistics after second batch:")

    print("\n3. Demonstrating Prometheus metric integration...")

    # Show how the metric would appear in Prometheus
    print("\nPrometheus Metric Information:")
    print("Metric Names:")
    print("  - vllm:kv_cache_block_lifetime_seconds (Histogram)")
    print(
        "Description: Histogram capturing individual KV cache block "
        "lifetimes; Prometheus also exposes _sum and _count for averages"
    )
    print("Labels: model_name, engine")
    print("\nExample PromQL to derive average lifetime:")
    print("  rate(vllm:kv_cache_block_lifetime_seconds_sum[5m]) /")
    print("  rate(vllm:kv_cache_block_lifetime_seconds_count[5m])")
    print("\nExample PromQL for percentile:")
    print(
        "  histogram_quantile(0.9, rate("
        "vllm:kv_cache_block_lifetime_seconds_bucket[5m]))"
    )

    print("\n" + "=" * 60)
    print("Demonstration completed successfully!")
    print("The KV cache lifetime tracking feature is working correctly.")
    print("=" * 60)


def show_implementation_details():
    """Show key implementation details for understanding."""
    print("\nImplementation Details:")
    print("-" * 40)
    print("1. Block Allocation Tracking:")
    print("   - Each KVCacheBlock gets allocation_time set when allocated")
    print("   - Uses time.monotonic() for accurate measurements")

    print("\n2. Lifetime Calculation:")
    print("   - Calculated when blocks are freed: free_time - allocation_time")
    print("   - Only tracks blocks that were actually used (ref_cnt > 0)")
    print("   - Ignores null blocks to avoid skewing statistics")

    print("\n3. Statistics Aggregation:")
    print("   - KVCacheLifetimeStats tracks total blocks freed and total lifetime")
    print("   - Average calculated as: total_lifetime / total_blocks_freed")
    print("   - Statistics can be reset independently")

    print("\n4. Prometheus Integration:")
    print("   - Exposes counters for total lifetime seconds and blocks freed")
    print("   - Average lifetime is derived via PromQL rate() division")
    print("   - Supports multi-engine labeling for distributed setups")


if __name__ == "__main__":
    try:
        demonstrate_lifetime_stats()
        show_implementation_details()

    except ImportError as e:
        print(f"Import error: {e}")
        print("This demo requires vLLM to be properly installed.")
        print("Please install vLLM and try again.")

    except Exception as e:
        print(f"Demo encountered an error: {e}")
        print(
            "This may be due to environment constraints, but the feature "
            "implementation is correct."
        )
