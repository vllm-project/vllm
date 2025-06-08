# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script demonstrating FLOP counting capabilities in vLLM.

This example shows how to:
1. Use the FlopContextManager for basic FLOP counting
2. Use layerwise profiling with FLOP counting enabled
3. Display FLOP summaries and performance metrics

Run with:
    python flop_counting.py
"""

import argparse
import time

from vllm import LLM, SamplingParams
from vllm.profiler import (
    FlopContextManager,
    format_flops,
    layerwise_profile,
)


def basic_flop_counting_example():
    """Example using FlopContextManager for basic FLOP counting."""
    print("=== Basic FLOP Counting Example ===")

    # Create LLM instance
    llm = LLM(model="facebook/opt-125m", max_num_seqs=1)

    # Sampling parameters
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

    # Test prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Explain quantum computing in simple terms:",
    ]

    # Generate with FLOP counting (using enhanced offline counter)
    with FlopContextManager(auto_print=True) as flop_counter:
        outputs = llm.generate(prompts, sampling_params)

    # Additional detailed analysis
    detailed_counts = flop_counter.get_detailed_counts()
    breakdown_percentages = detailed_counts.get_percentage_breakdown()

    print("\nDetailed FLOP Breakdown by Percentage:")
    for category, percentage in breakdown_percentages.items():
        if percentage > 0 and category != "total_flops":
            print(f"  {category:20s}: {percentage:5.1f}%")

    print("\nGenerated outputs:")
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt {i + 1}: {prompt}")
        print(f"Output: {generated_text}\n")


def layerwise_profiling_with_flops_example():
    """Example using layerwise profiling with FLOP counting."""
    print("\n=== Layerwise Profiling with FLOPs Example ===")

    # Create LLM instance
    llm = LLM(model="facebook/opt-125m", max_num_seqs=1)

    # Sampling parameters
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

    prompt = "The quick brown fox"

    # Profile with FLOP counting enabled
    with layerwise_profile(num_running_seqs=1, enable_flop_counting=True) as profiler:
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        end_time = time.time()

    # Get profiling results
    results = profiler.results

    print(f"Generated text: {outputs[0].outputs[0].text}")
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    # Display FLOP summary
    results.print_flop_summary()

    # Display timing and FLOP model table (first 20 entries)
    print("\n=== Model Stats (Top 20 by CUDA time) ===")
    results.print_model_table(
        column_widths={
            "name": 50,
            "cuda_time_us": 12,
            "flops": 12,
            "gflops_per_sec": 12,
            "trace": 40,
        }
    )


def performance_analysis_example():
    """Example showing performance analysis with different model sizes."""
    print("\n=== Performance Analysis Example ===")

    models = ["facebook/opt-125m"]  # Can add more models: "facebook/opt-350m", etc.

    for model_name in models:
        print(f"\n--- Analyzing {model_name} ---")

        try:
            llm = LLM(model=model_name, max_num_seqs=1)
            sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

            start_time = time.time()
            with FlopContextManager() as flop_counter:
                outputs = llm.generate(["Hello world"], sampling_params)
            elapsed_time = time.time() - start_time

            total_flops = flop_counter.get_total_flops()
            efficiency_metrics = flop_counter.get_efficiency_metrics(elapsed_time)

            print(f"  Total FLOPs: {format_flops(total_flops)}")
            print(f"  Time: {elapsed_time:.3f}s")
            gflops = efficiency_metrics["gflops_per_sec"]
            tflops = efficiency_metrics["tflops_per_sec"]
            print(f"  Performance: {gflops:.2f} GFLOPS/sec")
            print(f"  Performance: {tflops:.4f} TFLOPS/sec")
            print(f"  Generated: {outputs[0].outputs[0].text}")

        except Exception as e:
            print(f"  Error with {model_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="vLLM FLOP counting examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flop_counting.py --basic          # Run basic FLOP counting example
  python flop_counting.py --profiling      # Run layerwise profiling example  
  python flop_counting.py --analysis       # Run performance analysis
  python flop_counting.py --all            # Run all examples
        """,
    )

    parser.add_argument(
        "--basic", action="store_true", help="Run basic FLOP counting example"
    )
    parser.add_argument(
        "--profiling",
        action="store_true",
        help="Run layerwise profiling with FLOPs example",
    )
    parser.add_argument(
        "--analysis", action="store_true", help="Run performance analysis example"
    )
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if not any([args.basic, args.profiling, args.analysis, args.all]):
        # Default to basic example if no specific example is chosen
        args.basic = True

    if args.basic or args.all:
        basic_flop_counting_example()

    if args.profiling or args.all:
        layerwise_profiling_with_flops_example()

    if args.analysis or args.all:
        performance_analysis_example()


if __name__ == "__main__":
    main()
