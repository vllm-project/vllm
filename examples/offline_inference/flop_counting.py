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


def get_generation_stats(llm, prompts, sampling_params):
    """Helper function to compute generation_stats for FLOP estimation.

    Args:
        llm: The vLLM LLM instance
        prompts: List of prompts or single prompt string
        sampling_params: SamplingParams with max_tokens set

    Returns:
        dict: generation_stats dictionary for FlopContextManager
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    # Calculate input statistics
    tokenizer = llm.llm_engine.tokenizer
    input_lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
    avg_input_length = sum(input_lengths) // len(input_lengths)
    batch_size = len(prompts)

    return {
        "input_shape": (batch_size, avg_input_length),
        "num_generated_tokens": sampling_params.max_tokens,
    }


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
    model_config = llm.llm_engine.model_config.hf_config
    generation_stats = get_generation_stats(llm, prompts, sampling_params)

    with FlopContextManager(
        auto_print=False, model_config=model_config, generation_stats=generation_stats
    ) as flop_counter:
        outputs = llm.generate(prompts, sampling_params)

    # Additional detailed analysis
    detailed_counts = flop_counter.get_detailed_counts()
    total_flops = flop_counter.get_total_flops()

    print(f"\nTotal FLOPs: {format_flops(total_flops)}")

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

    # Set up model estimation parameters
    model_config = llm.llm_engine.model_config.hf_config
    generation_stats = get_generation_stats(llm, [prompt], sampling_params)

    # Profile with FLOP counting enabled
    with layerwise_profile(num_running_seqs=1, enable_flop_counting=True) as profiler:
        # Set up model-based FLOP estimation as fallback
        profiler.flop_counter.set_model_for_estimation(model_config, generation_stats)

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

            # Set up model estimation parameters
            model_config = llm.llm_engine.model_config.hf_config
            generation_stats = get_generation_stats(
                llm, ["Hello world"], sampling_params
            )

            start_time = time.time()
            with FlopContextManager(
                model_config=model_config, generation_stats=generation_stats
            ) as flop_counter:
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


def multi_model_comparison_example():
    """Example comparing FLOP usage across different model architectures."""
    print("\n=== Multi-Model Architecture Comparison ===")

    # Define test models for different architectures
    test_models = [
        {
            "name": "Standard Transformer (OPT-125M)",
            "model_id": "facebook/opt-125m",
            "type": "transformer",
        },
        # Note: Uncomment these if you have access to the models
        # {
        #     "name": "Mixture of Experts (Mixtral-8x7B)",
        #     "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        #     "type": "moe"
        # },
        # {
        #     "name": "Embedding Model (E5-Mistral)",
        #     "model_id": "intfloat/e5-mistral-7b-instruct",
        #     "type": "embedding"
        # },
        # {
        #     "name": "Multi-modal (LLaVA)",
        #     "model_id": "llava-hf/llava-1.5-7b-hf",
        #     "type": "multimodal"
        # }
    ]

    results = []
    test_prompt = "Explain the concept of artificial intelligence."

    for model_info in test_models:
        print(f"\n--- Testing {model_info['name']} ---")

        try:
            # Create LLM instance
            llm = LLM(model=model_info["model_id"], max_num_seqs=1)
            sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

            # Set up model estimation parameters
            model_config = llm.llm_engine.model_config.hf_config
            generation_stats = get_generation_stats(llm, [test_prompt], sampling_params)

            # Set up FLOP counting with model-specific estimation
            with FlopContextManager(
                model_config=model_config, generation_stats=generation_stats
            ) as flop_counter:
                start_time = time.time()
                outputs = llm.generate([test_prompt], sampling_params)
                end_time = time.time()

            # Collect results
            detailed_counts = flop_counter.get_detailed_counts()
            total_flops = flop_counter.get_total_flops()
            elapsed_time = end_time - start_time

            result = {
                "model": model_info["name"],
                "type": model_info["type"],
                "total_flops": total_flops,
                "time": elapsed_time,
                "breakdown": detailed_counts.get_breakdown_dict(),
                "generated_text": outputs[0].outputs[0].text,
            }
            results.append(result)

            # Print detailed analysis
            print(f"  Model Type: {model_info['type']}")
            print(f"  Total FLOPs: {format_flops(total_flops)}")
            print(f"  Inference Time: {elapsed_time:.3f}s")
            print(
                f"  FLOP Efficiency: {total_flops / elapsed_time / 1e9:.2f} GFLOPS/sec"
            )

            # Show FLOP breakdown
            breakdown = detailed_counts.get_percentage_breakdown()
            print("  FLOP Breakdown:")
            for category, percentage in breakdown.items():
                if percentage > 0 and category != "total_flops":
                    flops_in_category = detailed_counts.get_breakdown_dict()[category]
                    flop_str = format_flops(flops_in_category)
                    print(f"    {category:20s}: {percentage:5.1f}% ({flop_str})")

            print(f"  Generated: {outputs[0].outputs[0].text[:50]}...")

        except Exception as e:
            print(f"  Error testing {model_info['name']}: {e}")
            continue

    # Summary comparison
    if len(results) > 1:
        print("\n=== Model Comparison Summary ===")
        print(
            f"{'Model':<30} {'Type':<12} {'Total FLOPs':<15} {'Time':<8} "
            f"{'Efficiency':<12}"
        )
        print("-" * 85)

        for result in results:
            efficiency = result["total_flops"] / result["time"] / 1e9
            print(
                f"{result['model'][:29]:<30} {result['type']:<12} "
                f"{format_flops(result['total_flops']):<15} {result['time']:.2f}s   "
                f"{efficiency:.1f} GFLOPS/s"
            )


def main():
    parser = argparse.ArgumentParser(
        description="vLLM FLOP counting examples for multiple model architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flop_counting.py --basic          # Run basic FLOP counting example
  python flop_counting.py --profiling      # Run layerwise profiling example  
  python flop_counting.py --analysis       # Run performance analysis
  python flop_counting.py --multimodel     # Compare different model architectures
  python flop_counting.py --moe            # Show MoE-specific analysis features
  python flop_counting.py --embedding      # Show embedding model analysis
  python flop_counting.py --multimodal     # Show multi-modal model analysis
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
    parser.add_argument(
        "--multimodel",
        action="store_true",
        help="Compare FLOP usage across different model architectures",
    )
    parser.add_argument(
        "--moe",
        action="store_true",
        help="Show Mixture-of-Experts FLOP analysis features",
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Show embedding model FLOP analysis features",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Show multi-modal model FLOP analysis features",
    )
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if not any(
        [
            args.basic,
            args.profiling,
            args.analysis,
            args.multimodel,
            args.moe,
            args.embedding,
            args.multimodal,
            args.all,
        ]
    ):
        # Default to basic example if no specific example is chosen
        args.basic = True

    if args.basic or args.all:
        basic_flop_counting_example()

    if args.profiling or args.all:
        layerwise_profiling_with_flops_example()

    if args.analysis or args.all:
        performance_analysis_example()

    if args.multimodel or args.all:
        multi_model_comparison_example()


if __name__ == "__main__":
    main()
