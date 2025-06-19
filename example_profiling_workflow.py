#!/usr/bin/env python3
"""
Complete vLLM Profiling Workflow Example

This script demonstrates a complete workflow for profiling vLLM performance
and generating custom schedule plans based on the profiling results.

The workflow includes:
1. Workload generation with different configurations
2. Profiling execution with detailed metrics collection
3. Performance analysis and bottleneck identification
4. Custom schedule plan generation
5. Results visualization and reporting
"""

import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Import our custom modules
from profiling_workload_generator import (
    ProfilingConfig,
    ProfilingRunner,
    create_profiling_engine,
)
from custom_schedule_planner import CustomSchedulePlanner, SchedulePlan

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CompleteProfilingWorkflow:
    """Complete workflow for vLLM profiling and schedule plan generation."""

    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.profiling_results = []
        self.schedule_plans = []

    def run_profiling_suite(
        self,
        configs: List[Dict[str, Any]],
        model_name: str = "microsoft/DialoGPT-medium",
    ) -> List[Dict[str, Any]]:
        """Run profiling for multiple configurations."""
        logger.info(
            f"Starting profiling suite with {len(configs)} configurations"
        )

        results = []

        for i, config_dict in enumerate(configs):
            logger.info(
                f"Running configuration {i + 1}/{len(configs)}: {config_dict}"
            )

            # Create profiling configuration
            config = ProfilingConfig(
                C=config_dict["C"],
                M=config_dict["M"],
                B=config_dict["B"],
                block_size=config_dict.get("block_size", 16),
                num_warmup_runs=config_dict.get("num_warmup", 3),
                num_measurement_runs=config_dict.get("num_measurements", 10),
                max_model_len=config_dict.get("max_model_len", 8192),
                results_file=f"config_{i + 1}_results.json",
            )

            # Create engine
            engine = create_profiling_engine(config, model_name)

            # Create runner
            runner = ProfilingRunner(engine, config)

            # Run profiling
            profiling_type = config_dict.get("profiling_type", "mixed")
            logger.info(f"Running {profiling_type} profiling...")

            if profiling_type == "prefill":
                result = runner.run_prefill_profiling()
            elif profiling_type == "decode":
                result = runner.run_decode_profiling()
            else:  # mixed
                result = runner.run_mixed_profiling()

            # Save result
            result_dict = result.to_dict()
            result_dict["config_name"] = f"config_{i + 1}"
            result_dict["config_params"] = config_dict

            results.append(result_dict)

            # Save to file
            result_file = self.output_dir / f"config_{i + 1}_results.json"
            with open(result_file, "w") as f:
                json.dump(result_dict, f, indent=2)

            logger.info(
                f"Configuration {i + 1} completed. Results saved to {result_file}"
            )

        self.profiling_results = results
        return results

    def generate_custom_schedule_plans(
        self, target_optimization: str = "balanced"
    ) -> List[SchedulePlan]:
        """Generate custom schedule plans based on profiling results."""
        logger.info(
            f"Generating custom schedule plans for {target_optimization} optimization"
        )

        all_plans = []

        for result_dict in self.profiling_results:
            # Create ProfilingResult object
            config = ProfilingConfig(
                C=result_dict["config"]["C"],
                M=result_dict["config"]["M"],
                B=result_dict["config"]["B"],
                block_size=result_dict["config"]["block_size"],
            )

            # Create a simplified ProfilingResult (you might need to adjust this based on your actual ProfilingResult structure)
            from profiling_workload_generator import ProfilingResult

            profiling_result = ProfilingResult(
                config=config,
                prefill_latencies=result_dict.get("prefill_latencies", []),
                decode_latencies=result_dict.get("decode_latencies", []),
                total_latencies=result_dict.get("total_latencies", []),
                tokens_per_second=result_dict.get("tokens_per_second", 0),
                requests_per_second=result_dict.get("requests_per_second", 0),
            )

            # Create planner
            planner = CustomSchedulePlanner(profiling_result)

            # Generate plans based on target optimization
            if target_optimization == "throughput":
                plans = [planner._create_throughput_optimized_plan()]
            elif target_optimization == "latency":
                plans = [planner._create_latency_optimized_plan()]
            elif target_optimization == "memory":
                plans = [planner._create_memory_optimized_plan()]
            elif target_optimization == "cache":
                plans = [planner._create_cache_optimized_plan()]
            else:  # balanced
                plans = planner.generate_optimized_schedule_plans()

            # Add configuration info to plans
            for plan in plans:
                plan.name = f"{plan.name} - {result_dict['config_name']}"
                plan.description = f"{plan.description} (based on {result_dict['config_name']})"

            all_plans.extend(plans)

        self.schedule_plans = all_plans
        return all_plans

    def save_schedule_plans(self, filename: str = "custom_schedule_plans.json"):
        """Save all generated schedule plans to file."""
        plans_dict = {
            "profiling_results_summary": [
                {
                    "config_name": result["config_name"],
                    "config_params": result["config_params"],
                    "performance_summary": result.get(
                        "performance_summary", {}
                    ),
                }
                for result in self.profiling_results
            ],
            "schedule_plans": [plan.to_dict() for plan in self.schedule_plans],
        }

        output_file = self.output_dir / filename
        with open(output_file, "w") as f:
            json.dump(plans_dict, f, indent=2)

        logger.info(f"Schedule plans saved to {output_file}")
        return output_file

    def generate_workload_configurations(self) -> List[Dict[str, Any]]:
        """Generate a comprehensive set of workload configurations for profiling."""
        configs = []

        # Base configurations for different scenarios
        base_configs = [
            # Short prompts, small batches (low latency scenario)
            {
                "C": 64,
                "M": 4,
                "B": 8,
                "block_size": 16,
                "profiling_type": "mixed",
            },
            {
                "C": 128,
                "M": 8,
                "B": 16,
                "block_size": 16,
                "profiling_type": "mixed",
            },
            # Medium prompts, medium batches (balanced scenario)
            {
                "C": 512,
                "M": 32,
                "B": 32,
                "block_size": 16,
                "profiling_type": "mixed",
            },
            {
                "C": 1024,
                "M": 64,
                "B": 64,
                "block_size": 16,
                "profiling_type": "mixed",
            },
            # Long prompts, large batches (high throughput scenario)
            {
                "C": 2048,
                "M": 128,
                "B": 128,
                "block_size": 16,
                "profiling_type": "mixed",
            },
            {
                "C": 4096,
                "M": 256,
                "B": 256,
                "block_size": 16,
                "profiling_type": "mixed",
            },
            # Prefill-focused configurations
            {
                "C": 1024,
                "M": 64,
                "B": 32,
                "block_size": 16,
                "profiling_type": "prefill",
            },
            {
                "C": 2048,
                "M": 128,
                "B": 64,
                "block_size": 16,
                "profiling_type": "prefill",
            },
            # Decode-focused configurations
            {
                "C": 512,
                "M": 32,
                "B": 128,
                "block_size": 16,
                "profiling_type": "decode",
            },
            {
                "C": 1024,
                "M": 64,
                "B": 256,
                "block_size": 16,
                "profiling_type": "decode",
            },
            # Different block sizes
            {
                "C": 1024,
                "M": 64,
                "B": 64,
                "block_size": 8,
                "profiling_type": "mixed",
            },
            {
                "C": 1024,
                "M": 64,
                "B": 64,
                "block_size": 32,
                "profiling_type": "mixed",
            },
        ]

        # Add configurations with different measurement parameters
        for base_config in base_configs:
            # Standard measurement
            configs.append(
                {
                    **base_config,
                    "num_warmup": 3,
                    "num_measurements": 10,
                }
            )

            # Quick measurement (for testing)
            configs.append(
                {
                    **base_config,
                    "num_warmup": 1,
                    "num_measurements": 3,
                }
            )

        return configs

    def print_workflow_summary(self):
        """Print a summary of the complete workflow."""
        print("\n" + "=" * 80)
        print("COMPLETE PROFILING WORKFLOW SUMMARY")
        print("=" * 80)

        print(f"Output Directory: {self.output_dir}")
        print(
            f"Number of Profiling Configurations: {len(self.profiling_results)}"
        )
        print(f"Number of Generated Schedule Plans: {len(self.schedule_plans)}")
        print()

        print("Profiling Results Summary:")
        for result in self.profiling_results:
            config_name = result["config_name"]
            config_params = result["config_params"]
            performance = result.get("performance_summary", {})

            print(f"  {config_name}:")
            print(
                f"    C={config_params['C']}, M={config_params['M']}, "
                f"B={config_params['B']}, block_size={config_params.get('block_size', 16)}"
            )
            print(
                f"    Profiling Type: {config_params.get('profiling_type', 'mixed')}"
            )

            if "tokens_per_second" in result:
                print(
                    f"    Throughput: {result['tokens_per_second']:.2f} tokens/sec"
                )
            if "prefill_latencies" in result and result["prefill_latencies"]:
                avg_prefill = sum(result["prefill_latencies"]) / len(
                    result["prefill_latencies"]
                )
                print(f"    Avg Prefill Latency: {avg_prefill:.2f}ms")
            if "decode_latencies" in result and result["decode_latencies"]:
                avg_decode = sum(result["decode_latencies"]) / len(
                    result["decode_latencies"]
                )
                print(f"    Avg Decode Latency: {avg_decode:.2f}ms")
            print()

        print("Generated Schedule Plans:")
        for i, plan in enumerate(self.schedule_plans, 1):
            print(f"  {i}. {plan.name}")
            print(f"     Description: {plan.description}")
            print(f"     Key Parameters:")
            print(
                f"       max_num_batched_tokens: {plan.max_num_batched_tokens}"
            )
            print(f"       max_num_seqs: {plan.max_num_seqs}")
            print(
                f"       enable_chunked_prefill: {plan.enable_chunked_prefill}"
            )
            print(f"       block_size: {plan.block_size}")
            print(f"       scheduling_policy: {plan.scheduling_policy}")
            print()

        print("=" * 80)


def main():
    """Main function for running the complete profiling workflow."""
    parser = argparse.ArgumentParser(
        description="Complete vLLM Profiling Workflow"
    )

    # Workflow parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiling_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="Model to use for profiling",
    )
    parser.add_argument(
        "--target-optimization",
        choices=["throughput", "latency", "memory", "cache", "balanced"],
        default="balanced",
        help="Target optimization for schedule plans",
    )

    # Profiling parameters
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with minimal configurations",
    )
    parser.add_argument(
        "--custom-configs",
        type=str,
        help="Path to JSON file with custom configurations",
    )

    args = parser.parse_args()

    # Create workflow
    workflow = CompleteProfilingWorkflow(args.output_dir)

    # Generate or load configurations
    if args.custom_configs:
        with open(args.custom_configs, "r") as f:
            configs = json.load(f)
        logger.info(
            f"Loaded {len(configs)} custom configurations from {args.custom_configs}"
        )
    else:
        configs = workflow.generate_workload_configurations()
        if args.quick_test:
            # Use only a subset for quick testing
            configs = configs[:3]
        logger.info(f"Generated {len(configs)} configurations for profiling")

    # Run profiling suite
    logger.info("Starting profiling suite...")
    start_time = time.time()

    results = workflow.run_profiling_suite(configs, args.model)

    profiling_time = time.time() - start_time
    logger.info(f"Profiling suite completed in {profiling_time:.2f} seconds")

    # Generate custom schedule plans
    logger.info("Generating custom schedule plans...")
    plans = workflow.generate_custom_schedule_plans(args.target_optimization)

    # Save results
    workflow.save_schedule_plans()

    # Print summary
    workflow.print_workflow_summary()

    logger.info("Complete profiling workflow finished successfully!")


if __name__ == "__main__":
    main()


# Example usage functions
def example_quick_profiling():
    """Example of running a quick profiling session."""
    print("Running quick profiling example...")

    # Create a simple configuration
    config = ProfilingConfig(
        C=512,  # 512 input tokens
        M=32,  # 32 KV cache blocks
        B=32,  # Batch size of 32
        block_size=16,  # 16 tokens per block
        num_warmup_runs=2,
        num_measurement_runs=5,
        max_model_len=2048,
    )

    # Create engine
    engine = create_profiling_engine(config, "microsoft/DialoGPT-medium")

    # Create runner
    runner = ProfilingRunner(engine, config)

    # Run profiling
    result = runner.run_mixed_profiling()

    # Print results
    summary = result.get_summary_stats()
    print(f"\nQuick Profiling Results:")
    print(
        f"  Configuration: C={config.C}, M={config.M}, B={config.B}, block_size={config.block_size}"
    )
    print(
        f"  Prefill P95 Latency: {summary['prefill_stats']['p95_latency_ms']:.2f}ms"
    )
    print(
        f"  Decode P95 Latency: {summary['decode_stats']['p95_latency_ms']:.2f}ms"
    )
    print(
        f"  Throughput: {summary['throughput']['tokens_per_second']:.2f} tokens/sec"
    )

    return result


def example_custom_schedule_generation():
    """Example of generating custom schedule plans."""
    print("Running custom schedule generation example...")

    # Create a sample profiling result
    config = ProfilingConfig(
        C=1024,
        M=64,
        B=64,
        block_size=16,
        num_warmup_runs=3,
        num_measurement_runs=10,
    )

    # Create a sample ProfilingResult (you would normally get this from actual profiling)
    from profiling_workload_generator import ProfilingResult

    profiling_result = ProfilingResult(
        config=config,
        prefill_latencies=[50.0, 52.0, 48.0, 51.0, 49.0],
        decode_latencies=[5.0, 5.2, 4.8, 5.1, 4.9],
        total_latencies=[55.0, 57.2, 52.8, 56.1, 53.9],
        tokens_per_second=1200.0,
        requests_per_second=18.5,
    )

    # Create planner
    planner = CustomSchedulePlanner(profiling_result)

    # Generate plans
    plans = planner.generate_optimized_schedule_plans()

    # Print plans
    print(f"\nGenerated {len(plans)} custom schedule plans:")
    for i, plan in enumerate(plans, 1):
        print(f"  {i}. {plan.name}")
        print(f"     Description: {plan.description}")
        print(f"     max_num_batched_tokens: {plan.max_num_batched_tokens}")
        print(f"     max_num_seqs: {plan.max_num_seqs}")
        print(f"     enable_chunked_prefill: {plan.enable_chunked_prefill}")
        print()

    return plans


if __name__ == "__main__":
    # Uncomment to run examples
    # example_quick_profiling()
    # example_custom_schedule_generation()
    main()
