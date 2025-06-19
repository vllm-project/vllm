#!/usr/bin/env python3
"""
Custom Schedule Planner for vLLM

This module generates optimized schedule plans based on profiling results
and workload characteristics, considering the key factors:
- C: Number of input tokens/prompt
- M: Number of KV Cache blocks for precomputed tokens
- B: Batch size
- block_size: KV Cache block size
"""

import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
import logging

from profiling_workload_generator import ProfilingConfig, ProfilingResult

logger = logging.getLogger(__name__)


@dataclass
class SchedulePlan:
    """A custom schedule plan configuration."""

    name: str
    description: str

    # Core scheduling parameters
    max_num_batched_tokens: int
    max_num_seqs: int
    max_model_len: int

    # Advanced scheduling parameters
    enable_chunked_prefill: bool
    max_num_partial_prefills: int
    max_long_partial_prefills: int
    long_prefill_token_threshold: int
    scheduling_policy: str

    # Cache parameters
    block_size: int
    gpu_memory_utilization: float
    swap_space: float

    # Performance targets
    target_latency_ms: Optional[float] = None
    target_throughput_tokens_per_sec: Optional[float] = None
    target_memory_mb: Optional[float] = None

    # Expected performance
    expected_prefill_latency_ms: Optional[float] = None
    expected_decode_latency_ms: Optional[float] = None
    expected_throughput_tokens_per_sec: Optional[float] = None
    expected_memory_usage_mb: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class WorkloadProfile:
    """Profile of a specific workload."""

    # Workload characteristics
    C: int  # Input tokens per prompt
    M: int  # KV cache blocks
    B: int  # Batch size
    block_size: int  # KV cache block size

    # Performance characteristics
    avg_prefill_latency_ms: float
    avg_decode_latency_ms: float
    avg_throughput_tokens_per_sec: float
    avg_memory_usage_mb: float

    # Resource utilization
    gpu_utilization_percent: float
    cache_hit_rate: float
    batch_efficiency: float

    # Bottlenecks
    is_compute_bound: bool
    is_memory_bound: bool
    is_scheduling_bound: bool
    is_cache_bound: bool


class CustomSchedulePlanner:
    """Generates custom schedule plans based on workload analysis."""

    def __init__(self, profiling_result: ProfilingResult):
        self.profiling_result = profiling_result
        self.config = profiling_result.config

        # Create workload profile
        self.workload_profile = self._create_workload_profile()

    def _create_workload_profile(self) -> WorkloadProfile:
        """Create a workload profile from profiling results."""
        summary = self.profiling_result.get_summary_stats()

        return WorkloadProfile(
            C=self.config.C,
            M=self.config.M,
            B=self.config.B,
            block_size=self.config.block_size,
            avg_prefill_latency_ms=summary["prefill_stats"]["mean_latency_ms"],
            avg_decode_latency_ms=summary["decode_stats"]["mean_latency_ms"],
            avg_throughput_tokens_per_sec=summary["throughput"][
                "tokens_per_second"
            ],
            avg_memory_usage_mb=8000,  # Placeholder - get from actual profiling
            gpu_utilization_percent=85,  # Placeholder - get from actual profiling
            cache_hit_rate=0.9,  # Placeholder - get from actual profiling
            batch_efficiency=0.8,  # Placeholder - get from actual profiling
            is_compute_bound=False,  # Will be determined by analysis
            is_memory_bound=False,  # Will be determined by analysis
            is_scheduling_bound=False,  # Will be determined by analysis
            is_cache_bound=False,  # Will be determined by analysis
        )

    def analyze_workload_characteristics(self) -> Dict[str, Any]:
        """Analyze workload characteristics to determine bottlenecks."""
        profile = self.workload_profile

        analysis = {
            "compute_intensity": self._calculate_compute_intensity(),
            "memory_intensity": self._calculate_memory_intensity(),
            "scheduling_complexity": self._calculate_scheduling_complexity(),
            "cache_efficiency": self._calculate_cache_efficiency(),
            "bottlenecks": self._identify_bottlenecks(),
            "optimization_opportunities": self._identify_optimization_opportunities(),
        }

        return analysis

    def _calculate_compute_intensity(self) -> float:
        """Calculate compute intensity of the workload."""
        # Compute intensity = (C * B) / (prefill_latency + decode_latency)
        total_tokens = self.config.C * self.config.B
        total_latency = (
            self.workload_profile.avg_prefill_latency_ms
            + self.workload_profile.avg_decode_latency_ms
        )

        if total_latency > 0:
            return total_tokens / total_latency
        return 0.0

    def _calculate_memory_intensity(self) -> float:
        """Calculate memory intensity of the workload."""
        # Memory intensity = (M * B * block_size) / memory_usage
        kv_cache_size = self.config.M * self.config.B * self.config.block_size
        memory_usage = (
            self.workload_profile.avg_memory_usage_mb * 1024 * 1024
        )  # Convert to bytes

        if memory_usage > 0:
            return kv_cache_size / memory_usage
        return 0.0

    def _calculate_scheduling_complexity(self) -> float:
        """Calculate scheduling complexity."""
        # Complexity based on batch size, sequence length, and scheduling policy
        complexity = (self.config.B * self.config.C) / (
            self.config.max_num_seqs * self.config.max_model_len
        )
        return min(complexity, 1.0)

    def _calculate_cache_efficiency(self) -> float:
        """Calculate cache efficiency."""
        # Based on cache hit rate and batch efficiency
        return (
            self.workload_profile.cache_hit_rate
            + self.workload_profile.batch_efficiency
        ) / 2

    def _identify_bottlenecks(self) -> Dict[str, bool]:
        """Identify performance bottlenecks."""
        bottlenecks = {
            "compute_bound": False,
            "memory_bound": False,
            "scheduling_bound": False,
            "cache_bound": False,
        }

        # Compute-bound: high latency relative to token count
        if (
            self.workload_profile.avg_prefill_latency_ms > 100
            or self.workload_profile.avg_decode_latency_ms > 50
        ):
            bottlenecks["compute_bound"] = True

        # Memory-bound: high memory usage relative to KV cache size
        kv_cache_size_mb = (
            self.config.M * self.config.B * self.config.block_size
        ) / (1024 * 1024)
        if self.workload_profile.avg_memory_usage_mb > kv_cache_size_mb * 2:
            bottlenecks["memory_bound"] = True

        # Scheduling-bound: low batch efficiency
        if self.workload_profile.batch_efficiency < 0.7:
            bottlenecks["scheduling_bound"] = True

        # Cache-bound: low cache hit rate
        if self.workload_profile.cache_hit_rate < 0.8:
            bottlenecks["cache_bound"] = True

        return bottlenecks

    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []

        # Check for underutilization
        if self.workload_profile.gpu_utilization_percent < 70:
            opportunities.append(
                "GPU underutilization - consider increasing batch size"
            )

        if self.workload_profile.batch_efficiency < 0.8:
            opportunities.append(
                "Low batch efficiency - consider adjusting max_num_batched_tokens"
            )

        if self.workload_profile.cache_hit_rate < 0.9:
            opportunities.append(
                "Low cache hit rate - consider increasing block_size or enabling prefix caching"
            )

        # Check for memory inefficiency
        kv_cache_size_mb = (
            self.config.M * self.config.B * self.config.block_size
        ) / (1024 * 1024)
        if self.workload_profile.avg_memory_usage_mb < kv_cache_size_mb * 0.5:
            opportunities.append(
                "Memory underutilization - consider increasing max_num_seqs"
            )

        return opportunities

    def generate_optimized_schedule_plans(self) -> List[SchedulePlan]:
        """Generate optimized schedule plans based on workload analysis."""
        analysis = self.analyze_workload_characteristics()
        bottlenecks = analysis["bottlenecks"]

        plans = []

        # Plan 1: Throughput-optimized
        if (
            bottlenecks["compute_bound"]
            or self.workload_profile.gpu_utilization_percent < 80
        ):
            plans.append(self._create_throughput_optimized_plan())

        # Plan 2: Latency-optimized
        if (
            self.workload_profile.avg_prefill_latency_ms > 50
            or self.workload_profile.avg_decode_latency_ms > 20
        ):
            plans.append(self._create_latency_optimized_plan())

        # Plan 3: Memory-optimized
        if (
            bottlenecks["memory_bound"]
            or self.workload_profile.avg_memory_usage_mb > 15000
        ):
            plans.append(self._create_memory_optimized_plan())

        # Plan 4: Cache-optimized
        if (
            bottlenecks["cache_bound"]
            or self.workload_profile.cache_hit_rate < 0.85
        ):
            plans.append(self._create_cache_optimized_plan())

        # Plan 5: Balanced
        plans.append(self._create_balanced_plan())

        return plans

    def _create_throughput_optimized_plan(self) -> SchedulePlan:
        """Create a throughput-optimized schedule plan."""
        # Increase batch size and token budget for higher throughput
        new_max_batched_tokens = int(self.config.max_num_batched_tokens * 1.5)
        new_max_seqs = int(self.config.max_num_seqs * 1.3)

        # Enable chunked prefill for better batching
        new_max_partial_prefills = min(8, new_max_seqs // 4)

        return SchedulePlan(
            name="Throughput Optimized",
            description="Optimized for maximum tokens per second",
            max_num_batched_tokens=new_max_batched_tokens,
            max_num_seqs=new_max_seqs,
            max_model_len=self.config.max_model_len,
            enable_chunked_prefill=True,
            max_num_partial_prefills=new_max_partial_prefills,
            max_long_partial_prefills=new_max_partial_prefills // 2,
            long_prefill_token_threshold=int(self.config.max_model_len * 0.1),
            scheduling_policy="fcfs",
            block_size=self.config.block_size,
            gpu_memory_utilization=0.95,
            swap_space=2.0,
            target_throughput_tokens_per_sec=self.workload_profile.avg_throughput_tokens_per_sec
            * 1.3,
            expected_throughput_tokens_per_sec=self.workload_profile.avg_throughput_tokens_per_sec
            * 1.2,
        )

    def _create_latency_optimized_plan(self) -> SchedulePlan:
        """Create a latency-optimized schedule plan."""
        # Reduce batch size and token budget for lower latency
        new_max_batched_tokens = int(self.config.max_num_batched_tokens * 0.7)
        new_max_seqs = int(self.config.max_num_seqs * 0.8)

        # Use priority scheduling for better latency
        new_max_partial_prefills = 2

        return SchedulePlan(
            name="Latency Optimized",
            description="Optimized for minimum response time",
            max_num_batched_tokens=new_max_batched_tokens,
            max_num_seqs=new_max_seqs,
            max_model_len=self.config.max_model_len,
            enable_chunked_prefill=True,
            max_num_partial_prefills=new_max_partial_prefills,
            max_long_partial_prefills=1,
            long_prefill_token_threshold=int(self.config.max_model_len * 0.05),
            scheduling_policy="priority",
            block_size=min(32, self.config.block_size * 2),
            gpu_memory_utilization=0.85,
            swap_space=1.0,
            target_latency_ms=self.workload_profile.avg_prefill_latency_ms
            * 0.7,
            expected_prefill_latency_ms=self.workload_profile.avg_prefill_latency_ms
            * 0.8,
            expected_decode_latency_ms=self.workload_profile.avg_decode_latency_ms
            * 0.8,
        )

    def _create_memory_optimized_plan(self) -> SchedulePlan:
        """Create a memory-optimized schedule plan."""
        # Reduce memory usage through smaller batches and CPU offloading
        new_max_batched_tokens = int(self.config.max_num_batched_tokens * 0.5)
        new_max_seqs = int(self.config.max_num_seqs * 0.6)

        return SchedulePlan(
            name="Memory Optimized",
            description="Optimized for low memory usage",
            max_num_batched_tokens=new_max_batched_tokens,
            max_num_seqs=new_max_seqs,
            max_model_len=self.config.max_model_len,
            enable_chunked_prefill=False,  # Disable to reduce memory overhead
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            long_prefill_token_threshold=0,
            scheduling_policy="fcfs",
            block_size=self.config.block_size,
            gpu_memory_utilization=0.7,
            swap_space=8.0,  # Increase swap space
            target_memory_mb=self.workload_profile.avg_memory_usage_mb * 0.6,
            expected_memory_usage_mb=self.workload_profile.avg_memory_usage_mb
            * 0.7,
        )

    def _create_cache_optimized_plan(self) -> SchedulePlan:
        """Create a cache-optimized schedule plan."""
        # Optimize for better cache efficiency
        new_block_size = max(16, self.config.block_size * 2)
        new_max_batched_tokens = int(self.config.max_num_batched_tokens * 1.2)

        return SchedulePlan(
            name="Cache Optimized",
            description="Optimized for cache efficiency",
            max_num_batched_tokens=new_max_batched_tokens,
            max_num_seqs=self.config.max_num_seqs,
            max_model_len=self.config.max_model_len,
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=int(self.config.max_model_len * 0.08),
            scheduling_policy="fcfs",
            block_size=new_block_size,
            gpu_memory_utilization=0.9,
            swap_space=2.0,
            expected_throughput_tokens_per_sec=self.workload_profile.avg_throughput_tokens_per_sec
            * 1.1,
        )

    def _create_balanced_plan(self) -> SchedulePlan:
        """Create a balanced schedule plan."""
        # Moderate adjustments for balanced performance
        new_max_batched_tokens = int(self.config.max_num_batched_tokens * 1.1)
        new_max_seqs = int(self.config.max_num_seqs * 1.1)

        return SchedulePlan(
            name="Balanced",
            description="Balanced throughput and latency",
            max_num_batched_tokens=new_max_batched_tokens,
            max_num_seqs=new_max_seqs,
            max_model_len=self.config.max_model_len,
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=int(self.config.max_model_len * 0.06),
            scheduling_policy="fcfs",
            block_size=self.config.block_size,
            gpu_memory_utilization=0.9,
            swap_space=3.0,
            expected_throughput_tokens_per_sec=self.workload_profile.avg_throughput_tokens_per_sec
            * 1.05,
            expected_prefill_latency_ms=self.workload_profile.avg_prefill_latency_ms
            * 0.95,
            expected_decode_latency_ms=self.workload_profile.avg_decode_latency_ms
            * 0.95,
        )

    def generate_parameter_sweep_plans(
        self,
        C_range: List[int] = None,
        M_range: List[int] = None,
        B_range: List[int] = None,
        block_size_range: List[int] = None,
    ) -> List[SchedulePlan]:
        """Generate schedule plans for parameter sweep analysis."""
        if C_range is None:
            C_range = [self.config.C // 2, self.config.C, self.config.C * 2]
        if M_range is None:
            M_range = [self.config.M // 2, self.config.M, self.config.M * 2]
        if B_range is None:
            B_range = [self.config.B // 2, self.config.B, self.config.B * 2]
        if block_size_range is None:
            block_size_range = [8, 16, 32]

        plans = []

        for C in C_range:
            for M in M_range:
                for B in B_range:
                    for block_size in block_size_range:
                        # Skip invalid combinations
                        if C > self.config.max_model_len or M * block_size < C:
                            continue

                        plan = SchedulePlan(
                            name=f"Parameter Sweep C{C}_M{M}_B{B}_BS{block_size}",
                            description=f"Parameter sweep: C={C}, M={M}, B={B}, block_size={block_size}",
                            max_num_batched_tokens=C * B,
                            max_num_seqs=B * 2,
                            max_model_len=self.config.max_model_len,
                            enable_chunked_prefill=True,
                            max_num_partial_prefills=min(4, B // 2),
                            max_long_partial_prefills=min(2, B // 4),
                            long_prefill_token_threshold=int(
                                self.config.max_model_len * 0.05
                            ),
                            scheduling_policy="fcfs",
                            block_size=block_size,
                            gpu_memory_utilization=0.9,
                            swap_space=2.0,
                        )
                        plans.append(plan)

        return plans

    def save_schedule_plans(
        self,
        plans: List[SchedulePlan],
        filename: str = "custom_schedule_plans.json",
    ):
        """Save schedule plans to file."""
        plans_dict = {
            "workload_profile": asdict(self.workload_profile),
            "original_config": asdict(self.config),
            "schedule_plans": [plan.to_dict() for plan in plans],
        }

        with open(filename, "w") as f:
            json.dump(plans_dict, f, indent=2)

        logger.info(f"Schedule plans saved to {filename}")

    def print_schedule_plans_summary(self, plans: List[SchedulePlan]):
        """Print a summary of generated schedule plans."""
        print("\n" + "=" * 80)
        print("CUSTOM SCHEDULE PLANS SUMMARY")
        print("=" * 80)

        print(f"Workload Profile:")
        print(
            f"  C={self.workload_profile.C}, M={self.workload_profile.M}, "
            f"B={self.workload_profile.B}, block_size={self.workload_profile.block_size}"
        )
        print(
            f"  Avg Prefill Latency: {self.workload_profile.avg_prefill_latency_ms:.2f}ms"
        )
        print(
            f"  Avg Decode Latency: {self.workload_profile.avg_decode_latency_ms:.2f}ms"
        )
        print(
            f"  Avg Throughput: {self.workload_profile.avg_throughput_tokens_per_sec:.2f} tokens/sec"
        )
        print()

        print("Generated Schedule Plans:")
        for i, plan in enumerate(plans, 1):
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

            if plan.expected_throughput_tokens_per_sec:
                print(
                    f"     Expected Throughput: {plan.expected_throughput_tokens_per_sec:.2f} tokens/sec"
                )
            if plan.expected_prefill_latency_ms:
                print(
                    f"     Expected Prefill Latency: {plan.expected_prefill_latency_ms:.2f}ms"
                )
            if plan.expected_decode_latency_ms:
                print(
                    f"     Expected Decode Latency: {plan.expected_decode_latency_ms:.2f}ms"
                )
            print()

        print("=" * 80)


def main():
    """Main function for generating custom schedule plans."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Custom Schedule Planner for vLLM"
    )

    # Input parameters
    parser.add_argument(
        "--profiling-results",
        type=str,
        required=True,
        help="Path to profiling results JSON file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="custom_schedule_plans.json",
        help="Output file for schedule plans",
    )
    parser.add_argument(
        "--generate-parameter-sweep",
        action="store_true",
        help="Generate parameter sweep plans",
    )

    args = parser.parse_args()

    # Load profiling results
    with open(args.profiling_results, "r") as f:
        results_data = json.load(f)

    # Create ProfilingResult object (simplified)
    config = ProfilingConfig(
        C=results_data["config"]["C"],
        M=results_data["config"]["M"],
        B=results_data["config"]["B"],
        block_size=results_data["config"]["block_size"],
    )

    profiling_result = ProfilingResult(
        config=config,
        prefill_latencies=results_data.get("prefill_latencies", []),
        decode_latencies=results_data.get("decode_latencies", []),
        total_latencies=results_data.get("total_latencies", []),
        tokens_per_second=results_data.get("tokens_per_second", 0),
        requests_per_second=results_data.get("requests_per_second", 0),
    )

    # Create planner
    planner = CustomSchedulePlanner(profiling_result)

    # Generate optimized plans
    optimized_plans = planner.generate_optimized_schedule_plans()

    # Generate parameter sweep plans if requested
    if args.generate_parameter_sweep:
        sweep_plans = planner.generate_parameter_sweep_plans()
        all_plans = optimized_plans + sweep_plans
    else:
        all_plans = optimized_plans

    # Save plans
    planner.save_schedule_plans(all_plans, args.output_file)

    # Print summary
    planner.print_schedule_plans_summary(all_plans)


if __name__ == "__main__":
    main()
