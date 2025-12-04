# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Benchmark the __slots__ optimization impact on vLLM v1 performance.

This benchmark measures the real-world performance improvements from adding
__slots__ to Request and KVCacheBlock classes.

Example usage:
    python benchmarks/benchmark_slots_optimization.py \
        --num-requests 10000 \
        --num-blocks 10000 \
        --num-iterations 50000

    python benchmarks/benchmark_slots_optimization.py \
        --output-json results.json

Run all benchmarks with default settings:
    python benchmarks/benchmark_slots_optimization.py
"""

import argparse
import dataclasses
import gc
import json
import time
import tracemalloc

try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


@dataclasses.dataclass
class BenchmarkResults:
    """Results from the __slots__ optimization benchmark."""

    memory_baseline_bytes: float = 0.0
    memory_optimized_bytes: float = 0.0
    memory_savings_percent: float = 0.0
    creation_baseline_throughput: float = 0.0
    creation_optimized_throughput: float = 0.0
    creation_speedup: float = 0.0
    access_baseline_ops_per_sec: float = 0.0
    access_optimized_ops_per_sec: float = 0.0
    access_speedup: float = 0.0
    gc_baseline_time_sec: float = 0.0
    gc_optimized_time_sec: float = 0.0
    gc_improvement_percent: float = 0.0


class RequestBaseline:
    """Request without __slots__ (baseline)."""

    def __init__(self, request_id: str, num_tokens: int = 100):
        self.request_id = request_id
        self.client_index = 0
        self.priority = 0
        self.sampling_params = None
        self.pooling_params = None
        self.eos_token_id = 2
        self.lora_request = None
        self.structured_output_request = None
        self.arrival_time = time.time()
        self.status = 1
        self.events = []
        self.stop_reason = None
        self.kv_transfer_params = None
        self.max_tokens = 100
        self.prompt_token_ids = list(range(num_tokens))
        self.prompt_embeds = None
        self.num_prompt_tokens = num_tokens
        self._output_token_ids = []
        self._all_token_ids = list(range(num_tokens))
        self.num_output_placeholders = 0
        self.discard_latest_async_tokens = False
        self.spec_token_ids = []
        self.num_computed_tokens = 0
        self.cache_salt = None
        self.mm_features = []
        self.num_encoder_inputs = 0
        self.has_encoder_inputs = False
        self.output_token_ids = []
        self.all_token_ids = list(range(num_tokens))
        self.trace_headers = None
        self.num_cached_tokens = -1
        self.num_nans_in_logits = 0
        self.num_preemptions = 0
        self.num_external_computed_tokens = 0
        self.block_hashes = []
        self.get_hash_new_full_blocks = None
        self.skip_reading_prefix_cache = False


@dataclasses.dataclass
class KVCacheBlockBaseline:
    """KVCacheBlock without slots (baseline)."""

    block_id: int
    ref_cnt: int = 0
    _block_hash: bytes | None = None
    prev_free_block: "KVCacheBlockBaseline | None" = None
    next_free_block: "KVCacheBlockBaseline | None" = None
    is_null: bool = False


class RequestOptimized:
    """Request with __slots__ (optimized)."""

    __slots__ = (
        "_all_token_ids",
        "_output_token_ids",
        "all_token_ids",
        "arrival_time",
        "block_hashes",
        "cache_salt",
        "client_index",
        "discard_latest_async_tokens",
        "eos_token_id",
        "events",
        "get_hash_new_full_blocks",
        "has_encoder_inputs",
        "kv_transfer_params",
        "lora_request",
        "max_tokens",
        "mm_features",
        "num_cached_tokens",
        "num_computed_tokens",
        "num_encoder_inputs",
        "num_external_computed_tokens",
        "num_nans_in_logits",
        "num_output_placeholders",
        "num_preemptions",
        "num_prompt_tokens",
        "output_token_ids",
        "pooling_params",
        "priority",
        "prompt_embeds",
        "prompt_token_ids",
        "request_id",
        "sampling_params",
        "skip_reading_prefix_cache",
        "spec_token_ids",
        "status",
        "stop_reason",
        "structured_output_request",
        "trace_headers",
    )

    def __init__(self, request_id: str, num_tokens: int = 100):
        self.request_id = request_id
        self.client_index = 0
        self.priority = 0
        self.sampling_params = None
        self.pooling_params = None
        self.eos_token_id = 2
        self.lora_request = None
        self.structured_output_request = None
        self.arrival_time = time.time()
        self.status = 1
        self.events = []
        self.stop_reason = None
        self.kv_transfer_params = None
        self.max_tokens = 100
        self.prompt_token_ids = list(range(num_tokens))
        self.prompt_embeds = None
        self.num_prompt_tokens = num_tokens
        self._output_token_ids = []
        self._all_token_ids = list(range(num_tokens))
        self.num_output_placeholders = 0
        self.discard_latest_async_tokens = False
        self.spec_token_ids = []
        self.num_computed_tokens = 0
        self.cache_salt = None
        self.mm_features = []
        self.num_encoder_inputs = 0
        self.has_encoder_inputs = False
        self.output_token_ids = []
        self.all_token_ids = list(range(num_tokens))
        self.trace_headers = None
        self.num_cached_tokens = -1
        self.num_nans_in_logits = 0
        self.num_preemptions = 0
        self.num_external_computed_tokens = 0
        self.block_hashes = []
        self.get_hash_new_full_blocks = None
        self.skip_reading_prefix_cache = False


@dataclasses.dataclass(slots=True)
class KVCacheBlockOptimized:
    """KVCacheBlock with slots (optimized)."""

    block_id: int
    ref_cnt: int = 0
    _block_hash: bytes | None = None
    prev_free_block: "KVCacheBlockOptimized | None" = None
    next_free_block: "KVCacheBlockOptimized | None" = None
    is_null: bool = False


def format_number(n: float) -> str:
    """Format large numbers with commas."""
    return f"{n:,.0f}"


def format_bytes(bytes_val: float) -> str:
    """Format bytes to human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val:.1f} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    else:
        return f"{bytes_val / (1024 * 1024):.1f} MB"


def format_duration(seconds: float) -> str:
    """Format duration in appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f} us"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def benchmark_memory(
    request_class,
    block_class,
    num_requests: int,
    num_blocks: int,
) -> dict:
    """Measure memory consumption of object creation."""
    tracemalloc.start()
    tracemalloc.reset_peak()

    requests = [request_class(f"req_{i}", num_tokens=100) for i in range(num_requests)]
    blocks = [block_class(block_id=i) for i in range(num_blocks)]

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    del requests, blocks

    return {
        "peak_memory": peak,
        "per_instance": peak / (num_requests + num_blocks),
    }


def benchmark_creation_speed(
    obj_class,
    num_iterations: int,
    is_request: bool,
) -> dict:
    """Measure object creation throughput."""
    gc.collect()
    start = time.perf_counter()

    if is_request:
        for i in range(num_iterations):
            obj = obj_class(f"req_{i}", num_tokens=100)
            del obj
    else:
        for i in range(num_iterations):
            obj = obj_class(block_id=i)
            del obj

    elapsed = time.perf_counter() - start

    return {
        "total_time": elapsed,
        "per_object": elapsed / num_iterations,
        "throughput": num_iterations / elapsed,
    }


def benchmark_attribute_access(
    obj_class,
    num_accesses: int,
    is_request: bool,
) -> dict:
    """Measure attribute access speed."""
    if is_request:
        obj = obj_class("test_request", num_tokens=100)
        attrs = [
            "request_id",
            "client_index",
            "status",
            "max_tokens",
            "num_prompt_tokens",
            "num_computed_tokens",
            "arrival_time",
        ]
    else:
        obj = obj_class(block_id=1)
        attrs = ["block_id", "ref_cnt", "_block_hash", "is_null"]

    start = time.perf_counter()
    for _ in range(num_accesses):
        for attr in attrs:
            _ = getattr(obj, attr)
    read_time = time.perf_counter() - start

    start = time.perf_counter()
    for i in range(num_accesses):
        if is_request:
            obj.status = i % 5
            obj.num_computed_tokens = i
        else:
            obj.ref_cnt = i % 100
            obj.is_null = i % 2 == 0
    write_time = time.perf_counter() - start

    total_ops = num_accesses * (len(attrs) + 2)

    return {
        "read_time": read_time,
        "write_time": write_time,
        "total_time": read_time + write_time,
        "ops_per_sec": total_ops / (read_time + write_time),
    }


def benchmark_gc_pressure(
    obj_class,
    num_iterations: int,
    is_request: bool,
) -> dict:
    """Measure garbage collection pressure."""
    gc.collect()
    gc.disable()

    start = time.perf_counter()
    for i in range(num_iterations):
        if is_request:
            obj = obj_class(f"req_{i}", num_tokens=100)
        else:
            obj = obj_class(block_id=i)
        del obj
    elapsed = time.perf_counter() - start

    gc_start = time.perf_counter()
    gc.collect()
    gc_time = time.perf_counter() - gc_start

    gc.enable()

    return {
        "creation_time": elapsed,
        "gc_time": gc_time,
    }


def run_benchmarks(args: argparse.Namespace) -> BenchmarkResults:
    """Run all benchmarks and return results."""
    results = BenchmarkResults()

    print("=" * 70)
    print("vLLM __slots__ Optimization Benchmark")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  num_requests:   {format_number(args.num_requests)}")
    print(f"  num_blocks:     {format_number(args.num_blocks)}")
    print(f"  num_iterations: {format_number(args.num_iterations)}")
    print(f"  num_accesses:   {format_number(args.num_accesses)}")
    print()

    # Benchmark 1: Memory Usage
    if not args.skip_memory:
        print("Benchmark 1: Memory Usage")
        print("-" * 70)

        print(
            f"  Creating {format_number(args.num_requests)} requests "
            f"and {format_number(args.num_blocks)} blocks..."
        )

        baseline_mem = benchmark_memory(
            RequestBaseline,
            KVCacheBlockBaseline,
            args.num_requests,
            args.num_blocks,
        )
        optimized_mem = benchmark_memory(
            RequestOptimized,
            KVCacheBlockOptimized,
            args.num_requests,
            args.num_blocks,
        )

        savings = baseline_mem["peak_memory"] - optimized_mem["peak_memory"]
        reduction = (savings / baseline_mem["peak_memory"]) * 100

        results.memory_baseline_bytes = baseline_mem["peak_memory"]
        results.memory_optimized_bytes = optimized_mem["peak_memory"]
        results.memory_savings_percent = reduction

        print(f"  Baseline:  {format_bytes(baseline_mem['peak_memory'])}")
        print(f"  Optimized: {format_bytes(optimized_mem['peak_memory'])}")
        print(f"  Savings:   {format_bytes(savings)} ({reduction:.1f}% reduction)")
        print()
    else:
        print("Benchmark 1: Memory Usage [SKIPPED]")
        print()

    # Benchmark 2: Object Creation Speed
    if not args.skip_creation:
        print("Benchmark 2: Object Creation Speed")
        print("-" * 70)

        print(
            f"  Creating and destroying {format_number(args.num_iterations)} objects..."
        )

        baseline_create = benchmark_creation_speed(
            RequestBaseline, args.num_iterations, True
        )
        optimized_create = benchmark_creation_speed(
            RequestOptimized, args.num_iterations, True
        )

        speedup = baseline_create["total_time"] / optimized_create["total_time"]

        results.creation_baseline_throughput = baseline_create["throughput"]
        results.creation_optimized_throughput = optimized_create["throughput"]
        results.creation_speedup = speedup

        print(
            f"  Baseline:  {format_duration(baseline_create['total_time'])} "
            f"({format_number(baseline_create['throughput'])} req/s)"
        )
        print(
            f"  Optimized: {format_duration(optimized_create['total_time'])} "
            f"({format_number(optimized_create['throughput'])} req/s)"
        )
        print(f"  Speedup:   {speedup:.2f}x faster")
        print()
    else:
        print("Benchmark 2: Object Creation Speed [SKIPPED]")
        print()

    # Benchmark 3: Attribute Access Speed
    if not args.skip_access:
        print("Benchmark 3: Attribute Access Speed")
        print("-" * 70)

        print(f"  Performing {format_number(args.num_accesses)} attribute accesses...")

        baseline_access = benchmark_attribute_access(
            RequestBaseline, args.num_accesses, True
        )
        optimized_access = benchmark_attribute_access(
            RequestOptimized, args.num_accesses, True
        )

        speedup = baseline_access["total_time"] / optimized_access["total_time"]

        results.access_baseline_ops_per_sec = baseline_access["ops_per_sec"]
        results.access_optimized_ops_per_sec = optimized_access["ops_per_sec"]
        results.access_speedup = speedup

        print(
            f"  Baseline:  {format_duration(baseline_access['total_time'])} "
            f"({format_number(baseline_access['ops_per_sec'])} ops/s)"
        )
        print(
            f"  Optimized: {format_duration(optimized_access['total_time'])} "
            f"({format_number(optimized_access['ops_per_sec'])} ops/s)"
        )
        print(f"  Speedup:   {speedup:.2f}x faster")
        print()
    else:
        print("Benchmark 3: Attribute Access Speed [SKIPPED]")
        print()

    # Benchmark 4: GC Pressure
    if not args.skip_gc:
        print("Benchmark 4: Garbage Collection Pressure")
        print("-" * 70)

        print(
            f"  Measuring GC pressure with "
            f"{format_number(args.num_iterations)} objects..."
        )

        baseline_gc = benchmark_gc_pressure(RequestBaseline, args.num_iterations, True)
        optimized_gc = benchmark_gc_pressure(
            RequestOptimized, args.num_iterations, True
        )

        improvement = (
            (baseline_gc["gc_time"] - optimized_gc["gc_time"])
            / baseline_gc["gc_time"]
            * 100
        )

        results.gc_baseline_time_sec = baseline_gc["gc_time"]
        results.gc_optimized_time_sec = optimized_gc["gc_time"]
        results.gc_improvement_percent = improvement

        print(f"  Baseline GC time:  {format_duration(baseline_gc['gc_time'])}")
        print(f"  Optimized GC time: {format_duration(optimized_gc['gc_time'])}")
        print(f"  Improvement:       {improvement:.1f}% less GC time")
        print()
    else:
        print("Benchmark 4: Garbage Collection Pressure [SKIPPED]")
        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    if results.memory_savings_percent > 0:
        print(f"  Memory:   {results.memory_savings_percent:.1f}% reduction")
    if results.creation_speedup > 1.0:
        print(f"  Creation: {results.creation_speedup:.2f}x faster")
    if results.access_speedup > 1.0:
        print(f"  Access:   {results.access_speedup:.2f}x faster")
    if results.gc_improvement_percent > 0:
        print(f"  GC:       {results.gc_improvement_percent:.1f}% improvement")

    print("=" * 70)

    # Save results to JSON if requested
    if args.output_json:
        output_dict = dataclasses.asdict(results)
        with open(args.output_json, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    return results


def main(args: argparse.Namespace):
    """Main entry point."""
    run_benchmarks(args)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark __slots__ optimization for vLLM v1"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10000,
        help="Number of Request objects to create for benchmarking",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=10000,
        help="Number of KVCacheBlock objects to create for benchmarking",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=50000,
        help="Number of iterations for creation/destruction benchmark",
    )
    parser.add_argument(
        "--num-accesses",
        type=int,
        default=1_000_000,
        help="Number of attribute access operations to perform",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save benchmark results as JSON",
    )
    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help="Skip memory usage benchmark",
    )
    parser.add_argument(
        "--skip-creation",
        action="store_true",
        help="Skip object creation benchmark",
    )
    parser.add_argument(
        "--skip-access",
        action="store_true",
        help="Skip attribute access benchmark",
    )
    parser.add_argument(
        "--skip-gc",
        action="store_true",
        help="Skip garbage collection benchmark",
    )

    args = parser.parse_args()
    main(args)
