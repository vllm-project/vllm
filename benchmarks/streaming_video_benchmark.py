# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming Video Benchmark for SSM + Sliding Window Attention.

This benchmark demonstrates the memory efficiency of SSM-based video processing
compared to standard full attention. Key metrics:

1. Memory scaling: O(1) for SSM vs O(n) for full attention
2. Concurrent query throughput with shared SSM state
3. Long video processing capability

Usage:
    # Basic streaming benchmark with SSM
    python benchmarks/streaming_video_benchmark.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --num-frames 64 \
        --use-hybrid-attention

    # Compare standard vs hybrid with memory tracking
    python benchmarks/streaming_video_benchmark.py \
        --compare-modes \
        --num-frames 32 \
        --concurrent-queries 5

    # Stress test with long video
    python benchmarks/streaming_video_benchmark.py \
        --scenario long-video \
        --num-frames 128 \
        --use-hybrid-attention

Requirements:
    pip install opencv-python
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from benchmarks.streaming_video_context import (
    StreamingVideoContext,
    get_gpu_memory_info,
)


@dataclass
class StreamingBenchmarkResult:
    """Results from a streaming video benchmark run."""

    scenario: str
    config: str
    model: str
    num_frames: int
    concurrent_queries: int
    
    # Frame processing metrics
    total_frame_time_seconds: float = 0.0
    avg_frame_time_ms: float = 0.0
    frame_times_ms: list[float] = field(default_factory=list)
    
    # Query metrics
    total_queries: int = 0
    avg_query_latency_seconds: float = 0.0
    query_latencies_seconds: list[float] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    queries_per_second: float = 0.0
    
    # Memory metrics
    memory_samples: list[dict[str, float]] = field(default_factory=list)
    memory_growth_rate_gib_per_frame: float = 0.0
    peak_memory_gib: float = 0.0
    initial_memory_gib: float = 0.0
    final_memory_gib: float = 0.0
    
    # Comparison metrics (for SSM vs standard)
    memory_savings_percent: float | None = None
    speedup_factor: float | None = None
    
    error: str | None = None


@dataclass
class ScenarioConfig:
    """Configuration for a benchmark scenario."""

    name: str
    description: str
    num_frames: int
    concurrent_queries: int
    questions: list[str]
    frame_batch_size: int = 1
    query_interval_frames: int = 0  # 0 = query only at end


# Predefined scenarios
SCENARIOS = {
    "single-query": ScenarioConfig(
        name="single-query",
        description="Process video incrementally, single query at end",
        num_frames=32,
        concurrent_queries=1,
        questions=["Describe what happened in this video in detail."],
        frame_batch_size=4,
        query_interval_frames=0,
    ),
    "multi-query": ScenarioConfig(
        name="multi-query",
        description="Multiple concurrent queries sharing SSM state",
        num_frames=32,
        concurrent_queries=5,
        questions=[
            "What is happening at the beginning of the video?",
            "Describe the main activity in this video.",
            "What objects are visible in the video?",
            "How many people are in this video?",
            "What is the setting or location of this video?",
        ],
        frame_batch_size=4,
        query_interval_frames=0,
    ),
    "continuous-query": ScenarioConfig(
        name="continuous-query",
        description="Query at regular intervals as video progresses",
        num_frames=48,
        concurrent_queries=1,
        questions=[
            "What just happened?",
            "Describe the current scene.",
            "What changed since the last frame?",
        ],
        frame_batch_size=1,
        query_interval_frames=8,  # Query every 8 frames
    ),
    "long-video": ScenarioConfig(
        name="long-video",
        description="Stress test with long video to show memory efficiency",
        num_frames=128,
        concurrent_queries=3,
        questions=[
            "Summarize what happened throughout the entire video.",
            "What was the most significant event in this video?",
            "Describe the progression of events from start to finish.",
        ],
        frame_batch_size=8,
        query_interval_frames=0,
    ),
    "memory-scaling": ScenarioConfig(
        name="memory-scaling",
        description="Track memory at every frame to measure scaling behavior",
        num_frames=64,
        concurrent_queries=1,
        questions=["Describe the video."],
        frame_batch_size=1,  # Process one at a time for accurate memory tracking
        query_interval_frames=0,
    ),
}


def load_video_frames(num_frames: int = 16, verbose: bool = True) -> np.ndarray:
    """Load frames from the baby_reading video asset.

    If the video has fewer frames than requested, frames will be repeated
    to simulate a longer video.

    Args:
        num_frames: Number of frames to return.
        verbose: Whether to print loading progress.

    Returns:
        Numpy array of video frames with shape (num_frames, H, W, C).
    """
    from vllm.assets.video import VideoAsset
    from vllm.multimodal.video import sample_frames_from_video

    if verbose:
        print(f"Loading baby_reading video asset...")
    
    video_asset = VideoAsset(name="baby_reading")
    raw_frames = video_asset.np_ndarrays
    
    if verbose:
        print(f"Raw video has {len(raw_frames)} frames")

    if len(raw_frames) >= num_frames:
        # Sample uniformly if we have enough frames
        frames = sample_frames_from_video(raw_frames, num_frames)
    else:
        # Repeat frames to simulate longer video
        repeat_factor = (num_frames // len(raw_frames)) + 1
        extended_frames = np.tile(raw_frames, (repeat_factor, 1, 1, 1))
        frames = sample_frames_from_video(extended_frames[:num_frames * 2], num_frames)

    if verbose:
        print(f"Prepared {len(frames)} frames for benchmark")

    return frames


def create_llm(
    model: str,
    use_hybrid_attention: bool = False,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
):
    """Create and configure the LLM instance.

    Args:
        model: Model name or path.
        use_hybrid_attention: Whether to enable hybrid SSM + attention.
        gpu_memory_utilization: GPU memory fraction to use.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        max_model_len: Maximum context length.

    Returns:
        Configured LLM instance.
    """
    from vllm import LLM

    config_name = "hybrid_attention" if use_hybrid_attention else "standard_attention"
    print(f"\nInitializing LLM with {config_name}...")

    engine_kwargs = {
        "model": model,
        "max_model_len": max_model_len or 8192,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "limit_mm_per_prompt": {"video": 1},
        "mm_processor_kwargs": {
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        "trust_remote_code": True,
    }

    if use_hybrid_attention:
        engine_kwargs["hf_overrides"] = {
            "use_hybrid_attention": True,
        }
        print("  Hybrid attention enabled via hf_overrides")

    init_start = time.perf_counter()
    llm = LLM(**engine_kwargs)
    init_time = time.perf_counter() - init_start
    print(f"  Initialization time: {init_time:.2f}s")

    return llm


def run_streaming_benchmark(
    model: str,
    scenario: ScenarioConfig,
    use_hybrid_attention: bool = False,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    max_tokens: int = 128,
    num_warmup: int = 1,
    verbose: bool = True,
) -> StreamingBenchmarkResult:
    """Run a streaming video benchmark scenario.

    Args:
        model: Model name or path.
        scenario: Scenario configuration.
        use_hybrid_attention: Whether to use hybrid SSM + attention.
        gpu_memory_utilization: GPU memory fraction.
        tensor_parallel_size: Tensor parallel size.
        max_model_len: Maximum model context length.
        max_tokens: Maximum tokens to generate per query.
        num_warmup: Number of warmup iterations.
        verbose: Whether to print progress.

    Returns:
        StreamingBenchmarkResult with all metrics.
    """
    config_name = "hybrid_attention" if use_hybrid_attention else "standard_attention"

    result = StreamingBenchmarkResult(
        scenario=scenario.name,
        config=config_name,
        model=model,
        num_frames=scenario.num_frames,
        concurrent_queries=scenario.concurrent_queries,
    )

    try:
        # Load video frames
        frames = load_video_frames(scenario.num_frames, verbose=verbose)

        # Create LLM
        llm = create_llm(
            model=model,
            use_hybrid_attention=use_hybrid_attention,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )

        # Record initial memory
        result.initial_memory_gib = get_gpu_memory_info().get("used_memory_gib", 0)

        # Create streaming context
        ctx = StreamingVideoContext(
            stream_id=f"bench_{scenario.name}",
            llm=llm,
            use_hybrid_attention=use_hybrid_attention,
            frame_batch_size=scenario.frame_batch_size,
        )

        # Warmup
        if num_warmup > 0 and verbose:
            print(f"\nRunning {num_warmup} warmup iterations...")
            warmup_frames = frames[:min(4, len(frames))]
            for i in range(num_warmup):
                ctx.add_frames(warmup_frames)
                ctx.query(scenario.questions[0], max_tokens=32)
                ctx.clear()
                print(f"  Warmup {i + 1}/{num_warmup} complete")

        # Main benchmark: process frames incrementally
        if verbose:
            print(f"\nProcessing {scenario.num_frames} frames...")

        frame_times = []
        query_results = []
        memory_samples = []
        total_frame_start = time.perf_counter()

        for i, frame in enumerate(frames):
            # Process frame
            frame_start = time.perf_counter()
            metadata = ctx.add_frame(frame, process_immediately=True)
            frame_time = (time.perf_counter() - frame_start) * 1000
            frame_times.append(frame_time)

            # Record memory
            memory_info = get_gpu_memory_info()
            if memory_info.get("available"):
                memory_samples.append({
                    "frame_idx": i,
                    "used_memory_gib": memory_info.get("used_memory_gib", 0),
                })

            # Query at intervals if configured
            if (
                scenario.query_interval_frames > 0
                and (i + 1) % scenario.query_interval_frames == 0
            ):
                question = scenario.questions[i % len(scenario.questions)]
                query_result = ctx.query(question, max_tokens=max_tokens)
                query_results.append(query_result)
                if verbose:
                    print(f"  Frame {i + 1}: query latency {query_result.latency_seconds:.3f}s")
            elif verbose and (i + 1) % 10 == 0:
                print(f"  Processed frame {i + 1}/{scenario.num_frames}")

        total_frame_time = time.perf_counter() - total_frame_start

        # Run concurrent queries at end if no interval queries
        if scenario.query_interval_frames == 0:
            if verbose:
                print(f"\nRunning {scenario.concurrent_queries} concurrent queries...")

            query_start = time.perf_counter()
            questions = scenario.questions[:scenario.concurrent_queries]

            # Run queries (simulated concurrency via sequential for simplicity)
            for q_idx, question in enumerate(questions):
                query_result = ctx.query(question, max_tokens=max_tokens)
                query_results.append(query_result)
                if verbose:
                    print(f"  Query {q_idx + 1}: {query_result.latency_seconds:.3f}s")

            total_query_time = time.perf_counter() - query_start
        else:
            total_query_time = sum(q.latency_seconds for q in query_results)

        # Calculate results
        result.frame_times_ms = frame_times
        result.total_frame_time_seconds = total_frame_time
        result.avg_frame_time_ms = float(np.mean(frame_times)) if frame_times else 0

        result.query_latencies_seconds = [q.latency_seconds for q in query_results]
        result.total_queries = len(query_results)
        result.avg_query_latency_seconds = (
            float(np.mean(result.query_latencies_seconds))
            if result.query_latencies_seconds
            else 0
        )
        result.total_input_tokens = sum(q.input_tokens for q in query_results)
        result.total_output_tokens = sum(q.output_tokens for q in query_results)
        result.queries_per_second = (
            len(query_results) / total_query_time if total_query_time > 0 else 0
        )

        # Memory metrics
        result.memory_samples = memory_samples
        if memory_samples:
            memory_values = [s["used_memory_gib"] for s in memory_samples]
            result.peak_memory_gib = max(memory_values)
            result.final_memory_gib = memory_values[-1]

            # Calculate memory growth rate
            if len(memory_values) >= 2:
                x = np.arange(len(memory_values))
                y = np.array(memory_values)
                slope = np.polyfit(x, y, 1)[0]
                result.memory_growth_rate_gib_per_frame = float(slope)

        # Get stats from context
        stats = ctx.get_stats()

        # Print summary
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Streaming Benchmark Results ({config_name})")
            print(f"{'=' * 60}")
            print(f"Scenario: {scenario.name}")
            print(f"Model: {model}")
            print(f"Frames: {scenario.num_frames}")
            print(f"Concurrent queries: {scenario.concurrent_queries}")
            print(f"\n--- Frame Processing ---")
            print(f"Total frame time: {total_frame_time:.2f}s")
            print(f"Avg frame time: {result.avg_frame_time_ms:.1f}ms")
            print(f"\n--- Query Performance ---")
            print(f"Total queries: {result.total_queries}")
            print(f"Avg query latency: {result.avg_query_latency_seconds * 1000:.1f}ms")
            print(f"Queries/second: {result.queries_per_second:.2f}")
            print(f"\n--- Memory ---")
            print(f"Initial memory: {result.initial_memory_gib:.2f} GiB")
            print(f"Peak memory: {result.peak_memory_gib:.2f} GiB")
            print(f"Final memory: {result.final_memory_gib:.2f} GiB")
            print(f"Growth rate: {result.memory_growth_rate_gib_per_frame * 1000:.3f} MiB/frame")
            print(f"{'=' * 60}")

        # Cleanup
        ctx.clear()
        del llm
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    except Exception as e:
        result.error = str(e)
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

    return result


def run_comparison_benchmark(
    model: str,
    scenario: ScenarioConfig,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    max_tokens: int = 128,
    num_warmup: int = 1,
    verbose: bool = True,
) -> tuple[StreamingBenchmarkResult, StreamingBenchmarkResult]:
    """Run comparison between standard and hybrid attention.

    Args:
        model: Model name or path.
        scenario: Scenario configuration.
        gpu_memory_utilization: GPU memory fraction.
        tensor_parallel_size: Tensor parallel size.
        max_model_len: Maximum model context length.
        max_tokens: Maximum tokens to generate.
        num_warmup: Number of warmup iterations.
        verbose: Whether to print progress.

    Returns:
        Tuple of (standard_result, hybrid_result).
    """
    print("\n" + "=" * 70)
    print("RUNNING STANDARD ATTENTION BENCHMARK")
    print("=" * 70)

    standard_result = run_streaming_benchmark(
        model=model,
        scenario=scenario,
        use_hybrid_attention=False,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        num_warmup=num_warmup,
        verbose=verbose,
    )

    # Force cleanup between runs
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    time.sleep(2)  # Allow GPU to settle

    print("\n" + "=" * 70)
    print("RUNNING HYBRID ATTENTION BENCHMARK")
    print("=" * 70)

    hybrid_result = run_streaming_benchmark(
        model=model,
        scenario=scenario,
        use_hybrid_attention=True,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        num_warmup=num_warmup,
        verbose=verbose,
    )

    # Calculate comparison metrics
    if standard_result.error is None and hybrid_result.error is None:
        # Memory savings
        if standard_result.peak_memory_gib > 0:
            memory_saved = standard_result.peak_memory_gib - hybrid_result.peak_memory_gib
            hybrid_result.memory_savings_percent = (
                memory_saved / standard_result.peak_memory_gib * 100
            )

        # Speedup
        if standard_result.avg_query_latency_seconds > 0:
            hybrid_result.speedup_factor = (
                standard_result.avg_query_latency_seconds
                / hybrid_result.avg_query_latency_seconds
            )

        # Print comparison summary
        if verbose:
            print("\n" + "=" * 70)
            print("COMPARISON SUMMARY")
            print("=" * 70)
            print(f"\n{'Metric':<35} {'Standard':<15} {'Hybrid':<15} {'Delta':<15}")
            print("-" * 80)

            # Memory comparison
            std_mem = standard_result.peak_memory_gib
            hyb_mem = hybrid_result.peak_memory_gib
            mem_delta = ((hyb_mem - std_mem) / std_mem * 100) if std_mem > 0 else 0
            print(f"{'Peak Memory (GiB)':<35} {std_mem:>12.2f}   {hyb_mem:>12.2f}   {mem_delta:>+12.1f}%")

            # Memory growth
            std_growth = standard_result.memory_growth_rate_gib_per_frame * 1000
            hyb_growth = hybrid_result.memory_growth_rate_gib_per_frame * 1000
            print(f"{'Memory Growth (MiB/frame)':<35} {std_growth:>12.3f}   {hyb_growth:>12.3f}")

            # Latency comparison
            std_lat = standard_result.avg_query_latency_seconds * 1000
            hyb_lat = hybrid_result.avg_query_latency_seconds * 1000
            lat_delta = ((hyb_lat - std_lat) / std_lat * 100) if std_lat > 0 else 0
            print(f"{'Avg Query Latency (ms)':<35} {std_lat:>12.1f}   {hyb_lat:>12.1f}   {lat_delta:>+12.1f}%")

            # Throughput
            std_qps = standard_result.queries_per_second
            hyb_qps = hybrid_result.queries_per_second
            qps_delta = ((hyb_qps - std_qps) / std_qps * 100) if std_qps > 0 else 0
            print(f"{'Queries/second':<35} {std_qps:>12.2f}   {hyb_qps:>12.2f}   {qps_delta:>+12.1f}%")

            # Frame processing
            std_frame = standard_result.avg_frame_time_ms
            hyb_frame = hybrid_result.avg_frame_time_ms
            frame_delta = ((hyb_frame - std_frame) / std_frame * 100) if std_frame > 0 else 0
            print(f"{'Avg Frame Time (ms)':<35} {std_frame:>12.1f}   {hyb_frame:>12.1f}   {frame_delta:>+12.1f}%")

            print("\n" + "=" * 70)

            # Key insights
            if hybrid_result.memory_savings_percent and hybrid_result.memory_savings_percent > 0:
                print(f"\nKey Insight: Hybrid SSM reduces peak memory by {hybrid_result.memory_savings_percent:.1f}%")
            
            if hyb_growth < std_growth:
                print(f"Key Insight: Hybrid SSM memory grows {std_growth/hyb_growth:.1f}x slower per frame")

    return standard_result, hybrid_result


def main(args: argparse.Namespace) -> None:
    """Main entry point for streaming video benchmark."""
    # Get scenario
    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available scenarios: {list(SCENARIOS.keys())}")
            sys.exit(1)
        scenario = SCENARIOS[args.scenario]
    else:
        # Create custom scenario from args
        scenario = ScenarioConfig(
            name="custom",
            description="Custom benchmark configuration",
            num_frames=args.num_frames,
            concurrent_queries=args.concurrent_queries,
            questions=[args.question] * args.concurrent_queries,
            frame_batch_size=args.frame_batch_size,
            query_interval_frames=args.query_interval,
        )

    # Override scenario parameters if explicitly provided
    if args.num_frames:
        scenario = ScenarioConfig(
            name=scenario.name,
            description=scenario.description,
            num_frames=args.num_frames,
            concurrent_queries=scenario.concurrent_queries if not args.concurrent_queries else args.concurrent_queries,
            questions=scenario.questions,
            frame_batch_size=scenario.frame_batch_size,
            query_interval_frames=scenario.query_interval_frames,
        )

    results = []

    if args.compare_modes:
        # Run comparison benchmark
        standard_result, hybrid_result = run_comparison_benchmark(
            model=args.model,
            scenario=scenario,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            num_warmup=args.num_warmup,
            verbose=not args.quiet,
        )
        results.append(asdict(standard_result))
        results.append(asdict(hybrid_result))
    else:
        # Run single benchmark
        result = run_streaming_benchmark(
            model=args.model,
            scenario=scenario,
            use_hybrid_attention=args.use_hybrid_attention,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            num_warmup=args.num_warmup,
            verbose=not args.quiet,
        )
        results.append(asdict(result))

    # Save results
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        output_data = {
            "benchmark_type": "streaming_video",
            "model": args.model,
            "scenario": scenario.name,
            "scenario_description": scenario.description,
            "results": results,
        }

        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the streaming video benchmark."""
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=list(SCENARIOS.keys()),
        help=f"Predefined benchmark scenario. Available: {list(SCENARIOS.keys())}",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="Number of frames to process",
    )
    parser.add_argument(
        "--concurrent-queries",
        type=int,
        default=3,
        help="Number of concurrent queries at end of video",
    )
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=4,
        help="Number of frames to batch together",
    )
    parser.add_argument(
        "--query-interval",
        type=int,
        default=0,
        help="Query every N frames (0 = only at end)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Describe what is happening in this video.",
        help="Question to ask about the video",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per query",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=1,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--use-hybrid-attention",
        action="store_true",
        help="Use hybrid SSM + sliding window attention",
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Compare standard vs hybrid attention",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path for JSON results",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Streaming Video Benchmark for SSM + Sliding Window Attention"
    )
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)

