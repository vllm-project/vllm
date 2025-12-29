# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Video inference benchmark for Hybrid Attention VL models.

This script benchmarks video inference using Qwen2.5-VL with both standard
and hybrid attention backends. It uses the baby_reading video asset and
measures throughput, latency, and memory usage.

Usage:
    # Standard Qwen2.5-VL benchmark
    python benchmarks/video_benchmark.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --num-frames 16 \
        --num-iterations 5

    # Hybrid attention benchmark
    python benchmarks/video_benchmark.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --use-hybrid-attention \
        --num-frames 16 \
        --num-iterations 5

    # Compare both modes
    python benchmarks/video_benchmark.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --compare-modes \
        --num-frames 16 \
        --output-file video_benchmark_results.json

Requirements:
    pip install opencv-python
"""

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# Note: We avoid importing torch at module level to prevent CUDA initialization
# issues with vLLM's multiprocessing. Torch is imported lazily where needed.


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: str
    model: str
    num_frames: int
    num_iterations: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_latency_seconds: float = 0.0
    min_latency_seconds: float = 0.0
    max_latency_seconds: float = 0.0
    p50_latency_seconds: float = 0.0
    p90_latency_seconds: float = 0.0
    p99_latency_seconds: float = 0.0
    throughput_tokens_per_second: float = 0.0
    generation_tokens_per_second: float = 0.0
    all_latencies: list[float] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def get_gpu_memory_info() -> dict[str, float]:
    """Get GPU memory usage information."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}

        device = torch.cuda.current_device()
        free_memory = torch.cuda.mem_get_info(device)[0]
        total_memory = torch.cuda.mem_get_info(device)[1]
        used_memory = total_memory - free_memory

        return {
            "available": True,
            "free_memory_gib": free_memory / (1024**3),
            "total_memory_gib": total_memory / (1024**3),
            "used_memory_gib": used_memory / (1024**3),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def load_video_frames(num_frames: int = 16) -> np.ndarray:
    """Load frames from the baby_reading video asset.

    Args:
        num_frames: Number of frames to sample from the video.

    Returns:
        Numpy array of video frames with shape (num_frames, H, W, C).
    """
    from vllm.assets.video import VideoAsset
    from vllm.multimodal.video import sample_frames_from_video

    print(f"Loading baby_reading video asset...")
    video_asset = VideoAsset(name="baby_reading")
    print(f"Video path: {video_asset.video_path}")

    # Get raw frames
    raw_frames = video_asset.np_ndarrays
    print(f"Raw video has {len(raw_frames)} frames")

    # Sample frames uniformly
    frames = sample_frames_from_video(raw_frames, num_frames)
    print(f"Sampled {len(frames)} frames for inference")

    return frames


def build_qwen25_vl_prompt(num_frames: int, question: str) -> str:
    """Build the prompt for Qwen2.5-VL video inference.

    Args:
        num_frames: Number of frames (not used directly but kept for API consistency).
        question: The question to ask about the video.

    Returns:
        Formatted prompt string.
    """
    video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
    prompt = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{video_placeholder}{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def run_benchmark(
    model: str,
    num_frames: int,
    num_iterations: int,
    num_warmup: int,
    max_tokens: int,
    question: str,
    use_hybrid_attention: bool = False,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
) -> BenchmarkResult:
    """Run video inference benchmark.

    Args:
        model: Model name or path.
        num_frames: Number of video frames to use.
        num_iterations: Number of benchmark iterations.
        num_warmup: Number of warmup iterations.
        max_tokens: Maximum tokens to generate.
        question: Question to ask about the video.
        use_hybrid_attention: Whether to use hybrid attention.
        gpu_memory_utilization: GPU memory utilization fraction.
        tensor_parallel_size: Tensor parallel size.
        max_model_len: Maximum model context length.

    Returns:
        BenchmarkResult with metrics.
    """
    config_name = "hybrid_attention" if use_hybrid_attention else "standard_attention"

    result = BenchmarkResult(
        config=config_name,
        model=model,
        num_frames=num_frames,
        num_iterations=num_iterations,
    )

    try:
        from vllm import LLM, SamplingParams

        # Load video frames
        frames = load_video_frames(num_frames)

        # Build prompt
        prompt = build_qwen25_vl_prompt(num_frames, question)

        # NOTE: Do NOT call get_gpu_memory_info() here - it initializes CUDA
        # in the parent process, which causes "CUDA device busy" errors when
        # vLLM spawns its subprocess. We'll capture memory after LLM init.

        # Initialize LLM
        print(f"\nInitializing LLM with {config_name}...")
        init_start = time.perf_counter()

        # Build engine kwargs
        engine_kwargs = {
            "model": model,
            "max_model_len": max_model_len or 4096,
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

        # For hybrid attention, we need to use the hybrid model
        if use_hybrid_attention:
            # Override HuggingFace config to enable hybrid attention
            engine_kwargs["hf_overrides"] = {
                "use_hybrid_attention": True,
            }
            print("  Hybrid attention enabled via hf_overrides")

        llm = LLM(**engine_kwargs)
        init_time = time.perf_counter() - init_start
        print(f"  Initialization time: {init_time:.2f}s")

        # Now safe to check GPU memory - LLM subprocess is already running
        result.memory["post_init"] = get_gpu_memory_info()
        result.memory["baseline"] = result.memory["post_init"]  # Best we can do

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic for benchmarking
            max_tokens=max_tokens,
            top_p=1.0,
        )

        # Warmup
        print(f"\nRunning {num_warmup} warmup iterations...")
        for i in range(num_warmup):
            _ = llm.generate(
                {"prompt": prompt, "multi_modal_data": {"video": frames}},
                sampling_params=sampling_params,
            )
            print(f"  Warmup {i + 1}/{num_warmup} complete")

        # Clear any cached memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        # Benchmark iterations
        print(f"\nRunning {num_iterations} benchmark iterations...")
        latencies = []
        total_input = 0
        total_output = 0

        for i in range(num_iterations):
            start_time = time.perf_counter()
            outputs = llm.generate(
                {"prompt": prompt, "multi_modal_data": {"video": frames}},
                sampling_params=sampling_params,
            )
            end_time = time.perf_counter()

            latency = end_time - start_time
            latencies.append(latency)

            # Count tokens
            for output in outputs:
                total_input += len(output.prompt_token_ids)
                total_output += len(output.outputs[0].token_ids)

            print(f"  Iteration {i + 1}/{num_iterations}: {latency:.3f}s")

        # Calculate statistics
        latencies_arr = np.array(latencies)
        total_time = sum(latencies)

        result.all_latencies = latencies
        result.total_input_tokens = total_input
        result.total_output_tokens = total_output
        result.avg_latency_seconds = float(np.mean(latencies_arr))
        result.min_latency_seconds = float(np.min(latencies_arr))
        result.max_latency_seconds = float(np.max(latencies_arr))
        result.p50_latency_seconds = float(np.percentile(latencies_arr, 50))
        result.p90_latency_seconds = float(np.percentile(latencies_arr, 90))
        result.p99_latency_seconds = float(np.percentile(latencies_arr, 99))
        result.throughput_tokens_per_second = (total_input + total_output) / total_time
        result.generation_tokens_per_second = total_output / total_time

        result.memory["post_benchmark"] = get_gpu_memory_info()

        # Print summary
        print(f"\n{'=' * 50}")
        print(f"Benchmark Results ({config_name})")
        print(f"{'=' * 50}")
        print(f"Model: {model}")
        print(f"Num frames: {num_frames}")
        print(f"Iterations: {num_iterations}")
        print(f"Total input tokens: {total_input}")
        print(f"Total output tokens: {total_output}")
        print(f"Avg latency: {result.avg_latency_seconds * 1000:.1f}ms")
        print(f"P50 latency: {result.p50_latency_seconds * 1000:.1f}ms")
        print(f"P90 latency: {result.p90_latency_seconds * 1000:.1f}ms")
        print(f"P99 latency: {result.p99_latency_seconds * 1000:.1f}ms")
        print(f"Throughput: {result.throughput_tokens_per_second:.1f} tokens/s")
        print(f"Generation: {result.generation_tokens_per_second:.1f} tokens/s")
        print(f"{'=' * 50}")

        # Cleanup
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


def main(args: argparse.Namespace) -> None:
    """Main benchmark entry point."""
    results = []

    if args.compare_modes:
        # Run both standard and hybrid benchmarks
        print("\n" + "=" * 60)
        print("Running STANDARD attention benchmark")
        print("=" * 60)
        standard_result = run_benchmark(
            model=args.model,
            num_frames=args.num_frames,
            num_iterations=args.num_iterations,
            num_warmup=args.num_warmup,
            max_tokens=args.max_tokens,
            question=args.question,
            use_hybrid_attention=False,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
        results.append(asdict(standard_result))

        print("\n" + "=" * 60)
        print("Running HYBRID attention benchmark")
        print("=" * 60)
        hybrid_result = run_benchmark(
            model=args.model,
            num_frames=args.num_frames,
            num_iterations=args.num_iterations,
            num_warmup=args.num_warmup,
            max_tokens=args.max_tokens,
            question=args.question,
            use_hybrid_attention=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
        results.append(asdict(hybrid_result))

        # Print comparison
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        if standard_result.error is None and hybrid_result.error is None:
            speedup = (
                standard_result.avg_latency_seconds
                / hybrid_result.avg_latency_seconds
            )
            print(f"Standard avg latency: {standard_result.avg_latency_seconds * 1000:.1f}ms")
            print(f"Hybrid avg latency:   {hybrid_result.avg_latency_seconds * 1000:.1f}ms")
            print(f"Speedup: {speedup:.2f}x")
            print(f"\nStandard throughput: {standard_result.throughput_tokens_per_second:.1f} tok/s")
            print(f"Hybrid throughput:   {hybrid_result.throughput_tokens_per_second:.1f} tok/s")
        else:
            if standard_result.error:
                print(f"Standard benchmark failed: {standard_result.error}")
            if hybrid_result.error:
                print(f"Hybrid benchmark failed: {hybrid_result.error}")

    else:
        # Run single benchmark
        result = run_benchmark(
            model=args.model,
            num_frames=args.num_frames,
            num_iterations=args.num_iterations,
            num_warmup=args.num_warmup,
            max_tokens=args.max_tokens,
            question=args.question,
            use_hybrid_attention=args.use_hybrid_attention,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
        results.append(asdict(result))

    # Save results
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output_file, "w") as f:
            json.dump(
                {
                    "benchmark_type": "video_inference",
                    "model": args.model,
                    "num_frames": args.num_frames,
                    "question": args.question,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to: {args.output_file}")


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the video benchmark."""
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to sample from the video",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=2,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Describe what is happening in this video in detail.",
        help="Question to ask about the video",
    )
    parser.add_argument(
        "--use-hybrid-attention",
        action="store_true",
        help="Use hybrid attention (SSM + sliding window)",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video inference benchmark for Hybrid Attention VL models"
    )
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)

