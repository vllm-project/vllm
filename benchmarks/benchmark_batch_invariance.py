#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark to measure the performance overhead of VLLM_BATCH_INVARIANT mode.

This benchmark runs the same workload twice:
1. With VLLM_BATCH_INVARIANT=0 (baseline)
2. With VLLM_BATCH_INVARIANT=1 (batch invariant mode)

And reports the timing and throughput metrics for comparison.

Environment variables:
    VLLM_BENCH_MODEL: Model to benchmark (default: "Qwen/Qwen3-1.7B")
    VLLM_BENCH_TP_SIZE: Tensor parallel size (default: 1, use 8 for deepseek)
    VLLM_BENCH_BATCH_SIZE: Max batch size (default: 128)
    VLLM_BENCH_NUM_TRIALS: Number of trials to run (default: 5)
    VLLM_BENCH_MIN_PROMPT: Min prompt length in words (default: 1024)
    VLLM_BENCH_MAX_PROMPT: Max prompt length in words (default: 2048)
    VLLM_BENCH_MAX_TOKENS: Max tokens to generate (default: 128)
    VLLM_BENCH_TEMPERATURE: Temperature for sampling (default: 0.0)
    VLLM_BENCH_GPU_MEMORY_UTILIZATION: GPU memory utilization (default: 0.4)
    VLLM_BENCH_MAX_MODEL_LEN: Max model length (default: 5120)
    VLLM_BENCH_BACKEND: Attention backend (default: FLASH_ATTN)

Example usage:
    # Benchmark qwen3 (default)
    python benchmarks/benchmark_batch_invariance.py

    # Benchmark deepseek with 8 GPUs
    VLLM_BENCH_MODEL="deepseek-ai/DeepSeek-V3" VLLM_BENCH_TP_SIZE=8 \\
        python benchmarks/benchmark_batch_invariance.py

    # Quick test with fewer trials
    VLLM_BENCH_NUM_TRIALS=2 VLLM_BENCH_BATCH_SIZE=32 \\
        python benchmarks/benchmark_batch_invariance.py
"""

import contextlib
import os
import random
import time

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform


def _random_prompt(min_words: int = 1024, max_words: int = 1024 * 2) -> str:
    """Generate a random prompt for benchmarking."""
    prompt_templates = [
        "Question: What is the capital of France?\nAnswer: The capital of France is",
        "Q: How does photosynthesis work?\nA: Photosynthesis is the process by which",
        "User: Can you explain quantum mechanics?\nAssistant: Quantum mechanics is",
        "Once upon a time in a distant galaxy, there lived",
        "The old man walked slowly down the street, remembering",
        "In the year 2157, humanity finally discovered",
        "To implement a binary search tree in Python, first we need to",
        "The algorithm works by iterating through the array and",
        "Here's how to optimize database queries using indexing:",
        "The Renaissance was a period in European history that",
        "Climate change is caused by several factors including",
        "The human brain contains approximately 86 billion neurons which",
        "I've been thinking about getting a new laptop because",
        "Yesterday I went to the store and bought",
        "My favorite thing about summer is definitely",
    ]

    base_prompt = random.choice(prompt_templates)

    if max_words < min_words:
        max_words = min_words
    target_words = random.randint(min_words, max_words)

    if target_words > 50:
        padding_text = (
            " This is an interesting topic that deserves more explanation. "
            * (target_words // 50)
        )
        base_prompt = base_prompt + padding_text

    return base_prompt


def run_benchmark_with_batch_invariant(
    model: str,
    tp_size: int,
    max_batch_size: int,
    num_trials: int,
    min_prompt: int,
    max_prompt: int,
    max_tokens: int,
    temperature: float,
    gpu_mem_util: float,
    max_model_len: int,
    backend: str,
    batch_invariant: bool,
    seed: int = 12345,
) -> dict:
    """
    Run the benchmark with the specified configuration.

    Returns a dict with timing and throughput metrics.
    """
    random.seed(seed)

    # Set environment variables
    os.environ["VLLM_ATTENTION_BACKEND"] = backend
    if batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    else:
        os.environ["VLLM_BATCH_INVARIANT"] = "0"

    print(f"\n{'=' * 80}")
    print(f"BENCHMARK: VLLM_BATCH_INVARIANT={int(batch_invariant)}")
    print(f"  Model: {model}")
    print(f"  TP Size: {tp_size}")
    print(f"  Backend: {backend}")
    print(f"  Max Batch Size: {max_batch_size}")
    print(f"  Trials: {num_trials}")
    print(f"  Max Tokens: {max_tokens}")
    print(f"{'=' * 80}\n")

    sampling = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        seed=20240919,
    )

    needle_prompt = "There once was a "

    llm = None
    try:
        # Create LLM engine
        start_init = time.perf_counter()
        llm = LLM(
            model=model,
            max_num_seqs=max_batch_size,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            dtype="bfloat16",
            tensor_parallel_size=tp_size,
            enable_prefix_caching=False,
        )
        init_time = time.perf_counter() - start_init
        print(f"Engine initialization time: {init_time:.2f}s\n")

        # Generate baseline
        print("Generating baseline (warmup)...")
        baseline_out = llm.generate([needle_prompt], sampling)
        assert len(baseline_out) == 1
        baseline_text = baseline_out[0].outputs[0].text
        print(f"Baseline output: '{baseline_text[:50]}...'\n")

        # Run trials and measure timing
        trial_times: list[float] = []
        total_tokens = 0
        total_prompts = 0

        for trial in range(num_trials):
            # Create a batch
            prompts: list[str] = []
            batch_size = random.randint(max_batch_size // 2, max_batch_size)
            needle_pos = random.randint(0, batch_size - 1)
            for i in range(batch_size):
                if i == needle_pos:
                    prompts.append(needle_prompt)
                else:
                    prompts.append(_random_prompt(min_prompt, max_prompt))

            # Measure time for this trial
            start_time = time.perf_counter()
            outputs = llm.generate(prompts, sampling)
            trial_time = time.perf_counter() - start_time

            trial_times.append(trial_time)
            total_prompts += len(prompts)

            # Count tokens
            for output in outputs:
                if output.outputs:
                    total_tokens += len(output.outputs[0].token_ids)

            print(
                f"Trial {trial + 1}/{num_trials}: "
                f"batch_size={batch_size}, "
                f"time={trial_time:.2f}s"
            )

            # Verify needle output still matches
            needle_output = outputs[needle_pos]
            assert needle_output.prompt == needle_prompt

        # Compute statistics
        avg_time = sum(trial_times) / len(trial_times)
        min_time = min(trial_times)
        max_time = max(trial_times)
        throughput = total_tokens / sum(trial_times)
        prompts_per_sec = total_prompts / sum(trial_times)

        print(f"\n{'=' * 80}")
        print("RESULTS:")
        print(f"  Average time per trial: {avg_time:.2f}s")
        print(f"  Min time: {min_time:.2f}s")
        print(f"  Max time: {max_time:.2f}s")
        print(f"  Total tokens generated: {total_tokens}")
        print(f"  Total prompts processed: {total_prompts}")
        print(f"  Throughput: {throughput:.2f} tokens/s")
        print(f"  Prompts/s: {prompts_per_sec:.2f}")
        print(f"{'=' * 80}\n")

        return {
            "init_time": init_time,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "total_tokens": total_tokens,
            "total_prompts": total_prompts,
            "throughput": throughput,
            "prompts_per_sec": prompts_per_sec,
            "trial_times": trial_times,
        }

    finally:
        # Cleanup
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()


def main():
    # Check platform support
    if not (current_platform.is_cuda() and current_platform.has_device_capability(90)):
        print("ERROR: Requires CUDA and >= Hopper (SM90)")
        print(f"Current platform: {current_platform.device_type}")
        if current_platform.is_cuda():
            print(f"Device capability: {current_platform.get_device_capability()}")
        return 1

    # Read configuration from environment
    model = os.getenv("VLLM_BENCH_MODEL", "Qwen/Qwen3-1.7B")
    tp_size = int(os.getenv("VLLM_BENCH_TP_SIZE", "1"))
    max_batch_size = int(os.getenv("VLLM_BENCH_BATCH_SIZE", "128"))
    num_trials = int(os.getenv("VLLM_BENCH_NUM_TRIALS", "5"))
    min_prompt = int(os.getenv("VLLM_BENCH_MIN_PROMPT", "1024"))
    max_prompt = int(os.getenv("VLLM_BENCH_MAX_PROMPT", "2048"))
    max_tokens = int(os.getenv("VLLM_BENCH_MAX_TOKENS", "128"))
    temperature = float(os.getenv("VLLM_BENCH_TEMPERATURE", "0.0"))
    gpu_mem_util = float(os.getenv("VLLM_BENCH_GPU_MEMORY_UTILIZATION", "0.4"))
    max_model_len = int(os.getenv("VLLM_BENCH_MAX_MODEL_LEN", "5120"))
    backend = os.getenv("VLLM_BENCH_BACKEND", "FLASH_ATTN")

    print("\n" + "=" * 80)
    print("VLLM BATCH INVARIANCE BENCHMARK")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Tensor Parallel Size: {tp_size}")
    print(f"  Attention Backend: {backend}")
    print(f"  Max Batch Size: {max_batch_size}")
    print(f"  Number of Trials: {num_trials}")
    print(f"  Prompt Length Range: {min_prompt}-{max_prompt} words")
    print(f"  Max Tokens to Generate: {max_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  GPU Memory Utilization: {gpu_mem_util}")
    print(f"  Max Model Length: {max_model_len}")
    print("=" * 80)

    # Run benchmark WITHOUT batch invariance (baseline)
    print("\n" + "=" * 80)
    print("PHASE 1: Running WITHOUT batch invariance (baseline)")
    print("=" * 80)
    baseline_results = run_benchmark_with_batch_invariant(
        model=model,
        tp_size=tp_size,
        max_batch_size=max_batch_size,
        num_trials=num_trials,
        min_prompt=min_prompt,
        max_prompt=max_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        gpu_mem_util=gpu_mem_util,
        max_model_len=max_model_len,
        backend=backend,
        batch_invariant=False,
    )

    # Run benchmark WITH batch invariance
    print("\n" + "=" * 80)
    print("PHASE 2: Running WITH batch invariance")
    print("=" * 80)
    batch_inv_results = run_benchmark_with_batch_invariant(
        model=model,
        tp_size=tp_size,
        max_batch_size=max_batch_size,
        num_trials=num_trials,
        min_prompt=min_prompt,
        max_prompt=max_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        gpu_mem_util=gpu_mem_util,
        max_model_len=max_model_len,
        backend=backend,
        batch_invariant=True,
    )

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON: Batch Invariance vs Baseline")
    print("=" * 80)

    init_overhead_pct = (
        (batch_inv_results["init_time"] - baseline_results["init_time"])
        / baseline_results["init_time"]
        * 100
    )
    time_overhead_pct = (
        (batch_inv_results["avg_time"] - baseline_results["avg_time"])
        / baseline_results["avg_time"]
        * 100
    )
    throughput_change_pct = (
        (batch_inv_results["throughput"] - baseline_results["throughput"])
        / baseline_results["throughput"]
        * 100
    )

    print("\nInitialization Time:")
    print(f"  Baseline:         {baseline_results['init_time']:.2f}s")
    print(f"  Batch Invariant:  {batch_inv_results['init_time']:.2f}s")
    print(f"  Overhead:         {init_overhead_pct:+.2f}%")

    print("\nAverage Trial Time:")
    print(f"  Baseline:         {baseline_results['avg_time']:.2f}s")
    print(f"  Batch Invariant:  {batch_inv_results['avg_time']:.2f}s")
    print(f"  Overhead:         {time_overhead_pct:+.2f}%")

    print("\nThroughput (tokens/s):")
    print(f"  Baseline:         {baseline_results['throughput']:.2f}")
    print(f"  Batch Invariant:  {batch_inv_results['throughput']:.2f}")
    print(f"  Change:           {throughput_change_pct:+.2f}%")

    print("\nPrompts/s:")
    print(f"  Baseline:         {baseline_results['prompts_per_sec']:.2f}")
    print(f"  Batch Invariant:  {batch_inv_results['prompts_per_sec']:.2f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if time_overhead_pct > 0:
        print(
            f"Batch invariance mode adds approximately {time_overhead_pct:.1f}% "
            "overhead"
        )
    else:
        print(
            f"Batch invariance mode is approximately {-time_overhead_pct:.1f}% "
            "faster (unexpected!)"
        )

    if abs(throughput_change_pct) < 1.0:
        print("Throughput difference is negligible (< 1%)")
    elif throughput_change_pct < 0:
        print(
            f"Throughput decreased by {-throughput_change_pct:.1f}% "
            "with batch invariance"
        )
    else:
        print(
            f"Throughput increased by {throughput_change_pct:.1f}% "
            "with batch invariance (unexpected!)"
        )

    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
