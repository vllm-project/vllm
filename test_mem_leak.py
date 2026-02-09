"""
Minimal reproduction script for CPU memory leak with prefix caching.
Tests whether CPU RSS memory grows unboundedly during inference with
prefix caching enabled vs disabled.
"""
import gc
import os
import resource
import tracemalloc

# Use a single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams


def get_rss_mb():
    """Get current RSS memory in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def get_tracemalloc_mb():
    """Get current traced memory in MB."""
    current, peak = tracemalloc.get_traced_memory()
    return current / (1024 * 1024), peak / (1024 * 1024)


def run_test(enable_prefix_caching: bool, num_rounds: int = 5,
             prompts_per_round: int = 200):
    """Run inference test and monitor memory."""
    print(f"\n{'='*60}")
    print(f"Testing with prefix_caching={'enabled' if enable_prefix_caching else 'disabled'}")
    print(f"{'='*60}")

    tracemalloc.start()

    # Use a small model
    llm = LLM(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=enable_prefix_caching,
        limit_mm_per_prompt={"video": 0},
        enforce_eager=True,
    )

    sampling_params = SamplingParams(temperature=0.8, max_tokens=64)

    rss_before = get_rss_mb()
    traced_before = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
    print(f"Initial RSS: {rss_before:.1f} MB, Traced: {traced_before:.1f} MB")

    for round_idx in range(num_rounds):
        # Create prompts that share common prefixes (to exercise prefix caching)
        # Use varying suffixes to create different block hashes
        common_prefix = "You are a helpful assistant. " * 50
        prompts = [
            f"{common_prefix} Question {round_idx * prompts_per_round + i}: "
            f"What is {i * 7 + round_idx}? Answer briefly."
            for i in range(prompts_per_round)
        ]

        outputs = llm.generate(prompts, sampling_params)

        # Force garbage collection
        del outputs
        gc.collect()

        rss_after = get_rss_mb()
        traced_current, traced_peak = get_tracemalloc_mb()
        print(
            f"Round {round_idx + 1}: RSS={rss_after:.1f} MB "
            f"(delta={rss_after - rss_before:+.1f} MB), "
            f"Traced={traced_current:.1f} MB (peak={traced_peak:.1f} MB)"
        )

    # Take a snapshot of the top memory consumers
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print(f"\nTop 15 memory consumers:")
    for stat in top_stats[:15]:
        print(f"  {stat}")

    # Also check by filename
    top_stats_file = snapshot.statistics('filename')
    print(f"\nTop 10 memory consumers by file:")
    for stat in top_stats_file[:10]:
        print(f"  {stat}")

    tracemalloc.stop()

    final_rss = get_rss_mb()
    print(f"\nFinal RSS: {final_rss:.1f} MB (total delta: {final_rss - rss_before:+.1f} MB)")

    del llm
    gc.collect()

    return final_rss - rss_before


if __name__ == "__main__":
    # Test with prefix caching enabled
    delta_with_caching = run_test(enable_prefix_caching=True)

    gc.collect()

    # Test with prefix caching disabled
    delta_without_caching = run_test(enable_prefix_caching=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"RSS growth with prefix caching:    {delta_with_caching:+.1f} MB")
    print(f"RSS growth without prefix caching: {delta_without_caching:+.1f} MB")
    print(f"Difference: {delta_with_caching - delta_without_caching:+.1f} MB")
