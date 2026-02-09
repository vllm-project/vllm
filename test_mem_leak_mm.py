"""
Reproduction script for CPU memory leak with prefix caching + multimodal inputs.
Tests whether CPU RSS memory grows unboundedly during VLM inference.
"""
import gc
import os
import resource
import tracemalloc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams


def get_rss_mb():
    """Get current RSS memory in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_test(enable_prefix_caching: bool, num_rounds: int = 5,
             prompts_per_round: int = 100):
    """Run inference test with multimodal inputs and monitor memory."""
    print(f"\n{'='*60}")
    print(f"Testing with prefix_caching={'enabled' if enable_prefix_caching else 'disabled'}")
    print(f"{'='*60}")

    tracemalloc.start()

    llm = LLM(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=enable_prefix_caching,
        limit_mm_per_prompt={"image": 2, "video": 0},
        enforce_eager=True,
    )

    sampling_params = SamplingParams(temperature=0.8, max_tokens=32)

    rss_before = get_rss_mb()
    traced_before = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
    print(f"Initial RSS: {rss_before:.1f} MB, Traced: {traced_before:.1f} MB")

    # Use different image URLs to simulate variety in multimodal inputs
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.png",
    ]

    for round_idx in range(num_rounds):
        # Create multimodal prompts with images
        prompts = []
        for i in range(prompts_per_round):
            img_url = image_urls[i % len(image_urls)]
            prompt = {
                "prompt": f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image briefly. Question {round_idx * prompts_per_round + i}.<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {
                    "image": img_url,
                },
            }
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)

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
    print(f"\nTop 20 memory consumers:")
    for stat in top_stats[:20]:
        print(f"  {stat}")

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


def get_tracemalloc_mb():
    current, peak = tracemalloc.get_traced_memory()
    return current / (1024 * 1024), peak / (1024 * 1024)


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
