# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams

# Larger set of text prompts for batch testing
PROMPTS = [
    "The future of artificial intelligence is",
    "The quick brown fox jumps over",
    "To be or not to be, that is",
    "In a galaxy far, far away",
    "The capital of France is",
    "Python is a programming language that",
    "The weather today is",
    "I sat down at the table and",
    "The best way to learn cooking is",
    "Once upon a time in a land",
    "The speed of light is approximately",
    "Machine learning differs from traditional programming because",
    "The Roman Empire fell due to",
    "Quantum computing will revolutionize",
    "The deepest part of the ocean is",
    "Renewable energy sources include",
    "The human brain has approximately",
    "Climate change is primarily caused by",
    "The invention of the printing press",
    "Space exploration began when",
]


def main():
    # Configuration
    model_dir = "meta-llama/Llama-3.1-8B-Instruct"
    eagle_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
    num_spec_tokens = 5
    acceptance_rate_threshold = 0.4

    # Max concurrent sequences: controls how many requests are batched together
    # vLLM's continuous batching will process all prompts, but with at most
    # max_num_seqs running concurrently at any time.
    max_num_seqs = 10  # Try 1, 2, 5, 10, 20 to test different batch sizes

    print("Running Dynamic Speculative Decoding with:")
    print(f"  Target Model: {model_dir}")
    print(f"  Eagle Model: {eagle_dir}")
    print(f"  Max Speculative Tokens (k): {num_spec_tokens}")
    print(f"  Acceptance Rate Threshold: {acceptance_rate_threshold}")
    print(f"  Max Concurrent Seqs: {max_num_seqs}")
    print(f"  Total Prompts: {len(PROMPTS)}")
    print("-" * 50)

    # Configure Speculative Decoding
    speculative_config = {
        "method": "eagle_dynamic",
        "model": eagle_dir,
        "num_speculative_tokens": num_spec_tokens,
        "acceptance_rate_threshold": acceptance_rate_threshold,
    }

    # Initialize LLM with max_num_seqs to control concurrent batch size
    llm = LLM(
        model=model_dir,
        speculative_config=speculative_config,
        max_model_len=4096,
        max_num_seqs=max_num_seqs,  # Controls concurrent batch size
        limit_mm_per_prompt={},
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

    # Generate all prompts (vLLM handles batching internally)
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)

    # Print Outputs
    for i, output in enumerate(outputs):
        print(f"[{i}] Prompt: {output.prompt!r}")
        print(f"    Generated: {output.outputs[0].text!r}")
        print("-" * 30)

    # Collect and Print Metrics
    metrics = llm.get_metrics()

    num_drafts = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * num_spec_tokens

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            for pos in range(len(metric.values)):
                if pos < len(acceptance_counts):
                    acceptance_counts[pos] += metric.values[pos]

    print("\nMetrics Summary:")
    print(f"Max Concurrent Seqs: {max_num_seqs}")
    print(f"Total Prompts Processed: {len(PROMPTS)}")
    print(f"Total Draft Steps: {num_drafts}")
    print(f"Total Accepted Tokens: {num_accepted_tokens}")

    if num_drafts > 0:
        mean_acceptance = 1 + (num_accepted_tokens / num_drafts)
        print(f"Mean Acceptance Length: {mean_acceptance:.2f}")

        print("\nAcceptance Rate per Speculative Token Position:")
        for i, count in enumerate(acceptance_counts):
            rate = count / num_drafts
            print(f"  k={i + 1}: {rate:.2%}")
    else:
        print("No draft steps recorded.")


if __name__ == "__main__":
    main()
