# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import random
import numpy as np
import torch
import sys
from vllm.inputs import TokensPrompt

#from benchmark_dataset import AIMODataset
# Set environment variables
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

from transformers import AutoTokenizer


from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector


def load_prompts(args, tokenizer):
    """Load prompts based on dataset name."""
    dataset_name = args.dataset_name

    if dataset_name == "debug":
        prompts = [
            #"The future of AI is", 
            #"The future of technology is",
            "The mission of a PhD student is",
            "Can you please repeat the following sentence: 'The future of AI is' for 3 times' ?",
            #"9 out of 10 cheerleaders are 64 tall.  The 10th cheerleader is 60 tall.  If they build a human pyramid, where 4 girls are on the bottom,  3 stand on top of the 4, 2 stand on top of the 3 and the shortest girl is at the top, how tall is the human pyramid in feet?"
        ]
        return prompts[:args.num_prompts]


    raise ValueError(f"Unknown dataset name: {dataset_name}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Self-speculative decoding inference script")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="debug",
        choices=["debug", "aimo", "cropped_aimo"],
        help="Name of the dataset to use.",
    )
    parser.add_argument("--max_num_seqs", type=int, default=64, help="Maximum number of sequences")
    parser.add_argument("--num_prompts", type=int, default=64, help="Number of prompts to process")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--enforce_eager", action="store_true", help="Enforce eager execution")
    parser.add_argument("--enable_chunked_prefill", action="store_true", help="Enable chunked prefill")
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192, help="Maximum batched tokens")
    parser.add_argument("--temp", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--enable_sspec", action="store_true", help="Enable self-speculative decoding")
    parser.add_argument("--sspec_method", type=str, default="self_specs",
                        choices=["self_specs", "self_spec_ngram"],
                        help="Self-speculative method: self_specs (baseline) or self_spec_ngram (with n-gram assistance)")
    parser.add_argument("--num_speculative_tokens", type=int, default=8,
                        help="Self-spec threshold: number of tokens to accumulate before VERIFYING (default: 8)")
    parser.add_argument("--ngram_draft_tokens", type=int, default=3,
                        help="Number of draft tokens proposed by n-gram per step (for self_spec_ngram, default: 3)")
    parser.add_argument("--prompt_lookup_max", type=int, default=None,
                        help="Max n-gram window size (for self_spec_ngram). If not set, uses ngram_draft_tokens")
    parser.add_argument("--prompt_lookup_min", type=int, default=None,
                        help="Min n-gram window size (for self_spec_ngram). If not set, uses ngram_draft_tokens")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--disable_log_stats", action="store_true", help="Disable log stats")
    parser.add_argument("--recent_ratio", type=int, default=0.05, help="Number of recent tokens to keep for sparse attention (default: 128)")
    parser.add_argument("--sink_size", type=int, default=32, help="Number of sink tokens to keep for sparse attention (default: 32)")
    parser.add_argument(
        "--input-file",
        type=str,
        help="File to read prompts from for the cropped_aimo dataset.")
    return parser.parse_args()


def get_speculative_config(args):
    """Get self-speculative decoding configuration based on arguments."""
    if args.enable_sspec:
        config = {
            "method": args.sspec_method,
            #"model": None,
            "num_speculative_tokens": args.num_speculative_tokens,  # Self-spec threshold (ACCUMULATING → VERIFYING)
        }

        # Add n-gram parameters for self_spec_ngram
        if args.sspec_method == "self_spec_ngram":
            # Set num_ngram_draft_tokens (controls draft size per step)
            config["num_ngram_draft_tokens"] = args.ngram_draft_tokens

            # Set prompt_lookup_max/min (controls n-gram window size for matching)
            # If not provided, use ngram_draft_tokens as default
            lookup_max = args.prompt_lookup_max if args.prompt_lookup_max is not None else args.ngram_draft_tokens
            lookup_min = args.prompt_lookup_min if args.prompt_lookup_min is not None else args.ngram_draft_tokens

            config["prompt_lookup_max"] = lookup_max  # N-gram window size (max)
            config["prompt_lookup_min"] = lookup_min  # N-gram window size (min)

        return config
    return None


def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    args = parse_args()

    model_dir = "Qwen/Qwen3-8B"
    max_model_len = 40960
    # Load tokenizer and prepare prompts
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompts = load_prompts(args, tokenizer)

    print(f"Loaded {len(prompts)} prompts")

    if args.dataset_name == "cropped_aimo":
        prompt_ids = prompts
        # prompt_ids = [
        #     tokenizer.encode(prompt)
        #     for prompt in prompts
        # ]
        # print(f"Prompt: {prompts[0]}")
        # print(f"Prompt IDs: {prompt_ids[0]}")
    else:
        prompt_ids = [
            tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
            for prompt in prompts
        ]

    # Get speculative config
    speculative_config = get_speculative_config(args)

    # Initialize LLM
    llm_kwargs = {
        "model": model_dir,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tp,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enforce_eager": args.enforce_eager,
        "max_model_len": max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": args.enable_prefix_caching,
        "disable_log_stats": args.disable_log_stats,
        "block_size": 1, # NOTE(brian1009): Set to 1 to disable prefix caching
        "recent_ratio": args.recent_ratio,
        "sink_size": args.sink_size,
    }

    if speculative_config is not None:
        llm_kwargs["speculative_config"] = speculative_config
        print(f"\n{'='*70}")
        print(f"SELF-SPECULATIVE DECODING CONFIGURATION")
        print(f"{'='*70}")
        print(f"Method: {speculative_config['method']}")
        print(f"Self-spec threshold (ACCUMULATING → VERIFYING): {speculative_config['num_speculative_tokens']} tokens")
        if speculative_config['method'] == 'self_spec_ngram':
            print(f"N-gram draft tokens per step: {speculative_config.get('num_ngram_draft_tokens', 'NOT SET')} tokens")
            print(f"N-gram window size: max={speculative_config['prompt_lookup_max']}, min={speculative_config['prompt_lookup_min']}")
            print(f"")
            print(f"Expected behavior:")
            ngram_drafts = speculative_config.get('num_ngram_draft_tokens', 3)
            print(f"  - During ACCUMULATING: N-gram proposes up to {ngram_drafts} drafts/step")
            print(f"  - Reaches threshold in ~{speculative_config['num_speculative_tokens'] // ngram_drafts} steps (with good acceptance)")
            print(f"  - Then VERIFYING: All {speculative_config['num_speculative_tokens']} tokens verified with full KV")
        print(f"Sparse attention config: sink_size={args.sink_size}, recent_ratio={args.recent_ratio}")
        print(f"{'='*70}\n")
    else:
        print("Self-speculative decoding disabled")

    llm = LLM(**llm_kwargs)

    # Set up sampling parameters
    sampling_params = SamplingParams(temperature=args.temp, max_tokens=96, top_p=1.0, ignore_eos=False)


    # Generate outputs
    print("Starting generation...")
    import time
    start_time = time.time()
    #llm.start_profile()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=x) for x in prompt_ids], 
        sampling_params=sampling_params
    )
    #llm.stop_profile()
    end_time = time.time()
    print(f"Generation time: {end_time - start_time} seconds")

    # Print generated text
    if args.verbose:

        # print model outputs
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            print(f"\n{'='*60}")
            print(f"Output {i+1}/{len(outputs)}")
            print(f"{'='*60}")
            generated_text = output.outputs[0].text
            generated_ids = output.outputs[0].token_ids
            prompt_ids = output.prompt_token_ids
            finish_reason = output.outputs[0].finish_reason

            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            print(f"Prompt length: {len(prompt_ids)}")
            print(f"Generated length: {len(generated_ids)}")
            print(f"Generated IDs: {generated_ids}")
            print(f"Finish Reason: {finish_reason}")

         # Print metrics if available
        try:
            metrics = llm.get_metrics()

            # Display self-spec request state statistics from Prometheus metrics
            print("\n" + "="*80)
            print("SELF-SPEC STATISTICS FROM PROMETHEUS METRICS")
            print("="*80)

            # Extract metrics from Prometheus snapshot (similar to math_eval.py)
            total_draft_tokens = 0
            total_accepted_tokens = 0
            num_drafts = 0

            for metric in metrics:
                metric_name = metric.name if hasattr(metric, 'name') else str(metric)

                if 'spec_decode_num_draft_tokens' in metric_name and hasattr(metric, 'value'):
                    total_draft_tokens = int(metric.value)
                elif 'spec_decode_num_accepted_tokens' in metric_name and 'per_pos' not in metric_name and hasattr(metric, 'value'):
                    total_accepted_tokens = int(metric.value)
                elif 'spec_decode_num_drafts' in metric_name and hasattr(metric, 'value'):
                    num_drafts = int(metric.value)

            if total_draft_tokens > 0 or total_accepted_tokens > 0:
                acceptance_rate = (total_accepted_tokens / total_draft_tokens) if total_draft_tokens > 0 else 0.0

                print("SPECULATIVE DECODING METRICS:")
                print(f"  Draft tokens proposed: {total_draft_tokens:,}")
                print(f"  Tokens accepted: {total_accepted_tokens:,}")
                print(f"  Tokens rejected: {total_draft_tokens - total_accepted_tokens:,}")
                print(f"  Acceptance rate: {acceptance_rate * 100:.2f}%")
                print(f"  Number of drafts: {num_drafts:,}")
                if num_drafts > 0:
                    print(f"  Avg draft tokens per draft: {total_draft_tokens / num_drafts:.2f}")
                    print(f"  Avg accepted per draft: {total_accepted_tokens / num_drafts:.2f}")
                print()
            else:
                print("No speculative decoding metrics available yet.")
                print()

            # # Show iteration-by-iteration breakdown (first 10 and last 10 iterations)
            # print("ITERATION-BY-ITERATION BREAKDOWN:")
            # print(f"{'Iter':<6} {'Tokens':<8} {'Total':<7} {'Accum':<7} {'Verify':<8} {'Normal':<8} {'Accum%':<8} {'Verify%':<9} {'Normal%':<8}")
            # print("-" * 80)

            # # Show first 10 iterations
            # show_iterations = min(10, len(total_reqs_history))
            # for i in range(show_iterations):
            #     tokens = total_tokens_history[i]
            #     total_reqs = total_reqs_history[i]
            #     accumulating = accumulating_history[i]
            #     verifying = verifying_history[i]
            #     normal = total_reqs - accumulating - verifying

            #     accum_pct = (accumulating / total_reqs * 100) if total_reqs > 0 else 0
            #     verify_pct = (verifying / total_reqs * 100) if total_reqs > 0 else 0
            #     normal_pct = (normal / total_reqs * 100) if total_reqs > 0 else 0

            #     print(f"{i+1:<6} {tokens:<8} {total_reqs:<7} {accumulating:<7} {verifying:<8} {normal:<8} {accum_pct:<7.1f}% {verify_pct:<8.1f}% {normal_pct:<7.1f}%")

            # # Show last 10 iterations if there are more than 10 total
            # if len(total_reqs_history) > 10:
            #     # if len(total_reqs_history) > 20:
            #     #     print("  ...")
            #     #start_idx = max(10, len(total_reqs_history) - 10)
            #     start_idx = 10
            #     for i in range(start_idx, len(total_reqs_history)):
            #         tokens = total_tokens_history[i]
            #         total_reqs = total_reqs_history[i]
            #         accumulating = accumulating_history[i]
            #         verifying = verifying_history[i]
            #         normal = total_reqs - accumulating - verifying

            #         accum_pct = (accumulating / total_reqs * 100) if total_reqs > 0 else 0
            #         verify_pct = (verifying / total_reqs * 100) if total_reqs > 0 else 0
            #         normal_pct = (normal / total_reqs * 100) if total_reqs > 0 else 0

            #         print(f"{i+1:<6} {tokens:<8} {total_reqs:<7} {accumulating:<7} {verifying:<8} {normal:<8} {accum_pct:<7.1f}% {verify_pct:<8.1f}% {normal_pct:<7.1f}%")

            # print("="*80)
            print_metrics(metrics)
        except Exception as e:
            if "Stat logging disabled" in str(e):
                print("\nWarning: Stat logging is disabled. Cannot retrieve metrics.")
            else:
                print(f"\nCould not retrieve metrics: {e}")


def print_metrics(metrics):
    """Print self-speculative decoding metrics."""
    num_drafts = num_accepted = 0
    acceptance_counts = [0] * 312 # Track acceptance at each position

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    if num_drafts > 0:
        print(f"\n{'='*60}")
        print("SELF-SPECULATIVE DECODING METRICS")
        print(f"{'='*60}")
        print(f"Mean acceptance length: {1 + (num_accepted / num_drafts):.2f}")
        print(f"Total drafts: {num_drafts}")
        print(f"Total accepted: {num_accepted}")

        print("\nAcceptance rate by token position:")
        for i in range(len(acceptance_counts)):
            if acceptance_counts[i] > 0:
                rate = acceptance_counts[i] / num_drafts
                print(f"  Position {i}: {rate:.3f} ({acceptance_counts[i]}/{num_drafts})")


if __name__ == "__main__":
    main()
