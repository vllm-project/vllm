# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Trace-replay with vLLM offline inference.

Trace-replay lets you supply a known sequence of decode token IDs alongside
the prompt.  Instead of sampling from the model distribution, the engine
injects each decode token deterministically, step by step.  All other
outputs — logprobs, token ranks, text decoding — are computed faithfully
from the real logit distribution for that token.

How it works:
  1. Set ``SamplingParams.trace_decode_token_ids`` to the list of decode
     token IDs you want to force.
  2. The engine will output exactly those tokens and stop; ``max_tokens``
     is ignored.  EOS tokens inside the trace sequence do **not** halt
     generation early.

Incompatible features (fall back to normal sampling with a warning):
  * Speculative decoding
  * n > 1

Typical use-cases:
  * Reproduce exact outputs from a previous run for benchmarking.
  * Compute logprobs for an already-known output (e.g. reference answers).
  * Dataset annotation: given (prompt, response) pairs, obtain per-token
    logprob scores without altering the response.

Usage:
    python examples/generate/trace_replay_offline.py
    python examples/generate/trace_replay_offline.py --model facebook/opt-125m
"""

import argparse

from vllm import LLM, SamplingParams

DEFAULT_PROMPT = "Hello, my name is"


def build_llm(args: argparse.Namespace) -> LLM:
    """Construct an LLM from common CLI args."""
    llm_kwargs: dict = {
        "model": args.model,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    return LLM(**llm_kwargs)


def run_normal_generation(llm: LLM, prompt: str, max_tokens: int) -> list[int]:
    """Run a standard greedy generation and return the output token IDs."""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        # Request logprobs for the greedy token at each step so we can
        # compare them against the trace-replay logprobs below.
        logprobs=1,
    )
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    result = outputs[0].outputs[0]
    output_token_ids = list(result.token_ids)
    print("[Normal generation]")
    print(f"  Prompt           : {prompt!r}")
    print(f"  Output token IDs : {output_token_ids}")
    print(f"  Output text      : {result.text!r}")
    return output_token_ids


def run_trace_replay(llm: LLM, prompt: str, decode_token_ids: list[int]) -> None:
    """Replay a known decode sequence and print per-token logprobs."""
    sampling_params = SamplingParams(
        # Provide the decode tokens to replay.
        trace_decode_token_ids=decode_token_ids,
        # Request top-5 logprobs so we can inspect the distribution.
        logprobs=5,
    )
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    result = outputs[0].outputs[0]
    replayed_ids = list(result.token_ids)

    print("\n[Trace-replay]")
    print(f"  Requested decode token IDs : {decode_token_ids}")
    print(f"  Replayed  output token IDs : {replayed_ids}")
    print(f"  Replayed  output text      : {result.text!r}")

    # Verify the replayed tokens match exactly.
    assert replayed_ids == decode_token_ids, (
        f"Mismatch!\n  expected: {decode_token_ids}\n  got:      {replayed_ids}"
    )
    print("  Replayed tokens match the requested trace exactly.")

    # Show per-token logprobs (computed from the real distribution).
    if result.logprobs:
        print("\n  Per-token logprobs (trace token):")
        for step, (token_id, logprob_dict) in enumerate(
            zip(replayed_ids, result.logprobs)
        ):
            sampled_lp = logprob_dict.get(token_id)
            lp_value = f"{sampled_lp.logprob:.4f}" if sampled_lp is not None else "n/a"
            rank = sampled_lp.rank if sampled_lp is not None else "n/a"
            print(
                f"    step {step:2d}: token_id={token_id:6d}  "
                f"logprob={lp_value}  rank={rank}"
            )


def run_demo(args: argparse.Namespace) -> None:
    print(f"Loading model: {args.model}\n")
    llm = build_llm(args)
    prompt = args.prompt

    print("=" * 60)
    print("Step 1 — Normal greedy generation (captures decode tokens)")
    print("=" * 60)
    decode_token_ids = run_normal_generation(llm, prompt, max_tokens=8)

    print("\n" + "=" * 60)
    print("Step 2 — Trace-replay with the captured tokens")
    print("=" * 60)
    run_trace_replay(llm, prompt, decode_token_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace-replay with vLLM offline inference"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt to use for generation (default: %(default)r)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Name or path of the HuggingFace model to use",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of tensor parallel replicas",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Always use eager-mode PyTorch (disable CUDA graph)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Model context length",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences per iteration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
