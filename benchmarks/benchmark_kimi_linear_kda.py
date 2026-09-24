# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Kimi Linear end-to-end latency with selectable KDA backends.

Examples:
    python benchmarks/benchmark_kimi_linear_kda.py \
        --tp 1 --prefill triton --decode triton \
        --prompt-lengths 128 --output-json /tmp/kimi-linear-triton.json

    python benchmarks/benchmark_kimi_linear_kda.py \
        --tp 1 --prefill helion --decode helion \
        --prompt-lengths 17 31 47 63 79 95 111 127 \
        --output-lengths 1 16 --output-json /tmp/kimi-linear-varlen.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="moonshotai/Kimi-Linear-48B-A3B-Instruct",
    )
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument(
        "--prefill",
        choices=("triton", "helion"),
        required=True,
    )
    parser.add_argument(
        "--decode",
        choices=("triton", "helion"),
        required=True,
    )
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128])
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if any(length <= 0 for length in args.prompt_lengths):
        parser.error("--prompt-lengths values must be positive")
    if any(length <= 0 for length in args.output_lengths):
        parser.error("--output-lengths values must be positive")

    max_model_len = max(512, max(args.prompt_lengths) + max(args.output_lengths))
    max_num_batched_tokens = args.max_num_batched_tokens or max(
        512,
        1 << (sum(args.prompt_lengths) - 1).bit_length(),
    )
    max_num_seqs = max(args.max_num_seqs, len(args.prompt_lengths))
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        seed=0,
        tensor_parallel_size=args.tp,
        attention_backend="TRITON_MLA",
        disable_custom_all_reduce=args.tp > 1,
        enable_prefix_caching=False,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kda_prefill_backend=args.prefill,
        kda_decode_backend=args.decode,
    )
    prompts = [
        TokensPrompt(
            prompt_token_ids=list(
                range(100 + index * 1024, 100 + index * 1024 + length)
            )
        )
        for index, length in enumerate(args.prompt_lengths)
    ]
    results: dict[str, Any] = {
        "model": args.model,
        "device": torch.cuda.get_device_name(),
        "tensor_parallel_size": args.tp,
        "prefill_backend": args.prefill,
        "decode_backend": args.decode,
        "prompt_lengths": args.prompt_lengths,
        "output_lengths": args.output_lengths,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "warmup": args.warmup,
        "rep": args.rep,
    }
    for output_len in args.output_lengths:
        sampling_params = SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=output_len,
            detokenize=False,
        )

        def run(params: SamplingParams = sampling_params) -> float:
            start = time.perf_counter()
            llm.generate(prompts, params, use_tqdm=False)
            return time.perf_counter() - start

        for _ in range(args.warmup):
            run()
        samples = np.array([run() for _ in range(args.rep)])
        result = {
            "mean_seconds": float(samples.mean()),
            "median_seconds": float(np.median(samples)),
            "p10_seconds": float(np.percentile(samples, 10)),
            "p90_seconds": float(np.percentile(samples, 90)),
            "samples_seconds": samples.tolist(),
        }
        results[str(output_len)] = result
        print(output_len, result, flush=True)

    args.output_json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
