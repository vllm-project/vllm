#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the V1 GPU worker Gumbel-max sampler.

This covers the hot path used by Model Runner V2 sampling and Eagle-style
speculative decoding. The default shapes mirror the regression report in
https://github.com/vllm-project/vllm/issues/40755:

* 40 tokens: one sampled token for 40 concurrent requests.
* 240 tokens: six sampled tokens for 40 concurrent requests with Eagle3.

The benchmark can optionally include the processed-logits writeback used by
draft sampling, and can compare the default fp32 Gumbel noise path against the
opt-in fp64 path.
"""

import argparse
from dataclasses import dataclass

import torch

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample


@dataclass(frozen=True)
class GumbelBenchConfig:
    num_tokens: int
    vocab_size: int
    num_reqs: int
    output_processed_logits: bool
    scalar_output_col: bool
    use_fp64: bool

    @property
    def name(self) -> str:
        if self.output_processed_logits:
            col_mode = "scalar_col" if self.scalar_output_col else "per_token_col"
            processed = f"processed_logits/{col_mode}"
        else:
            processed = "sample_only"
        precision = "fp64" if self.use_fp64 else "fp32"
        return (
            f"tokens={self.num_tokens}, vocab={self.vocab_size}, "
            f"reqs={self.num_reqs}, {processed}, {precision}"
        )


def make_inputs(config: GumbelBenchConfig, device: str, dtype: torch.dtype):
    logits = torch.randn(
        config.num_tokens, config.vocab_size, dtype=dtype, device=device
    )

    # Map consecutive sampled tokens back to request states. For the Eagle3
    # shape, this models six draft positions per request.
    repeats = triton.cdiv(config.num_tokens, config.num_reqs)
    expanded_idx_mapping = torch.arange(
        config.num_reqs, dtype=torch.int32, device=device
    ).repeat_interleave(repeats)[: config.num_tokens]

    temperature = torch.ones(config.num_reqs, dtype=torch.float32, device=device)
    seeds = torch.arange(config.num_reqs, dtype=torch.int64, device=device) + 0xABCD
    pos = torch.arange(config.num_tokens, dtype=torch.int64, device=device)

    output_logits = None
    output_col = None
    if config.output_processed_logits:
        num_cols = max(1, triton.cdiv(config.num_tokens, config.num_reqs))
        output_logits = torch.empty(
            config.num_reqs,
            num_cols,
            config.vocab_size,
            dtype=dtype,
            device=device,
        )
        if config.scalar_output_col:
            output_col = torch.zeros((), dtype=torch.int64, device=device)
        else:
            output_col = (
                torch.arange(config.num_tokens, dtype=torch.int64, device=device)
                % num_cols
            )

    return (
        logits,
        expanded_idx_mapping,
        temperature,
        seeds,
        pos,
        output_logits,
        output_col,
    )


def run_one(config: GumbelBenchConfig, args: argparse.Namespace) -> None:
    inputs = make_inputs(config, args.device, getattr(torch, args.dtype))
    (
        logits,
        expanded_idx_mapping,
        temperature,
        seeds,
        pos,
        output_logits,
        output_col,
    ) = inputs

    def fn():
        return gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seeds,
            pos,
            apply_temperature=True,
            output_processed_logits=output_logits,
            output_processed_logits_col=output_col,
            use_fp64=config.use_fp64,
        )

    # Compile/warm up before timing.
    fn()
    torch.cuda.synchronize()

    quantiles = (0.5, 0.2, 0.8)
    ms, min_ms, max_ms = triton.testing.do_bench(
        fn,
        warmup=args.warmup,
        rep=args.repeat,
        quantiles=quantiles,
    )
    print(
        f"{config.name}: median={ms * 1000:.2f}us "
        f"p20={min_ms * 1000:.2f}us p80={max_ms * 1000:.2f}us"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM Gumbel sampling")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--num-reqs", type=int, default=40)
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[40, 240])
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--include-fp64",
        action="store_true",
        help="Also benchmark the opt-in fp64 Gumbel noise path.",
    )
    parser.add_argument(
        "--include-sample-only",
        action="store_true",
        help="Also benchmark sampling without processed-logits writeback.",
    )
    parser.add_argument(
        "--scalar-output-col",
        action="store_true",
        help=(
            "Use the current scalar draft-step column for processed-logits "
            "writeback. By default, per-token columns are used when "
            "--num-tokens is larger than --num-reqs to avoid repeated writes "
            "to the same request/column slot."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    configs: list[GumbelBenchConfig] = []
    for num_tokens in args.num_tokens:
        output_modes = [True]
        if args.include_sample_only:
            output_modes.append(False)
        precision_modes = [False]
        if args.include_fp64:
            precision_modes.append(True)

        for output_processed_logits in output_modes:
            for use_fp64 in precision_modes:
                configs.append(
                    GumbelBenchConfig(
                        num_tokens=num_tokens,
                        vocab_size=args.vocab_size,
                        num_reqs=args.num_reqs,
                        output_processed_logits=output_processed_logits,
                        scalar_output_col=(
                            args.scalar_output_col or num_tokens <= args.num_reqs
                        ),
                        use_fp64=use_fp64,
                    )
                )

    for config in configs:
        run_one(config, args)


if __name__ == "__main__":
    main()
