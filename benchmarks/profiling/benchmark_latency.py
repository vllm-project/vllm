# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the latency of processing a single batch of requests."""

import argparse
import dataclasses
import json
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser


def main(args: argparse.Namespace):
    print(args)

    @contextmanager
    def rpd_profiler_context():
        from rpdTracerControl import rpdTracerControl as rpd

        llm.start_profile()
        yield
        llm.stop_profile()
        rpd.top_totals()

    @contextmanager
    def torch_profiler_context(profile_result_dir: str | None = None):
        p = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(profile_result_dir)
            ),
        )
        p.start()
        try:
            with torch.no_grad():
                yield p
        finally:
            p.stop()
            print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    def get_profiling_context(profile_result_dir: str | None = None):
        if args.profile_torch:
            return torch_profiler_context(profile_result_dir)
        elif args.profile_rpd:
            return rpd_profiler_context()
        else:
            return nullcontext()

    if args.profile_torch or args.profile_rpd:
        profile_result_dir = Path(
            args.profile_result_dir or "./vllm_benchmark_latency_result"
        )
        profile_result_dir.mkdir(parents=True, exist_ok=True)
        name = os.path.basename(os.path.normpath(args.model))
        model_trace_name = (
            f"{name}_in_{args.input_len}_out_{args.output_len}_"
            f"batch_{args.batch_size}_tp_{args.tensor_parallel_size}"
        )
        print(f"Profiling (results will be saved to '{profile_result_dir}')...")
        if args.profile_rpd:
            profile_result_dir /= f"{model_trace_name}.rpd"
            os.environ["VLLM_RPD_PROFILER_DIR"] = str(profile_result_dir)

    engine_args = EngineArgs.from_cli_args(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(**dataclasses.asdict(engine_args))

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(
        10000, size=(args.batch_size, args.input_len)
    )
    dummy_prompts: list[PromptType] = [
        {"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()
    ]

    def llm_generate():
        if not args.use_beam_search:
            llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)
        else:
            llm.beam_search(
                dummy_prompts,
                BeamSearchParams(
                    beam_width=args.n,
                    max_tokens=args.output_len,
                    ignore_eos=True,
                ),
            )

    def run_to_completion(profile_dir: str | None = None):
        if profile_dir:
            with get_profiling_context(profile_dir):
                llm_generate()
        else:
            start_time = time.perf_counter()
            llm_generate()
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile_dir=None)

    if args.profile_torch or args.profile_rpd:
        run_to_completion(profile_dir=profile_result_dir)
        return

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile_dir=None))
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f"Avg latency: {np.mean(latencies)} seconds")
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile} seconds")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion."
    )
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n", type=int, default=1, help="Number of generated sequences per prompt."
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument(
        "--num-iters", type=int, default=30, help="Number of iterations to run."
    )
    parser.add_argument(
        "--profile-torch",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--profile-rpd",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--profile-result-dir",
        type=str,
        default=os.getenv("VLLM_RPD_PROFILER_DIR", default=None),
        help=(
            "path to save the profiler output. Can be visualized "
            "with ui.perfetto.dev or Tensorboard."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the latency results in JSON format.",
    )

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
