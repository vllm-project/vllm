# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate dynamic speculative decoding config by profiling.

This script:
1. Profiles acceptance rates per position using offline inference
2. Runs benchmark sweeps to measure ITL for different (batch_size, K) combinations
3. Saves the config to a JSON file for use with dynamic SD

Example usage:
    python vllm/v1/spec_decode/dynamic/generate_config.py \
        --method eagle \
        --model-dir meta-llama/Llama-3.1-8B-Instruct \
        --draft-dir yuhuili/EAGLE-LLaMA3.1-Instruct-8B \
        --tp 1 \
        --max-vllm-batch-size 256 \
        --batch-size-list 1 4 16 64 256 \
        --num-speculative-tokens-list 1 3 5 \
        --dataset-name hf \
        --dataset-path philschmid/mt-bench \
        --result-dir ./log/dynamic_sd
"""

import dataclasses
import json
import pprint
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from vllm import LLM
from vllm.benchmarks.datasets import add_dataset_parser, get_samples
from vllm.benchmarks.sweep.param_sweep import ParameterSweep
from vllm.benchmarks.sweep.serve import SweepServeArgs, run_main
from vllm.config.speculative import DynamicSpeculativeConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.metrics.reader import Counter, Vector


def build_serve_params(
    method: str,
    draft_dir: str | None,
    tp: int,
    num_speculative_tokens_list: list[int],
    prompt_lookup_max: int,
    prompt_lookup_min: int,
) -> ParameterSweep:
    """Build serve parameter sweep for vanilla + speculative decode configs."""
    records: list[dict[str, object]] = []

    # Vanilla config (no speculative decoding, K=0)
    records.append({"_benchmark_name": "vanilla"})

    # Speculative decoding configs with varying num_speculative_tokens
    if method == "ngram":
        for k in num_speculative_tokens_list:
            records.append(
                {
                    "_benchmark_name": f"ngram-k-{k}",
                    "speculative_config": {
                        "method": "ngram",
                        "num_speculative_tokens": k,
                        "prompt_lookup_max": prompt_lookup_max,
                        "prompt_lookup_min": prompt_lookup_min,
                    },
                }
            )
    elif method in ("eagle", "eagle3"):
        for k in num_speculative_tokens_list:
            records.append(
                {
                    "_benchmark_name": f"{method}-k-{k}",
                    "speculative_config": {
                        "method": method,
                        "model": draft_dir,
                        "num_speculative_tokens": k,
                        "draft_tensor_parallel_size": tp,
                    },
                }
            )
    elif method == "mtp":
        for k in num_speculative_tokens_list:
            records.append(
                {
                    "_benchmark_name": f"mtp-k-{k}",
                    "speculative_config": {
                        "method": "mtp",
                        "num_speculative_tokens": k,
                    },
                }
            )

    return ParameterSweep.from_records(records)


def build_bench_params(
    batch_size_list: list[int],
    num_batches: int,
) -> ParameterSweep:
    """Build benchmark parameter sweep for different concurrencies."""
    records = []
    for bs in batch_size_list:
        num_prompts = num_batches * bs
        records.append(
            {
                "_benchmark_name": f"bs-{bs}",
                "max_concurrency": bs,
                "num_prompts": num_prompts,
            }
        )
    return ParameterSweep.from_records(records)


def parse_itl_from_dataframe(result_df) -> dict[int, dict[int, float]]:
    """Parse ITL from sweep result DataFrame into batch_stats format.

    Returns:
        batch_stats: dict of {batch_size: {num_drafts: median_itl_ms}}
        where num_drafts=0 corresponds to vanilla (no speculation).
    """
    batch_stats: dict[int, dict[int, float]] = {}
    for _, row in result_df.iterrows():
        bs = int(row["max_concurrency"])

        # Determine k from speculative_config
        spec_config = row.get("speculative_config")
        if isinstance(spec_config, dict):
            k = int(spec_config["num_speculative_tokens"])
        elif isinstance(spec_config, str):
            k = int(json.loads(spec_config)["num_speculative_tokens"])
        else:
            k = 0  # vanilla

        if bs not in batch_stats:
            batch_stats[bs] = {}
        batch_stats[bs][k] = row["median_itl_ms"]

    return batch_stats


def run_profiling_sweep(args):
    """Run profiling benchmarks using vllm bench sweep serve."""
    serve_cmd = [
        "vllm",
        "serve",
        args.model_dir,
        "--gpu-memory-utilization",
        "0.95",
        "--max-num-seqs",
        str(args.max_vllm_batch_size),
        "--tensor-parallel-size",
        str(args.tp),
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
    ]

    bench_cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        args.model_dir,
        "--backend",
        "openai-chat",
        "--endpoint",
        "/v1/chat/completions",
        "--dataset-name",
        args.dataset_name,
        "--dataset-path",
        args.dataset_path,
        f"--temperature={args.temp}",
        f"--top-p={args.top_p}",
        f"--top-k={args.top_k}",
    ]

    serve_params = build_serve_params(
        method=args.method,
        draft_dir=args.draft_dir,
        tp=args.tp,
        num_speculative_tokens_list=args.num_speculative_tokens_list,
        prompt_lookup_max=args.prompt_lookup_max,
        prompt_lookup_min=args.prompt_lookup_min,
    )

    bench_params = build_bench_params(
        batch_size_list=args.batch_size_list,
        num_batches=args.num_batches,
    )

    sweep_args = SweepServeArgs(
        serve_cmd=serve_cmd,
        bench_cmd=bench_cmd,
        after_bench_cmd=[],
        show_stdout=True,
        serve_params=serve_params,
        bench_params=bench_params,
        output_dir=Path(args.result_dir),
        num_runs=1,
        dry_run=False,
        resume=False,
        link_vars=[],
        server_ready_timeout=600,
        experiment_name=f"{args.method}-{args.num_spec_tokens}",
    )

    return run_main(sweep_args)


def get_acceptance_rate_per_pos(args) -> tuple[float, list[float]]:
    """Profile acceptance rate per position using offline inference."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    prompts = get_samples(args, tokenizer)

    llm_prompts: list[Any]
    if args.enable_multimodal_chat:
        llm_prompts = [p.prompt for p in prompts]
    else:
        llm_prompts = [
            {
                "prompt_token_ids": tokenizer.encode(
                    prompt.prompt, add_special_tokens=False
                ),
                "multi_modal_data": prompt.multi_modal_data,
            }
            for prompt in prompts
        ]

    # Build speculative config based on method
    if args.method in ("eagle", "eagle3"):
        eagle_dir = args.draft_dir
        if eagle_dir is None:
            eagle_dir = (
                "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
                if args.method == "eagle"
                else "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
            )
        speculative_config = {
            "method": args.method,
            "model": eagle_dir,
            "num_speculative_tokens": args.num_spec_tokens,
        }
    elif args.method == "ngram":
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": args.num_spec_tokens,
            "prompt_lookup_max": args.prompt_lookup_max,
            "prompt_lookup_min": args.prompt_lookup_min,
        }
    elif args.method == "mtp":
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": args.num_spec_tokens,
        }
    else:
        raise ValueError(f"unknown method: {args.method}")

    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enable_chunked_prefill=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
        disable_log_stats=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_vllm_batch_size,
    )

    sampling_params = SamplingParams(temperature=args.temp, max_tokens=args.output_len)
    _ = llm.chat(llm_prompts, sampling_params=sampling_params)

    metrics = llm.get_metrics()

    num_drafts = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * args.num_spec_tokens

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    print(f"Mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    acceptance_rate_per_pos = []
    for i, count in enumerate(acceptance_counts):
        rate = count / num_drafts if num_drafts > 0 else 0
        print(f"Acceptance at position {i}: {rate:.2f}")
        acceptance_rate_per_pos.append(rate)

    return acceptance_length, acceptance_rate_per_pos


def main():
    parser = FlexibleArgumentParser(description=__doc__)
    add_dataset_parser(parser)

    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--draft-dir", type=str, default=None)
    parser.add_argument(
        "--method",
        type=str,
        default="eagle",
        choices=["ngram", "eagle", "eagle3", "mtp"],
    )
    parser.add_argument(
        "--num-speculative-tokens-list",
        nargs="*",
        type=int,
        default=[1, 3, 5],
    )
    parser.add_argument(
        "--batch-size-list",
        nargs="*",
        type=int,
        default=[1, 4, 16, 64, 256],
    )
    parser.add_argument(
        "--max-vllm-batch-size",
        type=int,
        required=True,
        help="Max vLLM server batch size (must equal max of batch-size-list)",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--result-dir", type=str, default="./log/dynamic_sd")
    parser.add_argument("--prompt-lookup-max", type=int, default=5)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=10)

    args = parser.parse_args()
    args.print_output = False
    args.enable_multimodal_chat = False
    args.num_spec_tokens = max(args.num_speculative_tokens_list)

    # Construct result directory path
    args.result_dir = (
        f"{args.result_dir}/tp-{args.tp}_temp-{args.temp}"
        f"_top_p-{args.top_p}_top_k-{args.top_k}"
        f"/{args.dataset_path.replace('/', '_')}/"
    )
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    assert args.max_vllm_batch_size == max(args.batch_size_list), (
        "max_vllm_batch_size must equal max of batch_size_list"
    )

    pprint.pprint(vars(args))
    start = time.time()

    # Step 1: Profile acceptance rates
    print("\n" + "=" * 60)
    print("Step 1: Profiling acceptance rates per position")
    print("=" * 60)
    acceptance_length, acceptance_rate_per_pos = get_acceptance_rate_per_pos(args)
    print(f"\nAcceptance length: {acceptance_length:.2f}")
    print(f"Acceptance rates: {acceptance_rate_per_pos}")

    # Step 2: Run benchmark sweep for ITL profiling
    print("\n" + "=" * 60)
    print("Step 2: Running benchmark sweep for ITL profiling")
    print("=" * 60)
    result_df = run_profiling_sweep(args)

    # Step 3: Parse batch_stats from benchmark results
    print("\n" + "=" * 60)
    print("Step 3: Parsing batch statistics")
    print("=" * 60)
    batch_stats = parse_itl_from_dataframe(result_df)
    print(json.dumps(batch_stats, indent=2))

    # Step 4: Save config
    print("\n" + "=" * 60)
    print("Step 4: Saving config")
    print("=" * 60)

    dynamic_config = DynamicSpeculativeConfig(
        batch_stats=batch_stats,
        max_num_speculative_tokens=len(acceptance_rate_per_pos),
        acceptance_rate_per_pos=acceptance_rate_per_pos,
        warmup_steps=args.warmup_steps,
    )

    config_path = f"{args.result_dir}/dynamic_speculative_config.json"
    with open(config_path, "w") as f:
        json.dump(dataclasses.asdict(dynamic_config), f, indent=2)

    print(f"Config saved to: {config_path}")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
