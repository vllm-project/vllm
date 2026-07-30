# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-node DP/EP model correctness and latency gate for SharedEP."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import time


def run_rank(
    rank: int,
    world_size: int,
    port: int,
    args: argparse.Namespace,
    result_queue: mp.Queue,
) -> None:
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(world_size)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = str(port)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        enable_expert_parallel=True,
        all2all_backend=args.backend,
        max_model_len=args.input_len + args.output_len,
        max_num_batched_tokens=32,
        enable_chunked_prefill=True,
        max_num_seqs=32,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        disable_log_stats=False,
        compilation_config={
            "cudagraph_capture_sizes": [1, 4, 16, 32],
            "max_cudagraph_capture_size": 32,
        },
    )
    if args.prompt is None:
        tokenizer = llm.get_tokenizer()
        seed_ids = tokenizer.encode(
            "The capital of France is Paris. ",
            add_special_tokens=False,
        )
        prompt_ids = (
            seed_ids * ((args.input_len + len(seed_ids) - 1) // len(seed_ids))
        )[: args.input_len]
        prompts = [
            {"prompt_token_ids": prompt_ids} for _ in range(args.batch_size_per_rank)
        ]
    else:
        prompts = [args.prompt for _ in range(args.batch_size_per_rank)]
    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.output_len,
        ignore_eos=True,
    )
    for _ in range(args.warmup):
        llm.generate(prompts, sampling, use_tqdm=False)
    trials = []
    reference_ids = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed_ms = (time.perf_counter() - started) * 1000
        ttfts = []
        tpots = []
        output_ids = []
        for output in outputs:
            metrics = output.metrics
            if metrics is None:
                raise RuntimeError("request metrics were not collected")
            ids = output.outputs[0].token_ids
            if len(ids) != args.output_len:
                raise AssertionError(
                    f"rank {rank} generated {len(ids)}, expected {args.output_len}"
                )
            output_ids.append(ids)
            ttfts.append(metrics.first_token_latency * 1000)
            tpots.append(
                (metrics.last_token_ts - metrics.first_token_ts)
                * 1000
                / max(1, len(ids) - 1)
            )
        if reference_ids is None:
            reference_ids = output_ids
        elif output_ids != reference_ids:
            raise AssertionError(f"rank {rank} output changed between trials")
        trials.append(
            {
                "elapsed_ms": elapsed_ms,
                "median_ttft_ms": statistics.median(ttfts),
                "median_tpot_ms": statistics.median(tpots),
            }
        )
    result_queue.put((rank, trials, reference_ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="shared_ep")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--input-len", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument(
        "--prompt",
        help="Literal prompt. By default, construct exactly --input-len token IDs.",
    )
    parser.add_argument("--batch-size-per-rank", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--port", type=int, default=29671)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=run_rank,
            args=(rank, args.world_size, args.port, args, result_queue),
        )
        for rank in range(args.world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"SharedEP model workers failed with exit codes {failed}")
    results = sorted(result_queue.get() for _ in processes)
    for rank, trials, output_ids in results:
        print(f"rank={rank} trials={trials} output_ids={output_ids}")
    reference_ids = results[0][2]
    if any(output_ids != reference_ids for _, _, output_ids in results[1:]):
        raise AssertionError("greedy outputs differed across DP ranks")
    max_rank_trials = [
        {
            metric: max(result[1][trial][metric] for result in results)
            for metric in ("elapsed_ms", "median_ttft_ms", "median_tpot_ms")
        }
        for trial in range(args.repeats)
    ]
    median_elapsed_ms = statistics.median(
        trial["elapsed_ms"] for trial in max_rank_trials
    )
    median_ttft_ms = statistics.median(
        trial["median_ttft_ms"] for trial in max_rank_trials
    )
    median_tpot_ms = statistics.median(
        trial["median_tpot_ms"] for trial in max_rank_trials
    )
    print(
        f"model={args.model} backend={args.backend} world_size={args.world_size} "
        f"input_len={args.input_len} output_len={args.output_len} "
        f"batch_size_per_rank={args.batch_size_per_rank} "
        f"median_elapsed_ms={median_elapsed_ms:.3f} "
        f"median_ttft_ms={median_ttft_ms:.3f} "
        f"median_tpot_ms={median_tpot_ms:.3f} "
        f"trials={max_rank_trials} output_ids={reference_ids}"
    )


if __name__ == "__main__":
    main()
