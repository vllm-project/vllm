#!/usr/bin/env python3
"""MPI/external-launcher vLLM KV-cache benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from vllm import LLM, SamplingParams, envs

MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("baseline", "dram", "legomem"), required=True)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--distractors", type=int, default=8)
    parser.add_argument("--kv-cache-mib", type=int, default=32)
    parser.add_argument("--offload-mib", type=int, default=255)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--require-amx", action="store_true")
    return parser.parse_args()


def timed_generate(
    llm: LLM, prompt: list[int], params: SamplingParams
) -> tuple[float, int]:
    if dist.is_initialized():
        dist.barrier()
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params=params, use_tqdm=False)
    if dist.is_initialized():
        dist.barrier()
    elapsed = torch.tensor([time.perf_counter() - start], dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    cached = torch.tensor(
        [int(outputs[0].num_cached_tokens or 0)], dtype=torch.int64
    )
    if dist.is_initialized():
        dist.all_reduce(cached, op=dist.ReduceOp.MIN)
    return float(elapsed.item()), int(cached.item())


def make_prompt(seed: int, length: int, vocab_size: int) -> list[int]:
    # Avoid low-numbered special tokens while producing disjoint block hashes.
    usable = max(1, vocab_size - 1024)
    return [1024 + ((seed * 7919 + i * 104729) % usable) for i in range(length)]


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    if world_size != args.tp:
        raise ValueError(f"MPI world size {world_size} != tensor parallel size {args.tp}")

    amx_supported = bool(torch.cpu._is_amx_tile_supported())
    int4_w4a8_enabled = bool(envs.VLLM_CPU_INT4_W4A8)
    if args.require_amx and not amx_supported:
        raise RuntimeError(f"rank {rank} does not expose AMX tile support")
    if args.require_amx and not int4_w4a8_enabled:
        raise RuntimeError("VLLM_CPU_INT4_W4A8 must be enabled for the AMX run")
    print(
        "VLLM_AMX_AUDIT="
        + json.dumps(
            {
                "rank": rank,
                "amx_tile_supported": amx_supported,
                "vllm_cpu_int4_w4a8": int4_w4a8_enabled,
                "dtype": args.dtype,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    connector = None
    if args.mode != "baseline":
        per_rank_bytes = args.offload_mib * MIB
        connector = {
            "kv_connector": "SimpleCPUOffloadConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": per_rank_bytes * world_size,
                "cpu_bytes_to_use_per_rank": per_rank_bytes,
                # Offload completed prefix blocks as they approach eviction.
                # This matches an inference-service workload where prefixes
                # survive request completion and are reused by later calls.
                "lazy_offload": True,
                "kv_offload_backend": "cpu" if args.mode == "dram" else "legomem",
                "legomem_library_path": "/home/ubuntu/legomem/lib/liblegomem_kv.so",
                "legomem_host": "127.0.0.1",
                "legomem_port": 9999,
                "legomem_num_nodes": world_size,
                "legomem_node_capacity_bytes": 256 * MIB,
            },
        }

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        distributed_executor_backend="external_launcher",
        max_model_len=args.input_tokens + args.output_tokens,
        max_num_batched_tokens=args.input_tokens + args.output_tokens,
        max_num_seqs=1,
        kv_cache_memory_bytes=args.kv_cache_mib * MIB,
        enable_prefix_caching=True,
        kv_transfer_config=connector,
        enforce_eager=True,
        disable_log_stats=False,
    )
    tokenizer = llm.get_tokenizer()
    vocab_size = int(getattr(tokenizer, "vocab_size", 32000))
    params = SamplingParams(
        temperature=0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
        detokenize=False,
    )

    # Warm kernels without polluting the measured cache state.
    timed_generate(llm, make_prompt(9001, min(32, args.input_tokens), vocab_size), params)
    if not llm.reset_prefix_cache(reset_connector=True):
        raise RuntimeError("failed to reset vLLM prefix caches before measurement")

    target = make_prompt(1, args.input_tokens, vocab_size)
    cold_seconds, cold_cached_tokens = timed_generate(llm, target, params)

    eviction_seconds = 0.0
    for index in range(args.distractors):
        distractor_seconds, _ = timed_generate(
            llm, make_prompt(100 + index, args.input_tokens, vocab_size), params
        )
        eviction_seconds += distractor_seconds

    replay_seconds, replay_cached_tokens = timed_generate(llm, target, params)
    result = {
        "model": args.model,
        "mode": args.mode,
        "ranks": world_size,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "distractors": args.distractors,
        "kv_cache_mib_per_rank": args.kv_cache_mib,
        "offload_mib_per_rank": 0 if args.mode == "baseline" else args.offload_mib,
        "cold_seconds": cold_seconds,
        "cold_cached_tokens": cold_cached_tokens,
        "eviction_seconds": eviction_seconds,
        "replay_seconds": replay_seconds,
        "replay_cached_tokens": replay_cached_tokens,
        "replay_speedup_vs_cold": cold_seconds / replay_seconds,
        "estimated_ttft_seconds": replay_seconds,
        "amx_required": args.require_amx,
        "amx_tile_supported": amx_supported,
        "vllm_cpu_int4_w4a8": int4_w4a8_enabled,
    }
    if rank == 0:
        print("VLLM_MPI_KV_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
