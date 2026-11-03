# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark segment-layout reads against separate reads.

One "group" is a shared prefix and N candidates. A packed read sends the group
as one request with ``segment_layout_params`` (one forward, FlexAttention);
separate reads send N ``prefix + candidate`` requests. Both run on the
``token_embed`` task of an encoder-only model and return one row per
candidate.

Reported per (prefix length, candidate length, N):
  * latency: median wall time of one group, batch size 1;
  * throughput: candidates per second over ``--num-groups`` groups at once.

Packed prompts are longer than any single ``prefix + candidate`` prompt while
their positions restart after the prefix, so ``--max-model-len`` may exceed
the model's position limit; set ``VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`` in that
case (positions stay below ``prefix + candidate`` length).

Example:
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python benchmarks/benchmark_segment_layout.py \\
        --model BAAI/bge-base-en-v1.5 \\
        --shapes 256x32 32x128 --num-candidates 4 16 64

"""

import argparse
import gc
import json
import statistics
import time

import torch

from vllm import LLM, PoolingParams
from vllm.config import PoolerConfig
from vllm.pooling_params import SegmentLayoutParams


def make_groups(num_groups, prefix_len, cand_len, num_candidates, vocab, seed):
    generator = torch.Generator().manual_seed(seed)

    def ids(n):
        return torch.randint(1000, vocab - 1, (n,), generator=generator).tolist()

    return [
        (ids(prefix_len), [ids(cand_len) for _ in range(num_candidates)])
        for _ in range(num_groups)
    ]


def packed_inputs(groups):
    prompts, params = [], []
    for prefix, candidates in groups:
        prompts.append(
            {"prompt_token_ids": prefix + [t for c in candidates for t in c]}
        )
        params.append(
            PoolingParams(
                segment_layout_params=SegmentLayoutParams(
                    [len(prefix)] + [len(c) for c in candidates]
                )
            )
        )
    return prompts, params


def separate_inputs(groups):
    prompts = [
        {"prompt_token_ids": prefix + c} for prefix, cands in groups for c in cands
    ]
    return prompts, [PoolingParams() for _ in prompts]


def run(llm, prompts, params, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        llm.encode(
            prompts, pooling_params=params, pooling_task="token_embed", use_tqdm=False
        )
        times.append(time.perf_counter() - start)
    return times


def bench_mode(args, mode, backend, shapes):
    llm = LLM(
        args.model,
        runner="pooling",
        pooler_config=PoolerConfig(task="token_embed"),
        attention_backend=backend,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    vocab = llm.llm_engine.model_config.get_vocab_size()
    build = packed_inputs if mode == "packed" else separate_inputs
    rows = []
    for prefix_len, cand_len in shapes:
        for n in args.num_candidates:
            one = make_groups(1, prefix_len, cand_len, n, vocab, seed=0)
            many = make_groups(args.num_groups, prefix_len, cand_len, n, vocab, 1)
            run(llm, *build(one), repeats=3)  # warm up
            latency = statistics.median(run(llm, *build(one), args.repeats))
            throughput = run(llm, *build(many), repeats=3)
            rows.append(
                {
                    "mode": mode,
                    "backend": backend,
                    "prefix_len": prefix_len,
                    "candidate_len": cand_len,
                    "num_candidates": n,
                    "latency_ms": round(latency * 1e3, 2),
                    "candidates_per_s": round(
                        args.num_groups * n / statistics.median(throughput), 1
                    ),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["256x32", "32x128"],
        help="PREFIXxCANDIDATE token lengths",
    )
    parser.add_argument("--num-candidates", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--num-groups", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--separate-backends",
        nargs="+",
        default=["FLASH_ATTN", "FLEX_ATTENTION"],
        help="attention backends for the separate-reads baseline",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes]
    rows = bench_mode(args, "packed", "FLEX_ATTENTION", shapes)
    for backend in args.separate_backends:
        rows += bench_mode(args, "separate", backend, shapes)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
