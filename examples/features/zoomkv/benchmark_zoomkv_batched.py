#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate ZoomKV batched sparse decode scaling (BS=1 vs BS=N).

Runs a fixed-length prefill then a short decode sweep under
``VLLM_ZOOMKV_STAGE_TIMER=1`` so we can confirm ZoomKV internal cost no longer
scales ~linearly with batch size after the batched fast path.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_ZOOMKV_STAGE_TIMER", "1")

import torch

from vllm import LLM, SamplingParams
from vllm.config.attention import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops.zoomkv import stage_timer as _zt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--mode", choices=("native", "dense", "sparse"), default="sparse"
    )
    p.add_argument("--batch-sizes", type=str, default="1,2,4,8")
    p.add_argument("--prompt-tokens", type=int, default=16384)
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=17408)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--threshold", type=int, default=512)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile / piecewise CUDA graphs.",
    )
    p.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Reuse the warmup prompt blocks for focused long-context decode.",
    )
    p.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Bracket each measured generate with cudaProfilerStart/Stop.",
    )
    p.add_argument(
        "--layerwise-nvtx",
        action="store_true",
        help="Emit model/module NVTX ranges for single-layer inspection.",
    )
    p.add_argument("--output-json")
    return p.parse_args()


def make_prompt(approx_tokens: int) -> str:
    unit = "The capital of France is a well-known European city. "
    # Empirically ~11 tokens/unit for this prose on Qwen tokenizers; stay
    # under the target so max_model_len - output_tokens still has headroom.
    repeats = max(1, int(approx_tokens / 11.5))
    return (unit * repeats) + "What is the capital of France? Answer with one word."


def main() -> None:
    args = parse_args()
    batches = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    prompt = make_prompt(args.prompt_tokens)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,
        enable_layerwise_nvtx_tracing=args.layerwise_nvtx,
        enable_prefix_caching=args.enable_prefix_caching,
        max_num_seqs=max(batches),
        attention_config=AttentionConfig(
            backend=(
                AttentionBackendEnum.FLASH_ATTN
                if args.mode == "native"
                else AttentionBackendEnum.ZOOMKV
            ),
            zoomkv_sink_size=64,
            zoomkv_local_size=256,
            zoomkv_final_topk=100,
            zoomkv_full_attention_threshold=args.threshold,
            zoomkv_dense_fallback=args.mode == "dense",
            zoomkv_enable_offload=False,
        ),
    )

    # Warmup / measure actual prompt length.
    warm = llm.generate(
        [prompt],
        SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True),
    )[0]
    prompt_tokens = len(warm.prompt_token_ids)
    print(f"prompt_tokens={prompt_tokens} mode={args.mode}")

    results = []
    for bs in batches:
        prompts = [prompt] * bs
        # Drop warm-up accumulate from stage timer.
        _zt.dump_and_reset(label="warmup-drop")
        sampling = SamplingParams(
            max_tokens=args.output_tokens,
            temperature=0.0,
            ignore_eos=True,
        )
        if args.cuda_profiler_range:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sampling)
        wall_s = time.perf_counter() - t0
        if args.cuda_profiler_range:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        decode_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        tpots_ms = []
        for out in outs:
            metrics = out.metrics
            num_output_tokens = len(out.outputs[0].token_ids)
            if metrics is not None and num_output_tokens > 1:
                tpots_ms.append(
                    1000.0
                    * (metrics.last_token_ts - metrics.first_token_ts)
                    / (num_output_tokens - 1)
                )
        report = _zt.dump_and_reset(
            label=f"bs={bs}", decode_tokens=max(1, decode_tokens // max(1, bs))
        )
        print(report)
        # Parse total sparse GPU ms from the report lines if present.
        row = {
            "batch_size": bs,
            "prompt_tokens": prompt_tokens,
            "output_tokens_per_req": args.output_tokens,
            "wall_s": wall_s,
            "decode_tokens_total": decode_tokens,
            "tok_s_output": decode_tokens / wall_s if wall_s > 0 else 0.0,
            "mean_tpot_ms": (
                sum(tpots_ms) / len(tpots_ms) if tpots_ms else None
            ),
            "stage_report": report,
        }
        results.append(row)
        print(
            f"BS={bs}: wall={wall_s:.2f}s output_tok/s={row['tok_s_output']:.2f} "
            f"mean_tpot_ms={row['mean_tpot_ms']} "
            f"total_decode_tokens={decode_tokens}"
        )

    # Scaling summary: compare BS=1 vs largest BS using wall output throughput.
    by_bs = {r["batch_size"]: r for r in results}
    if 1 in by_bs:
        base = by_bs[1]["tok_s_output"]
        for bs, row in by_bs.items():
            if bs == 1:
                continue
            ideal = base * bs
            eff = row["tok_s_output"] / ideal if ideal > 0 else 0.0
            print(
                f"scaling BS={bs}: tok/s={row['tok_s_output']:.2f} "
                f"vs ideal {ideal:.2f} (efficiency={eff:.2%})"
            )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"mode": args.mode, "results": results}, f, indent=2)
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
