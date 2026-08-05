#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke-test dense and sparse paths of the GPU-only ZoomKV backend."""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams
from vllm.config.attention import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--threshold", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=180)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--strict-kernels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of using Triton/PyTorch kernel fallbacks.",
    )
    parser.add_argument(
        "--enable-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable K-only CPU offload of cold Key blocks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        attention_config=AttentionConfig(
            backend=AttentionBackendEnum.ZOOMKV,
            zoomkv_sink_size=64,
            zoomkv_local_size=256,
            zoomkv_final_topk=100,
            zoomkv_quest_chunk=16,
            zoomkv_quest_large_chunk=256,
            zoomkv_full_attention_threshold=args.threshold,
            zoomkv_strict_kernels=args.strict_kernels,
            zoomkv_enable_offload=args.enable_offload,
        ),
    )
    load_s = time.perf_counter() - load_start
    sampling = SamplingParams(
        max_tokens=args.output_tokens,
        temperature=0.0,
        ignore_eos=True,
    )

    short_prompt = "What is the capital of France? Answer with one word."
    filler = "The capital of France is a well-known European city. " * args.repeat
    long_prompt = filler + short_prompt

    started = time.perf_counter()
    short_output = llm.generate([short_prompt], sampling)[0]
    dense_s = time.perf_counter() - started

    started = time.perf_counter()
    long_output = llm.generate([long_prompt], sampling)[0]
    sparse_s = time.perf_counter() - started

    result = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "threshold": args.threshold,
        "load_s": load_s,
        "dense": {
            "prompt_tokens": len(short_output.prompt_token_ids),
            "wall_s": dense_s,
            "text": short_output.outputs[0].text,
        },
        "sparse": {
            "prompt_tokens": len(long_output.prompt_token_ids),
            "wall_s": sparse_s,
            "text": long_output.outputs[0].text,
        },
    }
    if result["dense"]["prompt_tokens"] >= args.threshold:
        raise RuntimeError("Dense smoke prompt unexpectedly exceeds the threshold")
    if result["sparse"]["prompt_tokens"] < args.threshold:
        raise RuntimeError(
            "Sparse smoke prompt is shorter than --threshold; increase --repeat"
        )

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            output_file.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
