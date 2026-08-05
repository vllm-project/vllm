#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark dense fallback or ZoomKV on an identical long prompt.

The headline number is *decode latency*, isolated from prefill. vLLM's
``generate()`` always runs a full prefill+decode, and with
``enable_prefix_caching=False`` every call re-runs prefill, so a single
wall-clock reading is dominated by the (large) prefill of a long prompt.

To recover decode-only cost we measure wall time at several output-token
counts and linear-fit ``T(N) = prefill_overhead + N * decode_per_token``.
The slope is the per-token decode latency (prefill, tokenization and
scheduler fixed costs live in the intercept and cancel out). Over a short
sweep the context length changes by <1%, so decode cost is effectively flat
and the fit is accurate; the reported slope is the average decode cost across
the sweep range.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from vllm import LLM, SamplingParams
from vllm.config.attention import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("dense", "sparse"), required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--prompt-repeats", type=int, default=700)
    parser.add_argument(
        "--decode-sweep",
        type=str,
        default="4,16,32,64",
        help=(
            "Comma-separated output-token counts. Wall time is measured at each "
            "count and linear-fit to isolate per-token decode latency."
        ),
    )
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threshold", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--enable-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable K-only CPU offload of cold Key blocks.",
    )
    parser.add_argument(
        "--cpu-bytes-per-rank",
        type=int,
        default=8 * 1024**3,
        help="Pinned host Key pool budget per rank when offload is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep = sorted({int(x) for x in args.decode_sweep.split(",") if x.strip()})
    if len(sweep) < 2:
        raise ValueError("--decode-sweep needs at least two distinct token counts")

    prompt = (
        "The capital of France is a well-known European city. " * args.prompt_repeats
    ) + "What is the capital of France? Answer with one word."

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
            zoomkv_full_attention_threshold=args.threshold,
            zoomkv_dense_fallback=args.mode == "dense",
            zoomkv_enable_offload=args.enable_offload,
            zoomkv_cpu_bytes_per_rank=args.cpu_bytes_per_rank,
        ),
    )

    warmup = llm.generate(
        [prompt],
        SamplingParams(
            max_tokens=args.warmup_output_tokens,
            temperature=0.0,
            ignore_eos=True,
        ),
    )[0]
    prompt_tokens = len(warmup.prompt_token_ids)
    if prompt_tokens + sweep[-1] > args.max_model_len:
        raise RuntimeError(
            f"prompt_tokens({prompt_tokens}) + max decode({sweep[-1]}) "
            f"exceeds max_model_len({args.max_model_len})"
        )

    def timed_generate(n_tokens: int) -> dict:
        sampling = SamplingParams(
            max_tokens=n_tokens,
            temperature=0.0,
            ignore_eos=True,
        )
        walls: list[float] = []
        produced = n_tokens
        text = ""
        for _ in range(args.runs):
            started = time.perf_counter()
            output = llm.generate([prompt], sampling)[0]
            walls.append(time.perf_counter() - started)
            produced = len(output.outputs[0].token_ids)
            text = output.outputs[0].text
        return {
            "requested_tokens": n_tokens,
            "produced_tokens": produced,
            "runs_wall_s": walls,
            "median_wall_s": statistics.median(walls),
            "text": text,
        }

    points = [timed_generate(n) for n in sweep]

    # Fit against *actually produced* token counts. ignore_eos keeps them equal
    # to the request, but stay honest if a model ever emits fewer.
    xs = [p["produced_tokens"] for p in points]
    ys = [p["median_wall_s"] for p in points]
    fit = statistics.linear_regression(xs, ys)
    decode_s_per_tok = fit.slope
    prefill_overhead_s = fit.intercept

    # Two-point cross-check between the smallest and largest sweep points.
    lo, hi = points[0], points[-1]
    dtok = hi["produced_tokens"] - lo["produced_tokens"]
    two_point_decode_s = (
        (hi["median_wall_s"] - lo["median_wall_s"]) / dtok if dtok > 0 else float("nan")
    )

    result = {
        "mode": args.mode,
        "prompt_tokens": prompt_tokens,
        "final_context_len": prompt_tokens + sweep[-1],
        "decode_sweep": sweep,
        "runs_per_point": args.runs,
        "points": points,
        # Isolated decode latency (prefill / tokenization / scheduler cancel out).
        "decode_ms_per_token_fit": decode_s_per_tok * 1e3,
        "decode_tok_s_fit": (
            (1.0 / decode_s_per_tok) if decode_s_per_tok > 0 else float("nan")
        ),
        "decode_ms_per_token_two_point": two_point_decode_s * 1e3,
        "prefill_overhead_s_fit": prefill_overhead_s,
        # Prefill-inclusive throughput at the largest point (NOT decode-only).
        "e2e_tok_s_at_max": hi["produced_tokens"] / hi["median_wall_s"],
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            output_file.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
