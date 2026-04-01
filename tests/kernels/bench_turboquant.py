# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark TurboQuant vs baseline throughput."""

import time

import torch

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = "Explain the theory of general relativity in detail."


def benchmark(kv_cache_dtype="auto", label="baseline"):
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    llm = LLM(
        MODEL,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
    )
    params = SamplingParams(max_tokens=128, temperature=0.0)
    llm.generate([PROMPT], params)  # warmup
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([PROMPT] * 8, params)
    torch.accelerator.synchronize()
    t1 = time.perf_counter()
    toks = sum(len(o.outputs[0].token_ids) for o in out)
    tp = toks / (t1 - t0)
    print(f"  {toks} tokens in {t1 - t0:.2f}s = {tp:.1f} tok/s")
    print(f"  Sample: {out[0].outputs[0].text[:80]}...")
    del llm
    torch.accelerator.empty_cache()
    return tp


if __name__ == "__main__":
    t1 = benchmark("auto", "BASELINE (bf16)")
    t2 = benchmark("turboquant", "TURBOQUANT (4-bit)")
    print(f"\nRatio: {t2 / t1:.2f}x")
