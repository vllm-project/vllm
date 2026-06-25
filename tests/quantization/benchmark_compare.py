#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comparison benchmark: baseline (bf16) vs TurboQuant vs OSCAR INT2 KV-cache.

Runs two measurement passes per config:
  1. Throughput  -- synthetic generations, reports tok/s and GPU memory
  2. Perplexity  -- wikitext-2-raw-v1 (first 50 passages), reports PPL

Usage:
    python tests/quantization/benchmark_compare.py \
        --model Qwen/Qwen3-32B \
        --gpu-memory-utilization 0.92 \
        --batch-size 8 \
        --input-len 512 \
        --output-len 128

Run just one backend (skip others):
    --backends baseline turboquant   # any subset
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KV-cache quant comparison benchmark")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument(
        "--backends",
        nargs="+",
        default=["baseline", "turboquant", "oscar"],
        choices=["baseline", "turboquant", "oscar"],
        help="Which backends to benchmark (default: all three)",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type for model weights and activations",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--bench-iters", type=int, default=5)
    # PPL settings
    p.add_argument(
        "--ppl-passages",
        type=int,
        default=50,
        help="Number of wikitext passages for perplexity (0 = skip)",
    )
    p.add_argument(
        "--ppl-max-len",
        type=int,
        default=1024,
        help="Max token length per passage for PPL",
    )
    # OSCAR rotation files (optional, generated dummy if missing)
    p.add_argument(
        "--oscar-k-rotation", default=None, help="Path to K rotation .pt file"
    )
    p.add_argument(
        "--oscar-v-rotation", default=None, help="Path to V rotation .pt file"
    )
    p.add_argument(
        "--turboquant-dtype",
        default="turboquant_k3v4_nc",
        help="TurboQuant kv_cache_dtype to use",
    )
    p.add_argument(
        "--oscar-dtype", default="oscar_int2", help="OSCAR kv_cache_dtype to use"
    )
    p.add_argument(
        "--output-json",
        default="benchmark_compare_results.json",
        help="Where to write JSON results",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------


@dataclass
class BackendResult:
    backend: str
    kv_cache_dtype: str
    gpu_memory_used_gb: float
    kv_cache_tokens: int
    throughput_tok_s: float
    latency_p50_ms: float
    latency_p99_ms: float
    perplexity: float | None
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_rotation(head_dim: int, num_heads: int, path: str) -> None:
    """Generate a dummy identity rotation and save it."""
    R = torch.eye(head_dim).unsqueeze(0).expand(num_heads, -1, -1)
    torch.save(R, path)
    print(f"  [dummy rotation] saved {R.shape} → {path}")


def _setup_oscar_env(
    args: argparse.Namespace, head_dim: int = 128, num_heads: int = 8
) -> None:
    """Ensure OSCAR rotation env vars are set."""
    k_path = args.oscar_k_rotation or "/tmp/oscar_dummy_k.pt"
    v_path = args.oscar_v_rotation or "/tmp/oscar_dummy_v.pt"
    if not Path(k_path).exists():
        _dummy_rotation(head_dim, num_heads, k_path)
    if not Path(v_path).exists():
        _dummy_rotation(head_dim, num_heads, v_path)
    os.environ["VLLM_OSCAR_K_ROTATION_PATH"] = k_path
    os.environ["VLLM_OSCAR_V_ROTATION_PATH"] = v_path


def _gpu_memory_used_gb() -> float:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / (1024**3)
    return 0.0


def _build_prompts(batch_size: int, input_len: int) -> list[str]:
    """Simple repeating-word prompts of roughly `input_len` tokens."""
    word = "the"
    # ~1 token per word on average for simple English
    return [(" ".join([word] * input_len)) for _ in range(batch_size)]


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------


def run_throughput(
    llm,
    batch_size: int,
    input_len: int,
    output_len: int,
    warmup_iters: int,
    bench_iters: int,
) -> dict:
    from vllm import SamplingParams

    prompts = _build_prompts(batch_size, input_len)
    sp = SamplingParams(max_tokens=output_len, temperature=0.0)

    print(f"  Warming up ({warmup_iters} iters)...")
    for _ in range(warmup_iters):
        llm.generate(prompts, sp)

    print(f"  Benchmarking ({bench_iters} iters)...")
    latencies = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    total_output_tokens = sum(
        len(o.outputs[0].token_ids) for out in [outputs] for o in out
    )
    total_tokens_per_iter = total_output_tokens + batch_size * input_len
    avg_latency_ms = sum(latencies) / len(latencies)
    throughput = total_tokens_per_iter / (avg_latency_ms / 1000)

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

    return {
        "throughput_tok_s": round(throughput, 1),
        "latency_p50_ms": round(p50, 1),
        "latency_p99_ms": round(p99, 1),
    }


# ---------------------------------------------------------------------------
# Perplexity benchmark
# ---------------------------------------------------------------------------


def run_perplexity(llm, num_passages: int, max_len: int) -> float | None:
    """Compute perplexity on wikitext-2 using vLLM's logprobs."""
    if num_passages <= 0:
        return None

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "  [ppl] datasets not installed, skipping perplexity. pip install datasets"
        )
        return None

    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        print(f"  [ppl] Could not load wikitext: {e}")
        return None

    from vllm import SamplingParams

    texts = [row["text"] for row in ds if len(row["text"].strip()) > 100]
    texts = texts[:num_passages]
    if not texts:
        print("  [ppl] No valid passages found.")
        return None

    # vLLM logprobs: request 1 logprob per token (the token itself)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0)

    total_nll = 0.0
    total_tokens = 0

    print(f"  Computing PPL on {len(texts)} passages...")
    for i, text in enumerate(texts):
        try:
            out = llm.generate([text[: max_len * 4]], sp)  # rough char limit
            result = out[0]
            if result.prompt_logprobs is None:
                continue
            # prompt_logprobs is List[Optional[Dict[token_id, Logprob]]]
            # None for the first token, then one entry per subsequent token
            nlls = []
            for tok_logprobs in result.prompt_logprobs:
                if tok_logprobs is None:
                    continue
                # The top entry is the actual token
                logp = next(iter(tok_logprobs.values())).logprob
                nlls.append(-logp)

            if nlls:
                total_nll += sum(nlls)
                total_tokens += len(nlls)
        except Exception as e:
            print(f"  [ppl] passage {i} failed: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  [ppl] {i + 1}/{len(texts)} passages done...")

    if total_tokens == 0:
        return None

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return round(ppl, 2)


# ---------------------------------------------------------------------------
# Per-backend runner
# ---------------------------------------------------------------------------

BACKEND_CONFIGS = {
    "baseline": {"kv_cache_dtype": "auto"},
    "turboquant": {},  # dtype filled from args at runtime
    "oscar": {},  # dtype filled from args at runtime
}


def run_backend(
    name: str, kv_cache_dtype: str, args: argparse.Namespace
) -> BackendResult:
    print(f"\n{'=' * 60}")
    print(f"  Backend: {name.upper()}  |  kv_cache_dtype={kv_cache_dtype}")
    print(f"{'=' * 60}")

    if name == "oscar":
        _setup_oscar_env(args)

    from vllm import LLM

    extra_kwargs: dict[str, Any] = {}
    if kv_cache_dtype != "auto":
        extra_kwargs["kv_cache_dtype"] = kv_cache_dtype

    # OSCAR now uses a fused Triton decode attention kernel, so it supports
    # CUDA Graphs and no longer requires eager execution.
    enforce_eager = False
    if "enforce_eager" in extra_kwargs:
        enforce_eager = extra_kwargs.pop("enforce_eager")

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        disable_log_stats=True,
        enforce_eager=enforce_eager,
        **extra_kwargs,
    )

    kv_tokens = 0
    try:
        # V1 engine: cache_config is on vllm_config
        vllm_cfg = llm.llm_engine.vllm_config
        cache_cfg = vllm_cfg.cache_config
        num_blocks = getattr(cache_cfg, "num_gpu_blocks", 0)
        block_size = getattr(cache_cfg, "block_size", 16)
        if num_blocks > 0:
            kv_tokens = num_blocks * block_size
    except Exception:
        pass

    torch.accelerator.synchronize()
    gpu_mem_gb = _gpu_memory_used_gb()

    # Throughput
    tput = run_throughput(
        llm,
        args.batch_size,
        args.input_len,
        args.output_len,
        args.warmup_iters,
        args.bench_iters,
    )

    # Perplexity
    ppl = run_perplexity(llm, args.ppl_passages, args.ppl_max_len)

    del llm
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()
    import gc

    gc.collect()
    import ray

    if ray.is_initialized():
        ray.shutdown()
    torch.accelerator.empty_cache()

    return BackendResult(
        backend=name,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_used_gb=round(gpu_mem_gb, 2),
        kv_cache_tokens=kv_tokens,
        perplexity=ppl,
        **tput,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    dtype_map = {
        "baseline": "auto",
        "turboquant": args.turboquant_dtype,
        "oscar": args.oscar_dtype,
    }

    results: list[BackendResult] = []

    for backend in args.backends:
        dtype = dtype_map[backend]
        try:
            result = run_backend(backend, dtype, args)
        except Exception as exc:
            import traceback

            print(f"\n[ERROR] Backend '{backend}' failed:\n{traceback.format_exc()}")
            result = BackendResult(
                backend=backend,
                kv_cache_dtype=dtype,
                gpu_memory_used_gb=0.0,
                kv_cache_tokens=0,
                throughput_tok_s=0.0,
                latency_p50_ms=0.0,
                latency_p99_ms=0.0,
                perplexity=None,
                error=str(exc),
            )
        results.append(result)

    # -----------------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print(f"  RESULTS  |  Model: {args.model}")
    print("=" * 80)
    header = (
        f"{'Backend':<12} {'KV dtype':<22} {'GPU Mem GB':>10} "
        f"{'KV Tokens':>12} {'Tput tok/s':>12} {'P50 ms':>8} {'PPL':>8}"
    )
    print(header)
    print("-" * 80)

    baseline_tput = None
    for r in results:
        if r.backend == "baseline" and r.throughput_tok_s > 0:
            baseline_tput = r.throughput_tok_s

    for r in results:
        speedup = ""
        if baseline_tput and r.throughput_tok_s > 0 and r.backend != "baseline":
            ratio = r.throughput_tok_s / baseline_tput
            speedup = f"  ({ratio:.2f}x)"
        ppl_str = f"{r.perplexity:.1f}" if r.perplexity else "N/A"
        err_str = f"  !! {r.error[:40]}" if r.error else ""
        print(
            f"{r.backend:<12} {r.kv_cache_dtype:<22} {r.gpu_memory_used_gb:>10.1f} "
            f"{r.kv_cache_tokens:>12,} {r.throughput_tok_s:>12.1f}{speedup} "
            f"{r.latency_p50_ms:>8.1f} {ppl_str:>8}{err_str}"
        )

    print("=" * 80)

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    out = {
        "model": args.model,
        "config": {
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "bench_iters": args.bench_iters,
            "ppl_passages": args.ppl_passages,
        },
        "results": [asdict(r) for r in results],
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to: {args.output_json}")


if __name__ == "__main__":
    main()
