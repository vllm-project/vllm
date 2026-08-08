# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Speculative Decoding Analysis Benchmark
========================================
Profiles throughput, acceptance rate, and latency across different
speculation lengths (gamma) and draft model configurations.

Unlike the general throughput benchmark, this script is specifically
designed to characterise the *speculative decoding efficiency curve*:
how acceptance rate, tokens/s, and memory overhead change as gamma
(number of draft tokens per step) is swept from 1 to a configurable max.

Usage
-----
    python benchmarks/benchmark_spec_decode_analysis.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --speculative-model meta-llama/Llama-3.2-1B-Instruct \\
        --gamma-sweep 1 2 4 6 8 \\
        --num-prompts 200 \\
        --output-json spec_decode_results.json

    # Baseline without speculative decoding for comparison
    python benchmarks/benchmark_spec_decode_analysis.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --no-spec-decode \\
        --num-prompts 200

Output columns
--------------
gamma | acceptance_rate | tokens_per_sec | mean_latency_ms | p50_ms |
p95_ms | p99_ms | gpu_memory_gb | speedup_vs_baseline
"""

import argparse
import dataclasses
import json
import time
from typing import List, Optional

import numpy as np

try:
    from vllm import LLM, SamplingParams
    from vllm.spec_decode.metrics import SpecDecodeWorkerMetrics
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SpecDecodeResult:
    gamma: int
    acceptance_rate: float          # mean accepted draft tokens / gamma
    tokens_per_sec: float           # total output tokens / total wall time
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    gpu_memory_gb: float
    speedup_vs_baseline: Optional[float] = None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _load_prompts(num_prompts: int, input_len: int) -> List[str]:
    """Generate synthetic prompts of fixed token length."""
    # Repeating 'the' ~= 1 token per word; good enough for throughput tests.
    word = "the quick brown fox "
    prompt = (word * (input_len // len(word.split()) + 1)).strip()
    return [prompt] * num_prompts


def run_spec_decode_sweep(
    model: str,
    speculative_model: Optional[str],
    gamma_values: List[int],
    num_prompts: int,
    input_len: int,
    output_len: int,
    tensor_parallel_size: int,
    seed: int,
    trust_remote_code: bool,
    dtype: str,
    num_speculative_tokens_fallback: int,
) -> List[SpecDecodeResult]:
    """
    Run inference for each gamma value and collect metrics.

    Returns one SpecDecodeResult per gamma (plus gamma=0 as baseline
    if speculative_model is provided).
    """
    if not _VLLM_AVAILABLE:
        raise ImportError(
            "vLLM is required to run this benchmark. "
            "Install with: pip install vllm"
        )

    prompts = _load_prompts(num_prompts, input_len)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        seed=seed,
    )
    results: List[SpecDecodeResult] = []

    # -----------------------------------------------------------------
    # Baseline (no speculative decoding)
    # -----------------------------------------------------------------
    print("\n[baseline] Running without speculative decoding ...")
    llm_baseline = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        seed=seed,
    )
    latencies = []
    t0 = time.perf_counter()
    for prompt in prompts:
        t_start = time.perf_counter()
        llm_baseline.generate([prompt], sampling_params)
        latencies.append((time.perf_counter() - t_start) * 1000)
    total_time = time.perf_counter() - t0
    total_output_tokens = num_prompts * output_len

    baseline_tps = total_output_tokens / total_time
    baseline = SpecDecodeResult(
        gamma=0,
        acceptance_rate=1.0,
        tokens_per_sec=baseline_tps,
        mean_latency_ms=float(np.mean(latencies)),
        p50_latency_ms=float(np.percentile(latencies, 50)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        gpu_memory_gb=_get_gpu_memory_gb(),
        speedup_vs_baseline=1.0,
    )
    results.append(baseline)
    print(
        f"  [baseline] {baseline_tps:.1f} tok/s  "
        f"p50={baseline.p50_latency_ms:.0f}ms"
    )
    del llm_baseline

    if speculative_model is None:
        return results

    # -----------------------------------------------------------------
    # Sweep gamma values
    # -----------------------------------------------------------------
    for gamma in sorted(gamma_values):
        print(f"\n[gamma={gamma}] Loading speculative engine ...")
        llm = LLM(
            model=model,
            speculative_model=speculative_model,
            num_speculative_tokens=gamma,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            seed=seed,
        )

        latencies = []
        t0 = time.perf_counter()
        for prompt in prompts:
            t_start = time.perf_counter()
            llm.generate([prompt], sampling_params)
            latencies.append((time.perf_counter() - t_start) * 1000)
        total_time = time.perf_counter() - t0
        tps = total_output_tokens / total_time

        # Acceptance rate from vLLM internal metrics (if available)
        acc_rate = _get_acceptance_rate(llm) or _estimate_acceptance_rate(
            tps, baseline_tps, gamma
        )

        gpu_mem = _get_gpu_memory_gb()
        speedup = tps / baseline_tps

        result = SpecDecodeResult(
            gamma=gamma,
            acceptance_rate=acc_rate,
            tokens_per_sec=tps,
            mean_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            gpu_memory_gb=gpu_mem,
            speedup_vs_baseline=speedup,
        )
        results.append(result)
        print(
            f"  [gamma={gamma}] {tps:.1f} tok/s  "
            f"speedup={speedup:.2f}x  "
            f"accept={acc_rate:.2%}  "
            f"p50={result.p50_latency_ms:.0f}ms"
        )
        del llm

    return results


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_gpu_memory_gb() -> float:
    """Return current GPU memory usage in GB (best-effort)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _get_acceptance_rate(llm) -> Optional[float]:
    """
    Attempt to read acceptance rate from vLLM's internal spec decode stats.
    Returns None if unavailable (older vLLM versions).
    """
    try:
        stats = llm.llm_engine.stat_logger
        if hasattr(stats, "spec_decode_metrics"):
            m: SpecDecodeWorkerMetrics = stats.spec_decode_metrics
            if m is not None and m.draft_acceptance_rate is not None:
                return float(m.draft_acceptance_rate)
    except Exception:
        pass
    return None


def _estimate_acceptance_rate(
    observed_tps: float, baseline_tps: float, gamma: int
) -> float:
    """
    Back-calculate approximate acceptance rate from observed speedup.

    Uses the draft-model-free approximation (valid when draft << target cost):
        speedup ≈ 1 + alpha * gamma

    Solving for alpha:
        alpha = (speedup - 1) / gamma
    """
    speedup = observed_tps / baseline_tps
    alpha = (speedup - 1.0) / gamma
    return max(0.0, min(1.0, alpha))


# ---------------------------------------------------------------------------
# Output / reporting
# ---------------------------------------------------------------------------

def print_summary_table(results: List[SpecDecodeResult]) -> None:
    header = (
        f"{'gamma':>5}  {'tok/s':>8}  {'speedup':>8}  "
        f"{'accept':>7}  {'p50ms':>7}  {'p95ms':>7}  {'p99ms':>7}  {'GPU GB':>7}"
    )
    print("\n" + "=" * len(header))
    print("Speculative Decoding Analysis Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        label = "baseline" if r.gamma == 0 else str(r.gamma)
        speedup_str = f"{r.speedup_vs_baseline:.2f}x" if r.speedup_vs_baseline else "1.00x"
        print(
            f"{label:>5}  {r.tokens_per_sec:>8.1f}  {speedup_str:>8}  "
            f"{r.acceptance_rate:>7.2%}  {r.p50_latency_ms:>7.0f}  "
            f"{r.p95_latency_ms:>7.0f}  {r.p99_latency_ms:>7.0f}  "
            f"{r.gpu_memory_gb:>7.2f}"
        )
    print("=" * len(header))


def save_json(results: List[SpecDecodeResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump([r.as_dict() for r in results], f, indent=2)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Target (verifier) model name or path.",
    )
    parser.add_argument(
        "--speculative-model",
        type=str,
        default=None,
        help="Draft model name or path. If omitted, only the baseline is run.",
    )
    parser.add_argument(
        "--gamma-sweep",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8],
        metavar="GAMMA",
        help="List of speculation lengths (num_speculative_tokens) to sweep. "
             "Default: 1 2 4 6 8",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=200,
        help="Number of prompts to benchmark per gamma value. Default: 200",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=128,
        help="Approximate number of input tokens per prompt. Default: 128",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Number of output tokens to generate per prompt. Default: 128",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel degree. Default: 1",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype. Default: auto",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Trust remote code when loading the model.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save results as JSON. Default: not saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Model:            {args.model}")
    print(f"Draft model:      {args.speculative_model or '(none — baseline only)'}")
    print(f"Gamma sweep:      {args.gamma_sweep}")
    print(f"Num prompts:      {args.num_prompts}")
    print(f"Input / output:   {args.input_len} / {args.output_len} tokens")
    print(f"Tensor parallel:  {args.tensor_parallel_size}")
    print(f"Dtype:            {args.dtype}")

    results = run_spec_decode_sweep(
        model=args.model,
        speculative_model=args.speculative_model,
        gamma_values=args.gamma_sweep,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        num_speculative_tokens_fallback=max(args.gamma_sweep),
    )

    print_summary_table(results)

    if args.output_json:
        save_json(results, args.output_json)


if __name__ == "__main__":
    main()
