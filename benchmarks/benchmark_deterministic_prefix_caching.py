#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark: Deterministic prefix caching overhead measurement.

Measures the wall-clock cost of splitting cache-miss prefills at block
boundaries versus computing all tokens in a single pass. This quantifies
the overhead introduced by --deterministic-prefix-caching.

The benchmark isolates three components:
  1. Raw GEMM decomposition cost: single GEMM(M=N) vs GEMM(M=N-R) + GEMM(M=R)
  2. Full transformer layer simulation (QKV + MLP projections)
  3. Throughput projection under varying cache hit rates
  4. Determinism validation confirming split-pass suffix matches cache-hit suffix

Usage:
  # Full benchmark suite
  python benchmarks/benchmark_deterministic_prefix_caching.py

  # GEMM microbenchmark only
  python benchmarks/benchmark_deterministic_prefix_caching.py --gemm-only

  # Layer simulation only
  python benchmarks/benchmark_deterministic_prefix_caching.py --layer-only

  # Determinism validation only
  python benchmarks/benchmark_deterministic_prefix_caching.py --determinism-only

  # Custom parameters
  python benchmarks/benchmark_deterministic_prefix_caching.py \
      --prompt-lengths 31 127 511 2048 --block-size 16 --iters 100

Requirements:
  pip install torch
"""

import argparse
import json
import statistics
import sys
from dataclasses import dataclass

import torch

DEFAULT_WARMUP = 5
DEFAULT_ITERS = 50
DEFAULT_BLOCK_SIZES = [16]
DEFAULT_PROMPT_LENGTHS = [31, 63, 127, 255, 511, 1023, 2048, 4095]

# (hidden_dim, intermediate_dim, num_layers, label)
MODEL_CONFIGS = [
    (1024, 3072, 28, "Qwen3-0.6B-scale"),
    (4096, 11008, 32, "Llama-3-8B-scale"),
    (8192, 28672, 80, "Llama-3-70B-scale"),
]


@dataclass
class SplitResult:
    prompt_len: int
    block_size: int
    remainder: int
    single_us: float
    single_std: float
    split_step1_us: float
    split_step2_us: float
    split_total_us: float
    split_std: float
    overhead_pct: float
    model_label: str


def bench_gemm_cuda_events(
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
) -> tuple[float, float]:
    """Benchmark a single GEMM using CUDA events. Returns (mean_us, std_us)."""
    device = torch.device("cuda")
    a = torch.randn(m, k, dtype=dtype, device=device)
    w = torch.randn(k, n, dtype=dtype, device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        torch.mm(a, w)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start.record()
        torch.mm(a, w)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)  # ms -> us

    del a, w
    torch.cuda.empty_cache()
    return statistics.mean(times), statistics.stdev(times)


def bench_split_gemm_cuda_events(
    m1: int,
    m2: int,
    k: int,
    n: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
) -> tuple[float, float, float, float]:
    """
    Benchmark two sequential GEMMs simulating the split.
    Returns (step1_mean_us, step2_mean_us, total_mean_us, total_std_us).
    """
    device = torch.device("cuda")
    a1 = torch.randn(m1, k, dtype=dtype, device=device)
    a2 = torch.randn(m2, k, dtype=dtype, device=device)
    w = torch.randn(k, n, dtype=dtype, device=device)

    s1_evt = torch.cuda.Event(enable_timing=True)
    mid_evt = torch.cuda.Event(enable_timing=True)
    e2_evt = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        torch.mm(a1, w)
        torch.mm(a2, w)
    torch.cuda.synchronize()

    step1_times: list[float] = []
    step2_times: list[float] = []
    total_times: list[float] = []
    for _ in range(iters):
        s1_evt.record()
        torch.mm(a1, w)
        mid_evt.record()
        torch.mm(a2, w)
        e2_evt.record()
        torch.cuda.synchronize()
        s1 = s1_evt.elapsed_time(mid_evt) * 1e3
        s2 = mid_evt.elapsed_time(e2_evt) * 1e3
        step1_times.append(s1)
        step2_times.append(s2)
        total_times.append(s1 + s2)

    del a1, a2, w
    torch.cuda.empty_cache()
    return (
        statistics.mean(step1_times),
        statistics.mean(step2_times),
        statistics.mean(total_times),
        statistics.stdev(total_times),
    )


def run_gemm_microbenchmark(
    model_configs: list[tuple[int, int, int, str]] | None = None,
    prompt_lengths: list[int] | None = None,
    block_sizes: list[int] | None = None,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
) -> list[SplitResult]:
    """Run the GEMM decomposition benchmark across model configs and prompt lengths."""
    if model_configs is None:
        model_configs = MODEL_CONFIGS
    if prompt_lengths is None:
        prompt_lengths = DEFAULT_PROMPT_LENGTHS
    if block_sizes is None:
        block_sizes = DEFAULT_BLOCK_SIZES

    results: list[SplitResult] = []

    for hidden, intermediate, num_layers, label in model_configs:
        print(f"\n{'=' * 80}")
        print(
            f"GEMM Microbenchmark: {label}  "
            f"(hidden={hidden}, intermediate={intermediate}, layers={num_layers})"
        )
        print(f"{'=' * 80}")
        print(
            f"{'Prompt':>7} {'BS':>4} {'R':>3} {'Single(µs)':>12} "
            f"{'Step1(µs)':>12} {'Step2(µs)':>12} {'Split(µs)':>12} "
            f"{'Overhead':>10} {'×layers(ms)':>12}"
        )
        print("-" * 100)

        for block_size in block_sizes:
            for prompt_len in prompt_lengths:
                remainder = prompt_len % block_size
                if remainder == 0:
                    single_us, single_std = bench_gemm_cuda_events(
                        prompt_len,
                        hidden,
                        intermediate,
                        warmup=warmup,
                        iters=iters,
                    )
                    results.append(
                        SplitResult(
                            prompt_len=prompt_len,
                            block_size=block_size,
                            remainder=0,
                            single_us=single_us,
                            single_std=single_std,
                            split_step1_us=0,
                            split_step2_us=0,
                            split_total_us=single_us,
                            split_std=single_std,
                            overhead_pct=0.0,
                            model_label=label,
                        )
                    )
                    print(
                        f"{prompt_len:>7} {block_size:>4} {remainder:>3} "
                        f"{single_us:>12.1f} {'—':>12} {'—':>12} {'—':>12} "
                        f"{'0% (aligned)':>10} {'—':>12}"
                    )
                    continue

                m_prefix = prompt_len - remainder
                m_suffix = remainder

                single_us, single_std = bench_gemm_cuda_events(
                    prompt_len,
                    hidden,
                    intermediate,
                    warmup=warmup,
                    iters=iters,
                )

                s1_us, s2_us, split_us, split_std = bench_split_gemm_cuda_events(
                    m_prefix,
                    m_suffix,
                    hidden,
                    intermediate,
                    warmup=warmup,
                    iters=iters,
                )

                overhead = ((split_us - single_us) / single_us) * 100
                layer_overhead_ms = (split_us - single_us) * num_layers / 1e3

                r = SplitResult(
                    prompt_len=prompt_len,
                    block_size=block_size,
                    remainder=remainder,
                    single_us=single_us,
                    single_std=single_std,
                    split_step1_us=s1_us,
                    split_step2_us=s2_us,
                    split_total_us=split_us,
                    split_std=split_std,
                    overhead_pct=overhead,
                    model_label=label,
                )
                results.append(r)

                print(
                    f"{prompt_len:>7} {block_size:>4} {remainder:>3} "
                    f"{single_us:>12.1f} {s1_us:>12.1f} {s2_us:>12.1f} "
                    f"{split_us:>12.1f} {overhead:>+9.1f}% "
                    f"{layer_overhead_ms:>+11.3f}"
                )

        print()

    return results


def _run_layer(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Simulate a single transformer layer's GEMM workload."""
    qkv = torch.mm(x, w_qkv)  # noqa: F841
    attn_out = torch.mm(x, w_o)
    residual = x + attn_out
    gate_up = torch.mm(residual, w_gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    mlp_out = torch.mm(activated, w_down)
    return residual + mlp_out


def bench_transformer_layer_split(
    prompt_len: int,
    hidden: int,
    intermediate: int,
    block_size: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
) -> tuple[float, float, float]:
    """
    Simulate a single transformer layer's GEMM workload.

    Models: QKV projection, output projection, gate+up projection, down
    projection. Attention itself is skipped (not affected by the split).

    Returns (single_pass_us, split_pass_us, overhead_pct).
    """
    device = torch.device("cuda")
    remainder = prompt_len % block_size

    w_qkv = torch.randn(hidden, 3 * hidden, dtype=dtype, device=device)
    w_o = torch.randn(hidden, hidden, dtype=dtype, device=device)
    w_gate_up = torch.randn(hidden, 2 * intermediate, dtype=dtype, device=device)
    w_down = torch.randn(intermediate, hidden, dtype=dtype, device=device)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # --- Single pass ---
    x_full = torch.randn(prompt_len, hidden, dtype=dtype, device=device)
    for _ in range(warmup):
        _run_layer(x_full, w_qkv, w_o, w_gate_up, w_down)
    torch.cuda.synchronize()

    single_times: list[float] = []
    for _ in range(iters):
        start_evt.record()
        _run_layer(x_full, w_qkv, w_o, w_gate_up, w_down)
        end_evt.record()
        torch.cuda.synchronize()
        single_times.append(start_evt.elapsed_time(end_evt) * 1e3)

    single_us = statistics.mean(single_times)

    if remainder == 0:
        del x_full, w_qkv, w_o, w_gate_up, w_down
        torch.cuda.empty_cache()
        return single_us, single_us, 0.0

    # --- Split pass ---
    m_prefix = prompt_len - remainder
    x_prefix = torch.randn(m_prefix, hidden, dtype=dtype, device=device)
    x_suffix = torch.randn(remainder, hidden, dtype=dtype, device=device)

    for _ in range(warmup):
        _run_layer(x_prefix, w_qkv, w_o, w_gate_up, w_down)
        _run_layer(x_suffix, w_qkv, w_o, w_gate_up, w_down)
    torch.cuda.synchronize()

    split_times: list[float] = []
    for _ in range(iters):
        start_evt.record()
        _run_layer(x_prefix, w_qkv, w_o, w_gate_up, w_down)
        _run_layer(x_suffix, w_qkv, w_o, w_gate_up, w_down)
        end_evt.record()
        torch.cuda.synchronize()
        split_times.append(start_evt.elapsed_time(end_evt) * 1e3)

    split_us = statistics.mean(split_times)
    overhead = ((split_us - single_us) / single_us) * 100

    del x_full, x_prefix, x_suffix, w_qkv, w_o, w_gate_up, w_down
    torch.cuda.empty_cache()

    return single_us, split_us, overhead


def run_layer_benchmark(
    model_configs: list[tuple[int, int, int, str]] | None = None,
    prompt_lengths: list[int] | None = None,
    block_size: int = 16,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
) -> list[dict]:
    """Full transformer layer simulation benchmark."""
    if model_configs is None:
        model_configs = MODEL_CONFIGS
    if prompt_lengths is None:
        prompt_lengths = DEFAULT_PROMPT_LENGTHS

    results: list[dict] = []

    for hidden, intermediate, num_layers, label in model_configs:
        print(f"\n{'=' * 80}")
        print(
            f"Layer simulation: {label}  "
            f"(hidden={hidden}, intermediate={intermediate}, layers={num_layers})"
        )
        print(f"{'=' * 80}")
        print(
            f"{'Prompt':>7} {'R':>3} {'Single/layer':>14} "
            f"{'Split/layer':>14} {'Overhead':>10} "
            f"{'Full model Δ':>14}"
        )
        print("-" * 70)

        for prompt_len in prompt_lengths:
            remainder = prompt_len % block_size
            single_us, split_us, overhead = bench_transformer_layer_split(
                prompt_len,
                hidden,
                intermediate,
                block_size,
                warmup=warmup,
                iters=iters,
            )
            model_delta_ms = (split_us - single_us) * num_layers / 1e3

            results.append(
                {
                    "model": label,
                    "prompt_len": prompt_len,
                    "remainder": remainder,
                    "single_layer_us": single_us,
                    "split_layer_us": split_us,
                    "overhead_pct": overhead,
                    "model_delta_ms": model_delta_ms,
                    "num_layers": num_layers,
                }
            )

            aligned = " (aligned)" if remainder == 0 else ""
            print(
                f"{prompt_len:>7} {remainder:>3} {single_us:>13.1f}µs "
                f"{split_us:>13.1f}µs {overhead:>+9.1f}% "
                f"{model_delta_ms:>+13.3f}ms{aligned}"
            )

        print()

    return results


def simulate_throughput_impact(
    gemm_results: list[SplitResult],
    hit_rates: list[float] | None = None,
    decode_tokens: int = 256,
    decode_token_time_us: float = 50.0,
) -> None:
    """Project throughput impact under various cache hit rates."""
    if hit_rates is None:
        hit_rates = [0.0, 0.5, 0.8, 0.95]

    print(f"\n{'=' * 100}")
    print("THROUGHPUT IMPACT PROJECTION")
    print(
        f"Decode tokens: {decode_tokens}, "
        f"per-token decode time: {decode_token_time_us:.0f}µs"
    )
    print(f"{'=' * 100}")

    models = sorted(set(r.model_label for r in gemm_results))

    for model in models:
        model_results = [r for r in gemm_results if r.model_label == model]
        print(f"\n--- {model} ---")

        header = f"{'Prompt':>7} {'R':>3} {'Prefill Δ':>10}"
        for hr in hit_rates:
            header += f" {'HR=' + str(int(hr * 100)) + '%':>12}"
        print(header)
        print("-" * (30 + 13 * len(hit_rates)))

        for r in model_results:
            if r.remainder == 0:
                row = f"{r.prompt_len:>7} {r.remainder:>3} {'0 (align)':>10}"
                for _ in hit_rates:
                    row += f" {'0.00%':>12}"
                print(row)
                continue

            prefill_delta_us = r.split_total_us - r.single_us
            decode_total_us = decode_tokens * decode_token_time_us
            prefill_us = r.single_us

            row = f"{r.prompt_len:>7} {r.remainder:>3} {prefill_delta_us:>+9.1f}µs"
            for hr in hit_rates:
                miss_rate = 1.0 - hr
                effective_delta = prefill_delta_us * miss_rate
                e2e_base = prefill_us + decode_total_us
                e2e_pct = (effective_delta / e2e_base) * 100
                row += f" {e2e_pct:>+11.3f}%"
            print(row)

    print()


def validate_determinism(
    hidden: int = 4096,
    intermediate: int = 11008,
    prompt_len: int = 127,
    block_size: int = 16,
    num_trials: int = 20,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Validate that split-pass suffix computation is bitwise identical to
    cache-hit suffix computation across trials, while single-pass suffix
    slicing may produce different bit patterns.
    """
    device = torch.device("cuda")
    remainder = prompt_len % block_size
    m_prefix = prompt_len - remainder

    print(f"\n{'=' * 80}")
    print("DETERMINISM VALIDATION")
    print(
        f"hidden={hidden}, intermediate={intermediate}, "
        f"prompt_len={prompt_len}, block_size={block_size}"
    )
    print(f"prefix M={m_prefix}, suffix M={remainder}")
    print(f"{'=' * 80}")

    # Fixed input and weights
    x_full = torch.randn(prompt_len, hidden, dtype=dtype, device=device)
    x_suffix_only = x_full[m_prefix:].clone()
    w = torch.randn(hidden, intermediate, dtype=dtype, device=device)

    # Single pass: extract suffix rows from full-M output
    single_outputs = []
    for _ in range(num_trials):
        out = torch.mm(x_full, w)
        single_outputs.append(out[m_prefix:].clone())

    # Split pass: compute suffix independently (M=R)
    split_outputs = []
    for _ in range(num_trials):
        out = torch.mm(x_suffix_only, w)
        split_outputs.append(out.clone())

    # Cache-hit simulation: also M=R, same as split
    cache_hit_outputs = []
    for _ in range(num_trials):
        out = torch.mm(x_suffix_only, w)
        cache_hit_outputs.append(out.clone())

    single_consistent = all(
        torch.equal(single_outputs[0], o) for o in single_outputs[1:]
    )
    split_consistent = all(torch.equal(split_outputs[0], o) for o in split_outputs[1:])
    cross_match = torch.equal(single_outputs[0], split_outputs[0])
    fix_match = all(torch.equal(split_outputs[0], o) for o in cache_hit_outputs)

    print(
        f"\nSingle-pass suffix (M={prompt_len}) consistency: "
        f"{'✓' if single_consistent else '✗'}"
    )
    print(
        f"Split-pass suffix  (M={remainder}) consistency:  "
        f"{'✓' if split_consistent else '✗'}"
    )
    print(
        f"Split == cache-hit (both M={remainder}):         "
        f"{'✓ bitwise identical' if fix_match else '✗ differs'}"
    )
    print(
        f"Single-pass == split-pass:                       "
        f"{'✓ match' if cross_match else '✗ differs (expected)'}"
    )

    if not cross_match:
        diff = (single_outputs[0] - split_outputs[0]).abs()
        num_diff = (diff > 0).sum().item()
        max_diff = diff.max().item()
        total = diff.numel()
        print(f"  → {num_diff}/{total} elements differ, max_diff={max_diff:.6f}")
        print(
            "  → This is the tiling divergence that "
            "--deterministic-prefix-caching eliminates."
        )

    del x_full, x_suffix_only, w
    torch.cuda.empty_cache()


def print_summary_table(gemm_results: list[SplitResult]) -> None:
    print(f"\n{'=' * 100}")
    print("SUMMARY: GEMM SPLIT OVERHEAD BY MODEL SCALE AND PROMPT LENGTH")
    print(f"{'=' * 100}")

    models = sorted(set(r.model_label for r in gemm_results))

    header = f"{'Prompt':>7}"
    for m in models:
        header += f" {m:>20}"
    print(header)
    print("-" * (8 + 21 * len(models)))

    prompt_lens = sorted(set(r.prompt_len for r in gemm_results))
    for pl in prompt_lens:
        row = f"{pl:>7}"
        for m in models:
            matches = [
                r for r in gemm_results if r.model_label == m and r.prompt_len == pl
            ]
            if matches:
                r = matches[0]
                if r.remainder == 0:
                    row += f" {'0% (aligned)':>20}"
                else:
                    row += f" {r.overhead_pct:>+19.1f}%"
            else:
                row += f" {'—':>20}"
        print(row)

    print()


def save_results(
    gemm_results: list[SplitResult],
    layer_results: list[dict],
    output_path: str,
) -> None:
    data = {
        "device": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A"
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "hip_version": getattr(torch.version, "hip", None) or "N/A",
        "gemm_results": [
            {
                "model": r.model_label,
                "prompt_len": r.prompt_len,
                "block_size": r.block_size,
                "remainder": r.remainder,
                "single_us": round(r.single_us, 2),
                "split_total_us": round(r.split_total_us, 2),
                "split_step1_us": round(r.split_step1_us, 2),
                "split_step2_us": round(r.split_step2_us, 2),
                "overhead_pct": round(r.overhead_pct, 2),
            }
            for r in gemm_results
        ],
        "layer_results": [
            {k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}
            for r in layer_results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark deterministic prefix caching overhead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gemm-only", action="store_true", help="Run only the raw GEMM microbenchmark"
    )
    parser.add_argument(
        "--layer-only", action="store_true", help="Run only the full-layer simulation"
    )
    parser.add_argument(
        "--determinism-only",
        action="store_true",
        help="Run only the determinism validation",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Custom prompt lengths to benchmark",
    )
    parser.add_argument(
        "--block-size", type=int, default=16, help="Block size (default: 16)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup iterations (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help=f"Benchmark iterations (default: {DEFAULT_ITERS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="deterministic_prefix_cache_benchmark.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this benchmark.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name()
    print(f"Device: {device_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda or 'N/A'}")
    hip_version = getattr(torch.version, "hip", None)
    if hip_version:
        print(f"ROCm/HIP: {hip_version}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")

    prompt_lengths = args.prompt_lengths or DEFAULT_PROMPT_LENGTHS
    block_sizes = [args.block_size]

    # ---- Determinism only ----
    if args.determinism_only:
        validate_determinism()
        return

    gemm_results: list[SplitResult] = []
    layer_results: list[dict] = []

    # ---- GEMM microbenchmark ----
    if not args.layer_only:
        gemm_results = run_gemm_microbenchmark(
            prompt_lengths=prompt_lengths,
            block_sizes=block_sizes,
            warmup=args.warmup,
            iters=args.iters,
        )

    # ---- Layer simulation ----
    if not args.gemm_only:
        layer_results = run_layer_benchmark(
            prompt_lengths=prompt_lengths,
            block_size=args.block_size,
            warmup=args.warmup,
            iters=args.iters,
        )

    # ---- Summary and projections ----
    if gemm_results:
        print_summary_table(gemm_results)
        simulate_throughput_impact(gemm_results)

    # ---- Determinism validation ----
    validate_determinism()

    # ---- Save ----
    save_results(gemm_results, layer_results, args.output)


if __name__ == "__main__":
    main()
