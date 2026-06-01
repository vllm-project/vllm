# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: weighted vs weightless vLLM RMSNorm CUDA kernels.

Targets FlashNorm issue #41430 justification — compares
torch.ops._C.rms_norm / fused_add_rms_norm against their weightless siblings
when the weight tensor is all-ones (mathematically equivalent, less work).

Example (A100, bf16, Llama-scale hidden size):

    .venv/bin/python benchmarks/kernels/benchmark_rmsnorm_weightless.py \\
        --hidden-size 4096 \\
        --tokens 1 16 128 1024 \\
        --dtype bfloat16 \\
        --use-residual

Requires a CUDA build with weightless kernels (not VLLM_USE_PRECOMPILED).
"""

from __future__ import annotations

import argparse
import statistics

import torch

from vllm import _custom_ops as ops


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. Run on an NVIDIA GPU (e.g. A100).")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")


def _require_weightless_ops() -> None:
    if not hasattr(torch.ops._C, "rms_norm_weightless"):
        raise RuntimeError(
            "torch.ops._C.rms_norm_weightless not found. "
            "Rebuild from source: uv pip install -e . --torch-backend=auto"
        )


def _bench_us(
    fn,
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # ms -> us per iter


def _bench_rms_norm(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    *,
    use_residual: bool,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    device = "cuda"
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    residual = torch.randn_like(x) if use_residual else None
    eps = 1e-5

    if use_residual:
        x_w = x.clone()
        r_w = residual.clone()  # type: ignore[union-attr]
        x_l = x.clone()
        r_l = residual.clone()  # type: ignore[union-attr]

        weighted_us = _bench_us(
            lambda: ops.fused_add_rms_norm(x_w, r_w, weight, eps),
            warmup=warmup,
            iters=iters,
        )
        weightless_us = _bench_us(
            lambda: ops.fused_add_rms_norm_weightless(x_l, r_l, eps),
            warmup=warmup,
            iters=iters,
        )
    else:
        out_w = torch.empty_like(x)
        out_l = torch.empty_like(x)

        weighted_us = _bench_us(
            lambda: ops.rms_norm(out_w, x, weight, eps),
            warmup=warmup,
            iters=iters,
        )
        weightless_us = _bench_us(
            lambda: ops.rms_norm_weightless(out_l, x, eps),
            warmup=warmup,
            iters=iters,
        )

    speedup_pct = 100.0 * (weighted_us / weightless_us - 1.0)
    return weighted_us, weightless_us, speedup_pct


def _check_correctness(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    *,
    use_residual: bool,
) -> None:
    device = "cuda"
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    residual = torch.randn_like(x) if use_residual else None
    eps = 1e-5

    if use_residual:
        x_w, r_w = x.clone(), residual.clone()  # type: ignore[union-attr]
        x_l, r_l = x.clone(), residual.clone()  # type: ignore[union-attr]
        ops.fused_add_rms_norm(x_w, r_w, weight, eps)
        ops.fused_add_rms_norm_weightless(x_l, r_l, eps)
        torch.testing.assert_close(x_w, x_l, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(r_w, r_l, atol=1e-2, rtol=1e-2)
    else:
        out_w = torch.empty_like(x)
        out_l = torch.empty_like(x)
        ops.rms_norm(out_w, x, weight, eps)
        ops.rms_norm_weightless(out_l, x, eps)
        torch.testing.assert_close(out_w, out_l, atol=1e-2, rtol=1e-2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark weighted vs weightless vLLM RMSNorm kernels."
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 16, 128, 1024],
        help="num_tokens values to sweep (decode batch sizes)",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--use-residual",
        action="store_true",
        help="Benchmark fused_add_rms_norm (Llama decoder path)",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Repeat each config and report median us",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    _require_cuda()
    _require_weightless_ops()

    mode = "fused_add_rms_norm" if args.use_residual else "rms_norm"
    print(f"mode={mode}, dtype={args.dtype}, hidden_size={args.hidden_size}")
    print(f"warmup={args.warmup}, iters={args.iters}, trials={args.trials}")
    print()

    # Quick correctness on largest token count
    _check_correctness(
        max(args.tokens),
        args.hidden_size,
        dtype,
        use_residual=args.use_residual,
    )
    print("correctness: weighted(ones) == weightless  OK")
    print()

    header = (
        f"{'tokens':>8}  {'weighted_us':>12}  {'weightless_us':>14}  "
        f"{'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    for num_tokens in args.tokens:
        trial_weighted: list[float] = []
        trial_weightless: list[float] = []
        for _ in range(args.trials):
            w_us, l_us, _ = _bench_rms_norm(
                num_tokens,
                args.hidden_size,
                dtype,
                use_residual=args.use_residual,
                warmup=args.warmup,
                iters=args.iters,
            )
            trial_weighted.append(w_us)
            trial_weightless.append(l_us)

        weighted_us = statistics.median(trial_weighted)
        weightless_us = statistics.median(trial_weightless)
        speedup_pct = 100.0 * (weighted_us / weightless_us - 1.0)
        print(
            f"{num_tokens:8d}  {weighted_us:12.2f}  {weightless_us:14.2f}  "
            f"{speedup_pct:+7.2f}%"
        )


if __name__ == "__main__":
    main()
