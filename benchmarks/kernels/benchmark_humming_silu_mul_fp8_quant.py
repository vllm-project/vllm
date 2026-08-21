# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark Humming-compatible SwiGLU + per-token FP8 quantization."""

import argparse

import torch

from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func
from vllm.triton_utils import triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routed-rows",
        type=int,
        nargs="+",
        default=[6, 48, 96, 192, 384, 768, 1536, 3072, 6144],
        help="Rows seen by the activation kernel (input tokens x DSV4 top-k 6)",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[2048, 3072],
        help="DSV4-Flash and DSV4-Pro routed-expert intermediate sizes",
    )
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    return parser.parse_args()


def benchmark_case(
    routed_rows: int,
    hidden_size: int,
    swiglu_limit: float | None,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[float, float]:
    from humming import ops as humming_ops

    input = torch.randn(
        (routed_rows, hidden_size * 2),
        dtype=torch.bfloat16,
        device="cuda",
    )
    activation = torch.empty(
        (routed_rows, hidden_size),
        dtype=input.dtype,
        device=input.device,
    )
    baseline_output = torch.empty_like(activation, dtype=torch.float8_e4m3fn)
    fused_output = torch.empty_like(baseline_output)

    def baseline() -> None:
        if swiglu_limit is None:
            torch.ops._C.silu_and_mul(activation, input)
        else:
            swiglu_limit_func(activation, input, swiglu_limit)
        humming_ops.quant_input(
            inputs=activation,
            outputs=baseline_output,
            dtype="float8e4m3",
            group_size=None,
            scale_dtype="float32",
        )

    def fused() -> None:
        scale = torch.empty((routed_rows, 1), dtype=torch.float32, device=input.device)
        torch.ops._C.silu_and_mul_per_token_quant(
            fused_output,
            input,
            scale,
            None,
            swiglu_limit,
            1e-30,
            True,
        )

    baseline_ms = triton.testing.do_bench(
        baseline,
        warmup=warmup_ms,
        rep=rep_ms,
        return_mode="median",
    )
    fused_ms = triton.testing.do_bench(
        fused,
        warmup=warmup_ms,
        rep=rep_ms,
        return_mode="median",
    )
    return baseline_ms * 1000, fused_ms * 1000


def main() -> None:
    args = parse_args()
    import humming

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU: {properties.name}, SM {properties.major}.{properties.minor}")
    print(f"Humming: {humming.__file__}")
    print(f"SwiGLU limit: {args.swiglu_limit}")
    print("M,H,baseline_us,fused_us,speedup,saved_bf16_bytes")

    for hidden_size in args.hidden_sizes:
        for routed_rows in args.routed_rows:
            baseline_us, fused_us = benchmark_case(
                routed_rows=routed_rows,
                hidden_size=hidden_size,
                swiglu_limit=args.swiglu_limit,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
            saved_bytes = routed_rows * hidden_size * torch.bfloat16.itemsize
            print(
                f"{routed_rows},{hidden_size},{baseline_us:.3f},"
                f"{fused_us:.3f},{baseline_us / fused_us:.3f},"
                f"{saved_bytes}"
            )


if __name__ == "__main__":
    main()
