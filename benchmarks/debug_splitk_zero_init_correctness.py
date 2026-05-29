#!/usr/bin/env python3
"""Direct correctness checks for BlockScale SplitK zero-init fusion.

This script exercises the same custom-op contract used by
BlockScaleSplitKZeroInitFusionPass:

  producer_with_zero_init(..., gemm_out_zero_init=Y)
  gemm_a8w8_blockscale_splitk(..., output=Y, y_is_zeroed=True)

It checks that the producer really zeros Y and that the fused path matches the
functional producer + functional blockscale GEMM path.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable

os.environ.setdefault("VLLM_TARGET_DEVICE", "rocm")

import torch

from vllm._aiter_ops import rocm_aiter_ops


GROUP_SIZE = 128
EPS = 1e-6


@dataclass
class CaseResult:
    producer: str
    m: int
    n: int
    k: int
    zero_nonzero_count: int
    producer_fp8_equal: bool
    producer_scale_max_abs: float
    fused_gemm_max_abs: float
    fused_gemm_mean_abs: float
    passed: bool


def _sync() -> None:
    torch.cuda.synchronize()


def _make_inputs(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    return x, w_bf16


def _quantize_weight(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.rocm_aiter_group_fp8_quant(w_bf16, GROUP_SIZE)


def _functional_group_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.rocm_aiter_group_fp8_quant(x, GROUP_SIZE)


def _zero_init_group_quant(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.rocm_aiter_group_fp8_quant_with_zero_init(
        x, y, GROUP_SIZE
    )


def _functional_rmsnorm_quant(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant(
        x, weight, EPS, GROUP_SIZE
    )


def _zero_init_rmsnorm_quant(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant_with_zero_init(
        x, y, weight, EPS, GROUP_SIZE
    )


def _functional_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
        a, b, a_scale, b_scale, torch.bfloat16
    )


def _splitk_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale_splitk(
        a, b, a_scale, b_scale, output, torch.bfloat16, 0, True
    )
    return output


def _run_case(
    producer: str,
    m: int,
    n: int,
    k: int,
    functional_producer: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    zero_init_producer: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    producer_extra_arg: torch.Tensor | None = None,
) -> CaseResult:
    x, w_bf16 = _make_inputs(m, n, k)
    b, b_scale = _quantize_weight(w_bf16)
    _sync()

    if producer_extra_arg is None:
        a_ref, a_scale_ref = functional_producer(x)
    else:
        a_ref, a_scale_ref = functional_producer(x, producer_extra_arg)
    ref = _functional_gemm(a_ref, b, a_scale_ref, b_scale)
    _sync()

    y = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    y.fill_(123.0)
    _sync()

    if producer_extra_arg is None:
        a_fused, a_scale_fused = zero_init_producer(x, y)
    else:
        a_fused, a_scale_fused = zero_init_producer(x, y, producer_extra_arg)
    _sync()
    zero_nonzero_count = int(torch.count_nonzero(y).item())

    fused = _splitk_gemm(a_fused, b, a_scale_fused, b_scale, y)
    _sync()

    diff = (fused.float() - ref.float()).abs()
    scale_diff = (a_scale_fused.float() - a_scale_ref.float()).abs()
    fp8_equal = bool(torch.equal(a_fused, a_ref))
    scale_max_abs = float(scale_diff.max().item()) if scale_diff.numel() else 0.0
    fused_max_abs = float(diff.max().item()) if diff.numel() else 0.0
    fused_mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    passed = zero_nonzero_count == 0 and fp8_equal and fused_max_abs == 0.0

    return CaseResult(
        producer=producer,
        m=m,
        n=n,
        k=k,
        zero_nonzero_count=zero_nonzero_count,
        producer_fp8_equal=fp8_equal,
        producer_scale_max_abs=scale_max_abs,
        fused_gemm_max_abs=fused_max_abs,
        fused_gemm_mean_abs=fused_mean_abs,
        passed=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", nargs="+", type=int, default=[1, 2, 4, 8, 16, 64])
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.k % GROUP_SIZE != 0:
        raise ValueError(f"k must be divisible by {GROUP_SIZE}, got {args.k}")

    torch.manual_seed(args.seed)
    rocm_aiter_ops.register_ops_once()

    results: list[CaseResult] = []
    rms_weight = torch.ones((args.k,), device="cuda", dtype=torch.bfloat16)

    for m in args.m:
        results.append(
            _run_case(
                "group_fp8_quant",
                m,
                args.n,
                args.k,
                _functional_group_quant,
                _zero_init_group_quant,
            )
        )
        results.append(
            _run_case(
                "rmsnorm_fp8_group_quant",
                m,
                args.n,
                args.k,
                _functional_rmsnorm_quant,
                _zero_init_rmsnorm_quant,
                rms_weight,
            )
        )

    payload = [asdict(result) for result in results]
    print(json.dumps(payload, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
