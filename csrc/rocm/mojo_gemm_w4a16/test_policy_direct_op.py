#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vllm.model_executor.kernels.linear.mixed_precision import mojo_w4a16


def _accelerator_device() -> torch.device:
    return torch.accelerator.current_accelerator()


def _make_inputs(
    *,
    m: int,
    n: int,
    k: int,
    group_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(123)
    device = _accelerator_device()
    a = torch.randn((m, k), device=device, dtype=dtype)
    qweight = torch.randint(
        -(1 << 31),
        (1 << 31) - 1,
        (k, n // 8),
        device=device,
        dtype=torch.int32,
    )
    scales = torch.randn((k // group_size, n), device=device, dtype=dtype)
    return a, qweight, scales


def _dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=12288)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--expected-variant")
    parser.add_argument("--expected-policy-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dtype = _dtype(args.dtype)
    a, qweight, scales = _make_inputs(
        m=args.m,
        n=args.n,
        k=args.k,
        group_size=args.group_size,
        dtype=dtype,
    )

    cfg = mojo_w4a16._select_config_for_tensors(  # noqa: SLF001
        a, qweight, scales, None, args.group_size, 8, False
    )
    if args.expected_variant and cfg.variant != args.expected_variant:
        raise AssertionError(
            f"expected variant {args.expected_variant}, selected {cfg.variant}"
        )
    if args.expected_policy_file and not args.expected_policy_file.exists():
        raise AssertionError(
            f"expected policy file missing: {args.expected_policy_file}"
        )

    mojo_w4a16.prepare_mojo_w4a16_gemm(a, qweight, scales, None, args.group_size)
    out = mojo_w4a16.mojo_w4a16_gemm(a, qweight, scales, None, args.group_size)
    torch.accelerator.synchronize()

    if out.shape != (args.m, args.n):
        raise AssertionError(f"unexpected output shape: {tuple(out.shape)}")
    if out.dtype != dtype:
        raise AssertionError(f"unexpected output dtype: {out.dtype}")
    if not bool(torch.isfinite(out.float()).all().item()):
        raise AssertionError("output contains non-finite values")

    print(
        "direct_op:ok",
        f"cfg={cfg.key}",
        f"shape={tuple(out.shape)}",
        f"dtype={out.dtype}",
        f"sum={float(out.float().sum().item())}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
