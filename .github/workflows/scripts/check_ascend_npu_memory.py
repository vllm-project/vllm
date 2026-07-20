#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys


GIB = 1024**3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail fast when an Ascend device lacks memory for vLLM startup."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--utilization", type=float, required=True)
    parser.add_argument("--insufficient-exit-code", type=int, default=87)
    return parser.parse_args()


def _gib(value: int) -> float:
    return value / GIB


def main() -> int:
    args = _parse_args()
    if not 0 < args.utilization <= 1:
        print(
            f"Ascend NPU memory utilization must be in (0, 1], got {args.utilization}",
            file=sys.stderr,
        )
        return 2
    if not 1 <= args.insufficient_exit_code <= 255:
        print("Ascend NPU insufficient exit code must be in [1, 255]", file=sys.stderr)
        return 2

    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as error:
        print(f"Ascend NPU memory preflight import failed: {error}", file=sys.stderr)
        return 1

    try:
        if not torch.npu.is_available():
            raise RuntimeError("torch.npu.is_available() returned False")
        torch.npu.set_device(args.device)
        torch.zeros(1, device=args.device)
    except Exception as error:
        print(
            "Ascend NPU allocation preflight failed: "
            f"device={args.device} error={error}",
            file=sys.stderr,
        )
        return 1

    mem_get_info = getattr(torch.npu, "mem_get_info", None)
    if not callable(mem_get_info):
        print(
            "ASCEND_NPU_MEMORY_STATUS=unavailable "
            f"device={args.device} utilization={args.utilization:.4f} "
            "fallback=allocation-only"
        )
        return 0

    try:
        free_bytes, total_bytes = mem_get_info()
        free_bytes = int(free_bytes)
        total_bytes = int(total_bytes)
    except Exception as error:
        print(
            "ASCEND_NPU_MEMORY_STATUS=unavailable "
            f"device={args.device} utilization={args.utilization:.4f} "
            f"fallback=allocation-only error={type(error).__name__}"
        )
        return 0

    if free_bytes < 0 or total_bytes <= 0 or free_bytes > total_bytes:
        print(
            "Ascend NPU memory preflight returned invalid values: "
            f"device={args.device} free_bytes={free_bytes} total_bytes={total_bytes}",
            file=sys.stderr,
        )
        return 1

    required_bytes = int(total_bytes * args.utilization)
    fields = (
        f"device={args.device} free_gib={_gib(free_bytes):.2f} "
        f"total_gib={_gib(total_bytes):.2f} "
        f"required_gib={_gib(required_bytes):.2f} "
        f"utilization={args.utilization:.4f}"
    )
    if free_bytes < required_bytes:
        print(f"ASCEND_NPU_MEMORY_STATUS=insufficient {fields}", file=sys.stderr)
        return args.insufficient_exit_code

    print(f"ASCEND_NPU_MEMORY_STATUS=ok {fields}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
