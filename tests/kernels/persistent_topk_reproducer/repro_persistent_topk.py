#!/usr/bin/env python3
import argparse
import json
import math
import os
import time

import torch

try:
    import vllm._C  # noqa: F401
except ModuleNotFoundError:
    import vllm._C_stable_libtorch  # noqa: F401


WORKSPACE_BYTES = 1 << 20


def ordered_fp16_coarse_bin(values: torch.Tensor) -> torch.Tensor:
    bits = values.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    ordered = torch.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits | 0x8000)
    return ordered >> 8


def threshold_bin_population(logits: torch.Tensor, k: int) -> int:
    kth = torch.topk(logits, k, dim=1).values[:, -1]
    bins = ordered_fp16_coarse_bin(logits)
    kth_bins = ordered_fp16_coarse_bin(kth).unsqueeze(1)
    return int((bins == kth_bins).sum(dim=1).max().item())


def load_backend(name: str):
    if name == "persistent":
        return torch.ops._C.persistent_topk
    if name == "cooperative":
        return torch.ops._C.cooperative_topk
    if name == "exact-extension":
        from torch.utils.cpp_extension import load

        source = os.environ["EXACT_TOPK_SOURCE"]
        include = os.environ["EXACT_TOPK_INCLUDE"]
        module = load(
            name="vllm_51782_exact_topk",
            sources=[source],
            extra_include_paths=[include],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=True,
        )
        return module.exact_persistent_topk
    if name == "overflow-extension":
        from torch.utils.cpp_extension import load

        source = os.environ["PATCHED_TOPK_SOURCE"]
        include = os.environ["PATCHED_TOPK_INCLUDE"]
        module = load(
            name=os.environ.get(
                "PATCHED_TOPK_MODULE", "vllm_51782_overflow_topk_v2"
            ),
            sources=[source],
            extra_include_paths=[include],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=True,
        )
        return module.patched_persistent_topk
    raise ValueError(f"unknown backend: {name}")


def run_case(logits: torch.Tensor, k: int, repeats: int, op) -> dict:
    rows, width = logits.shape
    lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    output = torch.full((rows, k), -1, dtype=torch.int32, device="cuda")
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")

    op(logits, lengths, output, workspace, k, width)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        output.fill_(-1)
        workspace.zero_()
        op(logits, lengths, output, workspace, k, width)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / repeats

    invalid = int(((output < 0) | (output >= width)).sum().item())
    duplicate_rows = int(
        sum(torch.unique(output[row]).numel() != k for row in range(rows))
    )
    safe_output = output.clamp(0, width - 1).long()
    selected = torch.gather(logits, 1, safe_output)
    expected = torch.topk(logits, k, dim=1).values
    selected = torch.sort(selected, dim=1, descending=True).values
    expected = torch.sort(expected, dim=1, descending=True).values
    mismatches = selected != expected
    mismatch_per_row = mismatches.sum(dim=1)
    delta = (selected - expected).abs()

    return {
        "rows": rows,
        "width": width,
        "k": k,
        "threshold_bin_population": threshold_bin_population(logits, k),
        "bad_rows": int((mismatch_per_row > 0).sum().item()),
        "max_mismatches": int(mismatch_per_row.max().item()),
        "max_abs_delta": float(delta.max().item()),
        "invalid_indices": invalid,
        "rows_with_duplicates": duplicate_rows,
        "mean_kernel_ms": round(elapsed_ms, 6),
    }


def normal_case(rows: int, width: int, sigma: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(rows, width, generator=generator) * sigma + 10.0).cuda()


def single_bin_case(rows: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    upper = math.nextafter(0.245, 0.0)
    return (1.0 + torch.rand(rows, width, generator=generator) * upper).cuda()


def fine_bin_case(rows: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    upper = math.nextafter(0.01, 0.0)
    return (1.0 + torch.rand(rows, width, generator=generator) * upper).cuda()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--backend",
        choices=(
            "persistent",
            "cooperative",
            "exact-extension",
            "overflow-extension",
        ),
        default="persistent",
    )
    args = parser.parse_args()
    op = load_backend(args.backend)

    device = torch.cuda.get_device_properties(0)
    print(json.dumps({
        "environment": {
            "torch": torch.__version__,
            "vllm": __import__("vllm").__version__,
            "gpu": device.name,
            "compute_capability": f"{device.major}.{device.minor}",
            "total_memory_bytes": device.total_memory,
            "backend": args.backend,
        }
    }))

    cases = [
        ("normal", 64, 16384, 2048, 0.1),
        ("normal", 64, 32768, 2048, 0.1),
        ("normal", 64, 131072, 2048, 0.1),
        ("single_bin", 33, 16384, 2048, None),
        ("single_bin", 33, 16385, 2048, None),
    ]
    if args.full:
        cases = [
            ("normal", rows, width, k, sigma)
            for k in (512, 1024, 2048)
            for rows, width in ((64, 16384), (33, 32768), (64, 32768), (64, 131072))
            for sigma in (1.0, 0.3, 0.1, 0.03)
        ] + [
            ("single_bin", 33, width, k, None)
            for k in (512, 1024, 2048)
            for width in (16383, 16384, 16385, 16512)
        ] + [
            ("fine_bin", rows, width, k, None)
            for k in (512, 1024, 2048)
            for rows, width in (
                (1, 4096),
                (8, 8192),
                (8, 12288),
                (8, 32768),
                (32, 32768),
                (33, 8192),
                (33, 16384),
                (33, 32768),
                (33, 32769),
            )
        ]

    failures = 0
    for kind, rows, width, k, sigma in cases:
        if kind == "normal":
            logits = normal_case(rows, width, float(sigma), seed=args.seed)
        elif kind == "single_bin":
            logits = single_bin_case(rows, width, seed=args.seed)
        else:
            logits = fine_bin_case(rows, width, seed=args.seed)
        result = run_case(logits, k, args.repeats, op)
        result.update({"distribution": kind, "sigma": sigma})
        failures += int(result["bad_rows"] > 0)
        print(json.dumps(result, sort_keys=True))
        del logits
        torch.cuda.empty_cache()

    print(json.dumps({
        "summary": {
            "cases": len(cases),
            "failing_cases": failures,
            "seed": args.seed,
        }
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
