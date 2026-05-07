#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generic benchmark harness for vLLM IR ops.

Usage:
    python benchmarks/kernels/ir/bench_ir_ops.py
    python benchmarks/kernels/ir/bench_ir_ops.py --ops rms_norm
    python benchmarks/kernels/ir/bench_ir_ops.py --ops rms_norm,silu_mul
    python benchmarks/kernels/ir/bench_ir_ops.py --no-cuda-graph
    python benchmarks/kernels/ir/bench_ir_ops.py --ops rms_norm --save-path ./results/
"""

import argparse
import contextlib
import csv
import dataclasses
import datetime
import math
import os
import subprocess
import sys
import tempfile

# Ensure repo root is on sys.path so `benchmarks` is importable as a package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Suppress noisy C++ warnings from vllm kernel registration (written to fd 2
# directly by the dynamic linker, so Python-level sys.stderr redirect won't
# catch them).
_saved_fd = os.dup(2)
try:
    with open(os.devnull, "w") as _devnull:
        os.dup2(_devnull.fileno(), 2)
        import torch

        import vllm.kernels  # noqa: E402, F401
finally:
    os.dup2(_saved_fd, 2)
    os.close(_saved_fd)

from tqdm import tqdm  # noqa: E402

from benchmarks.kernels.ir.shapes import SHAPE_CONFIGS  # noqa: E402  # isort: skip
from vllm.ir.op import IrOp  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402


@dataclasses.dataclass(frozen=True)
class BenchConfig:
    use_cuda_graph: bool = True
    warmup: int = 25
    rep: int = 100


def _pkg_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    with contextlib.suppress(PackageNotFoundError):
        return version(name)
    return "not installed"


_METADATA_LABELS = {
    "timestamp": "Timestamp",
    "git_commit": "Git commit",
    "vllm": "vLLM",
    "pytorch": "PyTorch",
    "cuda_runtime": "CUDA runtime",
    "triton": "Triton",
    "cutlass": "CUTLASS",
    "helion": "Helion",
    "device": "Device",
    "bench_mode": "Bench mode",
    "warmup": "Warmup",
    "rep": "Repetitions",
}


def collect_env_metadata(cfg: BenchConfig) -> dict[str, str]:
    from vllm.collect_env import get_env_info

    env = get_env_info()

    git_sha = "unknown"
    with contextlib.suppress(subprocess.CalledProcessError, FileNotFoundError):
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

    device_name = current_platform.get_device_name()

    warmup_note = " ms" if not cfg.use_cuda_graph else " ms (ignored)"
    rep_note = " replays" if cfg.use_cuda_graph else " ms"

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_sha,
        "vllm": str(env.vllm_version),
        "pytorch": str(env.torch_version),
        "cuda_runtime": str(env.cuda_runtime_version),
        "triton": triton.__version__,
        "cutlass": _pkg_version("nvidia-cutlass-dsl"),
        "helion": _pkg_version("helion"),
        "device": device_name,
        "bench_mode": "cuda_graph" if cfg.use_cuda_graph else "eager",
        "warmup": f"{cfg.warmup}{warmup_note}",
        "rep": f"{cfg.rep}{rep_note}",
    }


def print_metadata(metadata: dict[str, str]):
    print("=" * 60)
    for key, val in metadata.items():
        print(f"{_METADATA_LABELS.get(key, key) + ':':<16}{val}")
    print("=" * 60)


def _clone_args(args: tuple) -> tuple:
    return tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)


# TODO(gmagogsfm): When the `maybe_inplace` PR lands, ops marked as
# inplace=True will mutate bench_args across iterations. Both CUDA graph
# and eager modes will accumulate drift from repeated in-place mutation.
# We need to re-clone inputs per iteration for inplace ops.
def _bench_one(fn, args, cfg: BenchConfig) -> float:
    bench_args = _clone_args(args)
    bench_fn = lambda: fn(*bench_args)

    if cfg.use_cuda_graph:
        ms = triton.testing.do_bench_cudagraph(bench_fn, rep=cfg.rep, quantiles=[0.5])
    else:
        ms = triton.testing.do_bench(
            bench_fn, warmup=cfg.warmup, rep=cfg.rep, quantiles=[0.5]
        )
    return ms * 1000


# TODO(gmagogsfm): Once compiled native implementation lands (#38775),
# the benchmark baseline should be the compiled native (what vLLM runs by
# default) rather than the uncompiled native implementation.
def collect_timings(
    op: IrOp, shape_configs: list[dict], cfg: BenchConfig
) -> tuple[list[str], list[str], dict[str, dict[str, float]]]:
    def fmt(v) -> str:
        return str(v).split(".")[-1] if isinstance(v, torch.dtype) else str(v)

    case_names = [
        "_".join(f"{k}={fmt(v)}" for k, v in kwargs.items()) for kwargs in shape_configs
    ]
    providers = [n for n, impl in op.impls.items() if impl.supported]

    results: dict[str, dict[str, float]] = {c: {} for c in case_names}
    for provider in providers:
        impl = op.impls[provider]
        desc = f"{op.name} / {provider}"
        for case_name, kwargs in tqdm(
            zip(case_names, shape_configs),
            desc=desc,
            total=len(case_names),
            unit=" cases",
        ):
            args = op.generate_inputs(**kwargs)
            if impl.supports_args(*args):
                results[case_name][provider] = _bench_one(impl.impl_fn, args, cfg)
            else:
                results[case_name][provider] = float("nan")

    return case_names, providers, results


def analyze_results(
    op_name: str,
    case_names: list[str],
    providers: list[str],
    results: dict[str, dict[str, float]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    native_col = "native"
    non_native = [p for p in providers if p != native_col]

    header_cols = ["case"]
    for p in providers:
        header_cols.append(f"{p} (us)")
    for p in non_native:
        header_cols.append(f"{p} speedup")

    detail_rows: list[dict[str, str]] = []
    speedup_data: dict[str, list[tuple[float, str]]] = {p: [] for p in non_native}

    for case_name in case_names:
        timings = results[case_name]
        row: dict[str, str] = {"case": case_name}

        for p in providers:
            val = timings.get(p, float("nan"))
            row[f"{p} (us)"] = f"{val:.2f}" if not math.isnan(val) else "n/a"

        native_us = timings.get(native_col, float("nan"))
        for p in non_native:
            p_us = timings.get(p, float("nan"))
            if not math.isnan(native_us) and not math.isnan(p_us) and p_us > 0:
                speedup = native_us / p_us
                row[f"{p} speedup"] = f"{speedup:.2f}x"
                speedup_data[p].append((speedup, case_name))
            else:
                row[f"{p} speedup"] = "n/a"

        detail_rows.append(row)

    summary_rows: list[dict[str, str]] = []
    for p in non_native:
        entries = speedup_data[p]
        if not entries:
            continue
        speedups = [s for s, _ in entries]
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        best_val, best_case = max(entries)
        worst_val, worst_case = min(entries)
        wins = sum(1 for s in speedups if s > 1.0)
        losses = sum(1 for s in speedups if s < 1.0)
        total = len(speedups)

        print(f"\n{p} vs native ({wins}/{total} faster, {losses}/{total} slower):")
        print(f"  geomean speedup: {geomean:.2f}x")
        print(f"  best:            {best_val:.2f}x  ({best_case})")
        print(f"  worst:           {worst_val:.2f}x  ({worst_case})")

        summary_rows.append(
            {
                "op": op_name,
                "provider": p,
                "geomean_speedup": f"{geomean:.2f}",
                "best_speedup": f"{best_val:.2f}",
                "best_case": best_case,
                "worst_speedup": f"{worst_val:.2f}",
                "worst_case": worst_case,
                "wins": str(wins),
                "losses": str(losses),
                "total": str(total),
            }
        )

    return detail_rows, summary_rows, header_cols


def write_csv(path: str, rows: list[dict[str, str]], fieldnames: list[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results(
    save_dir: str,
    op_name: str,
    detail_rows: list[dict[str, str]],
    header_cols: list[str],
    all_summary_rows: list[dict[str, str]],
    metadata: dict[str, str],
):
    write_csv(
        os.path.join(save_dir, f"{op_name}_detail.csv"),
        detail_rows,
        header_cols,
    )
    if all_summary_rows:
        write_csv(
            os.path.join(save_dir, "summary.csv"),
            all_summary_rows,
            list(all_summary_rows[0].keys()),
        )
    write_csv(
        os.path.join(save_dir, "metadata.csv"),
        [metadata],
        list(metadata.keys()),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark vLLM IR ops")
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Comma-separated list of op names to benchmark (substring match)",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA graph; use do_bench with L2 cache flushing instead",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup time in ms (do_bench) or ignored with CUDA graph (default: 25)",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Repetition time in ms (do_bench) or number of graph replays "
        "(do_bench_cudagraph) (default: 100)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory to save results (default: auto-created temp dir)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = BenchConfig(
        use_cuda_graph=not args.no_cuda_graph,
        warmup=args.warmup,
        rep=args.rep,
    )

    torch.set_default_device(current_platform.device_type)

    metadata = collect_env_metadata(cfg)
    print_metadata(metadata)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_path or os.path.join(
        tempfile.gettempdir(), f"vllm_ir_bench_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)

    op_filters = [f.strip() for f in args.ops.split(",")] if args.ops else None
    all_summary_rows: list[dict[str, str]] = []

    for op in IrOp.registry.values():
        if op_filters and not any(f in op.name for f in op_filters):
            continue
        if not op.has_input_generator:
            print(f"Skipping op '{op.name}': no input generator registered")
            continue
        if op.name not in SHAPE_CONFIGS:
            raise RuntimeError(
                f"No benchmark shape config for op '{op.name}'. "
                f"Add it to benchmarks/kernels/ir/shapes.py"
            )

        case_names, providers, results = collect_timings(
            op, SHAPE_CONFIGS[op.name], cfg
        )
        detail_rows, summary_rows, header_cols = analyze_results(
            op.name, case_names, providers, results
        )
        all_summary_rows.extend(summary_rows)

        save_results(
            save_dir,
            op.name,
            detail_rows,
            header_cols,
            all_summary_rows,
            metadata,
        )

    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
