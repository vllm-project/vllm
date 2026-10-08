# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune or validate native per-token FP8 GEMM with transposed weights.

Example: .venv/bin/python benchmarks/kernels/benchmark_w8a8_per_token_fp8.py
    --shapes 4096,3840 3840,4096 --output-dir /tmp/per-token-tuning

Use --out-dtype float16 to tune FP16 output, --m to select batch sizes,
--resume to continue an interrupted sweep, or --validate-only to compare
installed configs against the default heuristic through the runtime custom op.

Configs are written to output-dir/configs. Copy them to
vllm/model_executor/layers/quantization/utils/configs after validation.
Tuning and independent finalist measurements use GPU graphs and hot operands;
quantization, attention, and tensor-parallel communication are excluded.
"""

import argparse
import hashlib
import itertools
import json
import random
import statistics
import time
from functools import partial
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    triton_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    get_w8a8_per_token_fp8_config,
    get_w8a8_per_token_fp8_config_filename,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.platform_utils import get_device_name_as_file_name


def candidates():
    keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages")
    return [
        dict(zip(keys, values))
        for values in itertools.product(
            (16, 32, 64, 128), (32, 64, 128), (64, 128, 256), (4, 8), (1, 2, 3)
        )
    ]


def call(a, b, sa, sb, config=None, out_dtype=torch.bfloat16):
    kwargs = {}
    if config is not None:
        kwargs = {
            "use_heuristic": False,
            "block_size_m": config["BLOCK_SIZE_M"],
            "block_size_n": config["BLOCK_SIZE_N"],
            "block_size_k": config["BLOCK_SIZE_K"],
            "num_warps": config["num_warps"],
            "num_stages": config["num_stages"],
        }
    return triton_scaled_mm(a, b, sa, sb, out_dtype, **kwargs)


def reference(a, w, sa, sb):
    rows = torch.linspace(0, a.shape[0] - 1, min(a.shape[0], 64), device=a.device)
    cols = torch.linspace(0, w.shape[0] - 1, min(w.shape[0], 128), device=w.device)
    rows, cols = rows.long().unique(), cols.long().unique()
    ref = (a[rows].float() @ w[cols].float().t()) * sa[rows] * sb[cols].t()
    return rows, cols, ref


def check(output, ref, rows, cols):
    for chunk in output.reshape(-1).split(16 * 1024 * 1024):
        if not torch.isfinite(chunk).all().item():
            raise ValueError("Non-finite output")
    sample = output[rows[:, None], cols[None, :]].float()
    error = ((sample - ref).square().mean() / ref.square().mean()).sqrt().item()
    if error >= 0.01:
        raise ValueError(f"Relative RMS error {error} >= 0.01")
    return error


def bench(fn, rep_ms):
    benchmark = (
        triton.testing.do_bench
        if current_platform.is_xpu()
        else triton.testing.do_bench_cudagraph
    )
    return 1000 * benchmark(fn, rep=rep_ms, return_mode="median")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", required=True, help="N,K pairs")
    parser.add_argument(
        "--m", "--batch-sizes", type=int, nargs="+", default=[2**i for i in range(16)]
    )
    parser.add_argument("--output-dir", "--save-path", type=Path, required=True)
    parser.add_argument(
        "--out-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--search-ms", type=int, default=3)
    parser.add_argument("--rep-ms", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Benchmark installed configs through the native custom op vs heuristic",
    )
    args = parser.parse_args()
    if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
        parser.error("Requires a CUDA, ROCm, or XPU device")
    if any(m <= 0 for m in args.m):
        parser.error("Batch sizes must be positive")
    out_dtype = getattr(torch, args.out_dtype)
    run_mm = partial(call, out_dtype=out_dtype)
    device = current_platform.device_type
    torch.accelerator.set_device_index(args.device)
    torch.set_num_threads(4)
    shapes = sorted({tuple(map(int, s.split(","))) for s in args.shapes})
    if any(len(shape) != 2 or min(shape) <= 0 for shape in shapes):
        parser.error("Shapes must be positive N,K pairs")
    cases = [(m, n, k) for n, k in shapes for m in sorted(set(args.m))]
    if not cases:
        parser.error("No cases to run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.jsonl"
    if result_path.exists() and not args.resume:
        raise FileExistsError("Choose a fresh output directory")
    search = [] if args.validate_only else candidates()
    if args.validate_only:
        import vllm.model_executor.kernels.linear.scaled_mm.triton  # noqa: F401
    metadata = {
        "device": str(getattr(torch, device).get_device_properties(args.device)),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "input_dtype": str(current_platform.fp8_dtype()),
        "output_dtype": args.out_dtype,
        "weight_layout": "contiguous [N,K] storage, transposed [K,N] view",
        "method": ("event timing" if current_platform.is_xpu() else "CUDA graphs")
        + ", hot operands, median of randomized rounds",
        "bias": False,
        "quantization_included": False,
        "correctness": "all elements finite; <=64 rows x128 columns vs FP32",
        "cases": cases,
        "search_space": search,
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "sources": {},
    }
    for source in (
        Path(__file__),
        Path(triton_scaled_mm.__code__.co_filename),
        Path(get_w8a8_per_token_fp8_config.__code__.co_filename),
    ):
        metadata["sources"][str(source)] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    complete = {}
    metadata_path = args.output_dir / "metadata.json"
    if args.resume and metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        for key in (
            "device",
            "torch",
            "hip",
            "triton",
            "sources",
            "cases",
        ):
            assert json.loads(json.dumps(metadata.get(key))) == old.get(key), key
        old_args, new_args = dict(old["arguments"]), dict(metadata["arguments"])
        old_args.pop("resume")
        new_args.pop("resume")
        assert old_args == new_args, "Resume arguments differ"
        for path in checkpoints.glob("*.json"):
            row = json.loads(path.read_text())
            case = (row["M"], row["N"], row["K"])
            assert case in cases and case not in complete, path
            complete[case] = row
        # Atomic checkpoints also recover a partially written JSONL tail.
        result_path.write_text(
            "".join(json.dumps(complete[c]) + "\n" for c in cases if c in complete)
        )
    metadata["completed_cases"] = len(complete)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    started = time.monotonic()
    configs = {}

    def save(record):
        name = f"M={record['M']}-N={record['N']}-K={record['K']}.json"
        path = checkpoints / name
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(record) + "\n")
        temp.replace(path)
        with result_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    if not args.validate_only:
        for (m, n, k), row in complete.items():
            filename = get_w8a8_per_token_fp8_config_filename(
                n,
                k,
                get_device_name_as_file_name(),
                current_platform.fp8_dtype(),
                out_dtype,
            )
            configs.setdefault(filename, {})[m] = row["config"]
        # Rebuild exports even if the interruption followed the final checkpoint.
        folder = args.output_dir / "configs"
        folder.mkdir(exist_ok=True)
        for filename, config in configs.items():
            (folder / filename).write_text(json.dumps(config, indent=4) + "\n")
    for index, case in enumerate(cases):
        if case in complete:
            continue
        m, n, k = case
        seed_index = index
        torch.manual_seed(1234 + seed_index)
        dtype = current_platform.fp8_dtype()
        a = (torch.randn(m, k, device=device) * 0.2).to(dtype)
        w = (torch.randn(n, k, device=device) * 0.2).to(dtype)
        b = w.t()
        sa = torch.rand(m, 1, device=device) + 0.5
        sb = torch.rand(n, 1, device=device) + 0.5
        rows, cols, ref = reference(a, w, sa, sb)
        if args.validate_only:
            config = get_w8a8_per_token_fp8_config(m, n, k, dtype, out_dtype)
            if config is None:
                raise ValueError(f"No installed tuned config for {case}")
            methods = {
                "heuristic": partial(run_mm, a, b, sa, sb),
                "tuned": partial(
                    torch.ops.vllm.w8a8_triton_per_token_scaled_mm_func,
                    a,
                    b,
                    sa,
                    sb,
                    out_dtype,
                    None,
                ),
            }
            errors = {
                name: check(fn(), ref, rows, cols) for name, fn in methods.items()
            }
            timings = {name: [] for name in methods}
            rng = random.Random(1234 + seed_index)
            for _ in range(args.rounds):
                names = list(methods)
                rng.shuffle(names)
                for name in names:
                    timings[name].append(bench(methods[name], args.rep_ms))
            record = {
                "M": m,
                "N": n,
                "K": k,
                "config": config,
                "relative_rms": errors,
                "samples_us": timings,
                "median_us": {
                    name: statistics.median(t) for name, t in timings.items()
                },
            }
            record["speedup"] = (
                record["median_us"]["heuristic"] / record["median_us"]["tuned"]
            )
            save(record)
            print(
                f"[{index + 1}/{len(cases)}] M={m} N={n} K={k}: "
                f"{record['median_us']} ({record['speedup']:.3f}x)",
                flush=True,
            )
            methods.clear()
            del a, w, b, sa, sb, rows, cols, ref
            torch.accelerator.empty_cache()
            continue
        ranked, skipped = [], []
        order = list(search)
        random.Random(1234 + seed_index).shuffle(order)
        print(f"[{index + 1}/{len(cases)}] searching M={m} N={n} K={k}", flush=True)
        for candidate in order:
            fn = partial(run_mm, a, b, sa, sb, candidate)
            try:
                error = check(fn(), ref, rows, cols)
                latency = bench(fn, args.search_ms)
            except (
                triton.runtime.errors.OutOfResources,
                triton.CompilationError,
            ) as exc:
                skipped.append({"config": candidate, "reason": str(exc)})
                continue
            ranked.append(
                {"config": candidate, "search_us": latency, "relative_rms": error}
            )
        ranked.sort(key=lambda r: r["search_us"])
        if not ranked:
            raise RuntimeError(f"No valid config for {case}")
        methods = {"heuristic": partial(run_mm, a, b, sa, sb)}
        for i, row in enumerate(ranked[: args.finalists]):
            methods[f"finalist_{i}"] = partial(run_mm, a, b, sa, sb, row["config"])
        timings = {name: [] for name in methods}
        rng = random.Random(9999 + seed_index)
        for _ in range(args.rounds):
            names = list(methods)
            rng.shuffle(names)
            for name in names:
                timings[name].append(bench(methods[name], args.rep_ms))
        winner = min(
            (name for name in methods if name != "heuristic"),
            key=lambda name: statistics.median(timings[name]),
        )
        config = ranked[int(winner.removeprefix("finalist_"))]["config"]
        # Keep the heuristic if the independent finalist measurements do not win.
        baseline_us = statistics.median(timings["heuristic"])
        tuned_us = statistics.median(timings[winner])
        if tuned_us >= baseline_us:
            tile = (
                (64, 64 if n < 8192 else 128, 256)
                if m <= 32
                else (64, 64, 256)
                if m <= 64
                else (64, 128, 128)
                if m <= 128
                else (128, 128, 128)
            )
            config = {
                "BLOCK_SIZE_M": tile[0],
                "BLOCK_SIZE_N": tile[1],
                "BLOCK_SIZE_K": tile[2],
                "num_warps": 4,
                "num_stages": 2 if current_platform.is_rocm() else 3,
            }
            tuned_us = baseline_us
        record = {
            "M": m,
            "N": n,
            "K": k,
            "config": config,
            "heuristic_us": baseline_us,
            "tuned_us": tuned_us,
            "speedup": baseline_us / tuned_us,
            "samples_us": timings,
            "ranked": ranked,
            "skipped": skipped,
            "relative_rms": check(run_mm(a, b, sa, sb, config), ref, rows, cols),
        }
        save(record)
        filename = get_w8a8_per_token_fp8_config_filename(
            n, k, get_device_name_as_file_name(), dtype, out_dtype
        )
        configs.setdefault(filename, {})[m] = config
        folder = args.output_dir / "configs"
        folder.mkdir(exist_ok=True)
        (folder / filename).write_text(json.dumps(configs[filename], indent=4) + "\n")
        print(
            f"M={m} N={n} K={k}: {baseline_us:.2f} -> {tuned_us:.2f} us "
            f"({baseline_us / tuned_us:.3f}x), {config}",
            flush=True,
        )
        methods.clear()
        del fn, a, w, b, sa, sb, rows, cols, ref
        torch.accelerator.empty_cache()
    metadata["elapsed_seconds"] = time.monotonic() - started
    metadata["completed_cases"] = len(cases)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
