# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare compiled RowWise and native Triton over tuned N/K shapes and large M."""

import argparse
import gc
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
from benchmark_rdna4_fp8_linear import (
    discover_cases,
    make_call,
    make_kernel,
    sampled_reference,
)

from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonPerTokenFp8ScaledMMLinearKernel,
)
from vllm.triton_utils import triton


def fp8_input(rows, columns):
    result = torch.empty((rows, columns), device="cuda", dtype=torch.float8_e4m3fn)
    chunk_rows = max(1, 4_194_304 // columns)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        chunk = torch.randn(end - start, columns, device="cuda")
        result[start:end].copy_(chunk.mul_(0.2))
    return result


def check_output(output, rows, cols, reference):
    chunk_rows = max(1, 4_194_304 // output.shape[1])
    for start in range(0, output.shape[0], chunk_rows):
        if not torch.isfinite(output[start : start + chunk_rows]).all().item():
            raise ValueError("Nonfinite output")
    sample = output[rows[:, None], cols[None, :]].float()
    error = (sample - reference).square().mean() / reference.square().mean()
    error = error.sqrt().item()
    if error >= 0.01:
        raise ValueError(f"Relative RMS {error} >= 0.01")
    return error


def graph_bench(fn, rep_ms):
    # Release the ordinary allocator's large output before creating a graph pool.
    # This keeps the 15-GiB output cases from requiring two resident output pools.
    with torch.cuda.stream(torch.cuda.Stream()):
        fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            fn()
        end.record()
        torch.accelerator.synchronize()
        estimate_ms = start.elapsed_time(end) / 5
        repeats = max(1, int(rep_ms / max(estimate_ms, 0.001)))
        torch.accelerator.empty_cache()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                fn()
        torch.accelerator.synchronize()
        samples = []
        for _ in range(10):
            start.record()
            graph.replay()
            end.record()
            torch.accelerator.synchronize()
            samples.append(1000 * start.elapsed_time(end) / repeats)
        return statistics.median(samples)


def validate_call(fn, shape, rows, cols, reference):
    output = fn()
    assert output.shape == shape and output.dtype == torch.bfloat16
    return check_output(output, rows, cols, reference)


def benchmark_case(case, index, rounds, rep_ms):
    m, n, k, filename = case
    torch.manual_seed(1234 + index)
    a, w = fp8_input(m, k), fp8_input(n, k)
    sa = torch.rand(m, 1, device="cuda") + 0.5
    sb = torch.rand(n, 1, device="cuda") + 0.5
    rowwise = make_kernel("torch", n, k)
    native = make_kernel("triton", n, k)
    assert isinstance(rowwise, RowWiseTorchFP8ScaledMMLinearKernel)
    assert isinstance(native, TritonPerTokenFp8ScaledMMLinearKernel)
    methods = {
        "rowwise_compiled": torch.compile(
            make_call(rowwise, a, w.t(), sa, sb), fullgraph=True
        ),
        "native_triton": make_call(native, a, w.t(), sa, sb),
    }
    rows, cols, reference = sampled_reference(a, w, sa, sb)
    errors, failures = {}, {}
    for name in list(methods):
        try:
            errors[name] = validate_call(methods[name], (m, n), rows, cols, reference)
            for _ in range(3):
                methods[name]()
            torch.accelerator.synchronize()
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if "illegal memory" in str(exc).lower():
                raise
            failures[name] = str(exc)
            methods.pop(name)
            gc.collect()
            torch.accelerator.empty_cache()
    timings = {name: [] for name in methods}
    rng = random.Random(1234 + index)
    for _ in range(rounds):
        order = list(methods)
        rng.shuffle(order)
        for name in order:
            try:
                timings[name].append(graph_bench(methods[name], rep_ms))
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if "illegal memory" in str(exc).lower():
                    raise
                failures[name] = str(exc)
                methods.pop(name)
                timings.pop(name)
            finally:
                gc.collect()
                torch.accelerator.empty_cache()
    return {
        "M": m,
        "N": n,
        "K": k,
        "config_file": filename,
        "relative_rms": errors,
        "failures": failures,
        "samples_us": timings,
        "median_us": {name: statistics.median(t) for name, t in timings.items()},
        "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-m", type=int, default=262144)
    parser.add_argument("--min-m", type=int, default=1)
    parser.add_argument("--n", type=int, nargs="+")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep-ms", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if (
        any(value < 1 or value & (value - 1) for value in (args.min_m, args.max_m))
        or args.min_m > args.max_m
    ):
        parser.error("M bounds must be ordered positive powers of two")
    tuned_cases, files, excluded = discover_cases(args.config_dir, None)
    shapes = sorted({(n, k, filename) for _, n, k, filename in tuned_cases})
    if args.n is not None:
        shapes = [shape for shape in shapes if shape[0] in args.n]
    ms = [
        1 << power
        for power in range(args.min_m.bit_length() - 1, args.max_m.bit_length())
    ]
    cases = [(m, n, k, filename) for m in ms for n, k, filename in shapes]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases or args.rounds < 1 or args.rep_ms < 1:
        parser.error("Nonempty cases and positive rounds/rep-ms are required")
    if args.check:
        print(
            json.dumps(
                {
                    "cases": len(cases),
                    "shapes": shapes,
                    "M": ms,
                    "excluded_any_only_files": excluded,
                },
                indent=2,
            )
        )
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "metadata.json").exists():
        raise FileExistsError("Choose a fresh output directory")
    torch.accelerator.set_device_index(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.recompile_limit = 1024
    torch._dynamo.config.accumulated_recompile_limit = 4096
    repo = Path(__file__).parents[2]
    metadata = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "device": str(torch.cuda.get_device_properties(0)),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "initial_free_total_bytes": torch.accelerator.memory.get_memory_info(),
        "vllm_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "vllm_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
        ).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_dir": str(args.config_dir.resolve()),
        "configs": files,
        "excluded_any_only_files": excluded,
        "cases": cases,
        "rounds": args.rounds,
        "rep_ms": args.rep_ms,
        "seed_base": 1234,
        "method": (
            "CUDA graphs; median of 10 replays per randomized round; hot operands"
        ),
        "correctness": "all elements finite; <=64 rows x128 columns vs FP32",
        "input_dtype": "float8_e4m3fn",
        "output_dtype": "bfloat16",
        "bias": False,
        "quantization_included": False,
        "implementations": ["rowwise_compiled", "native_triton"],
        "scope": "tuned N/K shapes; M extended independently of AITER tuning coverage",
    }
    path = args.output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.monotonic()
    for index, case in enumerate(cases):
        torch.accelerator.reset_peak_memory_stats()
        record = benchmark_case(case, index, args.rounds, args.rep_ms)
        with (args.output_dir / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(
            f"[{index + 1}/{len(cases)}] M={case[0]} N={case[1]} K={case[2]} "
            + json.dumps(record["median_us"])
            + (f" FAILURES={record['failures']}" if record["failures"] else ""),
            flush=True,
        )
        gc.collect()
        torch.accelerator.empty_cache()
    metadata["completed_cases"] = len(cases)
    metadata["elapsed_seconds"] = time.monotonic() - started
    path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
