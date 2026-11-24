# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare batch-invariant BMM revisions with explicit cache and timing modes.

Run from a candidate checkout in a matching official vLLM environment:
    .venv/bin/python benchmarks/kernels/benchmark_batch_invariant_bmm.py \
        --baseline-root /path/to/unmodified-checkout --method cupti --cache cold

Only the baseline BMM module is imported from the second checkout. Its helper
imports use the same official environment. No AST extraction or kernel override
is used. A microbenchmark does not establish a model or serving speedup.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import subprocess
import sys
import warnings
from pathlib import Path

import torch

import vllm
from vllm.model_executor.determinism import batch_invariant as candidate

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
# (batch dimension, reduction dimension, output columns). In MLA, the batch
# dimension is normally the number of local heads, not the request batch size.
SHAPES = [(16, 128, 512), (16, 512, 128), (32, 128, 512), (32, 512, 128)]
ROWS = [1, 8, 16, 17, 32, 33, 64, 65, 128, 256]


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_baseline(root: Path):
    path = root / "vllm/model_executor/determinism/batch_invariant.py"
    spec = importlib.util.spec_from_file_location("baseline_batch_invariant_bmm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def gpu_processes() -> set[str]:
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    return {
        row.split(",")[1].strip()
        for row in rows
        if len(row.split(",")) == 2 and row.split(",")[0].strip() == uuid
    }


def graph_us(fn, repeat: int) -> list[float]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    calls_per_graph = 32
    with torch.cuda.graph(graph):
        for _ in range(calls_per_graph):
            fn()
    for _ in range(5):
        graph.replay()
    torch.accelerator.synchronize()
    values = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(25):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1e3 / (25 * calls_per_graph))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--method", choices=["cupti", "graph-events"], default="cupti")
    parser.add_argument("--cache", choices=["cold", "warm"], default="cold")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--m", type=int, nargs="+", default=ROWS)
    args = parser.parse_args()
    if args.rounds < 1 or any(m < 1 for m in args.m):
        parser.error("Rounds and row counts must be positive")
    if args.method == "graph-events" and args.cache != "warm":
        parser.error("graph-events uses warm repeated buffers; select --cache warm")
    if not torch.accelerator.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA is required")
    torch.accelerator.set_device_index(0)
    torch.manual_seed(4327)
    free, total = torch.accelerator.get_memory_info()
    if total - free > 2 * 1024**3:
        raise RuntimeError("GPU0 has an existing job; use an uncontended GPU window")
    allowed = gpu_processes()
    if len(allowed) > 1:
        raise RuntimeError("Multiple GPU0 processes already exist")
    baseline, baseline_file = load_baseline(args.baseline_root)
    modules = {"baseline": baseline, "candidate": candidate}
    props = torch.cuda.get_device_properties(0)
    import triton

    print(
        json.dumps(
            {
                "type": "environment",
                "gpu": props.name,
                "sm": [props.major, props.minor],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "triton": triton.__version__,
                "vllm_version": vllm.__version__,
                "vllm_file": vllm.__file__,
                "candidate_file": candidate.__file__,
                "baseline_file": str(baseline_file),
                "candidate_sha256": source_hash(Path(candidate.__file__)),
                "baseline_sha256": source_hash(baseline_file),
                "dtype": args.dtype,
                "method": args.method,
                "cache": args.cache,
                "rounds": args.rounds,
                "metric_scope": "preallocated GPU BMM; no allocation or Engine",
            }
        ),
        flush=True,
    )
    if args.method == "cupti":
        # Refuse silent fallback to CUDA events. CUPTI must be installed and
        # usable for this measurement to be called hardware-kernel timing.
        import cupti  # noqa: F401
        from flashinfer.testing import bench_gpu_time_with_cupti

    rng = random.Random(20261124)
    bit_dtype = torch.int32 if args.dtype == "fp32" else torch.int16
    for batch, k, n in SHAPES:
        for m in args.m:
            a = torch.randn(batch, m, k, device="cuda", dtype=DTYPES[args.dtype])
            b = torch.randn(batch, k, n, device="cuda", dtype=a.dtype)
            outputs = {
                name: torch.empty(batch, m, n, device="cuda", dtype=a.dtype)
                for name in modules
            }
            functions = {
                name: (
                    lambda mod=mod, out=outputs[name], a=a, b=b: (
                        mod.bmm_batch_invariant(a, b, out=out)
                    )
                )
                for name, mod in modules.items()
            }
            for fn in functions.values():
                for _ in range(5):
                    fn()
            torch.accelerator.synchronize()
            assert torch.equal(
                outputs["baseline"].view(bit_dtype),
                outputs["candidate"].view(bit_dtype),
            ), "Candidate changed output bits"
            timings = {name: [] for name in modules}
            for _ in range(args.rounds):
                order = list(modules)
                rng.shuffle(order)
                for name in order:
                    if gpu_processes() - allowed:
                        raise RuntimeError(
                            "New external GPU0 job appeared; discard this run"
                        )
                    if args.method == "cupti":
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "error", message=".*Falling back to CUDA events.*"
                            )
                            samples = bench_gpu_time_with_cupti(
                                functions[name],
                                use_cuda_graph=True,
                                cold_l2_cache=args.cache == "cold",
                            )
                        values = [float(value) * 1e3 for value in samples]
                    else:
                        values = graph_us(functions[name], repeat=1)
                    if gpu_processes() - allowed:
                        raise RuntimeError(
                            "New external GPU0 job appeared; discard this run"
                        )
                    timings[name].extend(values)
            old, new = (statistics.median(timings[name]) for name in modules)
            print(
                json.dumps(
                    {
                        "type": "measurement",
                        "B": batch,
                        "M": m,
                        "K": k,
                        "N": n,
                        "baseline_us": old,
                        "candidate_us": new,
                        "speedup": old / new,
                        "baseline_tflops": 2 * batch * m * k * n / (old * 1e6),
                        "candidate_tflops": 2 * batch * m * k * n / (new * 1e6),
                        "samples_us": timings,
                        "output_bits_equal": True,
                        "full_model_validation": False,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
