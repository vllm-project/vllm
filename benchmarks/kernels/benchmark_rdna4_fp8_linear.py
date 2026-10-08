# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare RowWise/ChannelWise Torch, native Triton and tuned AITER on RDNA4.

Run with VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_LINEAR=1. Inputs are
already quantized FP8; quantization, TP communication and attention are excluded.
"""

import argparse
import csv
import hashlib
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import regex as re
import torch
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.triton_utils import triton


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep-ms", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--m", type=int, nargs="+")
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=("rowwise", "channelwise", "native", "aiter"),
        default=["rowwise", "channelwise", "native", "aiter"],
    )
    return parser.parse_args()


def discover_cases(config_dir, ms):
    cases, files, excluded = [], {}, []
    for path in sorted(config_dir.glob("GEMM-A8W8-N=*-K=*.json")):
        data = json.loads(path.read_text())
        files[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "config": data,
        }
        n, k = map(int, re.findall(r"[NK]=(\d+)", path.name))
        bounds = sorted(
            int(key.removeprefix("M_LEQ_")) for key in data if key.startswith("M_LEQ_")
        )
        if not bounds:
            excluded.append(path.name)
        for m in bounds:
            if ms is None or m in ms:
                cases.append((m, n, k, path.name))
    return cases, files, excluded


def make_kernel(backend, n, k, force_kernel=None):
    config = VllmConfig(kernel_config=KernelConfig(linear_backend=backend))
    config.compilation_config.mode = CompilationMode.VLLM_COMPILE
    with set_current_vllm_config(config):
        return init_fp8_linear_kernel(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticChannelSym,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(n, k),
            force_kernel=force_kernel,
        )


def sampled_reference(a, w, sa, sb):
    # Check evenly spaced rows/columns, including both boundaries, for every M.
    rows = torch.linspace(0, a.shape[0] - 1, min(a.shape[0], 64), device="cuda")
    cols = torch.linspace(0, w.shape[0] - 1, min(w.shape[0], 128), device="cuda")
    rows, cols = rows.long().unique(), cols.long().unique()
    ref = (a[rows].float() @ w[cols].float().t()) * sa[rows] * sb[cols].t()
    return rows, cols, ref


def relative_rms(output, rows, cols, ref):
    if not torch.isfinite(output).all().item():
        raise ValueError("Non-finite output")
    sample = output[rows[:, None], cols[None, :]].float()
    error = ((sample - ref).square().mean() / ref.square().mean()).sqrt().item()
    if error >= 0.01:
        raise ValueError(f"Relative RMS error {error} >= 0.01")
    return error


def make_call(kernel, a, weight, sa, sb):
    output_shape = [a.shape[0], sb.shape[0]]

    def call():
        return kernel.apply_scaled_mm(
            A=a,
            B=weight,
            As=sa,
            Bs=sb,
            out_dtype=torch.bfloat16,
            bias=None,
            output_shape=output_shape,
        )

    return call


def main():
    args = parse_args()
    torch.accelerator.set_device_index(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Each shape is independently specialized, so this sweep deliberately has
    # more compiled functions than the normal Dynamo recompile limit.
    torch._dynamo.config.recompile_limit = 1024
    torch._dynamo.config.accumulated_recompile_limit = 4096
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "results.jsonl").exists():
        raise FileExistsError("Choose a fresh output directory to avoid mixed runs")
    cases, files, excluded = discover_cases(args.config_dir, args.m)
    if args.limit is not None:
        cases = cases[: args.limit]
    metadata = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "device_properties": str(torch.cuda.get_device_properties(args.device)),
        "gpu_index": args.device,
        "vllm_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2], text=True
        ).strip(),
        "config_dir": str(args.config_dir.resolve()),
        "configs": files,
        "excluded_any_only_files": excluded,
        "cases": cases,
        "rounds": args.rounds,
        "rep_ms": args.rep_ms,
        "method": "CUDA graphs, hot operands, median of randomized rounds",
        "correctness": "all elements finite; <=64 rows x128 columns vs FP32",
        "input_dtype": "float8_e4m3fn",
        "output_dtype": "bfloat16",
        "bias": False,
        "quantization_included": False,
        "implementations": args.implementations,
        "seed_base": 1234,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    results = []
    started = time.monotonic()
    for index, (m, n, k, filename) in enumerate(cases):
        torch.manual_seed(1234 + index)
        a = (torch.randn(m, k, device="cuda") * 0.2).to(torch.float8_e4m3fn)
        w = (torch.randn(n, k, device="cuda") * 0.2).to(torch.float8_e4m3fn)
        sa = torch.rand(m, 1, device="cuda") + 0.5
        sb = torch.rand(n, 1, device="cuda") + 0.5
        tuned_config, tuned = get_gemm_config("GEMM-A8W8", m, n, k)
        if not tuned:
            raise ValueError(f"AITER config is not tuned for {(m, n, k)}")
        methods = {}
        if "rowwise" in args.implementations:
            rowwise = make_kernel("torch", n, k)
            assert isinstance(rowwise, RowWiseTorchFP8ScaledMMLinearKernel)
            eager_rowwise = make_call(rowwise, a, w.t(), sa, sb)
            methods["rowwise_eager"] = eager_rowwise
            methods["rowwise_compiled"] = torch.compile(eager_rowwise, fullgraph=True)
        if "channelwise" in args.implementations:
            channel = make_kernel(
                "torch", n, k, ChannelWiseTorchFP8ScaledMMLinearKernel
            )
            assert isinstance(channel, ChannelWiseTorchFP8ScaledMMLinearKernel)
            eager_channel = make_call(channel, a, w.t(), sa, sb)
            methods["channelwise_eager"] = eager_channel
            methods["channelwise_compiled"] = torch.compile(
                eager_channel, fullgraph=True
            )
        if "native" in args.implementations:
            native = make_kernel("triton", n, k)
            assert isinstance(native, TritonPerTokenFp8ScaledMMLinearKernel)
            methods["native_triton"] = make_call(native, a, w.t(), sa, sb)
        if "aiter" in args.implementations:
            aiter = make_kernel("aiter", n, k, AiterPerTokenFp8ScaledMMLinearKernel)
            assert isinstance(aiter, AiterPerTokenFp8ScaledMMLinearKernel)
            methods["aiter_tuned"] = make_call(aiter, a, w, sa, sb)
        rows, cols, ref = sampled_reference(a, w, sa, sb)
        errors = {}
        for name, fn in methods.items():
            output = fn()
            assert output.shape == (m, n) and output.dtype == torch.bfloat16
            errors[name] = relative_rms(output, rows, cols, ref)
            del output
            for _ in range(3):
                fn()
        torch.accelerator.synchronize()
        timings = {name: [] for name in methods}
        rng = random.Random(1234 + index)
        for _ in range(args.rounds):
            order = list(methods)
            rng.shuffle(order)
            for name in order:
                timings[name].append(
                    1000
                    * triton.testing.do_bench_cudagraph(
                        methods[name], rep=args.rep_ms, return_mode="median"
                    )
                )
        record = {
            "M": m,
            "N": n,
            "K": k,
            "config_file": filename,
            "aiter_config": tuned_config,
            "relative_rms": errors,
            "samples_us": timings,
            "median_us": {name: statistics.median(t) for name, t in timings.items()},
        }
        results.append(record)
        with (args.output_dir / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(
            f"[{index + 1}/{len(cases)}] M={m} N={n} K={k} "
            + " ".join(
                f"{name}={value:.2f}us" for name, value in record["median_us"].items()
            ),
            flush=True,
        )
        methods.clear()
        del a, w, sa, sb, rows, cols, ref
        torch.accelerator.empty_cache()
    with (args.output_dir / "results.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["M", "N", "K", "config_file", *results[0]["median_us"]]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {key: row[key] for key in ("M", "N", "K", "config_file")}
                | row["median_us"]
            )
    metadata["elapsed_seconds"] = time.monotonic() - started
    metadata["completed_cases"] = len(results)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
