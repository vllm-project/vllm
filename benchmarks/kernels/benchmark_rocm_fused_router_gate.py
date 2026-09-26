# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare gfx950 routing with the gate GEMM plus selection baseline.

The default M values are the distinct products of --batch-sizes and
--multipliers. HIP graph replay excludes allocation and compilation. Cold
runs rotate 128 independent input sets (over 256 MiB of gate weights), while
warm runs reuse one input set. CUPTI is unavailable on ROCm.
"""

import argparse
import hashlib
import inspect
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.rocm_fused_router_gate import (
    rocm_fused_router_gate,
)
from vllm.triton_utils import triton

K, N = 7168, 384


def capture(functions, repeats):
    for fn in functions:
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            for fn in functions:
                fn()
    graph.replay()
    torch.accelerator.synchronize()
    return graph


def measure(graph, calls, samples):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / calls)
    return timings


def make_functions(x, weight, bias, topk):
    def baseline():
        logits = torch.mm(x, weight.T, out_dtype=torch.float32)
        return fused_topk_bias(
            x,
            logits,
            "sqrtsoftplus",
            bias,
            topk,
            True,
            routed_scaling_factor=1.5,
        )

    def candidate():
        return rocm_fused_router_gate(x, weight, bias, topk, True, 1.5)

    return baseline, candidate


def check(x, weight, bias, functions, topk):
    logits = x.float() @ weight.float().T
    scores = torch.nn.functional.softplus(logits).sqrt()
    ids = (scores + bias).argsort(dim=-1, descending=True, stable=True)[:, :topk]
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True) * 1.5
    for fn in functions:
        actual_weights, actual_ids = fn()
        torch.testing.assert_close(actual_ids.long(), ids, atol=0, rtol=0)
        torch.testing.assert_close(actual_weights, weights, atol=2e-5, rtol=2e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--multipliers", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6]
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        choices=[5120, 7168],
        default=7168,
        help="5120: DeepSeek-V4.1-Flash, 7168: DeepSeek-V4-Pro",
    )
    parser.add_argument("--topk", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--cache", choices=["warm", "cold", "both"], default="both")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--cold-sets", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    global K
    K = args.hidden_size
    if args.cold_sets * K * N * 2 <= 256 * 1024**2:
        parser.error("cold input sets must exceed the MI355X 256 MiB LLC")
    torch.manual_seed(0)
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "kernel_sha256": hashlib.sha256(
            Path(inspect.getfile(rocm_fused_router_gate)).read_bytes()
        ).hexdigest(),
        "environment": {
            k: os.environ[k]
            for k in (
                "HIP_VISIBLE_DEVICES",
                "HIP_FORCE_DEV_KERNARG",
                "ROCPROFILER_QUEUE_INTERPOSITION",
            )
            if k in os.environ
        },
        "K": K,
        "N": N,
        "dtype": "bfloat16",
        "accumulator": "float32",
        "batch_sizes": args.batch_sizes,
        "multipliers": args.multipliers,
        "cold_sets": args.cold_sets,
        "timing": "median HIP graph replay, events around entire graph",
        "peak_hbm_bytes_per_second": 8e12,
        "peak_dense_bf16_flops_per_second": 2.5166e15,
        "work": "2*M*N*K FLOPs; 2*(M*K+N*K)+4*N+8*M*topk bytes",
        "note": (
            "Ideal operation traffic excludes scratch and assumes each input is read "
            "once. Roofline excludes launch and top-k costs. Cold rotates independent "
            "inputs; warm uses LLC and cannot establish HBM efficiency."
        ),
    }
    print(json.dumps(metadata), flush=True)
    results = []
    rows = sorted({b * s for b in args.batch_sizes for s in args.multipliers})
    modes = ["warm", "cold"] if args.cache == "both" else [args.cache]
    for m in rows:
        x = torch.randn((m, K), device="cuda", dtype=torch.bfloat16)
        weight = (torch.randn((N, K), device="cuda") * K**-0.5).bfloat16()
        bias = torch.randn(N, device="cuda")
        for topk in args.topk:
            fns = make_functions(x, weight, bias, topk)
            check(x, weight, bias, fns, topk)
            for mode in modes:
                inputs = [(x, weight, bias)]
                if mode == "cold":
                    inputs.extend(
                        (x.clone(), weight.clone(), bias.clone())
                        for _ in range(args.cold_sets - 1)
                    )
                pairs = [make_functions(*tensors, topk) for tensors in inputs]
                repeats = 128 if mode == "warm" else 1
                graphs = [
                    capture([pair[i] for pair in pairs], repeats) for i in range(2)
                ]
                times = [[], []]
                for sample in range(3):
                    for i in [0, 1] if sample % 2 == 0 else [1, 0]:
                        times[i].extend(
                            measure(graphs[i], repeats * len(inputs), args.samples)
                        )
                baseline, candidate = (statistics.median(t) for t in times)
                byte_count = 2 * (m * K + N * K) + 4 * N + 8 * m * topk
                flops = 2 * m * N * K
                ideal_us = max(byte_count / 8e12, flops / 2.5166e15) * 1e6
                row = {
                    "m": m,
                    "topk": topk,
                    "cache": mode,
                    "baseline_us": baseline,
                    "candidate_us": candidate,
                    "baseline_p10_p90_us": [
                        statistics.quantiles(times[0], n=10)[i] for i in (0, 8)
                    ],
                    "candidate_p10_p90_us": [
                        statistics.quantiles(times[1], n=10)[i] for i in (0, 8)
                    ],
                    "speedup": baseline / candidate,
                    "candidate_tflops": flops / candidate / 1e6,
                    "ideal_bytes": byte_count,
                    "effective_gbps": byte_count / candidate / 1000,
                    "ideal_roofline_us": ideal_us,
                    "cold_roofline_fraction": ideal_us / candidate
                    if mode == "cold"
                    else None,
                }
                results.append(row)
                print(json.dumps(row), flush=True)
                del graphs, pairs, inputs
                torch.accelerator.empty_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
