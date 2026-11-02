# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the fused Qwen4Exp HC up projection + gate mix on SM100.

Compares ``hc_up_gate_mix`` with the serving chain it replaces: the Qwen4Exp
up-projection GEMM (planned skinny GEMM or ``F.linear``) followed by
``hc_gate_mix``. Shapes are Qwen3.8-Flash-Next's: hc_count 4, hidden 2560,
rank 320, with ``lowrank`` a column slice of the merged down+inject output.

    .venv/bin/python benchmarks/kernels/benchmark_qwen4_exp_hc_up_gate_mix.py

Timing is FlashInfer CUPTI with CUDA graphs and a cold L2. Each arm is
checked bitwise against the chain before it is timed.
"""

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti

import vllm._custom_ops  # noqa: F401
from vllm.models.qwen4_exp.nvidia.low_latency_gemm import (
    QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY,
)
from vllm.models.qwen4_exp.nvidia.ops.hc import (
    hc_gate_mix,
    hc_up_gate_mix,
    hc_up_gate_mix_supported,
)
from vllm.platforms import current_platform

HC, HIDDEN, RANK = 4, 2560, 320
D = HC * HIDDEN
DOWN_N = RANK + HC + 12  # merged down+inject output width, 16-row padded


def make_inputs(m: int, seed: int):
    torch.manual_seed(seed)
    down = torch.randn(m, DOWN_N, device="cuda", dtype=torch.bfloat16)
    xn = torch.randn(m, D, device="cuda", dtype=torch.bfloat16)
    w_up = torch.randn(D, RANK, device="cuda", dtype=torch.bfloat16) * RANK**-0.5
    return down[:, :RANK], xn, w_up


def bench_us(fn, args) -> list[float]:
    return [
        statistics.median(
            bench_gpu_time_with_cupti(
                fn,
                use_cuda_graph=True,
                cold_l2_cache=True,
                dry_run_iters=5,
                repeat_iters=args.repeat,
            )
        )
        * 1000
        for _ in range(args.trials)
    ]


def benchmark(m: int, args) -> list[dict]:
    lowrank, xn, w_up = make_inputs(m, args.seed)
    gemm = torch.ops.vllm.qwen4_exp_low_latency_gemm
    gate = gemm(lowrank, w_up)
    arms = {
        "unfused": lambda: hc_gate_mix(xn, gemm(lowrank, w_up), HC),
        "fused": lambda: hc_up_gate_mix(lowrank, xn, w_up),
        "gemm": lambda: gemm(lowrank, w_up),
        "gate_mix": lambda: hc_gate_mix(xn, gate, HC),
    }
    expected = arms["unfused"]()
    assert torch.equal(expected, hc_gate_mix(xn, F.linear(lowrank, w_up), HC))
    assert torch.equal(arms["fused"](), expected), f"fused != unfused at M={m}"

    # Ideal traffic reads every input and writes every output once.
    weight_bytes = D * RANK * 2
    io_bytes = {
        "fused": weight_bytes + m * (RANK + D + HIDDEN) * 2,
        "gemm": weight_bytes + m * (RANK + D) * 2,
        "gate_mix": m * (2 * D + HIDDEN) * 2,
    }
    io_bytes["unfused"] = io_bytes["gemm"] + io_bytes["gate_mix"]
    flops = 2 * m * D * RANK
    rows = []
    names = list(arms)
    random.Random(args.seed + m).shuffle(names)
    for name in names:
        trials = bench_us(arms[name], args)
        us = statistics.median(trials)
        rows.append(
            dict(
                m=m,
                arm=name,
                us=us,
                trials_us=trials,
                tflops=flops / (us * 1e6) if name != "gate_mix" else None,
                gbps=io_bytes[name] / (us * 1e3),
                # Lower bound on the time of the fused op at the given rates.
                sol_us=max(
                    io_bytes["fused"] / (args.peak_tbps * 1e6),
                    flops / (args.peak_tflops * 1e6),
                ),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 8, 64, 256, 512, 1024, 1536, 2048, 3072, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--peak-tbps", type=float, default=8.0)
    parser.add_argument("--peak-tflops", type=float, default=2250.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    w_up = torch.empty(D, RANK, device="cuda", dtype=torch.bfloat16)
    if not hc_up_gate_mix_supported(w_up, HC):
        raise SystemExit("hc_up_gate_mix needs an SM100-family GPU and build")
    plans = QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY.get(
        current_platform.get_device_capability(), {}
    )
    metadata = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        skinny_gemm_m=sorted(plans.get((D, RANK), {})),
        args=vars(args),
    )
    print(json.dumps(metadata, default=str))
    results = []
    for m in args.num_tokens:
        for row in benchmark(m, args):
            results.append(row)
            print(json.dumps(row), flush=True)
    if args.output:
        args.output.write_text(
            json.dumps(dict(metadata=metadata, results=results), indent=2, default=str)
            + "\n"
        )


if __name__ == "__main__":
    main()
