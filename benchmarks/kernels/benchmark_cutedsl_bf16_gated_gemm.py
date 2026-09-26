"""CUDA-graph benchmark for the fused BF16 gated GEMM."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.kernels.linear.cute_dsl.cutedsl_bf16_gated_gemm import (
    cutedsl_bf16_gated_gemm,
)

CASES = (
    (1, 1024, 2048, 2),
    (8, 1024, 2048, 2),
    (16, 1024, 2048, 8),
    (64, 6144, 7168, 19),
    (128, 6144, 7168, 20),
)


def _activation(gate, up, activation, linear_beta):
    gate = gate.float()
    up = up.float()
    if activation == "silu":
        return (gate * torch.sigmoid(gate) * up).bfloat16()
    up = up if linear_beta is None else linear_beta * torch.tanh(up / linear_beta)
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * up).bfloat16()


def _graph_us(fn, warmup, iterations):
    output = [None]
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        output[0] = fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples), output[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--tactics", nargs="+", type=int, default=[-1])
    args = parser.parse_args()

    rows = []
    for tokens, intermediate, k, tactic in CASES:
        for activation in ("silu", "situ"):
            linear_beta = 25.0 if activation == "situ" else 1.0
            torch.manual_seed(20260924 + tokens + intermediate + k)
            x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16) * 0.02
            weight = (
                torch.randn(2 * intermediate, k, device="cuda", dtype=torch.bfloat16)
                * 0.02
            )
            gate_weight, up_weight = weight.chunk(2, dim=0)

            def fused(
                x=x,
                weight=weight,
                activation=activation,
                linear_beta=linear_beta,
                tactic=tactic,
            ):
                return cutedsl_bf16_gated_gemm(
                    x,
                    weight,
                    activation=activation,
                    beta=4.0,
                    linear_beta=linear_beta,
                    tactic=tactic,
                )

            def separate(
                x=x,
                gate_weight=gate_weight,
                up_weight=up_weight,
                activation=activation,
                linear_beta=linear_beta,
            ):
                gate = torch.mm(x, gate_weight.T)
                up = torch.mm(x, up_weight.T)
                return _activation(gate, up, activation, linear_beta)

            fused_output = fused()
            separate_output = separate()
            torch.testing.assert_close(
                fused_output, separate_output, rtol=2e-2, atol=0.25
            )
            fused_us, _ = _graph_us(fused, args.warmup, args.iters)
            separate_us, _ = _graph_us(separate, args.warmup, args.iters)
            row = {
                "tokens": tokens,
                "intermediate": intermediate,
                "k": k,
                "tactic": tactic,
                "activation": activation,
                "fused_us": fused_us,
                "separate_bf16_us": separate_us,
                "speedup": separate_us / fused_us,
            }
            rows.append(row)
            print(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
