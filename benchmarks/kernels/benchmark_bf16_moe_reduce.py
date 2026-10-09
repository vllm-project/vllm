# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete CUTLASS BF16 finalize chains using hot CUDA graphs."""

import argparse
import json
import random
import statistics

import torch

from vllm.model_executor.layers.fused_moe.bf16_moe_reduce import (
    bf16_moe_weighted_sum,
)
from vllm.utils.torch_utils import set_random_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-values", default="1,4")
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--inner", type=int, default=32)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--json")
    args = parser.parse_args()
    ms = [int(m) for m in args.m_values.split(",")]
    if min(ms + [args.width, args.topk, args.inner, args.replays, args.trials]) < 1:
        parser.error("All sizes must be positive")
    if args.topk > 32:
        parser.error("topk must be <= 32")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    set_random_seed(0)
    rng = random.Random(0)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cache_mode": "hot",
        "args": vars(args),
        "cases": [],
    }
    for m in ms:
        experts = torch.randn(
            m, args.topk, args.width, device="cuda", dtype=torch.bfloat16
        )
        weights = torch.randn(m, args.topk, device="cuda").softmax(-1)
        output = torch.empty(m, args.width, device="cuda", dtype=torch.bfloat16)

        def reference(experts=experts, weights=weights, output=output):
            output.copy_(
                (experts * weights.bfloat16()[..., None]).sum(dim=1), non_blocking=True
            )
            return output

        def fused(experts=experts, weights=weights, output=output):
            bf16_moe_weighted_sum(experts, weights, output)
            return output

        expected = reference().clone()
        torch.testing.assert_close(fused(), expected, rtol=0.008, atol=1e-5)
        graphs = {}
        for name, fn in {"torch_chain": reference, "fused": fused}.items():
            for _ in range(10):
                fn()
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.inner):
                    fn()
            graphs[name] = graph

        trials = {name: [] for name in graphs}
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for _ in range(args.trials):
            names = list(graphs)
            rng.shuffle(names)
            for name in names:
                for _ in range(3):
                    graphs[name].replay()
                start.record()
                for _ in range(args.replays):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                trials[name].append(
                    start.elapsed_time(end) * 1000 / (args.replays * args.inner)
                )
        case = {
            "m": m,
            "times": {
                name: {"median_us": statistics.median(times), "trials_us": times}
                for name, times in trials.items()
            },
        }
        report["cases"].append(case)
        print(json.dumps(case))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
