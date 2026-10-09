# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the complete CUTLASS FP4 routing setup before/after rebuilding."""

import argparse
import json
import statistics

import torch

from vllm import _custom_ops as ops
from vllm.utils.torch_utils import set_random_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-values", default="1,4")
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--n", type=int, default=640)
    parser.add_argument("--k", type=int, default=2560)
    parser.add_argument("--inner", type=int, default=32)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--label", default="current")
    parser.add_argument("--diag", action="store_true")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    ms = [int(m) for m in args.m_values.split(",")]
    if (
        min(
            ms
            + [
                args.experts,
                args.topk,
                args.n,
                args.k,
                args.inner,
                args.replays,
                args.trials,
            ]
        )
        < 1
    ):
        parser.error("All sizes must be positive")
    if args.topk > args.experts:
        parser.error("topk must not exceed the number of experts")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    set_random_seed(0)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "args": vars(args),
        "cases": [],
    }
    print(
        "CUDA graph batch averages of complete setup, including zero/count/scan/sort."
    )
    for m in ms:
        topk_ids = (
            torch.rand(m, args.experts, device="cuda").topk(args.topk).indices.int()
        )
        expert_offsets = torch.empty(args.experts + 1, dtype=torch.int32, device="cuda")
        blockscale_offsets = torch.empty_like(expert_offsets)
        ps1 = torch.empty(args.experts, 3, dtype=torch.int32, device="cuda")
        ps2 = torch.empty_like(ps1)
        in_map = torch.empty(m * args.topk, dtype=torch.int32, device="cuda")
        out_map = torch.empty_like(in_map)

        def run(
            topk_ids=topk_ids,
            expert_offsets=expert_offsets,
            blockscale_offsets=blockscale_offsets,
            ps1=ps1,
            ps2=ps2,
            in_map=in_map,
            out_map=out_map,
        ):
            ops.get_cutlass_moe_mm_data(
                topk_ids,
                expert_offsets,
                ps1,
                ps2,
                in_map,
                out_map,
                args.experts,
                args.n,
                args.k,
                blockscale_offsets,
            )

        for _ in range(10):
            run()
        counts = torch.bincount(topk_ids.flatten().long(), minlength=args.experts).int()
        prefix = torch.cat([torch.zeros_like(counts[:1]), counts.cumsum(0).int()])
        padded = torch.div(counts + 127, 128, rounding_mode="floor") * 128
        block_prefix = torch.cat([torch.zeros_like(counts[:1]), padded.cumsum(0).int()])
        torch.testing.assert_close(expert_offsets, prefix, rtol=0, atol=0)
        torch.testing.assert_close(blockscale_offsets, block_prefix, rtol=0, atol=0)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.inner):
                run()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        trials = []
        for _ in range(args.trials):
            for _ in range(3):
                graph.replay()
            start.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            trials.append(start.elapsed_time(end) * 1000 / (args.inner * args.replays))
        case = {"m": m, "median_us": statistics.median(trials), "trials_us": trials}
        report["cases"].append(case)
        print(json.dumps(case))
        if args.diag:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                run()
                torch.accelerator.synchronize()
            print(
                prof.key_averages().table(
                    sort_by="self_device_time_total", row_limit=12
                )
            )
    with open(args.json, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
