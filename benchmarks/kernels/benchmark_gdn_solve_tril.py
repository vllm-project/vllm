# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare opt-in GDN inverse fusion with the original varlen solver."""

import argparse
import json
import os
import statistics

import torch

from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
from vllm.triton_utils import triton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1072, 3216])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timing", choices=["graph", "event"], default="graph")
    args = parser.parse_args()
    if args.repeats < 1 or any(length < 1 for length in args.lengths):
        parser.error("Positive lengths and repeat count are required")
    torch.manual_seed(42)
    results = []
    for length in args.lengths:
        a = torch.randn((1, length + 16, 32, 64), device="cuda") * 0.01
        rows = torch.arange(length + 16, device="cuda") % 64
        columns = torch.arange(64, device="cuda")
        a.masked_fill_(columns[None, None, None, :] >= rows[None, :, None, None], 0)
        a[:, length:] = 0
        cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        indices = torch.tensor(
            [(0, i) for i in range(triton.cdiv(length, 64))],
            device="cuda",
            dtype=torch.int32,
        )
        samples = {"baseline": [], "candidate": []}
        outputs = {}
        for trial in range(args.repeats):
            order = ["baseline", "candidate"]
            if trial % 2:
                order.reverse()
            for variant in order:
                os.environ["FLA_TRIL_PARALLEL_DIAGONALS"] = (
                    "1" if variant == "candidate" else "0"
                )

                def invoke(a=a, cu=cu, indices=indices):
                    return solve_tril(a, cu, indices, torch.bfloat16)

                outputs[variant] = invoke()
                if args.timing == "graph":
                    milliseconds = triton.testing.do_bench_cudagraph(invoke)
                else:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    for _ in range(10):
                        invoke()
                    start.record()
                    for _ in range(100):
                        invoke()
                    end.record()
                    end.synchronize()
                    milliseconds = start.elapsed_time(end) / 100
                samples[variant].append(milliseconds)
        torch.testing.assert_close(
            outputs["candidate"], outputs["baseline"], atol=2e-3, rtol=2e-3
        )
        baseline = statistics.median(samples["baseline"])
        candidate = statistics.median(samples["candidate"])
        results.append(
            {
                "length": length,
                "samples_ms": samples,
                "baseline_median_ms": baseline,
                "candidate_median_ms": candidate,
                "latency_reduction_percent": 100 * (1 - candidate / baseline),
            }
        )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "timing": args.timing,
                "results": results,
            }
        )
    )


if __name__ == "__main__":
    main()
