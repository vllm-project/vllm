# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare dense and expert-aware packed quantization with CUDA graph timing."""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_quant_fp8_packed_triton,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("baseline", args.baseline)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = module.silu_mul_quant_fp8_packed_triton
    warmups, iterations, graph_calls = 5, 25, 16
    results = []
    torch.manual_seed(0)
    cases = [
        (128, 1, 128, 256, 128),
        (512, 4, 128, 2048, 128),
        (4096, 32, 128, 2048, 128),
        (28672, 32, 896, 2048, 128),
        (28672, 32, 96, 2048, 128),
        (28672, 32, 1, 2048, 128),
        (28672, 32, 0, 2048, 128),
        (28672, 32, 896, 7168, 128),
        (28672, 32, 96, 7168, 128),
        (4096, 32, 128, 2048, 32),
    ]
    for capacity, experts, live_per_expert, hidden, group in cases:
        x = torch.randn((capacity, hidden * 2), device="cuda", dtype=torch.bfloat16)
        starts = torch.arange(experts, device="cuda", dtype=torch.int32)
        starts *= ((live_per_expert + 127) // 128) * 128
        ends = starts + live_per_expert
        graphs = {}
        for name, fn, kwargs in [
            ("baseline", baseline, {}),
            ("dense", silu_mul_quant_fp8_packed_triton, {}),
            (
                "expert",
                silu_mul_quant_fp8_packed_triton,
                {"expert_ends": ends, "expert_alignment": 128},
            ),
        ]:
            output = torch.empty(
                (capacity, hidden), device="cuda", dtype=torch.float8_e4m3fn
            )

            def invoke(fn=fn, output=output, kwargs=kwargs, x=x, group=group):
                return fn(
                    x, group_size=group, output_q=output, clamp_limit=10.0, **kwargs
                )

            for _ in range(warmups):
                invoke()
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(graph_calls):
                    q, s = invoke()
            graphs[name] = (graph, q, s)

        live = torch.zeros(capacity, device="cuda", dtype=torch.bool)
        for start in starts.tolist():
            live[start : start + live_per_expert] = True
        for name in ("dense", "expert"):
            torch.testing.assert_close(
                graphs[name][1].view(torch.uint8)[live],
                graphs["baseline"][1].view(torch.uint8)[live],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                graphs[name][2][live], graphs["baseline"][2][live], rtol=0, atol=0
            )
        samples = {name: [] for name in graphs}
        for iteration in range(iterations):
            order = list(graphs)
            if iteration % 2:
                order.reverse()
            for name in order:
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                begin.record()
                graphs[name][0].replay()
                end.record()
                end.synchronize()
                samples[name].append(begin.elapsed_time(end) * 1000 / graph_calls)
        result = dict(
            capacity=capacity,
            experts=experts,
            live_rows=experts * live_per_expert,
            hidden=hidden,
            group_size=group,
            samples_us=samples,
            median_us={k: statistics.median(v) for k, v in samples.items()},
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "warmups": warmups,
                "iterations": iterations,
                "graph_calls": graph_calls,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
