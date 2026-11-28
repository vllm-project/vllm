# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the native gate/up + SiLU chain with the gated-projection operation."""

import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[16, 528, 768, 1056, 1584, 2048, 4096]
    )
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(args.tokens) < 1 or args.rounds < 1 or args.replays < 1:
        parser.error("Token counts, rounds and replays must be positive")

    import torch
    import torch.nn.functional as F

    from vllm.model_executor.layers.fusion import gated_linear_silu  # noqa: F401
    from vllm.platforms import current_platform

    if not current_platform.is_cuda() or not current_platform.has_device_capability(80):
        raise RuntimeError("This benchmark requires NVIDIA BF16 tensor cores")
    torch.manual_seed(0)
    stream = torch.cuda.Stream()
    result = {
        "device": current_platform.get_device_name(),
        "capability": current_platform.get_device_capability(),
        "torch": torch.__version__,
        "scope": "Gate/up projection and activation; not full-model latency",
        "persistent_weight_copy_bytes": 0,
        "cases": [],
    }

    def capture(fn):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = fn()
        graph.replay()
        torch.accelerator.synchronize()
        return graph, output

    def measure(graph):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / args.replays

    with torch.inference_mode():
        weight = torch.randn(18432, 2560, dtype=torch.bfloat16, device="cuda") * 0.05
        for rows in args.tokens:
            x = torch.randn(rows, 2560, dtype=torch.bfloat16, device="cuda") * 0.05

            def native(x=x):
                projected = F.linear(x, weight)
                out = torch.empty((x.shape[0], 9216), dtype=x.dtype, device=x.device)
                torch.ops._C.silu_and_mul(out, projected)
                return out

            def candidate(x=x):
                return torch.ops.vllm.gated_linear_silu(x, weight)

            torch.testing.assert_close(candidate(), native(), rtol=0.02, atol=0.02)
            functions = {"native": native, "gated_op": candidate}
            graphs = {name: capture(fn) for name, fn in functions.items()}
            measurements = {
                name: {"graph_us": [], "eager_us": []} for name in functions
            }
            for repeat in range(args.rounds):
                order = list(functions)
                if repeat % 2:
                    order.reverse()
                for name in order:
                    measurements[name]["graph_us"].append(measure(graphs[name][0]))
                    torch.accelerator.synchronize()
                    start = time.perf_counter()
                    for _ in range(args.replays):
                        functions[name]()
                    torch.accelerator.synchronize()
                    measurements[name]["eager_us"].append(
                        (time.perf_counter() - start) * 1e6 / args.replays
                    )
            row = {
                "tokens": rows,
                "path": "fused_prefill" if 768 <= rows <= 4096 else "native_fallback",
                "samples": measurements,
            }
            for name, values in measurements.items():
                row[name] = {
                    key + "_median": statistics.median(samples)
                    for key, samples in values.items()
                }
            result["cases"].append(row)
            print(
                json.dumps(
                    {key: value for key, value in row.items() if key != "samples"}
                )
            )
            del graphs
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
