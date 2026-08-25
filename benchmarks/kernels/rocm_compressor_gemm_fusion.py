# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare serial compressor GEMMs with one fused GEMM on ROCm."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class Timing:
    median_us: float
    mean_us: float
    p05_us: float
    p95_us: float


class CompressorGemmFusion:
    """Run compressor projections separately or as one fused projection."""

    def __init__(
        self,
        m: int,
        n_sizes: tuple[int, ...],
        k: int,
    ) -> None:
        torch.manual_seed(0)
        self.n_sizes = n_sizes
        self.input = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        self.weights = tuple(
            torch.randn(n, k, dtype=torch.bfloat16, device="cuda") for n in n_sizes
        )
        self.fused_weight = torch.cat(self.weights, dim=0).contiguous()

    def serial(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.mm(self.input, weight.T, out_dtype=torch.float32)
            for weight in self.weights
        )

    def fused(self) -> tuple[torch.Tensor, ...]:
        output = torch.mm(
            self.input,
            self.fused_weight.T,
            out_dtype=torch.float32,
        )
        return output.split(self.n_sizes, dim=-1)


def capture(
    runner: Callable[[], Any],
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.Stream, Any]:
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    current_stream = torch.cuda.current_stream()
    capture_stream.wait_stream(current_stream)
    with (
        torch.cuda.stream(capture_stream),
        torch.cuda.graph(graph, stream=capture_stream),
    ):
        outputs = runner()
    current_stream.wait_stream(capture_stream)
    torch.accelerator.synchronize()
    return graph, capture_stream, outputs


def replay_snapshot(
    graph: torch.cuda.CUDAGraph,
    outputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    graph.replay()
    torch.accelerator.synchronize()
    return tuple(output.clone() for output in outputs)


def percentile(samples: list[float], fraction: float) -> float:
    index = round((len(samples) - 1) * fraction)
    return sorted(samples)[index]


def measure_interleaved(
    runners: dict[str, Callable[[], Any]],
    iterations: int,
    warmup: int,
) -> dict[str, Timing]:
    for _ in range(warmup):
        for runner in runners.values():
            runner()
    torch.accelerator.synchronize()

    samples = {name: [] for name in runners}
    names = tuple(runners)
    for iteration in range(iterations):
        offset = iteration % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runners[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0)

    return {
        name: Timing(
            median_us=statistics.median(values),
            mean_us=statistics.fmean(values),
            p05_us=percentile(values, 0.05),
            p95_us=percentile(values, 0.95),
        )
        for name, values in samples.items()
    }


def compare_outputs(
    expected: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    rtol = 1e-4
    atol = 1e-4
    comparisons = []
    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
        difference = (left - right).abs()
        comparisons.append(
            {
                "index": index,
                "shape": list(right.shape),
                "exact": torch.equal(left, right),
                "max_abs_difference": difference.max().item(),
                "mean_abs_difference": difference.mean().item(),
                "rtol": rtol,
                "atol": atol,
            }
        )
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
    return {
        "passed": True,
        "all_exact": all(item["exact"] for item in comparisons),
        "outputs": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument(
        "--n-sizes",
        default="2048,512",
        help="Comma-separated output widths.",
    )
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is required")

    n_sizes = tuple(int(value) for value in args.n_sizes.split(","))
    if not n_sizes or any(value <= 0 for value in n_sizes):
        parser.error("--n-sizes must contain positive integers")

    module = CompressorGemmFusion(
        args.m,
        n_sizes,
        args.k,
    )
    serial_eager = module.serial()
    fused_eager = module.fused()
    torch.accelerator.synchronize()
    correctness = {
        "fused_vs_serial": compare_outputs(serial_eager, fused_eager),
    }
    eager_runners: dict[str, Callable[[], Any]] = {
        "serial": module.serial,
        "fused": module.fused,
    }
    eager_timings = measure_interleaved(
        eager_runners,
        args.iterations,
        args.warmup,
    )

    serial_graph, serial_capture_stream, serial_outputs = capture(module.serial)
    serial_graph_outputs = replay_snapshot(serial_graph, serial_outputs)
    fused_graph, fused_capture_stream, fused_outputs = capture(module.fused)
    fused_graph_outputs = replay_snapshot(fused_graph, fused_outputs)
    correctness["serial_graph_vs_eager"] = compare_outputs(
        serial_eager,
        serial_graph_outputs,
    )
    correctness["fused_graph_vs_eager"] = compare_outputs(
        serial_eager,
        fused_graph_outputs,
    )
    graph_runners: dict[str, Callable[[], Any]] = {
        "serial": serial_graph.replay,
        "fused": fused_graph.replay,
    }
    graph_timings = measure_interleaved(
        graph_runners,
        args.iterations,
        args.warmup,
    )

    def timing_payload(timings: dict[str, Timing]) -> dict[str, Any]:
        payload = {name: asdict(timing) for name, timing in timings.items()}
        payload["fused_speedup_vs_serial"] = (
            timings["serial"].median_us / timings["fused"].median_us
        )
        return payload

    result = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(),
        "environment": {
            name: os.environ.get(name, "<unset>")
            for name in (
                "AMD_DIRECT_DISPATCH",
                "GPU_MAX_HW_QUEUES",
                "HIP_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "ROC_CPU_WAIT_FOR_SIGNAL",
            )
        },
        "input_shape": [args.m, args.k],
        "individual_weight_shapes": [[n, args.k] for n in n_sizes],
        "fused_weight_shape": [sum(n_sizes), args.k],
        "output_mapping": [
            [sum(n_sizes[:index]), sum(n_sizes[: index + 1])]
            for index in range(len(n_sizes))
        ],
        "dtype": "bfloat16",
        "output_dtype": "float32",
        "weight_fusion": "offline_concatenate_dim_0",
        "output_split": "zero_copy_views",
        "stream_mode": "single_stream",
        "execution_modes": list(eager_runners),
        "correctness": correctness,
        "eager": timing_payload(eager_timings),
        "graph": timing_payload(graph_timings),
        "iterations": args.iterations,
        "warmup": args.warmup,
    }
    _ = serial_capture_stream, fused_capture_stream
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
