# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import math
import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from benchmark_dataset import SampleRequest


def convert_to_pytorch_benchmark_format(
    args: argparse.Namespace, metrics: dict[str, list], extra_info: dict[str, Any]
) -> list:
    """
    Save the benchmark results in the format used by PyTorch OSS benchmark with
    on metric per record
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",
                "extra_info": {
                    "args": vars(args),
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }

        tp = record["benchmark"]["extra_info"]["args"].get("tensor_parallel_size")
        # Save tensor_parallel_size parameter if it's part of the metadata
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"]["tensor_parallel_size"] = (
                extra_info["tensor_parallel_size"]
            )

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):
    def clear_inf(self, o: Any):
        if isinstance(o, dict):
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    with open(filename, "w") as f:
        json.dump(
            records,
            f,
            cls=InfEncoder,
            default=lambda o: f"<{type(o).__name__} object is not JSON serializable>",
        )


def print_requests_statistics(requests: list["SampleRequest"]) -> None:
    def get_stats(lens):
        return {
            "p10": np.percentile(lens, 10),
            "p50": np.percentile(lens, 50),
            "p90": np.percentile(lens, 90),
            "p99": np.percentile(lens, 99),
            "min": np.min(lens),
            "max": np.max(lens),
            "avg": np.mean(lens),
        }

    input_lens = [request.prompt_len for request in requests]
    output_lens = [request.expected_output_len for request in requests]

    stats = {
        "Input": get_stats(input_lens),
        "Output": get_stats(output_lens),
    }

    headers = ["Type", "Count", "p10", "p50", "p90", "p99", "min", "max", "avg"]
    row_format = "{:<8} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}"

    print("=" * 70)
    print(f"Requests Statistics: {len(requests)} requests")
    print("-" * 70)
    print(row_format.format(*headers))
    print("-" * 70)
    for name, lens in [("Input", input_lens), ("Output", output_lens)]:
        s = stats[name]
        print(
            row_format.format(
                name,
                len(lens),
                f"{s['p10']:.0f}",
                f"{s['p50']:.0f}",
                f"{s['p90']:.0f}",
                f"{s['p99']:.0f}",
                f"{s['min']:.0f}",
                f"{s['max']:.0f}",
                f"{s['avg']:.0f}",
            )
        )
    print("=" * 70)
