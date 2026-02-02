# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import math
import os
from typing import Any


def extract_field(
    args: argparse.Namespace, extra_info: dict[str, Any], field_name: str
) -> str:
    if field_name in extra_info:
        return extra_info[field_name]

    v = args
    # For example, args.compilation_config.mode
    for nested_field in field_name.split("."):
        if not hasattr(v, nested_field):
            return ""
        v = getattr(v, nested_field)
    return v


def use_compile(args: argparse.Namespace, extra_info: dict[str, Any]) -> bool:
    """
    Check if the benchmark is run with torch.compile
    """
    return not (
        extract_field(args, extra_info, "compilation_config.mode") == "0"
        or "eager" in getattr(args, "output_json", "")
        or "eager" in getattr(args, "result_filename", "")
    )


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
                    "compilation_config.mode": extract_field(
                        args, extra_info, "compilation_config.mode"
                    ),
                    "optimization_level": extract_field(
                        args, extra_info, "optimization_level"
                    ),
                    # A boolean field used by vLLM benchmark HUD dashboard
                    "use_compile": use_compile(args, extra_info),
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
            return {
                str(k)
                if not isinstance(k, (str, int, float, bool, type(None)))
                else k: self.clear_inf(v)
                for k, v in o.items()
            }
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
            default=lambda o: f"<{type(o).__name__} is not JSON serializable>",
        )
