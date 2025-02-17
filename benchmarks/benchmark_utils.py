# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import Any, Dict, List


def convert_to_pytorch_benchmark_format(args: argparse.Namespace,
                                        metrics: Dict[str, List],
                                        extra_info: Dict[str, Any]) -> List:
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
        records.append(record)

    return records
