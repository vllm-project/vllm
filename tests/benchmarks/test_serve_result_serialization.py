# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from unittest.mock import patch

from vllm.benchmarks.serve import save_to_pytorch_benchmark_format


def test_pooling_e2el_is_tracked_without_per_request_arrays() -> None:
    results = {
        "mean_e2el_ms": 12.0,
        "median_e2el_ms": 11.0,
        "std_e2el_ms": 2.0,
        "p99_e2el_ms": 16.0,
        "failed": 2,
        "latencies": [0.01, 0.02],
        "queue_times": [0.001, 0.002],
    }
    args = argparse.Namespace(model="test-model")

    with (
        patch("vllm.benchmarks.serve.convert_to_pytorch_benchmark_format") as convert,
        patch("vllm.benchmarks.serve.write_to_json"),
    ):
        convert.return_value = [{"benchmark": {}}]
        save_to_pytorch_benchmark_format(args, results, "result.json")

    assert convert.call_args.kwargs["metrics"] == {
        "median_e2el_ms": [11.0],
        "mean_e2el_ms": [12.0],
        "std_e2el_ms": [2.0],
        "p99_e2el_ms": [16.0],
    }
    assert convert.call_args.kwargs["extra_info"] == {"failed": 2}
