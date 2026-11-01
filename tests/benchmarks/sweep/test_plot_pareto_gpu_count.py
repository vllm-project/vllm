# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path

import pytest

from vllm.benchmarks.sweep.plot_pareto import plot_pareto


def test_standard_gpu_counts_match_explicit_plot(tmp_path: Path):
    pytest.importorskip("seaborn")
    records = [
        {
            "output_throughput": 100,
            "max_concurrency": 1,
            "num_gpus": 8,
            "total_gpus": 8,
            "tensor_parallel_size": 2,
        },
        {
            "output_throughput": 200,
            "max_concurrency": 10,
            "gpu_count": 4,
            "total_gpus": 4,
            "tensor_parallel_size": 2,
        },
    ]

    (tmp_path / "summary.json").write_text(json.dumps(records), encoding="utf-8")
    plot_pareto(
        output_dir=tmp_path,
        user_count_var="max_concurrency",
        gpu_count_var="total_gpus",
        label_by=[],
        dry_run=False,
    )
    plot_path = next(tmp_path.rglob("*.png"))
    expected = plot_path.read_bytes()

    plot_pareto(
        output_dir=tmp_path,
        user_count_var="max_concurrency",
        gpu_count_var=None,
        label_by=[],
        dry_run=False,
    )
    assert plot_path.read_bytes() == expected
