# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.benchmarks.datasets import DEFAULT_NUM_PROMPTS
from vllm.benchmarks.sweep.param_sweep import ParameterSweepItem
from vllm.benchmarks.sweep.serve_workload import explore_comb_workloads


@pytest.mark.parametrize(
    "options,overrides,expected",
    [
        ([], {}, DEFAULT_NUM_PROMPTS),
        (["--num-prompts", "4"], {}, 4),
        (["--num-prompts=4"], {}, 4),
        (["--num_prompts", "4"], {}, 4),
        (["--num_prompts=4"], {}, 4),
        (["--num-prompts", "2", "--num-prompts", "4"], {}, 4),
        ([], {"num_prompts": 4}, 4),
        ([], {"num-prompts": 4}, 4),
        (["--num-prompts", "7"], {"num-prompts": 4}, 4),
    ],
)
def test_workload_uses_effective_prompt_count(
    options, overrides, expected, tmp_path, monkeypatch
):
    """Use the same prompt count as the benchmark after applying overrides."""
    run_workload = Mock(return_value=None)
    monkeypatch.setattr(
        "vllm.benchmarks.sweep.serve_workload.run_comb_workload", run_workload
    )

    explore_comb_workloads(
        None,
        ["vllm", "bench", "serve", *options],
        serve_comb=ParameterSweepItem(),
        bench_comb=ParameterSweepItem(overrides),
        link_vars=[],
        workload_var="request_rate",
        workload_iters=2,
        experiment_dir=tmp_path,
        num_runs=1,
        dry_run=True,
    )

    serial, batch = run_workload.call_args_list
    assert serial.kwargs["workload_value"] == 1
    assert batch.kwargs["workload_value"] == expected
    assert batch.kwargs["bench_comb"]["max_concurrency"] == expected
