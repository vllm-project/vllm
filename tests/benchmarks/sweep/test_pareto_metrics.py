# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.benchmarks.sweep.plot_pareto import _prepare_records


@pytest.mark.parametrize("request_rate", [2.0, 10.0, float("inf")])
def test_per_user_throughput_uses_concurrency(request_rate):
    records, skipped = _prepare_records(
        [
            {
                "output_throughput": 100.0,
                "request_rate": request_rate,
                "max_concurrent_requests": 5,
            }
        ],
        user_count_var="max_concurrency",
        gpu_count_var=None,
    )
    assert skipped == 0
    assert records[0]["user_count_estimate"] == 5
    assert records[0]["tokens_per_user"] == 20.0


def test_request_rate_alone_does_not_supply_a_user_count():
    records, skipped = _prepare_records(
        [{"output_throughput": 100.0, "request_rate": 2.0}],
        user_count_var="max_concurrency",
        gpu_count_var=None,
    )
    assert records == []
    assert skipped == 1


def test_explicit_user_count_takes_precedence():
    records, _ = _prepare_records(
        [
            {
                "output_throughput": 100.0,
                "users": 10,
                "request_rate": 2.0,
                "max_concurrent_requests": 5,
            }
        ],
        user_count_var="users",
        gpu_count_var=None,
    )
    assert records[0]["tokens_per_user"] == 10.0
