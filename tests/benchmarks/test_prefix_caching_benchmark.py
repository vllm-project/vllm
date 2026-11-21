# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from benchmarks.benchmark_prefix_caching import Request, repeat_and_sort_requests


def test_sort_repeated_requests_by_prompt_length():
    requests = [
        Request(prompt="long", prompt_len=4, output_len=2),
        Request(prompt="short", prompt_len=2, output_len=2),
    ]

    assert repeat_and_sort_requests(requests, repeat_count=2, sort=True) == [
        "short",
        "short",
        "long",
        "long",
    ]
