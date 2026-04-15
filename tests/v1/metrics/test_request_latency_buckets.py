# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.metrics.utils import build_request_latency_buckets


def test_build_request_latency_buckets_default_starts_at_300ms():
    b = build_request_latency_buckets(fine_low_end=False)
    assert b[0] == 0.3
    assert b == sorted(b)
    assert len(b) == len(set(b))


def test_build_request_latency_buckets_fine_includes_sub_300ms():
    b = build_request_latency_buckets(fine_low_end=True)
    assert 0.01 in b
    assert 0.25 in b
    assert 0.3 in b
    assert b[0] == 0.01
    assert b == sorted(b)
    assert len(b) == len(set(b))
