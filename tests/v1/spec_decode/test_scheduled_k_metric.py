# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The dynamic-SD schedule's K is recorded even when it selects 0."""

from types import SimpleNamespace

import prometheus_client
import pytest

from vllm.v1.spec_decode.metrics import SpecDecodingProm


@pytest.fixture
def prom():
    registry = prometheus_client.CollectorRegistry()
    original = SpecDecodingProm._counter_cls

    class _Counter(prometheus_client.Counter):
        def __init__(self, *args, **kwargs):
            kwargs["registry"] = registry
            super().__init__(*args, **kwargs)

    SpecDecodingProm._counter_cls = _Counter
    try:
        yield (
            SpecDecodingProm(
                speculative_config=SimpleNamespace(num_speculative_tokens=3),
                labelnames=["model"],
                per_engine_labelvalues={0: ["m"]},
            ),
            registry,
        )
    finally:
        SpecDecodingProm._counter_cls = original


def _value(registry, k):
    return registry.get_sample_value(
        "vllm:spec_decode_scheduled_steps_total",
        {"model": "m", "num_spec_tokens": str(k)},
    )


def test_every_k_including_zero_has_a_counter(prom):
    _, registry = prom
    for k in range(4):
        assert _value(registry, k) == 0.0


def test_k_zero_steps_are_counted(prom):
    p, registry = prom
    for _ in range(5):
        p.observe_scheduled_k(0)
    p.observe_scheduled_k(3)

    assert _value(registry, 0) == 5.0
    assert _value(registry, 3) == 1.0
    assert _value(registry, 1) == 0.0


def test_out_of_range_k_is_ignored(prom):
    p, registry = prom
    p.observe_scheduled_k(4)
    p.observe_scheduled_k(-1)

    for k in range(4):
        assert _value(registry, k) == 0.0


def test_unknown_engine_is_ignored(prom):
    p, registry = prom
    p.observe_scheduled_k(1, engine_idx=7)

    assert _value(registry, 1) == 0.0
