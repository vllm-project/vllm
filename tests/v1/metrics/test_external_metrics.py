# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
import pytest

from vllm.v1.metrics.external import (
    _reset_external_metrics_providers_for_tests,
    collect_external_metrics,
    has_external_metrics_providers,
    register_external_metrics_provider,
    unregister_external_metrics_provider,
)
from vllm.v1.metrics.stats import SchedulerStats

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def reset_external_metrics_providers():
    _reset_external_metrics_providers_for_tests()
    yield
    _reset_external_metrics_providers_for_tests()


def test_external_metrics_provider_is_namespaced_and_rate_limited():
    calls = 0

    def collect():
        nonlocal calls
        calls += 1
        return {"used_bytes": 42, "labels": {"pool": "kv"}}

    register_external_metrics_provider(
        "example.plugin", collect, collection_interval_s=2.0
    )

    assert collect_external_metrics(now=10.0) == {
        "example.plugin": {"used_bytes": 42, "labels": {"pool": "kv"}}
    }
    assert collect_external_metrics(now=11.9) is None
    assert collect_external_metrics(now=12.0) == {
        "example.plugin": {"used_bytes": 42, "labels": {"pool": "kv"}}
    }
    assert calls == 2


def test_external_metrics_provider_failures_are_isolated():
    def fail():
        raise RuntimeError("provider failed")

    register_external_metrics_provider("failing", fail)
    register_external_metrics_provider("healthy", lambda: {"value": 1})

    assert collect_external_metrics(now=10.0) == {"healthy": {"value": 1}}


@pytest.mark.parametrize(
    "payload",
    [
        {"bad": object()},
        {"bad": {1: "non-string key"}},
        {"bad": float("nan")},
        {"bad": [float("inf")]},
        {"bad": -(2**63) - 1},
        {"bad": 2**64},
        {"bad": {"nested": [2**64]}},
        ["not", "a", "mapping"],
    ],
)
def test_invalid_external_metrics_payload_is_isolated(payload: object):
    register_external_metrics_provider(
        "invalid",
        lambda: payload,  # type: ignore[arg-type,return-value]
    )
    register_external_metrics_provider("healthy", lambda: {"value": 1})

    assert collect_external_metrics(now=10.0) == {"healthy": {"value": 1}}


@pytest.mark.parametrize("name", ["", "1plugin", "bad name", "bad/name"])
def test_external_metrics_provider_name_is_validated(name: str):
    with pytest.raises(ValueError, match="provider names"):
        register_external_metrics_provider(name, lambda: {})


def test_external_metrics_provider_registration_is_unique():
    assert not has_external_metrics_providers()
    register_external_metrics_provider("example", lambda: {})
    assert has_external_metrics_providers()

    with pytest.raises(ValueError, match="already registered"):
        register_external_metrics_provider("example", lambda: {})

    unregister_external_metrics_provider("example")
    assert not has_external_metrics_providers()
    register_external_metrics_provider("example", lambda: {"value": 2})
    assert collect_external_metrics(now=10.0) == {"example": {"value": 2}}


def test_external_metrics_provider_configuration_is_validated():
    with pytest.raises(TypeError, match="must be callable"):
        register_external_metrics_provider("example", object())  # type: ignore[arg-type]

    for interval in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite value greater than zero"):
            register_external_metrics_provider(
                "example", lambda: {}, collection_interval_s=interval
            )


@pytest.mark.parametrize("integer", [-(2**63), 2**64 - 1])
def test_external_metrics_snapshot_is_msgpack_serializable(integer: int):
    register_external_metrics_provider(
        "example.plugin",
        lambda: {
            "used_bytes": integer,
            "labels": {"pool": "kv"},
            "buckets": [1.0, 2.0, integer],
            "enabled": True,
            "optional": None,
        },
    )
    stats = SchedulerStats(
        external_metrics=collect_external_metrics(now=10.0),
    )

    encoded = msgspec.msgpack.encode(stats)
    decoded = msgspec.msgpack.decode(encoded, type=SchedulerStats)

    assert decoded.external_metrics == stats.external_metrics
    assert decoded.external_metrics is not None
    assert decoded.external_metrics["example.plugin"]["used_bytes"] == integer
    assert decoded.external_metrics["example.plugin"]["enabled"] is True
