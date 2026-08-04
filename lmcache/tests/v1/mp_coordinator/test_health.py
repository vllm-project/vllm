# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the health-check eviction logic."""

# Standard
import time

# First Party
from lmcache.v1.mp_coordinator.app import evict_stale
from lmcache.v1.mp_coordinator.registry import InstanceRegistry, MPInstance


def _instance(instance_id: str, heartbeat: float) -> MPInstance:
    return MPInstance(
        instance_id=instance_id,
        ip="127.0.0.1",
        http_port=8080,
        registration_time=heartbeat,
        last_heartbeat_time=heartbeat,
    )


def test_evict_stale_removes_only_expired():
    registry = InstanceRegistry()
    now = time.monotonic()
    registry.register(_instance("fresh", now))
    registry.register(_instance("old", now - 100.0))

    evicted = evict_stale(registry, instance_timeout=30.0)

    assert evicted == ["old"]
    assert registry.contains("fresh")
    assert not registry.contains("old")


def test_evict_stale_noop_when_all_fresh():
    registry = InstanceRegistry()
    registry.register(_instance("a", time.monotonic()))
    assert evict_stale(registry, instance_timeout=30.0) == []
    assert registry.contains("a")
