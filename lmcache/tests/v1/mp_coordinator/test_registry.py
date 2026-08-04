# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the mp coordinator instance registry."""

# Standard
import time

# First Party
from lmcache.v1.mp_coordinator.registry import InstanceRegistry, MPInstance


def _instance(instance_id: str, heartbeat: float = 0.0) -> MPInstance:
    """Build a pure-membership MPInstance for tests."""
    now = heartbeat or time.time()
    return MPInstance(
        instance_id=instance_id,
        ip="127.0.0.1",
        http_port=8080,
        registration_time=now,
        last_heartbeat_time=now,
    )


def test_register_and_get():
    registry = InstanceRegistry()
    instance = _instance("a")
    registry.register(instance)
    assert registry.contains("a")
    assert registry.get("a") is instance
    assert registry.get("missing") is None


def test_register_returns_replaced_flag():
    registry = InstanceRegistry()
    assert registry.register(_instance("a")) is False
    assert registry.register(_instance("a")) is True


def test_deregister_returns_instance():
    registry = InstanceRegistry()
    instance = _instance("a")
    registry.register(instance)
    removed = registry.deregister("a")
    assert removed is instance
    assert not registry.contains("a")
    assert registry.deregister("a") is None


def test_all_instances_snapshot_is_independent():
    registry = InstanceRegistry()
    registry.register(_instance("a"))
    registry.register(_instance("b"))
    snapshot = registry.all_instances()
    assert {n.instance_id for n in snapshot} == {"a", "b"}
    registry.deregister("a")
    # Snapshot taken earlier is unaffected by later mutation.
    assert {n.instance_id for n in snapshot} == {"a", "b"}


def test_update_heartbeat():
    registry = InstanceRegistry()
    instance = _instance("a", heartbeat=100.0)
    registry.register(instance)
    assert registry.update_heartbeat("a", 200.0) is True
    assert registry.get("a").last_heartbeat_time == 200.0
    # Unknown instance signals a needed re-register.
    assert registry.update_heartbeat("missing", 300.0) is False


def test_stale_detects_expired():
    registry = InstanceRegistry()
    # stale() compares against the monotonic clock, so seed heartbeat times
    # from the same clock.
    now = time.monotonic()
    registry.register(_instance("fresh", heartbeat=now))
    registry.register(_instance("old", heartbeat=now - 100.0))
    stale = registry.stale(timeout=30.0)
    assert stale == ["old"]


def test_random_instance_empty_returns_none():
    registry = InstanceRegistry()
    assert registry.random_instance() is None


def test_random_instance_single_is_deterministic():
    registry = InstanceRegistry()
    inst = _instance("only")
    registry.register(inst)
    # Sampling a 1-element population is deterministic.
    assert registry.random_instance() is inst


def test_random_instance_samples_from_live_set():
    # Standard
    import random as _random

    registry = InstanceRegistry()
    for letter in "abc":
        registry.register(_instance(letter))
    # Deterministic seed → reproducible assertion that random_instance
    # actually samples from the registered set (and only it).
    _random.seed(7)
    seen = {registry.random_instance().instance_id for _ in range(50)}
    assert seen <= {"a", "b", "c"}
    # With 50 picks across 3 items the chance of missing one is
    # (2/3)**50 ≈ 1.6e-9 — close enough to "covers the set" for
    # a sanity test with a fixed seed.
    assert seen == {"a", "b", "c"}
