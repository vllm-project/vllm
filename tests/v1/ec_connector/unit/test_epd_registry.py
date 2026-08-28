# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic registration and liveness for the EPD proxy's instance registry."""

from collections import Counter
from unittest.mock import patch

import pytest

from vllm.distributed.ec_transfer.proxy.registry import (
    InstanceRecord,
    InstanceRegistry,
    InstanceRole,
)

ENCODE = InstanceRole.ENCODE
DECODE = InstanceRole.DECODE


@pytest.fixture
def registry():
    return InstanceRegistry(probe_interval=0, fail_threshold=3, evicted_ttl=60.0)


def _fill(registry, role, count, prefix="http://h"):
    for index in range(count):
        registry.register(InstanceRecord(role, f"{prefix}{index}:8000"))


async def _probe_round(registry, healthy: set[str], now: float = 0.0):
    """Run one probe pass where only `healthy` URLs answer."""

    async def fake_probe(self, session, url):
        return url in healthy

    with (
        patch.object(InstanceRegistry, "_probe", fake_probe),
        patch(
            "vllm.distributed.ec_transfer.proxy.registry.time.monotonic",
            return_value=now,
        ),
    ):
        await registry._probe_once(session=None)


class TestRegistration:
    def test_registering_is_idempotent_and_reports_novelty(self, registry):
        record = InstanceRecord(ENCODE, "http://e0:8000")
        assert registry.register(record) is True
        assert registry.register(record) is False
        assert registry.urls(ENCODE) == ["http://e0:8000"]

    def test_roles_are_routed_separately(self, registry):
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        registry.register(InstanceRecord(DECODE, "http://d0:8000"))
        assert registry.urls(ENCODE) == ["http://e0:8000"]
        assert registry.urls(DECODE) == ["http://d0:8000"]

    def test_consumer_reports_its_own_transfer_addresses(self, registry):
        """Only the EC consumer knows these, so they ride with its record.

        Deriving them from a positionally-aligned CLI list instead breaks the
        moment an instance is added or removed.
        """
        registry.register(
            InstanceRecord(
                DECODE,
                "http://d0:8000",
                ec_zmq_addrs=["tcp://d0:20001", "tcp://d0:20002"],
                dp_size=2,
            )
        )
        picked = registry.pick(DECODE)
        assert picked.ec_zmq_addrs == ["tcp://d0:20001", "tcp://d0:20002"]
        assert picked.dp_size == 2

    def test_unregister_is_terminal(self, registry):
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        assert registry.unregister("http://e0:8000") is True
        assert registry.unregister("http://e0:8000") is False
        assert registry.pick(ENCODE) is None


class TestRoundRobin:
    def test_cursor_survives_a_registration(self, registry):
        """A new instance must not restart the rotation at the first one.

        Rebuilding an `itertools.cycle` over the roster -- the usual way to
        round-robin a mutable list -- resets the position, so every
        registration hot-spots whichever instance happens to be first.
        """
        _fill(registry, ENCODE, 3, prefix="http://e")
        assert [r.url for r in registry.pick_many(ENCODE, 2)] == [
            "http://e0:8000",
            "http://e1:8000",
        ]
        registry.register(InstanceRecord(ENCODE, "http://e3:8000"))
        assert [r.url for r in registry.pick_many(ENCODE, 2)] == [
            "http://e2:8000",
            "http://e3:8000",
        ]

    def test_fan_out_spreads_evenly_over_single_item_requests(self, registry):
        _fill(registry, ENCODE, 3, prefix="http://e")
        hits = Counter(registry.pick_many(ENCODE, 1)[0].url for _ in range(9))
        assert set(hits.values()) == {3}

    def test_no_instances_yields_nothing(self, registry):
        assert registry.pick_many(ENCODE, 2) == []
        assert registry.pick(ENCODE) is None


class TestLiveness:
    @pytest.mark.asyncio
    async def test_one_missed_probe_does_not_evict(self, registry):
        """A busy encoder can miss a probe; only a run of them means death."""
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        for _ in range(registry._fail_threshold - 1):
            await _probe_round(registry, healthy=set())
        assert registry.urls(ENCODE) == ["http://e0:8000"]

    @pytest.mark.asyncio
    async def test_a_success_clears_the_failure_run(self, registry):
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        await _probe_round(registry, healthy=set())
        await _probe_round(registry, healthy=set())
        await _probe_round(registry, healthy={"http://e0:8000"})
        await _probe_round(registry, healthy=set())
        assert registry.urls(ENCODE) == ["http://e0:8000"]

    @pytest.mark.asyncio
    async def test_consecutive_failures_stop_routing(self, registry):
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        registry.register(InstanceRecord(ENCODE, "http://e1:8000"))
        for _ in range(registry._fail_threshold):
            await _probe_round(registry, healthy={"http://e1:8000"})
        assert registry.urls(ENCODE) == ["http://e1:8000"]
        assert registry.status()["encode"]["evicted"] == ["http://e0:8000"]

    @pytest.mark.asyncio
    async def test_a_recovered_instance_rejoins_without_re_registering(self, registry):
        """Restarting every encoder to recover from a blip is not acceptable."""
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        for _ in range(registry._fail_threshold):
            await _probe_round(registry, healthy=set())
        assert registry.urls(ENCODE) == []

        await _probe_round(registry, healthy={"http://e0:8000"})
        assert registry.urls(ENCODE) == ["http://e0:8000"]

    @pytest.mark.asyncio
    async def test_a_lasting_outage_is_eventually_forgotten(self, registry):
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        for _ in range(registry._fail_threshold):
            await _probe_round(registry, healthy=set(), now=0.0)
        assert registry.status()["encode"]["evicted"] == ["http://e0:8000"]

        await _probe_round(registry, healthy=set(), now=registry._evicted_ttl + 1)
        assert registry.status()["encode"]["evicted"] == []

    @pytest.mark.asyncio
    async def test_re_registering_revives_an_evicted_instance(self, registry):
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        for _ in range(registry._fail_threshold):
            await _probe_round(registry, healthy=set())
        assert registry.urls(ENCODE) == []

        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        assert registry.urls(ENCODE) == ["http://e0:8000"]
        assert registry.status()["encode"]["evicted"] == []

    @pytest.mark.asyncio
    async def test_probing_an_empty_registry_is_a_no_op(self, registry):
        await _probe_round(registry, healthy=set())
        assert registry.status()["encode"]["live"] == []
