# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic registration and liveness for the EPD proxy's instance registry."""

from unittest.mock import patch

import pytest

from examples.disaggregated.disaggregated_encoder.disagg_epd_proxy import (
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
            "examples.disaggregated.disaggregated_encoder.disagg_epd_proxy.time.monotonic",
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

    def test_transfer_addresses_stay_with_the_registered_consumer(self, registry):
        """Roster updates must not shift a consumer's transfer addresses."""
        registry.register(InstanceRecord(DECODE, "http://d0:8000"))
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

    def test_round_robin_survives_registration(self, registry):
        _fill(registry, ENCODE, 2)
        assert [record.url for record in registry.pick_many(ENCODE, 2)] == [
            "http://h0:8000",
            "http://h1:8000",
        ]
        registry.register(InstanceRecord(ENCODE, "http://h2:8000"))
        assert registry.pick(ENCODE).url == "http://h2:8000"


class TestLiveness:
    @pytest.mark.asyncio
    async def test_identical_reregistration_preserves_inflight_probe_results(
        self, registry
    ):
        """Repeated registrations must not suppress eviction or recovery."""
        url = "http://e0:8000"
        registry.register(InstanceRecord(ENCODE, url))
        healthy = False

        async def probe(self, session, url):
            registry.register(InstanceRecord(ENCODE, url))
            return healthy

        with patch.object(InstanceRegistry, "_probe", probe):
            for _ in range(registry._fail_threshold):
                await registry._probe_once(None)
            assert registry.status()["encode"] == {"live": [], "evicted": [url]}

            healthy = True
            await registry._probe_once(None)
            assert registry.status()["encode"] == {"live": [url], "evicted": []}

    @pytest.mark.asyncio
    async def test_consecutive_failures_stop_routing(self, registry):
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        registry.register(InstanceRecord(ENCODE, "http://e1:8000"))
        for _ in range(registry._fail_threshold):
            await _probe_round(registry, healthy={"http://e1:8000"})
        assert registry.urls(ENCODE) == ["http://e1:8000"]
        assert registry.status()["encode"]["evicted"] == ["http://e0:8000"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("role", [ENCODE, DECODE])
    async def test_a_recovered_instance_rejoins_without_re_registering(
        self, registry, role
    ):
        """Restarting every encoder to recover from a blip is not acceptable."""
        registry.register(InstanceRecord(role, "http://e0:8000"))
        for _ in range(registry._fail_threshold):
            await _probe_round(registry, healthy=set())
        assert registry.urls(role) == []

        await _probe_round(registry, healthy={"http://e0:8000"}, now=1)
        assert registry.urls(role) == ["http://e0:8000"]

    @pytest.mark.asyncio
    async def test_re_registration_does_not_override_failed_health_checks(
        self, registry
    ):
        record = InstanceRecord(ENCODE, "http://e0:8000")
        registry.register(record)
        for _ in range(registry._fail_threshold):
            await _probe_round(registry, healthy=set())
        registry.register(record)
        assert registry.urls(ENCODE) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("healthy", [False, True])
    async def test_inflight_probe_cannot_restore_unregistered_instance(
        self, registry, healthy
    ):
        record = InstanceRecord(ENCODE, "http://e0:8000")
        registry.register(record)
        for _ in range(registry._fail_threshold):
            await _probe_round(registry, healthy=set())

        async def probe(self, session, url):
            registry.unregister(url)
            return healthy

        with patch.object(InstanceRegistry, "_probe", probe):
            await registry._probe_once(None)
        assert registry.status()["encode"] == {"live": [], "evicted": []}
