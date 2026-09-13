# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry behaviour under roster churn, races and odd registrations."""

from unittest.mock import patch

import pytest

from vllm.distributed.ec_transfer.proxy.epd_proxy import EPDProxy, EPDProxyConfig
from vllm.distributed.ec_transfer.proxy.registry import (
    InstanceRecord,
    InstanceRegistry,
    InstanceRole,
)

ENCODE = InstanceRole.ENCODE
PREFILL = InstanceRole.PREFILL
DECODE = InstanceRole.DECODE


@pytest.fixture
def registry():
    return InstanceRegistry(probe_interval=0, fail_threshold=2, evicted_ttl=60.0)


async def _probe_round(registry, healthy, now=0.0):
    """One probe pass where only `healthy` URLs answer."""

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


def _probe_round_racing(registry, healthy, mutate, now=0.0):
    """Apply probe results to a registry that changed after they were issued.

    `_probe_once` snapshots its targets and then awaits, so anything that
    happens in that window sees results describing a registry that no longer
    exists. Reproducing that by racing coroutines is timing-dependent; doing
    the same three steps by hand is not.
    """
    targets = list(registry._live.values()) + list(registry._evicted.values())
    mutate()
    for record in targets:
        if record.url in healthy:
            registry._on_probe_success(record)
        else:
            registry._on_probe_failure(record, now)


class TestUnregisterIsTerminal:
    @pytest.fixture
    def registry(self):
        # `unregister` clears the failure run, so only a registry that evicts
        # on a single failure can be reached by one racing result.
        return InstanceRegistry(probe_interval=0, fail_threshold=1)

    def test_unregister_during_an_in_flight_probe_stays_gone(self, registry):
        """An operator draining an instance must not have it come back.

        The probe round snapshots its targets and then awaits, so a removal
        that lands inside that window is applied to a registry the results no
        longer describe. The run of failures is driven to the brink first, so
        the racing result is the one that would evict.
        """
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))

        _probe_round_racing(
            registry,
            healthy=set(),
            mutate=lambda: registry.unregister("http://e0:8000"),
        )

        assert registry.urls(ENCODE) == []
        assert registry.status()["encode"]["evicted"] == []

    @pytest.mark.asyncio
    async def test_a_drained_instance_is_not_revived_by_a_later_success(self, registry):
        """Landing on the evicted list is enough to come back on its own."""
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        _probe_round_racing(
            registry,
            healthy=set(),
            mutate=lambda: registry.unregister("http://e0:8000"),
        )

        await _probe_round(registry, healthy={"http://e0:8000"})
        assert registry.urls(ENCODE) == []


class TestRosterChurn:
    def test_re_registering_moves_an_instance_to_the_back(self, registry):
        """Fan-out order follows registration order, so it must stay sane."""
        for index in range(3):
            registry.register(InstanceRecord(ENCODE, f"http://e{index}:8000"))
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        assert registry.urls(ENCODE) == [
            "http://e0:8000",
            "http://e1:8000",
            "http://e2:8000",
        ]

    def test_an_instance_can_change_role(self, registry):
        """A PD instance restarted as a dedicated prefill must not be in both."""
        registry.register(InstanceRecord(DECODE, "http://x:8000"))
        registry.register(InstanceRecord(PREFILL, "http://x:8000"))
        assert registry.urls(DECODE) == []
        assert registry.urls(PREFILL) == ["http://x:8000"]

    def test_re_registering_updates_the_transfer_addresses(self, registry):
        """A restarted consumer may bind different ports."""
        registry.register(
            InstanceRecord(DECODE, "http://d:8000", ec_zmq_addrs=["tcp://d:1"])
        )
        registry.register(
            InstanceRecord(DECODE, "http://d:8000", ec_zmq_addrs=["tcp://d:2"])
        )
        assert registry.pick(DECODE).ec_zmq_addrs == ["tcp://d:2"]

    def test_fan_out_larger_than_the_roster_reuses_instances(self, registry):
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        registry.register(InstanceRecord(ENCODE, "http://e1:8000"))
        picked = [record.url for record in registry.pick_many(ENCODE, 5)]
        assert len(picked) == 5
        assert set(picked) == {"http://e0:8000", "http://e1:8000"}

    @pytest.mark.asyncio
    async def test_losing_every_encoder_is_a_refusal_not_a_crash(self, registry):
        proxy = EPDProxy(EPDProxyConfig(), registry)
        registry.register(InstanceRecord(DECODE, "http://d:8000"))
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        for _ in range(registry._fail_threshold):
            await _probe_round(registry, healthy={"http://d:8000"})
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as excinfo:
            proxy.route(num_items=1)
        assert excinfo.value.status_code == 503
        # Text-only traffic is unaffected by the encoders being gone.
        assert proxy.route(num_items=0).decode.url == "http://d:8000"


class TestProbeRobustness:
    @pytest.mark.asyncio
    async def test_a_probe_that_raises_counts_as_a_failure(self, registry):
        """A connection error is how a dead instance usually presents."""
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))

        async def boom(self, session, url):
            raise ConnectionRefusedError(url)

        for _ in range(registry._fail_threshold):
            with patch.object(InstanceRegistry, "_probe", boom):
                await registry._probe_once(session=None)
        assert registry.urls(ENCODE) == []

    @pytest.mark.asyncio
    async def test_re_registering_clears_the_failure_run(self, registry):
        """A restarted instance starts from zero, not one probe from eviction."""
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        for _ in range(registry._fail_threshold - 1):
            await _probe_round(registry, healthy=set())

        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        await _probe_round(registry, healthy=set())
        assert registry.urls(ENCODE) == ["http://e0:8000"]


class TestReplicaSelection:
    def test_data_parallel_replicas_share_the_push_load(self, registry):
        """A consumer with several replicas must not funnel every push to one.

        The encoder push has to land on the replica that will run the
        request, so naming replica 0 every time both hot-spots it and leaves
        the others unreachable.
        """
        proxy = EPDProxy(EPDProxyConfig(), registry)
        registry.register(InstanceRecord(ENCODE, "http://e:8000"))
        registry.register(
            InstanceRecord(
                DECODE,
                "http://d:8000",
                ec_zmq_addrs=["tcp://d:1", "tcp://d:2", "tcp://d:3"],
                dp_size=3,
            )
        )
        routes = [proxy.route(num_items=1) for _ in range(6)]
        assert {route.consumer_zmq for route in routes} == {
            "tcp://d:1",
            "tcp://d:2",
            "tcp://d:3",
        }
        # The request must run on the replica the push was aimed at, so the
        # rank is named to the consumer too.
        assert [route.dp_rank for route in routes] == [0, 1, 2, 0, 1, 2]
