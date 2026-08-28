# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic registration and liveness for the EPD proxy's instance registry."""

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


class TestSelfRegistration:
    """What an instance reports, and when it reports at all."""

    @staticmethod
    def _state(ec_extra=None, ec_role=None, kv_role=None, port=8000):
        from types import SimpleNamespace

        from vllm.config.ec_transfer import ECTransferConfig
        from vllm.config.kv_transfer import KVTransferConfig

        ec_config = None
        if ec_extra is not None or ec_role is not None:
            ec_config = ECTransferConfig(
                ec_connector="ECExampleConnector" if ec_role else None,
                ec_role=ec_role,
                ec_connector_extra_config=ec_extra or {},
            )
        kv_config = (
            KVTransferConfig(kv_connector="NixlConnector", kv_role=kv_role)
            if kv_role
            else None
        )
        return SimpleNamespace(
            vllm_config=SimpleNamespace(
                ec_transfer_config=ec_config,
                kv_transfer_config=kv_config,
                parallel_config=SimpleNamespace(data_parallel_size=1),
            ),
            args=SimpleNamespace(host="127.0.0.1", port=port, ssl_certfile=None),
        )

    def test_a_statically_wired_deployment_announces_nothing(self):
        from vllm.distributed.ec_transfer.proxy import register as mod

        with patch.object(mod.ProxyRegistrar, "start"):
            assert mod.maybe_start(self._state()) is None
            assert mod.maybe_start(self._state(ec_role="ec_producer")) is None

    def test_an_instance_with_no_ec_role_still_registers(self):
        """A decode instance carries an EC config only to name the proxy.

        It moves no embeddings, but the proxy still has to know where to
        forward, so an absent role must not silence the announcement.
        """
        from vllm.distributed.ec_transfer.proxy import register as mod

        state = self._state(ec_extra={"proxy_url": "http://proxy:8000"})
        with patch.object(mod.ProxyRegistrar, "start"):
            registrar = mod.maybe_start(state)
        assert registrar is not None
        assert registrar.payload["role"] == "decode"
        assert registrar.payload["url"] == "http://127.0.0.1:8000"

    def test_roles_follow_what_the_instance_was_configured_to_do(self):
        from vllm.distributed.ec_transfer.proxy.register import infer_role

        encode = self._state(ec_role="ec_producer").vllm_config
        assert infer_role(encode) is InstanceRole.ENCODE
        prefill = self._state(ec_role="ec_consumer", kv_role="kv_producer").vllm_config
        assert infer_role(prefill) is InstanceRole.PREFILL
        decode = self._state(kv_role="kv_consumer").vllm_config
        assert infer_role(decode) is InstanceRole.DECODE
