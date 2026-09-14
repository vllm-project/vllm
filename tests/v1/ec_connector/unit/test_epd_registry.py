# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic registration and liveness for the EPD proxy's instance registry."""

from unittest.mock import Mock, patch

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
    async def test_consecutive_failures_stop_routing(self, registry):
        registry.register(InstanceRecord(ENCODE, "http://e0:8000"))
        registry.register(InstanceRecord(ENCODE, "http://e1:8000"))
        for _ in range(registry._fail_threshold):
            await _probe_round(registry, healthy={"http://e1:8000"})
        assert registry.urls(ENCODE) == ["http://e1:8000"]
        assert registry.status()["encode"]["evicted"] == ["http://e0:8000"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_static", [False, True])
    async def test_a_recovered_instance_rejoins_without_re_registering(
        self, registry, is_static
    ):
        """Restarting every encoder to recover from a blip is not acceptable."""
        role = DECODE if is_static else ENCODE
        registry.register(InstanceRecord(role, "http://e0:8000", is_static=is_static))
        for _ in range(registry._fail_threshold):
            await _probe_round(registry, healthy=set())
        assert registry.urls(role) == []

        if is_static:
            await _probe_round(registry, healthy=set(), now=120)

        await _probe_round(registry, healthy={"http://e0:8000"}, now=121)
        assert registry.urls(role) == ["http://e0:8000"]

    @pytest.mark.asyncio
    async def test_a_heartbeat_does_not_override_failed_health_checks(self, registry):
        record = InstanceRecord(ENCODE, "http://e0:8000", engine_id="engine")
        registry.register(record)
        for _ in range(registry._fail_threshold):
            await _probe_round(registry, healthy=set())
        registry.register(record)
        assert registry.urls(ENCODE) == []


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
                parallel_config=SimpleNamespace(
                    data_parallel_size=1, data_parallel_index=0
                ),
            ),
            args=SimpleNamespace(host="127.0.0.1", port=port, ssl_certfile=None),
        )

    def test_a_statically_wired_deployment_announces_nothing(self):
        from vllm.distributed.ec_transfer.proxy import register as mod

        with patch.object(mod.ProxyRegistrar, "start"):
            assert mod.start_worker_registration(self._state().vllm_config) is None
            assert (
                mod.start_worker_registration(
                    self._state(ec_role="ec_producer").vllm_config
                )
                is None
            )

    def test_an_instance_with_no_ec_role_does_not_register(self):
        from vllm.distributed.ec_transfer.proxy import register as mod

        state = self._state(
            ec_extra={
                "proxy_registry_addr": "tcp://proxy:14580",
                "_http_address": "http://127.0.0.1:8000",
            }
        )
        with patch.object(mod.ProxyRegistrar, "start"):
            assert mod.start_worker_registration(state.vllm_config) is None

    @pytest.mark.parametrize("rank", [0, 1])
    @pytest.mark.parametrize("backend", ["ECExampleConnector", "ECCPUConnector"])
    def test_worker_registration_is_owned_by_one_rank(self, rank, backend):
        from vllm.distributed.ec_transfer import ec_transfer_state as state_mod
        from vllm.distributed.ec_transfer.proxy import register as mod

        config = self._state(
            ec_role="ec_consumer",
            ec_extra={
                "proxy_registry_addr": "tcp://proxy:14580",
                "_http_address": "http://127.0.0.1:8000",
            },
        ).vllm_config
        config.ec_transfer_config.ec_connector = backend
        parallel = "vllm.distributed.parallel_state"
        with (
            patch.object(state_mod, "_EC_CONNECTOR_AGENT", None),
            patch.object(state_mod, "_EC_REGISTRAR", None),
            patch.object(state_mod.ECConnectorFactory, "create_connector") as factory,
            patch(f"{parallel}.get_tp_group", return_value=Mock(rank_in_group=rank)),
            patch(f"{parallel}.get_pp_group", return_value=Mock(rank_in_group=0)),
            patch(f"{parallel}.get_pcp_group", return_value=Mock(rank_in_group=0)),
            patch.object(mod.ProxyRegistrar, "start") as start,
            patch.object(mod.ProxyRegistrar, "close") as close,
        ):
            state_mod.ensure_ec_transfer_initialized(config)
            state_mod.ensure_ec_transfer_initialized(config)
            assert start.call_count == (1 if rank == 0 else 0)
            state_mod.ensure_ec_transfer_shutdown()
            assert close.call_count == start.call_count
            factory.return_value.shutdown.assert_called_once()

    def test_roles_follow_what_the_instance_was_configured_to_do(self):
        from vllm.distributed.ec_transfer.proxy.register import infer_role

        encode = self._state(ec_role="ec_producer").vllm_config
        assert infer_role(encode) is InstanceRole.ENCODE
        prefill = self._state(ec_role="ec_consumer", kv_role="kv_producer").vllm_config
        assert infer_role(prefill) is InstanceRole.PREFILL
        decode = self._state(kv_role="kv_consumer").vllm_config
        assert infer_role(decode) is InstanceRole.DECODE
