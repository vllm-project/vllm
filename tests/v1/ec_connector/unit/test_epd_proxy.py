# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing and registration behaviour of the EPD proxy."""

import asyncio
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from examples.disaggregated.disaggregated_encoder import (
    disagg_epd_proxy as proxy_module,
)
from examples.disaggregated.disaggregated_encoder.disagg_epd_proxy import (
    EPDProxy,
    EPDProxyConfig,
    InstanceRecord,
    InstanceRegistry,
    InstanceRole,
    RegistrationServer,
    build_app,
    extract_mm_items,
)

ENCODE = InstanceRole.ENCODE
PREFILL = InstanceRole.PREFILL
DECODE = InstanceRole.DECODE

IMAGE_ITEM = {"type": "image_url", "image_url": {"url": "http://img/0.png"}}


@pytest.fixture
def proxy():
    registry = InstanceRegistry(probe_interval=0)
    return EPDProxy(EPDProxyConfig(), registry)


class TestRouting:
    def test_no_decode_instance_is_service_unavailable(self, proxy):
        """The proxy comes up before anything registers, so this is normal."""
        with pytest.raises(HTTPException) as excinfo:
            proxy.route(num_items=0)
        assert excinfo.value.status_code == 503

    def test_media_without_an_encoder_is_service_unavailable(self, proxy):
        proxy.registry.register(InstanceRecord(DECODE, "http://d0:8000"))
        assert proxy.route(num_items=0).decode.url == "http://d0:8000"
        with pytest.raises(HTTPException) as excinfo:
            proxy.route(num_items=1)
        assert excinfo.value.status_code == 503

    def test_prefill_is_optional(self, proxy):
        """An E+PD deployment registers no prefill instance at all."""
        proxy.registry.register(InstanceRecord(DECODE, "http://d0:8000"))
        assert proxy.route(num_items=0).prefill is None

    def test_encoder_roster_tracks_new_instances(self, proxy):
        proxy.registry.register(InstanceRecord(DECODE, "http://d0:8000"))
        for index in range(2):
            proxy.registry.register(InstanceRecord(ENCODE, f"http://e{index}:8000"))
        assert proxy.route(num_items=3).encoder_urls == [
            "http://e0:8000",
            "http://e1:8000",
        ]
        proxy.registry.unregister("http://e0:8000")
        assert proxy.route(num_items=3).encoder_urls == ["http://e1:8000"]

    def test_static_decode_requires_a_registered_prefill(self):
        app = build_app(EPDProxyConfig(decode_servers_urls=["http://d0:8000/"]))
        proxy = app.state.proxy
        with pytest.raises(HTTPException) as excinfo:
            proxy.route(num_items=0)
        assert excinfo.value.status_code == 503
        proxy.registry.register(InstanceRecord(PREFILL, "http://p0:8000"))
        route = proxy.route(num_items=0)
        assert route.decode.url == "http://d0:8000"
        assert route.consumer.url == "http://p0:8000"
        proxy.registry.unregister("http://p0:8000")
        with pytest.raises(HTTPException):
            proxy.route(num_items=0)


class TestConsumerAddress:
    """Which stage receives the embedding depends on the topology."""

    def test_shared_storage_connectors_name_no_target(self, proxy):
        """Pin the registered replica without requiring a push endpoint."""
        proxy.registry.register(
            InstanceRecord(DECODE, "http://d0:8000", dp_rank=1, dp_size=2)
        )
        route = proxy.route(num_items=0)
        assert route.consumer_zmq is None
        assert route.dp_rank == 1

    def test_decode_is_the_consumer_when_prefill_is_not_split_out(self, proxy):
        proxy.registry.register(
            InstanceRecord(DECODE, "http://d0:8000", ec_zmq_addrs=["tcp://d0:20001"])
        )
        assert proxy.route(num_items=0).consumer_zmq == "tcp://d0:20001"

    def test_prefill_is_the_consumer_when_it_is_split_out(self, proxy):
        """In E+P+D the prefill instance consumes the embedding, not decode.

        Reading the address out of whichever list it came from -- rather than
        from the record that reported one -- sends the push to the wrong
        instance in this topology.
        """
        proxy.registry.register(
            InstanceRecord(PREFILL, "http://p0:8000", ec_zmq_addrs=["tcp://p0:20001"])
        )
        proxy.registry.register(InstanceRecord(DECODE, "http://d0:8000"))
        assert proxy.route(num_items=0).consumer_zmq == "tcp://p0:20001"


class TestProxyApi:
    @pytest.fixture
    def client(self):
        config = EPDProxyConfig(registry_address="tcp://127.0.0.1:0", probe_interval=0)
        with TestClient(build_app(config)) as client:
            yield client

    def test_proxy_serves_before_anything_registers(self, client):
        assert client.get("/health").status_code == 200
        assert client.get("/instances").json()["encode"]["live"] == []

    def test_requests_are_refused_until_a_decoder_registers(self, client):
        response = client.post(
            "/v1/chat/completions", json={"model": "m", "messages": []}
        )
        assert response.status_code == 503
        assert client.get("/v1/models").status_code == 503


def test_registration_server_updates_registry(proxy):
    server = RegistrationServer("tcp://127.0.0.1:0", proxy.registry)
    payload = {
        "operation": "register",
        "role": "decode",
        "url": "http://d0:8000/",
        "engine_id": "engine",
        "dp_rank": 0,
        "dp_size": 1,
        "ec_zmq_addrs": ["tcp://d0:20001"],
    }
    server._handle(payload)
    assert proxy.route(0).consumer_zmq == "tcp://d0:20001"
    server._handle({**payload, "operation": "unregister"})
    assert proxy.registry.pick(DECODE) is None


def test_extract_mm_items_finds_media_across_messages():
    req = {
        "messages": [
            {"role": "user", "content": "plain text"},
            {"role": "user", "content": [IMAGE_ITEM, {"type": "text", "text": "hi"}]},
            {"role": "user", "content": [IMAGE_ITEM]},
        ]
    }
    assert extract_mm_items(req) == [IMAGE_ITEM, IMAGE_ITEM]


@pytest.mark.parametrize("push", [False, True])
def test_encoder_handles_and_json_metadata_survive_rewrite(proxy, push, monkeypatch):
    """Keep main's hash-keyed handles and per-item push IDs after the proxy move."""
    proxy.registry.register(InstanceRecord(ENCODE, "http://encoder"))
    proxy.registry.register(
        InstanceRecord(
            DECODE,
            "http://decode",
            ec_zmq_addrs=["tcp://decode:14579"] if push else [],
        )
    )
    handle = {"metadata": {"image_grid_thw": [[1, 2, 3]]}, "peer_port": 5601}
    response = Mock(
        status=200,
        read=AsyncMock(
            return_value=msgspec.json.encode(
                {"ec_transfer_params": {"engine-hash": handle}}
            )
        ),
    )
    session = Mock(post=AsyncMock(return_value=response))
    monkeypatch.setattr(proxy_module, "encode_session", session)
    original = {"messages": [{"role": "user", "content": [IMAGE_ITEM, IMAGE_ITEM]}]}
    route = proxy.route(2)
    prepared, _, _ = asyncio.run(
        proxy_module.prepare_for_decode(
            original, "request", route.encoder_urls, None, route.consumer_zmq
        )
    )
    assert "ec_transfer_params" not in original
    assert prepared["ec_transfer_params"]["engine-hash"] == handle
    items = prepared["ec_transfer_params"]["ec_items"]
    assert [item["mm_hash"] for item in items] == ["engine-hash", "engine-hash"]
    assert items[0]["transfer_id"] != items[1]["transfer_id"]
    for index, call in enumerate(session.post.call_args_list):
        sent = msgspec.json.decode(call.kwargs["data"]).get("ec_transfer_params")
        if push:
            assert sent == {
                "consumer_zmq": "tcp://decode:14579",
                "ec_items": [{"transfer_id": items[index]["transfer_id"]}],
            }
        else:
            assert sent is None
        assert prepared["messages"][0]["content"][index]["image_embeds"] == {
            "image_grid_thw": [1, 2, 3]
        }
