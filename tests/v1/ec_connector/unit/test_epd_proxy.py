# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing and registration behaviour of the EPD proxy."""

import asyncio
from typing import Any
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
    build_app,
    extract_mm_items,
)

ENCODE = InstanceRole.ENCODE
PREFILL = InstanceRole.PREFILL
DECODE = InstanceRole.DECODE
PD = InstanceRole.PREFILL_DECODE

IMAGE_ITEM = {"type": "image_url", "image_url": {"url": "http://img/0.png"}}


@pytest.fixture
def proxy():
    registry = InstanceRegistry(probe_interval=0)
    return EPDProxy(registry)


class TestRouting:
    def test_no_decode_instance_is_service_unavailable(self, proxy):
        """The proxy comes up before anything registers, so this is normal."""
        with pytest.raises(HTTPException) as excinfo:
            proxy.route(num_items=0)
        assert excinfo.value.status_code == 503

    def test_media_without_an_encoder_is_service_unavailable(self, proxy):
        proxy.registry.register(InstanceRecord(PD, "http://d0:8000"))
        assert proxy.route(num_items=0).decode.url == "http://d0:8000"
        with pytest.raises(HTTPException) as excinfo:
            proxy.route(num_items=1)
        assert excinfo.value.status_code == 503

    def test_prefill_is_optional(self, proxy):
        """An E+PD deployment registers no prefill instance at all."""
        proxy.registry.register(InstanceRecord(PD, "http://d0:8000"))
        assert proxy.route(num_items=0).prefill is None

    def test_encoder_roster_tracks_new_instances(self, proxy):
        proxy.registry.register(InstanceRecord(PD, "http://d0:8000"))
        for index in range(2):
            proxy.registry.register(InstanceRecord(ENCODE, f"http://e{index}:8000"))
        assert proxy.route(num_items=3).encoder_urls == [
            "http://e0:8000",
            "http://e1:8000",
        ]
        proxy.registry.unregister("http://e0:8000")
        assert proxy.route(num_items=3).encoder_urls == ["http://e1:8000"]

    @pytest.mark.parametrize("unregister", [False, True])
    def test_standalone_decode_always_requires_healthy_prefill(self, unregister):
        app = build_app()
        proxy = app.state.proxy
        proxy.registry.register(InstanceRecord(DECODE, "http://d0:8000"))
        with pytest.raises(HTTPException) as excinfo:
            proxy.route(num_items=0)
        assert excinfo.value.status_code == 503
        proxy.registry.register(InstanceRecord(PREFILL, "http://p0:8000"))
        route = proxy.route(num_items=0)
        assert route.decode.url == "http://d0:8000"
        assert route.consumer.url == "http://p0:8000"
        if unregister:
            proxy.registry.unregister("http://p0:8000")
        else:
            for _ in range(proxy.registry._fail_threshold):
                proxy.registry._on_probe_failure(route.prefill, 0)
        with pytest.raises(HTTPException) as excinfo:
            proxy.route(num_items=0)
        assert excinfo.value.status_code == 503


class TestConsumerAddress:
    """Which stage receives the embedding depends on the topology."""

    def test_shared_storage_connectors_name_no_target(self, proxy):
        """Rotate replicas without requiring a push endpoint."""
        proxy.registry.register(InstanceRecord(PD, "http://d0:8000", dp_size=2))
        route = proxy.route(num_items=0)
        assert route.consumer_zmq is None
        assert route.dp_rank == 0
        assert proxy.route(num_items=0).dp_rank == 1

    def test_decode_is_the_consumer_when_prefill_is_not_split_out(self, proxy):
        proxy.registry.register(
            InstanceRecord(PD, "http://d0:8000", ec_zmq_addrs=["tcp://d0:20001"])
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
    def client(self, monkeypatch):
        monkeypatch.setenv("ADMIN_API_KEY", "test-key")
        config = EPDProxyConfig(probe_interval=0)
        with TestClient(build_app(config)) as client:
            client.headers["X-API-Key"] = "test-key"
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

    def test_http_registration_keeps_consumer_replica_and_address_together(
        self, client
    ):
        payload: dict[str, Any] = {
            "role": "prefill_decode",
            "url": "http://d0:8000/",
            "dp_size": 2,
            "ec_zmq_addrs": ["tcp://d0:20001", "tcp://d0:20003"],
        }
        assert client.post("/instances", json=payload).json() == {"registered": True}
        assert client.post("/instances", json=payload).json() == {"registered": False}
        proxy = client.app.state.proxy
        for rank in [0, 1, 0]:
            route = proxy.route(0)
            assert route.dp_rank == rank
            assert route.consumer_zmq == payload["ec_zmq_addrs"][rank]
        assert client.delete("/instances", params={"url": payload["url"]}).json() == {
            "removed": True
        }
        assert proxy.registry.pick(PD) is None

    def test_registration_rejects_invalid_topology(self, client):
        cases: list[tuple[dict[str, Any], int]] = [
            ({"dp_size": 0}, 422),
            ({"dp_size": 2, "ec_zmq_addrs": ["tcp://d0:20001"]}, 422),
            ({"role": "decode", "ec_zmq_addrs": ["tcp://d0:20001"]}, 400),
            ({"role": "encode", "ec_zmq_addrs": ["tcp://d0:20001"]}, 400),
        ]
        for fields, status in cases:
            response = client.post(
                "/instances",
                json={"url": "http://d0:8000", "role": "prefill_decode", **fields},
            )
            assert response.status_code == status

    def test_registration_requires_admin_key(self, client):
        client.headers.pop("X-API-Key")
        assert (
            client.post(
                "/instances", json={"role": "encode", "url": "http://e0:8000"}
            ).status_code
            == 403
        )
        assert (
            client.delete("/instances", params={"url": "http://e0:8000"}).status_code
            == 403
        )

    @pytest.mark.parametrize("split_role", ["prefill", "decode"])
    @pytest.mark.parametrize("combined_first", [False, True])
    def test_registration_rejects_mixed_topologies(
        self, client, split_role, combined_first
    ):
        roles = ["prefill_decode", split_role]
        if not combined_first:
            roles.reverse()
        first = {"role": roles[0], "url": "http://first:8000"}
        second = {"role": roles[1], "url": "http://second:8000"}
        assert client.post("/instances", json=first).status_code == 200
        assert client.post("/instances", json=second).status_code == 409
        # Health eviction must not silently permit a topology switch.
        registry = client.app.state.registry
        record = registry._live[first["url"]]
        for _ in range(registry._fail_threshold):
            registry._on_probe_failure(record, 0)
        assert client.post("/instances", json=second).status_code == 409
        assert client.delete("/instances", params={"url": first["url"]}).json() == {
            "removed": True
        }
        assert client.post("/instances", json=second).status_code == 200


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
            PD,
            "http://decode",
            ec_zmq_addrs=["tcp://decode:14579"] if push else [],
        )
    )
    handle = {
        "metadata": {"image_grid_thw": [[1, 2, 3]]},
        "item_indices": [0, 1],
        "peer_port": 5601,
    }
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
    assert session.post.call_count == 1
    sent = msgspec.json.decode(session.post.call_args.kwargs["data"]).get(
        "ec_transfer_params"
    )
    if push:
        assert sent == {
            "consumer_zmq": "tcp://decode:14579",
            "ec_items": [{"transfer_id": item["transfer_id"]} for item in items],
        }
    else:
        assert sent is None
    for index in range(2):
        assert prepared["messages"][0]["content"][index]["image_embeds"] == {
            "image_grid_thw": [1, 2, 3]
        }
