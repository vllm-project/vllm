# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing and registration behaviour of the EPD proxy."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from vllm.distributed.ec_transfer.proxy.epd_proxy import (
    EPDProxy,
    EPDProxyConfig,
    build_app,
    extract_mm_items,
)
from vllm.distributed.ec_transfer.proxy.registry import (
    InstanceRecord,
    InstanceRegistry,
    InstanceRole,
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

    def test_one_encoder_is_assigned_per_item(self, proxy):
        proxy.registry.register(InstanceRecord(DECODE, "http://d0:8000"))
        for index in range(2):
            proxy.registry.register(InstanceRecord(ENCODE, f"http://e{index}:8000"))
        route = proxy.route(num_items=3)
        assert [record.url for record in route.encoders] == [
            "http://e0:8000",
            "http://e1:8000",
            "http://e0:8000",
        ]


class TestConsumerAddress:
    """Which stage receives the embedding depends on the topology."""

    def test_shared_storage_connectors_name_no_target(self, proxy):
        """Nothing registered a receive address, so the encoder is told none."""
        proxy.registry.register(InstanceRecord(DECODE, "http://d0:8000"))
        assert proxy.route(num_items=0).consumer_zmq is None

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


class TestRegistrationApi:
    @pytest.fixture
    def client(self):
        with TestClient(build_app(EPDProxyConfig(probe_interval=0))) as client:
            yield client

    def test_proxy_serves_before_anything_registers(self, client):
        assert client.get("/health").status_code == 200
        assert client.get("/instances").json()["encode"]["live"] == []

    def test_an_instance_joins_and_leaves(self, client):
        body = {"role": "encode", "url": "http://e0:8000"}
        assert client.post("/instances", json=body).status_code == 200
        assert client.get("/instances").json()["encode"]["live"] == ["http://e0:8000"]
        assert client.request("DELETE", "/instances", json=body).json()["found"]
        assert client.get("/instances").json()["encode"]["live"] == []

    def test_a_consumer_reports_its_receive_addresses(self, client):
        client.post(
            "/instances",
            json={
                "role": "decode",
                "url": "http://d0:8000/",
                "ec_zmq_addrs": ["tcp://d0:20001", "tcp://d0:20002"],
                "dp_size": 2,
            },
        )
        # The trailing slash would otherwise produce "http://d0:8000//v1/...".
        assert client.get("/instances").json()["decode"]["live"] == ["http://d0:8000"]

    def test_an_unknown_role_is_rejected(self, client):
        response = client.post(
            "/instances", json={"role": "embed", "url": "http://x:8000"}
        )
        assert response.status_code == 422

    def test_requests_are_refused_until_a_decoder_registers(self, client):
        response = client.post(
            "/v1/chat/completions", json={"model": "m", "messages": []}
        )
        assert response.status_code == 503
        assert client.get("/v1/models").status_code == 503


def test_extract_mm_items_finds_media_across_messages():
    req = {
        "messages": [
            {"role": "user", "content": "plain text"},
            {"role": "user", "content": [IMAGE_ITEM, {"type": "text", "text": "hi"}]},
            {"role": "user", "content": [IMAGE_ITEM]},
        ]
    }
    assert extract_mm_items(req) == [IMAGE_ITEM, IMAGE_ITEM]
