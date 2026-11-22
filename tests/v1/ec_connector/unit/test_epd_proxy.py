# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing and registration behaviour of the EPD proxy."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest
from aiohttp import web
from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

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


@pytest.mark.parametrize("modality", ["image", "video"])
def test_generate_splice_preserves_source_order_and_duplicate_hash(modality):
    continuations = [
        ([1], [21, 22, 23], ["blue"], ["m1"], [3]),
        ([0, 2], [11, 12, 11, 12], ["red", "red"], ["m0", "m2"], [2, 2]),
    ]
    paired = modality == "video"
    if paired:
        continuations[1:] = [
            ([0], [11, 12], ["red"], ["m0"], [2]),
            ([2], [11, 12], ["red"], ["m2"], [2]),
        ]
    fragments = []
    for indices, ids, hashes, metadata, lengths in continuations:
        if paired:
            ids = [90, *ids, 91]  # Structural tokens are part of the replacement.
        spans = [
            {
                "offset": sum(lengths[:i]) + int(paired),
                "length": n,
                "is_embed": [True] * (n - 1) + [not paired],
            }
            for i, n in enumerate(lengths)
        ]
        placeholders = {modality: spans}
        feature_hashes = {modality: hashes}
        feature_metadata = {modality: metadata}
        if paired:
            placeholders["audio"] = [
                {**span, "is_embed": [False] * (n - 1) + [True]}
                for span, n in zip(spans, lengths, strict=True)
            ]
            feature_hashes["audio"] = [f"a-{h}" for h in hashes]
            feature_metadata["audio"] = [f"a-{m}" for m in metadata]
            if indices == [1]:
                # Equal-offset features have the opposite local transfer order.
                placeholders = {"audio": placeholders["audio"], modality: spans}
        fragments.append(
            (
                indices,
                f"e{indices[0]}",
                {
                    "prompt_token_ids": ids,
                    "features": {
                        "mm_hashes": feature_hashes,
                        "mm_placeholders": placeholders,
                        "mm_metadata": feature_metadata,
                    },
                },
            )
        )

    result = proxy_module._splice_media_fragments(
        [1, 9, 2, 9, 3, 9, 4], [1, 3, 5], fragments
    )
    assert result is not None
    ids, features, transfers = result
    assert features["mm_hashes"][modality] == ["red", "blue", "red"]
    assert features["mm_metadata"][modality] == ["m0", "m1", "m2"]
    if paired:
        assert ids == [1, 90, 11, 12, 91, 2, 90, 21, 22, 23, 91, 3, 90, 11, 12, 91, 4]
        assert features["mm_placeholders"]["video"] == [
            {"offset": 2, "length": 2, "is_embed": [True, False]},
            {"offset": 7, "length": 3, "is_embed": [True, True, False]},
            {"offset": 13, "length": 2, "is_embed": [True, False]},
        ]
        assert features["mm_hashes"]["audio"] == ["a-red", "a-blue", "a-red"]
        assert [p["offset"] for p in features["mm_placeholders"]["audio"]] == [2, 7, 13]
        assert transfers == ["e0:0", "e0:1", "e1:1", "e1:0", "e2:0", "e2:1"]
    else:
        assert ids == [1, 11, 12, 2, 21, 22, 23, 3, 11, 12, 4]
        assert features["mm_placeholders"]["image"] == [
            {"offset": 1, "length": 2, "is_embed": [True, True]},
            {"offset": 4, "length": 3, "is_embed": [True, True, True]},
            {"offset": 8, "length": 2, "is_embed": [True, True]},
        ]
        assert transfers == ["e0:0", "e1:0", "e0:1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("modality", "fragment_status"),
    [("image", status) for status in [200, 501, 503, 500, "raw", "batch-gap"]]
    + [("audio", 200), ("video", 200)],
)
async def test_generate_proxy_hands_d_only_kv(
    monkeypatch, unused_tcp_port, modality, fragment_status
):
    monkeypatch.setenv("ADMIN_API_KEY", "test-key")
    monkeypatch.setattr(proxy_module, "NO_REWRITE", fragment_status == "raw")
    keys = ["a", "b", "c"] if fragment_status == "batch-gap" else ["a", "b"]
    prompt_ids = [1, 9, 2, 9, 3, 9, 4] if len(keys) == 3 else [1, 9, 2, 9, 3]
    calls = []

    async def worker(request):
        stage = request.match_info["stage"]
        body = await request.json()
        calls.append((stage, body))
        if stage.startswith("e"):
            fragment = body.get("encode_media_fragment")
            if fragment and fragment_status in (501, 503):
                return web.json_response(
                    {"error": {"type": "MediaFragmentUnsupported"}},
                    status=fragment_status,
                )
            token_ids = body["token_ids"]
            if fragment and fragment_status == "batch-gap" and len(token_ids) == 2:
                token_ids = [9, 99, 9]  # Token 99 cannot be assigned to one image.
            offsets = [i for i, token in enumerate(token_ids) if token == 9]
            hashes = [part["uuid"] for part in body["content_parts"]]
            return web.json_response(
                {
                    "media_fragment": fragment is not None,
                    "prompt_token_ids": token_ids,
                    "features": {
                        "mm_hashes": {modality: hashes},
                        "mm_placeholders": {
                            modality: [{"offset": i, "length": 1} for i in offsets],
                        },
                        "mm_metadata": {modality: ["metadata"] * len(offsets)},
                    },
                    "ec_transfer_params": {
                        key: {"handle": request.headers["x-request-id"]}
                        for key in hashes
                    },
                }
            )
        if stage == "p":
            return web.json_response(
                {
                    "kv_transfer_params": {
                        "do_remote_prefill": True,
                        "handoff": sum(s == "p" for s, _ in calls),
                    }
                }
            )
        if fragment_status == 500:
            if sum(s == "d" for s, _ in calls) == 1:
                return web.Response(status=500)
            return web.Response(
                text="data: [DONE]\n\n", content_type="text/event-stream"
            )
        return web.json_response({"choices": [{"token_ids": [42]}]})

    upstream = web.Application()
    upstream.router.add_post("/{stage}/inference/v1/generate", worker)
    runner = web.AppRunner(upstream)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
        app = build_app(
            EPDProxyConfig(probe_interval=0, media_fanout_token_ids={modality: 9})
        )
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://proxy") as c,
        ):
            for stage, role in [
                ("e0", "encode"),
                ("e1", "encode"),
                ("p", "prefill"),
                ("d", "decode"),
            ]:
                response = await c.post(
                    "/instances",
                    json={
                        "role": role,
                        "url": f"http://127.0.0.1:{unused_tcp_port}/{stage}",
                    },
                    headers={"X-API-Key": "test-key"},
                )
                assert response.is_success, response.text
            response = await c.post(
                proxy_module.GENERATE_PATH,
                json={
                    "token_ids": prompt_ids,
                    "content_parts": [
                        {
                            "type": f"{modality}_url",
                            "uuid": key,
                            "url": f"http://media/{key}",
                        }
                        for key in keys
                    ],
                    "sampling_params": {"max_tokens": 20},
                    "mm_processor_kwargs": {"do_resize": True},
                    "media_io_kwargs": {"image": {"example": True}},
                    "stream": fragment_status == 500,
                },
            )
    finally:
        await runner.cleanup()

    stages = [stage for stage, _ in calls]
    assert set(stages[:2]) == {"e0", "e1"}
    assert all(b["encode_media_fragment"] == modality for _, b in calls[:2])
    assert sorted(len(b["content_parts"]) for _, b in calls[:2]) == [1, len(keys) - 1]
    if fragment_status == 503:
        assert response.status_code == 502
        assert len(stages) == 2
        return

    for stage, body in calls:
        if stage.startswith("e"):
            assert body["mm_processor_kwargs"] == {"do_resize": True}
            assert body["media_io_kwargs"] == {"image": {"example": True}}
        elif stage != "p" or fragment_status != "raw":
            assert "mm_processor_kwargs" not in body
            assert "media_io_kwargs" not in body

    if fragment_status == 500:
        assert response.text == "data: [DONE]\n\n"
        assert set(stages[4:6]) == {"e0", "e1"}
        assert stages[2:4] == stages[6:8] == ["p", "d"]
        assert calls[2][1]["ec_transfer_params"] != calls[6][1]["ec_transfer_params"]
        assert calls[3][1]["kv_transfer_params"]["handoff"] == 1
    else:
        assert response.json()["choices"] == [{"token_ids": [42]}]
        assert len(stages) == (5 if fragment_status in (501, "batch-gap") else 4)
    assert stages[-2:] == ["p", "d"]
    prefill, decode = calls[-2][1], calls[-1][1]
    if fragment_status == "raw":
        assert "features" not in prefill
        assert [p["uuid"] for p in prefill["content_parts"]] == keys
        assert "content_parts" not in decode
    else:
        assert prefill["features"] == decode["features"]
    assert decode["features"]["mm_hashes"][modality] == keys
    assert decode["token_ids"] == prompt_ids
    assert "ec_transfer_params" in prefill and "ec_transfer_params" not in decode
    assert decode["kv_transfer_params"] == {
        "do_remote_prefill": True,
        "handoff": 2 if fragment_status == 500 else 1,
    }
    assert decode["sampling_params"]["max_tokens"] == 20


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
