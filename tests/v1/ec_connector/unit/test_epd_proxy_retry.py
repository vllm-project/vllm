# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Body handling across the EPD proxy's decode retries.

Exercises the REAL helpers loaded from the ``examples/`` proxy, so a future
change to them is what these tests catch.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import msgspec
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

PROXY_REL = "examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py"


@pytest.fixture(scope="module")
def proxy():
    path = Path(__file__).parents[4] / PROXY_REL
    spec = importlib.util.spec_from_file_location("disagg_epd_proxy_retry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, params):
        self._params = params

    async def read(self):
        return msgspec.json.encode({"kv_transfer_params": self._params})


def test_maybe_prefill_leaves_the_caller_body_untouched(proxy, monkeypatch):
    """A decode retry re-enters this function with the body it was given.

    Mutating that body in place let one attempt's `remote_block_ids` survive
    into the next, so a retry whose prefill returns nothing sent decode blocks
    the prefiller may already have freed.
    """
    served = [{"remote_block_ids": [1, 2]}, {}]

    async def _stage(req_data, p_url, req_id, dp_rank=None):
        assert "kv_transfer_params" not in req_data
        return _Response(served.pop(0))

    monkeypatch.setattr(proxy, "process_prefill_stage", _stage)

    body = {"messages": [], "stream": False}
    first = asyncio.run(proxy.maybe_prefill(body, "http://prefill", "r1"))
    assert first["kv_transfer_params"] == {"remote_block_ids": [1, 2]}
    assert "kv_transfer_params" not in body

    second = asyncio.run(proxy.maybe_prefill(body, "http://prefill", "r1"))
    assert "kv_transfer_params" not in second


class _EncoderResponse:
    def __init__(self, params):
        self.status = 200
        self._params = params

    async def read(self):
        return msgspec.json.encode({"ec_transfer_params": self._params})

    async def text(self):
        return ""


class _EncoderSession:
    """Serve one canned encoder reply per attempt."""

    def __init__(self, replies):
        self._replies = list(replies)

    async def post(self, url, data=None, headers=None):
        return _EncoderResponse(self._replies.pop(0))


@pytest.mark.parametrize("no_rewrite", [False, True])
@pytest.mark.parametrize(
    "batch_size, expected_sizes", [(0, [3, 3]), (1, [1] * 6), (2, [2, 2, 1, 1])]
)
def test_image_batches_preserve_item_identity(
    proxy, monkeypatch, no_rewrite, batch_size, expected_sizes
):
    """Rehashed and repeated images retain metadata and per-occurrence transfers."""
    from vllm.distributed.ec_transfer.ec_connector.mooncake.scheduler import (
        ECMooncakeScheduler,
    )

    seen = []

    class Session:
        async def post(self, url, data=None, headers=None):
            body = msgspec.json.decode(data)
            seen.append((url, body))
            params: dict[str, dict[str, Any]] = {}
            for index, item in enumerate(body["messages"][0]["content"]):
                image = item["image_url"]["url"]
                entry = params.setdefault(
                    image + "-processed",
                    {
                        "metadata": {"image_grid_thw": [1, 2, ord(image)]},
                        "item_indices": [],
                        "peer_port": 1234,
                    },
                )
                entry["item_indices"].append(index)
            # Response ordering must not affect the image-to-metadata mapping.
            return _EncoderResponse(dict(reversed(list(params.items()))))

    monkeypatch.setattr(proxy, "encode_session", Session())
    monkeypatch.setattr(proxy, "NO_REWRITE", no_rewrite)
    monkeypatch.setattr(proxy, "ENCODER_MAX_BATCH_SIZE", batch_size)
    monkeypatch.setattr(proxy, "encoder_rr_idx", 0)
    images = ["A", "A", "C", "D", "A", "F"]
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}}
                    for image in images
                ],
            }
        ],
        "mm_processor_kwargs": {"max_pixels": 262144},
    }
    meta, handles = asyncio.run(
        proxy.fanout_encoder_primer(
            body, ["http://e0", "http://e1"], "r", "tcp://consumer:1"
        )
    )
    assert [len(b["messages"][0]["content"]) for _, b in seen] == expected_sizes
    assert (
        len(
            {
                x["transfer_id"]
                for _, b in seen
                for x in b["ec_transfer_params"]["ec_items"]
            }
        )
        == 6
    )
    for url, batch in seen:
        assert batch["mm_processor_kwargs"] == body["mm_processor_kwargs"]
        for item in batch["messages"][0]["content"]:
            assert item["image_url"]["url"] in (
                {"A", "C"} if url.startswith("http://e0/") else {"A", "D", "F"}
            )
    assert [meta[i]["image_grid_thw"][-1] for i in range(6)] == list(map(ord, images))
    assert [meta[i]["ec_mm_hash"] for i in range(6)] == [
        x + "-processed" for x in images
    ]
    assert handles["A-processed"]["peer_port"] == 1234
    assert "ec_transfer_params" not in body

    transfers_by_encoder: list[list[dict[str, str]]] = [[], []]
    for url, batch in seen:
        rank = 0 if url.startswith("http://e0/") else 1
        transfers_by_encoder[rank].extend(batch["ec_transfer_params"]["ec_items"])
    expected_transfers = [
        transfers_by_encoder[i % 2][i // 2]["transfer_id"] for i in range(len(images))
    ]
    params = (
        handles
        if no_rewrite
        else proxy.rewrite_for_decode(body, meta)["ec_transfer_params"]
    )
    request = SimpleNamespace(
        ec_transfer_params=params,
        mm_features=[
            SimpleNamespace(identifier=image + "-processed") for image in images
        ],
    )
    assert [
        ECMooncakeScheduler._request_transfer_id(request, i) for i in range(len(images))
    ] == expected_transfers


def test_batch_rejects_ambiguous_metadata(proxy, monkeypatch):
    monkeypatch.setattr(
        proxy,
        "encode_session",
        _EncoderSession([{"unknown": {"metadata": {"image_grid_thw": [1, 2, 2]}}}]),
    )
    body = {
        "messages": [
            {
                "content": [
                    {"type": "image_url", "image_url": {"url": image}}
                    for image in ["A", "B"]
                ]
            }
        ]
    }
    with pytest.raises(proxy.HTTPException, match="cannot be matched"):
        asyncio.run(proxy.fanout_encoder_primer(body, ["http://e0"], "r"))


def test_a_decode_retry_does_not_inherit_the_previous_handles(proxy, monkeypatch):
    """The retry loop re-enters `prepare_for_decode` with the same body.

    Recording the encoder's connector handles on that body in place let
    attempt 1's handle survive into attempt 2, so a second encode that
    reported nothing still sent decode a handle on an embedding the encoder
    no longer publishes -- the exact state the retry exists to leave behind.
    """
    handle = {"metadata": {"image_grid_thw": [1, 2, 2]}, "peer_port": 1234}
    monkeypatch.setattr(
        proxy,
        "encode_session",
        _EncoderSession([{"encoder-side-hash": handle}, {}]),
    )

    async def _no_prefill(req_data, p_url, req_id, dp_rank=None):
        return req_data

    monkeypatch.setattr(proxy, "maybe_prefill", _no_prefill)

    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "image"}}],
            }
        ],
        "stream": False,
    }
    args = ("r1", ["http://encoder"], "http://prefill", None)

    first, _, _ = asyncio.run(proxy.prepare_for_decode(body, *args))
    reported = first["ec_transfer_params"]
    assert {key: value for key, value in reported.items() if key != "ec_items"} == {
        "encoder-side-hash": handle
    }
    assert "ec_transfer_params" not in body

    # Attempt 2's encode reports nothing: decode must be told nothing.
    second, _, _ = asyncio.run(proxy.prepare_for_decode(body, *args))
    assert "ec_transfer_params" not in second


def test_raw_media_keeps_encoder_transfer_identity(proxy, monkeypatch):
    handle = {"metadata": {"image_grid_thw": [1, 2, 2]}}
    monkeypatch.setattr(proxy, "NO_REWRITE", True)
    monkeypatch.setattr(
        proxy, "encode_session", _EncoderSession([{"encoded-hash": handle}])
    )
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "image"}}],
            }
        ]
    }
    prepared, _, _ = asyncio.run(
        proxy.prepare_for_decode(
            body, "request", ["http://encoder"], "", "tcp://consumer:1"
        )
    )
    assert prepared["messages"] == body["messages"]
    assert "ec_transfer_params" not in body
    params = prepared["ec_transfer_params"]
    assert params["encoded-hash"] == handle
    assert params["ec_items"][0]["mm_hash"] == "encoded-hash"
    assert params["ec_items"][0]["transfer_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "request_options",
    [
        {},
        {
            "mm_processor_kwargs": {"max_pixels": 262144},
            "media_io_kwargs": {"image": {"image_mode": "RGB"}},
            "priority": -3,
            "session_id": "session-123",
        },
    ],
)
async def test_http_roundtrip_preserves_payload_and_response_bytes(
    proxy, monkeypatch, stream, prefill, dynamic, request_options
):
    """Preserve media and scheduling context across HTTP hops and retries."""
    seen: dict[str, list[dict]] = {"encode": [], "prefill": [], "decode": []}
    prefix = b'data: {"content":"'
    # Split a multibyte character at the old 1024-byte forwarding boundary.
    expected = (
        prefix + b"x" * (1023 - len(prefix)) + '图片🌍"}\n\ndata: [DONE]\n\n'.encode()
        if stream
        else '{ "choices": [{"message": {"content": "图片🌍"}}] }\n'.encode()
    )

    async def backend(request):
        stage = request.match_info["stage"]
        if dynamic:
            consumer = "prefill" if prefill else "decode"
            expected_rank = "0" if stage == consumer else None
            assert request.headers.get("X-data-parallel-rank") == expected_rank
        assert request.content_type == "application/json"
        body = await request.json()
        seen[stage].append(body)
        if stage == "encode":
            params: dict[str, dict[str, Any]] = {}
            for index, item in enumerate(body["messages"][0]["content"]):
                mm_hash = item["uuid"]
                if body.get("mm_processor_kwargs"):
                    mm_hash += "-processed"
                entry = params.setdefault(
                    mm_hash,
                    {
                        "metadata": {"image_grid_thw": [[1, 2, 2]]},
                        "peer_port": len(seen[stage]),
                        "item_indices": [],
                    },
                )
                entry["item_indices"].append(index)
            return web.json_response({"ec_transfer_params": params})
        if stage == "prefill":
            assert body["max_tokens"] == 1 and body["stream"] is False
            return web.json_response(
                {"kv_transfer_params": {"remote_block_ids": [len(seen[stage])]}}
            )
        if len(seen[stage]) == 1:
            return web.Response(status=500, text="retry")
        return web.Response(
            body=expected,
            content_type="text/event-stream" if stream else "application/json",
        )

    backend_app = web.Application()
    backend_app.router.add_post("/{stage}/v1/chat/completions", backend)
    async with TestServer(backend_app) as server:
        for stage, field in [
            ("encode", "e_urls"),
            ("prefill", "p_urls"),
            ("decode", "d_urls"),
        ]:
            urls = [str(server.make_url(f"/{stage}"))]
            if stage == "prefill" and not prefill:
                urls = []
            monkeypatch.setattr(proxy.app.state, field, urls, raising=False)
        monkeypatch.setattr(proxy.app.state, "d_ec_urls", [], raising=False)
        monkeypatch.setattr(proxy.app.state, "ec_consumer_dp_size", 1, raising=False)
        monkeypatch.setattr(proxy, "DECODE_RETRIES", 1)
        monkeypatch.setattr(proxy, "NO_REWRITE", False)
        target_app = proxy.app
        if dynamic:
            monkeypatch.setenv("ADMIN_API_KEY", "test-key")
            target_app = proxy.build_app(proxy.EPDProxyConfig(probe_interval=0))
        item = {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}}
        body = {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "描述图片🌍"},
                        item,
                        item,
                    ],
                }
            ],
            "stream": stream,
            "temperature": 0.01,
            "max_tokens": 32,
            "seed": 42,
            "structured_outputs": {"choice": ["A", "B"]},
            **request_options,
        }
        await proxy.on_startup()
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=target_app), base_url="http://proxy"
            ) as client:
                if dynamic:
                    registrations = [("encode", "encode")]
                    registrations += (
                        [("prefill", "prefill"), ("decode", "decode")]
                        if prefill
                        else [("prefill_decode", "decode")]
                    )
                    for role, stage in registrations:
                        consumer = "prefill" if prefill else "decode"
                        registered = await client.post(
                            "/instances",
                            headers={"X-API-Key": "test-key"},
                            json={
                                "role": role,
                                "url": str(server.make_url(f"/{stage}")),
                                "dp_size": 2 if stage == consumer else 1,
                            },
                        )
                        assert registered.status_code == 200
                response = await client.post("/v1/chat/completions", json=body)
                if dynamic:
                    removed = await client.delete(
                        "/instances",
                        headers={"X-API-Key": "test-key"},
                        params={"url": str(server.make_url("/decode"))},
                    )
                    assert removed.json() == {"removed": True}
                    unavailable = await client.post("/v1/chat/completions", json=body)
                    assert unavailable.status_code == 503
            assert response.status_code == 200
            assert response.content == expected
            assert response.headers["content-type"].startswith(
                "text/event-stream" if stream else "application/json"
            )
        finally:
            await proxy.on_shutdown()

    assert len(seen["encode"]) == 2
    assert len(seen["decode"]) == 2
    for requests in seen.values():
        for forwarded in requests:
            for key in (
                "mm_processor_kwargs",
                "media_io_kwargs",
                "priority",
                "session_id",
            ):
                assert forwarded.get(key) == body.get(key)
    final = seen["decode"][-1]
    for key in [
        "model",
        "stream",
        "temperature",
        "max_tokens",
        "seed",
        "structured_outputs",
    ]:
        assert final[key] == body[key]
    content = final["messages"][0]["content"]
    assert content[0] == body["messages"][0]["content"][0]
    for reference in content[1:]:
        assert reference == {
            "type": "image_embeds",
            "image_embeds": {"image_grid_thw": [1, 2, 2]},
            "uuid": proxy.content_uuid(item),
        }
    mm_hash = proxy.content_uuid(item)
    if request_options:
        mm_hash += "-processed"
    assert set(final["ec_transfer_params"]) - {"ec_items"} == {mm_hash}
    assert final["ec_transfer_params"][mm_hash]["peer_port"] == 2
    if prefill:
        assert final["kv_transfer_params"] == {"remote_block_ids": [2]}


@pytest.mark.parametrize("server_keep_alive", ["0.1", "1", "5", "2", "30"])
def test_pooled_connections_are_retired_before_the_server_closes_them(
    proxy, monkeypatch, server_keep_alive
):
    """The proxy must not hand a request a connection the server has dropped.

    vLLM closes idle keep-alive connections at `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`
    seconds while aiohttp pools them for 15, so a slow hop leaves a dead
    connection in the pool. The next request fails with
    ServerDisconnectedError and the server logs nothing, because it closed the
    socket before the request arrived.
    """
    monkeypatch.setenv("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", server_keep_alive)
    monkeypatch.setattr(proxy.app.state, "p_urls", [], raising=False)

    asyncio.run(proxy.on_startup())
    try:
        # No public accessor for the pool's idle timeout.
        pooled_for = proxy.encode_session.connector._keepalive_timeout
        assert pooled_for < float(server_keep_alive)
        assert proxy.decode_session.connector._keepalive_timeout == pooled_for
    finally:
        asyncio.run(proxy.on_shutdown())


@pytest.mark.parametrize(
    "no_rewrite, transfer",
    [(False, None), (True, None), (False, "push"), (False, "handle")],
)
def test_video_audio_fallback(proxy, monkeypatch, no_rewrite, transfer):
    """Keep raw video and neighboring images, but never discard transfer handles."""
    import copy

    from vllm.distributed.ec_transfer.ec_connector.utils import collect_ec_item_metadata

    video = {"type": "video_url", "video_url": {"url": "video-with-audio"}}
    images = [{"type": "image_url", "image_url": {"url": key}} for key in ("A", "B")]
    body: dict[str, Any] = {
        "messages": [{"content": [images[0], video, images[1]]}],
        "mm_processor_kwargs": {"use_audio_in_video": True},
    }
    original = copy.deepcopy(body)
    # Collector indices describe two processed features from one video item.
    video_metadata = collect_ec_item_metadata(
        [SimpleNamespace(identifier=key, data=None) for key in ("video", "audio")], None
    )
    if transfer == "handle":
        video_metadata["audio"]["transfer_id"] = "reservation"
    image_metadata = {
        key: {"metadata": {"image_grid_thw": [1, 2, size]}, "item_indices": [i]}
        for i, (key, size) in enumerate((("A", 2), ("B", 4)))
    }
    # Images form the first group, the singleton video the second.
    monkeypatch.setattr(proxy, "ENCODER_MAX_BATCH_SIZE", 0)
    monkeypatch.setattr(proxy, "NO_REWRITE", no_rewrite)
    replies = _EncoderSession([image_metadata, video_metadata])
    monkeypatch.setattr(proxy, "encode_session", replies)
    consumer = "tcp://consumer:1" if transfer == "push" else None
    preparation = proxy.prepare_for_decode(body, "r", ["http://e0"], "", consumer)
    if transfer:
        with pytest.raises(proxy.HTTPException, match="cannot be matched"):
            asyncio.run(preparation)
        return

    prepared, _, _ = asyncio.run(preparation)
    assert body == original
    assert prepared["mm_processor_kwargs"] == body["mm_processor_kwargs"]
    final = prepared["messages"][0]["content"]
    assert final[1] == video
    handles = prepared["ec_transfer_params"]
    assert not video_metadata.keys() & handles.keys()
    if no_rewrite:
        assert final == original["messages"][0]["content"]
        assert "ec_items" not in handles
    else:
        for position, image, key in ((0, images[0], "A"), (2, images[1], "B")):
            assert final[position] == {
                "type": "image_embeds",
                "image_embeds": image_metadata[key]["metadata"],
                "uuid": proxy.content_uuid(image),
            }
        assert [item["mm_hash"] for item in handles["ec_items"]] == ["A", "B"]
