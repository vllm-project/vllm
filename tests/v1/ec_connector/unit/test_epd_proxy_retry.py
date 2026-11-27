# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Body handling across the EPD proxy's decode retries.

Exercises the REAL helpers loaded from the `examples/` proxy, so a future
change to them is what these tests catch.
"""

import asyncio
import copy
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

from vllm.multimodal.parse import MultiModalDataItems, VideoProcessorItems
from vllm.multimodal.processing.inputs import ProcessorInputs

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


@pytest.mark.parametrize("second_type", ["image_url", "audio_url", "input_audio"])
@pytest.mark.parametrize("transfer", ["example", "nixl", "mooncake"])
def test_partial_audio_metadata_falls_back_with_all_transfers(
    proxy, monkeypatch, second_type, transfer
):
    """Never mix metadata-only audio with raw media, or lose EC identities."""
    import copy

    seen = []
    reported = {}

    class Session:
        async def post(self, url, data=None, headers=None):
            body = msgspec.json.decode(data)
            seen.append(body)
            item = body["messages"][0]["content"][0]
            mm_hash = item["uuid"] + "-processed"
            entry: dict[str, Any] = {
                "metadata": {"audio_feature_lengths": [100]} if len(seen) == 1 else {},
                "item_indices": [0],
            }
            if transfer == "nixl":
                entry.update(peer_host="encoder", peer_port=1234, size_bytes=4096)
            reported[mm_hash] = entry
            return _EncoderResponse({mm_hash: entry})

    second = (
        {"type": second_type, "input_audio": {"data": "large", "format": "wav"}}
        if second_type == "input_audio"
        else {"type": second_type, second_type: {"url": "large"}}
    )
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "small"},
                        "uuid": "user-audio",
                    },
                    second,
                ],
            }
        ],
        "mm_processor_kwargs": {"sampling_rate": 16000},
    }
    original = copy.deepcopy(body)
    monkeypatch.setattr(proxy, "encode_session", Session())
    monkeypatch.setattr(proxy, "NO_REWRITE", False)
    consumer = "tcp://consumer:1" if transfer == "mooncake" else None
    prepared, _, _ = asyncio.run(
        proxy.prepare_for_decode(body, "r", ["http://encoder"], "", consumer)
    )
    assert body == original
    assert prepared["mm_processor_kwargs"] == body["mm_processor_kwargs"]
    content = prepared["messages"][0]["content"]
    assert [item["type"] for item in content] == ["audio_url", second_type]
    assert content[0]["uuid"] == "user-audio"
    for item, encoded in zip(content, seen):
        assert item == encoded["messages"][0]["content"][0]
    params = prepared["ec_transfer_params"]
    for mm_hash, entry in reported.items():
        assert params[mm_hash] == entry
    if transfer == "mooncake":
        assert params["ec_items"] == [
            {
                "mm_hash": mm_hash,
                "transfer_id": encoded["ec_transfer_params"]["ec_items"][0][
                    "transfer_id"
                ],
            }
            for mm_hash, encoded in zip(reported, seen)
        ]


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
                # Mooncake aggregates by hash, retaining the last occurrence's ID.
                entry["transfer_id"] = body["ec_transfer_params"]["ec_items"][index][
                    "transfer_id"
                ]
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
    assert [meta[i][0]["metadata"]["image_grid_thw"][-1] for i in range(6)] == list(
        map(ord, images)
    )
    assert [meta[i][0]["mm_hash"] for i in range(6)] == [
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


@pytest.mark.parametrize("consumer", [None, "tcp://consumer:1"])
@pytest.mark.parametrize("has_indices", [False, True])
@pytest.mark.parametrize("has_metadata", [False, True])
def test_audio_in_video_accepts_single_video_feature(
    proxy, monkeypatch, consumer, has_indices, has_metadata
):
    """A video-only encoder can ignore use_audio_in_video without breaking decode."""
    seen = []
    metadata = {"video_grid_thw": [1, 2, 4]} if has_metadata else {}
    entry: dict[str, Any] = {"modality": "video", "metadata": metadata}
    if has_indices:
        entry["item_indices"] = [0]

    class Session:
        async def post(self, url, data=None, headers=None):
            seen.append(msgspec.json.decode(data))
            return _EncoderResponse({"video-hash": entry})

    monkeypatch.setattr(proxy, "encode_session", Session())
    monkeypatch.setattr(proxy, "NO_REWRITE", False)
    video = {"type": "video_url", "video_url": {"url": "video-only"}}
    body = {
        "messages": [{"role": "user", "content": [video]}],
        "mm_processor_kwargs": {"use_audio_in_video": True},
    }
    prepared, _, _ = asyncio.run(
        proxy.prepare_for_decode(body, "r", ["http://encoder"], "", consumer)
    )
    content = prepared["messages"][0]["content"]
    expected = (
        {"type": "video_embeds", "video_embeds": metadata} if has_metadata else video
    )
    assert content == [{**expected, "uuid": proxy.content_uuid(video)}]
    if consumer is not None:
        assert prepared["ec_transfer_params"]["ec_items"] == [
            {
                "mm_hash": "video-hash",
                "transfer_id": seen[0]["ec_transfer_params"]["ec_items"][0][
                    "transfer_id"
                ],
            }
        ]


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
    [
        (False, None),
        (True, None),
        (False, "push"),
        (True, "push"),
        (False, "handle"),
    ],
)
@pytest.mark.parametrize("has_metadata", [False, True])
def test_video_audio_rewrite(proxy, monkeypatch, no_rewrite, transfer, has_metadata):
    """One video publishes two features; forward both even on raw fallback."""
    import copy

    from vllm.distributed.ec_transfer.ec_connector.utils import collect_ec_item_metadata
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

    video = {"type": "video_url", "video_url": {"url": "video-with-audio"}}
    images = [{"type": "image_url", "image_url": {"url": key}} for key in ("A", "B")]
    body: dict[str, Any] = {
        "messages": [{"content": [images[0], video, images[1]]}],
        "mm_processor_kwargs": {"use_audio_in_video": True},
    }
    original = copy.deepcopy(body)
    # Collector indices describe two processed features from one video item.
    video_metadata = collect_ec_item_metadata(
        [
            MultiModalFeatureSpec(
                data=None,
                modality=key,
                identifier=key,
                mm_position=PlaceholderRange(offset=0, length=0),
            )
            for key in ("video", "audio")
        ],
        SimpleNamespace(fields_for=lambda _: set()),
    )
    if has_metadata:
        video_metadata["video"]["metadata"] = {
            "video_grid_thw": [4, 2, 2],
            "second_per_grid_ts": [0.5],
        }
        video_metadata["audio"]["metadata"] = {"audio_feature_lengths": [100]}
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
    prepared, _, _ = asyncio.run(preparation)
    assert body == original
    assert prepared["mm_processor_kwargs"] == body["mm_processor_kwargs"]
    final = prepared["messages"][0]["content"]
    handles = prepared["ec_transfer_params"]
    assert all(handles[key] == value for key, value in video_metadata.items())
    if no_rewrite:
        assert final == original["messages"][0]["content"]
        if consumer is None:
            assert "ec_items" not in handles
        else:
            assert {entry["mm_hash"] for entry in handles["ec_items"]} == {
                "A",
                "B",
                "video",
                "audio",
            }
    elif not has_metadata:
        assert [item["type"] for item in final] == [
            "image_url",
            "video_url",
            "image_url",
        ]
        assert final[1]["uuid"] == proxy.content_uuid(video)
        assert {entry["mm_hash"] for entry in handles["ec_items"]} == {
            "A",
            "B",
            "video",
            "audio",
        }
    else:
        assert [item["type"] for item in final] == [
            "image_embeds",
            "video_embeds",
            "image_embeds",
        ]
        assert final[1] == {
            "type": "video_embeds",
            "video_embeds": video_metadata["video"]["metadata"],
            "audio_embeds": video_metadata["audio"]["metadata"],
            "uuid": proxy.content_uuid(video),
        }
        for position, image, key in ((0, images[0], "A"), (2, images[1], "B")):
            assert final[position] == {
                "type": "image_embeds",
                "image_embeds": image_metadata[key]["metadata"],
                "uuid": proxy.content_uuid(image),
            }
        assert [item["mm_hash"] for item in handles["ec_items"]] == [
            "A",
            "video",
            "audio",
            "B",
        ]


def _vid(url: str) -> dict[str, Any]:
    return {"type": "video_url", "video_url": {"url": url}}


def _img(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


class _FanoutSession:
    """Record each encoder child body; reply with one entry per item."""

    def __init__(self, seen: list):
        self._seen = seen

    async def post(self, url, data=None, headers=None):
        body = msgspec.json.decode(data)
        self._seen.append((url, body))
        params: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(body["messages"][0]["content"]):
            media = item.get("video_url") or item.get("image_url") or {}
            entry = params.setdefault(
                media["url"],
                {"metadata": {"video_grid_thw": [1, 2, 3]}, "item_indices": []},
            )
            entry["item_indices"].append(index)
        return _EncoderResponse(params)


@pytest.mark.parametrize("nested", [True, False], ids=["nested", "flat"])
def test_fanout_passes_original_video_position(proxy, monkeypatch, nested):
    """The encoder receives the full kwargs and its position among the videos."""
    seen: list = []
    monkeypatch.setattr(proxy, "encode_session", _FanoutSession(seen))
    monkeypatch.setattr(proxy, "encoder_rr_idx", 0)
    scoped = {
        "do_sample_frames": True,
        "num_frames": [16, 32],
        "fps": [2.0, 4.0],
        "size": [420, 420],
    }
    kwargs = {"videos_kwargs": dict(scoped)} if nested else dict(scoped)
    body = {
        "messages": [{"role": "user", "content": [_vid("a"), _vid("b")]}],
        "mm_processor_kwargs": kwargs,
        "media_io_kwargs": {"video": {"fps": 12}},
    }
    original = copy.deepcopy(body)
    asyncio.run(
        proxy.fanout_encoder_primer(
            body, ["http://e0"], "r", enable_per_video_kwargs_slicing=True
        )
    )
    assert body == original

    by_url = {b["messages"][0]["content"][0]["video_url"]["url"]: b for _, b in seen}
    assert len(by_url) == 2
    for index, url in enumerate(("a", "b")):
        assert by_url[url]["mm_processor_kwargs"] == {
            **kwargs,
            "_epd_video_partition": [index, 2],
        }
        assert by_url[url]["media_io_kwargs"] == {"video": {"fps": 12}}


def test_fanout_keeps_scalar_and_unrelated_kwargs(proxy, monkeypatch):
    """The proxy does not use the request's kwargs to decide when to index."""
    seen: list = []
    monkeypatch.setattr(proxy, "encode_session", _FanoutSession(seen))
    monkeypatch.setattr(proxy, "encoder_rr_idx", 0)
    kwargs = {"videos_kwargs": {"fps": 2.0, "size": [420, 420]}}
    body = {
        "messages": [{"role": "user", "content": [_vid("a"), _vid("b")]}],
        "mm_processor_kwargs": kwargs,
    }
    asyncio.run(
        proxy.fanout_encoder_primer(
            body, ["http://e0"], "r", enable_per_video_kwargs_slicing=True
        )
    )
    assert len(seen) == 2
    for index, (_, child) in enumerate(seen):
        assert child["mm_processor_kwargs"] == {
            **kwargs,
            "_epd_video_partition": [index, 2],
        }


@pytest.mark.parametrize("nested", [True, False], ids=["nested", "flat"])
def test_fanout_keeps_kwargs_for_image_only_request(proxy, monkeypatch, nested):
    """Without videos there is no per-video partition to perform."""
    seen: list = []
    monkeypatch.setattr(proxy, "encode_session", _FanoutSession(seen))
    monkeypatch.setattr(proxy, "encoder_rr_idx", 0)
    kwargs: dict[str, Any] = {"num_frames": [16, 32]}
    if nested:
        kwargs = {"videos_kwargs": kwargs}
    body = {
        "messages": [{"role": "user", "content": [_img("a"), _img("b")]}],
        "mm_processor_kwargs": kwargs,
    }
    original = copy.deepcopy(body)
    asyncio.run(
        proxy.fanout_encoder_primer(
            body, ["http://e0", "http://e1"], "r", enable_per_video_kwargs_slicing=True
        )
    )
    assert len(seen) == 2
    for _, child in seen:
        assert child["mm_processor_kwargs"] == kwargs
    assert body == original


@pytest.mark.parametrize("nested", [True, False], ids=["nested", "flat"])
def test_fanout_interleaved_media_and_two_encoders(proxy, monkeypatch, nested):
    """Video positions count videos only; image children preserve all kwargs."""
    seen: list = []
    monkeypatch.setattr(proxy, "encode_session", _FanoutSession(seen))
    monkeypatch.setattr(proxy, "encoder_rr_idx", 0)
    kwargs = {"do_sample_frames": True, "num_frames": [16, 32]}
    if nested:
        kwargs = {"videos_kwargs": kwargs}
    body = {
        "messages": [
            {"role": "user", "content": [_img("x"), _vid("a")]},
            {"role": "user", "content": [_vid("b"), _img("y")]},
        ],
        "mm_processor_kwargs": kwargs,
    }
    original = copy.deepcopy(body)
    asyncio.run(
        proxy.fanout_encoder_primer(
            body, ["http://e0", "http://e1"], "r", enable_per_video_kwargs_slicing=True
        )
    )
    assert body == original

    for _, batch in seen:
        videos = [
            i for i in batch["messages"][0]["content"] if i["type"] == "video_url"
        ]
        images = [
            i for i in batch["messages"][0]["content"] if i["type"] == "image_url"
        ]
        if videos:
            # One video per child; position among the request's two videos.
            index = 0 if videos[0]["video_url"]["url"] == "a" else 1
            assert batch["mm_processor_kwargs"] == {
                **kwargs,
                "_epd_video_partition": [index, 2],
            }
            assert not images
        else:
            assert batch["mm_processor_kwargs"] == kwargs
    covered = {
        (i["type"], (i.get("video_url") or i.get("image_url"))["url"])
        for _, b in seen
        for i in b["messages"][0]["content"]
    }
    assert covered == {
        ("image_url", "x"),
        ("video_url", "a"),
        ("video_url", "b"),
        ("image_url", "y"),
    }


def test_fanout_rejects_short_per_video_list(proxy, monkeypatch):
    """An undersized per-video list fails fast instead of slicing garbage."""
    monkeypatch.setattr(proxy, "encode_session", _FanoutSession([]))
    monkeypatch.setattr(proxy, "encoder_rr_idx", 0)
    kwargs = {"videos_kwargs": {"do_sample_frames": True, "num_frames": [16]}}
    body = {
        "messages": [{"role": "user", "content": [_vid("a"), _vid("b")]}],
        "mm_processor_kwargs": kwargs,
    }
    with pytest.raises(proxy.HTTPException) as exc:
        asyncio.run(
            proxy.fanout_encoder_primer(
                body, ["http://e0"], "r", enable_per_video_kwargs_slicing=True
            )
        )
    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    ("flags", "allowed"),
    [
        ({"load_audio_from_video": True, "use_audio_in_video": False}, False),
        ({"use_audio_in_video": True}, False),
        ({"load_audio_from_video": False, "use_audio_in_video": True}, True),
    ],
)
def test_epd_video_audio_validation_preserves_flag_precedence(proxy, flags, allowed):
    body = {
        "messages": [{"role": "user", "content": [_vid("a"), _vid("b")]}],
        "mm_processor_kwargs": {**flags, "fps": [2.0, 4.0]},
    }
    if allowed:
        proxy._validate_proxy_request(body, True)
    else:
        with pytest.raises(proxy.HTTPException, match="Move mm_processor_kwargs.fps"):
            proxy._validate_proxy_request(body, True)

    body["mm_processor_kwargs"] = {**flags, "videos_kwargs": {"fps": [2.0, 4.0]}}
    proxy._validate_proxy_request(body, True)


def test_fanout_positions_include_metadata_only_video_without_sending_it(
    proxy, monkeypatch
):
    seen: list = []
    monkeypatch.setattr(proxy, "encode_session", _FanoutSession(seen))
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_embeds", "video_embeds": None, "uuid": "cached"},
                    _vid("b"),
                ],
            }
        ],
    }
    asyncio.run(
        proxy.fanout_encoder_primer(
            body, ["http://e0"], "r", enable_per_video_kwargs_slicing=True
        )
    )
    [(_, child)] = seen
    assert child["mm_processor_kwargs"] == {"_epd_video_partition": [1, 2]}
    assert len(child["messages"][0]["content"]) == 1


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("short", [False, True])
def test_fanout_default_preserves_base_hash_kwargs(proxy, monkeypatch, nested, short):
    """Without opt-in, even image children keep the original kwargs."""
    seen: list = []
    monkeypatch.setattr(proxy, "NO_REWRITE", False)
    monkeypatch.setattr(proxy, "encode_session", _FanoutSession(seen))
    values = [16] if short else [16, 32]
    kwargs: dict[str, Any] = {"num_frames": values}
    if nested:
        kwargs = {"videos_kwargs": kwargs}
    body = {
        "messages": [{"role": "user", "content": [_vid("a"), _img("x"), _vid("b")]}],
        "mm_processor_kwargs": kwargs,
    }
    original = copy.deepcopy(body)
    asyncio.run(proxy.fanout_encoder_primer(body, ["http://e0"], "r"))
    assert len(seen) == 3
    assert all(child["mm_processor_kwargs"] == kwargs for _, child in seen)
    assert body == original

    def hashes(uuids, options):
        inputs = ProcessorInputs(
            prompt=[],
            mm_data_items=MultiModalDataItems(
                {"video": VideoProcessorItems([None] * len(uuids))}
            ),
            mm_uuid_items={"video": uuids},
            hf_processor_mm_kwargs=options,
        )
        return inputs.get_mm_hashes("base-hash-model", "sha256")["video"]

    videos = [_vid("a"), _vid("b")]
    uuids = [proxy.content_uuid(video) for video in videos]
    parent_hashes = hashes(uuids, kwargs)
    for _, child in seen:
        item = child["messages"][0]["content"][0]
        if item["type"] != "video_url":
            continue
        index = uuids.index(item["uuid"])
        child_hash = hashes([item["uuid"]], child["mm_processor_kwargs"])[0]
        assert child_hash == parent_hashes[index]


@pytest.mark.asyncio
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("enabled", "kwargs", "invalid_at_proxy"),
    [
        (True, {"num_frames": [16]}, True),
        (True, {"videos_kwargs": {"num_frames": [16]}}, True),
        (True, {"fps": [2.0]}, True),
        (True, {"videos_kwargs": {"fps": [2.0]}}, True),
        (True, {"num_frames": [16, 32]}, False),
        (True, {"num_frames": [99], "videos_kwargs": {"num_frames": [16, 32]}}, False),
        (False, {"num_frames": [16]}, False),
        (False, {"videos_kwargs": {"num_frames": [16, 32]}}, False),
        (True, {"_epd_video_partition": [0, 2]}, True),
        (False, {"_epd_video_partition": [0, 2]}, True),
        pytest.param(
            True, {"second_per_grid_ts": [0.5]}, False, id="encoder-rejects-hy4"
        ),
    ],
)
async def test_per_video_kwargs_http_validation(
    proxy, monkeypatch, dynamic, stream, enabled, kwargs, invalid_at_proxy
):
    """Distinguish proxy-side 400s from encoder 400s after SSE starts."""
    seen: dict[str, list[dict]] = {"encode": [], "prefill": [], "decode": []}
    expected = b"data: [DONE]\n\n" if stream else b'{"choices": []}'
    encoder_rejects = "second_per_grid_ts" in kwargs
    encoder_error = "second_per_grid_ts must match video count"
    stream_finished = False

    async def backend(request):
        stage = request.match_info["stage"]
        seen[stage].append(await request.json())
        if stage == "encode" and encoder_rejects:
            return web.json_response({"error": encoder_error}, status=400)
        if stage in ("encode", "prefill"):
            return web.json_response({})
        return web.Response(
            body=expected,
            content_type="text/event-stream" if stream else "application/json",
        )

    backend_app = web.Application()
    backend_app.router.add_post("/{stage}/v1/chat/completions", backend)
    async with TestServer(backend_app) as server:
        for stage, field in (
            ("encode", "e_urls"),
            ("prefill", "p_urls"),
            ("decode", "d_urls"),
        ):
            monkeypatch.setattr(
                proxy.app.state,
                field,
                [str(server.make_url(f"/{stage}"))],
                raising=False,
            )
        monkeypatch.setattr(proxy.app.state, "d_ec_urls", [], raising=False)
        monkeypatch.setattr(proxy.app.state, "ec_consumer_dp_size", 1, raising=False)
        monkeypatch.setitem(
            proxy.app.state._state, "enable_per_video_kwargs_slicing", enabled
        )
        monkeypatch.setattr(proxy, "NO_REWRITE", True)
        target_app = proxy.app
        if dynamic:
            monkeypatch.setenv("ADMIN_API_KEY", "test-key")
            target_app = proxy.build_app(
                proxy.EPDProxyConfig(
                    probe_interval=0, enable_per_video_kwargs_slicing=enabled
                )
            )
        body = {
            "model": "served-model-alias",
            "messages": [{"role": "user", "content": [_vid("a"), _vid("b")]}],
            "mm_processor_kwargs": kwargs,
            "stream": stream,
        }
        await proxy.on_startup()
        try:

            async def observe_response(scope, receive, send):
                async def observe_send(message):
                    nonlocal stream_finished
                    if (
                        scope["path"] == "/v1/chat/completions"
                        and message["type"] == "http.response.body"
                    ):
                        stream_finished |= not message.get("more_body", False)
                    await send(message)

                await target_app(scope, receive, observe_send)

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=observe_response, raise_app_exceptions=False
                ),
                base_url="http://proxy",
            ) as client:
                if dynamic:
                    for stage in ("encode", "prefill", "decode"):
                        registered = await client.post(
                            "/instances",
                            headers={"X-API-Key": "test-key"},
                            json={
                                "role": stage,
                                "url": str(server.make_url(f"/{stage}")),
                            },
                        )
                        assert registered.status_code == 200
                response = await client.post("/v1/chat/completions", json=body)
        finally:
            await proxy.on_shutdown()

    if invalid_at_proxy:
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert (
            "is reserved by EPD"
            if "_epd_video_partition" in kwargs
            else "entries, one per video"
        ) in detail
        assert seen == {"encode": [], "prefill": [], "decode": []}
    elif encoder_rejects:
        assert len(seen["encode"]) == 2
        assert seen["prefill"] == seen["decode"] == []
        if stream:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.content == b""
            assert not stream_finished
        else:
            assert response.status_code == 400
            assert encoder_error in response.json()["detail"]
    else:
        assert response.status_code == 200
        assert response.content == expected
        assert stream_finished
        assert len(seen["encode"]) == 2
        assert len(seen["prefill"]) == len(seen["decode"]) == 1
        for stage in ("prefill", "decode"):
            assert seen[stage][0]["mm_processor_kwargs"] == kwargs
        for child in seen["encode"]:
            url = child["messages"][0]["content"][0]["video_url"]["url"]
            index = 0 if url == "a" else 1
            expected_kwargs = dict(kwargs)
            if enabled:
                expected_kwargs["_epd_video_partition"] = [index, 2]
            assert child["mm_processor_kwargs"] == expected_kwargs
