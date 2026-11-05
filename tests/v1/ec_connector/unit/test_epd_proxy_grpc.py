# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real rendering and gRPC calls through the proxy's public registration API.

Only the external inference workers are simulated: these tests exercise token
expansion, metadata serialization, routing, OpenAI output and cancellation.
"""

import asyncio
import copy
import itertools
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

import grpc
import httpx
import msgspec
import pybase64
import pytest
from google.protobuf.json_format import ParseDict
from PIL import Image

from examples.disaggregated.disaggregated_encoder import (
    disagg_epd_proxy as proxy,
)
from examples.disaggregated.disaggregated_encoder.grpc_backend import (
    GrpcBackend,
    load_protocol,
    transfer_params,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.v1.serial_utils import MsgpackDecoder

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


@asynccontextmanager
async def worker(
    backend,
    role,
    seen,
    *,
    blocked=None,
    started=None,
    aborted=None,
    stream_blocked=None,
):
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc

    pb, inference, control_pb, control = load_protocol()
    render, derender = await backend.frontend(MODEL)
    tokenizer = derender.renderer.get_tokenizer()
    answer = tokenizer.encode("blue", add_special_tokens=False)

    class Inference(inference.InferenceServicer):  # type: ignore[name-defined]
        async def Generate(self, request, context):
            seen.append((role, request, dict(context.invocation_metadata())))
            if started is not None:
                started.set()
            if blocked is not None:
                await blocked.wait()
            finish = pb.FinishInfo(
                finish_reason=pb.FinishInfo.STOP,
                num_output_tokens=len(answer) if role in ("decode", "pd") else 0,
            )
            if role == "encode":
                # The worker, not the proxy, fetches and preprocesses raw images.
                content = []
                for media in request.media:
                    assert media.WhichOneof("source") in ("url", "data_uri")
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": media.url or media.data_uri},
                            "uuid": media.uuid or None,
                        }
                    )
                if content:
                    _, inputs = await render.render_chat(
                        ChatCompletionRequest(
                            model=MODEL,
                            messages=[{"role": "user", "content": content}],
                        ),
                        skip_mm_cache=True,
                    )
                    prompt = inputs[0]
                    prompt_ids = prompt["prompt_token_ids"]
                    image_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")
                    unexpanded = [
                        token
                        for token, group in itertools.groupby(prompt_ids)
                        for _ in ([None] if token == image_token else list(group))
                    ]
                    assert list(request.token_ids.ids) == unexpanded
                else:
                    prompt_ids = list(request.token_ids.ids)
                reports: dict[str, Any] = {}
                params = transfer_params(request.kv.ec_transfer_params)
                for index, media in enumerate(request.media):
                    kwargs = prompt["mm_kwargs"]["image"][index]
                    assert "pixel_values" in kwargs
                    entry = reports.setdefault(
                        prompt["mm_hashes"]["image"][index],
                        {
                            "modality": "image",
                            "metadata": {"image_grid_thw": []},
                            "item_indices": [],
                            "peer_port": 19000,
                        },
                    )
                    entry["item_indices"].append(index)
                    entry["metadata"]["image_grid_thw"] = [
                        kwargs["image_grid_thw"].data.tolist()
                    ]
                    if params:
                        entry["transfer_id"] = params["ec_items"][index]["transfer_id"]
                ParseDict(reports, finish.ec_transfer_params)
            else:
                prompt_ids = list(request.token_ids.ids)
            if role in ("prefill", "pd"):
                assert request.kv.HasField("ec_transfer_params")
                for media in request.media:
                    kwargs = MsgpackDecoder(t=MultiModalKwargsItem).decode(
                        media.features.kwargs
                    )
                    assert "pixel_values" not in kwargs
                    assert "image_grid_thw" in kwargs
                if role == "prefill":
                    ParseDict(
                        {
                            "do_remote_prefill": True,
                            "remote_port": 19001,
                            "remote_block_ids": [1, 2],
                        },
                        finish.kv_transfer_params,
                    )
            if role == "decode" and request.media:
                assert not request.kv.HasField("ec_transfer_params")
                assert (
                    transfer_params(request.kv.kv_transfer_params)["remote_port"]
                    == 19001
                )
            return pb.GenerateResponse(
                prompt_info=pb.PromptInfo(
                    num_prompt_tokens=len(prompt_ids), token_ids=prompt_ids
                ),
                outputs=pb.SequenceOutput(
                    token_ids=answer if role in ("decode", "pd") else [],
                    finish_info=finish,
                ),
            )

        async def GenerateStream(self, request, context):
            response = await self.Generate(request, context)
            yield pb.GenerateResponse(
                prompt_info=response.prompt_info,
                outputs=pb.SequenceOutput(token_ids=answer),
            )
            response.ClearField("prompt_info")
            if stream_blocked is not None:
                await stream_blocked.wait()
            response.outputs.ClearField("token_ids")
            yield response

    class Control(control.ControlServicer):  # type: ignore[name-defined]
        async def GetServerInfo(self, request, context):
            return control_pb.ServerInfo(engine_version="test")

        async def GetModelInfo(self, request, context):
            return control_pb.ModelInfo(model_id=MODEL, served_model_name=MODEL)

        async def Abort(self, request, context):
            if aborted is not None:
                aborted.set()
            return control_pb.AbortResponse()

    server = grpc.aio.server()
    inference.add_InferenceServicer_to_server(Inference(), server)
    control.add_ControlServicer_to_server(Control(), server)
    health_service = health.aio.HealthServicer()
    await health_service.set("vllm.Inference", health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        yield f"grpc://127.0.0.1:{port}"
    finally:
        await server.stop(0)


def image_part(color):
    buf = BytesIO()
    Image.new("RGB", (56, 56), color).save(buf, format="PNG")
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,"
            + pybase64.b64encode(buf.getvalue()).decode()
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    ["http://unreachable.invalid/image.png", "data:image/png;base64,not-an-image"],
)
async def test_grpc_proxy_forwards_images_without_fetching_or_decoding(source):
    """Only E may fetch/decode pixels, including for URL and data-URI inputs."""
    backend = GrpcBackend(MODEL)
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": source}}],
            }
        ],
        "max_tokens": 4,
    }
    original = copy.deepcopy(data)
    try:
        _, request, _ = await backend.prepare(data, "raw-image")
        assert len(request.media) == 1
        media = request.media[0]
        assert media.WhichOneof("source") == (
            "data_uri" if source.startswith("data:") else "url"
        )
        assert (media.url or media.data_uri) == source
        assert data == original
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_grpc_usage_counts_worker_expanded_prompt():
    """The proxy's single image marker is not the worker's expanded token count."""
    backend = GrpcBackend(MODEL)
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [image_part("blue")]}],
        "max_tokens": 4,
    }
    seen: list[tuple[str, Any, dict[str, str]]] = []
    try:
        async with worker(backend, "encode", seen) as url:
            response = await backend.request(url, data, {})
            body = msgspec.json.decode(await response.read())
            assert body["usage"]["prompt_tokens"] > len(seen[0][1].token_ids.ids)
            assert body["ec_transfer_params"]
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("split", [False, True])
async def test_dynamic_grpc_epd_preserves_features_and_selected_replica(
    monkeypatch, stream, split
):
    """E/P/D register over HTTP, but every inference hop is native gRPC."""
    backend = GrpcBackend(MODEL)
    monkeypatch.setattr(proxy, "grpc_backend", backend)
    monkeypatch.setenv("ADMIN_API_KEY", "test-key")
    seen: list[tuple[str, Any, dict[str, str]]] = []
    app = proxy.build_app(proxy.EPDProxyConfig(probe_interval=0))
    async with (
        worker(backend, "encode", seen) as e,
        worker(backend, "prefill", seen) as p,
        worker(backend, "decode" if split else "pd", seen) as d,
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client,
    ):
        instances = [("encode", e)] + (
            [("prefill", p), ("decode", d)] if split else [("prefill_decode", d)]
        )
        for role, url in instances:
            payload = {"role": role, "url": url}
            if role in ("prefill", "prefill_decode"):
                payload.update(
                    dp_size=2, ec_zmq_addrs=["tcp://p:19000", "tcp://p:19001"]
                )
            response = await client.post(
                "/instances", json=payload, headers={"X-API-Key": "test-key"}
            )
            assert response.status_code == 200, response.text
        assert await backend.healthy(p)
        assert (await client.get("/v1/models")).json()["data"][0]["id"] == MODEL
        for rank in range(2):
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                image_part("blue"),
                                {"type": "text", "text": "Compare."},
                                image_part("red"),
                                image_part("blue"),
                            ],
                        }
                    ],
                    "max_tokens": 8,
                    "temperature": 0,
                    "stream": stream,
                    **({"stream_options": {"include_usage": True}} if stream else {}),
                },
            )
            assert response.status_code == 200, response.text
            if stream:
                events = [
                    line[6:]
                    for line in response.text.splitlines()
                    if line.startswith("data: ")
                ]
                assert events[-1] == "[DONE]"
                chunks = [msgspec.json.decode(event) for event in events[:-1]]
                content = "".join(
                    c.get("delta", {}).get("content", "")
                    for ch in chunks
                    for c in ch["choices"]
                )
                assert content == "blue"
                assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
                assert chunks[-1]["choices"] == []
                usage = chunks[-1]["usage"]
            else:
                body = response.json()
                assert body["choices"][0]["message"]["content"] == "blue"
                usage = body["usage"]
            if split:
                encode, prefill, decode = seen[-3:]
                assert "x-data-parallel-rank" not in decode[2]
                assert prefill[1].token_ids == decode[1].token_ids
            else:
                encode, prefill = seen[-2:]
            assert prefill[2]["x-data-parallel-rank"] == str(rank)
            assert (
                transfer_params(encode[1].kv.ec_transfer_params)["consumer_zmq"]
                == f"tcp://p:1900{rank}"
            )
            assert usage["prompt_tokens"] == len(prefill[1].token_ids.ids)
        response = await client.delete(
            "/instances",
            params={"url": p if split else d},
            headers={"X-API-Key": "test-key"},
        )
        assert response.json() == {"removed": True}
        assert (
            await client.post("/v1/chat/completions", json={"messages": []})
        ).status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_abandoned_grpc_encoder_is_explicitly_aborted(cancel):
    """Cancelling a unary call must also release engine-side work."""
    backend = GrpcBackend(MODEL, request_timeout=2)
    seen: list[tuple[str, Any, dict[str, str]]] = []
    started, blocked, aborted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    try:
        async with worker(
            backend, "encode", seen, started=started, blocked=blocked, aborted=aborted
        ) as url:
            task = asyncio.create_task(
                backend.request(
                    url,
                    {
                        "model": MODEL,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 1,
                    },
                    {},
                )
            )
            await asyncio.wait_for(started.wait(), 10)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                from fastapi import HTTPException

                with pytest.raises(HTTPException) as error:
                    await task
                assert error.value.status_code == 504
            await asyncio.wait_for(aborted.wait(), 10)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_closing_grpc_stream_aborts_worker():
    """An SSE consumer can disconnect after reading only the first chunk."""
    backend = GrpcBackend(MODEL)
    blocked, aborted = asyncio.Event(), asyncio.Event()
    try:
        async with worker(
            backend, "decode", [], stream_blocked=blocked, aborted=aborted
        ) as url:
            response = await backend.request(
                url,
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 4,
                    "stream": True,
                },
                {},
            )
            async with response:
                first = await asyncio.wait_for(anext(response.content.iter_any()), 10)
                assert b'"content":"blue"' in first
            await asyncio.wait_for(aborted.wait(), 10)
    finally:
        await backend.close()
