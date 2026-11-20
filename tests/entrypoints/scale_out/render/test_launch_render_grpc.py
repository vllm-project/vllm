# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for the `vllm.Render` gRPC service on `vllm launch render`.

Each gRPC `Render` result must match what the HTTP `/render` route of the same
server returns for the same JSON body.
"""

import io
import json

import httpx
import pybase64 as base64
import pytest
import pytest_asyncio

grpc = pytest.importorskip("grpc")
pytest.importorskip("grpc_health")
render_pb2 = pytest.importorskip(
    "vllm.entrypoints.scale_out.render.grpc.render_pb2",
    reason="vllm.Render stubs not generated (run compile_protos)",
)

from google.protobuf import json_format  # noqa: E402
from grpc_health.v1 import health_pb2, health_pb2_grpc  # noqa: E402

from tests.utils import RemoteLaunchRenderServer  # noqa: E402
from vllm.entrypoints.scale_out.render.grpc import render_pb2_grpc  # noqa: E402
from vllm.utils.network_utils import get_open_port  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-0.6B"
VL_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


class _GrpcRenderServer:
    def __init__(self, model: str, args: list[str]):
        self.grpc_port = get_open_port()
        self.server = RemoteLaunchRenderServer(
            model, [*args, "--grpc-port", str(self.grpc_port)], max_wait_seconds=240
        )

    def __enter__(self):
        self.server.__enter__()
        return self

    def __exit__(self, *exc):
        self.server.__exit__(*exc)


@pytest.fixture(scope="module")
def server():
    args = [
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    with _GrpcRenderServer(MODEL_NAME, args) as srv:
        yield srv


@pytest.fixture(scope="module")
def vl_server():
    with _GrpcRenderServer(VL_MODEL_NAME, ["--max-model-len", "4096"]) as srv:
        yield srv


@pytest_asyncio.fixture
async def http_client(server):
    async with httpx.AsyncClient(base_url=server.server.url_for(""), timeout=60) as c:
        yield c


@pytest_asyncio.fixture
async def stub(server):
    async with grpc.aio.insecure_channel(f"localhost:{server.grpc_port}") as channel:
        yield render_pb2_grpc.RenderStub(channel)


async def _render(stub, api, body: dict, mm_payload=render_pb2.MM_PAYLOAD_UNSPECIFIED):
    return await stub.Render(
        render_pb2.RenderRequest(
            api=api, body=json.dumps(body).encode(), mm_payload=mm_payload
        )
    )


async def _http_render(client, path: str, body: dict) -> dict:
    response = await client.post(path, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _assert_matches_http(response, http: dict) -> None:
    assert list(response.prompt_token_ids) == http["token_ids"]

    structured = (http["sampling_params"] or {}).get("structured_outputs")
    if structured:
        kind = response.structured_output.WhichOneof("kind")
        assert kind is not None
        value = getattr(response.structured_output, kind)
        if kind == "json" and not isinstance(structured["json"], str):
            assert json.loads(value) == structured["json"]
        elif kind == "choice":
            assert list(value.choices) == structured["choice"]
        else:
            assert value == structured[kind]
    else:
        assert not response.HasField("structured_output")

    kwargs = (http.get("reasoning_parser_kwargs") or {}).get("chat_template_kwargs")
    gate = response.engine_reasoning_gate
    if http.get("reasoning_ended") is None and not kwargs:
        assert not response.HasField("engine_reasoning_gate")
    else:
        assert gate.HasField("reasoning_ended") == (
            http.get("reasoning_ended") is not None
        )
        if http.get("reasoning_ended") is not None:
            assert gate.reasoning_ended == http["reasoning_ended"]
        assert json_format.MessageToDict(gate.chat_template_kwargs) == (kwargs or {})


CHAT_BODIES = {
    "plain": {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    },
    "no_thinking": {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "tool_required": {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": TOOLS,
        "tool_choice": "required",
    },
    "json_schema": {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Give me a city."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "city",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        },
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(CHAT_BODIES))
async def test_render_chat_matches_http(http_client, stub, case):
    body = CHAT_BODIES[case]
    http = await _http_render(http_client, "/v1/chat/completions/render", body)
    response = await _render(stub, render_pb2.API_CHAT, body)
    _assert_matches_http(response, http)


@pytest.mark.asyncio
async def test_render_completions_matches_http(http_client, stub):
    body = {"model": MODEL_NAME, "prompt": "Once upon a time", "max_tokens": 8}
    http = await _http_render(http_client, "/v1/completions/render", body)
    assert len(http) == 1
    response = await _render(stub, render_pb2.API_COMPLETIONS, body)
    _assert_matches_http(response, http[0])


@pytest.mark.asyncio
async def test_render_completions_rejects_batched_prompts(stub):
    body = {"model": MODEL_NAME, "prompt": ["a", "b"], "max_tokens": 8}
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _render(stub, render_pb2.API_COMPLETIONS, body)
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_render_messages_matches_http(http_client, stub):
    body = {
        "model": MODEL_NAME,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    http = await _http_render(http_client, "/v1/messages/render", body)
    response = await _render(stub, render_pb2.API_ANTHROPIC_MESSAGES, body)
    _assert_matches_http(response, http)


@pytest.mark.asyncio
async def test_render_responses_matches_http(http_client, stub):
    body = {"model": MODEL_NAME, "input": "Tell me a joke."}
    http = await _http_render(http_client, "/v1/responses/render", body)
    response = await _render(stub, render_pb2.API_RESPONSES, body)
    _assert_matches_http(response, http)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api", "body", "code"),
    [
        (
            render_pb2.API_UNSPECIFIED,
            {"model": MODEL_NAME, "messages": []},
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
        (render_pb2.API_CHAT, {"messages": "nope"}, grpc.StatusCode.INVALID_ARGUMENT),
        (
            render_pb2.API_CHAT,
            {"model": "no-such-model", "messages": [{"role": "user", "content": "x"}]},
            grpc.StatusCode.NOT_FOUND,
        ),
        (
            render_pb2.API_RESPONSES,
            {"model": MODEL_NAME, "input": "x", "previous_response_id": "resp_1"},
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
    ],
)
async def test_render_errors(stub, api, body, code):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _render(stub, api, body)
    assert exc_info.value.code() == code


@pytest.mark.asyncio
async def test_render_rejects_malformed_json(stub):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await stub.Render(render_pb2.RenderRequest(api=render_pb2.API_CHAT, body=b"{"))
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_render_chat_typed_rpc_unimplemented(stub):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await stub.RenderChat(render_pb2.RenderChatRequest())
    assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED


@pytest.mark.asyncio
async def test_health(server):
    async with grpc.aio.insecure_channel(f"localhost:{server.grpc_port}") as channel:
        health = health_pb2_grpc.HealthStub(channel)
        for service in ("", "vllm.Render"):
            response = await health.Check(
                health_pb2.HealthCheckRequest(service=service)
            )
            assert response.status == health_pb2.HealthCheckResponse.SERVING


def _image_data_uri() -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(200, 30, 30)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mm_payload", [render_pb2.MM_PAYLOAD_TENSORS, render_pb2.MM_PAYLOAD_NONE]
)
async def test_render_image_matches_http(vl_server, mm_payload):
    body = {
        "model": VL_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_uri()}},
                    {"type": "text", "text": "What color is this?"},
                ],
            }
        ],
    }
    async with httpx.AsyncClient(
        base_url=vl_server.server.url_for(""), timeout=120
    ) as client:
        http = await _http_render(client, "/v1/chat/completions/render", body)
    async with grpc.aio.insecure_channel(f"localhost:{vl_server.grpc_port}") as channel:
        response = await _render(
            render_pb2_grpc.RenderStub(channel), render_pb2.API_CHAT, body, mm_payload
        )

    assert list(response.prompt_token_ids) == http["token_ids"]
    features = http["features"]
    [item] = response.media
    [placeholder] = features["mm_placeholders"]["image"]
    assert item.features.identifier == features["mm_hashes"]["image"][0]
    assert item.features.offset == placeholder["offset"]
    assert item.features.length == placeholder["length"]
    if mm_payload == render_pb2.MM_PAYLOAD_NONE:
        assert not item.features.HasField("kwargs")
    else:
        [kwargs_b64] = features["kwargs_data"]["image"]
        assert item.features.kwargs == base64.b64decode(kwargs_b64)
