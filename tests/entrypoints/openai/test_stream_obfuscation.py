# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream obfuscation routing: each streaming OpenAI endpoint honors
``stream_options.include_obfuscation`` and falls back to
``--enable-stream-obfuscation``."""

import json
from argparse import Namespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

from vllm.entrypoints.openai.chat_completion import api_router as chat_router
from vllm.entrypoints.openai.completion import api_router as completion_router
from vllm.entrypoints.openai.responses import api_router as responses_router

_CHUNK = 'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\n'


class _OtherEvent(BaseModel):
    type: str = "response.completed"


async def _chunks():
    yield _CHUNK
    yield "data: [DONE]\n\n"


async def _events():
    yield ResponseTextDeltaEvent(
        content_index=0,
        delta="hi",
        item_id="msg_1",
        logprobs=[],
        output_index=0,
        sequence_number=1,
        type="response.output_text.delta",
    )
    yield _OtherEvent()


class _Handler:
    async def create_chat_completion(self, request, raw_request):
        return _chunks()

    async def create_completion(self, request, raw_request):
        return _chunks()

    async def create_responses(self, request, raw_request):
        return _events()

    async def retrieve_responses(self, response_id, *, starting_after, stream):
        return _events()


def _client(server_default: bool) -> TestClient:
    app = FastAPI()
    app.state.args = Namespace(enable_stream_obfuscation=server_default)
    for module in (chat_router, completion_router, responses_router):
        module.attach_router(app)
    handler = _Handler()
    app.state.openai_serving_chat = handler
    app.state.openai_serving_completion = handler
    app.state.openai_serving_responses = handler
    return TestClient(app)


def _data_payloads(body: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.split("\n")
        if line.startswith("data: {")
    ]


def _post(client: TestClient, path: str, include_obfuscation: bool | None) -> str:
    body: dict = {"model": "m", "stream": True}
    if path == "/v1/chat/completions":
        body["messages"] = [{"role": "user", "content": "hi"}]
    elif path == "/v1/completions":
        body["prompt"] = "hi"
    else:
        body["input"] = "hi"
    if include_obfuscation is not None:
        body["stream_options"] = {"include_obfuscation": include_obfuscation}
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    return response.text


@pytest.mark.parametrize(
    "path", ["/v1/chat/completions", "/v1/completions", "/v1/responses"]
)
@pytest.mark.parametrize(
    ("server_default", "include_obfuscation", "expected"),
    [
        (False, None, False),
        (True, None, True),
        (True, False, False),
        (False, True, True),
    ],
)
def test_request_option_and_server_default(
    path, server_default, include_obfuscation, expected
):
    body = _post(_client(server_default), path, include_obfuscation)
    first = _data_payloads(body)[0]
    assert ("obfuscation" in first) is expected


def test_responses_pads_only_delta_events():
    body = _post(_client(server_default=True), "/v1/responses", None)
    delta, other = _data_payloads(body)
    assert "obfuscation" in delta
    assert "obfuscation" not in other


def test_retrieve_responses_honors_query_parameter():
    client = _client(server_default=False)
    url = "/v1/responses/resp_1?stream=true"
    assert "obfuscation" not in client.get(url).text
    padded = client.get(url + "&include_obfuscation=true").text
    assert "obfuscation" in _data_payloads(padded)[0]
