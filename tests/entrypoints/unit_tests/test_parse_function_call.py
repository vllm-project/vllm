# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the native /parse_function_call endpoint."""

import json
from argparse import Namespace
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.parser.api_router import register_parser_api_router

HERMES_OUTPUT = (
    "Let me check.\n"
    "<tool_call>\n"
    '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
    "</tool_call>"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def _client(tool_call_parser: str | None) -> TestClient:
    app = FastAPI()
    app.state.args = Namespace(tool_call_parser=tool_call_parser)
    app.state.serving_tokenization = SimpleNamespace(
        renderer=SimpleNamespace(get_tokenizer=lambda: SimpleNamespace())
    )
    register_parser_api_router(app)
    return TestClient(app)


def test_parses_tool_call_with_explicit_parser():
    response = _client(None).post(
        "/parse_function_call",
        json={
            "text": HERMES_OUTPUT,
            "tools": TOOLS,
            "tool_call_parser": "hermes",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["normal_text"] == "Let me check.\n"
    assert len(body["calls"]) == 1
    call = body["calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}


def test_falls_back_to_server_parser():
    response = _client("hermes").post(
        "/parse_function_call",
        json={"text": HERMES_OUTPUT, "tools": TOOLS},
    )

    assert response.status_code == 200
    assert response.json()["calls"][0]["function"]["name"] == "get_weather"


def test_text_without_tool_call_is_returned_unchanged():
    response = _client("hermes").post(
        "/parse_function_call",
        json={"text": "Just a plain answer.", "tools": TOOLS},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["calls"] == []
    assert body["normal_text"] == "Just a plain answer."


def test_missing_parser_is_a_bad_request():
    response = _client(None).post(
        "/parse_function_call",
        json={"text": HERMES_OUTPUT, "tools": TOOLS},
    )

    assert response.status_code == 400
    assert "No tool parser selected" in response.json()["error"]["message"]


def test_unknown_parser_is_a_bad_request():
    response = _client(None).post(
        "/parse_function_call",
        json={
            "text": HERMES_OUTPUT,
            "tools": TOOLS,
            "tool_call_parser": "not-a-real-parser",
        },
    )

    assert response.status_code == 400
    assert "Unknown tool parser" in response.json()["error"]["message"]
