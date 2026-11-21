# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`kv_hints` is accepted and validated on the OpenAI generate endpoints."""

import pytest
import requests

from ...utils import RemoteOpenAIServer

MODEL_NAME = "unsloth/llama-3.2-1b-Instruct"

VALID_KV_HINTS = {
    "protocol_version": "0.1",
    "message_id": "msg-1",
    "actions": [
        {
            "action_id": "act-1",
            "action_type": "kv.fetch",
            "action_version": "1.0",
            "payload": {"block_hashes": [1234, 5678]},
        }
    ],
}

# Missing the required `message_id`.
MALFORMED_KV_HINTS = {"protocol_version": "0.1", "actions": []}


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.4",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


ENDPOINTS = [
    ("/v1/completions", {"prompt": "Hello, my name is"}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]}),
]


@pytest.mark.parametrize(("path", "body"), ENDPOINTS)
def test_valid_kv_hints_are_accepted(server, path, body):
    response = requests.post(
        server.url_for(path),
        json={
            "model": MODEL_NAME,
            "max_tokens": 4,
            "temperature": 0,
            "kv_hints": VALID_KV_HINTS,
            **body,
        },
    )

    assert response.status_code == 200, response.text


@pytest.mark.parametrize(("path", "body"), ENDPOINTS)
def test_malformed_kv_hints_are_rejected(server, path, body):
    response = requests.post(
        server.url_for(path),
        json={
            "model": MODEL_NAME,
            "max_tokens": 4,
            "temperature": 0,
            "kv_hints": MALFORMED_KV_HINTS,
            **body,
        },
    )

    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == "kv_hints"
    assert "Invalid kv_hints envelope" in error["message"]
