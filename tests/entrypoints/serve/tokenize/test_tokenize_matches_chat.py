# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`/tokenize` must render the same prompt as `/v1/chat/completions`."""

import json

import pytest
import requests

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
MESSAGES = [{"role": "user", "content": "What is 2+2?"}]
DOCUMENTS_TEMPLATE = (
    "{% for d in (documents or []) %}[DOC {{ d.title }}: {{ d.text }}]{% endfor %}"
    "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--max-num-seqs",
        "8",
        "--trust-request-chat-template",
        "--reasoning-effort-map",
        json.dumps({"minimal": "none"}),
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def _tokenize(server: RemoteOpenAIServer, body: dict) -> list[int]:
    response = requests.post(server.url_for("tokenize"), json=body)
    response.raise_for_status()
    return response.json()["tokens"]


def _chat_prompt_ids(server: RemoteOpenAIServer, body: dict) -> list[int]:
    response = requests.post(
        server.url_for("v1/chat/completions"),
        json=body | {"max_tokens": 1, "return_token_ids": True},
    )
    response.raise_for_status()
    return response.json()["prompt_token_ids"]


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"reasoning_effort": "none"},
        {"reasoning_effort": "high"},
        # Rewritten to "none" by --reasoning-effort-map.
        {"reasoning_effort": "minimal"},
        # An explicit enable_thinking wins over reasoning_effort.
        {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": True}},
    ],
)
def test_tokenize_matches_chat_reasoning_effort(
    server: RemoteOpenAIServer, extra: dict
):
    body = {"model": MODEL_NAME, "messages": MESSAGES} | extra
    assert _tokenize(server, body) == _chat_prompt_ids(server, body)


def test_tokenize_reasoning_effort_changes_prompt(server: RemoteOpenAIServer):
    """Guard against the comparison above passing with both fields ignored."""
    base = {"model": MODEL_NAME, "messages": MESSAGES}
    default_ids = _tokenize(server, base)
    no_think_ids = _tokenize(server, base | {"reasoning_effort": "none"})
    assert no_think_ids != default_ids
    assert len(no_think_ids) > len(default_ids)


def test_tokenize_matches_chat_documents(server: RemoteOpenAIServer):
    body = {
        "model": MODEL_NAME,
        "messages": MESSAGES,
        "chat_template": DOCUMENTS_TEMPLATE,
        "documents": [{"title": "Math", "text": "2+2 equals 4."}],
    }
    ids = _tokenize(server, body)
    assert ids == _chat_prompt_ids(server, body)

    without_documents = _tokenize(
        server, {k: v for k, v in body.items() if k != "documents"}
    )
    assert len(ids) > len(without_documents)
