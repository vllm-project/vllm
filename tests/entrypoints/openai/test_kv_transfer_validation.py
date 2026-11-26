# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
from fastapi.testclient import TestClient

from vllm.entrypoints.common.cli_args import FrontendArgs
from vllm.entrypoints.launchers.app import build_app
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError

REMOTE_KV_SOURCE = {
    "remote_kv_source": {
        "kv_request_id": "shared",
        "remote_host": "127.0.0.1",
        "remote_port": 1234,
    }
}

COMPLETION_BATCH = {"prompt": ["first", "second"]}
CHAT = {"messages": [{"role": "user", "content": "single"}]}
BATCH_CHAT = {
    "messages": [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]
}


@pytest.mark.parametrize(
    "request_type,request_args",
    [
        (CompletionRequest, COMPLETION_BATCH),
        (CompletionRequest, {"prompt": [[1], [2]]}),
        (CompletionRequest, {"prompt_embeds": [b"first", b"second"]}),
        (
            CompletionRequest,
            {"prompt": "single", "prompt_embeds": b"embedding"},
        ),
        (CompletionRequest, {"prompt": "single", "n": 2}),
        (ChatCompletionRequest, {**CHAT, "n": 2}),
        (BatchChatCompletionRequest, BATCH_CHAT),
    ],
)
def test_rejects_remote_kv_source_fanout(request_type, request_args):
    with pytest.raises(
        VLLMValidationError,
        match=r"parameter=kv_transfer_params\.remote_kv_source",
    ):
        request_type(
            model="test-model",
            kv_transfer_params=REMOTE_KV_SOURCE,
            **request_args,
        )


@pytest.mark.parametrize(
    "request_type,request_args",
    [
        (CompletionRequest, {"prompt": "single"}),
        (
            CompletionRequest,
            {"prompt": "single", "n": 2, "use_beam_search": True},
        ),
        (ChatCompletionRequest, CHAT),
        (ChatCompletionRequest, {**CHAT, "n": 2, "use_beam_search": True}),
        (
            BatchChatCompletionRequest,
            {"messages": [[{"role": "user", "content": "single"}]]},
        ),
    ],
)
def test_allows_remote_kv_source_without_fanout(request_type, request_args):
    request = request_type(
        model="test-model",
        kv_transfer_params=REMOTE_KV_SOURCE,
        **request_args,
    )

    assert request.kv_transfer_params == REMOTE_KV_SOURCE


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    return TestClient(
        build_app(FrontendArgs(), supported_tasks=("generate",)),
        raise_server_exceptions=False,
    )


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/v1/completions", COMPLETION_BATCH),
        (
            "/v1/completions",
            {"prompt": "single", "prompt_embeds": "embedding"},
        ),
        ("/v1/completions", {"prompt": "single", "n": 2}),
        ("/v1/chat/completions", {**CHAT, "n": 2}),
        ("/v1/chat/completions/batch", BATCH_CHAT),
    ],
)
def test_remote_kv_source_fanout_returns_400(
    api_client: TestClient, path: str, payload: dict[str, Any]
) -> None:
    response = api_client.post(
        path,
        json={
            "model": "test-model",
            "kv_transfer_params": REMOTE_KV_SOURCE,
            **payload,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == ("kv_transfer_params.remote_kv_source")
