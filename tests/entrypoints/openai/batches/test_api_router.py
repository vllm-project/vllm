# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import time
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.batches.api_router import (
    attach_router as attach_batches_router,
)
from vllm.entrypoints.openai.batches.api_router import init_batch_state
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.files.api_router import (
    attach_router as attach_files_router,
)
from vllm.entrypoints.serve.engine.protocol import UsageInfo

CHAT_URL = "/v1/chat/completions"
REQUEST_LINE = json.dumps(
    {
        "custom_id": "req-0",
        "method": "POST",
        "url": CHAT_URL,
        "body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    }
)


async def _slow_chat_response(*args, **kwargs) -> ChatCompletionResponse:
    await asyncio.sleep(2)
    return ChatCompletionResponse(
        id="chatcmpl-test",
        created=int(time.time()),
        model="m",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


@pytest.fixture
def client(tmp_path):
    app = FastAPI()
    app.state.openai_serving_chat = MagicMock()
    app.state.openai_serving_chat.create_chat_completion = AsyncMock(
        side_effect=_slow_chat_response
    )
    args = Namespace(
        batch_storage_dir=str(tmp_path), batch_priority=0, batch_retention_hours=24
    )
    init_batch_state(app.state, args, ("generate",))
    attach_files_router(app)
    attach_batches_router(app)
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient) -> str:
    response = client.post(
        "/v1/files",
        files={"file": ("in.jsonl", REQUEST_LINE.encode())},
        data={"purpose": "batch"},
    )
    assert response.status_code == 200
    return response.json()["id"]


def _wait_until_finished(client: TestClient, batch_id: str) -> dict:
    for _ in range(50):
        batch = client.get(f"/v1/batches/{batch_id}").json()
        if batch["status"] in ("completed", "failed", "cancelled"):
            return batch
        time.sleep(0.1)
    raise TimeoutError(f"Batch {batch_id} did not finish")


def test_batch_workflow(client):
    file_id = _upload(client)

    created = client.post(
        "/v1/batches", json={"input_file_id": file_id, "endpoint": CHAT_URL}
    )
    assert created.status_code == 200
    batch_id = created.json()["id"]

    in_use = client.delete(f"/v1/files/{file_id}")
    assert in_use.status_code == 409

    batch = _wait_until_finished(client, batch_id)
    assert batch["status"] == "completed"
    output = client.get(f"/v1/files/{batch['output_file_id']}/content")
    assert json.loads(output.content)["custom_id"] == "req-0"

    listed = client.get("/v1/batches").json()
    assert [b["id"] for b in listed["data"]] == [batch_id]
    assert client.delete(f"/v1/files/{file_id}").json()["deleted"] is True


def test_cancel_batch(client):
    file_id = _upload(client)
    batch_id = client.post(
        "/v1/batches", json={"input_file_id": file_id, "endpoint": CHAT_URL}
    ).json()["id"]

    cancelled = client.post(f"/v1/batches/{batch_id}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelling"
    assert _wait_until_finished(client, batch_id)["status"] == "cancelled"


def test_not_found(client):
    assert client.get("/v1/files/file-missing").status_code == 404
    assert client.get("/v1/files/file-missing/content").status_code == 404
    assert client.delete("/v1/files/file-missing").status_code == 404
    assert client.get("/v1/batches/batch-missing").status_code == 404
    assert client.post("/v1/batches/batch-missing/cancel").status_code == 404
    created = client.post(
        "/v1/batches", json={"input_file_id": "file-missing", "endpoint": CHAT_URL}
    )
    assert created.status_code == 404


def test_rejects_unsupported_endpoint_and_purpose(client):
    file_id = _upload(client)

    created = client.post(
        "/v1/batches", json={"input_file_id": file_id, "endpoint": "/v1/completions"}
    )
    uploaded = client.post(
        "/v1/files",
        files={"file": ("in.jsonl", b"{}")},
        data={"purpose": "fine-tune"},
    )

    assert created.status_code == 422
    assert uploaded.status_code == 422
