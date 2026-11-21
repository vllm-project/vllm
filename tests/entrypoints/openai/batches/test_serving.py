# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.batches.protocol import BatchCreateRequest, BatchObject
from vllm.entrypoints.openai.batches.serving import OpenAIServingBatches
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.files.serving import OpenAIServingFiles
from vllm.entrypoints.serve import create_error_response
from vllm.entrypoints.serve.engine.protocol import ErrorResponse, UsageInfo

CHAT_URL = "/v1/chat/completions"


def _chat_response() -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-test",
        created=int(time.time()),
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


async def _slow_chat_response(*args, **kwargs) -> ChatCompletionResponse:
    await asyncio.sleep(10)
    return _chat_response()


def _chat_handler(side_effect=None) -> MagicMock:
    handler = MagicMock()
    handler.create_chat_completion = AsyncMock(
        return_value=_chat_response(), side_effect=side_effect
    )
    return handler


def _chat_lines(n: int) -> bytes:
    return "\n".join(
        json.dumps(
            {
                "custom_id": f"req-{i}",
                "method": "POST",
                "url": CHAT_URL,
                "body": {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": f"Hello {i}"}],
                },
            }
        )
        for i in range(n)
    ).encode()


@pytest.fixture
def storage_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def serving_files(storage_dir):
    return OpenAIServingFiles(storage_dir=storage_dir)


def _make_serving_batches(
    storage_dir, serving_files, serving_chat=None, serving_scores=None
):
    return OpenAIServingBatches(
        storage_dir=storage_dir,
        serving_files=serving_files,
        serving_chat=serving_chat,
        serving_embedding=None,
        serving_scores=serving_scores,
    )


@pytest_asyncio.fixture
async def serving_batches(storage_dir, serving_files):
    serving = _make_serving_batches(storage_dir, serving_files, _chat_handler())
    yield serving
    await serving.shutdown()


async def _create_batch(
    serving_batches, serving_files, content: bytes, endpoint: str = CHAT_URL
) -> BatchObject:
    file_obj = await serving_files.upload_file(content, "in.jsonl", "batch")
    batch = await serving_batches.create_batch(
        BatchCreateRequest(input_file_id=file_obj.id, endpoint=endpoint)
    )
    assert isinstance(batch, BatchObject)
    return batch


@pytest.mark.asyncio
async def test_create_batch(serving_batches, serving_files):
    batch = await _create_batch(serving_batches, serving_files, _chat_lines(2))

    assert batch.id.startswith("batch-")
    assert batch.status == "validating"
    assert await serving_batches.get_batch(batch.id) is batch


@pytest.mark.asyncio
async def test_create_batch_with_missing_file(serving_batches):
    result = await serving_batches.create_batch(
        BatchCreateRequest(input_file_id="file-missing", endpoint=CHAT_URL)
    )

    assert isinstance(result, ErrorResponse)
    assert result.error.code == 404


@pytest.mark.asyncio
async def test_create_batch_with_unserved_endpoint(serving_batches, serving_files):
    file_obj = await serving_files.upload_file(_chat_lines(1), "in.jsonl", "batch")

    result = await serving_batches.create_batch(
        BatchCreateRequest(input_file_id=file_obj.id, endpoint="/v1/embeddings")
    )

    assert isinstance(result, ErrorResponse)
    assert result.error.code == 400


@pytest.mark.asyncio
async def test_batch_runs_to_completion(serving_batches, serving_files):
    batch = await _create_batch(serving_batches, serving_files, _chat_lines(2))
    await serving_batches.wait_for_batch(batch.id, timeout=10)

    assert batch.status == "completed"
    assert batch.completed_at is not None
    assert batch.request_counts.model_dump() == {
        "total": 2,
        "completed": 2,
        "failed": 0,
    }
    assert batch.error_file_id is None
    assert batch.output_file_id is not None
    output = await serving_files.get_file_content(batch.output_file_id)
    lines = [json.loads(line) for line in output.decode().splitlines()]
    assert {line["custom_id"] for line in lines} == {"req-0", "req-1"}
    assert lines[0]["response"]["body"]["choices"][0]["message"]["content"] == "hello"


@pytest.mark.asyncio
async def test_handler_errors_go_to_error_file(storage_dir, serving_files):
    serving_chat = _chat_handler()
    serving_chat.create_chat_completion.return_value = create_error_response("bad")
    serving = _make_serving_batches(storage_dir, serving_files, serving_chat)

    batch = await _create_batch(serving, serving_files, _chat_lines(1))
    await serving.wait_for_batch(batch.id, timeout=10)

    assert batch.status == "completed"
    assert batch.request_counts.failed == 1
    assert batch.output_file_id is None
    errors = await serving_files.get_file_content(batch.error_file_id)
    assert json.loads(errors)["response"]["status_code"] == 400
    await serving.shutdown()


@pytest.mark.asyncio
async def test_rerank_uses_scores_handler(storage_dir, serving_files):
    serving_scores = AsyncMock(
        side_effect=[
            JSONResponse(content={"results": []}),
            JSONResponse(
                content=create_error_response("bad").model_dump(), status_code=400
            ),
        ]
    )
    serving = _make_serving_batches(
        storage_dir, serving_files, serving_scores=serving_scores
    )
    content = "\n".join(
        json.dumps(
            {
                "custom_id": f"req-{i}",
                "method": "POST",
                "url": "/v1/rerank",
                "body": {"model": "m", "query": "q", "documents": ["d"]},
            }
        )
        for i in range(2)
    ).encode()

    batch = await _create_batch(serving, serving_files, content, "/v1/rerank")
    await serving.wait_for_batch(batch.id, timeout=10)

    assert serving_scores.await_count == 2
    assert batch.request_counts.completed == 1
    assert batch.request_counts.failed == 1
    output = await serving_files.get_file_content(batch.output_file_id)
    assert json.loads(output)["response"]["body"] == {"results": []}
    await serving.shutdown()


@pytest.mark.asyncio
async def test_invalid_lines_fail_the_batch(serving_batches, serving_files):
    content = _chat_lines(1) + b'\n{"custom_id": "bad"}'

    batch = await _create_batch(serving_batches, serving_files, content)
    await serving_batches.wait_for_batch(batch.id, timeout=10)

    assert batch.status == "failed"
    assert batch.errors is not None
    assert batch.errors.data[0].line == 2


@pytest.mark.asyncio
async def test_request_validator_errors_report_the_line(serving_batches, serving_files):
    invalid = {
        "custom_id": "bad",
        "method": "POST",
        "url": CHAT_URL,
        "body": {"model": "m", "messages": [], "top_logprobs": 5},
    }
    content = _chat_lines(1) + b"\n" + json.dumps(invalid).encode()

    batch = await _create_batch(serving_batches, serving_files, content)
    await serving_batches.wait_for_batch(batch.id, timeout=10)

    assert batch.status == "failed"
    assert batch.errors is not None
    assert batch.errors.data[0].line == 2
    assert "top_logprobs" in batch.errors.data[0].message


@pytest.mark.asyncio
async def test_cleanup_keeps_files_of_active_batches(storage_dir, serving_files):
    serving = _make_serving_batches(
        storage_dir, serving_files, _chat_handler(side_effect=_slow_chat_response)
    )
    file_obj = await serving_files.upload_file(_chat_lines(1), "in.jsonl", "batch")
    request = BatchCreateRequest(input_file_id=file_obj.id, endpoint=CHAT_URL)
    expired = await serving.create_batch(request)
    active = await serving.create_batch(request)
    assert isinstance(expired, BatchObject)
    assert isinstance(active, BatchObject)
    await asyncio.sleep(0.2)
    expired.status = "failed"
    expired.expires_at = int(time.time()) - 1

    await serving._cleanup_expired()

    assert await serving.get_batch(expired.id) is None
    assert await serving_files.get_file(file_obj.id) is not None
    await serving.cancel_batch(active.id)
    await serving.shutdown()


@pytest.mark.asyncio
async def test_list_batches_pagination(serving_batches, serving_files):
    for _ in range(3):
        await _create_batch(serving_batches, serving_files, _chat_lines(1))

    first_page = await serving_batches.list_batches(limit=2)
    second_page = await serving_batches.list_batches(limit=2, after=first_page.last_id)

    assert len(first_page.data) == 2
    assert first_page.has_more
    assert len(second_page.data) == 1
    assert not second_page.has_more


@pytest.mark.asyncio
async def test_cancel_batch(storage_dir, serving_files):
    serving = _make_serving_batches(
        storage_dir, serving_files, _chat_handler(side_effect=_slow_chat_response)
    )
    batch = await _create_batch(serving, serving_files, _chat_lines(5))
    await asyncio.sleep(0.2)

    assert serving.is_file_in_active_batch(batch.input_file_id)
    result = await serving.cancel_batch(batch.id)
    assert isinstance(result, BatchObject)
    assert result.status == "cancelling"

    await serving.wait_for_batch(batch.id, timeout=15)
    assert batch.status == "cancelled"
    assert batch.cancelled_at is not None
    assert not serving.is_file_in_active_batch(batch.input_file_id)
    await serving.shutdown()


@pytest.mark.asyncio
async def test_cancel_finished_batch_is_rejected(serving_batches, serving_files):
    batch = await _create_batch(serving_batches, serving_files, _chat_lines(1))
    await serving_batches.wait_for_batch(batch.id, timeout=10)

    result = await serving_batches.cancel_batch(batch.id)

    assert isinstance(result, ErrorResponse)
    assert await serving_batches.cancel_batch("batch-missing") is None


@pytest.mark.asyncio
async def test_expired_batches_are_cleaned_up(serving_batches, serving_files):
    batch = await _create_batch(serving_batches, serving_files, _chat_lines(1))
    await serving_batches.wait_for_batch(batch.id, timeout=10)
    batch.expires_at = int(time.time()) - 1

    await serving_batches._cleanup_expired()

    assert await serving_batches.get_batch(batch.id) is None
    assert await serving_files.get_file(batch.input_file_id) is None
    assert await serving_files.get_file(batch.output_file_id) is None


@pytest.mark.asyncio
async def test_metadata_persists_across_instances(storage_dir, serving_files):
    serving = _make_serving_batches(storage_dir, serving_files, _chat_handler())
    batch = await _create_batch(serving, serving_files, _chat_lines(1))
    await serving.wait_for_batch(batch.id, timeout=10)
    await serving.shutdown()

    reloaded = _make_serving_batches(storage_dir, serving_files)

    retrieved = await reloaded.get_batch(batch.id)
    assert retrieved is not None
    assert retrieved.status == "completed"


@pytest.mark.asyncio
async def test_interrupted_batches_fail_on_restart(storage_dir, serving_files):
    metadata_dir = os.path.join(storage_dir, "metadata")
    with open(os.path.join(metadata_dir, "batches.json"), "w") as f:
        json.dump(
            [
                {
                    "id": "batch-interrupted",
                    "endpoint": CHAT_URL,
                    "input_file_id": "file-x",
                    "status": "in_progress",
                    "completion_window": "24h",
                    "created_at": int(time.time()),
                }
            ],
            f,
        )

    serving = _make_serving_batches(storage_dir, serving_files)

    batch = await serving.get_batch("batch-interrupted")
    assert batch is not None
    assert batch.status == "failed"
    assert batch.errors is not None
    assert batch.errors.data[0].code == "server_restart"
