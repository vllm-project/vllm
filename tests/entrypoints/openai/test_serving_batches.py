"""Tests for OpenAIServingBatches."""
import asyncio
import json
import os
import tempfile
import time
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from vllm.entrypoints.openai.batch.protocol import (
    BatchObject,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_batches import OpenAIServingBatches
from vllm.entrypoints.openai.serving_files import OpenAIServingFiles


@pytest.fixture
def storage_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def serving_files(storage_dir):
    return OpenAIServingFiles(storage_dir=storage_dir)


def _make_chat_response():
    """Helper to create a minimal ChatCompletionResponse."""
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


def _make_mock_chat_handler():
    handler = AsyncMock()
    handler.create_chat_completion = AsyncMock(
        return_value=_make_chat_response())
    return handler


@pytest_asyncio.fixture
async def serving_batches(storage_dir, serving_files):
    sb = OpenAIServingBatches(
        storage_dir=storage_dir,
        serving_files=serving_files,
        serving_chat=_make_mock_chat_handler(),
        serving_embedding=None,
        serving_score=None,
        batch_priority=0,
        retention_hours=24,
    )
    yield sb
    # Ensure cleanup task is stopped
    await sb.shutdown()


def _make_batch_jsonl(n=2):
    """Create JSONL content with n chat completion requests."""
    lines = []
    for i in range(n):
        lines.append(json.dumps({
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "test-model",
                "messages": [{"role": "user", "content": f"Hello {i}"}],
                "max_tokens": 10,
            },
        }))
    return "\n".join(lines).encode()


@pytest.mark.asyncio
async def test_create_batch(serving_batches, serving_files):
    content = _make_batch_jsonl(2)
    file_obj = await serving_files.upload_file(content, "in.jsonl", "batch")

    batch = await serving_batches.create_batch(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    assert isinstance(batch, BatchObject)
    assert batch.id.startswith("batch-")
    assert batch.status == "validating"
    assert batch.input_file_id == file_obj.id
    assert batch.request_counts.total == 0  # Set during async processing


@pytest.mark.asyncio
async def test_create_batch_invalid_file(serving_batches):
    result = await serving_batches.create_batch(
        input_file_id="file-nonexistent",
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_create_batch_invalid_endpoint(serving_batches, serving_files):
    content = _make_batch_jsonl(1)
    file_obj = await serving_files.upload_file(content, "in.jsonl", "batch")
    result = await serving_batches.create_batch(
        input_file_id=file_obj.id,
        endpoint="/v1/invalid",
        completion_window="24h",
    )
    assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_get_batch(serving_batches, serving_files):
    content = _make_batch_jsonl(1)
    file_obj = await serving_files.upload_file(content, "in.jsonl", "batch")
    batch = await serving_batches.create_batch(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    retrieved = await serving_batches.get_batch(batch.id)
    assert retrieved is not None
    assert retrieved.id == batch.id


@pytest.mark.asyncio
async def test_get_batch_not_found(serving_batches):
    result = await serving_batches.get_batch("batch-nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_list_batches(serving_batches, serving_files):
    content = _make_batch_jsonl(1)
    f1 = await serving_files.upload_file(content, "a.jsonl", "batch")
    f2 = await serving_files.upload_file(content, "b.jsonl", "batch")
    await serving_batches.create_batch(f1.id, "/v1/chat/completions", "24h")
    await serving_batches.create_batch(f2.id, "/v1/chat/completions", "24h")
    batches, has_more = await serving_batches.list_batches(limit=20)
    assert len(batches) == 2
    assert not has_more


@pytest.mark.asyncio
async def test_list_batches_pagination(serving_batches, serving_files):
    content = _make_batch_jsonl(1)
    ids = []
    for i in range(3):
        f = await serving_files.upload_file(content, f"{i}.jsonl", "batch")
        b = await serving_batches.create_batch(
            f.id, "/v1/chat/completions", "24h")
        ids.append(b.id)

    batches, has_more = await serving_batches.list_batches(limit=2)
    assert len(batches) == 2
    assert has_more

    batches2, has_more2 = await serving_batches.list_batches(
        limit=2, after=batches[-1].id)
    assert len(batches2) == 1
    assert not has_more2


@pytest.mark.asyncio
async def test_batch_processes_to_completion(serving_batches, serving_files):
    content = _make_batch_jsonl(2)
    file_obj = await serving_files.upload_file(content, "in.jsonl", "batch")
    batch = await serving_batches.create_batch(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    # Wait for background processing to complete
    await serving_batches.wait_for_batch(batch.id, timeout=10)

    updated = await serving_batches.get_batch(batch.id)
    assert updated.status == "completed"
    assert updated.request_counts.completed == 2
    assert updated.request_counts.failed == 0
    assert updated.output_file_id is not None
    assert updated.completed_at is not None

    # Output file should exist and contain valid JSONL
    output_content = await serving_files.get_file_content(
        updated.output_file_id)
    assert output_content is not None
    lines = output_content.decode().strip().split("\n")
    assert len(lines) == 2


@pytest.mark.asyncio
async def test_cancel_batch(serving_batches, serving_files):
    # Use a slow handler to give time to cancel
    slow_handler = AsyncMock()

    async def slow_response(*args, **kwargs):
        await asyncio.sleep(10)
        return _make_chat_response()

    slow_handler.create_chat_completion = AsyncMock(side_effect=slow_response)
    serving_batches._serving_chat = slow_handler

    content = _make_batch_jsonl(50)
    file_obj = await serving_files.upload_file(content, "in.jsonl", "batch")
    batch = await serving_batches.create_batch(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    # Let it start processing
    await asyncio.sleep(0.2)

    result = await serving_batches.cancel_batch(batch.id)
    assert result is not None
    assert result.status in ("cancelling", "cancelled")

    # Wait for cancellation to finalize
    await serving_batches.wait_for_batch(batch.id, timeout=10)
    updated = await serving_batches.get_batch(batch.id)
    assert updated.status == "cancelled"
    assert updated.cancelled_at is not None


@pytest.mark.asyncio
async def test_is_file_in_active_batch(serving_batches, serving_files):
    # Use a slow handler so batch stays in_progress
    slow_handler = AsyncMock()

    async def slow_response(*args, **kwargs):
        await asyncio.sleep(10)
        return _make_chat_response()

    slow_handler.create_chat_completion = AsyncMock(side_effect=slow_response)
    serving_batches._serving_chat = slow_handler

    content = _make_batch_jsonl(5)
    file_obj = await serving_files.upload_file(content, "in.jsonl", "batch")
    batch = await serving_batches.create_batch(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    await asyncio.sleep(0.2)

    assert serving_batches.is_file_in_active_batch(file_obj.id) is True
    assert serving_batches.is_file_in_active_batch("file-other") is False

    # Cancel to clean up
    await serving_batches.cancel_batch(batch.id)
    await serving_batches.wait_for_batch(batch.id, timeout=10)


@pytest.mark.asyncio
async def test_metadata_persistence(storage_dir, serving_files):
    sb1 = OpenAIServingBatches(
        storage_dir=storage_dir,
        serving_files=serving_files,
        serving_chat=_make_mock_chat_handler(),
        serving_embedding=None,
        serving_score=None,
        batch_priority=0,
        retention_hours=24,
    )
    content = _make_batch_jsonl(1)
    f = await serving_files.upload_file(content, "in.jsonl", "batch")
    batch = await sb1.create_batch(f.id, "/v1/chat/completions", "24h")
    await sb1.wait_for_batch(batch.id, timeout=10)
    await sb1.shutdown()

    # Re-instantiate — should load from disk
    sb2 = OpenAIServingBatches(
        storage_dir=storage_dir,
        serving_files=serving_files,
        serving_chat=_make_mock_chat_handler(),
        serving_embedding=None,
        serving_score=None,
        batch_priority=0,
        retention_hours=24,
    )
    retrieved = await sb2.get_batch(batch.id)
    assert retrieved is not None
    assert retrieved.status == "completed"
    await sb2.shutdown()


@pytest.mark.asyncio
async def test_crash_recovery_marks_in_progress_as_failed(
        storage_dir, serving_files):
    """Simulate a crash by writing metadata with in_progress batch."""
    metadata_dir = os.path.join(storage_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    batch_data = [{
        "id": "batch-crashed",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-x",
        "status": "in_progress",
        "completion_window": "24h",
        "created_at": int(time.time()),
        "request_counts": {"total": 10, "completed": 3, "failed": 0},
    }]
    with open(os.path.join(metadata_dir, "batches.json"), "w") as f:
        json.dump(batch_data, f)

    sb = OpenAIServingBatches(
        storage_dir=storage_dir,
        serving_files=serving_files,
        serving_chat=None,
        serving_embedding=None,
        serving_score=None,
        batch_priority=0,
        retention_hours=24,
    )
    recovered = await sb.get_batch("batch-crashed")
    assert recovered.status == "failed"
    assert recovered.failed_at is not None
    await sb.shutdown()
