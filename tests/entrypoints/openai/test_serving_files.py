"""Tests for OpenAIServingFiles."""
import os
import tempfile

import pytest

from vllm.entrypoints.openai.batch.protocol import (
    FileDeleteResponse,
    FileObject,
)
from vllm.entrypoints.openai.serving_files import OpenAIServingFiles


@pytest.fixture
def storage_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def serving_files(storage_dir):
    return OpenAIServingFiles(storage_dir=storage_dir)


@pytest.mark.asyncio
async def test_upload_file(serving_files, storage_dir):
    content = (b'{"custom_id": "r1", "method": "POST", '
               b'"url": "/v1/chat/completions", "body": {}}\n')
    result = await serving_files.upload_file(
        content=content,
        filename="batch.jsonl",
        purpose="batch",
    )
    assert isinstance(result, FileObject)
    assert result.id.startswith("file-")
    assert result.filename == "batch.jsonl"
    assert result.purpose == "batch"
    assert result.bytes == len(content)
    assert os.path.exists(
        os.path.join(storage_dir, "files", result.id + ".jsonl"))


@pytest.mark.asyncio
async def test_list_files(serving_files):
    await serving_files.upload_file(b"line1\n", "a.jsonl", "batch")
    await serving_files.upload_file(b"line2\n", "b.jsonl", "batch_output")
    all_files = await serving_files.list_files()
    assert len(all_files) == 2
    batch_only = await serving_files.list_files(purpose="batch")
    assert len(batch_only) == 1
    assert batch_only[0].filename == "a.jsonl"


@pytest.mark.asyncio
async def test_get_file(serving_files):
    uploaded = await serving_files.upload_file(b"data\n", "f.jsonl", "batch")
    retrieved = await serving_files.get_file(uploaded.id)
    assert retrieved is not None
    assert retrieved.id == uploaded.id


@pytest.mark.asyncio
async def test_get_file_not_found(serving_files):
    result = await serving_files.get_file("file-nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_get_file_content(serving_files):
    content = b"hello world\n"
    uploaded = await serving_files.upload_file(content, "f.jsonl", "batch")
    retrieved_content = await serving_files.get_file_content(uploaded.id)
    assert retrieved_content == content


@pytest.mark.asyncio
async def test_get_file_content_not_found(serving_files):
    result = await serving_files.get_file_content("file-nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_delete_file(serving_files, storage_dir):
    uploaded = await serving_files.upload_file(b"data\n", "f.jsonl", "batch")
    result = await serving_files.delete_file(uploaded.id)
    assert isinstance(result, FileDeleteResponse)
    assert result.id == uploaded.id
    assert result.deleted is True
    assert await serving_files.get_file(uploaded.id) is None
    assert not os.path.exists(
        os.path.join(storage_dir, "files", uploaded.id + ".jsonl"))


@pytest.mark.asyncio
async def test_delete_file_not_found(serving_files):
    result = await serving_files.delete_file("file-nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_delete_file_in_active_batch(serving_files):
    uploaded = await serving_files.upload_file(b"data\n", "f.jsonl", "batch")

    class _BatchHandler:

        def is_file_in_active_batch(self, file_id):
            return file_id == uploaded.id

    result = await serving_files.delete_file(uploaded.id, _BatchHandler())
    assert isinstance(result, dict)
    assert result["error"]["code"] == 409
    # File must survive the rejected deletion.
    assert await serving_files.get_file(uploaded.id) is not None


@pytest.mark.asyncio
async def test_metadata_persistence(storage_dir):
    """Files survive re-instantiation (loaded from disk)."""
    sf1 = OpenAIServingFiles(storage_dir=storage_dir)
    uploaded = await sf1.upload_file(b"data\n", "f.jsonl", "batch")

    sf2 = OpenAIServingFiles(storage_dir=storage_dir)
    retrieved = await sf2.get_file(uploaded.id)
    assert retrieved is not None
    assert retrieved.id == uploaded.id
