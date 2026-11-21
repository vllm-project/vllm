# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest

from vllm.entrypoints.openai.files.protocol import FileDeleteResponse
from vllm.entrypoints.openai.files.serving import OpenAIServingFiles


@pytest.fixture
def serving_files(tmp_path):
    return OpenAIServingFiles(storage_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_upload_file(serving_files):
    content = b'{"custom_id": "r1"}\n'
    file_obj = await serving_files.upload_file(content, "batch.jsonl", "batch")

    assert file_obj.id.startswith("file-")
    assert file_obj.filename == "batch.jsonl"
    assert file_obj.purpose == "batch"
    assert file_obj.bytes == len(content)
    assert os.path.exists(serving_files.file_path(file_obj.id))


@pytest.mark.asyncio
async def test_list_files_filters_by_purpose(serving_files):
    await serving_files.upload_file(b"a\n", "a.jsonl", "batch")
    await serving_files.upload_file(b"b\n", "b.jsonl", "batch_output")

    assert len(await serving_files.list_files()) == 2
    batch_files = await serving_files.list_files(purpose="batch")
    assert [f.filename for f in batch_files] == ["a.jsonl"]


@pytest.mark.asyncio
async def test_get_file_and_content(serving_files):
    uploaded = await serving_files.upload_file(b"hello\n", "f.jsonl", "batch")

    file_obj = await serving_files.get_file(uploaded.id)
    assert file_obj is not None
    assert file_obj.id == uploaded.id
    assert await serving_files.get_file_content(uploaded.id) == b"hello\n"


@pytest.mark.asyncio
async def test_missing_file(serving_files):
    assert await serving_files.get_file("file-missing") is None
    assert await serving_files.get_file_content("file-missing") is None
    assert await serving_files.delete_file("file-missing") is None


@pytest.mark.asyncio
async def test_delete_file(serving_files):
    uploaded = await serving_files.upload_file(b"data\n", "f.jsonl", "batch")

    result = await serving_files.delete_file(uploaded.id)

    assert result == FileDeleteResponse(id=uploaded.id)
    assert await serving_files.get_file(uploaded.id) is None
    assert not os.path.exists(serving_files.file_path(uploaded.id))


@pytest.mark.asyncio
async def test_allocate_and_register_file(serving_files):
    file_id, path = serving_files.allocate_file()
    with open(path, "wb") as f:
        f.write(b"result\n")

    assert await serving_files.get_file(file_id) is None
    file_obj = await serving_files.register_file(
        file_id, "out.jsonl", "batch_output", 7
    )
    assert file_obj.bytes == 7
    assert await serving_files.get_file_content(file_id) == b"result\n"


@pytest.mark.asyncio
async def test_metadata_persists_across_instances(tmp_path):
    uploaded = await OpenAIServingFiles(str(tmp_path)).upload_file(
        b"data\n", "f.jsonl", "batch"
    )

    reloaded = OpenAIServingFiles(str(tmp_path))

    assert await reloaded.get_file(uploaded.id) == uploaded
