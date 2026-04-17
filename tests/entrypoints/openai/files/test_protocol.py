# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.files.protocol import (
    FileDeleteResponse,
    FileList,
    FileObject,
)


def test_file_object_shape_matches_openai_files_api():
    obj = FileObject(
        id="file-abc123",
        bytes=18432100,
        created_at=1743800000,
        expires_at=1743803600,
        filename="video.mp4",
        purpose="vision",
    )
    data = json.loads(obj.model_dump_json())
    assert data == {
        "id": "file-abc123",
        "object": "file",
        "bytes": 18432100,
        "created_at": 1743800000,
        "expires_at": 1743803600,
        "filename": "video.mp4",
        "purpose": "vision",
        "status": "processed",
        "status_details": None,
    }


def test_file_object_expires_at_omitted_when_none():
    """When TTL is disabled, responses should omit `expires_at` per OpenAI
    convention (caller uses exclude_none=True at serialization time)."""
    obj = FileObject(
        id="file-xyz",
        bytes=100,
        created_at=1,
        filename="x.png",
        purpose="user_data",
        expires_at=None,
    )
    data = json.loads(obj.model_dump_json(exclude_none=True))
    assert "expires_at" not in data
    assert data["purpose"] == "user_data"


@pytest.mark.parametrize(
    "bad_purpose",
    ["assistants", "fine-tune", "batch", "assistants_output", "", "VISION"],
)
def test_file_object_rejects_unsupported_purpose(bad_purpose):
    with pytest.raises(ValidationError):
        FileObject(
            id="file-x",
            bytes=1,
            created_at=1,
            filename="x",
            purpose=bad_purpose,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("good_purpose", ["vision", "user_data"])
def test_file_object_accepts_supported_purposes(good_purpose):
    obj = FileObject(
        id="file-x",
        bytes=1,
        created_at=1,
        filename="x",
        purpose=good_purpose,  # type: ignore[arg-type]
    )
    assert obj.purpose == good_purpose


def test_file_list_default_is_empty():
    fl = FileList()
    assert fl.object == "list"
    assert fl.data == []


def test_file_list_populated():
    obj = FileObject(
        id="file-a",
        bytes=1,
        created_at=1,
        filename="a.mp4",
        purpose="vision",
    )
    fl = FileList(data=[obj])
    assert len(fl.data) == 1
    assert fl.data[0].id == "file-a"


def test_file_delete_response():
    dr = FileDeleteResponse(id="file-abc")
    data = json.loads(dr.model_dump_json())
    assert data == {"id": "file-abc", "object": "file", "deleted": True}
