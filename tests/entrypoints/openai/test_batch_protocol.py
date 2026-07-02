"""Tests for Batch API protocol models."""
import time

from vllm.entrypoints.openai.batch.protocol import (
    BatchError,
    BatchErrors,
    BatchListResponse,
    BatchObject,
    BatchRequestCounts,
    FileListResponse,
    FileObject,
)


def test_file_object_creation():
    f = FileObject(
        id="file-abc123",
        bytes=1024,
        created_at=int(time.time()),
        filename="batch_input.jsonl",
        purpose="batch",
    )
    assert f.object == "file"
    assert f.id == "file-abc123"
    assert f.purpose == "batch"


def test_file_object_allows_batch_purposes():
    for purpose in ("batch", "batch_output", "batch_error"):
        f = FileObject(
            id="file-x",
            bytes=0,
            created_at=0,
            filename="f.jsonl",
            purpose=purpose,
        )
        assert f.purpose == purpose


def test_batch_request_counts_defaults():
    c = BatchRequestCounts(total=10, completed=0, failed=0)
    assert c.total == 10


def test_batch_error_model():
    e = BatchError(code="invalid_request", message="bad", param=None, line=3)
    assert e.code == "invalid_request"
    assert e.line == 3


def test_batch_errors_model():
    errs = BatchErrors(data=[
        BatchError(code="err", message="msg", param=None, line=1),
    ])
    assert errs.object == "list"
    assert len(errs.data) == 1


def test_batch_object_creation():
    now = int(time.time())
    b = BatchObject(
        id="batch-abc",
        endpoint="/v1/chat/completions",
        input_file_id="file-123",
        completion_window="24h",
        status="validating",
        created_at=now,
        request_counts=BatchRequestCounts(total=5, completed=0, failed=0),
    )
    assert b.object == "batch"
    assert b.status == "validating"
    assert b.output_file_id is None
    assert b.error_file_id is None
    assert b.metadata is None
    assert b.expires_at is None


def test_batch_object_with_metadata():
    b = BatchObject(
        id="batch-abc",
        endpoint="/v1/chat/completions",
        input_file_id="file-123",
        completion_window="24h",
        status="validating",
        created_at=0,
        request_counts=BatchRequestCounts(total=1, completed=0, failed=0),
        metadata={"user": "test", "job": "nightly"},
    )
    assert b.metadata == {"user": "test", "job": "nightly"}


def test_batch_list_response():
    r = BatchListResponse(
        data=[],
        has_more=False,
        first_id=None,
        last_id=None,
    )
    assert r.object == "list"
    assert r.data == []


def test_file_list_response():
    r = FileListResponse(data=[])
    assert r.object == "list"
