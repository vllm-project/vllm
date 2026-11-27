# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for client-processed multimodal tensors loaded from S3."""

import asyncio
import gc
import hashlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from botocore.exceptions import ClientError
from safetensors.torch import save as save_safetensors

from vllm.config.mm_tensor_refs import MMTensorRefsConfig
from vllm.exceptions import RetryableRequestError
from vllm.multimodal import s3_transport
from vllm.multimodal.s3_transport import (
    S3MMReader,
    S3MMReference,
)
from vllm.multimodal.tensor_transport import (
    MMTensorMemoryBudget,
    MMTensorTransportError,
    ReleasingStream,
    run_in_read_slot,
    validate_mm_tensor_request_size,
)
from vllm.utils.import_utils import PlaceholderModule

pytestmark = pytest.mark.cpu_test

URI = "s3://tensor-bucket/tenant/mm/request/image-0"


class FakeBody(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.read_sizes: list[int | None] = []

    def read(self, size: int | None = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)


class FakeS3Client:
    def __init__(
        self,
        payload: bytes | None,
        content_length: int | None = None,
        error: Exception | None = None,
    ):
        self.body = None if payload is None else FakeBody(payload)
        self.content_length = content_length
        self.error = error
        self.calls: list[dict[str, str]] = []

    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"Body": self.body, "ContentLength": self.content_length}


def _tensors() -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "num_patches": torch.tensor([3], dtype=torch.int64),
    }


def _reference(
    payload: bytes,
    *,
    uri: str = URI,
    nbytes: int | None = None,
    digest: str | None = None,
) -> S3MMReference:
    return S3MMReference(
        uri=uri,
        format="safetensors-v1",
        nbytes=len(payload) if nbytes is None else nbytes,
        digest=hashlib.sha256(payload).hexdigest() if digest is None else digest,
    )


CONFIG = MMTensorRefsConfig(
    s3_allowed_uri_prefix="s3://tensor-bucket/tenant/mm",
    max_object_bytes=1024**2,
    max_request_bytes=3 * 1024**2,
    max_reserved_bytes=6 * 1024**2,
)


def _reader(client=None, config=CONFIG) -> S3MMReader:
    return S3MMReader(config, object() if client is None else client)


def test_reader_reads_at_most_declared_size_and_decodes():
    expected = _tensors()
    payload = save_safetensors(expected)
    client = FakeS3Client(payload, content_length=len(payload))

    tensors = _reader(client).read(_reference(payload))

    assert client.calls == [
        {"Bucket": "tensor-bucket", "Key": "tenant/mm/request/image-0"}
    ]
    assert client.body is not None
    assert client.body.read_sizes == [len(payload) + 1]
    assert client.body.closed
    assert tensors.keys() == expected.keys()
    assert all(torch.equal(tensors[k], v) for k, v in expected.items())


@pytest.mark.parametrize(
    "uri",
    [
        "s3://other-bucket/tenant/mm/request/image-0",
        "s3://tensor-bucket/other/request/image-0",
        "s3://tensor-bucket/tenant/mm-elsewhere/image-0",
        "s3://tensor-bucket/tenant/mm/../other/image-0",
        "https://tensor-bucket/tenant/mm/request/image-0",
    ],
)
def test_reference_outside_allowed_prefix_is_rejected(uri):
    with pytest.raises(ValueError, match="configured prefix|Invalid S3 URI"):
        _reader().validate(_reference(b"x", uri=uri))


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({}, "s3_allowed_uri_prefix must be set"),
        ({"s3_allowed_uri_prefix": "tensor-bucket/mm"}, "s3_allowed_uri_prefix"),
        (
            {"s3_allowed_uri_prefix": "s3://b/p", "max_reserved_bytes": 2**30},
            "at least twice max_request_bytes",
        ),
    ],
)
def test_config_rejects_bad_settings(fields, match):
    with pytest.raises(ValueError, match=match):
        MMTensorRefsConfig(**fields)


def test_bucket_only_prefix_allows_any_key():
    config = MMTensorRefsConfig(s3_allowed_uri_prefix="s3://tensor-bucket")

    assert _reader(config=config).validate(_reference(b"x", uri=URI)) == (
        "tensor-bucket",
        "tenant/mm/request/image-0",
    )


def test_reference_over_size_limit_is_rejected():
    with pytest.raises(ValueError, match="configured limit"):
        _reader().validate(_reference(b"x", nbytes=1024**2 + 1))


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (404, "NoSuchKey", MMTensorTransportError),
        (503, "SlowDown", RetryableRequestError),
    ],
)
def test_reader_maps_store_errors_by_status(status, code, expected):
    error = ClientError(
        {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "GetObject",
    )
    reader = _reader(FakeS3Client(None, error=error))

    with pytest.raises(expected, match="Could not load"):
        reader.read(_reference(b"x"))


UNAVAILABLE = f"Could not load multimodal tensor object '{URI}'"


@pytest.mark.parametrize(
    ("served", "content_length", "digest"),
    [
        ("missing", None, None),
        ("payload", None, None),
        ("payload", 1, None),
        ("short", "declared", None),
        ("long", "declared", None),
        ("payload", "declared", "0" * 64),
    ],
)
def test_reader_errors_dont_reveal_the_object(served, content_length, digest):
    """Missing and mismatched objects fail alike, without their size or status."""
    payload = save_safetensors(_tensors())
    body = {"payload": payload, "short": payload[:-1], "long": payload + b"x"}
    if served == "missing":
        error = ClientError(
            {
                "Error": {"Code": "NoSuchKey"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "GetObject",
        )
        client = FakeS3Client(None, error=error)
    else:
        length = len(payload) if content_length == "declared" else content_length
        client = FakeS3Client(body[served], content_length=length)

    with pytest.raises(MMTensorTransportError) as excinfo:
        _reader(client).read(_reference(payload, digest=digest))

    assert str(excinfo.value) == UNAVAILABLE
    if client.body is not None:
        assert client.body.closed


def test_reader_without_boto3_fails_at_startup_not_per_request(monkeypatch):
    """Without boto3, a bad reference is still a 400; only the client needs it."""
    monkeypatch.setattr(s3_transport, "boto3", PlaceholderModule("boto3_missing"))
    monkeypatch.setattr(s3_transport, "botocore", PlaceholderModule("botocore_missing"))

    with pytest.raises(ValueError, match="configured prefix"):
        _reader().read(_reference(b"x", uri="s3://other-bucket/k"))
    with pytest.raises(ImportError):
        S3MMReader(CONFIG)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"not safetensors", "Could not decode"),
        (save_safetensors({}), "contains no tensors"),
    ],
)
def test_reader_rejects_bad_payload(payload, match):
    client = FakeS3Client(payload, content_length=len(payload))

    with pytest.raises(MMTensorTransportError, match=match):
        _reader(client).read(_reference(payload))


def test_request_size_limit_counts_every_reference():
    references = [_reference(b"x", nbytes=1024**2) for _ in range(3)]
    assert validate_mm_tensor_request_size(references, 3 * 1024**2) == 3 * 1024**2

    with pytest.raises(ValueError, match="in one request declare 4194304 bytes"):
        validate_mm_tensor_request_size([*references, references[0]], 3 * 1024**2)


@pytest.mark.asyncio
async def test_cancelled_read_keeps_its_slot_until_the_read_finishes():
    """A client disconnect cancels the request, but not the executor thread."""
    slots = asyncio.Semaphore(1)
    unblock = threading.Event()
    started: list[str] = []

    def blocked_read() -> str:
        started.append("first")
        unblock.wait(timeout=30)
        return "first"

    def next_read() -> str:
        started.append("second")
        return "second"

    first = asyncio.create_task(run_in_read_slot(slots, blocked_read))
    async with asyncio.timeout(10):
        while not started:
            await asyncio.sleep(0.01)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    # Still held by the cancelled read, before anyone else asks for it.
    assert slots.locked()

    second = asyncio.create_task(run_in_read_slot(slots, next_read))
    await asyncio.sleep(0)
    assert not second.done()
    assert started == ["first"]

    unblock.set()
    async with asyncio.timeout(10):
        assert await second == "second"
    assert started == ["first", "second"]


@pytest.mark.asyncio
async def test_memory_budget_waits_in_order_and_survives_cancellation():
    budget = MMTensorMemoryBudget(100)
    first = await budget.reserve(60)

    cancelled = asyncio.create_task(budget.reserve(50))
    queued = asyncio.create_task(budget.reserve(40))
    await asyncio.sleep(0)
    assert not cancelled.done() and not queued.done()

    # The waiter at the head leaves; the next one fits and gets its bytes.
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    second = await asyncio.wait_for(queued, timeout=10)
    assert budget.used_nbytes == 100

    first.release()
    second.release()
    assert budget.used_nbytes == 0
    with pytest.raises(ValueError, match="max_reserved_bytes"):
        await budget.reserve(101)


@pytest.mark.asyncio
async def test_reservation_outlives_a_cancelled_read():
    """The read thread keeps its bytes after the request lets go of them."""
    budget = MMTensorMemoryBudget(100)
    reservation = await budget.reserve(100)
    started = threading.Event()
    unblock = threading.Event()

    def blocked_read() -> bool:
        started.set()
        return unblock.wait(timeout=30)

    read = asyncio.create_task(
        run_in_read_slot(asyncio.Semaphore(1), blocked_read, reservation)
    )
    async with asyncio.timeout(10):
        while not started.is_set():
            await asyncio.sleep(0.01)
    read.cancel()
    with pytest.raises(asyncio.CancelledError):
        await read
    reservation.release()
    assert budget.used_nbytes == 100

    unblock.set()
    async with asyncio.timeout(10):
        while budget.used_nbytes:
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("consume", ["fully", "partly_then_close", "never"])
async def test_releasing_stream_returns_its_reservation(consume):
    """Including a stream dropped unread, e.g. after an early disconnect."""
    budget = MMTensorMemoryBudget(10)
    reservation = await budget.reserve(10)

    async def chunks():
        yield "a"
        yield "b"

    stream = ReleasingStream(chunks(), [reservation])
    if consume == "fully":
        assert [chunk async for chunk in stream] == ["a", "b"]
    elif consume == "partly_then_close":
        assert await anext(stream) == "a"
        await stream.aclose()
    else:
        del stream
        gc.collect()

    async with asyncio.timeout(10):
        while budget.used_nbytes:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_reads_run_on_the_given_executor():
    with ThreadPoolExecutor(1, thread_name_prefix="test-read") as executor:
        name = await run_in_read_slot(
            asyncio.Semaphore(1),
            lambda: threading.current_thread().name,
            executor=executor,
        )
    assert name.startswith("test-read")
