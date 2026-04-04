# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import regex as re

from vllm.config import FileUploadConfig
from vllm.entrypoints.openai.files.store import (
    ConcurrencyLimitExceeded,
    FileTooLarge,
    FileUploadStore,
    InvalidMime,
    QuotaExceeded,
    new_file_id,
    sanitize_filename,
)

# Minimal real signatures — enough for the MIME sniffer to classify. The
# store writes these as the full file content, so sizes are tiny.
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1024  # ~1KiB PNG-shaped
_MP4_BYTES = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 2048  # ~2KiB MP4
_TEXT_BYTES = b"this is definitely not a media file" + b"\x00" * 128


async def _stream(data: bytes, chunk: int = 256) -> AsyncIterator[bytes]:
    for i in range(0, len(data), chunk):
        yield data[i : i + chunk]


def _mk_config(tmp_path: Path, **overrides: object) -> FileUploadConfig:
    defaults: dict[str, object] = {
        "enabled": True,
        "dir": str(tmp_path / "uploads"),
        "ttl_seconds": 3600,
        "max_size_mb": 1,  # tight for test control
        "max_total_gb": 1,
        "max_concurrent": 4,
        "scope_header": "",
        "disable_listing": False,
    }
    defaults.update(overrides)
    return FileUploadConfig(**defaults)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_file_ids_are_128_bit_random():
    """Security invariant: 128-bit capability handles, never colliding."""
    ids = {new_file_id() for _ in range(10_000)}
    assert len(ids) == 10_000
    pattern = re.compile(r"^file-[0-9a-f]{32}$")
    for file_id in ids:
        assert pattern.match(file_id), file_id
        # 32 hex chars = 128 bits; "file-" prefix = 5 chars; total = 37.
        assert len(file_id) == 37


def test_filename_sanitization_drops_path_components():
    assert sanitize_filename("../etc/passwd/video.mp4") == "video.mp4"
    assert sanitize_filename("/absolute/path/to/clip.webm") == "clip.webm"
    assert sanitize_filename(r"C:\Windows\System32\evil.mp4") == "evil.mp4"
    assert sanitize_filename("./a/b/c/d/e.png") == "e.png"


def test_filename_sanitization_caps_length():
    long = "a" * 500 + ".mp4"
    result = sanitize_filename(long)
    assert len(result.encode("utf-8")) <= 255


def test_filename_sanitization_strips_control_chars():
    assert sanitize_filename("video\x00\r\n.mp4") == "video.mp4"
    assert sanitize_filename("x\x1bo.png") == "xo.png"


def test_filename_sanitization_empty_input_returns_placeholder():
    assert sanitize_filename("") == "upload.bin"
    assert sanitize_filename("\x00\x00") == "upload.bin"


def test_filename_sanitization_preserves_unicode():
    assert sanitize_filename("vidéo.mp4") == "vidéo.mp4"


# ---------------------------------------------------------------------------
# upload + retrieve round trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_get_delete_round_trip(tmp_path):
    store = FileUploadStore(_mk_config(tmp_path))
    rec = await store.create_file(
        stream=_stream(_PNG_BYTES),
        filename="cat.png",
        purpose="vision",
        scope=None,
    )
    assert rec.filename == "cat.png"
    assert rec.mime_type == "image/png"
    assert rec.bytes == len(_PNG_BYTES)
    assert rec.on_disk.exists()

    fetched = store.get(rec.id, scope=None)
    assert fetched is not None
    assert fetched.id == rec.id

    deleted = await store.delete(rec.id, scope=None)
    assert deleted is True
    assert store.get(rec.id, scope=None) is None
    assert not rec.on_disk.exists()


@pytest.mark.asyncio
async def test_delete_returns_false_for_unknown_id(tmp_path):
    store = FileUploadStore(_mk_config(tmp_path))
    assert await store.delete("file-" + "0" * 32, scope=None) is False


# ---------------------------------------------------------------------------
# streaming limits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_oversize_upload(tmp_path):
    # max_size_mb=1 → 1 MiB cap. Send 2 MiB.
    store = FileUploadStore(_mk_config(tmp_path, max_size_mb=1))
    big = _PNG_BYTES + b"\x00" * (2 * 1024 * 1024)
    with pytest.raises(FileTooLarge):
        await store.create_file(
            stream=_stream(big),
            filename="big.png",
            purpose="vision",
            scope=None,
        )
    # No orphan file on disk.
    # Only the startup marker file should remain — no orphaned upload bytes.
    assert [p.name for p in Path(store._dir).iterdir()] == [".vllm-upload-store"]  # noqa: SLF001


@pytest.mark.asyncio
async def test_rejects_non_media_mime(tmp_path):
    store = FileUploadStore(_mk_config(tmp_path))
    with pytest.raises(InvalidMime):
        await store.create_file(
            stream=_stream(_TEXT_BYTES),
            filename="note.txt",
            purpose="user_data",
            scope=None,
        )
    # Only the startup marker file should remain — no orphaned upload bytes.
    assert [p.name for p in Path(store._dir).iterdir()] == [".vllm-upload-store"]  # noqa: SLF001


# ---------------------------------------------------------------------------
# scoping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scope_mismatch_returns_none(tmp_path):
    """Capability non-disclosure: a file uploaded under scope A is invisible
    to scope B (and to unscoped callers)."""
    store = FileUploadStore(_mk_config(tmp_path))
    rec = await store.create_file(
        stream=_stream(_PNG_BYTES),
        filename="a.png",
        purpose="vision",
        scope="team-alpha",
    )
    assert store.get(rec.id, scope="team-alpha") is not None
    assert store.get(rec.id, scope="team-bravo") is None
    assert store.get(rec.id, scope=None) is None


@pytest.mark.asyncio
async def test_list_is_scope_filtered(tmp_path):
    store = FileUploadStore(_mk_config(tmp_path))
    await store.create_file(
        stream=_stream(_PNG_BYTES), filename="a", purpose="vision", scope="a"
    )
    await store.create_file(
        stream=_stream(_PNG_BYTES), filename="b", purpose="vision", scope="a"
    )
    await store.create_file(
        stream=_stream(_PNG_BYTES), filename="c", purpose="vision", scope="b"
    )
    assert len(store.list(scope="a")) == 2
    assert len(store.list(scope="b")) == 1
    assert len(store.list(scope="c")) == 0
    assert len(store.list(scope=None)) == 0


@pytest.mark.asyncio
async def test_delete_requires_matching_scope(tmp_path):
    store = FileUploadStore(_mk_config(tmp_path))
    rec = await store.create_file(
        stream=_stream(_PNG_BYTES), filename="a", purpose="vision", scope="x"
    )
    # Wrong scope cannot delete.
    assert await store.delete(rec.id, scope="y") is False
    assert store.get(rec.id, scope="x") is not None
    # Correct scope can.
    assert await store.delete(rec.id, scope="x") is True


# ---------------------------------------------------------------------------
# quota / LRU eviction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quota_eviction_drops_oldest_lru(tmp_path, monkeypatch):
    # 3MiB quota, 1MiB cap. Upload 3 files of ~1MiB, then a 4th — oldest
    # must be evicted.
    config = _mk_config(tmp_path, max_size_mb=1, max_total_gb=1)
    # Shrink the total-bytes budget under the test's control by monkeypatching.
    store = FileUploadStore(config)
    store._max_total_bytes = 3 * (len(_MP4_BYTES) + 2)  # noqa: SLF001

    recs = []
    for i in range(3):
        recs.append(
            await store.create_file(
                stream=_stream(_MP4_BYTES),
                filename=f"clip{i}.mp4",
                purpose="vision",
                scope=None,
            )
        )
        # Advance last_accessed ordering by touching time.
        await asyncio.sleep(0.01)

    # 4th upload should evict the oldest (recs[0]).
    await store.create_file(
        stream=_stream(_MP4_BYTES),
        filename="clip3.mp4",
        purpose="vision",
        scope=None,
    )
    assert store.get(recs[0].id, scope=None) is None, "oldest should be evicted"
    assert store.get(recs[1].id, scope=None) is not None
    assert store.get(recs[2].id, scope=None) is not None
    assert not recs[0].on_disk.exists(), "on-disk bytes unlinked after eviction"


@pytest.mark.asyncio
async def test_eviction_does_not_block_concurrent_reads(tmp_path):
    """Two-phase eviction: while the unlink runs outside the lock, other
    callers must still be able to `get()` non-evicted files. This verifies
    the snapshot-then-unlink pattern preserves read availability."""
    config = _mk_config(tmp_path, max_size_mb=1, max_total_gb=1)
    store = FileUploadStore(config)
    store._max_total_bytes = 2 * (len(_MP4_BYTES) + 2)  # noqa: SLF001

    # Populate two records, then race an eviction against concurrent gets.
    a = await store.create_file(
        stream=_stream(_MP4_BYTES), filename="a.mp4", purpose="vision", scope=None
    )
    await asyncio.sleep(0.01)
    b = await store.create_file(
        stream=_stream(_MP4_BYTES), filename="b.mp4", purpose="vision", scope=None
    )

    # Start the upload that will evict `a`, and concurrently call get(b).
    async def upload_that_evicts():
        return await store.create_file(
            stream=_stream(_MP4_BYTES),
            filename="c.mp4",
            purpose="vision",
            scope=None,
        )

    async def read_survivor():
        # Poll rapidly during the upload so we land inside the critical path.
        results = []
        for _ in range(50):
            results.append(store.get(b.id, scope=None))
            await asyncio.sleep(0)
        return results

    upload_task, read_task = await asyncio.gather(upload_that_evicts(), read_survivor())

    # The surviving record `b` must never have returned None during the
    # eviction — proof that reads are not blocked by the eviction unlink.
    assert all(r is not None for r in read_task), "reads blocked during eviction"
    assert store.get(a.id, scope=None) is None, "oldest should have been evicted"
    assert store.get(upload_task.id, scope=None) is not None


@pytest.mark.asyncio
async def test_upload_alone_exceeds_quota_is_rejected(tmp_path):
    store = FileUploadStore(_mk_config(tmp_path))
    store._max_total_bytes = 512  # noqa: SLF001 — smaller than one upload
    with pytest.raises(QuotaExceeded):
        await store.create_file(
            stream=_stream(_MP4_BYTES),
            filename="clip.mp4",
            purpose="vision",
            scope=None,
        )


# ---------------------------------------------------------------------------
# TTL + touching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_touches_last_accessed(tmp_path):
    store = FileUploadStore(_mk_config(tmp_path))
    rec = await store.create_file(
        stream=_stream(_PNG_BYTES), filename="a", purpose="vision", scope=None
    )
    original = rec.last_accessed
    await asyncio.sleep(0.02)
    store.get(rec.id, scope=None)
    assert rec.last_accessed > original


@pytest.mark.asyncio
async def test_sweeper_evicts_expired(tmp_path):
    # TTL of 0.05 seconds; manually invoke sweeper (no real sleep loop).
    store = FileUploadStore(_mk_config(tmp_path))
    store._ttl_seconds = 0  # noqa: SLF001 — everything expires immediately
    rec = await store.create_file(
        stream=_stream(_PNG_BYTES), filename="a", purpose="vision", scope=None
    )
    # Nudge last_accessed into the past.
    rec.last_accessed = time.time() - 10
    await store._sweep_expired()  # noqa: SLF001
    assert store.get(rec.id, scope=None) is None
    assert not rec.on_disk.exists()


@pytest.mark.asyncio
async def test_ttl_disabled_skips_sweeper(tmp_path):
    store = FileUploadStore(_mk_config(tmp_path, ttl_seconds=-1))
    rec = await store.create_file(
        stream=_stream(_PNG_BYTES), filename="a", purpose="vision", scope=None
    )
    rec.last_accessed = time.time() - 10_000
    await store._sweep_expired()  # noqa: SLF001
    assert store.get(rec.id, scope=None) is not None  # still present


# ---------------------------------------------------------------------------
# security invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_disk_name_does_not_expose_file_id(tmp_path):
    store = FileUploadStore(_mk_config(tmp_path))
    rec = await store.create_file(
        stream=_stream(_PNG_BYTES),
        filename="secret.png",
        purpose="vision",
        scope=None,
    )
    # The on-disk name is a sha256 of the id — never the id or filename.
    disk_name = rec.on_disk.name
    assert rec.id not in disk_name
    assert "secret" not in disk_name
    assert len(disk_name) == 64  # sha256 hex


@pytest.mark.asyncio
async def test_upload_dir_is_wiped_on_startup(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    # Drop the marker first, simulating a prior store run (the safety
    # check refuses to rmtree unmarked user directories).
    (upload_dir / ".vllm-upload-store").touch()
    (upload_dir / "leftover").write_bytes(b"previous server run")
    store = FileUploadStore(_mk_config(tmp_path, dir=str(upload_dir)))
    assert not (upload_dir / "leftover").exists()
    assert store._dir == upload_dir  # noqa: SLF001
    # Marker is re-dropped so subsequent restarts still work.
    assert (upload_dir / ".vllm-upload-store").exists()


def test_startup_refuses_to_wipe_unmarked_user_dir(tmp_path):
    """Safety: if an operator points --file-upload-dir at a non-empty
    directory that was NOT previously initialised by the store (no
    marker file), refuse to rmtree it rather than silently destroying
    unrelated data."""
    upload_dir = tmp_path / "important_user_data"
    upload_dir.mkdir()
    (upload_dir / "do_not_delete.txt").write_bytes(b"important")
    with pytest.raises(ValueError, match="Refusing to clear"):
        FileUploadStore(_mk_config(tmp_path, dir=str(upload_dir)))
    # User's data is still there.
    assert (upload_dir / "do_not_delete.txt").exists()


# ---------------------------------------------------------------------------
# audit logging
# ---------------------------------------------------------------------------


class _ListHandler(logging.Handler):
    """vllm's loggers set propagate=False, so pytest's caplog can't see
    their records. Attach this handler directly to the store's logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def audit_records():
    store_logger = logging.getLogger("vllm.entrypoints.openai.files.store")
    handler = _ListHandler()
    store_logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        store_logger.removeHandler(handler)


def _parse_audit(records: list[logging.LogRecord]) -> list[dict]:
    out = []
    for r in records:
        try:
            out.append(json.loads(r.getMessage()))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


@pytest.mark.asyncio
async def test_audit_log_emits_on_every_operation_with_required_fields(
    tmp_path, audit_records
):
    store = FileUploadStore(_mk_config(tmp_path))

    rec = await store.create_file(
        stream=_stream(_PNG_BYTES),
        filename="a.png",
        purpose="vision",
        scope="team",
        client_host="203.0.113.7",
        request_id="req-123",
    )
    # Retrieve + delete emit audit lines too.
    async for _ in await store.stream_content(
        rec.id, scope="team", client_host="203.0.113.7", request_id="req-124"
    ):
        pass
    await store.delete(
        rec.id, scope="team", client_host="203.0.113.7", request_id="req-125"
    )

    required = {"op", "file_id", "bytes", "scope", "client_host", "request_id", "ts"}
    payloads = _parse_audit(audit_records)
    ops_seen = [p["op"] for p in payloads]
    for p in payloads:
        assert required <= set(p), f"missing fields in {p}"
    assert ops_seen == ["file.upload", "file.retrieve", "file.delete"]


@pytest.mark.asyncio
async def test_audit_log_includes_reason_on_rejection(tmp_path, audit_records):
    store = FileUploadStore(_mk_config(tmp_path))
    with pytest.raises(InvalidMime):
        await store.create_file(
            stream=_stream(_TEXT_BYTES),
            filename="note.txt",
            purpose="user_data",
            scope=None,
            client_host="203.0.113.7",
        )
    rejects = [p for p in _parse_audit(audit_records) if p.get("op") == "file.reject"]
    assert len(rejects) == 1
    assert rejects[0]["reason"] == "mime_mismatch"


# ---------------------------------------------------------------------------
# concurrency + TOCTOU race handling (Copilot review fixes)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_rejects_when_concurrent_limit_hit(tmp_path):
    """Fail-fast contract: when `max_concurrent` uploads are already in
    flight, a new upload raises ConcurrencyLimitExceeded rather than
    queueing indefinitely (surfaced to the API layer as 503)."""
    # Semaphore capacity 1 makes the saturation point deterministic.
    store = FileUploadStore(_mk_config(tmp_path, max_concurrent=1))

    # Slow-stream that never completes until the test cancels it.
    started = asyncio.Event()
    never_finishes = asyncio.Event()

    async def _hang() -> AsyncIterator[bytes]:
        yield _PNG_BYTES
        started.set()
        await never_finishes.wait()
        yield b""

    first = asyncio.create_task(
        store.create_file(
            stream=_hang(), filename="slow.png", purpose="vision", scope=None
        )
    )
    await started.wait()

    # Second upload must be rejected immediately — not queued.
    with pytest.raises(ConcurrencyLimitExceeded):
        await store.create_file(
            stream=_stream(_PNG_BYTES),
            filename="x.png",
            purpose="vision",
            scope=None,
        )

    never_finishes.set()
    await first  # let the slow upload drain so pytest doesn't warn


@pytest.mark.asyncio
async def test_read_bytes_by_id_handles_concurrent_eviction(tmp_path):
    """TOCTOU: the record may be observed in `_records` but its on-disk
    file could be unlinked before `read_bytes()` runs (two-phase delete
    releases the lock between dict drop and unlink). The read must
    return None cleanly rather than propagating FileNotFoundError."""
    store = FileUploadStore(_mk_config(tmp_path))
    rec = await store.create_file(
        stream=_stream(_PNG_BYTES), filename="a.png", purpose="vision", scope=None
    )

    # Manually unlink while the record still exists in memory — simulates
    # the narrow window between metadata drop and actual unlink.
    rec.on_disk.unlink()
    assert rec.id in store._records  # noqa: SLF001 — invariant we're probing

    # Sync variant: returns None gracefully.
    assert store.read_bytes_by_id(rec.id) is None
    # Async variant: also returns None gracefully.
    assert await store.read_bytes_by_id_async(rec.id) is None


@pytest.mark.asyncio
async def test_stream_content_fails_fast_when_file_evicted_before_open(tmp_path):
    """TOCTOU: if the file is unlinked between `get()` and `open()`,
    `stream_content` must raise FileNotFoundError from the call itself,
    BEFORE returning the iterator — otherwise a 200 response starts
    streaming and then breaks mid-flight."""
    store = FileUploadStore(_mk_config(tmp_path))
    rec = await store.create_file(
        stream=_stream(_PNG_BYTES), filename="a.png", purpose="vision", scope=None
    )

    # Unlink the bytes without dropping the metadata record.
    rec.on_disk.unlink()

    with pytest.raises(FileNotFoundError):
        await store.stream_content(rec.id, scope=None)


@pytest.mark.asyncio
async def test_read_bytes_by_id_async_does_not_race_with_deletes(tmp_path):
    """Metadata access happens on the event-loop thread (only the disk
    read is dispatched to an executor), so concurrent deletes mutating
    `_records` cannot race with the lookup."""
    store = FileUploadStore(_mk_config(tmp_path))
    rec = await store.create_file(
        stream=_stream(_PNG_BYTES), filename="a.png", purpose="vision", scope=None
    )

    # Happy path: async read returns the bytes.
    data = await store.read_bytes_by_id_async(rec.id)
    assert data == _PNG_BYTES

    # Delete, then async read must return None (not raise).
    assert await store.delete(rec.id, scope=None) is True
    assert await store.read_bytes_by_id_async(rec.id) is None
