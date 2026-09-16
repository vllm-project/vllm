# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import sqlite3
import time
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import pytest

from vllm.entrypoints.openai.responses.store.crypto import DataDecryptionError
from vllm.entrypoints.openai.responses.store.service import ResponsesStoreService


def _write_key_file(path: Path, key: bytes) -> None:
    path.write_bytes(base64.b64encode(key))


def _store_args(db_path: Path, key_path: Path | None) -> Namespace:
    return Namespace(
        responses_store_disk_path=str(db_path),
        responses_store_key_file=None if key_path is None else str(key_path),
        responses_store_config={
            "enabled": True,
            "disk_enabled": True,
            "memory_capacity_mb": 1,
            "disk_capacity_mb": 2,
            "memory_low_watermark": 0.5,
            "memory_high_watermark": 0.8,
            "disk_low_watermark": 0.6,
            "disk_high_watermark": 0.9,
            "memory_ttl_seconds": 30,
            "disk_ttl_seconds": 60,
            "cleanup_interval_seconds": 60,
            "cleanup_max_candidates": 8,
            "cleanup_max_bytes_mb": 1,
            "num_shards": 4,
            "disk_write_interval_seconds": 0.01,
        },
    )


async def _wait_for_disk_copy(service: ResponsesStoreService, session_id: str) -> None:
    disk_store = service.store.disk_store
    assert disk_store is not None
    for _ in range(100):
        if await disk_store.is_complete(session_id):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("session was not persisted before the test deadline")


@pytest.mark.asyncio
async def test_restart_clears_sessions_but_preserves_key_metadata(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "responses.sqlite3"
    key_path = tmp_path / "responses.key"
    _write_key_file(key_path, b"a" * 32)
    args = _store_args(db_path, key_path)

    first_service = ResponsesStoreService.from_cli_args(args)
    first_service.start()
    assert first_service.key_rotation is not None
    assert first_service.key_rotation.is_running
    await first_service.store.save("session-1", "response-1", [10, 20])
    await _wait_for_disk_copy(first_service, "session-1")
    await first_service.close()

    with sqlite3.connect(db_path) as connection:
        original_metadata = connection.execute(
            """
            SELECT schema_version, key_check, key_rotated_at
            FROM store_metadata WHERE singleton_id = 1
            """
        ).fetchone()
        assert connection.execute("SELECT COUNT(*) FROM session_state").fetchone() == (
            1,
        )

    restarted_service = ResponsesStoreService.from_cli_args(args)
    assert await restarted_service.store.get("session-1") is None
    await restarted_service.close()

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM session_state").fetchone() == (
            0,
        )
        current_metadata = connection.execute(
            """
            SELECT schema_version, key_check, key_rotated_at
            FROM store_metadata WHERE singleton_id = 1
            """
        ).fetchone()
    assert current_metadata == original_metadata


@pytest.mark.asyncio
async def test_wrong_key_fails_before_session_table_is_cleared(tmp_path: Path) -> None:
    db_path = tmp_path / "responses.sqlite3"
    key_path = tmp_path / "responses.key"
    original_key = b"a" * 32
    _write_key_file(key_path, original_key)
    args = _store_args(db_path, key_path)

    service = ResponsesStoreService.from_cli_args(args)
    await service.store.save("session-1", "response-1", [10, 20])
    await _wait_for_disk_copy(service, "session-1")
    await service.close()

    _write_key_file(key_path, b"b" * 32)
    with pytest.raises(ValueError, match="key does not match"):
        ResponsesStoreService.from_cli_args(args)
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM session_state").fetchone() == (
            1,
        )

    _write_key_file(key_path, original_key)
    restarted_service = ResponsesStoreService.from_cli_args(args)
    assert await restarted_service.store.get("session-1") is None
    await restarted_service.close()


@pytest.mark.asyncio
async def test_ephemeral_mode_refuses_persistent_key_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "responses.sqlite3"
    key_path = tmp_path / "responses.key"
    _write_key_file(key_path, b"a" * 32)

    service = ResponsesStoreService.from_cli_args(_store_args(db_path, key_path))
    await service.close()

    with pytest.raises(RuntimeError, match="Refusing to reset"):
        ResponsesStoreService.from_cli_args(_store_args(db_path, None))


@pytest.mark.asyncio
async def test_database_key_rotates_atomically_after_ninety_days(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "responses.sqlite3"
    key_path = tmp_path / "responses.key"
    _write_key_file(key_path, b"a" * 32)
    service = ResponsesStoreService.from_cli_args(_store_args(db_path, key_path))
    await service.store.save("session-1", "response-1", [10, 20])
    await _wait_for_disk_copy(service, "session-1")

    disk_store = service.store.disk_store
    assert disk_store is not None
    before_rotation = key_path.read_bytes()
    assert not await disk_store.rotate_key_if_due(int(time.time()))
    assert key_path.read_bytes() == before_rotation

    rotation_time = int(time.time()) + disk_store.KEY_ROTATION_INTERVAL_SECONDS + 1
    assert await disk_store.rotate_key_if_due(rotation_time)
    assert key_path.read_bytes() != before_rotation
    assert not key_path.with_name(f"{key_path.name}.pending").exists()
    rotated_key_file = key_path.read_bytes()
    assert not await disk_store.rotate_key_if_due(rotation_time)
    assert key_path.read_bytes() == rotated_key_file

    await service.store.memory_store.delete("session-1")
    assert await service.store.get("session-1") == [10, 20]
    await service.close()


@pytest.mark.asyncio
async def test_committed_rotation_recovers_pending_key_but_clears_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "responses.sqlite3"
    key_path = tmp_path / "responses.key"
    original_key_file = base64.b64encode(b"a" * 32)
    key_path.write_bytes(original_key_file)
    args = _store_args(db_path, key_path)

    service = ResponsesStoreService.from_cli_args(args)
    await service.store.save("session-1", "response-1", [10, 20])
    await _wait_for_disk_copy(service, "session-1")
    disk_store = service.store.disk_store
    assert disk_store is not None
    key_provider = disk_store._key_file_provider
    assert key_provider is not None
    monkeypatch.setattr(
        key_provider,
        "promote_pending_key",
        Mock(side_effect=RuntimeError("interrupted promotion")),
    )

    rotation_time = int(time.time()) + disk_store.KEY_ROTATION_INTERVAL_SECONDS + 1
    with pytest.raises(RuntimeError, match="interrupted promotion"):
        await disk_store.rotate_key_if_due(rotation_time)
    assert key_path.read_bytes() == original_key_file
    assert key_path.with_name(f"{key_path.name}.pending").exists()
    await service.close()

    restarted_service = ResponsesStoreService.from_cli_args(args)
    assert await restarted_service.store.get("session-1") is None
    assert key_path.read_bytes() != original_key_file
    assert not key_path.with_name(f"{key_path.name}.pending").exists()
    await restarted_service.close()


@pytest.mark.asyncio
async def test_restart_migrates_key_metadata_and_clears_legacy_sessions(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "responses.sqlite3"
    key_path = tmp_path / "responses.key"
    _write_key_file(key_path, b"a" * 32)
    args = _store_args(db_path, key_path)

    service = ResponsesStoreService.from_cli_args(args)
    await service.store.save("session-1", "response-1", [10, 20])
    await _wait_for_disk_copy(service, "session-1")
    await service.close()

    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            ALTER TABLE store_metadata RENAME TO store_metadata_v2;
            CREATE TABLE store_metadata (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                schema_version INTEGER NOT NULL,
                key_check BLOB NOT NULL
            );
            INSERT INTO store_metadata (singleton_id, schema_version, key_check)
            SELECT singleton_id, 1, key_check FROM store_metadata_v2;
            DROP TABLE store_metadata_v2;
            """
        )

    restarted_service = ResponsesStoreService.from_cli_args(args)
    assert await restarted_service.store.get("session-1") is None
    await restarted_service.close()

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT schema_version, key_rotated_at
            FROM store_metadata WHERE singleton_id = 1
            """
        ).fetchone()
        session_count = connection.execute(
            "SELECT COUNT(*) FROM session_state"
        ).fetchone()
    assert row is not None
    assert row[0] == 2
    assert row[1] is not None
    assert session_count == (0,)


@pytest.mark.asyncio
async def test_key_rotation_rolls_back_if_any_session_is_corrupt(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "responses.sqlite3"
    key_path = tmp_path / "responses.key"
    _write_key_file(key_path, b"a" * 32)
    service = ResponsesStoreService.from_cli_args(_store_args(db_path, key_path))
    await service.store.save("session-1", "response-1", [10, 20])
    await _wait_for_disk_copy(service, "session-1")

    original_key_file = key_path.read_bytes()
    with sqlite3.connect(db_path) as connection:
        original_key_check = connection.execute(
            "SELECT key_check FROM store_metadata WHERE singleton_id = 1"
        ).fetchone()
        connection.execute(
            "UPDATE session_state SET token_ids = ? WHERE session_id = ?",
            (b"corrupt", "session-1"),
        )

    disk_store = service.store.disk_store
    assert disk_store is not None
    rotation_time = int(time.time()) + disk_store.KEY_ROTATION_INTERVAL_SECONDS + 1
    with pytest.raises(DataDecryptionError, match="Unable to rotate corrupted"):
        await disk_store.rotate_key_if_due(rotation_time)

    assert key_path.read_bytes() == original_key_file
    assert not key_path.with_name(f"{key_path.name}.pending").exists()
    with sqlite3.connect(db_path) as connection:
        current_key_check = connection.execute(
            "SELECT key_check FROM store_metadata WHERE singleton_id = 1"
        ).fetchone()
    assert current_key_check == original_key_check
    await service.close()
