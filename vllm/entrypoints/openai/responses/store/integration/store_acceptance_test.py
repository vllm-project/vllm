"""ResponseStore standalone acceptance tests.

Run module tests in the same environment used by ``vllm serve``::

    .venv/bin/python store_acceptance_test.py module --sessions 500

Run only eviction scenarios with INFO logging enabled::

    VLLM_LOGGING_LEVEL=INFO .venv/bin/python -u store_acceptance_test.py \
        module --eviction-only

Eviction details are logged by this test process, not a separate API server.

Run live API tests after the server has started::

    .venv/bin/python store_acceptance_test.py api \
        --base-url http://127.0.0.1:8000 --model MODEL_NAME \
        --sessions 500 --concurrency 10

API sessions are saved as a group before reads and deletion. Each session uses
a distinct verification code. All test sessions are deleted at the end, even
after failures. This checks API behavior; eviction under pressure is not treated
as a successful context hit.

Verify read-triggered memory TTL renewal against idle controls::

    .venv/bin/python store_acceptance_test.py api --sessions 5 --concurrency 5 \
        --verify-ttl-refresh --ttl-seconds 30 --ttl-cleanup-interval 1

For this mode, start a dedicated server with::

    --responses-store-config '{"enabled": true, "disk_enabled": false,
        "memory_ttl_seconds": 30, "cleanup_interval_seconds": 1}'

Give it sufficient memory capacity.
Each --sessions entry creates one active session and one idle control (2N IDs).
This mode replaces the ordinary API phases. Reads use use_store=False; only the
active session is touched. Its original expiry must pass while it remains alive
and the control expires. It must then expire after reads stop. The server's
configuration cannot be inspected through the existing API, so these settings
must match. This test does not isolate disk TTL renewal. Increase TTL or reduce
concurrency if a request takes more than one third of the TTL window.

The script intentionally uses only the Python standard library and the local
vLLM package. It does not require pytest and does not test restart recovery.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import sqlite3
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from vllm.entrypoints.openai.responses.store.cleanup import (
    CleanupRunResult,
    PeriodicCleanupConfig,
    PeriodicSessionStoreCleanup,
    TierCleanupConfig,
)
from vllm.entrypoints.openai.responses.store.disk import SQLiteSessionStore
from vllm.entrypoints.openai.responses.store.eviction import (
    CapacityWaterMarks,
    EvictionSelectionBudget,
)
from vllm.entrypoints.openai.responses.store.memory import MemorySessionStore
from vllm.entrypoints.openai.responses.store.tiered import TieredSessionStore
from vllm.logger import init_logger

logger = init_logger(__name__)

TestCase = Callable[[], Awaitable[None]]


def check(condition: bool, message: str) -> None:
    """Raise a readable failure instead of relying on Python assert."""
    if not condition:
        raise AssertionError(message)


async def wait_until(
    predicate: Callable[[], Awaitable[bool]],
    description: str,
    timeout: float = 10.0,
) -> None:
    """Wait for an asynchronous condition with a bounded timeout."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {description}")


async def wait_for_disk(
    disk: SQLiteSessionStore,
    session_ids: list[str],
    timeout: float = 10.0,
) -> None:
    """Wait until every requested session has a complete disk copy."""

    async def all_complete() -> bool:
        states = await asyncio.gather(
            *(disk.is_complete(session_id) for session_id in session_ids)
        )
        return all(states)

    await wait_until(all_complete, "complete SQLite copies", timeout)


async def test_memory_crud() -> None:
    memory = MemorySessionStore(
        max_capacity_bytes=8 * 1024 * 1024,
        mem_idle_ttl_seconds=60,
    )
    source = [11, 12]
    await memory.save("session-a", "response-1", source)
    source.append(999)
    await memory.save("session-a", "response-2", [13, 14])

    result = await memory.get("session-a")
    check(result == [11, 12, 13, 14], "memory did not append deltas")
    check(result is not None, "saved session was not found")
    result.append(888)
    check(
        await memory.get("session-a") == [11, 12, 13, 14],
        "get() exposed the internal token list",
    )

    snapshots = await memory.list()
    snapshots[0].token_ids.append(777)
    check(
        await memory.get("session-a") == [11, 12, 13, 14],
        "list() exposed the internal token list",
    )
    check(memory.used_bytes > 0, "memory size accounting was not updated")
    check(await memory.exists("session-a"), "exists() missed a saved session")
    check(await memory.delete("session-a"), "delete() reported false")
    check(not await memory.exists("session-a"), "deleted session still exists")
    check(not await memory.delete("session-a"), "second delete should be false")
    check(memory.used_bytes == 0, "memory size accounting did not return to zero")


async def test_tiered_roundtrip_and_encryption() -> None:
    with tempfile.TemporaryDirectory(prefix="responses-store-roundtrip-") as root:
        db_path = str(Path(root) / "store.sqlite3")
        memory = MemorySessionStore(max_capacity_bytes=8 * 1024 * 1024)
        disk = SQLiteSessionStore(
            db_path=db_path,
            disk_idle_ttl_seconds=60,
            write_interval_seconds=0.05,
        )
        store = TieredSessionStore(memory, disk, num_shards=8)
        try:
            first = [100_001, 100_002, 100_003]
            second = [200_001, 200_002]
            await store.save("session-roundtrip", "response-1", first)

            check(
                await disk.get("session-roundtrip") is None,
                "disk exposed a version before the async write completed",
            )
            await store.save("session-roundtrip", "response-2", second)
            await wait_for_disk(disk, ["session-roundtrip"])

            expected = first + second
            check(
                await store.get("session-roundtrip") == expected,
                "tiered store returned incorrect accumulated tokens",
            )
            states = await store.list()
            check(len(states) == 1, "tiered list() returned the wrong row count")
            check(
                states[0].session_id == "session-roundtrip"
                and states[0].response_id == "response-2"
                and states[0].token_ids == expected,
                "tiered list() returned stale session data",
            )
            metadata = await store.list_metadata()
            check(
                len(metadata) == 1
                and metadata[0].mem_size_bytes > 0
                and metadata[0].disk_size_bytes > 0,
                "tiered list_metadata() did not merge both tiers",
            )

            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    "SELECT token_ids FROM session_state WHERE session_id = ?",
                    ("session-roundtrip",),
                ).fetchone()
            check(row is not None, "SQLite row was not created")
            encrypted_blob = bytes(row[0])
            plaintext = array.array("I", expected).tobytes()
            if sys.byteorder != "little":
                values = array.array("I", expected)
                values.byteswap()
                plaintext = values.tobytes()
            check(
                encrypted_blob.startswith(b"VRS1"),
                "token blob does not use the encrypted frame format",
            )
            check(
                plaintext not in encrypted_blob,
                "plaintext token bytes were found in the SQLite token blob",
            )

            check(
                await memory.delete("session-roundtrip"),
                "could not remove the memory copy for restore testing",
            )
            check(
                await store.get("session-roundtrip") == expected,
                "memory miss did not restore the complete SQLite copy",
            )
            check(
                await memory.exists("session-roundtrip"),
                "restored session was not placed back in memory",
            )
        finally:
            await store.close()


async def test_delete_pending_and_recreate() -> None:
    with tempfile.TemporaryDirectory(prefix="responses-store-delete-") as root:
        memory = MemorySessionStore(max_capacity_bytes=8 * 1024 * 1024)
        disk = SQLiteSessionStore(
            db_path=str(Path(root) / "store.sqlite3"),
            write_interval_seconds=0.25,
        )
        store = TieredSessionStore(memory, disk, num_shards=8)
        try:
            session_id = "session-delete"
            await store.save(session_id, "old-response", [1, 2, 3])
            check(await store.delete(session_id), "pending session was not deleted")
            await asyncio.sleep(0.35)
            check(
                not await store.exists(session_id),
                "an in-flight write recreated a deleted session",
            )
            check(await store.get(session_id) is None, "deleted session was readable")

            await store.save(session_id, "new-response", [9, 10])
            await wait_for_disk(disk, [session_id])
            await memory.delete(session_id)
            check(
                await store.get(session_id) == [9, 10],
                "recreated session contains data from before deletion",
            )
            check(await store.delete(session_id), "recreated session delete failed")
            check(
                not await store.delete(session_id),
                "deleting an absent tiered session should return false",
            )
        finally:
            await store.close()


async def test_corrupt_disk_copy_and_full_rerender() -> None:
    with tempfile.TemporaryDirectory(prefix="responses-store-corrupt-") as root:
        db_path = str(Path(root) / "store.sqlite3")
        memory = MemorySessionStore(max_capacity_bytes=8 * 1024 * 1024)
        disk = SQLiteSessionStore(
            db_path=db_path,
            write_interval_seconds=0.02,
        )
        store = TieredSessionStore(memory, disk, num_shards=8)
        session_id = "session-corrupt"
        try:
            await store.save(session_id, "response-old", [1, 2, 3])
            await wait_for_disk(disk, [session_id])

            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    "SELECT token_ids FROM session_state WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                check(row is not None, "SQLite row was not created")
                damaged = bytearray(row[0])
                damaged[-1] ^= 1
                connection.execute(
                    "UPDATE session_state SET token_ids = ? WHERE session_id = ?",
                    (bytes(damaged), session_id),
                )
                connection.commit()

            await memory.delete(session_id)
            check(
                await store.get(session_id) is None,
                "corrupt disk data should produce a context miss",
            )
            check(
                not await store.exists(session_id),
                "corrupt disk data was not invalidated",
            )

            full_history = [101, 102, 103, 104]
            await store.save(session_id, "response-rebuilt", full_history)
            await wait_for_disk(disk, [session_id])
            await memory.delete(session_id)
            check(
                await store.get(session_id) == full_history,
                "full-history rerender did not rebuild the session",
            )
        finally:
            await store.close()


async def test_concurrent_sessions(session_count: int, rounds: int) -> None:
    with tempfile.TemporaryDirectory(prefix="responses-store-concurrency-") as root:
        memory = MemorySessionStore(max_capacity_bytes=256 * 1024 * 1024)
        disk = SQLiteSessionStore(
            db_path=str(Path(root) / "store.sqlite3"),
            disk_idle_ttl_seconds=60,
            write_interval_seconds=0.02,
        )
        store = TieredSessionStore(memory, disk, num_shards=64)
        session_ids = [f"concurrent-{index}" for index in range(session_count)]

        async def write_session(index: int, session_id: str) -> None:
            for round_number in range(rounds):
                await store.save(
                    session_id,
                    f"response-{index}-{round_number}",
                    [index, round_number],
                )

        started = time.perf_counter()
        try:
            await asyncio.gather(
                *(
                    write_session(index, session_id)
                    for index, session_id in enumerate(session_ids)
                )
            )
            await wait_for_disk(disk, session_ids, timeout=30.0)

            expected = [
                value for round_number in range(rounds) for value in (0, round_number)
            ]
            first = await store.get(session_ids[0])
            check(first == expected, "concurrent writes lost or reordered tokens")

            for session_id in session_ids:
                await memory.delete(session_id)
            restored = await asyncio.gather(
                *(store.get(session_id) for session_id in session_ids)
            )
            for index, token_ids in enumerate(restored):
                expected = [
                    value
                    for round_number in range(rounds)
                    for value in (index, round_number)
                ]
                check(
                    token_ids == expected,
                    f"disk restore mismatch for {session_ids[index]}",
                )

            shared_session = "concurrent-shared-session"
            shared_write_count = 100
            await asyncio.gather(
                *(
                    store.save(shared_session, f"shared-{index}", [index])
                    for index in range(shared_write_count)
                )
            )
            await wait_for_disk(disk, [shared_session])
            write_seconds = time.perf_counter() - started
            shared_tokens = await store.get(shared_session)
            check(
                shared_tokens is not None
                and len(shared_tokens) == shared_write_count
                and sorted(shared_tokens) == list(range(shared_write_count)),
                "concurrent writes to one session lost or duplicated tokens",
            )
            await memory.delete(shared_session)
            restored_shared_tokens = await store.get(shared_session)
            check(
                restored_shared_tokens is not None
                and sorted(restored_shared_tokens) == list(range(shared_write_count)),
                "SQLite copy lost same-session concurrent writes",
            )

            operation_count = session_count * rounds + shared_write_count
            rate = operation_count / max(write_seconds, 0.000_001)
            print(
                f"      {session_count} sessions, {operation_count} saves, "
                f"{write_seconds:.3f}s, {rate:.1f} save calls/s"
            )
        finally:
            await store.close()


def log_cleanup_result(result: CleanupRunResult, label: str) -> None:
    for tier in ("memory", "disk"):
        decision = getattr(result, f"{tier}_decision")
        batch = getattr(result, f"{tier}_result")
        logger.info(
            "event=eviction_test label=%s tier=%s reason=%s before=%d "
            "target=%s selected=%d evicted=%d skipped=%d freed=%d "
            "after=%d target_satisfied=%s budget_exhausted=%s",
            label,
            tier,
            decision.trigger_reason.value if decision.trigger_reason else "none",
            decision.used_bytes,
            decision.target_used_bytes,
            batch.selected_count if batch else 0,
            batch.evicted_count if batch else 0,
            batch.skipped_count if batch else 0,
            batch.actual_free_bytes if batch else 0,
            batch.final_used_bytes if batch else decision.used_bytes,
            batch.actual_target_satisfied if batch else None,
            batch.budget_exhausted if batch else False,
        )
        if batch:
            for item in batch.results:
                logger.info(
                    "event=eviction_test_session label=%s tier=%s "
                    "session_id=%s status=%s freed=%d",
                    label,
                    tier,
                    item.session_id,
                    item.status.value,
                    item.freed_bytes,
                )
    logger.info(
        "event=eviction_test_pressure label=%s protected=%d reclaimable=%d blocked=%s",
        label,
        result.disk_pressure.protected_used_bytes,
        result.disk_pressure.reclaimable_used_bytes,
        result.disk_pressure.pressure_blocked,
    )


async def run_cleanup_logged(
    cleanup: PeriodicSessionStoreCleanup,
    label: str,
) -> CleanupRunResult:
    result = await cleanup.run_once()
    log_cleanup_result(result, label)
    return result


def check_evicted(result: CleanupRunResult, tier: str, expected: list[str]) -> None:
    batch = getattr(result, f"{tier}_result")
    actual = (
        []
        if batch is None
        else [item.session_id for item in batch.results if item.evicted]
    )
    check(actual == expected, f"{tier}: expected {expected}, evicted {actual}")
    if expected:
        check(batch.actual_free_bytes > 0, f"{tier} freed no bytes")
        check(batch.skipped_count == 0, f"{tier} unexpectedly skipped candidates")


@asynccontextmanager
async def eviction_store():
    with tempfile.TemporaryDirectory(prefix="responses-store-eviction-") as root:
        memory = MemorySessionStore(8 * 1024 * 1024, mem_idle_ttl_seconds=1)
        disk = SQLiteSessionStore(
            db_path=str(Path(root) / "store.sqlite3"),
            disk_idle_ttl_seconds=60,
            write_interval_seconds=0.02,
        )
        store = TieredSessionStore(memory, disk, num_shards=8)
        try:
            yield store, memory, disk
        finally:
            await store.close()


async def test_no_cleanup_below_watermark() -> None:
    async with eviction_store() as (store, memory, disk):
        with patch("time.time_ns", return_value=1_800_000_000_000_000_000):
            await store.save("keep", "response", [1, 2, 3])
            await wait_for_disk(disk, ["keep"])
            before = (memory.used_bytes, await store.disk_used_bytes())
            cleanup = make_cleanup(store, 8 * 1024 * 1024, 8 * 1024 * 1024)
            result = await run_cleanup_logged(cleanup, "below-high-watermark")
            check(not result.memory_decision.should_evict, "memory triggered")
            check(not result.disk_decision.should_evict, "disk triggered")
            check(
                result.memory_result is None and result.disk_result is None,
                "unexpected eviction batch",
            )
            check(
                await memory.exists("keep") and await disk.exists("keep"),
                "live session was removed",
            )
            check(
                before == (memory.used_bytes, await store.disk_used_bytes()),
                "usage changed without eviction",
            )


async def test_lru_and_low_watermark() -> None:
    async with eviction_store() as (store, memory, disk):
        base_ms = 1_800_000_000_000
        with patch("time.time_ns") as clock:
            for offset, sid in enumerate(("a", "b", "c")):
                clock.return_value = (base_ms + offset) * 1_000_000
                await store.save(sid, "response", [1] * 128)
            await wait_for_disk(disk, ["a", "b", "c"])
            clock.return_value = (base_ms + 3) * 1_000_000
            await store.get("a")
            sizes = {
                m.session_id: m.mem_size_bytes for m in await memory.list_metadata()
            }
            before = memory.used_bytes
            low = before - sizes["b"]
            cleanup = make_cleanup(
                store,
                8 * 1024 * 1024,
                8 * 1024 * 1024,
                memory_watermarks=CapacityWaterMarks(
                    max_bytes=8 * 1024 * 1024,
                    high_watermark_bytes=before,
                    low_watermark_bytes=low,
                ),
            )
            result = await run_cleanup_logged(cleanup, "lru-low-watermark")
            check(
                result.memory_decision.trigger_reason.value == "high_watermark",
                "expected high watermark trigger without TTL expiry",
            )
            check_evicted(result, "memory", ["b"])
            check(memory.used_bytes == low, "did not stop at low watermark")
            check(
                result.memory_result.actual_target_satisfied is True,
                "low watermark not reported satisfied",
            )
            check(
                await memory.exists("a") and await memory.exists("c"),
                "newer sessions were evicted",
            )
            again = await run_cleanup_logged(cleanup, "after-low-watermark")
            check(not again.memory_decision.should_evict, "cleanup kept evicting")
            check(await store.get("b") == [1] * 128, "evicted copy not recoverable")


async def test_ttl_priority_and_budget() -> None:
    async with eviction_store() as (store, memory, disk):
        base_ms = 1_800_000_000_000
        with patch("time.time_ns") as clock:
            for offset, sid in ((0, "expired"), (500, "live")):
                clock.return_value = (base_ms + offset) * 1_000_000
                await store.save(sid, "response", [1] * 128)
            await wait_for_disk(disk, ["expired", "live"])
            clock.return_value = (base_ms + 1_000) * 1_000_000
            cleanup = make_cleanup(store, 2, 8 * 1024 * 1024, max_candidates=1)
            result = await run_cleanup_logged(cleanup, "ttl-first-budget-one")
            check(
                result.memory_decision.trigger_reason.value == "ttl_and_high_watermark",
                "expected combined trigger",
            )
            check_evicted(result, "memory", ["expired"])
            check(
                result.memory_result.candidates_limit_reached,
                "candidate budget was not reached",
            )
            check(
                result.memory_result.actual_target_satisfied is False,
                "budget-limited batch incorrectly reported reaching low watermark",
            )
            check(await memory.exists("live"), "live entry evicted before expired")
            result = await run_cleanup_logged(cleanup, "budget-next-round")
            check_evicted(result, "memory", ["live"])
            check(memory.used_bytes == 0, "next batch did not finish cleanup")


async def test_disk_capacity_and_protection() -> None:
    async with eviction_store() as (store, memory, disk):
        with patch("time.time_ns", return_value=1_800_000_000_000_000_000):
            await store.save("disk-victim", "response", [1] * 128)
            await wait_for_disk(disk, ["disk-victim"])
            cleanup = make_cleanup(store, 2, 2)
            first = await run_cleanup_logged(cleanup, "disk-protected-snapshot")
            check_evicted(first, "memory", ["disk-victim"])
            check_evicted(first, "disk", [])
            check(
                first.disk_pressure.blocked_by_protected_bytes,
                "memory-resident disk copy was not protected",
            )
            check(await disk.exists("disk-victim"), "protected copy was deleted")
            second = await run_cleanup_logged(cleanup, "disk-high-watermark")
            check(
                second.disk_decision.trigger_reason.value == "high_watermark",
                "expected disk capacity pressure without TTL expiry",
            )
            check_evicted(second, "disk", ["disk-victim"])
            check(
                second.disk_result.actual_target_satisfied is True,
                "disk did not reach low watermark",
            )
            check(await store.disk_used_bytes() == 0, "disk usage not reclaimed")
            check(not await store.exists("disk-victim"), "session still exists")
            check(await store.get("disk-victim") is None, "deleted session restored")


async def test_background_cleanup() -> None:
    async with eviction_store() as (store, memory, disk):
        cleanup = make_cleanup(store, 2, 2, interval_seconds=0.05)
        await store.save("background", "response", [1] * 128)
        await wait_for_disk(disk, ["background"])
        original_run_once = cleanup.run_once
        runs: list[CleanupRunResult] = []

        async def observed_run_once() -> CleanupRunResult:
            result = await original_run_once()
            runs.append(result)
            log_cleanup_result(result, f"background-round-{len(runs)}")
            return result

        try:
            with patch.object(cleanup, "run_once", side_effect=observed_run_once):
                cleanup.start()

                async def fully_evicted() -> bool:
                    return len(runs) >= 2 and not await store.exists("background")

                await wait_until(fully_evicted, "background eviction of both tiers")
                check(cleanup.is_running, "background worker stopped unexpectedly")
                check_evicted(runs[0], "memory", ["background"])
                check_evicted(runs[1], "disk", ["background"])
                check(not await memory.exists("background"), "memory copy remains")
                check(await store.disk_used_bytes() == 0, "disk copy remains")
        finally:
            await cleanup.stop()
        check(not cleanup.is_running, "cleanup worker did not stop")


def make_cleanup(
    store: TieredSessionStore,
    memory_max: int,
    disk_max: int,
    *,
    memory_watermarks: CapacityWaterMarks | None = None,
    disk_watermarks: CapacityWaterMarks | None = None,
    max_candidates: int = 10_000,
    max_bytes: int = 512 * 1024 * 1024,
    interval_seconds: float = 60,
) -> PeriodicSessionStoreCleanup:
    """Build a cleanup runner with a generous per-run budget."""
    budget = EvictionSelectionBudget(
        max_candidates=max_candidates,
        max_bytes=max_bytes,
    )
    return PeriodicSessionStoreCleanup(
        store,
        PeriodicCleanupConfig(
            interval_seconds=interval_seconds,
            memory=TierCleanupConfig(
                watermarks=memory_watermarks
                or CapacityWaterMarks(
                    max_bytes=memory_max,
                    low_watermark_bytes=0,
                    high_watermark_bytes=max(1, memory_max - 1),
                ),
                budget=budget,
            ),
            disk=TierCleanupConfig(
                watermarks=disk_watermarks
                or CapacityWaterMarks(
                    max_bytes=disk_max,
                    low_watermark_bytes=0,
                    high_watermark_bytes=max(1, disk_max - 1),
                ),
                budget=budget,
            ),
        ),
    )


async def test_ttl_cleanup() -> None:
    with tempfile.TemporaryDirectory(prefix="responses-store-ttl-") as root:
        memory = MemorySessionStore(
            max_capacity_bytes=8 * 1024 * 1024,
            mem_idle_ttl_seconds=1,
        )
        disk = SQLiteSessionStore(
            db_path=str(Path(root) / "store.sqlite3"),
            disk_idle_ttl_seconds=1,
            write_interval_seconds=0.02,
        )
        store = TieredSessionStore(memory, disk, num_shards=8)
        cleanup = make_cleanup(
            store,
            memory_max=1024 * 1024 * 1024,
            disk_max=1024 * 1024 * 1024,
        )
        try:
            session_id = "session-ttl"
            await store.save(session_id, "response-ttl", [31, 32, 33])
            await wait_for_disk(disk, [session_id])
            await asyncio.sleep(1.1)
            first_cleanup = await run_cleanup_logged(cleanup, "ttl-memory")
            check_evicted(first_cleanup, "memory", [session_id])
            check(
                first_cleanup.memory_result is not None,
                "expired memory session did not trigger cleanup",
            )
            check(
                not await memory.exists(session_id),
                "expired memory copy was not evicted",
            )

            second_cleanup = await run_cleanup_logged(cleanup, "ttl-disk")
            check_evicted(second_cleanup, "disk", [session_id])
            check(
                second_cleanup.disk_result is not None,
                "expired disk-only session did not trigger cleanup",
            )
            check(
                not await store.exists(session_id),
                "expired session still exists after both tiers were cleaned",
            )
            check(
                await store.get(session_id) is None,
                "TTL-cleaned session should signal a context miss",
            )
        finally:
            await cleanup.stop()
            await store.close()


async def test_millisecond_timestamps() -> None:
    with tempfile.TemporaryDirectory(prefix="responses-store-milliseconds-") as root:
        db_path = str(Path(root) / "store.sqlite3")
        memory = MemorySessionStore(
            max_capacity_bytes=8 * 1024 * 1024,
            mem_idle_ttl_seconds=1,
        )
        disk = SQLiteSessionStore(
            db_path=db_path,
            disk_idle_ttl_seconds=2,
            write_interval_seconds=0.02,
        )
        store = TieredSessionStore(memory, disk, num_shards=8)
        cleanup = make_cleanup(store, 1024 * 1024, 1024 * 1024)
        created_ms = 1_800_000_000_123
        session_id = "session-milliseconds"
        try:
            with patch("time.time_ns", return_value=created_ms * 1_000_000) as clock:
                await store.save(session_id, "response-ms", [1, 2, 3])
                await wait_for_disk(disk, [session_id])
                metadata = (await store.list_metadata())[0]
                check(metadata.created_at == created_ms, "creation lost milliseconds")
                check(metadata.updated_at == created_ms, "update lost milliseconds")
                check(
                    metadata.mem_idle_expires_at == created_ms + 1_000,
                    "memory TTL was not converted from seconds to milliseconds",
                )
                with sqlite3.connect(db_path) as connection:
                    row = connection.execute(
                        "SELECT created_at, updated_at, disk_idle_expires_at "
                        "FROM session_state WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                check(
                    row == (created_ms, created_ms, created_ms + 2_000),
                    "SQLite did not persist millisecond timestamps",
                )

                clock.return_value = (created_ms + 1) * 1_000_000
                await memory.get(session_id)
                check(
                    (await memory.list_metadata())[0].updated_at == created_ms + 1,
                    "same-second access did not update the millisecond timestamp",
                )

                clock.return_value = (created_ms + 2) * 1_000_000
                await memory.delete(session_id)
                check(await store.get(session_id) == [1, 2, 3], "restore failed")
                metadata = (await store.list_metadata())[0]
                check(
                    metadata.created_at == created_ms
                    and metadata.updated_at == created_ms + 2
                    and metadata.mem_idle_expires_at == created_ms + 1_002
                    and metadata.disk_idle_expires_at == created_ms + 2_002,
                    "restore did not refresh both TTLs in milliseconds",
                )

                for offset, memory_evicted, disk_evicted in (
                    (1_001, False, False),
                    (1_002, True, False),
                    (2_001, False, False),
                    (2_002, False, True),
                ):
                    clock.return_value = (created_ms + offset) * 1_000_000
                    result = await run_cleanup_logged(cleanup, f"ttl-{offset}ms")
                    check(
                        result.started_at == created_ms + offset,
                        "cleanup clock does not use milliseconds",
                    )
                    for tier, expected in (
                        ("memory", memory_evicted),
                        ("disk", disk_evicted),
                    ):
                        batch = getattr(result, f"{tier}_result")
                        count = 0 if batch is None else batch.evicted_count
                        check(
                            count == int(expected),
                            f"incorrect {tier} TTL eviction at offset {offset} ms",
                        )
                check(not await store.exists(session_id), "expired session survived")
        finally:
            await cleanup.stop()
            await store.close()


async def test_capacity_cleanup() -> None:
    with tempfile.TemporaryDirectory(prefix="responses-store-capacity-") as root:
        memory = MemorySessionStore(max_capacity_bytes=8 * 1024 * 1024)
        disk = SQLiteSessionStore(
            db_path=str(Path(root) / "store.sqlite3"),
            disk_idle_ttl_seconds=60,
            write_interval_seconds=0.02,
        )
        store = TieredSessionStore(memory, disk, num_shards=8)
        cleanup = make_cleanup(store, memory_max=2, disk_max=1024 * 1024 * 1024)
        try:
            session_id = "session-capacity"
            expected = list(range(128))
            await store.save(session_id, "response-capacity", expected)
            await wait_for_disk(disk, [session_id])
            result = await run_cleanup_logged(cleanup, "memory-high-watermark")
            check_evicted(result, "memory", [session_id])
            check(
                result.memory_result.actual_target_satisfied is True,
                "memory did not reach low watermark",
            )
            check(
                result.memory_decision.high_watermark_reached,
                "memory high watermark was not detected",
            )
            check(
                not await memory.exists(session_id),
                "capacity cleanup did not evict the memory copy",
            )
            check(
                await store.get(session_id) == expected,
                "capacity eviction lost a recoverable session",
            )
        finally:
            await cleanup.stop()
            await store.close()


async def test_memory_only_mode() -> None:
    memory = MemorySessionStore(max_capacity_bytes=8 * 1024 * 1024)
    store = TieredSessionStore(memory, None, num_shards=4)
    await store.save("memory-only", "response-1", [1, 2])
    await store.save("memory-only", "response-2", [3])
    check(await store.get("memory-only") == [1, 2, 3], "memory-only save failed")
    check(not store.disk_enabled, "memory-only store reports disk enabled")
    check(await store.delete("memory-only"), "memory-only delete failed")
    check(await store.get("memory-only") is None, "memory-only delete was not final")
    await store.close()


async def test_close_flushes_pending_writes() -> None:
    with tempfile.TemporaryDirectory(prefix="responses-store-close-") as root:
        db_path = str(Path(root) / "store.sqlite3")
        disk = SQLiteSessionStore(
            db_path=db_path,
            write_interval_seconds=0.5,
        )
        await disk.save_snapshot(
            _session_state("session-close", "response-close", [71, 72])
        )
        await disk.close()

        with sqlite3.connect(db_path) as connection:
            row = connection.execute(
                "SELECT token_ids FROM session_state WHERE session_id = ?",
                ("session-close",),
            ).fetchone()
        check(row is not None, "close() did not flush the pending write")

        try:
            await disk.get("session-close")
        except RuntimeError:
            pass
        else:
            raise AssertionError("closed disk store accepted a read")


def _session_state(
    session_id: str,
    response_id: str,
    token_ids: list[int],
) -> Any:
    from vllm.entrypoints.openai.responses.store.base import SessionState

    now = time.time_ns() // 1_000_000
    return SessionState(
        session_id=session_id,
        response_id=response_id,
        token_ids=token_ids,
        created_at=now,
        updated_at=now,
        memory_resident=True,
    )


async def run_module_tests(
    session_count: int,
    rounds: int,
    eviction_only: bool = False,
) -> bool:
    """Run all direct module acceptance tests."""
    cases: list[tuple[str, TestCase]] = [
        ("memory CRUD and copy isolation", test_memory_crud),
        (
            "tiered roundtrip, disk restore, encryption",
            test_tiered_roundtrip_and_encryption,
        ),
        ("delete pending write and recreate", test_delete_pending_and_recreate),
        (
            "corrupt disk copy and full-history rerender",
            test_corrupt_disk_copy_and_full_rerender,
        ),
        (
            "concurrent session isolation",
            lambda: test_concurrent_sessions(session_count, rounds),
        ),
        ("TTL cleanup", test_ttl_cleanup),
        ("millisecond timestamps and TTL boundaries", test_millisecond_timestamps),
        ("capacity cleanup", test_capacity_cleanup),
        ("no cleanup below high watermark", test_no_cleanup_below_watermark),
        ("LRU order and low watermark stop", test_lru_and_low_watermark),
        ("TTL priority and bounded batches", test_ttl_priority_and_budget),
        ("disk capacity and protected copies", test_disk_capacity_and_protection),
        ("background periodic cleanup", test_background_cleanup),
        ("memory-only mode", test_memory_only_mode),
        ("close flushes pending writes", test_close_flushes_pending_writes),
    ]
    if eviction_only:
        eviction_tests = {
            test_ttl_cleanup,
            test_millisecond_timestamps,
            test_capacity_cleanup,
            test_no_cleanup_below_watermark,
            test_lru_and_low_watermark,
            test_ttl_priority_and_budget,
            test_disk_capacity_and_protection,
            test_background_cleanup,
        }
        cases = [(name, case) for name, case in cases if case in eviction_tests]
    return await run_cases("MODULE ACCEPTANCE", cases)


async def run_cases(title: str, cases: list[tuple[str, TestCase]]) -> bool:
    """Run cases independently and print a compact summary."""
    print(f"\n=== {title} ===")
    failures: list[str] = []
    for index, (name, case) in enumerate(cases, start=1):
        started = time.perf_counter()
        try:
            await case()
        except Exception:
            failures.append(name)
            print(f"[{index:02d}] FAIL {name}")
            traceback.print_exc()
        else:
            elapsed = time.perf_counter() - started
            print(f"[{index:02d}] PASS {name} ({elapsed:.3f}s)")

    print(f"\nResult: {len(cases) - len(failures)}/{len(cases)} passed")
    if failures:
        print("Failed cases: " + ", ".join(failures))
        return False
    print(f"ALL {title} TESTS PASSED")
    return True


def http_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> tuple[int, dict[str, Any]]:
    """Send one JSON request and return errors as ordinary responses."""
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"raw_body": raw}
        return error.code, payload


def discover_model(base_url: str) -> str:
    status, payload = http_json("GET", f"{base_url}/v1/models")
    check(status == 200, f"GET /v1/models returned {status}: {payload}")
    models = payload.get("data") or []
    check(bool(models), "GET /v1/models returned no models")
    return str(models[0]["id"])


def flatten_strings(value: Any) -> list[str]:
    """Collect text fields from a Responses API JSON object."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(flatten_strings(item))
        return result
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            if key in {"text", "output_text", "content"} or isinstance(
                item, (dict, list)
            ):
                result.extend(flatten_strings(item))
        return result
    return []


async def run_api_tests(
    base_url: str,
    model: str | None,
    session_count: int = 1,
    concurrency: int = 1,
) -> bool:
    """Exercise a fixed number of distinct sessions in bounded HTTP phases."""
    check(session_count > 0, "--sessions must be greater than zero")
    check(concurrency > 0, "--concurrency must be greater than zero")
    base_url = base_url.rstrip("/")
    selected_model = model or await asyncio.to_thread(discover_model, base_url)
    run_id = uuid.uuid4().hex
    session_ids = [f"acceptance-{run_id}-{i}" for i in range(session_count)]
    failed_sessions: set[int] = set()

    async def request(
        index: int,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        return await asyncio.to_thread(
            http_json,
            method,
            f"{base_url}{path}",
            body,
            {"x-session-id": session_ids[index]},
        )

    async def query_session(index: int, method: str = "GET") -> dict[str, Any]:
        query = urllib.parse.urlencode({"session_id": session_ids[index]})
        endpoint = "delete" if method == "DELETE" else "get"
        status, payload = await request(index, method, f"/session/{endpoint}?{query}")
        check(status == 200, f"{method} session returned {status}: {payload}")
        return payload

    async def save(index: int) -> None:
        code = str(731900 + index)
        status, payload = await request(
            index,
            "POST",
            "/v1/responses",
            {
                "model": selected_model,
                "input": f"Remember this verification code: {code}.",
                "temperature": 0,
                "max_output_tokens": 32,
                "store": False,
                "use_store": True,
                "use_incremental_token": False,
            },
        )
        check(status == 200, f"save returned {status}: {payload}")

    async def exists(index: int) -> None:
        payload = await query_session(index)
        check(payload.get("exists") is True, f"saved session is missing: {payload}")

    async def incremental_hit(index: int) -> None:
        status, payload = await request(
            index,
            "POST",
            "/v1/responses",
            {
                "model": selected_model,
                "input": "Reply with only the verification code I gave you.",
                "temperature": 0,
                "max_output_tokens": 32,
                "store": False,
                "use_store": True,
                "use_incremental_token": True,
            },
        )
        check(status == 200, f"incremental request returned {status}: {payload}")
        output = " ".join(flatten_strings(payload.get("output", [])))
        check(str(731900 + index) in output, f"wrong session context: {output}")

    async def delete_and_miss(index: int) -> None:
        payload = await query_session(index, "DELETE")
        check(payload.get("deleted") is True, f"delete not acknowledged: {payload}")
        payload = await query_session(index)
        check(
            payload.get("exists") is False, f"deleted session still exists: {payload}"
        )
        status, payload = await request(
            index,
            "POST",
            "/v1/responses",
            {
                "model": selected_model,
                "input": "This request should miss.",
                "max_output_tokens": 8,
                "store": False,
                "use_store": False,
                "use_incremental_token": True,
            },
        )
        check(
            status == 500 and "incremental_context_miss" in json.dumps(payload),
            f"expected incremental_context_miss, got {status}: {payload}",
        )

    async def rebuild(index: int) -> None:
        await save(index)
        await exists(index)

    async def cleanup_session(index: int) -> None:
        await query_session(index, "DELETE")
        payload = await query_session(index)
        check(payload.get("exists") is False, f"cleanup failed: {payload}")

    async def phase(
        name: str,
        operation: Callable[[int], Awaitable[None]],
        include_failed: bool = False,
    ) -> None:
        indices = iter(
            i
            for i in range(session_count)
            if include_failed or i not in failed_sessions
        )
        errors: list[str] = []
        completed = 0

        async def worker() -> None:
            nonlocal completed
            for index in indices:
                try:
                    await operation(index)
                except Exception as error:
                    failed_sessions.add(index)
                    errors.append(session_ids[index])
                    print(
                        f"[API] FAIL phase={name} session_id={session_ids[index]} "
                        f"error={error}",
                        flush=True,
                    )
                else:
                    print(
                        f"[API] PASS phase={name} session_id={session_ids[index]}",
                        flush=True,
                    )
                completed += 1

        await asyncio.gather(
            *(worker() for _ in range(min(concurrency, session_count)))
        )
        print(
            f"[API] phase={name} passed={completed - len(errors)} "
            f"failed={len(errors)} skipped={session_count - completed}",
            flush=True,
        )
        check(completed > 0, f"{name}: no eligible sessions remain")
        check(not errors, f"{name}: {len(errors)} session(s) failed")

    print(
        f"Using model: {selected_model}; sessions={session_count}; "
        f"concurrency={concurrency}; run_id={run_id}",
        flush=True,
    )
    succeeded = False
    cleanup_ok = True
    try:
        succeeded = await run_cases(
            "LIVE API ACCEPTANCE",
            [
                ("save all sessions", lambda: phase("save", save)),
                ("query all sessions", lambda: phase("exists", exists)),
                (
                    "incremental context isolation",
                    lambda: phase("hit", incremental_hit),
                ),
                (
                    "delete and verify context miss",
                    lambda: phase("delete-miss", delete_and_miss),
                ),
                ("rebuild full context", lambda: phase("rebuild", rebuild)),
            ],
        )
    finally:
        try:
            await phase("cleanup", cleanup_session, include_failed=True)
        except Exception:
            cleanup_ok = False
            traceback.print_exc()
    print(
        f"API result: {session_count - len(failed_sessions)}/{session_count} "
        "sessions passed",
        flush=True,
    )
    return succeeded and cleanup_ok


async def run_api_ttl_refresh_tests(
    base_url: str,
    model: str | None,
    sessions: int,
    concurrency: int,
    ttl_seconds: float,
    cleanup_interval: float,
) -> bool:
    """Verify read renewal against an idle control on a memory-only server."""
    check(sessions > 0 and concurrency > 0, "sessions and concurrency must be positive")
    check(
        ttl_seconds > 0 and cleanup_interval > 0,
        "TTL and cleanup interval must be positive and match the server",
    )
    base_url = base_url.rstrip("/")
    selected_model = model or await asyncio.to_thread(discover_model, base_url)
    run_id = uuid.uuid4().hex
    indices = iter(range(sessions))
    failures: set[int] = set()
    margin = 3 * cleanup_interval + 1

    async def request(
        sid: str,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        return await asyncio.to_thread(
            http_json,
            method,
            f"{base_url}{path}",
            body,
            {"x-session-id": sid},
            min(120.0, ttl_seconds / 3),
        )

    async def present(sid: str) -> bool:
        query = urllib.parse.urlencode({"session_id": sid})
        status, payload = await request(sid, "GET", f"/session/get?{query}")
        check(
            status == 200 and isinstance(payload.get("exists"), bool),
            f"session query failed: {status}: {payload}",
        )
        return payload["exists"]

    async def generate(sid: str, incremental: bool) -> None:
        started = time.monotonic()
        status, payload = await request(
            sid,
            "POST",
            "/v1/responses",
            {
                "model": selected_model,
                "input": "Reply OK." if incremental else "Remember the word orchid.",
                "max_output_tokens": 1,
                "temperature": 0,
                "store": False,
                "use_store": not incremental,
                "use_incremental_token": incremental,
            },
        )
        check(status == 200, f"context request failed: {status}: {payload}")
        check(
            time.monotonic() - started < ttl_seconds / 3,
            "request took too long for the TTL window; increase the server and "
            "test TTL or reduce concurrency",
        )

    async def test_pair(index: int) -> None:
        active = f"ttl-{run_id}-{index}-active"
        control = f"ttl-{run_id}-{index}-control"
        try:
            await generate(control, False)
            await generate(active, False)
            saved_at = time.monotonic()
            check(
                await present(active) and await present(control),
                "new sessions are missing",
            )
            # Response completion is an upper bound on the initial save time.
            # Cross its original TTL plus several cleanup intervals before checking.
            original_expiry = saved_at + ttl_seconds + margin
            while True:
                remaining = original_expiry - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(ttl_seconds / 3, remaining))
                if time.monotonic() >= original_expiry:
                    break
                await generate(active, True)
                print(
                    f"[API TTL] refreshed session_id={active} use_store=False",
                    flush=True,
                )

            check(
                not await present(control),
                "idle control did not expire; check disk is disabled, TTL and "
                "cleanup interval match, and background cleanup is running",
            )
            check(
                await present(active),
                "accessed session expired at its original deadline: TTL not renewed",
            )
            print(
                f"[API TTL] original-deadline-passed active={active} "
                "active_exists=True control_exists=False",
                flush=True,
            )

            async def expired() -> bool:
                # The management exists endpoint does not refresh idle TTL.
                if not await present(active):
                    return True
                await asyncio.sleep(0.25)
                return False

            await wait_until(
                expired,
                "idle expiration after last refresh",
                timeout=ttl_seconds + margin,
            )
            status, payload = await request(
                active,
                "POST",
                "/v1/responses",
                {
                    "model": selected_model,
                    "input": "Reply OK.",
                    "max_output_tokens": 1,
                    "store": False,
                    "use_store": False,
                    "use_incremental_token": True,
                },
            )
            check(
                status == 500 and "incremental_context_miss" in json.dumps(payload),
                f"expected context miss after idle TTL: {status}: {payload}",
            )
            print(
                f"[API TTL] idle-expired session_id={active} "
                "incremental_context_miss=True",
                flush=True,
            )
        finally:
            cleanup_errors = []
            for sid in (active, control):
                try:
                    query = urllib.parse.urlencode({"session_id": sid})
                    status, payload = await request(
                        sid, "DELETE", f"/session/delete?{query}"
                    )
                    check(status == 200, f"cleanup returned {status}: {payload}")
                    check(not await present(sid), "session survived cleanup")
                except Exception as error:
                    cleanup_errors.append(f"{sid}: {error}")
            if cleanup_errors:
                failures.add(index)
                print(
                    "[API TTL] FAIL cleanup: " + "; ".join(cleanup_errors), flush=True
                )

    async def worker() -> None:
        for index in indices:
            try:
                await test_pair(index)
            except Exception as error:
                failures.add(index)
                print(f"[API TTL] FAIL pair={index} error={error}", flush=True)
            else:
                if index not in failures:
                    print(f"[API TTL] PASS pair={index}", flush=True)

    print(
        f"TTL refresh test: {sessions} pairs ({2 * sessions} session IDs), "
        f"concurrency={concurrency}, ttl={ttl_seconds}s, "
        f"cleanup_interval={cleanup_interval}s. Requires a memory-only server "
        "with sufficient capacity and matching settings.",
        flush=True,
    )
    await asyncio.gather(*(worker() for _ in range(min(sessions, concurrency))))
    print(f"TTL refresh result: {sessions - len(failures)}/{sessions} pairs passed")
    return not failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    module_parser = subparsers.add_parser("module", help="test store classes directly")
    module_parser.add_argument("--sessions", type=int, default=500)
    module_parser.add_argument("--rounds", type=int, default=3)
    module_parser.add_argument(
        "--eviction-only",
        action="store_true",
        help="run only eviction scenarios",
    )

    api_parser = subparsers.add_parser("api", help="test a running vLLM server")
    api_parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    api_parser.add_argument("--model", default=None)
    api_parser.add_argument(
        "--verify-ttl-refresh",
        action="store_true",
        help="run paired read-TTL renewal tests instead of the ordinary API suite",
    )
    api_parser.add_argument(
        "--ttl-seconds",
        type=float,
        default=None,
        help="memory TTL configured on the memory-only server; required for TTL tests",
    )
    api_parser.add_argument(
        "--ttl-cleanup-interval",
        type=float,
        default=1,
        help="server cleanup interval in seconds for TTL tests (default: 1)",
    )
    api_parser.add_argument(
        "--sessions",
        type=int,
        default=1,
        help="number of distinct API sessions to create and verify (default: 1)",
    )
    api_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="maximum concurrent API session workers (default: 1)",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> bool:
    if args.mode == "module":
        check(args.sessions > 0, "--sessions must be greater than zero")
        check(args.rounds > 0, "--rounds must be greater than zero")
        return await run_module_tests(args.sessions, args.rounds, args.eviction_only)
    if args.verify_ttl_refresh:
        check(args.ttl_seconds is not None, "--ttl-seconds is required")
        return await run_api_ttl_refresh_tests(
            args.base_url,
            args.model,
            args.sessions,
            args.concurrency,
            args.ttl_seconds,
            args.ttl_cleanup_interval,
        )
    return await run_api_tests(
        args.base_url,
        args.model,
        args.sessions,
        args.concurrency,
    )


def main() -> int:
    args = parse_args()
    try:
        succeeded = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
