# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the --sleep-idle-ttl idle sleep manager and middleware."""

import asyncio

import pytest

from vllm.entrypoints.serve.idle_sleep.manager import IdleSleepManager
from vllm.entrypoints.serve.idle_sleep.middleware import is_inference_request

TTL = 0.05
# Long enough for the idle loop (polling at TTL / 10) to fire reliably.
IDLE_WAIT = 0.5


class FakeEngineClient:
    """Records sleep/wake calls; models partial (tag-based) wake-up."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.sleeping = False
        self._awake_tags: set[str] = set()

    async def sleep(self, level=1, mode="abort"):
        self.calls.append(("sleep", level, mode))
        self.sleeping = True
        self._awake_tags = set()

    async def wake_up(self, tags=None):
        self.calls.append(("wake_up", tags))
        if tags is None:
            self.sleeping = False
        else:
            self._awake_tags.update(tags)
            if {"weights", "kv_cache"} <= self._awake_tags:
                self.sleeping = False

    async def is_sleeping(self):
        return self.sleeping

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls.append(("collective_rpc", method))


async def start_manager(level: int = 1) -> tuple[IdleSleepManager, FakeEngineClient]:
    engine = FakeEngineClient()
    manager = IdleSleepManager(engine, ttl=TTL, level=level)
    manager.start()
    return manager, engine


@pytest.mark.asyncio
async def test_sleeps_after_idle_ttl():
    manager, engine = await start_manager()
    try:
        await asyncio.sleep(IDLE_WAIT)
        assert ("sleep", 1, "wait") in engine.calls
        assert engine.sleeping
        assert manager.auto_slept
    finally:
        manager.stop()


@pytest.mark.asyncio
async def test_request_wakes_engine():
    manager, engine = await start_manager()
    try:
        await asyncio.sleep(IDLE_WAIT)
        assert engine.sleeping

        await manager.on_request_start()
        assert not engine.sleeping
        assert not manager.auto_slept
        assert ("wake_up", None) in engine.calls
        manager.on_request_end()
    finally:
        manager.stop()


@pytest.mark.asyncio
async def test_in_flight_request_blocks_sleep():
    manager, engine = await start_manager()
    try:
        await manager.on_request_start()
        await asyncio.sleep(IDLE_WAIT)
        assert not engine.sleeping

        manager.on_request_end()
        await asyncio.sleep(IDLE_WAIT)
        assert engine.sleeping
    finally:
        manager.stop()


@pytest.mark.asyncio
async def test_activity_resets_idle_timer():
    manager, engine = await start_manager()
    try:
        for _ in range(5):
            await asyncio.sleep(TTL / 2)
            await manager.on_request_start()
            manager.on_request_end()
        assert not engine.sleeping
    finally:
        manager.stop()


@pytest.mark.asyncio
async def test_level2_wake_reloads_weights():
    manager, engine = await start_manager(level=2)
    try:
        await asyncio.sleep(IDLE_WAIT)
        assert ("sleep", 2, "wait") in engine.calls

        await manager.on_request_start()
        manager.on_request_end()
        wake_start = engine.calls.index(("wake_up", ["weights"]))
        assert engine.calls[wake_start : wake_start + 3] == [
            ("wake_up", ["weights"]),
            ("collective_rpc", "reload_weights"),
            ("wake_up", ["kv_cache"]),
        ]
        assert not engine.sleeping
    finally:
        manager.stop()


@pytest.mark.asyncio
async def test_externally_slept_engine_left_alone():
    engine = FakeEngineClient()
    engine.sleeping = True  # slept via the dev /sleep endpoint
    manager = IdleSleepManager(engine, ttl=TTL)
    manager.start()
    try:
        await asyncio.sleep(IDLE_WAIT)
        assert engine.calls == []  # no auto-sleep on top, no auto-wake

        await manager.on_request_start()
        manager.on_request_end()
        assert ("wake_up", None) not in engine.calls
    finally:
        manager.stop()


@pytest.mark.asyncio
async def test_external_wake_resyncs_state():
    manager, engine = await start_manager()
    try:
        await asyncio.sleep(IDLE_WAIT)
        assert manager.auto_slept

        # Woken directly via the dev /wake_up endpoint. The manager
        # resyncs (clearing auto_slept), then auto-sleeps again once a
        # fresh idle TTL elapses with no traffic.
        await engine.wake_up()
        await asyncio.sleep(IDLE_WAIT)
        assert engine.calls.count(("sleep", 1, "wait")) == 2
        assert engine.sleeping
    finally:
        manager.stop()


@pytest.mark.asyncio
async def test_concurrent_requests_single_wake():
    manager, engine = await start_manager()
    try:
        await asyncio.sleep(IDLE_WAIT)
        assert engine.sleeping

        await asyncio.gather(*(manager.on_request_start() for _ in range(10)))
        assert engine.calls.count(("wake_up", None)) == 1
        assert manager.in_flight == 10
        for _ in range(10):
            manager.on_request_end()
    finally:
        manager.stop()


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("POST", "/v1/chat/completions", True),
        ("POST", "/v1/completions", True),
        ("POST", "/v1/embeddings", True),
        ("POST", "/score", True),
        ("POST", "/invocations", True),
        ("GET", "/v1/models", False),
        ("GET", "/health", False),
        ("GET", "/metrics", False),
        ("POST", "/tokenize", False),
        ("POST", "/sleep", False),
        ("POST", "/wake_up", False),
    ],
)
def test_is_inference_request(method, path, expected):
    assert is_inference_request({"method": method, "path": path}) is expected
