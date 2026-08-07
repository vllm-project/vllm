# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AsyncHostBuffer."""

import threading
import time

import pytest

from vllm.v1.kv_offload.async_host_buffer import AsyncHostBuffer

pytestmark = pytest.mark.cpu_test

POLL_TIMEOUT_S = 5.0


def _poll_until_ready(buf: AsyncHostBuffer, timeout: float = POLL_TIMEOUT_S):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resource = buf.poll()
        if resource is not None:
            return resource
        time.sleep(0.005)
    pytest.fail("poll() never returned the built resource")


# ---------------------------------------------------------------------------
# poll() — single-owner handoff
# ---------------------------------------------------------------------------


def test_poll_returns_none_while_building():
    """poll() must return None before allocate() has finished."""
    release = threading.Event()
    buf = AsyncHostBuffer(
        allocate=lambda: release.wait(timeout=POLL_TIMEOUT_S) and "resource",
        cleanup=lambda r: None,
        thread_name="t",
    )
    try:
        assert buf.poll() is None
    finally:
        release.set()
        buf.close()


def test_poll_returns_resource_once_built():
    """poll() must return the resource once allocate() completes."""
    buf = AsyncHostBuffer(
        allocate=lambda: "resource", cleanup=lambda r: None, thread_name="t"
    )
    assert _poll_until_ready(buf) == "resource"


def test_second_poll_returns_none_after_adoption():
    """Once adopted, later poll() calls must return None — single owner."""
    buf = AsyncHostBuffer(
        allocate=lambda: "resource", cleanup=lambda r: None, thread_name="t"
    )
    _poll_until_ready(buf)
    assert buf.poll() is None


def test_poll_sets_failed_and_returns_none_when_allocate_raises():
    """A raising allocate() must not be adopted, and must flip `failed`."""

    def allocate():
        raise RuntimeError("boom")

    buf = AsyncHostBuffer(allocate=allocate, cleanup=lambda r: None, thread_name="t")
    buf._thread.join(timeout=POLL_TIMEOUT_S)
    assert not buf.failed  # not observed until poll() is called
    assert buf.poll() is None
    assert buf.failed


# ---------------------------------------------------------------------------
# close() — exactly-once cleanup
# ---------------------------------------------------------------------------


def test_close_cleans_up_unadopted_resource():
    """close() must invoke cleanup() on a built-but-never-adopted resource."""
    cleaned = threading.Event()
    buf = AsyncHostBuffer(
        allocate=lambda: "resource",
        cleanup=lambda r: cleaned.set(),
        thread_name="t",
    )
    buf.close(timeout=POLL_TIMEOUT_S)
    assert cleaned.is_set()


def test_close_after_adoption_does_not_clean_up():
    """close() must be a no-op once the caller has adopted the resource."""
    cleaned = threading.Event()
    buf = AsyncHostBuffer(
        allocate=lambda: "resource",
        cleanup=lambda r: cleaned.set(),
        thread_name="t",
    )
    _poll_until_ready(buf)
    buf.close(timeout=POLL_TIMEOUT_S)
    assert not cleaned.is_set()


def test_close_timeout_abandons_without_blocking_and_still_cleans_up():
    """close() must return by `timeout` even if allocate() is still running,
    and the background thread must clean up the resource once it finishes."""
    release = threading.Event()
    cleaned = threading.Event()
    buf = AsyncHostBuffer(
        allocate=lambda: release.wait(timeout=POLL_TIMEOUT_S) and "resource",
        cleanup=lambda r: cleaned.set(),
        thread_name="t",
    )

    start = time.monotonic()
    buf.close(timeout=0.05)
    assert time.monotonic() - start < POLL_TIMEOUT_S / 2

    release.set()
    assert cleaned.wait(timeout=POLL_TIMEOUT_S), (
        "abandoned resource was never cleaned up"
    )
