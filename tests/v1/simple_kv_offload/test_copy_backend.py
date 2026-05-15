# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffloadConnector copy backend."""

import queue
import threading
import time
from types import SimpleNamespace

import pytest

from vllm.v1.simple_kv_offload import copy_backend
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend


@pytest.mark.skip_global_cleanup
def test_copy_loop_preallocates_timing_events_outside_copy_loop(monkeypatch):
    """The copy thread should not allocate timing events per copy item.

    This guards against allocating ``torch.Event(enable_timing=True)`` inside
    the hot copy loop. The pool should be populated before the thread starts
    consuming copy items, and copies within that pool size should not allocate
    additional events.
    """

    allocation_count = 0

    class FakeEvent:
        def __init__(self, enable_timing=False):
            nonlocal allocation_count
            assert enable_timing is True
            allocation_count += 1

        def record(self, stream):
            pass

    monkeypatch.setattr(copy_backend.torch, "Event", FakeEvent)
    monkeypatch.setattr(
        copy_backend.current_platform, "set_device", lambda device: None
    )
    monkeypatch.setattr(copy_backend, "copy_blocks", lambda *args, **kwargs: None)

    q: queue.Queue = queue.Queue()
    events = []
    errors = []

    def run_copy_loop():
        try:
            DmaCopyBackend._copy_loop(
                q,
                device="cuda:0",  # type: ignore[arg-type]
                load_stream="load_stream",  # type: ignore[arg-type]
                store_stream="store_stream",  # type: ignore[arg-type]
            )
        except Exception as exc:  # pragma: no cover - re-raised by assertion
            errors.append(exc)

    thread = threading.Thread(target=run_copy_loop)
    thread.start()

    expected_initial_events = DmaCopyBackend._EVENT_POOL_INITIAL_SIZE * 2
    deadline = time.monotonic() + 5
    while allocation_count < expected_initial_events and time.monotonic() < deadline:
        time.sleep(0.01)

    assert errors == []
    assert allocation_count == expected_initial_events

    params = SimpleNamespace()
    for event_idx in range(3):
        q.put(
            (
                [event_idx],
                [event_idx],
                params,
                False,
                event_idx,
                events,
                1024,
            )
        )
    q.put(None)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert len(events) == 3
    assert allocation_count == expected_initial_events
