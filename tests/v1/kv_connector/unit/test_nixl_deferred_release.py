# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeferredBlockRelease: blocks of a finished request are reported free only
after the guard event recorded GUARD_AGE steps later has completed."""

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    DeferredBlockRelease,
)


class _FakeEvent:
    recorded: list["_FakeEvent"] = []

    def __init__(self):
        self.done = False

    def record(self):
        _FakeEvent.recorded.append(self)

    def query(self):
        return self.done


@pytest.fixture
def fake_events(monkeypatch):
    _FakeEvent.recorded = []
    monkeypatch.setattr("torch.cuda.Event", _FakeEvent)
    return _FakeEvent


def test_release_waits_for_guard_event(fake_events):
    dbr = DeferredBlockRelease()
    dbr.on_step(["a"])  # step 0: request a finished
    assert dbr.ready() == set()
    dbr.on_step([])  # step 1: too early for a guard
    assert dbr.ready() == set() and fake_events.recorded == []
    dbr.on_step(["b"])  # step 2: a gets its guard event, b arrives
    assert len(fake_events.recorded) == 1
    assert dbr.ready() == set()  # guard not complete yet
    fake_events.recorded[0].done = True
    assert dbr.ready() == {"a"}
    assert dbr.ready() == set()  # released once
    dbr.on_step([])
    dbr.on_step([])  # b's guard (one event shared by all requests guarded in a step)
    assert len(fake_events.recorded) == 2
    fake_events.recorded[1].done = True
    assert dbr.ready() == {"b"}


def test_no_event_recorded_without_pending(fake_events):
    dbr = DeferredBlockRelease()
    for _ in range(3):
        dbr.on_step([])
    assert fake_events.recorded == [] and dbr.ready() == set()
