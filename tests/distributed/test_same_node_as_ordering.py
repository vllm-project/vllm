# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordering regression tests for `in_the_same_node_as` (#53718)."""

import pytest

from vllm.distributed import parallel_state


class _RecordingShm:
    def __init__(self, events: list[str], name: str = "seg"):
        self.events = events
        self.name = name
        self.buf = bytearray(128)

    def close(self) -> None:
        self.events.append("close")

    def unlink(self) -> None:
        self.events.append("unlink")


class _FakeStatelessGroup:
    def __init__(
        self,
        events: list[str],
        rank: int,
        world_size: int,
        fail_barrier: bool = False,
    ):
        self.events = events
        self.rank = rank
        self.world_size = world_size
        self.fail_barrier = fail_barrier

    def broadcast_obj(self, obj, src):
        return "seg" if obj is None else obj

    def barrier(self) -> None:
        self.events.append("barrier")
        if self.fail_barrier:
            raise RuntimeError("barrier failed")


def _patch_shm(monkeypatch, events: list[str]) -> None:
    def factory(*args, **kwargs):
        return _RecordingShm(events)

    monkeypatch.setattr(parallel_state.shared_memory, "SharedMemory", factory)


def test_source_rank_holds_shm_until_after_barrier(monkeypatch):
    """The creator must not close its handle before peers have opened it."""
    events: list[str] = []
    _patch_shm(monkeypatch, events)
    pg = _FakeStatelessGroup(events, rank=0, world_size=1)

    assert parallel_state.in_the_same_node_as(pg) == [True]
    assert events == ["barrier", "close", "unlink"]


def test_peer_rank_closes_but_does_not_unlink(monkeypatch):
    events: list[str] = []
    _patch_shm(monkeypatch, events)
    pg = _FakeStatelessGroup(events, rank=1, world_size=2)

    assert parallel_state.in_the_same_node_as(pg) == [False, False]
    assert events == ["barrier", "close"]


def test_cleanup_still_runs_when_barrier_fails(monkeypatch):
    events: list[str] = []
    _patch_shm(monkeypatch, events)
    pg = _FakeStatelessGroup(events, rank=0, world_size=1, fail_barrier=True)

    with pytest.raises(RuntimeError, match="barrier failed"):
        parallel_state.in_the_same_node_as(pg)
    assert events == ["barrier", "close", "unlink"]
