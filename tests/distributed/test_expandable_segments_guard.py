# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.device_communicators import expandable_segments as es


@pytest.mark.parametrize(
    "environ,expected",
    [
        ({}, False),
        ({"PYTORCH_ALLOC_CONF": "expandable_segments:True"}, True),
        ({"PYTORCH_ALLOC_CONF": "expandable_segments:true"}, True),
        ({"PYTORCH_ALLOC_CONF": "expandable_segments:1"}, True),
        ({"PYTORCH_ALLOC_CONF": "expandable_segments:False"}, False),
        ({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, True),
        (
            {"PYTORCH_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True"},
            True,
        ),
        ({"PYTORCH_ALLOC_CONF": "max_split_size_mb:512"}, False),
        ({"PYTORCH_ALLOC_CONF": "not_expandable_segments_really:True"}, False),
        ({"PYTORCH_ALLOC_CONF": ""}, False),
    ],
)
def test_env_parsing(environ, expected):
    assert es.expandable_segments_enabled_from_env(environ) is expected


class _FakeAllocator:
    def __init__(self, enabled: bool, honours_writes: bool = True):
        self.enabled = enabled
        self.honours_writes = honours_writes
        self.writes: list[bool] = []

    def get(self) -> bool:
        return self.enabled

    def set(self, value: bool) -> None:
        self.writes.append(value)
        if self.honours_writes:
            self.enabled = value


@pytest.fixture
def fake(monkeypatch):
    def _install(enabled: bool, honours_writes: bool = True) -> _FakeAllocator:
        alloc = _FakeAllocator(enabled, honours_writes)
        monkeypatch.setattr(es, "expandable_segments_enabled", alloc.get)
        monkeypatch.setattr(es, "_set_expandable_segments", alloc.set)
        monkeypatch.setattr(es, "_depth", 0)
        return alloc

    return _install


def test_disabled_flag_is_a_noop(fake):
    alloc = fake(enabled=True)
    with es.non_expandable_allocations(False) as applied:
        assert applied is False
        assert alloc.enabled is True
    assert alloc.writes == []


def test_noop_when_expandable_was_already_off(fake):
    alloc = fake(enabled=False)
    with es.non_expandable_allocations(True) as applied:
        assert applied is False
    assert alloc.writes == []


def test_toggles_off_and_restores(fake):
    alloc = fake(enabled=True)
    with es.non_expandable_allocations(True) as applied:
        assert applied is True
        assert alloc.enabled is False, "body must run with expandable segments off"
    assert alloc.enabled is True, "must be restored on exit"
    assert alloc.writes == [False, True]


def test_restores_even_if_body_raises(fake):
    alloc = fake(enabled=True)
    with pytest.raises(RuntimeError), es.non_expandable_allocations(True):
        assert alloc.enabled is False
        raise RuntimeError("capture blew up")
    assert alloc.enabled is True
    assert alloc.writes == [False, True]


def test_reentrant_only_outermost_toggles(fake):
    alloc = fake(enabled=True)
    with es.non_expandable_allocations(True) as outer:
        assert outer is True
        with es.non_expandable_allocations(True) as inner:
            assert inner is False
            assert alloc.enabled is False
        assert alloc.enabled is False, "inner exit must not restore"
    assert alloc.enabled is True
    assert alloc.writes == [False, True]


def test_depth_is_reset_after_exception(fake):
    alloc = fake(enabled=True)
    with pytest.raises(RuntimeError), es.non_expandable_allocations(True):
        raise RuntimeError("boom")
    with es.non_expandable_allocations(True) as applied:
        assert applied is True
    assert alloc.writes == [False, True, False, True]


def test_platform_ignoring_the_write_is_detected_and_not_restored(fake):
    alloc = fake(enabled=True, honours_writes=False)
    with es.non_expandable_allocations(True) as applied:
        assert applied is False
    assert alloc.writes == [False], "must not write True back"


def test_env_var_is_registered_and_defaults_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_NON_EXPANDABLE_CUDAGRAPH_POOL", raising=False)
    assert envs.VLLM_NON_EXPANDABLE_CUDAGRAPH_POOL is False
    monkeypatch.setenv("VLLM_NON_EXPANDABLE_CUDAGRAPH_POOL", "1")
    assert envs.VLLM_NON_EXPANDABLE_CUDAGRAPH_POOL is True
    monkeypatch.setenv("VLLM_NON_EXPANDABLE_CUDAGRAPH_POOL", "True")
    assert envs.VLLM_NON_EXPANDABLE_CUDAGRAPH_POOL is True
    monkeypatch.setenv("VLLM_NON_EXPANDABLE_CUDAGRAPH_POOL", "False")
    assert envs.VLLM_NON_EXPANDABLE_CUDAGRAPH_POOL is False


def test_graph_capture_enters_guard_before_group_contexts():
    import inspect

    from vllm.distributed import parallel_state

    src = inspect.getsource(parallel_state.graph_capture)
    assert "non_expandable_allocations" in src
    guard_at = src.index("non_expandable_allocations(envs.")
    tp_at = src.index("get_tp_group().graph_capture(context)")
    assert guard_at < tp_at
