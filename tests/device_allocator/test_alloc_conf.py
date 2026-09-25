# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.device_allocator import alloc_conf

pytestmark = pytest.mark.cpu_test

ES = alloc_conf.EXPANDABLE_SEGMENTS


@pytest.mark.parametrize(
    "environ,expected",
    [
        ({}, False),
        ({"PYTORCH_ALLOC_CONF": "expandable_segments:True"}, True),
        ({"PYTORCH_ALLOC_CONF": "expandable_segments:true"}, True),
        ({"PYTORCH_ALLOC_CONF": "expandable_segments:1"}, True),
        ({"PYTORCH_ALLOC_CONF": "expandable_segments:False"}, False),
        ({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, True),
        # ROCm users set this one; it was previously not looked at.
        ({"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}, True),
        (
            {"PYTORCH_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True"},
            True,
        ),
        ({"PYTORCH_ALLOC_CONF": "max_split_size_mb:512"}, False),
        ({"PYTORCH_ALLOC_CONF": "not_expandable_segments_really:True"}, False),
        ({"PYTORCH_ALLOC_CONF": ""}, False),
        # torch's precedence: the accelerator-agnostic variable wins.
        (
            {
                "PYTORCH_ALLOC_CONF": "expandable_segments:False",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
            False,
        ),
    ],
)
def test_expandable_segments_from_env(environ, expected):
    assert alloc_conf.expandable_segments_enabled_from_env(environ) is expected


@pytest.mark.parametrize(
    "conf,key,expected",
    [
        ("", ES, False),
        ("expandable_segments:True", ES, True),
        ("expandable_segments:TRUE", ES, True),
        ("expandable_segments:1", ES, True),
        ("expandable_segments:0", ES, False),
        (" expandable_segments : True ", ES, True),
        ("max_split_size_mb:512", ES, False),
        ("max_split_size_mb:512", "max_split_size_mb", False),
        ("garbage_collection_threshold:0.9", ES, False),
    ],
)
def test_conf_flag_enabled(conf, key, expected):
    assert alloc_conf.conf_flag_enabled(conf, key) is expected


@pytest.mark.parametrize(
    "conf,enabled,expected",
    [
        ("", False, "expandable_segments:False"),
        ("expandable_segments:True", False, "expandable_segments:False"),
        ("expandable_segments:False", True, "expandable_segments:True"),
        (
            "max_split_size_mb:512,expandable_segments:True",
            False,
            "max_split_size_mb:512,expandable_segments:False",
        ),
        (
            "garbage_collection_threshold:0.9,max_split_size_mb:512",
            False,
            "garbage_collection_threshold:0.9,max_split_size_mb:512,"
            "expandable_segments:False",
        ),
    ],
)
def test_with_conf_flag_preserves_other_fields(conf, enabled, expected):
    assert alloc_conf.with_conf_flag(conf, ES, enabled) == expected


def test_with_conf_flag_round_trips():
    """The regression this guards: torch resets every option absent from the
    string it is handed, so a one-field write drops the rest permanently."""
    conf = (
        "max_split_size_mb:512,expandable_segments:True,"
        "garbage_collection_threshold:0.9"
    )
    off = alloc_conf.with_conf_flag(conf, ES, False)
    assert "max_split_size_mb:512" in off
    assert "garbage_collection_threshold:0.9" in off
    assert alloc_conf.with_conf_flag(off, ES, True) == conf


def test_alloc_conf_from_env_precedence():
    assert (
        alloc_conf.alloc_conf_from_env(
            {
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
                "PYTORCH_HIP_ALLOC_CONF": "max_split_size_mb:128",
            }
        )
        == "max_split_size_mb:256"
    )
    assert alloc_conf.alloc_conf_from_env({}) == ""


def test_current_alloc_conf_prefers_live_state(monkeypatch):
    monkeypatch.setattr(alloc_conf, "live_alloc_conf", lambda: "max_split_size_mb:64")
    monkeypatch.setattr(alloc_conf, "alloc_conf_from_env", lambda: "stale:True")
    assert alloc_conf.current_alloc_conf() == "max_split_size_mb:64"

    monkeypatch.setattr(alloc_conf, "live_alloc_conf", lambda: None)
    assert alloc_conf.current_alloc_conf() == "stale:True"


def test_expandable_segments_enabled_returns_none_when_unreadable(monkeypatch):
    """'cannot read' must stay distinct from 'disabled'."""

    class _NoSnapshot:
        @staticmethod
        def _snapshot():
            raise RuntimeError("no allocator here")

    monkeypatch.setattr(alloc_conf.torch.cuda, "memory", _NoSnapshot)
    assert alloc_conf.expandable_segments_enabled() is None


class _FakeAllocator:
    """Stands in for the torch allocator: records writes, reports live state."""

    def __init__(self, conf: str, honours_writes: bool = True, readable: bool = True):
        self.conf = conf
        self.honours_writes = honours_writes
        self.readable = readable
        self.writes: list[str] = []

    def read(self) -> bool | None:
        if not self.readable:
            return None
        return alloc_conf.conf_flag_enabled(self.conf, ES)

    def write(self, conf: str) -> None:
        self.writes.append(conf)
        if self.honours_writes:
            self.conf = conf


@pytest.fixture
def fake(monkeypatch):
    def _install(conf: str, honours_writes: bool = True, readable: bool = True):
        alloc = _FakeAllocator(conf, honours_writes, readable)
        monkeypatch.setattr(alloc_conf, "expandable_segments_enabled", alloc.read)
        monkeypatch.setattr(alloc_conf, "set_alloc_conf", alloc.write)
        monkeypatch.setattr(alloc_conf, "current_alloc_conf", lambda: conf)
        monkeypatch.setattr(alloc_conf, "_non_expandable_depth", 0)
        return alloc

    return _install


def test_guard_disabled_flag_is_a_noop(fake):
    alloc = fake("expandable_segments:True")
    with alloc_conf.non_expandable_allocations(False) as applied:
        assert applied is False
    assert alloc.writes == []


def test_guard_noop_when_already_off(fake):
    alloc = fake("expandable_segments:False")
    with alloc_conf.non_expandable_allocations(True) as applied:
        assert applied is False
    assert alloc.writes == []


def test_guard_toggles_off_and_restores(fake):
    alloc = fake("expandable_segments:True")
    with alloc_conf.non_expandable_allocations(True) as applied:
        assert applied is True
        assert alloc.read() is False, "body must run with expandable segments off"
    assert alloc.read() is True, "must be restored on exit"
    assert alloc.writes == ["expandable_segments:False", "expandable_segments:True"]


def test_guard_preserves_other_allocator_fields(fake):
    conf = (
        "max_split_size_mb:512,garbage_collection_threshold:0.9,"
        "expandable_segments:True"
    )
    alloc = fake(conf)
    with alloc_conf.non_expandable_allocations(True):
        assert "max_split_size_mb:512" in alloc.conf
        assert "garbage_collection_threshold:0.9" in alloc.conf
    assert alloc.conf == conf


def test_guard_restores_when_the_body_raises(fake):
    alloc = fake("expandable_segments:True")
    with pytest.raises(RuntimeError), alloc_conf.non_expandable_allocations(True):
        assert alloc.read() is False
        raise RuntimeError("capture blew up")
    assert alloc.read() is True


def test_guard_is_reentrant_and_only_the_outermost_toggles(fake):
    alloc = fake("expandable_segments:True")
    with alloc_conf.non_expandable_allocations(True) as outer:
        assert outer is True
        with alloc_conf.non_expandable_allocations(True) as inner:
            assert inner is False
            assert alloc.read() is False
        assert alloc.read() is False, "inner exit must not restore"
    assert alloc.read() is True
    assert len(alloc.writes) == 2


def test_guard_depth_is_reset_after_an_exception(fake):
    alloc = fake("expandable_segments:True")
    with pytest.raises(RuntimeError), alloc_conf.non_expandable_allocations(True):
        raise RuntimeError("boom")
    with alloc_conf.non_expandable_allocations(True) as applied:
        assert applied is True
    assert len(alloc.writes) == 4


def test_guard_restores_even_when_the_write_was_ignored(fake):
    """A platform that drops the write must not be left mid-toggle."""
    alloc = fake("expandable_segments:True", honours_writes=False)
    with alloc_conf.non_expandable_allocations(True) as applied:
        assert applied is False
    assert alloc.writes == [
        "expandable_segments:False",
        "expandable_segments:True",
    ], "the restore must be unconditional"


def test_guard_restores_when_live_state_is_unreadable(fake):
    """Treating 'cannot read' as 'the write failed' would leave expandable
    segments off for the rest of the process."""
    alloc = fake("expandable_segments:True", readable=False)
    with alloc_conf.non_expandable_allocations(True) as applied:
        assert applied is True
    assert alloc.writes == ["expandable_segments:False", "expandable_segments:True"]


def test_guard_degrades_when_the_setter_raises(fake, monkeypatch):
    alloc = fake("expandable_segments:True")

    def boom(conf):
        raise RuntimeError("torch said no")

    monkeypatch.setattr(alloc_conf, "set_alloc_conf", boom)
    with alloc_conf.non_expandable_allocations(True) as applied:
        assert applied is False
    assert alloc.conf == "expandable_segments:True"


def test_env_var_is_registered_and_defaults_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_ROCM_NON_EXPANDABLE_CUDAGRAPH_POOL", raising=False)
    assert envs.VLLM_ROCM_NON_EXPANDABLE_CUDAGRAPH_POOL is False
    monkeypatch.setenv("VLLM_ROCM_NON_EXPANDABLE_CUDAGRAPH_POOL", "1")
    assert envs.VLLM_ROCM_NON_EXPANDABLE_CUDAGRAPH_POOL is True
    monkeypatch.setenv("VLLM_ROCM_NON_EXPANDABLE_CUDAGRAPH_POOL", "True")
    assert envs.VLLM_ROCM_NON_EXPANDABLE_CUDAGRAPH_POOL is True
    monkeypatch.setenv("VLLM_ROCM_NON_EXPANDABLE_CUDAGRAPH_POOL", "False")
    assert envs.VLLM_ROCM_NON_EXPANDABLE_CUDAGRAPH_POOL is False


def test_guard_is_off_when_the_env_var_is_unset(monkeypatch):
    from vllm.distributed import parallel_state

    monkeypatch.delenv("VLLM_ROCM_NON_EXPANDABLE_CUDAGRAPH_POOL", raising=False)
    assert parallel_state._use_non_expandable_graph_pool() is False


def test_guard_is_off_when_aiter_cannot_consume_it(monkeypatch):
    """The flag must no-op rather than silently claim a win the installed
    AITER cannot deliver."""
    from vllm.distributed import parallel_state
    from vllm.distributed.device_communicators import aiter_custom_all_reduce
    from vllm.platforms import current_platform

    monkeypatch.setenv("VLLM_ROCM_NON_EXPANDABLE_CUDAGRAPH_POOL", "1")
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        aiter_custom_all_reduce.AiterCustomAllreduce,
        "build_defers_capture_registration",
        staticmethod(lambda: False),
    )
    assert parallel_state._use_non_expandable_graph_pool() is False

    monkeypatch.setattr(
        aiter_custom_all_reduce.AiterCustomAllreduce,
        "build_defers_capture_registration",
        staticmethod(lambda: True),
    )
    assert parallel_state._use_non_expandable_graph_pool() is True


def test_graph_capture_enters_the_guard_before_the_group_contexts():
    """The guard must wrap the group contexts, not sit inside them: the
    communicators allocate their capture buffers when they are entered.

    Asserted on the parsed `with` items rather than on source text, so that
    reformatting cannot break it.
    """
    import ast
    import inspect

    from vllm.distributed import parallel_state

    src = inspect.getsource(parallel_state.graph_capture)
    fn = ast.parse(inspect.cleandoc(src)).body[0]
    withs = [n for n in ast.walk(fn) if isinstance(n, ast.With)]
    assert withs, "graph_capture should use a with statement"
    calls = [ast.unparse(item.context_expr) for item in withs[0].items]
    guard_at = next(
        (i for i, c in enumerate(calls) if "non_expandable_allocations" in c), None
    )
    assert guard_at is not None, calls
    tp_at = next((i for i, c in enumerate(calls) if "get_tp_group" in c), None)
    assert tp_at is not None, calls
    assert guard_at < tp_at, calls


def test_guard_keeps_the_environment_in_step(fake, monkeypatch):
    """A consumer that falls back to parsing the env must not read a stale
    True while the live allocator says False -- across ranks of one job that
    is a collective mismatch, not a local inefficiency."""
    fake("expandable_segments:True")
    monkeypatch.setenv(
        "PYTORCH_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:True"
    )
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_HIP_ALLOC_CONF", raising=False)

    import os

    with alloc_conf.non_expandable_allocations(True) as applied:
        assert applied is True
        assert (
            os.environ["PYTORCH_ALLOC_CONF"]
            == "max_split_size_mb:512,expandable_segments:False"
        )
    assert (
        os.environ["PYTORCH_ALLOC_CONF"]
        == "max_split_size_mb:512,expandable_segments:True"
    )


def test_guard_does_not_invent_env_vars_that_were_unset(fake, monkeypatch):
    fake("expandable_segments:True")
    for name in alloc_conf.ALLOC_CONF_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    import os

    with alloc_conf.non_expandable_allocations(True):
        for name in alloc_conf.ALLOC_CONF_ENV_VARS:
            assert name not in os.environ
