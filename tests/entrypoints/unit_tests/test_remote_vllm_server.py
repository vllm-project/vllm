# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

import tests.utils as test_utils
from tests.utils import (
    RemoteLaunchRenderServer,
    RemoteOpenAIServer,
    RemoteVLLMServer,
)

GIB = 1024**3
GPU_SCOPE = ("cuda", ("0",))


@pytest.fixture(autouse=True)
def reset_remote_server_cleanup_state() -> Iterator[None]:
    with RemoteVLLMServer._active_servers_lock:
        saved_failures = RemoteVLLMServer._failed_gpu_cleanup.copy()
        saved_active_servers = RemoteVLLMServer._active_servers.copy()
        RemoteVLLMServer._failed_gpu_cleanup.clear()
        RemoteVLLMServer._active_servers.clear()
    try:
        yield
    finally:
        with RemoteVLLMServer._active_servers_lock:
            RemoteVLLMServer._failed_gpu_cleanup.clear()
            RemoteVLLMServer._failed_gpu_cleanup.update(saved_failures)
            RemoteVLLMServer._active_servers.clear()
            RemoteVLLMServer._active_servers.update(saved_active_servers)


def make_server(memory_used: float | None) -> RemoteVLLMServer:
    server = object.__new__(RemoteVLLMServer)
    server._gpu_device_scope = GPU_SCOPE
    server._pre_server_gpu_memory = 1 * GIB
    server._pre_server_gpu_memory_by_device = {"0": 1 * GIB}
    server._get_gpu_memory_used = (  # type: ignore[method-assign]
        lambda device_ids=None: memory_used
    )
    return server


def test_gpu_device_ids_use_canonical_child_visibility(
    monkeypatch: pytest.MonkeyPatch,
):
    server = object.__new__(RemoteVLLMServer)
    monkeypatch.setattr(
        test_utils,
        "current_platform",
        SimpleNamespace(
            is_rocm=lambda: True,
            is_cuda=lambda: False,
            is_xpu=lambda: False,
            device_control_env_var="TEST_VISIBLE_DEVICES",
            device_type="cuda",
        ),
    )

    assert server._get_gpu_device_ids({"TEST_VISIBLE_DEVICES": "3,1,3"}) == ("1", "3")


def test_xpu_device_ids_use_parent_logical_ordinals(
    monkeypatch: pytest.MonkeyPatch,
):
    server = object.__new__(RemoteVLLMServer)
    monkeypatch.setattr(
        test_utils,
        "current_platform",
        SimpleNamespace(
            is_rocm=lambda: False,
            is_cuda=lambda: False,
            is_xpu=lambda: True,
            device_count=lambda: 2,
            device_control_env_var="ZE_AFFINITY_MASK",
            device_type="xpu",
        ),
    )

    assert server._get_gpu_device_ids({"ZE_AFFINITY_MASK": "2.0,3.0"}) == (
        "0",
        "1",
    )


def test_gpu_release_timeout_records_failed_cleanup():
    server = make_server(5 * GIB)

    with pytest.raises(RuntimeError, match="GPU memory did not release"):
        server._wait_for_gpu_memory_release(timeout=0)

    assert RemoteVLLMServer._failed_gpu_cleanup[GPU_SCOPE] == (
        1 * GIB,
        3 * GIB,
        5 * GIB,
    )


def test_gpu_release_timeout_with_unavailable_telemetry_fails_closed():
    server = make_server(None)

    with pytest.raises(RuntimeError, match="Current: unavailable"):
        server._wait_for_gpu_memory_release(timeout=0)

    assert RemoteVLLMServer._failed_gpu_cleanup[GPU_SCOPE] == (
        1 * GIB,
        3 * GIB,
        None,
    )


@pytest.mark.parametrize("memory_used", [5 * GIB, None])
def test_failed_cleanup_blocks_next_server(memory_used: float | None):
    RemoteVLLMServer._failed_gpu_cleanup[GPU_SCOPE] = (
        1 * GIB,
        3 * GIB,
        5 * GIB,
    )
    server = make_server(memory_used)

    with pytest.raises(RuntimeError, match="Refusing to start"):
        server._ensure_failed_gpu_cleanup_recovered()

    assert GPU_SCOPE in RemoteVLLMServer._failed_gpu_cleanup


def test_recovered_gpu_memory_clears_failed_cleanup():
    RemoteVLLMServer._failed_gpu_cleanup[GPU_SCOPE] = (
        1 * GIB,
        3 * GIB,
        5 * GIB,
    )
    server = make_server(3 * GIB)

    server._ensure_failed_gpu_cleanup_recovered()

    assert GPU_SCOPE not in RemoteVLLMServer._failed_gpu_cleanup


def test_active_server_defers_failed_cleanup_check():
    active_server = make_server(5 * GIB)
    with RemoteVLLMServer._active_servers_lock:
        RemoteVLLMServer._active_servers.add(active_server)
        RemoteVLLMServer._failed_gpu_cleanup[GPU_SCOPE] = (
            1 * GIB,
            3 * GIB,
            5 * GIB,
        )
    server = make_server(5 * GIB)

    server._ensure_failed_gpu_cleanup_recovered()

    assert GPU_SCOPE in RemoteVLLMServer._failed_gpu_cleanup


def test_partially_covered_failed_scope_blocks_new_server():
    failed_scope = ("cuda", ("0", "1"))
    active_server = make_server(5 * GIB)
    active_server._gpu_device_scope = ("cuda", ("1",))
    with RemoteVLLMServer._active_servers_lock:
        RemoteVLLMServer._active_servers.add(active_server)
        RemoteVLLMServer._failed_gpu_cleanup[failed_scope] = (
            2 * GIB,
            4 * GIB,
            6 * GIB,
        )
    server = make_server(5 * GIB)

    with pytest.raises(RuntimeError, match="cover only part"):
        server._ensure_failed_gpu_cleanup_recovered()

    assert failed_scope in RemoteVLLMServer._failed_gpu_cleanup


def test_overlapping_scope_cannot_bypass_failed_cleanup():
    RemoteVLLMServer._failed_gpu_cleanup[GPU_SCOPE] = (
        1 * GIB,
        3 * GIB,
        5 * GIB,
    )
    server = make_server(5 * GIB)
    server._gpu_device_scope = ("cuda", ("0", "1"))
    measured_device_ids = []
    server._get_gpu_memory_used = (  # type: ignore[method-assign]
        lambda device_ids=None: measured_device_ids.append(device_ids) or 5 * GIB
    )

    with pytest.raises(RuntimeError, match="Refusing to start"):
        server._ensure_failed_gpu_cleanup_recovered()

    assert GPU_SCOPE in RemoteVLLMServer._failed_gpu_cleanup
    assert measured_device_ids == [("0",)]


def test_render_server_has_no_gpu_cleanup_scope():
    server = object.__new__(RemoteLaunchRenderServer)

    assert server._get_gpu_device_ids(None) is None


def test_failed_cleanup_keeps_lowest_recovery_target():
    server = make_server(5 * GIB)
    server._record_failed_gpu_cleanup(2 * GIB, 4 * GIB, 6 * GIB)
    server._record_failed_gpu_cleanup(1 * GIB, 3 * GIB, 5 * GIB)

    assert RemoteVLLMServer._failed_gpu_cleanup[GPU_SCOPE] == (
        1 * GIB,
        3 * GIB,
        5 * GIB,
    )


def test_shutdown_many_waits_once_per_device_scope():
    terminated = []
    waited = []

    def make_shutdown_server(
        name: str,
        scope: tuple[str, tuple[str, ...]],
        baseline: float,
        baselines_by_device: dict[str, float] | None = None,
    ) -> RemoteVLLMServer:
        server = make_server(0)
        server._gpu_device_scope = scope
        server._pre_server_gpu_memory = baseline
        server._pre_server_gpu_memory_by_device = baselines_by_device or {
            scope[1][0]: baseline
        }
        server.proc = SimpleNamespace(pid=name)  # type: ignore[assignment]
        server._terminate_process_tree = (  # type: ignore[method-assign]
            lambda: terminated.append(name)
        )
        server._wait_for_gpu_memory_release = (  # type: ignore[method-assign]
            lambda **kwargs: waited.append((name, kwargs))
        )
        return server

    first = make_shutdown_server("first", ("cuda", ("0",)), 1 * GIB)
    later = make_shutdown_server("later", ("cuda", ("0",)), 2 * GIB)
    other = make_shutdown_server("other", ("cuda", ("1",)), 1 * GIB)

    RemoteVLLMServer.shutdown_many([later, other, first])

    assert sorted(terminated) == ["first", "later", "other"]
    assert sorted(name for name, _ in waited) == ["first", "other"]


def test_shutdown_many_uses_per_device_baseline_for_overlapping_scopes():
    waited = []

    def make_shutdown_server(
        name: str,
        scope: tuple[str, tuple[str, ...]],
        baselines: dict[str, float],
    ) -> RemoteVLLMServer:
        server = make_server(0)
        server._gpu_device_scope = scope
        server._pre_server_gpu_memory_by_device = baselines
        server._pre_server_gpu_memory = sum(baselines.values())
        server.proc = SimpleNamespace(pid=name)  # type: ignore[assignment]
        server._terminate_process_tree = lambda: None  # type: ignore[method-assign]
        server._wait_for_gpu_memory_release = (  # type: ignore[method-assign]
            lambda **kwargs: waited.append(kwargs)
        )
        return server

    first = make_shutdown_server("first", ("cuda", ("0",)), {"0": 1 * GIB})
    broader = make_shutdown_server(
        "broader",
        ("cuda", ("0", "1")),
        {"0": 10 * GIB, "1": 1 * GIB},
    )

    RemoteVLLMServer.shutdown_many([broader, first])

    assert waited == [
        {
            "device_ids": ("0", "1"),
            "baseline": 2 * GIB,
            "cleanup_scope": ("cuda", ("0", "1")),
        }
    ]


@pytest.mark.parametrize(
    ("request_timeout", "process_timeout", "expected_timeout"),
    [(0, 60.0, 75.0), (0, 0, 15.0), (30, 30, 45.0)],
)
def test_openai_server_wait_covers_engine_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    request_timeout: float,
    process_timeout: float,
    expected_timeout: float,
):
    server = object.__new__(RemoteOpenAIServer)
    server._request_shutdown_timeout = request_timeout
    monkeypatch.setattr(
        "tests.utils.get_engine_process_shutdown_timeout",
        lambda request_timeout, manager_timeout: process_timeout,
    )

    assert server._get_process_termination_timeout() == expected_timeout
