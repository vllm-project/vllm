# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for asynchronous CPU cache initialization (``async_init``).

Scheduler-side gating and the rank-readiness handshake are generic and covered
by tests/v1/kv_connector/unit/test_connector_init_status.py; these tests cover
the SimpleCPUOffloadConnector wiring and the worker-side state machine.
"""

from __future__ import annotations

import mmap
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

import vllm.v1.simple_kv_offload.worker as worker_module
from tests.v1.simple_kv_offload.test_scheduler import (
    _make_kv_cache_config,
    _make_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    ConnectorInitState,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker, _CpuCaches

pytestmark = pytest.mark.cpu_test

_POLL_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_connector(
    role: KVConnectorRole, **extra_config: Any
) -> SimpleCPUOffloadConnector:
    vllm_config = _make_vllm_config()
    vllm_config.kv_transfer_config.kv_connector_extra_config = extra_config
    return SimpleCPUOffloadConnector(vllm_config, role, _make_kv_cache_config(16))


def _make_worker(async_init: bool = False) -> SimpleCPUOffloadWorker:
    return SimpleCPUOffloadWorker(
        MagicMock(), MagicMock(), cpu_capacity_bytes=1024, async_init=async_init
    )


def _wait_for_state(worker: SimpleCPUOffloadWorker) -> ConnectorInitState:
    """Poll until the background initialization resolves, driving adoption."""
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while (state := worker.get_init_state()) is ConnectorInitState.INITIALIZING:
        assert time.monotonic() < deadline, "async init did not resolve in time"
        time.sleep(0.01)
    return state


def _cuda_platform(monkeypatch: pytest.MonkeyPatch, is_cuda: bool) -> None:
    import vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector as connector_module  # noqa: E501

    monkeypatch.setattr(connector_module.current_platform, "is_cuda", lambda: is_cuda)


# ---------------------------------------------------------------------------
# Connector-level config validation and wiring
# ---------------------------------------------------------------------------


def test_async_init_must_be_boolean():
    with pytest.raises(ValueError, match="boolean"):
        _make_connector(KVConnectorRole.SCHEDULER, async_init="yes")


def test_async_init_rejected_with_disk_backend(monkeypatch, tmp_path):
    _cuda_platform(monkeypatch, True)
    with pytest.raises(ValueError, match="disk"):
        _make_connector(
            KVConnectorRole.SCHEDULER,
            async_init=True,
            kv_offload_backend="disk",
            disk_path=str(tmp_path / "kv"),
        )


def test_async_init_rejected_off_cuda(monkeypatch):
    _cuda_platform(monkeypatch, False)
    with pytest.raises(ValueError, match="CUDA"):
        _make_connector(KVConnectorRole.SCHEDULER, async_init=True)


def test_sync_connector_is_ready_immediately():
    connector = _make_connector(KVConnectorRole.WORKER)
    assert connector.is_connector_ready() is True
    assert connector.get_connector_init_state() is None


def test_async_worker_connector_reports_ready_after_init(monkeypatch):
    """Readiness must follow the worker handler's init state end to end."""
    _cuda_platform(monkeypatch, True)
    connector = _make_connector(KVConnectorRole.WORKER, async_init=True)
    assert connector.is_connector_ready() is False

    # Nothing to offload resolves initialization instantly.
    connector.register_kv_caches({})
    assert connector.get_connector_init_state() is ConnectorInitState.READY
    assert connector.is_connector_ready() is True

    status = connector.build_connector_init_status()
    assert status is not None and status.ready_ranks == {0}


# ---------------------------------------------------------------------------
# Worker-side state machine
# ---------------------------------------------------------------------------


def test_worker_sync_mode_has_no_init_state():
    assert _make_worker(async_init=False).get_init_state() is None


def test_worker_initializing_before_register_kv_caches():
    worker = _make_worker(async_init=True)
    assert worker.get_init_state() is ConnectorInitState.INITIALIZING


def test_worker_ready_when_nothing_to_offload():
    worker = _make_worker(async_init=True)
    worker.register_kv_caches({})
    assert worker.get_init_state() is ConnectorInitState.READY


def test_worker_adopts_caches_and_builds_backend_once(monkeypatch):
    backend_cls = MagicMock()
    monkeypatch.setattr(worker_module, "DmaCopyBackend", backend_cls)
    worker = _make_worker(async_init=True)
    worker.gpu_kv_caches = {"layer_0": torch.zeros(2, 32, dtype=torch.int8)}
    caches = _CpuCaches({"layer_0": torch.zeros(2, 32, dtype=torch.int8)}, [])
    buffer = MagicMock()
    buffer.poll.side_effect = [None, caches]
    buffer.failed = False
    worker._cpu_init = buffer

    assert worker.get_init_state() is ConnectorInitState.INITIALIZING
    assert worker.get_init_state() is ConnectorInitState.READY
    assert worker.cpu_kv_caches is caches.tensors

    # Readiness must stick, and the backend must be built exactly once.
    assert worker.get_init_state() is ConnectorInitState.READY
    backend_cls.assert_called_once()


def test_worker_backend_failure_at_adoption_is_terminal(monkeypatch):
    """A backend-init failure must raise, unpin, and stay terminal."""
    backend_cls = MagicMock()
    backend_cls.return_value.init.side_effect = RuntimeError("backend boom")
    monkeypatch.setattr(worker_module, "DmaCopyBackend", backend_cls)
    worker = _make_worker(async_init=True)
    worker.gpu_kv_caches = {"layer_0": torch.zeros(2, 32, dtype=torch.int8)}
    caches = _CpuCaches({"layer_0": torch.zeros(2, 32, dtype=torch.int8)}, [1000])
    buffer = MagicMock()
    buffer.poll.return_value = caches
    buffer.failed = False
    worker._cpu_init = buffer
    unregister = MagicMock()
    monkeypatch.setattr(worker_module, "host_unregister", unregister)

    with pytest.raises(RuntimeError, match="backend boom"):
        worker.get_init_state()
    unregister.assert_called_once_with([1000])
    assert worker.cpu_kv_caches is None
    with pytest.raises(RuntimeError, match="initialization failed"):
        worker.get_init_state()


def test_worker_shutdown_closes_unadopted_buffer():
    worker = _make_worker(async_init=True)
    buffer = MagicMock()
    worker._cpu_init = buffer
    worker.shutdown()
    buffer.close.assert_called_once()


def test_allocation_passes_copy_unit_and_aligned_base(monkeypatch):
    """The worker must hand pin_tensor a page-aligned tensor and the block-row
    size as the copy unit, so no block copy can straddle pinned regions."""
    pins: list[tuple[int, int, int | None, int | None]] = []

    def _record_pin(
        tensor: torch.Tensor,
        chunk: int | None = None,
        copy_unit_bytes: int | None = None,
    ) -> list[int]:
        pins.append((tensor.data_ptr(), tensor.nbytes, chunk, copy_unit_bytes))
        return []

    monkeypatch.setattr(worker_module, "pin_tensor", _record_pin)
    shape, dtype = (7, 61, 256), torch.float16  # row = 31232 B, not a page multiple
    caches = worker_module._allocate_cpu_caches(
        {"layer_0": (shape, dtype)}, pin_memory=True, chunk_size_bytes=256 * 2**20
    )

    ((ptr, nbytes, _chunk, copy_unit),) = pins
    row_bytes = 61 * 256 * dtype.itemsize
    assert ptr % mmap.PAGESIZE == 0
    assert copy_unit == row_bytes
    tensor = caches.tensors["layer_0"]
    assert tensor.data_ptr() == ptr and tensor.shape == shape
    assert nbytes == 7 * row_bytes


# ---------------------------------------------------------------------------
# Background allocation through a real AsyncHostBuffer
# ---------------------------------------------------------------------------


def test_async_build_starts_at_first_poll_not_at_register(monkeypatch):
    """register_kv_caches must not start the background build: engine warmup
    captures CUDA graphs after it, and a concurrent cudaHostRegister
    invalidates an in-progress capture."""
    monkeypatch.setattr(worker_module, "PIN_MEMORY", False)
    builds: list[MagicMock] = []

    def _fake_buffer(**kwargs: Any) -> MagicMock:
        buffer = MagicMock()
        buffer.poll.return_value = None
        buffer.failed = False
        builds.append(buffer)
        return buffer

    monkeypatch.setattr(worker_module, "AsyncHostBuffer", _fake_buffer)
    worker = _make_worker(async_init=True)
    worker.gpu_kv_caches = {"layer_0": torch.zeros(4, 32, dtype=torch.int8)}
    worker.num_cpu_blocks = 2

    worker._init_cpu_mode(worker.gpu_kv_caches, 32)
    assert not builds

    assert worker.get_init_state() is ConnectorInitState.INITIALIZING
    assert len(builds) == 1


def test_async_allocation_resolves_to_ready(monkeypatch):
    monkeypatch.setattr(worker_module, "PIN_MEMORY", False)
    monkeypatch.setattr(worker_module, "DmaCopyBackend", MagicMock())
    worker = _make_worker(async_init=True)
    worker.gpu_kv_caches = {"layer_0": torch.zeros(4, 32, dtype=torch.int8)}
    worker.num_cpu_blocks = 2

    worker._init_cpu_mode(worker.gpu_kv_caches, 32)

    assert _wait_for_state(worker) is ConnectorInitState.READY
    assert worker.cpu_kv_caches is not None
    assert worker.cpu_kv_caches["layer_0"].shape == (2, 32)


def test_async_allocation_failure_is_terminal(monkeypatch):
    monkeypatch.setattr(worker_module, "PIN_MEMORY", False)

    def boom(*_args: Any, **_kwargs: Any) -> _CpuCaches:
        raise RuntimeError("allocation boom")

    monkeypatch.setattr(worker_module, "_allocate_cpu_caches", boom)
    worker = _make_worker(async_init=True)
    worker.gpu_kv_caches = {"layer_0": torch.zeros(4, 32, dtype=torch.int8)}
    worker.num_cpu_blocks = 2

    worker._init_cpu_mode(worker.gpu_kv_caches, 32)

    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            state = worker.get_init_state()
        except RuntimeError:
            break
        assert state is ConnectorInitState.INITIALIZING
        time.sleep(0.01)
    else:
        pytest.fail("failed async init never raised")
    with pytest.raises(RuntimeError, match="initialization failed"):
        worker.get_init_state()
