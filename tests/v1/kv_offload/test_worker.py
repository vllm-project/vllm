# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
    TransferSpec,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CpuOffloadingWorker


class DummyLoadStoreSpec(LoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "dummy"


def _make_spec() -> TransferSpec:
    return (DummyLoadStoreSpec(), DummyLoadStoreSpec())


class FakeDirectionHandler:
    """Stand-in for SingleDirectionOffloadingHandler.

    Records submitted transfers and fan-out calls so we can assert how
    CpuOffloadingWorker routes between its two internal handlers.
    """

    def __init__(self):
        self.transfers: list[tuple[int, TransferSpec]] = []
        self.waited: list[set[int]] = []
        self.shutdown_called = False
        self._finished: list[TransferResult] = []

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        self.transfers.append((job_id, spec))
        return True

    def get_finished(self) -> list[TransferResult]:
        finished = self._finished
        self._finished = []
        return finished

    def wait(self, job_ids: set[int]) -> None:
        self.waited.append(set(job_ids))

    def shutdown(self) -> None:
        self.shutdown_called = True


def _make_worker() -> tuple[
    CpuOffloadingWorker, FakeDirectionHandler, FakeDirectionHandler
]:
    """Build a CpuOffloadingWorker with fake handlers.

    CpuOffloadingWorker.__init__ allocates GPU/CPU tensors, so we bypass it
    and inject fakes to exercise the (CPU-runnable) routing/aggregation logic
    added by this refactor.
    """
    worker = object.__new__(CpuOffloadingWorker)
    store_handler = FakeDirectionHandler()
    load_handler = FakeDirectionHandler()
    worker._store_handler = store_handler  # type: ignore[assignment]
    worker._load_handler = load_handler  # type: ignore[assignment]
    return worker, store_handler, load_handler


def test_offloading_worker_abc_cannot_instantiate():
    """OffloadingWorker is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        OffloadingWorker()  # type: ignore[abstract]


def test_submit_store_routes_to_store_handler():
    """submit_store drives only the GPU->CPU (store) handler."""
    worker, store_handler, load_handler = _make_worker()
    spec = _make_spec()

    assert worker.submit_store(1, spec)

    assert store_handler.transfers == [(1, spec)]
    assert load_handler.transfers == []


def test_submit_load_routes_to_load_handler():
    """submit_load drives only the CPU->GPU (load) handler."""
    worker, store_handler, load_handler = _make_worker()
    spec = _make_spec()

    assert worker.submit_load(2, spec)

    assert load_handler.transfers == [(2, spec)]
    assert store_handler.transfers == []


def test_get_finished_merges_both_handlers():
    """get_finished aggregates results from both internal handlers."""
    worker, store_handler, load_handler = _make_worker()
    store_result = TransferResult(job_id=1, success=True)
    load_result = TransferResult(job_id=2, success=True)
    store_handler._finished = [store_result]
    load_handler._finished = [load_result]

    assert worker.get_finished() == [store_result, load_result]
    # both handlers were drained
    assert worker.get_finished() == []


def test_wait_fans_out_to_both_handlers():
    """wait blocks on both internal handlers."""
    worker, store_handler, load_handler = _make_worker()

    worker.wait({1, 2})

    assert store_handler.waited == [{1, 2}]
    assert load_handler.waited == [{1, 2}]


def test_shutdown_fans_out_to_both_handlers():
    """shutdown releases both internal handlers."""
    worker, store_handler, load_handler = _make_worker()

    worker.shutdown()

    assert store_handler.shutdown_called
    assert load_handler.shutdown_called
