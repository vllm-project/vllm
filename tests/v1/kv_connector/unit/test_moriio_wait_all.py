# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transfer-completion semantics of the MoRIIO wrapper.

Covers the two behaviours the READ barrier depends on:
  * poll_transfer_batch considers EVERY status, not just the newest one
  * waiting_for_transfer_complete blocks until terminal and raises on failure,
    identically on a mori that exposes the batched wait and on one that does
    not.
"""

import importlib
import importlib.util
import threading
import time
from enum import Enum

import pytest

from vllm.platforms import current_platform

mori_available = importlib.util.find_spec("mori") is not None

if not (current_platform.is_rocm() and mori_available):
    pytest.skip(
        "MoRIIOs are only available on ROCm with mori package installed",
        allow_module_level=True,
    )

moriio_common = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common"
)
moriio_engine = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine"
)

MoRIIOWrapper = moriio_engine.MoRIIOWrapper
TransferBatchState = moriio_common.TransferBatchState
TransferError = moriio_common.TransferError


class FakeStatusCode(Enum):
    """The mori.io StatusCode values the wrapper compares against."""

    SUCCESS = "success"
    IN_PROGRESS = "in_progress"
    ERR_RDMA_OP = "err_rdma_op"


class FakeStatus:
    """Mirrors the mori TransferStatus surface the wrapper uses."""

    def __init__(self, state: str = "in_progress", message: str = ""):
        self.state = state
        self.message = message

    def Succeeded(self) -> bool:
        return self.state == "success"

    def Failed(self) -> bool:
        return self.state == "failed"

    def Message(self) -> str:
        return self.message

    def Code(self) -> str:
        return self.state


class Mori341Engine:
    """IOEngine exposing the batched WaitAll."""

    def __init__(self):
        self.calls: list[int] = []

    def wait_all(self, statuses, timeout_ms: int = -1):
        # Real WaitAll blocks on a condition variable until every status is
        # terminal or the shared deadline passes, with failure winning over
        # still-in-flight. Modelled by polling so the fake cannot pass a test
        # that the real API would fail.
        self.calls.append(timeout_ms)
        deadline = time.monotonic() + (timeout_ms / 1000.0 if timeout_ms > 0 else 0)
        while True:
            failed = False
            in_progress = False
            for status in statuses:
                if status.Failed():
                    failed = True
                elif not status.Succeeded():
                    in_progress = True
            if failed:
                return FakeStatusCode.ERR_RDMA_OP
            if not in_progress:
                return FakeStatusCode.SUCCESS
            if time.monotonic() >= deadline:
                return FakeStatusCode.IN_PROGRESS
            time.sleep(0.005)


class LegacyEngine:
    """Older build: no wait_all attribute at all."""


@pytest.fixture(autouse=True)
def _stub_status_code(monkeypatch):
    """Pin the StatusCode the wrapper compares against to the fake enum."""
    monkeypatch.setattr(moriio_engine, "StatusCode", FakeStatusCode)


def _ok():
    return FakeStatus("success")


def _pending():
    return FakeStatus("in_progress")


def _bad():
    return FakeStatus("failed", "SQ full")


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ([], TransferBatchState.DONE),
        ([_ok(), _ok()], TransferBatchState.DONE),
        # Each case below puts the interesting status BEFORE a success, which a
        # statuses[-1] check would have missed entirely.
        ([_bad(), _ok()], TransferBatchState.FAILED),
        ([_pending(), _ok()], TransferBatchState.PENDING),
        ([_pending(), _bad(), _ok()], TransferBatchState.FAILED),
    ],
)
def test_poll_transfer_batch_considers_every_status(statuses, expected):
    wrapper = MoRIIOWrapper(moriio_engine=object(), transfer_timeout=0.05)
    assert wrapper.poll_transfer_batch(statuses) is expected


@pytest.mark.parametrize(
    "engine_factory,expect_batched",
    [(Mori341Engine, True), (LegacyEngine, False)],
)
def test_batch_wait_probe(engine_factory, expect_batched):
    wrapper = MoRIIOWrapper(moriio_engine=engine_factory(), transfer_timeout=0.05)
    assert wrapper._batch_wait_available() is expect_batched


@pytest.mark.parametrize(
    "engine_factory", [Mori341Engine, LegacyEngine], ids=["mori341", "legacy"]
)
@pytest.mark.parametrize(
    "statuses_factory,expected_message",
    [
        (lambda: [_ok(), _bad()], "SQ full"),
        (lambda: [_ok(), _pending()], "timed out"),
    ],
    ids=["failed", "timeout"],
)
def test_wait_raises_with_detail(engine_factory, statuses_factory, expected_message):
    """A batch return code alone loses the per-status detail; recover it."""
    wrapper = MoRIIOWrapper(moriio_engine=engine_factory(), transfer_timeout=0.05)
    with pytest.raises(TransferError, match=expected_message):
        wrapper.waiting_for_transfer_complete(statuses_factory())


@pytest.mark.parametrize(
    "engine_factory", [Mori341Engine, LegacyEngine], ids=["mori341", "legacy"]
)
def test_wait_returns_immediately_when_nothing_pending(engine_factory):
    wrapper = MoRIIOWrapper(moriio_engine=engine_factory(), transfer_timeout=0.05)
    wrapper.waiting_for_transfer_complete([])
    wrapper.waiting_for_transfer_complete([_ok(), _ok()])


@pytest.mark.parametrize(
    "engine_factory", [Mori341Engine, LegacyEngine], ids=["mori341", "legacy"]
)
def test_wait_blocks_until_terminal(engine_factory):
    """The barrier is only worth anything if it actually waits."""
    wrapper = MoRIIOWrapper(moriio_engine=engine_factory(), transfer_timeout=5.0)
    late = FakeStatus("in_progress")
    timer = threading.Timer(0.2, lambda: setattr(late, "state", "success"))
    timer.start()
    try:
        start = time.monotonic()
        wrapper.waiting_for_transfer_complete([late])
        assert time.monotonic() - start >= 0.15
    finally:
        timer.cancel()


def test_batched_path_calls_into_mori():
    engine = Mori341Engine()
    wrapper = MoRIIOWrapper(moriio_engine=engine, transfer_timeout=0.05)
    wrapper.waiting_for_transfer_complete([_ok(), _ok()])
    assert engine.calls, "wait_all was never called"
    # timeout_ms == 0 means "poll once" to mori, so it must never round down.
    assert all(timeout_ms > 0 for timeout_ms in engine.calls)


def test_poll_stays_in_python_even_with_wait_all():
    """poll_transfer_batch must not drive mori's progress from this thread."""
    engine = Mori341Engine()
    wrapper = MoRIIOWrapper(moriio_engine=engine, transfer_timeout=0.05)
    assert wrapper.poll_transfer_batch([_bad(), _ok()]) is TransferBatchState.FAILED
    assert not engine.calls
