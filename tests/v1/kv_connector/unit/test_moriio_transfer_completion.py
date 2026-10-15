# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transfer-completion semantics of the MoRIIO wrapper.

Covers the two behaviours the READ barrier depends on:
  * poll_transfer_batch considers EVERY status, not just the newest one
  * waiting_for_transfer_complete blocks until terminal and raises on failure.
"""

import importlib
import importlib.util
import threading
import time

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


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_timeout_must_be_positive_and_finite(value):
    with pytest.raises(ValueError, match="must be finite and greater than zero"):
        moriio_common._positive_finite_timeout("test_timeout", value)


def test_positive_finite_timeout_is_preserved():
    assert moriio_common._positive_finite_timeout("test_timeout", "1.5") == 1.5


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
    "statuses_factory,expected_message",
    [
        (lambda: [_ok(), _bad()], "SQ full"),
        (lambda: [_ok(), _pending()], "timed out"),
    ],
    ids=["failed", "timeout"],
)
def test_wait_raises_with_detail(statuses_factory, expected_message):
    wrapper = MoRIIOWrapper(moriio_engine=object(), transfer_timeout=0.05)
    with pytest.raises(TransferError, match=expected_message):
        wrapper.waiting_for_transfer_complete(statuses_factory())


def test_wait_returns_immediately_when_nothing_pending():
    wrapper = MoRIIOWrapper(moriio_engine=object(), transfer_timeout=0.05)
    wrapper.waiting_for_transfer_complete([])
    wrapper.waiting_for_transfer_complete([_ok(), _ok()])


def test_wait_blocks_until_terminal():
    """The barrier is only worth anything if it actually waits."""
    wrapper = MoRIIOWrapper(moriio_engine=object(), transfer_timeout=5.0)
    late = FakeStatus("in_progress")
    timer = threading.Timer(0.2, lambda: setattr(late, "state", "success"))
    timer.start()
    try:
        start = time.monotonic()
        wrapper.waiting_for_transfer_complete([late])
        assert time.monotonic() - start >= 0.15
    finally:
        timer.cancel()
