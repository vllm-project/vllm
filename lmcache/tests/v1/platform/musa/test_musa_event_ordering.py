# SPDX-License-Identifier: Apache-2.0
"""Hardware-gated cross-process ordering test for MUSA IPC events."""

# Standard
from typing import Protocol
import multiprocessing as mp
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.musa import ipc_wrapper as musa_ipc


class _QueueWriter(Protocol):
    def put(self, value: object) -> None:
        """Write a value to the queue."""


class _QueueReader(Protocol):
    def get(self, timeout: float | None = None) -> object:
        """Read a value from the queue."""


def _musa_event_ipc_available() -> bool:
    """Return whether this runner has the TorchMUSA event IPC API."""
    if not musa_ipc.is_torch_musa_available():
        return False
    module = musa_ipc.get_torch_musa_module()
    return module is not None and musa_ipc.check_torch_musa_event_support(module)


pytestmark = pytest.mark.skipif(
    not _musa_event_ipc_available(),
    reason="MUSA hardware and the TorchMUSA event IPC API are required",
)


def _produce_event_handle(
    handle_queue: _QueueWriter,
    done_queue: _QueueReader,
) -> None:
    """Record an event and keep it alive until the receiver finishes."""
    # Third Party
    import torch
    import torch_musa  # noqa: F401

    # First Party
    from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
    from lmcache.v1.platform.musa.ipc_wrapper import ENV_MUSA_HANDLE_TRANSFER

    os.environ[ENV_MUSA_HANDLE_TRANSFER] = "1"
    device = torch.device("musa:0")
    torch.musa.set_device(device)  # type: ignore[attr-defined]
    backend = get_event_ipc_backend(device)
    backend.check_event_support(device)

    stream = torch.musa.Stream()  # type: ignore[attr-defined]
    source = torch.zeros(1, device=device)
    event = backend.create_event(device)
    with torch.musa.stream(stream):  # type: ignore[attr-defined]
        source.fill_(7)
        backend.record_event(event, stream)

    handle_queue.put(backend.export_event(event, device))
    done_queue.get(timeout=30)


def test_musa_ipc_event_wait_orders_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A receiver process waits on an event recorded by a producer process."""
    monkeypatch.setenv(musa_ipc.ENV_MUSA_HANDLE_TRANSFER, "1")
    ctx = mp.get_context("spawn")
    handle_queue = ctx.Queue()
    done_queue = ctx.Queue()
    process = ctx.Process(
        target=_produce_event_handle,
        args=(handle_queue, done_queue),
    )
    process.start()

    device = torch.device("musa:0")
    torch.musa.set_device(device)  # type: ignore[attr-defined]

    try:
        event_handle = handle_queue.get(timeout=30)
        assert isinstance(event_handle, bytes)

        backend = get_event_ipc_backend(device)
        backend.check_event_support(device)
        imported = backend.import_event(event_handle, device)

        consumer = torch.musa.Stream()  # type: ignore[attr-defined]
        marker = torch.zeros(1, device=device)
        with torch.musa.stream(consumer):  # type: ignore[attr-defined]
            backend.wait_event(imported, consumer)
            marker.fill_(1)
        consumer.synchronize()

        assert float(marker.cpu().item()) == pytest.approx(1.0)
        assert backend.query_event(imported)
    finally:
        if process.is_alive():
            done_queue.put("done")
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join()

    assert process.exitcode == 0
