# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Protocol
import multiprocessing as mp

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.musa import ipc_wrapper as musa_ipc


class _ResultQueue(Protocol):
    """Queue surface required by the spawned receiver."""

    def put(self, value: tuple[tuple[int, ...], float]) -> None:
        """Send a test result to the parent process."""
        ...


def _musa_memory_ipc_available() -> bool:
    """Return whether this runner has the TorchMUSA memory IPC API."""
    if not musa_ipc.is_torch_musa_available():
        return False
    module = musa_ipc.get_torch_musa_module()
    return module is not None and musa_ipc.check_torch_musa_ipc_support(module)


pytestmark = pytest.mark.skipif(
    not _musa_memory_ipc_available(),
    reason="MUSA hardware and the TorchMUSA memory IPC API are required",
)


def _receiver(wrapper: musa_ipc.MusaIPCWrapper, queue: _ResultQueue) -> None:
    """Reconstruct a MUSA tensor from ``wrapper`` in a spawned process."""
    tensor = wrapper.to_tensor()
    result = (tuple(tensor.shape), float(tensor.sum().item()))
    del tensor
    wrapper.close()
    queue.put(result)


def test_musa_ipc_wrapper_two_process_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A receiver process can reconstruct and read a sender MUSA tensor."""
    monkeypatch.setenv(musa_ipc.ENV_MUSA_HANDLE_TRANSFER, "1")
    tensor = torch.arange(16, device="musa:0", dtype=torch.float32).reshape(4, 4)
    wrapper = musa_ipc.MusaIPCWrapper(tensor)
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_receiver, args=(wrapper, queue))
    process.start()
    try:
        shape, total = queue.get(timeout=10)
        process.join(timeout=10)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)

    assert process.exitcode == 0
    assert shape == (4, 4)
    assert total == pytest.approx(float(tensor.sum().item()))
