# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DBO comm stream on ROCm: a CU-masked stream with its own hardware queue."""

import ctypes

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import dbo_comm_stream_priority

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="CU-masked streams are a HIP API"
)


def _comm_stream() -> tuple[torch.device, torch.cuda.Stream]:
    device = torch.device("cuda", torch.accelerator.current_device_index())
    stream = current_platform.create_ubatch_comm_stream(
        device, dbo_comm_stream_priority()
    )
    return device, stream


def test_dbo_comm_stream_is_cu_masked_on_rocm():
    """The comm stream comes from hipExtStreamCreateWithCUMask, not the pooled
    fallback, and work on it orders with the compute stream through events."""
    device, comm_stream = _comm_stream()
    assert isinstance(comm_stream, torch.cuda.ExternalStream)

    x = torch.arange(1 << 20, device=device, dtype=torch.float32)
    ready, done = torch.cuda.Event(), torch.cuda.Event()
    ready.record()
    with torch.cuda.stream(comm_stream):
        comm_stream.wait_event(ready)
        y = x * 2
        done.record()
    torch.cuda.current_stream().wait_event(done)
    assert torch.equal(y, x * 2)


def test_failed_cu_mask_create_falls_back_to_a_pooled_stream(monkeypatch):
    """When hipExtStreamCreateWithCUMask fails, the comm stream is a pooled
    stream and the HIP error does not fail the next kernel launch."""
    real_cdll = ctypes.CDLL

    class FailingCreate:
        """The HIP runtime, except that the create call gets a null stream
        pointer, which is hipErrorInvalidValue."""

        def __init__(self, lib):
            self._lib = lib

        def hipExtStreamCreateWithCUMask(self, handle, size, mask):
            return self._lib.hipExtStreamCreateWithCUMask(None, size, mask)

        def __getattr__(self, name):
            return getattr(self._lib, name)

    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda *args, **kwargs: FailingCreate(real_cdll(*args, **kwargs)),
    )
    device, comm_stream = _comm_stream()
    assert not isinstance(comm_stream, torch.cuda.ExternalStream)

    x = torch.ones(1 << 10, device=device)
    assert torch.equal(x * 2, torch.full_like(x, 2))


def test_static_hip_queues_keep_the_pooled_stream(monkeypatch):
    """With HIP's static queues and stream priority honoured, the high-priority
    pooled stream already has a queue of its own, so the hook keeps it."""
    monkeypatch.setenv("DEBUG_HIP_DYNAMIC_QUEUES", "0")
    monkeypatch.setenv("DEBUG_HIP_IGNORE_STREAM_PRIORITY", "0")
    _, comm_stream = _comm_stream()
    assert not isinstance(comm_stream, torch.cuda.ExternalStream)
    assert comm_stream.priority == dbo_comm_stream_priority()

    monkeypatch.setenv("DEBUG_HIP_IGNORE_STREAM_PRIORITY", "1")
    _, comm_stream = _comm_stream()
    assert isinstance(comm_stream, torch.cuda.ExternalStream)
