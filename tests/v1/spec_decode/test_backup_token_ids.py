# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the drafter's backup next-token ids.

The backup token of a request is read from host-side request state, which with
async scheduling holds -1 placeholders, and it is uploaded with a non-blocking
copy from pinned memory. Each step must upload its own values (no refill of a
host buffer whose copy is still pending), and an out-of-vocab backup token must
never reach the draft model's embedding.
"""

import pytest
import torch

import vllm.v1.spec_decode.utils as spec_utils
from vllm.platforms import current_platform
from vllm.v1.spec_decode.utils import DoubleBufferedCpuGpuBuffer


class _FakeEvent:
    """Records the order of record/synchronize calls instead of touching a device."""

    log: list[tuple[str, int]] = []
    count = 0

    def __init__(self):
        self.id = _FakeEvent.count
        _FakeEvent.count += 1

    def record(self):
        _FakeEvent.log.append(("record", self.id))

    def synchronize(self):
        _FakeEvent.log.append(("synchronize", self.id))


@pytest.fixture
def fake_event(monkeypatch: pytest.MonkeyPatch):
    _FakeEvent.log = []
    _FakeEvent.count = 0
    monkeypatch.setattr(spec_utils.torch, "Event", _FakeEvent)
    return _FakeEvent


def test_double_buffer_alternates_and_waits_for_own_copy(fake_event):
    # Any non-CPU device makes the buffer use events; meta needs no hardware.
    buf = DoubleBufferedCpuGpuBuffer(4, dtype=torch.int32, device=torch.device("meta"))

    first = buf.acquire()
    first.np[:2] = [11, 12]
    buf.copy_to_gpu(2)
    second = buf.acquire()
    # The next step fills the other host buffer, so the values of the pending
    # upload of the first step are left alone.
    assert second is not first
    second.np[:2] = [-1, -1]
    buf.copy_to_gpu(2)
    assert first.cpu[:2].tolist() == [11, 12]
    assert second.cpu[:2].tolist() == [-1, -1]
    # No buffer has been reused yet, so nothing waited.
    assert fake_event.log == [("record", 0), ("record", 1)]

    third = buf.acquire()
    # Reusing the first buffer waits for the copy issued from it, not for the
    # most recent one.
    assert third is first
    assert fake_event.log[-1] == ("synchronize", 0)
    buf.copy_to_gpu(2)
    assert fake_event.log[-1] == ("record", 0)
    assert fake_event.count == 2


class _NoAcceleratorEvent:
    def __init__(self):
        raise RuntimeError("torch.Event needs an accelerator")


def test_double_buffer_on_cpu_uses_no_events(monkeypatch: pytest.MonkeyPatch):
    # On a CPU-only wheel torch.Event() raises; the CPU backend's copies are
    # synchronous, so the buffer must not create events there.
    monkeypatch.setattr(spec_utils.torch, "Event", _NoAcceleratorEvent)
    buf = DoubleBufferedCpuGpuBuffer(4, dtype=torch.int32, device=torch.device("cpu"))
    for step in range(3):
        buf.acquire().np[:2] = [step, step + 1]
        assert buf.copy_to_gpu(2).tolist() == [step, step + 1]


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Triton kernel requires a GPU"
)
@pytest.mark.parametrize("backup", [-1, 100, 7])
def test_prepare_next_token_padded_keeps_backup_in_vocab(backup: int):
    device = torch.device(current_platform.device_type)
    vocab_size = 100
    # Row 0 is discarded, row 1 has no valid sampled token: both use the backup.
    # Row 2 accepts its sampled tokens and ignores the backup.
    sampled = torch.tensor([[5, 6], [-1, -1], [8, 9]], dtype=torch.int32, device=device)
    discard = torch.tensor([True, False, False], device=device)
    backup_ids = torch.full((3,), backup, dtype=torch.int32, device=device)
    next_ids = torch.empty(3, dtype=torch.int32, device=device)
    valid = torch.empty(3, dtype=torch.int32, device=device)

    spec_utils._eagle_prepare_next_token_padded(
        sampled,
        discard,
        backup_ids,
        next_ids,
        valid,
        vocab_size,
        sampled.shape[1],
        sampled.shape[0],
        sampled.stride(0),
        BLOCK_SIZE_TOKENS=spec_utils.next_power_of_2(sampled.shape[1]),
    )

    expected_backup = backup if 0 <= backup < vocab_size else 0
    assert next_ids.tolist() == [expected_backup, expected_backup, 9]
    assert valid.tolist() == [0, 0, 2]
