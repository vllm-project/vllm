# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the PCIe comm-stack gates: the native-P2P-atomics
platform query and the stream-memops symm-mem barrier replacement."""

from types import SimpleNamespace
from unittest import mock

import pytest

import vllm.distributed.device_communicators.symm_mem_pcie_barrier as spb
from vllm.platforms.cuda import NvmlCudaPlatform


def _fake_pynvml(statuses):
    """pynvml stand-in whose nvmlDeviceGetP2PStatus pops from `statuses`."""
    fake = mock.MagicMock()
    fake.NVML_P2P_CAPS_INDEX_ATOMICS = 3
    fake.NVML_P2P_STATUS_OK = 0
    fake.NVMLError = RuntimeError
    fake.nvmlDeviceGetHandleByIndex.side_effect = lambda i: f"h{i}"

    def get_status(h1, h2, caps):
        assert caps == 3
        res = statuses.pop(0)
        if isinstance(res, Exception):
            raise res
        return res

    fake.nvmlDeviceGetP2PStatus.side_effect = get_status
    return fake


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ([0, 0, 0, 0, 0, 0], True),  # all six pairs of 4 GPUs OK
        ([0, 5], False),  # second pair reports NOT_SUPPORTED
        ([RuntimeError("nvml")], False),  # query failure -> assume absent
    ],
)
def test_has_native_p2p_atomics(statuses, expected):
    fake = _fake_pynvml(statuses)
    with mock.patch("vllm.platforms.cuda.pynvml", fake):
        assert NvmlCudaPlatform.has_native_p2p_atomics([0, 1, 2, 3]) is expected


class _FakeDriver:
    """Records memops; mimics the cuda.bindings.driver surface we use."""

    def __init__(self):
        self.calls = []
        self.CUstreamWaitValue_flags = SimpleNamespace(CU_STREAM_WAIT_VALUE_EQ=2)

    def CUstream(self, s):
        return s

    def CUdeviceptr(self, p):
        return p

    def cuStreamWriteValue32(self, stream, addr, value, flags):
        self.calls.append(("write", addr, value))
        return (0,)

    def cuStreamWaitValue32(self, stream, addr, value, flags):
        assert flags == 2  # EQ
        self.calls.append(("wait", addr, value))
        return (0,)


def _fake_handle(world=4, rank=1):
    return SimpleNamespace(
        world_size=world,
        rank=rank,
        signal_pad_ptrs=[10_000 * (r + 1) for r in range(world)],
        signal_pad_size=4096,
    )


@pytest.fixture
def barrier_env(monkeypatch):
    drv = _FakeDriver()
    monkeypatch.setattr(spb, "_drv", drv)
    spb.reset_pcie_barrier_state()
    monkeypatch.setattr("torch.cuda.is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        "torch.cuda.current_stream", lambda: SimpleNamespace(cuda_stream=0)
    )
    return drv


def test_pcie_barrier_ring_slots_and_sequence(barrier_env):
    drv = barrier_env
    h = _fake_handle(world=4, rank=1)
    channel = 1
    W, R = 4, spb._RING

    def base(seq):
        return 4 * W * (channel * R + seq % R)

    spb._pcie_safe_barrier(h, channel=channel)
    writes = [c for c in drv.calls if c[0] == "write"]
    waits = [c for c in drv.calls if c[0] == "wait"]
    assert len(writes) == 4 and len(waits) == 4
    # every write lands in the peer's pad, in this rank's sender lane
    assert {a for _, a, _ in writes} == {
        h.signal_pad_ptrs[p] + base(1) + 4 * h.rank for p in range(4)
    }
    # every wait polls this rank's own pad, one slot per sender
    assert {a for _, a, _ in waits} == {
        h.signal_pad_ptrs[h.rank] + base(1) + 4 * p for p in range(4)
    }
    # all writes are issued before the first wait
    kinds = [k for k, _, _ in drv.calls]
    assert kinds.index("wait") == 4
    assert all(v == 1 for _, _, v in drv.calls)

    # consecutive same-channel instances use different ring slots
    drv.calls.clear()
    spb._pcie_safe_barrier(h, channel=channel)
    assert all(v == 2 for _, _, v in drv.calls)
    assert {a for k, a, _ in drv.calls if k == "write"} == {
        h.signal_pad_ptrs[p] + base(2) + 4 * h.rank for p in range(4)
    }
    assert base(2) != base(1)

    # the ring wraps after _RING instances, values keep growing
    for seq in range(3, R + 2):
        drv.calls.clear()
        spb._pcie_safe_barrier(h, channel=channel)
        assert all(v == seq for _, _, v in drv.calls)
    assert base(R + 1) == base(1)  # wrapped slot, distinguished by value

    spb.reset_pcie_barrier_state()
    drv.calls.clear()
    spb._pcie_safe_barrier(h, channel=channel)
    assert all(v == 1 for _, _, v in drv.calls)


def test_pcie_barrier_rejects_graph_capture(barrier_env, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="captured"):
        spb._pcie_safe_barrier(_fake_handle(), channel=0)


def test_pcie_barrier_rejects_bad_channel(barrier_env):
    with pytest.raises(ValueError):
        spb._pcie_safe_barrier(_fake_handle(), channel=spb._MAX_CHANNELS)
