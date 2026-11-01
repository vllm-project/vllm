# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the `torch.distributed` transport of the NCCL weight-transfer backend.

The backend moves weights with `PyNcclCommunicator` on CUDA/ROCm and over
`torch.distributed` (`xccl`, ...) on accelerators that have no NCCL. These tests
cover the platform dispatch and the transport's own behaviour without needing an
accelerator; the collectives themselves are exercised by the two-device tests.
"""

import atexit
import threading
from unittest.mock import MagicMock

import pytest
import torch

import vllm.distributed.utils as distributed_utils
import vllm.distributed.weight_transfer.nccl_common as nccl_common
import vllm.distributed.weight_transfer.torch_dist_transport as torch_dist_transport
from vllm.distributed.weight_transfer.torch_dist_transport import TorchDistTransport
from vllm.platforms import current_platform


@pytest.fixture
def not_cuda_alike(monkeypatch: pytest.MonkeyPatch):
    """Make the current platform look like an accelerator without NCCL."""
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: False)


def _make_transport(process_group=None) -> TorchDistTransport:
    """A communicator over a stand-in group, with its atexit hook unregistered."""
    comm = TorchDistTransport(
        process_group if process_group is not None else MagicMock(),
        torch.device("cpu"),
        rank=0,
        world_size=2,
        backend="gloo",
    )
    atexit.unregister(comm.destroy)
    return comm


class TestPlatformDispatch:
    """Which transport each platform gets, for each rendezvous mode."""

    def test_tcp_rendezvous_uses_torch_dist_off_cuda(
        self, not_cuda_alike, monkeypatch: pytest.MonkeyPatch
    ):
        sentinel = object()
        seen: dict[str, object] = {}

        def fake_create(master_address, master_port, rank, world_size, device):
            seen.update(
                master_address=master_address,
                master_port=master_port,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            return sentinel

        monkeypatch.setattr(
            torch_dist_transport.TorchDistTransport, "create", fake_create
        )

        got = nccl_common.stateless_init_process_group(
            "127.0.0.1", 12345, rank=1, world_size=2, device=0
        )

        assert got is sentinel
        assert seen == {
            "master_address": "127.0.0.1",
            "master_port": 12345,
            "rank": 1,
            "world_size": 2,
            "device": 0,
        }

    def test_uid_rendezvous_raises_off_cuda(self, not_cuda_alike):
        # An ncclUniqueId is only meaningful to NCCL, so there is nothing to fall
        # back to here -- unlike the TCPStore rendezvous, which any backend joins.
        with pytest.raises(NotImplementedError, match="ncclGetUniqueId"):
            nccl_common.uid_init_process_group(
                b"\x00" * 128, rank=0, world_size=2, device=0
            )


class TestCreate:
    @pytest.fixture(autouse=True)
    def named_platform(self, monkeypatch: pytest.MonkeyPatch):
        """Pin a platform that names a device type and a collective backend.

        `create()` resolves an integer device index against the platform, and a
        host with no accelerator at all reports an empty device type, which is not
        a device string torch accepts. Pinning keeps these cases running anywhere.
        """
        monkeypatch.setattr(current_platform, "device_type", "xpu", raising=False)
        monkeypatch.setattr(current_platform, "dist_backend", "xccl", raising=False)

    def test_joins_over_the_platform_backend(self, monkeypatch: pytest.MonkeyPatch):
        seen: dict[str, object] = {}

        def fake_init(**kwargs):
            seen.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(
            distributed_utils,
            "stateless_init_torch_distributed_process_group",
            fake_init,
        )

        comm = TorchDistTransport.create("127.0.0.1", 12345, 1, 2, device=0)
        atexit.unregister(comm.destroy)

        assert seen == {
            "host": "127.0.0.1",
            "port": 12345,
            "rank": 1,
            "world_size": 2,
            "backend": current_platform.dist_backend,
        }
        # An integer device index becomes a device on the platform's own type,
        # so the caller does not have to know which that is.
        assert comm.device == torch.device(current_platform.device_type, 0)
        assert comm.rank == 1
        assert comm.world_size == 2
        # Never the no-op state that is the reason this transport exists:
        # `PyNcclCommunicator` sets `disabled = True` off CUDA and every
        # collective on it silently does nothing.
        assert comm.available
        assert not comm.disabled


class TestBroadcast:
    def test_addresses_the_sender_within_the_group(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        calls: list[tuple[str, int | None]] = []
        entered: list[object] = []

        class FakeStreamCtx:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                entered.append(self.stream)

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(current_platform, "stream", FakeStreamCtx)
        monkeypatch.setattr(
            torch_dist_transport.dist,
            "broadcast",
            lambda tensor, group_src, group: calls.append(("broadcast", group_src)),
        )
        monkeypatch.setattr(
            torch_dist_transport.dist,
            "barrier",
            lambda group: calls.append(("barrier", None)),
        )

        comm = _make_transport()
        comm.broadcast(torch.zeros(4), src=0)
        stream = object()
        comm.broadcast(torch.zeros(4), src=0, stream=stream)

        # `group_src`, not `src`: the trainer is rank 0 of the transfer group but
        # has no rank in the workers' global group. And one barrier per tensor,
        # which oneCCL needs to stay coupled to the far end.
        assert calls == [("broadcast", 0), ("barrier", None)] * 2
        # A caller that names a stream gets the collective ordered against it, and
        # one that does not is left on the current stream rather than given some
        # other one.
        assert entered == [stream]


class TestDestroy:
    def test_aborts_and_unregisters_once_off_the_caller(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        unregistered: list[str] = []
        monkeypatch.setattr(
            "torch.distributed.distributed_c10d._unregister_process_group",
            lambda name: unregistered.append(name),
        )
        threads: list[str] = []
        process_group = MagicMock()
        process_group.group_name = "weight-transfer"
        process_group.abort.side_effect = lambda: threads.append(
            threading.current_thread().name
        )

        comm = _make_transport(process_group)
        comm.destroy()
        # Idempotent: a trainer may close explicitly and still hit the atexit hook.
        comm.destroy()
        assert comm._reaper is not None
        comm._reaper.join(timeout=10)

        # `abort()`, never `shutdown()`: the peer is still in the group, and its
        # finalization handshake is what deadlocks (or dies) on oneCCL.
        process_group.abort.assert_called_once_with()
        process_group.shutdown.assert_not_called()
        assert unregistered == ["weight-transfer"]
        # Both steps can block on the peer, so neither may run on the caller.
        assert threads == ["weight-transfer-pg-reaper"]
        assert comm.disabled
        # The handle a trainer holds goes with it, so a late `close_communicator`
        # cannot reach a group that is already gone.
        assert comm.group.process_group is None

    def test_does_not_wait_for_the_destructor(self, monkeypatch: pytest.MonkeyPatch):
        # Destroying the group is what blocks (oneCCL finalizes against a peer
        # that is still in the group), so it must not happen on the caller's
        # thread. The reaper holds the last reference until it can let go.
        monkeypatch.setattr(
            "torch.distributed.distributed_c10d._unregister_process_group",
            lambda name: None,
        )
        released = threading.Event()

        class SlowToDestroy:
            group_name = "weight-transfer"

            def abort(self):
                pass

            def __del__(self):
                released.set()

        comm = _make_transport(SlowToDestroy())
        comm.destroy()

        assert comm._reaper is not None
        assert comm._reaper.daemon
        comm._reaper.join(timeout=10)
        assert released.is_set(), "the reaper should end up dropping the group"
