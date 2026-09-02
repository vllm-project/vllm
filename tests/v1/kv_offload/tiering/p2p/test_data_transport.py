# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DataTransport base class and NixlTransport."""

from __future__ import annotations

import ctypes
from unittest.mock import MagicMock, patch

import numpy as np

from vllm.v1.kv_offload.tiering.p2p.data.base import PollResult
from vllm.v1.kv_offload.tiering.p2p.data.nixl import NixlTransport

# ---------------------------------------------------------------------------
# DataTransport base class tests
# ---------------------------------------------------------------------------


class TestDataTransportBase:
    """Tests for the DataTransport abstract base properties."""

    def _make_view(self, num_blocks: int = 8, block_len: int = 1024) -> memoryview:
        """Create a memoryview with the given shape."""
        buf = np.zeros((num_blocks, block_len), dtype=np.uint8)
        return memoryview(buf)

    def test_properties(self):
        """base_addr, num_blocks, block_len are set from memoryview shape."""
        view = self._make_view(num_blocks=4, block_len=2048)

        # Use NixlTransport (concrete) with NIXL mocked away
        with patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", None):
            transport = NixlTransport("test:1", view)

        assert transport.num_blocks == 4
        assert transport.block_len == 2048
        assert transport.base_addr == ctypes.addressof(ctypes.c_char.from_buffer(view))

    def test_config_fingerprint_empty_when_no_fields(self):
        """No config fields → empty fingerprint."""
        view = self._make_view()
        with patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", None):
            transport = NixlTransport("test:1", view, config_fields=None)
        assert transport.config_fingerprint == ""

    def test_config_fingerprint_deterministic(self):
        """Same config fields → same fingerprint."""
        view = self._make_view()
        fields = {"model": "llama", "dtype": "float16", "blocks_per_chunk": 1}
        with patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", None):
            t1 = NixlTransport("test:1", view, config_fields=fields)
            t2 = NixlTransport("test:2", view, config_fields=fields)
        assert t1.config_fingerprint == t2.config_fingerprint
        assert len(t1.config_fingerprint) == 16

    def test_config_fingerprint_differs_for_different_fields(self):
        """Different config fields → different fingerprint."""
        view = self._make_view()
        with patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", None):
            t1 = NixlTransport("test:1", view, config_fields={"model": "a"})
            t2 = NixlTransport("test:2", view, config_fields={"model": "b"})
        assert t1.config_fingerprint != t2.config_fingerprint


# ---------------------------------------------------------------------------
# NixlTransport tests (with mocked NIXL agent)
# ---------------------------------------------------------------------------


class TestNixlTransportWithMockedAgent:
    """Tests for NixlTransport logic with a mocked NIXL agent."""

    def _make_transport(self) -> NixlTransport:
        """Create a NixlTransport with mocked NIXL internals."""
        view = memoryview(np.zeros((8, 1024), dtype=np.uint8))

        with patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", None):
            transport = NixlTransport("test:1", view)

        # Manually set up a mock agent after construction
        agent = MagicMock()
        agent.add_remote_agent.return_value = "nixl-peer-name"
        agent.get_xfer_descs.return_value = MagicMock()
        agent.prep_xfer_dlist.return_value = MagicMock()
        agent.make_prepped_xfer.return_value = MagicMock(name="handle")
        agent.transfer.return_value = None
        agent.check_xfer_state.return_value = "PROC"
        agent.get_agent_metadata.return_value = b"test-metadata"

        transport._agent = agent
        transport._local_dlist = MagicMock()
        return transport

    def test_available_false_without_nixl(self):
        """Without NIXL installed, available is False."""
        view = memoryview(np.zeros((4, 512), dtype=np.uint8))
        with patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", None):
            transport = NixlTransport("test:1", view)
        assert transport.available is False

    def test_available_true_with_agent(self):
        transport = self._make_transport()
        assert transport.available is True

    def test_get_agent_metadata(self):
        transport = self._make_transport()
        assert transport.get_agent_metadata() == b"test-metadata"

    def test_write_blocks_returns_none_for_unknown_peer(self):
        """write_blocks returns None if peer not registered."""
        transport = self._make_transport()
        result = transport.write_blocks("unknown:1", [0, 1], [2, 3])
        assert result is None

    def test_write_blocks_returns_transfer_id(self):
        """write_blocks returns an integer transfer_id on success."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0, 1], [2, 3])
        assert tid is not None
        assert isinstance(tid, int)

    def test_write_blocks_increments_transfer_id(self):
        """Each write_blocks call gets a unique transfer_id."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid1 = transport.write_blocks("peer:1", [0], [1])
        tid2 = transport.write_blocks("peer:1", [2], [3])
        assert tid1 != tid2

    def test_poll_empty_when_no_inflight(self):
        """poll returns empty when nothing is inflight."""
        transport = self._make_transport()
        result = transport.poll()
        assert result == PollResult(done=(), failed=())

    def test_poll_returns_done_when_transfer_completes(self):
        """Completed transfer appears in poll().done."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])

        # Simulate completion
        transport._agent.check_xfer_state.return_value = "DONE"
        result = transport.poll()

        assert tid in result.done
        assert result.failed == ()
        # Handle released
        transport._agent.release_xfer_handle.assert_called()

    def test_poll_returns_failed_for_error_state(self):
        """Transfer in error state appears in poll().failed."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])

        transport._agent.check_xfer_state.return_value = "ERR"
        result = transport.poll()

        assert result.done == ()
        assert tid in result.failed

    def test_poll_ignores_in_progress(self):
        """Transfers in PROC/PEND state stay inflight."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        transport.write_blocks("peer:1", [0], [1])

        transport._agent.check_xfer_state.return_value = "PROC"
        result = transport.poll()
        assert result.done == ()
        assert result.failed == ()

        transport._agent.check_xfer_state.return_value = "PEND"
        result = transport.poll()
        assert result.done == ()
        assert result.failed == ()

    def test_poll_peer_id_scopes_to_peer(self):
        """poll(peer_id) drains only that peer's transfers.

        Regression: the transport is shared across peer sessions (e.g. a
        single prefiller serving a DP>1 decoder). An unscoped poll by one
        session used to consume and discard sibling sessions' completions,
        starving them until timeout. poll(peer_id) must leave other peers'
        transfers inflight.
        """
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        transport.add_remote_peer("peer:2", b"meta", 0x2000, 8, 1024)

        tid1 = transport.write_blocks("peer:1", [0], [1])
        tid2 = transport.write_blocks("peer:2", [2], [3])
        transport._agent.check_xfer_state.return_value = "DONE"

        # Polling peer:1 must not consume peer:2's completed transfer.
        result = transport.poll(peer_id="peer:1")
        assert tid1 in result.done
        assert tid2 not in result.done
        assert tid1 not in transport._inflight
        assert tid2 in transport._inflight

        # peer:2 sees its own completion when it polls.
        result2 = transport.poll(peer_id="peer:2")
        assert tid2 in result2.done
        assert tid2 not in transport._inflight

    def test_cancel_removes_inflight(self):
        """cancel removes transfers and releases handles."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid in transport._inflight

        result = transport.cancel([tid])
        assert result == []
        assert tid not in transport._inflight
        transport._agent.release_xfer_handle.assert_called()

    def test_cancel_ignores_unknown_ids(self):
        """cancel with unknown IDs doesn't crash."""
        transport = self._make_transport()
        assert transport.cancel([999, 1000]) == []
        assert transport.cancel([999, 1000], mode="wait") == []

    def test_cancel_wait_release_succeeds(self):
        """wait-mode cancel that succeeds pops the entry and returns []."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid in transport._inflight

        result = transport.cancel([tid], mode="wait")
        assert result == []
        assert tid not in transport._inflight
        transport._agent.release_xfer_handle.assert_called_once()

    def test_cancel_wait_release_raises(self):
        """wait-mode cancel keeps the entry and returns the tid on raise."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        transport._agent.release_xfer_handle.side_effect = RuntimeError(
            "NIXL_ERR_REPOST_ACTIVE"
        )

        result = transport.cancel([tid], mode="wait")
        assert result == [tid]
        assert tid in transport._inflight

    def test_cancel_wait_then_poll_completes(self):
        """A wait-cancel that left a tid pending later completes via poll."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        transport._agent.release_xfer_handle.side_effect = RuntimeError("busy")
        assert transport.cancel([tid], mode="wait") == [tid]
        assert tid in transport._inflight

        transport._agent.release_xfer_handle.side_effect = None
        transport._agent.check_xfer_state.return_value = "DONE"

        result = transport.poll()
        assert tid in result.done
        assert tid not in transport._inflight

    def test_add_and_remove_remote_peer(self):
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        assert "peer:1" in transport._remote_dlists

        transport.remove_remote_peer("peer:1")
        assert "peer:1" not in transport._remote_dlists
        transport._agent.release_dlist_handle.assert_called()
        transport._agent.remove_remote_agent.assert_called()

    def test_close_releases_everything(self):
        """close releases all handles and clears state."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        transport.write_blocks("peer:1", [0], [1])

        transport.close()
        assert transport._agent is None
        assert transport._inflight == {}
        assert transport._remote_dlists == {}


# ---------------------------------------------------------------------------
# NIXL agent-config selection
# ---------------------------------------------------------------------------


class TestNixlAgentConfigSelection:
    """Tests that backends/num_threads pick the right nixl_agent_config call.

    Mirrors the conditional in
    vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:325-329.
    """

    def _make_view(self) -> memoryview:
        return memoryview(np.zeros((4, 512), dtype=np.uint8))

    def test_non_ucx_backends_passes_backends_kwarg(self):
        """When any non-UCX backend is requested, pass backends + telemetry."""
        agent_cls = MagicMock()
        config_fn = MagicMock(return_value=MagicMock(name="cfg"))
        with (
            patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", agent_cls),
            patch(
                "vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgentConfig", config_fn
            ),
        ):
            NixlTransport("test:1", self._make_view(), backends=["MOONCAKE"])

        config_fn.assert_called_once_with(backends=["MOONCAKE"], capture_telemetry=True)
        # num_threads must NOT be passed on the non-UCX branch.
        assert "num_threads" not in config_fn.call_args.kwargs

    def test_ucx_only_passes_num_threads(self):
        """UCX-only configuration passes num_threads + telemetry, no backends."""
        agent_cls = MagicMock()
        config_fn = MagicMock(return_value=MagicMock(name="cfg"))
        with (
            patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", agent_cls),
            patch(
                "vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgentConfig", config_fn
            ),
        ):
            NixlTransport("test:1", self._make_view(), num_threads=8)

        config_fn.assert_called_once_with(num_threads=8, capture_telemetry=True)
        assert "backends" not in config_fn.call_args.kwargs

    def test_default_backends_is_ucx_only(self):
        """No backends arg → defaults to UCX-only branch."""
        agent_cls = MagicMock()
        config_fn = MagicMock(return_value=MagicMock(name="cfg"))
        with (
            patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", agent_cls),
            patch(
                "vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgentConfig", config_fn
            ),
        ):
            NixlTransport("test:1", self._make_view())

        # Default num_threads=4, no backends kwarg.
        config_fn.assert_called_once_with(num_threads=4, capture_telemetry=True)
