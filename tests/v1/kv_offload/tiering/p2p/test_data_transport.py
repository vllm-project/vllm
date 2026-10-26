# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DataTransport base class and NixlTransport."""

from __future__ import annotations

import ctypes
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vllm.v1.kv_offload.tiering.p2p.data import nixl as nixl_module
from vllm.v1.kv_offload.tiering.p2p.data.base import PollResult
from vllm.v1.kv_offload.tiering.p2p.data.nixl import NixlTransport


class nixlRemoteDisconnectError(Exception):  # noqa: N801
    """Stand-in for NIXL's binding exception of the same name."""


def _remote_disconnect() -> Exception:
    return nixlRemoteDisconnectError("NIXL_ERR_REMOTE_DISCONNECT")


@pytest.fixture
def registration_transport():
    """Track native registrations while constructing the real transport."""
    agent = MagicMock()
    registered = set()

    def add_remote_agent(metadata):
        name = metadata.decode()
        registered.add(name)
        return name

    agent.add_remote_agent.side_effect = add_remote_agent
    agent.remove_remote_agent.side_effect = registered.remove
    agent.check_xfer_state.return_value = "DONE"
    with (
        patch.object(nixl_module, "_NixlAgent", return_value=agent),
        patch.object(nixl_module, "_NixlAgentConfig"),
    ):
        transport = NixlTransport(
            "test:1", memoryview(np.zeros((8, 1024), dtype=np.uint8))
        )
    try:
        yield transport, agent, registered
    finally:
        transport.close()


class TestNixlRegistrationRollback:
    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize("stage", ["get_xfer_descs", "prep_xfer_dlist"])
    def test_failed_registration_rolls_back_and_can_retry(
        self, registration_transport, asynchronous, stage
    ):
        """Failed descriptor setup must not leak an agent or damage healthy peers."""
        transport, agent, registered = registration_transport
        transport.add_remote_peer("healthy:1", b"healthy", 0x2000, 8, 1024)
        error = RuntimeError("descriptor setup failed")
        getattr(agent, stage).side_effect = error
        register = (
            transport.add_remote_peer_async
            if asynchronous
            else transport.add_remote_peer
        )
        with pytest.raises(RuntimeError) as exc_info:
            result = register("peer:1", b"peer", 0x1000, 8, 1024)
            if asynchronous:
                result.result(timeout=5)
        assert exc_info.value is error
        assert registered == {"healthy"}
        assert transport.write_blocks("peer:1", [0], [1]) is None
        assert transport.write_blocks("healthy:1", [0], [1]) is not None

        getattr(agent, stage).side_effect = None
        result = register("peer:1", b"peer", 0x1000, 8, 1024)
        if asynchronous:
            result.result(timeout=5)
        transfer = transport.write_blocks("peer:1", [0], [1])
        assert transfer is not None
        assert transfer in transport.poll().done
        transport.remove_remote_peer("peer:1")
        assert registered == {"healthy"}

    @pytest.mark.parametrize("stage", ["get_xfer_descs", "prep_xfer_dlist"])
    def test_failed_duplicate_registration_keeps_published_agent(
        self, registration_transport, stage
    ):
        """Loading existing metadata does not transfer its cleanup ownership."""
        transport, agent, registered = registration_transport
        transport.add_remote_peer("peer:1", b"peer", 0x1000, 8, 1024)
        getattr(agent, stage).side_effect = RuntimeError("descriptor setup failed")

        with pytest.raises(RuntimeError, match="descriptor setup failed"):
            transport.add_remote_peer("peer:1", b"peer", 0x1000, 8, 1024)

        assert registered == {"peer"}
        transfer = transport.write_blocks("peer:1", [0], [1])
        assert transfer is not None
        assert transfer in transport.poll().done

    @pytest.mark.parametrize("stage", ["get_xfer_descs", "prep_xfer_dlist"])
    def test_failed_rollback_preserves_registration_error(
        self, registration_transport, stage, caplog
    ):
        """A cleanup error must not replace the descriptor setup error."""
        transport, agent, _ = registration_transport
        error = RuntimeError("descriptor setup failed")
        getattr(agent, stage).side_effect = error
        agent.remove_remote_agent.side_effect = RuntimeError("rollback failed")

        future = transport.add_remote_peer_async("peer:1", b"peer", 0x1000, 8, 1024)
        with pytest.raises(RuntimeError) as exc_info:
            future.result(timeout=5)
        assert exc_info.value is error
        assert transport.write_blocks("peer:1", [0], [1]) is None
        assert "rollback failed" in caplog.text


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

    def test_add_remote_peer_async_registers_off_the_caller_thread(self):
        """The O(num_blocks) registration must not run on the caller thread.

        This is the whole point of the async path: the scheduler thread
        hands the work off rather than paying prep_xfer_dlist inline.
        """
        transport = self._make_transport()
        caller = threading.get_ident()
        worker_threads: list[int] = []

        def record_thread(*args, **kwargs):
            worker_threads.append(threading.get_ident())
            return MagicMock(name="remote_dlist")

        transport._agent.prep_xfer_dlist.side_effect = record_thread

        future = transport.add_remote_peer_async("peer:1", b"meta", 0x1000, 8, 1024)
        future.result(timeout=5)

        assert worker_threads and caller not in worker_threads
        assert transport.write_blocks("peer:1", [0], [1]) is not None

    def test_registration_landing_after_remove_is_discarded(self):
        """A late registration must not revive a peer that was reaped.

        The session is torn down and remove_remote_peer runs while the
        worker is still building descriptors; publishing afterwards would
        leave a registered peer with no session and leak its handles.
        """
        transport = self._make_transport()
        building = threading.Event()
        release = threading.Event()

        def blocking_prep(*args, **kwargs):
            building.set()
            release.wait(timeout=5)
            return MagicMock(name="remote_dlist")

        transport._agent.prep_xfer_dlist.side_effect = blocking_prep

        future = transport.add_remote_peer_async("peer:1", b"meta", 0x1000, 8, 1024)
        assert building.wait(timeout=5)
        transport.remove_remote_peer("peer:1")
        release.set()

        with pytest.raises(RuntimeError, match="superseded"):
            future.result(timeout=5)
        assert "peer:1" not in transport._remote_dlists
        assert transport.write_blocks("peer:1", [0], [1]) is None
        # The handles the worker built were released rather than leaked.
        assert transport._agent.release_dlist_handle.called
        assert transport._agent.remove_remote_agent.called

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

    @pytest.mark.parametrize(
        "failure_method,release_fails",
        [
            ("make_prepped_xfer", False),
            ("transfer", False),
            ("transfer", True),
        ],
    )
    def test_write_blocks_submission_exception_returns_failure(
        self, failure_method, release_fails
    ):
        """Setup failures surface even if releasing a created handle also fails."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        agent = transport._agent
        getattr(agent, failure_method).side_effect = RuntimeError("submission failed")
        if release_fails:
            agent.release_xfer_handle.side_effect = RuntimeError("release failed")

        assert transport.write_blocks("peer:1", [0], [1]) is None
        assert not transport._inflight
        assert transport.poll() == PollResult(done=(), failed=())
        if failure_method == "make_prepped_xfer":
            agent.transfer.assert_not_called()
            agent.release_xfer_handle.assert_not_called()
        else:
            agent.release_xfer_handle.assert_called_once_with(
                agent.make_prepped_xfer.return_value
            )

    def test_write_blocks_transfer_err_status_returns_failure(self):
        """transfer() reports a failed post as "ERR" instead of raising."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        transport._agent.transfer.return_value = "ERR"

        assert transport.write_blocks("peer:1", [0], [1]) is None
        assert not transport._inflight
        transport._agent.release_xfer_handle.assert_called_once_with(
            transport._agent.make_prepped_xfer.return_value
        )

    def test_write_blocks_increments_transfer_id(self):
        """Each write_blocks call gets a unique transfer_id."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid1 = transport.write_blocks("peer:1", [0], [1])
        tid2 = transport.write_blocks("peer:1", [2], [3])
        assert tid1 != tid2

    def test_poll_empty_when_no_inflight(self):
        """Poll returns empty when nothing is inflight."""
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
        """Cancel removes transfers and releases handles."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid in transport._inflight

        result = transport.cancel([tid])
        assert result == []
        assert tid not in transport._inflight
        transport._agent.release_xfer_handle.assert_called()

    def test_cancel_ignores_unknown_ids(self):
        """Cancel with unknown IDs doesn't crash."""
        transport = self._make_transport()
        assert transport.cancel([999, 1000]) == []
        assert transport.cancel([999, 1000], mode="wait") == []

    @pytest.mark.parametrize("state", ["DONE", "ERR"])
    def test_cancel_wait_release_succeeds(self, state):
        """wait-mode cancel of a stopped transfer pops it and returns []."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid in transport._inflight

        transport._agent.check_xfer_state.return_value = state
        result = transport.cancel([tid], mode="wait")
        assert result == []
        assert tid not in transport._inflight
        transport._agent.release_xfer_handle.assert_called_once()

    @pytest.mark.parametrize("state", ["PROC", "PEND"])
    def test_cancel_wait_keeps_running_transfer(self, state):
        """Releasing a running NIXL WRITE does not stop it, so wait-mode
        cancel must not release it or report it as cancelled."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        transport._agent.check_xfer_state.return_value = state

        assert transport.cancel([tid], mode="wait") == [tid]
        assert tid in transport._inflight
        transport._agent.release_xfer_handle.assert_not_called()

    def test_cancel_wait_state_check_raises(self):
        """A failed state check is not proof the transfer stopped."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        transport._agent.check_xfer_state.side_effect = RuntimeError("NOT_FOUND")

        assert transport.cancel([tid], mode="wait") == [tid]
        assert tid in transport._inflight
        transport._agent.release_xfer_handle.assert_not_called()

    def test_cancel_wait_remote_disconnect_is_terminal(self):
        """A dead peer can no longer be written to: wait-mode cancel drops
        the transfer even if its handle can no longer be released."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        transport._agent.check_xfer_state.side_effect = _remote_disconnect()
        transport._agent.release_xfer_handle.side_effect = RuntimeError("gone")

        assert transport.cancel([tid], mode="wait") == []
        assert tid not in transport._inflight
        assert transport.poll() == PollResult(done=(), failed=())

    @pytest.mark.parametrize(
        "exc",
        [
            _remote_disconnect(),
            # Recognized by status code even if the binding class is renamed.
            RuntimeError("getXferStatus: NIXL_ERR_REMOTE_DISCONNECT"),
        ],
    )
    def test_poll_reports_remote_disconnect_as_failed(self, exc):
        """check_xfer_state raises instead of returning ERR once the peer
        disconnected; poll must surface that as a failure, not keep the
        transfer inflight forever."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        transport._agent.check_xfer_state.side_effect = exc

        result = transport.poll()
        assert result.done == ()
        assert tid in result.failed
        assert tid not in transport._inflight

    def test_poll_keeps_transfer_on_other_state_check_errors(self):
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        transport._agent.check_xfer_state.side_effect = RuntimeError("NOT_FOUND")

        assert transport.poll() == PollResult(done=(), failed=())
        assert tid in transport._inflight

    def test_cancel_wait_release_raises(self):
        """wait-mode cancel keeps the entry and returns the tid on raise."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        transport._agent.check_xfer_state.return_value = "DONE"
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
        assert tid is not None
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

    def test_remove_peer_removes_agent_even_if_release_fails(self):
        """A dlist release failure must not skip the agent removal.

        Teardown runs two independent NIXL operations. If
        release_dlist_handle() raises, remove_remote_agent() must still be
        attempted with the registered agent name; skipping it would leave
        a stale NIXL registration behind after the peer is gone.
        """
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        transport._agent.release_dlist_handle.side_effect = RuntimeError(
            "simulated release failure"
        )

        with patch.object(nixl_module.logger, "warning") as mock_warning:
            transport.remove_remote_peer("peer:1")

        # The release failure was logged rather than swallowed silently.
        mock_warning.assert_called_once()
        # Agent removal was still attempted with the registered name.
        transport._agent.release_dlist_handle.assert_called_once()
        transport._agent.remove_remote_agent.assert_called_once_with("nixl-peer-name")
        assert "peer:1" not in transport._remote_dlists

    def test_superseded_registration_removes_agent_even_if_release_fails(self):
        """A release failure while discarding must still remove the agent.

        _publish_peer discards handles built under a stale generation when
        the peer was removed or re-registered mid-registration. Its
        descriptor-list release and agent removal are independent too: if
        release_dlist_handle() raises, remove_remote_agent() must still be
        attempted for the superseded registration, otherwise its remote
        agent stays registered forever.
        """
        transport = self._make_transport()
        building = threading.Event()
        release = threading.Event()

        def blocking_prep(*args, **kwargs):
            building.set()
            release.wait(timeout=5)
            return MagicMock(name="remote_dlist")

        transport._agent.prep_xfer_dlist.side_effect = blocking_prep

        future = transport.add_remote_peer_async("peer:1", b"meta", 0x1000, 8, 1024)
        assert building.wait(timeout=5)
        # Supersede the registration while its worker still builds handles.
        transport.remove_remote_peer("peer:1")
        transport._agent.release_dlist_handle.side_effect = RuntimeError(
            "simulated release failure"
        )

        with patch.object(nixl_module.logger, "warning") as mock_warning:
            release.set()
            with pytest.raises(RuntimeError, match="superseded"):
                future.result(timeout=5)

            # The release failure was logged rather than swallowed silently.
            mock_warning.assert_called_once()
        # The superseded registration's agent was still removed.
        transport._agent.release_dlist_handle.assert_called_once()
        transport._agent.remove_remote_agent.assert_called_once_with("nixl-peer-name")
        assert "peer:1" not in transport._remote_dlists

    def test_close_releases_everything(self):
        """Close releases all handles and clears state."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        transport.write_blocks("peer:1", [0], [1])

        transport.close()
        assert transport._agent is None
        assert transport._inflight == {}
        assert transport._remote_dlists == {}


# ---------------------------------------------------------------------------
# Failure escalation: transfer scope vs peer scope
# ---------------------------------------------------------------------------


class nixlNotFoundError(Exception):  # noqa: N801
    """Stand-in for NIXL's binding exception of the same name."""


class nixlBackendError(Exception):  # noqa: N801
    """Stand-in for NIXL's binding exception of the same name: what a failed
    UCX transfer raises from check_xfer_state (the bindings raise for every
    NIXL error status rather than return "ERR")."""


class TestNixlPeerFaults:
    """The transport classifies each failure and reports a peer fault.

    See "Failure model" in data/base.py: a peer-scope NIXL error faults the
    peer at once; transfer-scope failures fault it only after
    _MAX_CONSECUTIVE_TRANSFER_FAILURES of them with no success between.
    """

    def _make_transport(self) -> NixlTransport:
        transport = TestNixlTransportWithMockedAgent()._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        transport.add_remote_peer("peer:2", b"meta", 0x2000, 8, 1024)
        return transport

    def _fail_one(self, transport: NixlTransport, peer_id: str = "peer:1"):
        """One transfer reported in state "ERR": the transport contract's
        transfer-scope failure (NIXL 1.5.0 raises instead; see below)."""
        tid = transport.write_blocks(peer_id, [0], [1])
        transport._agent.check_xfer_state.return_value = "ERR"
        result = transport.poll(peer_id=peer_id)
        assert tid in result.failed
        return result

    def test_consecutive_transfer_failures_fault_the_peer(self):
        transport = self._make_transport()
        assert self._fail_one(transport).peer_fault is None
        assert self._fail_one(transport).peer_fault is None
        fault = self._fail_one(transport).peer_fault
        assert fault is not None and "3 consecutive" in fault
        # Reported on every scoped poll, even with nothing inflight, and
        # never on a sibling's or the unscoped poll.
        assert transport.poll(peer_id="peer:1").peer_fault == fault
        assert transport.poll(peer_id="peer:2").peer_fault is None
        assert transport.poll().peer_fault is None

    def test_success_resets_the_strike_count(self):
        transport = self._make_transport()
        self._fail_one(transport)
        self._fail_one(transport)
        transport.write_blocks("peer:1", [0], [1])
        transport._agent.check_xfer_state.return_value = "DONE"
        transport.poll(peer_id="peer:1")
        self._fail_one(transport)
        assert self._fail_one(transport).peer_fault is None

    def test_strikes_are_per_peer(self):
        transport = self._make_transport()
        self._fail_one(transport, "peer:1")
        self._fail_one(transport, "peer:2")
        self._fail_one(transport, "peer:1")
        assert self._fail_one(transport, "peer:2").peer_fault is None

    def test_remote_disconnect_faults_at_once(self):
        transport = self._make_transport()
        tid = transport.write_blocks("peer:1", [0], [1])
        transport._agent.check_xfer_state.side_effect = _remote_disconnect()
        result = transport.poll(peer_id="peer:1")
        assert tid in result.failed
        assert result.peer_fault == "NIXL_ERR_REMOTE_DISCONNECT"

    def test_remote_disconnect_in_cancel_faults_the_peer(self):
        transport = self._make_transport()
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        transport._agent.check_xfer_state.side_effect = _remote_disconnect()
        assert transport.cancel([tid], mode="wait") == []
        transport._agent.check_xfer_state.side_effect = None
        fault = transport.poll(peer_id="peer:1").peer_fault
        assert fault == "NIXL_ERR_REMOTE_DISCONNECT"

    def test_submit_not_found_faults_at_once(self):
        transport = self._make_transport()
        transport._agent.make_prepped_xfer.side_effect = nixlNotFoundError(
            "NIXL_ERR_NOT_FOUND"
        )
        assert transport.write_blocks("peer:1", [0], [1]) is None
        assert transport._inflight == {}
        fault = transport.poll(peer_id="peer:1").peer_fault
        assert fault == "NIXL_ERR_NOT_FOUND"

    def test_failed_submissions_fault_the_peer(self):
        transport = self._make_transport()
        transport._agent.transfer.side_effect = RuntimeError("post failed")
        for _ in range(2):
            assert transport.write_blocks("peer:1", [0], [1]) is None
            assert transport.poll(peer_id="peer:1").peer_fault is None
        assert transport.write_blocks("peer:1", [0], [1]) is None
        assert transport.poll(peer_id="peer:1").peer_fault is not None

    def test_raising_state_check_faults_the_peer_and_keeps_the_transfer(self):
        """A failed transfer's state check raises. The WRITE may still be
        landing, so the transfer stays inflight, but its peer is faulted at
        once: strikes could be reset by other successes forever."""
        transport = self._make_transport()
        tid = transport.write_blocks("peer:1", [0], [1])
        transport._agent.check_xfer_state.side_effect = nixlBackendError(
            "NIXL_ERR_BACKEND"
        )
        for _ in range(3):
            result = transport.poll(peer_id="peer:1")
            assert result.failed == ()
            assert result.peer_fault == (
                "check_xfer_state raised nixlBackendError: NIXL_ERR_BACKEND"
            )
        assert tid in transport._inflight
        assert transport.poll(peer_id="peer:2").peer_fault is None

    def test_raise_faults_despite_a_success_in_the_batch(self):
        transport = self._make_transport()
        self._distinct_handles(
            transport, ["DONE", nixlBackendError("NIXL_ERR_BACKEND"), "DONE"]
        )
        result = transport.poll(peer_id="peer:1")
        assert len(result.done) == 2
        assert result.peer_fault is not None

    def _distinct_handles(self, transport: NixlTransport, states: list):
        """Submit one transfer per entry in *states* to peer:1; each one's
        check_xfer_state returns (or raises) its entry. Returns the tids."""
        handles = [object() for _ in states]
        transport._agent.make_prepped_xfer.side_effect = handles
        tids = [transport.write_blocks("peer:1", [0], [1]) for _ in states]
        by_handle = dict(zip(map(id, handles), states))

        def check(handle):
            state = by_handle[id(handle)]
            if isinstance(state, Exception):
                raise state
            return state

        transport._agent.check_xfer_state.side_effect = check
        return tids

    def test_success_in_the_same_batch_prevents_the_fault(self):
        """A DONE in the batch that also holds the failure reaching the
        threshold resets the strikes first, whatever the iteration order."""
        transport = self._make_transport()
        self._fail_one(transport)
        self._fail_one(transport)
        # The ERR is the oldest transfer, so it is iterated first.
        tids = self._distinct_handles(transport, ["ERR", "DONE", "DONE", "DONE"])
        result = transport.poll(peer_id="peer:1")
        assert result.failed == [tids[0]]
        assert sorted(result.done) == sorted(tids[1:])
        assert result.peer_fault is None
        assert transport._peer_strikes == {"peer:1": 1}

    def test_batch_of_failures_still_faults_despite_a_success(self):
        transport = self._make_transport()
        self._distinct_handles(transport, ["ERR", "ERR", "ERR", "DONE"])
        assert transport.poll(peer_id="peer:1").peer_fault is not None

    def test_not_found_state_check_keeps_the_transfer(self):
        """#59790 measured a WRITE still landing while its handle raised
        NOT_FOUND, so NOT_FOUND faults the peer but is not terminal."""
        transport = self._make_transport()
        tid = transport.write_blocks("peer:1", [0], [1])
        transport._agent.check_xfer_state.side_effect = nixlNotFoundError(
            "NIXL_ERR_NOT_FOUND"
        )
        result = transport.poll(peer_id="peer:1")
        assert result.failed == ()
        assert result.peer_fault == "NIXL_ERR_NOT_FOUND"
        assert tid in transport._inflight

    def test_not_found_in_cancel_keeps_the_transfer(self):
        transport = self._make_transport()
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        transport._agent.check_xfer_state.side_effect = nixlNotFoundError(
            "NIXL_ERR_NOT_FOUND"
        )
        assert transport.cancel([tid], mode="wait") == [tid]
        assert tid in transport._inflight
        transport._agent.check_xfer_state.side_effect = None
        assert transport.poll(peer_id="peer:1").peer_fault == "NIXL_ERR_NOT_FOUND"

    def test_raising_state_check_warns_once(self, monkeypatch):
        warnings: list[str] = []
        monkeypatch.setattr(
            nixl_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg)
        )
        transport = self._make_transport()
        transport.write_blocks("peer:1", [0], [1])
        transport._agent.check_xfer_state.side_effect = nixlBackendError(
            "NIXL_ERR_BACKEND"
        )
        for _ in range(5):
            transport.poll(peer_id="peer:1")
        assert sum("check_xfer_state failed" in w for w in warnings) == 1

    def test_remove_remote_peer_clears_the_fault(self):
        transport = self._make_transport()
        for _ in range(3):
            self._fail_one(transport)
        transport.remove_remote_peer("peer:1")
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        assert transport.poll(peer_id="peer:1").peer_fault is None
        assert self._fail_one(transport).peer_fault is None


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
