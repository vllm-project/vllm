# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for P2PClientSession and P2PServerSession.

Validates that client and server session protocols are compatible —
messages sent by one are correctly handled by the other.
"""

from __future__ import annotations

import time

import pytest

from vllm.v1.kv_offload.tiering.p2p.session.client import (
    LoadResult,
    P2PClientSession,
)
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    AbortAckMsg,
    AbortLookupFetchMsg,
    ConnectAckMsg,
    ConnectMsg,
    DisconnectMsg,
    LookupFetchMsg,
    TransferDoneMsg,
)
from vllm.v1.kv_offload.tiering.p2p.session.server import (
    P2PServerSession,
    StoreResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeDataTransport:
    """Minimal fake DataTransport for testing sessions."""

    def __init__(
        self,
        base_addr: int = 0x1000,
        num_blocks: int = 16,
        block_len: int = 4096,
        config_fingerprint: str = "",
    ) -> None:
        self._base_addr = base_addr
        self._num_blocks = num_blocks
        self._block_len = block_len
        self._config_fingerprint = config_fingerprint
        self._remote_peers: dict[str, dict] = {}
        self._transfers: dict[int, tuple] = {}
        self._next_id = 0
        self._poll_done: list[int] = []
        self._poll_failed: list[int] = []

    @property
    def base_addr(self) -> int:
        return self._base_addr

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def block_len(self) -> int:
        return self._block_len

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    def get_agent_metadata(self) -> bytes:
        return b"fake-metadata"

    def add_remote_peer(
        self, peer_id, agent_metadata, base_addr, num_blocks, block_len
    ) -> None:
        self._remote_peers[peer_id] = {
            ConnectMsg.AGENT_METADATA: agent_metadata,
            ConnectMsg.BASE_ADDR: base_addr,
            "num_blocks": num_blocks,
            "block_len": block_len,
        }

    def remove_remote_peer(self, peer_id: str) -> None:
        self._remote_peers.pop(peer_id, None)

    def write_blocks(self, peer_id, local_idxs, remote_idxs) -> int | None:
        if peer_id not in self._remote_peers:
            return None
        tid = self._next_id
        self._next_id += 1
        self._transfers[tid] = (peer_id, local_idxs, remote_idxs)
        return tid

    def poll(self):
        from vllm.v1.kv_offload.tiering.p2p.data.base import PollResult

        result = PollResult(done=list(self._poll_done), failed=list(self._poll_failed))
        self._poll_done.clear()
        self._poll_failed.clear()
        return result

    def cancel(self, transfer_ids) -> None:
        for tid in transfer_ids:
            self._transfers.pop(tid, None)

    def close(self) -> None:
        pass


class FakeConnection:
    """Fake ZmqConnection that captures sent messages."""

    def __init__(self, peer_id: str = "server:8000") -> None:
        self.peer_id = peer_id
        self._inbox: list[dict] = []
        self._sent: list[dict] = []
        self._closed = False

    @property
    def alive(self) -> bool:
        return not self._closed

    def send(self, msg: dict) -> None:
        self._sent.append(msg)

    def recv(self) -> list[dict]:
        msgs = self._inbox
        self._inbox = []
        return msgs

    def enqueue(self, msg: dict) -> None:
        self._inbox.append(msg)

    def mark_dead(self) -> None:
        self._closed = True

    def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Protocol compatibility tests
# ---------------------------------------------------------------------------


class TestProtocolCompatibility:
    """Validate that messages sent by client are understood by server
    and vice versa."""

    def _make_client(
        self, conn: FakeConnection, transport: FakeDataTransport
    ) -> P2PClientSession:
        """Create a client session (sends connect on init)."""
        return P2PClientSession(
            peer_id=conn.peer_id,
            conn=conn,  # type: ignore[arg-type]
            local_id="client:9000",
            transport=transport,  # type: ignore[arg-type]
        )

    def _make_server(
        self, conn: FakeConnection, transport: FakeDataTransport
    ) -> P2PServerSession:
        """Create a server session from a connect message."""
        # Simulate the connect message arriving
        connect_msg = {
            TYPE_KEY: ConnectMsg.TYPE,
            ConnectMsg.PEER_ID: "client:9000",
            ConnectMsg.AGENT_METADATA: b"client-metadata",
            ConnectMsg.BASE_ADDR: 0x2000,
            ConnectMsg.NUM_BLOCKS: 16,
            ConnectMsg.BLOCK_LEN: 4096,
        }
        conn.enqueue(connect_msg)
        return P2PServerSession(
            peer_id="client:9000",
            conn=conn,  # type: ignore[arg-type]
            local_id="server:8000",
            transport=transport,  # type: ignore[arg-type]
            local_block_len=4096,
        )

    def test_client_sends_valid_connect(self):
        """Client sends a connect message on init."""
        conn = FakeConnection()
        transport = FakeDataTransport()
        self._make_client(conn, transport)

        assert len(conn._sent) == 1
        msg = conn._sent[0]
        assert msg[TYPE_KEY] == ConnectMsg.TYPE
        assert msg[ConnectMsg.PEER_ID] == "client:9000"
        assert msg[ConnectMsg.NUM_BLOCKS] == 16
        assert msg[ConnectMsg.BLOCK_LEN] == 4096
        assert ConnectMsg.AGENT_METADATA in msg

    def test_server_accepts_connect_and_sends_ack(self):
        """Server processes connect and sends connect_ack."""
        conn = FakeConnection(peer_id="client:9000")
        transport = FakeDataTransport()
        server = self._make_server(conn, transport)

        assert server.peer_id == "client:9000"
        assert "client:9000" in transport._remote_peers

        # Server sends connect_ack
        assert len(conn._sent) == 1
        ack = conn._sent[0]
        assert ack[TYPE_KEY] == ConnectAckMsg.TYPE
        assert ack[ConnectAckMsg.PEER_ID] == "server:8000"

    def test_client_activates_on_connect_ack(self):
        """Client becomes ready after receiving connect_ack."""
        conn = FakeConnection()
        transport = FakeDataTransport()
        client = self._make_client(conn, transport)

        assert not client.ready

        # Simulate connect_ack arrival
        conn.enqueue(
            {TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "server:8000"}
        )
        client.poll()

        assert client.ready

    def test_client_queues_messages_before_ack(self):
        """Messages sent before connect_ack are queued and flushed after."""
        conn = FakeConnection()
        transport = FakeDataTransport()
        client = self._make_client(conn, transport)

        # Request blocks before ack — should queue
        client.request_blocks(
            job_id=1,
            kv_request_id="req-1",
            keys=[b"key1"],
            block_ids=[0],
        )

        # Only connect message sent so far
        assert len(conn._sent) == 1
        assert conn._sent[0][TYPE_KEY] == ConnectMsg.TYPE

        # Now ack arrives
        conn.enqueue(
            {TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "server:8000"}
        )
        client.poll()

        # Queued lookup_fetch should now be sent
        assert len(conn._sent) == 2
        assert conn._sent[1][TYPE_KEY] == LookupFetchMsg.TYPE

    def test_lookup_fetch_round_trip(self):
        """Client sends lookup_fetch, server receives and matches blocks."""
        server_conn = FakeConnection(peer_id="client:9000")
        server_transport = FakeDataTransport()
        server = self._make_server(server_conn, server_transport)

        # Pre-store blocks on server
        server.add_stored_blocks(
            kv_request_id="req-1",
            keys=[b"key1", b"key2"],
            block_ids=[0, 1],
            job_id=10,
        )

        # Simulate lookup_fetch from client
        lookup_msg = {
            TYPE_KEY: LookupFetchMsg.TYPE,
            LookupFetchMsg.KV_REQUEST_ID: "req-1",
            LookupFetchMsg.BLOCK_HASHES: [b"key1", b"key2"],
            LookupFetchMsg.BLOCK_INDEXES: [5, 6],
        }
        server_conn.enqueue(lookup_msg)

        # Poll triggers processing
        server.poll()

        # Transfer should have been submitted
        assert len(server_transport._transfers) == 1

    def test_transfer_done_round_trip(self):
        """Server sends transfer_done, client receives completion."""
        conn = FakeConnection()
        transport = FakeDataTransport()
        client = self._make_client(conn, transport)

        # Activate client
        conn.enqueue(
            {TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "server:8000"}
        )
        client.poll()

        # Request blocks
        client.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )

        # Simulate transfer_done from server
        conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.KV_REQUEST_ID: "req-1",
                TransferDoneMsg.SUCCESS: True,
            }
        )

        results = client.poll()
        assert len(results) == 1
        assert results[0] == LoadResult(job_id=1, kv_request_id="req-1", success=True)

    def test_abort_round_trip(self):
        """Client aborts, server sends abort_ack, client receives failure."""
        # Server side
        server_conn = FakeConnection(peer_id="client:9000")
        server_transport = FakeDataTransport()
        server = self._make_server(server_conn, server_transport)

        # Client sends abort
        abort_msg = {
            TYPE_KEY: AbortLookupFetchMsg.TYPE,
            AbortLookupFetchMsg.KV_REQUEST_ID: "req-1",
        }
        server_conn.enqueue(abort_msg)
        server.poll()

        # Server should have sent abort_ack
        abort_ack = next(
            m for m in server_conn._sent if m[TYPE_KEY] == AbortAckMsg.TYPE
        )
        assert abort_ack[AbortAckMsg.KV_REQUEST_ID] == "req-1"

        # Client receives abort_ack
        client_conn = FakeConnection()
        client_transport = FakeDataTransport()
        client = self._make_client(client_conn, client_transport)
        client_conn.enqueue(
            {TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "server:8000"}
        )
        client.poll()

        client.request_blocks(
            job_id=2, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        client_conn.enqueue(abort_ack)
        results = client.poll()

        assert len(results) == 1
        assert results[0].success is False

    def test_disconnect_marks_dead(self):
        """Disconnect message marks connection dead on both sides."""
        # Client side
        client_conn = FakeConnection()
        client_transport = FakeDataTransport()
        client = self._make_client(client_conn, client_transport)
        client_conn.enqueue(
            {TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "server:8000"}
        )
        client.poll()

        client_conn.enqueue({TYPE_KEY: DisconnectMsg.TYPE})
        client.poll()
        assert not client.alive

        # Server side
        server_conn = FakeConnection(peer_id="client:9000")
        server_transport = FakeDataTransport()
        server = self._make_server(server_conn, server_transport)

        server_conn.enqueue({TYPE_KEY: DisconnectMsg.TYPE})
        server.poll()
        assert not server.alive


# ---------------------------------------------------------------------------
# Server session unit tests
# ---------------------------------------------------------------------------


class TestP2PServerSession:
    """Unit tests for P2PServerSession."""

    def _make_server(
        self,
    ) -> tuple[P2PServerSession, FakeConnection, FakeDataTransport]:
        conn = FakeConnection(peer_id="client:9000")
        transport = FakeDataTransport()
        connect_msg = {
            TYPE_KEY: ConnectMsg.TYPE,
            ConnectMsg.PEER_ID: "client:9000",
            ConnectMsg.AGENT_METADATA: b"meta",
            ConnectMsg.BASE_ADDR: 0x2000,
            ConnectMsg.NUM_BLOCKS: 16,
            ConnectMsg.BLOCK_LEN: 4096,
        }
        conn.enqueue(connect_msg)
        server = P2PServerSession(
            peer_id="client:9000",
            conn=conn,  # type: ignore[arg-type]
            local_id="server:8000",
            transport=transport,  # type: ignore[arg-type]
            local_block_len=4096,
        )
        return server, conn, transport

    def test_block_len_mismatch_raises(self):
        """Mismatched block_len raises ValueError."""
        conn = FakeConnection(peer_id="client:9000")
        transport = FakeDataTransport(block_len=4096)
        connect_msg = {
            TYPE_KEY: ConnectMsg.TYPE,
            ConnectMsg.PEER_ID: "client:9000",
            ConnectMsg.AGENT_METADATA: b"meta",
            ConnectMsg.BASE_ADDR: 0x2000,
            ConnectMsg.NUM_BLOCKS: 16,
            ConnectMsg.BLOCK_LEN: 8192,  # mismatch!
        }
        conn.enqueue(connect_msg)
        with pytest.raises(ValueError, match="block_len mismatch"):
            P2PServerSession(
                peer_id="client:9000",
                conn=conn,  # type: ignore[arg-type]
                local_id="server:8000",
                transport=transport,  # type: ignore[arg-type]
                local_block_len=4096,
            )

    def test_config_fingerprint_mismatch_raises(self):
        """Mismatched config fingerprint raises ValueError."""
        conn = FakeConnection(peer_id="client:9000")
        transport = FakeDataTransport(config_fingerprint="abc123")
        connect_msg = {
            TYPE_KEY: ConnectMsg.TYPE,
            ConnectMsg.PEER_ID: "client:9000",
            ConnectMsg.AGENT_METADATA: b"meta",
            ConnectMsg.BASE_ADDR: 0x2000,
            ConnectMsg.NUM_BLOCKS: 16,
            ConnectMsg.BLOCK_LEN: 4096,
            ConnectMsg.CONFIG_FINGERPRINT: "different_fingerprint",
        }
        conn.enqueue(connect_msg)
        with pytest.raises(ValueError, match="config fingerprint mismatch"):
            P2PServerSession(
                peer_id="client:9000",
                conn=conn,  # type: ignore[arg-type]
                local_id="server:8000",
                transport=transport,  # type: ignore[arg-type]
                local_block_len=4096,
            )

    def test_config_fingerprint_match_succeeds(self):
        """Matching config fingerprints allow connection."""
        conn = FakeConnection(peer_id="client:9000")
        transport = FakeDataTransport(config_fingerprint="same_fp")
        connect_msg = {
            TYPE_KEY: ConnectMsg.TYPE,
            ConnectMsg.PEER_ID: "client:9000",
            ConnectMsg.AGENT_METADATA: b"meta",
            ConnectMsg.BASE_ADDR: 0x2000,
            ConnectMsg.NUM_BLOCKS: 16,
            ConnectMsg.BLOCK_LEN: 4096,
            ConnectMsg.CONFIG_FINGERPRINT: "same_fp",
        }
        conn.enqueue(connect_msg)
        server = P2PServerSession(
            peer_id="client:9000",
            conn=conn,  # type: ignore[arg-type]
            local_id="server:8000",
            transport=transport,  # type: ignore[arg-type]
            local_block_len=4096,
        )
        assert server.peer_id == "client:9000"

    def test_missing_fingerprint_allowed(self):
        """Missing fingerprint (empty) on either side is allowed."""
        conn = FakeConnection(peer_id="client:9000")
        transport = FakeDataTransport(config_fingerprint="abc123")
        connect_msg = {
            TYPE_KEY: ConnectMsg.TYPE,
            ConnectMsg.PEER_ID: "client:9000",
            ConnectMsg.AGENT_METADATA: b"meta",
            ConnectMsg.BASE_ADDR: 0x2000,
            ConnectMsg.NUM_BLOCKS: 16,
            ConnectMsg.BLOCK_LEN: 4096,
            # No fingerprint field — should be allowed
        }
        conn.enqueue(connect_msg)
        server = P2PServerSession(
            peer_id="client:9000",
            conn=conn,  # type: ignore[arg-type]
            local_id="server:8000",
            transport=transport,  # type: ignore[arg-type]
            local_block_len=4096,
        )
        assert server.peer_id == "client:9000"

    def test_store_then_fetch_matches(self):
        """Blocks stored before fetch are matched immediately."""
        server, conn, transport = self._make_server()

        server.add_stored_blocks("req-1", [b"k1", b"k2"], [0, 1], job_id=1)

        # Fetch arrives — should match
        conn.enqueue(
            {
                TYPE_KEY: LookupFetchMsg.TYPE,
                LookupFetchMsg.KV_REQUEST_ID: "req-1",
                LookupFetchMsg.BLOCK_HASHES: [b"k1", b"k2"],
                LookupFetchMsg.BLOCK_INDEXES: [10, 11],
            }
        )
        server.poll()

        assert len(transport._transfers) == 1
        tid, (peer, local, remote) = next(iter(transport._transfers.items()))
        assert local == [0, 1]
        assert remote == [10, 11]

    def test_fetch_then_store_matches(self):
        """Fetch arriving before store creates demand, store fulfills it."""
        server, conn, transport = self._make_server()

        # Fetch arrives first
        conn.enqueue(
            {
                TYPE_KEY: LookupFetchMsg.TYPE,
                LookupFetchMsg.KV_REQUEST_ID: "req-1",
                LookupFetchMsg.BLOCK_HASHES: [b"k1"],
                LookupFetchMsg.BLOCK_INDEXES: [5],
            }
        )
        server.poll()

        # No transfer yet (blocks not available)
        assert len(transport._transfers) == 0

        # Store arrives — should match
        server.add_stored_blocks("req-1", [b"k1"], [3], job_id=1)
        assert len(transport._transfers) == 1

    def test_transfer_completion_reports_store_result(self):
        """Completed transfer reports StoreResult."""
        server, conn, transport = self._make_server()

        server.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        conn.enqueue(
            {
                TYPE_KEY: LookupFetchMsg.TYPE,
                LookupFetchMsg.KV_REQUEST_ID: "req-1",
                LookupFetchMsg.BLOCK_HASHES: [b"k1"],
                LookupFetchMsg.BLOCK_INDEXES: [5],
            }
        )
        server.poll()

        # Mark transfer as done
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)

        results = server.poll()
        assert StoreResult(job_id=1, success=True) in results

    def test_store_timeout(self):
        """Store jobs that exceed timeout are reported as failed."""
        server, conn, transport = self._make_server()

        server.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)

        # Manually backdate the submitted_at
        server._store_jobs[1] = time.monotonic() - 60.0

        results = server.poll()
        assert StoreResult(job_id=1, success=False) in results

    def test_close_returns_pending_job_ids(self):
        """Close returns job_ids of pending store jobs."""
        server, conn, transport = self._make_server()

        server.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        server.add_stored_blocks("req-2", [b"k2"], [1], job_id=2)

        failed = list(server.close())
        assert set(failed) == {1, 2}


# ---------------------------------------------------------------------------
# Client session unit tests
# ---------------------------------------------------------------------------


class TestP2PClientSession:
    """Unit tests for P2PClientSession."""

    def _make_client(
        self,
    ) -> tuple[P2PClientSession, FakeConnection, FakeDataTransport]:
        conn = FakeConnection()
        transport = FakeDataTransport()
        client = P2PClientSession(
            peer_id=conn.peer_id,
            conn=conn,  # type: ignore[arg-type]
            local_id="client:9000",
            transport=transport,  # type: ignore[arg-type]
        )
        # Activate
        conn.enqueue(
            {TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "server:8000"}
        )
        client.poll()
        return client, conn, transport

    def test_request_blocks_sends_lookup_fetch(self):
        """request_blocks sends a lookup_fetch message."""
        client, conn, _ = self._make_client()

        client.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k1", b"k2"], block_ids=[0, 1]
        )

        lookup = conn._sent[-1]
        assert lookup[TYPE_KEY] == LookupFetchMsg.TYPE
        assert lookup[LookupFetchMsg.KV_REQUEST_ID] == "req-1"
        assert lookup[LookupFetchMsg.BLOCK_HASHES] == [b"k1", b"k2"]
        assert lookup[LookupFetchMsg.BLOCK_INDEXES] == [0, 1]

    def test_transfer_done_success(self):
        """transfer_done with success=True reports LoadResult."""
        client, conn, _ = self._make_client()

        client.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )

        conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.KV_REQUEST_ID: "req-1",
                TransferDoneMsg.SUCCESS: True,
            }
        )
        results = client.poll()
        assert results == [LoadResult(job_id=1, kv_request_id="req-1", success=True)]

    def test_transfer_done_failure(self):
        """transfer_done with success=False reports failure."""
        client, conn, _ = self._make_client()

        client.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )

        conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.KV_REQUEST_ID: "req-1",
                TransferDoneMsg.SUCCESS: False,
            }
        )
        results = client.poll()
        assert results == [LoadResult(job_id=1, kv_request_id="req-1", success=False)]

    def test_cancel_request_sends_abort(self):
        """cancel_request sends abort_lookup_fetch."""
        client, conn, _ = self._make_client()

        client.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        client.cancel_request("req-1")

        abort = conn._sent[-1]
        assert abort[TYPE_KEY] == AbortLookupFetchMsg.TYPE
        assert abort[AbortLookupFetchMsg.KV_REQUEST_ID] == "req-1"

    def test_load_timeout_sends_abort(self):
        """Timed-out load sends abort_lookup_fetch."""
        client, conn, _ = self._make_client()

        client.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )

        # Backdate submitted_at
        client._inbound["req-1"].submitted_at = time.monotonic() - 60.0

        client.poll()

        abort = conn._sent[-1]
        assert abort[TYPE_KEY] == AbortLookupFetchMsg.TYPE

    def test_close_returns_pending_requests(self):
        """Close returns all pending (job_id, kv_request_id) pairs."""
        client, conn, _ = self._make_client()

        client.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k1"], block_ids=[0]
        )
        client.request_blocks(
            job_id=2, kv_request_id="req-2", keys=[b"k2"], block_ids=[1]
        )

        failed = list(client.close())
        assert set(failed) == {(1, "req-1"), (2, "req-2")}

    def test_unknown_transfer_done_logged(self):
        """transfer_done for unknown kv_request_id doesn't crash."""
        client, conn, _ = self._make_client()

        conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.KV_REQUEST_ID: "unknown",
                TransferDoneMsg.SUCCESS: True,
            }
        )
        # Should not raise, just log warning
        results = client.poll()
        assert results == []


# ---------------------------------------------------------------------------
# Adversarial peer tests
# ---------------------------------------------------------------------------


class TestAdversarialServer:
    """Tests for a malicious/buggy server sending invalid messages to client."""

    def _make_active_client(self) -> tuple[P2PClientSession, FakeConnection]:
        conn = FakeConnection()
        transport = FakeDataTransport()
        client = P2PClientSession(
            peer_id=conn.peer_id,
            conn=conn,  # type: ignore[arg-type]
            local_id="client:9000",
            transport=transport,  # type: ignore[arg-type]
        )
        conn.enqueue(
            {TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "server:8000"}
        )
        client.poll()
        return client, conn

    def test_missing_kv_request_id_in_transfer_done(self):
        """transfer_done without kv_request_id doesn't crash."""
        client, conn = self._make_active_client()
        conn.enqueue({TYPE_KEY: TransferDoneMsg.TYPE, TransferDoneMsg.SUCCESS: True})
        results = client.poll()
        assert results == []

    def test_missing_type_field(self):
        """Message without type field doesn't crash."""
        client, conn = self._make_active_client()
        conn.enqueue({"random": "garbage"})
        results = client.poll()
        assert results == []

    def test_wrong_type_for_kv_request_id(self):
        """kv_request_id as int instead of str doesn't crash."""
        client, conn = self._make_active_client()
        conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.KV_REQUEST_ID: 12345,
                TransferDoneMsg.SUCCESS: True,
            }
        )
        results = client.poll()
        assert results == []

    def test_unknown_message_type(self):
        """Unknown message type is logged, not crashed."""
        client, conn = self._make_active_client()
        conn.enqueue({TYPE_KEY: "evil_command", "payload": "x" * 1000})
        results = client.poll()
        assert results == []

    def test_duplicate_connect_ack(self):
        """Multiple connect_ack messages don't crash."""
        client, conn = self._make_active_client()
        # Already activated, send another ack
        conn.enqueue(
            {TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "server:8000"}
        )
        results = client.poll()
        assert results == []
        assert client.ready  # still ready

    def test_empty_message(self):
        """Empty dict message doesn't crash."""
        client, conn = self._make_active_client()
        conn.enqueue({})
        results = client.poll()
        assert results == []

    def test_non_dict_message(self):
        """Non-dict message in inbox doesn't crash."""
        client, conn = self._make_active_client()
        conn._inbox.append("not a dict")  # type: ignore[arg-type]
        results = client.poll()
        assert results == []


class TestAdversarialClient:
    """Tests for a malicious/buggy client sending invalid messages to server."""

    def _make_server(
        self,
    ) -> tuple[P2PServerSession, FakeConnection, FakeDataTransport]:
        conn = FakeConnection(peer_id="client:9000")
        transport = FakeDataTransport()
        connect_msg = {
            TYPE_KEY: ConnectMsg.TYPE,
            ConnectMsg.PEER_ID: "client:9000",
            ConnectMsg.AGENT_METADATA: b"meta",
            ConnectMsg.BASE_ADDR: 0x2000,
            ConnectMsg.NUM_BLOCKS: 16,
            ConnectMsg.BLOCK_LEN: 4096,
        }
        conn.enqueue(connect_msg)
        server = P2PServerSession(
            peer_id="client:9000",
            conn=conn,  # type: ignore[arg-type]
            local_id="server:8000",
            transport=transport,  # type: ignore[arg-type]
            local_block_len=4096,
        )
        return server, conn, transport

    def test_lookup_fetch_missing_fields(self):
        """lookup_fetch without required fields doesn't crash."""
        server, conn, _ = self._make_server()
        conn.enqueue({TYPE_KEY: LookupFetchMsg.TYPE})
        results = server.poll()
        # Should not crash, just log
        assert isinstance(results, list)

    def test_lookup_fetch_mismatched_lengths(self):
        """lookup_fetch with mismatched hashes/indexes doesn't crash."""
        server, conn, transport = self._make_server()
        conn.enqueue(
            {
                TYPE_KEY: LookupFetchMsg.TYPE,
                LookupFetchMsg.KV_REQUEST_ID: "req-bad",
                LookupFetchMsg.BLOCK_HASHES: [b"k1", b"k2"],
                LookupFetchMsg.BLOCK_INDEXES: [1],  # mismatch!
            }
        )
        server.poll()
        # No transfer submitted
        assert len(transport._transfers) == 0

    def test_lookup_fetch_non_bytes_hashes(self):
        """lookup_fetch with non-bytes block_hashes doesn't crash."""
        server, conn, transport = self._make_server()
        conn.enqueue(
            {
                TYPE_KEY: LookupFetchMsg.TYPE,
                LookupFetchMsg.KV_REQUEST_ID: "req-x",
                LookupFetchMsg.BLOCK_HASHES: [123, None],  # wrong types
                LookupFetchMsg.BLOCK_INDEXES: [0, 1],
            }
        )
        results = server.poll()
        assert isinstance(results, list)

    def test_abort_missing_kv_request_id(self):
        """abort_lookup_fetch without kv_request_id doesn't crash."""
        server, conn, _ = self._make_server()
        conn.enqueue({TYPE_KEY: AbortLookupFetchMsg.TYPE})
        results = server.poll()
        assert isinstance(results, list)

    def test_unknown_message_type(self):
        """Unknown message type from client doesn't crash server."""
        server, conn, _ = self._make_server()
        conn.enqueue({TYPE_KEY: "hack_the_planet"})
        results = server.poll()
        assert isinstance(results, list)

    def test_empty_message(self):
        """Empty dict from client doesn't crash server."""
        server, conn, _ = self._make_server()
        conn.enqueue({})
        results = server.poll()
        assert isinstance(results, list)

    def test_non_dict_message(self):
        """Non-dict in inbox doesn't crash server."""
        server, conn, _ = self._make_server()
        conn._inbox.append(42)  # type: ignore[arg-type]
        results = server.poll()
        assert isinstance(results, list)

    def test_massive_block_list(self):
        """Very large block list doesn't crash (may be slow)."""
        server, conn, transport = self._make_server()
        # Pre-store one block to enable matching
        server.add_stored_blocks("req-big", [b"k"], [0], job_id=1)
        conn.enqueue(
            {
                TYPE_KEY: LookupFetchMsg.TYPE,
                LookupFetchMsg.KV_REQUEST_ID: "req-big",
                LookupFetchMsg.BLOCK_HASHES: [b"k"] * 10000,
                LookupFetchMsg.BLOCK_INDEXES: list(range(10000)),
            }
        )
        results = server.poll()
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Protocol validation tests
# ---------------------------------------------------------------------------


class TestConnectMsgValidation:
    """Unit tests for ConnectMsg.validate()."""

    def _valid_msg(self) -> dict:
        return {
            TYPE_KEY: ConnectMsg.TYPE,
            ConnectMsg.PEER_ID: "peer:1",
            ConnectMsg.AGENT_METADATA: b"meta",
            ConnectMsg.BASE_ADDR: 0x1000,
            ConnectMsg.NUM_BLOCKS: 8,
            ConnectMsg.BLOCK_LEN: 4096,
        }

    def test_valid_message_passes(self):
        ConnectMsg.validate(self._valid_msg())

    def test_missing_peer_id(self):
        msg = self._valid_msg()
        del msg[ConnectMsg.PEER_ID]
        with pytest.raises(ValueError, match="peer_id"):
            ConnectMsg.validate(msg)

    def test_missing_agent_metadata(self):
        msg = self._valid_msg()
        del msg[ConnectMsg.AGENT_METADATA]
        with pytest.raises(ValueError, match="agent_metadata"):
            ConnectMsg.validate(msg)

    def test_agent_metadata_wrong_type(self):
        msg = self._valid_msg()
        msg[ConnectMsg.AGENT_METADATA] = "not bytes"
        with pytest.raises(ValueError, match="agent_metadata"):
            ConnectMsg.validate(msg)

    def test_base_addr_negative(self):
        msg = self._valid_msg()
        msg[ConnectMsg.BASE_ADDR] = -1
        with pytest.raises(ValueError, match="base_addr"):
            ConnectMsg.validate(msg)

    def test_base_addr_not_int(self):
        msg = self._valid_msg()
        msg[ConnectMsg.BASE_ADDR] = "0x1000"
        with pytest.raises(ValueError, match="base_addr"):
            ConnectMsg.validate(msg)

    def test_num_blocks_zero(self):
        msg = self._valid_msg()
        msg[ConnectMsg.NUM_BLOCKS] = 0
        with pytest.raises(ValueError, match="num_blocks"):
            ConnectMsg.validate(msg)

    def test_num_blocks_negative(self):
        msg = self._valid_msg()
        msg[ConnectMsg.NUM_BLOCKS] = -1
        with pytest.raises(ValueError, match="num_blocks"):
            ConnectMsg.validate(msg)

    def test_block_len_zero(self):
        msg = self._valid_msg()
        msg[ConnectMsg.BLOCK_LEN] = 0
        with pytest.raises(ValueError, match="block_len"):
            ConnectMsg.validate(msg)


class TestLookupFetchMsgValidation:
    """Unit tests for LookupFetchMsg.validate()."""

    def _valid_msg(self) -> dict:
        return {
            TYPE_KEY: LookupFetchMsg.TYPE,
            LookupFetchMsg.KV_REQUEST_ID: "req-1",
            LookupFetchMsg.BLOCK_HASHES: [b"k1", b"k2"],
            LookupFetchMsg.BLOCK_INDEXES: [0, 1],
        }

    def test_valid_message_passes(self):
        LookupFetchMsg.validate(self._valid_msg())

    def test_missing_kv_request_id(self):
        msg = self._valid_msg()
        del msg[LookupFetchMsg.KV_REQUEST_ID]
        with pytest.raises(ValueError, match="kv_request_id"):
            LookupFetchMsg.validate(msg)

    def test_kv_request_id_wrong_type(self):
        msg = self._valid_msg()
        msg[LookupFetchMsg.KV_REQUEST_ID] = 123
        with pytest.raises(ValueError, match="kv_request_id"):
            LookupFetchMsg.validate(msg)

    def test_block_hashes_not_list(self):
        msg = self._valid_msg()
        msg[LookupFetchMsg.BLOCK_HASHES] = "not a list"
        with pytest.raises(ValueError, match="block_hashes"):
            LookupFetchMsg.validate(msg)

    def test_block_indexes_not_list(self):
        msg = self._valid_msg()
        msg[LookupFetchMsg.BLOCK_INDEXES] = 42
        with pytest.raises(ValueError, match="block_indexes"):
            LookupFetchMsg.validate(msg)

    def test_length_mismatch(self):
        msg = self._valid_msg()
        msg[LookupFetchMsg.BLOCK_INDEXES] = [0]
        with pytest.raises(ValueError, match="length mismatch"):
            LookupFetchMsg.validate(msg)

    def test_negative_index(self):
        msg = self._valid_msg()
        msg[LookupFetchMsg.BLOCK_INDEXES] = [0, -1]
        with pytest.raises(ValueError, match="invalid index"):
            LookupFetchMsg.validate(msg)

    def test_non_int_index(self):
        msg = self._valid_msg()
        msg[LookupFetchMsg.BLOCK_INDEXES] = [0, "bad"]
        with pytest.raises(ValueError, match="invalid index"):
            LookupFetchMsg.validate(msg)


class TestConnectAckMsgValidation:
    """Unit tests for ConnectAckMsg.validate()."""

    def test_valid_message_passes(self):
        msg = {TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "s:1"}
        ConnectAckMsg.validate(msg)

    def test_missing_peer_id(self):
        with pytest.raises(ValueError, match="peer_id"):
            ConnectAckMsg.validate({TYPE_KEY: ConnectAckMsg.TYPE})

    def test_peer_id_wrong_type(self):
        msg = {TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: 123}
        with pytest.raises(ValueError, match="peer_id"):
            ConnectAckMsg.validate(msg)


class TestTransferDoneMsgValidation:
    """Unit tests for TransferDoneMsg.validate()."""

    def test_valid_message_passes(self):
        msg = {
            TYPE_KEY: TransferDoneMsg.TYPE,
            TransferDoneMsg.KV_REQUEST_ID: "req-1",
            TransferDoneMsg.SUCCESS: True,
        }
        TransferDoneMsg.validate(msg)

    def test_missing_kv_request_id(self):
        msg = {TYPE_KEY: TransferDoneMsg.TYPE, TransferDoneMsg.SUCCESS: True}
        with pytest.raises(ValueError, match="kv_request_id"):
            TransferDoneMsg.validate(msg)

    def test_success_wrong_type(self):
        msg = {
            TYPE_KEY: TransferDoneMsg.TYPE,
            TransferDoneMsg.KV_REQUEST_ID: "req-1",
            TransferDoneMsg.SUCCESS: 1,
        }
        with pytest.raises(ValueError, match="success"):
            TransferDoneMsg.validate(msg)

    def test_success_missing(self):
        msg = {TYPE_KEY: TransferDoneMsg.TYPE, TransferDoneMsg.KV_REQUEST_ID: "req-1"}
        with pytest.raises(ValueError, match="success"):
            TransferDoneMsg.validate(msg)


class TestAbortLookupFetchMsgValidation:
    """Unit tests for AbortLookupFetchMsg.validate()."""

    def test_valid_message_passes(self):
        msg = {
            TYPE_KEY: AbortLookupFetchMsg.TYPE,
            AbortLookupFetchMsg.KV_REQUEST_ID: "req-1",
        }
        AbortLookupFetchMsg.validate(msg)

    def test_missing_kv_request_id(self):
        with pytest.raises(ValueError, match="kv_request_id"):
            AbortLookupFetchMsg.validate({TYPE_KEY: AbortLookupFetchMsg.TYPE})

    def test_kv_request_id_wrong_type(self):
        msg = {
            TYPE_KEY: AbortLookupFetchMsg.TYPE,
            AbortLookupFetchMsg.KV_REQUEST_ID: 42,
        }
        with pytest.raises(ValueError, match="kv_request_id"):
            AbortLookupFetchMsg.validate(msg)


class TestAbortAckMsgValidation:
    """Unit tests for AbortAckMsg.validate()."""

    def test_valid_message_passes(self):
        msg = {TYPE_KEY: AbortAckMsg.TYPE, AbortAckMsg.KV_REQUEST_ID: "req-1"}
        AbortAckMsg.validate(msg)

    def test_missing_kv_request_id(self):
        with pytest.raises(ValueError, match="kv_request_id"):
            AbortAckMsg.validate({TYPE_KEY: AbortAckMsg.TYPE})

    def test_kv_request_id_wrong_type(self):
        msg = {TYPE_KEY: AbortAckMsg.TYPE, AbortAckMsg.KV_REQUEST_ID: None}
        with pytest.raises(ValueError, match="kv_request_id"):
            AbortAckMsg.validate(msg)
