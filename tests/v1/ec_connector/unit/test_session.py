# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ProducerXfer, ProducerSession, ConsumerXfer, ConsumerSession."""

import time
from unittest.mock import MagicMock

import pytest

from vllm.distributed.ec_transfer.ec_connector.cpu.protocol import (
    XferAck,
    XferReq,
    XferStatus,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.embedding_cache import (
    EmbeddingCache,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.session import (
    _CONSUMER_QUARANTINE_TIMEOUT_S,
    ConsumerSession,
    ConsumerXfer,
    ProducerSession,
    ProducerXfer,
    XferState,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_data(xfer_state: str = "PROC") -> MagicMock:
    data = MagicMock()
    data.check_xfer_state.return_value = xfer_state
    data.post_read.return_value = 99
    return data


def _ok_ack(
    mm_hash: str = "h1",
    session_id: str = "sess-1",
    src_indices: list | None = None,
) -> XferAck:
    return XferAck(
        mm_hash=mm_hash,
        status=XferStatus.OK,
        session_id=session_id,
        src_block_indices=src_indices or [0, 1],
        agent_metadata=b"meta",
        mem_descriptor=b"desc",
    )


def _xfer(
    data: MagicMock | None = None,
    deadline: float | None = None,
    consumer_session_id: str = "sess-1",
) -> ConsumerXfer:
    return ConsumerXfer(
        mm_hash="h1",
        block_indices=[10, 11],
        addr=("host", 1234),
        deadline=deadline if deadline is not None else time.monotonic() + 60,
        data=data or _make_data(),
        consumer_session_id=consumer_session_id,
    )


def _started(
    data: MagicMock | None = None, deadline: float | None = None
) -> ConsumerXfer:
    """ConsumerXfer that has already received an OK XferAck."""
    x = _xfer(data, deadline)
    x.handle_ack(_ok_ack(session_id="sess-1", src_indices=[5, 6]), agent_name="agent-1")
    return x


# ── ProducerXfer ──────────────────────────────────────────────────────────────


def test_producer_xfer_not_expired_before_deadline():
    x = ProducerXfer("h1", [0, 1], deadline=time.monotonic() + 60)
    assert not x.is_expired()


def test_producer_xfer_expired_after_deadline():
    x = ProducerXfer("h1", [0, 1], deadline=time.monotonic() - 1)
    assert x.is_expired()


# ── ProducerSession ───────────────────────────────────────────────────────────


def _xfer_req(mm_hash: str = "h1", session_id: str = "sess-1") -> XferReq:
    return XferReq(
        mm_hash=mm_hash, compatibility_hash="hash-abc", session_id=session_id
    )


def _make_producer_session(cache: EmbeddingCache | None = None) -> ProducerSession:
    data = _make_data()
    data.get_agent_metadata.return_value = b"meta"
    data.get_mem_descriptor.return_value = b"desc"
    return ProducerSession(
        transport=MagicMock(),
        data=data,
        cache=cache or EmbeddingCache(num_blocks=8),
        compat_hash="hash-abc",
    )


def test_producer_session_grant_pins_ready_entry_and_returns_blocks():
    cache = EmbeddingCache(num_blocks=8)
    cache.alloc("h1", 2)
    cache.mark_ready("h1")
    s = _make_producer_session(cache)

    ack = s._grant_or_nack(_xfer_req(mm_hash="h1"))

    assert ack.status == XferStatus.OK
    entry = cache.get("h1")
    assert not entry.evictable  # pinned by the grant
    assert list(ack.src_block_indices) == list(entry.block_ids)


def test_producer_session_grant_nacks_missing_entry():
    s = _make_producer_session()
    ack = s._grant_or_nack(_xfer_req(mm_hash="nope"))
    assert ack.status == XferStatus.NACK_MISSING


def test_producer_session_grant_nacks_not_ready_entry():
    cache = EmbeddingCache(num_blocks=8)
    cache.alloc("h1", 2)  # not ready
    s = _make_producer_session(cache)
    ack = s._grant_or_nack(_xfer_req(mm_hash="h1"))
    assert ack.status == XferStatus.NACK_MISSING


def test_producer_session_grant_nacks_version_mismatch():
    cache = EmbeddingCache(num_blocks=8)
    cache.alloc("h1", 2)
    cache.mark_ready("h1")
    s = _make_producer_session(cache)
    req = XferReq(
        mm_hash="h1",
        compatibility_hash="hash-abc",
        session_id="sess-1",
        connector_version=-1,
    )

    ack = s._grant_or_nack(req)

    assert ack.status == XferStatus.NACK_VERSION
    entry = cache.get("h1")
    assert entry.evictable  # nothing was pinned


def test_producer_session_grant_nacks_compat_hash_mismatch():
    cache = EmbeddingCache(num_blocks=8)
    cache.alloc("h1", 2)
    cache.mark_ready("h1")
    s = _make_producer_session(cache)
    req = XferReq(mm_hash="h1", compatibility_hash="wrong-hash", session_id="sess-1")

    ack = s._grant_or_nack(req)

    assert ack.status == XferStatus.NACK_INCOMPAT
    entry = cache.get("h1")
    assert entry.evictable  # nothing was pinned


def test_producer_session_notif_unpins_via_cache():
    cache = EmbeddingCache(num_blocks=8)
    cache.alloc("h1", 2)
    cache.mark_ready("h1")
    s = _make_producer_session(cache)
    s._grant_or_nack(_xfer_req(mm_hash="h1", session_id="sess-1"))
    entry = cache.get("h1")
    assert not entry.evictable  # pinned

    s._data.get_new_notifs.return_value = {"agent": [b"sess-1:h1"]}
    s._drain_notifs()

    assert entry.evictable  # unpinned via cache
    assert s._active_xfers == {}


def test_producer_session_sweep_timeouts_unpins_via_cache():
    cache = EmbeddingCache(num_blocks=8)
    cache.alloc("h1", 2)
    cache.mark_ready("h1")
    s = _make_producer_session(cache)
    s._grant_or_nack(_xfer_req(mm_hash="h1", session_id="sess-1"))
    key = next(iter(s._active_xfers))
    s._active_xfers[key].deadline = time.monotonic() - 1

    s._sweep_timeouts()

    assert cache.get("h1").evictable
    assert s._active_xfers == {}


# ── ConsumerXfer.handle_ack ───────────────────────────────────────────────────


def test_consumer_xfer_handle_ack_ok_starts_read():
    data = _make_data()
    x = _xfer(data)
    assert x.handle_ack(_ok_ack(), agent_name="agent-1") is True
    assert x.transfer_handle == 99


def test_consumer_xfer_handle_ack_notif_msg_is_session_id_colon_mm_hash():
    data = _make_data()
    x = _xfer(data, consumer_session_id="my-sess")
    x.handle_ack(_ok_ack(mm_hash="h1"), agent_name="agent-1")
    _, kwargs = data.post_read.call_args
    assert kwargs["notif_msg"] == b"my-sess:h1"


def test_consumer_xfer_handle_ack_forwards_correct_args():
    data = _make_data()
    x = _xfer(data)
    x.handle_ack(_ok_ack(src_indices=[5, 6]), agent_name="agent-1")
    args, _ = data.post_read.call_args
    assert args[0] == [10, 11]  # local block_indices
    assert args[1] == "agent-1"  # agent_name (not a raw handle)
    assert args[2] == [5, 6]  # remote src_block_indices


def test_consumer_xfer_handle_ack_resets_deadline():
    original = time.monotonic() + 2
    x = _xfer(deadline=original)
    x.handle_ack(_ok_ack(), agent_name="agent-1")
    assert x.deadline > original


def test_consumer_xfer_handle_ack_nack_returns_false():
    data = _make_data()
    x = _xfer(data)
    assert (
        x.handle_ack(XferAck(mm_hash="h1", status=XferStatus.NACK_MISSING), "a")
        is False
    )
    assert x.transfer_handle is None
    data.post_read.assert_not_called()


# ── ConsumerXfer.poll — WAITING_ACK ──────────────────────────────────────────


def test_consumer_xfer_poll_waiting_ack_before_deadline():
    assert _xfer().poll(time.monotonic()) == XferState.WAITING_ACK


def test_consumer_xfer_poll_waiting_ack_timeout():
    assert (
        _xfer(deadline=time.monotonic() - 1).poll(time.monotonic())
        == XferState.ACK_TIMEOUT
    )


# ── ConsumerXfer.poll — READING ───────────────────────────────────────────────


def test_consumer_xfer_poll_reading_proc():
    assert _started().poll(time.monotonic()) == XferState.READING


def test_consumer_xfer_poll_reading_done_releases_handle():
    data = _make_data(xfer_state="DONE")
    x = _started(data)
    assert x.poll(time.monotonic()) == XferState.DONE
    data.release_xfer_handle.assert_called_once_with(99)
    assert x.transfer_handle is None


def test_consumer_xfer_poll_reading_unexpected_state_fails():
    data = _make_data(xfer_state="ERR")
    x = _started(data)
    assert x.poll(time.monotonic()) == XferState.READ_FAILED
    data.release_xfer_handle.assert_called_once_with(99)


def test_consumer_xfer_poll_reading_exception_fails_gracefully():
    data = _make_data()
    data.check_xfer_state.side_effect = RuntimeError("nixl crash")
    x = _started(data)
    assert x.poll(time.monotonic()) == XferState.READ_FAILED


def test_consumer_xfer_poll_timeout_quarantines_without_releasing():
    data = _make_data(xfer_state="PROC")
    x = _started(data)
    assert x.poll(time.monotonic() + 9999) == XferState.QUARANTINED
    data.release_xfer_handle.assert_not_called()
    assert x.transfer_handle == 99


# ── ConsumerXfer.poll — QUARANTINED ───────────────────────────────────────────


def test_consumer_xfer_quarantined_proc_stays():
    data = _make_data(xfer_state="PROC")
    x = _started(data)
    x.poll(time.monotonic() + 9999)  # → QUARANTINED
    assert x.poll(time.monotonic()) == XferState.QUARANTINED
    data.release_xfer_handle.assert_not_called()


def test_consumer_xfer_quarantined_terminal_settles():
    data = _make_data(xfer_state="PROC")
    x = _started(data)
    x.poll(time.monotonic() + 9999)  # → QUARANTINED
    data.check_xfer_state.return_value = "DONE"
    assert x.poll(time.monotonic()) == XferState.SETTLED
    data.release_xfer_handle.assert_called_once_with(99)
    assert x.transfer_handle is None


def test_consumer_xfer_quarantined_exception_settles():
    data = _make_data(xfer_state="PROC")
    x = _started(data)
    x.poll(time.monotonic() + 9999)
    data.check_xfer_state.side_effect = RuntimeError("gone")
    assert x.poll(time.monotonic()) == XferState.SETTLED


def test_consumer_xfer_quarantined_timeout_evicts_blocks():
    """Quarantine timeout should release handle and return READ_FAILED."""

    data = _make_data(xfer_state="PROC")
    x = _started(data)
    # First poll transitions to QUARANTINED (deadline extended by 60s)
    x.poll(time.monotonic() + 9999)
    # Second poll with expired quarantine deadline should evict
    expired_time = time.monotonic() + 9999 + _CONSUMER_QUARANTINE_TIMEOUT_S + 1
    assert x.poll(expired_time) == XferState.READ_FAILED
    data.release_xfer_handle.assert_called_once_with(99)
    assert x.transfer_handle is None


# ── ConsumerXfer.cancel / release ────────────────────────────────────────────


def test_consumer_xfer_cancel_valid_in_waiting_ack():
    _xfer().cancel()  # must not raise


def test_consumer_xfer_cancel_raises_when_read_started():
    with pytest.raises(AssertionError):
        _started().cancel()


def test_consumer_xfer_release_releases_handle():
    data = _make_data()
    x = _started(data)
    x.release()
    data.release_xfer_handle.assert_called_once_with(99)


def test_consumer_xfer_release_noop_when_no_handle():
    data = _make_data()
    _xfer(data).release()
    data.release_xfer_handle.assert_not_called()


# ── ConsumerSession ───────────────────────────────────────────────────────────


def _make_consumer_session(data=None):
    zmq_conn = MagicMock()
    zmq_conn.recv.return_value = []
    return ConsumerSession(
        addr=("host", 1234),
        zmq_conn=zmq_conn,
        transport=MagicMock(),
        data=data or _make_data(),
        compat_hash="hash-abc",
    )


def test_consumer_session_has_stable_session_id():
    s = _make_consumer_session()
    assert isinstance(s._session_id, str) and len(s._session_id) > 0
    assert s._session_id == s._session_id  # stable


def test_consumer_session_start_xfer_sends_req_with_session_id():
    import msgspec

    s = _make_consumer_session()
    s.start_xfer("h1", [0, 1], deadline=time.monotonic() + 10)
    raw = s._zmq.send.call_args[0][0]
    req = msgspec.msgpack.decode(raw, type=XferReq)
    assert req.mm_hash == "h1"
    assert req.session_id == s._session_id


def test_consumer_session_poll_done_goes_to_completed():
    import msgspec

    data = _make_data(xfer_state="PROC")
    s = _make_consumer_session(data)
    s.start_xfer("h1", [0, 1], deadline=time.monotonic() + 10)

    ack = XferAck(
        mm_hash="h1",
        status=XferStatus.OK,
        session_id=s._session_id,
        src_block_indices=[5, 6],
        agent_metadata=b"meta",
        mem_descriptor=b"desc",
    )
    data.add_remote_peer.return_value = "agent-1"
    raw_ack = msgspec.msgpack.encode(ack)

    data.check_xfer_state.return_value = "DONE"
    s.poll([raw_ack], time.monotonic())

    results = s.take_results()
    assert "h1" in results.completed
    assert not results.tombstoned


def test_consumer_session_poll_nack_goes_to_tombstoned():
    import msgspec

    s = _make_consumer_session()
    s.start_xfer("h1", [0, 1], deadline=time.monotonic() + 10)
    nack = XferAck(mm_hash="h1", status=XferStatus.NACK_MISSING)
    s.poll([msgspec.msgpack.encode(nack)], time.monotonic())
    results = s.take_results()
    assert "h1" in results.tombstoned
    assert not results.completed


def test_consumer_session_take_results_clears_state():
    s = _make_consumer_session()
    s.start_xfer("h1", [0, 1], deadline=time.monotonic() - 1)
    s.poll([], time.monotonic())
    r1 = s.take_results()
    assert "h1" in r1.tombstoned
    r2 = s.take_results()
    assert not r2.tombstoned  # cleared after first take


def test_consumer_session_on_peer_down_cancels_waiting_ack():
    s = _make_consumer_session()
    s.start_xfer("h1", [0, 1], deadline=time.monotonic() + 60)
    s.on_peer_down()
    results = s.take_results()
    assert "h1" in results.cancelled
    assert not results.tombstoned
