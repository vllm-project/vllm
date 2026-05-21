# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECCPUScheduler.

Mocking policy
--------------
Only the parts that require external I/O are mocked:
  - NixlWrapper / nixl_agent_config (nixl package may not be installed)
  - zmq.Context and the sockets it creates (would bind to real ports)
  - make_zmq_socket / make_zmq_path (same reason)
  - setup_ec_region (lets us inject a real ECSharedRegion with known dims)

Everything else is real: ECSharedRegion, threading primitives, msgspec codecs,
all scheduler logic. The router thread is replaced by a sentinel that parks
on _stop_event so it starts, can be joined, and never touches the mocked
sockets — the router-loop body itself is out of scope for unit tests; cover
it with an integration test using real inproc:// sockets.
"""

import contextlib
import logging
import uuid
from unittest.mock import MagicMock, Mock, patch

import msgspec
import pytest
import zmq
from utils import (  # noqa: E402  (test-local helper module)
    _BLOCK_SIZE,
    _ELEMENT_SIZE,
    _HIDDEN_DIM,
    _NUM_BLOCKS,
    _feature,
    _info,
    _make_layout,
    _make_nixl_mock,
    _make_vllm_config,
    _request_for,
)

from vllm.distributed.ec_transfer.ec_connector.cpu.metadata import (
    EC_CONNECTOR_VERSION,
    XferAck,
    XferReq,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    ConsumerPeer,
    ProducerPeer,
    serialize_mem_descriptor,
)
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    AllocationError,
    ECSharedRegion,
)
from vllm.v1.core.sched.output import SchedulerOutput

# ── scheduler-specific helpers ────────────────────────────────────────────────


def _inject_in_flight(
    sched: ECCPUScheduler, mm_hash: str, identity: bytes
) -> tuple[str, object]:
    """Add a fake in-flight entry; return (xfer_id, handle)."""
    handle = object()
    xfer_id = str(uuid.uuid4())
    sched._in_flight[xfer_id] = (identity, mm_hash, handle)
    return xfer_id, handle


_ack_encoder = msgspec.msgpack.Encoder()


def _dealer_returning(frames: list) -> MagicMock:
    """Mock DEALER that yields each frame list then raises zmq.Again."""
    dealer = MagicMock()
    dealer.recv_multipart.side_effect = frames + [zmq.Again]
    return dealer


def _ack_frames(mm_hash: str, ok: bool) -> list[bytes]:
    return [b"", _ack_encoder.encode(XferAck(mm_hash=mm_hash, ok=ok))]


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def make_scheduler():
    regions: list[ECSharedRegion] = []
    schedulers: list[ECCPUScheduler] = []

    def factory(
        is_producer: bool = False,
        is_consumer: bool = True,
    ) -> ECCPUScheduler:
        layout = _make_layout()
        regions.append(layout.region)

        mock_nixl = _make_nixl_mock()
        mock_router_sock = MagicMock()
        mock_inproc_recv = MagicMock()
        mock_inproc_send = MagicMock()
        mock_ctx = MagicMock()
        # Producer __init__ creates two PAIR sockets in this order:
        #   1) self._inproc_recv = ctx.socket(zmq.PAIR)
        #   2) self._inproc_send = ctx.socket(zmq.PAIR)
        # The side_effect order matches that creation order.
        mock_ctx.socket.side_effect = [mock_inproc_recv, mock_inproc_send]

        with (
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.NixlWrapper",
                return_value=mock_nixl,
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.nixl_agent_config",
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.setup_ec_region",
                return_value=layout,
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.make_zmq_socket",
                return_value=mock_router_sock,
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.make_zmq_path",
                return_value="tcp://mock:5000",
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.zmq.Context",
                return_value=mock_ctx,
            ),
        ):
            sched = ECCPUScheduler(_make_vllm_config(is_producer, is_consumer))

        schedulers.append(sched)
        return sched

    yield factory

    for sched in schedulers:
        with contextlib.suppress(Exception):
            sched.shutdown()
    for region in regions:
        with contextlib.suppress(Exception):
            region.cleanup()


@pytest.fixture
def producer(make_scheduler) -> ECCPUScheduler:
    return make_scheduler(is_producer=True, is_consumer=False)


@pytest.fixture
def consumer(make_scheduler) -> ECCPUScheduler:
    return make_scheduler(is_producer=False, is_consumer=True)


@pytest.fixture
def both(make_scheduler) -> ECCPUScheduler:
    return make_scheduler(is_producer=True, is_consumer=True)


# ── role gating (behavioral, not hasattr) ─────────────────────────────────────


def test_ensure_cache_available_returns_true_on_producer_only(producer):
    req = _request_for(_feature("h"))
    assert producer.ensure_cache_available(req, num_computed_tokens=0) is True
    # Producer has no consumer state; has_cache_item always returns False.
    assert producer.has_cache_item("h") is False


def test_producer_methods_no_op_on_consumer(consumer):
    done, params = consumer.request_finished(MagicMock())
    assert done is False and params is None
    consumer.update_state_after_alloc(MagicMock(), index=0)  # must not raise


# ── _producer_fifo_alloc ──────────────────────────────────────────────────────


def test_producer_alloc_succeeds_when_pool_has_space(producer):
    indices = producer._producer_fifo_alloc(3)
    assert len(indices) == 3


def test_producer_alloc_evicts_oldest_unpinned(producer):
    """Eviction iterates _local_encodings in insertion order (FIFO)."""
    region = producer._region
    old = region.alloc(4)
    new = region.alloc(4)
    producer._local_encodings["old"] = old
    producer._local_encodings["new"] = new

    result = producer._producer_fifo_alloc(3)

    assert len(result) == 3
    assert "old" not in producer._local_encodings  # FIFO entry was evicted
    assert "new" in producer._local_encodings


def test_producer_alloc_skips_pinned_blocks(producer):
    region = producer._region
    pinned = region.alloc(4)
    not_pinned = region.alloc(4)
    region.pin(pinned)
    producer._local_encodings["pinned"] = pinned
    producer._local_encodings["not_pinned"] = not_pinned

    result = producer._producer_fifo_alloc(3)

    assert len(result) == 3
    assert "pinned" in producer._local_encodings  # skipped: still pinned
    assert "not_pinned" not in producer._local_encodings
    region.unpin(pinned)


def test_producer_alloc_raises_when_all_encodings_pinned(producer):
    region = producer._region
    all_idx = region.alloc(_NUM_BLOCKS)
    region.pin(all_idx)
    producer._local_encodings["only"] = all_idx

    with pytest.raises(AllocationError):
        producer._producer_fifo_alloc(1)

    region.unpin(all_idx)


def test_producer_alloc_evicts_multiple_to_satisfy_request(producer):
    """A single eviction may not free enough; the loop must continue evicting."""
    region = producer._region
    a = region.alloc(2)
    b = region.alloc(2)
    c = region.alloc(2)
    d = region.alloc(2)  # pool now empty
    producer._local_encodings["a"] = a
    producer._local_encodings["b"] = b
    producer._local_encodings["c"] = c
    producer._local_encodings["d"] = d

    # Request 5 blocks: must evict a (2 freed → still short), b (4 → still
    # short), c (6 → satisfies). d should survive.
    result = producer._producer_fifo_alloc(5)

    assert len(result) == 5
    assert "a" not in producer._local_encodings
    assert "b" not in producer._local_encodings
    assert "c" not in producer._local_encodings
    assert "d" in producer._local_encodings


# ── update_state_after_alloc ──────────────────────────────────────────────────


def test_update_state_queues_size_bytes(producer):
    req = _request_for(_feature("h1", length=2))
    producer.update_state_after_alloc(req, index=0)

    expected_size = 2 * _HIDDEN_DIM * _ELEMENT_SIZE
    assert producer._pending_new_encodings == {"h1": expected_size}


def test_update_state_dedups_against_pending_save(producer):
    producer._pending_save["h1"] = [0]
    producer.update_state_after_alloc(_request_for(_feature("h1")), index=0)
    assert "h1" not in producer._pending_new_encodings


def test_update_state_dedups_against_local_encodings(producer):
    producer._local_encodings["h1"] = [0]
    producer.update_state_after_alloc(_request_for(_feature("h1")), index=0)
    assert "h1" not in producer._pending_new_encodings


# ── build_connector_meta (producer) ───────────────────────────────────────────


def test_build_meta_promotes_pending_save(producer):
    producer._pending_save["h1"] = [0, 1]

    producer.build_connector_meta(Mock(spec=SchedulerOutput))

    assert producer._local_encodings["h1"] == [0, 1]
    assert producer._pending_save == {}


def test_build_meta_allocates_for_pending_new_encodings(producer):
    producer._pending_new_encodings["h2"] = _BLOCK_SIZE  # exactly 1 block

    meta = producer.build_connector_meta(Mock(spec=SchedulerOutput))

    assert "h2" in meta.saves
    assert len(meta.saves["h2"]) == 1
    assert "h2" in producer._pending_save
    assert producer._pending_new_encodings == {}  # drained


def test_build_meta_rounds_up_partial_block(producer):
    producer._pending_new_encodings["h"] = _BLOCK_SIZE + 1
    meta = producer.build_connector_meta(Mock(spec=SchedulerOutput))
    assert len(meta.saves["h"]) == 2


# ── _handle_xfer_req ──────────────────────────────────────────────────────────


_EMPTY_MEM_DESCRIPTOR = serialize_mem_descriptor([])


def _make_xfer_req(
    *,
    mm_hash: str = "h",
    dst_indices: list[int] | None = None,
    consumer_agent_name: str = "consumer-agent",
    consumer_nixl_metadata: bytes = b"meta",
    consumer_mem_descriptor: bytes = _EMPTY_MEM_DESCRIPTOR,
    compatibility_hash: str | None = None,
    connector_version: int = EC_CONNECTOR_VERSION,
) -> XferReq:
    return XferReq(
        mm_hash=mm_hash,
        dst_block_indices=dst_indices if dst_indices is not None else [0],
        consumer_agent_name=consumer_agent_name,
        consumer_nixl_metadata=consumer_nixl_metadata,
        consumer_mem_descriptor=consumer_mem_descriptor,
        compatibility_hash=compatibility_hash if compatibility_hash is not None else "",
        connector_version=connector_version,
    )


def test_handle_xfer_req_version_mismatch_nacks(producer):
    req = _make_xfer_req(connector_version=EC_CONNECTOR_VERSION + 99)
    producer._handle_xfer_req(b"peer-id", req)
    assert producer._pending_nacks == [(b"peer-id", req.mm_hash)]
    assert not producer._in_flight


def test_handle_xfer_req_compat_hash_mismatch_nacks(producer):
    req = _make_xfer_req(compatibility_hash="not-a-real-hash")
    producer._handle_xfer_req(b"peer-id", req)
    assert producer._pending_nacks == [(b"peer-id", req.mm_hash)]
    assert not producer._in_flight


def test_handle_xfer_req_unknown_mm_hash_nacks(producer):
    req = _make_xfer_req(mm_hash="unknown", compatibility_hash=producer._compat_hash)
    producer._handle_xfer_req(b"peer-id", req)
    assert producer._pending_nacks == [(b"peer-id", "unknown")]


def test_handle_xfer_req_success_pins_and_records_in_flight(producer):
    indices = producer._region.alloc(1)
    producer._local_encodings["h"] = indices
    req = _make_xfer_req(
        mm_hash="h",
        dst_indices=[7],
        compatibility_hash=producer._compat_hash,
    )

    producer._handle_xfer_req(b"peer-id", req)

    # Pinned for the in-flight transfer.
    assert all(idx in producer._region._ref_count for idx in indices)
    # Recorded in _in_flight under a fresh xfer_id.
    assert len(producer._in_flight) == 1
    [(_xfer_id, (identity, mm_hash, _handle))] = producer._in_flight.items()
    assert identity == b"peer-id"
    assert mm_hash == "h"
    # NIXL was instructed to WRITE.
    producer._nixl.make_prepped_xfer.assert_called_once()
    producer._nixl.transfer.assert_called_once()


def test_handle_xfer_req_post_failure_unpins_and_nacks(producer):
    indices = producer._region.alloc(1)
    producer._local_encodings["h"] = indices

    producer._nixl.make_prepped_xfer.side_effect = RuntimeError("nixl error")

    req = _make_xfer_req(mm_hash="h", compatibility_hash=producer._compat_hash)
    producer._handle_xfer_req(b"peer-id", req)

    # Block was unpinned so eviction can reclaim it.
    assert not producer._region._ref_count
    # NACK queued for the consumer.
    assert producer._pending_nacks == [(b"peer-id", "h")]
    # Nothing in flight.
    assert not producer._in_flight


# ── _ensure_remote_peer ───────────────────────────────────────────────────────


def test_ensure_remote_peer_first_contact_adds_agent(producer):
    req = _make_xfer_req(
        consumer_agent_name="consumer", consumer_nixl_metadata=b"new-meta"
    )
    peer = producer._ensure_remote_peer(req)

    producer._nixl.add_remote_agent.assert_called_once_with(b"new-meta")
    assert peer.nixl_metadata_bytes == b"new-meta"
    assert producer._remote_peers["consumer"] is peer


def test_ensure_remote_peer_returns_cached_on_metadata_match(producer):
    req = _make_xfer_req(consumer_nixl_metadata=b"meta-A")
    first = producer._ensure_remote_peer(req)

    second = producer._ensure_remote_peer(req)

    assert second is first
    # add_remote_agent must have been called exactly once (first contact only).
    producer._nixl.add_remote_agent.assert_called_once()


def test_ensure_remote_peer_metadata_change_replaces_agent(producer):
    req1 = _make_xfer_req(consumer_nixl_metadata=b"meta-A")
    first = producer._ensure_remote_peer(req1)

    req2 = _make_xfer_req(consumer_nixl_metadata=b"meta-B")
    second = producer._ensure_remote_peer(req2)

    producer._nixl.remove_remote_agent.assert_called_once_with(first.nixl_agent_name)
    assert second is not first
    assert second.nixl_metadata_bytes == b"meta-B"


# ── _post_nixl_write ──────────────────────────────────────────────────────────


def test_post_nixl_write_block_count_mismatch_raises(producer):
    peer = ProducerPeer(
        nixl_agent_name="a", nixl_metadata_bytes=b"m", nixl_xfer_handle=1
    )
    with pytest.raises(ValueError, match="block count mismatch"):
        producer._post_nixl_write(peer, [0, 1], [9])


def test_post_nixl_write_invokes_make_and_transfer(producer):
    peer = ProducerPeer(
        nixl_agent_name="a", nixl_metadata_bytes=b"m", nixl_xfer_handle=1
    )
    producer._nixl.make_prepped_xfer.return_value = "handle-77"
    handle = producer._post_nixl_write(peer, [0], [9])

    assert handle == "handle-77"
    producer._nixl.make_prepped_xfer.assert_called_once()
    producer._nixl.transfer.assert_called_once_with("handle-77")


# ── _sweep_completions ────────────────────────────────────────────────────────


def _setup_in_flight_with_pin(
    producer: ECCPUScheduler, mm_hash: str = "h", n_blocks: int = 1
) -> tuple[list[int], object]:
    indices = producer._region.alloc(n_blocks)
    producer._local_encodings[mm_hash] = indices
    producer._region.pin(indices)
    _, handle = _inject_in_flight(producer, mm_hash, b"peer-id")
    return indices, handle


def test_sweep_done_unpins_releases_handle_and_acks_ok(producer):
    indices, handle = _setup_in_flight_with_pin(producer, "h", n_blocks=2)
    producer._nixl.check_xfer_state.return_value = "DONE"

    producer._sweep_completions()

    producer._inproc_send.send_multipart.assert_called_once_with(
        [b"peer-id", b"\x01", b"h"]
    )
    producer._nixl.release_xfer_handle.assert_called_once_with(handle)
    assert all(idx not in producer._region._ref_count for idx in indices)
    assert not producer._in_flight


def test_sweep_proc_does_not_release_or_ack(producer):
    indices, _handle = _setup_in_flight_with_pin(producer, "h")
    producer._nixl.check_xfer_state.return_value = "PROC"

    producer._sweep_completions()

    producer._inproc_send.send_multipart.assert_not_called()
    producer._nixl.release_xfer_handle.assert_not_called()
    # Still pinned and in flight.
    assert all(idx in producer._region._ref_count for idx in indices)
    assert producer._in_flight
    producer._region.unpin(indices)


def test_sweep_unexpected_state_acks_fail_and_releases(producer):
    indices, handle = _setup_in_flight_with_pin(producer, "h")
    producer._nixl.check_xfer_state.return_value = "WAT"

    producer._sweep_completions()

    producer._inproc_send.send_multipart.assert_called_once_with(
        [b"peer-id", b"\x00", b"h"]
    )
    producer._nixl.release_xfer_handle.assert_called_once_with(handle)
    assert not producer._in_flight
    # Blocks must be unpinned so eviction can reclaim them.
    assert all(idx not in producer._region._ref_count for idx in indices)


def test_sweep_check_xfer_state_exception_treated_as_failure(producer):
    indices, handle = _setup_in_flight_with_pin(producer, "h")
    producer._nixl.check_xfer_state.side_effect = RuntimeError("rpc lost")

    producer._sweep_completions()

    producer._inproc_send.send_multipart.assert_called_once_with(
        [b"peer-id", b"\x00", b"h"]
    )
    producer._nixl.release_xfer_handle.assert_called_once_with(handle)
    # Blocks must be unpinned so eviction can reclaim them.
    assert all(idx not in producer._region._ref_count for idx in indices)


def test_sweep_drains_pending_nacks(producer):
    producer._pending_nacks.append((b"peer-id", "x"))
    producer._sweep_completions()
    producer._inproc_send.send_multipart.assert_called_once_with(
        [b"peer-id", b"\x00", b"x"]
    )
    assert producer._pending_nacks == []


# ── request_finished (producer) ───────────────────────────────────────────────


def test_request_finished_returns_no_params_when_nothing_known(producer):
    req = _request_for(_feature("h"))
    done, params = producer.request_finished(req)
    assert done is False
    assert params is None


def test_request_finished_emits_for_local_encoding_hits(producer):
    producer._local_encodings["h"] = [0]
    req = _request_for(_feature("h", length=1))

    done, params = producer.request_finished(req)

    assert done is False
    assert params is not None and "h" in params
    info = params["h"]
    assert info["peer_host"] == producer._peer_host
    assert info["peer_port"] == producer._peer_port
    assert info["size_bytes"] == _HIDDEN_DIM * _ELEMENT_SIZE
    assert isinstance(info["nixl_agent_metadata_b64"], str)


def test_request_finished_emits_for_pending_save_hits(producer):
    """Race window: a request may finish before its save promotion. The
    producer must still announce so the consumer can fetch it next step."""
    producer._pending_save["h"] = [0]
    _, params = producer.request_finished(_request_for(_feature("h")))
    assert params is not None and "h" in params


def test_request_finished_uses_identifier_when_mm_hash_falsy(producer):
    feat = _feature("ident-only")
    feat.mm_hash = None
    producer._local_encodings["ident-only"] = [0]
    _, params = producer.request_finished(_request_for(feat))
    assert "ident-only" in params


# ── _drain_acks (consumer) ────────────────────────────────────────────────────


def _inject_peer_with_remote(
    sched: ECCPUScheduler, mm_hash: str, ok: bool
) -> list[int]:
    indices = sched._region.alloc(2)
    sched._remote_encodings[mm_hash] = indices
    dealer = _dealer_returning([_ack_frames(mm_hash, ok)])
    sched._peer_pool[("host", 1234)] = ConsumerPeer(
        zmq_dealer=dealer,
        nixl_agent_name="remote-agent",
        nixl_metadata_bytes=b"meta",
    )
    return indices


def test_drain_ok_ack_moves_hash_to_ready(consumer):
    _inject_peer_with_remote(consumer, "h", ok=True)
    consumer._drain_acks()

    assert "h" in consumer._ready
    # Blocks stay allocated in _remote_encodings until build_connector_meta
    # promotes them to _loaded; the ok path must NOT free them early.
    assert "h" in consumer._remote_encodings


def test_drain_fail_ack_frees_blocks(consumer):
    _inject_peer_with_remote(consumer, "h", ok=False)
    consumer._drain_acks()

    assert "h" not in consumer._remote_encodings
    assert "h" not in consumer._ready
    assert set(consumer._region._free) == set(range(_NUM_BLOCKS))


def test_drain_ignores_ack_for_unknown_hash(consumer):
    dealer = _dealer_returning([_ack_frames("unknown", ok=True)])
    consumer._peer_pool[("host", 1234)] = ConsumerPeer(
        zmq_dealer=dealer, nixl_agent_name="a", nixl_metadata_bytes=b"m"
    )
    consumer._drain_acks()

    assert "unknown" not in consumer._ready


def test_drain_drops_malformed_payload(consumer):
    indices = consumer._region.alloc(1)
    consumer._remote_encodings["h"] = indices
    dealer = _dealer_returning([[b"", b"\xff\xff\xff"]])  # not a valid msgpack
    consumer._peer_pool[("host", 1234)] = ConsumerPeer(
        zmq_dealer=dealer, nixl_agent_name="a", nixl_metadata_bytes=b"m"
    )

    consumer._drain_acks()

    assert "h" not in consumer._ready
    # Blocks must not have been freed on decode error.
    assert consumer._remote_encodings["h"] == indices


def test_drain_processes_multiple_acks_from_same_peer(consumer):
    """Two acks arrive in the same drain cycle from the same peer (e.g. two
    requests to the same producer completed in the same step). Both must be
    processed; the loop must not break after the first valid ack."""
    idx_a = consumer._region.alloc(1)
    idx_b = consumer._region.alloc(1)
    consumer._remote_encodings["a"] = idx_a
    consumer._remote_encodings["b"] = idx_b
    dealer = _dealer_returning([_ack_frames("a", ok=True), _ack_frames("b", ok=False)])
    consumer._peer_pool[("host", 1234)] = ConsumerPeer(
        zmq_dealer=dealer, nixl_agent_name="agent", nixl_metadata_bytes=b"m"
    )

    consumer._drain_acks()

    assert "a" in consumer._ready
    assert "a" in consumer._remote_encodings  # blocks held for build_meta
    assert "b" not in consumer._remote_encodings  # NACK: blocks freed
    assert "b" not in consumer._ready


# ── _get_or_add_peer (consumer) ───────────────────────────────────────────────


def test_get_or_add_peer_first_contact_creates_dealer(consumer):
    info = _info(metadata=b"new-meta")

    fake_dealer = MagicMock()
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.make_zmq_socket",
        return_value=fake_dealer,
    ):
        peer = consumer._get_or_add_peer(info)

    consumer._nixl.add_remote_agent.assert_called_once_with(b"new-meta")
    assert peer.zmq_dealer is fake_dealer
    assert peer.nixl_metadata_bytes == b"new-meta"
    # nixl_agent_name must capture the handle returned by add_remote_agent —
    # it is used by remove_remote_agent on metadata change and shutdown.
    assert peer.nixl_agent_name == consumer._nixl.add_remote_agent.return_value
    assert consumer._peer_pool[("host", 1234)] is peer


def test_get_or_add_peer_returns_cached_on_metadata_match(consumer):
    fake_dealer = MagicMock()
    consumer._peer_pool[("host", 1234)] = ConsumerPeer(
        zmq_dealer=fake_dealer,
        nixl_agent_name="agent",
        nixl_metadata_bytes=b"meta-A",
    )

    info = _info(metadata=b"meta-A")
    peer = consumer._get_or_add_peer(info)

    assert peer.zmq_dealer is fake_dealer
    consumer._nixl.add_remote_agent.assert_not_called()


def test_get_or_add_peer_metadata_change_replaces_dealer(consumer):
    stale_dealer = MagicMock()
    consumer._peer_pool[("host", 1234)] = ConsumerPeer(
        zmq_dealer=stale_dealer,
        nixl_agent_name="old-agent",
        nixl_metadata_bytes=b"old-meta",
    )

    new_dealer = MagicMock()
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.make_zmq_socket",
        return_value=new_dealer,
    ):
        peer = consumer._get_or_add_peer(_info(metadata=b"new-meta"))

    consumer._nixl.remove_remote_agent.assert_called_once_with("old-agent")
    stale_dealer.close.assert_called_once_with(linger=0)
    assert peer.zmq_dealer is new_dealer
    assert peer.nixl_metadata_bytes == b"new-meta"
    # New entry's agent name must reflect the fresh add_remote_agent call.
    assert peer.nixl_agent_name == consumer._nixl.add_remote_agent.return_value
    # Pool must be updated to the new entry so subsequent calls hit the cache.
    assert consumer._peer_pool[("host", 1234)] is peer


def test_get_or_add_peer_remove_agent_failure_still_creates_new_entry(consumer):
    """If remove_remote_agent raises , the error is swallowed and a fresh
    entry is created regardless.
    Propagating the exception would leave the consumer with a dead peer and
    no path to reconnect."""
    stale_dealer = MagicMock()
    consumer._peer_pool[("host", 1234)] = ConsumerPeer(
        zmq_dealer=stale_dealer,
        nixl_agent_name="old-agent",
        nixl_metadata_bytes=b"old-meta",
    )
    consumer._nixl.remove_remote_agent.side_effect = RuntimeError("already gone")

    new_dealer = MagicMock()
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.make_zmq_socket",
        return_value=new_dealer,
    ):
        peer = consumer._get_or_add_peer(_info(metadata=b"new-meta"))

    # New entry created despite the exception.
    assert peer.zmq_dealer is new_dealer
    assert peer.nixl_metadata_bytes == b"new-meta"
    assert consumer._peer_pool[("host", 1234)] is peer


# ── _alloc_and_start_xfer (consumer) ──────────────────────────────────────────


def test_alloc_and_start_xfer_sends_correct_xfer_req(consumer):
    fake_dealer = MagicMock()
    consumer._peer_pool[("host", 1234)] = ConsumerPeer(
        zmq_dealer=fake_dealer,
        nixl_agent_name="agent",
        nixl_metadata_bytes=b"meta",
    )

    consumer._alloc_and_start_xfer("h", _info(metadata=b"meta"), _BLOCK_SIZE)

    fake_dealer.send_multipart.assert_called_once()
    sent_frames = fake_dealer.send_multipart.call_args[0][0]
    # Frame 0 is the empty DEALER delimiter; frame 1 is the XferReq payload.
    assert sent_frames[0] == b""
    req = msgspec.msgpack.decode(sent_frames[1], type=XferReq)
    assert req.mm_hash == "h"
    assert req.compatibility_hash == consumer._compat_hash
    # dst_block_indices must match what _remote_encodings recorded — the
    # producer writes to these indices and the NACK/ACK path frees them.
    assert req.dst_block_indices == consumer._remote_encodings["h"]
    assert len(req.dst_block_indices) == 1  # _BLOCK_SIZE / block_size_bytes = 1


def test_alloc_and_start_xfer_send_failure_frees_blocks_and_reraises(consumer):
    fake_dealer = MagicMock()
    fake_dealer.send_multipart.side_effect = RuntimeError("socket dead")
    consumer._peer_pool[("host", 1234)] = ConsumerPeer(
        zmq_dealer=fake_dealer,
        nixl_agent_name="agent",
        nixl_metadata_bytes=b"meta",
    )

    free_before = list(consumer._region._free)
    with pytest.raises(RuntimeError, match="socket dead"):
        consumer._alloc_and_start_xfer("h", _info(metadata=b"meta"), _BLOCK_SIZE)

    # Blocks were returned to the pool; nothing recorded in _remote_encodings.
    assert sorted(consumer._region._free) == sorted(free_before)
    assert "h" not in consumer._remote_encodings


# ── _consumer_fifo_alloc ──────────────────────────────────────────────────────


def test_consumer_fifo_alloc_succeeds_when_pool_has_space(consumer):
    indices = consumer._consumer_fifo_alloc(3)
    assert len(indices) == 3


def test_consumer_fifo_alloc_evicts_loaded_in_insertion_order(consumer):
    region = consumer._region
    cold = region.alloc(_NUM_BLOCKS // 2)
    hot = region.alloc(_NUM_BLOCKS // 2)
    consumer._loaded["cold"] = cold
    consumer._loaded["hot"] = hot

    indices = consumer._consumer_fifo_alloc(1)

    assert len(indices) == 1
    assert "cold" not in consumer._loaded  # FIFO: oldest evicted first
    assert "hot" in consumer._loaded


def test_consumer_fifo_alloc_protects_pending_reload(consumer):
    """Bug regression: _consumer_fifo_alloc must NOT evict a _loaded entry
    whose mm_hash is in _pending_reload — the worker is about to copy
    those blocks this step. Without protection, build_connector_meta step
    (c) would silently drop the request from meta.loads."""
    region = consumer._region
    full = region.alloc(_NUM_BLOCKS)  # exhaust the pool
    consumer._loaded["A"] = full
    consumer._pending_reload.add("A")

    with pytest.raises(AllocationError):
        consumer._consumer_fifo_alloc(1)

    # Both the cache entry AND the underlying block indices survive.
    assert "A" in consumer._loaded
    assert consumer._loaded["A"] == full


def test_consumer_fifo_alloc_evicts_unprotected_when_others_protected(consumer):
    region = consumer._region
    cold = region.alloc(_NUM_BLOCKS // 2)
    hot = region.alloc(_NUM_BLOCKS // 2)
    consumer._loaded["cold"] = cold
    consumer._loaded["hot"] = hot
    consumer._pending_reload.add("hot")

    indices = consumer._consumer_fifo_alloc(1)

    assert len(indices) == 1
    assert "hot" in consumer._loaded  # protected, survived
    assert "cold" not in consumer._loaded  # evicted to satisfy alloc


# ── build_connector_meta (consumer) ───────────────────────────────────────────


def test_consumer_build_meta_promotes_ready_to_loads_and_loaded(consumer):
    indices = consumer._region.alloc(2)
    consumer._remote_encodings["h"] = indices
    consumer._ready.add("h")

    meta = consumer.build_connector_meta(Mock(spec=SchedulerOutput))

    assert meta.loads["h"] == indices
    assert consumer._loaded["h"] == indices  # cached for future re-serve
    assert consumer._ready == set()  # drained
    assert "h" not in consumer._remote_encodings  # popped


def test_consumer_build_meta_re_emits_pending_reload(consumer):
    """Cache hit: mm_hash already in _loaded, requested again this step →
    re-emit the same block indices in meta.loads without new transfer."""
    indices = consumer._region.alloc(1)
    consumer._loaded["h"] = indices
    consumer._pending_reload.add("h")
    free_before = len(consumer._region._free)

    meta = consumer.build_connector_meta(Mock(spec=SchedulerOutput))

    assert meta.loads["h"] is indices  # identity: same list object from _loaded
    assert consumer._pending_reload == set()  # cleared after build
    assert consumer._loaded["h"] is indices  # still cached for next time
    # No new transfer was started — the reload path serves from the mmap
    # cache without allocating blocks or sending a XferReq.
    assert "h" not in consumer._remote_encodings
    assert len(consumer._region._free) == free_before  # pool untouched


def test_consumer_build_meta_drops_stale_ready(consumer):
    """A mm_hash in _ready but with no _remote_encodings entry must be
    dropped quietly (e.g., a NACK raced with an OK ack)."""
    consumer._ready.add("h")
    # _remote_encodings does NOT have "h"

    meta = consumer.build_connector_meta(Mock(spec=SchedulerOutput))

    assert "h" not in meta.loads
    assert "h" not in consumer._ready
    assert "h" not in consumer._loaded


# ── ensure_cache_available (consumer) ─────────────────────────────────────────


def test_ensure_returns_true_when_no_announcements(consumer):
    """Empty params → no consumer involvement → admit immediately."""
    req = _request_for(_feature("h"))
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is True


def test_ensure_skips_already_computed_feature(consumer):
    """A feature whose mm_position is already covered by num_computed_tokens
    is not the consumer's concern; admit even if announced."""
    params = {"h": _info()}
    req = _request_for(_feature("h", length=1, offset=0), params=params)
    # offset+length=1 ≤ num_computed_tokens=1 → skipped
    assert consumer.ensure_cache_available(req, num_computed_tokens=1) is True


def test_ensure_skips_unannounced_feature(consumer):
    """Feature has no entry in announced params → fall through to local
    encode (admit, no transfer kicked off)."""
    params = {"other": _info()}
    req = _request_for(_feature("h"), params=params)
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert "h" not in consumer._remote_encodings


def test_ensure_falls_through_on_size_mismatch(consumer):
    """A producer that announces the wrong size_bytes for a feature should not
    be trusted — fall through to local encode."""
    params = {"h": _info(size_bytes=_BLOCK_SIZE * 99)}
    req = _request_for(_feature("h", length=1), params=params)
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert "h" not in consumer._remote_encodings


def test_ensure_defers_when_already_in_remote_encodings(consumer):
    """Transfer already in flight from a previous step → defer this step."""
    consumer._remote_encodings["h"] = [0]
    params = {"h": _info()}
    req = _request_for(_feature("h"), params=params)
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is False


def test_ensure_admits_when_already_loaded_and_marks_pending_reload(consumer):
    """Cache hit on _loaded → admit AND mark for re-serve in build_meta."""
    consumer._loaded["h"] = [0]
    params = {"h": _info()}
    req = _request_for(_feature("h"), params=params)

    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert "h" in consumer._pending_reload


def test_ensure_admits_when_just_arrived(consumer):
    """mm_hash in _ready (arrived but not yet handed to worker): _loaded
    miss but has_cache_item True → no _pending_reload, no defer."""
    consumer._remote_encodings["h"] = [0]
    consumer._ready.add("h")
    params = {"h": _info()}
    req = _request_for(_feature("h"), params=params)

    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert "h" not in consumer._pending_reload  # not yet in _loaded


def test_ensure_starts_xfer_for_uncached_announced_feature(consumer):
    fake_dealer = MagicMock()
    consumer._peer_pool[("host", 1234)] = ConsumerPeer(
        zmq_dealer=fake_dealer,
        nixl_agent_name="agent",
        nixl_metadata_bytes=b"meta",
    )
    params = {"h": _info(metadata=b"meta")}
    req = _request_for(_feature("h"), params=params)

    result = consumer.ensure_cache_available(req, num_computed_tokens=0)

    assert result is False  # pending — kicked off a transfer
    assert "h" in consumer._remote_encodings
    fake_dealer.send_multipart.assert_called_once()


def test_ensure_alloc_failure_falls_through_to_local_encode(consumer, caplog_vllm):
    """If allocation fails (region exhausted by protected entries), log and
    fall through; do not propagate AllocationError."""
    consumer._loaded["protected"] = consumer._region.alloc(_NUM_BLOCKS)
    consumer._pending_reload.add("protected")

    fake_dealer = MagicMock()
    consumer._peer_pool[("host", 1234)] = ConsumerPeer(
        zmq_dealer=fake_dealer,
        nixl_agent_name="agent",
        nixl_metadata_bytes=b"meta",
    )
    params = {"new": _info(metadata=b"meta")}
    req = _request_for(_feature("new"), params=params)

    # Must NOT raise — falls through to local encode for "new".
    with caplog_vllm.at_level(logging.ERROR):
        result = consumer.ensure_cache_available(req, num_computed_tokens=0)

    # Operator-visible diagnostic must have fired.
    assert any(
        "new" in r.message for r in caplog_vllm.records if r.levelno == logging.ERROR
    )
    # "new" was not transferred; "protected" still safe.
    assert "new" not in consumer._remote_encodings
    assert "protected" in consumer._loaded
    # The "new" feature fell through; pending stays False for that one.
    assert result is True


# ── has_cache_item ────────────────────────────────────────────────────────────


def test_has_cache_item_true_for_loaded(consumer):
    consumer._loaded["h"] = [0]
    assert consumer.has_cache_item("h") is True


def test_has_cache_item_true_for_ready(consumer):
    consumer._ready.add("h")
    assert consumer.has_cache_item("h") is True


def test_has_cache_item_false_for_in_flight(consumer):
    """A request mid-transfer is NOT yet a cache hit — the consumer still
    needs to wait on the ack before serving."""
    consumer._remote_encodings["h"] = [0]
    assert consumer.has_cache_item("h") is False


# ── shutdown ──────────────────────────────────────────────────────────────────


def test_shutdown_producer_stops_router_thread(producer):
    assert producer._router_t.is_alive()
    producer.shutdown()
    assert not producer._router_t.is_alive()


def test_shutdown_producer_releases_in_flight_handles(producer):
    handle = object()
    producer._in_flight["xid"] = (b"peer", "h", handle)

    producer.shutdown()

    producer._nixl.release_xfer_handle.assert_called_once_with(handle)
    assert producer._in_flight == {}


def test_shutdown_producer_releases_remaining_handles_if_one_raises(producer):
    """suppress(Exception) is per-item — first failure must not skip later handles."""
    h1, h2 = object(), object()
    producer._in_flight["xid1"] = (b"peer", "a", h1)
    producer._in_flight["xid2"] = (b"peer", "b", h2)
    producer._nixl.release_xfer_handle.side_effect = [RuntimeError("gone"), None]

    producer.shutdown()

    assert producer._nixl.release_xfer_handle.call_count == 2
    assert producer._in_flight == {}


def test_shutdown_producer_removes_remote_agents(producer):
    producer._remote_peers["c"] = ProducerPeer(
        nixl_agent_name="remote-1", nixl_metadata_bytes=b"m", nixl_xfer_handle=1
    )

    producer.shutdown()

    producer._nixl.remove_remote_agent.assert_called_once_with("remote-1")
    assert producer._remote_peers == {}


def test_shutdown_producer_removes_remaining_agents_if_one_raises(producer):
    """try/except is per-item — first failure must not skip later agents."""
    producer._remote_peers["a"] = ProducerPeer(
        nixl_agent_name="agent-1", nixl_metadata_bytes=b"m", nixl_xfer_handle=1
    )
    producer._remote_peers["b"] = ProducerPeer(
        nixl_agent_name="agent-2", nixl_metadata_bytes=b"m", nixl_xfer_handle=2
    )
    producer._nixl.remove_remote_agent.side_effect = [RuntimeError("gone"), None]

    producer.shutdown()

    assert producer._nixl.remove_remote_agent.call_count == 2
    assert producer._remote_peers == {}


def test_shutdown_consumer_closes_all_dealers(consumer):
    d1, d2 = MagicMock(), MagicMock()
    consumer._peer_pool[("h1", 1)] = ConsumerPeer(
        zmq_dealer=d1, nixl_agent_name="a", nixl_metadata_bytes=b""
    )
    consumer._peer_pool[("h2", 2)] = ConsumerPeer(
        zmq_dealer=d2, nixl_agent_name="b", nixl_metadata_bytes=b""
    )

    consumer.shutdown()

    d1.close.assert_called_once_with(linger=0)
    d2.close.assert_called_once_with(linger=0)
    assert consumer._peer_pool == {}


def test_shutdown_consumer_closes_remaining_dealers_if_one_raises(consumer):
    """try/except is per-item — first failure must not skip later dealers."""
    d1, d2 = MagicMock(), MagicMock()
    d1.close.side_effect = RuntimeError("already closed")
    consumer._peer_pool[("h1", 1)] = ConsumerPeer(
        zmq_dealer=d1, nixl_agent_name="a", nixl_metadata_bytes=b""
    )
    consumer._peer_pool[("h2", 2)] = ConsumerPeer(
        zmq_dealer=d2, nixl_agent_name="b", nixl_metadata_bytes=b""
    )

    consumer.shutdown()

    d2.close.assert_called_once_with(linger=0)
    assert consumer._peer_pool == {}


def test_shutdown_calls_nixl_deregister_and_region_cleanup(consumer):
    consumer.shutdown()
    consumer._nixl.deregister_memory.assert_called_once()
    assert consumer._region._base is None
