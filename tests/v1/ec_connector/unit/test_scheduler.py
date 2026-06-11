# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECCPUScheduler (consumer-side NIXL READ model).

Mocking policy
--------------
Only the parts that require external I/O are mocked:
  - NixlWrapper / nixl_agent_config (nixl package may not be installed)
  - zmq.Context and the sockets it creates (would bind to real ports)
  - make_zmq_socket / make_zmq_path (same reason)
  - setup_ec_region (lets us inject a real ECSharedRegion with known dims)

Everything else is real: ECSharedRegion, threading primitives, msgspec codecs,
all scheduler logic. The producer's router thread is replaced by a sentinel
that parks on _stop_event so it starts, can be joined, and never touches the
mocked sockets — the router-loop body itself is covered by an integration test
using real inproc:// sockets.
"""

import contextlib
import logging
import time
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

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.metadata import (
    EC_CONNECTOR_VERSION,
    XferAck,
    XferReq,
    XferStatus,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.zmq_transport import (
    ZmqProducerTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    ConsumerPeer,
    PendingRead,
    QuarantinedRead,
    serialize_mem_descriptor,
)
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    AllocationError,
    ECSharedRegion,
)
from vllm.v1.core.sched.output import SchedulerOutput

# ── helpers ───────────────────────────────────────────────────────────────────

_encoder = msgspec.msgpack.Encoder()

# A valid serialized mem descriptor: the real engine deserializes it in
# add_remote_source (only NIXL itself is mocked), so it must decode cleanly.
_MEM_DESC = serialize_mem_descriptor([(0, _BLOCK_SIZE, 0)])


def _dealer_returning(frames: list) -> MagicMock:
    """Mock DEALER that yields each frame list then raises zmq.Again."""
    dealer = MagicMock()
    dealer.recv_multipart.side_effect = frames + [zmq.Again]
    return dealer


def _ack_frames(
    mm_hash: str,
    status: XferStatus,
    *,
    src_block_indices: list[int] | None = None,
    agent_metadata: bytes = b"agent-meta",
    mem_descriptor: bytes = _MEM_DESC,
) -> list[bytes]:
    ack = XferAck(
        mm_hash=mm_hash,
        status=status,
        src_block_indices=src_block_indices or [],
        agent_metadata=agent_metadata if status == XferStatus.OK else b"",
        mem_descriptor=mem_descriptor if status == XferStatus.OK else b"",
    )
    return [b"", _encoder.encode(ack)]


def _silent_dealer() -> MagicMock:
    """Mock DEALER whose recv_multipart raises zmq.Again (no acks pending)."""
    dealer = MagicMock()
    dealer.recv_multipart.side_effect = zmq.Again
    return dealer


def _put_peer(
    sched: ECCPUScheduler, addr=("host", 1234), **peer_kwargs
) -> ConsumerPeer:
    """Insert a ConsumerPeer straight into the pool.

    The DEALER defaults to one that yields no acks, so a drain() that touches
    this peer does not spin on a bare MagicMock.
    """
    dealer = peer_kwargs.pop("zmq_dealer", None) or _silent_dealer()
    peer = ConsumerPeer(zmq_dealer=dealer, **peer_kwargs)
    sched._consumer._transport._peer_pool[addr] = peer
    return peer


def _pending(
    addr=("host", 1234), *, dst_indices, read_handle=None, ttl=999.0
) -> PendingRead:
    return PendingRead(
        addr=addr,
        dst_indices=dst_indices,
        deadline=time.monotonic() + ttl,
        read_handle=read_handle,
    )


def _make_xfer_req(
    *,
    mm_hash: str = "h",
    compatibility_hash: str = "",
    connector_version: int = EC_CONNECTOR_VERSION,
) -> XferReq:
    return XferReq(
        mm_hash=mm_hash,
        compatibility_hash=compatibility_hash,
        connector_version=connector_version,
    )


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _patch_router_run():
    """Replace ZmqProducerTransport._run with a sentinel that parks on
    _stop_event so the router thread starts, stays alive, and exits cleanly
    on stop() — without touching mocked ZMQ sockets."""

    def _sentinel(self):
        self._stop_event.wait()

    with patch.object(ZmqProducerTransport, "_run", _sentinel):
        yield


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
        mock_ctx = MagicMock()

        with (
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.NixlWrapper",
                new=MagicMock(),
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.nixl_agent_config",
                new=MagicMock(),
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.nixl_engine.NixlWrapper",
                return_value=mock_nixl,
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.nixl_engine.nixl_agent_config",
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.setup_ec_region",
                return_value=layout,
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.zmq_transport.make_zmq_socket",
                return_value=mock_router_sock,
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.zmq_transport.make_zmq_path",
                return_value="tcp://mock:5000",
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.zmq_transport.zmq.Context",
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


# ── role gating ───────────────────────────────────────────────────────────────


def test_ensure_cache_available_returns_true_on_producer_only(producer):
    req = _request_for(_feature("h"))
    assert producer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert producer.has_cache_item("h") is False


def test_producer_methods_no_op_on_consumer(consumer):
    done, params = consumer.request_finished(MagicMock())
    assert done is False and params is None
    consumer.update_state_after_alloc(MagicMock(), index=0)  # must not raise


# ── producer _fifo_alloc ──────────────────────────────────────────────────────


def test_producer_alloc_succeeds_when_pool_has_space(producer):
    assert len(producer._producer._fifo_alloc(3)) == 3


def test_producer_alloc_evicts_oldest_unpinned(producer):
    region = producer._memory_context.region
    producer._producer._local_encodings["old"] = region.alloc(4)
    producer._producer._local_encodings["new"] = region.alloc(4)

    result = producer._producer._fifo_alloc(3)

    assert len(result) == 3
    assert "old" not in producer._producer._local_encodings
    assert "new" in producer._producer._local_encodings


def test_producer_alloc_skips_pinned_blocks(producer):
    region = producer._memory_context.region
    pinned = region.alloc(4)
    not_pinned = region.alloc(4)
    region.pin(pinned)
    producer._producer._local_encodings["pinned"] = pinned
    producer._producer._local_encodings["not_pinned"] = not_pinned

    result = producer._producer._fifo_alloc(3)

    assert len(result) == 3
    assert "pinned" in producer._producer._local_encodings
    assert "not_pinned" not in producer._producer._local_encodings
    region.unpin(pinned)


def test_producer_alloc_raises_when_all_encodings_pinned(producer):
    region = producer._memory_context.region
    all_idx = region.alloc(_NUM_BLOCKS)
    region.pin(all_idx)
    producer._producer._local_encodings["only"] = all_idx
    with pytest.raises(AllocationError):
        producer._producer._fifo_alloc(1)
    region.unpin(all_idx)


# ── ensure_cache_available (producer) ────────────────────────────────────────


def test_producer_ensure_probes_and_returns_blocks(producer):
    """Probe returns True and leaves the free pool unchanged (alloc + free)."""
    free_before = len(producer._memory_context.region._free)
    req = _request_for(_feature("h", length=1))
    assert producer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert len(producer._memory_context.region._free) == free_before


def test_producer_ensure_evicts_unpinned_to_make_room(producer):
    """When the region is full the probe evicts unpinned encodings to make room."""
    region = producer._memory_context.region
    producer._producer._local_encodings["old"] = region.alloc(_NUM_BLOCKS)

    req = _request_for(_feature("h", length=1))
    assert producer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert "old" not in producer._producer._local_encodings


def test_producer_ensure_returns_false_when_all_blocks_pinned(producer):
    """When all blocks are pinned by in-flight reads, returns False to defer."""
    region = producer._memory_context.region
    all_idx = region.alloc(_NUM_BLOCKS)
    region.pin(all_idx)
    producer._producer._local_encodings["pinned"] = all_idx

    req = _request_for(_feature("h", length=1))
    assert producer.ensure_cache_available(req, num_computed_tokens=0) is False
    region.unpin(all_idx)


def test_producer_ensure_skips_feature_in_local_encodings(producer):
    """mm_hash already stored — no probe, region free count unchanged."""
    producer._producer._local_encodings["h"] = [0]
    free_before = len(producer._memory_context.region._free)
    assert producer.ensure_cache_available(_request_for(_feature("h")), 0) is True
    assert len(producer._memory_context.region._free) == free_before


def test_producer_ensure_skips_feature_in_pending_save(producer):
    """mm_hash already allocated last step — no probe."""
    producer._producer._pending_save["h"] = [0]
    free_before = len(producer._memory_context.region._free)
    assert producer.ensure_cache_available(_request_for(_feature("h")), 0) is True
    assert len(producer._memory_context.region._free) == free_before


def test_producer_ensure_skips_feature_in_pending_new_encodings(producer):
    """mm_hash already registered by a running request — no double-probe."""
    producer._producer._pending_new_encodings["h"] = _BLOCK_SIZE
    free_before = len(producer._memory_context.region._free)
    assert producer.ensure_cache_available(_request_for(_feature("h")), 0) is True
    assert len(producer._memory_context.region._free) == free_before


def test_producer_ensure_skips_already_computed_feature(producer):
    """Feature whose range is fully computed is not probed."""
    req = _request_for(_feature("h", length=1, offset=0))
    assert producer.ensure_cache_available(req, num_computed_tokens=1) is True


def test_scheduler_ensure_false_when_producer_blocked(make_scheduler):
    """ECCPUScheduler.ensure_cache_available returns
    False when producer can't reserve."""
    sched = make_scheduler(is_producer=True, is_consumer=False)
    region = sched._memory_context.region
    all_idx = region.alloc(_NUM_BLOCKS)
    region.pin(all_idx)
    sched._producer._local_encodings["pinned"] = all_idx

    req = _request_for(_feature("h"))
    assert sched.ensure_cache_available(req, num_computed_tokens=0) is False
    region.unpin(all_idx)


# ── update_state_after_alloc ──────────────────────────────────────────────────


def test_update_state_queues_size_bytes(producer):
    req = _request_for(_feature("h1", length=2))
    producer.update_state_after_alloc(req, index=0)
    expected_size = 2 * _HIDDEN_DIM * _ELEMENT_SIZE
    assert producer._producer._pending_new_encodings == {"h1": expected_size}


def test_update_state_dedups_against_pending_save(producer):
    producer._producer._pending_save["h1"] = [0]
    producer.update_state_after_alloc(_request_for(_feature("h1")), index=0)
    assert "h1" not in producer._producer._pending_new_encodings


def test_update_state_dedups_against_local_encodings(producer):
    producer._producer._local_encodings["h1"] = [0]
    producer.update_state_after_alloc(_request_for(_feature("h1")), index=0)
    assert "h1" not in producer._producer._pending_new_encodings


# ── build_connector_meta (producer) ───────────────────────────────────────────


def test_build_meta_promotes_pending_save(producer):
    producer._producer._pending_save["h1"] = [0, 1]
    producer.build_connector_meta(Mock(spec=SchedulerOutput))
    assert producer._producer._local_encodings["h1"] == [0, 1]
    assert producer._producer._pending_save == {}


def test_build_meta_allocates_for_pending_new_encodings(producer):
    producer._producer._pending_new_encodings["h2"] = _BLOCK_SIZE  # exactly 1 block
    meta = producer.build_connector_meta(Mock(spec=SchedulerOutput))
    assert "h2" in meta.saves
    assert len(meta.saves["h2"]) == 1
    assert "h2" in producer._producer._pending_save
    assert producer._producer._pending_new_encodings == {}


def test_build_meta_rounds_up_partial_block(producer):
    producer._producer._pending_new_encodings["h"] = _BLOCK_SIZE + 1
    meta = producer.build_connector_meta(Mock(spec=SchedulerOutput))
    assert len(meta.saves["h"]) == 2


# ── handle_xfer_req (producer grants) ─────────────────────────────────────────


def test_handle_xfer_req_version_mismatch_nacks(producer):
    req = _make_xfer_req(connector_version=EC_CONNECTOR_VERSION + 99)
    ack = producer._producer.handle_xfer_req(b"peer-id", req)
    assert ack.status == XferStatus.NACK_VERSION
    assert not producer._producer._pinned_encodings


def test_handle_xfer_req_compat_hash_mismatch_nacks(producer):
    req = _make_xfer_req(compatibility_hash="not-a-real-hash")
    ack = producer._producer.handle_xfer_req(b"peer-id", req)
    assert ack.status == XferStatus.NACK_INCOMPAT
    assert not producer._producer._pinned_encodings


def test_handle_xfer_req_unknown_mm_hash_nacks(producer):
    req = _make_xfer_req(mm_hash="unknown", compatibility_hash=producer._compat_hash)
    ack = producer._producer.handle_xfer_req(b"peer-id", req)
    assert ack.status == XferStatus.NACK_MISSING
    assert ack.mm_hash == "unknown"
    assert not producer._producer._pinned_encodings


def test_handle_xfer_req_success_pins_and_returns_grant(producer):
    indices = producer._memory_context.region.alloc(2)
    producer._producer._local_encodings["h"] = indices
    req = _make_xfer_req(mm_hash="h", compatibility_hash=producer._compat_hash)

    ack = producer._producer.handle_xfer_req(b"peer-id", req)

    assert ack.status == XferStatus.OK
    assert ack.src_block_indices == indices
    # Fresh producer metadata + mem descriptor travel on the grant.
    assert ack.agent_metadata == producer._engine._agent_metadata
    assert ack.mem_descriptor == producer._engine._mem_descriptor_bytes
    # Source blocks pinned, one deadline recorded.
    assert all(idx in producer._memory_context.region._ref_count for idx in indices)
    assert len(producer._producer._pinned_encodings["h"].deadlines) == 1


def test_handle_xfer_req_concurrent_reads_refcount(producer):
    """Two grants for the same mm_hash nest the pin (two deadlines)."""
    indices = producer._memory_context.region.alloc(1)
    producer._producer._local_encodings["h"] = indices
    req = _make_xfer_req(mm_hash="h", compatibility_hash=producer._compat_hash)

    producer._producer.handle_xfer_req(b"c1", req)
    producer._producer.handle_xfer_req(b"c2", req)

    assert len(producer._producer._pinned_encodings["h"].deadlines) == 2
    assert producer._memory_context.region._ref_count[indices[0]] == 2


# ── poll: notif unpin + lease sweep (producer) ────────────────────────────────


def _grant(producer, mm_hash="h", n=1):
    indices = producer._memory_context.region.alloc(n)
    producer._producer._local_encodings[mm_hash] = indices
    req = _make_xfer_req(mm_hash=mm_hash, compatibility_hash=producer._compat_hash)
    producer._producer.handle_xfer_req(b"peer-id", req)
    return indices


def test_poll_notif_unpins_source(producer):
    indices = _grant(producer, "h", n=2)
    producer._engine._nixl.get_new_notifs.return_value = {"peer": [b"h"]}

    producer._producer.poll()

    assert all(idx not in producer._memory_context.region._ref_count for idx in indices)
    assert "h" not in producer._producer._pinned_encodings


def test_poll_notif_for_unknown_hash_is_noop(producer):
    producer._engine._nixl.get_new_notifs.return_value = {"peer": [b"ghost"]}
    producer._producer.poll()  # must not raise
    assert not producer._producer._pinned_encodings


def test_poll_lease_expiry_force_unpins(producer):
    indices = _grant(producer, "h", n=1)
    # Force the lease into the past.
    producer._producer._pinned_encodings["h"].deadlines = [time.monotonic() - 1.0]

    producer._producer.poll()

    assert all(idx not in producer._memory_context.region._ref_count for idx in indices)
    assert "h" not in producer._producer._pinned_encodings


def test_poll_notif_then_late_lease_does_not_double_unpin(producer):
    """Once a notif fully unpins an encoding, a later lease sweep is a no-op."""
    indices = _grant(producer, "h", n=1)
    producer._engine._nixl.get_new_notifs.return_value = {"peer": [b"h"]}
    producer._producer.poll()  # notif unpins
    # A second poll (lease sweep over an empty table) must not over-unpin.
    producer._engine._nixl.get_new_notifs.return_value = {}
    producer._producer.poll()
    assert all(idx not in producer._memory_context.region._ref_count for idx in indices)


def test_has_pending_pins(producer):
    assert producer._producer.has_pending_pins() is False
    _grant(producer, "h")
    assert producer._producer.has_pending_pins() is True


# ── engine.post_read ──────────────────────────────────────────────────────────


def test_post_read_block_count_mismatch_raises(consumer):
    with pytest.raises(ValueError, match="block count mismatch"):
        consumer._engine.post_read([0, 1], 7, [9], notif_msg=b"h")


def test_post_read_invokes_make_read_and_transfer(consumer):
    consumer._engine._nixl.make_prepped_xfer.return_value = "handle-77"
    handle = consumer._engine.post_read([0], 7, [9], notif_msg=b"h")
    assert handle == "handle-77"
    op = consumer._engine._nixl.make_prepped_xfer.call_args[0][0]
    assert op == "READ"
    consumer._engine._nixl.transfer.assert_called_once_with("handle-77")


# ── request_finished (producer) ───────────────────────────────────────────────


def test_request_finished_returns_no_params_when_nothing_known(producer):
    done, params = producer.request_finished(_request_for(_feature("h")))
    assert done is False
    assert params is None


def test_request_finished_emits_address_and_size_only(producer):
    producer._producer._local_encodings["h"] = [0]
    done, params = producer.request_finished(_request_for(_feature("h", length=1)))

    assert done is False
    info = params["h"]
    assert info["peer_host"] == producer._peer_host
    assert info["peer_port"] == producer._peer_port
    assert info["size_bytes"] == _HIDDEN_DIM * _ELEMENT_SIZE
    # No NIXL metadata is ever announced through the routing layer.
    assert "nixl_agent_metadata_b64" not in info
    assert set(info) == {"peer_host", "peer_port", "size_bytes"}


def test_request_finished_emits_for_pending_save_hits(producer):
    producer._producer._pending_save["h"] = [0]
    _, params = producer.request_finished(_request_for(_feature("h")))
    assert params is not None and "h" in params


def test_request_finished_uses_identifier_when_mm_hash_falsy(producer):
    feat = _feature("ident-only")
    feat.mm_hash = None
    producer._producer._local_encodings["ident-only"] = [0]
    _, params = producer.request_finished(_request_for(feat))
    assert "ident-only" in params


# ── _start_read (consumer) ────────────────────────────────────────────────────


def test_start_read_sends_xfer_req(consumer):
    dealer = MagicMock()
    _put_peer(consumer, zmq_dealer=dealer)

    consumer._consumer._start_read("h", _info(), _BLOCK_SIZE)

    dealer.send_multipart.assert_called_once()
    sent = dealer.send_multipart.call_args[0][0]
    assert sent[0] == b""
    req = msgspec.msgpack.decode(sent[1], type=XferReq)
    assert req.mm_hash == "h"
    assert req.compatibility_hash == consumer._compat_hash
    pr = consumer._consumer._remote_encodings["h"]
    assert pr.read_handle is None  # awaiting ack
    assert len(pr.dst_indices) == 1


def test_start_read_uses_ensure_dealer_on_first_contact(consumer):
    new_dealer = MagicMock()
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.zmq_transport.make_zmq_socket",
        return_value=new_dealer,
    ):
        consumer._consumer._start_read("h", _info(), _BLOCK_SIZE)

    # DEALER created but NIXL registration deferred until the ack.
    peer = consumer._consumer._transport._peer_pool[("host", 1234)]
    assert peer.remote_read_handle is None
    consumer._engine._nixl.add_remote_agent.assert_not_called()
    new_dealer.send_multipart.assert_called_once()


def test_start_read_send_failure_frees_blocks_and_reraises(consumer):
    dealer = MagicMock()
    dealer.send_multipart.side_effect = RuntimeError("socket dead")
    _put_peer(consumer, zmq_dealer=dealer)

    free_before = sorted(consumer._memory_context.region._free)
    with pytest.raises(RuntimeError, match="socket dead"):
        consumer._consumer._start_read("h", _info(), _BLOCK_SIZE)

    assert sorted(consumer._memory_context.region._free) == free_before
    assert "h" not in consumer._consumer._remote_encodings


# ── _handle_ack (consumer) ────────────────────────────────────────────────────


def test_handle_ack_ok_registers_fresh_and_starts_read(consumer):
    addr = ("host", 1234)
    _put_peer(consumer, addr)
    indices = consumer._memory_context.region.alloc(1)
    consumer._consumer._remote_encodings["h"] = _pending(addr, dst_indices=indices)

    ack = XferAck(
        mm_hash="h",
        status=XferStatus.OK,
        src_block_indices=[3],
        agent_metadata=b"fresh-meta",
        mem_descriptor=_MEM_DESC,
    )
    consumer._consumer._handle_ack(addr, ack)

    # The lone add_remote_agent is fed the ack's fresh metadata.
    consumer._engine._nixl.add_remote_agent.assert_called_once_with(b"fresh-meta")
    # READ issued; handle recorded.
    op = consumer._engine._nixl.make_prepped_xfer.call_args[0][0]
    assert op == "READ"
    assert consumer._consumer._remote_encodings["h"].read_handle is not None


@pytest.mark.parametrize(
    "status",
    [XferStatus.NACK_MISSING, XferStatus.NACK_INCOMPAT, XferStatus.NACK_INTERNAL],
)
def test_handle_ack_nack_frees_and_tombstones(consumer, status):
    addr = ("host", 1234)
    _put_peer(consumer, addr)
    indices = consumer._memory_context.region.alloc(2)
    consumer._consumer._remote_encodings["h"] = _pending(addr, dst_indices=indices)
    free_before = len(consumer._memory_context.region._free)

    consumer._consumer._handle_ack(addr, XferAck(mm_hash="h", status=status))

    assert consumer._consumer._remote_encodings["h"] is None  # tombstone
    assert len(consumer._memory_context.region._free) == free_before + 2
    # A NACK never registers a remote agent.
    consumer._engine._nixl.add_remote_agent.assert_not_called()


def test_handle_ack_ignores_unknown_hash(consumer):
    consumer._consumer._handle_ack(
        ("host", 1234), XferAck(mm_hash="ghost", status=XferStatus.OK)
    )
    consumer._engine._nixl.add_remote_agent.assert_not_called()


def test_handle_ack_duplicate_after_read_started_is_ignored(consumer):
    addr = ("host", 1234)
    _put_peer(consumer, addr)
    indices = consumer._memory_context.region.alloc(1)
    consumer._consumer._remote_encodings["h"] = _pending(
        addr, dst_indices=indices, read_handle="existing"
    )

    consumer._consumer._handle_ack(
        addr, XferAck(mm_hash="h", status=XferStatus.OK, agent_metadata=b"m")
    )

    consumer._engine._nixl.make_prepped_xfer.assert_not_called()


# ── drain: poll_responses + read polling (consumer) ───────────────────────────


def _peer_with_ack(consumer, mm_hash, status, *, addr=("host", 1234), src=None):
    indices = consumer._memory_context.region.alloc(2)
    consumer._consumer._remote_encodings[mm_hash] = _pending(addr, dst_indices=indices)
    dealer = _dealer_returning([_ack_frames(mm_hash, status, src_block_indices=src)])
    _put_peer(consumer, addr, zmq_dealer=dealer)
    return indices


def test_drain_ok_ack_starts_read(consumer):
    _peer_with_ack(consumer, "h", XferStatus.OK, src=[1, 2])
    consumer._consumer.drain()
    pr = consumer._consumer._remote_encodings["h"]
    assert pr is not None and pr.read_handle is not None  # reading


def test_drain_nack_frees_and_tombstones(consumer):
    _peer_with_ack(consumer, "h", XferStatus.NACK_MISSING)
    consumer._consumer.drain()
    assert consumer._consumer._remote_encodings.get("h") is None
    assert set(consumer._memory_context.region._free) == set(range(_NUM_BLOCKS))


def test_drain_completed_read_promotes_to_ready(consumer):
    addr = ("host", 1234)
    _put_peer(consumer, addr)
    indices = consumer._memory_context.region.alloc(1)
    consumer._consumer._remote_encodings["h"] = _pending(
        addr, dst_indices=indices, read_handle="rh"
    )
    consumer._engine._nixl.check_xfer_state.return_value = "DONE"

    consumer._consumer.drain()

    assert "h" in consumer._consumer._ready
    consumer._engine._nixl.release_xfer_handle.assert_called_once_with("rh")


def test_drain_read_failure_frees_and_tombstones(consumer):
    addr = ("host", 1234)
    _put_peer(consumer, addr)
    indices = consumer._memory_context.region.alloc(1)
    consumer._consumer._remote_encodings["h"] = _pending(
        addr, dst_indices=indices, read_handle="rh"
    )
    consumer._engine._nixl.check_xfer_state.return_value = "ERR"
    free_before = len(consumer._memory_context.region._free)

    consumer._consumer.drain()

    assert consumer._consumer._remote_encodings["h"] is None
    assert len(consumer._memory_context.region._free) == free_before + 1


def test_drain_ack_timeout_frees_and_tombstones(consumer):
    addr = ("host", 1234)
    _put_peer(consumer, addr)
    indices = consumer._memory_context.region.alloc(1)
    # read_handle None + deadline in the past → ack timeout.
    pr = _pending(addr, dst_indices=indices)
    pr.deadline = time.monotonic() - 1.0
    consumer._consumer._remote_encodings["h"] = pr
    free_before = len(consumer._memory_context.region._free)

    consumer._consumer.drain()

    assert consumer._consumer._remote_encodings["h"] is None
    assert len(consumer._memory_context.region._free) == free_before + 1


def test_drain_drops_malformed_payload(consumer):
    addr = ("host", 1234)
    indices = consumer._memory_context.region.alloc(1)
    consumer._consumer._remote_encodings["h"] = _pending(addr, dst_indices=indices)
    _put_peer(consumer, addr, zmq_dealer=_dealer_returning([[b"", b"\xff\xff\xff"]]))

    consumer._consumer.drain()

    # Decode error must not advance or free the read.
    pr = consumer._consumer._remote_encodings["h"]
    assert pr is not None and pr.read_handle is None


# ── in-flight read timeout → quarantine (consumer) ────────────────────────────


def test_read_timeout_quarantines_and_tombstones(consumer):
    addr = ("host", 1234)
    _put_peer(consumer, addr)
    indices = consumer._memory_context.region.alloc(1)
    pr = _pending(addr, dst_indices=indices, read_handle="rh")
    pr.deadline = time.monotonic() - 1.0  # read budget exhausted
    consumer._consumer._remote_encodings["h"] = pr
    consumer._engine._nixl.check_xfer_state.return_value = "PROC"  # still running
    free_before = len(consumer._memory_context.region._free)

    consumer._consumer.drain()

    # Tombstoned for local encode, blocks NOT freed (still quarantined).
    assert consumer._consumer._remote_encodings["h"] is None
    assert len(consumer._memory_context.region._free) == free_before
    assert len(consumer._consumer._quarantine) == 1


def test_drain_quarantine_frees_on_terminal(consumer):
    indices = consumer._memory_context.region.alloc(2)
    consumer._consumer._quarantine.append(QuarantinedRead(indices, "rh"))
    consumer._engine._nixl.check_xfer_state.return_value = "DONE"
    free_before = len(consumer._memory_context.region._free)

    consumer._consumer.drain()

    assert consumer._consumer._quarantine == []
    consumer._engine._nixl.release_xfer_handle.assert_called_once_with("rh")
    assert len(consumer._memory_context.region._free) == free_before + 2


def test_drain_quarantine_keeps_proc(consumer):
    indices = consumer._memory_context.region.alloc(1)
    consumer._consumer._quarantine.append(QuarantinedRead(indices, "rh"))
    consumer._engine._nixl.check_xfer_state.return_value = "PROC"
    free_before = len(consumer._memory_context.region._free)

    consumer._consumer.drain()

    assert len(consumer._consumer._quarantine) == 1
    assert len(consumer._memory_context.region._free) == free_before


# ── register_source (consumer transport) ──────────────────────────────────────


def test_register_source_first_contact_adds_agent(consumer):
    addr = ("host", 1234)
    _put_peer(consumer, addr)
    peer = consumer._consumer._transport.register_source(addr, b"meta", _MEM_DESC)
    consumer._engine._nixl.add_remote_agent.assert_called_once_with(b"meta")
    assert peer.nixl_metadata_bytes == b"meta"
    assert peer.remote_read_handle is not None


def test_register_source_reuses_on_metadata_match(consumer):
    addr = ("host", 1234)
    _put_peer(consumer, addr)
    consumer._consumer._transport.register_source(addr, b"meta", _MEM_DESC)
    consumer._consumer._transport.register_source(addr, b"meta", _MEM_DESC)
    consumer._engine._nixl.add_remote_agent.assert_called_once()


def test_register_source_metadata_change_reregisters(consumer):
    addr = ("host", 1234)
    _put_peer(consumer, addr)
    consumer._consumer._transport.register_source(addr, b"old", _MEM_DESC)
    consumer._consumer._transport.register_source(addr, b"new", _MEM_DESC)
    consumer._engine._nixl.remove_remote_agent.assert_called_once()
    peer = consumer._consumer._transport._peer_pool[addr]
    assert peer.nixl_metadata_bytes == b"new"


# ── consumer _fifo_alloc ──────────────────────────────────────────────────────


def test_consumer_fifo_alloc_evicts_loaded_in_insertion_order(consumer):
    region = consumer._memory_context.region
    consumer._consumer._loaded["cold"] = region.alloc(_NUM_BLOCKS // 2)
    consumer._consumer._loaded["hot"] = region.alloc(_NUM_BLOCKS // 2)

    indices = consumer._consumer._fifo_alloc(1)

    assert len(indices) == 1
    assert "cold" not in consumer._consumer._loaded
    assert "hot" in consumer._consumer._loaded


def test_consumer_fifo_alloc_protects_pending_reload(consumer):
    region = consumer._memory_context.region
    consumer._consumer._loaded["A"] = region.alloc(_NUM_BLOCKS)
    consumer._consumer._pending_reload.add("A")
    with pytest.raises(AllocationError):
        consumer._consumer._fifo_alloc(1)
    assert "A" in consumer._consumer._loaded


# ── build_connector_meta (consumer) ───────────────────────────────────────────


def test_consumer_build_meta_promotes_ready_to_loads_and_loaded(consumer):
    indices = consumer._memory_context.region.alloc(2)
    consumer._consumer._remote_encodings["h"] = _pending(
        dst_indices=indices, read_handle="rh"
    )
    consumer._consumer._ready.add("h")

    meta = consumer.build_connector_meta(Mock(spec=SchedulerOutput))

    assert meta.loads["h"] == indices
    assert consumer._consumer._loaded["h"] == indices
    assert consumer._consumer._ready == set()
    assert "h" not in consumer._consumer._remote_encodings


def test_consumer_build_meta_re_emits_pending_reload(consumer):
    indices = consumer._memory_context.region.alloc(1)
    consumer._consumer._loaded["h"] = indices
    consumer._consumer._pending_reload.add("h")
    free_before = len(consumer._memory_context.region._free)

    meta = consumer.build_connector_meta(Mock(spec=SchedulerOutput))

    assert meta.loads["h"] is indices
    assert consumer._consumer._pending_reload == set()
    assert len(consumer._memory_context.region._free) == free_before


def test_consumer_build_meta_drops_stale_ready(consumer):
    consumer._consumer._ready.add("h")  # no _remote_encodings entry
    meta = consumer.build_connector_meta(Mock(spec=SchedulerOutput))
    assert "h" not in meta.loads
    assert "h" not in consumer._consumer._ready


# ── ensure_cache_available (consumer) ─────────────────────────────────────────


def test_ensure_returns_true_when_no_announcements(consumer):
    req = _request_for(_feature("h"))
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is True


def test_ensure_skips_already_computed_feature(consumer):
    params = {"h": _info()}
    req = _request_for(_feature("h", length=1, offset=0), params=params)
    assert consumer.ensure_cache_available(req, num_computed_tokens=1) is True


def test_ensure_skips_unannounced_feature(consumer):
    params = {"other": _info()}
    req = _request_for(_feature("h"), params=params)
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert "h" not in consumer._consumer._remote_encodings


def test_ensure_falls_through_on_size_mismatch(consumer):
    params = {"h": _info(size_bytes=_BLOCK_SIZE * 99)}
    req = _request_for(_feature("h", length=1), params=params)
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert "h" not in consumer._consumer._remote_encodings


def test_ensure_defers_when_already_in_flight(consumer):
    consumer._consumer._remote_encodings["h"] = _pending(dst_indices=[0])
    req = _request_for(_feature("h"), params={"h": _info()})
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is False


def test_ensure_consumes_nack_tombstone(consumer):
    consumer._consumer._remote_encodings["h"] = None  # tombstone
    req = _request_for(_feature("h"), params={"h": _info()})
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert "h" not in consumer._consumer._remote_encodings  # consumed


def test_ensure_admits_when_already_loaded_and_marks_pending_reload(consumer):
    consumer._consumer._loaded["h"] = [0]
    req = _request_for(_feature("h"), params={"h": _info()})
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert "h" in consumer._consumer._pending_reload


def test_ensure_starts_read_for_uncached_announced_feature(consumer):
    dealer = MagicMock()
    _put_peer(consumer, zmq_dealer=dealer)
    req = _request_for(_feature("h"), params={"h": _info()})

    result = consumer.ensure_cache_available(req, num_computed_tokens=0)

    assert result is False  # deferred — kicked off a read
    assert "h" in consumer._consumer._remote_encodings
    dealer.send_multipart.assert_called_once()


def test_ensure_alloc_failure_falls_through_to_local_encode(consumer, caplog_vllm):
    consumer._consumer._loaded["protected"] = consumer._memory_context.region.alloc(
        _NUM_BLOCKS
    )
    consumer._consumer._pending_reload.add("protected")
    _put_peer(consumer)
    req = _request_for(_feature("new"), params={"new": _info()})

    with caplog_vllm.at_level(logging.ERROR):
        result = consumer.ensure_cache_available(req, num_computed_tokens=0)

    assert any(
        "new" in r.message for r in caplog_vllm.records if r.levelno == logging.ERROR
    )
    assert "new" not in consumer._consumer._remote_encodings
    assert result is True


# ── has_cache_item ────────────────────────────────────────────────────────────


def test_has_cache_item_true_for_loaded(consumer):
    consumer._consumer._loaded["h"] = [0]
    assert consumer.has_cache_item("h") is True


def test_has_cache_item_true_for_ready(consumer):
    consumer._consumer._ready.add("h")
    assert consumer.has_cache_item("h") is True


def test_has_cache_item_false_for_in_flight(consumer):
    consumer._consumer._remote_encodings["h"] = _pending(dst_indices=[0])
    assert consumer.has_cache_item("h") is False


# ── on_peer_down (consumer) ───────────────────────────────────────────────────


def test_on_peer_down_awaiting_ack_frees_and_retries(consumer):
    addr = ("host", 1234)
    indices = consumer._memory_context.region.alloc(2)
    consumer._consumer._remote_encodings["h"] = _pending(addr, dst_indices=indices)
    _put_peer(consumer, addr, nixl_agent_name="agent")
    free_before = len(consumer._memory_context.region._free)

    consumer._consumer.on_peer_down(addr)

    # Forgotten (not tombstoned) so the next ensure retries; blocks freed.
    assert "h" not in consumer._consumer._remote_encodings
    assert len(consumer._memory_context.region._free) == free_before + 2
    assert addr not in consumer._consumer._transport._peer_pool


def test_on_peer_down_in_flight_quarantines_and_retries(consumer):
    addr = ("host", 1234)
    indices = consumer._memory_context.region.alloc(1)
    consumer._consumer._remote_encodings["h"] = _pending(
        addr, dst_indices=indices, read_handle="rh"
    )
    _put_peer(consumer, addr, nixl_agent_name="agent")
    free_before = len(consumer._memory_context.region._free)

    consumer._consumer.on_peer_down(addr)

    assert "h" not in consumer._consumer._remote_encodings  # forgotten → retry
    assert (
        len(consumer._memory_context.region._free) == free_before
    )  # NOT freed (quarantined)
    assert len(consumer._consumer._quarantine) == 1


def test_on_peer_down_preserves_ready_entries(consumer):
    addr = ("host", 1234)
    indices = consumer._memory_context.region.alloc(2)
    consumer._consumer._remote_encodings["h"] = _pending(addr, dst_indices=indices)
    consumer._consumer._ready.add("h")
    _put_peer(consumer, addr, nixl_agent_name="agent")
    free_before = len(consumer._memory_context.region._free)

    consumer._consumer.on_peer_down(addr)

    assert len(consumer._memory_context.region._free) == free_before  # not freed
    assert "h" in consumer._consumer._remote_encodings  # left for promotion


def test_on_peer_down_ignores_other_peer_entries(consumer):
    dead, alive = ("dead", 1), ("alive", 2)
    di = consumer._memory_context.region.alloc(1)
    ai = consumer._memory_context.region.alloc(1)
    consumer._consumer._remote_encodings["d"] = _pending(dead, dst_indices=di)
    consumer._consumer._remote_encodings["a"] = _pending(alive, dst_indices=ai)
    _put_peer(consumer, dead, nixl_agent_name="agent-dead")

    consumer._consumer.on_peer_down(dead)

    assert "d" not in consumer._consumer._remote_encodings
    assert consumer._consumer._remote_encodings["a"].addr == alive


def test_on_peer_down_removes_remote_agent(consumer):
    addr = ("host", 1234)
    _put_peer(consumer, addr, nixl_agent_name="agent-77")
    consumer._consumer.on_peer_down(addr)
    consumer._engine._nixl.remove_remote_agent.assert_called_once_with("agent-77")
    assert addr not in consumer._consumer._transport._peer_pool


def test_on_peer_down_tolerates_unknown_addr(consumer):
    consumer._consumer.on_peer_down(("ghost", 9999))  # must not raise


# ── poll_dead_peers wiring (consumer) ─────────────────────────────────────────


def test_drain_triggers_on_peer_down_on_disconnect_event(consumer):
    addr = ("host", 1234)
    indices = consumer._memory_context.region.alloc(2)
    consumer._consumer._remote_encodings["h"] = _pending(addr, dst_indices=indices)
    _put_peer(consumer, addr, nixl_agent_name="agent", zmq_monitor=MagicMock())
    free_before = len(consumer._memory_context.region._free)

    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.zmq_transport.recv_monitor_message",
        return_value={"event": zmq.EVENT_DISCONNECTED},
    ):
        consumer._consumer.drain()

    assert "h" not in consumer._consumer._remote_encodings  # forgotten → retry
    assert len(consumer._memory_context.region._free) == free_before + 2
    assert addr not in consumer._consumer._transport._peer_pool


def test_poll_dead_peers_skips_monitor_none(consumer):
    _put_peer(consumer, ("host", 1234), nixl_agent_name="agent")  # zmq_monitor None
    assert consumer._consumer._transport.poll_dead_peers() == []


# ── regression: stale metadata can never reach add_remote_agent ───────────────


def test_dead_peer_never_calls_add_remote_agent(consumer):
    """A request to a dead producer must fall back to local encode without
    ever feeding metadata to add_remote_agent (the original crash path)."""
    _put_peer(consumer)  # silent dealer — no XferAck ever arrives
    req = _request_for(_feature("h"), params={"h": _info()})

    # Kick off the read, then drain repeatedly past the ack timeout.
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is False
    consumer._consumer._remote_encodings["h"].deadline = time.monotonic() - 1.0
    consumer._consumer.drain()

    assert consumer._consumer._remote_encodings.get("h") is None  # tombstone
    consumer._engine._nixl.add_remote_agent.assert_not_called()


# ── shutdown ──────────────────────────────────────────────────────────────────


def test_shutdown_producer_stops_router_thread(producer):
    assert producer._producer_transport._router_t.is_alive()
    producer.shutdown()
    assert not producer._producer_transport._router_t.is_alive()


def test_shutdown_producer_releases_pins(producer):
    indices = _grant(producer, "h", n=1)
    producer.shutdown()
    assert all(idx not in producer._memory_context.region._ref_count for idx in indices)
    assert producer._producer._pinned_encodings == {}


def test_shutdown_consumer_closes_all_dealers(consumer):
    d1 = _put_peer(consumer, ("h1", 1), nixl_agent_name="a").zmq_dealer
    d2 = _put_peer(consumer, ("h2", 2), nixl_agent_name="b").zmq_dealer

    consumer.shutdown()

    d1.close.assert_called_once_with(linger=0)
    d2.close.assert_called_once_with(linger=0)
    assert consumer._consumer._transport._peer_pool == {}


def test_shutdown_consumer_releases_quarantine(consumer):
    indices = consumer._memory_context.region.alloc(1)
    consumer._consumer._quarantine.append(QuarantinedRead(indices, "rh"))
    consumer.shutdown()
    consumer._engine._nixl.release_xfer_handle.assert_any_call("rh")
    assert consumer._consumer._quarantine == []


def test_shutdown_calls_nixl_deregister_and_region_cleanup(consumer):
    consumer.shutdown()
    consumer._engine._nixl.deregister_memory.assert_called_once()
    assert consumer._memory_context.region._base is None
