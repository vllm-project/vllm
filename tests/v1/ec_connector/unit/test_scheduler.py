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
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.protocol import (
    EC_CONNECTOR_VERSION,
    XferAck,
    XferReq,
    XferStatus,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.session import (
    ProducerSession,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
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


def _put_session(
    sched: ECCPUScheduler,
    addr=("host", 1234),
    dealer=None,
    monitor=None,
):
    """Insert a ConsumerSession with a mock DEALER into the consumer's _sessions."""
    from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq import (
        ZmqClientConnection,
    )
    from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.session import (
        ConsumerSession,
    )

    conn_dealer = dealer or _silent_dealer()
    conn = ZmqClientConnection(dealer=conn_dealer, monitor=monitor)
    transport = sched._consumer._transport
    session = ConsumerSession(
        addr=addr,
        zmq_conn=conn,
        transport=transport,
        data=sched._data,
        compat_hash=sched._compat_hash,
    )
    sched._consumer._sessions[addr] = session
    transport._connections[addr] = conn
    return session


def _make_xfer_req(
    *,
    mm_hash: str = "h",
    compatibility_hash: str = "",
    connector_version: int = EC_CONNECTOR_VERSION,
    session_id: str = "test-session",
) -> XferReq:
    return XferReq(
        mm_hash=mm_hash,
        compatibility_hash=compatibility_hash,
        connector_version=connector_version,
        session_id=session_id,
    )


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _patch_router_run():
    """Replace ProducerSession._run with a sentinel that parks on _stop so the
    background thread starts, stays alive, and exits cleanly without touching
    mocked ZMQ sockets."""

    def _sentinel(self):
        self._stop.wait()

    with patch.object(ProducerSession, "_run", _sentinel):
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
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.data.nixl.NixlWrapper",
                return_value=mock_nixl,
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.data.nixl.nixl_agent_config",
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.setup_ec_region",
                return_value=layout,
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_socket",
                return_value=mock_router_sock,
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_path",
                return_value="tcp://mock:5000",
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.zmq.Context",
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
    producer._blocks["old"] = region.alloc(4)
    producer._local_encodings["old"] = None
    producer._blocks["new"] = region.alloc(4)
    producer._local_encodings["new"] = None

    result = producer._producer._fifo_alloc(3)

    assert len(result) == 3
    assert "old" not in producer._local_encodings
    assert "new" in producer._local_encodings


def test_producer_alloc_skips_pinned_blocks(producer):
    region = producer._memory_context.region
    pinned = region.alloc(4)
    not_pinned = region.alloc(4)
    region.pin(pinned)
    producer._blocks["pinned"] = pinned
    producer._local_encodings["pinned"] = None
    producer._blocks["not_pinned"] = not_pinned
    producer._local_encodings["not_pinned"] = None

    result = producer._producer._fifo_alloc(3)

    assert len(result) == 3
    assert "pinned" in producer._local_encodings
    assert "not_pinned" not in producer._local_encodings
    region.unpin(pinned)


def test_producer_alloc_raises_when_all_encodings_pinned(producer):
    region = producer._memory_context.region
    all_idx = region.alloc(_NUM_BLOCKS)
    region.pin(all_idx)
    producer._blocks["only"] = all_idx
    producer._local_encodings["only"] = None
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
    producer._blocks["old"] = region.alloc(_NUM_BLOCKS)
    producer._local_encodings["old"] = None

    req = _request_for(_feature("h", length=1))
    assert producer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert "old" not in producer._local_encodings


def test_producer_ensure_returns_false_when_all_blocks_pinned(producer):
    """When all blocks are pinned by in-flight reads, returns False to defer."""
    region = producer._memory_context.region
    all_idx = region.alloc(_NUM_BLOCKS)
    region.pin(all_idx)
    producer._blocks["pinned"] = all_idx
    producer._local_encodings["pinned"] = None

    req = _request_for(_feature("h", length=1))
    assert producer.ensure_cache_available(req, num_computed_tokens=0) is False
    region.unpin(all_idx)


def test_producer_ensure_skips_feature_in_local_encodings(producer):
    """mm_hash already stored — no probe, region free count unchanged."""
    producer._local_encodings["h"] = None
    free_before = len(producer._memory_context.region._free)
    assert producer.ensure_cache_available(_request_for(_feature("h")), 0) is True
    assert len(producer._memory_context.region._free) == free_before


def test_producer_ensure_skips_feature_in_pending_save(producer):
    """mm_hash already allocated last step — no probe."""
    producer._producer._pending_save.add("h")
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
    sched._blocks["pinned"] = all_idx
    sched._local_encodings["pinned"] = None

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
    producer._producer._pending_save.add("h1")
    producer.update_state_after_alloc(_request_for(_feature("h1")), index=0)
    assert "h1" not in producer._producer._pending_new_encodings


def test_update_state_dedups_against_local_encodings(producer):
    producer._local_encodings["h1"] = None
    producer.update_state_after_alloc(_request_for(_feature("h1")), index=0)
    assert "h1" not in producer._producer._pending_new_encodings


# ── build_connector_meta (producer) ───────────────────────────────────────────


def test_build_meta_promotes_pending_save(producer):
    producer._blocks["h1"] = [0, 1]
    producer._producer._pending_save.add("h1")
    producer.build_connector_meta(Mock(spec=SchedulerOutput))
    assert "h1" in producer._local_encodings
    assert producer._local_encodings["h1"] is None
    assert producer._blocks["h1"] == [0, 1]
    assert producer._producer._pending_save == set()


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


# ── ProducerSession: grant/NACK ───────────────────────────────────────────────


def test_handle_xfer_req_version_mismatch_nacks(producer):
    req = _make_xfer_req(connector_version=EC_CONNECTOR_VERSION + 99)
    ack = producer._producer_session._grant_or_nack(req)
    assert ack.status == XferStatus.NACK_VERSION
    assert not producer._producer_session._active_xfers


def test_handle_xfer_req_compat_hash_mismatch_nacks(producer):
    req = _make_xfer_req(compatibility_hash="not-a-real-hash")
    ack = producer._producer_session._grant_or_nack(req)
    assert ack.status == XferStatus.NACK_INCOMPAT
    assert not producer._producer_session._active_xfers


def test_handle_xfer_req_unknown_mm_hash_nacks(producer):
    req = _make_xfer_req(mm_hash="unknown", compatibility_hash=producer._compat_hash)
    ack = producer._producer_session._grant_or_nack(req)
    assert ack.status == XferStatus.NACK_MISSING
    assert ack.mm_hash == "unknown"
    assert not producer._producer_session._active_xfers


def test_handle_xfer_req_success_pins_and_returns_grant(producer):
    indices = producer._memory_context.region.alloc(2)
    producer._blocks["h"] = indices
    producer._local_encodings["h"] = None
    req = _make_xfer_req(
        mm_hash="h", compatibility_hash=producer._compat_hash, session_id="sess-1"
    )

    ack = producer._producer_session._grant_or_nack(req)

    assert ack.status == XferStatus.OK
    assert ack.src_block_indices == indices
    assert ack.agent_metadata == producer._data.get_agent_metadata()
    assert ack.mem_descriptor == producer._data.get_mem_descriptor()
    assert all(idx in producer._memory_context.region._ref_count for idx in indices)
    assert "sess-1:h" in producer._producer_session._active_xfers


def test_handle_xfer_req_two_consumers_same_mm_hash(producer):
    """Two consumers (different session_ids) each pin the source once."""
    indices = producer._memory_context.region.alloc(1)
    producer._blocks["h"] = indices
    producer._local_encodings["h"] = None

    producer._producer_session._grant_or_nack(
        _make_xfer_req(
            mm_hash="h", compatibility_hash=producer._compat_hash, session_id="c1"
        )
    )
    producer._producer_session._grant_or_nack(
        _make_xfer_req(
            mm_hash="h", compatibility_hash=producer._compat_hash, session_id="c2"
        )
    )

    assert "c1:h" in producer._producer_session._active_xfers
    assert "c2:h" in producer._producer_session._active_xfers
    assert producer._memory_context.region._ref_count[indices[0]] == 2


# ── ProducerSession: poll (notif drain + timeout sweep) ───────────────────────


def _grant(producer, mm_hash="h", n=1, session_id="test-session"):
    """Grant a read and return (block_indices, xfer_key)."""
    indices = producer._memory_context.region.alloc(n)
    producer._blocks[mm_hash] = indices
    producer._local_encodings[mm_hash] = None
    req = _make_xfer_req(
        mm_hash=mm_hash, compatibility_hash=producer._compat_hash, session_id=session_id
    )
    ack = producer._producer_session._grant_or_nack(req)
    assert ack.status == XferStatus.OK
    return indices, f"{session_id}:{mm_hash}"


def test_poll_notif_unpins_source(producer):
    indices, key = _grant(producer, "h", n=2)
    producer._data._nixl.get_new_notifs.return_value = {"peer": [key.encode()]}

    producer._producer_session.poll([])

    assert all(idx not in producer._memory_context.region._ref_count for idx in indices)
    assert key not in producer._producer_session._active_xfers


def test_poll_notif_for_unknown_key_is_noop(producer):
    producer._data._nixl.get_new_notifs.return_value = {"peer": [b"ghost:unknown"]}
    producer._producer_session.poll([])  # must not raise
    assert not producer._producer_session._active_xfers


def test_poll_lease_expiry_force_unpins(producer):
    indices, key = _grant(producer, "h", n=1)
    producer._producer_session._active_xfers[key].deadline = time.monotonic() - 1.0

    producer._producer_session.poll([])

    assert all(idx not in producer._memory_context.region._ref_count for idx in indices)
    assert key not in producer._producer_session._active_xfers


def test_poll_notif_then_late_lease_does_not_double_unpin(producer):
    indices, key = _grant(producer, "h", n=1)
    producer._data._nixl.get_new_notifs.return_value = {"peer": [key.encode()]}
    producer._producer_session.poll([])  # notif unpins
    producer._data._nixl.get_new_notifs.return_value = {}
    producer._producer_session.poll([])  # sweep over empty table — must not raise
    assert all(idx not in producer._memory_context.region._ref_count for idx in indices)


def test_has_pending_xfers(producer):
    assert not producer._producer_session._active_xfers
    _grant(producer, "h")
    assert producer._producer_session._active_xfers


# DataTransport.post_read behavior is tested in test_data.py.


# ── request_finished (producer) ───────────────────────────────────────────────


def test_request_finished_returns_no_params_when_nothing_known(producer):
    done, params = producer.request_finished(_request_for(_feature("h")))
    assert done is False
    assert params is None


def test_request_finished_emits_address_and_size_only(producer):
    producer._local_encodings["h"] = None
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
    producer._producer._pending_save.add("h")
    _, params = producer.request_finished(_request_for(_feature("h")))
    assert params is not None and "h" in params


def test_request_finished_uses_identifier_when_mm_hash_falsy(producer):
    feat = _feature("ident-only")
    feat.mm_hash = None
    producer._local_encodings["ident-only"] = None
    _, params = producer.request_finished(_request_for(feat))
    assert "ident-only" in params


# ── _start_xfer (consumer) ────────────────────────────────────────────────────
# Session-level behavior (XferAck dispatch, drain, quarantine, NIXL registration)
# is tested in test_session.py. These tests cover the scheduler-level wiring.


def test_start_xfer_creates_session_and_sends_xfer_req(consumer):
    """_start_xfer lazily creates a ConsumerSession and sends XferReq."""
    new_dealer = MagicMock()
    new_dealer.recv_multipart.side_effect = zmq.Again
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_socket",
        return_value=new_dealer,
    ):
        consumer._consumer._start_xfer("h", _info(), _BLOCK_SIZE)

    addr = ("host", 1234)
    assert addr in consumer._consumer._sessions
    session = consumer._consumer._sessions[addr]
    assert "h" in session._xfers
    # XferReq sent over the DEALER, with correct mm_hash and session_id.
    new_dealer.send_multipart.assert_called_once()
    sent = new_dealer.send_multipart.call_args[0][0]
    req = msgspec.msgpack.decode(sent[1], type=XferReq)
    assert req.mm_hash == "h"
    assert req.session_id == session._session_id
    # Blocks allocated and tracked.
    assert len(consumer._consumer._blocks["h"]) == 1


def test_start_xfer_reuses_existing_session(consumer):
    """Second _start_xfer to the same addr reuses the existing session."""
    new_dealer = MagicMock()
    new_dealer.recv_multipart.side_effect = zmq.Again
    addr = ("host", 1234)
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_socket",
        return_value=new_dealer,
    ):
        consumer._consumer._start_xfer("h1", _info(), _BLOCK_SIZE)
        consumer._consumer._start_xfer("h2", _info(), _BLOCK_SIZE)

    assert len(consumer._consumer._sessions) == 1
    session = consumer._consumer._sessions[addr]
    assert "h1" in session._xfers
    assert "h2" in session._xfers


def test_start_xfer_send_failure_frees_blocks_and_reraises(consumer):
    new_dealer = MagicMock()
    new_dealer.send_multipart.side_effect = RuntimeError("socket dead")
    new_dealer.recv_multipart.side_effect = zmq.Again
    free_before = sorted(consumer._memory_context.region._free)
    with (
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_socket",
            return_value=new_dealer,
        ),
        pytest.raises(RuntimeError, match="socket dead"),
    ):
        consumer._consumer._start_xfer("h", _info(), _BLOCK_SIZE)

    assert sorted(consumer._memory_context.region._free) == free_before
    assert "h" not in consumer._consumer._in_flight


# ── consumer _fifo_alloc ──────────────────────────────────────────────────────


def test_consumer_fifo_alloc_evicts_local_encodings_in_insertion_order(consumer):
    region = consumer._memory_context.region
    cold_blocks = region.alloc(_NUM_BLOCKS // 2)
    consumer._consumer._blocks["cold"] = cold_blocks
    consumer._consumer._local_encodings["cold"] = None
    hot_blocks = region.alloc(_NUM_BLOCKS // 2)
    consumer._consumer._blocks["hot"] = hot_blocks
    consumer._consumer._local_encodings["hot"] = None

    indices = consumer._consumer._fifo_alloc(1)

    assert len(indices) == 1
    assert "cold" not in consumer._consumer._local_encodings
    assert "cold" not in consumer._consumer._blocks
    assert "hot" in consumer._consumer._local_encodings


def test_consumer_fifo_alloc_protects_pending_reload(consumer):
    region = consumer._memory_context.region
    consumer._blocks["A"] = region.alloc(_NUM_BLOCKS)
    consumer._local_encodings["A"] = None
    region.pin(consumer._blocks["A"])
    consumer._consumer._pending_reload.add("A")
    with pytest.raises(AllocationError):
        consumer._consumer._fifo_alloc(1)
    assert "A" in consumer._local_encodings


# ── build_connector_meta (consumer) ───────────────────────────────────────────


def test_consumer_build_meta_promotes_completed_to_loads_and_local_encodings(consumer):
    indices = consumer._memory_context.region.alloc(2)
    consumer._consumer._blocks["h"] = indices
    # Simulate a transfer that completed during this step.
    consumer._consumer._step_completed.add("h")

    meta = consumer.build_connector_meta(Mock(spec=SchedulerOutput))

    assert meta.loads["h"] == indices
    assert "h" in consumer._consumer._local_encodings
    assert consumer._consumer._blocks["h"] == indices
    assert not consumer._consumer._step_completed  # cleared for next step


def test_consumer_build_meta_re_emits_pending_reload(consumer):
    region = consumer._memory_context.region
    indices = region.alloc(1)
    consumer._consumer._blocks["h"] = indices
    consumer._consumer._local_encodings["h"] = None
    region.pin(indices)
    consumer._consumer._pending_reload.add("h")
    free_before = len(region._free)

    meta = consumer.build_connector_meta(Mock(spec=SchedulerOutput))

    assert meta.loads["h"] is consumer._consumer._blocks["h"]
    assert consumer._consumer._pending_reload == set()
    assert len(region._free) == free_before


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
    assert "h" not in consumer._consumer._in_flight


def test_ensure_falls_through_on_size_mismatch(consumer):
    params = {"h": _info(size_bytes=_BLOCK_SIZE * 99)}
    req = _request_for(_feature("h", length=1), params=params)
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert "h" not in consumer._consumer._in_flight


def test_ensure_defers_when_already_in_flight(consumer):
    consumer._consumer._in_flight.add("h")
    req = _request_for(_feature("h"), params={"h": _info()})
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is False


def test_ensure_consumes_nack_tombstone(consumer):
    consumer._consumer._tombstones.add("h")
    req = _request_for(_feature("h"), params={"h": _info()})
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert "h" not in consumer._consumer._tombstones  # consumed


def test_ensure_admits_when_already_local_encodings_and_marks_pending_reload(consumer):
    consumer._consumer._blocks["h"] = [0]
    consumer._consumer._local_encodings["h"] = None
    req = _request_for(_feature("h"), params={"h": _info()})
    assert consumer.ensure_cache_available(req, num_computed_tokens=0) is True
    assert "h" in consumer._consumer._pending_reload


def test_ensure_starts_read_for_uncached_announced_feature(consumer):
    new_dealer = MagicMock()
    new_dealer.recv_multipart.side_effect = zmq.Again
    req = _request_for(_feature("h"), params={"h": _info()})
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_socket",
        return_value=new_dealer,
    ):
        result = consumer.ensure_cache_available(req, num_computed_tokens=0)

    assert result is False  # deferred — read kicked off
    assert "h" in consumer._consumer._in_flight
    new_dealer.send_multipart.assert_called_once()


def test_ensure_alloc_failure_falls_through_to_local_encode(consumer, caplog_vllm):
    _region = consumer._memory_context.region
    _protected = _region.alloc(_NUM_BLOCKS)
    consumer._local_encodings["protected"] = None
    consumer._blocks["protected"] = _protected
    _region.pin(_protected)
    consumer._consumer._pending_reload.add("protected")
    req = _request_for(_feature("new"), params={"new": _info()})

    with caplog_vllm.at_level(logging.WARNING):
        result = consumer.ensure_cache_available(req, num_computed_tokens=0)

    assert "new" not in consumer._consumer._in_flight
    assert result is True


# ── has_cache_item ────────────────────────────────────────────────────────────


def test_has_cache_item_true_for_local_encodings(consumer):
    consumer._consumer._local_encodings["h"] = None
    assert consumer.has_cache_item("h") is True


def test_has_cache_item_false_for_in_flight(consumer):
    consumer._consumer._in_flight.add("h")
    assert consumer.has_cache_item("h") is False


# ── on_peer_down (consumer) ───────────────────────────────────────────────────


def test_on_peer_down_awaiting_ack_frees_and_retries(consumer):
    addr = ("host", 1234)
    session = _put_session(consumer, addr)
    indices = consumer._memory_context.region.alloc(2)
    consumer._consumer._blocks["h"] = indices
    consumer._consumer._in_flight.add("h")
    # Inject a WAITING_ACK ConsumerXfer into the session.
    from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.session import (
        ConsumerXfer,
    )

    session._xfers["h"] = ConsumerXfer(
        mm_hash="h",
        block_indices=indices,
        addr=addr,
        deadline=time.monotonic() + 60,
        data=consumer._data,
        consumer_session_id="test-sess",
    )
    free_before = len(consumer._memory_context.region._free)

    consumer._consumer._on_peer_down(addr)

    # WAITING_ACK → cancelled: blocks freed, no tombstone (retry ok).
    assert "h" not in consumer._consumer._in_flight
    assert "h" not in consumer._consumer._tombstones
    assert len(consumer._memory_context.region._free) == free_before + 2
    assert addr not in consumer._consumer._sessions


def test_on_peer_down_does_not_affect_local_encodings_entries(consumer):
    addr = ("host", 1234)
    indices = consumer._memory_context.region.alloc(2)
    consumer._consumer._blocks["h"] = indices
    consumer._consumer._local_encodings["h"] = None
    _put_session(consumer, addr)  # no xfer in flight for "h"
    free_before = len(consumer._memory_context.region._free)

    consumer._consumer._on_peer_down(addr)

    assert len(consumer._memory_context.region._free) == free_before
    assert "h" in consumer._consumer._local_encodings


def test_on_peer_down_tolerates_unknown_addr(consumer):
    consumer._consumer._on_peer_down(("ghost", 9999))  # must not raise


# ── poll_dead wiring (consumer) ───────────────────────────────────────────────


def test_poll_step_triggers_on_peer_down_on_disconnect_event(consumer):
    addr = ("host", 1234)
    mock_monitor = MagicMock()
    session = _put_session(consumer, addr, monitor=mock_monitor)
    indices = consumer._memory_context.region.alloc(2)
    consumer._consumer._blocks["h"] = indices
    consumer._consumer._in_flight.add("h")
    from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.session import (
        ConsumerXfer,
    )

    session._xfers["h"] = ConsumerXfer(
        mm_hash="h",
        block_indices=indices,
        addr=addr,
        deadline=time.monotonic() + 60,
        data=consumer._data,
        consumer_session_id="test-sess",
    )

    consumer._consumer._step_polled = False  # allow a fresh poll
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.recv_monitor_message",
        return_value={"event": zmq.EVENT_DISCONNECTED},
    ):
        consumer._consumer._poll_step()

    # Peer went down → WAITING_ACK cancelled → blocks freed
    assert "h" not in consumer._consumer._in_flight
    assert addr not in consumer._consumer._sessions


def test_poll_dead_skips_conn_without_monitor(consumer):
    _put_session(consumer)  # ZmqClientConnection has monitor=None
    assert consumer._consumer._transport.poll_dead() == []


# ── regression: ack timeout tombstones without calling add_remote_peer ────────


def test_ack_timeout_never_calls_add_remote_peer(consumer):
    """A timed-out XferAck must tombstone without touching NIXL registration."""
    new_dealer = MagicMock()
    new_dealer.recv_multipart.side_effect = zmq.Again
    req = _request_for(_feature("h"), params={"h": _info()})
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_socket",
        return_value=new_dealer,
    ):
        consumer.ensure_cache_available(
            req,
            num_computed_tokens=0,
        )

    # Force ack deadline into the past.
    addr = ("host", 1234)
    session = consumer._consumer._sessions[addr]
    session._xfers["h"].deadline = time.monotonic() - 1.0
    consumer._consumer._step_polled = False  # allow a fresh poll

    consumer._consumer._poll_step()

    assert "h" not in consumer._consumer._in_flight
    assert "h" in consumer._consumer._tombstones
    consumer._data._nixl.add_remote_agent.assert_not_called()


# ── shutdown ──────────────────────────────────────────────────────────────────


def test_shutdown_producer_stops_router_thread(producer):
    assert producer._producer_session._thread.is_alive()
    producer.shutdown()
    assert not producer._producer_session._thread.is_alive()


def test_shutdown_producer_releases_pins(producer):
    indices, key = _grant(producer, "h", n=1)
    producer.shutdown()
    assert all(idx not in producer._memory_context.region._ref_count for idx in indices)
    assert not producer._producer_session._active_xfers


def test_shutdown_consumer_closes_all_sessions(consumer):
    s1 = _put_session(consumer, ("h1", 1))
    s2 = _put_session(consumer, ("h2", 2))
    consumer.shutdown()
    assert not s1._zmq.alive
    assert not s2._zmq.alive
    assert not consumer._consumer._sessions


def test_shutdown_consumer_releases_quarantined_xfers(consumer):
    """Shutdown must release NIXL handles for sessions with quarantined xfers."""
    from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.session import (
        ConsumerXfer,
    )

    session = _put_session(consumer)
    indices = consumer._memory_context.region.alloc(1)
    xfer = ConsumerXfer(
        mm_hash="h",
        block_indices=indices,
        addr=("host", 1234),
        deadline=time.monotonic() - 1,
        data=consumer._data,
        consumer_session_id="test-sess",
    )
    xfer.transfer_handle = "rh"
    xfer._quarantined = True
    session._quarantined.append(xfer)

    consumer.shutdown()

    consumer._data._nixl.release_xfer_handle.assert_any_call("rh")


def test_shutdown_calls_nixl_deregister_and_region_cleanup(consumer):
    consumer.shutdown()
    consumer._data._nixl.deregister_memory.assert_called_once()
    assert consumer._memory_context.region._base is None
