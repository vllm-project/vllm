# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The consumer must not read remote KV after the producer's lease ran out.

A router can hold a prefilled request outside the decode engine (no heartbeats)
for longer than P's `kv_lease_duration`. P then releases and recycles the
blocks, so the stale `remote_block_ids` address other requests' KV: the NIXL
read succeeds and silently loads corrupted KV and SSM state. The pull worker
declines such reads (`failed_recving` -> kv_load_failure_policy) unless its
own heartbeats kept the lease alive, which it models exactly.
"""

import time
from unittest.mock import patch

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlConnector,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    _KV_BLOCKS_EXPIRY_SAFETY_MARGIN,
    RemoteLeaseModel,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    HeartbeatInfo,
    LookupPinReply,
    LookupPinRequest,
    PeerKVIdentity,
    PeerMissReason,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    kv_lease_extension,
)
from vllm.forward_context import ForwardContext

from .test_nixl_connector import FakeNixlConnectorWorker, FakeNixlWrapper
from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
    make_nixl_scheduler,
    maybe_update_block_size,
)

_REMOTE = FakeNixlConnectorWorker.REMOTE_ENGINE_ID
_REMOTE_REQ = "prefill-req"
_KEY = (_REMOTE, _REMOTE_REQ)
_MARGIN = _KV_BLOCKS_EXPIRY_SAFETY_MARGIN
_LEASE = 30  # producer's kv_lease_duration
_EXTENSION = kv_lease_extension(_LEASE)  # 20s per heartbeat

_FAKE_WRAPPER = patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.NixlWrapper",
    FakeNixlWrapper,
)


def _make_connector():
    """Worker-side NixlConnector (pull mode, no bidirectional transfer) with a
    handshaken fake remote and a zero clock offset."""
    block_size = maybe_update_block_size(32)
    vllm_config = create_vllm_config(block_size=block_size)
    kv_cache_config = make_kv_cache_config(block_size=block_size, num_blocks=2)
    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
    connector.connector_worker = FakeNixlConnectorWorker(
        vllm_config,
        connector.engine_id,
        hand_shake_latency=0,
        kv_cache_config=kv_cache_config,
    )
    worker = connector.connector_worker
    assert isinstance(worker.nixl_wrapper, FakeNixlWrapper)
    worker.kv_cache_layout = "LBHNC"
    remote_agents, _ = worker._nixl_handshake(
        host="localhost", port=1234, remote_tp_size=1, expected_engine_id=_REMOTE
    )
    worker._remote_agents[_REMOTE] = remote_agents
    worker._engine_clock_offset[_REMOTE] = 0.0
    return connector, worker


def _recv_meta(expiry_delta: float | None, local_blocks=([10, 11],)):
    """D-side metadata for a remote-prefill read whose producer deadline is
    `expiry_delta` seconds from now (None: deadline not exported)."""
    params = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": ([20, 21],),
        "remote_engine_id": _REMOTE,
        "remote_request_id": _REMOTE_REQ,
        "remote_host": "localhost",
        "remote_port": 1234,
        "tp_size": 1,
        "remote_tp_size": 1,
    }
    if expiry_delta is not None:
        params["remote_blocks_expiry_time"] = time.perf_counter() + expiry_delta
        params["remote_lease_extension"] = _EXTENSION
    meta = NixlConnectorMetadata()
    meta.add_new_req_to_recv(
        request_id="req", local_block_ids=local_blocks, kv_transfer_params=params
    )
    return meta


def _hb_meta(*req_ids: str, leases=None, engine_id: str = _REMOTE):
    """Scheduler heartbeat batch for `req_ids`; `leases` maps a remote
    request id to (remote deadline, extension) as the scheduler ships it."""
    meta = NixlConnectorMetadata()
    meta.heartbeat_by_engine = {
        engine_id: HeartbeatInfo(
            req_ids=set(req_ids),
            host="localhost",
            port=1234,
            tp_size=1,
            leases=dict(leases or {}),
        )
    }
    return meta


def _load(connector, meta):
    connector.bind_connector_metadata(meta)
    ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    connector.start_load_kv(ctx)
    return connector.connector_worker.get_transfer_results()


def _assert_declined(worker, result, declined: bool):
    assert worker.xfer_stats.data["num_declined_expired_reads"] == (
        [1] if declined else []
    )
    # A declined read is neither a transport failure nor a P-side expiry.
    assert worker.xfer_stats.data["num_failed_transfers"] == []
    assert worker.xfer_stats.data["num_kv_expired_reqs"] == []
    assert "req" in result.finished_recving
    assert ("req" in result.failed_recving) is declined


# ---------------------------------------------------------------------------
# Read gate on the exported deadline
# ---------------------------------------------------------------------------


@_FAKE_WRAPPER
@pytest.mark.parametrize(
    "expiry_delta, expect_declined",
    [
        (-10.0, True),  # lease ran out 10s ago
        (_MARGIN / 2, True),  # inside the safety margin
        (20.0, False),  # lease still live
        (None, False),  # deadline not exported (older peer / router dropped it)
    ],
)
def test_remote_prefill_read_gate(dist_init, expiry_delta, expect_declined):
    connector, worker = _make_connector()
    result = _load(connector, _recv_meta(expiry_delta))
    _assert_declined(worker, result, expect_declined)


@_FAKE_WRAPPER
def test_full_local_prefix_hit_skips_gate(dist_init):
    """No read is issued for a full local hit, so an expired deadline is moot."""
    connector, worker = _make_connector()
    result = _load(connector, _recv_meta(-10.0, local_blocks=()))
    assert worker.xfer_stats.data["num_declined_expired_reads"] == []
    assert "req" not in result.failed_recving


@_FAKE_WRAPPER
def test_unknown_clock_offset_reads_as_usual(dist_init):
    connector, worker = _make_connector()
    worker._engine_clock_offset.pop(_REMOTE)
    result = _load(connector, _recv_meta(-10.0))
    _assert_declined(worker, result, False)


@_FAKE_WRAPPER
def test_clock_offset_rebases_the_deadline(dist_init):
    """The exported deadline is in the remote's perf_counter domain
    (offset = remote - local)."""
    connector, worker = _make_connector()
    # Remote clock runs 100s ahead: a deadline 90s ahead on its clock is 10s
    # in the past on ours.
    worker._engine_clock_offset[_REMOTE] = 100.0
    result = _load(connector, _recv_meta(90.0))
    _assert_declined(worker, result, True)


# ---------------------------------------------------------------------------
# Lease model fed by the heartbeats this worker sent
# ---------------------------------------------------------------------------


@_FAKE_WRAPPER
@pytest.mark.parametrize(
    "expiry_delta, model_deadline_delta, expect_declined",
    [
        # Renewed well past now: read.
        (-10.0, 20.0, False),
        # Renewed, but the renewal ends inside the margin: decline.
        (-10.0, _MARGIN / 2, True),
        # A lapsed lease keeps the deadline it ran out at.
        (-10.0, -10.0, True),
        # Initial lease still far away: the model cannot shorten it.
        (100.0, -10.0, False),
    ],
)
def test_lease_model_decides_the_read(
    dist_init, expiry_delta, model_deadline_delta, expect_declined
):
    connector, worker = _make_connector()
    worker._remote_leases[_KEY] = RemoteLeaseModel(
        deadline=time.perf_counter() + model_deadline_delta
    )
    result = _load(connector, _recv_meta(expiry_delta))
    _assert_declined(worker, result, expect_declined)
    # Bookkeeping is dropped once the read is decided either way.
    assert _KEY not in worker._remote_leases


@_FAKE_WRAPPER
def test_model_for_another_engine_does_not_count(dist_init):
    connector, worker = _make_connector()
    worker._remote_leases[("other-engine", _REMOTE_REQ)] = RemoteLeaseModel(
        deadline=time.perf_counter() + 100.0
    )
    result = _load(connector, _recv_meta(-10.0))
    _assert_declined(worker, result, True)


@_FAKE_WRAPPER
@pytest.mark.parametrize(
    "heartbeat_times, read_time, expect_declined",
    [
        # Every 5s from prefill time (deadline 30); read 60s later.
        ([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], 60.0, False),
        # Last heartbeat at 55 renews to 75: a read at 70.5 is inside the
        # margin (70.5 + 5 >= 75), so it is declined.
        ([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], 70.5, True),
        # A 19s pause inside the initial lease is harmless: the remote renews
        # at 20 (to 40), 25 (to 45) and 30 (to 50).
        ([1, 20, 25, 30], 34.0, False),
        # The heartbeat at 40 finds the lease (30) gone: lapsed at 30.
        ([0, 5, 10, 40, 45], 48.0, True),
        # Started only at 28: 28 + margin > 30, so the model does not credit
        # it although the remote did renew at ~28 (conservative by design).
        ([28, 33, 38], 40.0, True),
        # Without any heartbeat the initial deadline stands.
        ([], 31.0, True),
        ([], 20.0, False),
    ],
)
def test_heartbeat_chain_end_to_end(
    dist_init, monkeypatch, heartbeat_times, read_time, expect_declined
):
    """Drive the real heartbeat path (_send_heartbeats) on a fake clock and
    check the read decision the resulting model leads to."""
    connector, worker = _make_connector()
    now = {"t": 0.0}
    monkeypatch.setattr(time, "perf_counter", lambda: now["t"])
    # Prefill finished at t=0 on a peer with the same clock: deadline t=30.
    meta = _recv_meta(_LEASE)
    leases = {_REMOTE_REQ: (float(_LEASE), _EXTENSION)}
    for t in heartbeat_times:
        now["t"] = float(t)
        worker._send_heartbeats(_hb_meta(_REMOTE_REQ, leases=leases))
    now["t"] = read_time
    result = _load(connector, meta)
    _assert_declined(worker, result, expect_declined)


@_FAKE_WRAPPER
def test_lease_model_mirrors_the_remote_renewal(dist_init, monkeypatch):
    """Deadline after each heartbeat equals what the remote's _handle_heartbeat
    would hold, with the send time standing in for the arrival time."""
    _, worker = _make_connector()
    now = {"t": 0.0}
    monkeypatch.setattr(time, "perf_counter", lambda: now["t"])
    leases = {_REMOTE_REQ: (30.0, 20)}
    expected = [(1.0, 30.0), (20.0, 40.0), (24.0, 44.0), (38.0, 58.0)]
    for t, deadline in expected:
        now["t"] = t
        worker._send_heartbeats(_hb_meta(_REMOTE_REQ, leases=leases))
        model = worker._remote_leases[_KEY]
        assert (model.deadline, model.alive) == (deadline, True)
    # 54 + margin > 58: this heartbeat finds the lease gone; it and all later
    # ones renew nothing, and the deadline stays where the lease ran out.
    for t in (54.0, 59.0):
        now["t"] = t
        worker._send_heartbeats(_hb_meta(_REMOTE_REQ, leases=leases))
    model = worker._remote_leases[_KEY]
    assert (model.deadline, model.alive) == (58.0, False)


@_FAKE_WRAPPER
def test_heartbeats_without_lease_info_build_no_model(dist_init):
    """A remote that exported no deadline or no renewal amount (older peer,
    fixed TTL) is never modelled: the read sees the plain deadline."""
    connector, worker = _make_connector()
    worker._send_heartbeats(_hb_meta(_REMOTE_REQ))
    assert worker._remote_leases == {}
    result = _load(connector, _recv_meta(-10.0))
    _assert_declined(worker, result, True)


@_FAKE_WRAPPER
def test_request_id_with_comma_is_never_credited(dist_init):
    """The heartbeat payload is comma-separated, so the remote cannot match
    such an id and renews nothing for it."""
    _, worker = _make_connector()
    now = time.perf_counter()
    leases = {"tenant,42": (now + 30.0, _EXTENSION), "plain": (now + 30.0, 20)}
    worker._send_heartbeats(_hb_meta("tenant,42", "plain", leases=leases))
    assert set(worker._remote_leases) == {(_REMOTE, "plain")}


@_FAKE_WRAPPER
def test_unknown_clock_offset_builds_no_model(dist_init):
    _, worker = _make_connector()
    worker._engine_clock_offset.pop(_REMOTE)
    leases = {_REMOTE_REQ: (time.perf_counter() + 30.0, _EXTENSION)}
    worker._send_heartbeats(_hb_meta(_REMOTE_REQ, leases=leases))
    assert worker._remote_leases == {}


@_FAKE_WRAPPER
def test_send_heartbeats_prunes_finished_requests(dist_init, monkeypatch):
    _, worker = _make_connector()
    now = {"t": 100.0}
    monkeypatch.setattr(time, "perf_counter", lambda: now["t"])
    leases = {"a": (130.0, 20), "b": (130.0, 20)}
    worker._send_heartbeats(_hb_meta("a", "b", leases=leases))
    now["t"] = 112.0
    worker._send_heartbeats(_hb_meta("a", "b", leases=leases))
    assert worker._remote_leases[(_REMOTE, "a")].deadline == 132.0
    # "b" left the scheduler's heartbeat set: its model goes away.
    worker._send_heartbeats(_hb_meta("a", leases={"a": leases["a"]}))
    assert set(worker._remote_leases) == {(_REMOTE, "a")}
    # An empty heartbeat round (throttled) must not prune anything.
    worker._send_heartbeats(NixlConnectorMetadata())
    assert set(worker._remote_leases) == {(_REMOTE, "a")}


@_FAKE_WRAPPER
def test_failed_heartbeat_send_is_not_recorded(dist_init):
    """A heartbeat that never left cannot have renewed anything."""
    _, worker = _make_connector()
    leases = {"a": (time.perf_counter() + 30.0, 20)}
    with patch.object(
        worker.nixl_wrapper, "send_notif", side_effect=RuntimeError("down")
    ):
        worker._send_heartbeats(_hb_meta("a", leases=leases))
    assert worker._remote_leases == {}


@_FAKE_WRAPPER
def test_heartbeat_round_missing_one_rank_is_not_recorded(dist_init):
    """Every remote rank must have received the round for it to count."""
    _, worker = _make_connector()
    agents = worker._remote_agents[_REMOTE]
    agents[(1, 0)] = "second-rank-agent"
    assert len(agents) == 2
    leases = {"a": (time.perf_counter() + 30.0, 20)}

    def send_notif(agent_name, notif_msg):
        if agent_name == "second-rank-agent":
            raise RuntimeError("down")

    with patch.object(worker.nixl_wrapper, "send_notif", side_effect=send_notif):
        worker._send_heartbeats(_hb_meta("a", leases=leases))
    assert worker._remote_leases == {}
    # Once every rank gets the round, it is recorded.
    worker._send_heartbeats(_hb_meta("a", leases=leases))
    assert (_REMOTE, "a") in worker._remote_leases


@_FAKE_WRAPPER
def test_replicated_pcp_rank_keeps_a_lease_model(dist_init, monkeypatch):
    """Replicated-PCP ranks > 0 leave start_load_kv early but still gate their
    reads, so they must send (and model) heartbeats before that early return."""
    connector, worker = _make_connector()
    worker.pcp_size, worker.pcp_rank, worker.pcp_dcp_sharded = 2, 1, False
    now = {"t": 0.0}
    monkeypatch.setattr(time, "perf_counter", lambda: now["t"])
    leases = {_REMOTE_REQ: (30.0, 20)}
    ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    for t in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0):
        now["t"] = t
        connector.bind_connector_metadata(_hb_meta(_REMOTE_REQ, leases=leases))
        connector.start_load_kv(ctx)
        connector.clear_connector_metadata()
    assert worker._remote_leases[_KEY].deadline == 70.0


@_FAKE_WRAPPER
def test_heartbeat_round_in_the_read_step_is_credited_first(dist_init, monkeypatch):
    """A heartbeat round shipped in the same step as the read is folded into
    the model before the gate looks at it."""
    connector, worker = _make_connector()
    now = {"t": 0.0}
    monkeypatch.setattr(time, "perf_counter", lambda: now["t"])
    leases = {_REMOTE_REQ: (30.0, 20)}
    now["t"] = 20.0
    worker._send_heartbeats(_hb_meta(_REMOTE_REQ, leases=leases))  # -> 40
    # At t=35 the model (40) is inside the margin; the round carried with the
    # read renews it to 55 first, so the read goes ahead.
    now["t"] = 35.0
    meta = _recv_meta(30.0 - 35.0)  # producer deadline 30 (expired on its own)
    meta.heartbeat_by_engine = _hb_meta(_REMOTE_REQ, leases=leases).heartbeat_by_engine
    result = _load(connector, meta)
    _assert_declined(worker, result, False)


@_FAKE_WRAPPER
def test_model_is_dropped_when_the_transfer_completes(dist_init):
    """Heartbeats may still go out between the read and its completion; the
    model they recreate must not outlive the transfer."""
    connector, worker = _make_connector()
    leases = {_REMOTE_REQ: (time.perf_counter() + 30.0, 20)}
    connector.bind_connector_metadata(_recv_meta(20.0))
    ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    connector.start_load_kv(ctx)
    assert _KEY not in worker._remote_leases  # popped when the read was decided
    worker._send_heartbeats(_hb_meta(_REMOTE_REQ, leases=leases))
    assert _KEY in worker._remote_leases
    result = worker.get_transfer_results()
    assert "req" in result.finished_recving
    assert _KEY not in worker._remote_leases


# ---------------------------------------------------------------------------
# Scheduler side: what the producer exports and what D ships with heartbeats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kv_lease_duration", [30, 12])
def test_prefill_request_finished_exports_lease_deadline(kv_lease_duration):
    """P exports the lease deadline (its perf_counter) and how much each
    heartbeat renews it; the consumer-side rebasing is covered above."""
    vllm_config = create_vllm_config(
        kv_connector_extra_config={"kv_lease_duration": kv_lease_duration}
    )
    scheduler = create_scheduler(vllm_config)
    block_size = vllm_config.cache_config.block_size
    request = create_request(
        request_id=7,
        num_tokens=block_size * 2,
        block_size=block_size,
        do_remote_decode=True,
    )
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    before = time.perf_counter()
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, create_model_runner_output([request])
    )
    after = time.perf_counter()
    kv = engine_core_outputs[0].outputs[0].kv_transfer_params
    assert kv is not None and kv["do_remote_prefill"] is True
    expiry = kv["remote_blocks_expiry_time"]
    assert isinstance(expiry, float)
    assert before + kv_lease_duration <= expiry <= after + kv_lease_duration
    assert kv["remote_lease_extension"] == kv_lease_extension(kv_lease_duration)


def test_decode_scheduler_ships_the_lease_with_heartbeats():
    """on_new_request carries the producer's deadline and renewal amount into
    the engine's HeartbeatInfo, and _stop_heartbeat drops them."""
    scheduler = make_nixl_scheduler(heartbeat=True)
    req = create_request(request_id=1, do_remote_prefill=True)
    req.kv_transfer_params["remote_blocks_expiry_time"] = 1234.5
    req.kv_transfer_params["remote_lease_extension"] = 20
    bare = create_request(request_id=2, do_remote_prefill=True)
    bare.kv_transfer_params.pop("remote_blocks_expiry_time", None)
    bare.kv_transfer_params.pop("remote_lease_extension", None)
    scheduler.on_new_request(req)
    scheduler.on_new_request(bare)
    engine_id = req.kv_transfer_params["remote_engine_id"]
    info = scheduler._heartbeat_by_engine[engine_id]
    assert info.req_ids == {"prefill-1", "prefill-2"}
    assert info.leases == {"prefill-1": (1234.5, 20)}
    scheduler._stop_heartbeat(req.request_id)
    assert info.req_ids == {"prefill-2"}
    assert info.leases == {}


@pytest.mark.parametrize(
    "msg",
    [
        LookupPinRequest(
            block_hashes=[b"\x01" * 32, b"\x02" * 32],
            identity=PeerKVIdentity(
                "abc123", 16, hash_algo="sha256", cache_salt="salt"
            ),
            lease_duration=30.0,
        ),
        LookupPinReply(
            hit_tokens=32,
            remote_block_ids=[[3, 4], [5, 6]],
            remote_request_id="pin-1",
            miss_reason=PeerMissReason.NONE,
            remote_blocks_expiry_time=1234.5,
            remote_lease_extension=20,
        ),
        LookupPinReply(
            hit_tokens=0,
            remote_block_ids=[],
            remote_request_id="pin-2",
            miss_reason=PeerMissReason.IDENTITY_MISMATCH,
        ),
    ],
)
def test_lookup_pin_wire_roundtrip(msg):
    """Lookup-and-pin messages survive the side channel's msgpack encoding."""
    decoded = msgspec.msgpack.Decoder(type(msg)).decode(msgspec.msgpack.encode(msg))
    assert decoded == msg
