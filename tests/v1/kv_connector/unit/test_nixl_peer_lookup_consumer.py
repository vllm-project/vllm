# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL peer prefix lookup, consumer side: the pull scheduler's lookup flow."""

import time
from types import SimpleNamespace

import pytest

from tests.v1.core.test_prefix_pin import (
    BLOCK_SIZE,
    compute_and_free,
    make_manager,
    make_request,
    tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    PeerEndpoint,
    PeerLookupStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    LookupPinReply,
    LookupPinRequest,
    NixlConnectorMetadata,
    PeerMissReason,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.peer_lookup_client import (
    LookupResult,
    PeerLookupClient,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput

from .test_nixl_peer_lookup_listener import cycle, served, wait_for  # noqa: F401
from .test_nixl_peer_pins import (
    COMPAT,
    _init_hash,  # noqa: F401  (autouse fixture)
    make_wired_scheduler,
)

pytestmark = pytest.mark.cpu_test

HOST = "10.0.0.7"
PORT = 5601
ENGINE = "engine-a"
PIN = "peer-pin-abc"
THRESHOLD = 2 * BLOCK_SIZE


class FakeLookupClient:
    """The PeerLookupClient interface, scripted by the test."""

    def __init__(self) -> None:
        self.submits: list[tuple[str, str, int, LookupPinRequest]] = []
        self.live: set[str] = set()
        self.results: dict[str, LookupResult] = {}
        self.cancelled: list[str] = []
        self.orphans: list[tuple[str, int, LookupPinReply]] = []
        self.polls = 0
        self.engine_ids: list[str | None] = []

    max_latency_s = 4.0

    def submit(self, key, host, port, request, engine_id=None) -> None:
        self.submits.append((key, host, port, request))
        self.engine_ids.append(engine_id)
        self.live.add(key)

    def poll(self, key):
        self.polls += 1
        result = self.results.pop(key, None)
        if result is not None:
            self.live.discard(key)
        return result

    def cancel(self, key) -> None:
        self.cancelled.append(key)
        self.live.discard(key)
        self.results.pop(key, None)

    def take_orphaned_pins(self):
        orphans, self.orphans = self.orphans, []
        return orphans

    def has_in_flight(self) -> bool:
        return len(self.live) > len(self.results)

    def close(self) -> None:
        pass


def make_consumer(client=None, backoff: int = 0, threshold: int = THRESHOLD):
    sched = make_wired_scheduler(None)
    sched._compatibility_hash = COMPAT
    sched.kv_recompute_threshold = threshold
    sched._peer_lookup_timeout = 2.0
    sched.peer_lookup_stats = PeerLookupStats()
    sched._peer_endpoints = {}
    sched._lookup_client = client if client is not None else FakeLookupClient()
    if backoff:
        sched.vllm_config.num_reprefillable_tokens = backoff
    return sched


def peer_request(
    num_blocks: int = 4,
    extra: int = 5,
    rid: str = "r1",
    start: int = 0,
    **params,
):
    request = make_request(rid, tokens(num_blocks, start) + [999] * extra)
    request.kv_transfer_params = {
        "do_remote_prefill": True,
        "peer_lookup": True,
        "remote_engine_id": ENGINE,
        "remote_host": HOST,
        "remote_port": PORT,
        "tp_size": 1,
        **params,
    }
    return request


def hit_reply(num_blocks: int, pin: str = PIN, first_id: int = 100) -> LookupPinReply:
    return LookupPinReply(
        hit_tokens=num_blocks * BLOCK_SIZE,
        remote_block_ids=[list(range(first_id, first_id + num_blocks))],
        remote_request_id=pin,
        miss_reason=PeerMissReason.NONE,
        remote_blocks_expiry_time=time.perf_counter() + 30,
        remote_lease_extension=5,
    )


def miss(reason=PeerMissReason.NOT_CACHED) -> LookupPinReply:
    return LookupPinReply(0, [], "", reason)


def answer(sched, request, reply=None, error=None) -> None:
    sched._lookup_client.results[request.request_id] = LookupResult(reply, error)


def start_lookup(sched, request):
    assert sched.get_num_new_matched_tokens(request, 0) == (None, False)


def finish_hit(sched, request, num_blocks: int, num_computed: int = 0):
    start_lookup(sched, request)
    answer(sched, request, hit_reply(num_blocks))
    return sched.get_num_new_matched_tokens(request, num_computed)


def recv_entry(sched, request):
    return sched._reqs_need_recv[request.request_id]


# ---- lookup flow -----------------------------------------------------------


def test_submit_once_then_none_while_pending():
    sched = make_consumer()
    client = sched._lookup_client
    request = peer_request(cache_salt=None, kv_identity={"ignored": 1})
    request.cache_salt = "salty"

    for _ in range(4):
        assert sched.get_num_new_matched_tokens(request, 0) == (None, False)

    assert len(client.submits) == 1
    key, host, port, sent = client.submits[0]
    assert (key, host, port) == (request.request_id, HOST, PORT)
    # The request's own hashes, full prompt blocks only (not the 5 extra tokens).
    assert sent.block_hashes == [bytes(h) for h in request.block_hashes[:4]]
    assert len(sent.block_hashes) == 4
    assert sent.identity.compatibility_hash == COMPAT
    assert sent.identity.block_size == BLOCK_SIZE
    assert sent.identity.hash_algo == "sha256"
    assert sent.identity.cache_salt == "salty"
    assert sent.lease_duration == sched._kv_lease_duration
    # A fresh correlation id per lookup, and the engine id for the probe cache.
    assert sent.lookup_id
    assert client.engine_ids == [ENGINE]
    assert sched.peer_lookup_stats.lookups == 1
    assert request.kv_transfer_params["peer_lookup"] is True
    # An idle engine must keep stepping while the lookup is out.
    assert sched.has_pending_push_work()


def test_router_block_hashes_are_ignored():
    sched = make_consumer()
    request = peer_request(remote_block_hashes=[b"wrong"])
    start_lookup(sched, request)
    sent = sched._lookup_client.submits[0][3]
    assert b"wrong" not in sent.block_hashes
    assert sent.block_hashes == [bytes(h) for h in request.block_hashes[:4]]


@pytest.mark.parametrize(
    ("reply", "error", "reason"),
    [
        (miss(PeerMissReason.NOT_CACHED), None, "not_cached"),
        (miss(PeerMissReason.IDENTITY_MISMATCH), None, "identity_mismatch"),
        (None, "timeout", "timeout"),
        (None, "send_failed", "send_failed"),
        (None, "bad_reply", "bad_reply"),
        # The peer's probe said it serves no lookups (an older or disabled
        # engine): counted apart from every other miss.
        (None, "peer_unsupported", "peer_unsupported"),
        (None, "lookup_id_mismatch", "lookup_id_mismatch"),
        (None, "peer_engine_mismatch", "peer_engine_mismatch"),
        (None, "probe_timeout", "probe_timeout"),
    ],
)
def test_miss_computes_locally(reply, error, reason):
    sched = make_consumer()
    request = peer_request()
    start_lookup(sched, request)
    answer(sched, request, reply, error)

    assert sched.get_num_new_matched_tokens(request, 0) == (0, False)

    params = request.kv_transfer_params
    assert params["peer_lookup"] is False
    assert params["do_remote_prefill"] is False
    assert params["peer_miss_reason"] == reason
    assert sched.peer_lookup_stats.misses == {reason: 1}
    # Later calls (and a preempted retry) take the stock "no remote" path.
    assert sched.get_num_new_matched_tokens(request, 0) == (0, False)
    sched.update_state_after_alloc(request, KVCacheBlocks(blocks=((),)), 0)
    assert sched._reqs_need_recv == {}
    assert len(sched._lookup_client.submits) == 1


def test_stalled_client_times_out(monkeypatch):
    sched = make_consumer()
    request = peer_request()
    start_lookup(sched, request)
    params = request.kv_transfer_params
    assert sched.get_num_new_matched_tokens(request, 0) == (None, False)
    params["_peer_lookup_deadline"] = time.monotonic() - 1
    assert sched.get_num_new_matched_tokens(request, 0) == (0, False)
    assert sched._lookup_client.cancelled == [request.request_id]
    assert sched.peer_lookup_stats.misses == {"timeout": 1}


def test_full_hit():
    sched = make_consumer()
    request = peer_request()
    assert finish_hit(sched, request, 4) == (4 * BLOCK_SIZE, True)

    params = request.kv_transfer_params
    assert params["do_remote_prefill"] is True
    assert params["peer_lookup"] is False
    assert params["_peer_hit"] is True
    assert params["remote_block_ids"] == [[100, 101, 102, 103]]
    assert params["remote_request_id"] == PIN
    assert params["remote_num_tokens"] == 4 * BLOCK_SIZE
    assert params["remote_blocks_expiry_time"] is not None
    assert params["remote_lease_extension"] == 5
    assert sched.peer_lookup_stats.full_hits == 1
    # The pin is heartbeated like a finished prefill.
    info = sched._heartbeat_by_engine[ENGINE]
    assert info.req_ids == {PIN}
    assert (info.host, info.port, info.tp_size) == (HOST, PORT, 1)
    assert PIN in info.leases

    # Idempotent: the scheduler may call again before the request is allocated.
    polls = sched._lookup_client.polls
    assert sched.get_num_new_matched_tokens(request, 0) == (4 * BLOCK_SIZE, True)
    assert sched._lookup_client.polls == polls
    assert len(sched._lookup_client.submits) == 1
    assert sched.peer_lookup_stats.full_hits == 1


def test_hit_without_tp_size_skips_heartbeat():
    sched = make_consumer()
    request = peer_request()
    del request.kv_transfer_params["tp_size"]
    assert finish_hit(sched, request, 4) == (4 * BLOCK_SIZE, True)
    assert sched._heartbeat_by_engine == {}


def test_partial_hit():
    sched = make_consumer()
    request = peer_request()
    assert finish_hit(sched, request, 2) == (2 * BLOCK_SIZE, True)
    assert request.kv_transfer_params["remote_num_tokens"] == 2 * BLOCK_SIZE
    assert sched.peer_lookup_stats.partial_hits == 1
    assert sched.peer_lookup_stats.full_hits == 0


def test_hit_below_threshold_only_releases_the_pin():
    sched = make_consumer()
    request = peer_request()
    assert finish_hit(sched, request, 1) == (0, False)
    # Stays a remote prefill so that update_state_after_alloc frees the pin.
    assert request.kv_transfer_params["do_remote_prefill"] is True

    sched.update_state_after_alloc(request, KVCacheBlocks(blocks=((),)), 0)
    _, local_ids, cached, awaiting = recv_entry(sched, request)
    assert not any(local_ids) and awaiting is False
    assert sched.peer_lookup_stats.notify_only_releases == 1
    assert sched._heartbeat_by_engine == {}

    meta = sched.build_connector_meta(SchedulerOutput.make_empty())
    recv = meta.reqs_to_recv[request.request_id]
    assert recv.remote.request_id == PIN
    assert recv.remote.engine_id == ENGINE
    assert recv.awaiting_kvs is False
    assert not any(recv.local_block_ids)


def test_local_hit_covering_the_peer_hit_only_releases_the_pin():
    sched = make_consumer()
    request = peer_request()
    # A pinned 2 blocks; B already has 3 in its own cache.
    assert finish_hit(sched, request, 2, num_computed=3 * BLOCK_SIZE) == (0, False)

    sched.update_state_after_alloc(request, KVCacheBlocks(blocks=((),)), 0)
    assert not any(recv_entry(sched, request)[1])
    meta = sched.build_connector_meta(SchedulerOutput.make_empty())
    assert meta.reqs_to_recv[request.request_id].remote.request_id == PIN


@pytest.mark.parametrize(
    "case",
    [
        "no_hash",
        "no_host",
        "no_port",
        "no_engine",
        "short",
        "block_size",
        "groups",
        "no_remote_prefill",
        "remote_prefill_false",
    ],
)
def test_local_precondition_miss_sends_nothing(case):
    sched = make_consumer()
    request = (
        peer_request(num_blocks=0, extra=10) if case == "short" else peer_request()
    )
    params = request.kv_transfer_params
    if case == "no_hash":
        sched._compatibility_hash = None
    elif case == "no_host":
        del params["remote_host"]
    elif case == "no_port":
        del params["remote_port"]
    elif case == "no_engine":
        del params["remote_engine_id"]
    elif case == "no_remote_prefill":
        del params["do_remote_prefill"]
    elif case == "remote_prefill_false":
        params["do_remote_prefill"] = False
    elif case == "block_size":
        sched.kv_cache_config.hash_block_size = BLOCK_SIZE * 2
    elif case == "groups":
        sched.kv_cache_config.kv_cache_groups.append(
            sched.kv_cache_config.kv_cache_groups[0]
        )

    assert sched.get_num_new_matched_tokens(request, 0) == (0, False)

    assert sched._lookup_client.submits == []
    assert params["do_remote_prefill"] is False and params["peer_lookup"] is False
    assert sched.peer_lookup_stats.misses == {"local_precondition": 1}
    assert sched.peer_lookup_stats.lookups == 0


def test_unseeded_process_hash_is_a_local_miss(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    sched = make_consumer()
    sched.vllm_config.cache_config.prefix_caching_hash_algo = "xxhash"
    request = peer_request()
    assert sched.get_num_new_matched_tokens(request, 0) == (0, False)
    assert sched._lookup_client.submits == []
    assert sched.peer_lookup_stats.misses == {"local_precondition": 1}


def test_request_that_skips_external_kv_is_a_local_miss():
    sched = make_consumer()
    request = peer_request()
    request.prefix_cache_hit_limit = 0
    assert sched.get_num_new_matched_tokens(request, 0) == (0, False)
    assert sched._lookup_client.submits == []


@pytest.mark.parametrize(
    "reply",
    [
        # Wrong block count for the hit.
        LookupPinReply(32, [[100]], PIN, PeerMissReason.NONE),
        # More blocks than we asked about.
        LookupPinReply(5 * BLOCK_SIZE, [list(range(5))], PIN, PeerMissReason.NONE),
        # Not whole blocks.
        LookupPinReply(40, [[100, 101, 102]], PIN, PeerMissReason.NONE),
        # More than one group.
        LookupPinReply(16, [[100], [101]], PIN, PeerMissReason.NONE),
        # No pin id.
        LookupPinReply(16, [[100]], "", PeerMissReason.NONE),
    ],
)
def test_inconsistent_reply_is_not_loaded(reply):
    sched = make_consumer()
    request = peer_request()
    start_lookup(sched, request)
    answer(sched, request, reply)
    assert sched.get_num_new_matched_tokens(request, 0) == (0, False)
    assert request.kv_transfer_params["do_remote_prefill"] is False
    assert sched.peer_lookup_stats.misses == {"bad_reply": 1}
    assert sched._heartbeat_by_engine == {}


# ---- abort and orphans -------------------------------------------------------


def test_abort_before_reply_cancels_without_a_recv():
    sched = make_consumer()
    request = peer_request()
    start_lookup(sched, request)

    assert sched.request_finished(request, ((),)) == (False, None)

    assert sched._lookup_client.cancelled == [request.request_id]
    assert sched._reqs_need_recv == {}
    params = request.kv_transfer_params
    assert params["peer_lookup"] is False and params["do_remote_prefill"] is False
    assert sched.peer_lookup_stats.cancelled == 1
    sched.build_connector_meta(
        SchedulerOutput.make_empty()
    )  # must not trip on the request


def test_abort_before_the_lookup_was_submitted():
    sched = make_consumer()
    request = peer_request()
    assert sched.request_finished(request, ((),)) == (False, None)
    assert sched._reqs_need_recv == {}
    meta = sched.build_connector_meta(SchedulerOutput.make_empty())
    assert meta.reqs_to_recv == {}


def test_abort_without_a_client_does_not_create_one():
    sched = make_consumer()
    sched._lookup_client = None
    request = peer_request()
    assert sched.request_finished(request, ((),)) == (False, None)
    assert sched._lookup_client is None


def test_abort_after_hit_before_alloc_releases_the_pin():
    sched = make_consumer()
    request = peer_request()
    assert finish_hit(sched, request, 4) == (4 * BLOCK_SIZE, True)

    assert sched.request_finished(request, ((),)) == (False, None)

    assert sched._heartbeat_by_engine == {}
    _, local_ids, _, awaiting = recv_entry(sched, request)
    assert not any(local_ids) and awaiting is False
    assert sched.peer_lookup_stats.notify_only_releases == 1
    meta = sched.build_connector_meta(SchedulerOutput.make_empty())
    recv = meta.reqs_to_recv[request.request_id]
    assert recv.remote.request_id == PIN
    assert recv.remote.block_ids == [[100, 101, 102, 103]]


def test_orphaned_pin_is_released_with_a_notify_only_recv():
    sched = make_consumer()
    request = peer_request(dcp_size=2, pp_size=3)
    start_lookup(sched, request)
    sched.request_finished(request, ((),))
    sched._lookup_client.orphans.append((HOST, PORT, hit_reply(3, pin="peer-pin-o")))

    meta = sched.build_connector_meta(SchedulerOutput.make_empty())

    assert isinstance(meta, NixlConnectorMetadata)
    (req_id, recv) = next(iter(meta.reqs_to_recv.items()))
    assert len(meta.reqs_to_recv) == 1
    assert req_id.startswith("peer-release-")
    assert recv.awaiting_kvs is False
    assert not any(recv.local_block_ids)
    assert recv.remote.request_id == "peer-pin-o"
    assert recv.remote.engine_id == ENGINE
    assert (recv.remote.host, recv.remote.port) == (HOST, PORT)
    assert recv.remote.block_ids == [[100, 101, 102]]
    assert (recv.tp_size, recv.dcp_size, recv.pp_size) == (1, 2, 3)
    assert sched.peer_lookup_stats.orphans_released == 1
    # Drained: not released twice.
    assert sched.build_connector_meta(SchedulerOutput.make_empty()).reqs_to_recv == {}


@pytest.mark.parametrize("endpoint", [None, PeerEndpoint(ENGINE, None)])
def test_orphan_without_known_peer_topology_is_left_to_its_lease(endpoint):
    sched = make_consumer()
    if endpoint is not None:
        sched._peer_endpoints[(HOST, PORT)] = endpoint
    sched._lookup_client.orphans.append((HOST, PORT, hit_reply(2, pin="peer-pin-o")))
    meta = sched.build_connector_meta(SchedulerOutput.make_empty())
    assert meta.reqs_to_recv == {}
    assert sched.peer_lookup_stats.orphans_released == 0
    assert sched.peer_lookup_stats.orphans_unreleased == 1


def test_orphan_release_does_not_touch_a_missing_client():
    sched = make_consumer()
    sched._lookup_client = None
    assert sched.build_connector_meta(SchedulerOutput.make_empty()).reqs_to_recv == {}
    assert not sched.has_pending_push_work()


# ---- block alignment -----------------------------------------------------------


def pair(sched, request, local_blocks: int, hit_blocks: int):
    """Drive a peer hit through allocation on a real cache manager.

    Returns (local ids loaded into, remote ids they pair with, remote ids kept,
    external tokens). The pairing is the worker's own _apply_prefix_caching.
    """
    manager = make_manager()
    if local_blocks:
        seed = make_request("seed", tokens(local_blocks))
        compute_and_free(manager, seed)
    computed, num_local, _ = manager.get_computed_blocks(request)
    assert num_local == local_blocks * BLOCK_SIZE

    count, async_load = finish_hit(sched, request, hit_blocks, num_computed=num_local)
    assert async_load and count > 0
    new = manager.allocate_slots(
        request,
        0,
        num_new_computed_tokens=num_local,
        new_computed_blocks=computed,
        num_external_computed_tokens=count,
        delay_cache_blocks=True,
    )
    assert new is not None
    blocks = manager.get_blocks(request.request_id)
    sched.update_state_after_alloc(request, blocks, count)

    _, local_ids, cached, awaiting = recv_entry(sched, request)
    assert awaiting is True
    assert cached == (local_blocks,)
    params = request.kv_transfer_params
    worker = SimpleNamespace(_has_mamba=False)
    paired_local, paired_remote = NixlBaseConnectorWorker._apply_prefix_caching(
        worker,
        decode_block_ids=local_ids,
        prefill_block_ids=params["remote_block_ids"],
        decode_physical_per_logical=1,
        prefill_physical_per_logical=1,
    )
    return paired_local[0], paired_remote[0], params, count


def test_pairing_full_hit():
    sched = make_consumer()
    request = peer_request(num_blocks=4, extra=5)
    local, remote, params, count = pair(sched, request, 0, 4)
    assert count == 4 * BLOCK_SIZE
    assert remote == [100, 101, 102, 103]
    assert len(local) == 4
    assert params["remote_block_ids"] == [[100, 101, 102, 103]]
    assert params["remote_num_tokens"] == 4 * BLOCK_SIZE


def test_pairing_partial_hit():
    sched = make_consumer()
    request = peer_request(num_blocks=4, extra=5)
    local, remote, params, count = pair(sched, request, 0, 2)
    assert count == 2 * BLOCK_SIZE
    assert remote == [100, 101]
    assert len(local) == 2
    assert params["remote_num_tokens"] == 2 * BLOCK_SIZE


def test_pairing_when_the_peer_hit_is_longer_than_the_load():
    # 20 trailing prompt tokens are recomputed here, so only the first 28 are
    # loaded: 2 blocks. A pinned 3. Without the cut the 2 local blocks would
    # pair with the tail of A's list, [101, 102].
    sched = make_consumer(backoff=20, threshold=0)
    request = peer_request(num_blocks=3, extra=0)
    local, remote, params, count = pair(sched, request, 0, 3)
    assert count == 3 * BLOCK_SIZE - 20
    assert len(local) == 2
    assert remote == [100, 101]
    assert params["remote_block_ids"] == [[100, 101]]
    assert params["remote_num_tokens"] == count


def test_pairing_with_a_local_prefix_hit():
    sched = make_consumer()
    request = peer_request(num_blocks=4, extra=5)
    local, remote, params, count = pair(sched, request, 1, 4)
    assert count == 3 * BLOCK_SIZE
    assert len(local) == 3
    # A's block 0 is covered by B's own cache.
    assert remote == [101, 102, 103]
    assert params["remote_block_ids"] == [[100, 101, 102, 103]]
    assert params["remote_num_tokens"] == 4 * BLOCK_SIZE


def test_pairing_with_a_local_hit_and_a_longer_peer_hit():
    sched = make_consumer(backoff=20, threshold=0)
    request = peer_request(num_blocks=4, extra=0)
    local, remote, params, count = pair(sched, request, 1, 4)
    # Load ends at token 44: B has block 0, blocks 1 and 2 are loaded.
    assert count == 4 * BLOCK_SIZE - 20 - BLOCK_SIZE
    assert len(local) == 2
    assert remote == [101, 102]
    assert params["remote_block_ids"] == [[100, 101, 102]]
    assert params["remote_num_tokens"] == 44


def test_scheduler_trimming_the_load_is_followed():
    # The core scheduler can shrink the external count (eg replay alignment).
    sched = make_consumer()
    request = peer_request(num_blocks=4, extra=5)
    assert finish_hit(sched, request, 4) == (4 * BLOCK_SIZE, True)
    manager = make_manager()
    computed, _, _ = manager.get_computed_blocks(request)
    ext = 2 * BLOCK_SIZE
    manager.allocate_slots(
        request,
        0,
        new_computed_blocks=computed,
        num_external_computed_tokens=ext,
        delay_cache_blocks=True,
    )
    sched.update_state_after_alloc(request, manager.get_blocks(request.request_id), ext)
    assert request.kv_transfer_params["remote_block_ids"] == [[100, 101]]


def test_realloc_after_preemption_computes_locally():
    sched = make_consumer()
    request = peer_request(num_blocks=4, extra=5)
    pair(sched, request, 0, 4)
    params = request.kv_transfer_params
    assert params["do_remote_prefill"] is False
    # Rescheduled from scratch: no second lookup, no second pull.
    assert sched.get_num_new_matched_tokens(request, 0) == (0, False)
    sched.update_state_after_alloc(request, KVCacheBlocks(blocks=((),)), 0)
    assert len(sched._lookup_client.submits) == 1
    assert params["remote_block_ids"] == [[100, 101, 102, 103]]


# ---- against a real producer -----------------------------------------------------


def test_real_client_against_a_real_producer(served):  # noqa: F811
    a_sched, a_connector, a_manager, hashes, _ = served
    b = make_consumer(client=PeerLookupClient(timeout_s=10.0))
    try:
        request = make_request("b-req", tokens(3) + [999] * 5)
        assert [bytes(h) for h in request.block_hashes[:3]] == hashes
        request.kv_transfer_params = {
            "do_remote_prefill": True,
            "peer_lookup": True,
            "remote_engine_id": "producer-engine",
            "remote_host": a_sched.side_channel_host,
            "remote_port": a_sched.side_channel_port,
            "tp_size": 1,
        }
        assert b.get_num_new_matched_tokens(request, 0) == (None, False)
        wait_for(lambda: a_connector.has_pending_push_work())
        cycle(a_connector)  # A fences the pin to this step
        a_meta = cycle(a_connector)  # ...and ships it, with the reply
        (pin_id,) = a_meta.reqs_to_send

        def answered():
            return b.get_num_new_matched_tokens(request, 0) != (None, False)

        wait_for(answered)

        # B probed A first (A advertises lookups), and the reply echoed B's id.
        verdict = b._lookup_client._caps[
            (a_sched.side_channel_host, a_sched.side_channel_port)
        ]
        assert verdict.error is None
        assert b.get_num_new_matched_tokens(request, 0) == (3 * BLOCK_SIZE, True)
        params = request.kv_transfer_params
        pin = next(iter(a_manager._pins))
        assert params["remote_request_id"] == pin_id
        assert params["remote_block_ids"] == [list(pin.get_block_ids()[0])]
        assert params["remote_num_tokens"] == 3 * BLOCK_SIZE
        assert params["_peer_hit"] is True
        assert b._heartbeat_by_engine["producer-engine"].req_ids == {pin_id}
        assert b.peer_lookup_stats.full_hits == 1
        assert not b.has_pending_push_work()
    finally:
        b._lookup_client.close()
