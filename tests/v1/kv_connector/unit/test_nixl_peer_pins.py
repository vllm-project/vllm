# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL peer prefix lookup-and-pin: the producer-side PeerPinServer."""

import threading
from types import SimpleNamespace

import pytest

from tests.v1.core.test_prefix_pin import (
    BLOCK_SIZE,
    NUM_BLOCKS,
    compute_and_free,
    make_manager,
    make_request,
    tokens,
)
from vllm.config import CacheConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    PeerLookupChannel,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    LookupPinRequest,
    PeerKVIdentity,
    PeerMissReason,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.peer_pins import (
    PEER_PIN_ID_PREFIX,
    PeerPinServer,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    kv_lease_extension,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash

pytestmark = pytest.mark.cpu_test

COMPAT = "compat-hash"
LEASE = 30.0
USABLE = NUM_BLOCKS - 1


@pytest.fixture(autouse=True)
def _init_hash(monkeypatch):
    init_none_hash(sha256)
    monkeypatch.setenv("PYTHONHASHSEED", "0")


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_server(
    manager: KVCacheManager,
    max_fraction: float = 1.0,
    min_free_fraction: float = 0.0,
    hash_algo: str = "sha256",
    clock: Clock | None = None,
) -> PeerPinServer:
    server = PeerPinServer(
        manager,
        block_size=BLOCK_SIZE,
        hash_algo=hash_algo,
        lease_duration=LEASE,
        max_pinned_fraction=max_fraction,
        min_free_fraction=min_free_fraction,
        clock=clock or Clock(),
    )
    server.set_compatibility_hash(COMPAT)
    return server


def lookup(
    hashes,
    compat: str = COMPAT,
    block_size: int = BLOCK_SIZE,
    hash_algo: str = "sha256",
    lease: float | None = None,
) -> LookupPinRequest:
    return LookupPinRequest(
        block_hashes=list(hashes),
        identity=PeerKVIdentity(compat, block_size, hash_algo),
        lease_duration=lease,
    )


def cached_manager(num_blocks: int = 3) -> tuple[KVCacheManager, list[bytes]]:
    manager = make_manager()
    request = make_request("a", tokens(num_blocks))
    compute_and_free(manager, request)
    return manager, list(request.block_hashes)


def step(server: PeerPinServer) -> dict[str, float]:
    """One schedule + process cycle; returns what the schedule shipped."""
    server.on_step_scheduled()
    sends = server.take_new_sends()
    server.on_step_processed()
    return sends


def ship(server: PeerPinServer) -> dict[str, float]:
    """Sync-style steps until the pin ships: the step scheduled after the lookup
    is processed, and the next schedule then ships it (one extra step)."""
    assert step(server) == {}
    sends = step(server)
    assert sends
    return sends


def only_reply(server: PeerPinServer):
    replies = server.take_ready_replies()
    assert len(replies) == 1
    return replies[0]


def test_full_hit_is_held_until_the_fence_async():
    manager, hashes = cached_manager()
    clock = Clock()
    server = make_server(manager, clock=clock)
    free_before = manager.block_pool.get_num_free_blocks()

    server.serve("tok", lookup(hashes))

    # Pinned right away, but not exposed and no reply yet.
    assert manager.num_pinned_blocks == 3
    assert manager.block_pool.get_num_free_blocks() == free_before - 3
    assert server.num_pending == 1
    assert server.take_ready_replies() == []

    # Async scheduling: two steps scheduled before either is processed.
    server.on_step_scheduled()
    assert server.take_new_sends() == {}
    server.on_step_scheduled()
    assert server.take_new_sends() == {}
    assert server.take_ready_replies() == []

    # The fence is the first step scheduled after the lookup.
    server.on_step_processed()
    sends = server.take_new_sends()
    assert len(sends) == 1
    ((pin_id, expiry),) = sends.items()
    assert expiry == clock.now + LEASE
    assert server.num_pending == 0 and server.num_shipped == 1

    token, reply = only_reply(server)
    assert token == "tok"
    assert reply.remote_request_id == pin_id
    assert reply.remote_blocks_expiry_time == expiry
    assert reply.remote_lease_extension == kv_lease_extension(int(LEASE))
    assert reply.hit_tokens == 3 * BLOCK_SIZE
    assert reply.miss_reason is PeerMissReason.NONE
    pin = next(iter(manager._pins))
    assert reply.remote_block_ids == pin.get_block_ids()
    assert all(b.ref_cnt == 1 for b in pin.blocks[0])


def test_sync_style_one_step_ships():
    manager, hashes = cached_manager()
    server = make_server(manager)
    server.serve("tok", lookup(hashes))
    ship(server)
    assert only_reply(server)[1].hit_tokens == 3 * BLOCK_SIZE


def test_serve_inside_schedule_waits_for_the_next_step():
    manager, hashes = cached_manager()
    server = make_server(manager)
    server.on_step_scheduled()  # schedule() running; build_connector_meta ran
    server.serve("tok", lookup(hashes))
    server.on_step_processed()  # the step scheduled before the lookup
    assert server.take_new_sends() == {}
    ship(server)


def test_miss_replies_immediately_and_holds_nothing():
    manager = make_manager()
    server = make_server(manager)
    free_before = manager.block_pool.get_num_free_blocks()
    server.serve("tok", lookup(make_request("x", tokens(2)).block_hashes))

    _, reply = only_reply(server)
    assert reply.miss_reason is PeerMissReason.NOT_CACHED
    assert reply.hit_tokens == 0
    assert reply.remote_block_ids == []
    assert reply.remote_request_id == ""
    assert reply.remote_blocks_expiry_time is None
    assert server.num_pending == 0
    assert manager.num_pinned_blocks == 0
    assert manager.block_pool.get_num_free_blocks() == free_before
    assert not server.has_pending_work()
    assert step(server) == {}


def test_evicted_miss_reason():
    manager, hashes = cached_manager(2)
    server = make_server(manager)
    # Churn the pool so the cached blocks are evicted.
    for i in range(USABLE):
        compute_and_free(manager, make_request(f"f{i}", tokens(1, start=100 + i)))
    server.serve("tok", lookup(hashes))
    assert only_reply(server)[1].miss_reason is PeerMissReason.EVICTED


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(compat="other"),
        dict(block_size=BLOCK_SIZE * 2),
        dict(hash_algo="xxhash"),
    ],
)
def test_identity_mismatch_refuses_without_pinning(kwargs):
    manager, hashes = cached_manager()
    server = make_server(manager)
    server.serve("tok", lookup(hashes, **kwargs))
    _, reply = only_reply(server)
    assert reply.miss_reason is PeerMissReason.IDENTITY_MISMATCH
    assert reply.hit_tokens == 0 and reply.remote_request_id == ""
    assert manager.num_pinned_blocks == 0
    assert server.num_pending == 0


def test_unknown_compat_hash_refuses():
    manager, hashes = cached_manager()
    server = PeerPinServer(manager, BLOCK_SIZE, "sha256", LEASE, 1.0, 0.0)
    server.serve("tok", lookup(hashes))
    assert only_reply(server)[1].miss_reason is PeerMissReason.IDENTITY_MISMATCH
    assert manager.num_pinned_blocks == 0


@pytest.mark.parametrize("algo", ["xxhash", "xxhash_cbor"])
def test_unseeded_process_hash_refuses(monkeypatch, algo):
    manager, hashes = cached_manager()
    server = make_server(manager, hash_algo=algo)
    monkeypatch.delenv("PYTHONHASHSEED")
    server.serve("tok", lookup(hashes, hash_algo=algo))
    assert only_reply(server)[1].miss_reason is PeerMissReason.IDENTITY_MISMATCH
    assert manager.num_pinned_blocks == 0
    # With a seed the same hashes are acceptable (manager hashes use sha256 here,
    # so this only checks that the identity gate passes).
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    server.serve("tok", lookup(hashes, hash_algo=algo))
    assert server.num_pending == 1


def test_unsupported_layout_maps_to_identity_mismatch():
    manager, hashes = cached_manager()
    manager._supports_prefix_pinning = False
    server = make_server(manager)
    server.serve("tok", lookup(hashes))
    assert only_reply(server)[1].miss_reason is PeerMissReason.IDENTITY_MISMATCH


def test_budget_exhausted_is_pin_refused():
    manager, hashes = cached_manager()
    # 3 of 10 blocks may be pinned.
    server = make_server(manager, max_fraction=0.3)
    server.serve("t1", lookup(hashes))
    assert manager.num_pinned_blocks == 3
    server.serve("t2", lookup(hashes))
    replies = server.take_ready_replies()
    assert [(t, r.miss_reason) for t, r in replies] == [
        ("t2", PeerMissReason.PIN_REFUSED)
    ]
    assert manager.num_pinned_blocks == 3


def test_partial_budget_caps_the_pin():
    manager, hashes = cached_manager(3)
    server = make_server(manager, max_fraction=0.2)  # 2 blocks
    server.serve("tok", lookup(hashes))
    ship(server)
    _, reply = only_reply(server)
    assert reply.hit_tokens == 2 * BLOCK_SIZE
    assert len(reply.remote_block_ids[0]) == 2
    assert reply.miss_reason is PeerMissReason.PIN_REFUSED


def test_min_free_blocks_refuses():
    manager, hashes = cached_manager(3)
    # Pinning 3 free-queue blocks would leave fewer than 10 free.
    server = make_server(manager, min_free_fraction=1.0)
    server.serve("tok", lookup(hashes))
    assert only_reply(server)[1].miss_reason is PeerMissReason.PIN_REFUSED
    assert manager.num_pinned_blocks == 0


def test_on_finished_sending_strips_only_pin_ids_and_releases():
    manager, hashes = cached_manager()
    server = make_server(manager)
    free_before = manager.block_pool.get_num_free_blocks()
    server.serve("tok", lookup(hashes))
    (pin_id,) = ship(server)
    pin = next(iter(manager._pins))
    assert manager.block_pool.get_num_free_blocks() == free_before - 3

    rest = server.on_finished_sending({pin_id, "req-1", "req-2"})

    assert rest == {"req-1", "req-2"}
    assert server.num_shipped == 0
    assert manager.num_pinned_blocks == 0
    assert all(b.ref_cnt == 0 for b in pin.blocks[0])
    assert manager.block_pool.get_num_free_blocks() == free_before


def test_on_finished_sending_returns_a_new_set():
    manager, _ = cached_manager()
    server = make_server(manager)
    ids = {"req-1"}
    out = server.on_finished_sending(ids)
    assert out == ids and out is not ids


def test_only_shipped_pins_are_taken_out():
    manager, _ = cached_manager()
    server = make_server(manager)
    # Only pins this server shipped are taken out; a request whose id merely
    # starts with the prefix still reaches the scheduler.
    stranger = PEER_PIN_ID_PREFIX + "client-chosen-id"
    assert server.on_finished_sending({stranger, "req"}) == {stranger, "req"}


def test_duplicate_finished_sending_is_harmless():
    manager, hashes = cached_manager()
    server = make_server(manager)
    server.serve("tok", lookup(hashes))
    (pin_id,) = ship(server)
    assert server.on_finished_sending({pin_id}) == set()
    assert server.on_finished_sending({pin_id}) == set()
    assert manager.num_pinned_blocks == 0


def test_invalid_blocks_before_ship_revoke_with_a_miss_reply():
    manager, hashes = cached_manager()
    server = make_server(manager)
    server.serve("tok", lookup(hashes))
    block_id = next(iter(manager._pins)).get_block_ids()[0][1]

    server.on_invalid_blocks({block_id})

    assert server.num_pending == 0
    assert manager.num_pinned_blocks == 0
    token, reply = only_reply(server)
    assert token == "tok"
    assert reply.miss_reason is PeerMissReason.NOT_CACHED and reply.hit_tokens == 0
    assert step(server) == {}


def test_invalid_blocks_after_ship_are_kept_and_counted():
    manager, hashes = cached_manager()
    server = make_server(manager)
    server.serve("tok", lookup(hashes))
    (pin_id,) = ship(server)
    block_id = next(iter(manager._pins)).get_block_ids()[0][0]

    server.on_invalid_blocks({block_id})

    assert server.num_revoked_after_reply == 1
    assert server.num_shipped == 1
    assert manager.num_pinned_blocks == 3
    server.on_finished_sending({pin_id})
    assert manager.num_pinned_blocks == 0


def test_invalid_blocks_unrelated_to_pins_are_ignored():
    manager, hashes = cached_manager()
    server = make_server(manager)
    server.serve("tok", lookup(hashes))
    server.on_invalid_blocks({9999})
    assert server.num_pending == 1 and server.num_revoked_after_reply == 0


def test_has_pending_work_transitions():
    manager, hashes = cached_manager()
    server = make_server(manager)
    assert not server.has_pending_work()
    server.serve("tok", lookup(hashes))
    assert server.has_pending_work()  # pending pin
    ship(server)
    assert server.has_pending_work()  # reply ready, not taken
    server.take_ready_replies()
    # A shipped pin keeps the engine stepping: the worker reports its
    # read-done notification or lease expiry only from inside a step.
    assert server.has_pending_work()
    assert server.num_shipped == 1
    (pin_id,) = server._shipped
    assert server.on_finished_sending({pin_id}) == set()
    assert not server.has_pending_work()


def test_miss_reply_keeps_engine_alive_until_taken():
    manager = make_manager()
    server = make_server(manager)
    server.serve("tok", lookup(make_request("x", tokens(1)).block_hashes))
    assert server.has_pending_work()
    server.take_ready_replies()
    assert not server.has_pending_work()


def test_lease_is_capped_at_the_configured_value():
    manager, hashes = cached_manager(1)
    clock = Clock()
    server = make_server(manager, clock=clock)
    server.serve("a", lookup(hashes, lease=5.0))
    server.serve("b", lookup(hashes, lease=500.0))
    server.serve("c", lookup(hashes))
    sends = ship(server)
    leases = sorted(round(e - clock.now) for e in sends.values())
    assert leases == [5, 30, 30]
    by_token = {t: r for t, r in server.take_ready_replies()}
    assert by_token["a"].remote_lease_extension == kv_lease_extension(5)


def test_pin_id_format():
    manager, hashes = cached_manager()
    server = make_server(manager)
    server.serve("a", lookup(hashes))
    server.serve("b", lookup(hashes))
    ids = list(ship(server))
    assert len(set(ids)) == 2
    for pin_id in ids:
        assert pin_id.startswith(PEER_PIN_ID_PREFIX)
        assert ":" not in pin_id and "," not in pin_id


def test_server_enables_evicted_hash_tracking():
    manager = make_manager()
    make_server(manager)
    assert manager.block_pool._max_evicted_hashes == NUM_BLOCKS


def test_other_threads_are_rejected():
    manager, hashes = cached_manager()
    server = make_server(manager)
    server.serve("tok", lookup(hashes))
    errors: list[BaseException] = []

    def run():
        try:
            server.on_step_scheduled()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert len(errors) == 1 and isinstance(errors[0], AssertionError)


# ---------------------------------------------------------------------------
# Wiring in the NIXL scheduler and connector (no model config needed).
# ---------------------------------------------------------------------------


def make_wired_scheduler(manager: KVCacheManager, **overrides):
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
        NixlConnectorScheduler,
    )

    from .utils import make_nixl_scheduler

    sched = make_nixl_scheduler(heartbeat=True)
    assert isinstance(sched, NixlConnectorScheduler)
    sched.vllm_config.cache_config = CacheConfig(block_size=BLOCK_SIZE)
    sched.block_size = BLOCK_SIZE
    sched._peer_lookup_enabled = overrides.get("peer_lookup", True)
    sched._peer_pin_max_fraction = 1.0
    sched._peer_pin_min_free_fraction = 0.0
    sched._peer_pin_serve_timeout = overrides.get("serve_timeout", 5.0)
    sched._peer_pins = None
    sched._compatibility_hash = None
    sched._registration_generation = 0
    sched._lookup = PeerLookupChannel()
    sched._lookup.enabled = sched._peer_lookup_enabled
    return sched


def make_wired_connector(sched) -> NixlPullConnector:
    connector = object.__new__(NixlPullConnector)
    connector.connector_scheduler = sched
    connector._kv_cache_manager = None
    return connector


def test_connector_wiring_end_to_end():
    manager, hashes = cached_manager()
    sched = make_wired_scheduler(manager)
    connector = make_wired_connector(sched)

    connector.bind_kv_cache_manager(manager)
    assert connector._kv_cache_manager is manager
    server = sched._peer_pins
    assert server is not None
    server.set_compatibility_hash(COMPAT)

    sched.serve_peer_lookup("tok", lookup(hashes))
    assert connector.has_pending_push_work()

    # Step 1 is scheduled; the lookup's fence is step 1, not yet processed.
    meta = connector.build_connector_meta(SimpleNamespace())
    assert meta.reqs_to_send == {} and meta.reqs_in_batch == set()
    list(connector.take_events())  # update_from_output for step 1

    # Step 2 ships it.
    meta = connector.build_connector_meta(SimpleNamespace())
    assert len(meta.reqs_to_send) == 1
    (pin_id,) = meta.reqs_to_send
    assert meta.reqs_in_batch == {pin_id}
    # build_connector_meta handed the reply to the listener's outbox.
    assert sched.take_peer_lookup_replies() == []
    _, reply = sched._lookup.outbox.get_nowait()
    assert reply.remote_request_id == pin_id
    assert reply.remote_blocks_expiry_time == meta.reqs_to_send[pin_id]
    # Shipped, not yet read: the engine must keep stepping to hear about it.
    assert connector.has_pending_push_work()

    # Read done: the pin id is stripped, other ids are untouched.
    from vllm.v1.outputs import KVConnectorOutput

    out = KVConnectorOutput(finished_sending={pin_id, "req-1"})
    connector.update_connector_output(out)
    assert out.finished_sending == {"req-1"}
    assert manager.num_pinned_blocks == 0
    assert not connector.has_pending_push_work()

    none_out = KVConnectorOutput()
    connector.update_connector_output(none_out)
    assert none_out.finished_sending is None


def test_peer_lookup_disabled_by_default():
    manager, _ = cached_manager()
    sched = make_wired_scheduler(manager, peer_lookup=False)
    make_wired_connector(sched).bind_kv_cache_manager(manager)
    assert sched._peer_pins is None
    assert not sched.has_pending_push_work()
    assert sched.take_peer_lookup_replies() == []


def test_handshake_metadata_sets_compat_hash():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlHandshakePayload,
    )

    manager, hashes = cached_manager()
    sched = make_wired_scheduler(manager)
    sched._handshake_data = {}
    sched._nixl_handshake_listener_t = object()  # skip starting the listener
    make_wired_connector(sched).bind_kv_cache_manager(manager)

    sched.set_xfer_handshake_metadata({(0, 0): NixlHandshakePayload(COMPAT, b"x")})

    sched.serve_peer_lookup("tok", lookup(hashes))
    assert sched._peer_pins is not None and sched._peer_pins.num_pending == 1


# ---------------------------------------------------------------------------
# Core scheduler: a pin id in finished_sending must not reach its assertion.
# These build a VllmConfig from a model, so they need Hugging Face access.
# ---------------------------------------------------------------------------


def test_scheduler_update_from_output_strips_pin_ids():
    from vllm.v1.core.kv_cache_manager import PrefixPin, PrefixPinMiss

    from .utils import (
        create_model_runner_output,
        create_request,
        create_scheduler,
        create_vllm_config,
    )

    vllm_config = create_vllm_config(
        kv_role="kv_both", kv_connector_extra_config={"peer_lookup": True}
    )
    scheduler = create_scheduler(vllm_config)
    assert scheduler.connector is not None
    nixl_sched = scheduler.connector.connector_scheduler
    server = nixl_sched._peer_pins
    assert server is not None

    # A pin that the worker already has in flight.
    pin = PrefixPin(tuple([] for _ in range(1)), 0, PrefixPinMiss.NONE)
    pin_id = PEER_PIN_ID_PREFIX + "abc"
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.peer_pins import (
        _ShippedPin,
    )

    server._shipped[pin_id] = _ShippedPin(pin, pin_id)

    request = create_request(request_id=1, num_tokens=10)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    model_output = create_model_runner_output([request], finished_sending={pin_id})

    scheduler.update_from_output(scheduler_output, model_output)  # must not assert

    assert server.num_shipped == 0
    assert pin.released
