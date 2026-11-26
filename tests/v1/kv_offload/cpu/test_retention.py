# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections import OrderedDict
from collections.abc import Sequence

import pytest

from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    StoreTrigger,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu import hints as hints_module
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkStatus
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory

_TRIGGER = StoreTrigger.ON_COMPUTE
_STEP = ScheduleEndContext(new_req_ids=(), preempted_req_ids=())
_IDS = itertools.count()


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, delta_s: float) -> None:
        self.now += delta_s


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(hints_module.time, "monotonic", fake)
    return fake


class _ExternalFIFOPolicy(CachePolicy):
    """Out-of-tree style policy that ignores chunk.retained."""

    def __init__(self, cache_capacity: int) -> None:
        super().__init__(cache_capacity)
        self.chunks: OrderedDict[OffloadKey, ChunkStatus] = OrderedDict()
        self.retention_events: list[tuple[OffloadKey, bool]] = []
        self.inserted_retained: list[tuple[OffloadKey, bool]] = []

    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.chunks.get(key)

    def insert(
        self, key: OffloadKey, chunk: ChunkStatus, req_context: ReqContext
    ) -> None:
        self.inserted_retained.append((key, chunk.retained))
        self.chunks[key] = chunk

    def remove(self, key: OffloadKey) -> None:
        self.chunks.pop(key, None)

    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        return

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        victims = [
            (k, c)
            for k, c in self.chunks.items()
            if c.ref_cnt == 0 and k not in protected
        ][:n]
        if len(victims) < n:
            return None
        for k, _ in victims:
            del self.chunks[k]
        return victims

    def clear(self) -> None:
        self.chunks.clear()

    def on_retention_changed(self, key: OffloadKey, retained: bool) -> None:
        self.retention_events.append((key, retained))


def _key(n: int) -> OffloadKey:
    return make_offload_key(f"k{n}".encode(), 0)


def _act(
    kind: str,
    action_id: str | None = None,
    *,
    lease_id: object = ...,
    ttl_s: object = ...,
    version: str = "1.0",
) -> KvHintAction:
    payload: dict[str, object] = {}
    if lease_id is not ...:
        payload["lease_id"] = lease_id
    if ttl_s is not ...:
        payload["ttl_s"] = ttl_s
    return KvHintAction(action_id or f"a{next(_IDS)}", kind, version, payload)


def _env(actions: list[KvHintAction], protocol_version: str = "0.1") -> KvHintsEnvelope:
    return KvHintsEnvelope(protocol_version, f"m{next(_IDS)}", actions)


def _ctx(*actions: KvHintAction, keys: Sequence[OffloadKey] = ()) -> ReqContext:
    ctx = ReqContext(
        f"r{next(_IDS)}", kv_hints=_env(list(actions)) if actions else None
    )
    for i, key in enumerate(keys):
        ctx.set_offload_key_position(key, 16 * (i + 1))
    return ctx


def _store(
    mgr: CPUOffloadingManager,
    keys: list[OffloadKey],
    *,
    retain: str | None = None,
    ttl_s: object = ...,
    ctx: ReqContext | None = None,
    complete: bool = True,
) -> list[OffloadKey]:
    """Run one request that stores keys and finishes; return evicted keys."""
    if ctx is None:
        actions = (
            [] if retain is None else [_act("cpu.retain", lease_id=retain, ttl_s=ttl_s)]
        )
        ctx = _ctx(*actions)
    for key in keys:
        if ctx.get_offload_key_position(key) is None:
            ctx.set_offload_key_position(
                key, 16 * (len(ctx._offload_key_positions) + 1)
            )
    mgr.on_new_request(ctx)
    out = mgr.prepare_store(keys, ctx, trigger=_TRIGGER)
    assert out is not None
    if complete and out.keys_to_store:
        mgr.complete_store(out.keys_to_store, ctx, trigger=_TRIGGER)
    mgr.on_request_finished(ctx)
    return out.evicted_keys


def _send(mgr: CPUOffloadingManager, *actions: KvHintAction) -> None:
    ctx = _ctx(*actions)
    mgr.on_new_request(ctx)
    mgr.on_request_finished(ctx)


def _metrics(mgr: CPUOffloadingManager) -> tuple[dict[tuple[str, str], int], int]:
    stats = mgr.get_stats()
    assert stats is not None
    data = stats.data["data"]
    return (
        dict(data.get(CPUOffloadingMetrics.CPU_HINT_ACTIONS, {})),
        int(data.get(CPUOffloadingMetrics.CPU_HINT_KEYS_SKIPPED, {}).get((), 0)),
    )


def test_invalid_hints_fail_open_with_metrics(clock: _FakeClock) -> None:
    mgr = CPUOffloadingManager(num_chunks=4)
    bad_proto = ReqContext("bad", kv_hints=_env([_act("cpu.retain")], "0.2"))
    _store(mgr, [_key(0)], ctx=bad_proto)
    assert _metrics(mgr) == ({("envelope", "ignored"): 1}, 0)

    retain = "cpu.retain"
    _store(
        mgr,
        [_key(1)],
        ctx=_ctx(
            KvHintAction("o1", "kvcr.retain", "1.0", {}),  # other tier: silent
            KvHintAction("u1", "cpu.unknown_op", "1.0", {}),
            KvHintAction("u2", "cpu.invalidate", "1.0", {"lease_id": "L1"}),
            _act(retain, lease_id="L1", version="0.1"),
            _act(retain, lease_id="L1", version="2.0"),
            _act(retain, lease_id="L1", version=""),
            _act(retain, lease_id=""),
            _act(retain, lease_id=123),
            _act(retain),  # no lease_id
            *(
                _act(retain, lease_id="L1", ttl_s=ttl)
                for ttl in (0, -1, True, "60", float("nan"), float("inf"))
            ),
            _act(retain, lease_id="L1", ttl_s=30, version="1.2"),
        ),
    )
    assert _metrics(mgr) == (
        {
            ("cpu.unknown", "ignored"): 2,
            (retain, "ignored"): 3,
            (retain, "rejected"): 9,
            (retain, "applied"): 1,
        },
        0,
    )


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_retained_evicted_last_and_still_evictable(
    clock: _FakeClock, policy: str
) -> None:
    mgr = CPUOffloadingManager(num_chunks=2, cache_policy=policy)
    k1, k2, k3, k4 = _key(1), _key(2), _key(3), _key(4)
    _store(mgr, [k1], retain="L1")
    _store(mgr, [k2])
    assert _store(mgr, [k3], retain="L2") == [k2]
    # Whole cache retained: stores still succeed by evicting a retained chunk.
    assert _store(mgr, [k4]) == [k1]
    assert mgr.lookup(k4, _ctx()) is LookupResult.HIT


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_ttl_default_and_clamp(clock: _FakeClock, policy: str) -> None:
    mgr = CPUOffloadingManager(num_chunks=2, cache_policy=policy)
    k1, k2, k3, k4 = _key(1), _key(2), _key(3), _key(4)
    _store(mgr, [k1], retain="L1", ttl_s=10**9)
    _store(mgr, [k2])

    clock.advance(hints_module.MAX_RETENTION_TTL_S - 1)
    mgr.on_schedule_end(_STEP)
    assert _store(mgr, [k3]) == [k2]
    clock.advance(1)
    mgr.on_schedule_end(_STEP)
    assert _store(mgr, [k4]) == [k1]

    # Omitted ttl_s uses the default.
    _store(mgr, [_key(5)], retain="L5")
    clock.advance(hints_module.DEFAULT_RETENTION_TTL_S - 1)
    mgr.on_schedule_end(_STEP)
    assert _store(mgr, [_key(6)]) == [k4]
    clock.advance(1)
    mgr.on_schedule_end(_STEP)
    assert _store(mgr, [_key(7)]) == [_key(5)]


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_release_applies_in_on_new_request(clock: _FakeClock, policy: str) -> None:
    mgr = CPUOffloadingManager(num_chunks=2, cache_policy=policy)
    k1, k2, k3 = _key(1), _key(2), _key(3)
    _store(mgr, [k1], retain="L1")
    _store(mgr, [k2])
    _metrics(mgr)

    # The release lands before this request's own prepare_store evicts.
    assert _store(mgr, [k3], ctx=_ctx(_act("cpu.release", lease_id="L1"))) == [k1]
    _send(mgr, _act("cpu.release", lease_id="L1"))
    assert _metrics(mgr) == (
        {("cpu.release", "applied"): 1, ("cpu.release", "noop"): 1},
        0,
    )


def test_release_cancels_earlier_retain_in_envelope_and_null_ttl(
    clock: _FakeClock,
) -> None:
    mgr = CPUOffloadingManager(num_chunks=2)
    k1, k2, k3 = _key(1), _key(2), _key(3)
    retain_then_release = _ctx(
        _act("cpu.retain", lease_id="L1"), _act("cpu.release", lease_id="L1")
    )
    _store(mgr, [k1], ctx=retain_then_release)
    # A release before a retain does not cancel it; a null ttl_s uses the default.
    release_then_retain = _ctx(
        _act("cpu.release", lease_id="L2"),
        _act("cpu.retain", lease_id="L2", ttl_s=None),
    )
    _store(mgr, [k2], ctx=release_then_retain)
    assert _store(mgr, [k3]) == [k1]
    assert _metrics(mgr) == (
        {
            ("cpu.retain", "noop"): 1,
            ("cpu.release", "noop"): 2,
            ("cpu.retain", "applied"): 1,
        },
        0,
    )


def test_released_leases_do_not_disturb_live_expiry(clock: _FakeClock) -> None:
    mgr = CPUOffloadingManager(num_chunks=2)
    k1, k2, k3, k4 = _key(1), _key(2), _key(3), _key(4)
    _store(mgr, [k1], retain="live", ttl_s=10.0)
    _store(mgr, [k2])
    # Enough create/release cycles to compact the expiry heap.
    for i in range(100):
        hit = _ctx(_act("cpu.retain", lease_id=f"S{i}"), keys=[k2])
        mgr.on_new_request(hit)
        mgr.on_request_finished(hit)
        _send(mgr, _act("cpu.release", lease_id=f"S{i}"))
    assert _store(mgr, [k3]) == [k2]
    clock.advance(10.0)
    mgr.on_schedule_end(_STEP)
    assert _store(mgr, [k4]) == [k1]


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_lease_expires_before_eviction_without_schedule_end(
    clock: _FakeClock, policy: str
) -> None:
    mgr = CPUOffloadingManager(num_chunks=2, cache_policy=policy)
    k1, k2, k3 = _key(1), _key(2), _key(3)
    _store(mgr, [k1], retain="L1", ttl_s=10.0)
    _store(mgr, [k2])
    clock.advance(10.0)
    assert _store(mgr, [k3]) == [k1]


def test_policy_hook_fires_on_edges_and_external_policy_still_evicts(
    clock: _FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[_ExternalFIFOPolicy] = []

    class _Tracked(_ExternalFIFOPolicy):
        def __init__(self, cache_capacity: int) -> None:
            super().__init__(cache_capacity)
            created.append(self)

    monkeypatch.setitem(CachePolicyFactory._registry, "ext_fifo", lambda: _Tracked)
    mgr = CPUOffloadingManager(num_chunks=2, cache_policy="ext_fifo")
    policy = created[0]
    k1, k2, k3 = _key(1), _key(2), _key(3)

    _store(mgr, [k1], retain="L1", ttl_s=10.0)
    assert policy.retention_events == [(k1, True)]

    # A second lease on k1 (1->2) and releasing the first (2->1) stay silent.
    r2 = _ctx(_act("cpu.retain", lease_id="L2", ttl_s=20.0), keys=[k1])
    mgr.on_new_request(r2)
    mgr.prepare_load([k1], r2)
    mgr.complete_load([k1], r2)
    mgr.on_request_finished(r2)
    _send(mgr, _act("cpu.release", lease_id="L1"))
    assert policy.retention_events == [(k1, True)]

    clock.advance(20.0)
    mgr.on_schedule_end(_STEP)
    assert policy.retention_events == [(k1, True), (k1, False)]

    _store(mgr, [k2], retain="L3")
    assert _store(mgr, [k3]) == [k1]

    # A failed store's lease that expired is swept before the key is re-inserted.
    mgr.reset_cache()
    rf = _ctx(_act("cpu.retain", lease_id="L4", ttl_s=10.0))
    _store(mgr, [k1], ctx=rf, complete=False)
    mgr.complete_store([k1], rf, trigger=_TRIGGER, success=False)
    clock.advance(10.0)
    _store(mgr, [k1])
    assert policy.inserted_retained[-1] == (k1, False)


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_refresh_adds_new_keys_and_never_shortens_ttl(
    clock: _FakeClock, policy: str
) -> None:
    mgr = CPUOffloadingManager(num_chunks=3, cache_policy=policy)
    k1, k2, k3, k4 = _key(1), _key(2), _key(3), _key(4)
    _store(mgr, [k1], retain="sess", ttl_s=100.0)

    # Turn 2 reuses k1, adds k2 and asks for an earlier deadline (60 < 100).
    clock.advance(10.0)
    t2 = _ctx(_act("cpu.retain", lease_id="sess", ttl_s=50.0), keys=[k1, k2])
    mgr.on_new_request(t2)
    mgr.prepare_load([k1], t2)
    mgr.complete_load([k1], t2)
    out = mgr.prepare_store([k2], t2, trigger=_TRIGGER)
    assert out is not None
    mgr.complete_store(out.keys_to_store, t2, trigger=_TRIGGER)
    mgr.on_request_finished(t2)

    _store(mgr, [k3])
    clock.advance(70.0)
    mgr.on_schedule_end(_STEP)
    assert _store(mgr, [k4], retain="other", ttl_s=500.0) == [k3]

    # Every refresh finds the live lease, even after its first deadline.
    _metrics(mgr)
    for _ in range(70):
        clock.advance(1.0)
        _send(mgr, _act("cpu.retain", "same-id", lease_id="sess", ttl_s=100.0))
    assert _metrics(mgr) == ({("cpu.retain", "applied"): 70}, 0)

    # The refreshes compacted the heap; "other" still expires afterwards.
    clock.advance(500.0)
    mgr.on_schedule_end(_STEP)
    _send(mgr, _act("cpu.release", lease_id="other"))
    assert _metrics(mgr) == ({("cpu.release", "noop"): 1}, 0)


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_restored_key_is_retained_while_lease_is_live(
    clock: _FakeClock, policy: str
) -> None:
    mgr = CPUOffloadingManager(num_chunks=2, cache_policy=policy)
    k1, k2, k3, k4 = _key(1), _key(2), _key(3), _key(4)
    _store(mgr, [k1], retain="L1", ttl_s=60.0)
    _store(mgr, [k2], retain="L2", ttl_s=60.0)
    assert _store(mgr, [k3], retain="L3", ttl_s=60.0) == [k1]
    _send(mgr, _act("cpu.release", lease_id="L2"), _act("cpu.release", lease_id="L3"))

    clock.advance(10.0)
    _store(mgr, [k1])
    assert _store(mgr, [k4]) == [k3]
    # k1 is now the oldest chunk, but L1 retains it again.
    assert _store(mgr, [_key(5)]) == [k4]


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_refresh_drops_keys_evicted_since_last_grant(
    clock: _FakeClock, policy: str
) -> None:
    mgr = CPUOffloadingManager(num_chunks=2, cache_policy=policy)
    k1, k2, k3, k4 = _key(1), _key(2), _key(3), _key(4)
    _store(mgr, [k1], retain="L1")
    _store(mgr, [k2], retain="L2")
    assert _store(mgr, [k3]) == [k1]
    # Refreshing L1 forgets the evicted k1, so re-storing k1 does not retain it.
    _send(mgr, _act("cpu.retain", lease_id="L1"))
    assert _store(mgr, [k1]) == [k3]
    assert _store(mgr, [k4]) == [k1]


def test_action_ids_dedupe_within_one_envelope_only(clock: _FakeClock) -> None:
    mgr = CPUOffloadingManager(num_chunks=3)
    k1, k2, k3 = _key(1), _key(2), _key(3)
    _store(
        mgr,
        [k1],
        ctx=_ctx(
            _act("cpu.retain", "a1", lease_id="L1"),
            _act("cpu.retain", "a1", lease_id="L2"),
        ),
    )
    assert _metrics(mgr) == (
        {("cpu.retain", "applied"): 1, ("cpu.retain", "duplicate"): 1},
        0,
    )

    # n>1 sampling: children share one envelope, and each retains its own keys.
    envelope = _env([_act("cpu.retain", "a1", lease_id="S")])
    children = [ReqContext(f"{i}_parent", kv_hints=envelope) for i in range(2)]
    for child in children:
        mgr.on_new_request(child)
    for child, key in zip(children, (k2, k3)):
        child.set_offload_key_position(key, 16)
        out = mgr.prepare_store([key], child, trigger=_TRIGGER)
        assert out is not None
        mgr.complete_store(out.keys_to_store, child, trigger=_TRIGGER)
        mgr.on_request_finished(child)
    # All three chunks are retained, so the oldest (k1) goes rather than k3.
    assert _store(mgr, [_key(4)]) == [k1]


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_write_pending_in_flight_load_and_failed_store(
    clock: _FakeClock, policy: str
) -> None:
    mgr = CPUOffloadingManager(num_chunks=2, cache_policy=policy)
    k1, k2, k3 = _key(1), _key(2), _key(3)

    # Finishing while the write is pending still retains the chunk.
    r1 = _ctx(_act("cpu.retain", lease_id="L1", ttl_s=30.0))
    _store(mgr, [k1], ctx=r1, complete=False)
    mgr.complete_store([k1], r1, trigger=_TRIGGER, success=True)
    _store(mgr, [k2])
    assert _store(mgr, [k3]) == [k2]

    # Releasing during a load clears retention once the load ends.
    loader = _ctx()
    mgr.prepare_load([k1], loader)
    _send(mgr, _act("cpu.release", lease_id="L1"))
    mgr.complete_load([k1], loader)
    assert _store(mgr, [_key(4)]) == [k1]


def test_gpu_prefix_hit_missing_keys_and_max_leases(
    clock: _FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hints_module, "MAX_LEASES", 1)
    mgr = CPUOffloadingManager(num_chunks=3)
    k1, k2, k3 = _key(1), _key(2), _key(3)
    _store(mgr, [k1])

    # k1 is a GPU prefix-cache hit (no CPU load/store); k2 was never stored.
    hit = _ctx(_act("cpu.retain", lease_id="L1"), keys=[k1, k2])
    mgr.on_new_request(hit)
    mgr.on_request_finished(hit)
    assert _metrics(mgr) == ({("cpu.retain", "applied"): 1}, 1)
    _store(mgr, [k2])
    _store(mgr, [k3])
    assert _store(mgr, [_key(4)]) == [k2]

    # A second lease is rejected at the cap.
    _store(mgr, [_key(5)], retain="L2")
    assert _metrics(mgr) == ({("cpu.retain", "rejected"): 1}, 0)

    # Retaining only missing keys creates no lease.
    _send(mgr, _act("cpu.release", lease_id="L1"))
    _metrics(mgr)
    miss = _ctx(_act("cpu.retain", lease_id="LE"), keys=[_key(99)])
    mgr.on_new_request(miss)
    mgr.on_request_finished(miss)
    _send(mgr, _act("cpu.release", lease_id="LE"))
    assert _metrics(mgr) == ({("cpu.retain", "noop"): 1, ("cpu.release", "noop"): 1}, 1)


def test_reset_cache_drops_leases_and_pending_retains(clock: _FakeClock) -> None:
    mgr = CPUOffloadingManager(num_chunks=2)
    k1, k2, k3 = _key(1), _key(2), _key(3)
    _store(mgr, [k1], retain="L1")
    active = _ctx(_act("cpu.retain", lease_id="L2"), keys=[k2])
    mgr.on_new_request(active)
    mgr.reset_cache()

    # Neither re-storing k1 (L1) nor finishing `active` (L2) retains anything.
    _store(mgr, [k1])
    _store(mgr, [k2])
    mgr.on_request_finished(active)
    assert _store(mgr, [k3]) == [k1]
    assert _store(mgr, [_key(4)]) == [k2]

    assert _metrics(mgr)[0]
    assert _metrics(mgr) == ({}, 0)
