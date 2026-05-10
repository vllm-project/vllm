# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the inline-vectors auto-promotion path.

Two-strikes promotion: the first time we see a ``(prefill_hash,
decode_hash)`` pair we just record the sighting and let the request go
through inline-pack as normal.  Only on the *second* sight do we
broadcast ``register_steering_modules`` and start rewriting affected
requests to use ``steering_module_ref``.

This protects genuinely-unique-per-request workloads (research sweeps
with a fresh spec each time) from paying a wasted synchronous RPC,
while still capturing the common case where a batch of requests
shares a spec.

Covers:

- LRU state transitions: first-sight (no name) → second-sight registers
  → registered hits return the name; eviction returns the prior name
  only when it had been registered
- Eligibility (already-named, no-vectors, already-packed all skip)
- Two-strikes: first observation does NOT broadcast; second does
- Repeated-spec dedup after the second sight registers it
- Distinct prefill+decode hashes don't collide on the dedup key
- LRU eviction of a registered entry triggers paired unregister; LRU
  eviction of a not-yet-registered entry triggers neither
- Idempotency on the mutated SamplingParams (re-running is a no-op)
- Async variant matches sync semantics
"""

import asyncio

import pytest

from vllm.config.steering_types import (
    SteeringAutoPromoteLRU,
    maybe_auto_promote_steering_modules,
    maybe_auto_promote_steering_modules_async,
)
from vllm.sampling_params import SamplingParams

# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


class _RpcSpy:
    """Records ``rpc_fn(method, kwargs=...)`` calls in arrival order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, method: str, *, kwargs: dict | None = None, **_):
        self.calls.append((method, kwargs or {}))


class _AsyncRpcSpy:
    """Async counterpart of :class:`_RpcSpy`."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, method: str, *, kwargs: dict | None = None, **_):
        self.calls.append((method, kwargs or {}))


def _spec(hook: str, layer_to_vec: dict[int, list[float]]):
    return {hook: dict(layer_to_vec)}


def _fresh_sp(spec_hook: str = "post_mlp",
              spec_vec: dict[int, list[float]] | None = None,
              **kw) -> SamplingParams:
    if spec_vec is None:
        spec_vec = {0: [1.0, 2.0]}
    return SamplingParams(steering_vectors=_spec(spec_hook, spec_vec), **kw)


# ---------------------------------------------------------------------------
# LRU state machine
# ---------------------------------------------------------------------------


class TestSteeringAutoPromoteLRU:
    def test_first_sight_returns_first_status(self):
        lru = SteeringAutoPromoteLRU(capacity=4)
        status, name, evicted = lru.observe((1, 2))
        assert status == "first"
        assert name is None
        assert evicted is None
        # The entry is now tracked but has no name.
        assert (1, 2) in lru
        assert lru.get((1, 2)) is None

    def test_second_sight_returns_second_then_mark_registers(self):
        lru = SteeringAutoPromoteLRU(capacity=4)
        lru.observe((1, 2))  # first
        status, name, evicted = lru.observe((1, 2))
        assert status == "second"
        assert name is None
        assert evicted is None
        # Caller now picks the name and records it.
        lru.mark_registered((1, 2), "auto_xyz")
        # Third observation hits the registered entry.
        status, name, evicted = lru.observe((1, 2))
        assert status == "registered"
        assert name == "auto_xyz"
        assert evicted is None

    def test_registered_get_returns_name(self):
        lru = SteeringAutoPromoteLRU(capacity=4)
        lru.observe((1, 2))
        lru.observe((1, 2))
        lru.mark_registered((1, 2), "auto_a")
        assert lru.get((1, 2)) == "auto_a"

    def test_get_returns_none_for_unregistered_first_sight(self):
        lru = SteeringAutoPromoteLRU(capacity=4)
        lru.observe((1, 2))
        # First-sight entry has name=None; get returns None.
        assert lru.get((1, 2)) is None

    def test_first_sight_overflow_evicts_unnamed_entry(self):
        lru = SteeringAutoPromoteLRU(capacity=2)
        lru.observe((1, 2))
        lru.observe((3, 4))
        # Both held with name=None.
        status, _, evicted = lru.observe((5, 6))
        assert status == "first"
        # Evicted the LRU end (key=(1,2) with no name).
        assert evicted == ((1, 2), None)

    def test_registered_lru_evicted_returns_name(self):
        lru = SteeringAutoPromoteLRU(capacity=2)
        # Register (1, 2).
        lru.observe((1, 2))
        lru.observe((1, 2))
        lru.mark_registered((1, 2), "auto_a")
        # Touch (3, 4) so (1, 2) becomes the LRU.
        lru.observe((3, 4))
        lru.observe((3, 4))  # second sight, but we only assert eviction below
        # Insert (5, 6) — should evict (1, 2) which has the registered name.
        status, _, evicted = lru.observe((5, 6))
        assert status == "first"
        assert evicted == ((1, 2), "auto_a")

    def test_observe_refreshes_recency(self):
        lru = SteeringAutoPromoteLRU(capacity=2)
        lru.observe((1, 2))
        lru.observe((3, 4))
        # Touch (1, 2) — second sight makes it MRU.
        lru.observe((1, 2))
        # Insert a third — should evict (3, 4).
        _, _, evicted = lru.observe((5, 6))
        assert evicted == ((3, 4), None)

    def test_mark_registered_unknown_key_raises(self):
        lru = SteeringAutoPromoteLRU(capacity=4)
        with pytest.raises(KeyError):
            lru.mark_registered((9, 9), "auto_x")

    def test_capacity_must_be_positive(self):
        with pytest.raises(ValueError):
            SteeringAutoPromoteLRU(capacity=0)


# ---------------------------------------------------------------------------
# Sync auto-promote
# ---------------------------------------------------------------------------


class TestSyncAutoPromote:
    def test_first_sight_does_not_register(self):
        # Single one-shot inline request: must NOT pay a register RPC.
        # This is the unique-per-request workload's protection.
        sp = _fresh_sp()
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        maybe_auto_promote_steering_modules(sp, rpc, lru)
        assert rpc.calls == []
        # Inline fields untouched — caller still goes through inline-pack.
        assert sp.steering_vectors is not None
        assert sp.steering_module_ref is None

    def test_second_sight_registers_and_promotes(self):
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        sp_a = _fresh_sp()
        sp_b = _fresh_sp()  # identical spec ⇒ same hash
        # First sight: no broadcast, no promotion.
        maybe_auto_promote_steering_modules(sp_a, rpc, lru)
        assert rpc.calls == []
        assert sp_a.steering_module_ref is None
        # Second sight: register + promote sp_b.
        maybe_auto_promote_steering_modules(sp_b, rpc, lru)
        assert len(rpc.calls) == 1
        method, kwargs = rpc.calls[0]
        assert method == "register_steering_modules"
        assert sp_b.steering_module_ref is not None
        name = sp_b.steering_module_ref[0]
        assert name.startswith("_auto_")
        assert name in kwargs["modules"]
        # sp_b's inline tier fields cleared so wire payload doesn't double-ship.
        assert sp_b.steering_vectors is None
        assert sp_b.prefill_steering_vectors is None
        assert sp_b.decode_steering_vectors is None

    def test_third_sight_hits_cache_no_extra_register(self):
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        sp_a = _fresh_sp()
        sp_b = _fresh_sp()
        sp_c = _fresh_sp()
        maybe_auto_promote_steering_modules(sp_a, rpc, lru)  # first
        maybe_auto_promote_steering_modules(sp_b, rpc, lru)  # second → register
        maybe_auto_promote_steering_modules(sp_c, rpc, lru)  # registered hit
        # Only ONE register — sp_a fell through to inline-pack, sp_b/sp_c share.
        assert sum(1 for m, _ in rpc.calls if m == "register_steering_modules") == 1
        # sp_b and sp_c share the same module name.
        assert sp_b.steering_module_ref == sp_c.steering_module_ref

    def test_distinct_phase_specs_do_not_collide(self):
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        # Same prefill, different decode — distinct dedup key.
        sp_a1 = SamplingParams(
            steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
            decode_steering_vectors=_spec("post_mlp", {0: [3.0, 4.0]}),
        )
        sp_a2 = SamplingParams(
            steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
            decode_steering_vectors=_spec("post_mlp", {0: [3.0, 4.0]}),
        )
        sp_b1 = SamplingParams(
            steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
            decode_steering_vectors=_spec("post_mlp", {0: [5.0, 6.0]}),
        )
        sp_b2 = SamplingParams(
            steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
            decode_steering_vectors=_spec("post_mlp", {0: [5.0, 6.0]}),
        )
        # Two distinct keys, each needs two sightings to register.
        maybe_auto_promote_steering_modules(sp_a1, rpc, lru)  # first key A
        maybe_auto_promote_steering_modules(sp_b1, rpc, lru)  # first key B
        maybe_auto_promote_steering_modules(sp_a2, rpc, lru)  # second key A → reg
        maybe_auto_promote_steering_modules(sp_b2, rpc, lru)  # second key B → reg
        # Two registers — different keys.
        assert sum(1 for m, _ in rpc.calls if m == "register_steering_modules") == 2
        assert sp_a2.steering_module_ref != sp_b2.steering_module_ref

    def test_unregistered_eviction_emits_no_unregister(self):
        # Two distinct first-sight entries with capacity=1: the second
        # observation evicts the first.  Since the first never registered,
        # no unregister RPC should be issued.
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=1)
        sp_a = _fresh_sp(spec_vec={0: [1.0, 2.0]})
        sp_b = _fresh_sp(spec_vec={0: [3.0, 4.0]})
        maybe_auto_promote_steering_modules(sp_a, rpc, lru)
        maybe_auto_promote_steering_modules(sp_b, rpc, lru)
        # Both fell through to inline-pack; zero RPC calls.
        assert rpc.calls == []
        assert sp_a.steering_module_ref is None
        assert sp_b.steering_module_ref is None

    def test_registered_eviction_unregisters(self):
        # Register key A (two sightings), then evict it by filling LRU.
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=1)
        sp_a1 = _fresh_sp(spec_vec={0: [1.0, 2.0]})
        sp_a2 = _fresh_sp(spec_vec={0: [1.0, 2.0]})
        sp_b1 = _fresh_sp(spec_vec={0: [3.0, 4.0]})
        sp_b2 = _fresh_sp(spec_vec={0: [3.0, 4.0]})
        maybe_auto_promote_steering_modules(sp_a1, rpc, lru)  # first A
        maybe_auto_promote_steering_modules(sp_a2, rpc, lru)  # second A → register
        # Now A is registered, capacity=1 — observing B evicts A.
        maybe_auto_promote_steering_modules(sp_b1, rpc, lru)
        # The B observation is "first" but it evicts a *registered* A entry,
        # so we expect an unregister RPC for A's name.
        methods = [m for m, _ in rpc.calls]
        assert methods == [
            "register_steering_modules",
            "unregister_steering_modules",
        ]
        unreg_kwargs = rpc.calls[-1][1]
        assert unreg_kwargs["names"] == [sp_a2.steering_module_ref[0]]
        # B is still in first-sight; second observation registers it.
        maybe_auto_promote_steering_modules(sp_b2, rpc, lru)
        methods = [m for m, _ in rpc.calls]
        assert methods == [
            "register_steering_modules",
            "unregister_steering_modules",
            "register_steering_modules",
        ]

    def test_named_request_is_no_op(self):
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        sp = SamplingParams(steering_module_ref=("preset", 1.0))
        maybe_auto_promote_steering_modules(sp, rpc, lru)
        assert rpc.calls == []
        assert sp.steering_module_ref == ("preset", 1.0)

    def test_no_steering_is_no_op(self):
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        sp = SamplingParams()
        maybe_auto_promote_steering_modules(sp, rpc, lru)
        assert rpc.calls == []
        assert sp.steering_module_ref is None

    def test_already_packed_observes_for_two_strikes(self):
        # An sp that already went through pack still participates in the
        # two-strikes count: this is the [sp]*N shared-object case where
        # request 0's pack call mutates sp before request 1's auto-promote
        # runs.  First-sight on a packed sp must not register (no RPC) but
        # MUST be recorded so the second sight can promote.
        sp = _fresh_sp()
        # Force the packed fields directly to simulate prior packing.
        sp._effective_prefill_steering_packed = {}
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        maybe_auto_promote_steering_modules(sp, rpc, lru)
        # First sight on a packed sp: no broadcast, sp untouched.
        assert rpc.calls == []
        assert sp.steering_module_ref is None
        # The LRU must have recorded the sighting so a second observation
        # promotes.
        assert len(lru) == 1

    def test_packed_sp_second_sight_promotes_and_clears_packed(self):
        # The [sp]*N case: req 0 leaves sp packed; req 1 (same sp object,
        # same hash key but observed afresh by another request flowing
        # through ``LLM._add_request``) must register, install the
        # module_ref, AND clear the packed fields so the wire payload
        # doesn't double-ship.
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        sp = _fresh_sp()
        # Simulate prior pass through pack: prime the cached hashes,
        # set the packed sentinel, clear inline.
        _ = sp.prefill_steering_config_hash
        _ = sp.decode_steering_config_hash
        sp._effective_prefill_steering_packed = {"post_mlp": {}}
        sp._effective_decode_steering_packed = {"post_mlp": {}}
        sp.steering_vectors = None
        # First observation (simulating request 0 in a different sp with
        # the same key): record sighting, no broadcast.
        # Use a sibling sp for the first sighting so we don't mutate the
        # shared one yet.
        sp_seed = _fresh_sp()
        maybe_auto_promote_steering_modules(sp_seed, rpc, lru)
        assert rpc.calls == []
        # Second observation against the packed sp: register + promote.
        maybe_auto_promote_steering_modules(sp, rpc, lru)
        assert sp.steering_module_ref is not None
        assert sp._effective_prefill_steering_packed is None
        assert sp._effective_decode_steering_packed is None
        assert sum(1 for m, _ in rpc.calls if m == "register_steering_modules") == 1

    def test_idempotent_after_promotion(self):
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        sp_warm = _fresh_sp()
        sp_target = _fresh_sp()
        # Warm the cache — sp_warm is the first sighting, sp_target the second.
        maybe_auto_promote_steering_modules(sp_warm, rpc, lru)
        maybe_auto_promote_steering_modules(sp_target, rpc, lru)
        first_calls = list(rpc.calls)
        first_ref = sp_target.steering_module_ref
        assert first_ref is not None
        # Run again on the now-promoted sp_target — should be a no-op.
        maybe_auto_promote_steering_modules(sp_target, rpc, lru)
        assert rpc.calls == first_calls
        assert sp_target.steering_module_ref == first_ref


# ---------------------------------------------------------------------------
# Async auto-promote — same semantics as sync
# ---------------------------------------------------------------------------


class TestAsyncAutoPromote:
    def test_async_first_sight_does_not_register(self):
        async def go():
            rpc = _AsyncRpcSpy()
            lru = SteeringAutoPromoteLRU(capacity=4)
            sp = _fresh_sp()
            await maybe_auto_promote_steering_modules_async(sp, rpc, lru)
            return sp, rpc

        sp, rpc = asyncio.run(go())
        assert rpc.calls == []
        assert sp.steering_module_ref is None

    def test_async_second_sight_promotes_and_dedups(self):
        async def go():
            rpc = _AsyncRpcSpy()
            lru = SteeringAutoPromoteLRU(capacity=4)
            sp_a = _fresh_sp()
            sp_b = _fresh_sp()
            sp_c = _fresh_sp()
            await maybe_auto_promote_steering_modules_async(sp_a, rpc, lru)  # first
            await maybe_auto_promote_steering_modules_async(sp_b, rpc, lru)  # second
            await maybe_auto_promote_steering_modules_async(sp_c, rpc, lru)  # cached
            return sp_a, sp_b, sp_c, rpc

        sp_a, sp_b, sp_c, rpc = asyncio.run(go())
        assert sp_a.steering_module_ref is None  # first sight: not promoted
        assert sp_b.steering_module_ref is not None
        assert sp_b.steering_module_ref == sp_c.steering_module_ref
        assert sum(1 for m, _ in rpc.calls if m == "register_steering_modules") == 1

    def test_async_registered_eviction_calls_unregister(self):
        async def go():
            rpc = _AsyncRpcSpy()
            lru = SteeringAutoPromoteLRU(capacity=1)
            # Register A (two sightings).
            sp_a1 = _fresh_sp(spec_vec={0: [1.0, 2.0]})
            sp_a2 = _fresh_sp(spec_vec={0: [1.0, 2.0]})
            sp_b = _fresh_sp(spec_vec={0: [3.0, 4.0]})
            await maybe_auto_promote_steering_modules_async(sp_a1, rpc, lru)
            await maybe_auto_promote_steering_modules_async(sp_a2, rpc, lru)
            # B's first sighting evicts A (registered).
            await maybe_auto_promote_steering_modules_async(sp_b, rpc, lru)
            return rpc

        rpc = asyncio.run(go())
        methods = [m for m, _ in rpc.calls]
        assert methods == [
            "register_steering_modules",
            "unregister_steering_modules",
        ]
