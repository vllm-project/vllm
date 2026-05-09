# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the inline-vectors auto-promotion path.

Auto-promotion turns an inline ``steering_vectors`` request into a
named-module reference on first sight, dedup'ing repeated specs.
Covers:

- Eligibility (already-named, no-vectors, already-packed all skip)
- LRU cache hit avoids a second register_steering_modules broadcast
- LRU eviction on overflow triggers paired unregister_steering_modules
- Hash dedup key uses BOTH prefill and decode hashes — distinct phase
  specs do not collide
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


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------


class TestSteeringAutoPromoteLRU:
    def test_get_miss_returns_none(self):
        lru = SteeringAutoPromoteLRU(capacity=4)
        assert lru.get((1, 2)) is None

    def test_put_then_get(self):
        lru = SteeringAutoPromoteLRU(capacity=4)
        evicted = lru.put((1, 2), "a")
        assert evicted is None
        assert lru.get((1, 2)) == "a"

    def test_put_existing_returns_none_and_refreshes(self):
        lru = SteeringAutoPromoteLRU(capacity=2)
        lru.put((1, 2), "a")
        lru.put((3, 4), "b")
        # Re-put existing — should not evict.
        evicted = lru.put((1, 2), "a")
        assert evicted is None
        # Insert a third — would evict (3, 4) since (1, 2) was just refreshed.
        evicted = lru.put((5, 6), "c")
        assert evicted == ((3, 4), "b")

    def test_get_refreshes_recency(self):
        lru = SteeringAutoPromoteLRU(capacity=2)
        lru.put((1, 2), "a")
        lru.put((3, 4), "b")
        # Touch (1, 2) — makes it the MRU.
        assert lru.get((1, 2)) == "a"
        # Insert a third — should evict (3, 4).
        evicted = lru.put((5, 6), "c")
        assert evicted == ((3, 4), "b")

    def test_capacity_must_be_positive(self):
        with pytest.raises(ValueError):
            SteeringAutoPromoteLRU(capacity=0)


# ---------------------------------------------------------------------------
# Sync auto-promote
# ---------------------------------------------------------------------------


class TestSyncAutoPromote:
    def test_inline_request_gets_promoted(self):
        sp = SamplingParams(
            steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
        )
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        maybe_auto_promote_steering_modules(sp, rpc, lru)
        assert sp.steering_module_ref is not None
        name, scale = sp.steering_module_ref
        assert name.startswith("_auto_")
        assert scale == 1.0
        # Inline tier fields cleared so wire payload doesn't double-ship.
        assert sp.steering_vectors is None
        assert sp.prefill_steering_vectors is None
        assert sp.decode_steering_vectors is None
        # One register, no unregister.
        assert len(rpc.calls) == 1
        method, kwargs = rpc.calls[0]
        assert method == "register_steering_modules"
        assert name in kwargs["modules"]

    def test_repeated_spec_hits_cache(self):
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        sp_a = SamplingParams(
            steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
        )
        sp_b = SamplingParams(
            steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
        )
        maybe_auto_promote_steering_modules(sp_a, rpc, lru)
        maybe_auto_promote_steering_modules(sp_b, rpc, lru)
        # Only ONE register — the second hit the cache.
        assert sum(1 for m, _ in rpc.calls if m == "register_steering_modules") == 1
        # Both requests share the same module name.
        assert sp_a.steering_module_ref == sp_b.steering_module_ref

    def test_distinct_phase_specs_do_not_collide(self):
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        # Same prefill, different decode — distinct dedup key.
        sp_a = SamplingParams(
            steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
            decode_steering_vectors=_spec("post_mlp", {0: [3.0, 4.0]}),
        )
        sp_b = SamplingParams(
            steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
            decode_steering_vectors=_spec("post_mlp", {0: [5.0, 6.0]}),
        )
        maybe_auto_promote_steering_modules(sp_a, rpc, lru)
        maybe_auto_promote_steering_modules(sp_b, rpc, lru)
        # Two registers — different keys.
        assert sum(1 for m, _ in rpc.calls if m == "register_steering_modules") == 2
        assert sp_a.steering_module_ref != sp_b.steering_module_ref

    def test_lru_eviction_unregisters(self):
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=1)
        sp_a = SamplingParams(
            steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
        )
        sp_b = SamplingParams(
            steering_vectors=_spec("post_mlp", {0: [3.0, 4.0]}),
        )
        maybe_auto_promote_steering_modules(sp_a, rpc, lru)
        maybe_auto_promote_steering_modules(sp_b, rpc, lru)
        methods = [m for m, _ in rpc.calls]
        # Expect: register A, register B, unregister A.
        assert methods == [
            "register_steering_modules",
            "register_steering_modules",
            "unregister_steering_modules",
        ]
        unreg_kwargs = rpc.calls[-1][1]
        assert unreg_kwargs["names"] == [sp_a.steering_module_ref[0]]

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

    def test_already_packed_is_no_op(self):
        # Simulate a request that already went through pack_effective_steering.
        sp = SamplingParams(
            steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
        )
        # Force the packed fields directly to simulate prior packing.
        # We don't actually need real arrays — non-None is enough to bail.
        sp._effective_prefill_steering_packed = {}
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        maybe_auto_promote_steering_modules(sp, rpc, lru)
        assert rpc.calls == []

    def test_idempotent_after_promotion(self):
        rpc = _RpcSpy()
        lru = SteeringAutoPromoteLRU(capacity=4)
        sp = SamplingParams(
            steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
        )
        maybe_auto_promote_steering_modules(sp, rpc, lru)
        first_calls = list(rpc.calls)
        first_ref = sp.steering_module_ref
        # Run again — should be a no-op (sp now has module_ref set).
        maybe_auto_promote_steering_modules(sp, rpc, lru)
        assert rpc.calls == first_calls
        assert sp.steering_module_ref == first_ref


# ---------------------------------------------------------------------------
# Async auto-promote — same semantics as sync
# ---------------------------------------------------------------------------


class TestAsyncAutoPromote:
    def test_async_promotes_and_dedups(self):
        async def go():
            rpc = _AsyncRpcSpy()
            lru = SteeringAutoPromoteLRU(capacity=4)
            sp_a = SamplingParams(
                steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
            )
            sp_b = SamplingParams(
                steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
            )
            await maybe_auto_promote_steering_modules_async(sp_a, rpc, lru)
            await maybe_auto_promote_steering_modules_async(sp_b, rpc, lru)
            return sp_a, sp_b, rpc

        sp_a, sp_b, rpc = asyncio.run(go())
        assert sp_a.steering_module_ref is not None
        assert sp_a.steering_module_ref == sp_b.steering_module_ref
        assert sum(1 for m, _ in rpc.calls if m == "register_steering_modules") == 1

    def test_async_eviction_calls_unregister(self):
        async def go():
            rpc = _AsyncRpcSpy()
            lru = SteeringAutoPromoteLRU(capacity=1)
            sp_a = SamplingParams(
                steering_vectors=_spec("post_mlp", {0: [1.0, 2.0]}),
            )
            sp_b = SamplingParams(
                steering_vectors=_spec("post_mlp", {0: [3.0, 4.0]}),
            )
            await maybe_auto_promote_steering_modules_async(sp_a, rpc, lru)
            await maybe_auto_promote_steering_modules_async(sp_b, rpc, lru)
            return rpc

        rpc = asyncio.run(go())
        methods = [m for m, _ in rpc.calls]
        assert methods == [
            "register_steering_modules",
            "register_steering_modules",
            "unregister_steering_modules",
        ]
