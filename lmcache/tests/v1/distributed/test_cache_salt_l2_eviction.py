# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for L2 eviction scoped by ``cache_salt``.

Drives :class:`L2EvictionController` end-to-end against a real
:class:`MockL2Adapter` with :class:`IsolatedLRUEvictionPolicy` and a
:class:`QuotaManager`, covering:

* over-budget salts get evicted; within-budget salts are left alone
* unregistered salts (no quota entry) have effective limit 0 and are
  wiped on the next eviction cycle
* salt isolation: touching alice's keys doesn't move bob's
"""

# Standard
import os
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.eviction import L2EvictionPolicy
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L2AdapterEvictionState,
    L2EvictionController,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _key(chunk_id: int, cache_salt: str) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
        cache_salt=cache_salt,
    )


def _memory_obj(n_floats: int = 128) -> TensorMemoryObj:
    """Small (~0.5 KB with 128 floats) tensor so quotas fit comfortably
    inside the mock adapter's 1 MB capacity."""
    raw = torch.empty(n_floats, dtype=torch.float32)
    raw.fill_(1.0)
    metadata = MemoryObjMetadata(
        shape=torch.Size([n_floats]),
        dtype=torch.float32,
        address=0,
        phy_size=n_floats * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, metadata, parent_allocator=None)


def _wait_fd(fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if not events:
        return False
    try:
        os.eventfd_read(fd)
    except BlockingIOError:
        pass
    return True


def _store_sync(adapter: MockL2Adapter, key: ObjectKey, obj: TensorMemoryObj):
    """Submit a store and wait for it to land in the base-class
    accounting (``_notify_keys_stored`` fires in ``_execute_store_in_the_loop``)."""
    adapter.submit_store_task([key], [obj])
    assert _wait_fd(adapter.get_store_event_fd()), "store event timed out"
    adapter.pop_completed_store_tasks()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_lru_setup():
    """Mock adapter + IsolatedLRU policy + QuotaManager wired through a
    controller — same topology as production, minus the background
    thread. Tests drive eviction synchronously via
    ``_check_and_evict`` so they don't race the 1-second loop."""
    adapter_config = MockL2AdapterConfig(
        max_size_gb=0.001,  # 1 MB — big enough for salted traffic
        mock_bandwidth_gb=10.0,
    )
    adapter = MockL2Adapter(adapter_config)

    policy = IsolatedLRUEvictionPolicy()
    # Wire the adapter → policy bridge so on_keys_* updates fire on
    # store / delete / access events. Normally StorageManager does
    # this; the test sets up its own bridge so the controller sees
    # accurate per-bucket LRU state.
    adapter.register_listener(L2EvictionPolicy(policy))

    eviction_config = EvictionConfig(
        eviction_policy="IsolatedLRU",
        trigger_watermark=0.8,
        eviction_ratio=0.5,
    )
    state = L2AdapterEvictionState(
        adapter_id=0, adapter=adapter, eviction_config=eviction_config
    )
    # L2AdapterEvictionState creates its own policy internally; swap in
    # the one we made to observe it from the test side. The listener
    # it attached is stale, but the policy reference the controller
    # uses is the one on ``state``.
    state.eviction_policy = policy

    qm = QuotaManager()
    controller = L2EvictionController([state], quota_manager=qm)
    # Don't call ``controller.start()`` — the test drives
    # ``_check_and_evict`` directly so timing is deterministic.

    yield adapter, policy, qm, controller, state
    adapter.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOverQuotaEviction:
    def test_over_budget_salt_gets_evicted(self, isolated_lru_setup):
        """Alice exceeds 50% of her 2 KB quota (watermark=0.8) — the
        controller evicts enough of her keys to bring her below."""
        adapter, policy, qm, controller, state = isolated_lru_setup

        # 2 KB quota; watermark 0.8 → threshold 1.6 KB; eviction
        # ratio 0.5 → evict half of alice's keys.
        qm.set_quota("alice", 2 * 1024)
        qm.set_quota("bob", 10 * 1024)  # plenty of headroom

        for i in range(4):
            _store_sync(adapter, _key(i, "alice"), _memory_obj())
        _store_sync(adapter, _key(100, "bob"), _memory_obj())

        before = adapter.get_usage().bytes_by_cache_salt
        assert before["alice"] > 0.8 * (2 * 1024)
        assert before["bob"] < 0.8 * (10 * 1024)

        controller._check_and_evict(state)

        after = adapter.get_usage().bytes_by_cache_salt
        assert after.get("alice", 0) < before["alice"], (
            "alice's bytes should have dropped"
        )
        assert after["bob"] == before["bob"], (
            "bob was within quota — nothing should have moved"
        )

    def test_unregistered_salt_gets_wiped(self, isolated_lru_setup):
        """A salt with no quota entry is fully evicted on the next
        cycle (allowlist semantics)."""
        adapter, policy, qm, controller, state = isolated_lru_setup

        # Only alice is registered.
        qm.set_quota("alice", 10 * 1024)

        _store_sync(adapter, _key(1, "alice"), _memory_obj())
        _store_sync(adapter, _key(2, "stranger"), _memory_obj())
        _store_sync(adapter, _key(3, "stranger"), _memory_obj())

        controller._check_and_evict(state)

        after = adapter.get_usage().bytes_by_cache_salt
        assert "stranger" not in after, (
            "unregistered salt should have been fully evicted"
        )
        assert after.get("alice", 0) > 0, "alice was under quota"

    def test_under_budget_does_nothing(self, isolated_lru_setup):
        """Nobody is over threshold → no eviction, accounting is
        stable across the cycle."""
        adapter, policy, qm, controller, state = isolated_lru_setup
        qm.set_quota("alice", 10 * 1024)

        _store_sync(adapter, _key(1, "alice"), _memory_obj())
        before = dict(adapter.get_usage().bytes_by_cache_salt)

        controller._check_and_evict(state)

        after = dict(adapter.get_usage().bytes_by_cache_salt)
        assert after == before


class TestSaltIsolation:
    def test_alice_over_quota_does_not_touch_bob(self, isolated_lru_setup):
        """Scoped eviction — only alice's keys are candidates, even
        though bob has some keys that are globally older."""
        adapter, policy, qm, controller, state = isolated_lru_setup

        qm.set_quota("alice", 2 * 1024)
        qm.set_quota("bob", 10 * 1024)

        # bob goes first → bob's keys are older globally.
        for i in range(2):
            _store_sync(adapter, _key(100 + i, "bob"), _memory_obj())
        for i in range(4):
            _store_sync(adapter, _key(i, "alice"), _memory_obj())

        controller._check_and_evict(state)

        after = adapter.get_usage().bytes_by_cache_salt
        assert after["bob"] == 2 * 512, (
            "bob should still have both keys — scoped eviction "
            "shouldn't leak into his bucket"
        )


class TestPolicyDispatch:
    def test_lru_policy_ignores_quota_manager(self, isolated_lru_setup):
        """Sanity check: even if a QuotaManager is wired in, a
        non-isolation policy falls through to the aggregate
        watermark branch. Here that branch is a no-op (adapter well
        under global capacity)."""
        adapter, policy, qm, controller, state = isolated_lru_setup

        # Swap the state's policy for one without isolation support.
        # First Party
        from lmcache.v1.distributed.eviction_policy.lru import (
            LRUEvictionPolicy,
        )

        state.eviction_policy = LRUEvictionPolicy()
        adapter.register_listener(L2EvictionPolicy(state.eviction_policy))

        # Set a quota that would normally trigger eviction.
        qm.set_quota("alice", 1)  # 1 byte limit

        _store_sync(adapter, _key(1, "alice"), _memory_obj())
        before = dict(adapter.get_usage().bytes_by_cache_salt)
        controller._check_and_evict(state)
        after = dict(adapter.get_usage().bytes_by_cache_salt)

        # LRU policy doesn't look at the quota → under aggregate
        # watermark → no eviction.
        assert after == before

    def test_isolated_lru_without_quota_manager_falls_through_to_global(
        self, isolated_lru_setup
    ):
        """Defensive: IsolatedLRU configured on a controller that wasn't
        given a QuotaManager falls through to the aggregate-watermark
        branch. This path is unreachable through normal ``StorageManager``
        wiring (which always creates a QuotaManager) but exists as a
        safety fallback."""
        adapter, policy, _qm, _controller, state = isolated_lru_setup

        # Build a fresh controller without a QuotaManager.
        no_qm_controller = L2EvictionController([state], quota_manager=None)

        _store_sync(adapter, _key(1, "alice"), _memory_obj())
        before = dict(adapter.get_usage().bytes_by_cache_salt)

        # Even with IsolatedLRU policy, no quota_manager → global branch.
        # Adapter is well under capacity → no eviction.
        no_qm_controller._check_and_evict(state)

        after = dict(adapter.get_usage().bytes_by_cache_salt)
        assert after == before
