# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the eager-cache-registration zombie protection.
"""

from __future__ import annotations

import pytest

from tests.v1.core.test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_config_hybrid_model,
    make_request,
)
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _uncommitted_blocks_for(manager: KVCacheManager, req_id: str) -> list:
    """Flatten all uncommitted buckets and return req_id's tracked blocks."""
    out: list = []
    for bucket in manager.block_pool._uncommitted:
        out.extend(bucket.get(req_id, []))
    return out


# ---------------------------------------------------------------------------
# Test 1: full-attention single-group case
# ---------------------------------------------------------------------------


def test_rollback_uncommitted_evicts_preempt_zombies():
    """
    Schedule req_A, then explicitly invoke the preempt cleanup
    (``rollback_uncommitted`` + ``free``). The eager-registered hash entries
    must be evicted from the cache map and the blocks' ``block_hash`` must be
    reset, so that a later req_B with the same prefix does NOT cache-hit on
    never-written blocks.
    """
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    # --- req_A: 32 tokens = 2 full blocks --------------------------------
    common_token_ids = [i for i in range(2) for _ in range(block_size)]
    req_a = make_request("a", common_token_ids, block_size, sha256)
    computed_a, num_computed_a = manager.get_computed_blocks(req_a)
    assert num_computed_a == 0  # cold cache
    blocks_a = manager.allocate_slots(req_a, 32, 0, computed_a)
    assert blocks_a is not None

    # After allocate_slots: 2 hashes registered eagerly, both tracked in the
    # current step's bucket.
    cache_map = manager.block_pool.cached_block_hash_to_block._cache
    assert len(cache_map) == 2
    assert len(_uncommitted_blocks_for(manager, "a")) == 2

    block_ids_a = blocks_a.get_block_ids()[0]
    eager_blocks = [manager.block_pool.blocks[bid] for bid in block_ids_a]

    # --- Simulate preempt before worker write ----------------------------
    # The preempt/abort code paths invoke ``rollback_uncommitted`` *before*
    # ``free`` so that any hash entries the worker hasn't yet backed with
    # K/V bytes are evicted from the cache map.
    manager.rollback_uncommitted(req_a.request_id)
    manager.free(req_a)

    assert len(cache_map) == 0, (
        f"rollback must evict both eager hash entries, got {len(cache_map)}."
    )
    assert _uncommitted_blocks_for(manager, "a") == []
    for blk in eager_blocks:
        assert blk.ref_cnt == 0, "free() did decrement ref_cnt"
        assert blk.block_hash is None, (
            "block.block_hash must be reset by rollback path."
        )

    # --- req_B: superset prefix → cache MISS -----------------------------
    req_b_tokens = common_token_ids + [99] * block_size
    req_b = make_request("b", req_b_tokens, block_size, sha256)
    _, num_computed_b = manager.get_computed_blocks(req_b)
    assert num_computed_b == 0, (
        f"req_b must not cache-hit on rolled-back entries; got "
        f"{num_computed_b} cached tokens."
    )


# ---------------------------------------------------------------------------
# Test 2: hybrid (full-attention + Mamba) case
# ---------------------------------------------------------------------------


def test_rollback_uncommitted_covers_mamba_hybrid_groups():
    """
    Same race as test 1, but for full-attention + 2 Mamba groups. Confirms
    the rollback path covers every manager that calls ``cache_blocks``
    during ``allocate_slots``, not just full-attention.
    """
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config_hybrid_model(
            block_size,
            num_blocks=16,
            sliding_window_blocks=0,
            second_spec_type="mamba",
        ),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    common_token_ids = [i for i in range(2) for _ in range(block_size)]
    req_a = make_request("a", common_token_ids, block_size, sha256)
    computed_a, _ = manager.get_computed_blocks(req_a)
    blocks_a = manager.allocate_slots(req_a, 32, 0, computed_a)
    assert blocks_a is not None

    cache_map = manager.block_pool.cached_block_hash_to_block._cache
    eager_entry_count = len(cache_map)
    assert eager_entry_count > 0
    assert len(_uncommitted_blocks_for(manager, "a")) == eager_entry_count

    # Preempt before worker write.
    manager.rollback_uncommitted(req_a.request_id)
    manager.free(req_a)

    assert len(cache_map) == 0, (
        f"rollback must clear all groups' eager entries; got {len(cache_map)}."
    )
    assert _uncommitted_blocks_for(manager, "a") == []

    req_b = make_request("b", common_token_ids, block_size, sha256)
    _, num_computed_b = manager.get_computed_blocks(req_b)
    assert num_computed_b == 0


# ---------------------------------------------------------------------------
# Test 3: normal-finish path keeps cache entries (no rollback called)
# ---------------------------------------------------------------------------


def test_finish_path_free_alone_preserves_cache_entries():
    """
    Control: the normal-finish path calls only ``free()`` (no rollback).
    Cache entries the worker has confirmed must remain hittable for future
    requests, with or without an intervening ``commit_step``.
    """
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    common_token_ids = [i for i in range(2) for _ in range(block_size)]
    req_a = make_request("a", common_token_ids, block_size, sha256)
    computed_a, _ = manager.get_computed_blocks(req_a)
    manager.allocate_slots(req_a, 32, 0, computed_a)
    assert len(_uncommitted_blocks_for(manager, "a")) == 2

    # ``free`` on its own must not touch the cache map. The pending bucket
    # entries get cleaned up later when the matching commit_step runs (or
    # stay tracked but harmless until then).
    manager.free(req_a)

    cache_map = manager.block_pool.cached_block_hash_to_block._cache
    assert len(cache_map) == 2, (
        "Normal free should leave the cache entries intact for future hits."
    )

    # commit_step pops the oldest bucket. After that, no rollback can affect
    # these entries even if rollback_uncommitted were called.
    manager.commit_step()
    assert _uncommitted_blocks_for(manager, "a") == []
    assert len(cache_map) == 2


# ---------------------------------------------------------------------------
# Test 4: committed=True kwarg skips _uncommitted tracking
# ---------------------------------------------------------------------------


def test_cache_blocks_committed_kwarg_skips_uncommitted_tracking():
    """
    When ``cache_blocks(committed=True)`` is called from a path that knows the
    worker has already confirmed the K/V writes (AsyncScheduler post-output,
    KV connector load completion), the new cache entries must NOT be tracked
    in ``_uncommitted``. A subsequent ``rollback_uncommitted`` for the same
    request must leave those entries intact in ``cached_block_hash_to_block``.
    """
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    tokens = [i for i in range(2) for _ in range(block_size)]
    req = make_request("a", tokens, block_size, sha256)
    computed, _ = manager.get_computed_blocks(req)
    blocks = manager.allocate_slots(req, 32, 0, computed)
    assert blocks is not None

    cache_map = manager.block_pool.cached_block_hash_to_block._cache
    assert len(cache_map) == 2
    # The eager (default) path tracks the 2 new registrations.
    assert len(_uncommitted_blocks_for(manager, "a")) == 2

    # Now drop the eager tracking by committing the step, then invoke a
    # committed-mode cache_blocks call (mimicking the AsyncScheduler/connector
    # post-confirm paths). It must register into cache_map but NOT add to
    # _uncommitted.
    manager.commit_step()
    assert _uncommitted_blocks_for(manager, "a") == []

    # Construct a longer prompt request with the same first 32 tokens; calling
    # cache_blocks(committed=True) for those tokens should be a no-op (already
    # cached) but importantly must not poison _uncommitted.
    tokens_longer = tokens + [99] * block_size
    req_longer = make_request("b", tokens_longer, block_size, sha256)
    computed_longer, num_computed_longer = manager.get_computed_blocks(req_longer)
    assert num_computed_longer == 32  # hits the two committed blocks
    manager.allocate_slots(req_longer, 16, 32, computed_longer)

    # Now simulate a committed-path registration for req_longer (the bytes are
    # already worker-confirmed for the suffix). This should NOT add to
    # _uncommitted["b"].
    before = len(_uncommitted_blocks_for(manager, "b"))
    manager.cache_blocks(req_longer, 48, committed=True)
    after = len(_uncommitted_blocks_for(manager, "b"))
    assert after == before, (
        f"committed=True must not grow _uncommitted; before={before}, after={after}"
    )

    # And subsequent rollback for req_longer must not evict any of the
    # committed entries from cache_map.
    cache_map_before_rollback = dict(cache_map)
    manager.rollback_uncommitted(req_longer.request_id)
    # The cache_map should still contain at least the two original committed
    # entries (req_a's prefix). The eager block(s) for req_longer's suffix
    # would be evicted by rollback, which is expected.
    for key, blk in cache_map_before_rollback.items():
        if blk in [manager.block_pool.blocks[bid] for bid in blocks.get_block_ids()[0]]:
            assert key in cache_map, "rollback evicted a committed entry from cache_map"


# ---------------------------------------------------------------------------
# Test 5: reset_prefix_cache also clears _uncommitted
# ---------------------------------------------------------------------------


def test_reset_prefix_cache_clears_uncommitted():
    """
    ``reset_prefix_cache`` wipes ``cached_block_hash_to_block`` and resets every
    block's ``block_hash``. ``_uncommitted`` would dangle if not cleared
    alongside — pointing at blocks whose hashes have been reset to None — so
    a subsequent ``rollback_uncommitted`` call would walk stale entries and
    potentially evict blocks that have been re-allocated to other requests.

    Confirm reset zeroes ``_uncommitted`` too.
    """
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    tokens = [i for i in range(2) for _ in range(block_size)]
    req = make_request("a", tokens, block_size, sha256)
    computed, _ = manager.get_computed_blocks(req)
    manager.allocate_slots(req, 32, 0, computed)

    # Sanity: eager registration populated _uncommitted.
    assert len(_uncommitted_blocks_for(manager, "a")) == 2
    assert len(manager.block_pool._uncommitted) == 1

    # Free the request so reset_prefix_cache succeeds (it requires zero
    # outstanding blocks except the null block).
    manager.free(req)
    assert manager.reset_prefix_cache()

    # Both the cache map and the uncommitted queue must be empty.
    assert len(manager.block_pool.cached_block_hash_to_block) == 0
    assert len(manager.block_pool._uncommitted) == 0, (
        "reset_prefix_cache must clear _uncommitted; otherwise stale entries "
        "point at blocks whose hashes have been reset and a subsequent "
        "rollback_uncommitted could touch the wrong physical blocks."
    )

    # Sanity follow-through: a fresh request after reset goes through the
    # normal eager-registration path without surprises.
    req2 = make_request("b", tokens, block_size, sha256)
    computed2, _ = manager.get_computed_blocks(req2)
    assert _ == 0  # cache was cleared by reset
    manager.allocate_slots(req2, 32, 0, computed2)
    assert len(_uncommitted_blocks_for(manager, "b")) == 2


# ---------------------------------------------------------------------------
# Test 6: FIFO bucket discipline across multiple in-flight scheduler steps
# ---------------------------------------------------------------------------


def test_fifo_bucket_discipline_under_multi_in_flight_steps():
    """
    Drive ``Scheduler.schedule()`` twice without an intervening
    ``update_from_output``, simulating the async batch-queue path. Each
    schedule must push a bucket via ``begin_step``; ``update_from_output``
    for the OLDEST step must pop only that bucket via ``commit_step``,
    leaving the newer in-flight bucket intact and its entries available for
    ``rollback_uncommitted`` if that newer step's request is later aborted.
    """
    # async_scheduling + PP>=2 lets the scheduler keep multiple in-flight
    # schedule() calls outstanding before any update_from_output fires —
    # the precise condition that exposes the FIFO discipline.
    # enable_prefix_caching is needed for cache_blocks (and therefore the
    # eager registrations we want to track) to fire at all.
    scheduler = create_scheduler(
        async_scheduling=True,
        pipeline_parallel_size=2,
        enable_prefix_caching=True,
    )
    manager = scheduler.kv_cache_manager
    cache_map = manager.block_pool.cached_block_hash_to_block._cache

    # Two requests with full-block prompts so each schedule() actually
    # registers eager cache entries (block_size=16 default → 32 tokens =
    # 2 full blocks per request).
    reqs = create_requests(num_requests=2, num_tokens=32)

    assert len(manager.block_pool._uncommitted) == 0

    # --- Step N: admit and schedule req_a --------------------------------
    scheduler.add_request(reqs[0])
    output_n = scheduler.schedule()
    assert len(manager.block_pool._uncommitted) == 1, (
        "schedule() must push a bucket via begin_step"
    )
    assert reqs[0].request_id in manager.block_pool._uncommitted[0]
    entries_after_n = len(cache_map)
    assert entries_after_n > 0

    # --- Step N+1: admit and schedule req_b WITHOUT update_from_output ---
    scheduler.add_request(reqs[1])
    scheduler.schedule()
    assert len(manager.block_pool._uncommitted) == 2, (
        "second schedule() must push another bucket while the first remains"
    )
    # Newer bucket carries req_b's eager registrations.
    assert reqs[1].request_id in manager.block_pool._uncommitted[-1]
    # Older bucket still carries req_a's, unchanged.
    assert reqs[0].request_id in manager.block_pool._uncommitted[0]

    # --- Worker N completes; update_from_output(N) commits oldest bucket -
    req_ids_n = list(output_n.num_scheduled_tokens.keys())
    scheduler.update_from_output(
        output_n,
        ModelRunnerOutput(
            req_ids=req_ids_n,
            req_id_to_index={r: i for i, r in enumerate(req_ids_n)},
            sampled_token_ids=[[] for _ in req_ids_n],  # still prefilling
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert len(manager.block_pool._uncommitted) == 1, (
        "update_from_output must commit only the oldest bucket via commit_step"
    )
    assert reqs[0].request_id not in manager.block_pool._uncommitted[0], (
        "commit_step must remove req_a's tracking; req_a's entries stay in "
        "cache_map as committed."
    )
    assert reqs[1].request_id in manager.block_pool._uncommitted[0]

    # --- Abort req_b mid-flight (its worker batch has NOT confirmed) -----
    # rollback must reach into the still-pending bucket and evict req_b's
    # entries; req_a's committed entries are untouched.
    manager.rollback_uncommitted(reqs[1].request_id)
    manager.free(reqs[1])
    assert _uncommitted_blocks_for(manager, reqs[1].request_id) == []
    assert len(cache_map) < entries_after_n + 1, (
        "rollback must evict req_b's eager entries from cache_map"
    )
    # req_a's entries (committed) still hittable.
    assert _uncommitted_blocks_for(manager, reqs[0].request_id) == []
    rollback_a = manager.rollback_uncommitted(reqs[0].request_id)
    assert rollback_a == 0, "rollback after commit must be a no-op for req_a"


# ---------------------------------------------------------------------------
# Test 5: scheduler-level preemption test (real Scheduler.schedule() path)
# ---------------------------------------------------------------------------


def test_scheduler_preempt_rolls_back_target_step_eager_cache():
    """
    Exercise the rollback through a real ``Scheduler.schedule()`` call.

    Setup seeds two RUNNING chunked-prefill requests with one already-
    committed block each. In the target schedule call, req_A is scheduled
    first and eagerly caches two new full blocks. req_B then cannot
    allocate, so priority preemption removes req_A from the same scheduler
    output before any worker can see or execute that work.

    The preempt path (`_preempt_request`) calls ``rollback_uncommitted``
    before ``free``, evicting the two target-step entries. The prior-step
    seeded block remains in the cache map. A future req_C with req_A's
    first two blocks as prefix therefore hits exactly the one seeded block.
    """
    block_size = 16
    # Resource budget is intentionally tight to force the exact race:
    #   max_num_batched_tokens=48 = 32 (req_a's 2 remaining blocks)
    #                             + 16 (req_b's 1 remaining block).
    #   num_blocks=5 = 1 null + 2 seeded committed + 2 free.
    #     The 2 free blocks are exactly enough for req_a's target-step
    #     allocation (2 new full blocks); req_b then has 0 free → triggers
    #     preempt of the lowest-priority running request, which is req_a.
    scheduler = create_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=48,
        max_model_len=8192,
        enable_prefix_caching=True,
        num_blocks=5,
        block_size=block_size,
    )
    # PRIORITY policy is required: under FCFS req_a would not be preempted
    # for req_b, and the race window we want to test would not open.
    # ``create_scheduler`` does not expose ``policy`` directly, so swap it
    # post-construction and rebuild the queues that depend on the policy.
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)

    # 48-token prompt = 3 full blocks; 1 seeded as committed, 2 to be
    # eagerly cached in the target step (these become the zombies).
    tokens_a = [0] * block_size + [1] * block_size + [2] * block_size
    # 32-token prompt = 2 full blocks; 1 seeded as committed, 1 needed in
    # the target step (this is what fails to allocate and triggers preempt).
    tokens_b = [10] * block_size + [11] * block_size
    req_a = make_request("a", tokens_a, block_size, sha256)
    req_b = make_request("b", tokens_b, block_size, sha256)
    # In vLLM priority semantics, smaller value = higher priority. req_b
    # outranks req_a, so when free blocks run out, req_a is the one
    # preempted to make room for req_b.
    req_a.priority = 5
    req_b.priority = 0
    req_a.arrival_time = 1.0
    req_b.arrival_time = 2.0

    # Plant both requests directly in RUNNING with their first block already
    # computed. This bypasses the normal admission path so we start the
    # test mid-chunked-prefill (the only state where this race opens)
    # without having to drive multiple scheduler steps to set it up.
    manager = scheduler.kv_cache_manager
    for req in (req_a, req_b):
        seeded_blocks = manager.allocate_slots(req, block_size)
        assert seeded_blocks is not None
        req.num_computed_tokens = block_size
        req.status = RequestStatus.RUNNING
        scheduler.requests[req.request_id] = req
    scheduler.running = [req_a, req_b]

    # The seed allocations registered hashes in the BlockPool's first lazy
    # bucket. Production puts seed entries in a *prior* committed step, not
    # the current step. Commit here to model that.
    manager.commit_step()

    # Sanity: post-seeding state matches the budget plan above.
    cache_map = manager.block_pool.cached_block_hash_to_block._cache
    assert len(cache_map) == 2  # one committed hash per seeded request.
    assert manager.block_pool.get_num_free_blocks() == 2  # the race window.

    output = scheduler.schedule()

    assert output.preempted_req_ids == {"a"}
    assert "a" not in output.num_scheduled_tokens, (
        "Sanity: req_a was removed from the scheduler output before worker "
        "execution, so its target-step KV writes cannot happen."
    )
    assert output.num_scheduled_tokens == {"b": block_size}
    assert req_a.status == RequestStatus.PREEMPTED

    # A future request with A's first two blocks as a prefix should hit
    # exactly one block: the prior-step seeded block (committed via
    # commit_step before this scheduler step). The target-step eager entry
    # for the would-be second block was rolled back on preempt.
    req_c = make_request(
        "c",
        tokens_a[: 2 * block_size] + [99] * block_size,
        block_size,
        sha256,
    )
    computed_c, num_computed_c = manager.get_computed_blocks(req_c)
    assert num_computed_c == block_size, (
        "Scheduler-level preemption must roll back the target-step eager "
        "cache entry; only the prior-step committed block should hit. Got "
        f"{num_computed_c} cached tokens, expected {block_size}."
    )
    hit_blocks = computed_c.blocks[0]
    assert len(hit_blocks) == 1, (
        f"Expected 1 hit (seeded committed block), got {len(hit_blocks)}."
    )
