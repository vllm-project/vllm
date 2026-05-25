# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the eager-cache-registration zombie problem.
"""

from __future__ import annotations

import pytest

from tests.v1.core.test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_config_hybrid_model,
    make_request,
)
from tests.v1.core.utils import create_scheduler
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


# ---------------------------------------------------------------------------
# Test 1: full-attention single-group case
# ---------------------------------------------------------------------------


def test_rollback_on_preempt_before_worker_write():
    """
    Schedule req_A then ``free`` it immediately (simulating preempt/abort
    before the worker has executed). The eager-registered hash entries
    must be evicted from the cache map and the blocks' ``block_hash``
    must be reset, so that a later req_B with the same prefix does NOT
    cache-hit on never-written blocks.
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

    # After allocate_slots: 2 hashes registered eagerly and both tracked
    # in ``_uncommitted[req_a]``.
    cache_map = manager.block_pool.cached_block_hash_to_block._cache
    assert len(cache_map) == 2, (
        "Sanity: 2 full blocks should each register a hash on allocate_slots"
    )
    assert len(manager.block_pool._uncommitted["a"]) == 2, (
        "Sanity: both eager registrations should be tracked as uncommitted "
        "until commit_step or rollback runs."
    )

    block_ids_a = blocks_a.get_block_ids()[0]
    eager_blocks = [manager.block_pool.blocks[bid] for bid in block_ids_a]

    # --- Simulate preempt/abort BEFORE worker writes K/V bytes -----------
    # In real life the worker would normally run between allocate_slots and
    # the next free(). Here we go directly from allocate_slots → free
    # *without* an intervening ``commit_step``, mirroring the race window
    # where the request is preempted while the worker has not yet executed
    # for this step.
    manager.free(req_a)

    # --- Rollback evicts the uncommitted entries -------------------------
    assert len(cache_map) == 0, (
        "free(uncommitted req) must evict both eager hash entries; got "
        f"{len(cache_map)} left in cache map."
    )
    assert "a" not in manager.block_pool._uncommitted, (
        "_uncommitted[req_a] must be cleared by rollback."
    )
    for blk in eager_blocks:
        assert blk.ref_cnt == 0, "free() did decrement ref_cnt"
        assert blk.block_hash is None, (
            "block.block_hash must be reset by rollback path."
        )

    # --- req_B: same prefix → cache MISS ---------------------------------
    # 48 tokens: first 32 would match req_a's now-rolled-back hashes, last
    # 16 are new. Without zombies, this is a full cold miss.
    req_b_tokens = common_token_ids + [99] * block_size
    req_b = make_request("b", req_b_tokens, block_size, sha256)
    _, num_computed_b = manager.get_computed_blocks(req_b)

    assert num_computed_b == 0, (
        f"req_b must not cache-hit on rolled-back entries; got "
        f"{num_computed_b} cached tokens, expected 0."
    )


# ---------------------------------------------------------------------------
# Test 2: hybrid (full-attention + Mamba) case
# ---------------------------------------------------------------------------


def test_rollback_on_preempt_for_mamba_hybrid():
    """
    Same race as test 1, but for full-attention + 2 Mamba groups. Confirms
    the rollback path covers every manager that calls ``cache_blocks``
    during ``allocate_slots``, not just full-attention.
    """
    block_size = 16
    # 1 full-attention group + 2 Mamba groups (slice 0/1).
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
    assert eager_entry_count > 0, (
        "Sanity: hybrid manager registers at least some eager entries on alloc"
    )
    assert len(manager.block_pool._uncommitted["a"]) == eager_entry_count, (
        "Sanity: every eager registration across all groups should be tracked "
        "as uncommitted."
    )

    # Preempt before worker write.
    manager.free(req_a)

    assert len(cache_map) == 0, (
        f"Rollback must clear all groups' eager entries; got "
        f"{len(cache_map)} left in the cache map."
    )
    assert "a" not in manager.block_pool._uncommitted

    # Future request cache MISS — no entries left to match.
    req_b = make_request("b", common_token_ids, block_size, sha256)
    _, num_computed_b = manager.get_computed_blocks(req_b)
    assert num_computed_b == 0, (
        f"req_b must not cache-hit on rolled-back entries; got "
        f"{num_computed_b} cached tokens, expected 0."
    )


# ---------------------------------------------------------------------------
# Test 3: control case — successful step + free should keep cache (no bug)
# ---------------------------------------------------------------------------


def test_commit_step_keeps_cache_across_normal_free():
    """
    Control: when a request runs the worker to completion (modelled here
    by an explicit ``commit_step()`` call) and is then freed normally,
    the cache entries SHOULD remain. ``commit_step`` clears
    ``_uncommitted`` so the subsequent ``free`` does not roll anything
    back.

    Without the ``commit_step`` call, ``free`` would correctly roll the
    entries back — that's exactly the eager-rollback behavior exercised
    by tests 1 and 2 above. This test pins down the other side: the
    commit hook is what preserves cache hits across normal completion.
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
    assert len(manager.block_pool._uncommitted["a"]) == 2

    # Simulate successful worker execution by committing the step.
    # Scheduler.schedule() calls this at the start of every step; here we
    # call it manually to model "previous step's worker has confirmed".
    manager.commit_step()
    assert manager.block_pool._uncommitted == {}, (
        "commit_step should clear all pending uncommitted registrations."
    )

    manager.free(req_a)

    cache_map = manager.block_pool.cached_block_hash_to_block._cache
    assert len(cache_map) == 2, (
        "Normal completion (commit_step + free) should leave the cache "
        f"entries intact, got {len(cache_map)} entries."
    )


# ---------------------------------------------------------------------------
# Test 4: scheduler-level preemption test
# ---------------------------------------------------------------------------


def test_scheduler_preempt_rolls_back_target_step_eager_cache():
    """
    Exercise the rollback through a real ``Scheduler.schedule()`` call.

    Setup seeds two RUNNING chunked-prefill requests with one already-committed
    block each. In the target schedule call, req_A is scheduled first and
    eagerly caches two new full blocks. req_B then cannot allocate, so priority
    preemption removes req_A from the same scheduler output before any worker
    can see or execute that work.

    The two target-step entries must be evicted on preempt; the prior-step
    seeded block remains in the cache map (it was promoted to committed when
    ``schedule`` called ``commit_step`` at the start of this step). A future
    req_C with req_A's first two blocks as prefix therefore hits exactly the
    one seeded block.
    """
    block_size = 16
    # Resource budget is intentionally tight to force the exact race:
    #   max_num_batched_tokens=48 = 32 (req_a's 2 remaining blocks)
    #                             + 16 (req_b's 1 remaining block).
    #     Both requests *want* to advance fully in the target step.
    #   num_blocks=5 = 1 null
    #                + 1 seeded committed block for req_a
    #                + 1 seeded committed block for req_b
    #                + 2 free blocks.
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
    # computed. This bypasses the normal admission path so we start the test
    # mid-chunked-prefill (the only state where this race opens) without
    # having to drive multiple scheduler steps to set it up.
    manager = scheduler.kv_cache_manager
    for req in (req_a, req_b):
        seeded_blocks = manager.allocate_slots(req, block_size)
        assert seeded_blocks is not None
        req.num_computed_tokens = block_size
        req.status = RequestStatus.RUNNING
        scheduler.requests[req.request_id] = req
    scheduler.running = [req_a, req_b]

    # Sanity: post-seeding state matches the budget plan above.
    cache_map = manager.block_pool.cached_block_hash_to_block._cache
    assert len(cache_map) == 2  # one committed hash per seeded request.
    assert manager.block_pool.get_num_free_blocks() == 2  # exactly the race window.

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
    # commit_step at the start of this scheduler step). The target-step
    # eager entry for the would-be second block was rolled back on preempt.
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
