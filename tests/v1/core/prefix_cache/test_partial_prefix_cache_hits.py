# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fine-grained partial prefix-cache hits for hybrid (full attention + mamba
"align") models: scheduler chunk splitting, partial tail registration, CoW
on partial hits, and same-step deferral."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlockCopy,
    get_block_hash,
    get_group_id,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCachePoolConfig,
    MambaSpec,
    MemoryModel,
)


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def test_connector_without_divergent_hit_support_uses_common_lookup():
    common_blocks = MagicMock()
    manager = MagicMock()
    manager.get_computed_blocks.return_value = (common_blocks, 0, 0)
    scheduler = SimpleNamespace(
        connector=SimpleNamespace(supports_divergent_local_hybrid_hits=False),
        kv_cache_manager=manager,
    )

    result = Scheduler._get_local_prefix_cache_hit(scheduler, MagicMock())

    assert result == (common_blocks, 0, 0, False)
    manager.get_computed_blocks_for_connector.assert_not_called()


def test_capable_connector_uses_divergent_partial_hit_lookup():
    per_group_blocks = MagicMock()
    manager = MagicMock()
    manager.get_computed_blocks_for_connector.return_value = (
        per_group_blocks,
        6,
        0,
        True,
    )
    scheduler = SimpleNamespace(
        connector=SimpleNamespace(supports_divergent_local_hybrid_hits=True),
        kv_cache_manager=manager,
    )

    result = Scheduler._get_local_prefix_cache_hit(scheduler, MagicMock())

    assert result == (per_group_blocks, 6, 0, True)
    manager.get_computed_blocks.assert_not_called()


def test_mamba_align_split_partial_tail_schedule():
    """Chunk ends with partial hits on: block-aligned chunks, one extra stop
    at the prompt's last hash boundary (registering the partial tail), then
    the remaining tokens. block=512, hash=32, prompt=10000, budget=8192:
    0 -> 8192 -> 9728 -> 9984 -> 10000."""
    block_size = 512
    hash_block_size = 32
    mock = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=False,
        hash_block_size=hash_block_size,
        mamba_partial_cache_hit=True,
    )
    split = Scheduler._mamba_block_aligned_split

    req = make_request("0", [0] * 10000, hash_block_size, sha256)
    req.num_computed_tokens = 0
    assert split(self=mock, request=req, num_new_tokens=8192) == 8192
    req.num_computed_tokens = 8192
    # Stop at the last block boundary (9728).
    assert split(self=mock, request=req, num_new_tokens=1808) == 1536
    req.num_computed_tokens = 9728
    # Extra stop at the prompt's last hash boundary (9984).
    assert split(self=mock, request=req, num_new_tokens=272) == 256
    req.num_computed_tokens = 9984
    # Final 16 tokens run unchanged (no mid-block-resume stop: the next
    # block boundary is past the last block boundary).
    assert split(self=mock, request=req, num_new_tokens=16) == 16

    # Partial hits off: no extra stop, the tail runs in one chunk.
    mock.mamba_partial_cache_hit = False
    req.num_computed_tokens = 9728
    assert split(self=mock, request=req, num_new_tokens=272) == 272
    mock.mamba_partial_cache_hit = True

    # A request resumed mid-block (partial hash hit at 9984): the first chunk
    # stops at the next block boundary (10240), later chunk ends re-align.
    req2 = make_request("1", [0] * 12000, hash_block_size, sha256)
    req2.num_computed_tokens = 9984
    assert split(self=mock, request=req2, num_new_tokens=2016) == 256
    req2.num_computed_tokens = 10240
    assert split(self=mock, request=req2, num_new_tokens=1000) == 512


def test_mamba_align_split_when_block_exceeds_scheduling_budget():
    """Sub-block chunks make progress only when no step can fit a full block."""
    block_size = 11392
    token_budget = 8192
    prompt_length = 30000
    mock = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        max_num_scheduled_tokens=token_budget,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=False,
        hash_block_size=32,
        mamba_partial_cache_hit=False,
    )
    req = make_request("0", [0] * prompt_length, 32, sha256)
    split = Scheduler._mamba_block_aligned_split

    mock.max_num_scheduled_tokens = block_size
    assert split(self=mock, request=req, num_new_tokens=token_budget) == 0
    mock.max_num_scheduled_tokens = token_budget

    scheduled_chunks = []
    while req.num_computed_tokens < prompt_length:
        num_new_tokens = min(token_budget, prompt_length - req.num_computed_tokens)
        num_scheduled_tokens = split(
            self=mock,
            request=req,
            num_new_tokens=num_new_tokens,
        )
        assert 0 < num_scheduled_tokens <= token_budget
        scheduled_chunks.append(num_scheduled_tokens)
        req.num_computed_tokens += num_scheduled_tokens

    assert scheduled_chunks == [8192, 3200, 8192, 3200, 7216]


def test_mamba_align_split_when_block_exceeds_long_prefill_threshold():
    """A long-prefill cap below the block size permits sub-block progress."""
    block_size = 512
    token_budget = 8192
    long_prefill_threshold = 384
    prompt_length = 1300
    mock = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        max_num_scheduled_tokens=token_budget,
        scheduler_config=SimpleNamespace(
            long_prefill_token_threshold=long_prefill_threshold
        ),
        use_eagle=False,
        hash_block_size=32,
        mamba_partial_cache_hit=False,
    )
    req = make_request("0", [0] * prompt_length, 32, sha256)
    split = Scheduler._mamba_block_aligned_split

    scheduled_chunks = []
    while req.num_computed_tokens < prompt_length:
        num_new_tokens = min(
            long_prefill_threshold,
            prompt_length - req.num_computed_tokens,
        )
        num_scheduled_tokens = split(
            self=mock,
            request=req,
            num_new_tokens=num_new_tokens,
        )
        assert 0 < num_scheduled_tokens <= long_prefill_threshold
        scheduled_chunks.append(num_scheduled_tokens)
        req.num_computed_tokens += num_scheduled_tokens

    assert scheduled_chunks == [384, 128, 384, 128, 276]


def test_hybrid_mamba_align_partial_hash_hit():
    hash_block_size = 2
    mamba_block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=20,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    blocks = manager.allocate_slots(req0, 6, num_computed, computed_blocks)
    assert blocks is not None
    manager.free(req0)
    manager.new_step_starts()

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert partial_mamba_block is not None
    assert partial_mamba_block[0].block_hash_num_tokens == 6

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert [len(group) for group in computed_blocks.blocks] == [3, 2]

    new_blocks = manager.allocate_slots(req1, 2, num_computed, computed_blocks)
    assert new_blocks is not None
    mamba_new_block_ids = new_blocks.get_block_ids()[1]
    assert len(mamba_new_block_ids) == 1
    assert mamba_new_block_ids[0] != partial_mamba_block[0].block_id
    assert manager.get_blocks("1").get_block_ids()[1][1] == mamba_new_block_ids[0]
    assert partial_mamba_block[0].block_hash is not None
    assert get_block_hash(partial_mamba_block[0].block_hash) == partial_mamba_hash
    assert get_group_id(partial_mamba_block[0].block_hash) == 1
    assert partial_mamba_block[0].block_hash_num_tokens == 6
    copies, _ = manager.take_kv_cache_block_copies()
    assert (
        KVCacheBlockCopy(
            src_block_id=partial_mamba_block[0].block_id,
            dst_block_id=mamba_new_block_ids[0],
        )
        in copies
    )
    assert manager.get_blocks("1").blocks[1][1].block_hash_num_tokens == 8


def test_hybrid_mamba_partial_tail_owner_uses_cow_on_continue():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert partial_mamba_block is not None
    partial_mamba_block_id = partial_mamba_block[0].block_id
    assert manager.get_blocks("0").get_block_ids()[1][1] == partial_mamba_block_id

    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    new_blocks = manager.allocate_slots(req0, 1)
    assert new_blocks is not None

    # Reversed CoW for the owning request: it keeps its own block (the
    # worker's block table is append-only), and no new mamba block is handed
    # to the worker. The prefix-cache entry is moved to a private copy that
    # the queued block copy fills before the next forward.
    assert new_blocks.get_block_ids()[1] == []
    assert manager.get_blocks("0").get_block_ids()[1][1] == partial_mamba_block_id
    copies, _ = manager.take_kv_cache_block_copies()
    cow_copy = next(c for c in copies if c.src_block_id == partial_mamba_block_id)
    assert cow_copy.dst_block_id != partial_mamba_block_id
    # The source block gave up the hash; the copy target now owns the entry.
    assert partial_mamba_block[0].block_hash is None
    moved = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert moved is not None
    assert moved[0].block_id == cow_copy.dst_block_id
    assert get_block_hash(moved[0].block_hash) == partial_mamba_hash
    assert get_group_id(moved[0].block_hash) == 1
    assert moved[0].block_hash_num_tokens == 6


def test_external_mamba_hit_same_block_uses_running_cow_on_continue():
    """An external mid-block hit must become a running request even when its
    first continuation does not need another Mamba block."""
    hash_block_size = 2
    mamba_block_size = 4 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    request = make_request("0", [0] * 15, hash_block_size, sha256)
    loaded_blocks = manager.allocate_slots(
        request,
        num_new_tokens=0,
        num_external_computed_tokens=10,
        delay_cache_blocks=True,
    )
    assert loaded_blocks is not None

    request.num_computed_tokens = 10
    first_step_blocks = manager.allocate_slots(request, num_new_tokens=4)
    assert first_step_blocks is not None

    source_block_id = manager.get_blocks("0").get_block_ids()[1][1]
    partial_hash = request.block_hashes[14 // hash_block_size - 1]
    partial_block = manager.block_pool.get_cached_block(
        partial_hash, kv_cache_group_ids=[1]
    )
    assert partial_block is not None
    assert partial_block[0].block_id == source_block_id

    request.num_computed_tokens = 14
    continuation_blocks = manager.allocate_slots(request, num_new_tokens=1)
    assert continuation_blocks is not None

    assert continuation_blocks.get_block_ids()[1] == []
    assert manager.get_blocks("0").get_block_ids()[1][1] == source_block_id
    copies, _ = manager.take_kv_cache_block_copies()
    cow_copy = next(c for c in copies if c.src_block_id == source_block_id)
    assert cow_copy.dst_block_id != source_block_id

    moved = manager.block_pool.get_cached_block(partial_hash, kv_cache_group_ids=[1])
    assert moved is not None
    assert moved[0].block_id == cow_copy.dst_block_id


def test_take_partial_tail_offloads_returns_cow_target():
    """The connector offload hand-off exposes the mamba CoW *target* block Y
    (the durable boundary state), not the overwritten source X, and only at
    the CoW step."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    # Step A registered the partial tail but has not CoW'd yet: no offload.
    assert manager.take_partial_tail_offloads() == {}

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    source_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert source_block is not None
    source_block_id = source_block[0].block_id

    # Step B: the producer continues, triggering the CoW X->Y.
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None

    offloads = manager.take_partial_tail_offloads()
    assert list(offloads.keys()) == ["0"]
    assert len(offloads["0"]) == 1
    group_id, block_id, boundary_tokens = offloads["0"][0]
    assert group_id == 1  # the mamba group
    assert boundary_tokens == 6
    copies, _ = manager.take_kv_cache_block_copies()
    cow_copy = next(c for c in copies if c.src_block_id == source_block_id)
    # The offload points at the durable CoW target Y, not the overwritten X.
    assert block_id == cow_copy.dst_block_id
    assert block_id != source_block_id
    # Draining clears it.
    assert manager.take_partial_tail_offloads() == {}

    # The hand-off pinned Y (its CoW retention is released after this step,
    # and Y is off the request block table); freeing the request unpins it.
    cow_block = manager.block_pool.blocks[block_id]
    pinned_ref = cow_block.ref_cnt
    assert pinned_ref >= 1
    manager.free(req0)
    assert cow_block.ref_cnt == pinned_ref - 1


def test_partial_tail_pin_survives_released_cow_retention():
    """If the CoW retention is released before the hand-off is drained
    (immediate-free mode), the drain must rescue the cow block from the free
    queue: a raw ref increment would leave a ref>0 block allocatable, and the
    next allocation would pop it and assert."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None

    # Retention released before the drain (defer_block_free=False ordering).
    _copies, retained = manager.take_kv_cache_block_copies()
    manager.free_popped_blocks(retained)

    offloads = manager.take_partial_tail_offloads()
    ((_group_id, block_id, boundary_tokens),) = offloads["0"]
    assert boundary_tokens == 6
    cow_block = manager.block_pool.blocks[block_id]
    assert cow_block.ref_cnt == 1

    # The pinned block is out of the free queue: draining every free block
    # neither trips the allocator's ref_cnt assert nor hands it out.
    new_blocks = manager.block_pool.get_new_blocks(
        manager.block_pool.get_num_free_blocks()
    )
    assert block_id not in {b.block_id for b in new_blocks}


@pytest.mark.parametrize("deferred_free", [False, True])
def test_partial_tail_pin_uses_owning_block_pool(deferred_free):
    """A Mamba partial-tail pin must never touch or free the same block ID
    from another group's pool."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
        pool_configs=[
            KVCachePoolConfig(
                pool_id=0,
                memory_model=MemoryModel.TOKEN_PROPORTIONAL,
                group_ids=[0],
                num_blocks=24,
            ),
            KVCachePoolConfig(
                pool_id=1,
                memory_model=MemoryModel.REQUEST_CONSTANT,
                group_ids=[1],
                num_blocks=24,
            ),
        ],
        group_to_pool_id=[0, 1],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    full_pool, mamba_pool = manager.coordinator.block_pools
    initial_free_counts = (
        full_pool.get_num_free_blocks(),
        mamba_pool.get_num_free_blocks(),
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None

    (cow_copy,), retained = manager.take_kv_cache_block_copies()
    cow_block = mamba_pool.blocks[cow_copy.dst_block_id]
    same_id_other_pool_block = full_pool.blocks[cow_copy.dst_block_id]
    manager.free_popped_blocks(retained)
    assert cow_block.ref_cnt == 0
    other_pool_ref_cnt = same_id_other_pool_block.ref_cnt
    free_counts_before_pin = (
        full_pool.get_num_free_blocks(),
        mamba_pool.get_num_free_blocks(),
    )

    offloads = manager.take_partial_tail_offloads()
    ((group_id, block_id, boundary_tokens),) = offloads["0"]
    assert (group_id, block_id, boundary_tokens) == (
        1,
        cow_copy.dst_block_id,
        6,
    )
    assert cow_block.ref_cnt == 1
    assert same_id_other_pool_block.ref_cnt == other_pool_ref_cnt
    assert full_pool.get_num_free_blocks() == free_counts_before_pin[0]
    assert mamba_pool.get_num_free_blocks() == free_counts_before_pin[1] - 1

    if deferred_free:
        popped = manager.pop_blocks_for_free(req0)
        assert any(block is cow_block for block in popped.blocks[1])
        assert all(block is not cow_block for block in popped.blocks[0])
        manager.free_popped_blocks(popped)
    else:
        manager.free(req0)
    assert cow_block.ref_cnt == 0
    assert (
        full_pool.get_num_free_blocks(),
        mamba_pool.get_num_free_blocks(),
    ) == initial_free_counts


def test_partial_tail_offload_dropped_when_request_freed_before_drain():
    """A hand-off recorded in the same scheduling pass as the request's death
    must not be drained: its release hook has already run, so draining would
    leak a pinned block."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None

    # The request dies (preempt/abort) before the scheduler drains.
    manager.free_popped_blocks(manager.pop_blocks_for_free(req0))
    assert manager.take_partial_tail_offloads() == {}


def test_take_partial_tail_offloads_empty_without_partial_tail():
    """A prompt ending on a block boundary registers no partial tail, so there
    is nothing to offload."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    # 4-token prompt ends exactly on the mamba block boundary (block_size=4).
    req0 = make_request("0", [0, 0, 1, 1], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 4, num_computed, computed_blocks) is not None
    assert manager.take_partial_tail_offloads() == {}

    req0.num_computed_tokens = 4
    req0.append_output_token_ids([2])
    assert manager.allocate_slots(req0, 1) is not None
    assert manager.take_partial_tail_offloads() == {}


def test_truncate_computed_blocks_preserves_sparse_prefix_positions():
    """truncate_computed_blocks slices each group by its own block size,
    keeps null placeholders in the retained prefix, and leaves the original
    lookup result untouched (pure view, no refcount changes)."""
    hash_block_size = 2
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=2 * hash_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    producer = make_request("producer", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    blocks, num_computed, _ = manager.get_computed_blocks(producer)
    assert manager.allocate_slots(producer, 6, num_computed, blocks) is not None
    manager.free(producer)
    manager.new_step_starts()

    consumer = make_request(
        "consumer", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256
    )
    blocks, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed == 6
    assert [len(group) for group in blocks.blocks] == [3, 2]
    assert blocks.blocks[1][0].is_null

    truncated = manager.truncate_computed_blocks(blocks, 4)

    assert [len(group) for group in truncated.blocks] == [2, 1]
    assert truncated.blocks[1][0].is_null
    assert [len(group) for group in blocks.blocks] == [3, 2]


def test_truncate_computed_blocks_allows_short_mamba_group_only():
    """External state may replace a short Mamba hit, but other groups must
    cover the aligned local endpoint."""
    hash_block_size = 2
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=2 * hash_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    producer = make_request("producer", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    blocks, num_computed, _ = manager.get_computed_blocks(producer)
    assert manager.allocate_slots(producer, 6, num_computed, blocks) is not None
    manager.free(producer)
    manager.new_step_starts()

    consumer = make_request(
        "consumer", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256
    )
    blocks, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed == 6
    assert [len(group) for group in blocks.blocks] == [3, 2]

    short_mamba = manager.create_kv_cache_blocks((list(blocks.blocks[0]), []))
    truncated = manager.truncate_computed_blocks(short_mamba, 4)
    assert [len(group) for group in truncated.blocks] == [2, 0]

    short_full_attention = manager.create_kv_cache_blocks(
        (list(blocks.blocks[0][:1]), list(blocks.blocks[1]))
    )
    with pytest.raises(AssertionError):
        manager.truncate_computed_blocks(short_full_attention, 4)

    with pytest.raises(AssertionError):
        manager.truncate_computed_blocks(blocks, 6)

    # The lookup result itself is never mutated.
    assert [len(group) for group in blocks.blocks] == [3, 2]


def test_hybrid_mamba_partial_tail_owner_continue_preserves_later_hit():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert partial_mamba_block is not None
    partial_mamba_block_id = partial_mamba_block[0].block_id

    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None
    # The owner moved the prefix-cache entry to a private copy; capture its id.
    owner_copies, _ = manager.take_kv_cache_block_copies()
    cow_copy = next(c for c in owner_copies if c.src_block_id == partial_mamba_block_id)
    moved_block_id = cow_copy.dst_block_id
    manager.new_step_starts()

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 4, 4], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    # The later request hits the moved (private-copy) entry, not the source.
    assert computed_blocks.get_block_ids()[1][1] == moved_block_id

    new_blocks = manager.allocate_slots(req1, 2, num_computed, computed_blocks)
    assert new_blocks is not None
    mamba_new_block_ids = new_blocks.get_block_ids()[1]
    assert len(mamba_new_block_ids) == 1
    assert mamba_new_block_ids[0] != moved_block_id
    # The hitting request CoWs from the moved entry into its own private block.
    copies, _ = manager.take_kv_cache_block_copies()
    assert (
        KVCacheBlockCopy(
            src_block_id=moved_block_id,
            dst_block_id=mamba_new_block_ids[0],
        )
        in copies
    )


def test_hybrid_mamba_moved_partial_entry_defers_same_step_hit():
    """The owner's move re-arms the same-step guard: the moved entry is
    filled by this step's copy, and chained same-step copies read stale
    sources, so a request hitting it in the move step must be deferred."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    manager.new_step_starts()

    # The owning request continues decoding: the partial entry moves to a
    # private copy in this step.
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None

    # A request hitting the moved entry in the SAME step must be deferred.
    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 4, 4], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert manager.allocate_slots(req1, 2, num_computed, computed_blocks) is None

    # Next step the moved entry is consumable.
    manager.new_step_starts()
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert manager.allocate_slots(req1, 2, num_computed, computed_blocks) is not None


def test_hybrid_full_attention_partial_hash_hit_uses_cow():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    manager.free(req0)
    manager.new_step_starts()

    partial_full_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_full_block = manager.block_pool.get_cached_block(
        partial_full_hash, kv_cache_group_ids=[0]
    )
    assert partial_full_block is not None

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert [len(group) for group in computed_blocks.blocks] == [2, 2]

    new_blocks = manager.allocate_slots(req1, 2, num_computed, computed_blocks)
    assert new_blocks is not None
    full_new_block_ids = new_blocks.get_block_ids()[0]
    assert len(full_new_block_ids) == 1
    assert full_new_block_ids[0] != partial_full_block[0].block_id
    assert partial_full_block[0].block_hash is not None
    assert get_block_hash(partial_full_block[0].block_hash) == partial_full_hash
    assert get_group_id(partial_full_block[0].block_hash) == 0
    assert partial_full_block[0].block_hash_num_tokens == 6
    copies, retained = manager.take_kv_cache_block_copies()
    assert (
        KVCacheBlockCopy(
            src_block_id=partial_full_block[0].block_id,
            dst_block_id=full_new_block_ids[0],
        )
        in copies
    )
    assert partial_full_block[0].ref_cnt == 1
    manager.free_popped_blocks(retained)
    assert partial_full_block[0].ref_cnt == 0


def test_hybrid_partial_hit_cow_target_starts_uncached():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    manager.free(req0)
    manager.new_step_starts()

    partial_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_full_block = manager.block_pool.get_cached_block(
        partial_hash, kv_cache_group_ids=[0]
    )
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_hash, kv_cache_group_ids=[1]
    )
    assert partial_full_block is not None
    assert partial_mamba_block is not None

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6

    new_blocks = manager.allocate_slots(
        req1,
        2,
        num_computed,
        computed_blocks,
        delay_cache_blocks=True,
    )
    assert new_blocks is not None

    full_cow_block = manager.get_blocks("1").blocks[0][1]
    mamba_cow_block = manager.get_blocks("1").blocks[1][1]
    assert full_cow_block.block_id != partial_full_block[0].block_id
    assert mamba_cow_block.block_id != partial_mamba_block[0].block_id
    assert full_cow_block.block_hash is None
    assert full_cow_block.block_hash_num_tokens is None
    assert mamba_cow_block.block_hash is None
    assert mamba_cow_block.block_hash_num_tokens is None

    assert partial_full_block[0].block_hash is not None
    assert get_block_hash(partial_full_block[0].block_hash) == partial_hash
    assert get_group_id(partial_full_block[0].block_hash) == 0
    assert partial_full_block[0].block_hash_num_tokens == 6
    assert partial_mamba_block[0].block_hash is not None
    assert get_block_hash(partial_mamba_block[0].block_hash) == partial_hash
    assert get_group_id(partial_mamba_block[0].block_hash) == 1
    assert partial_mamba_block[0].block_hash_num_tokens == 6


def test_hybrid_partial_hash_truncates_full_attention_hit_length():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    pool = manager.block_pool
    req = make_request(
        "0",
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        hash_block_size,
        sha256,
    )

    full_blocks = pool.get_new_blocks(3)
    pool.cache_full_blocks(
        request=req,
        blocks=full_blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=0,
    )
    pool.cache_partial_block(
        request=req,
        block=full_blocks[2],
        num_tokens=10,
        kv_cache_group_id=0,
        block_size=block_size,
    )

    mamba_block = pool.get_new_blocks(1)[0]
    pool.cache_partial_block(
        request=req,
        block=mamba_block,
        num_tokens=6,
        kv_cache_group_id=1,
        block_size=block_size,
    )

    computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
    assert num_computed == 6
    assert [len(group) for group in computed_blocks.blocks] == [2, 2]


def test_cow_retained_blocks_returned_for_release():
    """new_step_starts returns the CoW copy retentions instead of freeing
    them; the scheduler owns releasing them once the copy has run."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    # The owner's move queues a copy and retains both endpoints.
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None
    (cow_copy,), retained = manager.take_kv_cache_block_copies()
    retained_blocks = [block for group in retained.blocks for block in group]
    assert {b.block_id for b in retained_blocks} == {
        cow_copy.src_block_id,
        cow_copy.dst_block_id,
    }
    # Not freed yet: the retention refs are still held.
    assert all(b.ref_cnt > 0 for b in retained_blocks)
    manager.free_popped_blocks(retained)


def test_free_cow_retained_blocks_defers_until_copy_step_processed():
    """Scheduler releases CoW retentions immediately when the copy's step has
    been processed (or deferral is off), and defers them otherwise."""
    from collections import deque

    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

    freed: list = []
    blocks = [SimpleNamespace(block_id=7), SimpleNamespace(block_id=9)]

    def free_popped_blocks(retained: KVCacheBlocks) -> None:
        freed.extend(reversed(retained.blocks[0]))

    mock = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            free_popped_blocks=free_popped_blocks,
        ),
        deferred_frees=deque(),
        defer_block_free=True,
        processed_step_seq=2,
    )
    free = Scheduler._free_cow_retained_blocks
    retained = KVCacheBlocks((blocks[::-1],))

    # Copy step still in flight: deferred with its fence.
    free(mock, retained, fence_seq=3)
    assert not freed
    assert mock.deferred_frees == deque([(3, retained)])

    # Copy step processed: freed immediately.
    mock.processed_step_seq = 3
    free(mock, retained, fence_seq=3)
    assert freed == blocks

    # Deferral disabled: freed immediately regardless of the fence.
    freed.clear()
    mock.deferred_frees.clear()
    mock.defer_block_free = False
    mock.processed_step_seq = 0
    free(mock, retained, fence_seq=3)
    assert freed == blocks


def test_full_attention_eagle_drops_one_hash_unit():
    """With fine-grained partial hits, eagle rewinds the hit by one hash unit
    instead of a whole cache block: the tail block's KV is append-only, so it
    still covers the reduced length and stays in the hit as a partial block."""
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    hash_block_size = 2
    block_size = 4
    pool = BlockPool(
        num_gpu_blocks=10, enable_caching=True, hash_block_size=hash_block_size
    )
    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    req = make_request("0", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)

    def find(drop_eagle_block):
        return FullAttentionManager.find_longest_cache_hit(
            block_hashes=req.block_hashes,
            max_length=8,
            kv_cache_group_ids=[0],
            block_pool=pool,
            kv_cache_spec=spec,
            drop_eagle_block=drop_eagle_block,
            alignment_tokens=hash_block_size,
        )

    # Two full cached blocks (hit 8): eagle rewinds to 6, keeping the last
    # block as a partial hit instead of dropping it to 4.
    blocks = pool.get_new_blocks(2)
    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=0,
    )
    hit_blocks, hit_length = find(drop_eagle_block=False)
    assert (hit_length, len(hit_blocks[0])) == (8, 2)
    hit_blocks, hit_length = find(drop_eagle_block=True)
    assert (hit_length, len(hit_blocks[0])) == (6, 2)

    # A partial tail at 6 (block 1 not fully cached): eagle rewinds to the
    # block boundary and trims the tail block.
    pool2 = BlockPool(
        num_gpu_blocks=10, enable_caching=True, hash_block_size=hash_block_size
    )
    pool = pool2
    blocks = pool.get_new_blocks(2)
    pool.cache_full_blocks(
        request=req,
        blocks=blocks[:1],
        num_cached_blocks=0,
        num_full_blocks=1,
        block_size=block_size,
        kv_cache_group_id=0,
    )
    assert (
        pool.cache_partial_block(
            request=req,
            block=blocks[1],
            num_tokens=6,
            kv_cache_group_id=0,
            block_size=block_size,
        )
        is not None
    )
    hit_blocks, hit_length = find(drop_eagle_block=False)
    assert (hit_length, len(hit_blocks[0])) == (6, 2)
    hit_blocks, hit_length = find(drop_eagle_block=True)
    assert (hit_length, len(hit_blocks[0])) == (4, 1)


def test_hybrid_partial_hit_with_eagle_stays_within_group_blocks():
    """Regression: with eagle, the mamba group must not receive the eagle
    lookup margin — its finder never applies the drop, so it could return a
    hit past the blocks the (dropped) full-attention group covers, crashing
    the consumer's CoW with block_idx >= len(req_blocks)."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
        use_eagle=True,
    )

    # The owner prefills in scheduler-split style: stop at the block boundary
    # (4), then at the prompt's last hash boundary (6, partial entries).
    req0 = make_request("0", [7] * 6, hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 4, num_computed, computed_blocks) is not None
    req0.num_computed_tokens = 4
    manager.new_step_starts()
    assert manager.allocate_slots(req0, 2) is not None
    req0.num_computed_tokens = 6
    manager.new_step_starts()

    # A longer request with eagle: full attention drops the partial tail, so
    # the joint hit must fall back to the block boundary the FA blocks cover.
    req1 = make_request("1", [7] * 6 + [9] * 2, hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 4
    assert all(
        len(group) * block_size >= num_computed for group in computed_blocks.blocks
    )
    assert manager.allocate_slots(req1, 4, num_computed, computed_blocks) is not None
