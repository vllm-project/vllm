# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``CacheConfig.evict_superseded_mamba_states``.

A multi-turn conversation's next turn resumes from the previous turn's Mamba
state and caches a deeper one, so once it finishes the state it resumed from is
dead for the conversation. The flag frees those states to the front of the free
queue, keeping their hashes until the pool hands the blocks out.

Shapes follow a TP8 hybrid deployment: three Mamba "align" groups with 896-token
blocks, speculative and prefill checkpoint blocks, one 7168-token attention
group, 128-token hashes and sparse retention.
"""

import random

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_group_id,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

HASH_BLOCK_SIZE = 128
MAMBA_GROUP_IDS = (0, 1, 2)
_rng = random.Random(0)
COLD = [_rng.randrange(1000, 160_000) for _ in range(20_000)]
WARM = COLD + [_rng.randrange(1000, 160_000) for _ in range(2_500)]
BRANCH = COLD + [_rng.randrange(1000, 160_000) for _ in range(2_500)]
# The cold and warm turns' prompt checkpoints.
COLD_STATE, WARM_STATE = 19_968, 22_400


def _manager(evict: bool) -> KVCacheManager:
    init_none_hash(sha256)
    mamba = MambaSpec(
        block_size=896,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=4,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=HASH_BLOCK_SIZE,
    )
    attention = FullAttentionSpec(
        block_size=7168, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    config = KVCacheConfig(
        num_blocks=200,
        kv_cache_tensors=[],
        kv_cache_groups=[
            *(KVCacheGroupSpec([f"mamba{i}"], mamba) for i in MAMBA_GROUP_IDS),
            KVCacheGroupSpec(["attn"], attention),
        ],
        prefix_cache_retention_interval=0,
    )
    return KVCacheManager(
        config,
        max_model_len=65_536,
        scheduler_block_size=7168,
        hash_block_size=HASH_BLOCK_SIZE,
        evict_superseded_mamba_states=evict,
    )


def _prefill(manager: KVCacheManager, request: Request, chunks: list[int]) -> int:
    """Schedule the prompt in ``chunks`` as the scheduler would; return the hit."""
    blocks, hit, request.shared_prefix_boundary = manager.get_computed_blocks(request)
    for i, num_tokens in enumerate(chunks):
        new_computed = (hit, blocks) if i == 0 else (0, None)
        assert manager.allocate_slots(request, num_tokens, *new_computed) is not None
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens += new_computed[0] + num_tokens
        # The step ran its copy-on-write copies.
        manager.block_pool.free_blocks(manager.take_kv_cache_block_copies()[1])
        manager.new_step_starts()
    return hit


def _turn(manager: KVCacheManager, request_id: str, tokens, chunks) -> int:
    request = make_request(request_id, tokens, HASH_BLOCK_SIZE, sha256)
    hit = _prefill(manager, request, chunks)
    assert request.num_computed_tokens == len(tokens)
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    manager.free(request)
    return hit


def _mamba_state_ids(manager: KVCacheManager, num_tokens: int) -> set[int]:
    return {
        block.block_id
        for block in manager.block_pool.blocks
        if block.block_hash is not None
        and get_group_id(block.block_hash) in MAMBA_GROUP_IDS
        and block.block_hash_num_tokens == num_tokens
    }


def _out_of_lru_order_ids(manager: KVCacheManager) -> set[int]:
    """Free cached blocks queued ahead of an uncached one."""
    queue = manager.block_pool.free_block_queue.get_all_free_blocks()
    last_uncached = max(i for i, b in enumerate(queue) if b.block_hash is None)
    return {b.block_id for b in queue[:last_uncached] if b.block_hash is not None}


@pytest.mark.parametrize("evict", [False, True])
def test_warm_turn_evicts_the_state_it_resumed_from_first(evict):
    manager = _manager(evict)
    assert _turn(manager, "cold", COLD, [11_648, 8_352]) == 0
    assert _turn(manager, "warm", WARM, [2_532]) == COLD_STATE
    superseded = _mamba_state_ids(manager, COLD_STATE)
    assert len(superseded) == len(MAMBA_GROUP_IDS)
    assert _out_of_lru_order_ids(manager) == (superseded if evict else set())

    # Still hashed, the superseded states serve a branch off the cold turn...
    branch = make_request("branch", BRANCH, HASH_BLOCK_SIZE, sha256)
    assert manager.get_computed_blocks(branch)[1] == COLD_STATE
    # ...until the pool hands them out, before any other cached block.
    pool = manager.block_pool
    cached = [
        b
        for b in pool.free_block_queue.get_all_free_blocks()
        if b.block_hash is not None
    ]
    pool.get_new_blocks(pool.get_num_free_blocks() - len(cached))
    evicted = {b.block_id for b in cached if b.block_hash is None}
    assert evicted == (superseded if evict else set())
    assert manager.get_computed_blocks(branch)[1] == (0 if evict else COLD_STATE)


def test_turn_without_a_deeper_state_supersedes_nothing():
    manager = _manager(evict=True)
    _turn(manager, "cold", COLD, [11_648, 8_352])
    _turn(manager, "warm", WARM, [2_532])
    superseded = _mamba_state_ids(manager, COLD_STATE)
    # Ends inside the 896-token block it resumed in, so caches no later state.
    assert _turn(manager, "short", WARM + COLD[:300], [400]) == WARM_STATE
    assert _out_of_lru_order_ids(manager) == superseded


@pytest.mark.parametrize("evict", [False, True])
def test_preempted_request_keeps_its_states(evict):
    manager = _manager(evict)
    _turn(manager, "cold", COLD, [11_648, 8_352])
    resumed_from = _mamba_state_ids(manager, COLD_STATE)
    warm = make_request("warm", WARM, HASH_BLOCK_SIZE, sha256)
    assert _prefill(manager, warm, [2_532]) == COLD_STATE
    # Preempted after caching a deeper state, it may still need the older one
    # if the deeper state is evicted while it waits.
    manager.free(warm)
    assert _out_of_lru_order_ids(manager) == set()

    warm.status = RequestStatus.PREEMPTED
    warm.num_computed_tokens = 0
    assert _prefill(manager, warm, [100]) == WARM_STATE
    warm.status = RequestStatus.FINISHED_LENGTH_CAPPED
    manager.free(warm)
    assert _out_of_lru_order_ids(manager) == (resumed_from if evict else set())


def test_states_up_to_a_shared_prefix_junction_are_kept():
    manager = _manager(evict=True)
    _turn(manager, "cold", COLD, [11_648, 8_352])
    _turn(manager, "prefix", COLD[:7_200], [7_200])
    shared = make_request(
        "shared", COLD[:16_000] + WARM[-2_000:], HASH_BLOCK_SIZE, sha256
    )
    # Resumes from the 7,168 state while attention matches up to 14,336.
    assert _prefill(manager, shared, [7_168, 3_664]) == 7_168
    assert shared.shared_prefix_boundary == 14_336
    shared.status = RequestStatus.FINISHED_LENGTH_CAPPED
    manager.free(shared)
    assert _out_of_lru_order_ids(manager) == set()


def test_superseded_state_held_elsewhere_is_evicted_first_once_released():
    manager = _manager(evict=True)
    pool = manager.block_pool
    _turn(manager, "cold", COLD, [11_648, 8_352])
    states = [pool.blocks[i] for i in _mamba_state_ids(manager, COLD_STATE)]
    pool.touch(states)  # e.g. a KV connector reading them
    _turn(manager, "warm", WARM, [2_532])
    assert _out_of_lru_order_ids(manager) == set()
    pool.free_blocks(states)
    assert _out_of_lru_order_ids(manager) == {b.block_id for b in states}


def test_prioritize_eviction():
    pool = BlockPool(num_gpu_blocks=6, enable_caching=True, hash_block_size=16)
    a, b, c, d = pool.get_new_blocks(4)
    for i, block in enumerate((a, b, c)):
        block_hash = make_block_hash_with_group_id(BlockHash(bytes([i])), 0)
        block.set_block_hash(block_hash, num_tokens=16)
        pool.cached_block_hash_to_block.insert(block_hash, block)
    (e,) = pool.free_block_queue.get_all_free_blocks()

    def queue():
        return [blk.block_id for blk in pool.free_block_queue.get_all_free_blocks()]

    pool.free_blocks([a, b])
    pool.prioritize_eviction([b, c])
    assert queue() == [b.block_id, e.block_id, a.block_id]
    # c follows once its last reference is freed, behind uncached blocks.
    pool.free_blocks([c, d])
    assert queue() == [d.block_id, c.block_id, b.block_id, e.block_id, a.block_id]
    assert all(blk.block_hash is not None for blk in (a, b, c))

    # A prefix-cache hit on a referenced block cancels it.
    pool.touch([a])
    pool.prioritize_eviction([a])
    pool.touch([a])
    pool.free_blocks([a, a])
    assert queue()[-1] == a.block_id
