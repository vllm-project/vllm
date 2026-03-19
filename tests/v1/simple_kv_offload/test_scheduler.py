# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffloadScheduler."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLOCK_SIZE = 16
HEAD_SIZE = 16
NUM_KV_HEADS = 1
DTYPE = torch.float16
# bytes per block per tensor:
# block_size * num_kv_heads * head_size * 2 (K+V) * element_size
_BYTES_PER_BLOCK = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * 2 * DTYPE.itemsize

# Ensure none_hash is initialized once
init_none_hash(sha256)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kv_cache_config(
    num_blocks: int,
    num_groups: int = 1,
) -> KVCacheConfig:
    """Build a KVCacheConfig with non-empty kv_cache_tensors."""
    groups = []
    tensors = []
    for g in range(num_groups):
        layer_names = [f"layer_{g}"]
        groups.append(
            KVCacheGroupSpec(
                layer_names,
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=NUM_KV_HEADS,
                    head_size=HEAD_SIZE,
                    dtype=DTYPE,
                ),
            )
        )
        tensors.append(
            KVCacheTensor(
                size=_BYTES_PER_BLOCK * num_blocks,
                shared_by=layer_names,
            )
        )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def _make_vllm_config(block_size: int = BLOCK_SIZE) -> VllmConfig:
    """Minimal VllmConfig for scheduler tests (no GPU)."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=10000,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
    )
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )


@dataclass
class SchedulerFixture:
    """Bundle returned by make_scheduler for convenient access."""

    scheduler: SimpleCPUOffloadScheduler
    gpu_block_pool: BlockPool
    vllm_config: VllmConfig
    kv_cache_config: KVCacheConfig


def make_scheduler(
    num_cpu_blocks: int = 8,
    num_gpu_blocks: int = 16,
    num_groups: int = 1,
    lazy: bool = False,
) -> SchedulerFixture:
    """Build a SimpleCPUOffloadScheduler with small block pools."""
    kv_cache_config = _make_kv_cache_config(num_gpu_blocks, num_groups)
    vllm_config = _make_vllm_config()
    cpu_capacity_bytes = _BYTES_PER_BLOCK * num_cpu_blocks * num_groups

    sched = SimpleCPUOffloadScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        cpu_capacity_bytes=cpu_capacity_bytes,
        lazy_offload=lazy,
    )

    # Build a real GPU block pool and bind it
    gpu_block_pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
    )
    sched.bind_gpu_block_pool(gpu_block_pool)

    return SchedulerFixture(
        scheduler=sched,
        gpu_block_pool=gpu_block_pool,
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
    )


_req_counter = 0


def make_request(
    num_blocks: int = 2,
    request_id: str | None = None,
) -> Request:
    """Create a Request with deterministic block hashes."""
    global _req_counter
    _req_counter += 1
    if request_id is None:
        request_id = f"req-{_req_counter}"

    num_tokens = num_blocks * BLOCK_SIZE
    start = _req_counter * 10000
    prompt_token_ids = list(range(start, start + num_tokens))
    sampling_params = SamplingParams(max_tokens=1)

    req = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    return req


def make_scheduler_output(
    req_id_to_num_tokens: dict[str, int],
) -> SchedulerOutput:
    """Build a minimal SchedulerOutput with num_scheduled_tokens."""
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=req_id_to_num_tokens,
        total_num_scheduled_tokens=sum(req_id_to_num_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def simulate_store_completion(
    scheduler: SimpleCPUOffloadScheduler,
    event_idx: int,
) -> None:
    """Simulate worker reporting a store event completion."""
    output = KVConnectorOutput(
        finished_sending={f"__store_done_{event_idx}"},
        finished_recving=set(),
    )
    scheduler.update_connector_output(output)


def simulate_load_completion(
    scheduler: SimpleCPUOffloadScheduler,
    req_ids: set[str],
) -> None:
    """Simulate worker reporting load completions for requests."""
    output = KVConnectorOutput(
        finished_sending=set(),
        finished_recving=req_ids,
    )
    scheduler.update_connector_output(output)


def get_cpu_free_blocks(scheduler: SimpleCPUOffloadScheduler) -> int:
    """Return number of free CPU blocks."""
    return scheduler.cpu_block_pool.get_num_free_blocks()


def _allocate_gpu_blocks(
    gpu_block_pool: BlockPool,
    request: Request,
    num_blocks: int,
    group_id: int = 0,
) -> list:
    """Allocate GPU blocks, cache them with hashes, return block list.

    Mimics what KVCacheManager does: allocate blocks from pool, then
    register them in the prefix cache via cache_full_blocks so that
    re-allocation properly evicts stale hashes.
    """
    blocks = gpu_block_pool.get_new_blocks(num_blocks)
    num_full = min(num_blocks, len(request.block_hashes))
    if num_full > 0:
        gpu_block_pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=0,
            num_full_blocks=num_full,
            block_size=BLOCK_SIZE,
            kv_cache_group_id=group_id,
        )
    return blocks


# ---------------------------------------------------------------------------
# Test 1a: Eager store-and-load roundtrip
# ---------------------------------------------------------------------------
def test_eager_store_and_load_roundtrip() -> None:
    """Eager mode: store blocks on compute, complete store, verify cache hit."""
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    num_blocks = 2
    req = make_request(num_blocks=num_blocks)

    gpu_blocks = _allocate_gpu_blocks(gpu_pool, req, num_blocks, group_id=0)
    kv_blocks = KVCacheBlocks(blocks=(gpu_blocks,))
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)
    sched_out = make_scheduler_output({req.request_id: num_blocks * BLOCK_SIZE})

    meta = sched.build_connector_meta(sched_out)
    assert meta.store_event >= 0, "Expected a store event to be scheduled"
    assert len(meta.store_gpu_blocks) > 0
    assert len(meta.store_cpu_blocks) == len(meta.store_gpu_blocks)
    simulate_store_completion(sched, meta.store_event)

    # New request with same tokens should get CPU cache hit
    req2 = Request(
        request_id="req-eager-load",
        prompt_token_ids=req.prompt_token_ids,
        sampling_params=req.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=req._block_hasher,
    )
    hit_tokens, is_async = sched.get_num_new_matched_tokens(req2, num_computed_tokens=0)
    assert hit_tokens == num_blocks * BLOCK_SIZE
    assert is_async is True

    gpu_blocks2 = gpu_pool.get_new_blocks(num_blocks)
    kv_blocks2 = KVCacheBlocks(blocks=(gpu_blocks2,))
    sched.update_state_after_alloc(req2, kv_blocks2, num_external_tokens=hit_tokens)

    sched_out2 = make_scheduler_output({req2.request_id: 1})
    meta2 = sched.build_connector_meta(sched_out2)
    assert meta2.load_event >= 0, "Expected a load event to be assigned"
    assert len(meta2.load_gpu_blocks) > 0
    assert len(meta2.load_cpu_blocks) == len(meta2.load_gpu_blocks)


# ---------------------------------------------------------------------------
# Test 1b: Lazy store-and-load roundtrip
# ---------------------------------------------------------------------------
def _flush_old_blocks_to_lru_head(
    gpu_pool: BlockPool,
    num_filler_blocks: int,
) -> list:
    """Allocate filler blocks so that previously-freed (hashed) blocks migrate
    to the LRU head of the free queue.  Returns the filler blocks (caller must
    free them later to restore pool capacity).

    In a real engine the same thing happens naturally: after one request
    finishes and frees its blocks, subsequent requests allocate from the LRU
    head, consuming the unhashed blocks and leaving the old hashed blocks at
    the front of the queue.
    """
    fillers = gpu_pool.get_new_blocks(num_filler_blocks)
    return fillers


def test_lazy_store_and_load_roundtrip() -> None:
    """Lazy mode: schedule a request, finish it so its hashed blocks are freed,
    then schedule new requests so the old blocks migrate to the LRU head.
    The lazy scanner offloads them to CPU.  Re-scheduling the old request
    triggers a CPU cache hit + load.

    GPU pool: 8 blocks (7 usable).  _target_free = ceil(64/16) = 4.
    """
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=8, lazy=True)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    num_blocks = 2

    # --- Step 1: Schedule req_old, compute, and finish ---
    req_old = make_request(num_blocks=num_blocks)
    gpu_blocks_old = _allocate_gpu_blocks(gpu_pool, req_old, num_blocks, group_id=0)
    gpu_pool.free_blocks(gpu_blocks_old)

    # Allocate filler blocks so req_old's hashed blocks move to LRU head.
    # 7 usable - 2 (req_old freed) = 5 other free blocks to consume.
    fillers = _flush_old_blocks_to_lru_head(gpu_pool, num_filler_blocks=5)

    # --- Step 2: Lazy scanner should offload req_old's blocks ---
    sched_out = make_scheduler_output({})
    meta = sched.build_connector_meta(sched_out)
    assert meta.store_event >= 0, "Expected lazy store to offload old blocks"
    assert len(meta.store_gpu_blocks) == num_blocks
    simulate_store_completion(sched, meta.store_event)

    # Free fillers to restore pool capacity.
    gpu_pool.free_blocks(fillers)

    # --- Step 3: Re-schedule req_old — should get CPU cache hit ---
    req_old2 = Request(
        request_id="req-old-reload",
        prompt_token_ids=req_old.prompt_token_ids,
        sampling_params=req_old.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=req_old._block_hasher,
    )
    hit_tokens, is_async = sched.get_num_new_matched_tokens(
        req_old2, num_computed_tokens=0
    )
    assert hit_tokens == num_blocks * BLOCK_SIZE, (
        f"Expected {num_blocks * BLOCK_SIZE} hit tokens, got {hit_tokens}"
    )
    assert is_async is True

    # Allocate fresh GPU blocks for the load.
    gpu_blocks_load = gpu_pool.get_new_blocks(num_blocks)
    kv_blocks_load = KVCacheBlocks(blocks=(gpu_blocks_load,))
    sched.update_state_after_alloc(
        req_old2, kv_blocks_load, num_external_tokens=hit_tokens
    )

    sched_out2 = make_scheduler_output({req_old2.request_id: 1})
    meta2 = sched.build_connector_meta(sched_out2)
    assert meta2.load_event >= 0, "Expected a load event to be assigned"
    assert len(meta2.load_gpu_blocks) > 0


# ---------------------------------------------------------------------------
# Test 2a: Eager duplicate store is skipped
# ---------------------------------------------------------------------------
def test_eager_duplicate_store_skipped() -> None:
    """Eager: storing the same block hashes twice should not allocate new CPU blocks."""
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    num_blocks = 2
    req = make_request(num_blocks=num_blocks)

    gpu_blocks = _allocate_gpu_blocks(gpu_pool, req, num_blocks, group_id=0)
    kv_blocks = KVCacheBlocks(blocks=(gpu_blocks,))
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)
    sched_out = make_scheduler_output({req.request_id: num_blocks * BLOCK_SIZE})

    meta1 = sched.build_connector_meta(sched_out)
    assert meta1.store_event >= 0
    simulate_store_completion(sched, meta1.store_event)
    cpu_free_after_first = get_cpu_free_blocks(sched)

    # Second request with identical hashes — should skip store
    req2 = Request(
        request_id="req-dup-eager",
        prompt_token_ids=req.prompt_token_ids,
        sampling_params=req.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=req._block_hasher,
    )
    gpu_blocks2 = _allocate_gpu_blocks(gpu_pool, req2, num_blocks, group_id=0)
    kv_blocks2 = KVCacheBlocks(blocks=(gpu_blocks2,))
    sched.update_state_after_alloc(req2, kv_blocks2, num_external_tokens=0)
    sched_out2 = make_scheduler_output({req2.request_id: num_blocks * BLOCK_SIZE})

    meta2 = sched.build_connector_meta(sched_out2)
    if meta2.store_event >= 0:
        assert len(meta2.store_cpu_blocks) == 0, (
            "Expected no new CPU blocks for duplicate hashes"
        )
    assert get_cpu_free_blocks(sched) == cpu_free_after_first


# ---------------------------------------------------------------------------
# Test 2b: Lazy duplicate store is skipped
# ---------------------------------------------------------------------------
def test_lazy_duplicate_store_skipped() -> None:
    """Lazy: blocks already offloaded to CPU should not be offloaded again.

    Same pattern as the lazy roundtrip: flush old blocks to LRU head, offload,
    then repeat with the same hashes and verify no new CPU allocation.
    """
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=8, lazy=True)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    num_blocks = 2
    req = make_request(num_blocks=num_blocks)

    # Schedule + finish → hashed blocks in free queue
    gpu_blocks = _allocate_gpu_blocks(gpu_pool, req, num_blocks, group_id=0)
    gpu_pool.free_blocks(gpu_blocks)

    # Flush old blocks to LRU head, then trigger lazy offload.
    fillers = _flush_old_blocks_to_lru_head(gpu_pool, num_filler_blocks=5)
    meta1 = sched.build_connector_meta(make_scheduler_output({}))
    assert meta1.store_event >= 0
    simulate_store_completion(sched, meta1.store_event)
    gpu_pool.free_blocks(fillers)
    cpu_free_after_first = get_cpu_free_blocks(sched)

    # Allocate blocks with the same hashes and free them again.
    # The scanner should see they are already in CPU cache and skip them.
    req2 = Request(
        request_id="req-dup-lazy",
        prompt_token_ids=req.prompt_token_ids,
        sampling_params=req.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=req._block_hasher,
    )
    gpu_blocks2 = _allocate_gpu_blocks(gpu_pool, req2, num_blocks, group_id=0)
    gpu_pool.free_blocks(gpu_blocks2)

    # Flush again so the hashed blocks are at LRU head for the scanner.
    fillers2 = _flush_old_blocks_to_lru_head(gpu_pool, num_filler_blocks=5)
    meta2 = sched.build_connector_meta(make_scheduler_output({}))
    gpu_pool.free_blocks(fillers2)

    # Either no store event, or zero new CPU blocks (already cached).
    if meta2.store_event >= 0:
        assert len(meta2.store_cpu_blocks) == 0, (
            "Expected no new CPU blocks for duplicate hashes"
        )
    assert get_cpu_free_blocks(sched) == cpu_free_after_first


# ---------------------------------------------------------------------------
# Test 3: LRU eviction order
# ---------------------------------------------------------------------------
def test_lru_eviction_order() -> None:
    """With limited CPU space, oldest blocks should be evicted first.

    CPU block pool: num_cpu_blocks=5 -> 4 free usable blocks (1 taken by null_block).
    After storing 4 blocks (2 req_a + 2 req_b), all free slots are occupied by
    cached blocks (ref_cnt=0, in hash map).  When 2 more are stored (req_c),
    2 LRU blocks from req_a get evicted from the cache to make room.
    """
    # 5 total = 4 usable (null_block takes 1), filling exactly with 4 blocks
    fix = make_scheduler(num_cpu_blocks=5, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    # Fill CPU with 4 blocks: 2 requests x 2 blocks (in LRU insertion order)
    req_a = make_request(num_blocks=2)
    req_b = make_request(num_blocks=2)

    gpu_blocks_a = _allocate_gpu_blocks(gpu_pool, req_a, 2, group_id=0)
    gpu_blocks_b = _allocate_gpu_blocks(gpu_pool, req_b, 2, group_id=0)

    kv_a = KVCacheBlocks(blocks=(gpu_blocks_a,))
    kv_b = KVCacheBlocks(blocks=(gpu_blocks_b,))
    sched.update_state_after_alloc(req_a, kv_a, num_external_tokens=0)
    sched.update_state_after_alloc(req_b, kv_b, num_external_tokens=0)

    sched_out = make_scheduler_output(
        {
            req_a.request_id: 2 * BLOCK_SIZE,
            req_b.request_id: 2 * BLOCK_SIZE,
        }
    )
    meta = sched.build_connector_meta(sched_out)
    assert meta.store_event >= 0
    simulate_store_completion(sched, meta.store_event)

    # Verify all 4 blocks are cached in CPU hash map
    for i, bhash in enumerate(req_a.block_hashes[:2]):
        bhash_with_group = make_block_hash_with_group_id(bhash, 0)
        assert (
            sched.cpu_block_pool.cached_block_hash_to_block.get_one_block(
                bhash_with_group
            )
            is not None
        ), f"req_a block {i} should be cached after store"
    for i, bhash in enumerate(req_b.block_hashes[:2]):
        bhash_with_group = make_block_hash_with_group_id(bhash, 0)
        assert (
            sched.cpu_block_pool.cached_block_hash_to_block.get_one_block(
                bhash_with_group
            )
            is not None
        ), f"req_b block {i} should be cached after store"

    # Store 2 more blocks from a new request - must evict 2 LRU blocks (req_a)
    req_c = make_request(num_blocks=2)
    gpu_blocks_c = _allocate_gpu_blocks(gpu_pool, req_c, 2, group_id=0)
    kv_c = KVCacheBlocks(blocks=(gpu_blocks_c,))
    sched.update_state_after_alloc(req_c, kv_c, num_external_tokens=0)

    sched_out2 = make_scheduler_output({req_c.request_id: 2 * BLOCK_SIZE})
    meta2 = sched.build_connector_meta(sched_out2)
    assert meta2.store_event >= 0
    simulate_store_completion(sched, meta2.store_event)

    # req_a hashes should be evicted from CPU (they were LRU)
    for i, bhash in enumerate(req_a.block_hashes[:2]):
        bhash_with_group = make_block_hash_with_group_id(bhash, 0)
        cache_map = sched.cpu_block_pool.cached_block_hash_to_block
        cached = cache_map.get_one_block(bhash_with_group)
        assert cached is None, f"req_a block {i} should have been evicted"

    # req_b and req_c hashes should be present
    for i, bhash in enumerate(req_b.block_hashes[:2]):
        bhash_with_group = make_block_hash_with_group_id(bhash, 0)
        cache_map = sched.cpu_block_pool.cached_block_hash_to_block
        cached = cache_map.get_one_block(bhash_with_group)
        assert cached is not None, f"req_b block {i} should still be cached"

    for i, bhash in enumerate(req_c.block_hashes[:2]):
        bhash_with_group = make_block_hash_with_group_id(bhash, 0)
        cache_map = sched.cpu_block_pool.cached_block_hash_to_block
        cached = cache_map.get_one_block(bhash_with_group)
        assert cached is not None, f"req_c block {i} should still be cached"


# ---------------------------------------------------------------------------
# Test 4: Touched blocks survive eviction
# ---------------------------------------------------------------------------
def test_touched_blocks_survive_eviction() -> None:
    """Touching CPU blocks updates their LRU position, protecting them from eviction."""
    # 5 total = 4 usable (null_block takes 1)
    fix = make_scheduler(num_cpu_blocks=5, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    # Fill CPU with 4 blocks (req_a: 2, req_b: 2) in LRU order
    req_a = make_request(num_blocks=2)
    req_b = make_request(num_blocks=2)

    gpu_blocks_a = _allocate_gpu_blocks(gpu_pool, req_a, 2, group_id=0)
    gpu_blocks_b = _allocate_gpu_blocks(gpu_pool, req_b, 2, group_id=0)

    kv_a = KVCacheBlocks(blocks=(gpu_blocks_a,))
    kv_b = KVCacheBlocks(blocks=(gpu_blocks_b,))
    sched.update_state_after_alloc(req_a, kv_a, num_external_tokens=0)
    sched.update_state_after_alloc(req_b, kv_b, num_external_tokens=0)

    sched_out = make_scheduler_output(
        {
            req_a.request_id: 2 * BLOCK_SIZE,
            req_b.request_id: 2 * BLOCK_SIZE,
        }
    )
    meta = sched.build_connector_meta(sched_out)
    simulate_store_completion(sched, meta.store_event)

    # Touch req_a's CPU blocks to make them most-recently-used
    cpu_pool = sched.cpu_block_pool
    for bhash in req_a.block_hashes[:2]:
        bhash_with_group = make_block_hash_with_group_id(bhash, 0)
        cached_blk = cpu_pool.cached_block_hash_to_block.get_one_block(bhash_with_group)
        assert cached_blk is not None
        cpu_pool.touch([cached_blk])
        # Undo touch to return ref_cnt to 0
        # (so it's a free candidate but at MRU position)
        cpu_pool.free_blocks([cached_blk])

    # Now store 2 more blocks; req_b (LRU front) should be evicted, not req_a
    req_c = make_request(num_blocks=2)
    gpu_blocks_c = _allocate_gpu_blocks(gpu_pool, req_c, 2, group_id=0)
    kv_c = KVCacheBlocks(blocks=(gpu_blocks_c,))
    sched.update_state_after_alloc(req_c, kv_c, num_external_tokens=0)

    sched_out2 = make_scheduler_output({req_c.request_id: 2 * BLOCK_SIZE})
    meta2 = sched.build_connector_meta(sched_out2)
    simulate_store_completion(sched, meta2.store_event)

    # req_b should be evicted (LRU), req_a and req_c should survive
    for i, bhash in enumerate(req_b.block_hashes[:2]):
        bhash_with_group = make_block_hash_with_group_id(bhash, 0)
        cached = cpu_pool.cached_block_hash_to_block.get_one_block(bhash_with_group)
        assert cached is None, f"req_b block {i} should have been evicted (it was LRU)"

    for i, bhash in enumerate(req_a.block_hashes[:2]):
        bhash_with_group = make_block_hash_with_group_id(bhash, 0)
        cached = cpu_pool.cached_block_hash_to_block.get_one_block(bhash_with_group)
        assert cached is not None, f"req_a block {i} should survive (was touched/MRU)"


# ---------------------------------------------------------------------------
# Test 5: Preemption no CPU block leak
# ---------------------------------------------------------------------------
def test_preemption_no_cpu_block_leak() -> None:
    """request_finished during in-flight load defers cleanup;
    completes after load done."""
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    num_blocks = 2

    # First: store blocks to CPU
    req = make_request(num_blocks=num_blocks)
    gpu_blocks = _allocate_gpu_blocks(gpu_pool, req, num_blocks, group_id=0)
    kv_blocks = KVCacheBlocks(blocks=(gpu_blocks,))
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)
    sched_out = make_scheduler_output({req.request_id: num_blocks * BLOCK_SIZE})
    meta = sched.build_connector_meta(sched_out)
    simulate_store_completion(sched, meta.store_event)

    # Create new request with same tokens, check hit
    req2 = Request(
        request_id="req-preempt-load",
        prompt_token_ids=req.prompt_token_ids,
        sampling_params=req.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=req._block_hasher,
    )
    hit_tokens, is_async = sched.get_num_new_matched_tokens(req2, num_computed_tokens=0)
    assert hit_tokens > 0

    gpu_blocks2 = gpu_pool.get_new_blocks(num_blocks)
    kv_blocks2 = KVCacheBlocks(blocks=(gpu_blocks2,))
    sched.update_state_after_alloc(req2, kv_blocks2, num_external_tokens=hit_tokens)

    # Assign load_event via build_connector_meta
    sched_out2 = make_scheduler_output({req2.request_id: 1})
    meta2 = sched.build_connector_meta(sched_out2)
    assert meta2.load_event >= 0

    # Request finishes BEFORE load completes -> deferred
    sched.request_finished(req2, block_ids=[])
    assert req2.request_id in sched._reqs_to_load
    assert sched._reqs_to_load[req2.request_id].finished is True

    # Now simulate load completion -> cleanup fires
    simulate_load_completion(sched, {req2.request_id})
    assert req2.request_id not in sched._reqs_to_load


# ---------------------------------------------------------------------------
# Test 6: Eager store preemption cleanup
# ---------------------------------------------------------------------------
def test_eager_store_preemption_cleanup() -> None:
    """In eager mode, finishing a request during in-flight store defers cleanup."""
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    num_blocks = 2
    req = make_request(num_blocks=num_blocks)
    gpu_blocks = _allocate_gpu_blocks(gpu_pool, req, num_blocks, group_id=0)
    kv_blocks = KVCacheBlocks(blocks=(gpu_blocks,))
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)

    sched_out = make_scheduler_output({req.request_id: num_blocks * BLOCK_SIZE})
    meta = sched.build_connector_meta(sched_out)
    store_event = meta.store_event
    assert store_event >= 0

    # The request gets store_events populated
    assert req.request_id in sched._reqs_to_store
    store_state = sched._reqs_to_store[req.request_id]
    assert store_event in store_state.store_events

    # Finish request while store still in-flight -> deferred
    sched.request_finished(req, block_ids=[])
    assert req.request_id in sched._reqs_to_store
    assert sched._reqs_to_store[req.request_id].finished is True

    # Simulate store completion -> deferred cleanup fires
    simulate_store_completion(sched, store_event)
    assert req.request_id not in sched._reqs_to_store


# ---------------------------------------------------------------------------
# Test 7: In-flight finish deferred cleanup (load variant)
# ---------------------------------------------------------------------------
def test_inflight_finish_deferred_cleanup() -> None:
    """Store, then start a load, request_finished defers,
    load completion fires cleanup."""
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    num_blocks = 2

    # Store
    req = make_request(num_blocks=num_blocks)
    gpu_blocks = _allocate_gpu_blocks(gpu_pool, req, num_blocks, group_id=0)
    kv_blocks = KVCacheBlocks(blocks=(gpu_blocks,))
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)
    sched_out = make_scheduler_output({req.request_id: num_blocks * BLOCK_SIZE})
    meta = sched.build_connector_meta(sched_out)
    simulate_store_completion(sched, meta.store_event)

    # Load
    req2 = Request(
        request_id="req-inflight-load",
        prompt_token_ids=req.prompt_token_ids,
        sampling_params=req.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=req._block_hasher,
    )
    hit_tokens, _ = sched.get_num_new_matched_tokens(req2, num_computed_tokens=0)
    assert hit_tokens > 0

    gpu_blocks2 = gpu_pool.get_new_blocks(num_blocks)
    kv_blocks2 = KVCacheBlocks(blocks=(gpu_blocks2,))
    sched.update_state_after_alloc(req2, kv_blocks2, num_external_tokens=hit_tokens)

    sched_out2 = make_scheduler_output({req2.request_id: 1})
    meta2 = sched.build_connector_meta(sched_out2)
    assert meta2.load_event >= 0

    # Finish before load completes
    sched.request_finished(req2, block_ids=[])
    assert req2.request_id in sched._reqs_to_load

    # Simulate load completion -> request removed
    simulate_load_completion(sched, {req2.request_id})
    assert req2.request_id not in sched._reqs_to_load


# ---------------------------------------------------------------------------
# Test 8: Null GPU blocks are skipped in store and load transfer pairs
# ---------------------------------------------------------------------------
def test_multi_group_null_blocks_skipped() -> None:
    """Null GPU blocks (no block_hash) must not appear in store or load pairs.

    In eager store mode, _prepare_eager_store_specs skips blocks whose
    block_hash is None (null blocks have no hash). We verify this by mixing
    real hashed blocks with unhashed (null-like) blocks in a single group and
    checking that only real blocks appear in the store list.
    """
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, num_groups=1, lazy=False)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    num_blocks = 2
    req = make_request(num_blocks=num_blocks)

    # Allocate real blocks (with hashes) and use the null_block as a placeholder
    gpu_blocks = _allocate_gpu_blocks(gpu_pool, req, num_blocks, group_id=0)
    null_block = gpu_pool.null_block

    # Mix: [real_block, null_block] — null_block has no hash, should be skipped
    mixed_blocks = [gpu_blocks[0], null_block]
    kv_blocks = KVCacheBlocks(blocks=(mixed_blocks,))
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)

    sched_out = make_scheduler_output({req.request_id: num_blocks * BLOCK_SIZE})
    meta = sched.build_connector_meta(sched_out)

    # Null block's ID should NOT appear in store_gpu_blocks
    null_block_id = null_block.block_id
    assert null_block_id not in meta.store_gpu_blocks, (
        f"Null block id {null_block_id} should not appear in store transfer pairs"
    )

    # Only real block should be scheduled for store
    assert len(meta.store_gpu_blocks) == 1
    assert gpu_blocks[0].block_id in meta.store_gpu_blocks

    # Complete the store
    assert meta.store_event >= 0
    simulate_store_completion(sched, meta.store_event)

    # Create matching request and get load hit
    req2 = Request(
        request_id="req-null-load",
        prompt_token_ids=req.prompt_token_ids,
        sampling_params=req.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=req._block_hasher,
    )
    hit_tokens, is_async = sched.get_num_new_matched_tokens(req2, num_computed_tokens=0)
    # Only 1 block was stored (the real one)
    assert hit_tokens == BLOCK_SIZE
    assert is_async is True

    # Allocate new GPU blocks for the load
    gpu_blocks2 = gpu_pool.get_new_blocks(1)
    kv_blocks2 = KVCacheBlocks(blocks=([gpu_blocks2[0], null_block],))
    sched.update_state_after_alloc(req2, kv_blocks2, num_external_tokens=hit_tokens)

    sched_out2 = make_scheduler_output({req2.request_id: 1})
    meta2 = sched.build_connector_meta(sched_out2)

    # Null block's ID should NOT appear in load_gpu_blocks
    assert null_block_id not in meta2.load_gpu_blocks, (
        f"Null block id {null_block_id} should not appear in load transfer pairs"
    )


# ---------------------------------------------------------------------------
# Test 9: Chunked prefill updates gpu_block_ids (not duplicated)
# ---------------------------------------------------------------------------
def test_chunked_prefill_updates_gpu_block_ids() -> None:
    """update_state_after_alloc called twice should update,
    not duplicate, gpu_block_ids."""
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    num_blocks = 4
    req = make_request(num_blocks=num_blocks)

    # First call: 2 blocks
    gpu_blocks_first = _allocate_gpu_blocks(gpu_pool, req, 2, group_id=0)
    kv_blocks_first = KVCacheBlocks(blocks=(gpu_blocks_first,))
    sched.update_state_after_alloc(req, kv_blocks_first, num_external_tokens=0)

    assert req.request_id in sched._reqs_to_store
    entry = sched._reqs_to_store[req.request_id]
    assert len(entry.gpu_block_ids[0]) == 2

    # Second call: 4 blocks (growing block list with chunked prefill)
    gpu_blocks_second = _allocate_gpu_blocks(gpu_pool, req, num_blocks, group_id=0)
    kv_blocks_second = KVCacheBlocks(blocks=(gpu_blocks_second,))
    sched.update_state_after_alloc(req, kv_blocks_second, num_external_tokens=0)

    # Should still be exactly 1 entry in _reqs_to_store
    assert list(sched._reqs_to_store.keys()).count(req.request_id) == 1
    entry = sched._reqs_to_store[req.request_id]
    # gpu_block_ids should be updated to the new set (4 blocks), not appended
    assert len(entry.gpu_block_ids[0]) == num_blocks


# ---------------------------------------------------------------------------
# Test 10: Partial GPU prefix hit + CPU load + new compute blocks
# ---------------------------------------------------------------------------
def test_partial_gpu_prefix_plus_cpu_load() -> None:
    """When GPU has a prefix cache hit for the first N blocks, CPU has a
    hit for the next M blocks, and there are P new blocks needing fresh
    compute, the block layout is:

        | comp (N) | ext_comp (M) | new (P) |

    External blocks sit in the middle — not at the beginning or end.
    The load path must target hashes at positions [N, N+M).

    Request: 6 blocks (0..5).
    - Store all 6 to CPU.
    - New request: GPU prefix cache hits blocks 0,1 (hashed).
      CPU hits blocks 2,3. Blocks 4,5 are new (need compute).
    - update_state_after_alloc receives 6 GPU blocks:
      [0,1] hashed (comp), [2,3] unhashed (ext_comp), [4,5] unhashed (new).
    - Load must target hash positions 2,3.
    """
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    num_blocks = 6
    req = make_request(num_blocks=num_blocks)

    # Store all 6 blocks to CPU via eager store.
    gpu_blocks = _allocate_gpu_blocks(gpu_pool, req, num_blocks, group_id=0)
    kv_blocks = KVCacheBlocks(blocks=(gpu_blocks,))
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)
    sched_out = make_scheduler_output({req.request_id: num_blocks * BLOCK_SIZE})
    meta = sched.build_connector_meta(sched_out)
    assert meta.store_event >= 0
    simulate_store_completion(sched, meta.store_event)

    # New request with same tokens — but only partial GPU prefix hit.
    req2 = Request(
        request_id="req-partial-gpu",
        prompt_token_ids=req.prompt_token_ids,
        sampling_params=req.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=req._block_hasher,
    )

    # GPU prefix cache hits the first 2 blocks.
    gpu_local_computed = 2 * BLOCK_SIZE
    hit_tokens, is_async = sched.get_num_new_matched_tokens(
        req2, num_computed_tokens=gpu_local_computed
    )
    # CPU should hit blocks 2,3 (not 4,5 — those are beyond the CPU range).
    num_cpu_hit_blocks = 2
    # Actually CPU has all 6 stored; it returns hits starting from position 2.
    # The number of CPU hit blocks = min(remaining request blocks, CPU cached).
    # Here remaining = 6 - 2 = 4 blocks are in CPU, so hit = 4 * BLOCK_SIZE.
    num_cpu_hit_blocks = 4
    assert hit_tokens == num_cpu_hit_blocks * BLOCK_SIZE, (
        f"Expected {num_cpu_hit_blocks * BLOCK_SIZE} CPU hit tokens, got {hit_tokens}"
    )
    assert is_async is True

    # Simulate what the real scheduler does: only accept 2 of the 4 CPU hit
    # blocks as external (e.g. due to budget constraints), leaving 2 new
    # blocks for fresh compute.
    num_ext_blocks = 2
    num_new_blocks = 2
    external_tokens = num_ext_blocks * BLOCK_SIZE

    # Build block list matching real layout: | comp(2) | ext_comp(2) | new(2) |
    # comp: GPU prefix cache hit — blocks with hashes
    gpu_comp = _allocate_gpu_blocks(gpu_pool, req2, 2, group_id=0)
    # ext_comp + new: freshly allocated, no hashes
    gpu_ext_and_new = gpu_pool.get_new_blocks(num_ext_blocks + num_new_blocks)
    all_gpu_blocks = gpu_comp + gpu_ext_and_new
    kv_blocks2 = KVCacheBlocks(blocks=(all_gpu_blocks,))

    # Critical call: with 2 hashed comp blocks and 2 external tokens worth
    # of blocks, the manager must derive skipped=2 and load hashes [2,3].
    sched.update_state_after_alloc(
        req2, kv_blocks2, num_external_tokens=external_tokens
    )

    sched_out2 = make_scheduler_output({req2.request_id: num_new_blocks * BLOCK_SIZE})
    meta2 = sched.build_connector_meta(sched_out2)
    assert meta2.load_event >= 0, "Expected a load event for partial GPU + CPU hit"
    assert len(meta2.load_gpu_blocks) == num_ext_blocks
    assert len(meta2.load_cpu_blocks) == num_ext_blocks

    # Verify the load targets the ext_comp GPU blocks (positions 2,3),
    # not the comp blocks (0,1) or new blocks (4,5).
    ext_block_ids = [b.block_id for b in gpu_ext_and_new[:num_ext_blocks]]
    for bid in meta2.load_gpu_blocks:
        assert bid in ext_block_ids, (
            f"Load GPU block {bid} should be an ext_comp block, not a comp or new block"
        )
