# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifetime of a hybrid request's reusable Mamba prompt-boundary state.

A hybrid (attention + Mamba "align") prefix-cache hit is the minimum over
groups, and the Mamba side of it is a single state block. That block must stay
out of the free queue for as long as the request's attention blocks do, and
must be the last of the request's blocks to be evicted afterwards; otherwise
the next turn finds every attention block and no state, and misses the whole
prompt.
"""

import itertools

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

HASH_BLOCK_SIZE = 16
NUM_BLOCKS = 32
CHURN_PROMPT_LEN = 8 * HASH_BLOCK_SIZE
CHURN_REQUESTS = 12  # cycles the pool several times

_token_ids = itertools.count(1000)


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


def _make_config(
    mamba_block_size: int, mamba_first: bool, retention_interval: int | None
) -> KVCacheConfig:
    attn = KVCacheGroupSpec(
        ["attn"],
        FullAttentionSpec(
            block_size=HASH_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )
    mamba = KVCacheGroupSpec(
        ["mamba"],
        MambaSpec(
            block_size=mamba_block_size,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
    )
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[mamba, attn] if mamba_first else [attn, mamba],
        prefix_cache_retention_interval=retention_interval,
    )


def _make_manager(
    mamba_block_size: int,
    mamba_first: bool = False,
    retention_interval: int | None = None,
) -> KVCacheManager:
    config = _make_config(mamba_block_size, mamba_first, retention_interval)
    return KVCacheManager(
        config,
        max_model_len=4096,
        scheduler_block_size=max(
            group.kv_cache_spec.block_size for group in config.kv_cache_groups
        ),
        hash_block_size=HASH_BLOCK_SIZE,
        enable_caching=True,
    )


def _make_request(request_id: str, prompt: list[int]) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=64),
        pooling_params=None,
        block_hasher=get_request_block_hasher(HASH_BLOCK_SIZE, sha256),
    )


def _release_cow_copies(manager: KVCacheManager) -> None:
    """The scheduler's release of CoW endpoints once the copy has run."""
    _, retained = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(retained)


def _prefill(manager: KVCacheManager, request: Request, chunks: list[int]) -> None:
    computed, num_computed, _ = manager.get_computed_blocks(request)
    for i, chunk in enumerate(chunks):
        blocks = manager.allocate_slots(
            request,
            chunk,
            num_computed if i == 0 else 0,
            computed if i == 0 else None,
        )
        assert blocks is not None
        # The scheduler commits the hit together with the first chunk.
        request.num_computed_tokens += chunk + (num_computed if i == 0 else 0)
        manager.coordinator.new_step_starts()
        _release_cow_copies(manager)


def _decode(manager: KVCacheManager, request: Request, num_steps: int) -> None:
    for _ in range(num_steps):
        request.append_output_token_ids(next(_token_ids))
        assert manager.allocate_slots(request, 1) is not None
        request.num_computed_tokens += 1
        manager.coordinator.new_step_starts()
        _release_cow_copies(manager)


def _churn(manager: KVCacheManager) -> None:
    """Admit and finish unrelated requests; each allocation pops the head of
    the free queue, so anything parked there is recycled."""
    for i in range(CHURN_REQUESTS):
        prompt = [next(_token_ids) for _ in range(CHURN_PROMPT_LEN)]
        other = _make_request(f"other-{i}", prompt)
        _prefill(manager, other, [CHURN_PROMPT_LEN])
        manager.free(other)


def _group_id(manager: KVCacheManager, spec_type: type) -> int:
    for group_id, group in enumerate(manager.kv_cache_config.kv_cache_groups):
        if isinstance(group.kv_cache_spec, spec_type):
            return group_id
    raise AssertionError(spec_type)


def _cached_blocks(manager: KVCacheManager, group_id: int):
    return [
        block
        for block in manager.block_pool.blocks
        if block.block_hash is not None and get_group_id(block.block_hash) == group_id
    ]


def _follow_up(request: Request, num_new_tokens: int) -> Request:
    tokens = list(request.all_token_ids)
    tokens += [next(_token_ids) for _ in range(num_new_tokens)]
    return _make_request(f"{request.request_id}-next", tokens)


# (mamba block size, prompt length, chunks, reusable boundary)
BOUNDARY_CASES = [
    # Boundary on a Mamba block boundary: the state block is retired by
    # ``remove_skipped_blocks`` at the first decode step.
    pytest.param(HASH_BLOCK_SIZE, 65, [64, 1], 64, id="block-aligned"),
    # Boundary inside a Mamba block: the state is CoW-copied out of the running
    # block, and the copy is released once the copy has run.
    pytest.param(2 * HASH_BLOCK_SIZE, 49, [48, 1], 48, id="partial-tail"),
]


# ``None`` keeps every state (dense); ``0`` keeps only the reusable boundary.
RETENTION = [pytest.param(None, id="dense"), pytest.param(0, id="sparse")]


@pytest.mark.parametrize("retention_interval", RETENTION)
@pytest.mark.parametrize(
    "mamba_block_size, prompt_len, chunks, boundary", BOUNDARY_CASES
)
def test_boundary_state_outlives_decode(
    mamba_block_size, prompt_len, chunks, boundary, retention_interval
):
    manager = _make_manager(mamba_block_size, retention_interval=retention_interval)
    pool = manager.block_pool
    mamba_group = _group_id(manager, MambaSpec)

    request = _make_request("req", [next(_token_ids) for _ in range(prompt_len)])
    _prefill(manager, request, chunks)
    [snapshot] = _cached_blocks(manager, mamba_group)
    key = snapshot.block_hash
    assert snapshot.block_hash_num_tokens == boundary

    _decode(manager, request, 2)
    # Held on the request's behalf: not in the free queue, hash intact.
    assert snapshot.ref_cnt >= 1
    assert snapshot.block_hash == key
    # ...at a bounded cost: the running state and the boundary state.
    held = [
        block
        for block in manager.get_blocks(request.request_id).blocks[mamba_group]
        if not block.is_null
    ]
    assert len(held) <= 2

    _churn(manager)
    assert snapshot.block_hash == key, "boundary state recycled mid-request"

    _decode(manager, request, 2)
    manager.free(request)
    # The hold is released with the request: nothing leaks.
    assert pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert snapshot.ref_cnt == 0
    assert snapshot.block_hash == key

    follow_up = _follow_up(request, 32)
    _, hit_lengths = manager.coordinator.find_longest_cache_hit_per_group(
        follow_up.block_hashes, follow_up.num_tokens - 1
    )
    assert hit_lengths[mamba_group] == boundary
    _, num_computed, _ = manager.get_computed_blocks(follow_up)
    assert num_computed == boundary


@pytest.mark.parametrize("retention_interval", RETENTION)
@pytest.mark.parametrize("resume", ["preempted", "sibling"])
@pytest.mark.parametrize(
    "mamba_block_size, prompt_len, chunks, boundary", BOUNDARY_CASES
)
def test_boundary_state_held_after_cache_hit(
    mamba_block_size, prompt_len, chunks, boundary, resume, retention_interval
):
    """A request that takes the boundary state over from the cache (resumed
    after preemption, or a sibling with the same prompt) holds it too."""
    manager = _make_manager(mamba_block_size, retention_interval=retention_interval)
    pool = manager.block_pool
    mamba_group = _group_id(manager, MambaSpec)

    prompt = [next(_token_ids) for _ in range(prompt_len)]
    producer = _make_request("producer", prompt)
    _prefill(manager, producer, chunks)
    _decode(manager, producer, 1)
    [snapshot] = _cached_blocks(manager, mamba_group)
    key = snapshot.block_hash
    if resume == "preempted":
        manager.free(producer)

    request = _make_request("req", prompt)
    _, num_computed, _ = manager.get_computed_blocks(request)
    assert num_computed == boundary
    _prefill(manager, request, [prompt_len - num_computed])
    _decode(manager, request, 2)
    if resume == "sibling":
        manager.free(producer)

    assert snapshot.ref_cnt >= 1
    _churn(manager)
    assert snapshot.block_hash == key, "boundary state recycled mid-request"

    manager.free(request)
    assert pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert snapshot.ref_cnt == 0
    follow_up = _follow_up(request, 32)
    _, num_computed, _ = manager.get_computed_blocks(follow_up)
    assert num_computed == boundary


@pytest.mark.parametrize("deferred_free", [False, True], ids=["free", "deferred"])
@pytest.mark.parametrize(
    "mamba_first", [False, True], ids=["attn-first", "mamba-first"]
)
def test_request_end_evicts_attention_before_boundary_state(mamba_first, deferred_free):
    manager = _make_manager(HASH_BLOCK_SIZE, mamba_first)
    pool = manager.block_pool
    attn_group = _group_id(manager, FullAttentionSpec)
    mamba_group = _group_id(manager, MambaSpec)

    request = _make_request("req", [next(_token_ids) for _ in range(65)])
    _prefill(manager, request, [64, 1])
    _decode(manager, request, 2)
    [snapshot] = _cached_blocks(manager, mamba_group)
    attn_cached = {
        block.block_id
        for block in manager.get_blocks(request.request_id).blocks[attn_group]
        if block.block_hash is not None
    }
    assert attn_cached

    if deferred_free:
        # Scheduler._free_request_blocks with ``defer_block_free``.
        blocks = manager.pop_blocks_for_free(request)
        pool.free_blocks(reversed(blocks))
    else:
        manager.free(request)
    assert pool.get_num_free_blocks() == NUM_BLOCKS - 1

    queue = [block.block_id for block in pool.free_block_queue.get_all_free_blocks()]
    snapshot_pos = queue.index(snapshot.block_id)
    attn_positions = [queue.index(block_id) for block_id in attn_cached]
    assert max(attn_positions) < snapshot_pos, (
        "boundary state evicted before the request's attention blocks"
    )
