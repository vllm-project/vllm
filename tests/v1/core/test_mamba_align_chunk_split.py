# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefill chunk splitting (`_mamba_block_aligned_split`).

Invariant: slot `p` holds the state after exactly `(p + 1) * block_size` tokens.
State is written at chunk ends, so a chunk ending mid-block leaves its slot at
the wrong offset, and a later chunk crossing that boundary publishes it anyway.
Requests resuming from it then restore a truncated state (#43559).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_group_id
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from .utils import create_requests

pytestmark = pytest.mark.cpu_test

# Mirrors the deployment where the poisoning was observed (Qwen3.6-27B): mamba
# block 1600, MTP with 3 draft tokens, prompts shorter than 2 mamba blocks.
ATTN_BLOCK_SIZE = 16
MAMBA_BLOCK_SIZE = 1600
NUM_SPEC = 3
PROMPT_LEN = 2002
MAMBA_GROUP_ID = 1


def _make_hybrid_kv_cache_manager(
    num_prefill_checkpoint_blocks: int = 0,
    retention_interval: int | None = None,
    attn_block_size: int = ATTN_BLOCK_SIZE,
) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        prefix_cache_retention_interval=retention_interval,
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=attn_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=MAMBA_BLOCK_SIZE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                    num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
                ),
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=262144,
        scheduler_block_size=MAMBA_BLOCK_SIZE,
        hash_block_size=attn_block_size,
        enable_caching=True,
        use_eagle=True,
    )


def _split(
    request: Request,
    num_new_tokens: int,
    use_eagle: bool = True,
    partial_hit: bool = False,
    num_prefill_checkpoint_blocks: int = 0,
    retention_interval: int | None = None,
    eagle_reach_margin: int = 0,
) -> int:
    """Call the real `Scheduler._mamba_block_aligned_split` on a stub self."""
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE),
        use_eagle=use_eagle,
        mamba_retention_interval=retention_interval,
        mamba_eagle_reach_margin=eagle_reach_margin,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        # `prefix_match_unit` finer than the block size (#46384).
        mamba_partial_cache_hit=partial_hit,
        hash_block_size=ATTN_BLOCK_SIZE,
        mamba_has_prefill_checkpoint_blocks=(
            num_prefill_checkpoint_blocks > 0 and not use_eagle
        ),
    )
    return Scheduler._mamba_block_aligned_split(stub, request, num_new_tokens)


@pytest.mark.parametrize(
    ("prompt_len", "num_new_tokens", "use_eagle", "expected"),
    [
        (2002, 2002, False, 2002),
        (3602, 2000, False, 2000),
        (3602, 3602, True, MAMBA_BLOCK_SIZE),
    ],
)
def test_internal_checkpoint_split(
    prompt_len: int, num_new_tokens: int, use_eagle: bool, expected: int
) -> None:
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    assert (
        _split(
            request,
            num_new_tokens,
            use_eagle=use_eagle,
            num_prefill_checkpoint_blocks=1,
        )
        == expected
    )
    if not use_eagle:
        manager = _make_hybrid_kv_cache_manager(num_prefill_checkpoint_blocks=1)
        assert manager.allocate_slots(request, expected) is not None
        mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
        blocks = mamba_manager.req_to_blocks[request.request_id]
        assert all(not block.is_null for block in blocks)  # checkpoint + running state


def _run_chunked_prefill(
    manager: KVCacheManager, request: Request, budgets: list[int]
) -> dict[int, int]:
    """Prefill `request`, one step per entry in `budgets`.

    A zero-token split means the budget cannot fund an aligned chunk; the
    scheduler defers the request to a later step, so this does too.

    Returns physical mamba block id -> the token offset of the state it holds,
    mirroring the GDN kernel: the running slot ends up at the chunk end.
    """
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    state_at: dict[int, int] = {}
    # `budgets` fragments the first steps; afterwards the request is alone and
    # gets as much as it can use, so the prefill always finishes.
    for step in range(len(budgets) + 64):
        computed = request.num_computed_tokens
        if computed >= request.num_tokens:
            break
        budget = budgets[step] if step < len(budgets) else request.num_tokens
        num_new = _split(request, min(request.num_tokens - computed, budget))
        if num_new == 0:
            continue
        assert (
            manager.allocate_slots(request, num_new, num_lookahead_tokens=NUM_SPEC)
            is not None
        )
        request.num_computed_tokens = computed + num_new
        blocks = mamba_manager.req_to_blocks[request.request_id]
        running = cdiv(request.num_computed_tokens, MAMBA_BLOCK_SIZE) - 1
        state_at[blocks[running].block_id] = request.num_computed_tokens
    return state_at


def _count_cached_boundary_states(
    manager: KVCacheManager, request: Request, state_at: dict[int, int]
) -> int:
    """Assert every hash-cached mamba slot holds the state its hash claims.

    Covers both full-block snapshots (`(p + 1) * block_size`) and the
    partial-tail entries align mode registers at an exact token count.

    Returns the number of cached slots checked.
    """
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    checked = 0
    for pos, block in enumerate(mamba_manager.req_to_blocks[request.request_id]):
        if block.is_null or block.block_hash is None:
            continue
        claimed = block.block_hash_num_tokens
        assert state_at.get(block.block_id) == claimed, (
            f"mamba slot {pos} is hashed as state@{claimed} but holds "
            f"state@{state_at.get(block.block_id)}"
        )
        checked += 1
    return checked


def _prefill(prompt_len: int, budgets: list[int]) -> int:
    manager = _make_hybrid_kv_cache_manager()
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    state_at = _run_chunked_prefill(manager, request, budgets)
    assert request.num_computed_tokens == prompt_len, "prefill did not complete"
    return _count_cached_boundary_states(manager, request, state_at)


def test_fragmented_first_chunk_does_not_poison_mamba_prefix_cache() -> None:
    """EAGLE zeroes `last_cache_position` for prompts under two blocks.

    Past that point any chunk end used to be accepted, so a short first chunk
    (concurrent prefills sharing the budget) left slot 0 at state@364 while the
    next chunk crossed 1600 and published slot 0 as state@1600.
    """
    _prefill(PROMPT_LEN, budgets=[364, PROMPT_LEN])


def test_fragmented_tail_chunk_does_not_poison_mamba_prefix_cache() -> None:
    """Same poisoning one block in, where a hit is still cacheable.

    `last_cache_position` is 1600, so the chunk ending there is cached. The next
    chunk used to be free to stop mid-block (slot 1 at state@2600) and the one
    after it crossed 3200, publishing slot 1 as state@3200.
    """
    assert _prefill(3602, budgets=[1600, 1000, 3602]) > 0


@pytest.mark.parametrize("first_chunk", [800, 900, 1599, 1601, 2000])
def test_intermediate_chunk_ends_stay_block_aligned(first_chunk: int) -> None:
    """Every non-final prefill chunk must end on a mamba block boundary."""
    _prefill(PROMPT_LEN, budgets=[first_chunk, PROMPT_LEN, PROMPT_LEN])


@pytest.mark.parametrize(
    ("block_size", "prompt_len", "budgets"),
    [
        # Kimi-K3-scale mamba blocks: TP8 shards the recurrent state 8 ways
        # (~1.5k block), DEP16 keeps it whole (~12k block). The first budget
        # walks the request up to `last_cache_position`; the second is the
        # sub-block fragment that lands in the unguarded tail region.
        (1536, 30000, [27648, 1024, 30000]),
        (12288, 30000, [12288, 4000, 30000]),
        (12288, 41000, [24576, 8000, 41000]),
    ],
)
def test_poisoning_is_block_size_independent(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
    prompt_len: int,
    budgets: list[int],
) -> None:
    """The invariant is per-block, so large mamba blocks are not safer."""
    import sys

    monkeypatch.setattr(sys.modules[__name__], "MAMBA_BLOCK_SIZE", block_size)
    assert _prefill(prompt_len, budgets=budgets) > 0


@pytest.mark.parametrize("partial_hit", [False, True])
@pytest.mark.parametrize("resume_at", [331, 1599, 1601, 2531, 3011])
@pytest.mark.parametrize(
    ("num_prefill_checkpoint_blocks", "use_eagle"),
    [(0, True), (1, False)],
)
def test_unaligned_resume_never_runs_past_its_block(
    partial_hit: bool,
    resume_at: int,
    num_prefill_checkpoint_blocks: int,
    use_eagle: bool,
) -> None:
    """A prefill resuming mid-block must re-align before crossing a boundary.

    Reachable with a finer `prefix_match_unit` (its partial-tail stop ends a
    chunk off-grid by design) and with unaligned external tokens from a KV
    connector.
    """
    prompt_len = 3602
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    tail_boundary = prompt_len // ATTN_BLOCK_SIZE * ATTN_BLOCK_SIZE

    pos, ends = resume_at, []
    while pos < prompt_len:
        request.num_computed_tokens = pos
        num_new = _split(
            request,
            prompt_len - pos,
            use_eagle=use_eagle,
            partial_hit=partial_hit,
            num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
        )
        assert num_new > 0, f"no progress at {pos}"
        if pos % MAMBA_BLOCK_SIZE != 0:
            block_end = (pos // MAMBA_BLOCK_SIZE + 1) * MAMBA_BLOCK_SIZE
            assert pos + num_new <= block_end, (
                f"chunk [{pos}, {pos + num_new}) starts mid-block and runs past "
                f"{block_end}; the slot holding state@{pos} gets hashed as "
                f"state@{block_end}"
            )
        pos += num_new
        ends.append(pos)

    for end in ends[:-1]:
        aligned = end % MAMBA_BLOCK_SIZE == 0
        assert aligned or (partial_hit and end == tail_boundary), (
            f"intermediate chunk end {end} is neither block-aligned nor the "
            f"partial-tail boundary"
        )


def test_split_stops_at_every_boundary_without_checkpoints() -> None:
    """Without internal checkpoints a reusable state exists only at chunk
    ends, so each chunk stops at its next block boundary; otherwise a sibling
    sharing fewer tokens than the deepest chunk end can never hit (#52897)."""
    prompt_len = 3 * MAMBA_BLOCK_SIZE + 500
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    ends = []
    while request.num_computed_tokens < prompt_len:
        num_new = _split(
            request, prompt_len - request.num_computed_tokens, use_eagle=False
        )
        request.num_computed_tokens += num_new
        ends.append(request.num_computed_tokens)
    assert ends == [
        MAMBA_BLOCK_SIZE,
        2 * MAMBA_BLOCK_SIZE,
        3 * MAMBA_BLOCK_SIZE,
        prompt_len,
    ]


def test_split_keeps_last_boundary_reachable_under_eagle() -> None:
    """The speculative one-block back-off forfeited a full mamba block of
    reusable prefix (measured as half the cache on aligned prompts, #52897).
    With a state at every crossed boundary, a lookup whose last block is
    pruned falls back one block instead of missing, so the back-off is gone:
    the deepest boundary below the prompt stays a chunk end."""
    prompt_len = 3 * MAMBA_BLOCK_SIZE + 500
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    ends = []
    while request.num_computed_tokens < prompt_len:
        num_new = _split(
            request, prompt_len - request.num_computed_tokens, use_eagle=True
        )
        request.num_computed_tokens += num_new
        ends.append(request.num_computed_tokens)
    assert 3 * MAMBA_BLOCK_SIZE in ends


def test_quiet_prefill_caches_state_at_every_boundary() -> None:
    """End to end through the manager: a single-budget ("quiet") prefill must
    leave a hash-consistent cached state at every crossed boundary, not only
    at the chunk ends the budget happens to produce (#52897)."""
    prompt_len = 3 * MAMBA_BLOCK_SIZE + 500
    manager = _make_hybrid_kv_cache_manager()
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    state_at = _run_chunked_prefill(manager, request, budgets=[prompt_len])
    assert set(state_at.values()) == {
        MAMBA_BLOCK_SIZE,
        2 * MAMBA_BLOCK_SIZE,
        3 * MAMBA_BLOCK_SIZE,
        prompt_len,
    }
    # How many of these the manager hashes during an in-flight prefill is
    # lookup-side policy (eagle lookahead defers hashing); every hashed slot
    # must still be consistent with the state it holds.
    assert _count_cached_boundary_states(manager, request, state_at) >= 1


def _chunk_ends(prompt_len: int, **split_kwargs) -> list[int]:
    """Chunk ends of an unbudgeted prefill through the real split."""
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    ends = []
    while request.num_computed_tokens < prompt_len:
        num_new = _split(
            request, prompt_len - request.num_computed_tokens, **split_kwargs
        )
        assert num_new > 0
        request.num_computed_tokens += num_new
        ends.append(request.num_computed_tokens)
    return ends


@pytest.mark.parametrize(
    ("prompt_len", "expected"),
    [
        # Unaligned: the eagle-reachable boundary, the replay boundary, the tail.
        (
            4 * MAMBA_BLOCK_SIZE + 892,
            [3 * MAMBA_BLOCK_SIZE, 4 * MAMBA_BLOCK_SIZE, 4 * MAMBA_BLOCK_SIZE + 892],
        ),
        # Aligned: the replay boundary is one block below the prompt end.
        (
            4 * MAMBA_BLOCK_SIZE,
            [2 * MAMBA_BLOCK_SIZE, 3 * MAMBA_BLOCK_SIZE, 4 * MAMBA_BLOCK_SIZE],
        ),
    ],
)
def test_split_default_retention_stops_only_where_states_are_kept(
    prompt_len: int, expected: list[int]
) -> None:
    """`prefix_cache_retention_interval=0` (the default) hashes only the
    replay boundary and shared-prefix junctions, so a boundary stop anywhere
    else splits the prefill for a state that is discarded. Under EAGLE/MTP the
    attention lookup drops one block, so the state a replay can reach is one
    block below the replay boundary: the split materializes both, and nothing
    else (#53479)."""
    ends = _chunk_ends(
        prompt_len,
        use_eagle=True,
        retention_interval=0,
        eagle_reach_margin=MAMBA_BLOCK_SIZE,
    )
    assert ends == expected


def test_split_segment_retention_stops_at_segment_boundaries() -> None:
    """A positive interval keeps one state per interval-sized segment plus
    the replay boundary, so those are the only mandatory chunk ends."""
    prompt_len = 5 * MAMBA_BLOCK_SIZE + 100
    ends = _chunk_ends(
        prompt_len, use_eagle=False, retention_interval=2 * MAMBA_BLOCK_SIZE
    )
    assert ends == [
        2 * MAMBA_BLOCK_SIZE,
        4 * MAMBA_BLOCK_SIZE,
        5 * MAMBA_BLOCK_SIZE,
        prompt_len,
    ]


def test_split_dense_retention_keeps_every_boundary_stop() -> None:
    """`None`, or an interval at/below the block size, hashes every boundary
    state, so every boundary stays a chunk end."""
    prompt_len = 3 * MAMBA_BLOCK_SIZE + 500
    dense = [MAMBA_BLOCK_SIZE, 2 * MAMBA_BLOCK_SIZE, 3 * MAMBA_BLOCK_SIZE, prompt_len]
    assert (
        _chunk_ends(
            prompt_len,
            use_eagle=True,
            retention_interval=None,
            eagle_reach_margin=MAMBA_BLOCK_SIZE,
        )
        == dense
    )
    assert (
        _chunk_ends(prompt_len, use_eagle=False, retention_interval=MAMBA_BLOCK_SIZE)
        == dense
    )


def _cached_mamba_states(manager: KVCacheManager) -> set[int]:
    """Token offsets of every hash-cached mamba state in the pool (align mode
    relocates the running state and frees earlier slots into the pool, where a
    hashed one stays findable)."""
    cache = manager.block_pool.cached_block_hash_to_block._cache
    return {
        block.block_hash_num_tokens
        for key, value in cache.items()
        if get_group_id(key) == MAMBA_GROUP_ID
        for block in (value.values() if isinstance(value, dict) else [value])
    }


@pytest.mark.parametrize("keep_eagle_reach", [True, False])
def test_default_retention_keeps_the_eagle_reachable_state(
    keep_eagle_reach: bool,
) -> None:
    """End to end through the manager, in "align" geometry (attention and
    mamba blocks equal, as the engine forces): with the default interval and
    EAGLE/MTP, sparse retention kept only the replay boundary's state (block 3
    for a 4-block-plus-tail prompt) while the pruned lookup reaches one block
    less, so an identical prompt only hit once a junction formed — on its
    third send. The state one block below must be kept too, so the second
    send hits (#53479). `keep_eagle_reach=False` reproduces the old behaviour
    by zeroing the margin, proving the test discriminates."""
    block = MAMBA_BLOCK_SIZE
    manager = _make_hybrid_kv_cache_manager(retention_interval=0, attn_block_size=block)
    assert manager.coordinator.eagle_reach_margin == block
    if not keep_eagle_reach:
        for single_type_manager in manager.coordinator.single_type_managers:
            single_type_manager.eagle_reach_margin = 0
    prompt_len = 4 * block + 892
    (request,) = create_requests(
        1, num_tokens=prompt_len, block_size=block, same_prompt=True, req_ids=["p0"]
    )
    _run_chunked_prefill(manager, request, budgets=[prompt_len])
    (repeat,) = create_requests(
        1, num_tokens=prompt_len, block_size=block, same_prompt=True, req_ids=["p1"]
    )
    # The lookup drops the last matched attention block (4 -> 3 blocks) and
    # needs the mamba state exactly there. Sparse retention keeps nothing else.
    if keep_eagle_reach:
        assert _cached_mamba_states(manager) == {3 * block, 4 * block}
        assert manager.get_computed_blocks(repeat)[1] == 3 * block
    else:
        assert _cached_mamba_states(manager) == {4 * block}
        assert manager.get_computed_blocks(repeat)[1] == 0
