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
) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=ATTN_BLOCK_SIZE,
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
        hash_block_size=ATTN_BLOCK_SIZE,
        enable_caching=True,
        use_eagle=True,
    )


def _split(
    request: Request,
    num_new_tokens: int,
    use_eagle: bool = True,
    use_eagle_block_drop: bool | None = None,
    partial_hit: bool = False,
    num_prefill_checkpoint_blocks: int = 0,
) -> int:
    """Call the real `Scheduler._mamba_block_aligned_split` on a stub self."""
    if use_eagle_block_drop is None:
        use_eagle_block_drop = use_eagle
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE),
        use_eagle_block_drop=use_eagle_block_drop,
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


def test_disabling_eagle_block_drop_keeps_the_trailing_cache_boundary() -> None:
    (request,) = create_requests(1, num_tokens=3602, block_size=ATTN_BLOCK_SIZE)

    with_drop = _split(request, request.num_tokens, use_eagle_block_drop=True)
    without_drop = _split(request, request.num_tokens, use_eagle_block_drop=False)

    assert with_drop == MAMBA_BLOCK_SIZE
    assert without_drop == 2 * MAMBA_BLOCK_SIZE


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


# Heterogeneous layouts: `cache_config.block_size` is the minimum over all
# groups and can be finer than the mamba state grid (page-size matching, a
# drafter/attention group with a smaller block, or an explicit --block-size;
# see also #53142 for the worker-side sibling). Production geometry that
# exposed this: draft attention block 816, mamba block 1648 — 816k equals
# 1648m only at the LCM (84048), so chunk stops on the generic grid never
# materialize a state and the mamba group publishes no prefix hashes.
HETERO_ATTN_BLOCK_SIZE = 816
HETERO_MAMBA_BLOCK_SIZE = 1648
HETERO_SCHEDULER_BLOCK_SIZE = 84048  # lcm(816, 1648)
HETERO_MAMBA_GROUP_ID = 1


def _make_heterogeneous_kv_cache_manager() -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention_layer"],
                FullAttentionSpec(
                    block_size=HETERO_ATTN_BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=HETERO_MAMBA_BLOCK_SIZE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                ),
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=262144,
        scheduler_block_size=HETERO_SCHEDULER_BLOCK_SIZE,
        hash_block_size=ATTN_BLOCK_SIZE,
        enable_caching=True,
        use_eagle=True,
    )


def _hetero_split(request: Request, num_new_tokens: int) -> int:
    """Real split on a heterogeneous-layout stub `self`.

    `cache_config.block_size` is the group minimum (816); the mamba state
    grid is 1648. `mamba_state_block_size` mirrors the attribute
    `Scheduler.__init__` derives from the mamba group's spec.
    """
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=HETERO_ATTN_BLOCK_SIZE),
        mamba_state_block_size=HETERO_MAMBA_BLOCK_SIZE,
        use_eagle=True,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        hash_block_size=ATTN_BLOCK_SIZE,
        mamba_has_prefill_checkpoint_blocks=False,
    )
    return Scheduler._mamba_block_aligned_split(stub, request, num_new_tokens)


def test_heterogeneous_block_sizes_stop_chunks_on_the_mamba_grid() -> None:
    """Chunk ends must follow MambaSpec.block_size, not cache_config.block_size."""
    prompt_len = 4 * HETERO_MAMBA_BLOCK_SIZE + 70
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    pos, ends = 0, []
    while pos < prompt_len:
        request.num_computed_tokens = pos
        num_new = _hetero_split(request, prompt_len - pos)
        assert num_new > 0, f"no progress at {pos}"
        pos += num_new
        ends.append(pos)
    for end in ends[:-1]:
        assert end % HETERO_MAMBA_BLOCK_SIZE == 0, (
            f"intermediate chunk end {end} is off the mamba state grid; the "
            f"worker never commits a state there (816-grid stop)"
        )
    assert ends[0] % HETERO_MAMBA_BLOCK_SIZE == 0, (
        f"first chunk ended at {ends[0]}, off the mamba state grid "
        f"(816-grid stop); ends={ends}"
    )


def _hetero_prefill(prompt_len: int) -> tuple[KVCacheManager, Request, dict[int, int]]:
    """Prefill one request through the real manager under the hetero layout.

    Budgets one mamba block per step, the shape the fixed split produces for
    a request sharing the token budget (and the equal-geometry tests above).
    """
    manager = _make_heterogeneous_kv_cache_manager()
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    mamba_manager = manager.coordinator.single_type_managers[HETERO_MAMBA_GROUP_ID]
    state_at: dict[int, int] = {}
    while request.num_computed_tokens < request.num_tokens:
        computed = request.num_computed_tokens
        budget = min(HETERO_MAMBA_BLOCK_SIZE, request.num_tokens - computed)
        num_new = _hetero_split(request, budget)
        assert num_new > 0, f"no progress at {computed}"
        assert (
            manager.allocate_slots(request, num_new, num_lookahead_tokens=NUM_SPEC)
            is not None
        )
        request.num_computed_tokens = computed + num_new
        blocks = mamba_manager.req_to_blocks[request.request_id]
        running = cdiv(request.num_computed_tokens, HETERO_MAMBA_BLOCK_SIZE) - 1
        state_at[blocks[running].block_id] = request.num_computed_tokens
    return manager, request, state_at


def test_heterogeneous_block_sizes_publish_mamba_states_for_immediate_reuse() -> None:
    """Mamba states must be published where the worker materializes them.

    On the generic 816 grid no chunk ever ends on the 1648 state grid, so the
    mamba group publishes zero prefix hashes and a request repeating the same
    prompt immediately reuses nothing.
    """
    prompt_len = 4 * HETERO_MAMBA_BLOCK_SIZE + 70
    manager, first, state_at = _hetero_prefill(prompt_len)
    mamba_manager = manager.coordinator.single_type_managers[HETERO_MAMBA_GROUP_ID]

    published = [
        block.block_hash_num_tokens
        for block in mamba_manager.req_to_blocks[first.request_id]
        if not block.is_null and block.block_hash is not None
    ]
    full_block_entries = [p for p in published if p % HETERO_MAMBA_BLOCK_SIZE == 0]
    assert full_block_entries, (
        f"no mamba full-block state published (entries={published}); "
        f"immediate reuse on this layout is impossible"
    )
    # Safety: every hashed slot holds the state its hash claims.
    for block in mamba_manager.req_to_blocks[first.request_id]:
        if block.is_null or block.block_hash is None:
            continue
        assert state_at.get(block.block_id) == block.block_hash_num_tokens, (
            f"mamba slot hashed as state@{block.block_hash_num_tokens} but "
            f"holds state@{state_at.get(block.block_id)}"
        )

    (second,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    # The scheduler hashes prompt token ids deterministically, so mirror it:
    second.all_token_ids = first.all_token_ids
    second.block_hashes = first.block_hashes
    _, num_computed, _ = manager.get_computed_blocks(second)
    assert num_computed >= HETERO_MAMBA_BLOCK_SIZE, (
        f"immediate repeat of the same prompt reused {num_computed} tokens "
        f"(expected at least one legal mamba boundary)"
    )
    assert num_computed % HETERO_MAMBA_BLOCK_SIZE == 0


def test_equal_block_sizes_control_still_reuses() -> None:
    """Control: with cache_config.block_size == MambaSpec.block_size the
    behavior is the same before and after a heterogeneous-layout fix."""
    prompt_len = 2 * MAMBA_BLOCK_SIZE + 70
    manager = _make_hybrid_kv_cache_manager()
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    while request.num_computed_tokens < request.num_tokens:
        computed = request.num_computed_tokens
        num_new = _split(request, request.num_tokens - computed)
        assert num_new > 0
        assert (
            manager.allocate_slots(request, num_new, num_lookahead_tokens=NUM_SPEC)
            is not None
        )
        request.num_computed_tokens = computed + num_new
    (second,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    second.all_token_ids = request.all_token_ids
    second.block_hashes = request.block_hashes
    _, num_computed, _ = manager.get_computed_blocks(second)
    assert num_computed >= MAMBA_BLOCK_SIZE
    assert num_computed % MAMBA_BLOCK_SIZE == 0
