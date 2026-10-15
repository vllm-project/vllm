# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefill chunk splitting (`_mamba_block_aligned_split`).

Invariant: slot `p` holds the state after exactly `(p + 1) * block_size` tokens.
State is written at chunk ends, so a chunk ending mid-block leaves its slot at
the wrong offset, and a later chunk crossing that boundary publishes it anyway.
Requests resuming from it then restore a truncated state (#43559).
"""

import sys
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
    is_mamba_prefill_checkpoint_valid,
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


@pytest.mark.parametrize(
    (
        "query_start",
        "query_end",
        "checkpoint_position",
        "hash_size",
        "block_size",
        "alignment",
        "expected",
    ),
    [
        (0, 100, 88, 8, 64, 16, False),
        (0, 100, 96, 8, 64, 16, True),
        (0, 100, 96, 8, 64, None, False),
        (8, 100, 96, 8, 64, 16, False),
        (15360, 16896, 16768, 128, 1536, 16, False),
        (15360, 16960, 16896, 128, 1536, 16, True),
        (15360, 16960, 16768, 128, 1536, 16, True),
        (16064, 18433, 18432, 128, 1536, 16, False),
    ],
)
def test_mamba_prefill_checkpoint_valid(
    query_start: int,
    query_end: int,
    checkpoint_position: int,
    hash_size: int,
    block_size: int,
    alignment: int | None,
    expected: bool,
) -> None:
    assert (
        is_mamba_prefill_checkpoint_valid(
            query_start=query_start,
            query_end=query_end,
            checkpoint_position=checkpoint_position,
            hash_block_size=hash_size,
            mamba_block_size=block_size,
            checkpoint_alignment=alignment,
        )
        is expected
    )


def _make_hybrid_kv_cache_manager(
    num_prefill_checkpoint_blocks: int = 0,
    use_eagle: bool = True,
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
                    prefill_checkpoint_alignment=(
                        16 if num_prefill_checkpoint_blocks > 0 else None
                    ),
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
        use_eagle=use_eagle,
    )


def _split(
    request: Request,
    num_new_tokens: int,
    use_eagle: bool = True,
    use_eagle_block_drop: bool | None = None,
    partial_hit: bool = False,
    num_prefill_checkpoint_blocks: int = 0,
    max_num_scheduled_tokens: int = 16384,
    num_prefill_lookahead: int = 0,
) -> int:
    """Call the real `Scheduler._mamba_block_aligned_split` on a stub self."""
    if use_eagle_block_drop is None:
        use_eagle_block_drop = use_eagle
    stub = SimpleNamespace(
        block_size=MAMBA_BLOCK_SIZE,
        cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE),
        use_eagle_block_drop=use_eagle_block_drop,
        num_prefill_lookahead=num_prefill_lookahead,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        # `prefix_match_unit` finer than the block size (#46384).
        mamba_partial_cache_hit=partial_hit,
        mamba_shared_prefix_checkpoint=False,
        hash_block_size=ATTN_BLOCK_SIZE,
        mamba_has_prefill_checkpoint_blocks=(num_prefill_checkpoint_blocks > 0),
        mamba_prefill_checkpoint_alignment=(
            16 if num_prefill_checkpoint_blocks > 0 else None
        ),
    )
    return Scheduler._mamba_block_aligned_split(stub, request, num_new_tokens)


@pytest.mark.parametrize(
    ("prompt_len", "num_new_tokens", "use_eagle", "expected"),
    [
        (2002, 2002, False, 2002),
        (3602, 2000, False, MAMBA_BLOCK_SIZE),
        (3602, 3602, True, 3602),
        (2002, 2002, True, 2002),
        (3602, 512, True, 0),
        (3602, 1700, True, MAMBA_BLOCK_SIZE),
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


def test_partial_checkpoint_resume_stops_at_mamba_block_boundary() -> None:
    prompt_len = 3602
    resume_at = 1984
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    request.num_computed_tokens = resume_at

    # Without an internal checkpoint export, the forward must stop at the
    # physical Mamba boundary so it does not publish a stale running state.
    assert (
        _split(
            request,
            prompt_len - resume_at,
            use_eagle=False,
            partial_hit=True,
        )
        == MAMBA_BLOCK_SIZE - resume_at % MAMBA_BLOCK_SIZE
    )

    # A hash-aligned resume inside a physical Mamba block cannot use the fixed
    # internal-checkpoint column: it aliases the initial-state column. Enabling
    # checkpoints must therefore preserve the normal aligned split.
    assert (
        _split(
            request,
            prompt_len - resume_at,
            use_eagle=False,
            partial_hit=True,
            num_prefill_checkpoint_blocks=1,
        )
        == MAMBA_BLOCK_SIZE - resume_at % MAMBA_BLOCK_SIZE
    )


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
    # Under eagle the partial-tail stop sits one hash unit below the prompt's
    # last hash boundary: eagle matches a unit past its candidate and drops it,
    # so nothing proves that last boundary.
    tail_stop = prompt_len // ATTN_BLOCK_SIZE * ATTN_BLOCK_SIZE
    if use_eagle:
        tail_stop -= ATTN_BLOCK_SIZE

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
        assert aligned or (partial_hit and end == tail_stop), (
            f"intermediate chunk end {end} is neither block-aligned nor the "
            f"partial-tail stop ({tail_stop})"
        )


def _prefill_with_prefill_lookahead(
    prompt_len: int,
    lookahead: int,
    budget: int,
    use_eagle_block_drop: bool,
) -> tuple[list[int], int]:
    """Chunk a prefill the way `Scheduler.schedule()` does for multi-module MTP.

    Every step runs the real `_mamba_block_aligned_split` and then the real
    `_reserve_prefill_lookahead` on the step's budget, allocating through a real
    hybrid `KVCacheManager`. Returns the chunk ends and the number of cached
    Mamba slots, each checked to hold the state its hash claims.
    """
    manager = _make_hybrid_kv_cache_manager(use_eagle=use_eagle_block_drop)
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    sched = SimpleNamespace(
        num_prefill_lookahead=lookahead,
        need_mamba_block_aligned_split=True,
        cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE),
    )
    ends: list[int] = []
    state_at: dict[int, int] = {}
    # Bounded: with a constant budget a stalled prefill never recovers.
    for _ in range(2 * cdiv(prompt_len, min(budget, MAMBA_BLOCK_SIZE)) + 2):
        computed = request.num_computed_tokens
        if computed >= prompt_len:
            break
        num_new = _split(
            request,
            min(prompt_len - computed, budget),
            use_eagle_block_drop=use_eagle_block_drop,
            max_num_scheduled_tokens=budget,
            num_prefill_lookahead=lookahead,
        )
        num_new = Scheduler._reserve_prefill_lookahead(
            sched, request, computed, num_new
        )
        if num_new == 0:
            continue
        assert (
            manager.allocate_slots(request, num_new, num_lookahead_tokens=NUM_SPEC)
            is not None
        )
        request.num_computed_tokens = computed + num_new
        ends.append(request.num_computed_tokens)
        blocks = mamba_manager.req_to_blocks[request.request_id]
        running = cdiv(request.num_computed_tokens, MAMBA_BLOCK_SIZE) - 1
        state_at[blocks[running].block_id] = request.num_computed_tokens
    return ends, _count_cached_boundary_states(manager, request, state_at)


@pytest.mark.parametrize("use_eagle_block_drop", [False, True])
@pytest.mark.parametrize("budget_blocks", [4.0, 1.0, 0.5])
@pytest.mark.parametrize("num_full_blocks", [1, 2])
@pytest.mark.parametrize("tail", ["one", "window_edge", "past_window"])
@pytest.mark.parametrize("lookahead", [2, 3, 5])
@pytest.mark.parametrize("block_size", [64, 1600])
def test_mamba_split_respects_multi_module_mtp_prefill_lookahead(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
    lookahead: int,
    tail: str,
    num_full_blocks: int,
    budget_blocks: float,
    use_eagle_block_drop: bool,
) -> None:
    """#61762: the Mamba split and the multi-module MTP lookahead must agree.

    `_reserve_prefill_lookahead` shortened a chunk ending within
    `num_prefill_lookahead` of the prompt end. Applied to a Mamba-aligned end,
    that left the request just short of the boundary, and the next step (start
    mid-block, stop at the boundary) was shortened to zero tokens, forever.
    """
    monkeypatch.setattr(sys.modules[__name__], "MAMBA_BLOCK_SIZE", block_size)
    # Tokens past the last full block: inside the window, at its edge, past it.
    tail_len = {"one": 1, "window_edge": lookahead - 1, "past_window": lookahead}[tail]
    prompt_len = num_full_blocks * block_size + tail_len
    budget = int(block_size * budget_blocks)

    ends, _ = _prefill_with_prefill_lookahead(
        prompt_len, lookahead, budget, use_eagle_block_drop
    )

    assert ends and ends[-1] == prompt_len, f"prefill stalled at {ends}"
    if budget >= block_size:
        assert all(end % block_size == 0 for end in ends[:-1]), ends
    # A chunk ends inside the lookahead window only when no block boundary
    # before the window can be reached and the prompt cannot be finished.
    in_window = [end for end in ends[:-1] if prompt_len - lookahead < end]
    assert all(end % block_size == 0 for end in in_window), ends
    if budget >= prompt_len:
        assert not in_window, ends


def test_mamba_split_lookahead_window_is_noop_without_multi_module_mtp() -> None:
    """EAGLE-family drafters (lookahead 1) and no drafter keep today's split."""
    (request,) = create_requests(1, num_tokens=3201, block_size=ATTN_BLOCK_SIZE)
    for lookahead in (0, 1):
        assert (
            _split(
                request,
                request.num_tokens,
                use_eagle_block_drop=False,
                num_prefill_lookahead=lookahead,
            )
            == 2 * MAMBA_BLOCK_SIZE
        )
