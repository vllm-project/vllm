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

from tests.v1.core.test_prefix_caching import make_request
from vllm.multimodal.inputs import PlaceholderRange
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    is_mamba_prefill_checkpoint_valid,
)
from vllm.v1.request import Request

from .utils import create_requests, create_scheduler, new_kv_cache_group

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
        (448, 665, 512, 256, 256, 64, True),
        (15360, 16896, 16768, 128, 1536, 16, False),
        (15360, 16960, 16896, 128, 1536, 16, True),
        (15360, 16960, 16768, 128, 1536, 16, True),
        (16064, 18433, 18432, 128, 1536, 16, True),
        (8, 100, 88, 16, 64, 16, False),
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
    num_hash_lookahead_tokens: int = 0,
) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            new_kv_cache_group(
                ["full_layer"],
                AttentionSpec(
                    block_size=ATTN_BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            new_kv_cache_group(
                ["mamba_layer"],
                MambaManager.layer_config(
                    MambaSpec(
                        block_size=MAMBA_BLOCK_SIZE,
                        shapes=((1, 1),),
                        dtypes=(torch.float32,),
                    ),
                    enable_prefix_caching=True,
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
        num_hash_lookahead_tokens=num_hash_lookahead_tokens,
    )


def _split(
    request: Request,
    num_new_tokens: int,
    partial_hit: bool = False,
    num_prefill_checkpoint_blocks: int = 0,
    max_num_scheduled_tokens: int = 16384,
    num_hash_lookahead_tokens: int = 0,
) -> int:
    """Exercise the chunk sizing policy for this backend."""
    stub = SimpleNamespace(
        block_size=MAMBA_BLOCK_SIZE,
        mamba_block_size=MAMBA_BLOCK_SIZE,
        num_hash_lookahead_tokens=num_hash_lookahead_tokens,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        scheduler_config=SimpleNamespace(
            long_prefill_token_threshold=0,
            disable_chunked_mm_input=False,
            max_num_batched_tokens=max_num_scheduled_tokens,
        ),
        num_lookahead_tokens=0,
        # `prefix_match_unit` finer than the block size (#46384).
        mamba_partial_cache_hit=partial_hit,
        mamba_shared_prefix_checkpoint=False,
        mamba_prefill_checkpoint_alignment=(
            16 if num_prefill_checkpoint_blocks else None
        ),
        hash_block_size=ATTN_BLOCK_SIZE,
        num_prefill_lookahead=0,
    )
    if not num_prefill_checkpoint_blocks:
        num_new_tokens = Scheduler._mamba_block_aligned_split(
            stub, request, num_new_tokens
        )
    return Scheduler._mamba_checkpoint_aligned_split(
        stub, request, request.num_computed_tokens, num_new_tokens
    )


@pytest.mark.parametrize(
    ("prompt_len", "num_new_tokens", "lookahead", "expected"),
    [
        (2002, 2002, 0, 2002),
        (3602, 2000, 0, 2000),
        (3602, 3602, 1, 3602),
        (3602, 512, 1, 512),
        (3602, 1700, 1, 1696),
    ],
)
def test_internal_checkpoint_split(
    prompt_len: int, num_new_tokens: int, lookahead: int, expected: int
) -> None:
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    assert (
        _split(
            request,
            num_new_tokens,
            num_hash_lookahead_tokens=lookahead,
            num_prefill_checkpoint_blocks=1,
        )
        == expected
    )


@pytest.mark.parametrize(
    "budget, other_tokens, prompt_len, threshold, lookahead, mm, steps",
    [
        (256, 226, 600, 0, 0, None, (0, 256)),  # Retry with the full step budget.
        (48, 18, 42, 0, 0, None, (30, 12)),  # Use the short budget near the tail.
        (256, 226, 42, 0, 0, None, (30, 12)),  # Large budget, short prompt.
        (64, 34, 160, 0, 0, None, (30, 34)),  # Budget is exactly one alignment unit.
        (32, 0, 160, 0, 0, None, (32, 32)),  # A full step cannot reach 64.
        (256, 1, 160, 32, 0, None, (32, 32)),  # Per-request chunk limit.
        (64, 0, 66, 0, 4, None, (62, 4)),  # MTP needs four trailing tokens.
        # MM requests advance even when the first image is in a later chunk.
        (256, 226, 600, 0, 0, (False, [(400, 16)]), (30, 226)),
        (256, 226, 600, 0, 0, (True, [(400, 16)]), (30, 226)),
        (576, 0, 880, 0, 0, (True, [(20, 576)]), (20, 576)),  # Indivisible image.
        # Both images fit separately. Deferring at 30 would retry the same pair.
        (576, 0, 880, 0, 0, (False, [(0, 20), (30, 576)]), (30, 546)),
    ],
)
def test_checkpoint_chunk_defers_only_with_alignment_headroom(
    budget, other_tokens, prompt_len, threshold, lookahead, mm, steps
):
    """Deferral requires a text-only request with room in the budget and the prompt."""
    disable_chunked_mm_input, images = mm or (False, [])
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf" if images else "facebook/opt-125m",
        max_num_batched_tokens=max(576 if images else 64, budget),
        max_model_len=1024,
        long_prefill_token_threshold=threshold,
        block_size=256,
        enable_prefix_caching=True,
        skip_tokenizer_init=not images,
        disable_chunked_mm_input=disable_chunked_mm_input,
        mamba_layer_config=MambaManager.layer_config(
            MambaSpec(block_size=256, shapes=((1, 1),), dtypes=(torch.float32,)),
            enable_prefix_caching=True,
            num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=64,
        ),
    )
    scheduler.max_num_scheduled_tokens = budget
    scheduler.num_prefill_lookahead = lookahead
    if other_tokens:
        (other,) = create_requests(
            1, num_tokens=other_tokens, req_ids=["other"], block_size=256
        )
        scheduler.add_request(other)
    (request,) = create_requests(
        1,
        num_tokens=prompt_len,
        block_size=256,
        mm_positions=[
            [PlaceholderRange(offset=start, length=length) for start, length in images]
        ],
    )
    scheduler.add_request(request)
    for expected in steps:
        output = scheduler.schedule()
        assert output.num_scheduled_tokens.get(request.request_id, 0) == expected


def test_scheduler_sends_only_committed_checkpoint_positions():
    """Only requests scheduled in this step contribute checkpoint positions."""
    scheduler = create_scheduler(
        max_num_batched_tokens=670,
        max_model_len=2048,
        block_size=256,
        enable_prefix_caching=True,
        skip_tokenizer_init=True,
        mamba_layer_config=MambaManager.layer_config(
            MambaSpec(block_size=256, shapes=((1, 1),), dtypes=(torch.float32,)),
            enable_prefix_caching=True,
            num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=64,
        ),
    )
    (request,) = create_requests(1, num_tokens=1160, block_size=256)
    scheduler.add_request(request)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {request.request_id: 640}
    assert first.mamba_ckpt_positions == {request.request_id: 512}
    assert first.get_mamba_ckpt_positions(["decode", request.request_id]) == [0, 512]
    request.shared_prefix_boundary = 768
    last = scheduler.schedule()
    assert last.num_scheduled_tokens == {request.request_id: 520}
    assert last.mamba_ckpt_positions == {request.request_id: 1024}

    # The first request still holds its checkpoint while only the short one runs.
    (short,) = create_requests(1, num_tokens=16, req_ids=["short"], block_size=256)
    scheduler.add_request(short)
    following = scheduler.schedule()
    assert following.num_scheduled_tokens == {"short": 16}
    assert following.mamba_ckpt_positions == {}

    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert not scheduler.kv_cache_manager.get_mamba_ckpt_positions([request.request_id])
    assert first.mamba_ckpt_positions == {request.request_id: 512}


@pytest.mark.parametrize(
    ("disable_chunked_mm_input", "image_start", "first_end", "second_end"),
    [(True, 580, 690, 1344), (False, 580, 640, 1280), (False, 646, 640, 1280)],
)
def test_checkpoint_alignment_respects_mm_chunking(
    disable_chunked_mm_input, image_start, first_end, second_end
):
    """Align splittable images and reuse any image encoded ahead of the decoder."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=690,
        max_model_len=2048,
        block_size=128,
        disable_chunked_mm_input=disable_chunked_mm_input,
        enable_prefix_caching=True,
        mamba_layer_config=MambaManager.layer_config(
            MambaSpec(block_size=128, shapes=((1, 1),), dtypes=(torch.float32,)),
            enable_prefix_caching=True,
            num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=64,
        ),
    )
    (request,) = create_requests(
        1,
        num_tokens=1590,
        block_size=128,
        mm_positions=[[PlaceholderRange(offset=image_start, length=690 - image_start)]],
    )
    scheduler.add_request(request)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {request.request_id: first_end}
    assert first.scheduled_encoder_inputs == {request.request_id: [0]}
    second = scheduler.schedule()
    assert not second.scheduled_encoder_inputs
    assert second.num_scheduled_tokens == {request.request_id: second_end - first_end}
    assert second.mamba_ckpt_positions == (
        {request.request_id: 1152} if first_end == 640 else {}
    )
    last = scheduler.schedule()
    assert last.num_scheduled_tokens == {request.request_id: 1590 - second_end}
    assert last.mamba_ckpt_positions == {request.request_id: 1536}


def test_block_aligned_split_requires_chunked_mm_input():
    with pytest.raises(AssertionError, match="require chunked MM input"):
        create_scheduler(
            model="llava-hf/llava-1.5-7b-hf",
            disable_chunked_mm_input=True,
            enable_prefix_caching=True,
            mamba_layer_config=MambaManager.layer_config(
                MambaSpec(block_size=16, shapes=((1, 1),), dtypes=(torch.float32,)),
                enable_prefix_caching=True,
            ),
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
            partial_hit=True,
        )
        == MAMBA_BLOCK_SIZE - resume_at % MAMBA_BLOCK_SIZE
    )

    # With checkpoint export, the checkpoint goes in the resumed column, whose
    # state is read before the forward, so the rest of the prompt runs at once.
    assert (
        _split(
            request,
            prompt_len - resume_at,
            partial_hit=True,
            num_prefill_checkpoint_blocks=1,
        )
        == prompt_len - resume_at
    )


def test_lookahead_keeps_the_trailing_cache_boundary() -> None:
    (request,) = create_requests(1, num_tokens=3602, block_size=ATTN_BLOCK_SIZE)
    assert (
        _split(request, request.num_tokens, num_hash_lookahead_tokens=1)
        == 2 * MAMBA_BLOCK_SIZE
    )


def _run_chunked_prefill(
    manager: KVCacheManager, request: Request, budgets: list[int], **split_kwargs
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
        num_new = _split(
            request, min(request.num_tokens - computed, budget), **split_kwargs
        )
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


def test_fragmented_tail_chunk_does_not_poison_mamba_prefix_cache() -> None:
    """A sub-block fragment after a cached block boundary must not leave a
    mid-block state (slot 1 at state@2600) that a later chunk publishes as
    state@3200."""
    assert _prefill(3602, budgets=[1600, 1000, 3602]) > 0


@pytest.mark.parametrize("lookahead", [1, 2])
def test_lookahead_split_caches_partial_tail_boundary(lookahead: int) -> None:
    """Chunks stop where lookahead hashes put the prompt's partial tail, so the
    manager caches that boundary and a resend hits it."""
    init_none_hash(sha256)
    prompt_len = 3601
    boundary = (prompt_len - lookahead) // ATTN_BLOCK_SIZE * ATTN_BLOCK_SIZE
    manager = _make_hybrid_kv_cache_manager(num_hash_lookahead_tokens=lookahead)

    def make(request_id: str) -> Request:
        return make_request(
            request_id,
            list(range(prompt_len)),
            ATTN_BLOCK_SIZE,
            sha256,
            num_hash_lookahead_tokens=lookahead,
        )

    producer = make("producer")
    state_at = _run_chunked_prefill(
        manager,
        producer,
        budgets=[],
        partial_hit=True,
        num_hash_lookahead_tokens=lookahead,
    )
    assert producer.num_computed_tokens == prompt_len
    _count_cached_boundary_states(manager, producer, state_at)
    manager.free(producer)
    manager.new_step_starts()

    _, num_computed, _ = manager.get_computed_blocks(make("resend"))
    assert num_computed == boundary


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
    ("num_prefill_checkpoint_blocks", "lookahead"),
    [(0, 1), (1, 0)],
)
def test_unaligned_resume_final_chunk_with_and_without_checkpoint_support(
    partial_hit: bool,
    resume_at: int,
    num_prefill_checkpoint_blocks: int,
    lookahead: int,
) -> None:
    """Backends without checkpoint export stop at a block boundary on resume."""
    prompt_len = 3602
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    # The trailing hash boundary has finalized lookahead in this prompt.
    tail_stop = prompt_len // ATTN_BLOCK_SIZE * ATTN_BLOCK_SIZE

    pos, ends = resume_at, []
    while pos < prompt_len:
        request.num_computed_tokens = pos
        num_new = _split(
            request,
            prompt_len - pos,
            num_hash_lookahead_tokens=lookahead,
            partial_hit=partial_hit,
            num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
        )
        assert num_new > 0, f"no progress at {pos}"
        if num_prefill_checkpoint_blocks:
            assert pos + num_new == prompt_len
        elif pos % MAMBA_BLOCK_SIZE != 0:
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


@pytest.mark.parametrize("resume_at", [400, 1200, 1216])
@pytest.mark.parametrize("checkpoint_blocks", [0, 1])
def test_final_chunk_with_block_floored_junction(
    resume_at: int, checkpoint_blocks: int
) -> None:
    """A final chunk with export support runs through the junction."""
    prompt_len = 3602
    junction = MAMBA_BLOCK_SIZE + 1200
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    request.num_computed_tokens = resume_at
    request.shared_prefix_boundary = junction
    num_new = _split(
        request,
        prompt_len - resume_at,
        num_hash_lookahead_tokens=1,
        num_prefill_checkpoint_blocks=checkpoint_blocks,
    )
    expected = (
        prompt_len
        if checkpoint_blocks
        else junction // MAMBA_BLOCK_SIZE * MAMBA_BLOCK_SIZE
    )
    assert resume_at + num_new == expected
