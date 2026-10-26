# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefix caching without a Mamba-specific chunk split.

Prefill chunks end wherever the token budget runs out. The worker writes the
running state into slot `(seq_len - 1) // block_size` at every chunk end, so a
chunk ending mid-block leaves that slot at the wrong offset for its position.
Only internal checkpoints and running slots that stopped exactly on their block
boundary may be published; a request resuming from anything else restores a
truncated state (#43559).
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.checkpoint import (
    compute_mamba_prefill_checkpoints,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
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
    block_size: int,
    retention_interval: int | None,
    use_eagle: bool,
    hash_block_size: int = ATTN_BLOCK_SIZE,
    num_prefill_lookahead: int = 0,
) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        prefix_cache_retention_interval=retention_interval,
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                    num_prefill_checkpoint_blocks=1,
                    prefill_checkpoint_alignment=16,
                ),
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=262144,
        scheduler_block_size=block_size,
        hash_block_size=hash_block_size,
        enable_caching=True,
        use_eagle=use_eagle,
        num_prefill_lookahead=num_prefill_lookahead,
    )


def _prefill(
    manager: KVCacheManager, request: Request, budgets: list[int]
) -> dict[int, int]:
    """Prefill one chunk per budget, with no Mamba-specific chunk split.

    Returns physical mamba block id -> the token offset of the state it holds,
    mirroring the worker: each chunk exports its internal checkpoint into the
    column `compute_mamba_prefill_checkpoints` picks, and the running slot ends
    up at the chunk end.
    """
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    block_size = mamba_manager.block_size
    state_at: dict[int, int] = {}
    for budget in budgets:
        start = request.num_computed_tokens
        num_new = min(request.num_tokens - start, budget)
        assert (
            manager.allocate_slots(request, num_new, num_lookahead_tokens=NUM_SPEC)
            is not None
        )
        end = start + num_new
        blocks = mamba_manager.req_to_blocks[request.request_id]
        (offset,), (col,) = compute_mamba_prefill_checkpoints(
            [end],
            [num_new],
            hash_block_size=mamba_manager.block_pool.hash_block_size,
            mamba_block_size=block_size,
            checkpoint_alignment=16,
            drop_eagle_block=mamba_manager.drop_eagle_checkpoint_block,
            cache_hit_alignment_tokens=mamba_manager.cache_hit_alignment_tokens,
        )
        if col >= 0 and not blocks[col].is_null:
            state_at[blocks[col].block_id] = start + offset
        request.num_computed_tokens = end
        state_at[blocks[cdiv(end, block_size) - 1].block_id] = end
    assert request.num_computed_tokens == request.num_tokens
    return state_at


def _assert_cached_states_exact(
    manager: KVCacheManager, request: Request, state_at: dict[int, int]
) -> None:
    """Every hash-cached mamba slot holds the state its hash claims."""
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    for pos, block in enumerate(mamba_manager.req_to_blocks[request.request_id]):
        if block.is_null or block.block_hash is None:
            continue
        claimed = block.block_hash_num_tokens
        assert state_at.get(block.block_id) == claimed, (
            f"mamba slot {pos} is hashed as state@{claimed} but holds "
            f"state@{state_at.get(block.block_id)}"
        )


@pytest.mark.parametrize("retention_interval", [None, 0])
@pytest.mark.parametrize("use_eagle", [True, False])
@pytest.mark.parametrize(
    ("block_size", "prompt_len", "budgets"),
    [
        (MAMBA_BLOCK_SIZE, 2002, [364, 2002]),
        (MAMBA_BLOCK_SIZE, 3602, [1000, 3602]),
        (MAMBA_BLOCK_SIZE, 3602, [1600, 1000, 3602]),
        (MAMBA_BLOCK_SIZE, 3602, [2000, 3602]),
        (MAMBA_BLOCK_SIZE, 5000, [900, 1700, 5000]),
        # Kimi-K3-scale mamba blocks: TP8 shards the recurrent state 8 ways
        # (~1.5k block), DEP16 keeps it whole (~12k block).
        (1536, 30000, [27648, 1024, 30000]),
        (12288, 30000, [12288, 4000, 30000]),
        (12288, 41000, [24576, 8000, 41000]),
    ],
)
def test_unsplit_prefill_publishes_only_exact_states(
    retention_interval: int | None,
    use_eagle: bool,
    block_size: int,
    prompt_len: int,
    budgets: list[int],
) -> None:
    manager = _make_hybrid_kv_cache_manager(block_size, retention_interval, use_eagle)
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    state_at = _prefill(manager, request, budgets)
    _assert_cached_states_exact(manager, request, state_at)


@pytest.mark.parametrize(
    ("hash_block_size", "num_prefill_lookahead", "use_eagle"),
    [
        # The hash block equals the Mamba block, so partial hash hits are off
        # and the cacheable count rounds down to a block boundary.
        (64, 0, False),
        # Multi-module MTP holds back its re-prefillable tail.
        (16, 3, True),
    ],
)
def test_cacheable_count_short_of_chunk_end_publishes_no_stale_state(
    hash_block_size: int, num_prefill_lookahead: int, use_eagle: bool
) -> None:
    """The cacheable token count can stop on a boundary the chunk ran past."""
    manager = _make_hybrid_kv_cache_manager(
        64,
        retention_interval=None,
        use_eagle=use_eagle,
        hash_block_size=hash_block_size,
        num_prefill_lookahead=num_prefill_lookahead,
    )
    (request,) = create_requests(1, num_tokens=140, block_size=hash_block_size)
    state_at = _prefill(manager, request, [100, 40])
    _assert_cached_states_exact(manager, request, state_at)
