# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse-retention grid checkpoints in align-mode Mamba allocation.

Regression: the align allocator recycles (relocates) the running state block
and back-fills the vacated slot with the null block, so the state blocks the
retention mask promises to keep never survived to be cached — retention
grids wider than the ~2-block relocation lag retained nothing (or only the
first checkpoint, erratically), and divergent-prefix resubmits re-prefilled
from scratch.
"""

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    retention_grid_block,
)
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.cpu_test

BLOCK = 1600
INTERVAL = 3 * BLOCK  # grid at blocks 2, 5, 8, ... (state at 4800, 9600, ...)
NUM_BLOCKS = 6


def _make_manager(retention_interval: int | None = INTERVAL) -> MambaManager:
    spec = MambaSpec(
        block_size=BLOCK,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=0,
    )
    block_pool = BlockPool(
        num_gpu_blocks=1000, enable_caching=True, hash_block_size=BLOCK
    )
    return MambaManager(
        spec,
        block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK,
        retention_interval=retention_interval,
    )


def _prefill(manager: MambaManager, req_id: str, num_blocks: int) -> None:
    """Drive block-aligned prefill chunks through the align allocator the way
    the scheduler does: growing num_tokens, one block per step."""
    manager.req_to_blocks[req_id]  # materialize the defaultdict entry
    for step in range(1, num_blocks + 1):
        num_tokens = step * BLOCK
        estimate = manager.get_num_blocks_to_allocate(
            req_id,
            num_tokens,
            new_computed_blocks=[],
            total_computed_tokens=(step - 1) * BLOCK,
            num_local_computed_tokens=(step - 1) * BLOCK,
            num_tokens_main_model=num_tokens,
        )
        new = manager.allocate_new_blocks(
            req_id, num_tokens, num_tokens_main_model=num_tokens
        )
        # Admission never under-estimates what allocation consumes.
        assert len(new) <= estimate
        manager.remove_skipped_blocks(req_id, num_tokens)


def test_grid_blocks_survive_relocation() -> None:
    manager = _make_manager()
    _prefill(manager, "req", NUM_BLOCKS)
    blocks = manager.req_to_blocks["req"]
    null = manager.block_pool.null_block
    for idx in range(NUM_BLOCKS - 2):  # interior positions (running tail exempt)
        on_grid = retention_grid_block(idx, INTERVAL, BLOCK)
        if on_grid:
            assert blocks[idx] != null, f"grid block {idx} was recycled"
        else:
            assert blocks[idx] == null, f"non-grid block {idx} unexpectedly kept"


def test_no_interval_recycles_everything() -> None:
    manager = _make_manager(retention_interval=None)
    _prefill(manager, "req", NUM_BLOCKS)
    blocks = manager.req_to_blocks["req"]
    null = manager.block_pool.null_block
    assert all(b == null for b in blocks[: NUM_BLOCKS - 2])


def test_allocator_and_mask_agree_on_the_grid() -> None:
    """The block the allocator spares must be exactly the block the retention
    mask registers — a drift between the two re-introduces the bug."""
    manager = _make_manager()
    spec = manager.kv_cache_spec
    mask = MambaManager.reachable_block_mask(
        start_block=0,
        end_block=NUM_BLOCKS,
        alignment_tokens=BLOCK,
        kv_cache_spec=spec,
        use_eagle=False,
        retention_interval=INTERVAL,
        reachable_boundaries=(),
    )
    grid = [retention_grid_block(i, INTERVAL, BLOCK) for i in range(NUM_BLOCKS)]
    assert mask == grid


def test_admission_accounts_for_grid_crossings() -> None:
    """A step whose token span crosses a grid boundary needs one extra block
    (the spared one is not recycled); the estimate must include it or pool
    pressure trips the allocator's post-condition."""
    manager = _make_manager()
    _prefill(manager, "req", 2)  # up to 3200 tokens, next step crosses 4800
    est_crossing = manager.get_num_blocks_to_allocate(
        "req",
        3 * BLOCK,
        new_computed_blocks=[],
        total_computed_tokens=2 * BLOCK,
        num_local_computed_tokens=2 * BLOCK,
        num_tokens_main_model=3 * BLOCK,
    )
    manager_plain = _make_manager(retention_interval=None)
    _prefill(manager_plain, "req", 2)
    est_plain = manager_plain.get_num_blocks_to_allocate(
        "req",
        3 * BLOCK,
        new_computed_blocks=[],
        total_computed_tokens=2 * BLOCK,
        num_local_computed_tokens=2 * BLOCK,
        num_tokens_main_model=3 * BLOCK,
    )
    assert est_crossing == est_plain + 1


def _make_request_for(manager: MambaManager, req_id: str, num_blocks: int):
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash

    from .test_prefix_caching import make_request

    del manager  # shape documented by BLOCK below
    init_none_hash(sha256)
    return make_request(
        request_id=req_id,
        prompt_token_ids=list(range(num_blocks * BLOCK)),
        block_size=BLOCK,
        hash_fn=sha256,
    )


def _prefill_with_caching(manager: MambaManager, request, num_blocks: int) -> None:
    """_prefill, plus the scheduler's per-step cache_blocks call so rolling
    decode-end checkpoints register as they freeze."""
    req_id = request.request_id
    manager.req_to_blocks[req_id]
    for step in range(1, num_blocks + 1):
        num_tokens = step * BLOCK
        estimate = manager.get_num_blocks_to_allocate(
            req_id,
            num_tokens,
            new_computed_blocks=[],
            total_computed_tokens=(step - 1) * BLOCK,
            num_local_computed_tokens=(step - 1) * BLOCK,
            num_tokens_main_model=num_tokens,
        )
        new = manager.allocate_new_blocks(
            req_id, num_tokens, num_tokens_main_model=num_tokens
        )
        assert len(new) <= estimate
        manager.cache_blocks(request, num_tokens, retention_interval=INTERVAL)
        manager.remove_skipped_blocks(req_id, num_tokens)


def test_rolling_checkpoint_registers_newest_boundary() -> None:
    """The newest frozen non-grid boundary must become a prefix-cache resume
    point (the decode-end checkpoint): an append-style resubmit then
    re-prefills at most one block instead of one retention interval."""
    manager = _make_manager()
    request = _make_request_for(manager, "req", NUM_BLOCKS)
    _prefill_with_caching(manager, request, NUM_BLOCKS)

    # Bookkeeping tracks the newest boundary (grid boundaries are recorded
    # as covered-by-the-mask; non-grid ones are registered ad hoc).
    assert manager._rolling_registered["req"] == NUM_BLOCKS - 1
    newest_non_grid = NUM_BLOCKS - 1
    while retention_grid_block(newest_non_grid, INTERVAL, BLOCK):
        newest_non_grid -= 1
    cached = manager.block_pool.get_cached_block(
        request.block_hashes[newest_non_grid], [0]
    )
    assert cached is not None, "rolling checkpoint hash not registered"


def test_rolling_checkpoint_survives_relocation_with_spec_window() -> None:
    """With a speculative window, the relocation loop must spare (not
    relocate) a hash-carrying block: relocation reuses the block object and
    would silently re-key newer state under the old hash."""
    spec = MambaSpec(
        block_size=BLOCK,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=2,
    )
    block_pool = BlockPool(
        num_gpu_blocks=1000, enable_caching=True, hash_block_size=BLOCK
    )
    manager = MambaManager(
        spec,
        block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK,
        retention_interval=INTERVAL,
    )
    request = _make_request_for(manager, "req", NUM_BLOCKS)
    _prefill_with_caching(manager, request, NUM_BLOCKS)

    blocks = manager.req_to_blocks["req"]
    null = manager.block_pool.null_block
    registered = manager._rolling_registered.get("req")
    assert registered is not None
    hashed_real = [
        i for i, b in enumerate(blocks) if b != null and b.block_hash is not None
    ]
    # Every hashed block still holds its own (unrelocated) state slot: the
    # rolling checkpoint and any grid checkpoints the run produced.
    assert registered in hashed_real or manager.block_pool.get_cached_block(
        request.block_hashes[registered], [0]
    ), "rolling checkpoint neither in-table nor on the cached free list"
