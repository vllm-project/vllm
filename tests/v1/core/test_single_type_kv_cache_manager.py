# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    CircularBufferManager,
    FullAttentionManager,
    MambaManager,
    RSWAManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    RSWASpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def test_external_computed_blocks_do_not_corrupt_free_pool():
    block_size = 4
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    block_pool = BlockPool(
        num_gpu_blocks=10,
        enable_caching=False,
        hash_block_size=block_size,
    )
    manager = FullAttentionManager(
        spec,
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    request_id = "request"
    manager.allocate_new_blocks(
        request_id,
        num_tokens=3 * block_size,
        num_tokens_main_model=3 * block_size,
    )
    num_free_blocks = block_pool.get_num_free_blocks()

    # Speculative allocations can exceed the blocks implied by the eventual
    # external computed-token count. This must not request a negative number
    # of blocks, which inflates the free-queue counter.
    manager.allocate_external_computed_blocks(
        request_id,
        num_local_computed_tokens=0,
        num_external_computed_tokens=block_size,
    )

    assert block_pool.get_num_free_blocks() == num_free_blocks
    assert len(manager.req_to_blocks[request_id]) == 3


def test_mamba_speculative_block_relocation_requires_exclusive_ownership():
    spec = MambaSpec(
        block_size=4,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=1,
    )
    block_pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=4)
    manager = MambaManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=4,
    )
    block = block_pool.get_new_blocks(1)[0]
    blocks = [block]

    manager._relocate_speculative_block(blocks, 0)

    assert blocks == [block_pool.null_block, block]
    assert block.ref_cnt == 1

    pinned_block = block_pool.get_new_blocks(1)[0]
    block_pool.touch((pinned_block,))
    with pytest.raises(AssertionError, match="exclusively owned and unhashed"):
        manager._relocate_speculative_block([pinned_block], 0)


def _align_manager(
    num_spec_blocks: int = 2,
    num_gpu_blocks: int = 16,
    mode: str = "align",
) -> tuple[MambaManager, BlockPool]:
    spec = MambaSpec(
        block_size=4,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode=mode,
        num_speculative_blocks=num_spec_blocks,
    )
    pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks, enable_caching=False, hash_block_size=4
    )
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=4,
    )
    return manager, pool


def test_mamba_tier0_admission_reserves_window_as_nulls():
    """A request priced down to the K=0 tier at admission gets its
    speculative window as null row slots (no pool blocks): the row keeps
    the standard layout — skip nulls, committed, window — so the state
    indexing formula holds from birth and the promotion refills the
    window."""
    manager, pool = _align_manager()
    assert manager.mark_spec_admission_tier0("r") is True
    assert manager.is_spec_released("r")
    free_before = pool.get_num_free_blocks()

    blocks = manager.allocate_new_blocks("r", num_tokens=8, num_tokens_main_model=8)
    row = manager.req_to_blocks["r"]
    assert row == blocks
    assert row[0] is pool.null_block
    assert row[1].ref_cnt == 1
    assert row[2:] == [pool.null_block, pool.null_block]
    assert pool.get_num_free_blocks() == free_before - 1
    assert "r" in manager._allocated_block_reqs


def test_mamba_priced_admission_mark_guards():
    """Only a fresh waiting request under a spec-decode align pool can be
    priced down: the mark refuses pools without a speculative window,
    non-align modes, and requests that already hold blocks."""
    no_window, _ = _align_manager(num_spec_blocks=0)
    assert no_window.mark_spec_admission_tier0("r") is False

    non_align, _ = _align_manager(mode="none")
    assert non_align.mark_spec_admission_tier0("r") is False

    manager, pool = _align_manager()
    assert manager.mark_spec_admission_tier0("r") is True
    free_before = pool.get_num_free_blocks()
    manager.allocate_new_blocks("r", num_tokens=8, num_tokens_main_model=8)
    assert manager.spec_admission_priceable("r") is False
    assert manager.is_spec_released("r")
    assert pool.get_num_free_blocks() == free_before - 1


def test_mamba_pinned_row_growth_fills_window_nulls():
    """Decoding advances the state boundary into the pinned window: the
    null positions must be refilled in place (the state chain needs real
    blocks there) while the row keeps growing one slot per boundary
    crossing."""
    manager, pool = _align_manager()
    assert manager.mark_spec_admission_tier0("r") is True
    manager.allocate_new_blocks("r", num_tokens=8, num_tokens_main_model=8)
    free_after_admission = pool.get_num_free_blocks()

    manager.allocate_new_blocks("r", num_tokens=12, num_tokens_main_model=12)
    row = manager.req_to_blocks["r"]
    assert len(row) == 5
    assert row[2].ref_cnt == 1 and row[3].ref_cnt == 1 and row[4].ref_cnt == 1
    assert manager.take_spec_row_rebuilds() == {"r"}
    assert pool.get_num_free_blocks() == free_after_admission - 3

    manager.allocate_new_blocks("r", num_tokens=16, num_tokens_main_model=16)
    row = manager.req_to_blocks["r"]
    assert len(row) == 6
    assert all(block.ref_cnt == 1 for block in row[1:])
    assert manager.take_spec_row_rebuilds() == set()
    assert pool.get_num_free_blocks() == free_after_admission - 4


def test_mamba_pinned_row_growth_survives_multi_block_advance():
    """A chunked prefill can advance the state boundary by several blocks
    in ONE step (a long prompt across max-batched-token chunks). A
    normal row absorbs the advance free via window relocation; a pinned
    row has no scratch to relocate, so its row-length growth is explicit
    — the per-step bound must budget the same window's worth or the row
    accounting assert fires."""
    manager, pool = _align_manager()
    assert manager.mark_spec_admission_tier0("r") is True
    manager.allocate_new_blocks("r", num_tokens=8, num_tokens_main_model=8)
    row = manager.req_to_blocks["r"]
    assert len(row) == 4
    free_after_admission = pool.get_num_free_blocks()

    # One step advances the boundary from cdiv(8,4)=2 to cdiv(20,4)=5
    # blocks: num_new=3 against a chain bound of 1 — legal only because
    # the pinned row budgets the window's worth.
    manager.allocate_new_blocks("r", num_tokens=20, num_tokens_main_model=20)
    row = manager.req_to_blocks["r"]
    # The advance jumped past the swallowed window nulls (live_start
    # lands beyond the row): they stay null as unexported checkpoint
    # slots, the running-state slot and the new tail append physical —
    # pure append, no in-place refill, no runner rebuild.
    assert [b.is_null for b in row] == [True, False, True, True, False, False, False]
    assert manager.take_spec_row_rebuilds() == set()
    assert pool.get_num_free_blocks() == free_after_admission - 3

    # The next single-block advance keeps the row accounting: the chain
    # grows one slot and the swallowed window nulls stay unexported.
    manager.allocate_new_blocks("r", num_tokens=24, num_tokens_main_model=24)
    row = manager.req_to_blocks["r"]
    assert len(row) == 8
    assert [b.is_null for b in row] == [
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert manager.take_spec_row_rebuilds() == set()
    assert pool.get_num_free_blocks() == free_after_admission - 4


def test_mamba_promotion_reallocates_fresh_blocks():
    """The priced promotion refills the pinned window's null positions
    with fresh hedge blocks: they are empty allocations, not computed
    data, so no replay is needed and the allocate path counts the row as
    a normal full-window row afterwards."""
    manager, pool = _align_manager()
    assert manager.mark_spec_admission_tier0("r") is True
    manager.allocate_new_blocks("r", num_tokens=8, num_tokens_main_model=8)
    row = manager.req_to_blocks["r"]
    assert row[2:] == [pool.null_block, pool.null_block]
    free_pinned = pool.get_num_free_blocks()

    assert manager.promote_spec_blocks("r") is True
    fresh = row[2:]
    assert len(fresh) == 2
    assert all(block.ref_cnt == 1 for block in fresh)
    assert pool.get_num_free_blocks() == free_pinned - 2
    assert not manager.is_spec_released("r")
    assert manager.take_spec_row_rebuilds() == {"r"}
    assert manager.promote_spec_blocks("r") is False

    assert (
        manager.get_num_blocks_to_allocate(
            request_id="r",
            num_tokens=8,
            new_computed_blocks=[],
            total_computed_tokens=8,
            num_local_computed_tokens=8,
            num_tokens_main_model=8,
        )
        == 0
    )


def test_mamba_promotion_prices_against_free_blocks():
    """The promotion is priced: it refills only when the pool fits the
    window (plus the reserve); a refused promotion leaves the row and
    the pin untouched."""
    manager, pool = _align_manager(num_gpu_blocks=8)
    assert manager.mark_spec_admission_tier0("r") is True
    manager.allocate_new_blocks("r", num_tokens=8, num_tokens_main_model=8)
    free_pinned = pool.get_num_free_blocks()
    row = list(manager.req_to_blocks["r"])

    assert manager.promote_spec_blocks("r", reserve=free_pinned - 1) is False
    assert manager.req_to_blocks["r"] == row
    assert manager.is_spec_released("r")
    assert pool.get_num_free_blocks() == free_pinned

    assert manager.promote_spec_blocks("r", reserve=free_pinned - 2) is True
    assert all(block.ref_cnt == 1 for block in manager.req_to_blocks["r"][2:])
    assert not manager.is_spec_released("r")


def test_mamba_promotion_never_replaces_committed_tail():
    """A row whose window positions are committed blocks (the boundary
    consumed them) has nothing to refill: the promotion must touch
    nothing and just clear the stale pin — a blind tail replacement
    would destroy committed state and leave the request unmarked with
    no hedge."""
    manager, pool = _align_manager()
    committed = pool.get_new_blocks(3)
    manager.req_to_blocks["r"] = list(committed)
    manager._allocated_block_reqs.add("r")
    manager._spec_released.add("r")
    free_before = pool.get_num_free_blocks()

    assert manager.promote_spec_blocks("r") is False
    assert manager.req_to_blocks["r"] == committed
    assert pool.get_num_free_blocks() == free_before
    assert not manager.is_spec_released("r")


def test_mamba_promotion_keeps_pin_when_row_missing():
    """The admission mark pins row-less requests at the K=0 tier. The
    promotion must not clear that pin before hedge blocks are actually
    placed — an unpinned hedge-less row walks the relocation loop off
    the end of the row."""
    manager, _ = _align_manager()
    assert manager.mark_spec_admission_tier0("r") is True
    assert manager.is_spec_released("r")

    assert manager.promote_spec_blocks("r") is False
    assert manager.is_spec_released("r")


def test_mamba_pin_cleared_on_free():
    """The pin dies with the row: a freed request must not leave its
    K=0 mark behind, or a later admission would silently inherit the
    tier-0 window substitution."""
    manager, _ = _align_manager()
    assert manager.mark_spec_admission_tier0("r") is True
    manager.allocate_new_blocks("r", num_tokens=8, num_tokens_main_model=8)
    assert manager.is_spec_released("r")

    manager.free("r")
    assert not manager.is_spec_released("r")


def test_mamba_k0_crossing_release_swaps_window_for_nulls():
    """At the K=0 crossing a running row's real window is an idle hedge:
    the release frees it into null positions (the born-null row shape),
    joins the pin set, and the priced promotion refills fresh blocks
    without replay."""
    manager, pool = _align_manager()
    manager.allocate_new_blocks("r", num_tokens=8, num_tokens_main_model=8)
    assert manager.req_to_blocks["r"][-2:] != [pool.null_block, pool.null_block]
    free_before = pool.get_num_free_blocks()

    assert manager.release_spec_blocks("r") is True
    assert manager.req_to_blocks["r"][-2:] == [pool.null_block, pool.null_block]
    assert pool.get_num_free_blocks() == free_before + 2
    assert manager.is_spec_released("r")
    assert manager.take_spec_row_rebuilds() == {"r"}
    assert manager.release_spec_blocks("r") is False

    assert manager.promote_spec_blocks("r") is True
    assert all(block.ref_cnt == 1 for block in manager.req_to_blocks["r"][-2:])
    assert not manager.is_spec_released("r")


def test_mamba_retirement_crosses_null_gaps():
    spec = MambaSpec(
        block_size=4,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    pool = BlockPool(num_gpu_blocks=8, enable_caching=False, hash_block_size=4)
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=4,
    )
    old, committed, in_flight = pool.get_new_blocks(3)
    manager.req_to_blocks["r"] = [old, pool.null_block, committed, in_flight]

    # The state at token 12 and the following in-flight state must survive.
    for _ in range(2):
        manager.remove_skipped_blocks("r", processed_computed_tokens=12)
        assert manager.req_to_blocks["r"] == [
            pool.null_block,
            pool.null_block,
            committed,
            in_flight,
        ]
        assert old.ref_cnt == 0
        assert committed.ref_cnt == in_flight.ref_cnt == 1
        assert pool.get_num_free_blocks() == 5

    manager.free("r")
    manager.req_to_blocks["r"] = pool.get_new_blocks(2)
    manager.remove_skipped_blocks("r", processed_computed_tokens=5)
    assert manager.req_to_blocks["r"][0].is_null
    assert manager.req_to_blocks["r"][1].ref_cnt == 1


@pytest.mark.parametrize("block_size", [3584, 4608])
@pytest.mark.parametrize("in_flight_chunks", [0, 1, 2])
def test_mamba_retirement_bounds_prefill_states(block_size, in_flight_chunks):
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=5,
    )
    pool = BlockPool(num_gpu_blocks=1000, enable_caching=True, hash_block_size=256)
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    initial_free = pool.get_num_free_blocks()
    chunk = 8192 // block_size * block_size
    peak_held = 0
    # Replay full prefill chunks with an async committed-token lag.
    for target in range(chunk, 480031 + 1, chunk):
        computed = target - chunk
        committed = max(0, computed - in_flight_chunks * chunk)
        manager.remove_skipped_blocks("r", committed)
        manager.get_num_blocks_to_allocate(
            "r", target + 5, [], computed, computed, target
        )
        manager.allocate_new_blocks("r", target + 5, target)
        peak_held = max(peak_held, initial_free - pool.get_num_free_blocks())

    # Five speculative blocks, current/previous states, and bounded in-flight states.
    assert peak_held == 7 + in_flight_chunks
    manager.free("r")
    assert pool.get_num_free_blocks() == initial_free


@pytest.mark.parametrize("block_size", [896, 1536])
@pytest.mark.parametrize("num_speculative_blocks", [0, 1, 4])
@pytest.mark.parametrize("prompt_tokens", [25121, 704547])
def test_mamba_checkpoint_admission_matches_allocation(
    block_size, num_speculative_blocks, prompt_tokens
):
    """Checkpoint admission must match the subsequent physical allocation."""
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=64,
    )
    pool = BlockPool(
        num_gpu_blocks=2048,
        enable_caching=True,
        hash_block_size=128,
    )
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    request_id = "prefill"
    computed_tokens = 23040

    def estimate(num_tokens, total_computed_tokens, apply_admission_cap):
        return manager.get_num_blocks_to_allocate(
            request_id=request_id,
            num_tokens=num_tokens,
            new_computed_blocks=[],
            total_computed_tokens=total_computed_tokens,
            num_local_computed_tokens=total_computed_tokens,
            num_tokens_main_model=num_tokens,
            apply_admission_cap=apply_admission_cap,
        )

    estimate(computed_tokens, 0, False)
    manager.allocate_new_blocks(request_id, computed_tokens, computed_tokens)
    assert request_id in manager._allocated_block_reqs

    admission_estimate = estimate(prompt_tokens, computed_tokens, True)
    allocation_estimate = estimate(prompt_tokens, computed_tokens, False)
    assert request_id in manager._checkpoints

    free_before = pool.get_num_free_blocks()
    manager.allocate_new_blocks(request_id, prompt_tokens, prompt_tokens)
    allocated = free_before - pool.get_num_free_blocks()

    assert admission_estimate == allocation_estimate == allocated


@pytest.mark.parametrize("num_speculative_blocks", [1, 2, 3])
@pytest.mark.parametrize("first_chunk_blocks", [2, 4])
@pytest.mark.parametrize("retention_interval", [0, None])
def test_mamba_checkpoint_hash_maps_only_to_checkpoint_block(
    num_speculative_blocks, first_chunk_blocks, retention_interval
):
    """A checkpoint step must not leave a never-written speculative block
    cached."""
    block_size = 128
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=16,
    )
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=128)
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    # MTP/EAGLE moves the checkpoint one block below the last full block.
    manager.drop_eagle_checkpoint_block = True
    init_none_hash(sha256)
    prompt_tokens = 6 * block_size + 50
    request = Request(
        request_id="r",
        prompt_token_ids=list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )

    def step(start, end):
        estimate = manager.get_num_blocks_to_allocate("r", end, [], start, start, end)
        free_before = pool.get_num_free_blocks()
        manager.allocate_new_blocks("r", end, end)
        assert free_before - pool.get_num_free_blocks() == estimate
        manager.cache_blocks(
            request,
            end,
            retention_interval=retention_interval,
            replay_boundaries=[5 * block_size],
        )
        request.num_computed_tokens = end

    step(0, first_chunk_blocks * block_size)
    speculative_blocks = manager.req_to_blocks["r"][-num_speculative_blocks:]
    manager.new_step_starts()
    step(first_chunk_blocks * block_size, prompt_tokens)

    checkpoint_position, checkpoint_idx = manager._checkpoints["r"]
    checkpoint_block = manager.req_to_blocks["r"][checkpoint_idx]
    assert all(
        b.block_hash is None or b is checkpoint_block for b in speculative_blocks
    )
    block_hash = request.block_hashes[checkpoint_position // block_size - 1]
    assert pool.get_cached_block(block_hash, [0]) == [checkpoint_block]


def get_sliding_window_manager(
    sliding_window_spec,
    block_pool,
    enable_caching=True,
    needs_kv_cache_zeroing=False,
):
    # Tests don't exercise admission gating; pass a large cap that is a no-op.
    return SlidingWindowManager(
        sliding_window_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
        scheduler_block_size=sliding_window_spec.block_size,
        needs_kv_cache_zeroing=needs_kv_cache_zeroing,
        max_admission_blocks_per_request=10**9,
    )


def get_chunked_local_attention_manager(
    chunked_local_attention_spec,
    block_pool,
    enable_caching=True,
    needs_kv_cache_zeroing=False,
):
    return ChunkedLocalAttentionManager(
        chunked_local_attention_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
        scheduler_block_size=chunked_local_attention_spec.block_size,
        needs_kv_cache_zeroing=needs_kv_cache_zeroing,
        max_admission_blocks_per_request=10**9,
    )


def test_circular_buffer_allocates_one_block_for_the_request_lifetime():
    block_size = 4
    spec = CircularBufferSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
    )
    block_pool = BlockPool(
        num_gpu_blocks=10, enable_caching=True, hash_block_size=block_size
    )
    manager = CircularBufferManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
        needs_kv_cache_zeroing=True,
        max_admission_blocks_per_request=1,
    )
    request_id = "request"

    assert (
        manager.get_num_blocks_to_allocate(
            request_id,
            num_tokens=128,
            new_computed_blocks=[block_pool.blocks[1]],
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=128,
        )
        == 1
    )
    blocks = manager.allocate_new_blocks(
        request_id, num_tokens=128, num_tokens_main_model=128
    )
    assert len(blocks) == 1
    assert not manager.records_new_block_ids
    assert manager.take_new_block_ids() == []

    for num_tokens in (256, 1024):
        assert (
            manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                [],
                total_computed_tokens=num_tokens - 1,
                num_local_computed_tokens=0,
                num_tokens_main_model=num_tokens,
            )
            == 0
        )
        assert manager.allocate_new_blocks(request_id, num_tokens, num_tokens) == []

    manager.cache_blocks(request_id, 1024, replay_boundaries=(0,))
    manager.remove_skipped_blocks(request_id, 1024)
    assert manager.get_num_skipped_tokens(1024) == 0
    assert manager.req_to_blocks[request_id] == blocks


@pytest.mark.parametrize("record_for_zeroing", [True, False])
def test_external_kpool_tail_zeroing(record_for_zeroing):
    """The kpool tail ring is never zeroed, transferred or not: a pool only
    reads ring slots its own request wrote."""
    spec = CircularBufferSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
    )
    manager = CircularBufferManager(
        spec,
        block_pool=BlockPool(10, enable_caching=True, hash_block_size=4),
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=4,
        needs_kv_cache_zeroing=True,
        max_admission_blocks_per_request=1,
    )
    manager.allocate_external_computed_blocks(
        "request", 16, 4, record_for_zeroing=record_for_zeroing
    )
    blocks = manager.req_to_blocks["request"]
    assert len(blocks) == 1
    assert manager.take_new_block_ids() == []
    assert manager.allocate_new_blocks("request", 20, 20) == []
    assert manager.take_new_block_ids() == []


def test_sliding_window_records_new_blocks_for_zeroing():
    block_size = 2
    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )
    block_pool = BlockPool(
        num_gpu_blocks=10, enable_caching=False, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(
        spec,
        block_pool,
        enable_caching=False,
        needs_kv_cache_zeroing=True,
    )

    blocks = manager.allocate_new_blocks(
        "request", num_tokens=4, num_tokens_main_model=4
    )

    assert manager.records_new_block_ids
    assert manager.take_new_block_ids() == [block.block_id for block in blocks]
    assert manager.take_new_block_ids() == []


def test_chunked_local_attention_records_new_blocks_for_zeroing():
    block_size = 2
    spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )
    block_pool = BlockPool(
        num_gpu_blocks=10, enable_caching=False, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(
        spec,
        block_pool,
        enable_caching=False,
        needs_kv_cache_zeroing=True,
    )

    blocks = manager.allocate_new_blocks(
        "request", num_tokens=4, num_tokens_main_model=4
    )

    assert manager.records_new_block_ids
    assert manager.take_new_block_ids() == [block.block_id for block in blocks]
    assert manager.take_new_block_ids() == []


def test_chunked_local_attention_possible_cached_prefix():
    block_size = 2
    chunked_local_attention_spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(
        chunked_local_attention_spec, block_pool
    )

    def run_one_case(block_is_cached, tail_token, expect_length):
        block_hash_list = [
            BlockHash(str(i).encode()) for i in range(len(block_is_cached))
        ]

        block_pool.cached_block_hash_to_block._cache.clear()

        # Mock the block pool with the cached blocks
        for i, (block_hash, is_cached) in enumerate(
            zip(block_hash_list, block_is_cached)
        ):
            if is_cached:
                block_pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(block_hash, 0),
                    block_pool.blocks[i + 10],
                )

        computed_blocks = manager.find_longest_cache_hit(
            block_hashes=block_hash_list,
            max_length=len(block_hash_list) * block_size + tail_token,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=chunked_local_attention_spec,
            drop_eagle_block=False,
            alignment_tokens=block_size,
        )[0][0]
        assert len(computed_blocks) == expect_length

        assert all(
            block == block_pool.null_block
            for block in computed_blocks[: (expect_length - 1) // 2]
        )

    run_one_case([True], 0, 1)
    run_one_case([True], 1, 1)
    run_one_case([True, False], 0, 2)
    run_one_case([True, False], 1, 2)
    run_one_case([True, True], 0, 2)
    run_one_case([True, True], 1, 2)
    run_one_case([True, True, False], 0, 2)
    run_one_case([True, True, False], 1, 2)
    run_one_case([True, True, True], 0, 3)
    run_one_case([True, True, True], 1, 3)
    run_one_case([True, True, True, False], 0, 4)
    run_one_case([True, True, True, False], 1, 4)
    run_one_case([random.choice([True, False])] * 8 + [True], 1, 9)
    run_one_case([random.choice([True, False])] * 8 + [False], 1, 8)
    run_one_case([random.choice([True, False])] * 8 + [True, True], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [True, False], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [True, False], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, True], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, True], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, False], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, False], 1, 10)


def test_sliding_window_possible_cached_prefix():
    block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    def run_one_case(block_is_cached, expect_length):
        block_hash_list = [
            BlockHash(str(i).encode()) for i in range(len(block_is_cached))
        ]

        block_pool.cached_block_hash_to_block._cache.clear()

        # Mock the block pool with the cached blocks
        for i, (block_hash, is_cached) in enumerate(
            zip(block_hash_list, block_is_cached)
        ):
            if is_cached:
                block_pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(block_hash, 0),
                    block_pool.blocks[i + 10],
                )

        computed_blocks = manager.find_longest_cache_hit(
            block_hashes=block_hash_list,
            max_length=len(block_hash_list) * block_size,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=sliding_window_spec,
            drop_eagle_block=False,
            alignment_tokens=block_size,
        )[0][0]
        assert len(computed_blocks) == expect_length

        assert all(
            block == block_pool.null_block
            for block in computed_blocks[: expect_length - 2]
        )
        for i in range(2):
            if i < expect_length:
                block_index = expect_length - i - 1
                assert computed_blocks[block_index].block_id == block_index + 10

    run_one_case([False] * 10, 0)
    run_one_case([True], 1)
    run_one_case([True, False], 1)
    run_one_case([True, True], 2)
    run_one_case([True, True, False], 2)
    run_one_case([True, True, True], 3)
    run_one_case([True, True, True, False], 3)
    run_one_case(
        [True, True, False, True, False, False, True, True, False, True, True, True], 12
    )
    run_one_case(
        [True, True, False, True, False, False, True, True, False, False, False], 8
    )
    run_one_case(
        [True, True, False, True, False, False, True, True, False, False, False, True],
        8,
    )


def test_sliding_window_cache_hit_with_finer_hash_alignment():
    """Sliding-window lookup uses full blocks with finer hybrid-cache hashes."""
    hash_block_size = 2
    block_size = 4
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=8,
    )
    block_pool = BlockPool(
        num_gpu_blocks=100,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)
    block_hashes = [BlockHash(str(i).encode()) for i in range(4)]

    # Each physical block uses the last chained hash in its block-size view.
    for block_hash, block in zip(
        (block_hashes[1], block_hashes[3]), block_pool.blocks[10:12]
    ):
        block_pool.cached_block_hash_to_block.insert(
            make_block_hash_with_group_id(block_hash, 0), block
        )

    computed_blocks, hit_length = manager.find_longest_cache_hit(
        block_hashes=block_hashes,
        max_length=8,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=sliding_window_spec,
        drop_eagle_block=False,
        alignment_tokens=hash_block_size,
    )

    assert hit_length == 8
    assert computed_blocks[0] == block_pool.blocks[10:12]


def test_chunked_local_attention_remove_skipped_blocks():
    attention_spec = ChunkedLocalAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=2)

    manager = get_chunked_local_attention_manager(attention_spec, block_pool)

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids) -> list[KVCacheBlock]:
        return [
            KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
            for id_ in ids
        ]

    def assert_block_id(block_table: list[KVCacheBlock], ids: list[int]):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
        1009,
        1010,
    ]
    block_table = id_to_block_table(original_block_ids)
    manager.req_to_blocks["test"] = block_table

    manager.remove_skipped_blocks("test", 0)
    assert_block_id(block_table, original_block_ids)

    # For 4th token (0-indexed), token 0-3 is out of the local attention window.
    manager.remove_skipped_blocks("test", 4)
    assert_block_id(block_table, [null_block_id] * 2)

    # For 6th token (0-indexed), token 4 - 6 are in local attention window,
    # token 0 - 3 are out, 2 blocks can be removed.
    manager.remove_skipped_blocks("test", 6)
    assert_block_id(block_table, [null_block_id] * 2 + original_block_ids[2:])
    # For 12th token (0-indexed),
    # token 0-11 are out, 6 block can be removed.
    manager.remove_skipped_blocks("test", 12)
    assert_block_id(block_table, [null_block_id] * 6)


def test_sliding_window_remove_skipped_blocks():
    sliding_window_spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=2)

    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids) -> list[KVCacheBlock]:
        return [
            KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
            for id_ in ids
        ]

    def assert_block_id(block_table: list[KVCacheBlock], ids: list[int]):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
        1009,
        1010,
    ]
    block_table = id_to_block_table(original_block_ids)
    manager.req_to_blocks["test"] = block_table

    manager.remove_skipped_blocks("test", 0)
    assert_block_id(block_table, original_block_ids)

    # 4 tokens are computed. Only token 0 is out of the sliding window. As
    # block 1000 also contains token 1 that is in the sliding window, block 1000
    # cannot be removed.
    manager.remove_skipped_blocks("test", 4)
    assert_block_id(block_table, original_block_ids)

    # 5 tokens are computed. Token 0 & 1 are out of the sliding window.
    # Block 1000 can be removed.
    manager.remove_skipped_blocks("test", 5)
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 6 tokens are computed. Token 0-2 are out of the sliding window.
    # Cannot remove new block as the block 1001 is still used by token 3.
    manager.remove_skipped_blocks("test", 6)
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 7 tokens are computed. Token 0-3 are out of the sliding window.
    # Block 1001 can be removed and block 1000 is already removed.
    manager.remove_skipped_blocks("test", 7)
    assert_block_id(block_table, [null_block_id] * 2 + original_block_ids[2:])

    # 11 tokens are computed. Token 0-7 are out of the sliding window.
    # Block 1002 & 1003 can be removed now. Block 1003 represents a longer
    # sequence, and is expected to be evicted earlier than 1002, so the order
    # of removed blocks should be [1003, 1002].
    manager.remove_skipped_blocks("test", 11)
    assert_block_id(block_table, [null_block_id] * 4 + original_block_ids[4:])


def test_rswa_remove_skipped_blocks_gap_range():
    block_size = 4
    rswa_spec = RSWASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        rswa_window=8,
    )
    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=4)
    manager = RSWAManager(
        rswa_spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )

    null_block_id = block_pool.null_block.block_id
    original_block_ids = list(range(1000, 1010))
    block_table = [
        KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
        for id_ in original_block_ids
    ]
    manager.req_to_blocks["test"] = block_table

    prefix_len = 16

    # Without num_prompt_tokens, R-SWA does not evict gap blocks.
    manager.remove_skipped_blocks("test", 28)
    assert [b.block_id for b in block_table] == original_block_ids

    # Gap = block 4 only (tokens [16, 20) fall in the gap).
    manager.remove_skipped_blocks("test", 28, num_prompt_tokens=prefix_len)
    expected = original_block_ids.copy()
    expected[4] = null_block_id
    assert [b.block_id for b in block_table] == expected

    # Window moves: blocks 5 and 6 also enter the gap; block 4 is already null.
    manager.remove_skipped_blocks("test", 36, num_prompt_tokens=prefix_len)
    expected[5] = null_block_id
    expected[6] = null_block_id
    assert [b.block_id for b in block_table] == expected


def test_get_num_blocks_to_allocate():
    block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,  # Placeholder value, not related to test result
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)
    cached_blocks_1 = [KVCacheBlock(i + 1) for i in range(10)]
    cached_blocks_2 = [block_pool.null_block for _ in range(5)] + [
        KVCacheBlock(i + 1) for i in range(5)
    ]

    assert (
        manager.get_num_blocks_to_allocate(
            "1", 20 * block_size, cached_blocks_1, 0, 0, 20 * block_size
        )
        == 20
    )
    assert (
        manager.get_num_blocks_to_allocate(
            "2", 20 * block_size, cached_blocks_2, 0, 0, 20 * block_size
        )
        == 15
    )


def test_evictable_cached_blocks_not_double_allocated():
    block_size = 2
    sliding_window_length = 2 * block_size
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window_length,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    request_id = "req"
    evictable_block = block_pool.blocks[1]  # ref_cnt == 0, eviction candidate

    num_blocks_to_allocate = manager.get_num_blocks_to_allocate(
        request_id=request_id,
        num_tokens=2 * block_size,
        new_computed_blocks=[evictable_block],
        total_computed_tokens=block_size,
        num_local_computed_tokens=block_size,
        num_tokens_main_model=2 * block_size,
    )
    # Free capacity check should count evictable cached blocks, but allocation
    # should only allocate the truly new block.
    assert num_blocks_to_allocate == 2

    manager.add_local_computed_blocks(
        request_id,
        [evictable_block],
        num_local_computed_tokens=block_size,
        num_external_computed_tokens=0,
    )
    new_blocks = manager.allocate_new_blocks(
        request_id, num_tokens=4, num_tokens_main_model=4
    )
    assert len(new_blocks) == 1
    assert len(manager.req_to_blocks[request_id]) == 2


def test_chunked_local_attention_get_num_blocks_to_allocate():
    block_size = 2
    attention_spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,  # Placeholder value, not related to test result
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(attention_spec, block_pool)
    cached_blocks_1 = [KVCacheBlock(i + 1) for i in range(10)]
    cached_blocks_2 = [block_pool.null_block for _ in range(5)] + [
        KVCacheBlock(i + 1) for i in range(5)
    ]

    assert (
        manager.get_num_blocks_to_allocate(
            "1", 20 * block_size, cached_blocks_1, 0, 0, 20 * block_size
        )
        == 20
    )
    assert (
        manager.get_num_blocks_to_allocate(
            "2", 20 * block_size, cached_blocks_2, 0, 0, 20 * block_size
        )
        == 15
    )


def test_predictor_matches_allocator_blocks_calculation_with_admission_cap():
    """In forward steps, `get_num_blocks_to_allocate` must return exactly what
    `allocate_new_blocks` will pull; otherwise `block_pool.get_new_blocks`
    raises `ValueError: Cannot get N free blocks from the pool`.
    """
    block_size = 2
    sliding_window = 8  # 4-block live window
    cap = sliding_window // block_size

    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = SlidingWindowManager(
        spec,
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=spec.block_size,
        max_admission_blocks_per_request=cap,
    )

    request_id = "req"
    total_computed = 0
    # Walk through request forward steps. Check num_blocks returned by
    # `get_num_blocks_to_allocate` matches what `allocate_new_blocks` pulls
    for num_tokens in (4, 8, 12, 16):
        predicted = manager.get_num_blocks_to_allocate(
            request_id=request_id,
            num_tokens=num_tokens,
            new_computed_blocks=[],
            total_computed_tokens=total_computed,
            num_local_computed_tokens=0,
            num_tokens_main_model=num_tokens,
        )
        new_blocks = manager.allocate_new_blocks(
            request_id, num_tokens=num_tokens, num_tokens_main_model=num_tokens
        )
        assert predicted == len(new_blocks), (
            f"num_tokens={num_tokens}: predictor returned {predicted} "
            f"but allocator pulled {len(new_blocks)}"
        )
        total_computed = num_tokens


def test_pop_blocks_for_free_trims_null_head():
    block_size = 2
    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )
    block_pool = BlockPool(
        num_gpu_blocks=10, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(spec, block_pool)
    null = block_pool.null_block
    real = block_pool.get_new_blocks(2)
    manager.req_to_blocks["swa"] = [null] * 3 + real
    manager.req_to_blocks["dense"] = dense = block_pool.get_new_blocks(2)

    assert manager.pop_blocks_for_free("swa") == real
    assert manager.pop_blocks_for_free("dense") is dense  # no null head, no copy
    assert "swa" not in manager.req_to_blocks
    assert all(b.ref_cnt == 1 for b in real + dense)

    # The null block is never pooled, so touch / free_blocks leave it alone.
    null_ref_cnt = null.ref_cnt
    num_free_blocks = block_pool.get_num_free_blocks()
    block_pool.touch([null])
    block_pool.free_blocks([null])
    assert null.ref_cnt == null_ref_cnt
    assert block_pool.get_num_free_blocks() == num_free_blocks
