# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CachedWeightProvider (LFRU expert cache)."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    CachedWeightProvider,
    ExpertWeightResult,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")

NUM_EXPERTS = [8, 64]
DTYPES = [torch.bfloat16, torch.float16]
CAPACITIES = [1, 4]
HIDDEN = 16
INTERMEDIATE = 32


def _make_weights(num_experts: int, dtype: torch.dtype):
    # Generate in BF16 then cast - torch.randn doesn't support FP8 dtypes
    w13 = torch.randn(num_experts, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
    w2 = torch.randn(num_experts, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16)
    return w13.to(dtype), w2.to(dtype)


def _make_scales(num_experts: int):
    w13_s = torch.rand(num_experts, 1, dtype=torch.float32)
    w2_s = torch.rand(num_experts, 1, dtype=torch.float32)
    return w13_s, w2_s


def _make_provider(
    num_experts: int = 8,
    capacity: int = 4,
    dtype: torch.dtype = torch.bfloat16,
    with_scales: bool = False,
    split: str = "token",
):
    set_random_seed(42)
    w13, w2 = _make_weights(num_experts, dtype)
    kwargs: dict = dict(capacity=capacity, w13_weight=w13, w2_weight=w2, split=split)
    scales = None
    if with_scales:
        w13_s, w2_s = _make_scales(num_experts)
        kwargs.update(w13_scale=w13_s, w2_scale=w2_s)
        scales = (w13_s, w2_s)
    return CachedWeightProvider(**kwargs), w13, w2, scales


def _topk(ids: list[int]) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int32, device="cuda").unsqueeze(0)


# -- Core cache behavior --


@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cold_miss_and_warm_hit(num_experts: int, capacity: int, dtype: torch.dtype):
    """Cold access misses, repeat access hits. GPU buffer matches source."""
    provider, w13, w2, _ = _make_provider(num_experts, capacity, dtype)
    expert_ids = list(range(min(capacity, num_experts)))

    # Cold miss
    result = provider.prepare(_topk(expert_ids))
    assert provider.misses == len(expert_ids)
    assert provider.hits == 0
    assert isinstance(result, ExpertWeightResult)
    assert result.w1 is provider.buf_w13
    assert result.w2 is provider.buf_w2
    assert result.expert_map.shape == (num_experts,)

    # Verify GPU buffer contents match source weights
    for eid in expert_ids:
        slot = provider._lru[eid][0]
        torch.testing.assert_close(result.w1[slot].cpu(), w13[eid])
        torch.testing.assert_close(result.w2[slot].cpu(), w2[eid])

    # Warm hit
    prev_misses = provider.misses
    provider.prepare(_topk(expert_ids))
    assert provider.hits == len(expert_ids)
    assert provider.misses == prev_misses


@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cache_full_equals_num_experts(num_experts: int, dtype: torch.dtype):
    """When capacity == num_experts, all fit with zero evictions."""
    provider, _, _, _ = _make_provider(num_experts, capacity=num_experts, dtype=dtype)
    all_ids = list(range(num_experts))
    provider.prepare(_topk(all_ids))
    assert provider.misses == num_experts
    assert len(provider._free_slots) == 0

    provider.prepare(_topk(all_ids))
    assert provider.hits == num_experts


@pytest.mark.parametrize("capacity", CAPACITIES)
def test_expert_map_points_at_slots(capacity: int):
    """expert_map sends resident experts to their slot and the rest to -1."""
    provider, _, _, _ = _make_provider(capacity=capacity)
    ids = list(range(min(capacity, 8)))
    result = provider.prepare(_topk(ids))

    mapping = result.expert_map.tolist()
    for eid in ids:
        assert mapping[eid] == provider._lru[eid][0]
    for eid in range(len(mapping)):
        if eid not in provider._lru:
            assert mapping[eid] == -1


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_accepts_either_topk_dtype(dtype: torch.dtype):
    """topk_ids may be int32 or int64; the map is always int32."""
    provider, *_ = _make_provider()
    ids = torch.tensor([[0, 1]], dtype=dtype, device="cuda")
    result = provider.prepare(ids)
    assert result.expert_map.dtype == torch.int32


# -- LFRU eviction semantics --


def test_lfru_prefers_evicting_low_frequency():
    """LFRU evicts the expert with lowest freq/age score, not pure LRU.
    A accessed 5x, B accessed 1x. When C arrives, B is evicted, not A.
    """
    provider, w13, _, _ = _make_provider(capacity=2)
    provider.prepare(_topk([0, 1]))
    for _ in range(4):
        provider.prepare(_topk([0]))  # A freq=5
    provider.prepare(_topk([1]))  # touch B for recency parity

    provider.prepare(_topk([2]))  # evicts B (lower freq/age score)
    assert 0 in provider._lru, "High-frequency expert A should survive"
    assert 2 in provider._lru, "New expert C should be cached"
    assert 1 not in provider._lru, "Low-frequency expert B should be evicted"
    slot_c = provider._lru[2][0]
    torch.testing.assert_close(provider.buf_w13[slot_c].cpu(), w13[2])


def test_lfru_evicts_stale_high_freq_expert():
    """High historical freq but old last-access loses to recent low-freq.
    Distinguishes LFRU (score=freq/age) from pure frequency-based caching.
    """
    provider, _, _, _ = _make_provider(capacity=2)

    # Expert 0: accessed 11x early, then becomes stale
    provider.prepare(_topk([0]))
    for _ in range(10):
        provider.prepare(_topk([0]))
    # Expert 1: loaded later, accessed 51x (0 becomes very stale)
    provider.prepare(_topk([1]))
    for _ in range(50):
        provider.prepare(_topk([1]))

    # Expert 0: freq=11, age~62 -> score~0.18. Expert 1: freq=51, age=1 -> 51
    provider.prepare(_topk([2]))
    assert 1 in provider._lru, "Recent high-freq expert should survive"
    assert 0 not in provider._lru, "Stale expert should be evicted"


def test_capacity_one_always_evicts():
    """With capacity=1, every new expert evicts the previous."""
    provider, *_ = _make_provider(capacity=1)
    for eid in range(5):
        provider.prepare(_topk([eid]))
    assert provider.misses == 5
    assert provider.hits == 0
    assert len(provider._lru) == 1
    assert 4 in provider._lru


# -- GPU buffer correctness under eviction --


def test_gpu_buffer_correct_after_eviction():
    """After eviction, the reused slot contains the new expert's weights."""
    provider, w13, w2, _ = _make_provider(capacity=4)
    provider.prepare(_topk([0, 1, 2, 3]))

    # Make 0 the eviction candidate (least recently used, lowest freq)
    provider.prepare(_topk([1, 2, 3]))
    slot_for_0 = provider._lru[0][0]

    provider.prepare(_topk([7]))
    assert provider._lru[7][0] == slot_for_0
    torch.testing.assert_close(provider.buf_w13[slot_for_0].cpu(), w13[7])
    torch.testing.assert_close(provider.buf_w2[slot_for_0].cpu(), w2[7])


def test_batch_experts_survive_intra_batch_eviction():
    """Every expert in a batch maps to a slot holding its own weights.

    A miss loads the expert with freq=1, the lowest possible LFRU score, so a
    later miss in the same batch would evict it and reuse its slot while
    _mapping still points there — feeding the kernel another expert's weights
    with no error raised.
    """
    provider, w13, w2, _ = _make_provider(num_experts=8, capacity=4)

    # Warm every slot with high-frequency experts so the buffer is full and
    # each resident entry outscores a freshly loaded one.
    for _ in range(50):
        provider.prepare(_topk([0, 1, 2, 3]))

    batch = [4, 5, 6, 7]
    result = provider.prepare(_topk(batch))

    for expert_id in batch:
        slot = provider._lru[expert_id][0]
        torch.testing.assert_close(provider.buf_w13[slot].cpu(), w13[expert_id])
        torch.testing.assert_close(provider.buf_w2[slot].cpu(), w2[expert_id])

    # Every batch expert must be resident, not mapped away.
    assert all(result.expert_map[e].item() >= 0 for e in batch)


# -- Scale buffer handling --


def test_scale_lifecycle():
    """Scales are allocated, copied on load, and updated on eviction."""
    if not current_platform.has_device_capability(89):
        pytest.skip("FP8 requires CUDA capability >= 89")

    provider, _, _, scales = _make_provider(
        capacity=4, dtype=torch.float8_e4m3fn, with_scales=True
    )
    w13_s, w2_s = scales

    # Buffers allocated on GPU
    assert provider.buf_w13_scale is not None
    assert provider.buf_w2_scale is not None
    assert provider.buf_w13_scale.device.type == "cuda"

    # Scales copied correctly on load
    result = provider.prepare(_topk([3, 6]))
    for eid in [3, 6]:
        slot = provider._lru[eid][0]
        torch.testing.assert_close(result.w1_scale[slot].cpu(), w13_s[eid])
        torch.testing.assert_close(result.w2_scale[slot].cpu(), w2_s[eid])

    # Fill cache and evict: scales must be updated in evicted slot
    provider.prepare(_topk([0, 1]))  # cache now full: [3, 6, 0, 1]
    provider.prepare(_topk([3, 6, 0]))  # boost freq on 3,6,0; expert 1 stale

    result = provider.prepare(_topk([7]))  # evicts 1
    assert 1 not in provider._lru
    slot_7 = provider._lru[7][0]
    torch.testing.assert_close(provider.buf_w13_scale[slot_7].cpu(), w13_s[7])
    torch.testing.assert_close(provider.buf_w2_scale[slot_7].cpu(), w2_s[7])


def test_no_scales_when_not_provided():
    """Without scale inputs, scale buffers remain None."""
    provider, *_ = _make_provider()
    assert provider.buf_w13_scale is None
    assert provider.buf_w2_scale is None
    result = provider.prepare(_topk([0]))
    assert result.w1_scale is None
    assert result.w2_scale is None


# -- Invalidation --


def test_invalidate_frees_slot():
    """invalidate() removes an expert and returns its slot to the free list."""
    provider, *_ = _make_provider()
    provider.prepare(_topk([0, 1, 2, 3]))
    old_slot = provider._lru[2][0]
    provider.invalidate(2)
    assert 2 not in provider._lru
    assert old_slot in provider._free_slots


def test_invalidate_noop_when_absent():
    """invalidate() on an uncached expert is a no-op."""
    provider, *_ = _make_provider()
    provider.invalidate(99)  # must not raise


# -- Overflow (unique experts > capacity) --


def test_overflow_raises():
    """When unique experts exceed capacity, raise RuntimeError immediately."""
    provider, *_ = _make_provider(capacity=2)
    with pytest.raises(RuntimeError, match="unique experts"):
        provider.prepare(_topk([0, 1, 2, 3]))


# -- CPU pinned memory --


def test_cpu_backing_is_pinned():
    """CPU weight tensors must be pinned for async H2D copies."""
    provider, *_ = _make_provider()
    assert provider._cpu_w13.is_pinned()
    assert provider._cpu_w2.is_pinned()


# -- Expert-group execution --


def _rows(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def test_single_group_when_experts_fit():
    """A forward within capacity runs as one group."""
    provider, *_ = _make_provider(num_experts=8, capacity=4, split="expert")
    groups = provider.plan_expert_groups(_rows([[0, 1], [1, 2], [2, 3]]))
    assert groups == [[0, 1, 2, 3]]


def test_groups_partition_all_experts():
    """Every expert appears in exactly one group, and no group overflows."""
    provider, *_ = _make_provider(num_experts=8, capacity=3, split="expert")
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    groups = provider.plan_expert_groups(topk)

    assert len(groups) > 1
    flat = [e for g in groups for e in g]
    assert sorted(flat) == sorted(topk.unique().tolist())
    assert len(flat) == len(set(flat)), "an expert landed in two groups"
    for g in groups:
        assert len(g) <= provider.capacity


def test_group_count_is_independent_of_token_count():
    """Group count follows the expert set, not the batch size."""
    provider, *_ = _make_provider(num_experts=8, capacity=4, split="expert")
    few = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])
    many = _rows([[0, 1], [2, 3], [4, 5], [6, 7]] * 50)
    assert len(provider.plan_expert_groups(few)) == len(
        provider.plan_expert_groups(many)
    )


def test_expert_map_hides_other_groups():
    """Within a group only that group's experts are resident."""
    provider, w13, _, _ = _make_provider(num_experts=8, capacity=3, split="expert")
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    for group in provider.plan_expert_groups(topk):
        result = provider.prepare(topk, group)
        mapping = result.expert_map.tolist()
        for eid in group:
            slot = mapping[eid]
            assert slot >= 0
            torch.testing.assert_close(result.w1[slot].cpu(), w13[eid])
        resident = {e for e, slot in enumerate(mapping) if slot >= 0}
        assert resident == set(group)


def test_group_execution_covers_every_pair():
    """Summing over groups touches each (token, expert) pair exactly once."""
    from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
        run_with_expert_cache,
    )

    provider, *_ = _make_provider(num_experts=8, capacity=3, split="expert")
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    seen: list[int] = []
    firsts: list[bool] = []

    def run(result, rows, include_shared):
        firsts.append(include_shared)
        seen.extend(e for e, slot in enumerate(result.expert_map.tolist()) if slot >= 0)
        return torch.zeros(topk.size(0), 4, device="cuda")

    out = run_with_expert_cache(provider, topk, run)

    assert sorted(seen) == sorted(topk.unique().tolist())
    assert firsts[0] is True and not any(firsts[1:]), "first flag must fire once"
    assert out.shape == (topk.size(0), 4)


# -- Token-split execution --


def test_token_split_single_chunk_when_batch_fits():
    """A batch within capacity is evaluated in one piece."""
    provider, *_ = _make_provider(num_experts=8, capacity=4)
    plan = provider.plan_chunks(_rows([[0, 1], [1, 2], [2, 3]]))
    assert [rows for rows, _ in plan] == [slice(0, 3)]
    assert plan[0][1] == [0, 1, 2, 3]


def test_token_split_covers_every_row_once():
    """Chunks tile the batch, and each fits the cache."""
    provider, *_ = _make_provider(num_experts=8, capacity=4)
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    plan = provider.plan_chunks(topk)
    chunks = [rows for rows, _ in plan]

    assert len(chunks) > 1
    assert chunks[0].start == 0
    assert chunks[-1].stop == topk.size(0)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.stop == nxt.start
    for rows, unique_ids in plan:
        assert topk[rows].unique().numel() <= provider.capacity
        assert unique_ids == sorted(topk[rows].unique().tolist())


def test_token_split_rejects_single_row_over_capacity():
    """No split helps when one token alone exceeds capacity."""
    provider, *_ = _make_provider(num_experts=8, capacity=2)
    with pytest.raises(RuntimeError, match="one token routes to"):
        provider.plan_chunks(_rows([[0, 1, 2, 3]]))


@pytest.mark.parametrize("split", ["token", "expert"])
def test_both_splits_reach_every_expert(split: str):
    """Whatever the split, every expert the batch needs gets loaded."""
    from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
        run_with_expert_cache,
    )

    provider, *_ = _make_provider(num_experts=8, capacity=3, split=split)
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    seen: set[int] = set()

    def run(result, rows, include_shared):
        seen.update(e for e, s in enumerate(result.expert_map.tolist()) if s >= 0)
        return torch.zeros(topk[rows].size(0), 4, device="cuda")

    out = run_with_expert_cache(provider, topk, run)

    assert seen == set(topk.unique().tolist())
    assert out.shape == (topk.size(0), 4)
