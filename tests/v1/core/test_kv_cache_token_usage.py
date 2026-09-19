# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KVCacheManager.token_usage.

The interesting property is that ``token_usage`` is *exact*, not sampled, and it
gets there without any per-block bookkeeping. Every test here checks the number
it reports against token counts computed by hand, so the assertions fail if the
identity behind it ever stops holding:

    held_tokens = used_blocks * block_size - sum(tail remainder)

That identity relies on only full blocks being shareable, so a partially filled
block is private to one request. The sharing tests below are the ones that would
catch a regression there: they are constructed so that a naive
``sum(num_computed_tokens)`` would double-count.
"""

import pytest

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash

from .test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_config_hybrid_model,
    make_kv_cache_manager,
    make_request,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _manager(block_size: int, num_blocks: int, **kwargs):
    return make_kv_cache_manager(
        make_kv_cache_config(block_size, num_blocks),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        **kwargs,
    )


def _admit(manager, request_id, token_ids, block_size):
    """Allocate for a request the way the scheduler does, and mark its tokens
    computed. ``allocate_slots`` does not advance ``num_computed_tokens``; the
    scheduler does that itself, so the test has to as well."""
    req = make_request(request_id, token_ids, block_size, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(req)
    blocks = manager.allocate_slots(
        req, len(token_ids) - num_computed, num_computed, computed
    )
    assert blocks is not None
    req.num_computed_tokens = len(token_ids)
    return req


def _held_tokens(manager, running, block_size, num_blocks):
    """Reconstruct the absolute token count from the reported fraction."""
    total_slots = (num_blocks - 1) * block_size
    return round(manager.token_usage(running) * total_slots)


def test_empty_cache_holds_no_tokens():
    manager = _manager(16, 100)
    assert manager.token_usage([]) == 0.0


def test_disjoint_requests_hold_exactly_the_tokens_they_brought():
    block_size, num_blocks = 16, 100
    manager = _manager(block_size, num_blocks)

    running = []
    expected = 0
    for i, n in enumerate([17, 20, 33, 64, 100]):
        # Disjoint token id ranges, so nothing is shared and the held token
        # count is simply the sum of the prompt lengths.
        tokens = list(range(i * 10_000, i * 10_000 + n))
        running.append(_admit(manager, f"r{i}", tokens, block_size))
        expected += n
        assert _held_tokens(manager, running, block_size, num_blocks) == expected


def test_full_blocks_only_agrees_with_block_usage():
    block_size, num_blocks = 16, 100
    manager = _manager(block_size, num_blocks)

    running = []
    for i, n in enumerate([16, 32, 64]):
        tokens = list(range(i * 10_000, i * 10_000 + n))
        running.append(_admit(manager, f"r{i}", tokens, block_size))

    # No request has a partial tail, so the two units coincide.
    assert manager.token_usage(running) == pytest.approx(manager.usage)


def test_block_size_one_agrees_with_block_usage():
    block_size, num_blocks = 1, 200
    manager = _manager(block_size, num_blocks)

    running = [
        _admit(manager, f"r{i}", list(range(i * 10_000, i * 10_000 + 17)), block_size)
        for i in range(3)
    ]

    # A one-token block can never be partially filled.
    assert manager.token_usage(running) == pytest.approx(manager.usage)


def test_shared_full_blocks_are_not_counted_twice():
    """Four identical 70-token requests, block_size 16.

    ``cacheable_length(70, 16) == 64``, so blocks 0-3 are cached and shared by
    every later request; block 4 holds 6 tokens and is private to each.

        distinct blocks = 4 shared + 4 private = 8
        held tokens     = 64 + 4 * 6 = 88

    Summing ``num_computed_tokens`` would give 4 * 70 = 280, more than three
    times the truth, so this test pins the property that makes the identity work.
    """
    block_size, num_blocks = 16, 100
    manager = _manager(block_size, num_blocks)

    tokens = list(range(70))
    running = [_admit(manager, f"r{i}", tokens, block_size) for i in range(4)]

    assert _held_tokens(manager, running, block_size, num_blocks) == 88
    assert sum(r.num_computed_tokens for r in running) == 280


def test_partial_blocks_make_block_usage_overstate_occupancy():
    """The case the metric exists for: 17-token prompts with block_size 16.

    Every request needs two blocks and fills 17 of the 32 slots, so block usage
    reads about twice the token usage.
    """
    block_size, num_blocks = 16, 65
    manager = _manager(block_size, num_blocks)

    running = []
    for i in range(32):
        tokens = list(range(i * 10_000, i * 10_000 + 17))
        running.append(_admit(manager, f"r{i}", tokens, block_size))

    # 32 requests * 2 blocks = 64 blocks, i.e. the whole pool.
    assert manager.usage == pytest.approx(1.0)
    # 32 * 17 = 544 of 64 * 16 = 1024 slots.
    assert _held_tokens(manager, running, block_size, num_blocks) == 544
    assert manager.token_usage(running) == pytest.approx(544 / 1024)
    assert manager.token_usage(running) < manager.usage


def test_token_usage_never_exceeds_block_usage():
    block_size, num_blocks = 32, 200
    manager = _manager(block_size, num_blocks)

    running = []
    for i, n in enumerate([1, 5, 31, 32, 33, 63, 64, 65, 200, 511]):
        tokens = list(range(i * 10_000, i * 10_000 + n))
        running.append(_admit(manager, f"r{i}", tokens, block_size))
        assert manager.token_usage(running) <= manager.usage + 1e-12


def test_hybrid_model_falls_back_to_block_usage():
    """With several KV cache groups the groups have different block sizes, so a
    single token-slot denominator is not well defined. Reporting the block
    figure is better than reporting a number nobody can interpret."""
    block_size = 16
    manager = make_kv_cache_manager(
        make_kv_cache_config_hybrid_model(block_size, 100, sliding_window_blocks=2),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    running = [
        _admit(manager, f"r{i}", list(range(i * 10_000, i * 10_000 + 17)), block_size)
        for i in range(3)
    ]

    assert manager.num_kv_cache_groups > 1
    assert manager.token_usage(running) == manager.usage
