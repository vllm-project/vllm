# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P0-shadow invalidation accounting on `MultiModalProcessorSenderCache`."""

from unittest.mock import MagicMock

from vllm.multimodal.cache import (
    MultiModalProcessorCacheItemMetadata,
    MultiModalProcessorSenderCache,
)


def _make_cache() -> MultiModalProcessorSenderCache:
    # A stub model_config whose `.get_multimodal_config()` returns an object
    # with a fixed `mm_processor_cache_gb`. Just enough surface for
    # `MultiModalProcessorSenderCache.__init__`.
    model_config = MagicMock()
    mm_config = MagicMock()
    mm_config.mm_processor_cache_gb = 1.0
    model_config.get_multimodal_config.return_value = mm_config
    return MultiModalProcessorSenderCache(model_config)


def _insert(cache: MultiModalProcessorSenderCache, mm_hash: str) -> None:
    # Direct-insert via the underlying LRU. `MultiModalProcessorCacheItemMetadata`
    # is a NamedTuple with (num_items, prompt_updates); shape doesn't matter
    # for accounting purposes.
    entry = MultiModalProcessorCacheItemMetadata(
        num_items=1,
        prompt_updates={},  # type: ignore[arg-type]
    )
    cache.cache_if_fits(cache._cache, mm_hash, entry)


def test_counter_starts_at_zero() -> None:
    cache = _make_cache()
    assert cache.num_invalidations() == 0
    assert cache.num_invalidations(delta=True) == 0


def test_invalidate_bumps_counter_only_on_actual_eviction() -> None:
    cache = _make_cache()
    _insert(cache, "hash-A")

    cache.invalidate("hash-A")
    assert cache.num_invalidations() == 1

    # Second invalidate on the same hash is a no-op eviction and must not
    # inflate the counter.
    cache.invalidate("hash-A")
    assert cache.num_invalidations() == 1

    # Invalidating a hash that was never inserted must also not bump.
    cache.invalidate("hash-nonexistent")
    assert cache.num_invalidations() == 1


def test_delta_polling_resets_the_window() -> None:
    cache = _make_cache()
    for mm_hash in ("h1", "h2", "h3"):
        _insert(cache, mm_hash)
        cache.invalidate(mm_hash)

    # First delta drains the whole window.
    assert cache.num_invalidations(delta=True) == 3
    # Second delta with no new invalidations is zero.
    assert cache.num_invalidations(delta=True) == 0
    # Cumulative getter still reflects the lifetime total.
    assert cache.num_invalidations() == 3

    _insert(cache, "h4")
    cache.invalidate("h4")
    # Only the new invalidation shows in the next delta.
    assert cache.num_invalidations(delta=True) == 1
    assert cache.num_invalidations() == 4
