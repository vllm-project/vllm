# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HF3FS metadata server data structures and allocation logic:
- RankFileMetadata : page allocation / release primitives
- KeyMetadata      : per-key rank-page tracking and completion detection
- GlobalMetadataState : coordinated allocation with cache-hit semantics
"""

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_metadata_server import (
    GlobalMetadataState,
    KeyMetadata,
    RankFileMetadata,
)

# ===========================================================================
# TestRankFileMetadata
# ===========================================================================


class TestRankFileMetadata:
    """Unit tests for RankFileMetadata page allocation primitives."""

    @pytest.mark.parametrize(
        "alloc_count, expected_pages",
        [(3, 3), (5, 0)],
        ids=["alloc_partial", "alloc_exceeds"],
    )
    def test_allocate_pages(self, alloc_count, expected_pages):
        """allocate_pages returns correct pages or empty list when insufficient."""
        rank_meta = RankFileMetadata(rank_id=0, num_pages=3, free_pages=list(range(3)))
        pages = rank_meta.allocate_pages(alloc_count)
        assert len(pages) == expected_pages
        if expected_pages > 0:
            rank_meta.release_pages(pages)
            assert rank_meta.get_free_page_count() == 3

    def test_release_pages_restores_count(self):
        """Releasing allocated pages returns them to the free pool."""
        rank_meta = RankFileMetadata(rank_id=0, num_pages=4, free_pages=list(range(4)))
        pages = rank_meta.allocate_pages(2)
        assert rank_meta.get_free_page_count() == 2
        rank_meta.release_pages(pages)
        assert rank_meta.get_free_page_count() == 4

    def test_release_pages_no_duplicates(self):
        """Releasing the same page twice must not create duplicates."""
        rank_meta = RankFileMetadata(rank_id=0, num_pages=3, free_pages=list(range(3)))
        rank_meta.allocate_pages(1)  # takes page 0
        rank_meta.release_pages([0])
        rank_meta.release_pages([0])  # second release of the same page
        assert rank_meta.get_free_page_count() == 3


# ===========================================================================
# TestKeyMetadata
# ===========================================================================


class TestKeyMetadata:
    """Unit tests for KeyMetadata completion tracking."""

    def test_is_complete_false_until_all_ranks(self):
        """is_complete() returns True only when all ranks confirmed."""
        key_meta = KeyMetadata(key="k", rank_to_page={}, tp_world_size=2)
        assert key_meta.is_complete() is False
        key_meta.add_rank_page(0, 5)
        assert key_meta.is_complete() is False
        key_meta.add_rank_page(1, 10)
        assert key_meta.is_complete() is True

    def test_get_rank_page_returns_none_for_missing_rank(self):
        """get_rank_page() returns None when the rank has no entry."""
        key_meta = KeyMetadata(key="k", rank_to_page={0: 3}, tp_world_size=2)
        assert key_meta.get_rank_page(0) == 3
        assert key_meta.get_rank_page(1) is None

    def test_get_all_pages(self):
        """get_all_pages() returns all (rank, page) pairs."""
        key_meta = KeyMetadata(key="k", rank_to_page={0: 1, 1: 2}, tp_world_size=2)
        pairs = key_meta.get_all_pages()
        assert set(pairs) == {(0, 1), (1, 2)}


# ===========================================================================
# TestGlobalMetadataStateAllocation
# ===========================================================================


class TestGlobalMetadataStateAllocation:
    """Tests for GlobalMetadataState allocation and cache-hit semantics."""

    def test_uninitialized_rank_raises_on_allocate(self):
        """allocate_pages_for_keys raises ValueError for unknown rank."""
        state = GlobalMetadataState()
        with pytest.raises((ValueError, Exception)):
            state.allocate_pages_for_keys(99, [("key", "")])

    def test_uninitialized_rank_raises_on_get_locations(self):
        """get_key_locations raises ValueError for unknown rank."""
        state = GlobalMetadataState()
        with pytest.raises((ValueError, Exception)):
            state.get_key_locations(99, ["any_key"])

    def test_basic_allocation_and_confirm(self):
        """Allocating a page and confirming it marks the key as complete."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 4)

        results = state.allocate_pages_for_keys(0, [("K", "")])
        assert results["K"] >= 0

        state.confirm_write_for_keys(0, [("K", results["K"])])
        assert state.batch_key_exists(["K"]) == [True]
        locations = state.get_key_locations(0, ["K"])
        assert locations == [results["K"]]

    def test_allocate_pages_cache_hit_does_not_leak_pages(self):
        """Cache-hit key must not consume a page from the free pool;
        the pre-allocated slot must be returned before reusing the existing page.
        """
        state = GlobalMetadataState()
        state.initialize_rank(0, 5)  # 5 free pages: [0,1,2,3,4]

        # Simulate a key that has already been fully written and confirmed.
        state.key_metadata["K_cached"] = KeyMetadata(
            key="K_cached", rank_to_page={0: 2}, tp_world_size=1
        )

        free_before = state.rank_metadata[0].get_free_page_count()  # 5

        results = state.allocate_pages_for_keys(0, [("K_cached", ""), ("K_new", "")])

        free_after = state.rank_metadata[0].get_free_page_count()

        # Cache-hit key must reuse its existing page.
        assert results["K_cached"] == 2, (
            f"Cache-hit key should reuse page 2, got {results['K_cached']}"
        )
        # New key must receive a valid page.
        assert results["K_new"] >= 0, (
            f"New key should get a valid page, got {results['K_new']}"
        )
        # Exactly one page consumed from the free pool.
        assert free_before - free_after == 1, (
            f"Expected 1 page consumed, got delta={free_before - free_after}"
        )

    def test_allocate_pages_all_cache_hits_frees_all_slots(self):
        """When every key in the batch is a cache hit, no pages are consumed."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 5)

        for key, page in (("K1", 0), ("K2", 1)):
            state.key_metadata[key] = KeyMetadata(
                key=key, rank_to_page={0: page}, tp_world_size=1
            )

        free_before = state.rank_metadata[0].get_free_page_count()
        results = state.allocate_pages_for_keys(0, [("K1", ""), ("K2", "")])
        free_after = state.rank_metadata[0].get_free_page_count()

        assert results["K1"] == 0
        assert results["K2"] == 1
        assert free_after == free_before, (
            f"All-cache-hit batch must not consume free pages; "
            f"before={free_before}, after={free_after}"
        )

    def test_allocate_returns_minus_one_when_pool_exhausted(self):
        """If the free pool is exhausted, all new keys receive -1."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 1)  # only 1 free page

        results = state.allocate_pages_for_keys(0, [("K1", ""), ("K2", "")])
        # allocate_pages uses all-or-nothing: 2 needed but only 1 available → []
        assert all(v == -1 for v in results.values()), f"Expected all -1, got {results}"

    @pytest.mark.parametrize("cached_first", [True, False])
    def test_mixed_batch_uses_capacity_only_for_misses(self, cached_first):
        """A cache hit must not prevent an otherwise fitting new allocation."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 2)
        cached_page = state.allocate_pages_for_keys(0, [("cached", "")])["cached"]
        state.confirm_write_for_keys(0, [("cached", cached_page)])

        keys = [("cached", ""), ("new", "")]
        if not cached_first:
            keys.reverse()
        results = state.allocate_pages_for_keys(0, keys)

        assert results["cached"] == cached_page
        assert results["new"] >= 0
        assert results["new"] != cached_page
        assert state.rank_metadata[0].get_free_page_count() == 0

    def test_insufficient_misses_preserve_cached_page_and_free_capacity(self):
        """An oversized miss batch must not consume pages or invalidate hits."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 2)
        cached_page = state.allocate_pages_for_keys(0, [("cached", "")])["cached"]
        state.confirm_write_for_keys(0, [("cached", cached_page)])

        results = state.allocate_pages_for_keys(
            0, [("new_a", ""), ("cached", ""), ("new_b", "")]
        )

        assert results == {"new_a": -1, "cached": cached_page, "new_b": -1}
        assert state.get_key_locations(0, ["cached"]) == [cached_page]
        assert state.rank_metadata[0].get_free_page_count() == 1

    def test_duplicate_new_key_consumes_one_page(self):
        """The key-based allocation response must not orphan a duplicate's page."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 2)

        results = state.allocate_pages_for_keys(0, [("new", ""), ("new", "")])

        assert results == {"new": 0}
        assert state.rank_metadata[0].get_free_page_count() == 1

    def test_confirm_write_releases_pages(self):
        """confirm_write_for_keys with pages_to_release returns them to pool."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 3)

        results = state.allocate_pages_for_keys(0, [("K", "")])
        page = results["K"]
        free_after_alloc = state.rank_metadata[0].get_free_page_count()

        state.confirm_write_for_keys(0, [("K", page)], pages_to_release=[page])
        free_after_release = state.rank_metadata[0].get_free_page_count()

        assert free_after_release == free_after_alloc + 1


# ===========================================================================
# TestGlobalMetadataStateEviction
# ===========================================================================


def _fill(state, keys, pages):
    """Write `keys` into pre-selected pages so the free pool is empty."""
    state.rank_metadata[0].free_pages = [
        p for p in state.rank_metadata[0].free_pages if p not in pages
    ]
    for key, page in zip(keys, pages):
        state.confirm_write_for_keys(0, [(key, page)])


class TestGlobalMetadataStateEviction:
    """Tests for LRU eviction, reader leases, and the unchanged all-or-nothing."""

    def test_eviction_frees_page_for_new_key(self):
        """A full pool with an unleased complete key makes room for a new key."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 2)
        _fill(state, ["K1", "K2"], [0, 1])
        assert state.rank_metadata[0].get_free_page_count() == 0

        results = state.allocate_pages_for_keys(0, [("K3", "")])

        assert results["K3"] == 0, results  # oldest key's page was reclaimed
        assert state.batch_key_exists(["K1"]) == [False]
        assert state.get_key_locations(0, ["K1"]) == [None]
        assert state.evicted_keys == 1

    def test_leased_key_is_not_evicted(self):
        """A key held by a reader survives eviction; allocation fails instead."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 1)
        _fill(state, ["K1"], [0])

        assert state.acquire_keys(0, ["K1"]) == {"K1": 0}
        assert state.allocate_pages_for_keys(0, [("K2", "")]) == {"K2": -1}
        assert state.batch_key_exists(["K1"]) == [True]

        state.release_keys(0, ["K1"])
        assert state.allocate_pages_for_keys(0, [("K2", "")]) == {"K2": 0}

    def test_expired_lease_is_reclaimed(self):
        """A lease from a crashed reader expires instead of pinning pages."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 1)
        _fill(state, ["K1"], [0])

        state.acquire_keys(0, ["K1"], ttl_s=-1.0)  # already expired
        assert state.allocate_pages_for_keys(0, [("K2", "")]) == {"K2": 0}
        assert state.batch_key_exists(["K1"]) == [False]

    def test_eviction_skips_partial_keys(self):
        """A key without a confirmed write is never reclaimed by eviction."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 2)
        partial_page = state.allocate_pages_for_keys(0, [("partial", "")])["partial"]
        _fill(state, ["done"], [1])

        results = state.allocate_pages_for_keys(0, [("new", "")])

        assert results["new"] == 1
        # The partial key still owns its page: only the complete key was evicted.
        assert state.rank_metadata[0].get_free_page_count() == 0
        assert state.batch_key_exists(["partial"]) == [False]
        assert state.get_key_locations(0, ["partial"]) == [None]
        assert partial_page == 0

    def test_hits_of_the_allocating_request_are_not_evicted(self):
        """Pages a request is about to read from cache must not be recycled."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 2)
        _fill(state, ["K1", "K2"], [0, 1])

        results = state.allocate_pages_for_keys(0, [("K1", ""), ("K3", "")])

        assert results["K1"] == 0
        assert results["K3"] == 1
        assert state.batch_key_exists(["K1"]) == [True]
        assert state.batch_key_exists(["K2"]) == [False]

    def test_eviction_prefers_least_recently_used(self):
        """Acquiring a key refreshes its LRU position."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 2)
        _fill(state, ["K1", "K2"], [0, 1])

        state.acquire_keys(0, ["K1"])
        state.release_keys(0, ["K1"])

        results = state.allocate_pages_for_keys(0, [("K3", "")])

        assert results["K3"] == 1
        assert state.batch_key_exists(["K1"]) == [True]
        assert state.batch_key_exists(["K2"]) == [False]

    def test_eviction_across_ranks_releases_every_page(self):
        """Evicting a key returns one page per rank, keeping the pools disjoint."""
        state = GlobalMetadataState()
        state.initialize_rank(0, 1)
        state.initialize_rank(1, 1)
        state.confirm_write_for_keys(0, [("K", 0)])
        state.confirm_write_for_keys(1, [("K", 0)])
        assert state.batch_key_exists(["K"]) == [True]

        assert state.allocate_pages_for_keys(0, [("K2", "")]) == {"K2": 0}

        assert state.rank_metadata[0].get_free_page_count() == 0
        assert state.rank_metadata[1].get_free_page_count() == 1
        assert state.rank_metadata[1].free_pages == [0]
