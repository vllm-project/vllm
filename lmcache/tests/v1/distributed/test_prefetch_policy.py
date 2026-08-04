# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for prefetch policy interface and DefaultPrefetchPolicy.

Tests are written against the PrefetchPolicy contract defined in
prefetch_policy.py.
"""

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
    RetainPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
)

# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_descriptor(index: int) -> AdapterDescriptor:
    """Create an AdapterDescriptor for testing."""
    config = MockL2AdapterConfig(max_size_gb=1.0, mock_bandwidth_gb=10.0)
    return AdapterDescriptor(index=index, config=config)


def make_bitmap(size: int, set_bits: list[int]) -> Bitmap:
    """Create a Bitmap with specific bits set."""
    bitmap = Bitmap(size)
    for i in set_bits:
        bitmap.set(i)
    return bitmap


def plan_to_indices(plan: dict[int, Bitmap]) -> dict[int, list[int]]:
    """Convert a Bitmap-based load plan to index lists for easy assertion."""
    return {
        adapter_idx: bitmap.get_indices_list() for adapter_idx, bitmap in plan.items()
    }


# =============================================================================
# DefaultPrefetchPolicy Tests
# =============================================================================


class TestDefaultPrefetchPolicy:
    """Test DefaultPrefetchPolicy.select_load_plan behavior."""

    def test_single_adapter_all_keys_found(self):
        """All found keys assigned to the single adapter."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0)]
        lookup_results = {0: make_bitmap(3, [0, 1, 2])}

        result = policy.select_load_plan(keys, lookup_results, adapters)

        assert plan_to_indices(result) == {0: [0, 1, 2]}

    def test_single_adapter_partial_hits(self):
        """Only found keys are in the plan."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(4)]
        adapters = [make_descriptor(0)]
        # Only keys 0 and 2 found
        lookup_results = {0: make_bitmap(4, [0, 2])}

        result = policy.select_load_plan(keys, lookup_results, adapters)

        assert plan_to_indices(result) == {0: [0, 2]}

    def test_multi_adapter_overlap_first_wins(self):
        """When key is in multiple adapters, lowest-index adapter gets it."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]
        # Both adapters have key 1
        lookup_results = {
            0: make_bitmap(3, [0, 1]),
            1: make_bitmap(3, [1, 2]),
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        # key 0 → adapter 0, key 1 → adapter 0 (first wins), key 2 → adapter 1
        assert plan_to_indices(result) == {0: [0, 1], 1: [2]}

    def test_multi_adapter_disjoint(self):
        """Each adapter gets its unique keys."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(4)]
        adapters = [make_descriptor(0), make_descriptor(1)]
        lookup_results = {
            0: make_bitmap(4, [0, 1]),
            1: make_bitmap(4, [2, 3]),
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        assert plan_to_indices(result) == {0: [0, 1], 1: [2, 3]}

    def test_no_hits_returns_empty(self):
        """Empty bitmaps → empty plan."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]
        lookup_results = {
            0: make_bitmap(3, []),
            1: make_bitmap(3, []),
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        assert plan_to_indices(result) == {}

    def test_empty_adapters_returns_empty(self):
        """No adapters means no plan."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(0)]

        result = policy.select_load_plan(keys, {}, [])

        assert plan_to_indices(result) == {}

    def test_empty_keys_returns_empty(self):
        """Empty keys list → empty plan."""
        policy = DefaultPrefetchPolicy()
        adapters = [make_descriptor(0)]
        lookup_results = {0: make_bitmap(0, [])}

        result = policy.select_load_plan([], lookup_results, adapters)

        assert plan_to_indices(result) == {}

    def test_adapter_order_matters(self):
        """Adapter with lower index always has priority, regardless of order
        in the adapters list."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(0)]
        # Adapter 1 listed first, but adapter 0 should win
        adapters = [make_descriptor(1), make_descriptor(0)]
        lookup_results = {
            0: make_bitmap(1, [0]),
            1: make_bitmap(1, [0]),
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        # Adapter 0 gets the key (lower index wins)
        assert plan_to_indices(result) == {0: [0]}

    def test_three_adapters_with_overlap(self):
        """Three adapters with partial overlaps."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(5)]
        adapters = [make_descriptor(0), make_descriptor(1), make_descriptor(2)]
        lookup_results = {
            0: make_bitmap(5, [0, 3]),  # has keys 0, 3
            1: make_bitmap(5, [1, 2, 3]),  # has keys 1, 2, 3
            2: make_bitmap(5, [2, 3, 4]),  # has keys 2, 3, 4
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        # key 0 → adapter 0
        # key 1 → adapter 1
        # key 2 → adapter 1 (lower than adapter 2)
        # key 3 → adapter 0 (lowest that has it)
        # key 4 → adapter 2
        assert plan_to_indices(result) == {0: [0, 3], 1: [1, 2], 2: [4]}

    def test_missing_lookup_result_for_adapter(self):
        """Adapter with no lookup result is skipped."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(2)]
        adapters = [make_descriptor(0), make_descriptor(1)]
        # Only adapter 1 has results
        lookup_results = {1: make_bitmap(2, [0, 1])}

        result = policy.select_load_plan(keys, lookup_results, adapters)

        assert plan_to_indices(result) == {1: [0, 1]}

    def test_each_key_appears_at_most_once(self):
        """No key should be assigned to multiple adapters."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1), make_descriptor(2)]
        # All adapters have all keys
        lookup_results = {
            0: make_bitmap(3, [0, 1, 2]),
            1: make_bitmap(3, [0, 1, 2]),
            2: make_bitmap(3, [0, 1, 2]),
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        # All keys should go to adapter 0 only
        assert plan_to_indices(result) == {0: [0, 1, 2]}
        assert 1 not in result
        assert 2 not in result


# =============================================================================
# DefaultPrefetchPolicy.select_l1_retentions Tests
# =============================================================================


class TestDefaultPrefetchPolicyRetentions:
    """Test DefaultPrefetchPolicy.select_l1_retentions."""

    def test_returns_all_false(self):
        """Default policy marks all keys as temporary."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(5)]
        result = policy.select_l1_retentions(keys)
        assert result == [False] * 5

    def test_empty_keys(self):
        """Empty keys list returns empty list."""
        policy = DefaultPrefetchPolicy()
        result = policy.select_l1_retentions([])
        assert result == []

    def test_single_key(self):
        """Single key returns single False."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(0)]
        result = policy.select_l1_retentions(keys)
        assert result == [False]

    def test_length_matches_input(self):
        """Output length always matches input length."""
        policy = DefaultPrefetchPolicy()
        for n in [0, 1, 10, 100]:
            keys = [make_object_key(i) for i in range(n)]
            result = policy.select_l1_retentions(keys)
            assert len(result) == n


# =============================================================================
# RetainPrefetchPolicy Tests
# =============================================================================


class TestRetainPrefetchPolicyRetentions:
    """Test RetainPrefetchPolicy.select_l1_retentions."""

    def test_returns_all_true(self):
        """Retain policy marks all keys as permanent."""
        policy = RetainPrefetchPolicy()
        keys = [make_object_key(i) for i in range(5)]
        result = policy.select_l1_retentions(keys)
        assert result == [True] * 5

    def test_empty_keys(self):
        """Empty keys list returns empty list."""
        policy = RetainPrefetchPolicy()
        result = policy.select_l1_retentions([])
        assert result == []

    def test_single_key(self):
        """Single key returns single True."""
        policy = RetainPrefetchPolicy()
        keys = [make_object_key(0)]
        result = policy.select_l1_retentions(keys)
        assert result == [True]

    def test_length_matches_input(self):
        """Output length always matches input length."""
        policy = RetainPrefetchPolicy()
        for n in [0, 1, 10, 100]:
            keys = [make_object_key(i) for i in range(n)]
            result = policy.select_l1_retentions(keys)
            assert len(result) == n


class TestRetainPrefetchPolicyLoadPlan:
    """RetainPrefetchPolicy inherits load plan from Default."""

    def test_inherits_load_plan(self):
        """Load plan should behave identically to Default."""
        policy = RetainPrefetchPolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]
        lookup_results = {
            0: make_bitmap(3, [0, 1]),
            1: make_bitmap(3, [1, 2]),
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        assert plan_to_indices(result) == {0: [0, 1], 1: [2]}
