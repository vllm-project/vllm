# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for PiecewiseBackend._find_range_for_shape hotpath dispatch.

These tests use object.__new__ to construct a minimal PiecewiseBackend without
requiring a full VllmConfig, FX graph, or GPU — making them fast and CI-safe.
"""

import types

from vllm.compilation.piecewise_backend import PiecewiseBackend, RangeEntry
from vllm.config.utils import Range

# Upper bound for preallocated index mapping in tests.
_TEST_MAX_TOKENS = 10000


def _build_size_to_range_index(
    compile_sizes: list[int] | None,
    compile_ranges: list[Range],
    max_tokens: int = _TEST_MAX_TOKENS,
) -> tuple[list[int], int]:
    """Build a (size_to_range_index, num_range_entries) pair.

    Mirrors the eager mapping logic in VllmBackend.__init__.
    """
    next_idx = len(compile_ranges)
    extra_size_indices: dict[int, int] = {}
    if compile_sizes is not None:
        for s in compile_sizes:
            if isinstance(s, int):
                r = Range(start=s, end=s)
                if r not in compile_ranges and s not in extra_size_indices:
                    extra_size_indices[s] = next_idx
                    next_idx += 1

    num_range_entries = next_idx

    mapping: list[int] = [-1] * (max_tokens + 1)
    for i in range(len(compile_ranges) - 1, -1, -1):
        cr = compile_ranges[i]
        lo = max(0, cr.start)
        hi = min(max_tokens, cr.end)
        for size in range(lo, hi + 1):
            mapping[size] = i
    for s, idx in extra_size_indices.items():
        if 0 <= s <= max_tokens:
            mapping[s] = idx

    return mapping, num_range_entries


def _make_backend(
    compile_sizes: list[int] | None,
    compile_ranges: list[Range],
    vllm_backend: object | None = None,
) -> PiecewiseBackend:
    """
    Construct a PiecewiseBackend with only the attributes needed by
    _find_range_for_shape, mirroring the population logic in __init__.

    Pass a shared *vllm_backend* stub (e.g. a SimpleNamespace with
    _size_to_range_index and _num_range_entries) to exercise the
    cross-instance cache path.  When omitted, a local stub is created
    automatically.
    """
    b = object.__new__(PiecewiseBackend)
    b.compile_sizes = compile_sizes
    b.compile_ranges = compile_ranges

    # Auto-create a local index mapping when no shared backend is provided.
    if vllm_backend is None:
        mapping, num_entries = _build_size_to_range_index(compile_sizes, compile_ranges)
        vllm_backend = types.SimpleNamespace(
            _size_to_range_index=mapping,
            _num_range_entries=num_entries,
        )
    b.vllm_backend = vllm_backend

    # Mirror __init__ range_entries population
    b.range_entries = {}
    if compile_sizes is not None:
        for s in compile_sizes:
            if isinstance(s, int):
                r = Range(start=s, end=s)
                if r not in compile_ranges:
                    b.range_entries[r] = RangeEntry(compile_range=r)
    for r in compile_ranges:
        b.range_entries[r] = RangeEntry(compile_range=r)

    b._build_find_range_caches()
    return b


def test_exact_compile_size_returns_single_point_entry():
    """An exact compile size returns its own single-point RangeEntry."""
    b = _make_backend(compile_sizes=[128, 256, 512], compile_ranges=[Range(1, 1024)])
    entry = b._find_range_for_shape(256)
    assert entry is not None
    assert entry.compile_range == Range(256, 256)


def test_exact_compile_size_covered_by_compile_range():
    """When Range(s,s) equals a compile_range entry, dispatch still works."""
    b = _make_backend(
        compile_sizes=[64],
        compile_ranges=[Range(64, 64), Range(1, 1024)],
    )
    entry = b._find_range_for_shape(64)
    assert entry is not None
    assert entry.compile_range == Range(64, 64)


def test_non_exact_shape_falls_through_to_range():
    """A shape not in compile_sizes but inside a compile_range returns that entry."""
    b = _make_backend(compile_sizes=[512], compile_ranges=[Range(1, 1024)])
    entry = b._find_range_for_shape(300)
    assert entry is not None
    assert entry.compile_range == Range(1, 1024)


def test_shape_outside_all_ranges_returns_none():
    """A shape outside all compile_ranges returns None."""
    b = _make_backend(compile_sizes=[512], compile_ranges=[Range(1, 1024)])
    assert b._find_range_for_shape(2000) is None


def test_compile_sizes_none_returns_none():
    """When compile_sizes is None, _find_range_for_shape always returns None."""
    b = _make_backend(compile_sizes=None, compile_ranges=[Range(1, 2048)])
    assert b._find_range_for_shape(100) is None


def test_multiple_ranges_first_match_returned():
    """Overlapping ranges: the first matching range in compile_ranges is returned."""
    b = _make_backend(
        compile_sizes=[],
        compile_ranges=[Range(1, 100), Range(50, 200)],
    )
    entry = b._find_range_for_shape(75)
    assert entry is not None
    assert entry.compile_range == Range(1, 100)


def test_exact_size_entry_distinct_from_range_entry():
    """compile_size entry and compile_range entry are distinct RangeEntry objects."""
    b = _make_backend(compile_sizes=[256], compile_ranges=[Range(1, 1024)])
    exact_entry = b._find_range_for_shape(256)
    range_entry = b._find_range_for_shape(300)
    assert exact_entry is not None
    assert range_entry is not None
    assert exact_entry is not range_entry
    assert exact_entry.compile_range == Range(256, 256)
    assert range_entry.compile_range == Range(1, 1024)


def test_empty_compile_sizes_uses_range_only():
    """Empty compile_sizes list falls through to range scan for all shapes."""
    b = _make_backend(compile_sizes=[], compile_ranges=[Range(1, 512)])
    entry = b._find_range_for_shape(200)
    assert entry is not None
    assert entry.compile_range == Range(1, 512)


def test_multiple_non_overlapping_ranges():
    """Shape is dispatched to the correct non-overlapping range."""
    b = _make_backend(
        compile_sizes=[],
        compile_ranges=[Range(1, 8), Range(9, 64), Range(65, 512)],
    )
    entry_5 = b._find_range_for_shape(5)
    assert entry_5 is not None
    assert entry_5.compile_range == Range(1, 8)
    entry_32 = b._find_range_for_shape(32)
    assert entry_32 is not None
    assert entry_32.compile_range == Range(9, 64)
    entry_200 = b._find_range_for_shape(200)
    assert entry_200 is not None
    assert entry_200.compile_range == Range(65, 512)
    assert b._find_range_for_shape(1000) is None


def test_shared_dispatch_cache_across_instances():
    """All instances sharing the same vllm_backend share one index mapping.

    Simulates the Llama3-70B scenario where 81 subgraph PiecewiseBackend
    instances share a single VllmBackend.  Both backends use the same shared
    _size_to_range_index list but have independent _range_index_to_entry arrays.
    """
    ranges = [Range(1, 8), Range(9, 64), Range(65, 512)]

    mapping, num_entries = _build_size_to_range_index(
        compile_sizes=[], compile_ranges=ranges
    )
    fake_backend = types.SimpleNamespace(
        _size_to_range_index=mapping,
        _num_range_entries=num_entries,
    )

    # Build two independent PiecewiseBackend instances that share fake_backend.
    b1 = _make_backend(
        compile_sizes=[], compile_ranges=ranges, vllm_backend=fake_backend
    )
    b2 = _make_backend(
        compile_sizes=[], compile_ranges=ranges, vllm_backend=fake_backend
    )

    # Both must reference the SAME underlying index list.
    assert b1._size_to_range_index is b2._size_to_range_index

    # But each has its own entry array (different RangeEntry objects).
    assert b1._range_index_to_entry is not b2._range_index_to_entry

    # Both dispatch shape 32 to the Range(9, 64) entry.
    entry1 = b1._find_range_for_shape(32)
    assert entry1 is not None
    assert entry1.compile_range == Range(9, 64)

    entry2 = b2._find_range_for_shape(32)
    assert entry2 is not None
    assert entry2.compile_range == Range(9, 64)

    # A shape that matches no range returns None.
    assert b1._find_range_for_shape(9999) is None
    assert b2._find_range_for_shape(9999) is None
