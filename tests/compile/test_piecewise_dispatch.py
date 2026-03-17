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


def _make_backend(
    compile_sizes: list[int] | None,
    compile_ranges: list[Range],
    vllm_backend: object | None = None,
) -> PiecewiseBackend:
    """
    Construct a PiecewiseBackend with only the attributes needed by
    _find_range_for_shape, mirroring the population logic in __init__.

    Pass a shared *vllm_backend* stub (e.g. a SimpleNamespace with
    _shape_dispatch_cache) to exercise the cross-instance cache path.
    """
    b = object.__new__(PiecewiseBackend)
    b.compile_sizes = compile_sizes
    b.compile_ranges = compile_ranges
    if vllm_backend is not None:
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
    """All instances sharing the same vllm_backend share one dispatch cache.

    Simulates the Llama3-70B scenario where 81 subgraph PiecewiseBackend
    instances share a single VllmBackend.  The first instance to look up a
    shape pays the O(#ranges) scan; subsequent ones get an O(1) dict hit.
    """
    # A minimal stub for VllmBackend — only needs _shape_dispatch_cache.
    fake_backend = types.SimpleNamespace(_shape_dispatch_cache={})
    ranges = [Range(1, 8), Range(9, 64), Range(65, 512)]

    # Build two independent PiecewiseBackend instances that share fake_backend.
    b1 = _make_backend(
        compile_sizes=[], compile_ranges=ranges, vllm_backend=fake_backend
    )
    b2 = _make_backend(
        compile_sizes=[], compile_ranges=ranges, vllm_backend=fake_backend
    )

    # Both must reference the SAME underlying dict.
    assert b1._shape_dispatch_cache is b2._shape_dispatch_cache

    # b1 performs the range scan and populates the shared cache.
    entry1 = b1._find_range_for_shape(32)
    assert entry1 is not None
    assert entry1.compile_range == Range(9, 64)
    assert 32 in fake_backend._shape_dispatch_cache  # cache was written
    assert fake_backend._shape_dispatch_cache[32] == Range(9, 64)

    # b2 sees the same shape: it gets an O(1) hit, no scan.
    entry2 = b2._find_range_for_shape(32)
    assert entry2 is not None
    assert entry2.compile_range == Range(9, 64)

    # A shape that matches no range is also cached (as None) after first scan.
    none_entry = b1._find_range_for_shape(9999)
    assert none_entry is None
    assert 9999 in fake_backend._shape_dispatch_cache
    assert fake_backend._shape_dispatch_cache[9999] is None
    assert b2._find_range_for_shape(9999) is None
