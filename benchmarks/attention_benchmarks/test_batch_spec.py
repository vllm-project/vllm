#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test suite for batch specification parser."""

import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from batch_spec import (
    format_batch_spec,
    get_batch_stats,
    parse_batch_spec,
)
from benchmark import generate_batch_specs_from_ranges


def test_basic_patterns():
    """Test basic batch specification patterns."""
    print("Testing basic patterns...")

    # Prefill
    result = parse_batch_spec("q2k")
    assert len(result) == 1
    assert result[0].q_len == 2048
    assert result[0].kv_len == 2048
    assert result[0].is_prefill
    print("  ✓ q2k -> [(2048, 2048)]")

    # Decode
    result = parse_batch_spec("8q1s1k")
    assert len(result) == 8
    assert all(r.q_len == 1 and r.kv_len == 1024 for r in result)
    assert all(r.is_decode for r in result)
    print("  ✓ 8q1s1k -> 8 x [(1, 1024)]")

    # Context extension
    result = parse_batch_spec("q1ks2k")
    assert len(result) == 1
    assert result[0].q_len == 1024
    assert result[0].kv_len == 2048
    assert result[0].is_extend
    print("  ✓ q1ks2k -> [(1024, 2048)]")


def test_combined_patterns():
    """Test combined batch specifications."""
    print("\nTesting combined patterns...")

    result = parse_batch_spec("2q1k_32q1s1k")
    assert len(result) == 34
    assert sum(1 for r in result if r.is_prefill) == 2
    assert sum(1 for r in result if r.is_decode) == 32
    print("  ✓ 2q1k_32q1s1k -> 2 prefill + 32 decode")

    result = parse_batch_spec("4q2k_8q4s1k_64q1s2k")
    assert len(result) == 76  # 4 + 8 + 64
    print("  ✓ 4q2k_8q4s1k_64q1s2k -> complex mix")


def test_extend_patterns():
    """Test context extension (extend) patterns."""
    print("\nTesting extend patterns...")

    # 4-token extension with 1k context
    result = parse_batch_spec("q4s1k")
    assert len(result) == 1
    assert result[0].q_len == 4
    assert result[0].kv_len == 1024
    assert result[0].is_extend
    assert not result[0].is_decode
    assert not result[0].is_prefill
    print("  ✓ q4s1k -> 4-token extend with 1k context")

    # 8 requests of 8-token extension
    result = parse_batch_spec("8q8s2k")
    assert len(result) == 8
    assert all(r.q_len == 8 and r.kv_len == 2048 for r in result)
    assert all(r.is_extend for r in result)
    print("  ✓ 8q8s2k -> 8 x 8-token extend with 2k context")


def test_formatting():
    """Test batch spec formatting."""
    print("\nTesting formatting...")

    requests = parse_batch_spec("2q2k_32q1s1k")
    formatted = format_batch_spec(requests)
    assert "2 prefill" in formatted
    assert "32 decode" in formatted
    print(f"  ✓ Format: {formatted}")

    requests = parse_batch_spec("q4s1k_8q1s1k")
    formatted = format_batch_spec(requests)
    assert "1 extend" in formatted
    assert "8 decode" in formatted
    print(f"  ✓ Format with extend: {formatted}")


def test_batch_stats():
    """Test batch statistics."""
    print("\nTesting batch statistics...")

    requests = parse_batch_spec("2q2k_32q1s1k")
    stats = get_batch_stats(requests)

    assert stats["total_requests"] == 34
    assert stats["num_prefill"] == 2
    assert stats["num_decode"] == 32
    assert stats["total_tokens"] == 2048 * 2 + 32 * 1
    print(
        f"  ✓ Stats: {stats['total_requests']} requests, {stats['total_tokens']} tokens"
    )


def test_error_handling():
    """Test error handling."""
    print("\nTesting error handling...")

    try:
        parse_batch_spec("invalid")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        print("  ✓ Invalid spec raises ValueError")


def test_range_generation_simple():
    """Test simple range generation."""
    print("\nTesting range generation (simple)...")

    ranges = [{"template": "q{q_len}ks1k", "q_len": {"start": 1, "stop": 5, "step": 1}}]
    specs = generate_batch_specs_from_ranges(ranges)
    expected = ["q1ks1k", "q2ks1k", "q3ks1k", "q4ks1k", "q5ks1k"]
    assert specs == expected, f"Expected {expected}, got {specs}"
    print(f"  ✓ Simple range: {len(specs)} specs generated")


def test_range_generation_multiple():
    """Test multiple range specifications."""
    print("\nTesting range generation (multiple ranges)...")

    ranges = [
        {"template": "q{q_len}ks1k", "q_len": {"start": 1, "stop": 3, "step": 1}},
        {"template": "q{q_len}ks1k", "q_len": {"start": 10, "stop": 20, "step": 5}},
    ]
    specs = generate_batch_specs_from_ranges(ranges)
    expected = ["q1ks1k", "q2ks1k", "q3ks1k", "q10ks1k", "q15ks1k", "q20ks1k"]
    assert specs == expected, f"Expected {expected}, got {specs}"
    print(f"  ✓ Multiple ranges: {len(specs)} specs generated")


def test_range_generation_large():
    """Test large range similar to study4 config."""
    print("\nTesting range generation (large range)...")

    ranges = [
        {"template": "q{q_len}ks1k", "q_len": {"start": 1, "stop": 16, "step": 1}},
        {"template": "q{q_len}ks1k", "q_len": {"start": 17, "stop": 64, "step": 2}},
        {"template": "q{q_len}ks1k", "q_len": {"start": 65, "stop": 128, "step": 4}},
    ]
    specs = generate_batch_specs_from_ranges(ranges)
    expected_count = 16 + 24 + 16  # (1-16) + (17,19,21...63) + (65,69,73...125)
    assert len(specs) == expected_count, (
        f"Expected {expected_count} specs, got {len(specs)}"
    )
    print(f"  ✓ Large range: {len(specs)} specs generated")


def test_range_generation_cartesian():
    """Test Cartesian product with multiple parameters."""
    print("\nTesting range generation (Cartesian product)...")

    ranges = [
        {
            "template": "q{q_len}ks{kv_len}k",
            "q_len": {"start": 1, "stop": 2, "step": 1},
            "kv_len": {"start": 1, "stop": 2, "step": 1},
        }
    ]
    specs = generate_batch_specs_from_ranges(ranges)
    # Should generate Cartesian product: (1,1), (1,2), (2,1), (2,2)
    expected = ["q1ks1k", "q1ks2k", "q2ks1k", "q2ks2k"]
    assert specs == expected, f"Expected {expected}, got {specs}"
    print(f"  ✓ Cartesian product: {len(specs)} specs generated")


def test_range_generation_end_inclusive():
    """Test end_inclusive parameter."""
    print("\nTesting range generation (end_inclusive)...")

    # Test inclusive (default)
    ranges_inclusive = [
        {"template": "q{q_len}ks1k", "q_len": {"start": 1, "stop": 3, "step": 1}}
    ]
    specs = generate_batch_specs_from_ranges(ranges_inclusive)
    expected = ["q1ks1k", "q2ks1k", "q3ks1k"]
    assert specs == expected, f"Expected {expected}, got {specs}"
    print(f"  ✓ end_inclusive default (true): {specs}")

    # Test explicit inclusive
    ranges_explicit_inclusive = [
        {
            "template": "q{q_len}ks1k",
            "q_len": {"start": 1, "stop": 5, "step": 1, "end_inclusive": True},
        }
    ]
    specs = generate_batch_specs_from_ranges(ranges_explicit_inclusive)
    expected = ["q1ks1k", "q2ks1k", "q3ks1k", "q4ks1k", "q5ks1k"]
    assert specs == expected, f"Expected {expected}, got {specs}"
    print("  ✓ end_inclusive=true: includes stop value")

    # Test exclusive
    ranges_exclusive = [
        {
            "template": "q{q_len}ks1k",
            "q_len": {"start": 1, "stop": 5, "step": 1, "end_inclusive": False},
        }
    ]
    specs = generate_batch_specs_from_ranges(ranges_exclusive)
    expected = ["q1ks1k", "q2ks1k", "q3ks1k", "q4ks1k"]
    assert specs == expected, f"Expected {expected}, got {specs}"
    print("  ✓ end_inclusive=false: excludes stop value")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Batch Specification Parser Tests")
    print("=" * 60)

    test_basic_patterns()
    test_combined_patterns()
    test_extend_patterns()
    test_formatting()
    test_batch_stats()
    test_error_handling()
    test_range_generation_simple()
    test_range_generation_multiple()
    test_range_generation_large()
    test_range_generation_cartesian()
    test_range_generation_end_inclusive()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
