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
    parse_manual_batch,
)


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
    result = parse_batch_spec("8s1k")
    assert len(result) == 8
    assert all(r.q_len == 1 and r.kv_len == 1024 for r in result)
    assert all(r.is_decode for r in result)
    print("  ✓ 8s1k -> 8 x [(1, 1024)]")

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

    result = parse_batch_spec("2q1k_32s1k")
    assert len(result) == 34
    assert sum(1 for r in result if r.is_prefill) == 2
    assert sum(1 for r in result if r.is_decode) == 32
    print("  ✓ 2q1k_32s1k -> 2 prefill + 32 decode")

    result = parse_batch_spec("4q2k_spec8s1k_64s2k")
    assert len(result) == 69
    print("  ✓ 4q2k_spec8s1k_64s2k -> complex mix")


def test_speculative_decode():
    """Test speculative decode patterns."""
    print("\nTesting speculative decode...")

    result = parse_batch_spec("spec4s1k")
    assert len(result) == 1
    assert result[0].q_len == 4
    assert result[0].kv_len == 1024
    assert result[0].is_speculative
    assert result[0].spec_length == 4
    print("  ✓ spec4s1k -> 4-token speculative")

    result = parse_batch_spec("8spec8s2k")
    assert len(result) == 8
    assert all(r.is_speculative and r.spec_length == 8 for r in result)
    print("  ✓ 8spec8s2k -> 8 x 8-token speculative")


def test_chunked_prefill():
    """Test chunked prefill patterns."""
    print("\nTesting chunked prefill...")

    result = parse_batch_spec("chunk8q16k")
    assert len(result) == 1
    assert result[0].q_len == 16384
    assert result[0].is_chunked
    assert result[0].chunk_size == 8
    print("  ✓ chunk8q16k -> chunked 16k prefill")

    result = parse_batch_spec("2chunk4q8k")
    assert len(result) == 2
    assert all(r.is_chunked and r.chunk_size == 4 for r in result)
    print("  ✓ 2chunk4q8k -> 2 x chunked 8k prefill")


def test_formatting():
    """Test batch spec formatting."""
    print("\nTesting formatting...")

    requests = parse_batch_spec("2q2k_32s1k")
    formatted = format_batch_spec(requests)
    assert "2 prefill" in formatted
    assert "32 decode" in formatted
    print(f"  ✓ Format: {formatted}")

    requests = parse_batch_spec("spec4s1k_8s1k")
    formatted = format_batch_spec(requests)
    assert "specdecode" in formatted
    print(f"  ✓ Format with spec: {formatted}")


def test_batch_stats():
    """Test batch statistics."""
    print("\nTesting batch statistics...")

    requests = parse_batch_spec("2q2k_32s1k")
    stats = get_batch_stats(requests)

    assert stats["total_requests"] == 34
    assert stats["num_prefill"] == 2
    assert stats["num_decode"] == 32
    assert stats["total_tokens"] == 2048 * 2 + 32 * 1
    print(
        f"  ✓ Stats: {stats['total_requests']} requests, {stats['total_tokens']} tokens"
    )


def test_manual_batch():
    """Test manual batch specification."""
    print("\nTesting manual batch...")

    requests = parse_manual_batch(["1,1024", "2048,2048", "1,2048"])
    assert len(requests) == 3
    assert requests[0].as_tuple() == (1, 1024)
    assert requests[1].as_tuple() == (2048, 2048)
    assert requests[2].as_tuple() == (1, 2048)
    print("  ✓ Manual batch: 3 requests")


def test_error_handling():
    """Test error handling."""
    print("\nTesting error handling...")

    try:
        parse_batch_spec("invalid")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        print("  ✓ Invalid spec raises ValueError")

    try:
        parse_manual_batch(["1024,512"])  # kv < q
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        print("  ✓ Invalid kv_len raises ValueError")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Batch Specification Parser Tests")
    print("=" * 60)

    test_basic_patterns()
    test_combined_patterns()
    test_speculative_decode()
    test_chunked_prefill()
    test_formatting()
    test_batch_stats()
    test_manual_batch()
    test_error_handling()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
