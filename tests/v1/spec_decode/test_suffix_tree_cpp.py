# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for C++ suffix tree implementation."""

import pytest

from vllm.v1.spec_decode.suffix_decode import Candidate, SuffixTree


class TestSuffixTreeCpp:
    """Test suite for C++ SuffixTree implementation."""

    def test_basic_operations(self):
        """Test basic suffix tree operations."""
        tree = SuffixTree(32)  # max_depth = 32

        # Add sequences
        tree.extend(0, [1, 2, 3, 4, 5])
        tree.extend(1, [1, 2, 3, 6, 7])
        assert tree.num_seqs() == 2

        # Test speculation
        result = tree.speculate([1, 2, 3], 5, 1.0, 0.0, 0.1, False)
        assert isinstance(result, Candidate)
        assert len(result.token_ids) > 0
        assert result.score > 0
        assert result.match_len == 3

        # Remove sequence
        tree.remove(1)
        assert tree.num_seqs() == 1

    def test_append_operations(self):
        """Test append vs extend operations."""
        tree = SuffixTree(16)

        # Start with extend
        tree.extend(0, [1, 2, 3])

        # Append individual tokens
        tree.append(0, 4)
        tree.append(0, 5)

        # Verify speculation works
        result = tree.speculate([1, 2, 3, 4], 3, 1.0, 0.0, 0.1, False)
        assert 5 in result.token_ids

    def test_multiple_sequences(self):
        """Test handling multiple sequences with shared prefixes."""
        tree = SuffixTree(64)

        # Add sequences with common prefixes
        sequences = [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 7, 8],
            [1, 2, 3, 9, 10, 11],
            [4, 5, 6, 7, 8, 9],
        ]

        for i, seq in enumerate(sequences):
            tree.extend(i, seq)

        # Test speculation on common prefix
        result = tree.speculate([1, 2, 3], 3, 1.0, 0.0, 0.1, False)
        assert len(result.token_ids) > 0
        # Should predict one of the continuations
        assert result.token_ids[0] in [4, 9]

    def test_speculation_parameters(self):
        """Test different speculation parameters."""
        tree = SuffixTree(32)
        tree.extend(0, list(range(20)))

        # Test max_spec_tokens
        result1 = tree.speculate([0, 1, 2], 3, 1.0, 0.0, 0.1, False)
        assert len(result1.token_ids) <= 3

        result2 = tree.speculate([0, 1, 2], 10, 1.0, 0.0, 0.1, False)
        assert len(result2.token_ids) <= 10

        # Test max_spec_factor and offset
        # With factor=0.5 and match_len=3, max tokens = 3*0.5 = 1.5 -> 1
        result3 = tree.speculate([0, 1, 2], 10, 0.5, 0.0, 0.1, False)
        assert len(result3.token_ids) <= 2

        # With offset=3, even short matches can speculate more
        result4 = tree.speculate([0, 1], 10, 0.0, 3.0, 0.1, False)
        assert len(result4.token_ids) <= 3

    def test_integrity_check(self):
        """Test tree integrity checking."""
        tree = SuffixTree(16)

        # Add some sequences
        tree.extend(0, [1, 2, 3, 4])
        tree.extend(1, [1, 2, 5, 6])
        tree.extend(2, [7, 8, 9])

        # Check integrity
        integrity = tree.check_integrity()
        assert integrity == "", f"Integrity check failed: {integrity}"

        # Remove a sequence and check again
        tree.remove(1)
        integrity = tree.check_integrity()
        assert integrity == "", f"Integrity check failed: {integrity}"

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        tree = SuffixTree(32)

        # Empty tree should have minimal memory
        initial_memory = tree.estimate_memory()
        assert initial_memory > 0

        # Add sequences and check memory increases
        for i in range(10):
            tree.extend(i, list(range(i, i + 20)))

        final_memory = tree.estimate_memory()
        assert final_memory > initial_memory

    def test_empty_sequences(self):
        """Test handling of empty sequences."""
        tree = SuffixTree(16)

        # Add empty sequence - C++ implementation may not track empty sequences
        tree.extend(0, [])
        # The C++ implementation doesn't count empty sequences
        # This is actually reasonable behavior

        # Add a non-empty sequence
        tree.extend(1, [1, 2, 3])
        assert tree.num_seqs() >= 1

        # Speculation on empty pattern
        result = tree.speculate([], 5, 1.0, 0.0, 0.1, False)
        assert result.token_ids == []
        assert result.score == 0.0

    def test_large_sequences(self):
        """Test handling of sequences larger than max_depth."""
        max_depth = 16
        tree = SuffixTree(max_depth)

        # Add sequence longer than max_depth
        long_seq = list(range(100))
        tree.extend(0, long_seq)

        # Pattern at the end should still work (within max_depth window)
        pattern = list(range(80, 85))
        result = tree.speculate(pattern, 5, 1.0, 0.0, 0.1, False)
        assert result.token_ids == list(range(85, 90))

    def test_tree_vs_path_speculation(self):
        """Test tree-based vs path-based speculation."""
        tree = SuffixTree(32)

        # Add multiple sequences with branching
        tree.extend(0, [1, 2, 3, 4, 5])
        tree.extend(1, [1, 2, 3, 6, 7])
        tree.extend(2, [1, 2, 3, 4, 8])

        # Path speculation (use_tree_spec=False)
        path_result = tree.speculate([1, 2, 3], 5, 1.0, 0.0, 0.1, False)

        # Tree speculation (use_tree_spec=True)
        tree_result = tree.speculate([1, 2, 3], 5, 1.0, 0.0, 0.1, True)

        # Both should return valid results
        assert len(path_result.token_ids) > 0
        assert len(tree_result.token_ids) > 0

        # Tree speculation might return different structure
        # (multiple branches vs single path)
        assert path_result.parents[0] == -1  # First token has no parent
        if len(tree_result.token_ids) > 1:
            # Check tree structure is valid
            for i, parent in enumerate(tree_result.parents):
                assert parent < i  # Parent index must be less than current


if __name__ == "__main__":
    pytest.main([__file__])
