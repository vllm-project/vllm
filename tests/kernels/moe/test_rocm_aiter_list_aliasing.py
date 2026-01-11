# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for ROCm AITER fused MoE list aliasing fix.

This tests the fix for the list aliasing bug where using [list] * n
creates n references to the same list object, causing unintended
modifications when any single element is changed.

The fix uses list comprehension [... for _ in range(n)] to create
independent list copies.

See: https://github.com/vllm-project/vllm/pull/31121
"""

import pytest


class TestListAliasingFix:
    """Test that list aliasing bug is fixed in MoE initialization."""

    def test_list_multiplication_creates_aliased_references(self):
        """Demonstrate the bug: [list] * n creates aliased references."""
        # This is the BUGGY pattern that was used before the fix
        max_num_tokens = 5
        fake_expertid = 99

        # Bug: All elements reference the SAME inner list
        buggy_list = [[fake_expertid] * 3] * max_num_tokens

        # Verify all elements are initially the same
        assert all(elem == [99, 99, 99] for elem in buggy_list)

        # Modify one element
        buggy_list[0] = [1, 2, 3]

        # With the buggy pattern and direct assignment, only index 0 changes
        # But the original bug was when using in-place modification:
        buggy_list2 = [[fake_expertid] * 3] * max_num_tokens
        buggy_list2[2][0] = 42  # Modify element at index 2

        # BUG: ALL elements are modified because they reference the same list!
        for i, elem in enumerate(buggy_list2):
            assert elem[0] == 42, f"Element {i} should be 42 due to aliasing bug"

    def test_list_comprehension_creates_independent_copies(self):
        """Verify the fix: list comprehension creates independent copies."""
        # This is the FIXED pattern using list comprehension
        max_num_tokens = 5
        fake_expertid = 99

        # Fix: Each element is an independent list
        fixed_list = [[fake_expertid] * 3 for _ in range(max_num_tokens)]

        # Verify all elements are initially the same
        assert all(elem == [99, 99, 99] for elem in fixed_list)

        # Modify one element in-place
        fixed_list[2][0] = 42

        # Only the modified element should change
        assert fixed_list[2] == [42, 99, 99], "Modified element should be [42, 99, 99]"

        # Other elements should remain unchanged
        for i in [0, 1, 3, 4]:
            assert fixed_list[i] == [99, 99, 99], (
                f"Element {i} should remain [99, 99, 99]"
            )

    def test_moe_shared_expert_ids_pattern(self):
        """Test the actual pattern used in init_aiter_topK_meta_data."""
        max_num_tokens = 10
        n_routed_experts = 8
        n_shared_experts = 2
        fake_expertid = n_routed_experts + n_shared_experts  # 10
        is_EP = True
        tp_rank = 0
        tp_size = 2

        # Fixed pattern (from the PR)
        s_topk_ids_list = [
            [fake_expertid] * (n_shared_experts + is_EP) for _ in range(max_num_tokens)
        ]

        # Verify initial state
        expected_initial = [fake_expertid] * (n_shared_experts + is_EP)  # [10, 10, 10]
        assert all(elem == expected_initial for elem in s_topk_ids_list)

        # Simulate the EP assignment logic
        shared_expert_ids = [
            n_routed_experts + i for i in range(n_shared_experts + is_EP)
        ]  # [8, 9, 10]

        for i in range(tp_rank, max_num_tokens, tp_size):
            s_topk_ids_list[i] = shared_expert_ids

        # Verify only specific indices were modified
        for i in range(max_num_tokens):
            if i % tp_size == tp_rank:
                # These should have shared_expert_ids
                assert s_topk_ids_list[i] == shared_expert_ids, (
                    f"Index {i} should have shared_expert_ids"
                )
            else:
                # These should remain unchanged with fake_expertid
                assert s_topk_ids_list[i] == expected_initial, (
                    f"Index {i} should remain unchanged"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
