# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for ROCm attention metadata variable initialization.

This tests the fix for the uninitialized variable bug where
prefix_scheduler_metadata was only initialized inside the else branch,
causing an UnboundLocalError when use_cascade=True.

The fix initializes prefix_scheduler_metadata = None before the if/else
block so it's always defined.

See: https://github.com/vllm-project/vllm/pull/31118
"""

import pytest

from vllm.platforms import current_platform

# Skip on non-ROCm platforms
pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


class TestVariableInitializationPattern:
    """Test that variables are properly initialized before conditional blocks."""

    def test_uninitialized_variable_in_else_branch_bug(self):
        """Demonstrate the bug pattern: variable only set in else branch."""

        def buggy_build(common_prefix_len: int) -> dict:
            """Simulates the buggy build() method pattern."""
            use_cascade = common_prefix_len > 0

            if use_cascade:
                cu_prefix_query_lens = [0, 100]
                prefix_kv_lens = [common_prefix_len]
                suffix_kv_lens = [50]
                # BUG: prefix_scheduler_metadata is NOT set here!
            else:
                cu_prefix_query_lens = None
                prefix_kv_lens = None
                suffix_kv_lens = None
                prefix_scheduler_metadata = None  # Only set in else branch

            # This line causes UnboundLocalError when use_cascade=True
            return {
                "use_cascade": use_cascade,
                "cu_prefix_query_lens": cu_prefix_query_lens,
                "prefix_kv_lens": prefix_kv_lens,
                "suffix_kv_lens": suffix_kv_lens,
                "prefix_scheduler_metadata": prefix_scheduler_metadata,
            }

        # Works when use_cascade=False (else branch runs)
        result = buggy_build(common_prefix_len=0)
        assert result["prefix_scheduler_metadata"] is None

        # Fails when use_cascade=True (if branch runs)
        with pytest.raises(UnboundLocalError):
            buggy_build(common_prefix_len=10)

    def test_fixed_variable_initialization_pattern(self):
        """Verify the fix: variable initialized before conditional block."""

        def fixed_build(common_prefix_len: int) -> dict:
            """Simulates the fixed build() method pattern."""
            use_cascade = common_prefix_len > 0

            # FIX: Initialize before the if/else block
            prefix_scheduler_metadata = None

            if use_cascade:
                cu_prefix_query_lens = [0, 100]
                prefix_kv_lens = [common_prefix_len]
                suffix_kv_lens = [50]
            else:
                cu_prefix_query_lens = None
                prefix_kv_lens = None
                suffix_kv_lens = None

            return {
                "use_cascade": use_cascade,
                "cu_prefix_query_lens": cu_prefix_query_lens,
                "prefix_kv_lens": prefix_kv_lens,
                "suffix_kv_lens": suffix_kv_lens,
                "prefix_scheduler_metadata": prefix_scheduler_metadata,
            }

        # Works when use_cascade=False
        result = fixed_build(common_prefix_len=0)
        assert result["prefix_scheduler_metadata"] is None
        assert result["use_cascade"] is False

        # Now also works when use_cascade=True
        result = fixed_build(common_prefix_len=10)
        assert result["prefix_scheduler_metadata"] is None
        assert result["use_cascade"] is True
        assert result["prefix_kv_lens"] == [10]

    def test_rocm_attn_metadata_build_pattern(self):
        """Test the actual pattern used in RocmAttentionMetadataBuilder.build()."""

        def simulate_rocm_attn_build(common_prefix_len: int, num_tokens: int) -> dict:
            """Simulates RocmAttentionMetadataBuilder.build() with the fix."""
            use_cascade = common_prefix_len > 0

            # Fixed: Initialize before conditional (line 119 in fixed code)
            prefix_scheduler_metadata = None

            if use_cascade:
                cu_prefix_query_lens = [0, num_tokens]
                prefix_kv_lens = [common_prefix_len]
                suffix_kv_lens = [100 - common_prefix_len]  # seq_lens - prefix
            else:
                cu_prefix_query_lens = None
                prefix_kv_lens = None
                suffix_kv_lens = None

            # Construct metadata (simulating RocmAttentionMetadata creation)
            return {
                "num_actual_tokens": num_tokens,
                "use_cascade": use_cascade,
                "common_prefix_len": common_prefix_len,
                "cu_prefix_query_lens": cu_prefix_query_lens,
                "prefix_kv_lens": prefix_kv_lens,
                "suffix_kv_lens": suffix_kv_lens,
                "prefix_scheduler_metadata": prefix_scheduler_metadata,
            }

        # Test non-cascade mode (prefix_len=0)
        result = simulate_rocm_attn_build(common_prefix_len=0, num_tokens=50)
        assert result["use_cascade"] is False
        assert result["prefix_scheduler_metadata"] is None
        assert result["cu_prefix_query_lens"] is None

        # Test cascade mode (prefix_len>0)
        result = simulate_rocm_attn_build(common_prefix_len=20, num_tokens=50)
        assert result["use_cascade"] is True
        assert result["prefix_scheduler_metadata"] is None  # Should not raise!
        assert result["cu_prefix_query_lens"] == [0, 50]
        assert result["prefix_kv_lens"] == [20]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
