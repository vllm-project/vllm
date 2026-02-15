# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Helix configuration (no GPU required).

Tests cover:
1. Helix configuration validation and derived properties
2. Helix parallel state functions
3. Helix LSE-weighted combination logic
4. Backend compatibility (known limitations)
"""

import pytest

from vllm.config.parallel import ParallelConfig


class TestHelixConfig:
    """Test Helix configuration validation and derived properties."""

    def test_helix_mode_disabled_by_default(self):
        """Helix mode should be disabled by default."""
        config = ParallelConfig()
        assert config.helix_mode is False
        assert config.helix_kvp_size == 1
        assert config.helix_tpa_size == 1

    def test_helix_mode_requires_dcp(self):
        """Helix mode requires decode_context_parallel_size > 1."""
        with pytest.raises(
            ValueError, match="requires decode_context_parallel_size > 1"
        ):
            ParallelConfig(
                helix_mode=True,
                tensor_parallel_size=4,
                decode_context_parallel_size=1,
            )

    def test_helix_mode_requires_divisible_tp(self):
        """TP must be divisible by DCP when Helix is enabled."""
        with pytest.raises(ValueError, match="must be divisible by"):
            ParallelConfig(
                helix_mode=True,
                tensor_parallel_size=6,
                decode_context_parallel_size=4,
            )

    def test_helix_derived_properties_tp4_dcp2(self):
        """Test derived properties: TP=4, DCP=2 -> TPA=2, KVP=2."""
        config = ParallelConfig(
            helix_mode=True,
            tensor_parallel_size=4,
            decode_context_parallel_size=2,
        )
        assert config.helix_kvp_size == 2
        assert config.helix_tpa_size == 2

    def test_helix_derived_properties_tp8_dcp4(self):
        """Test derived properties: TP=8, DCP=4 -> TPA=2, KVP=4."""
        config = ParallelConfig(
            helix_mode=True,
            tensor_parallel_size=8,
            decode_context_parallel_size=4,
        )
        assert config.helix_kvp_size == 4
        assert config.helix_tpa_size == 2

    def test_helix_derived_properties_tp8_dcp8(self):
        """Test derived properties: TP=8, DCP=8 -> TPA=1, KVP=8 (MLA case)."""
        config = ParallelConfig(
            helix_mode=True,
            tensor_parallel_size=8,
            decode_context_parallel_size=8,
        )
        assert config.helix_kvp_size == 8
        assert config.helix_tpa_size == 1

    def test_helix_disabled_properties_unchanged(self):
        """When helix_mode=False, DCP doesn't affect helix properties."""
        config = ParallelConfig(
            helix_mode=False,
            tensor_parallel_size=8,
            decode_context_parallel_size=4,
        )
        assert config.helix_kvp_size == 1  # Not using Helix
        assert config.helix_tpa_size == 8  # Full TP


class TestHelixParallelState:
    """Test Helix parallel state functions exist."""

    def test_helix_kvp_group_importable(self):
        """Verify get_helix_kvp_group is importable."""
        from vllm.distributed.parallel_state import get_helix_kvp_group

        assert callable(get_helix_kvp_group)


class TestHelixLSECombine:
    """Test Helix LSE-weighted combination logic (no GPU required)."""

    def test_lse_weighted_combine_importable(self):
        """Verify _lse_weighted_combine is importable."""
        from vllm.v1.attention.ops.helix import _lse_weighted_combine

        assert callable(_lse_weighted_combine)

    def test_lse_weighted_combine_single_rank(self):
        """Single rank case: output unchanged."""
        import torch

        from vllm.v1.attention.ops.helix import _lse_weighted_combine

        # Single rank: N=1, B=2, H=4, D=8
        outputs = torch.randn(1, 2, 4, 8)
        lses = torch.randn(1, 2, 4)

        result = _lse_weighted_combine(outputs, lses)

        # With single rank, weight is 1.0, so output should be unchanged
        assert result.shape == (2, 4, 8)
        torch.testing.assert_close(result, outputs.squeeze(0), rtol=1e-5, atol=1e-5)

    def test_lse_weighted_combine_equal_lse(self):
        """Equal LSE values: outputs averaged equally."""
        import torch

        from vllm.v1.attention.ops.helix import _lse_weighted_combine

        # Two ranks with equal LSE -> equal weights -> average
        _N, B, H, D = 2, 1, 1, 4
        outputs = torch.tensor(
            [
                [[[1.0, 2.0, 3.0, 4.0]]],  # Rank 0
                [[[5.0, 6.0, 7.0, 8.0]]],  # Rank 1
            ]
        )
        lses = torch.tensor(
            [
                [[0.0]],  # Rank 0: lse = 0
                [[0.0]],  # Rank 1: lse = 0
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        # Equal LSE -> weights are 0.5, 0.5 -> average
        expected = (outputs[0] + outputs[1]) / 2  # Shape: (B, H, D) = (1, 1, 4)
        assert result.shape == (B, H, D)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_lse_weighted_combine_different_lse(self):
        """Different LSE values: larger LSE gets more weight."""
        import torch

        from vllm.v1.attention.ops.helix import _lse_weighted_combine

        B, H, D = 1, 1, 2
        outputs = torch.tensor(
            [
                [[[0.0, 0.0]]],  # Rank 0: output is 0
                [[[1.0, 1.0]]],  # Rank 1: output is 1
            ]
        )
        # Large difference in LSE: rank 1 dominates (N=2 ranks)
        lses = torch.tensor(
            [
                [[-100.0]],  # Rank 0: very small contribution
                [[0.0]],  # Rank 1: dominant contribution
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        # Rank 1 should dominate, result close to [1, 1]
        assert result.shape == (B, H, D)
        assert torch.allclose(result, outputs[1].squeeze(0), atol=1e-5)

    def test_lse_weighted_combine_mathematically_correct(self):
        """Verify mathematical correctness of LSE combination."""
        import torch

        from vllm.v1.attention.ops.helix import _lse_weighted_combine

        # Manual computation to verify (N=2 ranks)
        outputs = torch.tensor(
            [
                [[[2.0, 4.0]]],
                [[[6.0, 8.0]]],
            ]
        )
        lses = torch.tensor(
            [
                [[1.0]],  # exp(1) ≈ 2.718
                [[2.0]],  # exp(2) ≈ 7.389
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        # Manual: w0 = exp(1)/(exp(1)+exp(2)), w1 = exp(2)/(exp(1)+exp(2))
        import math

        w0 = math.exp(1) / (math.exp(1) + math.exp(2))
        w1 = math.exp(2) / (math.exp(1) + math.exp(2))
        # Shape should be (B, H, D) = (1, 1, 2)
        expected = torch.tensor([[[w0 * 2.0 + w1 * 6.0, w0 * 4.0 + w1 * 8.0]]])

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_lse_weighted_combine_return_lse(self):
        """Test return_lse=True returns global LSE."""
        import torch

        from vllm.v1.attention.ops.helix import _lse_weighted_combine

        B, H, D = 1, 1, 2  # N=2 ranks
        outputs = torch.tensor(
            [
                [[[1.0, 2.0]]],
                [[[3.0, 4.0]]],
            ]
        )
        lses = torch.tensor(
            [
                [[1.0]],
                [[2.0]],
            ]
        )

        result, global_lse = _lse_weighted_combine(outputs, lses, return_lse=True)

        # global_lse should be logsumexp of input lses
        import math

        expected_global_lse = math.log(math.exp(1) + math.exp(2))

        assert result.shape == (B, H, D)
        assert global_lse.shape == (B, H)
        assert abs(global_lse.item() - expected_global_lse) < 1e-5


class TestHelixBackendCompatibility:
    """Test Helix backend compatibility documentation.

    These tests document known backend limitations for Helix mode.
    Actual validation happens at runtime in cuda.py.
    """

    def test_helix_gqa_flashinfer_not_supported(self):
        """Document: FlashInfer + Helix GQA is not supported.

        FlashInfer produces incorrect output with Helix GQA (TPA > 1).
        Users must use FLASH_ATTN for Helix GQA.

        Runtime validation in cuda.py:
        - Raises ValueError if user specifies --attention-backend FLASHINFER
        - Auto-selects FLASH_ATTN if backend is not specified
        """
        # This is a documentation test - actual validation is in cuda.py
        # Config creation itself doesn't validate backend compatibility
        config = ParallelConfig(
            helix_mode=True,
            tensor_parallel_size=8,
            decode_context_parallel_size=2,  # TPA = 4 > 1 (GQA case)
        )
        assert config.helix_tpa_size == 4
        assert config.helix_tpa_size > 1  # GQA case
        # Backend validation happens later in cuda.py check_and_update_config

    def test_helix_mla_flashinfer_supported(self):
        """Document: FlashInfer + Helix MLA is supported.

        When TPA=1 (MLA case), FlashInfer works correctly.
        The backend compatibility issue only affects TPA > 1 (GQA).
        """
        config = ParallelConfig(
            helix_mode=True,
            tensor_parallel_size=8,
            decode_context_parallel_size=8,  # TPA = 1 (MLA case)
        )
        assert config.helix_tpa_size == 1
        # FlashInfer is supported for TPA=1 (MLA)

    def test_helix_gqa_flash_attn_supported(self):
        """Document: FLASH_ATTN + Helix GQA is supported.

        FLASH_ATTN works correctly with Helix GQA at all TPA values.
        """
        config = ParallelConfig(
            helix_mode=True,
            tensor_parallel_size=8,
            decode_context_parallel_size=2,  # TPA = 4 (GQA case)
        )
        assert config.helix_tpa_size == 4
        # FLASH_ATTN is supported for all TPA values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
