# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FusedMoE weight loading with padded hidden dimensions.

When using DeepEP backends or NIXL EP with models like nemotron_h,
hidden_size may be rounded up (e.g., 2688 -> 3072) for backend requirements.
Weight parameters are created with the padded size, but checkpoint weights
have the original unpadded size. These tests verify that weight loading
correctly handles this mismatch.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE


class TestGetHiddenDim:
    """Unit tests for _get_hidden_dim."""

    def test_2d_non_transposed_w2(self):
        # w2: shard_dim=1 (intermediate), hidden=0
        assert FusedMoE._get_hidden_dim(shard_dim=1, ndim=2) == 0

    def test_2d_non_transposed_w13(self):
        # w1/w3: shard_dim=0 (intermediate), hidden=1
        assert FusedMoE._get_hidden_dim(shard_dim=0, ndim=2) == 1

    def test_2d_transposed_w2(self):
        # transposed w2: shard_dim=0, hidden=1
        assert FusedMoE._get_hidden_dim(shard_dim=0, ndim=2) == 1

    def test_2d_transposed_w13(self):
        # transposed w1/w3: shard_dim=1, hidden=0
        assert FusedMoE._get_hidden_dim(shard_dim=1, ndim=2) == 0

    def test_3d_non_transposed_w2(self):
        # 3D w2: shard_dim=2, hidden=1
        assert FusedMoE._get_hidden_dim(shard_dim=2, ndim=3) == 1

    def test_3d_non_transposed_w13(self):
        # 3D w1/w3: shard_dim=1, hidden=2
        assert FusedMoE._get_hidden_dim(shard_dim=1, ndim=3) == 2

    def test_3d_transposed_w2(self):
        # transposed 3D w2: shard_dim=1, hidden=2
        assert FusedMoE._get_hidden_dim(shard_dim=1, ndim=3) == 2

    def test_3d_transposed_w13(self):
        # transposed 3D w1/w3: shard_dim=2, hidden=1
        assert FusedMoE._get_hidden_dim(shard_dim=2, ndim=3) == 1

    def test_1d_returns_zero(self):
        # 1D per-channel scales: always returns 0
        assert FusedMoE._get_hidden_dim(shard_dim=0, ndim=1) == 0
        assert FusedMoE._get_hidden_dim(shard_dim=1, ndim=1) == 0

    def test_invalid_shard_dim_raises(self):
        # shard_dim outside the data dimensions should raise
        with pytest.raises(ValueError, match="not a valid data dimension"):
            FusedMoE._get_hidden_dim(shard_dim=0, ndim=3)


class TestNarrowExpertDataForPadding:
    """Unit tests for _narrow_expert_data_for_padding."""

    def test_no_narrowing_when_shapes_match(self):
        expert_data = torch.zeros(1024, 1024)
        loaded_weight = torch.randn(1024, 1024)
        result = FusedMoE._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        assert result.shape == loaded_weight.shape
        assert result.data_ptr() == expert_data.data_ptr()

    def test_narrow_w2_hidden_dim(self):
        # w2: (hidden_size, intermediate_size) - hidden_size padded at dim 0
        expert_data = torch.zeros(3072, 1024)
        loaded_weight = torch.randn(2688, 1024)
        result = FusedMoE._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        assert result.shape == (2688, 1024)

    def test_narrow_w13_hidden_dim(self):
        # w1/w3: (intermediate_size, hidden_size) - hidden_size padded at dim 1
        expert_data = torch.zeros(2048, 3072)
        loaded_weight = torch.randn(2048, 2688)
        result = FusedMoE._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=1
        )
        assert result.shape == (2048, 2688)

    def test_narrow_transposed_w2(self):
        # transposed w2: (intermediate_size, hidden_size) - hidden at dim 1
        expert_data = torch.zeros(1024, 3072)
        loaded_weight = torch.randn(1024, 2688)
        hidden_dim = FusedMoE._get_hidden_dim(shard_dim=0, ndim=2)
        result = FusedMoE._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=hidden_dim
        )
        assert result.shape == (1024, 2688)

    def test_narrow_3d_full_load(self):
        # 3D tensor for full_load path: w2 (num_experts, hidden_size, intermediate)
        expert_data = torch.zeros(8, 3072, 1024)
        loaded_weight = torch.randn(8, 2688, 1024)
        result = FusedMoE._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=1
        )
        assert result.shape == (8, 2688, 1024)

    def test_narrow_1d_scale(self):
        # 1D scale tensor: per-channel w2 scale (hidden_size,)
        expert_data = torch.zeros(3072)
        loaded_weight = torch.randn(2688)
        result = FusedMoE._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        assert result.shape == (2688,)

    def test_scalar_weight_no_op(self):
        # 0-dim tensor should be a no-op
        expert_data = torch.zeros(3072)
        loaded_weight = torch.tensor(1.0)
        result = FusedMoE._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        # ndim == 0, so no narrowing
        assert result.shape == (3072,)

    def test_no_narrowing_when_loaded_weight_larger(self):
        # Guard: don't narrow if loaded_weight is larger than expert_data
        expert_data = torch.zeros(2688, 1024)
        loaded_weight = torch.randn(3072, 1024)
        result = FusedMoE._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        assert result.shape == (2688, 1024)
        assert result.data_ptr() == expert_data.data_ptr()

    def test_negative_hidden_dim_is_noop(self):
        # Negative hidden_dim should be a safe no-op (0 <= check)
        expert_data = torch.zeros(3072, 1024)
        loaded_weight = torch.randn(2688, 1024)
        result = FusedMoE._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=-1
        )
        # -1 fails the 0 <= check, so no narrowing
        assert result.shape == (3072, 1024)
        assert result.data_ptr() == expert_data.data_ptr()

    def test_only_narrows_hidden_dim(self):
        # Verify that only the specified hidden_dim is narrowed,
        # even when other dimensions also differ
        expert_data = torch.zeros(3072, 2048)
        loaded_weight = torch.randn(2688, 1024)
        result = FusedMoE._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        # Only dim 0 (hidden) should be narrowed; dim 1 stays at 2048
        assert result.shape == (2688, 2048)

    def test_narrowed_data_shares_storage(self):
        # Verify narrowing returns a view (writes go to original tensor)
        expert_data = torch.zeros(3072, 1024)
        loaded_weight = torch.randn(2688, 1024)
        result = FusedMoE._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        result.copy_(loaded_weight)
        # The first 2688 rows of expert_data should now have loaded_weight
        assert torch.equal(expert_data[:2688, :], loaded_weight)
        # Padded region should remain zero
        assert torch.equal(expert_data[2688:, :], torch.zeros(3072 - 2688, 1024))


class TestWeightLoadingWithPaddedHiddenSize:
    """Integration-style tests that simulate padded weight loading."""

    def test_load_w2_with_padding(self):
        """Simulate loading w2 weights when hidden_size is padded."""
        padded_hidden = 3072
        original_hidden = 2688
        intermediate = 1024

        expert_data_full = torch.zeros(padded_hidden, intermediate)
        loaded_weight = torch.randn(original_hidden, intermediate)

        # w2 non-transposed: shard_dim=1, hidden_dim=0
        hidden_dim = FusedMoE._get_hidden_dim(shard_dim=1, ndim=2)
        expert_data = FusedMoE._narrow_expert_data_for_padding(
            expert_data_full, loaded_weight, hidden_dim=hidden_dim
        )
        expert_data.copy_(loaded_weight)

        assert torch.equal(expert_data_full[:original_hidden, :], loaded_weight)
        assert torch.equal(
            expert_data_full[original_hidden:, :],
            torch.zeros(padded_hidden - original_hidden, intermediate),
        )

    def test_load_w13_with_padding(self):
        """Simulate loading w1/w3 weights when hidden_size is padded."""
        padded_hidden = 3072
        original_hidden = 2688
        intermediate = 1024

        # w1/w3: (intermediate_size, hidden_size)
        expert_data_full = torch.zeros(intermediate, padded_hidden)
        loaded_weight = torch.randn(intermediate, original_hidden)

        # w1 non-transposed: shard_dim=0, hidden_dim=1
        hidden_dim = FusedMoE._get_hidden_dim(shard_dim=0, ndim=2)
        expert_data = FusedMoE._narrow_expert_data_for_padding(
            expert_data_full, loaded_weight, hidden_dim=hidden_dim
        )
        expert_data.copy_(loaded_weight)

        assert torch.equal(expert_data_full[:, :original_hidden], loaded_weight)
        assert torch.equal(
            expert_data_full[:, original_hidden:],
            torch.zeros(intermediate, padded_hidden - original_hidden),
        )

    def test_load_transposed_w2_with_padding(self):
        """Simulate loading transposed w2 (GPTQ) with padded hidden_size."""
        padded_hidden = 3072
        original_hidden = 2688
        intermediate = 1024

        # transposed w2: (intermediate_size, hidden_size), shard_dim=0
        expert_data_full = torch.zeros(intermediate, padded_hidden)
        loaded_weight = torch.randn(intermediate, original_hidden)

        hidden_dim = FusedMoE._get_hidden_dim(shard_dim=0, ndim=2)
        expert_data = FusedMoE._narrow_expert_data_for_padding(
            expert_data_full, loaded_weight, hidden_dim=hidden_dim
        )
        expert_data.copy_(loaded_weight)

        assert torch.equal(expert_data_full[:, :original_hidden], loaded_weight)

    def test_no_padding_is_noop(self):
        """Verify that when sizes match, behavior is unchanged."""
        hidden = 2048
        intermediate = 1024

        expert_data_full = torch.zeros(hidden, intermediate)
        loaded_weight = torch.randn(hidden, intermediate)

        hidden_dim = FusedMoE._get_hidden_dim(shard_dim=1, ndim=2)
        expert_data = FusedMoE._narrow_expert_data_for_padding(
            expert_data_full, loaded_weight, hidden_dim=hidden_dim
        )
        expert_data.copy_(loaded_weight)

        assert torch.equal(expert_data_full, loaded_weight)

    def test_bnb_shape_mismatch_raises(self):
        """BnB + padded hidden_size should raise via weight_loader."""
        from unittest.mock import MagicMock

        num_experts = 1
        padded_packed = 3072  # padded packed size
        original_packed = 2688  # original packed size

        # Build a param that looks like a BnB 4-bit MoE weight.
        param_data = torch.zeros(num_experts, padded_packed, 1, dtype=torch.uint8)
        param = torch.nn.Parameter(param_data, requires_grad=False)
        param.use_bitsandbytes_4bit = True

        loaded_weight = torch.randint(0, 255, (original_packed, 1), dtype=torch.uint8)

        # Minimal FusedMoE mock so weight_loader reaches the BnB path.
        moe = MagicMock(spec=FusedMoE)
        moe.quant_config = None
        moe.quant_method = MagicMock()
        moe.quant_method.__class__.__name__ = "BitsAndBytesMethod"
        moe._expert_map = None
        moe.tp_rank = 0

        # Call the real weight_loader (unbound) with our mock as self.
        with pytest.raises(ValueError, match="BitsAndBytes"):
            FusedMoE.weight_loader(
                moe,
                param,
                loaded_weight,
                weight_name="w2",
                shard_id="w2",
                expert_id=0,
            )
