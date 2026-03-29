# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FusedMoE weight loading with padded hidden dimensions.

When using DeepEP backends or NIXL EP with models like nemotron_h,
hidden_size may be rounded up (e.g., 2688 -> 3072) for backend requirements.
Weight parameters are created with the padded size, but checkpoint weights
have the original unpadded size. These tests verify that weight loading
correctly handles this mismatch.
"""

import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE


class TestNarrowExpertDataForPadding:
    """Unit tests for _narrow_expert_data_for_padding."""

    def test_no_narrowing_when_shapes_match(self):
        expert_data = torch.zeros(1024, 1024)
        loaded_weight = torch.randn(1024, 1024)
        result = FusedMoE._narrow_expert_data_for_padding(expert_data, loaded_weight)
        assert result.shape == loaded_weight.shape
        assert result.data_ptr() == expert_data.data_ptr()

    def test_narrow_w2_hidden_dim(self):
        # w2: (hidden_size, intermediate_size) - hidden_size padded
        expert_data = torch.zeros(3072, 1024)
        loaded_weight = torch.randn(2688, 1024)
        result = FusedMoE._narrow_expert_data_for_padding(expert_data, loaded_weight)
        assert result.shape == (2688, 1024)

    def test_narrow_w13_hidden_dim(self):
        # w1/w3: (intermediate_size, hidden_size) - hidden_size padded
        expert_data = torch.zeros(2048, 3072)
        loaded_weight = torch.randn(2048, 2688)
        result = FusedMoE._narrow_expert_data_for_padding(expert_data, loaded_weight)
        assert result.shape == (2048, 2688)

    def test_narrow_3d_full_load(self):
        # 3D tensor for full_load path
        expert_data = torch.zeros(8, 3072, 1024)
        loaded_weight = torch.randn(8, 2688, 1024)
        result = FusedMoE._narrow_expert_data_for_padding(expert_data, loaded_weight)
        assert result.shape == (8, 2688, 1024)

    def test_narrow_1d_scale(self):
        # 1D scale tensor
        expert_data = torch.zeros(3072)
        loaded_weight = torch.randn(2688)
        result = FusedMoE._narrow_expert_data_for_padding(expert_data, loaded_weight)
        assert result.shape == (2688,)

    def test_scalar_weight_no_op(self):
        # 0-dim tensor should be a no-op
        expert_data = torch.zeros(3072)
        loaded_weight = torch.tensor(1.0)
        result = FusedMoE._narrow_expert_data_for_padding(expert_data, loaded_weight)
        # ndim == 0, so loop doesn't execute
        assert result.shape == (3072,)

    def test_narrowed_data_shares_storage(self):
        # Verify narrowing returns a view (writes go to original tensor)
        expert_data = torch.zeros(3072, 1024)
        loaded_weight = torch.randn(2688, 1024)
        result = FusedMoE._narrow_expert_data_for_padding(expert_data, loaded_weight)
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

        # Simulate padded expert_data (as would be allocated by the layer)
        expert_data_full = torch.zeros(padded_hidden, intermediate)
        # Simulate checkpoint weight with original size
        loaded_weight = torch.randn(original_hidden, intermediate)

        # This is what _load_w2 does (without tp sharding for simplicity)
        expert_data = FusedMoE._narrow_expert_data_for_padding(
            expert_data_full, loaded_weight
        )
        expert_data.copy_(loaded_weight)

        # Verify: loaded region has correct values
        assert torch.equal(expert_data_full[:original_hidden, :], loaded_weight)
        # Verify: padded region remains zero
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

        expert_data = FusedMoE._narrow_expert_data_for_padding(
            expert_data_full, loaded_weight
        )
        expert_data.copy_(loaded_weight)

        assert torch.equal(expert_data_full[:, :original_hidden], loaded_weight)
        assert torch.equal(
            expert_data_full[:, original_hidden:],
            torch.zeros(intermediate, padded_hidden - original_hidden),
        )

    def test_no_padding_is_noop(self):
        """Verify that when sizes match, behavior is unchanged."""
        hidden = 2048
        intermediate = 1024

        expert_data_full = torch.zeros(hidden, intermediate)
        loaded_weight = torch.randn(hidden, intermediate)

        expert_data = FusedMoE._narrow_expert_data_for_padding(
            expert_data_full, loaded_weight
        )
        expert_data.copy_(loaded_weight)

        assert torch.equal(expert_data_full, loaded_weight)
