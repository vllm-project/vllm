# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for FusedMoEWithSharedOuterLoRA.

This module tests the shared outer LoRA pattern for MoE layers where:
- w13 (gate/up projections): LoRA A is shared across all experts
- w2 (down projection): LoRA B is shared across all experts

The sharing is implemented via expand() with stride=0 on the expert dimension.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config.lora import LoRAConfig
from vllm.lora.layers import (
    FusedMoEWithLoRA,
    FusedMoEWithSharedOuterLoRA,
)
from vllm.lora.utils import _all_lora_classes, from_layer
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.platforms import current_platform

DEVICE = "cuda" if current_platform.is_cuda_alike() else "cpu"


@pytest.fixture
def mock_base_layer():
    """Create a mock FusedMoE base layer."""
    base_layer = MagicMock()
    base_layer.use_ep = False
    base_layer.tp_size = 1
    base_layer.tp_rank = 0
    base_layer.local_num_experts = 8
    base_layer.global_num_experts = 8
    base_layer.hidden_size = 256
    base_layer.intermediate_size_per_partition = 512
    return base_layer


@pytest.fixture
def mock_lora_config():
    """Create a mock LoRA config."""
    config = MagicMock(spec=LoRAConfig)
    config.max_loras = 4
    config.max_lora_rank = 16
    config.lora_dtype = torch.float32
    config.fully_sharded_loras = False
    return config


@pytest.fixture
def shared_outer_lora_layer(mock_base_layer, mock_lora_config):
    """Create a FusedMoEWithSharedOuterLoRA layer with weights initialized."""
    with patch(
        "vllm.lora.layers.fused_moe._get_lora_device",
        return_value=torch.device(DEVICE),
    ), patch.object(
        FusedMoEWithSharedOuterLoRA, "_inject_lora_into_fused_moe"
    ):
        layer = FusedMoEWithSharedOuterLoRA(mock_base_layer)
        layer.create_lora_weights(
            mock_lora_config.max_loras, mock_lora_config, None
        )
    return layer


class TestLayerSelection:
    """Tests for layer selection logic."""

    def test_shared_outer_lora_registered(self):
        """Test that FusedMoEWithSharedOuterLoRA is registered in _all_lora_classes."""
        assert FusedMoEWithSharedOuterLoRA in _all_lora_classes

    def test_can_replace_layer_with_shared_moe_lora_enabled(self):
        """Test that FusedMoEWithSharedOuterLoRA is selected when use_shared_moe_lora=True."""
        mock_layer = MagicMock(spec=FusedMoE)
        mock_layer.__class__ = FusedMoE

        mock_lora_config = MagicMock(spec=LoRAConfig)
        mock_model_config = MagicMock()
        mock_model_config.use_shared_moe_lora = True

        result = FusedMoEWithSharedOuterLoRA.can_replace_layer(
            source_layer=mock_layer,
            lora_config=mock_lora_config,
            packed_modules_list=["experts", "shared_experts"],
            model_config=mock_model_config,
        )

        assert result is True

    def test_can_replace_layer_without_shared_moe_lora(self):
        """Test that FusedMoEWithSharedOuterLoRA is NOT selected when use_shared_moe_lora=False."""
        mock_layer = MagicMock(spec=FusedMoE)
        mock_layer.__class__ = FusedMoE

        mock_lora_config = MagicMock(spec=LoRAConfig)
        mock_model_config = MagicMock()
        mock_model_config.use_shared_moe_lora = False

        result = FusedMoEWithSharedOuterLoRA.can_replace_layer(
            source_layer=mock_layer,
            lora_config=mock_lora_config,
            packed_modules_list=["experts", "shared_experts"],
            model_config=mock_model_config,
        )

        assert result is False

    def test_regular_fused_moe_defers_when_shared_enabled(self):
        """Test that FusedMoEWithLoRA defers to FusedMoEWithSharedOuterLoRA when enabled."""
        mock_layer = MagicMock(spec=FusedMoE)
        mock_layer.__class__ = FusedMoE

        mock_lora_config = MagicMock(spec=LoRAConfig)
        mock_model_config = MagicMock()
        mock_model_config.use_shared_moe_lora = True

        result = FusedMoEWithLoRA.can_replace_layer(
            source_layer=mock_layer,
            lora_config=mock_lora_config,
            packed_modules_list=["experts", "shared_experts"],
            model_config=mock_model_config,
        )

        assert result is False

    def test_from_layer_selects_shared_outer_lora(self, mock_base_layer, mock_lora_config):
        """Test that from_layer selects FusedMoEWithSharedOuterLoRA when enabled."""
        mock_model_config = MagicMock()
        mock_model_config.use_shared_moe_lora = True

        # Need to make it look like a FusedMoE
        mock_base_layer.__class__ = FusedMoE

        with patch(
            "vllm.lora.layers.fused_moe._get_lora_device",
            return_value=torch.device(DEVICE),
        ), patch.object(
            FusedMoEWithSharedOuterLoRA, "_inject_lora_into_fused_moe"
        ):
            result = from_layer(
                layer=mock_base_layer,
                max_loras=4,
                lora_config=mock_lora_config,
                packed_modules_list=["experts", "shared_experts"],
                model_config=mock_model_config,
            )

        assert isinstance(result, FusedMoEWithSharedOuterLoRA)


class TestWeightShapesAndStrides:
    """Tests for weight tensor shapes and strides."""

    def test_shared_weight_strides_are_zero(self, shared_outer_lora_layer):
        """Test that shared weights have stride=0 on expert dimension."""
        # w13_lora_a is shared
        w13_a_stride = shared_outer_lora_layer.w13_lora_a_stacked[0].stride(1)
        assert w13_a_stride == 0, f"w13_lora_a expert stride should be 0, got {w13_a_stride}"

        # w2_lora_b is shared
        w2_b_stride = shared_outer_lora_layer.w2_lora_b_stacked[0].stride(1)
        assert w2_b_stride == 0, f"w2_lora_b expert stride should be 0, got {w2_b_stride}"

    def test_per_expert_weight_strides_are_nonzero(self, shared_outer_lora_layer):
        """Test that per-expert weights have non-zero stride on expert dimension."""
        # w13_lora_b is per-expert
        w13_b_stride = shared_outer_lora_layer.w13_lora_b_stacked[0].stride(1)
        assert w13_b_stride != 0, "w13_lora_b expert stride should be non-zero"

        # w2_lora_a is per-expert
        w2_a_stride = shared_outer_lora_layer.w2_lora_a_stacked[0].stride(1)
        assert w2_a_stride != 0, "w2_lora_a expert stride should be non-zero"

    def test_storage_shapes_have_expert_dim_one(self, shared_outer_lora_layer):
        """Test that storage tensors have expert_dim=1."""
        storage_w13_a = shared_outer_lora_layer._w13_lora_a_storage[0].shape
        assert storage_w13_a[1] == 1, f"Storage expert dim should be 1, got {storage_w13_a[1]}"

        storage_w2_b = shared_outer_lora_layer._w2_lora_b_storage[0].shape
        assert storage_w2_b[1] == 1, f"Storage expert dim should be 1, got {storage_w2_b[1]}"

    def test_expanded_shapes_have_full_experts(self, shared_outer_lora_layer, mock_base_layer):
        """Test that expanded views have full expert dimension."""
        num_experts = mock_base_layer.local_num_experts

        w13_a_shape = shared_outer_lora_layer.w13_lora_a_stacked[0].shape
        assert w13_a_shape[1] == num_experts, f"Expected {num_experts} experts, got {w13_a_shape[1]}"

        w2_b_shape = shared_outer_lora_layer.w2_lora_b_stacked[0].shape
        assert w2_b_shape[1] == num_experts, f"Expected {num_experts} experts, got {w2_b_shape[1]}"


class TestSetLora:
    """Tests for set_lora functionality."""

    def test_set_lora_with_shared_weights(self, shared_outer_lora_layer, mock_base_layer):
        """Test that set_lora correctly handles shared weights with expert_dim=1."""
        num_experts = mock_base_layer.local_num_experts
        rank = 16
        hidden_size = mock_base_layer.hidden_size
        intermediate_size = mock_base_layer.intermediate_size_per_partition

        torch.manual_seed(42)
        # Shared weights (expert_dim=1)
        w1_lora_a = torch.randn(1, rank, hidden_size, dtype=torch.float32, device=DEVICE)
        w3_lora_a = torch.randn(1, rank, hidden_size, dtype=torch.float32, device=DEVICE)
        w2_lora_b = torch.randn(1, hidden_size, rank, dtype=torch.float32, device=DEVICE)

        # Per-expert weights
        w1_lora_b = torch.randn(num_experts, intermediate_size, rank, dtype=torch.float32, device=DEVICE)
        w3_lora_b = torch.randn(num_experts, intermediate_size, rank, dtype=torch.float32, device=DEVICE)
        w2_lora_a = torch.randn(num_experts, rank, intermediate_size, dtype=torch.float32, device=DEVICE)

        lora_a = [w1_lora_a, w2_lora_a, w3_lora_a]
        lora_b = [w1_lora_b, w2_lora_b, w3_lora_b]

        shared_outer_lora_layer.set_lora(0, lora_a, lora_b)

        # Verify shared weights are broadcast to all experts
        stored_w13_a = shared_outer_lora_layer.w13_lora_a_stacked[0][0]
        for i in range(1, num_experts):
            assert torch.allclose(stored_w13_a[0], stored_w13_a[i]), \
                f"Expert 0 and {i} should have identical shared weights"

        # Verify stored value matches input
        assert torch.allclose(stored_w13_a[0], w1_lora_a.squeeze(0)), \
            "Stored value should match input"

    def test_set_lora_with_identical_expert_weights(self, shared_outer_lora_layer, mock_base_layer):
        """Test that set_lora accepts (num_experts, ...) with identical values."""
        num_experts = mock_base_layer.local_num_experts
        rank = 16
        hidden_size = mock_base_layer.hidden_size
        intermediate_size = mock_base_layer.intermediate_size_per_partition

        torch.manual_seed(42)
        # Create shared weight with expert dim but identical values
        base_w1_a = torch.randn(1, rank, hidden_size, dtype=torch.float32, device=DEVICE)
        w1_lora_a = base_w1_a.expand(num_experts, rank, hidden_size).clone()

        w3_lora_a = torch.randn(1, rank, hidden_size, dtype=torch.float32, device=DEVICE)
        w2_lora_b = torch.randn(1, hidden_size, rank, dtype=torch.float32, device=DEVICE)

        w1_lora_b = torch.randn(num_experts, intermediate_size, rank, dtype=torch.float32, device=DEVICE)
        w3_lora_b = torch.randn(num_experts, intermediate_size, rank, dtype=torch.float32, device=DEVICE)
        w2_lora_a = torch.randn(num_experts, rank, intermediate_size, dtype=torch.float32, device=DEVICE)

        lora_a = [w1_lora_a, w2_lora_a, w3_lora_a]
        lora_b = [w1_lora_b, w2_lora_b, w3_lora_b]

        # Should not raise - identical expert weights are accepted
        shared_outer_lora_layer.set_lora(0, lora_a, lora_b)

    def test_set_lora_rejects_different_expert_weights(self, shared_outer_lora_layer, mock_base_layer):
        """Test that set_lora rejects shared weights with different expert values."""
        num_experts = mock_base_layer.local_num_experts
        rank = 16
        hidden_size = mock_base_layer.hidden_size
        intermediate_size = mock_base_layer.intermediate_size_per_partition

        torch.manual_seed(42)
        # Create weights with different values per expert (WRONG for shared)
        w1_lora_a = torch.randn(num_experts, rank, hidden_size, dtype=torch.float32, device=DEVICE)
        w3_lora_a = torch.randn(1, rank, hidden_size, dtype=torch.float32, device=DEVICE)
        w2_lora_b = torch.randn(1, hidden_size, rank, dtype=torch.float32, device=DEVICE)

        w1_lora_b = torch.randn(num_experts, intermediate_size, rank, dtype=torch.float32, device=DEVICE)
        w3_lora_b = torch.randn(num_experts, intermediate_size, rank, dtype=torch.float32, device=DEVICE)
        w2_lora_a = torch.randn(num_experts, rank, intermediate_size, dtype=torch.float32, device=DEVICE)

        lora_a = [w1_lora_a, w2_lora_a, w3_lora_a]
        lora_b = [w1_lora_b, w2_lora_b, w3_lora_b]

        with pytest.raises(ValueError, match="experts have DIFFERENT values"):
            shared_outer_lora_layer.set_lora(0, lora_a, lora_b)

    def test_reset_lora(self, shared_outer_lora_layer, mock_base_layer):
        """Test that reset_lora zeros out the weights."""
        num_experts = mock_base_layer.local_num_experts
        rank = 16
        hidden_size = mock_base_layer.hidden_size
        intermediate_size = mock_base_layer.intermediate_size_per_partition

        torch.manual_seed(42)
        w1_lora_a = torch.randn(1, rank, hidden_size, dtype=torch.float32, device=DEVICE)
        w3_lora_a = torch.randn(1, rank, hidden_size, dtype=torch.float32, device=DEVICE)
        w2_lora_b = torch.randn(1, hidden_size, rank, dtype=torch.float32, device=DEVICE)
        w1_lora_b = torch.randn(num_experts, intermediate_size, rank, dtype=torch.float32, device=DEVICE)
        w3_lora_b = torch.randn(num_experts, intermediate_size, rank, dtype=torch.float32, device=DEVICE)
        w2_lora_a = torch.randn(num_experts, rank, intermediate_size, dtype=torch.float32, device=DEVICE)

        shared_outer_lora_layer.set_lora(0, [w1_lora_a, w2_lora_a, w3_lora_a], [w1_lora_b, w2_lora_b, w3_lora_b])

        # Verify weights are non-zero
        assert shared_outer_lora_layer._w13_lora_a_storage[0][0].abs().sum() > 0

        # Reset
        shared_outer_lora_layer.reset_lora(0)

        # Verify weights are zeroed
        assert shared_outer_lora_layer._w13_lora_a_storage[0][0].abs().sum() == 0
        assert shared_outer_lora_layer._w2_lora_b_storage[0][0].abs().sum() == 0
        assert shared_outer_lora_layer.w13_lora_b_stacked[0][0].abs().sum() == 0
        assert shared_outer_lora_layer.w2_lora_a_stacked[0][0].abs().sum() == 0


class TestStrideZeroEquivalence:
    """Tests verifying that stride=0 produces same results as explicit copies."""

    def test_expand_produces_stride_zero(self):
        """Test that expand() produces stride=0 on the expanded dimension."""
        storage = torch.randn(4, 1, 8, 128)
        expanded = storage.expand(4, 16, 8, 128)

        assert expanded.stride(1) == 0
        assert expanded.shape == (4, 16, 8, 128)

    def test_all_experts_see_same_data(self):
        """Test that all experts access the same underlying data."""
        storage = torch.randn(4, 1, 8, 128)
        expanded = storage.expand(4, 16, 8, 128)

        lora_id = 0
        for expert_id in range(1, 16):
            assert torch.allclose(expanded[lora_id, 0], expanded[lora_id, expert_id])

    def test_write_to_storage_affects_all_experts(self):
        """Test that writing to storage is visible to all expert views."""
        storage = torch.zeros(4, 1, 8, 128)
        expanded = storage.expand(4, 16, 8, 128)

        # Write to storage
        storage[0, 0, :, :] = torch.randn(8, 128)

        # All experts should see the same updated value
        for expert_id in range(16):
            assert torch.allclose(expanded[0, expert_id], storage[0, 0])


class TestExpertParallel:
    """Tests for Expert Parallel (EP) support in MoE LoRA layers."""

    @pytest.fixture
    def mock_ep_base_layer(self):
        """Create a mock FusedMoE base layer configured for EP."""
        base_layer = MagicMock()
        base_layer.use_ep = True
        base_layer.tp_size = 1
        base_layer.tp_rank = 0
        base_layer.local_num_experts = 8
        base_layer.global_num_experts = 64
        base_layer.hidden_size = 256
        base_layer.intermediate_size_per_partition = 512
        return base_layer

    def test_fused_moe_with_lora_init_succeeds_with_ep(
        self, mock_ep_base_layer
    ):
        """Test that FusedMoEWithLoRA initializes successfully with EP."""
        with patch(
            "vllm.lora.layers.fused_moe._get_lora_device",
            return_value=torch.device(DEVICE),
        ), patch.object(FusedMoEWithLoRA, "_inject_lora_into_fused_moe"):
            layer = FusedMoEWithLoRA(mock_ep_base_layer)

        assert layer.tp_size == 1
        assert layer.tp_rank == 0

    def test_shared_outer_lora_init_succeeds_with_ep(
        self, mock_ep_base_layer
    ):
        """Test that FusedMoEWithSharedOuterLoRA initializes successfully with EP."""
        with patch(
            "vllm.lora.layers.fused_moe._get_lora_device",
            return_value=torch.device(DEVICE),
        ), patch.object(
            FusedMoEWithSharedOuterLoRA, "_inject_lora_into_fused_moe"
        ):
            layer = FusedMoEWithSharedOuterLoRA(mock_ep_base_layer)

        assert layer.tp_size == 1
        assert layer.tp_rank == 0

    def test_ep_weight_shapes_use_local_num_experts(
        self, mock_ep_base_layer, mock_lora_config
    ):
        """Test that LoRA weight shapes use local_num_experts under EP."""
        with patch(
            "vllm.lora.layers.fused_moe._get_lora_device",
            return_value=torch.device(DEVICE),
        ), patch.object(
            FusedMoEWithSharedOuterLoRA, "_inject_lora_into_fused_moe"
        ):
            layer = FusedMoEWithSharedOuterLoRA(mock_ep_base_layer)
            layer.create_lora_weights(
                mock_lora_config.max_loras, mock_lora_config, None
            )

        # Per-expert weights should use local_num_experts (8, not 64)
        local_experts = mock_ep_base_layer.local_num_experts
        w13_b_shape = layer.w13_lora_b_stacked[0].shape
        assert w13_b_shape[1] == local_experts, (
            f"Expected {local_experts} local experts, got {w13_b_shape[1]}"
        )

        w2_a_shape = layer.w2_lora_a_stacked[0].shape
        assert w2_a_shape[1] == local_experts, (
            f"Expected {local_experts} local experts, got {w2_a_shape[1]}"
        )
