# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Gemma 4 multimodal performance optimizations (PR 6).

Covers:
- Gemma4VisionRMSNorm weight preservation and hidden_size resolution
- _optimize_hf_multimodal_tower Gemma4VisionAttention component swap
- Batched vision pooler for image and video inputs (including identity pooling)
- _clear_mm_prefix_for_full_attn_layers memoization and clearing logic
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4VisionModel,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4VisionAttention as HfGemma4VisionAttention,
)

from vllm.model_executor.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    Gemma4VisionAttention,
    Gemma4VisionRMSNorm,
    _optimize_hf_multimodal_tower,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GPU kernel tests"
)
def test_gemma4_vision_rmsnorm_weight_preservation(default_vllm_config):
    """Verify that _optimize_hf_multimodal_tower preserves pre-trained non-ones
    weights and GPU placement bit-for-bit (Bug 1 & Bug 4)."""
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    hidden_size = 128

    class DummyParent(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = hidden_size
            self.norm = Gemma4RMSNorm(hidden_size, eps=1e-6)

    model = DummyParent().to(device=device, dtype=dtype)
    # Fill with non-trivial pretrained weights
    torch.manual_seed(42)
    custom_weight = torch.randn(hidden_size, device=device, dtype=dtype) * 0.5 + 1.2
    model.norm.weight.data.copy_(custom_weight)

    _optimize_hf_multimodal_tower(model)

    assert isinstance(model.norm, Gemma4VisionRMSNorm)
    assert model.norm.has_weight is True
    assert model.norm.weight.device == device
    assert model.norm.weight.dtype == dtype
    assert torch.equal(model.norm.weight, custom_weight)

    x = torch.randn(2, 16, hidden_size, device=device, dtype=dtype)
    out = model.norm(x)
    assert out.shape == x.shape
    assert out.device == device
    assert torch.equal(model.norm.weight, custom_weight)


def test_optimize_tower_head_dim_resolution(default_vllm_config):
    """Verify safe hidden_size resolution when child.with_scale is False (Bug 2)."""

    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 48
            # v_norm in Gemma4VisionAttention has with_scale=False
            self.v_norm = Gemma4RMSNorm(48, eps=1e-6, with_scale=False)

    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 96
            self.unscaled_norm = Gemma4RMSNorm(96, eps=1e-6, with_scale=False)

    attn = DummyAttn()
    _optimize_hf_multimodal_tower(attn)
    assert isinstance(attn.v_norm, Gemma4VisionRMSNorm)
    assert attn.v_norm.has_weight is False
    assert attn.v_norm.hidden_size == 48

    layer = DummyLayer()
    _optimize_hf_multimodal_tower(layer)
    assert isinstance(layer.unscaled_norm, Gemma4VisionRMSNorm)
    assert layer.unscaled_norm.has_weight is False
    assert layer.unscaled_norm.hidden_size == 96


def test_vision_attention_component_swap(default_vllm_config):
    """Verify _optimize_hf_multimodal_tower swaps attention class to
    Gemma4VisionAttention without affecting standalone models or global
    transformers module functions."""
    import transformers.models.gemma4.modeling_gemma4 as modeling_gemma4

    vis_cfg = Gemma4VisionConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        head_dim=16,
    )
    model = Gemma4VisionModel(vis_cfg)
    orig_rope_fn = modeling_gemma4.apply_multidimensional_rope

    _optimize_hf_multimodal_tower(model)

    # 1. First layer self_attn in model is swapped to vLLM Gemma4VisionAttention
    attn = model.encoder.layers[0].self_attn
    assert isinstance(attn, Gemma4VisionAttention)
    assert isinstance(attn.q_norm, Gemma4VisionRMSNorm)
    assert isinstance(attn.k_norm, Gemma4VisionRMSNorm)
    assert isinstance(attn.v_norm, Gemma4VisionRMSNorm)

    # 2. Standalone model created after optimization is unaffected
    other_model = Gemma4VisionModel(vis_cfg)
    assert type(other_model.encoder.layers[0].self_attn) is HfGemma4VisionAttention

    # 3. Global transformers module function is untouched
    assert modeling_gemma4.apply_multidimensional_rope is orig_rope_fn


def test_clear_mm_prefix_for_full_attn_layers_memoization():
    """Verify _clear_mm_prefix_for_full_attn_layers correctly filters full attention
    layers, memoizes lookup keys, and handles cache eviction."""

    class DummyModel:
        def __init__(self):
            self._full_attn_layer_idxs = frozenset([0, 2])
            self._cached_full_attn_layer_names = {}

    dummy = DummyModel()
    clear_func = (
        Gemma4ForConditionalGeneration._clear_mm_prefix_for_full_attn_layers.__get__(
            dummy
        )
    )

    class MockMetadata:
        def __init__(self, prefix):
            self.mm_prefix_range = prefix
            self.mm_prefix_range_tensor = prefix
            self.mm_prefix_query_range_tensor = prefix

    layer_metadata = {
        "model.layers.0.self_attn": MockMetadata([0, 10]),
        "model.layers.1.self_attn": MockMetadata([0, 10]),  # sliding layer
        "model.layers.2.self_attn": MockMetadata([0, 10]),
    }

    mock_fc = MagicMock()
    mock_fc.attn_metadata = layer_metadata

    with patch("vllm.forward_context.get_forward_context", return_value=mock_fc):
        clear_func()

        # Layers 0 and 2 should have prefixes cleared
        assert layer_metadata["model.layers.0.self_attn"].mm_prefix_range is None
        assert layer_metadata["model.layers.2.self_attn"].mm_prefix_range is None
        # Layer 1 (sliding window) must retain its prefix
        assert layer_metadata["model.layers.1.self_attn"].mm_prefix_range == [0, 10]

        # Verify memoization
        keys_tuple = tuple(layer_metadata.keys())
        assert keys_tuple in dummy._cached_full_attn_layer_names
        cached = dummy._cached_full_attn_layer_names[keys_tuple]
        assert "model.layers.0.self_attn" in cached
        assert "model.layers.2.self_attn" in cached
        assert "model.layers.1.self_attn" not in cached

        # Verify cache bounding: simulate 9 different key sets
        for i in range(9):
            dummy_dict = {f"model.layers.{i}.self_attn": MockMetadata([0, 5])}
            mock_fc.attn_metadata = dummy_dict
            clear_func()

        assert len(dummy._cached_full_attn_layer_names) <= 8


def test_batched_vision_pooler_image_and_video_processing():
    """Verify batched vision pooler across single resolution, multi-resolution buckets,
    -1 padding, empty batches, and multiple videos."""

    class DummyVisionTower(nn.Module):
        def __init__(self, hidden_dim=64, pool_k=2):
            super().__init__()
            self.config = MagicMock()
            self.config.standardize = False
            self.config.position_embedding_size = 32
            self.hidden_dim = hidden_dim
            self.pool_k = pool_k
            self.patch_embedder = MagicMock(
                side_effect=lambda pv, pp, pad: torch.randn(*pv.shape[:-1], hidden_dim)
            )
            self.encoder = MagicMock(
                side_effect=lambda inputs_embeds, **kw: MagicMock(
                    last_hidden_state=inputs_embeds
                )
            )

            def mock_pooler(
                hidden_states, pixel_position_ids, padding_positions, output_length
            ):
                B = hidden_states.shape[0]
                pooled = torch.randn(B, output_length, hidden_dim)
                if output_length == hidden_states.shape[1]:
                    # HF identity pooling returns padding_positions (True for pad)
                    return pooled, padding_positions[:, :output_length].clone()
                # Otherwise standard pooling returns valid_mask (False for pad)
                valid = ~padding_positions[:, :output_length]
                return pooled, valid

            self.pooler = MagicMock(side_effect=mock_pooler)

    class DummyModel:
        def __init__(self, pool_k=2):
            self.config = MagicMock()
            self.config.vision_config = MagicMock()
            self.config.vision_config.pooling_kernel_size = pool_k
            self.config.vision_config.position_embedding_size = 32
            self.config.vision_config.default_output_length = 70
            self.vision_tower = DummyVisionTower(hidden_dim=64, pool_k=pool_k)
            self._enable_mm_lora = False
            self.model_dtype = torch.bfloat16
            self.embed_vision = MagicMock(
                side_effect=lambda inputs_embeds: inputs_embeds
            )
            self._encoder_chunk = staticmethod(lambda p, f, t, pos: 10)

    # 1. Image input: empty batch
    dummy = DummyModel(pool_k=2)
    process_img = Gemma4ForConditionalGeneration._process_image_input.__get__(dummy)
    assert process_img({"pixel_values": [], "pixel_position_ids": []}) == []

    # 2. Image input: single resolution with -1 padding
    pv_1 = torch.randn(16, 768)
    pp_1 = torch.zeros(16, 2, dtype=torch.long)
    pp_1[12:, :] = -1  # 4 padding tokens
    pv_2 = torch.randn(16, 768)
    pp_2 = torch.zeros(16, 2, dtype=torch.long)
    pp_2[14:, :] = -1  # 2 padding tokens

    out_single = process_img(
        {
            "pixel_values": [pv_1, pv_2],
            "pixel_position_ids": [pp_1, pp_2],
        }
    )
    assert len(out_single) == 2
    assert isinstance(out_single[0], torch.Tensor)
    assert isinstance(out_single[1], torch.Tensor)

    # 3. Image input: multi-resolution buckets
    pv_bucket2 = torch.randn(32, 768)
    pp_bucket2 = torch.zeros(32, 2, dtype=torch.long)
    out_multi = process_img(
        {
            "pixel_values": [pv_1, pv_bucket2],
            "pixel_position_ids": [pp_1, pp_bucket2],
        }
    )
    assert len(out_multi) == 2

    # 3b. Image input: identity pooling (output_length == seq_len, pool_k=1)
    dummy_identity = DummyModel(pool_k=1)
    dummy_identity.config.vision_config.default_output_length = 16
    process_img_identity = Gemma4ForConditionalGeneration._process_image_input.__get__(
        dummy_identity
    )
    out_identity = process_img_identity(
        {
            "pixel_values": [pv_1, pv_2],
            "pixel_position_ids": [pp_1, pp_2],
        }
    )
    assert len(out_identity) == 2
    # Verify valid_mask inversion: returns only 12 and 14 valid non-padding tokens
    assert out_identity[0].shape[0] == 12
    assert out_identity[1].shape[0] == 14

    # 4. Video input: empty batch
    process_vid = Gemma4ForConditionalGeneration._process_video_input.__get__(dummy)
    assert (
        process_vid(
            {
                "pixel_values_videos": torch.empty(0, 16, 768),
                "pixel_position_ids_videos": torch.empty(0, 16, 2, dtype=torch.long),
                "video_frame_counts": [],
            }
        )
        == []
    )

    # 5. Video input: multiple videos [2, 3] frames = 5 frames total
    total_frames = 5
    pv_vid = torch.randn(total_frames, 16, 768)
    pp_vid = torch.zeros(total_frames, 16, 2, dtype=torch.long)
    pp_vid[:, 12:, :] = -1  # padding
    fc = [2, 3]

    out_vid = process_vid(
        {
            "pixel_values_videos": pv_vid,
            "pixel_position_ids_videos": pp_vid,
            "video_frame_counts": fc,
        }
    )
    assert len(out_vid) == 2
    assert isinstance(out_vid[0], torch.Tensor)
    assert isinstance(out_vid[1], torch.Tensor)
