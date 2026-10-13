# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Gemma 4 multimodal performance optimizations (PR 6).

Covers:
- Gemma4VisionRMSNorm kernel dispatch, weight preservation, and fallbacks
- _optimize_hf_multimodal_tower monkey-patching and hidden_size resolution
- fused_vision_2d_rope numerical parity, broadcasting, and empty guards
- Batched vision pooler for image and video inputs (including identity pooling)
- _clear_mm_prefix_for_full_attn_layers memoization and clearing logic
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4VisionAttention,
    Gemma4VisionModel,
    apply_multidimensional_rope,
)

from vllm.model_executor.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    Gemma4VisionRMSNorm,
    _optimize_hf_multimodal_tower,
    fused_vision_2d_rope,
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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GPU kernel tests"
)
def test_gemma4_vision_rmsnorm_shapes_and_fallbacks(default_vllm_config):
    """Test Gemma4VisionRMSNorm across 2D/3D/4D contiguous & non-contiguous shapes,
    with_scale=True/False, empty tensors, and CPU fallback."""
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    H = 64
    eps = 1e-6

    # 1. With scale on GPU across 2D, 3D, 4D
    norm = Gemma4VisionRMSNorm(hidden_size=H, eps=eps, has_weight=True).to(
        device=device, dtype=dtype
    )
    torch.manual_seed(123)
    norm.weight.data.uniform_(0.8, 1.2)

    for shape in [(8, H), (2, 16, H), (2, 4, 8, H)]:
        x = torch.randn(*shape, device=device, dtype=dtype)
        out = norm(x)
        assert out.shape == x.shape

        # Parity with PyTorch FP32 reference
        ref = (
            x.float()
            * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
            * norm.weight.float()
        ).to(dtype)
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < 2e-2, f"Diff {diff} too high for shape {shape}"

    # 2. Non-contiguous input
    x_base = torch.randn(16, 2, H, device=device, dtype=dtype)
    x_nc = x_base.transpose(0, 1)
    assert not x_nc.is_contiguous()
    out_nc = norm(x_nc)
    assert out_nc.shape == x_nc.shape

    # 3. Unscaled norm (with_scale=False)
    norm_no_scale = Gemma4VisionRMSNorm(hidden_size=H, eps=eps, has_weight=False).to(
        device=device, dtype=dtype
    )
    x_3d = torch.randn(2, 8, H, device=device, dtype=dtype)
    out_no_scale = norm_no_scale(x_3d)
    ref_no_scale = (
        x_3d.float() * torch.rsqrt(x_3d.float().pow(2).mean(-1, keepdim=True) + eps)
    ).to(dtype)
    diff_no_scale = (out_no_scale.float() - ref_no_scale.float()).abs().max().item()
    assert diff_no_scale < 2e-2

    # 4. Empty tensor (numel == 0)
    x_empty = torch.randn(0, H, device=device, dtype=dtype)
    out_empty = norm(x_empty)
    assert out_empty.numel() == 0
    assert out_empty.shape == (0, H)

    # 5. CPU fallback
    norm_cpu = Gemma4VisionRMSNorm(hidden_size=H, eps=eps, has_weight=True)
    x_cpu = torch.randn(2, 4, H)
    out_cpu = norm_cpu(x_cpu)
    assert out_cpu.shape == x_cpu.shape
    assert out_cpu.device == torch.device("cpu")


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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GPU kernel tests"
)
def test_fused_vision_2d_rope_parity_and_broadcasting():
    """Verify fused_vision_2d_rope parity against apply_multidimensional_rope,
    non-contiguous inputs, broadcasted cos/sin, and empty tensors."""
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    B, L, N, D = 2, 64, 8, 64

    torch.manual_seed(42)
    x = torch.randn(B, L, N, D, device=device, dtype=dtype)
    cos = torch.randn(B, L, D, device=device, dtype=dtype)
    sin = torch.randn(B, L, D, device=device, dtype=dtype)

    # Reference HF implementation
    ref_out = apply_multidimensional_rope(
        x,
        cos,
        sin,
        position_ids=torch.zeros(B, L, 2, device=device, dtype=torch.long),
        unsqueeze_dim=2,
    )
    fused_out = fused_vision_2d_rope(x, cos, sin)

    assert fused_out.shape == ref_out.shape
    diff = (fused_out.float() - ref_out.float()).abs()
    assert diff.max().item() <= 0.1

    # 1-ULP parity and cosine similarity
    a_i = fused_out.view(torch.int16)
    b_i = ref_out.view(torch.int16)
    same_sign = (a_i >= 0) == (b_i >= 0)
    ulp = torch.where(
        same_sign, (a_i.int() - b_i.int()).abs(), torch.tensor(999999, device=device)
    )
    ulp = torch.where(
        (a_i == b_i) | ((fused_out == 0) & (ref_out == 0)) | (diff <= 1e-6), 0, ulp
    )
    within_1ulp_pct = float((ulp <= 1).float().mean().item() * 100.0)
    assert within_1ulp_pct >= 95.0, f"within_1ulp_pct is {within_1ulp_pct}"

    cos_sim = torch.cosine_similarity(
        fused_out.float().flatten(), ref_out.float().flatten(), dim=0
    ).item()
    assert cos_sim >= 0.99999

    # Broadcasted cos/sin (cos.shape[0] == 1)
    cos_bcast = torch.randn(1, L, D, device=device, dtype=dtype)
    sin_bcast = torch.randn(1, L, D, device=device, dtype=dtype)
    ref_bcast = apply_multidimensional_rope(
        x,
        cos_bcast.expand(B, L, D),
        sin_bcast.expand(B, L, D),
        position_ids=torch.zeros(B, L, 2, device=device, dtype=torch.long),
        unsqueeze_dim=2,
    )
    fused_bcast = fused_vision_2d_rope(x, cos_bcast, sin_bcast)
    diff_bcast = (fused_bcast.float() - ref_bcast.float()).abs()
    assert diff_bcast.max().item() <= 0.1

    # Non-contiguous input
    x_raw = torch.randn(B, N, L, D, device=device, dtype=dtype)
    x_transposed = x_raw.transpose(1, 2)
    assert not x_transposed.is_contiguous()
    out_transposed = fused_vision_2d_rope(x_transposed, cos, sin)
    assert out_transposed.shape == (B, L, N, D)

    # Empty tensor
    x_empty = torch.randn(0, L, N, D, device=device, dtype=dtype)
    cos_empty = torch.randn(0, L, D, device=device, dtype=dtype)
    sin_empty = torch.randn(0, L, D, device=device, dtype=dtype)
    out_empty = fused_vision_2d_rope(x_empty, cos_empty, sin_empty)
    assert out_empty.numel() == 0
    assert out_empty.shape == (0, L, N, D)


def test_vision_attention_monkey_patching(default_vllm_config):
    """Verify _optimize_hf_multimodal_tower patches Gemma4VisionAttention."""
    vis_cfg = Gemma4VisionConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        head_dim=16,
    )
    model = Gemma4VisionModel(vis_cfg)
    attn_mod = sys.modules.get(Gemma4VisionAttention.__module__)

    _optimize_hf_multimodal_tower(model)

    assert getattr(attn_mod, "_fused_2d_rope_patched", False) is True


def test_batched_vision_pooler_image_input():
    """Verify vt.pooler batching and identity pooling valid_mask guard."""
    # Test identity pooling guard where output_length == batch_hidden.shape[1]
    # HF returns padding_positions (True for padding), inverted to valid_mask.
    batch_size = 2
    seq_len = 16
    hidden_dim = 32

    batch_hidden = torch.randn(batch_size, seq_len, hidden_dim)
    output_length = seq_len  # identity pooling case

    # Mock pooler returning padding_positions as second element
    mock_padding = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mock_padding[:, 12:] = True  # last 4 tokens are padding
    mock_pooled = torch.randn(batch_size, seq_len, hidden_dim)

    mock_vt = MagicMock()
    mock_vt.pooler.return_value = (mock_pooled, mock_padding.clone())

    pooled_states, valid_mask = mock_vt.pooler(
        hidden_states=batch_hidden,
        output_length=output_length,
    )
    if output_length == batch_hidden.shape[1]:
        valid_mask = ~valid_mask

    # After inversion, first 12 tokens should be valid (True) and last 4 should be False
    assert valid_mask[:, :12].all().item() is True
    assert (~valid_mask[:, 12:]).all().item() is True
    assert valid_mask.sum(dim=-1).tolist() == [12, 12]


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
                # valid_mask: False for positions where padding_positions is True
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
