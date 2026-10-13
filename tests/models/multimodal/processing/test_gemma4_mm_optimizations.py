# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Gemma 4 multimodal performance optimizations (PR 6).

Covers:
- Gemma4VisionRMSNorm weight preservation and hidden_size resolution
- Gemma4VisionRMSNorm guard fallbacks (dtype, variance_size_override, strided weight)
- _optimize_hf_multimodal_tower Gemma4VisionAttention component swap
- Batched vision pooler for image and video inputs (including identity pooling)
- _get_device_memory_info OOT/meta/CPU safety and error propagation
- _clear_mm_prefix_for_full_attn_layers memoization and clearing logic
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
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

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    Gemma4VisionAttention,
    Gemma4VisionRMSNorm,
    _get_device_memory_info,
    _optimize_hf_multimodal_tower,
    _pool_vision_hidden_states,
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


def test_gemma4_vision_rmsnorm_forward_without_active_vllm_config():
    """Verify Gemma4VisionRMSNorm.forward works outside set_current_vllm_config()
    (as in GPUWorker.execute_model()), defaults to vllm_c even when
    custom_ops=['none'], and honors '-rms_norm'."""
    from vllm import envs, ir
    from vllm.config import (
        CompilationConfig,
        VllmConfig,
        get_current_vllm_config,
        set_current_vllm_config,
    )

    hidden_size = 64

    class DummyParent(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = hidden_size
            self.norm = Gemma4RMSNorm(hidden_size, eps=1e-6)

    none_cfg = VllmConfig(compilation_config=CompilationConfig(custom_ops=["none"]))
    assert "none" in none_cfg.compilation_config.custom_ops

    with set_current_vllm_config(none_cfg):
        cpu_model = DummyParent()
        _optimize_hf_multimodal_tower(cpu_model)
        cuda_model = None
        if torch.cuda.is_available():
            cuda_model = DummyParent().to(device="cuda:0", dtype=torch.bfloat16)
            _optimize_hf_multimodal_tower(cuda_model)

    assert isinstance(cpu_model.norm, Gemma4VisionRMSNorm)
    assert cpu_model.norm._allow_vllm_c is True

    with set_current_vllm_config(
        VllmConfig(compilation_config=CompilationConfig(custom_ops=["-rms_norm"]))
    ):
        disabled_model = DummyParent()
        _optimize_hf_multimodal_tower(disabled_model)
    assert isinstance(disabled_model.norm, Gemma4VisionRMSNorm)
    assert disabled_model.norm._allow_vllm_c is False

    with pytest.raises(AssertionError):
        get_current_vllm_config()

    cpu_x = torch.randn(2, 8, hidden_size)
    assert cpu_model.norm(cpu_x).shape == cpu_x.shape
    if cuda_model is not None:
        assert isinstance(cuda_model.norm, Gemma4VisionRMSNorm)
        assert cuda_model.norm._allow_vllm_c is True
        cuda_x = torch.randn(2, 8, hidden_size, device="cuda:0", dtype=torch.bfloat16)
        vllm_c_impl = ir.ops.rms_norm.impls.get("vllm_c")
        if (
            vllm_c_impl is not None
            and vllm_c_impl.supported
            and not envs.VLLM_BATCH_INVARIANT
        ):
            with patch.object(
                vllm_c_impl, "impl_fn", wraps=vllm_c_impl.impl_fn
            ) as spy_impl:
                cuda_out = cuda_model.norm(cuda_x)
            assert spy_impl.call_count == 1
        else:
            cuda_out = cuda_model.norm(cuda_x)
        assert cuda_out.shape == cuda_x.shape
        assert cuda_out.is_cuda


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
    assert attn.v_norm(torch.randn(2, 4, 48)).shape == (2, 4, 48)

    layer = DummyLayer()
    _optimize_hf_multimodal_tower(layer)
    assert isinstance(layer.unscaled_norm, Gemma4VisionRMSNorm)
    assert layer.unscaled_norm.has_weight is False
    assert layer.unscaled_norm.hidden_size == 96
    assert layer.unscaled_norm(torch.randn(2, 4, 96)).shape == (2, 4, 96)


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
    -1 padding, empty batches, multiple videos, and DP-encoder dispatch."""

    class DummyPatchEmbedder(nn.Module):
        def __init__(self, hidden_dim=64):
            super().__init__()
            self.input_proj = MagicMock(
                side_effect=lambda pv: torch.randn(
                    *pv.shape[:-1], hidden_dim, dtype=pv.dtype, device=pv.device
                )
            )
            self._position_embeddings = MagicMock(return_value=0.0)

    class DummyVisionTower(nn.Module):
        def __init__(self, hidden_dim=64, pool_k=2):
            super().__init__()
            self.config = MagicMock()
            self.config.standardize = False
            self.config.position_embedding_size = 32
            self.hidden_dim = hidden_dim
            self.pool_k = pool_k
            self.patch_embedder = DummyPatchEmbedder(hidden_dim=hidden_dim)
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
                # Otherwise standard pooling reduces k^2 patches per output token
                valid = ~padding_positions.reshape(B, output_length, -1).all(dim=-1)
                return pooled, valid

            self.pooler = MagicMock(side_effect=mock_pooler)

    def make_dummy_model(pool_k=2) -> Gemma4ForConditionalGeneration:
        model = object.__new__(Gemma4ForConditionalGeneration)  # type: ignore[type-abstract]
        nn.Module.__init__(model)
        model.config = MagicMock()
        model.config.vision_config = MagicMock()
        model.config.vision_config.pooling_kernel_size = pool_k
        model.config.vision_config.position_embedding_size = 32
        model.config.vision_config.default_output_length = 70
        model.config.vision_config.num_channels = 3
        model.vision_tower = DummyVisionTower(hidden_dim=64, pool_k=pool_k)
        model.vision_input_norm = MagicMock(side_effect=lambda pv, dtype: pv.to(dtype))
        model._enable_mm_lora = False
        model.use_data_parallel = False
        model.model_dtype = torch.bfloat16
        model.embed_vision = MagicMock(side_effect=lambda inputs_embeds: inputs_embeds)
        model._encoder_chunk = lambda p, f, t, pos: 10  # type: ignore[method-assign]
        return model

    # 1. Image input: empty batch
    dummy = make_dummy_model(pool_k=2)
    assert (
        dummy._process_image_input({"pixel_values": [], "pixel_position_ids": []}) == []
    )

    # 2. Image input: single resolution with -1 padding
    pv_1 = torch.randn(16, 768)
    pp_1 = torch.zeros(16, 2, dtype=torch.long)
    pp_1[12:, :] = -1  # 4 padding tokens
    pv_2 = torch.randn(16, 768)
    pp_2 = torch.zeros(16, 2, dtype=torch.long)
    pp_2[14:, :] = -1  # 2 padding tokens

    out_single = dummy._process_image_input(
        {
            "pixel_values": [pv_1, pv_2],
            "pixel_position_ids": [pp_1, pp_2],
        }
    )
    assert len(out_single) == 2
    assert isinstance(out_single[0], torch.Tensor)
    assert isinstance(out_single[1], torch.Tensor)
    assert out_single[0].dtype == torch.bfloat16

    # 3. Image input: interleaved multi-resolution buckets preserve order
    pv_3 = torch.randn(16, 768)
    pp_3 = torch.zeros(16, 2, dtype=torch.long)
    pp_3[8:, :] = -1  # 8 valid patches -> 2 pooled tokens (vs 3 for pv_1)
    pv_bucket2 = torch.randn(32, 768)
    pp_bucket2 = torch.zeros(32, 2, dtype=torch.long)  # 32 valid -> 8 pooled tokens
    out_multi = dummy._process_image_input(
        {
            "pixel_values": [pv_1, pv_bucket2, pv_3],
            "pixel_position_ids": [pp_1, pp_bucket2, pp_3],
        }
    )
    assert [t.shape[0] for t in out_multi] == [3, 8, 2]

    # 3b. Image & video identity pooling (output_length == seq_len, pool_k=1)
    dummy_identity = make_dummy_model(pool_k=1)
    dummy_identity.config.vision_config.default_output_length = 16
    out_identity = dummy_identity._process_image_input(
        {
            "pixel_values": [pv_1, pv_2],
            "pixel_position_ids": [pp_1, pp_2],
        }
    )
    assert [t.shape[0] for t in out_identity] == [12, 14]

    # Forward-compat: even if HF fixes identity pooling to return ~padding_positions,
    # using ~padding_positions directly avoids double-inversion.
    def fixed_hf_pooler(
        hidden_states, pixel_position_ids, padding_positions, output_length
    ):
        return (
            torch.randn(hidden_states.shape[0], output_length, 64),
            ~padding_positions[:, :output_length],
        )

    dummy_identity.vision_tower.pooler.side_effect = fixed_hf_pooler
    out_identity_fixed_hf = dummy_identity._process_image_input(
        {
            "pixel_values": [pv_1, pv_2],
            "pixel_position_ids": [pp_1, pp_2],
        }
    )
    assert [t.shape[0] for t in out_identity_fixed_hf] == [12, 14]

    # 4. Video input: empty batch
    assert (
        dummy._process_video_input(
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
    pp_vid[:, 12:, :] = -1  # 12 valid -> 3 pooled tokens per frame (pool_k=2)
    fc = [2, 3]

    out_vid = dummy._process_video_input(
        {
            "pixel_values_videos": pv_vid,
            "pixel_position_ids_videos": pp_vid,
            "video_frame_counts": fc,
        }
    )
    assert [t.shape[0] for t in out_vid] == [6, 9]
    assert out_vid[0].dtype == torch.bfloat16

    # 5b. Video input: padded frame shape (66 valid out of 70 soft tokens, pool_k=2)
    pv_vid_70 = torch.randn(2, 280, 768)
    pp_vid_70 = torch.zeros(2, 280, 2, dtype=torch.long)
    pp_vid_70[:, 264:, :] = -1  # 16 pad patches = 4 pad soft tokens -> 66 valid
    out_vid_70 = dummy._process_video_input(
        {
            "pixel_values_videos": pv_vid_70,
            "pixel_position_ids_videos": pp_vid_70,
            "video_frame_counts": [1, 1],
        }
    )
    assert [t.shape[0] for t in out_vid_70] == [66, 66]

    # 5c. Video input: identity pooling (pool_k=1)
    out_vid_identity = dummy_identity._process_video_input(
        {
            "pixel_values_videos": pv_vid,
            "pixel_position_ids_videos": pp_vid,
            "video_frame_counts": fc,
        }
    )
    assert [t.shape[0] for t in out_vid_identity] == [24, 36]

    # 5d. Memory-bounded pooler chunking splits large/tight-budget batches
    dummy_chunk = make_dummy_model(pool_k=2)
    seq_len_c, out_len_c = 16, 4
    per_item_bytes = seq_len_c * out_len_c * 12
    # total_bytes // 10 = 2 * per_item_bytes -> pool_chunk = 2 for 5 frames
    flat_chunked, lens_chunked = _pool_vision_hidden_states(
        dummy_chunk.vision_tower,
        torch.randn(5, seq_len_c, 64),
        pp_vid,
        (pp_vid == -1).all(dim=-1),
        output_length=out_len_c,
        free_bytes=100 * per_item_bytes,
        total_bytes=20 * per_item_bytes,
    )
    assert dummy_chunk.vision_tower.pooler.call_count == 3
    assert lens_chunked == [3, 3, 3, 3, 3]
    assert flat_chunked.shape[0] == 15

    # Empty batch edge guard
    flat_empty, lens_empty = _pool_vision_hidden_states(
        dummy_chunk.vision_tower,
        torch.empty(0, seq_len_c, 64),
        torch.empty(0, seq_len_c, 2, dtype=torch.long),
        torch.empty(0, seq_len_c, dtype=torch.bool),
        output_length=out_len_c,
        free_bytes=100 * per_item_bytes,
        total_bytes=20 * per_item_bytes,
    )
    assert flat_empty.shape == (0, 64)
    assert lens_empty == []

    # 6. DP-encoder mode dispatches cleanly to batched local pooler
    dummy.use_data_parallel = True
    dummy._dp_execute_and_gather = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda raw_kwargs, process_fn: process_fn(raw_kwargs)
    )
    out_dp_img = dummy._process_image_input(
        {
            "pixel_values": [pv_1, pv_2],
            "pixel_position_ids": [pp_1, pp_2],
        }
    )
    assert len(out_dp_img) == 2
    out_dp_vid = dummy._process_video_input(
        {
            "pixel_values_videos": pv_vid,
            "pixel_position_ids_videos": pp_vid,
            "video_frame_counts": fc,
        }
    )
    assert [t.shape[0] for t in out_dp_vid] == [6, 9]


def test_borrowed_methods_without_subclassing():
    """Verify models that borrow Gemma4ForConditionalGeneration vision/video
    methods without subclassing (DiffusionGemma, EmbeddingGemma2) run cleanly."""

    class DummyPatchEmbedder(nn.Module):
        def __init__(self, hidden_dim=64):
            super().__init__()
            self.input_proj = MagicMock(
                side_effect=lambda pv: torch.randn(
                    *pv.shape[:-1], hidden_dim, dtype=pv.dtype, device=pv.device
                )
            )
            self._position_embeddings = MagicMock(return_value=0.0)

    class DummyVisionTower(nn.Module):
        def __init__(self, hidden_dim=64):
            super().__init__()
            self.config = MagicMock()
            self.config.standardize = False
            self.config.position_embedding_size = 32
            self.patch_embedder = DummyPatchEmbedder(hidden_dim=hidden_dim)
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
                valid = ~padding_positions.reshape(B, output_length, -1).all(dim=-1)
                return pooled, valid

            self.pooler = MagicMock(side_effect=mock_pooler)

    class BorrowedModel:
        _parse_and_validate_image_input = (
            Gemma4ForConditionalGeneration._parse_and_validate_image_input
        )
        _parse_and_validate_video_input = (
            Gemma4ForConditionalGeneration._parse_and_validate_video_input
        )
        _parse_and_validate_multimodal_inputs = (
            Gemma4ForConditionalGeneration._parse_and_validate_multimodal_inputs
        )
        _encoder_chunk = staticmethod(Gemma4ForConditionalGeneration._encoder_chunk)
        _get_max_soft_tokens = Gemma4ForConditionalGeneration._get_max_soft_tokens
        _embed_patches = Gemma4ForConditionalGeneration._embed_patches
        _normalize_pixel_values = Gemma4ForConditionalGeneration._normalize_pixel_values
        _process_image_input = Gemma4ForConditionalGeneration._process_image_input_local
        _process_video_input = Gemma4ForConditionalGeneration._process_video_input_local
        _process_image_input_local = (
            Gemma4ForConditionalGeneration._process_image_input_local
        )
        _process_video_input_local = (
            Gemma4ForConditionalGeneration._process_video_input_local
        )
        embed_multimodal = Gemma4ForConditionalGeneration.embed_multimodal

        def __init__(self):
            self.config = MagicMock()
            self.config.vision_config = MagicMock()
            self.config.vision_config.pooling_kernel_size = 2
            self.config.vision_config.position_embedding_size = 32
            self.config.vision_config.default_output_length = 70
            self.config.vision_config.num_channels = 3
            self.vision_tower = DummyVisionTower(hidden_dim=64)
            self.vision_input_norm = MagicMock(
                side_effect=lambda pv, dtype: pv.to(dtype)
            )
            self._enable_mm_lora = False
            self.model_dtype = torch.bfloat16
            self.embed_vision = MagicMock(
                side_effect=lambda inputs_embeds: inputs_embeds
            )

    model: Any = BorrowedModel()
    assert not isinstance(model, Gemma4ForConditionalGeneration)
    assert not hasattr(model, "_get_device_memory_info")
    assert not hasattr(model, "_pool_vision_hidden_states")

    # 1. Single-resolution images via embed_multimodal & _process_image_input_local
    pv_a = torch.randn(16, 768)
    pp_a = torch.zeros(16, 2, dtype=torch.long)
    pp_a[12:, :] = -1
    pv_b = torch.randn(16, 768)
    pp_b = torch.zeros(16, 2, dtype=torch.long)
    pp_b[8:, :] = -1

    embs_single = model.embed_multimodal(
        pixel_values=[pv_a, pv_b],
        pixel_position_ids=[pp_a, pp_b],
    )
    assert [t.shape[0] for t in embs_single] == [3, 2]

    # 2. Multi-resolution interleaved images
    pv_wide = torch.randn(32, 768)
    pp_wide = torch.zeros(32, 2, dtype=torch.long)
    embs_multi = model._process_image_input_local(
        {
            "pixel_values": [pv_a, pv_wide, pv_b],
            "pixel_position_ids": [pp_a, pp_wide, pp_b],
        }
    )
    assert [t.shape[0] for t in embs_multi] == [3, 8, 2]

    # 3. Chunked pooling under tight memory budget
    model.vision_tower.pooler.reset_mock()
    per_item_bytes = 16 * 4 * 12
    with patch(
        "vllm.model_executor.models.gemma4_mm._get_device_memory_info",
        return_value=(100 * per_item_bytes, 20 * per_item_bytes),
    ):
        embs_chunked = model._process_image_input_local(
            {
                "pixel_values": [pv_a] * 5,
                "pixel_position_ids": [pp_a] * 5,
            }
        )
    assert [t.shape[0] for t in embs_chunked] == [3, 3, 3, 3, 3]
    assert model.vision_tower.pooler.call_count == 3

    # 4. Video input via embed_multimodal & _process_video_input_local
    pv_vid = torch.randn(5, 16, 768)
    pp_vid = torch.zeros(5, 16, 2, dtype=torch.long)
    pp_vid[:, 12:, :] = -1
    embs_vid = model.embed_multimodal(
        pixel_values_videos=pv_vid,
        pixel_position_ids_videos=pp_vid,
        video_frame_counts=[2, 3],
    )
    assert [t.shape[0] for t in embs_vid] == [6, 9]


def test_get_device_memory_info(monkeypatch: pytest.MonkeyPatch):
    """Verify _get_device_memory_info handles CPU/meta, missing/mismatched
    accelerators, OOT accelerators, NotImplementedError, and hardware errors."""
    sentinel = (1 << 60, 1 << 60)

    # 1. CPU and meta short-circuit without touching torch.accelerator
    mock_cur_acc = MagicMock()
    mock_get_mem = MagicMock()
    monkeypatch.setattr(torch.accelerator, "current_accelerator", mock_cur_acc)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", mock_get_mem)

    assert _get_device_memory_info(torch.device("cpu")) == sentinel
    assert _get_device_memory_info(torch.device("meta")) == sentinel
    mock_cur_acc.assert_not_called()
    mock_get_mem.assert_not_called()

    # 2. current_accelerator(check_available=True) returns None
    mock_cur_acc = MagicMock(return_value=None)
    mock_get_mem = MagicMock()
    monkeypatch.setattr(torch.accelerator, "current_accelerator", mock_cur_acc)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", mock_get_mem)

    assert _get_device_memory_info(torch.device("cuda:0")) == sentinel
    mock_cur_acc.assert_called_once_with(check_available=True)
    mock_get_mem.assert_not_called()

    # 3. Device type differs from active accelerator (e.g. cuda vs xpu)
    mock_cur_acc = MagicMock(return_value=SimpleNamespace(type="xpu"))
    mock_get_mem = MagicMock()
    monkeypatch.setattr(torch.accelerator, "current_accelerator", mock_cur_acc)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", mock_get_mem)

    assert _get_device_memory_info(torch.device("cuda:0")) == sentinel
    mock_cur_acc.assert_called_once_with(check_available=True)
    mock_get_mem.assert_not_called()

    # 4. Simulated OOT accelerator matching device.type returns exact (free, total)
    oot_device: Any = SimpleNamespace(type="npu")
    mock_cur_acc = MagicMock(return_value=SimpleNamespace(type="npu"))
    mock_get_mem = MagicMock(return_value=(1234, 5678))
    monkeypatch.setattr(torch.accelerator, "current_accelerator", mock_cur_acc)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", mock_get_mem)

    assert _get_device_memory_info(oot_device) == (1234, 5678)
    mock_cur_acc.assert_called_once_with(check_available=True)
    mock_get_mem.assert_called_once_with(oot_device)

    # 5. OOT accelerator raising NotImplementedError falls back to sentinel
    mock_get_mem = MagicMock(side_effect=NotImplementedError("unsupported"))
    monkeypatch.setattr(torch.accelerator, "get_memory_info", mock_get_mem)

    assert _get_device_memory_info(oot_device) == sentinel
    mock_get_mem.assert_called_once_with(oot_device)

    # 6. Non-NotImplementedError (e.g. sticky hardware RuntimeError) propagates
    mock_get_mem = MagicMock(side_effect=RuntimeError("device error"))
    monkeypatch.setattr(torch.accelerator, "get_memory_info", mock_get_mem)

    with pytest.raises(RuntimeError, match="device error"):
        _get_device_memory_info(oot_device)


def test_gemma4_vision_rmsnorm_guard_fallbacks(
    default_vllm_config, monkeypatch: pytest.MonkeyPatch
):
    """Verify Gemma4VisionRMSNorm falls back to native for float64 inputs,
    variance_size_override, and non-contiguous strided weights."""
    from vllm import envs, ir

    hidden_size = 64
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))

    vllm_c_impl = ir.ops.rms_norm.impls.get("vllm_c")
    spy_impl = None
    if vllm_c_impl is not None:
        spy_impl = MagicMock(wraps=vllm_c_impl.impl_fn)
        monkeypatch.setattr(vllm_c_impl, "impl_fn", spy_impl)

    with ir.ops.rms_norm.set_priority(["native"]):
        for device in devices:
            # 1. float64 input falls back to native and matches reference RMSNorm
            if spy_impl is not None:
                spy_impl.reset_mock()
            torch.manual_seed(0)
            norm_f64 = Gemma4VisionRMSNorm(
                hidden_size, eps=1e-6, dtype=torch.float64
            ).to(device=device)
            norm_f64.weight.data.copy_(
                torch.randn(hidden_size, device=device, dtype=torch.float64)
            )
            x_f64 = torch.randn(2, 8, hidden_size, device=device, dtype=torch.float64)
            out_f64 = norm_f64(x_f64)
            ref_f64 = RMSNorm.forward_native(norm_f64, x_f64)
            assert isinstance(ref_f64, torch.Tensor)
            assert out_f64.dtype == torch.float64
            assert torch.allclose(out_f64, ref_f64)
            if spy_impl is not None:
                assert spy_impl.call_count == 0

            # 2. var_hidden_size != hidden_size falls back to native and differs
            #    from full-width RMSNorm
            if not envs.VLLM_BATCH_INVARIANT:
                if spy_impl is not None:
                    spy_impl.reset_mock()
                var_hidden = hidden_size // 2
                norm_var = Gemma4VisionRMSNorm(
                    hidden_size,
                    eps=1e-6,
                    var_hidden_size=var_hidden,
                    dtype=torch.float32,
                ).to(device=device)
                norm_var.weight.data.copy_(
                    torch.randn(hidden_size, device=device, dtype=torch.float32)
                )
                x_var = torch.randn(
                    2, 8, hidden_size, device=device, dtype=torch.float32
                )
                x_var[..., :var_hidden] *= 0.1
                x_var[..., var_hidden:] *= 10.0

                out_var = norm_var(x_var)
                ref_var = RMSNorm.forward_native(norm_var, x_var)
                assert isinstance(ref_var, torch.Tensor)
                assert torch.allclose(out_var, ref_var)

                full_norm = RMSNorm(hidden_size, eps=1e-6, dtype=torch.float32).to(
                    device=device
                )
                full_norm.weight.data.copy_(norm_var.weight.data)
                full_out = full_norm.forward_native(x_var)
                assert isinstance(full_out, torch.Tensor)
                assert not torch.allclose(out_var, full_out, atol=1e-2, rtol=1e-2)
                if spy_impl is not None:
                    assert spy_impl.call_count == 0

            # 3. Non-contiguous strided weight falls back to native and matches
            #    reference RMSNorm
            if spy_impl is not None:
                spy_impl.reset_mock()
            norm_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            norm_strided = Gemma4VisionRMSNorm(
                hidden_size, eps=1e-6, dtype=norm_dtype
            ).to(device=device)
            strided_weight = torch.randn(
                hidden_size * 2, device=device, dtype=norm_dtype
            )[::2]
            assert not strided_weight.is_contiguous()
            norm_strided.weight = nn.Parameter(strided_weight)
            assert not norm_strided.weight.is_contiguous()

            x_strided = torch.randn(2, 8, hidden_size, device=device, dtype=norm_dtype)
            out_strided = norm_strided(x_strided)
            ref_strided = RMSNorm.forward_native(norm_strided, x_strided)
            assert isinstance(ref_strided, torch.Tensor)
            assert torch.allclose(out_strided, ref_strided)
            if spy_impl is not None:
                assert spy_impl.call_count == 0
