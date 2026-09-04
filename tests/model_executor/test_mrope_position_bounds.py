# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MRoPE position bounds checking (security fix).

Validates that:
1. MRotaryEmbedding rejects out-of-bounds positions with ValueError
2. Model get_mrope_input_positions clamps temporal positions when t_factor
   would produce indices exceeding the cache size
3. second_per_grid_ts is floored to prevent near-zero fps from producing
   unbounded temporal multipliers
"""

from dataclasses import dataclass, field

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLForConditionalGeneration,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True, scope="module")
def _force_cpu_default_device():
    original = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(original)


# --- MRotaryEmbedding bounds check tests ---


class TestMRotaryEmbeddingBoundsCheck:
    """Test that MRotaryEmbedding raises ValueError for OOB positions."""

    def _make_mrope(self, max_position_embeddings=128):
        return MRotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=max_position_embeddings,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
            mrope_section=[16, 8, 8],
        )

    def test_valid_positions_pass(self, default_vllm_config):
        mrope = self._make_mrope(max_position_embeddings=128)
        # Cache size = 128 * 4 = 512
        positions = torch.randint(0, 511, (3, 10))
        query = torch.randn(10, 64)
        key = torch.randn(10, 64)
        q, k = mrope.forward_native(positions, query, key)
        assert q.shape == query.shape
        assert k.shape == key.shape

    def test_oob_positions_raise_valueerror(self, default_vllm_config):
        mrope = self._make_mrope(max_position_embeddings=128)
        # Cache size = 128 * 4 = 512; position 1000 is out of bounds
        positions = torch.tensor([[0, 1, 1000], [0, 1, 2], [0, 1, 2]])
        query = torch.randn(3, 64)
        key = torch.randn(3, 64)
        with pytest.raises(ValueError, match="exceeds the rotary embedding cache"):
            mrope.forward_native(positions, query, key)

    def test_boundary_position_raises(self, default_vllm_config):
        mrope = self._make_mrope(max_position_embeddings=128)
        # Exactly at cache size (512) should fail
        positions = torch.tensor([[512], [0], [0]])
        query = torch.randn(1, 64)
        key = torch.randn(1, 64)
        with pytest.raises(ValueError, match="exceeds the rotary embedding cache"):
            mrope.forward_native(positions, query, key)

    def test_negative_positions_raise_valueerror(self, default_vllm_config):
        mrope = self._make_mrope(max_position_embeddings=128)
        positions = torch.tensor([[0, -1, 2], [0, 1, 2], [0, 1, 2]])
        query = torch.randn(3, 64)
        key = torch.randn(3, 64)
        with pytest.raises(ValueError, match="is negative"):
            mrope.forward_native(positions, query, key)

    def test_max_valid_position_passes(self, default_vllm_config):
        mrope = self._make_mrope(max_position_embeddings=128)
        # max_position_embeddings * 4 - 1 = 511 should pass
        positions = torch.tensor([[511], [0], [0]])
        query = torch.randn(1, 64)
        key = torch.randn(1, 64)
        q, k = mrope.forward_native(positions, query, key)
        assert q.shape == query.shape


# --- Model position clamping tests ---


@dataclass
class DummyVisionConfig:
    spatial_merge_size: int = 2
    tokens_per_second: float = 1.0
    patch_size: int = 14


@dataclass
class DummyConfig:
    image_token_id: int = 151655
    video_token_id: int = 151654
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    max_position_embeddings: int = 128
    vision_config: DummyVisionConfig = field(default_factory=DummyVisionConfig)


def _make_qwen25vl_model(config=None):
    if config is None:
        config = DummyConfig()
    model = object.__new__(Qwen2_5_VLForConditionalGeneration)
    model.config = config
    return model


def _make_qwen2vl_model(config=None):
    if config is None:
        config = DummyConfig()
    model = object.__new__(Qwen2VLForConditionalGeneration)
    model.config = config
    return model


def _make_video_feature(
    offset: int, grid_thw: tuple[int, int, int], second_per_grid_ts: float
) -> MultiModalFeatureSpec:
    data = {
        "video_grid_thw": MultiModalFieldElem(
            data=torch.tensor(grid_thw),
            field=None,
        ),
        "second_per_grid_ts": MultiModalFieldElem(
            data=torch.tensor(second_per_grid_ts),
            field=None,
        ),
    }
    t, h, w = grid_thw
    spatial_merge_size = 2
    length = t * (h // spatial_merge_size) * (w // spatial_merge_size)
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem(data),
        modality="video",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


class TestQwen25VLPositionClamping:
    """Test that Qwen2.5-VL clamps positions when t_factor is extreme."""

    def test_normal_t_factor(self):
        model = _make_qwen25vl_model()
        # Normal case: second_per_grid_ts=1.0, tokens_per_second=1.0
        # t_factor = 1.0, no clamping needed
        feature = _make_video_feature(
            offset=0, grid_thw=(4, 4, 4), second_per_grid_ts=1.0
        )
        input_tokens = list(range(4 * 2 * 2))  # t * (h/merge) * (w/merge)
        positions, delta = model.get_mrope_input_positions(input_tokens, [feature])
        assert positions.shape == (3, len(input_tokens))
        # All positions should be within bounds
        max_pos = model.config.max_position_embeddings * 4
        assert positions.max().item() < max_pos

    def test_extreme_t_factor_is_clamped(self):
        model = _make_qwen25vl_model()
        # Extreme second_per_grid_ts simulating near-zero fps
        # t_factor = 1e6 * 1.0 = 1e6, would produce indices >> cache size
        feature = _make_video_feature(
            offset=0, grid_thw=(4, 4, 4), second_per_grid_ts=1e6
        )
        input_tokens = list(range(4 * 2 * 2))
        positions, delta = model.get_mrope_input_positions(input_tokens, [feature])
        max_pos = model.config.max_position_embeddings * 4
        assert positions.max().item() < max_pos

    def test_second_per_grid_ts_floor(self):
        """Verify near-zero second_per_grid_ts is floored to 1e-3."""
        model = _make_qwen25vl_model()
        feature = _make_video_feature(
            offset=0, grid_thw=(4, 4, 4), second_per_grid_ts=1e-15
        )
        input_tokens = list(range(4 * 2 * 2))
        # Should not crash due to extreme values
        positions, delta = model.get_mrope_input_positions(input_tokens, [feature])
        assert positions.shape == (3, len(input_tokens))


class TestQwen2VLPositionClamping:
    """Test that Qwen2-VL has the same protection."""

    def test_extreme_t_factor_is_clamped(self):
        model = _make_qwen2vl_model()
        feature = _make_video_feature(
            offset=0, grid_thw=(4, 4, 4), second_per_grid_ts=1e6
        )
        input_tokens = list(range(4 * 2 * 2))
        positions, delta = model.get_mrope_input_positions(input_tokens, [feature])
        max_pos = model.config.max_position_embeddings * 4
        assert positions.max().item() < max_pos


class TestSecondPerGridTsValidation:
    """Test that the floor on second_per_grid_ts works across models."""

    @pytest.mark.parametrize(
        "bad_value",
        [0.0, -1.0, 1e-15, 1e-100, float("inf"), float("-inf"), float("nan")],
    )
    def test_near_zero_fps_does_not_crash(self, bad_value):
        model = _make_qwen25vl_model()
        feature = _make_video_feature(
            offset=0, grid_thw=(2, 4, 4), second_per_grid_ts=bad_value
        )
        input_tokens = list(range(2 * 2 * 2))
        positions, delta = model.get_mrope_input_positions(input_tokens, [feature])
        max_pos = model.config.max_position_embeddings * 4
        assert positions.max().item() < max_pos

    def test_large_second_per_grid_ts_clamped(self):
        """Simulate fps near zero → second_per_grid_ts near infinity."""
        config = DummyConfig()
        config.vision_config.tokens_per_second = 25.0
        model = _make_qwen25vl_model(config)
        # second_per_grid_ts=1000 with tokens_per_second=25 → t_factor=25000
        feature = _make_video_feature(
            offset=0, grid_thw=(10, 4, 4), second_per_grid_ts=1000.0
        )
        input_tokens = list(range(10 * 2 * 2))
        positions, delta = model.get_mrope_input_positions(input_tokens, [feature])
        max_pos = config.max_position_embeddings * 4
        assert positions.max().item() < max_pos
