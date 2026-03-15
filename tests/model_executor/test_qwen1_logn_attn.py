# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Qwen1 use_logn_attn and use_dynamic_ntk support.

Verifies that QWenAttention correctly applies LogN attention scaling and
that QWenBlock correctly translates use_dynamic_ntk into
DynamicNTKAlphaRotaryEmbedding parameters.
"""

import math
from dataclasses import dataclass, field

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.dynamic_ntk_alpha_rope import (
    DynamicNTKAlphaRotaryEmbedding,
)
from vllm.model_executor.models.qwen import QWenAttention, QWenBlock


@dataclass
class DummyQwen1Config:
    """Minimal config mimicking Qwen-7B's QWenConfig."""

    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    intermediate_size: int = 22016
    max_position_embeddings: int = 32768
    layer_norm_epsilon: float = 1e-6
    seq_length: int = 8192
    use_logn_attn: bool = True
    use_dynamic_ntk: bool = True
    rope_parameters: dict = field(
        default_factory=lambda: {"rope_type": "default", "rope_theta": 10000}
    )


# ── Pure math: LogN tensor values ──────────────────────────────────────


class TestLognFormula:
    """Verify the logn formula matches the original Qwen1 implementation."""

    SEQ_LENGTH = 8192
    MAX_POS = 32768

    @staticmethod
    def _compute_logn_list(seq_length, max_pos):
        return [
            max(1.0, math.log(i + 1) / math.log(seq_length)) for i in range(max_pos)
        ]

    def test_within_training_length_is_one(self):
        logn = self._compute_logn_list(self.SEQ_LENGTH, self.MAX_POS)
        for i in range(self.SEQ_LENGTH):
            assert logn[i] == 1.0, f"Position {i} should be 1.0, got {logn[i]}"

    def test_beyond_training_length_grows(self):
        logn = self._compute_logn_list(self.SEQ_LENGTH, self.MAX_POS)
        for i in range(self.SEQ_LENGTH, self.MAX_POS):
            assert logn[i] > 1.0, f"Position {i} should be > 1.0"

    def test_monotonically_increasing(self):
        logn = self._compute_logn_list(self.SEQ_LENGTH, self.MAX_POS)
        for i in range(1, self.MAX_POS):
            assert logn[i] >= logn[i - 1], f"Not monotonic at position {i}"

    @pytest.mark.parametrize("position", [0, 4095, 8191, 8192, 16383, 32767])
    def test_matches_original_qwen1(self, position):
        """Each value must match the original Qwen1 formula (1-indexed)."""
        logn = self._compute_logn_list(self.SEQ_LENGTH, self.MAX_POS)
        i = position + 1  # original uses 1-indexed
        expected = math.log(i, self.SEQ_LENGTH) if i > self.SEQ_LENGTH else 1.0
        assert abs(logn[position] - expected) < 1e-10, (
            f"pos={position}: expected={expected:.8f}, actual={logn[position]:.8f}"
        )


# ── Pure math: NTK alpha computation ───────────────────────────────────


class TestNTKAlphaFormula:
    """Verify the NTK alpha formula matches the original Qwen1."""

    @staticmethod
    def _compute_ntk_alpha(max_pos, seq_length):
        if max_pos <= seq_length:
            return None  # no dynamic NTK needed
        context_value = math.log(max_pos / seq_length, 2) + 1
        return 2 ** math.ceil(context_value) - 1

    @pytest.mark.parametrize(
        ("max_pos", "seq_len", "expected"),
        [
            (32768, 8192, 7),  # Qwen-7B default: log2(4)+1=3 → 2^3-1=7
            (16384, 8192, 3),  # 2x training: log2(2)+1=2 → 2^2-1=3
            (65536, 8192, 15),  # 8x training: log2(8)+1=4 → 2^4-1=15
            (8192, 8192, None),  # Equal → no scaling needed
            (4096, 8192, None),  # Smaller → no scaling needed
        ],
    )
    def test_alpha_values(self, max_pos, seq_len, expected):
        assert self._compute_ntk_alpha(max_pos, seq_len) == expected


# ── Integration: QWenAttention with logn buffer ─────────────────────────


class TestLognAttentionInit:
    """Test QWenAttention logn buffer registration (requires dist_init)."""

    SEQ_LENGTH = 8192
    MAX_POS = 32768

    def test_buffer_registered(self, dist_init):
        attn = QWenAttention(
            hidden_size=4096,
            num_heads=32,
            max_position_embeddings=self.MAX_POS,
            rope_parameters={"rope_type": "default", "rope_theta": 10000},
            use_logn_attn=True,
            seq_length=self.SEQ_LENGTH,
        )
        assert hasattr(attn, "logn_tensor")
        assert attn.logn_tensor.shape == (self.MAX_POS,)
        # Verify values within training length are 1.0
        assert torch.allclose(
            attn.logn_tensor[: self.SEQ_LENGTH],
            torch.ones(self.SEQ_LENGTH),
        )
        # Verify values beyond training length are > 1.0
        assert (attn.logn_tensor[self.SEQ_LENGTH :] > 1.0).all()

    def test_no_buffer_when_disabled(self, dist_init):
        attn = QWenAttention(
            hidden_size=4096,
            num_heads=32,
            max_position_embeddings=self.MAX_POS,
            rope_parameters={"rope_type": "default", "rope_theta": 10000},
            use_logn_attn=False,
            seq_length=self.SEQ_LENGTH,
        )
        assert not hasattr(attn, "logn_tensor")
        assert attn.use_logn_attn is False


# ── Integration: QWenBlock dynamic NTK ──────────────────────────────────


class TestDynamicNTKBlock:
    """Test QWenBlock translates use_dynamic_ntk correctly (requires dist_init)."""

    def test_dynamic_ntk_produces_correct_rope(self, dist_init):
        config = DummyQwen1Config()
        block = QWenBlock(config)
        rope = block.attn.rotary_emb
        assert isinstance(rope, DynamicNTKAlphaRotaryEmbedding)
        assert rope.scaling_alpha == 7.0

    def test_dynamic_ntk_disabled(self, dist_init):
        config = DummyQwen1Config(use_dynamic_ntk=False)
        block = QWenBlock(config)
        rope = block.attn.rotary_emb
        assert not isinstance(rope, DynamicNTKAlphaRotaryEmbedding)

    def test_no_scaling_when_equal(self, dist_init):
        config = DummyQwen1Config(
            max_position_embeddings=8192,
            seq_length=8192,
            use_dynamic_ntk=True,
        )
        block = QWenBlock(config)
        assert not isinstance(block.attn.rotary_emb, DynamicNTKAlphaRotaryEmbedding)

    def test_no_override_when_rope_type_non_default(self, dist_init):
        """use_dynamic_ntk should not clobber an explicitly set rope_type."""
        config = DummyQwen1Config(
            use_dynamic_ntk=True,
            rope_parameters={"rope_type": "linear", "rope_theta": 10000, "factor": 2.0},
        )
        block = QWenBlock(config)
        assert not isinstance(block.attn.rotary_emb, DynamicNTKAlphaRotaryEmbedding)


# ── Integration: Config threading ───────────────────────────────────────


class TestConfigThreading:
    """Verify QWenBlock passes config fields to QWenAttention correctly."""

    def test_logn_attn_enabled(self, dist_init):
        config = DummyQwen1Config(use_logn_attn=True, seq_length=8192)
        block = QWenBlock(config)
        assert block.attn.use_logn_attn is True
        assert hasattr(block.attn, "logn_tensor")

    def test_logn_attn_disabled(self, dist_init):
        config = DummyQwen1Config(use_logn_attn=False)
        block = QWenBlock(config)
        assert block.attn.use_logn_attn is False

    def test_missing_fields_use_defaults(self, dist_init):
        """Config without Qwen1-specific fields should use safe defaults."""

        @dataclass
        class MinimalConfig:
            hidden_size: int = 4096
            num_attention_heads: int = 32
            intermediate_size: int = 22016
            max_position_embeddings: int = 32768
            layer_norm_epsilon: float = 1e-6
            rope_parameters: dict = field(
                default_factory=lambda: {
                    "rope_type": "default",
                    "rope_theta": 10000,
                }
            )

        block = QWenBlock(MinimalConfig())
        assert block.attn.use_logn_attn is False
        assert not isinstance(block.attn.rotary_emb, DynamicNTKAlphaRotaryEmbedding)


# ── Integration: Forward pass logn scaling ──────────────────────────────


class TestLognScalingLogic:
    """Verify the logn scaling logic applied to queries.

    Tests the scaling math directly on tensors without running the full
    attention kernel (which requires CUDA).
    """

    SEQ_LENGTH = 64
    MAX_POS = 256
    NUM_TOKENS = 8
    HEAD_DIM = 64
    NUM_HEADS = 2

    def _make_logn_tensor(self):
        """Build the same logn_tensor that QWenAttention would register."""
        logn_list = [
            max(1.0, math.log(i + 1) / math.log(self.SEQ_LENGTH))
            for i in range(self.MAX_POS)
        ]
        return torch.tensor(logn_list, dtype=torch.float32)

    def test_short_context_identity(self):
        """Positions within training length have factor 1.0 → no change."""
        logn_tensor = self._make_logn_tensor()
        positions = torch.arange(0, self.NUM_TOKENS, dtype=torch.long)
        q = torch.randn(self.NUM_TOKENS, self.NUM_HEADS * self.HEAD_DIM)

        logn_scale = logn_tensor[positions].unsqueeze(-1)
        q_scaled = q * logn_scale

        assert torch.allclose(q, q_scaled), (
            "Queries should be unchanged for positions within training length"
        )

    def test_long_context_scaled(self):
        """Positions beyond training length should scale queries up."""
        logn_tensor = self._make_logn_tensor()
        pos = self.SEQ_LENGTH + 50
        positions = torch.tensor([pos], dtype=torch.long)
        q = torch.randn(1, self.NUM_HEADS * self.HEAD_DIM)

        logn_scale = logn_tensor[positions].unsqueeze(-1)
        q_scaled = q * logn_scale

        expected_factor = math.log(pos + 1) / math.log(self.SEQ_LENGTH)
        actual_ratio = (q_scaled / q).mean().item()
        assert abs(actual_ratio - expected_factor) < 1e-5, (
            f"Expected ratio ~{expected_factor:.6f}, got {actual_ratio:.6f}"
        )

    def test_mixed_positions(self):
        """Mix of short and long positions: only long ones are scaled."""
        logn_tensor = self._make_logn_tensor()
        positions = torch.tensor(
            [0, self.SEQ_LENGTH - 1, self.SEQ_LENGTH, self.MAX_POS - 1],
            dtype=torch.long,
        )
        q = torch.ones(4, self.NUM_HEADS * self.HEAD_DIM)

        logn_scale = logn_tensor[positions].unsqueeze(-1)
        q_scaled = q * logn_scale

        # First two positions (within training length) → unchanged
        assert torch.allclose(q_scaled[0], q[0])
        assert torch.allclose(q_scaled[1], q[1])

        # Last two positions (beyond training length) → scaled up
        expected_2 = max(1.0, math.log(self.SEQ_LENGTH + 1) / math.log(self.SEQ_LENGTH))
        assert abs(q_scaled[2, 0].item() - expected_2) < 1e-5

        expected_3 = math.log(self.MAX_POS) / math.log(self.SEQ_LENGTH)
        assert abs(q_scaled[3, 0].item() - expected_3) < 1e-5

    def test_broadcast_shape(self):
        """logn_scale broadcasts correctly across all head dimensions."""
        logn_tensor = self._make_logn_tensor()
        positions = torch.tensor([self.SEQ_LENGTH + 10], dtype=torch.long)
        q = torch.ones(1, self.NUM_HEADS * self.HEAD_DIM)

        logn_scale = logn_tensor[positions].unsqueeze(-1)
        q_scaled = q * logn_scale

        # All elements in the single token should be scaled by the same factor
        expected = logn_tensor[positions[0]].item()
        assert torch.allclose(q_scaled, torch.full_like(q_scaled, expected), atol=1e-6)
