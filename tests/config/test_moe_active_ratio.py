# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoE active parameter ratio calculation, headroom sizing,
and dense isolation.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from vllm.config.cache import CacheConfig
from vllm.config.model import ModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.mem_utils import calculate_moe_active_headroom

_getter = cast(property, ModelConfig.active_parameter_ratio).fget
assert callable(_getter)
_active_ratio_getter = cast(Callable[[Any], float], _getter)


def _create_mock_moe_model_config(
    *,
    hf_text_config: SimpleNamespace,
    model_arch_config: SimpleNamespace,
    num_experts: int,
    num_active_experts: int,
    expert_intermediate_size: int,
    shared_intermediate_size: int,
    hidden_size: int,
    num_layers: int,
    head_size: int,
    kv_heads: int,
    vocab_size: int,
) -> MagicMock:
    """Helper to wire mock methods on ModelConfig for testing active_parameter_ratio."""
    mock = MagicMock(spec=ModelConfig)
    mock.hf_text_config = hf_text_config
    mock.model_arch_config = model_arch_config
    mock.is_moe = True

    mock.get_num_experts = lambda: num_experts
    mock.get_num_active_experts = lambda: num_active_experts
    mock.get_expert_intermediate_size = lambda: expert_intermediate_size
    mock.get_shared_intermediate_size = lambda: shared_intermediate_size
    mock.get_hidden_size = lambda: hidden_size
    mock.get_total_num_hidden_layers = lambda: num_layers
    mock.get_head_size = lambda: head_size
    mock.get_total_num_kv_heads = lambda: kv_heads
    mock.get_vocab_size = lambda: vocab_size
    return mock


class TestMoEActiveRatio:
    """Test suite for MoE active parameter ratio computation."""

    def test_dense_model_ratio(self):
        """Verify that dense models strictly return alpha = 1.0 and 0 active experts."""
        mock_model_config = MagicMock(spec=ModelConfig)
        mock_model_config.is_moe = False
        mock_model_config.active_parameter_ratio = _active_ratio_getter(
            mock_model_config
        )

        assert mock_model_config.active_parameter_ratio == 1.0

    def test_gemma4_moe_ratio(self):
        """Verify Gemma 4 26B-A4B active parameter ratio (~0.14-0.20)."""
        hf_text_config = SimpleNamespace(
            num_experts=128,
            top_k_experts=8,
            moe_intermediate_size=704,
            intermediate_size=2112,
            hidden_size=2816,
            num_hidden_layers=30,
            num_attention_heads=16,
            head_dim=256,
            num_key_value_heads=8,
            vocab_size=262144,
            tie_word_embeddings=True,
        )
        model_arch_config = SimpleNamespace(
            num_experts=128,
            total_num_hidden_layers=30,
            total_num_attention_heads=16,
            total_num_kv_heads=8,
            head_size=256,
            hidden_size=2816,
            vocab_size=262144,
        )

        mock_model_config = _create_mock_moe_model_config(
            hf_text_config=hf_text_config,
            model_arch_config=model_arch_config,
            num_experts=128,
            num_active_experts=8,
            expert_intermediate_size=704,
            shared_intermediate_size=2112,
            hidden_size=2816,
            num_layers=30,
            head_size=256,
            kv_heads=8,
            vocab_size=262144,
        )

        assert mock_model_config.get_num_experts() == 128
        assert mock_model_config.get_num_active_experts() == 8
        assert mock_model_config.get_expert_intermediate_size() == 704
        assert mock_model_config.get_shared_intermediate_size() == 2112

        alpha = _active_ratio_getter(mock_model_config)
        assert 0.14 <= alpha <= 0.20, f"Expected alpha in [0.14, 0.20], got {alpha}"

    def test_mixtral_moe_ratio(self):
        """Verify Mixtral 8x7B active parameter ratio (~0.25-0.35)."""
        hf_text_config = SimpleNamespace(
            num_local_experts=8,
            num_experts_per_tok=2,
            intermediate_size=14336,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=32000,
            tie_word_embeddings=True,
        )
        model_arch_config = SimpleNamespace(
            num_experts=8,
            total_num_hidden_layers=32,
            total_num_attention_heads=32,
            total_num_kv_heads=8,
            head_size=128,
            hidden_size=4096,
            vocab_size=32000,
        )

        mock_model_config = _create_mock_moe_model_config(
            hf_text_config=hf_text_config,
            model_arch_config=model_arch_config,
            num_experts=8,
            num_active_experts=2,
            expert_intermediate_size=14336,
            shared_intermediate_size=0,
            hidden_size=4096,
            num_layers=32,
            head_size=128,
            kv_heads=8,
            vocab_size=32000,
        )

        assert mock_model_config.get_num_experts() == 8
        assert mock_model_config.get_num_active_experts() == 2
        assert mock_model_config.get_expert_intermediate_size() == 14336
        assert mock_model_config.get_shared_intermediate_size() == 0

        alpha = _active_ratio_getter(mock_model_config)
        assert 0.25 <= alpha <= 0.35, f"Expected alpha in [0.25, 0.35], got {alpha}"


class TestMoEHeadroomCalculation:
    """Test suite for calculate_moe_active_headroom and safety factor."""

    def test_headroom_sizing(self):
        peak = 5 * (1024**3)  # 5 GiB
        alpha = 0.16
        safety = 0.05
        expected = int(peak * alpha * 1.05)
        headroom = calculate_moe_active_headroom(peak, alpha, safety)
        assert headroom == expected
        reclaimed = peak - headroom
        assert reclaimed > 4 * (1024**3)  # Over 4 GiB reclaimed

    def test_headroom_assertions(self):
        with pytest.raises(AssertionError, match="non-negative"):
            calculate_moe_active_headroom(-1, 0.5, 0.05)
        with pytest.raises(AssertionError, match="between 0.0 and 1.0"):
            calculate_moe_active_headroom(1000, 1.5, 0.05)
        with pytest.raises(AssertionError, match="between 0.0 and 1.0"):
            calculate_moe_active_headroom(1000, -0.1, 0.05)
        with pytest.raises(AssertionError, match="non-negative"):
            calculate_moe_active_headroom(1000, 0.5, -0.05)

    def test_cache_config_default_safety_factor(self):
        config = CacheConfig()
        assert config.moe_activation_safety_factor == 0.05

    def test_cache_config_compute_hash_ignores_safety_factor(self):
        """Verify modifying moe_activation_safety_factor does not
        invalidate graph cache."""
        c1 = CacheConfig(moe_activation_safety_factor=0.05)
        c2 = CacheConfig(moe_activation_safety_factor=0.20)
        assert c1.compute_hash() == c2.compute_hash()

    def test_cli_argument_parsing(self):
        parser = FlexibleArgumentParser()
        EngineArgs.add_cli_args(parser)
        parsed = parser.parse_args(
            ["--model", "test-model", "--moe-activation-safety-factor", "0.10"]
        )
        assert parsed.moe_activation_safety_factor == 0.10
