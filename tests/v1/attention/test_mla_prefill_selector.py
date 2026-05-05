# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA prefill backend selector."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import AttentionConfig, ModelConfig, VllmConfig
from vllm.model_executor.layers.attention.mla_attention import get_mla_prefill_scale
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    yarn_get_mscale,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum
from vllm.v1.attention.backends.mla.prefill.selector import (
    MLAPrefillSelectorConfig,
    _auto_select_mla_prefill_backend,
    get_mla_prefill_backend,
    is_deepseek_r1_mla_compatible,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching."""
    _auto_select_mla_prefill_backend.cache_clear()


def _make_mock_model_config(
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> ModelConfig:
    mock_config = MagicMock(spec=ModelConfig)
    mock_config.dtype = dtype
    mock_config.hf_text_config = MagicMock()
    mock_config.hf_text_config.qk_nope_head_dim = qk_nope_head_dim
    mock_config.hf_text_config.qk_rope_head_dim = qk_rope_head_dim
    mock_config.hf_text_config.v_head_dim = v_head_dim
    return mock_config


def _make_vllm_config(
    model_config: ModelConfig | None = None,
    mla_prefill_backend: MLAPrefillBackendEnum | None = None,
) -> VllmConfig:
    if model_config is None:
        model_config = _make_mock_model_config()

    attention_config = AttentionConfig(mla_prefill_backend=mla_prefill_backend)
    mock_vllm_config = MagicMock(spec=VllmConfig)
    mock_vllm_config.model_config = model_config
    mock_vllm_config.attention_config = attention_config
    return mock_vllm_config


class TestMLAPrefillScale:
    """Tests for the MLA prefill softmax scale."""

    def test_uses_qk_head_dim_for_deepseek_v2_style_mla(self):
        model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(
                q_lora_rank=None,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                rope_parameters={"rope_type": "default"},
            )
        )

        assert get_mla_prefill_scale(model_config) == pytest.approx(192**-0.5)

    def test_applies_deepseek_yarn_mscale(self):
        model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(
                q_lora_rank=None,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                rope_parameters={
                    "rope_type": "yarn",
                    "factor": 40,
                    "mscale_all_dim": 0.707,
                },
            )
        )

        mscale = yarn_get_mscale(40, 0.707)
        assert get_mla_prefill_scale(model_config) == pytest.approx(
            192**-0.5 * mscale * mscale
        )

    def test_deepseek_v4_style_mla_does_not_apply_yarn_mscale(self):
        model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(
                compress_ratios=[4],
                q_lora_rank=1536,
                head_dim=128,
                qk_rope_head_dim=64,
                rope_parameters={
                    "rope_type": "yarn",
                    "factor": 40,
                    "mscale_all_dim": 0.707,
                },
            )
        )

        assert get_mla_prefill_scale(model_config) == pytest.approx(128**-0.5)


class TestGetMLAPrefillBackend:
    """Tests for get_mla_prefill_backend (public API)."""

    def test_no_device_capability_returns_flash_attn(self):
        vllm_config = _make_vllm_config()

        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = None

            backend = get_mla_prefill_backend(vllm_config)
            assert backend.get_name() == "FLASH_ATTN"

    def test_explicit_flash_attn_selection(self):
        try:
            flash_attn_cls = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
        except ImportError:
            pytest.skip("FLASH_ATTN backend not available")
            return

        vllm_config = _make_vllm_config(
            mla_prefill_backend=MLAPrefillBackendEnum.FLASH_ATTN,
        )

        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=9, minor=0
            )

            with patch.object(
                flash_attn_cls,
                "validate_configuration",
                return_value=[],
            ):
                backend = get_mla_prefill_backend(vllm_config)
                assert backend.get_name() == "FLASH_ATTN"

    def test_explicit_backend_invalid_raises_error(self):
        vllm_config = _make_vllm_config(
            mla_prefill_backend=MLAPrefillBackendEnum.FLASHINFER,
        )

        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=9, minor=0
            )

            with pytest.raises(ValueError, match="is not valid"):
                get_mla_prefill_backend(vllm_config)

    def test_explicit_backend_import_error_raises(self):
        vllm_config = _make_vllm_config(
            mla_prefill_backend=MLAPrefillBackendEnum.TRTLLM_RAGGED,
        )

        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=10, minor=0
            )

            with (
                patch.object(
                    MLAPrefillBackendEnum.TRTLLM_RAGGED,
                    "get_class",
                    side_effect=ImportError("trtllm not installed"),
                ),
                pytest.raises(ValueError, match="is not valid"),
            ):
                get_mla_prefill_backend(vllm_config)

    def test_auto_selection_on_hopper(self):
        try:
            flash_attn_cls = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
        except ImportError:
            pytest.skip("FLASH_ATTN backend not available")
            return

        vllm_config = _make_vllm_config()

        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=9, minor=0
            )

            with patch.object(
                flash_attn_cls,
                "validate_configuration",
                return_value=[],
            ):
                backend = get_mla_prefill_backend(vllm_config)
                assert backend.get_name() == "FLASH_ATTN"


class TestAutoSelectMLAPrefillBackend:
    """Tests for fallback and error paths in auto-selection."""

    def test_blackwell_falls_back_to_trtllm(self):
        vllm_config = _make_vllm_config()
        capability = DeviceCapability(major=10, minor=0)
        selector_config = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            is_r1_compatible=is_deepseek_r1_mla_compatible(vllm_config),
        )

        try:
            trtllm_cls = MLAPrefillBackendEnum.TRTLLM_RAGGED.get_class()
        except ImportError:
            pytest.skip("TRTLLM_RAGGED backend not available")
            return

        with (
            patch.object(
                MLAPrefillBackendEnum.FLASH_ATTN,
                "get_class",
                side_effect=ImportError("FLASH_ATTN not available"),
            ),
            patch.object(trtllm_cls, "validate_configuration", return_value=[]),
        ):
            backend = _auto_select_mla_prefill_backend(
                capability,
                selector_config,
            )
            assert backend.get_name() == "TRTLLM_RAGGED"

    def test_all_fail_raises_error(self):
        vllm_config = _make_vllm_config()
        capability = DeviceCapability(major=10, minor=0)
        selector_config = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            is_r1_compatible=is_deepseek_r1_mla_compatible(vllm_config),
        )

        def mock_get_class(backend_enum):  # noqa: ARG001
            cls = MagicMock()
            cls.validate_configuration.return_value = ["not available"]
            return cls

        with patch.object(MLAPrefillBackendEnum, "get_class", mock_get_class):
            _auto_select_mla_prefill_backend.cache_clear()
            with pytest.raises(ValueError, match="No valid MLA"):
                _auto_select_mla_prefill_backend(
                    capability,
                    selector_config,
                )


class TestBackendValidation:
    """Tests for backend validation logic."""

    def test_r1_dimension_requirement(self):
        try:
            from vllm.v1.attention.backends.mla.prefill.flashinfer import (
                FlashInferPrefillBackend,
            )
        except ImportError:
            pytest.skip("FlashInfer prefill backend not available")
            return

        assert FlashInferPrefillBackend.requires_r1_mla_dimensions is True

        vllm_config = _make_vllm_config(
            model_config=_make_mock_model_config(
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            )
        )
        capability = DeviceCapability(major=10, minor=0)
        selector_config = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            is_r1_compatible=is_deepseek_r1_mla_compatible(vllm_config),
        )

        with patch.object(FlashInferPrefillBackend, "is_available", return_value=True):
            invalid_reasons = FlashInferPrefillBackend.validate_configuration(
                capability,
                selector_config,
            )
            assert len(invalid_reasons) == 0

        vllm_config_invalid = _make_vllm_config(
            model_config=_make_mock_model_config(
                qk_nope_head_dim=64,
                qk_rope_head_dim=64,
                v_head_dim=128,
            )
        )
        selector_config_invalid = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            is_r1_compatible=is_deepseek_r1_mla_compatible(vllm_config_invalid),
        )

        with patch.object(FlashInferPrefillBackend, "is_available", return_value=True):
            invalid_reasons = FlashInferPrefillBackend.validate_configuration(
                capability,
                selector_config_invalid,
            )
            assert len(invalid_reasons) == 1
            assert "DeepSeek R1 MLA dimensions" in invalid_reasons[0]


class TestMLAPrefillBackendParsing:
    """Tests for string-based mla_prefill_backend parsing from CLI args."""

    def test_valid_string_parses_to_enum(self):
        config = AttentionConfig(
            mla_prefill_backend="FLASH_ATTN",  # type: ignore[arg-type]
        )
        assert config.mla_prefill_backend == MLAPrefillBackendEnum.FLASH_ATTN

    def test_invalid_string_raises_error(self):
        with pytest.raises(ValueError, match="Unknown MLA prefill backend"):
            AttentionConfig(
                mla_prefill_backend="NONEXISTENT",  # type: ignore[arg-type]
            )


class TestDeprecatedFlagMigration:
    """Tests for _migrate_deprecated_mla_prefill_flags in AttentionConfig."""

    def test_no_deprecated_flags_leaves_backend_none(self):
        config = AttentionConfig()
        assert config.mla_prefill_backend is None

    def test_use_trtllm_ragged_migrates_to_trtllm_ragged(self):
        config = AttentionConfig(use_trtllm_ragged_deepseek_prefill=True)
        assert config.mla_prefill_backend == MLAPrefillBackendEnum.TRTLLM_RAGGED

    def test_disable_flashinfer_prefill_migrates_to_flash_attn(self):
        config = AttentionConfig(disable_flashinfer_prefill=True)
        assert config.mla_prefill_backend == MLAPrefillBackendEnum.FLASH_ATTN

    def test_explicit_backend_ignores_deprecated_flags(self):
        config = AttentionConfig(
            mla_prefill_backend=MLAPrefillBackendEnum.FLASH_ATTN,
            use_cudnn_prefill=True,
        )
        assert config.mla_prefill_backend == MLAPrefillBackendEnum.FLASH_ATTN

    def test_cudnn_raises_error(self):
        match = "cuDNN MLA prefill backend has been removed"
        with pytest.raises(ValueError, match=match):
            AttentionConfig(use_cudnn_prefill=True)

    def test_trtllm_takes_priority_over_disable_flashinfer(self):
        config = AttentionConfig(
            use_trtllm_ragged_deepseek_prefill=True,
            disable_flashinfer_prefill=True,
        )
        assert config.mla_prefill_backend == MLAPrefillBackendEnum.TRTLLM_RAGGED
