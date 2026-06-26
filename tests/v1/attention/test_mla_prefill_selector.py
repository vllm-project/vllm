# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA prefill backend selector."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import AttentionConfig, ModelConfig, VllmConfig
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.prefill.base import MLADimensions
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum
from vllm.v1.attention.backends.mla.prefill.selector import (
    MLAPrefillSelectorConfig,
    _auto_select_mla_prefill_backend,
    get_mla_prefill_backend,
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
        capability = DeviceCapability(major=10, minor=0)
        selector_config = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            mla_dimensions=MLADimensions(
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            ),
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
        capability = DeviceCapability(major=10, minor=0)
        selector_config = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            mla_dimensions=MLADimensions(
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            ),
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

    def test_backend_supported_dimension_validation(self):
        try:
            from vllm.v1.attention.backends.mla.prefill.flashinfer import (
                FlashInferPrefillBackend,
            )
            from vllm.v1.attention.backends.mla.prefill.trtllm_ragged import (
                TrtllmRaggedPrefillBackend,
            )
        except ImportError:
            pytest.skip("MLA prefill backend not available")
            return

        capability = DeviceCapability(major=10, minor=0)
        selector_config = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            mla_dimensions=MLADimensions(
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            ),
        )

        with patch.object(FlashInferPrefillBackend, "is_available", return_value=True):
            invalid_reasons = FlashInferPrefillBackend.validate_configuration(
                capability,
                selector_config,
            )
            assert len(invalid_reasons) == 0

        selector_config_invalid = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            mla_dimensions=MLADimensions(
                qk_nope_head_dim=64,
                qk_rope_head_dim=64,
                v_head_dim=128,
            ),
        )

        with patch.object(FlashInferPrefillBackend, "is_available", return_value=True):
            invalid_reasons = FlashInferPrefillBackend.validate_configuration(
                capability,
                selector_config_invalid,
            )
            assert len(invalid_reasons) == 1
            assert "supported MLA dimensions" in invalid_reasons[0]

        selector_config_glm5 = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            mla_dimensions=MLADimensions(
                qk_nope_head_dim=192,
                qk_rope_head_dim=64,
                v_head_dim=256,
            ),
        )

        with patch.object(
            TrtllmRaggedPrefillBackend, "is_available", return_value=True
        ):
            invalid_reasons = TrtllmRaggedPrefillBackend.validate_configuration(
                capability,
                selector_config_glm5,
            )
            assert invalid_reasons == []


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


class TestMLAPrefillBackendConfig:
    """Tests for mla_prefill_backend configuration in AttentionConfig."""

    def test_default_backend_is_none(self):
        config = AttentionConfig()
        assert config.mla_prefill_backend is None

    def test_explicit_flash_attn_backend(self):
        config = AttentionConfig(
            mla_prefill_backend=MLAPrefillBackendEnum.FLASH_ATTN,
        )
        assert config.mla_prefill_backend == MLAPrefillBackendEnum.FLASH_ATTN

    def test_explicit_trtllm_ragged_backend(self):
        config = AttentionConfig(
            mla_prefill_backend=MLAPrefillBackendEnum.TRTLLM_RAGGED,
        )
        assert config.mla_prefill_backend == MLAPrefillBackendEnum.TRTLLM_RAGGED
