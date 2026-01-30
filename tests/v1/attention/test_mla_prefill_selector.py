# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA prefill backend selector."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import AttentionConfig, ModelConfig, VllmConfig
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum
from vllm.v1.attention.backends.mla.prefill.selector import (
    _auto_select_mla_prefill_backend,
    _get_mla_prefill_backend_priorities,
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
    """Create a mock ModelConfig with specified MLA dimensions."""
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
    """Create a VllmConfig for testing."""
    if model_config is None:
        model_config = _make_mock_model_config()

    attention_config = AttentionConfig(mla_prefill_backend=mla_prefill_backend)
    mock_vllm_config = MagicMock(spec=VllmConfig)
    mock_vllm_config.model_config = model_config
    mock_vllm_config.attention_config = attention_config
    return mock_vllm_config


class TestIsDeepseekR1MLACompatible:
    """Tests for is_deepseek_r1_mla_compatible function."""

    def test_compatible_dimensions(self):
        """Test with DeepSeek R1 compatible MLA dimensions."""
        vllm_config = _make_vllm_config(
            model_config=_make_mock_model_config(
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            )
        )
        assert is_deepseek_r1_mla_compatible(vllm_config) is True

    def test_incompatible_qk_nope_head_dim(self):
        """Test with incompatible qk_nope_head_dim."""
        vllm_config = _make_vllm_config(
            model_config=_make_mock_model_config(
                qk_nope_head_dim=64,  # Not 128
                qk_rope_head_dim=64,
                v_head_dim=128,
            )
        )
        assert is_deepseek_r1_mla_compatible(vllm_config) is False

    def test_incompatible_qk_rope_head_dim(self):
        """Test with incompatible qk_rope_head_dim."""
        vllm_config = _make_vllm_config(
            model_config=_make_mock_model_config(
                qk_nope_head_dim=128,
                qk_rope_head_dim=32,  # Not 64
                v_head_dim=128,
            )
        )
        assert is_deepseek_r1_mla_compatible(vllm_config) is False

    def test_incompatible_v_head_dim(self):
        """Test with incompatible v_head_dim."""
        vllm_config = _make_vllm_config(
            model_config=_make_mock_model_config(
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=64,  # Not 128
            )
        )
        assert is_deepseek_r1_mla_compatible(vllm_config) is False

    def test_missing_model_config(self):
        """Test with missing model config."""
        mock_vllm_config = MagicMock(spec=VllmConfig)
        mock_vllm_config.model_config = None
        assert is_deepseek_r1_mla_compatible(mock_vllm_config) is False


class TestGetMLAPrefillBackendPriorities:
    """Tests for _get_mla_prefill_backend_priorities function."""

    def test_blackwell_priorities(self):
        """Test backend priorities for Blackwell (SM10.x)."""
        capability = DeviceCapability(major=10, minor=0)
        priorities = _get_mla_prefill_backend_priorities(capability)

        assert len(priorities) == 4
        assert priorities[0] == MLAPrefillBackendEnum.TRTLLM_RAGGED
        assert priorities[1] == MLAPrefillBackendEnum.FLASHINFER
        assert priorities[2] == MLAPrefillBackendEnum.CUDNN
        assert priorities[3] == MLAPrefillBackendEnum.FLASH_ATTN

    def test_hopper_priorities(self):
        """Test backend priorities for Hopper (SM9.x)."""
        capability = DeviceCapability(major=9, minor=0)
        priorities = _get_mla_prefill_backend_priorities(capability)

        assert len(priorities) == 1
        assert priorities[0] == MLAPrefillBackendEnum.FLASH_ATTN

    def test_older_gpu_priorities(self):
        """Test backend priorities for older GPUs (SM8.x)."""
        capability = DeviceCapability(major=8, minor=0)
        priorities = _get_mla_prefill_backend_priorities(capability)

        assert len(priorities) == 1
        assert priorities[0] == MLAPrefillBackendEnum.FLASH_ATTN


class TestGetMLAPrefillBackend:
    """Tests for get_mla_prefill_backend function."""

    def test_no_device_capability_returns_flash_attn(self):
        """Test fallback to FlashAttention when device capability is unavailable."""
        vllm_config = _make_vllm_config()

        with patch(
            "vllm.v1.attention.backends.mla.prefill.selector.current_platform"
        ) as mock_platform:
            mock_platform.get_device_capability.return_value = None

            backend = get_mla_prefill_backend(vllm_config)
            assert backend.get_name() == "FLASH_ATTN_PREFILL"

    def test_explicit_flash_attn_selection(self):
        """Test explicit selection of FlashAttention backend."""
        try:
            flash_attn_cls = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
        except ImportError:
            pytest.skip("FLASH_ATTN backend not available")

        vllm_config = _make_vllm_config(
            mla_prefill_backend=MLAPrefillBackendEnum.FLASH_ATTN,
        )

        with patch(
            "vllm.v1.attention.backends.mla.prefill.selector.current_platform"
        ) as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=9, minor=0
            )

            # Mock the backend's validate_configuration to return no errors
            with patch.object(
                flash_attn_cls,
                "validate_configuration",
                return_value=[],
            ):
                backend = get_mla_prefill_backend(vllm_config)
                assert backend.get_name() == "FLASH_ATTN_PREFILL"

    def test_explicit_backend_invalid_falls_back_to_auto(self):
        """Test that invalid explicit backend falls back to auto-selection."""
        try:
            flash_attn_cls = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
        except ImportError:
            pytest.skip("FLASH_ATTN backend not available")

        vllm_config = _make_vllm_config(
            mla_prefill_backend=MLAPrefillBackendEnum.FLASHINFER,
        )

        with patch(
            "vllm.v1.attention.backends.mla.prefill.selector.current_platform"
        ) as mock_platform:
            # Hopper doesn't support FlashInfer prefill
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=9, minor=0
            )

            # FlashInfer will fail validation on Hopper, should fall back
            with patch.object(
                flash_attn_cls,
                "validate_configuration",
                return_value=[],
            ):
                backend = get_mla_prefill_backend(vllm_config)
                # Should fall back to FLASH_ATTN since it's the only option on Hopper
                assert backend.get_name() == "FLASH_ATTN_PREFILL"

    def test_auto_selection_on_hopper(self):
        """Test auto-selection on Hopper returns FlashAttention."""
        try:
            flash_attn_cls = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
        except ImportError:
            pytest.skip("FLASH_ATTN backend not available")

        vllm_config = _make_vllm_config()

        with patch(
            "vllm.v1.attention.backends.mla.prefill.selector.current_platform"
        ) as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=9, minor=0
            )

            with patch.object(
                flash_attn_cls,
                "validate_configuration",
                return_value=[],
            ):
                backend = get_mla_prefill_backend(vllm_config)
                assert backend.get_name() == "FLASH_ATTN_PREFILL"


class TestAutoSelectMLAPrefillBackend:
    """Tests for _auto_select_mla_prefill_backend function."""

    def test_blackwell_selects_first_valid_backend(self):
        """Test that Blackwell selects the first valid backend from priorities."""
        vllm_config = _make_vllm_config()
        capability = DeviceCapability(major=10, minor=0)

        try:
            trtllm_cls = MLAPrefillBackendEnum.TRTLLM_RAGGED.get_class()
        except ImportError:
            pytest.skip("TRTLLM_RAGGED backend not available")

        # Mock TRTLLM_RAGGED as available and valid
        with patch.object(
            trtllm_cls,
            "validate_configuration",
            return_value=[],
        ):
            backend = _auto_select_mla_prefill_backend(
                device_capability=capability,
                dtype=torch.bfloat16,
                vllm_config=vllm_config,
            )
            assert backend.get_name() == "TRTLLM_RAGGED_PREFILL"

    def test_blackwell_falls_back_when_trtllm_unavailable(self):
        """Test Blackwell falls back when TRTLLM is unavailable."""
        vllm_config = _make_vllm_config()
        capability = DeviceCapability(major=10, minor=0)

        try:
            flashinfer_cls = MLAPrefillBackendEnum.FLASHINFER.get_class()
        except ImportError:
            pytest.skip("FLASHINFER backend not available")

        # Mock TRTLLM_RAGGED as failing to import, FLASHINFER as valid
        with (
            patch.object(
                MLAPrefillBackendEnum.TRTLLM_RAGGED,
                "get_class",
                side_effect=ImportError("TRTLLM not available"),
            ),
            patch.object(flashinfer_cls, "validate_configuration", return_value=[]),
        ):
            backend = _auto_select_mla_prefill_backend(
                device_capability=capability,
                dtype=torch.bfloat16,
                vllm_config=vllm_config,
            )
            assert backend.get_name() == "FLASHINFER_PREFILL"

    def test_hopper_selects_flash_attn(self):
        """Test that Hopper only has FlashAttention available."""
        vllm_config = _make_vllm_config()
        capability = DeviceCapability(major=9, minor=0)

        try:
            flash_attn_cls = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
        except ImportError:
            pytest.skip("FLASH_ATTN backend not available")

        with patch.object(
            flash_attn_cls,
            "validate_configuration",
            return_value=[],
        ):
            backend = _auto_select_mla_prefill_backend(
                device_capability=capability,
                dtype=torch.bfloat16,
                vllm_config=vllm_config,
            )
            assert backend.get_name() == "FLASH_ATTN_PREFILL"

    def test_fallback_to_flash_attn_when_all_fail(self):
        """Test fallback to FlashAttention when all other backends fail."""
        vllm_config = _make_vllm_config()
        capability = DeviceCapability(major=10, minor=0)

        # Make all backends fail validation except FLASH_ATTN
        def mock_get_class(backend_enum):
            cls = MagicMock()
            if backend_enum == MLAPrefillBackendEnum.FLASH_ATTN:
                cls.validate_configuration.return_value = []
                cls.get_name.return_value = "FLASH_ATTN_PREFILL"
            else:
                cls.validate_configuration.return_value = ["not available"]
            return cls

        with patch.object(MLAPrefillBackendEnum, "get_class", mock_get_class):
            # Need to clear cache since we're changing behavior
            _auto_select_mla_prefill_backend.cache_clear()
            backend = _auto_select_mla_prefill_backend(
                device_capability=capability,
                dtype=torch.bfloat16,
                vllm_config=vllm_config,
            )
            assert backend.get_name() == "FLASH_ATTN_PREFILL"


class TestMLAPrefillBackendEnum:
    """Tests for MLAPrefillBackendEnum registry."""

    def test_all_backends_have_valid_paths(self):
        """Test that all registered backends have valid class paths."""
        for backend in MLAPrefillBackendEnum:
            path = backend.get_path()
            assert path is not None
            assert len(path) > 0
            assert "." in path  # Should be a qualified name

    def test_invalid_backend_raises_value_error(self):
        """Test that accessing invalid backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MLAPrefillBackendEnum["INVALID_BACKEND"]

        assert "Unknown MLA prefill backend" in str(exc_info.value)
        assert "INVALID_BACKEND" in str(exc_info.value)

    def test_flash_attn_backend_path(self):
        """Test FlashAttention backend has correct path."""
        assert "flash_attn" in MLAPrefillBackendEnum.FLASH_ATTN.get_path()

    def test_flashinfer_backend_path(self):
        """Test FlashInfer backend has correct path."""
        assert "flashinfer" in MLAPrefillBackendEnum.FLASHINFER.get_path()

    def test_cudnn_backend_path(self):
        """Test cuDNN backend has correct path."""
        assert "cudnn" in MLAPrefillBackendEnum.CUDNN.get_path()

    def test_trtllm_ragged_backend_path(self):
        """Test TRT-LLM Ragged backend has correct path."""
        assert "trtllm_ragged" in MLAPrefillBackendEnum.TRTLLM_RAGGED.get_path()


class TestBackendValidation:
    """Tests for backend validation logic.

    These tests validate the compute capability and dtype checks for each
    backend. They import the backend classes directly, which may fail if
    dependencies like flashinfer are not available.
    """

    def test_flash_attn_supports_all_capabilities(self):
        """Test FlashAttention supports all compute capabilities."""
        try:
            from vllm.v1.attention.backends.mla.prefill.flash_attn import (
                FlashAttnPrefillBackend,
            )
        except ImportError:
            pytest.skip("FlashAttention prefill backend not available")

        # FlashAttention should support all capabilities (fallback)
        for major in [8, 9, 10]:
            capability = DeviceCapability(major=major, minor=0)
            assert FlashAttnPrefillBackend.supports_compute_capability(capability)

    def test_flashinfer_only_supports_blackwell(self):
        """Test FlashInfer only supports Blackwell."""
        try:
            from vllm.v1.attention.backends.mla.prefill.flashinfer import (
                FlashInferPrefillBackend,
            )
        except ImportError:
            pytest.skip("FlashInfer prefill backend not available")

        # Only Blackwell (SM10)
        assert FlashInferPrefillBackend.supports_compute_capability(
            DeviceCapability(major=10, minor=0)
        )
        assert not FlashInferPrefillBackend.supports_compute_capability(
            DeviceCapability(major=9, minor=0)
        )
        assert not FlashInferPrefillBackend.supports_compute_capability(
            DeviceCapability(major=8, minor=0)
        )

    def test_cudnn_only_supports_blackwell(self):
        """Test cuDNN only supports Blackwell."""
        try:
            from vllm.v1.attention.backends.mla.prefill.cudnn import (
                CudnnPrefillBackend,
            )
        except ImportError:
            pytest.skip("cuDNN prefill backend not available")

        # Only Blackwell (SM10)
        assert CudnnPrefillBackend.supports_compute_capability(
            DeviceCapability(major=10, minor=0)
        )
        assert not CudnnPrefillBackend.supports_compute_capability(
            DeviceCapability(major=9, minor=0)
        )
        assert not CudnnPrefillBackend.supports_compute_capability(
            DeviceCapability(major=8, minor=0)
        )

    def test_trtllm_ragged_only_supports_blackwell(self):
        """Test TRT-LLM Ragged only supports Blackwell."""
        try:
            from vllm.v1.attention.backends.mla.prefill.trtllm_ragged import (
                TrtllmRaggedPrefillBackend,
            )
        except ImportError:
            pytest.skip("TRT-LLM Ragged prefill backend not available")

        # Only Blackwell (SM10)
        assert TrtllmRaggedPrefillBackend.supports_compute_capability(
            DeviceCapability(major=10, minor=0)
        )
        assert not TrtllmRaggedPrefillBackend.supports_compute_capability(
            DeviceCapability(major=9, minor=0)
        )
        assert not TrtllmRaggedPrefillBackend.supports_compute_capability(
            DeviceCapability(major=8, minor=0)
        )

    def test_r1_dimension_requirement(self):
        """Test that backends requiring R1 dimensions check correctly."""
        try:
            from vllm.v1.attention.backends.mla.prefill.flashinfer import (
                FlashInferPrefillBackend,
            )
        except ImportError:
            pytest.skip("FlashInfer prefill backend not available")

        assert FlashInferPrefillBackend.requires_r1_mla_dimensions is True

        # Valid R1 config
        vllm_config = _make_vllm_config(
            model_config=_make_mock_model_config(
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            )
        )
        capability = DeviceCapability(major=10, minor=0)

        with patch.object(FlashInferPrefillBackend, "is_available", return_value=True):
            invalid_reasons = FlashInferPrefillBackend.validate_configuration(
                device_capability=capability,
                dtype=torch.bfloat16,
                vllm_config=vllm_config,
            )
            assert len(invalid_reasons) == 0

        # Invalid R1 config
        vllm_config_invalid = _make_vllm_config(
            model_config=_make_mock_model_config(
                qk_nope_head_dim=64,  # Wrong dimension
                qk_rope_head_dim=64,
                v_head_dim=128,
            )
        )

        with patch.object(FlashInferPrefillBackend, "is_available", return_value=True):
            invalid_reasons = FlashInferPrefillBackend.validate_configuration(
                device_capability=capability,
                dtype=torch.bfloat16,
                vllm_config=vllm_config_invalid,
            )
            assert len(invalid_reasons) == 1
            assert "DeepSeek R1 MLA dimensions" in invalid_reasons[0]

    def test_dtype_validation(self):
        """Test dtype validation for backends."""
        try:
            from vllm.v1.attention.backends.mla.prefill.flash_attn import (
                FlashAttnPrefillBackend,
            )
        except ImportError:
            pytest.skip("FlashAttention prefill backend not available")

        # Supported dtypes
        assert FlashAttnPrefillBackend.supports_dtype(torch.float16)
        assert FlashAttnPrefillBackend.supports_dtype(torch.bfloat16)

        # Unsupported dtype
        assert not FlashAttnPrefillBackend.supports_dtype(torch.float32)
