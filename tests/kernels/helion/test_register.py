# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Helion kernel registration.

Tests ConfiguredHelionKernel, HelionKernelWrapper, and PresetConfigSearch
including config picker registration, custom autotuner integration, and
PyTorch op registration.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

import helion

from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.register import (
    ConfiguredHelionKernel,
    HelionKernelWrapper,
    validate_helion_settings,
)


@pytest.fixture
def sample_configs():
    """Create real Helion config objects for testing."""
    return {
        "hiddensize_4096_batchsize_32": helion.Config(
            block_sizes=[128],
            num_warps=4,
            num_stages=3,
        ),
        "hiddensize_4096_batchsize_64": helion.Config(
            block_sizes=[256],
            num_warps=8,
            num_stages=4,
        ),
        "hiddensize_4096_batchsize_128": helion.Config(
            block_sizes=[512],
            num_warps=16,
            num_stages=2,
        ),
        "default": helion.Config(
            block_sizes=[64],
            num_warps=2,
            num_stages=2,
        ),
    }


@pytest.fixture
def sample_kernel():
    """Create a simple test kernel function."""

    def test_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Simple test kernel that adds two tensors."""
        return x + y

    return test_kernel


@pytest.fixture
def config_manager_with_test_configs(sample_configs):
    """Set up ConfigManager with test configs for nvidia_h200 platform."""
    mock_config_manager = Mock(spec=ConfigManager)
    mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)
    return mock_config_manager


@pytest.fixture
def configured_kernel(sample_kernel, sample_configs, config_manager_with_test_configs):
    """Create a ConfiguredHelionKernel for testing."""

    def test_config_picker(args, config_keys):
        """Simple config picker that returns default."""
        return "default"

    with (
        patch(
            "vllm.kernels.helion.config_manager.ConfigManager.get_instance",
            return_value=config_manager_with_test_configs,
        ),
        patch(
            "vllm.kernels.helion.utils.get_canonical_gpu_name",
            return_value="nvidia_h200",
        ),
        patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
    ):
        # Mock just the helion.kernel decorator to avoid actual kernel compilation
        mock_decorated = Mock()
        mock_kernel.return_value = Mock(return_value=mock_decorated)

        return ConfiguredHelionKernel(
            op_name="test_kernel",
            config_picker=test_config_picker,
            raw_kernel_func=sample_kernel,
            helion_settings=None,
        )


class TestValidateHelionSettings:
    """Test suite for validate_helion_settings utility function."""

    def test_accepts_none_settings(self):
        """Test that None settings are accepted without error."""
        validate_helion_settings(None, "test_kernel")  # Should not raise

    def test_accepts_valid_settings(self):
        """Test that valid settings without conflicts are accepted."""
        settings = helion.Settings()
        settings.static_shapes = False
        settings.print_output_code = True
        validate_helion_settings(settings, "test_kernel")  # Should not raise

    def test_rejects_autotuner_fn(self):
        """Test that settings with custom autotuner_fn raise ValueError."""
        settings = helion.Settings()
        settings.autotuner_fn = lambda *args: None  # Set custom autotuner function

        with pytest.raises(ValueError, match="uses a custom autotuner"):
            validate_helion_settings(settings, "test_kernel")

    def test_warns_on_static_shapes_true(self):
        """Test that static_shapes=True emits a warning."""
        settings = helion.Settings()
        settings.static_shapes = True

        with patch("vllm.kernels.helion.register.logger") as mock_logger:
            validate_helion_settings(settings, "test_kernel")
            mock_logger.warning.assert_called_once()
            assert "static_shapes=True" in mock_logger.warning.call_args[0][0]


def create_configured_kernel_with_configs(
    op_name,
    config_picker,
    kernel_func,
    configs,
    platform="nvidia_h200",
    helion_settings=None,
):
    """Helper to create ConfiguredHelionKernel with real config objects."""
    mock_config_manager = Mock(spec=ConfigManager)
    mock_config_manager.get_platform_configs = Mock(return_value=configs)

    with (
        patch(
            "vllm.kernels.helion.config_manager.ConfigManager.get_instance",
            return_value=mock_config_manager,
        ),
        patch(
            "vllm.kernels.helion.utils.get_canonical_gpu_name",
            return_value=platform,
        ),
        patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
    ):
        mock_decorated = Mock()
        mock_kernel.return_value = Mock(return_value=mock_decorated)

        return ConfiguredHelionKernel(
            op_name=op_name,
            config_picker=config_picker,
            raw_kernel_func=kernel_func,
            helion_settings=helion_settings,
        )


class TestConfiguredHelionKernel:
    """Test suite for ConfiguredHelionKernel."""

    def test_init_raises_without_picker(self, sample_kernel, sample_configs):
        """Test that __init__ raises when no picker registered."""
        configs = {"default": sample_configs["default"]}
        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value=configs)

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager.get_instance",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            pytest.raises(RuntimeError, match="No config picker registered"),
        ):
            ConfiguredHelionKernel(
                op_name="test_kernel",
                config_picker=None,  # No picker registered
                raw_kernel_func=sample_kernel,
                helion_settings=None,
            )

    def test_config_selector_validates_picker_result(
        self, sample_kernel, sample_configs
    ):
        """Test that config selector validates picker returns valid key."""

        def invalid_picker(args, config_keys):
            return "invalid_key"

        kernel = create_configured_kernel_with_configs(
            op_name="test_kernel",
            config_picker=invalid_picker,
            kernel_func=sample_kernel,
            configs=sample_configs,
        )

        key_computer = kernel._create_key_computer()
        selector = kernel._create_config_selector(key_computer)

        with pytest.raises(
            ValueError, match="Config picker returned invalid config key"
        ):
            selector((torch.randn(32, 4096),))

    def test_config_selector_handles_none_from_picker(
        self, sample_kernel, sample_configs
    ):
        """Test that config selector falls back to 'default' on None."""

        def none_picker(args, config_keys):
            return None

        kernel = create_configured_kernel_with_configs(
            op_name="test_kernel",
            config_picker=none_picker,
            kernel_func=sample_kernel,
            configs=sample_configs,
        )

        key_computer = kernel._create_key_computer()
        selector = kernel._create_config_selector(key_computer)

        result = selector((torch.randn(32, 4096),))
        assert result is kernel.configs["default"]

    def test_create_decorated_kernel_passes_helion_settings(
        self, sample_kernel, sample_configs
    ):
        """Test that _create_decorated_kernel passes helion_settings."""

        def default_picker(args, config_keys):
            return "default"

        settings = helion.Settings()
        settings.print_output_code = True
        # Note: helion.Settings() defaults static_shapes to True

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)

        with (
            patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager.get_instance",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
        ):
            mock_decorated = Mock()
            mock_kernel.return_value = Mock(return_value=mock_decorated)

            ConfiguredHelionKernel(
                op_name="test_kernel",
                config_picker=default_picker,
                raw_kernel_func=sample_kernel,
                helion_settings=settings,
            )

            call_kwargs = mock_kernel.call_args[1]
            assert "print_output_code" in call_kwargs
            assert call_kwargs["print_output_code"] is True
            # helion.Settings() defaults to static_shapes=True, so it should remain True
            assert call_kwargs["static_shapes"] is True

    def test_create_decorated_kernel_preserves_static_shapes_true(
        self, sample_kernel, sample_configs
    ):
        """Test that explicit static_shapes=True is preserved."""

        def default_picker(args, config_keys):
            return "default"

        settings = helion.Settings()
        settings.static_shapes = True

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)

        with (
            patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager.get_instance",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
        ):
            mock_decorated = Mock()
            mock_kernel.return_value = Mock(return_value=mock_decorated)

            ConfiguredHelionKernel(
                op_name="test_kernel",
                config_picker=default_picker,
                raw_kernel_func=sample_kernel,
                helion_settings=settings,
            )

            call_kwargs = mock_kernel.call_args[1]
            assert call_kwargs["static_shapes"] is True

    def test_key_and_config_selector_use_same_logic(
        self, sample_kernel, sample_configs
    ):
        """Test that key and config_selector produce identical results."""

        def tracking_picker(args, config_keys):
            x = args[0]
            batch_size = x.shape[0]
            if batch_size <= 32:
                return "hiddensize_4096_batchsize_32"
            elif batch_size <= 64:
                return "hiddensize_4096_batchsize_64"
            return "hiddensize_4096_batchsize_128"

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)

        with (
            patch("vllm.kernels.helion.register.helion.kernel") as mock_helion_kernel,
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager.get_instance",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
        ):
            mock_decorated = Mock()
            mock_helion_kernel.return_value = Mock(return_value=mock_decorated)

            kernel = ConfiguredHelionKernel(
                op_name="test_kernel",
                config_picker=tracking_picker,
                raw_kernel_func=sample_kernel,
                helion_settings=None,
            )

            call_kwargs = mock_helion_kernel.call_args[1]
            key_fn = call_kwargs["key"]
            autotuner_fn = call_kwargs["autotuner_fn"]

            tensor = torch.randn(50, 4096)  # batch=50, should select batchsize_64

            # key receives unpacked args, autotuner receives args as tuple
            key_result = key_fn(tensor)
            autotuner = autotuner_fn(None, (tensor,))
            config = autotuner.autotune()

            assert key_result == "hiddensize_4096_batchsize_64"
            assert config is kernel.configs["hiddensize_4096_batchsize_64"]


class TestHelionKernelWrapper:
    """Test suite for HelionKernelWrapper."""

    def test_get_configured_op_validates_configs_available(self, sample_kernel):
        """Test get_configured_op validates configs are available."""

        def fake_impl(*args, **kwargs):
            return torch.zeros_like(args[0])

        wrapper = HelionKernelWrapper(
            raw_kernel_func=sample_kernel,
            op_name="test_kernel",
            fake_impl=fake_impl,
        )

        def default_picker(args, config_keys):
            return "default"

        wrapper._config_picker = default_picker

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(
            return_value={}
        )  # Empty configs

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager.get_instance",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            pytest.raises(ValueError, match="No configs available"),
        ):
            wrapper.get_configured_op()

    def test_get_configured_op_validates_config_picker(
        self, sample_kernel, sample_configs
    ):
        """Test get_configured_op validates config picker."""

        def fake_impl(*args, **kwargs):
            return torch.zeros_like(args[0])

        wrapper = HelionKernelWrapper(
            raw_kernel_func=sample_kernel,
            op_name="test_kernel",
            fake_impl=fake_impl,
        )
        # Don't set config picker - should raise assertion error

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager.get_instance",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            pytest.raises(AssertionError, match="No config picker registered"),
        ):
            wrapper.get_configured_op()

    def test_get_configured_op_returns_cached_op(self, sample_kernel, sample_configs):
        """Test get_configured_op returns cached op when already registered."""

        def fake_impl(*args, **kwargs):
            return torch.zeros_like(args[0])

        def default_picker(args, config_keys):
            return "default"

        wrapper = HelionKernelWrapper(
            raw_kernel_func=sample_kernel,
            op_name="test_kernel",
            fake_impl=fake_impl,
        )
        wrapper._config_picker = default_picker

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)

        existing_op = Mock()
        mock_namespace = Mock()
        mock_namespace.test_kernel = existing_op

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager.get_instance",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            patch.object(torch.ops, "vllm_helion", mock_namespace),
            patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
        ):
            mock_decorated = Mock()
            mock_kernel.return_value = Mock(return_value=mock_decorated)
            result = wrapper.get_configured_op()
            assert result is existing_op

    def test_get_configured_op_registers_new_op(self, sample_kernel, sample_configs):
        """Test get_configured_op creates and registers new op."""

        def fake_impl(*args, **kwargs):
            return torch.zeros_like(args[0])

        def default_picker(args, config_keys):
            return "default"

        wrapper = HelionKernelWrapper(
            raw_kernel_func=sample_kernel,
            op_name="test_kernel",
            fake_impl=fake_impl,
        )
        wrapper._config_picker = default_picker

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)

        new_op = Mock()
        registered_ops: dict[str, Mock] = {}

        class MockNamespace:
            def __getattr__(self, name):
                if name in registered_ops:
                    return registered_ops[name]
                raise AttributeError(name)

        mock_namespace = MockNamespace()

        def register_side_effect(op_name, op_func, **kwargs):
            registered_ops[op_name] = new_op

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager.get_instance",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            patch.object(torch.ops, "vllm_helion", mock_namespace),
            patch(
                "vllm.kernels.helion.register.direct_register_custom_op",
                side_effect=register_side_effect,
            ) as mock_register,
            patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
        ):
            mock_decorated = Mock()
            mock_kernel.return_value = Mock(return_value=mock_decorated)
            result = wrapper.get_configured_op()

            mock_register.assert_called_once()
            assert result is new_op
            # Check that op_func is the decorated kernel, not ConfiguredHelionKernel
            assert mock_register.call_args[1]["op_func"] is mock_decorated
