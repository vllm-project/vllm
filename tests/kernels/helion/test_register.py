# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Helion kernel registration.

Tests ConfiguredHelionKernel, HelionKernelWrapper, and PresetConfigSearch
including config picker registration and custom autotuner integration.
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
    _HOP_AVAILABLE,
    ConfiguredHelionKernel,
    HelionKernelWrapper,
    get_kernel,
    get_registered_kernels,
    register_kernel,
    resolve_kernel,
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
        """Test that static_shapes=True emits a warning about being overridden."""
        settings = helion.Settings()
        settings.static_shapes = True

        with patch("vllm.kernels.helion.register.logger") as mock_logger:
            validate_helion_settings(settings, "test_kernel")
            mock_logger.warning.assert_called_once()
            assert "overridden to False" in mock_logger.warning.call_args[0][0]


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
            # static_shapes is always forced to False by vLLM
            assert call_kwargs["static_shapes"] is False

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

    def test_get_configured_op_returns_cached_kernel(
        self, sample_kernel, sample_configs
    ):
        """Test get_configured_op returns cached ConfiguredHelionKernel."""

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

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager.get_instance",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
        ):
            mock_decorated = Mock()
            mock_kernel.return_value = Mock(return_value=mock_decorated)

            result1 = wrapper.get_configured_op()
            result2 = wrapper.get_configured_op()
            assert result1 is result2

    @pytest.mark.skipif(
        _HOP_AVAILABLE, reason="CustomOp path not used when HOP available"
    )
    def test_get_or_register_custom_op_returns_cached_op(
        self, sample_kernel, sample_configs
    ):
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
        # Custom op is now registered with versioned_name
        setattr(mock_namespace, wrapper.versioned_name, existing_op)

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
            result = wrapper._get_or_register_custom_op()
            assert result is existing_op

    @pytest.mark.skipif(
        _HOP_AVAILABLE, reason="CustomOp path not used when HOP available"
    )
    def test_get_or_register_custom_op_registers_new_op(
        self, sample_kernel, sample_configs
    ):
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
            result = wrapper._get_or_register_custom_op()

            mock_register.assert_called_once()
            assert result is new_op
            assert mock_register.call_args[1]["op_func"] is mock_decorated


class TestKernelRegistry:
    """Test suite for kernel registry functionality."""

    def setup_method(self):
        """Save and clear the registry before each test."""
        from vllm.kernels.helion.register import _REGISTERED_KERNELS

        self._saved_registry = dict(_REGISTERED_KERNELS)
        _REGISTERED_KERNELS.clear()

    def teardown_method(self):
        """Restore the registry after each test."""
        from vllm.kernels.helion.register import _REGISTERED_KERNELS

        _REGISTERED_KERNELS.clear()
        _REGISTERED_KERNELS.update(self._saved_registry)

    def test_get_registered_kernels_returns_copy(self):
        """Test get_registered_kernels returns copy of registry."""
        result1 = get_registered_kernels()
        result2 = get_registered_kernels()

        # Should be separate objects
        assert result1 is not result2
        # Should have same content
        assert result1 == result2

    def test_get_kernel_returns_kernel(self):
        """Test get_kernel returns specific version."""
        wrapper = HelionKernelWrapper(
            raw_kernel_func=Mock(),
            op_name="test_kernel",
            fake_impl=Mock(),
        )

        from vllm.kernels.helion.register import _REGISTERED_KERNELS

        _REGISTERED_KERNELS["test_kernel"] = {1: wrapper}

        result = get_kernel("test_kernel", 1)
        assert result is wrapper

    def test_get_kernel_returns_none_for_missing(self):
        """Test get_kernel returns None for missing kernel."""
        assert get_kernel("nonexistent", 1) is None
        assert get_kernel("nonexistent", 99) is None

    def test_register_kernel_auto_generates_fake_impl(self):
        """Test register_kernel auto-generates fake_impl when not provided."""
        with patch("vllm.kernels.helion.register.infer_fake_impl") as mock_infer:
            mock_fake = Mock()
            mock_infer.return_value = mock_fake

            def original_kernel(x):
                return x

            wrapper = register_kernel(original_kernel)

            mock_infer.assert_called_once_with(original_kernel, None)
            assert wrapper._fake_impl is mock_fake

    def test_register_kernel_creates_wrapper(self):
        """Test register_kernel creates HelionKernelWrapper."""

        def test_kernel(x):
            return x

        result = register_kernel("test_name")(test_kernel)

        assert isinstance(result, HelionKernelWrapper)
        assert result.op_name == "test_name"
        assert result.raw_kernel_func is test_kernel

    def test_register_kernel_auto_detects_name(self):
        """Test register_kernel uses function name when no name provided."""

        @register_kernel
        def my_test_kernel(x):
            return x

        assert my_test_kernel.op_name == "my_test_kernel"

    def test_register_kernel_registers_in_global_registry(self):
        """Test register_kernel adds wrapper to global registry."""

        @register_kernel
        def test_kernel(x):
            return x

        registered_kernels = get_registered_kernels()
        assert "test_kernel" in registered_kernels
        assert 1 in registered_kernels["test_kernel"]
        assert registered_kernels["test_kernel"][1] is test_kernel

    def test_register_kernel_passes_helion_settings(self):
        """Test register_kernel passes helion_settings to wrapper."""
        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"debug": True}

        @register_kernel("test_name", helion_settings=mock_settings)
        def test_kernel(x):
            return x

        assert test_kernel.helion_settings is mock_settings

    def test_register_kernel_supports_decorator_syntax(self):
        """Test register_kernel works with decorator arguments."""
        mock_fake = Mock()

        wrapper = register_kernel("custom_name", fake_impl=mock_fake)

        def test_kernel(x):
            return x

        result = wrapper(test_kernel)

        assert result.op_name == "custom_name"
        assert result._fake_impl is mock_fake

    def test_register_kernel_bare_decorator(self):
        """Test register_kernel works as bare decorator."""

        @register_kernel
        def test_kernel(x):
            return x

        assert isinstance(test_kernel, HelionKernelWrapper)
        assert test_kernel.op_name == "test_kernel"

    def test_registered_wrapper_can_register_config_picker(self):
        """Test that registered wrapper can register config picker."""

        @register_kernel
        def test_kernel(x):
            return x

        def my_picker(args, config_keys):
            return "default"

        result = test_kernel.register_config_picker(my_picker)

        assert result is my_picker
        assert test_kernel._config_picker is my_picker

    def test_register_kernel_raises_on_duplicate_registration(self):
        """Test register_kernel raises error on same name + same version."""

        @register_kernel("duplicate_name", ver=1)
        def kernel1(x):
            return x

        with pytest.raises(ValueError, match="already registered"):

            @register_kernel("duplicate_name", ver=1)
            def kernel2(x):
                return x

    def test_register_kernel_rejects_autotuner_fn_in_settings(self):
        """Test register_kernel rejects conflicting autotuner_fn."""
        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"autotuner_fn": Mock()}

        with pytest.raises(ValueError, match="uses a custom autotuner"):

            @register_kernel("test", helion_settings=mock_settings)
            def test_kernel(x):
                return x

    def test_register_kernel_no_warning_with_static_shapes_false(self):
        """Test register_kernel doesn't warn with static_shapes=False."""
        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"static_shapes": False}

        with patch("vllm.kernels.helion.register.logger") as mock_logger:

            @register_kernel("test", helion_settings=mock_settings)
            def test_kernel(x):
                return x

            # Should not call warning
            mock_logger.warning.assert_not_called()


class TestVersionedKernelRegistration(TestKernelRegistry):
    """Test suite for versioned kernel registration."""

    def test_register_same_kernel_multiple_versions(self):
        """Test registering multiple versions of the same kernel."""

        @register_kernel("multi_ver", ver=1)
        def kernel_v1(x):
            return x

        @register_kernel("multi_ver", ver=2)
        def kernel_v2(x):
            return x

        versions = get_registered_kernels()["multi_ver"]
        assert 1 in versions
        assert 2 in versions
        assert versions[1] is kernel_v1
        assert versions[2] is kernel_v2

    def test_register_duplicate_version_raises(self):
        """Test that registering the same name+version raises."""

        @register_kernel("dup_ver", ver=1)
        def kernel_v1(x):
            return x

        with pytest.raises(ValueError, match="already registered"):

            @register_kernel("dup_ver", ver=1)
            def kernel_v1_dup(x):
                return x

    def test_bare_decorator_defaults_to_ver_1(self):
        """Test bare @register_kernel defaults to ver=1."""

        @register_kernel
        def bare_kernel(x):
            return x

        assert bare_kernel.ver == 1

    def test_versioned_name_property(self):
        """Test versioned_name returns op_name_vN."""

        @register_kernel("vname_test", ver=3)
        def kernel_v3(x):
            return x

        assert kernel_v3.versioned_name == "vname_test_v3"

    def test_older_version_is_implicitly_deprecated(self):
        """Older versions are implicitly deprecated by existence of newer ones."""

        @register_kernel("implicit_dep", ver=1)
        def kernel_v1(x):
            return x

        @register_kernel("implicit_dep", ver=2)
        def kernel_v2(x):
            return x

        versions = get_registered_kernels()["implicit_dep"]
        assert max(versions) == 2
        assert 1 in versions  # v1 exists but is implicitly deprecated

    def test_ver_must_be_positive(self):
        """Test that ver < 1 raises ValueError."""
        with pytest.raises(ValueError, match="ver must be >= 1"):

            @register_kernel("bad_ver", ver=0)
            def kernel_v0(x):
                return x


class TestResolveKernel(TestKernelRegistry):
    """Test suite for resolve_kernel function."""

    def _make_config_manager(self, config_map):
        """Helper to create a mock ConfigManager for resolve tests.

        Args:
            config_map: dict mapping versioned_name to configs dict.
                        e.g. {"my_kernel_v2": {"default": config}, "my_kernel_v1": {}}

        Returns:
            A mock ConfigManager whose get_platform_configs looks up config_map.
        """
        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(
            side_effect=lambda name, platform: config_map.get(name, {})
        )
        return mock_config_manager

    def test_resolves_to_latest_with_configs(self):
        """Test resolve_kernel picks newest version that has configs."""

        @register_kernel("resolve_latest", ver=1)
        def kernel_v1(x):
            return x

        @register_kernel("resolve_latest", ver=2)
        def kernel_v2(x):
            return x

        cm = self._make_config_manager(
            {
                "resolve_latest_v2": {"default": Mock()},
                "resolve_latest_v1": {"default": Mock()},
            }
        )
        result = resolve_kernel("resolve_latest", cm, "nvidia_h200")
        assert result is kernel_v2

    def test_falls_back_to_older_version(self):
        """Test resolve_kernel falls back when newest has no configs."""

        @register_kernel("resolve_fallback", ver=1)
        def kernel_v1(x):
            return x

        @register_kernel("resolve_fallback", ver=2)
        def kernel_v2(x):
            return x

        cm = self._make_config_manager(
            {
                "resolve_fallback_v2": {},
                "resolve_fallback_v1": {"default": Mock()},
            }
        )
        result = resolve_kernel("resolve_fallback", cm, "nvidia_h200")
        assert result is kernel_v1

    def test_emits_deprecation_warning_on_fallback(self):
        """Test resolve_kernel emits DeprecationWarning on fallback."""
        import warnings as _warnings

        @register_kernel("resolve_warn", ver=1)
        def kernel_v1(x):
            return x

        @register_kernel("resolve_warn", ver=2)
        def kernel_v2(x):
            return x

        cm = self._make_config_manager(
            {
                "resolve_warn_v2": {},
                "resolve_warn_v1": {"default": Mock()},
            }
        )
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            result = resolve_kernel("resolve_warn", cm, "nvidia_h200")

        assert result is kernel_v1
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "v1" in str(w[0].message)
        assert "v2" in str(w[0].message)

    def test_raises_when_no_configs(self):
        """Test resolve_kernel raises when no version has configs."""

        @register_kernel("resolve_none", ver=1)
        def kernel_v1(x):
            return x

        cm = self._make_config_manager({"resolve_none_v1": {}})
        with pytest.raises(ValueError, match="No configs available"):
            resolve_kernel("resolve_none", cm, "nvidia_h200")
