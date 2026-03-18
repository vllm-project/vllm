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
import helion.language as hl

from tests.kernels.helion.helpers import dummy_kernel_registry
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.register import (
    _HOP_AVAILABLE,
    ConfiguredHelionKernel,
    HelionKernelWrapper,
    get_kernel_by_name,
    get_registered_kernels,
    register_kernel,
    validate_helion_settings,
)

if _HOP_AVAILABLE:
    from helion._compiler._dynamo.higher_order_ops import (
        helion_kernel_wrapper_mutation,
    )


def _add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


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
            "vllm.kernels.helion.config_manager.ConfigManager",
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
            "vllm.kernels.helion.config_manager.ConfigManager",
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
                "vllm.kernels.helion.config_manager.ConfigManager",
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
                "vllm.kernels.helion.config_manager.ConfigManager",
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
                "vllm.kernels.helion.config_manager.ConfigManager",
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

    def test_init_disables_on_missing_configs(self, sample_kernel):
        """Test __init__ marks wrapper as disabled when configs are missing."""

        def fake_impl(*args, **kwargs):
            return torch.zeros_like(args[0])

        def default_picker(args, config_keys):
            return "default"

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(
            return_value={}
        )  # Empty configs

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
        ):
            mock_kernel.return_value = Mock(return_value=sample_kernel)

            wrapper = HelionKernelWrapper(
                raw_kernel_func=sample_kernel,
                op_name="test_kernel",
                fake_impl=fake_impl,
                config_picker=default_picker,
            )

            assert wrapper._disabled is True
            assert "No configs available" in wrapper._disabled_reason

    def test_disabled_wrapper_raises_on_call(self, sample_kernel):
        """Test __call__ raises RuntimeError on a disabled wrapper."""

        def fake_impl(*args, **kwargs):
            return torch.zeros_like(args[0])

        def default_picker(args, config_keys):
            return "default"

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value={})

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
        ):
            mock_kernel.return_value = Mock(return_value=sample_kernel)

            wrapper = HelionKernelWrapper(
                raw_kernel_func=sample_kernel,
                op_name="test_kernel",
                fake_impl=fake_impl,
                config_picker=default_picker,
            )

        with pytest.raises(RuntimeError, match="is disabled"):
            wrapper(torch.randn(4, 4), torch.randn(4, 4))

    def test_disabled_wrapper_get_configured_op_raises(self, sample_kernel):
        """Test get_configured_op raises RuntimeError on a disabled wrapper."""

        def fake_impl(*args, **kwargs):
            return torch.zeros_like(args[0])

        def default_picker(args, config_keys):
            return "default"

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value={})

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
        ):
            mock_kernel.return_value = Mock(return_value=sample_kernel)

            wrapper = HelionKernelWrapper(
                raw_kernel_func=sample_kernel,
                op_name="test_kernel",
                fake_impl=fake_impl,
                config_picker=default_picker,
            )

        with pytest.raises(RuntimeError, match="is disabled"):
            wrapper.get_configured_op()

    def test_disabled_wrapper_supports_get_inputs(self, sample_kernel):
        """Test get_inputs works on a disabled wrapper."""

        def fake_impl(*args, **kwargs):
            return torch.zeros_like(args[0])

        def default_picker(args, config_keys):
            return "default"

        expected_inputs = {"key1": (torch.randn(4),)}
        input_gen = Mock(return_value=expected_inputs)

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value={})

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
        ):
            mock_kernel.return_value = Mock(return_value=sample_kernel)

            wrapper = HelionKernelWrapper(
                raw_kernel_func=sample_kernel,
                op_name="test_kernel",
                fake_impl=fake_impl,
                config_picker=default_picker,
                input_generator=input_gen,
            )

        assert wrapper._disabled is True
        result = wrapper.get_inputs()
        assert result is expected_inputs

    def test_disabled_wrapper_supports_run_autotune(self, sample_kernel):
        """Test run_autotune works on a disabled wrapper."""

        def fake_impl(*args, **kwargs):
            return torch.zeros_like(args[0])

        def default_picker(args, config_keys):
            return "default"

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value={})

        mock_config = Mock()

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
        ):
            mock_kernel.return_value = Mock(return_value=sample_kernel)

            wrapper = HelionKernelWrapper(
                raw_kernel_func=sample_kernel,
                op_name="test_kernel",
                fake_impl=fake_impl,
                config_picker=default_picker,
            )

        assert wrapper._disabled is True

        with patch(
            "vllm.kernels.helion.register.create_helion_decorated_kernel"
        ) as mock_create:
            mock_autotune_kernel = Mock()
            mock_autotune_kernel.autotune.return_value = mock_config
            mock_create.return_value = mock_autotune_kernel

            inputs = (torch.randn(4, 4),)
            result = wrapper.run_autotune(inputs)
            assert result is mock_config

    def test_init_caches_configured_kernel(self, sample_kernel, sample_configs):
        """Test __init__ eagerly builds and caches ConfiguredHelionKernel."""

        def fake_impl(*args, **kwargs):
            return torch.zeros_like(args[0])

        def default_picker(args, config_keys):
            return "default"

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
        ):
            mock_kernel.return_value = Mock(return_value=sample_kernel)

            wrapper = HelionKernelWrapper(
                raw_kernel_func=sample_kernel,
                op_name="test_kernel",
                fake_impl=fake_impl,
                config_picker=default_picker,
            )

            assert wrapper._configured_kernel is not None
            result1 = wrapper.get_configured_op()
            result2 = wrapper.get_configured_op()
            assert result1 is result2

    @pytest.mark.skipif(
        not _HOP_AVAILABLE, reason="HOP path only used when HOP available"
    )
    def test_init_eagerly_initializes_hop_path(self):
        """Test that register_kernel eagerly builds the configured kernel
        on the HOP path (no custom op registration needed)."""
        from vllm.kernels.helion.utils import get_canonical_gpu_name

        configs = {"default": helion.Config(block_sizes=[4, 4])}
        with (
            dummy_kernel_registry(configs=configs) as register,
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                wraps=get_canonical_gpu_name,
            ) as mock_gpu,
        ):
            wrapper = register(
                config_picker=lambda args, keys: "default",
            )(_add_kernel)

            mock_gpu.assert_called_once()
            assert wrapper._configured_kernel is not None

        with patch(
            "vllm.kernels.helion.utils.get_canonical_gpu_name",
            side_effect=AssertionError("get_canonical_gpu_name called during __call__"),
        ):
            x = torch.randn(4, 4, device="cuda")
            y = torch.randn(4, 4, device="cuda")
            result = wrapper(x, y)
            expected = x + y
            assert torch.allclose(result, expected)

    @pytest.mark.skipif(
        _HOP_AVAILABLE, reason="CustomOp path not used when HOP available"
    )
    def test_init_eagerly_initializes(self):
        """Test that register_kernel eagerly loads configs and detects GPU
        during construction so __call__ needs no further initialization."""
        from vllm.kernels.helion.utils import get_canonical_gpu_name

        with (
            dummy_kernel_registry() as register,
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                wraps=get_canonical_gpu_name,
            ) as mock_gpu,
        ):
            wrapper = register(
                config_picker=lambda args, keys: "default",
            )(_add_kernel)

            # Init must have detected GPU and built the kernel
            mock_gpu.assert_called_once()
            assert wrapper._configured_kernel is not None
            assert hasattr(torch.ops.vllm_helion, wrapper.op_name)

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

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)

        existing_op = Mock()
        mock_namespace = Mock()
        mock_namespace.test_kernel = existing_op

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager",
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

            wrapper = HelionKernelWrapper(
                raw_kernel_func=sample_kernel,
                op_name="test_kernel",
                fake_impl=fake_impl,
                config_picker=default_picker,
            )
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

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)

        new_op = Mock()
        registered_ops: dict[str, Mock] = {}
        mutates_args = ["y"]

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
                "vllm.kernels.helion.config_manager.ConfigManager",
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

            wrapper = HelionKernelWrapper(
                raw_kernel_func=sample_kernel,
                op_name="test_kernel",
                fake_impl=fake_impl,
                mutates_args=mutates_args,
                config_picker=default_picker,
            )
            result = wrapper._get_or_register_custom_op()

            mock_register.assert_called_once()
            assert result is new_op
            assert mock_register.call_args[1]["op_func"] is mock_decorated
            assert mock_register.call_args[1]["mutates_args"] is mutates_args


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

    def test_get_kernel_by_name_returns_kernel(self):
        """Test get_kernel_by_name returns registered kernel."""
        with dummy_kernel_registry() as register:
            wrapper = register(
                "test_kernel", config_picker=lambda args, keys: "default"
            )(_add_kernel)

        from vllm.kernels.helion.register import _REGISTERED_KERNELS

        _REGISTERED_KERNELS["test_kernel"] = wrapper

        result = get_kernel_by_name("test_kernel")
        assert result is wrapper

    def test_get_kernel_by_name_returns_none_for_missing(self):
        """Test get_kernel_by_name returns None for missing kernel."""
        result = get_kernel_by_name("nonexistent")
        assert result is None

    def test_register_kernel_auto_generates_fake_impl(self):
        """Test register_kernel auto-generates fake_impl when not provided."""
        with (
            dummy_kernel_registry() as register,
            patch("vllm.kernels.helion.register.infer_fake_impl") as mock_infer,
        ):
            mock_fake = Mock()
            mock_infer.return_value = mock_fake
            wrapper = register(
                config_picker=lambda args, keys: "default",
            )(_add_kernel)

        mock_infer.assert_called_once_with(_add_kernel, None)
        assert wrapper._fake_impl is mock_fake

    def test_register_kernel_creates_wrapper(self):
        """Test register_kernel creates HelionKernelWrapper."""
        with dummy_kernel_registry() as register:
            result = register("test_name", config_picker=lambda args, keys: "default")(
                _add_kernel
            )

        assert isinstance(result, HelionKernelWrapper)
        assert result.op_name == "test_name"
        assert result.raw_kernel_func is _add_kernel

    def test_register_kernel_auto_detects_name(self):
        """Test register_kernel uses function name when no name provided."""
        with dummy_kernel_registry() as register:
            wrapper = register(config_picker=lambda args, keys: "default")(_add_kernel)

        assert wrapper.op_name == "_add_kernel"

    def test_register_kernel_registers_in_global_registry(self):
        """Test register_kernel adds wrapper to global registry."""
        with dummy_kernel_registry() as register:
            wrapper = register(
                "test_kernel", config_picker=lambda args, keys: "default"
            )(_add_kernel)

        registered_kernels = get_registered_kernels()
        assert "test_kernel" in registered_kernels
        assert registered_kernels["test_kernel"] is wrapper

    def test_register_kernel_passes_helion_settings(self):
        """Test register_kernel passes helion_settings to wrapper."""
        settings = helion.Settings()
        settings.print_output_code = True

        with dummy_kernel_registry() as register:
            result = register(
                "test_name",
                config_picker=lambda args, keys: "default",
                helion_settings=settings,
            )(_add_kernel)

        assert result.helion_settings is settings

    def test_register_kernel_supports_decorator_syntax(self):
        """Test register_kernel works with decorator arguments."""
        mock_fake = Mock()

        with dummy_kernel_registry() as register:
            result = register(
                "custom_name",
                config_picker=lambda args, keys: "default",
                fake_impl=mock_fake,
            )(_add_kernel)

        assert result.op_name == "custom_name"
        assert result._fake_impl is mock_fake

    def test_register_kernel_raises_on_duplicate_registration(self):
        """Test register_kernel raises error on duplicate names."""
        with dummy_kernel_registry() as register:
            register("duplicate_name", config_picker=lambda args, keys: "default")(
                _add_kernel
            )

            with pytest.raises(ValueError, match="already registered"):
                register("duplicate_name", config_picker=lambda args, keys: "default")(
                    _add_kernel
                )

    def test_register_kernel_rejects_autotuner_fn_in_settings(self):
        """Test register_kernel rejects conflicting autotuner_fn."""
        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"autotuner_fn": Mock()}

        with pytest.raises(ValueError, match="uses a custom autotuner"):

            @register_kernel(
                "test",
                config_picker=lambda args, keys: "default",
                helion_settings=mock_settings,
            )
            def test_kernel(x):
                return x

    def test_register_kernel_no_warning_with_static_shapes_false(self):
        """Test register_kernel doesn't warn with static_shapes=False."""
        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"static_shapes": False}

        with (
            dummy_kernel_registry() as register,
            patch("vllm.kernels.helion.register.logger") as mock_logger,
        ):
            register(
                "test",
                config_picker=lambda args, keys: "default",
                helion_settings=mock_settings,
            )(_add_kernel)

        mock_logger.warning.assert_not_called()

    def test_disabled_kernel_appears_in_registry(self):
        """Test that a disabled wrapper is still in the global registry."""

        def fake_impl(*args, **kwargs):
            return torch.zeros_like(args[0])

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_platform_configs = Mock(return_value={})

        with (
            patch(
                "vllm.kernels.helion.config_manager.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
        ):
            mock_kernel.return_value = Mock(return_value=_add_kernel)

            wrapper = register_kernel(
                "disabled_kernel",
                config_picker=lambda args, keys: "default",
                fake_impl=fake_impl,
            )(_add_kernel)

        assert wrapper._disabled is True
        registered = get_registered_kernels()
        assert "disabled_kernel" in registered
        assert registered["disabled_kernel"] is wrapper


@pytest.mark.skipif(not _HOP_AVAILABLE, reason="Requires PyTorch >= 2.11 for HOP")
class TestTorchCompileHOP:
    """Test that HelionKernelWrapper emits the correct HOP under torch.compile."""

    def test_compiled_graph_contains_helion_hop(self):
        """Verify torch.compile on a HelionKernelWrapper emits a
        helion_kernel_wrapper_mutation HOP node in the FX graph."""
        configs = {"default": helion.Config(block_sizes=[4, 4])}

        with dummy_kernel_registry(configs=configs) as register:
            add_helion_kernel = register(
                op_name="test_torch_compile_add_kernel",
                config_picker=lambda args, keys: "default",
            )(_add_kernel)

        captured_graph: torch.fx.GraphModule | None = None

        def capturing_backend(gm, example_inputs):
            nonlocal captured_graph
            assert captured_graph is None, "Backend called multiple times"
            captured_graph = gm
            return gm.forward

        def f(x, y):
            return add_helion_kernel(x, y)

        torch._dynamo.reset()
        compiled_f = torch.compile(f, backend=capturing_backend, fullgraph=True)

        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")

        # Run compiled version and capture graph
        compiled_result = compiled_f(x, y)

        assert captured_graph is not None
        hop_nodes = [
            node
            for node in captured_graph.graph.nodes
            if node.op == "call_function"
            and node.target is helion_kernel_wrapper_mutation
        ]
        assert len(hop_nodes) > 0, (
            "Expected helion_kernel_wrapper_mutation HOP node in compiled graph, "
            f"but found none. Graph nodes: "
            f"{[(n.op, n.target) for n in captured_graph.graph.nodes]}"
        )

        # Verify compiled result matches eager execution
        eager_result = f(x, y)  # Run in eager mode

        assert torch.allclose(compiled_result, eager_result, atol=1e-5, rtol=1e-5), (
            "Compiled execution result doesn't match eager execution. "
            f"Max difference: {torch.max(torch.abs(compiled_result - eager_result))}"
        )
