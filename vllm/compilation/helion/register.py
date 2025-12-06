# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Helion kernel registration decorators.

This module provides decorators to automatically register Helion kernels
as PyTorch custom operations with automatic fake kernel generation.
"""

from collections.abc import Callable

import torch
from torch.library import Library

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Create a library to hold the Helion custom ops
vllm_helion_lib = Library("vllm_helion", "FRAGMENT")  # noqa


class HelionKernelWrapper:
    """
    Wrapper for Helion kernels that can create config-specific PyTorch custom ops.

    This wrapper manages the base Helion kernel and can create multiple PyTorch custom ops
    with different configurations using the pattern: {kernel_name}_{config_key}
    """

    def __init__(
        self,
        raw_kernel_func,
        op_name,
        namespace,
        helion_settings=None,
        default_config=None,
    ):
        self.raw_kernel_func = raw_kernel_func  # Store the raw undecorated function
        self.helion_settings = (
            helion_settings  # Store helion.Settings for dynamic decoration
        )
        self.default_config = default_config  # Store default helion.Config
        self._decorated_kernels = {}  # Dictionary to store decorated kernels by config hash
        self.op_name = op_name
        self.namespace = namespace
        self.base_op_name = f"{namespace}::{op_name}"

        # Registered implementations for autotuning methods
        self._autotune_inputs_generator = None
        self._config_picker = None

        # Registered benchmark class for this kernel
        self._benchmark_class = None

        # Fake implementation (set during registration)
        self._fake_impl = None

        # Copy attributes needed for PyTorch schema inference
        self.__name__ = getattr(raw_kernel_func, "__name__", op_name)
        self.__globals__ = getattr(raw_kernel_func, "__globals__", {})
        self.__annotations__ = getattr(raw_kernel_func, "__annotations__", {})

        # Import here to avoid circular imports
        import inspect

        try:
            # Copy the original function's signature for PyTorch schema inference
            self.__signature__ = inspect.signature(raw_kernel_func)
        except (ValueError, TypeError):
            # If signature extraction fails, PyTorch will handle schema inference differently
            pass

    def __call__(self, *args, **kwargs):
        """
        HelionKernelWrapper should never be called directly in the new pattern.

        This method always raises an error to prevent direct usage.
        """
        raise RuntimeError(
            f"HelionKernelWrapper '{self.op_name}' should not be called directly. "
            f"Instead use:\n"
            f"  - In HelionCustomOp subclasses: self.create_kernel({self.op_name})\n"
            f"  - For direct usage: kernel.create_configured_op_from_model(model_config, config_manager)\n"
            f"Both return a ConfiguredHelionKernel that can be called directly."
        )

    def autotune(self, *args, **kwargs):
        """
        Delegate autotuning to the underlying Helion kernel function.

        For autotuning, we apply @helion.kernel with just the settings (no config),
        since the point of autotuning is to discover the optimal config.
        """
        # Get or create decorated kernel for autotuning (no specific config needed)
        decorated_kernel = self._get_autotune_kernel()
        return decorated_kernel.autotune(*args, **kwargs)

    def _apply_helion_decorator(self, config):
        """
        Apply @helion.kernel decorator to the raw function with specific config.

        Args:
            config: helion.Config instance to use for this decoration
        """
        if not HELION_AVAILABLE:
            raise ImportError(
                "Helion is not available. Cannot apply helion.kernel decorator."
            )

        if self.helion_settings is None:
            raise ValueError(
                f"No helion_settings provided for kernel '{self.op_name}'. "
                "Cannot apply @helion.kernel decorator."
            )

        # Create helion.kernel decorator arguments from settings and config
        kernel_kwargs = {"config": config}

        # Pass all settings from helion.Settings object if available
        if self.helion_settings:
            # Convert helion.Settings to dict, excluding private/internal attributes
            settings_dict = self.helion_settings.to_dict()
            kernel_kwargs.update(settings_dict)

        # Apply the helion.kernel decorator
        decorated_kernel = helion.kernel(**kernel_kwargs)(self.raw_kernel_func)

        # Store in dictionary by config hash for reuse
        if config is not None:
            config_hash = hash(str(config.__dict__))
            self._decorated_kernels[config_hash] = decorated_kernel

        return decorated_kernel

    def _get_autotune_kernel(self):
        """
        Get or create a helion kernel decorated for autotuning (no specific config).
        """
        if not hasattr(self, "_autotune_kernel"):
            if not HELION_AVAILABLE:
                raise ImportError(
                    "Helion is not available. Cannot create autotune kernel."
                )

            # Create helion.kernel decorator arguments from settings only (no config)
            kernel_kwargs = {}
            if self.helion_settings:
                settings_dict = self.helion_settings.to_dict()
                kernel_kwargs.update(settings_dict)

            # Apply the helion.kernel decorator for autotuning
            self._autotune_kernel = helion.kernel(**kernel_kwargs)(self.raw_kernel_func)

        return self._autotune_kernel

    def _get_decorated_kernel(self, config):
        """
        Get or create a helion kernel decorated with a specific config.

        Args:
            config: helion.Config instance to use for decoration

        Returns:
            Decorated helion kernel for the given config
        """
        if config is None:
            return self._get_autotune_kernel()

        config_hash = hash(str(config.__dict__))
        if config_hash not in self._decorated_kernels:
            self._decorated_kernels[config_hash] = self._apply_helion_decorator(config)

        return self._decorated_kernels[config_hash]

    def register_autotune_inputs_generator(self, generator_func):
        """
        Register a function to generate autotune inputs.

        Args:
            generator_func: Function that returns dict[str, tuple] where:
                - key: Configuration identifier (e.g., "4096", "h4096_s8")
                - value: Tuple of concrete arguments to pass to forward()

        Returns:
            The registered function (for decorator usage)

        Example:
            @kernel_wrapper.register_autotune_inputs_generator
            def generate_inputs():
                return {
                    "4096": (input_tensor, scale),
                    "8192": (larger_input_tensor, scale)
                }
        """
        self._autotune_inputs_generator = generator_func
        return generator_func

    def register_config_picker(self, picker_func):
        """
        Register a function to pick the best config from available options.

        Args:
            picker_func: Function that takes (model_config, available_configs) and
                        returns (config_key, config) tuple or None

        Returns:
            The registered function (for decorator usage)

        Example:
            @kernel_wrapper.register_config_picker
            def pick_config(model_config, available_configs):
                target_size = model_config.get_hidden_size()
                key = str(target_size)
                if key in available_configs:
                    return (key, available_configs[key])
                return None
        """
        self._config_picker = picker_func
        return picker_func

    def get_autotune_inputs(self) -> dict[str, tuple]:
        """
        Return dictionary of inputs for autotuning.

        Returns:
            Dict where:
            - key: Configuration identifier (e.g., "4096", "h4096_s8")
            - value: Tuple of concrete arguments to pass to forward()

        Example:
            {
                "4096": (input_tensor, scale),
                "8192": (larger_input_tensor, scale)
            }
        """
        if self._autotune_inputs_generator is None:
            raise NotImplementedError(
                f"No autotune inputs generator registered for kernel '{self.op_name}'. "
                f"Use @{self.op_name}.register_autotune_inputs_generator to register one."
            )
        return self._autotune_inputs_generator()

    def get_best_config(
        self, model_config, available_configs: dict[str, "helion.Config"]
    ) -> tuple[str, "helion.Config"] | None:
        """
        Select the best config for model_config from available options.

        This is a pure function that performs config selection logic without any I/O.
        The registered picker function should implement kernel-specific selection strategies.

        Args:
            model_config: vLLM ModelConfig instance
            available_configs: Dictionary mapping config keys to loaded Helion configs

        Returns:
            Tuple of (config_key, config) for the best match, or None if no suitable match
        """
        if self._config_picker is None:
            raise NotImplementedError(
                f"No config picker registered for kernel '{self.op_name}'. "
                f"Use @{self.op_name}.register_config_picker to register one."
            )
        return self._config_picker(model_config, available_configs)

    def register_benchmark(self, benchmark_class):
        """
        Register a benchmark class for this kernel.

        Args:
            benchmark_class: KernelBenchmark subclass for benchmarking this kernel

        Returns:
            The registered benchmark class (for decorator usage)

        Example:
            @silu_mul_fp8.register_benchmark
            class SiluMulFp8Benchmark(KernelBenchmark):
                ...
        """
        self._benchmark_class = benchmark_class
        return benchmark_class

    def get_benchmark(self):
        """
        Get the registered benchmark class for this kernel.

        Returns:
            KernelBenchmark subclass, or None if no benchmark is registered
        """
        return self._benchmark_class

    def run_autotune(
        self, autotune_inputs: dict[str, tuple], tuner_kwargs: dict | None = None
    ) -> dict[str, "helion.Config"]:
        """
        Run autotuning and return configs (without saving).

        Args:
            autotune_inputs: Dictionary mapping config keys to input tuples for autotuning
            tuner_kwargs: Additional arguments for Helion tuner

        Returns:
            Dictionary mapping config keys to tuned Helion configs
        """
        if not HELION_AVAILABLE:
            raise ImportError(
                "Helion is not available. Please install Helion to use autotuning."
            )

        # Get the Helion kernel function for autotuning
        kernel_fn = self._get_autotune_kernel()

        # Set reasonable defaults for tuner
        tuner_kwargs = tuner_kwargs or {}
        default_tuner_kwargs = {
            "initial_population": 200,
            "copies": 10,
            "max_generations": 40,
        }
        default_tuner_kwargs.update(tuner_kwargs)

        results = {}

        for config_key, inputs in autotune_inputs.items():
            logger.info(f"Autotuning {self.op_name} for config: {config_key}")

            try:
                # Use Helion's built-in autotune method
                config = kernel_fn.autotune(inputs, **default_tuner_kwargs)
                results[config_key] = config

            except Exception as e:
                logger.error(
                    f"Autotuning failed for {self.op_name} config {config_key}: {e}"
                )

        return results

    def create_configured_op_from_model(self, model_config, config_manager):
        """
        Create a configured PyTorch op using optimal config for the model.

        This method handles the full config selection and op creation pipeline:
        1. Loads available configs using ConfigManager
        2. Selects optimal config for the model
        3. Creates and registers a PyTorch custom op
        4. Returns the torch.ops.vllm_helion.xxx callable

        Args:
            model_config: vLLM ModelConfig for optimal config selection
            config_manager: ConfigManager instance for loading configs

        Returns:
            PyTorch ops callable (torch.ops.vllm_helion.{configured_op_name})

        Example:
            torch_op = kernel_wrapper.create_configured_op_from_model(model_config, config_manager)
            # Now can call: torch_op(input, scale)
        """
        assert model_config is not None, (
            f"{self.op_name}.create_configured_op_from_model() requires model_config"
        )

        assert config_manager is not None, (
            f"{self.op_name}.create_configured_op_from_model() requires config_manager"
        )

        # Load available configs using ConfigManager
        available_configs = config_manager.load_all_configs(self.op_name)

        assert available_configs, (
            f"No configs available for kernel '{self.op_name}' - ensure configs are properly saved"
        )

        # Use our selection logic
        result = self.get_best_config(model_config, available_configs)

        # get_best_config() must return a (config_key, config) tuple if configs are available
        assert result is not None, (
            f"{self.op_name}.get_best_config() returned None with {len(available_configs)} configs available"
        )

        config_key, optimal_config = result

        # Create configured op (merged from create_configured_op_with_key)
        if not HELION_AVAILABLE:
            raise ImportError(
                "Helion is not available. Please install Helion to create configured ops."
            )

        configured_op_name = f"{self.op_name}_{config_key}"
        full_configured_op_name = f"{self.namespace}::{configured_op_name}"

        # Check if already registered
        try:
            # Try to access the op to see if it exists
            torch_op = getattr(getattr(torch.ops, self.namespace), configured_op_name)
            logger.debug(f"Op {full_configured_op_name} already registered")
            return torch_op
        except (AttributeError, RuntimeError):
            # Op doesn't exist, need to create it
            pass

        logger.info(f"Creating configured op: {full_configured_op_name}")

        # Get or create decorated kernel with the specific config
        decorated_kernel = self._get_decorated_kernel(optimal_config)

        # Register the decorated kernel directly as PyTorch custom op
        direct_register_custom_op(
            op_name=configured_op_name,
            op_func=decorated_kernel,  # Use the decorated kernel directly
            mutates_args=None,
            fake_impl=self._fake_impl,
            target_lib=vllm_helion_lib,
        )

        logger.info(f"Registered configured op: {full_configured_op_name}")

        # Return the registered PyTorch ops callable
        torch_op = getattr(getattr(torch.ops, self.namespace), configured_op_name)
        return torch_op


# Try to import Helion - it's an optional dependency
try:
    import helion

    HELION_AVAILABLE = True
except ImportError:
    helion = None
    HELION_AVAILABLE = False


# Global registry for tracking all registered HelionKernelWrapper instances
_REGISTERED_KERNELS: dict[str, "HelionKernelWrapper"] = {}


def get_registered_kernels() -> dict[str, "HelionKernelWrapper"]:
    """
    Get all registered Helion kernel wrappers from the global registry.

    Returns:
        Dictionary mapping kernel names to HelionKernelWrapper instances
    """
    return _REGISTERED_KERNELS.copy()


def get_kernel_by_name(kernel_name: str) -> "HelionKernelWrapper | None":
    """
    Get a specific registered kernel by name.

    Args:
        kernel_name: Name of the kernel to retrieve

    Returns:
        HelionKernelWrapper instance if found, None otherwise
    """
    return _REGISTERED_KERNELS.get(kernel_name)


def register_kernel(
    op_name: str | None = None,
    *,
    device_types: str = "cuda",
    mutates_args: tuple = (),
    fake_impl: Callable | None = None,
    helion_settings=None,
    default_config=None,
) -> Callable:
    """
    Decorator to register a Helion kernel as a PyTorch custom operation.

    This decorator automatically:
    1. Registers the kernel as a PyTorch custom op under the vllm_helion namespace
    2. Generates a fake implementation by running the kernel with _launcher=lambda *args, **kwargs: None
       OR uses the provided fake_impl if specified
    3. Provides proper error handling when Helion is not available

    Args:
        op_name: Name of the operation (will be registered as vllm_helion::op_name).
                If None or empty string, automatically uses the decorated function's __name__.
        device_types: Target device types (default: "cuda")
        mutates_args: Tuple of argument names that are mutated (default: empty)
        fake_impl: Optional custom fake implementation. If provided, uses this instead
                  of the auto-generated _launcher approach. Should be a callable that
                  takes (*args, **kwargs) and returns a tensor with correct shape/dtype.

    Returns:
        Decorator function that registers the Helion kernel

    Examples:
        # Explicit op name:
        @helion.kernel(...)
        @register_kernel("silu_mul_fp8")
        def silu_mul_fp8_kernel(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            pass

        # Auto-detect op name from function name:
        @helion.kernel(...)
        @register_kernel()
        def silu_mul_fp8(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            pass  # Will be registered as "silu_mul_fp8"

        # Custom fake kernel (for compatibility with symbolic shapes):
        def my_fake_impl(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            # Manual shape inference logic
            output_shape = input.shape[:-1] + (input.shape[-1] // 2,)
            return torch.empty(output_shape, dtype=torch.float8_e4m3fn, device=input.device)

        @helion.kernel(...)
        @register_kernel("silu_mul_fp8", fake_impl=my_fake_impl)
        def silu_mul_fp8_kernel(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            pass

        # Always accessible as: torch.ops.vllm_helion.silu_mul_fp8(input, scale)
    """

    def decorator(helion_kernel_func: Callable) -> Callable:
        # Determine the operation name
        final_op_name = op_name
        if final_op_name is None or final_op_name == "":
            # Auto-detect from function name
            final_op_name = helion_kernel_func.__name__

        if not HELION_AVAILABLE:
            logger.warning(
                "Helion not available, skipping registration of kernel '%s'",
                final_op_name,
            )
            return helion_kernel_func

        # The function should be a raw undecorated function
        # We will apply @helion.kernel dynamically when needed

        # Create the HelionKernelWrapper that will be registered as PyTorch custom op
        namespace = "vllm_helion"  # Fixed namespace for all Helion kernels
        kernel_wrapper = HelionKernelWrapper(
            helion_kernel_func,
            final_op_name,
            namespace,
            helion_settings,
            default_config,
        )

        # Register the wrapper as PyTorch custom op using vLLM's low-overhead function
        # Convert mutates_args tuple to list of strings for direct_register_custom_op
        mutates_args_list = list(mutates_args) if mutates_args else None

        # Determine fake implementation
        final_fake_impl = fake_impl
        if final_fake_impl is None:
            # Create auto-generated fake implementation using proper Helion bind/compile pattern
            def helion_fake_kernel(*args, **kwargs):
                """
                Run the Helion kernel with a dummy launcher to generate correct output shapes.
                Uses the proper Helion bind/compile pattern: _launcher is an argument of CompiledConfig.

                Note: This fake kernel only generates shapes for compilation - config selection
                happens in fusion passes where the actual Helion ops are inserted.
                """
                # Apply helion.kernel decorator temporarily for fake execution
                temp_config = default_config if default_config else helion.Config()

                # Create helion.kernel decorator arguments
                kernel_kwargs = {"config": temp_config}
                if helion_settings:
                    settings_dict = helion_settings.to_dict()
                    kernel_kwargs.update(settings_dict)

                # Apply the decorator
                temp_decorated_kernel = helion.kernel(**kernel_kwargs)(
                    helion_kernel_func
                )

                bound = temp_decorated_kernel.bind(args)
                compiled_runner = bound.compile_config(temp_config)

                return compiled_runner(
                    *args, **kwargs, _launcher=lambda *args, **kwargs: None
                )

            final_fake_impl = helion_fake_kernel

        # Store the fake implementation in the wrapper for reuse
        kernel_wrapper._fake_impl = final_fake_impl

        # Register using vLLM's direct_register_custom_op
        direct_register_custom_op(
            op_name=final_op_name,
            op_func=kernel_wrapper,
            mutates_args=mutates_args_list,
            fake_impl=final_fake_impl,
            target_lib=vllm_helion_lib,
        )

        if final_fake_impl == fake_impl:
            logger.info(
                "Registered custom fake implementation for Helion kernel '%s'",
                helion_kernel_func.__name__,
            )
        else:
            logger.info(
                "Registered auto-generated fake implementation for Helion kernel '%s'",
                helion_kernel_func.__name__,
            )

        logger.info(
            "Registered Helion kernel '%s' as PyTorch custom op '%s' using direct registration",
            helion_kernel_func.__name__,
            kernel_wrapper.base_op_name,
        )

        # Register in global registry for fast lookup
        _REGISTERED_KERNELS[final_op_name] = kernel_wrapper

        # Return the HelionKernelWrapper - this is what fusion passes will use
        return kernel_wrapper

    return decorator
