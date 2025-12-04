# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Helion kernel registration decorators.

This module provides decorators to automatically register Helion kernels
as PyTorch custom operations with automatic fake kernel generation.
"""

from collections.abc import Callable
from typing import Optional

import torch
from torch.library import Library

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Create a library to hold the Helion custom ops
vllm_helion_lib = Library("vllm_helion", "FRAGMENT")  # noqa


class HelionKernelWrapper:
    """
    Wrapper for Helion kernels that stores selected config and compiles accordingly.

    This wrapper is what gets registered as PyTorch custom op, not the raw Helion kernel.
    Fusion passes can set the config, and the __call__ method compiles with that config.
    """

    def __init__(self, helion_kernel_func, op_name, namespace):
        self.helion_kernel_func = helion_kernel_func
        self.op_name = op_name
        self.namespace = namespace
        self.full_op_name = f"{namespace}::{op_name}"
        self.config = None

        # Compilation cache: compile once when config is set
        self._compiled_kernel = None

        # References for advanced use and debugging (set later by register_kernel)
        self._pytorch_op_wrapper = None
        self._helion_kernel = None

        # Registered implementations for autotuning methods
        self._autotune_inputs_generator = None
        self._config_picker = None

        # Copy attributes needed for PyTorch schema inference
        self.__name__ = getattr(helion_kernel_func, "__name__", op_name)
        self.__globals__ = getattr(helion_kernel_func, "__globals__", {})
        self.__annotations__ = getattr(helion_kernel_func, "__annotations__", {})

        # Import here to avoid circular imports
        import inspect

        try:
            # Copy the original function's signature for PyTorch schema inference
            self.__signature__ = inspect.signature(helion_kernel_func)
        except (ValueError, TypeError):
            # If signature extraction fails, PyTorch will handle schema inference differently
            pass

    def set_config(self, config):
        """
        Set the config to use for this kernel execution.

        Can only be called once - subsequent calls will raise an error.
        This ensures consistent kernel behavior and simplifies caching.
        """
        if self.config is not None:
            raise RuntimeError(
                f"Config already set for kernel '{self.op_name}'. "
                "Kernel wrappers only allow setting config once to ensure consistent behavior."
            )

        self.config = config
        logger.debug(f"Config set for kernel '{self.op_name}'")

    def __call__(self, *args, **kwargs):
        """
        Execute the Helion kernel with the selected config.

        This method compiles the kernel once when first called, then reuses
        the compiled runner for all subsequent calls. Much simpler than
        cache invalidation since config can only be set once.
        """
        # Config must be set via set_config() before calling
        assert self.config is not None, (
            f"Config not set for kernel '{self.op_name}'. Call set_config() first."
        )

        logger.debug(f"Using configured config for {self.op_name}")

        # Compile once and cache the compiled kernel
        if self._compiled_kernel is None:
            logger.debug(
                f"First execution: compiling kernel {self.op_name} with config: {self.config}"
            )
            bound = self.helion_kernel_func.bind(args)
            self._compiled_kernel = bound.compile_config(self.config)
        else:
            logger.debug(f"Reusing compiled kernel for {self.op_name}")

        return self._compiled_kernel(*args, **kwargs)

    def autotune(self, *args, **kwargs):
        """
        Delegate autotuning to the underlying Helion kernel function.

        This method forwards the autotune call to the wrapped helion_kernel_func,
        which has the @helion.kernel decorator and supports autotuning.
        """
        return self.helion_kernel_func.autotune(*args, **kwargs)

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
                        returns the best helion.Config or None

        Returns:
            The registered function (for decorator usage)

        Example:
            @kernel_wrapper.register_config_picker
            def pick_config(model_config, available_configs):
                target_size = model_config.get_hidden_size()
                return available_configs.get(str(target_size))
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
    ) -> Optional["helion.Config"]:
        """
        Select the best config for model_config from available options.

        This is a pure function that performs config selection logic without any I/O.
        The registered picker function should implement kernel-specific selection strategies.

        Args:
            model_config: vLLM ModelConfig instance
            available_configs: Dictionary mapping config keys to loaded Helion configs

        Returns:
            Best matching Helion config from available_configs, or None if no suitable match
        """
        if self._config_picker is None:
            raise NotImplementedError(
                f"No config picker registered for kernel '{self.op_name}'. "
                f"Use @{self.op_name}.register_config_picker to register one."
            )
        return self._config_picker(model_config, available_configs)

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

        # Get the Helion kernel function
        kernel_fn = self.helion_kernel_func
        if kernel_fn is None:
            raise RuntimeError(
                f"No Helion kernel available for {self.op_name}"
            )

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
            logger.info(
                f"Autotuning {self.op_name} for config: {config_key}"
            )

            try:
                # Use Helion's built-in autotune method
                config = kernel_fn.autotune(inputs, **default_tuner_kwargs)
                results[config_key] = config

            except Exception as e:
                logger.error(
                    f"Autotuning failed for {self.op_name} config {config_key}: {e}"
                )

        return results

    def configure(self, model_config=None, config_manager=None):
        """
        Configure the kernel with optimal config for the given model.

        This is the main entry point for fusion passes - they call this method
        to configure the kernel with the best config, then can call the kernel directly.

        Args:
            model_config: vLLM ModelConfig for optimal config selection
            config_manager: ConfigManager instance for loading configs

        Example:
            # In fusion pass:
            helion_kernel.configure(vllm_config.model_config, config_manager)
            return helion_kernel(input, scale)
        """
        # Handle config selection logic here
        assert model_config is not None, (
            f"{self.op_name}.configure() requires model_config to be provided"
        )

        assert config_manager is not None, (
            f"{self.op_name}.configure() requires config_manager to be provided"
        )

        kernel_name = self.op_name
        assert kernel_name, (
            f"{self.op_name}.op_name returned None or empty string"
        )

        # Load available configs using ConfigManager
        available_configs = config_manager.load_all_configs(kernel_name)

        assert available_configs, (
            f"No configs available for kernel '{kernel_name}' - ensure configs are properly saved"
        )

        # Use our selection logic
        optimal_config = self.get_best_config(model_config, available_configs)

        # get_best_config() must return a config if configs are available
        assert optimal_config is not None, (
            f"{self.op_name}.get_best_config() returned None with {len(available_configs)} configs available"
        )

        # Set the config on the kernel
        self.set_config(optimal_config)


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
                "Helion not available, skipping registration of kernel '%s'", final_op_name
            )
            return helion_kernel_func

        from helion.runtime.kernel import Kernel as HelionKernel

        if not isinstance(helion_kernel_func, HelionKernel):
            raise ValueError(
                f"Function {helion_kernel_func.__name__} is not a Helion kernel. "
                "Make sure to apply @helion.kernel decorator before @register_kernel."
            )

        # Create the HelionKernelWrapper that will be registered as PyTorch custom op
        namespace = "vllm_helion"  # Fixed namespace for all Helion kernels
        kernel_wrapper = HelionKernelWrapper(helion_kernel_func, final_op_name, namespace)

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
                bound = helion_kernel_func.bind(args)

                # Use default config for fake execution (only for shape inference)
                config = (
                    helion_kernel_func.configs[0]
                    if hasattr(helion_kernel_func, "configs")
                    and helion_kernel_func.configs
                    else helion.Config()
                )

                compiled_runner = bound.compile_config(config)

                return compiled_runner(
                    *args, **kwargs, _launcher=lambda *args, **kwargs: None
                )

            final_fake_impl = helion_fake_kernel

        # Register using vLLM's direct_register_custom_op
        direct_register_custom_op(
            op_name=final_op_name,
            op_func=kernel_wrapper,
            mutates_args=mutates_args_list,
            fake_impl=final_fake_impl,
            target_lib=vllm_helion_lib,
        )

        # Preserve references for advanced use and debugging
        kernel_wrapper._pytorch_op_wrapper = None  # No longer applicable with direct registration
        kernel_wrapper._helion_kernel = helion_kernel_func

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
            kernel_wrapper.full_op_name,
        )

        # Register in global registry for fast lookup
        _REGISTERED_KERNELS[final_op_name] = kernel_wrapper

        # Return the HelionKernelWrapper - this is what fusion passes will use
        return kernel_wrapper

    return decorator
