# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Helion kernel registration decorators.

This module provides decorators to automatically register Helion kernels
as PyTorch custom operations with automatic fake kernel generation.
"""

from collections.abc import Callable

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Try to import Helion - it's an optional dependency
try:
    import helion

    HELION_AVAILABLE = True
except ImportError:
    HELION_AVAILABLE = False


def register_kernel(
    op_name: str,
    *,
    namespace: str = "vllm_helion",
    device_types: str = "cuda",
    mutates_args: tuple = (),
    fake_impl: Callable = None,
) -> Callable:
    """
    Decorator to register a Helion kernel as a PyTorch custom operation.

    This decorator automatically:
    1. Registers the kernel as a PyTorch custom op under the specified namespace
    2. Generates a fake implementation by running the kernel with _launcher=lambda *args, **kwargs: None
       OR uses the provided fake_impl if specified
    3. Provides proper error handling when Helion is not available

    Args:
        op_name: Name of the operation (will be registered as namespace::op_name)
        namespace: PyTorch custom op namespace (default: "vllm_helion")
        device_types: Target device types (default: "cuda")
        mutates_args: Tuple of argument names that are mutated (default: empty)
        fake_impl: Optional custom fake implementation. If provided, uses this instead
                  of the auto-generated _launcher approach. Should be a callable that
                  takes (*args, **kwargs) and returns a tensor with correct shape/dtype.

    Returns:
        Decorator function that registers the Helion kernel

    Examples:
        # Auto-generated fake kernel (uses _launcher approach):
        @helion.kernel(...)
        @register_kernel("silu_mul_fp8")
        def silu_mul_fp8_kernel(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            pass

        # Custom fake kernel (for compatibility with symbolic shapes):
        def my_fake_impl(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            # Manual shape inference logic
            output_shape = input.shape[:-1] + (input.shape[-1] // 2,)
            return torch.empty(output_shape, dtype=torch.float8_e4m3fn, device=input.device)

        @helion.kernel(...)
        @register_kernel("silu_mul_fp8", fake_impl=my_fake_impl)
        def silu_mul_fp8_kernel(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            pass

        # Now accessible as: torch.ops.vllm_helion.silu_mul_fp8(input, scale)
    """

    def decorator(helion_kernel_func: Callable) -> Callable:
        if not HELION_AVAILABLE:
            logger.warning(
                "Helion not available, skipping registration of kernel '%s'", op_name
            )
            return helion_kernel_func

        from helion.runtime.kernel import Kernel as HelionKernel

        if not isinstance(helion_kernel_func, HelionKernel):
            raise ValueError(
                f"Function {helion_kernel_func.__name__} is not a Helion kernel. "
                "Make sure to apply @helion.kernel decorator before @register_kernel."
            )

        # Full operation name with namespace
        full_op_name = f"{namespace}::{op_name}"

        # Apply the PyTorch custom op decorator
        pytorch_op_wrapper = torch.library.custom_op(
            full_op_name,
            mutates_args=mutates_args,
            device_types=device_types,
        )(helion_kernel_func)

        # Register fake implementation
        if fake_impl is not None:
            # Use user-provided fake implementation
            pytorch_op_wrapper.register_fake(fake_impl)
            logger.info(
                "Registered custom fake implementation for Helion kernel '%s'",
                helion_kernel_func.__name__,
            )
        else:
            # Create auto-generated fake implementation using proper Helion bind/compile pattern
            def helion_fake_kernel(*args, **kwargs):
                """
                Run the Helion kernel with a dummy launcher to generate correct output shapes.
                Uses the proper Helion bind/compile pattern: _launcher is an argument of CompiledConfig.
                This MUST work or the registration fails - no fallbacks.
                """
                bound = helion_kernel_func.bind(args)

                # Compile with a default configs should be fine as we are skipping all
                # on-device compute
                # TODO(gmagogsfm): In prod, we need to pick the right config before being
                # able to compile to config and run it.
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

            pytorch_op_wrapper.register_fake(helion_fake_kernel)
            logger.info(
                "Registered auto-generated fake implementation for Helion kernel '%s'",
                helion_kernel_func.__name__,
            )

        # Add a __call__ method to the pytorch_op_wrapper for direct invocation
        def pytorch_op_call_method(*args, **kwargs):
            """
            Convenience method to call the registered PyTorch custom op directly.

            This allows users to call the PyTorch wrapper directly like a function.

            Example:
                # Instead of: torch.ops.vllm_helion.silu_mul_fp8(input, scale)
                # You can do: decorated_kernel(input, scale)
            """
            # Get the registered op from torch.ops
            namespace_ops = getattr(torch.ops, namespace)
            registered_op = getattr(namespace_ops, op_name)
            return registered_op(*args, **kwargs)

        # Attach the call method to the pytorch_op_wrapper
        pytorch_op_wrapper.__call__ = pytorch_op_call_method

        # Preserve references for advanced use
        pytorch_op_wrapper._helion_kernel = helion_kernel_func
        pytorch_op_wrapper._op_name = op_name
        pytorch_op_wrapper._namespace = namespace

        logger.info(
            "Registered Helion kernel '%s' as PyTorch custom op '%s'",
            helion_kernel_func.__name__,
            full_op_name,
        )

        # Return the pytorch_op_wrapper instead of the Helion kernel
        return pytorch_op_wrapper

    return decorator
