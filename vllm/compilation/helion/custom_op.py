# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Base class for Helion-accelerated custom operations.

Helion custom ops extend vLLM's CustomOp infrastructure with:
- Helion kernel implementation (forward_helion)
- Enable/disable control via CompilationConfig.custom_ops
- Autotuning and config management
"""

from abc import abstractmethod

import torch

from vllm.compilation.helion.config_manager import ConfigManager
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp

logger = init_logger(__name__)

# Registry for CustomOp -> Benchmark class associations
_custom_op_benchmarks: dict[type["HelionCustomOp"], type] = {}

# Import Helion types conditionally
try:
    import helion

    HELION_AVAILABLE = True
except ImportError:
    helion = None
    HELION_AVAILABLE = False


# TODO(gmagogsfm): HelionCustomOp should also manage
# corresponding Helion kernel configs
class HelionCustomOp(CustomOp):
    """
    Base class for Helion-accelerated custom ops.

    This class extends vLLM's CustomOp to provide:
    - Helion kernel implementation (forward_helion)
    - Enable/disable checking via enabled() class method
    - Centralized config management via ConfigManager

    Example:
        @CustomOp.register("my_helion_op")
        class MyHelionOp(HelionCustomOp):
            def forward_helion(self, x):
                return torch.ops.vllm_helion.my_op(x)

    Checking if an op is enabled:
        # Class method (call on the class)
        if MyHelionOp.enabled():
            print("MyHelionOp is enabled")

        # Can also call on instance
        op = MyHelionOp()
        if op.enabled():
            print("This op is enabled")

    Controlling which ops are enabled:
        # Via config:
        --override-neuron-config '{"custom_ops": ["all"]}'  # Enable all (default)
        --override-neuron-config '{"custom_ops": ["none"]}'  # Disable all
        --override-neuron-config '{"custom_ops": ["all", "-my_helion_op"]}'  # Disable specific
        --override-neuron-config '{"custom_ops": ["none", "+my_helion_op"]}'  # Enable specific
    """

    def __init__(self, *args, **kwargs):
        """Initialize with singleton ConfigManager."""
        super().__init__(*args, **kwargs)
        self._config_manager = ConfigManager.get_instance()

    @abstractmethod
    def forward_helion(self, *args, **kwargs) -> torch.Tensor:
        """
        Helion kernel implementation.

        This method should call the Helion-compiled kernel.

        Args:
            *args: Input arguments for the kernel
            **kwargs: Keyword arguments for the kernel

        Returns:
            Output tensor from Helion kernel
        """
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs) -> torch.Tensor:
        """Route CUDA backend to Helion implementation."""
        if not HELION_AVAILABLE:
            raise ImportError(
                f"Helion is not installed. Please install Helion to use {self.__class__.__name__}. "
                "Alternatively, use the native implementation via forward_native()."
            )
        return self.forward_helion(*args, **kwargs)

    @classmethod
    def enabled(cls) -> bool:
        """
        Check if this Helion op is enabled.

        A Helion op is enabled only if:
        1. Helion is installed and importable
        2. The op is enabled via CompilationConfig.custom_ops (checked by base class)

        Returns:
            bool: True if both conditions are met, False otherwise
        """
        if not cls.is_helion_available():
            return False
        return super().enabled()

    @staticmethod
    def is_helion_available() -> bool:
        """
        Check if Helion is available in the current environment.

        Returns:
            bool: True if Helion can be imported, False otherwise

        Example:
            if HelionCustomOp.is_helion_available():
                from vllm.compilation.helion import SiluMulFp8Helion
                op = SiluMulFp8Helion()
        """
        try:
            import helion  # noqa: F401

            return True
        except ImportError:
            return False

    # Config Management Methods

    def configure(self, model_config=None):
        """
        Configure all kernels with optimal config for the given model.

        This is the main entry point for fusion passes - they call this method
        to configure all kernels with the best config, then can call the kernels directly.

        Args:
            model_config: vLLM ModelConfig for optimal config selection

        Example:
            # In fusion pass:
            helion_op.configure(vllm_config.model_config)
            return helion_op.forward_helion(input, scale)
        """
        assert self.enabled(), (
            f"{self.__class__.__name__} HelionCustomOp must be enabled"
        )

        # Handle case where model_config is None (e.g., during testing/compilation)
        if model_config is None:
            raise AssertionError(
                f"{self.__class__.__name__}.configure() requires model_config to be provided. "
                "Tests should create a mock ModelConfig with appropriate hidden size."
            )

        # Delegate to all helion kernels' configure methods
        helion_kernels = self.helion_kernels
        assert helion_kernels is not None and len(helion_kernels) > 0, (
            f"{self.__class__.__name__}.helion_kernels returned empty list - ensure Helion is available"
        )

        for kernel in helion_kernels:
            kernel.configure(model_config, self._config_manager)

    @property
    @abstractmethod
    def helion_kernels(self):
        """
        The Helion kernel functions for this custom operation.

        Subclasses should override this to return a list of their specific Helion kernel functions
        (those decorated with @helion.kernel). For single-kernel operations, return a list with one element.

        Returns:
            List[HelionKernelWrapper]: List of Helion kernel wrappers, or empty list if not available
        """
        raise NotImplementedError

    # Benchmark Registration Methods

    @classmethod
    def register_benchmark(cls, benchmark_class):
        """
        Register a benchmark class for this CustomOp.

        Args:
            benchmark_class: KernelBenchmark subclass to associate with this CustomOp

        Returns:
            The registered benchmark class (for decorator usage)

        Example:
            @SiluMulFp8Helion.register_benchmark
            class SiluMulFp8Benchmark(KernelBenchmark):
                ...
        """
        _custom_op_benchmarks[cls] = benchmark_class
        return benchmark_class

    @classmethod
    def get_benchmark(cls):
        """
        Get the registered benchmark class for this CustomOp.

        Returns:
            KernelBenchmark subclass, or None if no benchmark is registered
        """
        return _custom_op_benchmarks.get(cls)


def get_registered_custom_ops() -> dict[str, type["HelionCustomOp"]]:
    """
    Get all registered Helion CustomOps.

    Returns:
        Dictionary mapping CustomOp names to classes
    """
    # Get all registered CustomOps from vLLM's registry
    from vllm.model_executor.custom_op import CustomOp

    registered_ops = {}

    # Access the CustomOp registries
    for registry in [CustomOp.op_registry, CustomOp.op_registry_oot]:
        for op_name, op_class in registry.items():
            # Check if it's a HelionCustomOp
            if issubclass(op_class, HelionCustomOp):
                registered_ops[op_name] = op_class

    return registered_ops
