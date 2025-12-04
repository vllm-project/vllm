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
from typing import Optional

import torch

from vllm.compilation.helion.config_manager import ConfigManager
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp

logger = init_logger(__name__)

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

    # Autotuning and Config Management Methods

    def autotune(
        self, autotune_inputs: dict[str, tuple], tuner_kwargs: dict | None = None
    ) -> dict[str, "helion.Config"]:
        """
        Run autotuning and return configs (without saving).

        Delegates to the helion_kernel's run_autotune method.

        Args:
            autotune_inputs: Dictionary mapping config keys to input tuples for autotuning
            tuner_kwargs: Additional arguments for Helion tuner

        Returns:
            Dictionary mapping config keys to tuned Helion configs
        """
        helion_kernel = self.helion_kernel
        assert helion_kernel is not None, (
            f"{self.__class__.__name__}.helion_kernel returned None - ensure Helion is available"
        )

        return helion_kernel.run_autotune(autotune_inputs, tuner_kwargs)

    def configure(self, model_config=None):
        """
        Configure the kernel with optimal config for the given model.

        This is the main entry point for fusion passes - they call this method
        to configure the kernel with the best config, then can call the kernel directly.

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

        assert model_config is not None, (
            f"{self.__class__.__name__}.configure() requires model_config to be provided"
        )

        # Delegate to helion_kernel's configure method
        helion_kernel = self.helion_kernel
        assert helion_kernel is not None, (
            f"{self.__class__.__name__}.helion_kernel returned None - ensure Helion is available"
        )

        helion_kernel.configure(model_config, self._config_manager)


    @property
    @abstractmethod
    def helion_kernel(self):
        """
        The Helion kernel function for autotuning.

        Subclasses should override this to return their specific Helion kernel function
        (the one decorated with @helion.kernel).

        Returns:
            The Helion kernel function, or None if not available
        """
        raise NotImplementedError
