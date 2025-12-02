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
                return torch.ops.my_helion_lib.my_op(x)

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
        """Initialize with ConfigManager."""
        super().__init__(*args, **kwargs)
        self._config_manager = ConfigManager()

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

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def get_best_config(
        self, model_config, available_configs: dict[str, "helion.Config"]
    ) -> Optional["helion.Config"]:
        """
        Select the best config for model_config from available options.

        This is a pure function that performs config selection logic without any I/O.
        Subclasses should implement kernel-specific selection strategies.

        Args:
            model_config: vLLM ModelConfig instance
            available_configs: Dictionary mapping config keys to loaded Helion configs

        Returns:
            Best matching Helion config from available_configs, or None if no suitable match
        """
        raise NotImplementedError

    def autotune(
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
        kernel_fn = self.helion_kernel
        if kernel_fn is None:
            raise RuntimeError(
                f"No Helion kernel available for {self.__class__.__name__}"
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
                f"Autotuning {self.__class__.__name__} for config: {config_key}"
            )

            try:
                # Use Helion's built-in autotune method
                config = kernel_fn.autotune(inputs, **default_tuner_kwargs)
                results[config_key] = config

            except Exception as e:
                logger.error(
                    f"Autotuning failed for {self.__class__.__name__} config {config_key}: {e}"
                )

        return results

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
