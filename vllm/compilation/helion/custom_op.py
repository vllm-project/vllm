# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Base class for Helion-accelerated custom operations.

Helion custom ops extend vLLM's CustomOp infrastructure with:
- Helion kernel implementation (forward_helion)
- Enable/disable control via CompilationConfig.custom_ops
"""

from abc import abstractmethod

import torch

from vllm.model_executor.custom_op import CustomOp


# TODO(gmagogsfm): HelionCustomOp should also manage
# corresponding Helion kernel configs
class HelionCustomOp(CustomOp):
    """
    Base class for Helion-accelerated custom ops.

    This class extends vLLM's CustomOp to provide:
    - Helion kernel implementation (forward_helion)
    - Enable/disable checking via enabled() class method

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
