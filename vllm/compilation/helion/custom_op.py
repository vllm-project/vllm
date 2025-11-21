# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Base class for Helion-accelerated custom operations.

Helion custom ops extend vLLM's CustomOp infrastructure with:
- Helion kernel implementation
- PyTorch native fallback for torch.compile
"""

from abc import abstractmethod

import torch

from vllm.model_executor.custom_op import CustomOp


class HelionCustomOp(CustomOp):
    """
    Base class for Helion-accelerated custom ops.

    This class extends vLLM's CustomOp to provide:
    - Helion kernel implementation (forward_helion)
    - PyTorch native fallback (forward_native)

    Example:
        @CustomOp.register("my_helion_op")
        class MyHelionOp(HelionCustomOp):
            def forward_helion(self, x):
                return torch.ops.my_helion_lib.my_op(x)

            def forward_native(self, x):
                # PyTorch-native implementation
                return ...
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
        return self.forward_helion(*args, **kwargs)
