# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class UCCCommunicator:
    """
    UCCCommunicator provides a minimalistic interface for collective operations
    using PyTorch's native UCC (Unified Collective Communication) backend.

    UCC is a high-performance collective communication library that provides
    optimized implementations for various interconnects and can leverage
    hardware acceleration when available.
    """

    def __init__(self, group: ProcessGroup,
                 device: Union[int, str, torch.device]) -> None:
        """
        Initialize UCCCommunicator.

        Args:
            group: The process group to work on. Must be a UCC process group.
            device: The device to bind operations to.
        """
        self.disabled = True
        self.group = group

        # Validate that this is a UCC process group
        if dist.get_backend(group) != "ucc":
            logger.warning(
                "UCCCommunicator requires a UCC process group backend, "
                "but got backend: %s. Disabling UCC allreduce.",
                dist.get_backend(group))
            return

        # Set up device
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Store group size for convenience
        self.world_size = dist.get_world_size(group)

        # Enable UCC allreduce if validation passes
        self.disabled = False
        logger.info(
            "UCCCommunicator initialized successfully with UCC backend "
            "on device %s, world size: %d", self.device, self.world_size)

    def all_reduce(
            self,
            tensor: torch.Tensor,
            op: dist.ReduceOp = dist.ReduceOp.SUM) -> Optional[torch.Tensor]:
        """
        Perform allreduce operation using UCC backend.

        Args:
            tensor: Input tensor to reduce across all processes.
            op: Reduction operation (default: SUM).

        Returns:
            The reduced tensor or None if UCC allreduce is disabled.
        """
        if self.disabled:
            return None

        # Ensure tensor is on the correct device
        if tensor.device != self.device:
            tensor = tensor.to(self.device)

        # Perform allreduce using UCC backend
        try:
            dist.all_reduce(tensor, op=op, group=self.group)
            return tensor
        except Exception as e:
            logger.warning(
                "UCC allreduce failed: %s. Falling back to regular allreduce.",
                str(e))
            return None

    def should_use_ucc_allreduce(self, tensor: torch.Tensor) -> bool:
        """
        Determine if UCC allreduce should be used for the given tensor.

        Args:
            tensor: The tensor to check.

        Returns:
            True if UCC allreduce should be used, False otherwise.
        """
        if self.disabled:
            return False

        tensor_size = tensor.numel() * tensor.element_size()

        # Use UCC for tensors less than 512MB and when world size > 1
        return tensor_size < 512 * 1024 * 1024 and self.world_size > 1

    def close(self) -> None:
        """
        Clean up resources.
        """
        # UCC process group cleanup is handled by PyTorch
        pass

    @staticmethod
    def is_ucc_available() -> bool:
        """
        Check if UCC backend is available in PyTorch.

        Returns:
            True if UCC backend is available, False otherwise.
        """
        if hasattr(dist, "is_ucc_available"):
            return dist.is_ucc_available()
        try:
            return hasattr(dist.Backend,
                           "UCC") or "ucc" in dist.Backend.__dict__.values()
        except Exception:
            return False
