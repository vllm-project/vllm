# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file defines an interface `KVPipeBase`
that provides an abstraction for sending and receiving tensors, or None, via
distributed communications.

All classes instantiated from this interface are assumed to be a FIFO pipe.

If your distributed communication platform already supports key-value lookup,
you can bypass this interface and directly start from `kv_lookup_buffer`.
"""

from abc import ABC, abstractmethod

import torch


class KVPipeBase(ABC):
    """
    This class provides an interface for sending and receiving tensors, or
    None, by distributed communications.
    """

    @abstractmethod
    def send_tensor(self, tensor: torch.Tensor | None) -> None:
        """Send a tensor, or None, via the pipe.

        Need to support sending None -- important for error handling.

        TODO: add a `key` argument so that we can use traditional
        key-value database as the distributed communication mechanism behind
        the pipe.

        Args:
            tensor (Optional[torch.Tensor]): The tensor to be sent. Can be None.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def recv_tensor(self) -> torch.Tensor | None:
        """Receive a tensor (can be None) from the pipeline.

        Returns:
            Optional[torch.Tensor]: The tensor received from the pipeline. Can
                                    be None.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the pipeline and release resources.

        This method is responsible for closing the communication pipeline
        and releasing any resources associated with it.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError
