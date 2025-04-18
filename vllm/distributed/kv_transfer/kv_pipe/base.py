# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file defines an interface `KVPipeBase`
that provides an abstraction for sending and receiving tensors, or None, via
distributed communications.

All classes instantiated from this interface are assumed to be a FIFO pipe.

If your distributed communication platform already supports key-value lookup,
you can bypass this interface and directly start from `kv_lookup_buffer`.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class KVPipeBase(ABC):
    """
    This class provides an interface for sending and receiving tensors, or
    None, by distributed communications.
    """

    @abstractmethod
    def send_tensor(self, tensor: Optional[torch.Tensor], target_rank: int = 0) -> None:
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
    def recv_tensor(self, src_rank: int) -> Optional[torch.Tensor]:
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
