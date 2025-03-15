# SPDX-License-Identifier: Apache-2.0

import abc
from abc import ABC, abstractmethod

import torch


class Serializer(ABC):

    @abstractmethod
    def to_bytes(self, t: torch.Tensor) -> bytes:
        """
        Serialize a pytorch tensor to bytes. The serialized bytes should contain
        both the data and the metadata (shape, dtype, etc.) of the tensor.

        Input:
            t: the input pytorch tensor, can be on any device, in any shape,
               with any dtype

        Returns:
            bytes: the serialized bytes
        """
        raise NotImplementedError


class Deserializer(metaclass=abc.ABCMeta):

    def __init__(self, dtype):
        self.dtype = dtype

    @abstractmethod
    def from_bytes(self, bs: bytes) -> torch.Tensor:
        """
        Deserialize a pytorch tensor from bytes.

        Input:
            bytes: a stream of bytes

        Output:
            torch.Tensor: the deserialized pytorch tensor
        """
        raise NotImplementedError
