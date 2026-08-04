# SPDX-License-Identifier: Apache-2.0
# Standard
import abc
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


class Serializer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
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


class SerializerDebugWrapper(Serializer):
    def __init__(self, s: Serializer):
        self.s = s

    def to_bytes(self, t: torch.Tensor) -> bytes:
        start = time.perf_counter()
        bs = self.s.to_bytes(t)
        end = time.perf_counter()

        logger.debug(f"Serialization took {end - start:.2f} seconds")
        return bs


class Deserializer(metaclass=abc.ABCMeta):
    def __init__(self, dtype):
        self.dtype = dtype

    @abc.abstractmethod
    def from_bytes(self, bs: bytes) -> torch.Tensor:
        """
        Deserialize a pytorch tensor from bytes.

        Input:
            bytes: a stream of bytes

        Output:
            torch.Tensor: the deserialized pytorch tensor
        """
        raise NotImplementedError


class DeserializerDebugWrapper(Deserializer):
    def __init__(self, d: Deserializer):
        self.d = d

    @_lmcache_nvtx_annotate
    def from_bytes(self, t: bytes) -> torch.Tensor:
        start = time.perf_counter()
        ret = self.d.from_bytes(t)
        end = time.perf_counter()

        logger.debug(f"Deserialization took {(end - start) * 1000:.2f} ms")
        return ret
