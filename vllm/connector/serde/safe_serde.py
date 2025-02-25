# SPDX-License-Identifier: Apache-2.0

from typing import Union

import torch
from safetensors.torch import load, save

from vllm.connector.serde.serde import Deserializer, Serializer


class SafeSerializer(Serializer):

    def __init__(self):
        super().__init__()

    def to_bytes(self, t: torch.Tensor) -> bytes:
        return save({"tensor_bytes": t.cpu().contiguous()})


class SafeDeserializer(Deserializer):

    def __init__(self):
        # TODO: dtype options
        super().__init__(torch.float32)

    def from_bytes_normal(self, b: Union[bytearray, bytes]) -> torch.Tensor:
        return load(bytes(b))["tensor_bytes"]

    def from_bytes(self, b: Union[bytearray, bytes]) -> torch.Tensor:
        return self.from_bytes_normal(b)
