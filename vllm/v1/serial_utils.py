# SPDX-License-Identifier: Apache-2.0

import pickle
from typing import Any, Optional

import torch
from msgspec import msgpack

CUSTOM_TYPE_TENSOR = 1
CUSTOM_TYPE_PICKLE = 2


class MsgpackEncoder:
    """Encoder with custom torch tensor serialization."""

    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=custom_enc_hook)

    def encode(self, obj: Any) -> bytes:
        return self.encoder.encode(obj)

    def encode_into(self, obj: Any, buf: bytearray) -> None:
        self.encoder.encode_into(obj, buf)


class MsgpackDecoder:
    """Decoder with custom torch tensor serialization."""

    def __init__(self, t: Optional[Any] = None):
        args = () if t is None else (t, )
        self.decoder = msgpack.Decoder(*args, ext_hook=custom_ext_hook)

    def decode(self, obj: Any):
        return self.decoder.decode(obj)


def custom_enc_hook(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        # NOTE(rob): it is fastest to use numpy + pickle
        # when serializing torch tensors.
        # https://gist.github.com/tlrmchlsmth/8067f1b24a82b6e2f90450e7764fa103 # noqa: E501
        return msgpack.Ext(CUSTOM_TYPE_TENSOR, pickle.dumps(obj.numpy()))

    return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj))


def custom_ext_hook(code: int, data: memoryview) -> Any:
    if code == CUSTOM_TYPE_TENSOR:
        return torch.from_numpy(pickle.loads(data))
    if code == CUSTOM_TYPE_PICKLE:
        return pickle.loads(data)

    raise NotImplementedError(f"Extension type code {code} is not supported")
