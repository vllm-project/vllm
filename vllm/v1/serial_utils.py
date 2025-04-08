# SPDX-License-Identifier: Apache-2.0

import pickle
from types import FunctionType
from typing import Any, Optional

import cloudpickle
import torch
from msgspec import msgpack

from vllm.multimodal.inputs import MultiModalKwargs

CUSTOM_TYPE_TENSOR = 1
CUSTOM_TYPE_PICKLE = 2
CUSTOM_TYPE_CLOUDPICKLE = 3
CUSTOM_TYPE_TENSORDICT = 4 

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

    # clunky workaround for https://github.com/vllm-project/vllm/issues/16185
    # if the object is multimodal kwargs, and can be represented as a simple dict
    # then convert to numpy format before transmitting.
    # TODO: support other NestedTensors forms
    if isinstance(obj, MultiModalKwargs):
        numpydict = {}
        total_size = 0
        for key in obj:
            if isinstance(obj[key], torch.Tensor):
                numpydict[key] = obj[key].numpy()
                total_size += obj[key].shape.numel() * obj[key].dtype.itemsize
        # check if all keys were converted
        if len(obj) == len(numpydict):
            msg = msgpack.Ext(CUSTOM_TYPE_TENSORDICT, pickle.dumps(numpydict))
            return msg

    if isinstance(obj, FunctionType):
        return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

    return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj))


def custom_ext_hook(code: int, data: memoryview) -> Any:
    if code == CUSTOM_TYPE_TENSOR:
        return torch.from_numpy(pickle.loads(data))
    if code == CUSTOM_TYPE_PICKLE:
        return pickle.loads(data)
    if code == CUSTOM_TYPE_CLOUDPICKLE:
        return cloudpickle.loads(data)
    if code == CUSTOM_TYPE_TENSORDICT:
        numpydict = pickle.loads(data)
        for key in numpydict:
            numpydict[key] = torch.from_numpy(numpydict[key])
        return MultiModalKwargs(numpydict)

    raise NotImplementedError(f"Extension type code {code} is not supported")
