# SPDX-License-Identifier: Apache-2.0

import pickle
from types import FunctionType
from typing import Any, Optional, Union

import cloudpickle
import msgspec
import numpy
import torch
from msgspec import msgpack

from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalKwargs, NestedTensors

CUSTOM_TYPE_TENSOR = 1
CUSTOM_TYPE_PICKLE = 2
CUSTOM_TYPE_CLOUDPICKLE = 3
CUSTOM_TYPE_TENSORDICT = 4

logger = init_logger(__name__)


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


class CustomArray(msgspec.Struct):
    dtype: str
    shape: list[int]
    buffer: msgspec.Raw


def encode_torch(t: torch.Tensor) -> CustomArray:
    a = t.numpy()
    return CustomArray(dtype = a.dtype.str,
                       shape = list(a.shape),
                       buffer = msgspec.Raw(b"foo"))


def decode_torch(c: CustomArray) -> torch.Tensor:
    a = numpy.ndarray(buffer=c.buffer,
                      dtype=numpy.dtype(c.dtype),
                      shape=tuple(c.shape))
    return torch.from_numpy(a)


NestedArrays = Union[list["NestedArrays"], list[CustomArray], CustomArray,
                     tuple[CustomArray, ...]]


def encode_nested_tensors(nt: NestedTensors) -> NestedArrays:
    if isinstance(nt, torch.Tensor):
        return encode_torch(nt)
    if isinstance(nt, list):
        return [encode_nested_tensors(x) for x in nt]
    if isinstance(nt, tuple):
        lst = list(nt)
        lst[0] = encode_torch(lst[0])
        return tuple(lst)
    raise TypeError(f"Unexpected NestedTensors contents: {nt.type()}")


def decode_nested_tensors(na: NestedArrays) -> NestedTensors:
    if isinstance(na, CustomArray):
        return decode_torch(na)
    if isinstance(na, list):
        return [decode_nested_tensors(x) for x in na]
    if isinstance(na, tuple):
        lst = list(na)
        lst[0] = decode_torch(lst[0])
        return tuple(lst)
    raise TypeError(f"Unexpected NestedArrays contents: {na.type()}")


def custom_enc_hook(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return msgpack.Ext(CUSTOM_TYPE_TENSOR, msgpack.encode(encode_torch(obj)))

    # Clunkish workaround for https://github.com/vllm-project/vllm/issues/16185
    # if the object is multimodal kwargs, then convert to ndarrays before transmitting.
    if isinstance(obj, MultiModalKwargs):
        adict = {}
        try:
            for key in obj:
                adict[key] = encode_nested_tensors(obj[key])
            return msgpack.Ext(CUSTOM_TYPE_TENSORDICT, msgpack.encode(adict))
        except TypeError as err:
            # fall back to pickle serializer.
            logger.warning("Unable to convert MultiModalKwargs (%s), fall back to pickle", err)
            pass

    if isinstance(obj, FunctionType):
        return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

    return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj))


def custom_ext_hook(code: int, data: memoryview) -> Any:
    if code == CUSTOM_TYPE_TENSOR:
        return decode_torch(msgpack.decode(data, type=CustomArray))
    if code == CUSTOM_TYPE_PICKLE:
        return pickle.loads(data)
    if code == CUSTOM_TYPE_CLOUDPICKLE:
        return cloudpickle.loads(data)
    if code == CUSTOM_TYPE_TENSORDICT:
        adict = msgpack.decode(data, type=dict)
        for key in adict:
            adict[key] = decode_nested_tensors(adict[key])
        return MultiModalKwargs(adict)

    raise NotImplementedError(f"Extension type code {code} is not supported")

dd = { "foo": torch.zeros(1000) }
mm = MultiModalKwargs(dd)
zz = { "foo": encode_nested_tensors(dd["foo"]) }
encoder = MsgpackEncoder()
bb = encoder.encode(encode_torch(torch.zeros(1000)))
decoder = MsgpackDecoder(CustomArray)
ee = decoder.decode(bb)
print(ee)