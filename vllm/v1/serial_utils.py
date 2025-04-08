# SPDX-License-Identifier: Apache-2.0

import pickle
from types import FunctionType
from typing import Any, Optional, Union

import cloudpickle
import torch
import numpy
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


NestedArrays = Union[
    list["NestedArrays"], list[numpy.ndarray], numpy.ndarray, tuple[numpy.ndarray, ...]
]


def nested_tensors_to_numpy(nt: NestedTensors) -> NestedArrays:
    if isinstance(nt, torch.Tensor):
        return nt.numpy()
    if isinstance(nt, list):
        return [nested_tensors_to_numpy(x) for x in nt]
    if isinstance(nt, tuple):
        lst = list(nt)
        lst[0] = lst[0].numpy()
        return tuple(lst)
    raise TypeError(f"Unexpected NestedTensors contents: {nt.type()}")


def numpy_to_nested_tensors(nt: NestedArrays) -> NestedTensors:
    if isinstance(nt, numpy.ndarray):
        return torch.from_numpy(nt)
    if isinstance(nt, list):
        return [numpy_to_nested_tensors(x) for x in nt]
    if isinstance(nt, tuple):
        lst = list(nt)
        lst[0] = torch.from_numpy(lst[0])
        return tuple(lst)
    raise TypeError(f"Unexpected NestedArrays contents: {nt.type()}")


def custom_enc_hook(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        # NOTE(rob): it is fastest to use numpy + pickle
        # when serializing torch tensors.
        # https://gist.github.com/tlrmchlsmth/8067f1b24a82b6e2f90450e7764fa103 # noqa: E501
        return msgpack.Ext(CUSTOM_TYPE_TENSOR, pickle.dumps(obj.numpy()))

    # Clunkish workaround for https://github.com/vllm-project/vllm/issues/16185
    # if the object is multimodal kwargs, then convert to ndarrays before transmitting.
    if isinstance(obj, MultiModalKwargs):
        adict = {}
        try:
            for key in obj:
                adict[key] = nested_tensors_to_numpy(obj[key])
            return msgpack.Ext(CUSTOM_TYPE_TENSORDICT, pickle.dumps(adict))
        except TypeError as err:
            # fall back to pickle serializer.
            logger.warning(f"Unable to serialize MultiModalKwargs : {err}, falling back to pickle")
            pass

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
            numpydict[key] = numpy_to_nested_tensors(numpydict[key])
        return MultiModalKwargs(numpydict)

    raise NotImplementedError(f"Extension type code {code} is not supported")
