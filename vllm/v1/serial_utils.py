# SPDX-License-Identifier: Apache-2.0

import pickle
from collections.abc import Sequence
from inspect import isclass
from types import FunctionType
from typing import Any, Optional, Union

import cloudpickle
import msgspec
import numpy as np
import torch
import zmq
from msgspec import msgpack

from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalKwargs, NestedTensors

CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2

logger = init_logger(__name__)
bytestr = Union[bytes, bytearray, memoryview, zmq.Frame]


# explicit representation of the encoded tensor information
class CustomArray(msgspec.Struct):
    d: str
    s: tuple[int, ...]
    i: int


# msgspec confuses lists and tuples, so we need a struct rather than an union
class NestedArray(msgspec.Struct,
                  omit_defaults=True):  # type: ignore[call-arg]
    A: Optional[CustomArray] = None
    L: Optional[list[CustomArray]] = None
    T: Optional[tuple[CustomArray, ...]] = None


class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization."""

    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        self.aux_buffers: Optional[list[bytestr]] = None

    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            self.aux_buffers = bufs = [b'']
            bufs[0] = self.encoder.encode(obj)
            return bufs
        finally:
            self.aux_buffers = None

    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]:
        try:
            self.aux_buffers = [buf]
            bufs = self.aux_buffers
            self.encoder.encode_into(obj, buf)
            return bufs
        finally:
            self.aux_buffers = None

    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return self._encode_ndarray(obj.numpy())

        # Fall back to pickle for object or void kind ndarrays.
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ('O', 'V'):
            return self._encode_ndarray(obj)

        if isinstance(obj, MultiModalKwargs):
            d = {k: self._encode_nested(obj[k]) for k in obj}
            return d

        if isinstance(obj, FunctionType):
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        return msgpack.Ext(CUSTOM_TYPE_PICKLE,
                           pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_ndarray(self, obj: np.ndarray) -> CustomArray:
        assert self.aux_buffers is not None
        obj = np.ascontiguousarray(obj)
        index = len(self.aux_buffers)
        self.aux_buffers.append(obj.data)
        return CustomArray(obj.dtype.str, obj.shape, index)

    def _encode_nested(self, nt: NestedTensors) -> NestedArray:
        if isinstance(nt, torch.Tensor):
            return NestedArray(A=self._encode_ndarray(nt.numpy()))
        if isinstance(nt, list):
            return NestedArray(L=[self._encode_nested(x) for x in nt])
        if isinstance(nt, tuple):
            lst = list(nt)
            lst[0] = self._encode_ndarray(lst[0].numpy())
            return NestedArray(T=tuple(lst))
        raise TypeError(f"Unexpected NestedTensors contents: {nt.type()}")


class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization."""

    def __init__(self, t: Optional[Any] = None):
        args = () if t is None else (t, )
        self.decoder = msgpack.Decoder(*args,
                                       ext_hook=self.ext_hook,
                                       dec_hook=self.dec_hook)
        self.aux_buffers: Sequence[bytestr] = ()

    def decode(self, bufs: Union[bytestr, Sequence[bytestr]]) -> Any:
        if isinstance(bufs, (bytes, bytearray, memoryview, zmq.Frame)):
            return self.decoder.decode(bufs)

        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])
        finally:
            self.aux_buffers = ()

    def dec_hook(self, t: type, obj: Any) -> Any:
        if isclass(t):
            if issubclass(t, np.ndarray):
                return self._decode_ndarray(obj)
            if issubclass(t, MultiModalKwargs):
                return MultiModalKwargs(
                    {k: self._decode_nested(obj[k])
                     for k in obj})
            if issubclass(t, torch.Tensor):
                return torch.from_numpy(self._decode_ndarray(obj))
        return obj

    def _decode_ndarray(self, arr: CustomArray) -> np.ndarray:
        # msgspec doesn't reconstruct CustomArray properly, just returns a dict
        if isinstance(arr, dict):
            arr = CustomArray(arr['d'], arr['s'], arr['i'])
        return np.ndarray(buffer=self.aux_buffers[arr.i],
                          dtype=np.dtype(arr.d),
                          shape=arr.s)

    def _decode_nested(self, na: NestedArray) -> NestedTensors:
        # same - NestedArray is not known to msgspec so it gets decoded as dict
        if isinstance(na, dict):
            na = NestedArray(na.get('A', None), na.get('L', None),
                             na.get('T', None))
        if na.A:  #array
            return torch.from_numpy(self._decode_ndarray(na.A))
        if na.L:  #list
            return [self._decode_nested(x) for x in na.L]
        if na.T:  #tuple
            lst = list(na.T)
            lst[0] = torch.from_numpy(self._decode_ndarray(lst[0]))
            return tuple(lst)
        raise TypeError(f"Unexpected NestedArray contents: {na}")

    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_PICKLE:
            return pickle.loads(data)
        if code == CUSTOM_TYPE_CLOUDPICKLE:
            return cloudpickle.loads(data)

        raise NotImplementedError(
            f"Extension type code {code} is not supported")
