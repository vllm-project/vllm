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
CUSTOM_TYPE_MMDICT = 3

logger = init_logger(__name__)
bytestr = Union[bytes, bytearray, memoryview, zmq.Frame]

# explicit representation of the encoded tensor information
class CustomArray(msgspec.Struct):
    d: str  
    s: tuple[int, ...]
    i: int

# msgspec confuses lists and tuples, so we need explicit fields rather than an union
class NestedArray(msgspec.Struct):
    a: CustomArray = None
    l: list[CustomArray] = None
    t: tuple[CustomArray, ...] = None

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
            d = { k: self._encode_nested(obj[k]) for k in obj }
            # returning d here should just work, but it seems that if 
            # it's a top-level object, msgspec just ignores the decoder hook.
            return msgpack.Ext(CUSTOM_TYPE_MMDICT, msgpack.encode(d))

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
            return NestedArray(a = self._encode_ndarray(nt.numpy()))
        if isinstance(nt, list):
            return NestedArray(l = [self._encode_nested(x) for x in nt])
        if isinstance(nt, tuple):
            lst = list(nt)
            lst[0] = self._encode_ndarray(lst[0].numpy())
            return NestedArray(t = tuple(lst))
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
            # Note: this is not used - it doesn't seem to work for top-level MultiModalKwargs
            # (dec_hook is not invoked). Instead we use Ext, see ext_hook below. 
            if issubclass(t, MultiModalKwargs):
                return MultiModalKwargs({ k: self._decode_nested(obj[k]) for k in obj })
            if issubclass(t, torch.Tensor):
                return torch.from_numpy(self._decode_ndarray(obj))
        return obj

    def _decode_ndarray(self, arr: CustomArray) -> np.ndarray:
        # for some reason, msgpack doesn't reconstruct CustomArray properly, but returns a dict
        if isinstance(arr, dict):
            arr = CustomArray(arr['d'], arr['s'], arr['i'])
        return np.ndarray(buffer=self.aux_buffers[arr.i],
                          dtype=np.dtype(arr.d),
                          shape=arr.s)

    def _decode_nested(self, na: NestedArray) -> NestedTensors:
        if isinstance(na, dict):
            na = NestedArray(na['a'], na['l'], na['t'])
        if na.a: #array
            return torch.from_numpy(self._decode_ndarray(na.a))
        # tuples get converted to lists
        if na.l: #list
            return [self._decode_nested(x) for x in na.l]
        if na.t:
            lst = list(na.t)
            lst[0] = torch.from_numpy(self._decode_ndarray(lst[0]))
            return tuple(lst)
        raise TypeError(f"Unexpected NestedArray contents: {na}")

    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_PICKLE:
            return pickle.loads(data)
        if code == CUSTOM_TYPE_CLOUDPICKLE:
            return cloudpickle.loads(data)
        if code == CUSTOM_TYPE_MMDICT:
             obj = msgpack.decode(data)
             return MultiModalKwargs({ k: self._decode_nested(obj[k]) for k in obj })

        raise NotImplementedError(
            f"Extension type code {code} is not supported")
