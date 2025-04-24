# SPDX-License-Identifier: Apache-2.0

import dataclasses
import pickle
from collections.abc import Sequence
from inspect import isclass
from types import FunctionType
from typing import Any, Optional, Union

import cloudpickle
import numpy as np
import torch
import zmq
from msgspec import msgpack

from vllm import envs
from vllm.multimodal.inputs import (BaseMultiModalField,
                                    MultiModalBatchedField,
                                    MultiModalFieldConfig, MultiModalFieldElem,
                                    MultiModalFlatField, MultiModalKwargs,
                                    MultiModalKwargsItem,
                                    MultiModalSharedField, NestedTensors)

CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_RAW_VIEW = 3

# MultiModalField class serialization type map.
# These need to list all possible field types and match them
# to factory methods in `MultiModalFieldConfig`.
MMF_CLASS_TO_FACTORY: dict[type[BaseMultiModalField], str] = {
    MultiModalFlatField: "flat",
    MultiModalSharedField: "shared",
    MultiModalBatchedField: "batched",
}

bytestr = Union[bytes, bytearray, memoryview, zmq.Frame]


class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    By default, arrays below 256B are serialized inline Larger will get sent 
    via dedicated messages. Note that this is a per-tensor limit.
    """

    def __init__(self, size_threshold: Optional[int] = None):
        if size_threshold is None:
            size_threshold = envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        # This is used as a local stash of buffers that we can then access from
        # our custom `msgspec` hook, `enc_hook`. We don't have a way to
        # pass custom data to the hook otherwise.
        self.aux_buffers: Optional[list[bytestr]] = None
        self.size_threshold = size_threshold

    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            self.aux_buffers = bufs = [b'']
            bufs[0] = self.encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
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
            return self._encode_tensor(obj)

        # Fall back to pickle for object or void kind ndarrays.
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ('O', 'V'):
            return self._encode_ndarray(obj)

        if isinstance(obj, MultiModalKwargs):
            mm: MultiModalKwargs = obj
            if not mm.modalities:
                # just return the main dict if there are no modalities.
                return dict(mm)

            # ignore the main dict, it will be re-indexed.
            # Encode a list of MultiModalKwargsItems as plain dicts
            # + special handling for .field.
            # Any tensors *not* indexed by modality will be ignored.
            return [[{
                "modality": elem.modality,
                "key": elem.key,
                "data": self._encode_nested_tensors(elem.data),
                "field": self._encode_mm_field(elem.field),
            } for elem in item.values()]
                    for itemlist in mm._items_by_modality.values()
                    for item in itemlist]

        if isinstance(obj, FunctionType):
            # `pickle` is generally faster than cloudpickle, but can have
            # problems serializing methods.
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        return msgpack.Ext(CUSTOM_TYPE_PICKLE,
                           pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_ndarray(
        self, obj: np.ndarray
    ) -> tuple[str, tuple[int, ...], Union[int, memoryview]]:
        assert self.aux_buffers is not None
        # If the array is non-contiguous, we need to copy it first
        arr_data = obj.data if obj.data.c_contiguous else obj.tobytes()
        if not obj.shape or obj.nbytes < self.size_threshold:
            # Encode small arrays and scalars inline. Using this extension type
            # ensures we can avoid copying when decoding.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr_data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr_data)

        # We serialize the ndarray as a tuple of native types.
        # The data is either inlined if small, or an index into a list of
        # backing buffers that we've stashed in `aux_buffers`.
        return obj.dtype.str, obj.shape, data

    def _encode_tensor(
        self, obj: torch.Tensor
    ) -> tuple[str, tuple[int, ...], Union[int, memoryview]]:
        assert self.aux_buffers is not None
        # this creates a copy of the tensor if it's not already contiguous
        obj = obj.contiguous()
        #  view the tensor as a 1D array of bytes
        arr = obj.view((obj.numel(), )).view(torch.uint8).numpy()
        if obj.nbytes < self.size_threshold:
            # Smaller tensors are encoded inline, just like ndarrays.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr.data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr.data)
        dtype = str(obj.dtype)[6:]  # remove 'torch.' prefix
        return dtype, obj.shape, data

    def _encode_nested_tensors(self, nt: NestedTensors) -> Any:
        if isinstance(nt, torch.Tensor):
            return self._encode_tensor(nt)
        if isinstance(nt, (int, float)):
            # Although it violates NestedTensors type, MultiModalKwargs
            # values are sometimes floats.
            return nt
        return [self._encode_nested_tensors(x) for x in nt]

    def _encode_mm_field(self, field: BaseMultiModalField):
        # Figure out the factory name for the field type.
        name = MMF_CLASS_TO_FACTORY.get(field.__class__)
        if not name:
            raise TypeError(f"Unsupported field type: {field.__class__}")
        # We just need to copy all of the field values in order
        # which will be then used to reconstruct the field.
        field_values = (getattr(field, f.name)
                        for f in dataclasses.fields(field))
        return name, *field_values


class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    def __init__(self, t: Optional[Any] = None):
        args = () if t is None else (t, )
        self.decoder = msgpack.Decoder(*args,
                                       ext_hook=self.ext_hook,
                                       dec_hook=self.dec_hook)
        self.aux_buffers: Sequence[bytestr] = ()

    def decode(self, bufs: Union[bytestr, Sequence[bytestr]]) -> Any:
        if isinstance(bufs, (bytes, bytearray, memoryview, zmq.Frame)):
            # TODO - This check can become `isinstance(bufs, bytestr)`
            # as of Python 3.10.
            return self.decoder.decode(bufs)

        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])
        finally:
            self.aux_buffers = ()

    def dec_hook(self, t: type, obj: Any) -> Any:
        # Given native types in `obj`, convert to type `t`.
        if isclass(t):
            if issubclass(t, np.ndarray):
                return self._decode_ndarray(obj)
            if issubclass(t, torch.Tensor):
                return self._decode_tensor(obj)
            if issubclass(t, MultiModalKwargs):
                if isinstance(obj, list):
                    return MultiModalKwargs.from_items(
                        self._decode_mm_items(obj))
                return MultiModalKwargs({
                    k: self._decode_nested_tensors(v)
                    for k, v in obj.items()
                })
        return obj

    def _decode_ndarray(self, arr: Any) -> np.ndarray:
        dtype, shape, data = arr
        # zero-copy decode. We assume the ndarray will not be kept around,
        # as it now locks the whole received message buffer in memory.
        buffer = self.aux_buffers[data] if isinstance(data, int) else data
        return np.ndarray(buffer=buffer, dtype=np.dtype(dtype), shape=shape)

    def _decode_tensor(self, arr: Any) -> torch.Tensor:
        dtype, shape, data = arr
        # Copy from inline representation, to decouple the memory storage
        # of the message from the original buffer. And also make Torch
        # not complain about a readonly memoryview.
        buffer = self.aux_buffers[data] if isinstance(data, int) \
            else bytearray(data)
        # Create numpy wrapper around the bytes
        arr = np.ndarray(buffer=buffer, dtype=np.uint8, shape=(len(buffer), ))
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        # Convert back to proper shape & type
        return torch.from_numpy(arr).view(torch_dtype).view(shape)

    def _decode_mm_items(self, obj: list) -> list[MultiModalKwargsItem]:
        decoded_items = []
        for item in obj:
            elems = []
            for v in item:
                v["data"] = self._decode_nested_tensors(v["data"])
                # Reconstruct the field processor using MultiModalFieldConfig
                factory_meth_name, *field_args = v["field"]
                factory_meth = getattr(MultiModalFieldConfig,
                                       factory_meth_name)
                v["field"] = factory_meth(None, *field_args).field
                elems.append(MultiModalFieldElem(**v))
            decoded_items.append(MultiModalKwargsItem.from_elems(elems))
        return decoded_items

    def _decode_nested_tensors(self, obj: Any) -> NestedTensors:
        if isinstance(obj, (int, float)):
            # Although it violates NestedTensors type, MultiModalKwargs
            # values are sometimes floats.
            return obj
        if not isinstance(obj, list):
            raise TypeError(f"Unexpected NestedTensors contents: {type(obj)}")
        if obj and isinstance(obj[0], str):
            return self._decode_tensor(obj)
        return [self._decode_nested_tensors(x) for x in obj]

    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data
        if code == CUSTOM_TYPE_PICKLE:
            return pickle.loads(data)
        if code == CUSTOM_TYPE_CLOUDPICKLE:
            return cloudpickle.loads(data)

        raise NotImplementedError(
            f"Extension type code {code} is not supported")
