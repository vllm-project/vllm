# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import importlib
import pickle
from collections.abc import Sequence
from inspect import isclass
from types import FunctionType
from typing import Any, Callable, Optional, Union

import cloudpickle
import msgspec
import numpy as np
import torch
import zmq
from msgspec import msgpack

from vllm import envs
from vllm.logger import init_logger
# yapf: disable
from vllm.multimodal.inputs import (BaseMultiModalField,
                                    MultiModalBatchedField,
                                    MultiModalFieldConfig, MultiModalFieldElem,
                                    MultiModalFlatField, MultiModalKwargs,
                                    MultiModalKwargsItem,
                                    MultiModalKwargsItems,
                                    MultiModalSharedField, NestedTensors)
# yapf: enable
from vllm.v1.engine import UtilityResult

logger = init_logger(__name__)

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


def _log_insecure_serialization_warning():
    logger.warning_once("Allowing insecure serialization using pickle due to "
                        "VLLM_ALLOW_INSECURE_SERIALIZATION=1")


def _typestr(val: Any) -> Optional[tuple[str, str]]:
    if val is None:
        return None
    t = type(val)
    return t.__module__, t.__qualname__


def _encode_type_info_recursive(obj: Any) -> Any:
    """Recursively encode type information for nested structures of
    lists/dicts."""
    if obj is None:
        return None
    if type(obj) is list:
        return [_encode_type_info_recursive(item) for item in obj]
    if type(obj) is dict:
        return {k: _encode_type_info_recursive(v) for k, v in obj.items()}
    return _typestr(obj)


def _decode_type_info_recursive(
        type_info: Any, data: Any, convert_fn: Callable[[Sequence[str], Any],
                                                        Any]) -> Any:
    """Recursively decode type information for nested structures of
    lists/dicts."""
    if type_info is None:
        return data
    if isinstance(type_info, dict):
        assert isinstance(data, dict)
        return {
            k: _decode_type_info_recursive(type_info[k], data[k], convert_fn)
            for k in type_info
        }
    if isinstance(type_info, list) and (
            # Exclude serialized tensors/numpy arrays.
            len(type_info) != 2 or not isinstance(type_info[0], str)):
        assert isinstance(data, list)
        return [
            _decode_type_info_recursive(ti, d, convert_fn)
            for ti, d in zip(type_info, data)
        ]
    return convert_fn(type_info, data)


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
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

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

        if isinstance(obj, slice):
            # We are assuming only int-based values will be used here.
            return tuple(
                int(v) if v is not None else None
                for v in (obj.start, obj.stop, obj.step))

        if isinstance(obj, MultiModalKwargsItem):
            return self._encode_mm_item(obj)

        if isinstance(obj, MultiModalKwargsItems):
            return self._encode_mm_items(obj)

        if isinstance(obj, MultiModalKwargs):
            return self._encode_mm_kwargs(obj)

        if isinstance(obj, UtilityResult):
            result = obj.result
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                return None, result
            # Since utility results are not strongly typed, we recursively
            # encode type information for nested structures of lists/dicts
            # to help with correct msgspec deserialization.
            return _encode_type_info_recursive(result), result

        if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            raise TypeError(f"Object of type {type(obj)} is not serializable"
                            "Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow "
                            "fallback to pickle-based serialization.")

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
        arr_data = obj.data if obj.flags.c_contiguous else obj.tobytes()
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
        # view the tensor as a contiguous 1D array of bytes
        arr = obj.flatten().contiguous().cpu().view(torch.uint8).numpy()
        if obj.nbytes < self.size_threshold:
            # Smaller tensors are encoded inline, just like ndarrays.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr.data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr.data)
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, obj.shape, data

    def _encode_mm_items(self, items: MultiModalKwargsItems) -> dict[str, Any]:
        return {
            modality: [self._encode_mm_item(item) for item in itemlist]
            for modality, itemlist in items.items()
        }

    def _encode_mm_item(self,
                        item: MultiModalKwargsItem) -> list[dict[str, Any]]:
        return [self._encode_mm_field_elem(elem) for elem in item.values()]

    def _encode_mm_field_elem(self,
                              elem: MultiModalFieldElem) -> dict[str, Any]:
        return {
            "modality":
            elem.modality,
            "key":
            elem.key,
            "data": (None if elem.data is None else
                     self._encode_nested_tensors(elem.data)),
            "field":
            self._encode_mm_field(elem.field),
        }

    def _encode_mm_kwargs(self, kw: MultiModalKwargs) -> dict[str, Any]:
        return {
            modality: self._encode_nested_tensors(data)
            for modality, data in kw.items()
        }

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
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

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
            if t is slice:
                return slice(*obj)
            if issubclass(t, MultiModalKwargsItem):
                return self._decode_mm_item(obj)
            if issubclass(t, MultiModalKwargsItems):
                return self._decode_mm_items(obj)
            if issubclass(t, MultiModalKwargs):
                return self._decode_mm_kwargs(obj)
            if t is UtilityResult:
                return self._decode_utility_result(obj)
        return obj

    def _decode_utility_result(self, obj: Any) -> UtilityResult:
        result_type, result = obj
        if result_type is not None:
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise TypeError("VLLM_ALLOW_INSECURE_SERIALIZATION must "
                                "be set to use custom utility result types")
            # Use recursive decoding to handle nested structures
            result = _decode_type_info_recursive(result_type, result,
                                                 self._convert_result)
        return UtilityResult(result)

    def _convert_result(self, result_type: Sequence[str], result: Any) -> Any:
        if result_type is None:
            return result
        mod_name, name = result_type
        mod = importlib.import_module(mod_name)
        result_type = getattr(mod, name)
        return msgspec.convert(result, result_type, dec_hook=self.dec_hook)

    def _decode_ndarray(self, arr: Any) -> np.ndarray:
        dtype, shape, data = arr
        # zero-copy decode. We assume the ndarray will not be kept around,
        # as it now locks the whole received message buffer in memory.
        buffer = self.aux_buffers[data] if isinstance(data, int) else data
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)

    def _decode_tensor(self, arr: Any) -> torch.Tensor:
        dtype, shape, data = arr
        # Copy from inline representation, to decouple the memory storage
        # of the message from the original buffer. And also make Torch
        # not complain about a readonly memoryview.
        buffer = self.aux_buffers[data] if isinstance(data, int) \
            else bytearray(data)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        if not buffer:  # torch.frombuffer doesn't like empty buffers
            assert 0 in shape
            return torch.empty(shape, dtype=torch_dtype)
        # Create uint8 array
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)

    def _decode_mm_items(self, obj: dict[str, Any]) -> MultiModalKwargsItems:
        return MultiModalKwargsItems({
            modality: [self._decode_mm_item(item) for item in itemlist]
            for modality, itemlist in obj.items()
        })

    def _decode_mm_item(self, obj: list[Any]) -> MultiModalKwargsItem:
        return MultiModalKwargsItem.from_elems(
            [self._decode_mm_field_elem(v) for v in obj])

    def _decode_mm_field_elem(self, obj: dict[str,
                                              Any]) -> MultiModalFieldElem:
        if obj["data"] is not None:
            obj["data"] = self._decode_nested_tensors(obj["data"])

        # Reconstruct the field processor using MultiModalFieldConfig
        factory_meth_name, *field_args = obj["field"]
        factory_meth = getattr(MultiModalFieldConfig, factory_meth_name)

        # Special case: decode the union "slices" field of
        # MultiModalFlatField
        if factory_meth_name == "flat":
            field_args[0] = self._decode_nested_slices(field_args[0])

        obj["field"] = factory_meth(None, *field_args).field
        return MultiModalFieldElem(**obj)

    def _decode_mm_kwargs(self, obj: dict[str, Any]) -> MultiModalKwargs:
        return MultiModalKwargs({
            modality: self._decode_nested_tensors(data)
            for modality, data in obj.items()
        })

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

    def _decode_nested_slices(self, obj: Any) -> Any:
        assert isinstance(obj, (list, tuple))
        if obj and not isinstance(obj[0], (list, tuple)):
            return slice(*obj)
        return [self._decode_nested_slices(x) for x in obj]

    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data

        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            if code == CUSTOM_TYPE_PICKLE:
                return pickle.loads(data)
            if code == CUSTOM_TYPE_CLOUDPICKLE:
                return cloudpickle.loads(data)

        raise NotImplementedError(
            f"Extension type code {code} is not supported")
