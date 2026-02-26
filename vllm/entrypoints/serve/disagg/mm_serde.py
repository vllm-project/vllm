# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encode/decode utilities for multimodal tensors and field metadata
over JSON/HTTP, used by the disaggregated generate endpoint."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import pybase64
import torch

from vllm.entrypoints.serve.disagg.protocol import (
    FieldElemData,
    FieldInfo,
    MultiModalKwargsItemData,
    TensorData,
)
from vllm.multimodal.inputs import (
    BaseMultiModalField,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    NestedTensors,
)
from vllm.utils.serial_utils import EmbedDType, binary2tensor, tensor2binary
from vllm.v1.serial_utils import MMF_CLASS_TO_FACTORY

# Reverse mapping: factory name -> field class
_FACTORY_TO_MMF_CLASS = {v: k for k, v in MMF_CLASS_TO_FACTORY.items()}


# --- Tensor encode/decode ---


def encode_tensor(t: torch.Tensor) -> TensorData:
    """Encode a torch.Tensor to a JSON-serializable TensorData."""
    dtype_str = cast(EmbedDType, str(t.dtype).removeprefix("torch."))
    raw = tensor2binary(t, embed_dtype=dtype_str, endianness="native")
    return TensorData(
        dtype=dtype_str,
        shape=list(t.shape),
        data_b64=pybase64.b64encode(raw).decode("ascii"),
    )


def decode_tensor(td: TensorData) -> torch.Tensor:
    """Decode a TensorData back to a torch.Tensor."""
    raw = pybase64.b64decode(td.data_b64)
    return binary2tensor(
        raw,
        shape=tuple(td.shape),
        embed_dtype=cast(EmbedDType, td.dtype),
        endianness="native",
    )


# --- Nested tensors encode/decode ---


def encode_nested_tensors(
    nt: NestedTensors,
) -> TensorData | list[Any] | int | float:
    """Encode NestedTensors to JSON-serializable form."""
    if isinstance(nt, torch.Tensor):
        return encode_tensor(nt)
    if isinstance(nt, (int, float)):
        return nt
    return [encode_nested_tensors(x) for x in nt]


def decode_nested_tensors(
    obj: TensorData | list[Any] | int | float,
) -> NestedTensors:
    """Decode JSON-serializable form back to NestedTensors."""
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, TensorData):
        return decode_tensor(obj)
    if isinstance(obj, dict):
        # Pydantic may deserialize TensorData as a dict in nested unions
        return decode_tensor(TensorData(**obj))
    if isinstance(obj, list):
        return [decode_nested_tensors(x) for x in obj]
    raise TypeError(f"Unexpected nested tensor type: {type(obj)}")


# --- Slice encode/decode (mirrors v1/serial_utils.py:165-169, 416-420) ---


def encode_slice(s: slice) -> list[int | None]:
    return [s.start, s.stop, s.step]


def encode_nested_slices(
    slices: Any,
) -> list[list[int | None]] | list[list[list[int | None]]]:
    """Encode slice or sequence of slices to JSON-serializable form."""
    if not slices:
        return []
    first = slices[0]
    if isinstance(first, slice):
        # Simple slices: list[slice]
        return [encode_slice(s) for s in slices]
    # Nested slices: list[tuple[slice, ...]]
    return [[encode_slice(s) for s in group] for group in slices]


def decode_nested_slices(obj: Any) -> Any:
    """Decode JSON-serializable slices back to slice objects."""
    if not obj:
        return []
    first = obj[0]
    if not isinstance(first, (list, tuple)):
        # Single slice encoded as [start, stop, step]
        return slice(*obj)
    if first and isinstance(first[0], (list, tuple)):
        # Nested: list[list[list[int|None]]] -> list[tuple[slice, ...]]
        return [tuple(slice(*s) for s in group) for group in obj]
    # Simple: list[list[int|None]] -> list[slice]
    return [slice(*s) for s in obj]


# --- Field encode/decode ---


def encode_field(field: BaseMultiModalField) -> FieldInfo:
    """Encode a BaseMultiModalField to JSON-serializable FieldInfo."""
    name = MMF_CLASS_TO_FACTORY.get(field.__class__)
    if not name:
        raise TypeError(f"Unsupported field type: {field.__class__}")

    kwargs: dict[str, Any] = {
        f.name: getattr(field, f.name) for f in dataclasses.fields(field)
    }

    # Convert slices to JSON-serializable lists
    if "slices" in kwargs and kwargs["slices"] is not None:
        kwargs["slices"] = encode_nested_slices(kwargs["slices"])

    return FieldInfo(type=name, **kwargs)


def decode_field(info: FieldInfo) -> BaseMultiModalField:
    """Decode a FieldInfo back to a BaseMultiModalField."""
    factory = getattr(MultiModalFieldConfig, info.type)
    kwargs: dict[str, Any] = {}
    if info.keep_on_cpu:
        kwargs["keep_on_cpu"] = True
    if info.type == "flat":
        kwargs["slices"] = decode_nested_slices(info.slices)
        if info.dim is not None:
            kwargs["dim"] = info.dim
    elif info.type == "shared":
        kwargs["batch_size"] = info.batch_size
    return factory("", **kwargs).field


# --- FieldElem encode/decode ---


def encode_field_elem(elem: MultiModalFieldElem) -> FieldElemData:
    """Encode a MultiModalFieldElem to JSON-serializable FieldElemData."""
    data = None if elem.data is None else encode_nested_tensors(elem.data)
    return FieldElemData(data=data, field=encode_field(elem.field))


def decode_field_elem(elem_data: FieldElemData) -> MultiModalFieldElem:
    """Decode a FieldElemData back to a MultiModalFieldElem."""
    data = None if elem_data.data is None else decode_nested_tensors(elem_data.data)
    field = decode_field(elem_data.field)
    return MultiModalFieldElem(data=data, field=field)


# --- MultiModalKwargsItem encode/decode ---


def encode_mm_kwargs_item(
    item: MultiModalKwargsItem,
) -> MultiModalKwargsItemData:
    """Encode a MultiModalKwargsItem to JSON-serializable form."""
    return MultiModalKwargsItemData(
        items={key: encode_field_elem(elem) for key, elem in item.items()}
    )


def decode_mm_kwargs_item(
    data: MultiModalKwargsItemData,
) -> MultiModalKwargsItem:
    """Decode a MultiModalKwargsItemData back to a MultiModalKwargsItem."""
    return MultiModalKwargsItem(
        {key: decode_field_elem(elem) for key, elem in data.items.items()}
    )
