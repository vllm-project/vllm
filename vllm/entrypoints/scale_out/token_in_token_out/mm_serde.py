# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encode/decode utilities for multimodal tensors and field metadata
over JSON/HTTP, used by the disaggregated generate endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pybase64
from transformers import BatchFeature

from vllm.multimodal.inputs import (
    MultiModalFlatField,
    MultiModalKwargsItem,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

if TYPE_CHECKING:
    from vllm.multimodal.processing import BaseMultiModalProcessor

_encoder = MsgpackEncoder(size_threshold=2**62)  # force all tensors inline
_decoder = MsgpackDecoder(t=MultiModalKwargsItem)


def encode_mm_kwargs_item(item: MultiModalKwargsItem) -> str:
    """Serialize a MultiModalKwargsItem to a base64 string."""
    bufs = _encoder.encode(item)
    assert len(bufs) == 1, "All tensors should be inline"
    return pybase64.b64encode(bufs[0]).decode("ascii")


def _rebind_mm_fields(
    item: MultiModalKwargsItem,
    *,
    modality: str,
    mm_processor: BaseMultiModalProcessor,
) -> MultiModalKwargsItem:
    batched_data = {}
    for key, elem in item.items():
        if elem.data is None:
            raise ValueError(f"{key} field data must not be None")
        batched_data[key] = elem.field.reduce_data([elem], pin_memory=False)

    try:
        # The public generate route cannot trust caller-selected processor
        # kwargs. Current field schemas are context-independent.
        configs = mm_processor._get_mm_fields_config(BatchFeature(batched_data), {})
    except Exception as exc:
        raise ValueError("kwargs_data does not match the model field schema") from exc

    for key, elem in item.items():
        config = configs.get(key)
        if config is None or config.modality != modality:
            raise ValueError(f"{key} has no field processor for {modality}")

        actual = elem.field
        trusted = config.field
        layouts_match = (
            type(actual) is type(trusted)
            and actual.keep_on_cpu == trusted.keep_on_cpu
            and (
                not isinstance(actual, MultiModalFlatField)
                or (
                    isinstance(trusted, MultiModalFlatField)
                    and actual.dim == trusted.dim
                )
            )
        )
        if not layouts_match:
            raise ValueError(f"{key} field processor does not match the model schema")
        elem.field = trusted

    return item


def decode_mm_kwargs_item(
    data: str,
    *,
    modality: str | None = None,
    mm_processor: BaseMultiModalProcessor | None = None,
) -> MultiModalKwargsItem:
    """Deserialize a base64 string back to a MultiModalKwargsItem."""
    raw = pybase64.b64decode(data)
    item = _decoder.decode(raw)
    if mm_processor is None:
        return item
    if modality is None:
        raise ValueError("modality is required when rebinding field processors")
    return _rebind_mm_fields(
        item,
        modality=modality,
        mm_processor=mm_processor,
    )
