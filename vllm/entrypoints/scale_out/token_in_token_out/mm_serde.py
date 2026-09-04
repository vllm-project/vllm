# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encode/decode utilities for multimodal tensors and field metadata
over JSON/HTTP, used by the disaggregated generate endpoint."""

from __future__ import annotations

import pybase64

from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

_encoder = MsgpackEncoder(size_threshold=2**62)  # force all tensors inline
_decoder = MsgpackDecoder(t=MultiModalKwargsItem)


def encode_mm_kwargs_item(item: MultiModalKwargsItem) -> str:
    """Serialize a MultiModalKwargsItem to a base64 string."""
    bufs = _encoder.encode(item)
    assert len(bufs) == 1, "All tensors should be inline"
    return pybase64.b64encode(bufs[0]).decode("ascii")


def decode_mm_kwargs_item(data: str) -> MultiModalKwargsItem:
    """Deserialize a base64 string back to a MultiModalKwargsItem."""
    raw = pybase64.b64decode(data)
    return _decoder.decode(raw)
