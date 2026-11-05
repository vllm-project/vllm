# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import pybase64
import torch

from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

_encoder = MsgpackEncoder(size_threshold=2**62)  # force inline
_decoder = MsgpackDecoder(t=torch.Tensor)


def encode_tensor(tensor) -> str:
    """Serialize a raw tensor to a base64 string using vLLM msgpack logic."""
    bufs = _encoder.encode(tensor.clone())
    assert len(bufs) == 1, "Tensor was split into multiple buffers"
    # With the inline threshold above, the single buffer is always bytes.
    return pybase64.b64encode(cast(bytes, bufs[0])).decode("ascii")


def decode_tensor(encoded: str) -> torch.Tensor:
    raw = pybase64.b64decode(encoded.encode("ascii"))
    tensor = _decoder.decode(raw)
    return tensor.clone()
