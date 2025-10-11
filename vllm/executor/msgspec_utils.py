# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from array import array
from typing import Any

from vllm.multimodal.inputs import MultiModalKwargs
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE


def encode_hook(obj: Any) -> Any:
    """Custom msgspec enc hook that supports array types and MultiModalKwargs.

    See https://jcristharif.com/msgspec/api.html#msgspec.msgpack.Encoder
    """
    if isinstance(obj, array):
        assert obj.typecode == VLLM_TOKEN_ID_ARRAY_TYPE, (
            f"vLLM array type should use '{VLLM_TOKEN_ID_ARRAY_TYPE}' type. "
            f"Given array has a type code of {obj.typecode}."
        )
        return obj.tobytes()
    if isinstance(obj, MultiModalKwargs):
        return dict(obj)


def decode_hook(type: type, obj: Any) -> Any:
    """Custom msgspec dec hook that supports array types and MultiModalKwargs.

    See https://jcristharif.com/msgspec/api.html#msgspec.msgpack.Encoder
    """
    if type is array:
        deserialized = array(VLLM_TOKEN_ID_ARRAY_TYPE)
        deserialized.frombytes(obj)
        return deserialized
    if type is MultiModalKwargs:
        return MultiModalKwargs(obj)
