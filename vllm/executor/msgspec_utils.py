from array import array
from typing import Any, Type

from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE


def encode_hook(obj: Any) -> Any:
    """Custom msgspec enc hook that supports array types.

    See https://jcristharif.com/msgspec/api.html#msgspec.msgpack.Encoder
    """
    if isinstance(obj, array):
        assert obj.typecode == "l"
        return obj.tobytes()
    else:
        raise ValueError(f"Unsupported serialization type: {type(obj)}")


def decode_hook(type: Type, obj: Any) -> Any:
    """Custom msgspec dec hook that supports array types.

    See https://jcristharif.com/msgspec/api.html#msgspec.msgpack.Encoder
    """
    if type is array:
        deserialized = array(VLLM_TOKEN_ID_ARRAY_TYPE)
        deserialized.frombytes(obj)
        return deserialized
    else:
        raise ValueError(f"Unsupported deserialization type: {type(obj)}")
