from array import array
from typing import Any, Type
from vllm.sequence import SequenceData


def encode_hook(obj: Any) -> Any:
    """Custom msgspec enc hook that supports array types.

    See https://jcristharif.com/msgspec/api.html#msgspec.msgpack.Encoder
    """
    if isinstance(obj, array):
        return obj.tobytes()
    if isinstance(obj, SequenceData):
        # This can be reconstructed from __post_init__.
        obj._prompt_token_ids_tuple = tuple()
        obj._cached_all_token_ids = []
        obj._new_appended_tokens = []
    else:
        raise ValueError(f"Unsupported serialization type: {type(obj)}")


def decode_hook(type: Type, obj: Any) -> Any:
    """Custom msgspec dec hook that supports array types.

    See https://jcristharif.com/msgspec/api.html#msgspec.msgpack.Encoder
    """
    if type is array:
        deserialized = array('I')
        deserialized.frombytes(obj)
        return deserialized
    if isinstance(obj, SequenceData):
        obj.__post_init__()
        return obj
    else:
        raise ValueError(f"Unsupported deserialization type: {type(obj)}")
