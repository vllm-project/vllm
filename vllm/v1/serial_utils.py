import pickle
from typing import Any

import numpy as np
from msgspec import msgpack

CUSTOM_TYPE_CODE_PICKLE = 1
pickle_types = (np.ndarray, )


class PickleEncoder:

    def encode(self, obj):
        return pickle.dumps(obj)

    def decode(self, data):
        return pickle.loads(data)


def custom_enc_hook(obj: Any) -> Any:
    if isinstance(obj, pickle_types):
        # Return an `Ext` object so msgspec serializes it as an extension type.
        return msgpack.Ext(CUSTOM_TYPE_CODE_PICKLE, pickle.dumps(obj))
    else:
        # Raise a NotImplementedError for other types
        raise NotImplementedError(
            f"Objects of type {type(obj)} are not supported")


def custom_ext_hook(code: int, data: memoryview) -> Any:
    if code == CUSTOM_TYPE_CODE_PICKLE:
        # This extension type represents a complex number, decode the data
        # buffer accordingly.
        return pickle.loads(data)
    else:
        # Raise a NotImplementedError for other extension type codes
        raise NotImplementedError(
            f"Extension type code {code} is not supported")
