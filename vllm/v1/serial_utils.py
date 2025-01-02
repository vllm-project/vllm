import torch
import pickle
from typing import Any
from msgspec import msgpack

CUSTOM_TYPE_CODE_PICKLE = 1
PICKLE_TYPES = torch.Tensor


class PickleEncoder:

    def encode(self, obj):
        return pickle.dumps(obj)

    def decode(self, data):
        return pickle.loads(data)
    

class MsgpackEncoder:
    """Encoder with custom torch tensor serialization."""

    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=custom_enc_hook)
    
    def encode(self, obj: Any) -> bytes:
        return self.encoder.encode(obj)
    
    def encode_into(self, obj: Any, buf: bytearray) -> None:
        self.encoder.encode_into(obj, buf)


class MsgpackDecoder:
    """Decoder with custom torch tensor serialization."""

    def __init__(self, t: Any):
        self.decoder = msgpack.Decoder(t, ext_hook=custom_ext_hook)
    
    def decode(self, obj: Any):
        return self.decoder.decode(obj)


def custom_enc_hook(obj: Any) -> Any:
    if isinstance(obj, PICKLE_TYPES):
        # NOTE(rob): it is fastest to use numpy + pickle
        # when serializing torch tensors.
        # https://gist.github.com/tlrmchlsmth/8067f1b24a82b6e2f90450e7764fa103 # noqa: E501
        return msgpack.Ext(CUSTOM_TYPE_CODE_PICKLE,
                           pickle.dumps(obj.numpy()))
    else:
        raise NotImplementedError(
            f"Objects of type {type(obj)} are not supported")


def custom_ext_hook(code: int, data: memoryview) -> Any:
    if code == CUSTOM_TYPE_CODE_PICKLE:
        return torch.from_numpy(pickle.loads(data))
    else:
        raise NotImplementedError(
            f"Extension type code {code} is not supported")
