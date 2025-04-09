# SPDX-License-Identifier: Apache-2.0
from collections import UserDict
from dataclasses import dataclass
from typing import Optional

import msgspec
import numpy as np
import torch

from vllm.multimodal.inputs import MultiModalKwargs
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


class UnrecognizedType(UserDict):

    def __init__(self, an_int: int):
        super().__init__()
        self.an_int = an_int


@dataclass
class MyType:
    tensor1: torch.Tensor
    a_string: str
    list_of_tensors: list[torch.Tensor]
    numpy_array: np.ndarray
    unrecognized: UnrecognizedType


def test_encode_decode():
    """Test encode/decode loop with zero-copy tensors."""

    obj = MyType(
        tensor1=torch.randint(low=0, high=100, size=(10, ), dtype=torch.int32),
        a_string="hello",
        list_of_tensors=[
            torch.rand((1, 10), dtype=torch.float32),
            torch.rand((3, 5, 4), dtype=torch.float64)
        ],
        numpy_array=np.arange(20),
        unrecognized=UnrecognizedType(33),
    )

    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(MyType)

    encoded = encoder.encode(obj)

    # There should be the main buffer + 3 tensor buffers + one ndarray buffer
    assert len(encoded) == 5

    decoded: MyType = decoder.decode(encoded)

    assert_equal(decoded, obj)

    # Test encode_into case

    preallocated = bytearray()

    encoded2 = encoder.encode_into(obj, preallocated)

    assert len(encoded2) == 5
    assert encoded2[0] is preallocated

    decoded2: MyType = decoder.decode(encoded2)

    assert_equal(decoded2, obj)


class MyRequest(msgspec.Struct):
    mm: Optional[list[MultiModalKwargs]]

def test_multimodal_kwargs():
    d = { 
          "foo": torch.zeros(1000, dtype=torch.float16),
          "bar": [ torch.zeros(i * 1000, dtype=torch.int8) for i in range(3) ],
          "baz": ( torch.zeros(256, dtype=torch.int64), "i'm a tuple" )
         }

    # pack mm kwargs into a mock request so that it can be decoded properly
    req = MyRequest(mm = [ MultiModalKwargs(d) ] )

    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(MyRequest)

    encoded = encoder.encode(req)

    # 5 total tensors + top level buffer
    assert len(encoded) == 6

    total_len = sum([len(x) for x in encoded])
    
    # expected total encoding length, should be 4384, +-20 for minor changes
    assert total_len >= 4364 and total_len <= 4404

    decoded: MultiModalKwargs = decoder.decode(encoded).mm[0]
    assert torch.equal(d["foo"], decoded["foo"])


def assert_equal(obj1: MyType, obj2: MyType):
    assert torch.equal(obj1.tensor1, obj2.tensor1)
    assert obj1.a_string == obj2.a_string
    assert all(
        torch.equal(a, b)
        for a, b in zip(obj1.list_of_tensors, obj2.list_of_tensors))
    assert np.array_equal(obj1.numpy_array, obj2.numpy_array)
    assert obj1.unrecognized.an_int == obj2.unrecognized.an_int
