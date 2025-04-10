# SPDX-License-Identifier: Apache-2.0
from collections import UserDict
from dataclasses import dataclass

import numpy as np
import torch

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
        tensor1=torch.randint(low=0,
                              high=100,
                              size=(1024, ),
                              dtype=torch.int32),
        a_string="hello",
        list_of_tensors=[
            torch.rand((1, 10), dtype=torch.float32),
            torch.rand((3, 5, 4000), dtype=torch.float64),
            torch.tensor(1984),  # test scalar too
        ],
        numpy_array=np.arange(512),
        unrecognized=UnrecognizedType(33),
    )

    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(MyType)

    encoded = encoder.encode(obj)

    # There should be the main buffer + 2 large tensor buffers
    # + 1 large numpy array. "large" is <= 256 bytes.
    # The two small tensors are encoded inline.
    assert len(encoded) == 4

    decoded: MyType = decoder.decode(encoded)

    assert_equal(decoded, obj)

    # Test encode_into case

    preallocated = bytearray()

    encoded2 = encoder.encode_into(obj, preallocated)

    assert len(encoded2) == 4
    assert encoded2[0] is preallocated

    decoded2: MyType = decoder.decode(encoded2)

    assert_equal(decoded2, obj)


def assert_equal(obj1: MyType, obj2: MyType):
    assert torch.equal(obj1.tensor1, obj2.tensor1)
    assert obj1.a_string == obj2.a_string
    assert all(
        torch.equal(a, b)
        for a, b in zip(obj1.list_of_tensors, obj2.list_of_tensors))
    assert np.array_equal(obj1.numpy_array, obj2.numpy_array)
    assert obj1.unrecognized.an_int == obj2.unrecognized.an_int
