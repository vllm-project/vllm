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
    small_f_contig_tensor: torch.Tensor
    large_f_contig_tensor: torch.Tensor
    small_non_contig_tensor: torch.Tensor
    large_non_contig_tensor: torch.Tensor


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
        small_f_contig_tensor=torch.rand(5, 4).t(),
        large_f_contig_tensor=torch.rand(1024, 4).t(),
        small_non_contig_tensor=torch.rand(2, 4)[:, 1:3],
        large_non_contig_tensor=torch.rand(1024, 512)[:, 10:20],
    )

    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(MyType)

    encoded = encoder.encode(obj)

    # There should be the main buffer + 4 large tensor buffers
    # + 1 large numpy array. "large" is <= 512 bytes.
    # The two small tensors are encoded inline.
    assert len(encoded) == 6

    decoded: MyType = decoder.decode(encoded)

    assert_equal(decoded, obj)

    # Test encode_into case

    preallocated = bytearray()

    encoded2 = encoder.encode_into(obj, preallocated)

    assert len(encoded2) == 6
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
    assert torch.equal(obj1.small_f_contig_tensor, obj2.small_f_contig_tensor)
    assert torch.equal(obj1.large_f_contig_tensor, obj2.large_f_contig_tensor)
    assert torch.equal(obj1.small_non_contig_tensor,
                       obj2.small_non_contig_tensor)
    assert torch.equal(obj1.large_non_contig_tensor,
                       obj2.large_non_contig_tensor)
