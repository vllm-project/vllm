# SPDX-License-Identifier: Apache-2.0
from collections import UserDict
from dataclasses import dataclass
from typing import Optional

import msgspec
import numpy as np
import torch

from vllm.multimodal.inputs import (MultiModalBatchedField,
                                    MultiModalFieldElem, MultiModalKwargs,
                                    MultiModalKwargsItem, NestedTensors)
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


class MyRequest(msgspec.Struct):
    mm: Optional[list[MultiModalKwargs]]


def test_multimodal_kwargs():
    d = {
        "foo":
        torch.zeros(1000, dtype=torch.float16),
        "bar": [torch.zeros(i * 1000, dtype=torch.int8) for i in range(3)],
        "baz": [
            torch.rand((256), dtype=torch.float16),
            [
                torch.rand((1, 12), dtype=torch.float32),
                torch.rand((3, 5, 7), dtype=torch.float64),
            ], [torch.rand((4, 4), dtype=torch.float16)]
        ],
    }

    # pack mm kwargs into a mock request so that it can be decoded properly
    req = MyRequest(mm=[MultiModalKwargs(d)])

    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(MyRequest)

    encoded = encoder.encode(req)

    # 8 total tensors + top level buffer
    assert len(encoded) == 6

    total_len = sum([len(x) for x in encoded])

    # expected total encoding length, should be 4440, +-20 for minor changes
    assert total_len >= 4420 and total_len <= 4460
    decoded: MultiModalKwargs = decoder.decode(encoded).mm[0]
    assert all(nested_equal(d[k], decoded[k]) for k in d)


def test_multimodal_items_by_modality():
    e1 = MultiModalFieldElem("audio", "a0", torch.zeros(1000,
                                                        dtype=torch.int16),
                             MultiModalBatchedField())
    e2 = MultiModalFieldElem(
        "video",
        "v0",
        [torch.zeros(1000, dtype=torch.int8) for _ in range(4)],
        MultiModalBatchedField(),
    )
    e3 = MultiModalFieldElem("image", "i0", torch.zeros(1000,
                                                        dtype=torch.int32),
                             MultiModalBatchedField())
    e4 = MultiModalFieldElem("image", "i1", torch.zeros(1000,
                                                        dtype=torch.int32),
                             MultiModalBatchedField())
    audio = MultiModalKwargsItem.from_elems([e1])
    video = MultiModalKwargsItem.from_elems([e2])
    image = MultiModalKwargsItem.from_elems([e3, e4])
    mm = MultiModalKwargs.from_items([audio, video, image])

    # pack mm kwargs into a mock request so that it can be decoded properly
    req = MyRequest([mm])

    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(MyRequest)

    encoded = encoder.encode(req)

    # 5 total tensors + top level buffer
    assert len(encoded) == 8

    total_len = sum([len(x) for x in encoded])

    # expected total encoding length, should be 7507, +-20 for minor changes
    assert total_len >= 7487 and total_len <= 7527
    decoded: MultiModalKwargs = decoder.decode(encoded).mm[0]

    # check all modalities were recovered and do some basic sanity checks
    assert len(decoded.modalities) == 3
    images = decoded.get_items("image")
    assert len(images) == 1
    assert len(images[0].items()) == 2
    assert list(images[0].keys()) == ["i0", "i1"]

    # check the tensor contents and layout in the main dict
    assert all(nested_equal(mm[k], decoded[k]) for k in mm)


def nested_equal(a: NestedTensors, b: NestedTensors):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    else:
        return all([nested_equal(x, y) for (x, y) in zip(a, b)])


def assert_equal(obj1: MyType, obj2: MyType):
    assert torch.equal(obj1.tensor1, obj2.tensor1)
    assert obj1.a_string == obj2.a_string
    assert all(
        torch.equal(a, b)
        for a, b in zip(obj1.list_of_tensors, obj2.list_of_tensors))
    assert np.array_equal(obj1.numpy_array, obj2.numpy_array)
    assert obj1.unrecognized.an_int == obj2.unrecognized.an_int
