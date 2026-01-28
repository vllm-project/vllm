# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import UserDict
from dataclasses import dataclass

import msgspec
import numpy as np
import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    NestedTensors,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.cpu_test


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
    empty_tensor: torch.Tensor


def test_encode_decode(monkeypatch: pytest.MonkeyPatch):
    """Test encode/decode loop with zero-copy tensors."""

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        obj = MyType(
            tensor1=torch.randint(low=0, high=100, size=(1024,), dtype=torch.int32),
            a_string="hello",
            list_of_tensors=[
                torch.rand((1, 10), dtype=torch.float32),
                torch.rand((3, 5, 4000), dtype=torch.float64),
                torch.tensor(1984),  # test scalar too
                # Make sure to test bf16 which numpy doesn't support.
                torch.rand((3, 5, 1000), dtype=torch.bfloat16),
                torch.tensor(
                    [float("-inf"), float("inf")] * 1024, dtype=torch.bfloat16
                ),
            ],
            numpy_array=np.arange(512),
            unrecognized=UnrecognizedType(33),
            small_f_contig_tensor=torch.rand(5, 4).t(),
            large_f_contig_tensor=torch.rand(1024, 4).t(),
            small_non_contig_tensor=torch.rand(2, 4)[:, 1:3],
            large_non_contig_tensor=torch.rand(1024, 512)[:, 10:20],
            empty_tensor=torch.empty(0),
        )

        encoder = MsgpackEncoder(size_threshold=256)
        decoder = MsgpackDecoder(MyType)

        encoded = encoder.encode(obj)

        # There should be the main buffer + 4 large tensor buffers
        # + 1 large numpy array. "large" is <= 512 bytes.
        # The two small tensors are encoded inline.
        assert len(encoded) == 8

        decoded: MyType = decoder.decode(encoded)

        assert_equal(decoded, obj)

        # Test encode_into case

        preallocated = bytearray()

        encoded2 = encoder.encode_into(obj, preallocated)

        assert len(encoded2) == 8
        assert encoded2[0] is preallocated

        decoded2: MyType = decoder.decode(encoded2)

        assert_equal(decoded2, obj)


class MyRequest(msgspec.Struct):
    mm: list[MultiModalKwargsItems] | None


def test_multimodal_kwargs():
    e1 = MultiModalFieldElem(
        "audio",
        "a0",
        torch.zeros(1000, dtype=torch.bfloat16),
        MultiModalBatchedField(),
    )
    e2 = MultiModalFieldElem(
        "video",
        "v0",
        [torch.zeros(1000, dtype=torch.int8) for _ in range(4)],
        MultiModalFlatField(
            slices=[[slice(1, 2, 3), slice(4, 5, 6)], [slice(None, 2)]],
            dim=0,
        ),
    )
    e3 = MultiModalFieldElem(
        "image",
        "i0",
        torch.zeros(1000, dtype=torch.int32),
        MultiModalSharedField(batch_size=4),
    )
    e4 = MultiModalFieldElem(
        "image",
        "i1",
        torch.zeros(1000, dtype=torch.int32),
        MultiModalFlatField(slices=[slice(1, 2, 3), slice(4, 5, 6)], dim=2),
    )
    audio = MultiModalKwargsItem.from_elems([e1])
    video = MultiModalKwargsItem.from_elems([e2])
    image = MultiModalKwargsItem.from_elems([e3, e4])
    mm = MultiModalKwargsItems.from_seq([audio, video, image])

    # pack mm kwargs into a mock request so that it can be decoded properly
    req = MyRequest([mm])

    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(MyRequest)

    encoded = encoder.encode(req)

    assert len(encoded) == 8

    total_len = sum(memoryview(x).cast("B").nbytes for x in encoded)

    # expected total encoding length, should be 14395, +-20 for minor changes
    assert 14375 <= total_len <= 14425
    decoded = decoder.decode(encoded).mm[0]
    assert isinstance(decoded, MultiModalKwargsItems)

    # check all modalities were recovered and do some basic sanity checks
    assert len(decoded) == 3
    images = decoded["image"]
    assert len(images) == 1
    assert len(images[0].items()) == 2
    assert list(images[0].keys()) == ["i0", "i1"]

    # check the tensor contents and layout in the main dict
    mm_data = mm.get_data()
    decoded_data = decoded.get_data()
    assert all(nested_equal(mm_data[k], decoded_data[k]) for k in mm_data)


def nested_equal(a: NestedTensors, b: NestedTensors):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    return all(nested_equal(x, y) for x, y in zip(a, b))


def assert_equal(obj1: MyType, obj2: MyType):
    assert torch.equal(obj1.tensor1, obj2.tensor1)
    assert obj1.a_string == obj2.a_string
    assert all(
        torch.equal(a, b) for a, b in zip(obj1.list_of_tensors, obj2.list_of_tensors)
    )
    assert np.array_equal(obj1.numpy_array, obj2.numpy_array)
    assert obj1.unrecognized.an_int == obj2.unrecognized.an_int
    assert torch.equal(obj1.small_f_contig_tensor, obj2.small_f_contig_tensor)
    assert torch.equal(obj1.large_f_contig_tensor, obj2.large_f_contig_tensor)
    assert torch.equal(obj1.small_non_contig_tensor, obj2.small_non_contig_tensor)
    assert torch.equal(obj1.large_non_contig_tensor, obj2.large_non_contig_tensor)
    assert torch.equal(obj1.empty_tensor, obj2.empty_tensor)


def test_dict_serialization():
    """Test encoding and decoding of a generic Python object using pickle."""
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder()

    # Create a sample Python object
    obj = {"key": "value", "number": 42}

    # Encode the object
    encoded = encoder.encode(obj)

    # Decode the object
    decoded = decoder.decode(encoded)

    # Verify the decoded object matches the original
    assert obj == decoded, "Decoded object does not match the original object."


def test_tensor_serialization():
    """Test encoding and decoding of a torch.Tensor."""
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(torch.Tensor)

    # Create a sample tensor
    tensor = torch.rand(10, 10)

    # Encode the tensor
    encoded = encoder.encode(tensor)

    # Decode the tensor
    decoded = decoder.decode(encoded)

    # Verify the decoded tensor matches the original
    assert torch.allclose(tensor, decoded), (
        "Decoded tensor does not match the original tensor."
    )


def test_numpy_array_serialization():
    """Test encoding and decoding of a numpy array."""
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(np.ndarray)

    # Create a sample numpy array
    array = np.random.rand(10, 10)

    # Encode the numpy array
    encoded = encoder.encode(array)

    # Decode the numpy array
    decoded = decoder.decode(encoded)

    # Verify the decoded array matches the original
    assert np.allclose(array, decoded), (
        "Decoded numpy array does not match the original array."
    )


class CustomClass:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, CustomClass) and self.value == other.value


def test_custom_class_serialization_allowed_with_pickle(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that serializing a custom class succeeds when allow_pickle=True."""

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder(CustomClass)

        obj = CustomClass("test_value")

        # Encode the custom class
        encoded = encoder.encode(obj)

        # Decode the custom class
        decoded = decoder.decode(encoded)

        # Verify the decoded object matches the original
        assert obj == decoded, "Decoded object does not match the original object."


def test_custom_class_serialization_disallowed_without_pickle():
    """Test that serializing a custom class fails when allow_pickle=False."""
    encoder = MsgpackEncoder()

    obj = CustomClass("test_value")

    with pytest.raises(TypeError):
        # Attempt to encode the custom class
        encoder.encode(obj)


@dataclass
class RequestWithTensor:
    """Mock request with non-multimodal tensor field like EngineCoreRequest."""

    prompt_embeds: torch.Tensor | None
    data: str


def test_non_multimodal_tensor_with_ipc():
    """Test that non-multimodal tensor fields work correctly with IPC enabled.

    This reproduces the bug where fields like prompt_embeds: torch.Tensor | None
    would fail to decode when IPC is enabled because _decode_tensor expected a tuple
    but received a TensorIpcHandle dict.
    """
    import torch.multiprocessing as torch_mp

    # Create tensor queues for IPC
    tensor_queues = [torch_mp.Queue()]

    # Create encoder with IPC enabled
    encoder = MsgpackEncoder(tensor_queues=tensor_queues, multimodal_tensor_ipc="torch")
    encoder.set_target_engine(0)
    encoder.set_request_context("test_request_123")

    # Create decoder with IPC queue
    decoder = MsgpackDecoder(RequestWithTensor, tensor_queue=tensor_queues[0])

    # Create a request with a non-multimodal tensor
    original_tensor = torch.randn(5, 10, dtype=torch.float32)
    request = RequestWithTensor(prompt_embeds=original_tensor, data="test_data")

    # Encode the request - this should send the tensor via IPC
    encoded = encoder.encode(request)

    # Verify encoding succeeded
    assert len(encoded) > 0

    # Decode the request - this should retrieve the tensor from IPC queue
    # Previously this would fail with: TypeError: cannot unpack non-iterable dict object
    decoded = decoder.decode(encoded)

    # Verify the decoded request matches the original
    assert isinstance(decoded, RequestWithTensor)
    assert decoded.data == "test_data"
    assert decoded.prompt_embeds is not None
    assert torch.allclose(decoded.prompt_embeds, original_tensor), (
        "Decoded tensor does not match the original tensor."
    )


def test_non_multimodal_tensor_with_ipc_none_value():
    """Test that None values for tensor fields work correctly with IPC enabled."""
    import torch.multiprocessing as torch_mp

    # Create tensor queues for IPC
    tensor_queues = [torch_mp.Queue()]

    # Create encoder with IPC enabled
    encoder = MsgpackEncoder(tensor_queues=tensor_queues, multimodal_tensor_ipc="torch")
    encoder.set_target_engine(0)
    encoder.set_request_context("test_request_456")

    # Create decoder with IPC queue
    decoder = MsgpackDecoder(RequestWithTensor, tensor_queue=tensor_queues[0])

    # Create a request with None for the tensor field
    request = RequestWithTensor(prompt_embeds=None, data="test_data_with_none")

    # Encode and decode the request
    encoded = encoder.encode(request)
    decoded = decoder.decode(encoded)

    # Verify the decoded request matches the original
    assert isinstance(decoded, RequestWithTensor)
    assert decoded.data == "test_data_with_none"
    assert decoded.prompt_embeds is None
