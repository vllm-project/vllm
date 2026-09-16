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
        torch.zeros(1000, dtype=torch.bfloat16),
        MultiModalBatchedField(),
    )
    e2 = MultiModalFieldElem(
        [torch.zeros(1000, dtype=torch.int8) for _ in range(4)],
        MultiModalFlatField(
            slices=[[slice(1, 2, 3), slice(4, 5, 6)], [slice(None, 2)]],
            dim=0,
        ),
    )
    e3 = MultiModalFieldElem(
        torch.zeros(1000, dtype=torch.int32),
        MultiModalSharedField(batch_size=4),
    )
    e4 = MultiModalFieldElem(
        torch.zeros(1000, dtype=torch.int32),
        MultiModalFlatField(slices=[slice(1, 2, 3), slice(4, 5, 6)], dim=2),
    )
    mm = MultiModalKwargsItems(
        {
            "audio": [MultiModalKwargsItem({"a0": e1})],
            "video": [MultiModalKwargsItem({"v0": e2})],
            "image": [MultiModalKwargsItem({"i0": e3, "i1": e4})],
        }
    )

    # pack mm kwargs into a mock request so that it can be decoded properly
    req = MyRequest([mm])

    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(MyRequest)

    encoded = encoder.encode(req)

    assert len(encoded) == 8

    total_len = sum(memoryview(x).cast("B").nbytes for x in encoded)

    # expected total encoding length, should be 14319, +-20 for minor changes
    assert 14300 <= total_len <= 14340
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
    would fail to decode when IPC is enabled because _decode_tensor expected a
    raw tensor tuple but received a msgpack-decoded TensorIpcHandle list.
    """
    import torch.multiprocessing as torch_mp

    from vllm.v1.engine.tensor_ipc import TensorIpcReceiver, TensorIpcSender

    # Create tensor queues for IPC
    tensor_queues = [torch_mp.Queue()]

    # Create encoder with IPC sender
    sender = TensorIpcSender(tensor_queues[0])
    encoder = MsgpackEncoder(oob_tensor_consumer=sender)

    # Create decoder with IPC receiver
    receiver = TensorIpcReceiver(tensor_queues[0])
    decoder = MsgpackDecoder(RequestWithTensor, oob_tensor_provider=receiver)

    # Create a request with a non-multimodal tensor
    original_tensor = torch.randn(5, 10, dtype=torch.float32)
    request = RequestWithTensor(prompt_embeds=original_tensor, data="test_data")

    # Encode the request - this should send the tensor via IPC
    encoded = encoder.encode(request)

    # Verify encoding succeeded
    assert len(encoded) > 0

    # Decode the request - this should retrieve the tensor from IPC queue
    # Previously this would fail because the decoder tried to unpack the
    # handle list as raw tensor bytes metadata.
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

    from vllm.v1.engine.tensor_ipc import TensorIpcReceiver, TensorIpcSender

    # Create tensor queues for IPC
    tensor_queues = [torch_mp.Queue()]

    # Create encoder with IPC sender
    sender = TensorIpcSender(tensor_queues[0])
    encoder = MsgpackEncoder(oob_tensor_consumer=sender)

    # Create decoder with IPC receiver
    receiver = TensorIpcReceiver(tensor_queues[0])
    decoder = MsgpackDecoder(RequestWithTensor, oob_tensor_provider=receiver)

    # Create a request with None for the tensor field
    request = RequestWithTensor(prompt_embeds=None, data="test_data_with_none")

    # Encode and decode the request
    encoded = encoder.encode(request)
    decoded = decoder.decode(encoded)

    # Verify the decoded request matches the original
    assert isinstance(decoded, RequestWithTensor)
    assert decoded.data == "test_data_with_none"
    assert decoded.prompt_embeds is None


def test_multiple_senders_single_receiver_ipc():
    """Test N senders sharing a queue with a single receiver via msgpack.

    Simulates the real vLLM topology where multiple API server frontends
    each have their own MsgpackEncoder + TensorIpcSender, all putting
    tensors onto the same torch.mp queue, and a single engine core
    decodes them with one MsgpackDecoder + TensorIpcReceiver.
    """
    import torch.multiprocessing as torch_mp

    from vllm.v1.engine.tensor_ipc import TensorIpcReceiver, TensorIpcSender

    num_senders = 3
    num_messages_per_sender = 2
    tensor_queue = torch_mp.Queue()

    # Create N independent senders (each gets its own uuid-based sender_id)
    senders = []
    encoders = []
    for _ in range(num_senders):
        s = TensorIpcSender(tensor_queue)
        senders.append(s)
        encoders.append(MsgpackEncoder(oob_tensor_consumer=s))

    # Single receiver
    receiver = TensorIpcReceiver(tensor_queue)
    decoder = MsgpackDecoder(RequestWithTensor, oob_tensor_provider=receiver)

    # Encode messages from all senders, interleaving the order
    # so that tensors from different senders land on the queue interleaved.
    encoded_payloads: list[tuple[int, int, torch.Tensor, list]] = []
    for msg_idx in range(num_messages_per_sender):
        for sender_idx in range(num_senders):
            tensor = torch.full(
                (sender_idx + 1, msg_idx + 2),
                float(sender_idx * 100 + msg_idx),
                dtype=torch.float32,
            )
            req = RequestWithTensor(
                prompt_embeds=tensor,
                data=f"s{sender_idx}_m{msg_idx}",
            )
            encoded = encoders[sender_idx].encode(req)
            encoded_payloads.append((sender_idx, msg_idx, tensor, encoded))

    # Decode all messages — the receiver must correctly match each
    # tensor handle to the right TensorIpcData from the shared queue.
    for sender_idx, msg_idx, original_tensor, encoded in encoded_payloads:
        decoded = decoder.decode(encoded)
        assert isinstance(decoded, RequestWithTensor)
        assert decoded.data == f"s{sender_idx}_m{msg_idx}"
        assert decoded.prompt_embeds is not None
        assert decoded.prompt_embeds.shape == original_tensor.shape, (
            f"Shape mismatch for sender {sender_idx} msg {msg_idx}: "
            f"{decoded.prompt_embeds.shape} != {original_tensor.shape}"
        )
        assert torch.allclose(decoded.prompt_embeds, original_tensor), (
            f"Value mismatch for sender {sender_idx} msg {msg_idx}"
        )


def _logprobs_outputs(num_reqs: int, num_prompt_tokens: int):
    """An EngineCoreOutputs carrying prompt logprobs, as the engine core sends
    it: many requests, each with per-token tensors small enough that pyzmq
    copies their frames, while the accumulated payload frame is large enough
    that pyzmq sends it zero-copy.
    """
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
    from vllm.v1.outputs import LogprobsTensors

    outputs = []
    for req in range(num_reqs):
        num_tokens = num_prompt_tokens + req % 4
        outputs.append(
            EngineCoreOutput(
                request_id=f"req-{req:08d}",
                new_token_ids=[req],
                new_prompt_logprobs_tensors=LogprobsTensors(
                    logprob_token_ids=torch.arange(
                        num_tokens * 2, dtype=torch.int64
                    ).view(num_tokens, 2),
                    logprobs=torch.zeros(num_tokens, 2, dtype=torch.float32),
                    selected_token_ranks=torch.zeros(num_tokens, dtype=torch.int32),
                ),
            )
        )
    return EngineCoreOutputs(outputs=outputs)


def test_payload_buffer_reuse_does_not_corrupt_in_flight_messages():
    """The engine core recycles the msgpack payload buffer across messages
    (`MsgpackEncoder.encode_into`). It may only do so once zmq has finished
    sending that buffer, otherwise a newer payload is delivered alongside the
    older message's zero-copy tensor frames.

    `Socket.send_multipart(track=True)` cannot be used to detect this: it
    returns a tracker for the last frame only, and pyzmq copies frames below
    `zmq.COPY_THRESHOLD` and reports them as already-sent.
    """
    import zmq

    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.engine.core import EngineCoreProc

    num_msgs = 100
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(EngineCoreOutputs)
    # Enough requests that the payload frame is zero-copied rather than copied
    # by pyzmq, which is what makes early reuse observable.
    messages = [_logprobs_outputs(300, 24 + i % 8) for i in range(num_msgs)]
    assert len(encoder.encode(messages[0])[0]) >= zmq.COPY_THRESHOLD

    reuse_buffers: list[bytearray] = []
    pending: list[tuple[zmq.MessageTracker, bytearray]] = []
    with zmq.Context() as ctx:
        push = ctx.socket(zmq.PUSH)
        push.bind("inproc://test-payload-reuse")
        pull = ctx.socket(zmq.PULL)
        pull.connect("inproc://test-payload-reuse")

        for outputs in messages:
            while pending and pending[0][0].done:
                reuse_buffers.append(pending.pop(0)[1])
            buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
            buffers = encoder.encode_into(outputs, buffer)
            tracker = EngineCoreProc._send_msg_tracking_payload(push, buffers)
            if tracker.done:
                reuse_buffers.append(buffer)
            else:
                pending.append((tracker, buffer))

        for i, sent in enumerate(messages):
            received = decoder.decode(pull.recv_multipart(copy=False))
            assert len(received.outputs) == len(sent.outputs), f"message {i}"
            for expected, actual in zip(sent.outputs, received.outputs):
                sent_ids = expected.new_prompt_logprobs_tensors.logprob_token_ids
                got_ids = actual.new_prompt_logprobs_tensors.logprob_token_ids
                assert actual.request_id == expected.request_id, f"message {i}"
                assert torch.equal(got_ids, sent_ids), (
                    f"message {i} request {actual.request_id}: corrupted "
                    f"prompt logprobs, {got_ids.shape} vs {sent_ids.shape}"
                )
        push.close(linger=0)
        pull.close(linger=0)


def test_zero_copy_frames_survive_without_caller_side_references():
    """Callers don't need to retain the encoded object until zmq has sent it:
    for a zero-copy frame, zmq holds its own reference to the backing buffer.

    The engine core clients rely on this when sending requests that carry
    tensors (e.g. prompt embeds) without tracking the messages.

    What makes that safe is that `tensor_data()` hands zmq a memoryview which
    transitively references the source tensor, so refcounting - not timing -
    keeps the memory from being freed and reused underneath zmq.
    """
    import gc

    import zmq

    from vllm.v1.utils import tensor_data

    num_elems = 100_000  # comfortably over zmq.COPY_THRESHOLD
    expected = torch.arange(num_elems, dtype=torch.int64)
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(RequestWithTensor)

    # The buffer handed to zmq must keep the tensor's storage alive by itself.
    holder = tensor_data(expected).obj
    while getattr(holder, "base", None) is not None:
        holder = holder.base
    assert isinstance(holder, torch.Tensor)
    assert holder.data_ptr() == expected.data_ptr()

    with zmq.Context() as ctx:
        push = ctx.socket(zmq.PUSH)
        push.bind("inproc://test-zero-copy-lifetime")
        pull = ctx.socket(zmq.PULL)
        pull.connect("inproc://test-zero-copy-lifetime")

        request = RequestWithTensor(prompt_embeds=expected.clone(), data="req")
        buffers = encoder.encode(request)
        assert max(len(buf) for buf in buffers) >= zmq.COPY_THRESHOLD
        push.send_multipart(buffers, copy=False)

        # Drop every reference the sender holds, then churn the allocator.
        del request, buffers
        gc.collect()
        torch.arange(num_elems * 4, dtype=torch.int64)

        decoded = decoder.decode(pull.recv_multipart(copy=False))
        assert decoded.prompt_embeds is not None
        assert torch.equal(decoded.prompt_embeds, expected)
        push.close(linger=0)
        pull.close(linger=0)
