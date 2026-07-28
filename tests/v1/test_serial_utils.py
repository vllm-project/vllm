# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import time
from collections import UserDict
from dataclasses import dataclass

import msgspec
import numpy as np
import pytest
import torch
import zmq
import zmq.asyncio

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


def test_output_socket_drops_undecodable_frame():
    """A malformed frame on the engine output socket must not crash the engine.

    Regression test for issue #44486: an external port scanner injected junk
    bytes onto the engine->client output socket, msgspec raised a
    ``ValidationError`` ("Expected array, got int"), and that exception was
    forwarded to ``outputs_queue`` and re-raised, killing the engine.

    This drives the real ``AsyncMPClient._ensure_output_queue_task`` loop with a
    lightweight fake ``self`` and an in-process ZMQ socket pair. A garbage frame
    (an ``int`` where ``EngineCoreOutputs`` is expected, reproducing the exact
    error from the issue) is sent first, followed by a valid
    ``EngineCoreOutputs``. The loop must drop the bad frame and still deliver the
    valid one.
    """
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.engine.core_client import AsyncMPClient
    from vllm.v1.metrics.stats import SchedulerStats

    class _FakeResources:
        def __init__(self, output_socket):
            self.output_queue_task = None
            self.output_socket = output_socket

        def validate_alive(self, frames):
            # The real implementation only raises on the single-frame
            # ENGINE_CORE_DEAD sentinel; an arbitrary junk frame cannot
            # masquerade as it, so a no-op faithfully models that here.
            pass

    class _FakeClient:
        # Deliberately has no ``process_engine_outputs`` /
        # ``eep_process_engine_core_notification`` attributes, so the loop
        # takes the plain "enqueue outputs" path.
        def __init__(self, output_socket):
            self.resources = _FakeResources(output_socket)
            self.decoder = MsgpackDecoder(EngineCoreOutputs)
            self.utility_results: dict = {}
            self.outputs_queue: asyncio.Queue = asyncio.Queue()

    async def _run():
        ctx = zmq.asyncio.Context()
        addr = "inproc://test-44486-output"
        recv_socket = ctx.socket(zmq.PULL)
        recv_socket.bind(addr)
        send_socket = ctx.socket(zmq.PUSH)
        send_socket.connect(addr)

        fake = _FakeClient(recv_socket)

        encoder = MsgpackEncoder()

        try:
            # Start the real output-handling loop under test.
            AsyncMPClient._ensure_output_queue_task(fake)

            # 1) garbage frame (int where an array is expected), then
            # 2) a valid EngineCoreOutputs.
            await send_socket.send_multipart(list(encoder.encode(123)))
            valid = EngineCoreOutputs(engine_index=7, scheduler_stats=SchedulerStats())
            await send_socket.send_multipart(list(encoder.encode(valid)))

            return await asyncio.wait_for(fake.outputs_queue.get(), timeout=5.0)
        finally:
            task = fake.resources.output_queue_task
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            recv_socket.close(linger=0)
            send_socket.close(linger=0)
            ctx.term()

    received = asyncio.run(_run())

    # Without the fix, the queue's first item would be the forwarded
    # ValidationError instead of our valid output.
    assert not isinstance(received, Exception), received
    assert isinstance(received, EngineCoreOutputs)
    assert received.engine_index == 7


def test_dp_coordinator_drops_undecodable_frame():
    """A malformed frame on the DP coordinator's sockets must not crash it.

    Regression test for issue #44486: the DP coordinator decodes engine
    messages with ``decoder.decode(buffer)`` (and a raw ``msgspec.msgpack``
    decode on the front-end socket). ``run_coordinator`` only catches
    ``KeyboardInterrupt``, so a ``ValidationError`` from a garbage frame
    propagated uncaught and killed the coordinator process.

    This drives the real ``DPCoordinatorProc.process_input_socket`` loop. A fake
    engine subscribes so the coordinator reaches its steady-state poll loop;
    then a garbage frame (an ``int`` where ``EngineCoreOutputs`` is expected,
    reproducing "Expected array, got int") is pushed to the engine output
    socket, followed by a valid stats update. The coordinator must drop the bad
    frame and still publish the valid stats to the front-end.
    """
    import threading

    from vllm.utils.network_utils import get_open_zmq_ipc_path
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.engine.coordinator import DPCoordinatorProc
    from vllm.v1.metrics.stats import SchedulerStats

    front_addr = get_open_zmq_ipc_path()
    back_output_addr = get_open_zmq_ipc_path()
    back_publish_addr = get_open_zmq_ipc_path()

    coord = DPCoordinatorProc(engine_count=1, enable_wave_coordination=True)
    captured: dict = {}

    def run():
        try:
            # Mirrors run_coordinator(), which only catches KeyboardInterrupt;
            # any other exception would crash the coordinator process.
            coord.process_input_socket(
                front_addr, back_output_addr, back_publish_addr, None
            )
        except KeyboardInterrupt:
            captured["exc"] = "KeyboardInterrupt"
        except BaseException as e:  # noqa: BLE001
            captured["exc"] = f"{type(e).__name__}: {e}"

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    ctx = coord.ctx
    encoder = MsgpackEncoder()
    try:
        # Act as the single engine subscribing, so the coordinator finishes
        # startup and enters its steady-state poll loop.
        sub_back = ctx.socket(zmq.SUB)
        sub_back.setsockopt(zmq.SUBSCRIBE, b"")
        sub_back.connect(back_publish_addr)
        sub_back.RCVTIMEO = 5000
        assert sub_back.recv() == b"READY"

        # Subscribe to the front-end stats stream to observe liveness.
        sub_front = ctx.socket(zmq.SUB)
        sub_front.setsockopt(zmq.SUBSCRIBE, b"")
        sub_front.connect(front_addr)
        sub_front.RCVTIMEO = 8000
        time.sleep(0.2)  # let the SUB subscription propagate to the XPUB

        push = ctx.socket(zmq.PUSH)
        push.connect(back_output_addr)

        # 1) garbage frame (int where an EngineCoreOutputs array is expected).
        push.send(bytes(encoder.encode(123)[0]))
        # 2) a valid stats update from the engine.
        valid = EngineCoreOutputs(
            engine_index=0,
            scheduler_stats=SchedulerStats(num_waiting_reqs=5, num_running_reqs=3),
        )
        push.send(bytes(encoder.encode(valid)[0]))

        # The coordinator must survive the garbage frame and publish the stats
        # from the valid one ([waiting, running] == [5, 3]) to the front-end.
        seen = None
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                decoded = msgspec.msgpack.decode(sub_front.recv())
            except zmq.Again:
                break
            if decoded[0] == [[5, 3]]:
                seen = decoded[0]
                break

        assert "exc" not in captured, f"coordinator crashed: {captured.get('exc')}"
        assert thread.is_alive(), "coordinator loop exited unexpectedly"
        assert seen == [[5, 3]], (
            "coordinator did not publish stats from the valid frame "
            f"(captured={captured})"
        )
    finally:
        # The loop only exits on KeyboardInterrupt; destroying the context
        # force-closes its sockets so the thread unwinds cleanly.
        with contextlib.suppress(Exception):
            ctx.destroy(linger=0)
        thread.join(timeout=5)


def test_try_decode_frame_passes_through_valid():
    """A well-formed frame decodes normally and is returned unchanged."""
    from vllm.v1.serial_utils import DROP_FRAME, try_decode_frame

    buf = msgspec.msgpack.encode(["a", 1, 2])
    assert try_decode_frame(msgspec.msgpack.decode, buf, "test") == ["a", 1, 2]
    assert try_decode_frame(msgspec.msgpack.decode, buf, "test", expected_len=3) == [
        "a",
        1,
        2,
    ]
    assert try_decode_frame(msgspec.msgpack.decode, buf, "test") is not DROP_FRAME


def test_try_decode_frame_drops_undecodable():
    """Garbage bytes and typed-schema mismatches yield DROP_FRAME, not a raise.

    Covers the core #44486 crash: an ``int`` where an array is expected raises
    ``msgspec.ValidationError`` under a typed decoder; the helper must swallow it.
    """
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.serial_utils import DROP_FRAME, try_decode_frame

    # Raw junk bytes -> DecodeError.
    assert try_decode_frame(msgspec.msgpack.decode, b"\xc1\xff\xff", "test") is (
        DROP_FRAME
    )

    # Type mismatch against a typed decoder -> ValidationError
    # ("Expected array, got int"), the exact error from the issue.
    typed = MsgpackDecoder(EngineCoreOutputs)
    junk = bytes(MsgpackEncoder().encode(123)[0])
    assert try_decode_frame(typed.decode, junk, "test") is DROP_FRAME


def test_try_decode_frame_drops_wrong_shape():
    """A well-formed-but-misshapen untyped frame is dropped before the unpack.

    This is the second class of #44486 bug: ``msgspec.msgpack.decode`` succeeds
    on any valid msgpack, so a wrong-arity sequence (e.g. from a crafted frame)
    would crash the caller's fixed-arity tuple-unpack. ``expected_len`` turns
    that into a dropped frame instead.
    """
    from vllm.v1.serial_utils import DROP_FRAME, try_decode_frame

    # Right type, wrong length.
    buf = msgspec.msgpack.encode([1, 2])
    assert (
        try_decode_frame(msgspec.msgpack.decode, buf, "test", expected_len=3)
        is DROP_FRAME
    )

    # Decodes fine, but not a sequence at all.
    buf = msgspec.msgpack.encode(5)
    assert (
        try_decode_frame(msgspec.msgpack.decode, buf, "test", expected_len=2)
        is DROP_FRAME
    )
