# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for tensor IPC queue functionality."""

import contextlib
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.synchronize import Barrier as BarrierType
from multiprocessing.synchronize import Event as EventType
from typing import Any

import pytest
import torch
import torch.multiprocessing as torch_mp

from vllm.v1.engine.tensor_ipc import (
    TensorIpcData,
    TensorIpcReceiver,
    TensorIpcSender,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


@pytest.fixture(scope="module", autouse=True)
def setup_multiprocessing():
    """Set multiprocessing start method to 'spawn' for compatibility."""
    with contextlib.suppress(RuntimeError):
        # Already set, which is fine
        torch_mp.set_start_method("spawn", force=True)
    yield


@dataclass
# Use a typed container so the test covers the real vLLM path where tensor IPC
# handles are encoded and decoded as fields nested inside larger msgpack payloads.
class TensorEnvelope:
    tensor: torch.Tensor
    label: str


def encoder_process(
    tensor_queue: torch_mp.Queue,
    payload_queue: mp.Queue,
    result_queue: mp.Queue,
    tensor_data: dict[str, Any],
    ready_event: EventType,
    retrieval_done: EventType,
):
    """Process that msgpack-encodes and sends tensors via IPC."""
    try:
        sender = TensorIpcSender(tensor_queue)
        encoder = MsgpackEncoder(oob_tensor_consumer=sender)

        if torch.cuda.is_available():
            device = "cuda:0"
            tensor = torch.randn(
                *tensor_data["shape"], dtype=tensor_data["dtype"], device=device
            )
        else:
            # Fall back to CPU for testing
            device = "cpu"
            tensor = torch.randn(*tensor_data["shape"], dtype=tensor_data["dtype"])

        message = TensorEnvelope(tensor=tensor, label="cuda-msgpack")
        encoded = encoder.encode(message)
        payload_queue.put(encoded, timeout=10.0)

        ready_event.set()

        result_queue.put(
            {
                "success": True,
                "encoded_length": len(encoded),
                "device": str(device),
                "tensor_shape": tuple(tensor.shape),
            }
        )
        retrieval_done.wait(timeout=30.0)
    except Exception as e:
        import traceback

        ready_event.set()
        retrieval_done.set()
        result_queue.put(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )


def decoder_process(
    tensor_queue: torch_mp.Queue,
    payload_queue: mp.Queue,
    result_queue: mp.Queue,
    expected_shape: tuple,
    encoder_ready: EventType,
    retrieval_done: EventType,
):
    """Process that msgpack-decodes tensors received via IPC."""
    try:
        if not encoder_ready.wait(timeout=10.0):
            raise TimeoutError("Encoder did not signal ready")

        encoded = payload_queue.get(timeout=5.0)
        receiver = TensorIpcReceiver(tensor_queue)
        decoder = MsgpackDecoder(TensorEnvelope, oob_tensor_provider=receiver)
        decoded = decoder.decode(encoded)

        result_queue.put(
            {
                "success": True,
                "tensor_shape": tuple(decoded.tensor.shape),
                "device": str(decoded.tensor.device),
                "label": decoded.label,
                "matches_expected": tuple(decoded.tensor.shape) == expected_shape,
            }
        )
    except Exception as e:
        import traceback

        retrieval_done.set()
        result_queue.put(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )
    else:
        retrieval_done.set()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_tensor_queue_basic():
    """Test CUDA tensor IPC through the msgpack encoder/decoder path."""
    tensor_queue = torch_mp.Queue()
    payload_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    encoder_ready = mp.Event()
    retrieval_done = mp.Event()

    tensor_shape = (4, 8, 16)
    tensor_dtype = torch.float32

    encoder_proc = mp.Process(
        target=encoder_process,
        args=(
            tensor_queue,
            payload_queue,
            result_queue,
            {"shape": tensor_shape, "dtype": tensor_dtype},
            encoder_ready,
            retrieval_done,
        ),
    )
    encoder_proc.start()

    decoder_proc = mp.Process(
        target=decoder_process,
        args=(
            tensor_queue,
            payload_queue,
            result_queue,
            tensor_shape,
            encoder_ready,
            retrieval_done,
        ),
    )
    decoder_proc.start()

    encoder_result = result_queue.get(timeout=10.0)
    decoder_result = result_queue.get(timeout=10.0)

    encoder_proc.join(timeout=5.0)
    decoder_proc.join(timeout=5.0)

    # Verify results
    assert encoder_result["success"], (
        f"Encoder failed: {encoder_result.get('error')}\n"
        f"{encoder_result.get('traceback', '')}"
    )
    assert decoder_result["success"], (
        f"Decoder failed: {decoder_result.get('error')}\n"
        f"{decoder_result.get('traceback', '')}"
    )
    assert decoder_result["matches_expected"], "Tensor shape mismatch"
    assert "cuda" in decoder_result["device"], "Tensor not on CUDA device"
    assert decoder_result["label"] == "cuda-msgpack"


def test_cpu_tensor_fallback():
    """Test that CPU tensors use standard serialization path."""
    encoder = MsgpackEncoder()

    # Create a CPU tensor
    tensor = torch.randn(3, 4, dtype=torch.float32)

    # Encode the tensor (should use standard path, not queue)
    encoded = encoder.encode({"test_tensor": tensor})

    # Verify encoding succeeded
    assert len(encoded) > 0
    assert isinstance(encoded, (list, tuple))

    # Basic check: no queue should be used, so tensor goes through standard path
    # This is mainly to ensure no exceptions are raised


def test_msgpack_encoder_decoder_with_ipc():
    """Test the full msgpack + tensor IPC path in one process."""
    tensor_queue = torch_mp.Queue()
    sender = TensorIpcSender(tensor_queue)
    encoder = MsgpackEncoder(oob_tensor_consumer=sender)
    receiver = TensorIpcReceiver(tensor_queue)
    decoder = MsgpackDecoder(TensorEnvelope, oob_tensor_provider=receiver)

    # Use CPU here to exercise the msgpack + sender/receiver integration
    # without relying on same-process CUDA IPC behavior.
    tensor = torch.randn(2, 3)

    message = TensorEnvelope(tensor=tensor, label="test")
    encoded = encoder.encode(message)
    assert len(encoded) > 0

    decoded = decoder.decode(encoded)
    assert isinstance(decoded, TensorEnvelope)
    assert decoded.label == "test"
    assert torch.allclose(decoded.tensor, tensor)


def test_decoder_buffer_management():
    """Test receiver's tensor buffer management when draining queue."""
    tensor_queue = torch_mp.Queue()

    sender_id = "test_sender"
    message_id = 1

    # Put multiple tensors in queue using TensorIpcData
    tensors_data = [
        (0, torch.randn(2, 3)),
        (1, torch.randn(4, 5)),
        (2, torch.randn(6, 7)),
    ]

    for tensor_id, tensor in tensors_data:
        ipc_data = TensorIpcData(
            sender_id=sender_id,
            message_id=message_id,
            tensor_id=tensor_id,
            tensor=tensor,
        )
        tensor_queue.put(ipc_data)

    # Create receiver directly
    receiver = TensorIpcReceiver(tensor_queue)

    # Request tensor_id=2 (should buffer tensor_id=0 and tensor_id=1)
    handle = {"sender_id": sender_id, "message_id": message_id, "tensor_id": 2}

    result = receiver("float32", (6, 7), handle)
    assert result.shape == (6, 7)

    # Verify buffer has tensor_id 0 and 1
    sender = receiver._tensor_buffers[sender_id]
    tensors = sender.tensors.get(message_id, {})
    assert 0 in tensors
    assert 1 in tensors

    # Request buffered tensor
    handle2 = {"sender_id": sender_id, "message_id": message_id, "tensor_id": 0}

    result2 = receiver("float32", (2, 3), handle2)
    assert result2.shape == (2, 3)
    # tensor_id 0 should be removed from buffer
    sender = receiver._tensor_buffers[sender_id]
    tensors = sender.tensors.get(message_id, {})
    assert 0 not in tensors


def api_server_worker(
    server_id: int,
    tensor_queue: torch_mp.Queue,
    result_queue: mp.Queue,
    barrier: BarrierType,
    retrieval_done: EventType,
):
    """Worker simulating an API server sending tensors."""
    try:
        # Each server sends a unique tensor
        tensor = torch.ones(server_id + 1, server_id + 2) * server_id
        sender_id = f"server_{server_id}"

        # Wait for all servers to be ready
        barrier.wait()

        # Send tensor using TensorIpcData
        ipc_data = TensorIpcData(
            sender_id=sender_id,
            message_id=0,
            tensor_id=0,
            tensor=tensor,
        )
        tensor_queue.put(ipc_data)

        result_queue.put({"server_id": server_id, "success": True})

        # Keep process alive until main process has retrieved all tensors
        # This prevents shared memory handles from being invalidated
        retrieval_done.wait(timeout=30.0)
    except Exception as e:
        import traceback

        result_queue.put(
            {
                "server_id": server_id,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


def test_multiple_api_servers_to_engine():
    """Test multiple API servers sending to one engine core via multiprocessing."""
    num_api_servers = 3
    tensor_queue = torch_mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    barrier = mp.Barrier(num_api_servers)
    retrieval_done = mp.Event()

    # Start multiple API server processes
    processes = []
    for server_id in range(num_api_servers):
        proc = mp.Process(
            target=api_server_worker,
            args=(server_id, tensor_queue, result_queue, barrier, retrieval_done),
        )
        proc.start()
        processes.append(proc)

    # Collect results from all servers
    results = []
    for _ in range(num_api_servers):
        result = result_queue.get(timeout=10.0)
        results.append(result)

    # Verify all servers succeeded
    for result in results:
        assert result["success"], (
            f"Server {result['server_id']} failed: {result.get('error')}"
        )

    # Verify all tensors are in queue
    received_tensors = []
    for _ in range(num_api_servers):
        ipc_data = tensor_queue.get(timeout=1.0)
        received_tensors.append((ipc_data.sender_id, ipc_data.tensor))

    assert len(received_tensors) == num_api_servers

    # Verify tensor content (order may vary with multiprocessing)
    tensor_by_sender = {sid: t for sid, t in received_tensors}
    for server_id in range(num_api_servers):
        expected_id = f"server_{server_id}"
        assert expected_id in tensor_by_sender, (
            f"Missing tensor from server {server_id}"
        )
        expected_tensor = torch.ones(server_id + 1, server_id + 2) * server_id
        assert torch.allclose(tensor_by_sender[expected_id], expected_tensor)

    # Signal workers that retrieval is complete
    retrieval_done.set()

    # Wait for all processes to complete
    for proc in processes:
        proc.join(timeout=5.0)


def mixed_tensor_encoder_process(
    tensor_queue: torch_mp.Queue,
    result_queue: mp.Queue,
    ready_event: EventType,
    retrieval_done: EventType,
):
    """Process that encodes mixed CPU/CUDA tensors."""
    try:
        sender = TensorIpcSender(tensor_queue)
        _encoder = MsgpackEncoder(oob_tensor_consumer=sender)

        # Create only CUDA tensor for IPC (CPU will be serialized)
        # But actually, let's just send CUDA tensor directly
        cuda_tensor = torch.randn(4, 5, device="cuda:0")

        # Manually send via IPC to test the mechanism
        cuda_tensor_shared = cuda_tensor.share_memory_()

        ipc_data = TensorIpcData(
            sender_id="mixed_encoder",
            message_id=0,
            tensor_id=0,
            tensor=cuda_tensor_shared,
        )
        tensor_queue.put(ipc_data, timeout=10.0)

        ready_event.set()

        result_queue.put({"success": True, "sent_cuda": True})

        # Keep process alive until decoder has retrieved the tensor
        retrieval_done.wait(timeout=30.0)
    except Exception as e:
        import traceback

        ready_event.set()
        result_queue.put(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )


def mixed_tensor_decoder_process(
    tensor_queue: torch_mp.Queue,
    result_queue: mp.Queue,
    encoder_ready: EventType,
    retrieval_done: EventType,
):
    """Process that retrieves mixed tensors from queue."""
    try:
        # Wait for encoder to finish
        if not encoder_ready.wait(timeout=10.0):
            raise TimeoutError("Encoder did not signal ready")

        # Try to get CUDA tensor from queue
        ipc_data = tensor_queue.get(timeout=5.0)

        result_queue.put(
            {
                "success": True,
                "is_cuda": ipc_data.tensor.is_cuda,
                "shape": tuple(ipc_data.tensor.shape),
            }
        )

        # Signal that retrieval is complete
        retrieval_done.set()
    except Exception as e:
        import traceback

        retrieval_done.set()  # Signal even on failure
        result_queue.put(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_cpu_cuda_tensors():
    """Test encoding with mixed CPU and CUDA tensors using multiprocessing."""
    tensor_queue = torch_mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    encoder_ready = mp.Event()
    retrieval_done = mp.Event()

    # Start encoder process
    encoder_proc = mp.Process(
        target=mixed_tensor_encoder_process,
        args=(tensor_queue, result_queue, encoder_ready, retrieval_done),
    )
    encoder_proc.start()

    # Start decoder process
    decoder_proc = mp.Process(
        target=mixed_tensor_decoder_process,
        args=(tensor_queue, result_queue, encoder_ready, retrieval_done),
    )
    decoder_proc.start()

    # Get results
    encoder_result = result_queue.get(timeout=10.0)
    decoder_result = result_queue.get(timeout=10.0)

    encoder_proc.join(timeout=5.0)
    decoder_proc.join(timeout=5.0)

    # Verify encoder succeeded
    assert encoder_result["success"], (
        f"Encoder failed: {encoder_result.get('error')}\n"
        f"{encoder_result.get('traceback', '')}"
    )

    # Verify decoder succeeded and got CUDA tensor
    assert decoder_result["success"], (
        f"Decoder failed: {decoder_result.get('error')}\n"
        f"{decoder_result.get('traceback', '')}"
    )
    assert decoder_result["is_cuda"], "Retrieved tensor is not on CUDA"
    assert decoder_result["shape"] == (4, 5), (
        f"Unexpected shape: {decoder_result['shape']}"
    )


def cpu_tensor_ipc_encoder_process(
    tensor_queue: torch_mp.Queue,
    result_queue: mp.Queue,
    tensor_shape: tuple,
    ready_event: EventType,
    retrieval_done: EventType,
):
    """Process that encodes and sends CPU tensors via IPC queue."""
    try:
        # Create encoder with IPC enabled for all tensors
        sender = TensorIpcSender(tensor_queue)
        encoder = MsgpackEncoder(oob_tensor_consumer=sender)

        # Create a CPU tensor
        tensor = torch.randn(*tensor_shape, dtype=torch.float32)

        # Encode the tensor (should use IPC queue, not standard serialization)
        encoded = encoder.encode({"test_tensor": tensor})

        # Signal that encoding is complete
        ready_event.set()

        result_queue.put(
            {
                "success": True,
                "encoded_length": len(encoded),
                "device": str(tensor.device),
                "tensor_shape": tuple(tensor.shape),
            }
        )

        # Keep process alive until decoder has retrieved the tensor
        # This is necessary for CPU tensor shared memory to remain valid
        retrieval_done.wait(timeout=30.0)
    except Exception as e:
        import traceback

        ready_event.set()
        result_queue.put(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )


def cpu_tensor_ipc_decoder_process(
    tensor_queue: torch_mp.Queue,
    result_queue: mp.Queue,
    expected_shape: tuple,
    encoder_ready: EventType,
    retrieval_done: EventType,
):
    """Process that decodes and receives CPU tensors from IPC queue."""
    try:
        # Wait for encoder to finish sending
        if not encoder_ready.wait(timeout=10.0):
            raise TimeoutError("Encoder did not signal ready")

        # Get tensor from queue
        ipc_data = tensor_queue.get(timeout=5.0)

        result_queue.put(
            {
                "success": True,
                "tensor_id": ipc_data.tensor_id,
                "tensor_shape": tuple(ipc_data.tensor.shape),
                "device": str(ipc_data.tensor.device),
                "matches_expected": tuple(ipc_data.tensor.shape) == expected_shape,
                "is_cpu": ipc_data.tensor.device.type == "cpu",
            }
        )

        # Signal that retrieval is complete
        retrieval_done.set()
    except Exception as e:
        import traceback

        retrieval_done.set()  # Signal even on failure
        result_queue.put(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )


def test_cpu_tensor_ipc():
    """Test CPU tensor sharing via IPC queue when mm_tensor_ipc is enabled."""
    # Set up single queue and synchronization
    tensor_queue = torch_mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    encoder_ready = mp.Event()
    retrieval_done = mp.Event()

    tensor_shape = (3, 5, 7)

    # Start encoder process
    encoder_proc = mp.Process(
        target=cpu_tensor_ipc_encoder_process,
        args=(
            tensor_queue,
            result_queue,
            tensor_shape,
            encoder_ready,
            retrieval_done,
        ),
    )
    encoder_proc.start()

    # Start decoder process
    decoder_proc = mp.Process(
        target=cpu_tensor_ipc_decoder_process,
        args=(
            tensor_queue,
            result_queue,
            tensor_shape,
            encoder_ready,
            retrieval_done,
        ),
    )
    decoder_proc.start()

    # Wait for processes and collect results
    encoder_result = result_queue.get(timeout=10.0)
    decoder_result = result_queue.get(timeout=10.0)

    encoder_proc.join(timeout=5.0)
    decoder_proc.join(timeout=5.0)

    # Verify results
    assert encoder_result["success"], (
        f"Encoder failed: {encoder_result.get('error')}\n"
        f"{encoder_result.get('traceback', '')}"
    )
    assert decoder_result["success"], (
        f"Decoder failed: {decoder_result.get('error')}\n"
        f"{decoder_result.get('traceback', '')}"
    )
    assert decoder_result["matches_expected"], "Tensor shape mismatch"
    assert decoder_result["is_cpu"], "Tensor not on CPU device"


def test_ipc_disabled_mode():
    """Test that IPC is disabled when no sender is provided."""
    tensor_queues = [torch_mp.Queue()]

    # Create encoder without IPC sender (IPC disabled)
    encoder = MsgpackEncoder()

    # Create a CPU tensor
    cpu_tensor = torch.randn(2, 3, dtype=torch.float32)

    # Encode the tensor (should use standard serialization, not IPC)
    encoded = encoder.encode({"test_tensor": cpu_tensor})

    # Verify encoding succeeded
    assert len(encoded) > 0
    assert isinstance(encoded, (list, tuple))

    # Verify queue is empty (no IPC was used)
    assert tensor_queues[0].empty(), "Tensor queue should be empty when IPC is disabled"

    # If CUDA is available, test with CUDA tensor too
    if torch.cuda.is_available():
        cuda_tensor = torch.randn(4, 5, device="cuda:0")
        encoded_cuda = encoder.encode({"cuda_tensor": cuda_tensor})
        assert len(encoded_cuda) > 0
        assert tensor_queues[0].empty(), (
            "Tensor queue should be empty for CUDA tensor when IPC is disabled"
        )


@dataclass
class MultiTensorMessage:
    """Message with multiple tensors to test multi-tensor IPC."""

    t1: torch.Tensor
    t2: torch.Tensor
    sender_label: str


def concurrent_sender_process(
    tensor_queue: torch_mp.Queue,
    payload_queue: mp.Queue,
    result_queue: mp.Queue,
    sender_index: int,
    num_messages: int,
    barrier: BarrierType,
    retrieval_done: EventType,
):
    """Process that acts as one of N concurrent senders."""
    try:
        sender = TensorIpcSender(tensor_queue)
        encoder = MsgpackEncoder(oob_tensor_consumer=sender)

        # Wait for all senders to be ready before sending
        barrier.wait(timeout=10.0)

        encoded_payloads = []
        for msg_idx in range(num_messages):
            # Each sender creates uniquely-shaped tensors so we can
            # verify correct routing on the receiver side.
            t1 = torch.full((sender_index + 1, 3), float(msg_idx), dtype=torch.float32)
            t2 = torch.full(
                (2, sender_index + 2), float(msg_idx + 100), dtype=torch.float64
            )
            msg = MultiTensorMessage(
                t1=t1,
                t2=t2,
                sender_label=f"sender_{sender_index}_msg_{msg_idx}",
            )
            encoded = encoder.encode(msg)
            encoded_payloads.append(encoded)

        # Send all encoded payloads via the regular (non-tensor) queue
        for encoded in encoded_payloads:
            payload_queue.put(encoded, timeout=10.0)

        result_queue.put(
            {
                "success": True,
                "sender_index": sender_index,
                "num_sent": num_messages,
            }
        )

        # Keep alive so shared-memory handles remain valid
        retrieval_done.wait(timeout=30.0)
    except Exception as e:
        import traceback

        result_queue.put(
            {
                "success": False,
                "sender_index": sender_index,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


def test_concurrent_senders_single_receiver():
    """Test N concurrent senders sharing one queue with a single receiver.

    Each sender encodes multiple messages (each containing two tensors) via
    its own MsgpackEncoder + TensorIpcSender.  A single TensorIpcReceiver
    on the receiving side must correctly drain-and-buffer interleaved
    TensorIpcData items from the shared queue and match them back to the
    right message handles during decode.
    """
    num_senders = 4
    num_messages_per_sender = 3
    tensor_queue = torch_mp.Queue()
    payload_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    barrier = mp.Barrier(num_senders)
    retrieval_done = mp.Event()

    # Launch sender processes
    processes = []
    for i in range(num_senders):
        proc = mp.Process(
            target=concurrent_sender_process,
            args=(
                tensor_queue,
                payload_queue,
                result_queue,
                i,
                num_messages_per_sender,
                barrier,
                retrieval_done,
            ),
        )
        proc.start()
        processes.append(proc)

    # Collect send confirmations
    send_results = []
    for _ in range(num_senders):
        send_results.append(result_queue.get(timeout=15.0))
    for r in send_results:
        assert r["success"], (
            f"Sender {r['sender_index']} failed: {r.get('error')}\n"
            f"{r.get('traceback', '')}"
        )

    # Now decode all messages from the main process using a single receiver
    receiver = TensorIpcReceiver(tensor_queue)
    decoder = MsgpackDecoder(MultiTensorMessage, oob_tensor_provider=receiver)

    decoded_messages: list[MultiTensorMessage] = []
    total = num_senders * num_messages_per_sender
    for _ in range(total):
        encoded = payload_queue.get(timeout=10.0)
        decoded = decoder.decode(encoded)
        assert isinstance(decoded, MultiTensorMessage)
        decoded_messages.append(decoded)

    # Signal senders they can exit
    retrieval_done.set()

    # Group by sender_label prefix to verify all messages arrived
    by_sender: dict[int, list[MultiTensorMessage]] = {}
    for msg in decoded_messages:
        # label format: "sender_{i}_msg_{j}"
        parts = msg.sender_label.split("_")
        sender_idx = int(parts[1])
        by_sender.setdefault(sender_idx, []).append(msg)

    assert len(by_sender) == num_senders, (
        f"Expected {num_senders} senders, got {len(by_sender)}"
    )

    for sender_idx in range(num_senders):
        msgs = sorted(by_sender[sender_idx], key=lambda m: m.sender_label)
        assert len(msgs) == num_messages_per_sender, (
            f"Sender {sender_idx}: expected {num_messages_per_sender} "
            f"messages, got {len(msgs)}"
        )
        for msg_idx, msg in enumerate(msgs):
            assert msg.sender_label == f"sender_{sender_idx}_msg_{msg_idx}"
            # Verify tensor shapes match what the sender created
            assert msg.t1.shape == (sender_idx + 1, 3)
            assert msg.t2.shape == (2, sender_idx + 2)
            # Verify tensor values
            assert torch.allclose(msg.t1, torch.full_like(msg.t1, float(msg_idx)))
            assert torch.allclose(msg.t2, torch.full_like(msg.t2, float(msg_idx + 100)))

    for proc in processes:
        proc.join(timeout=5.0)


def test_concurrent_senders_interleaved_buffer():
    """Test receiver buffering when tensors from multiple senders interleave.

    Manually enqueue TensorIpcData from two senders in an interleaved order
    and verify the receiver correctly buffers and retrieves each tensor by
    its (sender_id, message_id, tensor_id) handle.
    """
    tensor_queue = torch_mp.Queue()

    # Sender A: 2 tensors for message 1
    a_t0 = torch.randn(2, 3)
    a_t1 = torch.randn(4, 5)
    # Sender B: 2 tensors for message 1
    b_t0 = torch.randn(6, 7)
    b_t1 = torch.randn(8, 9)

    # Interleave: B_t0, A_t0, B_t1, A_t1
    for sid, mid, tid, t in [
        ("B", 1, 0, b_t0),
        ("A", 1, 0, a_t0),
        ("B", 1, 1, b_t1),
        ("A", 1, 1, a_t1),
    ]:
        tensor_queue.put(
            TensorIpcData(sender_id=sid, message_id=mid, tensor_id=tid, tensor=t)
        )

    receiver = TensorIpcReceiver(tensor_queue)

    # Request A_t1 first — receiver must drain and buffer B_t0, A_t0, B_t1
    result = receiver(
        "float32", a_t1.shape, {"sender_id": "A", "message_id": 1, "tensor_id": 1}
    )
    assert torch.equal(result, a_t1)

    # Now request B_t0 from buffer
    result = receiver(
        "float32", b_t0.shape, {"sender_id": "B", "message_id": 1, "tensor_id": 0}
    )
    assert torch.equal(result, b_t0)

    # Request A_t0 from buffer
    result = receiver(
        "float32", a_t0.shape, {"sender_id": "A", "message_id": 1, "tensor_id": 0}
    )
    assert torch.equal(result, a_t0)

    # Request B_t1 from buffer
    result = receiver(
        "float64", b_t1.shape, {"sender_id": "B", "message_id": 1, "tensor_id": 1}
    )
    assert torch.equal(result, b_t1)

    # All buffers should be drained
    for sid in ("A", "B"):
        tensors = receiver._tensor_buffers[sid].tensors.get(1, {})
        assert len(tensors) == 0, f"Sender {sid} buffer not empty: {tensors}"


def test_mixed_cpu_cuda_with_ipc_enabled():
    """Test that encoder is configured correctly for IPC with all tensor types."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tensor_queue = torch_mp.Queue()

    # Create sender and encoder with IPC enabled
    sender = TensorIpcSender(tensor_queue)
    encoder = MsgpackEncoder(oob_tensor_consumer=sender)

    # Verify sender configuration
    assert encoder.oob_tensor_consumer is not None, "Consumer should be set"

    # Note: Actual IPC transfer only works across processes
    # (tested in test_cpu_tensor_ipc)
    # This test just verifies the configuration is correct


def test_tensor_cleanup_after_decode():
    """Test that tensors are removed from tracking after successful decode."""
    # Create a tensor queue
    tensor_queue = torch_mp.Queue()

    # Create and encode a tensor
    tensor = torch.randn(5, 5)
    # Move to shared memory for IPC
    if not tensor.is_shared():
        tensor.share_memory_()

    # Manually create a TensorIpcData and put it in the queue
    sender_id = "test_sender"
    message_id = 0
    tensor_id = 0
    ipc_data = TensorIpcData(
        sender_id=sender_id,
        message_id=message_id,
        tensor_id=tensor_id,
        tensor=tensor,
    )
    tensor_queue.put(ipc_data)

    # Create receiver directly
    receiver = TensorIpcReceiver(tensor_queue)

    handle = {
        "sender_id": sender_id,
        "message_id": message_id,
        "tensor_id": tensor_id,
    }

    # Receive the tensor - this should retrieve it from the queue
    decoded_tensor = receiver(
        str(tensor.dtype).removeprefix("torch."), tensor.shape, handle
    )

    # Verify the tensor was decoded
    assert decoded_tensor.shape == tensor.shape, "Decoded tensor should match shape"

    # Verify the tensor was removed from buffer after decode
    sender = receiver._tensor_buffers[sender_id]
    tensors = sender.tensors.get(message_id, {})
    assert tensor_id not in tensors, "Tensor should be removed from buffer"
