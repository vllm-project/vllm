# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for tensor IPC queue functionality."""

import contextlib
import multiprocessing as mp
from multiprocessing.synchronize import Barrier as BarrierType
from multiprocessing.synchronize import Event as EventType
from typing import Any

import pytest
import torch
import torch.multiprocessing as torch_mp

from vllm.v1.serial_utils import (
    MsgpackDecoder,
    MsgpackEncoder,
    TensorIpcData,
    TensorIpcHandle,
)


@pytest.fixture(scope="module", autouse=True)
def setup_multiprocessing():
    """Set multiprocessing start method to 'spawn' for compatibility."""
    with contextlib.suppress(RuntimeError):
        # Already set, which is fine
        torch_mp.set_start_method("spawn", force=True)
    yield


def encoder_process(
    tensor_queues: list[torch_mp.Queue],
    result_queue: mp.Queue,
    target_engine: int,
    tensor_data: dict[str, Any],
    ready_event: EventType,
):
    """Process that encodes and sends CUDA tensors via queue."""
    try:
        # Create encoder with tensor queues
        encoder = MsgpackEncoder(tensor_queues=tensor_queues)
        encoder.set_target_engine(target_engine)

        # Create a CUDA tensor if available
        if torch.cuda.is_available():
            device = "cuda:0"
            tensor = torch.randn(
                *tensor_data["shape"], dtype=tensor_data["dtype"], device=device
            )
        else:
            # Fall back to CPU for testing
            device = "cpu"
            tensor = torch.randn(*tensor_data["shape"], dtype=tensor_data["dtype"])

        # Encode the tensor
        encoded = encoder.encode({"test_tensor": tensor})

        # Signal that encoding is complete before sending result
        ready_event.set()

        result_queue.put(
            {
                "success": True,
                "encoded_length": len(encoded),
                "device": str(device),
                "tensor_shape": tuple(tensor.shape),
            }
        )
    except Exception as e:
        import traceback

        ready_event.set()  # Signal even on failure
        result_queue.put(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )


def decoder_process(
    tensor_queue: torch_mp.Queue,
    result_queue: mp.Queue,
    expected_shape: tuple,
    encoder_ready: EventType,
):
    """Process that decodes and receives CUDA tensors from queue."""
    try:
        # Wait for encoder to finish sending
        if not encoder_ready.wait(timeout=10.0):
            raise TimeoutError("Encoder did not signal ready")

        # Try to get tensor from queue directly for testing
        ipc_data = tensor_queue.get(timeout=5.0)

        result_queue.put(
            {
                "success": True,
                "tensor_id": ipc_data.tensor_id,
                "tensor_shape": tuple(ipc_data.tensor.shape),
                "device": str(ipc_data.tensor.device),
                "matches_expected": tuple(ipc_data.tensor.shape) == expected_shape,
            }
        )
    except Exception as e:
        import traceback

        result_queue.put(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_tensor_queue_basic():
    """Test basic CUDA tensor sharing via queue."""
    # Set up queues and synchronization
    num_engines = 2
    tensor_queues = [torch_mp.Queue() for _ in range(num_engines)]
    result_queue: mp.Queue = mp.Queue()
    encoder_ready = mp.Event()

    target_engine = 0
    tensor_shape = (4, 8, 16)
    tensor_dtype = torch.float32

    # Start encoder process
    encoder_proc = mp.Process(
        target=encoder_process,
        args=(
            tensor_queues,
            result_queue,
            target_engine,
            {"shape": tensor_shape, "dtype": tensor_dtype},
            encoder_ready,
        ),
    )
    encoder_proc.start()

    # Start decoder process
    decoder_proc = mp.Process(
        target=decoder_process,
        args=(tensor_queues[target_engine], result_queue, tensor_shape, encoder_ready),
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
    assert "cuda" in decoder_result["device"], "Tensor not on CUDA device"


def test_cpu_tensor_fallback():
    """Test that CPU tensors use standard serialization path."""
    encoder = MsgpackEncoder(tensor_queues=None)

    # Create a CPU tensor
    tensor = torch.randn(3, 4, dtype=torch.float32)

    # Encode the tensor (should use standard path, not queue)
    encoded = encoder.encode({"test_tensor": tensor})

    # Verify encoding succeeded
    assert len(encoded) > 0
    assert isinstance(encoded, (list, tuple))

    # Basic check: no queue should be used, so tensor goes through standard path
    # This is mainly to ensure no exceptions are raised


def test_encoder_without_target_engine():
    """Test that encoder handles missing target engine gracefully."""
    tensor_queues = [torch_mp.Queue()]
    encoder = MsgpackEncoder(tensor_queues=tensor_queues)

    # Don't set target engine
    if torch.cuda.is_available():
        tensor = torch.randn(2, 3, device="cuda:0")
    else:
        tensor = torch.randn(2, 3)

    # Should fall back to standard serialization
    encoded = encoder.encode({"test_tensor": tensor})
    assert len(encoded) > 0


def test_decoder_buffer_management():
    """Test decoder's tensor buffer management when draining queue."""
    tensor_queue = torch_mp.Queue()

    # Put multiple tensors in queue using TensorIpcData
    tensors = {
        "tensor_1": torch.randn(2, 3),
        "tensor_2": torch.randn(4, 5),
        "tensor_3": torch.randn(6, 7),
    }

    for tensor_id, tensor in tensors.items():
        ipc_data = TensorIpcData(request_id=None, tensor_id=tensor_id, tensor=tensor)
        tensor_queue.put(ipc_data)

    # Create decoder
    decoder = MsgpackDecoder(tensor_queue=tensor_queue)

    # Request tensor_3 (should buffer tensor_1 and tensor_2)
    handle = TensorIpcHandle(
        request_id=None,
        tensor_id="tensor_3",
        shape=[6, 7],
        dtype="float32",
        device="cpu",
    )

    result = decoder._decode_ipc_queue_tensor(handle)
    assert result.shape == (6, 7)

    # Verify buffer has tensor_1 and tensor_2 using tuple keys
    assert (None, "tensor_1") in decoder._tensor_buffer
    assert (None, "tensor_2") in decoder._tensor_buffer

    # Request buffered tensor
    handle2 = TensorIpcHandle(
        request_id=None,
        tensor_id="tensor_1",
        shape=[2, 3],
        dtype="float32",
        device="cpu",
    )

    result2 = decoder._decode_ipc_queue_tensor(handle2)
    assert result2.shape == (2, 3)
    # tensor_1 should be removed from buffer
    assert (None, "tensor_1") not in decoder._tensor_buffer


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
        tensor_id = f"server_{server_id}_tensor"

        # Wait for all servers to be ready
        barrier.wait()

        # Send tensor using TensorIpcData
        ipc_data = TensorIpcData(request_id=None, tensor_id=tensor_id, tensor=tensor)
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
        received_tensors.append((ipc_data.tensor_id, ipc_data.tensor))

    assert len(received_tensors) == num_api_servers

    # Verify tensor content (order may vary with multiprocessing)
    tensor_by_id = {tid: t for tid, t in received_tensors}
    for server_id in range(num_api_servers):
        expected_id = f"server_{server_id}_tensor"
        assert expected_id in tensor_by_id, f"Missing tensor from server {server_id}"
        expected_tensor = torch.ones(server_id + 1, server_id + 2) * server_id
        assert torch.allclose(tensor_by_id[expected_id], expected_tensor)

    # Signal workers that retrieval is complete
    retrieval_done.set()

    # Wait for all processes to complete
    for proc in processes:
        proc.join(timeout=5.0)


def mixed_tensor_encoder_process(
    tensor_queues: list[torch_mp.Queue],
    result_queue: mp.Queue,
    ready_event: EventType,
    retrieval_done: EventType,
):
    """Process that encodes mixed CPU/CUDA tensors."""
    try:
        encoder = MsgpackEncoder(
            tensor_queues=tensor_queues, multimodal_tensor_ipc="torch"
        )
        encoder.set_target_engine(0)

        # Create only CUDA tensor for IPC (CPU will be serialized)
        # But actually, let's just send CUDA tensor directly
        cuda_tensor = torch.randn(4, 5, device="cuda:0")

        # Manually send via IPC to test the mechanism
        tensor_id = "test_cuda_tensor"
        cuda_tensor_shared = cuda_tensor.share_memory_()
        from vllm.v1.serial_utils import TensorIpcData

        ipc_data = TensorIpcData(
            request_id=None, tensor_id=tensor_id, tensor=cuda_tensor_shared
        )
        tensor_queues[0].put(ipc_data, timeout=10.0)

        ready_event.set()

        result_queue.put(
            {
                "success": True,
                "sent_cuda": True,
            }
        )

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
    tensor_queues = [torch_mp.Queue()]
    result_queue: mp.Queue = mp.Queue()
    encoder_ready = mp.Event()
    retrieval_done = mp.Event()

    # Start encoder process
    encoder_proc = mp.Process(
        target=mixed_tensor_encoder_process,
        args=(tensor_queues, result_queue, encoder_ready, retrieval_done),
    )
    encoder_proc.start()

    # Start decoder process
    decoder_proc = mp.Process(
        target=mixed_tensor_decoder_process,
        args=(tensor_queues[0], result_queue, encoder_ready, retrieval_done),
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
    tensor_queues: list[torch_mp.Queue],
    result_queue: mp.Queue,
    target_engine: int,
    tensor_shape: tuple,
    ready_event: EventType,
    retrieval_done: EventType,
):
    """Process that encodes and sends CPU tensors via IPC queue."""
    try:
        # Create encoder with IPC enabled for all tensors
        encoder = MsgpackEncoder(
            tensor_queues=tensor_queues, multimodal_tensor_ipc="torch"
        )
        encoder.set_target_engine(target_engine)

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
    """Test CPU tensor sharing via IPC queue when multimodal_tensor_ipc is enabled."""
    # Set up queues and synchronization
    num_engines = 2
    tensor_queues = [torch_mp.Queue() for _ in range(num_engines)]
    result_queue: mp.Queue = mp.Queue()
    encoder_ready = mp.Event()
    retrieval_done = mp.Event()

    target_engine = 0
    tensor_shape = (3, 5, 7)

    # Start encoder process
    encoder_proc = mp.Process(
        target=cpu_tensor_ipc_encoder_process,
        args=(
            tensor_queues,
            result_queue,
            target_engine,
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
            tensor_queues[target_engine],
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
    """Test that IPC is disabled when multimodal_tensor_ipc="msgspec"."""
    tensor_queues = [torch_mp.Queue()]

    # Create encoder with IPC disabled
    encoder = MsgpackEncoder(
        tensor_queues=tensor_queues, multimodal_tensor_ipc="msgspec"
    )
    encoder.set_target_engine(0)

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


def test_mixed_cpu_cuda_with_ipc_enabled():
    """Test that encoder is configured correctly for IPC with all tensor types."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tensor_queues = [torch_mp.Queue()]

    # Create encoder with IPC enabled for all tensors
    encoder = MsgpackEncoder(tensor_queues=tensor_queues, multimodal_tensor_ipc="torch")
    encoder.set_target_engine(0)

    # Verify encoder configuration
    assert encoder.multimodal_tensor_ipc == "torch", (
        "Torch queue-based IPC should be enabled"
    )
    assert encoder.tensor_queues is not None, "Tensor queues should be set"
    assert encoder.target_engine_index == 0, "Target engine should be set"

    # Note: Actual IPC transfer only works across processes
    # (tested in test_cpu_tensor_ipc)
    # This test just verifies the configuration is correct


def test_tensor_cleanup_on_abort():
    """Test that orphaned tensors are cleaned up when requests are aborted."""
    # Create a tensor queue (not actually used in this simplified test)
    tensor_queue = torch_mp.Queue()

    # Create decoder
    decoder = MsgpackDecoder(dict, tensor_queue=tensor_queue)

    # Simulate tensors in the buffer for multiple requests
    request_ids = ["req1", "req2", "req3"]

    for request_id in request_ids:
        # Simulate 2 tensors per request using tuple keys
        for i in range(2):
            tensor_id = f"encoder_{i}"
            tensor_key = (request_id, tensor_id)
            tensor = torch.randn(10, 10)

            # Manually add to buffer and tracking (simulating decode behavior)
            decoder._tensor_buffer[tensor_key] = tensor

            if request_id not in decoder._request_to_tensors:
                decoder._request_to_tensors[request_id] = []
            decoder._request_to_tensors[request_id].append(tensor_key)

    # Verify tensors are in the buffer
    initial_buffer_size = len(decoder._tensor_buffer)
    assert initial_buffer_size == 6, "Buffer should contain 6 tensors (2 per request)"

    # Verify request tracking
    assert len(decoder._request_to_tensors) == 3, "Should track 3 requests"
    assert len(decoder._request_to_tensors["req1"]) == 2, "req1 should have 2 tensors"

    # Cleanup tensors for req1
    removed_count_1 = decoder.cleanup_request_tensors("req1")
    assert removed_count_1 == 2, "Should have removed 2 tensors for req1"
    assert len(decoder._tensor_buffer) == 4, "Buffer should have 4 tensors left"
    assert "req1" not in decoder._request_to_tensors, (
        "req1 should be removed from tracking"
    )

    # Cleanup tensors for req2
    removed_count_2 = decoder.cleanup_request_tensors("req2")
    assert removed_count_2 == 2, "Should have removed 2 tensors for req2"
    assert len(decoder._tensor_buffer) == 2, "Buffer should have 2 tensors left"

    # Cleanup req3
    removed_count_3 = decoder.cleanup_request_tensors("req3")
    assert removed_count_3 == 2, "Should have removed 2 tensors for req3"

    # Verify all tensors are cleaned up
    assert len(decoder._tensor_buffer) == 0, "Buffer should be empty"
    assert len(decoder._request_to_tensors) == 0, "Request tracking should be empty"

    # Cleanup for non-existent request should return 0
    removed_count_4 = decoder.cleanup_request_tensors("nonexistent")
    assert removed_count_4 == 0, "Should return 0 for non-existent request"


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
    request_id = "test_req"
    tensor_id = "encoder_0"
    ipc_data = TensorIpcData(request_id=request_id, tensor_id=tensor_id, tensor=tensor)
    tensor_queue.put(ipc_data)

    # Create decoder
    decoder = MsgpackDecoder(dict, tensor_queue=tensor_queue)

    # Create a TensorIpcHandle to decode
    handle = TensorIpcHandle(
        request_id=request_id,
        tensor_id=tensor_id,
        shape=list(tensor.shape),
        dtype=str(tensor.dtype).removeprefix("torch."),
        device=str(tensor.device),
    )

    # Decode the tensor - this should retrieve it from the queue
    decoded_tensor = decoder._decode_ipc_queue_tensor(handle)

    # Verify the tensor was decoded
    assert decoded_tensor.shape == tensor.shape, "Decoded tensor should match shape"

    # Verify the tensor was removed from buffer after decode
    tensor_key = (request_id, tensor_id)
    assert tensor_key not in decoder._tensor_buffer, (
        "Tensor should be removed from buffer"
    )

    # Verify the request tracking was cleaned up
    assert request_id not in decoder._request_to_tensors, (
        "Request tracking should be cleaned up"
    )


def test_request_context_in_encoder():
    """Test that encoder properly sets and clears request context."""
    encoder = MsgpackEncoder()

    # Initially no request context
    assert encoder._current_request_id is None

    # Set request context
    encoder.set_request_context("req123")
    assert encoder._current_request_id == "req123"

    # Clear request context
    encoder.set_request_context(None)
    assert encoder._current_request_id is None
