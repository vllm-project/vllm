# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for tensor IPC queue functionality."""

import multiprocessing as mp
from typing import Any

import pytest
import torch
import torch.multiprocessing as torch_mp

from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, TensorIpcData, TensorIpcHandle

# Set multiprocessing start method to 'spawn' for compatibility
torch_mp.set_start_method('spawn', force=True)


def encoder_process(
    tensor_queues: list[torch_mp.Queue],
    result_queue: mp.Queue,
    target_engine: int,
    tensor_data: dict[str, Any],
    ready_event: mp.Event,
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
        result_queue.put({
            "success": False, 
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def decoder_process(
    tensor_queue: torch_mp.Queue,
    result_queue: mp.Queue,
    expected_shape: tuple,
    encoder_ready: mp.Event,
):
    """Process that decodes and receives CUDA tensors from queue."""
    try:
        # Create decoder with tensor queue
        decoder = MsgpackDecoder(tensor_queue=tensor_queue)

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
        result_queue.put({
            "success": False, 
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_tensor_queue_basic():
    """Test basic CUDA tensor sharing via queue."""
    # Set up queues and synchronization
    num_engines = 2
    tensor_queues = [torch_mp.Queue() for _ in range(num_engines)]
    result_queue = mp.Queue()
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
    assert encoder_result["success"], f"Encoder failed: {encoder_result.get('error')}\n{encoder_result.get('traceback', '')}"
    assert decoder_result["success"], f"Decoder failed: {decoder_result.get('error')}\n{decoder_result.get('traceback', '')}"
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
    print(f"  Encoded CPU tensor with shape {tensor.shape} using standard path")


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
        ipc_data = TensorIpcData(tensor_id=tensor_id, tensor=tensor)
        tensor_queue.put(ipc_data)

    # Create decoder
    decoder = MsgpackDecoder(tensor_queue=tensor_queue)

    # Request tensor_3 (should buffer tensor_1 and tensor_2)
    handle = TensorIpcHandle(
        tensor_id="tensor_3",
        shape=[6, 7],
        dtype="float32",
        device="cpu",
    )

    result = decoder._decode_cuda_queue_tensor(handle)
    assert result.shape == (6, 7)

    # Verify buffer has tensor_1 and tensor_2
    assert "tensor_1" in decoder._tensor_buffer
    assert "tensor_2" in decoder._tensor_buffer

    # Request buffered tensor
    handle2 = TensorIpcHandle(
        tensor_id="tensor_1",
        shape=[2, 3],
        dtype="float32",
        device="cpu",
    )

    result2 = decoder._decode_cuda_queue_tensor(handle2)
    assert result2.shape == (2, 3)
    # tensor_1 should be removed from buffer
    assert "tensor_1" not in decoder._tensor_buffer


def api_server_worker(
    server_id: int,
    tensor_queue: torch_mp.Queue,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    retrieval_done: mp.Event,
):
    """Worker simulating an API server sending tensors."""
    try:
        # Each server sends a unique tensor
        tensor = torch.ones(server_id + 1, server_id + 2) * server_id
        tensor_id = f"server_{server_id}_tensor"
        
        # Wait for all servers to be ready
        barrier.wait()
        
        # Send tensor using TensorIpcData
        ipc_data = TensorIpcData(tensor_id=tensor_id, tensor=tensor)
        tensor_queue.put(ipc_data)
        
        result_queue.put({"server_id": server_id, "success": True})
        
        # Keep process alive until main process has retrieved all tensors
        # This prevents shared memory handles from being invalidated
        retrieval_done.wait(timeout=30.0)
    except Exception as e:
        import traceback
        result_queue.put({
            "server_id": server_id,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def test_multiple_api_servers_to_engine():
    """Test multiple API servers sending to one engine core via multiprocessing."""
    num_api_servers = 3
    tensor_queue = torch_mp.Queue()
    result_queue = mp.Queue()
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
        assert result["success"], f"Server {result['server_id']} failed: {result.get('error')}"

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
    ready_event: mp.Event,
):
    """Process that encodes mixed CPU/CUDA tensors."""
    try:
        encoder = MsgpackEncoder(tensor_queues=tensor_queues)
        encoder.set_target_engine(0)

        # Create mixed tensors
        data = {
            "cpu_tensor": torch.randn(2, 3),  # CPU
            "cuda_tensor": torch.randn(4, 5, device="cuda:0"),  # CUDA
            "scalar": 42,
            "list": [1, 2, 3],
        }

        # Encode
        encoded = encoder.encode(data)
        
        ready_event.set()
        
        result_queue.put({
            "success": True,
            "encoded_length": len(encoded),
        })
    except Exception as e:
        import traceback
        ready_event.set()
        result_queue.put({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def mixed_tensor_decoder_process(
    tensor_queue: torch_mp.Queue,
    result_queue: mp.Queue,
    encoder_ready: mp.Event,
):
    """Process that retrieves mixed tensors from queue."""
    try:
        # Wait for encoder to finish
        if not encoder_ready.wait(timeout=10.0):
            raise TimeoutError("Encoder did not signal ready")

        # Try to get CUDA tensor from queue
        ipc_data = tensor_queue.get(timeout=5.0)
        
        result_queue.put({
            "success": True,
            "is_cuda": ipc_data.tensor.is_cuda,
            "shape": tuple(ipc_data.tensor.shape),
        })
    except Exception as e:
        import traceback
        result_queue.put({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_cpu_cuda_tensors():
    """Test encoding with mixed CPU and CUDA tensors using multiprocessing."""
    tensor_queues = [torch_mp.Queue()]
    result_queue = mp.Queue()
    encoder_ready = mp.Event()

    # Start encoder process
    encoder_proc = mp.Process(
        target=mixed_tensor_encoder_process,
        args=(tensor_queues, result_queue, encoder_ready),
    )
    encoder_proc.start()

    # Start decoder process
    decoder_proc = mp.Process(
        target=mixed_tensor_decoder_process,
        args=(tensor_queues[0], result_queue, encoder_ready),
    )
    decoder_proc.start()

    # Get results
    encoder_result = result_queue.get(timeout=10.0)
    decoder_result = result_queue.get(timeout=10.0)

    encoder_proc.join(timeout=5.0)
    decoder_proc.join(timeout=5.0)

    # Verify encoder succeeded
    assert encoder_result["success"], f"Encoder failed: {encoder_result.get('error')}\n{encoder_result.get('traceback', '')}"
    
    # Verify decoder succeeded and got CUDA tensor
    assert decoder_result["success"], f"Decoder failed: {decoder_result.get('error')}\n{decoder_result.get('traceback', '')}"
    assert decoder_result["is_cuda"], "Retrieved tensor is not on CUDA"
    assert decoder_result["shape"] == (4, 5), f"Unexpected shape: {decoder_result['shape']}"


if __name__ == "__main__":
    # Run basic tests
    print("Running CPU tensor fallback test...")
    test_cpu_tensor_fallback()
    print("✓ CPU tensor fallback test passed")

    print("\nRunning encoder without target engine test...")
    test_encoder_without_target_engine()
    print("✓ Encoder without target engine test passed")

    print("\nRunning decoder buffer management test...")
    test_decoder_buffer_management()
    print("✓ Decoder buffer management test passed")

    print("\nRunning multiple API servers test...")
    test_multiple_api_servers_to_engine()
    print("✓ Multiple API servers test passed")

    if torch.cuda.is_available():
        print("\nRunning CUDA tensor queue basic test...")
        test_cuda_tensor_queue_basic()
        print("✓ CUDA tensor queue basic test passed")

        print("\nRunning mixed CPU/CUDA tensors test...")
        test_mixed_cpu_cuda_tensors()
        print("✓ Mixed CPU/CUDA tensors test passed")
    else:
        print("\nSkipping CUDA tests (CUDA not available)")

    print("\n✅ All tests passed!")

