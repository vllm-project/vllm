# SPDX-License-Identifier: Apache-2.0
# Standard
from multiprocessing import Queue
from typing import Any
import multiprocessing as mp

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    IPCCacheServerKey,
    get_customized_decoder,
    get_customized_encoder,
)


def _get_cuda_ipc_wrapper():
    # First Party
    from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper

    return CudaIPCWrapper


def test_ipc_cache_engine_key_serialization():
    """Test encoding and decoding of IPCCacheServerKey using msgspec."""
    # Create a sample IPCCacheServerKey
    original_key = IPCCacheServerKey.from_token_ids(
        model_name="test_model",
        world_size=4,
        worker_id=1,
        token_ids=list(range(256)),
        start=0,
        end=256,
        request_id="test_request",
    )

    # Encode the key
    encoded = msgspec.msgpack.encode(original_key)

    # Decode the key
    decoded_key = msgspec.msgpack.decode(encoded, type=IPCCacheServerKey)

    # Verify correctness
    assert original_key == decoded_key, "IPCCacheServerKeys do not match!"


def test_ipc_cache_engine_key_serialization_with_cache_salt():
    """Roundtrip must carry ``cache_salt`` verbatim — it is part of
    cache identity so eq must hold after encode/decode."""
    original_key = IPCCacheServerKey.from_token_ids(
        model_name="test_model",
        world_size=4,
        worker_id=1,
        token_ids=list(range(256)),
        start=0,
        end=256,
        request_id="test_request",
        cache_salt="alice",
    )

    encoded = msgspec.msgpack.encode(original_key)
    decoded_key = msgspec.msgpack.decode(encoded, type=IPCCacheServerKey)

    assert original_key == decoded_key
    assert decoded_key.cache_salt == "alice"


@pytest.mark.cuda
@pytest.mark.skipif(
    not (torch_dev.is_available() and torch_device_type == "cuda"),
    reason="requires available CUDA runtime",
)
def test_cudaipc_wrapper_serialization():
    """Test custom encoder/decoder for single CudaIPCWrapper object."""
    CudaIPCWrapper = _get_cuda_ipc_wrapper()
    encoder = get_customized_encoder(type=CudaIPCWrapper)
    decoder = get_customized_decoder(type=CudaIPCWrapper)

    # Create a sample tensor
    original_tensor = torch.randn(3, 4, device=torch_device_type)
    wrapper = CudaIPCWrapper(original_tensor)

    # Encode the wrapper
    encoded = encoder.encode(wrapper)

    # Decode the wrapper
    decoded_wrapper = decoder.decode(encoded)
    assert isinstance(decoded_wrapper, CudaIPCWrapper), (
        "Decoded object is not of type CudaIPCWrapper"
    )
    assert decoded_wrapper == wrapper, (
        "Decoded CudaIPCWrapper does not match the original"
    )


@pytest.mark.cuda
@pytest.mark.skipif(
    not (torch_dev.is_available() and torch_device_type == "cuda"),
    reason="requires available CUDA runtime",
)
def test_cudaipc_wrapper_list_serialization():
    """Test custom encoder/decoder for list of CudaIPCWrapper objects."""
    CudaIPCWrapper = _get_cuda_ipc_wrapper()
    wrappers = []
    for _ in range(5):
        tensor = torch.randn(2, 2, device=torch_device_type)
        wrapper = CudaIPCWrapper(tensor)
        wrappers.append(wrapper)

    encoder = get_customized_encoder(type=list[CudaIPCWrapper])
    decoder = get_customized_decoder(type=list[CudaIPCWrapper])

    # Encode the list of wrappers
    encoded = encoder.encode(wrappers)

    # Decode the list of wrappers
    decoded_wrappers = decoder.decode(encoded)

    assert len(decoded_wrappers) == len(wrappers), (
        "Decoded list length does not match original"
    )

    for original, decoded in zip(wrappers, decoded_wrappers, strict=False):
        assert original == decoded, "Decoded CudaIPCWrapper does not match the original"


def _worker_process_deserialize_and_reconstruct(
    encoded_data: bytes, result_queue: Queue
):
    """
    Worker function that runs in a separate process.
    Deserializes CudaIPCWrapper list and reconstructs tensors.
    """
    try:
        # Decode the list of wrappers
        torch_dev.init()
        decoder = get_customized_decoder(type=list[Any])
        decoded_wrappers = decoder.decode(encoded_data)

        # Convert each wrapper back to tensor and compute checksum
        checksums = []
        shapes = []
        for wrapper in decoded_wrappers:
            tensor = wrapper.to_tensor()
            # Compute checksum as sum of all elements
            checksum = float(tensor.sum().cpu().item())
            checksums.append(checksum)
            shapes.append(list(tensor.shape))

            # Do add 1 on the tensor to ensure it's writable
            tensor.add_(1)

        result_queue.put(("success", checksums, shapes))
    except Exception as e:
        result_queue.put(("error", str(e), None))


@pytest.mark.cuda
@pytest.mark.skipif(
    not (torch_dev.is_available() and torch_device_type == "cuda"),
    reason="requires available CUDA runtime",
)
def test_cudaipc_wrapper_multiprocess_serialization():
    """
    Test CudaIPCWrapper serialization across processes using spawn method.
    This verifies that CUDA IPC handles can be properly shared between processes.
    """
    # Set multiprocessing start method to spawn
    CudaIPCWrapper = _get_cuda_ipc_wrapper()
    ctx = mp.get_context("spawn")

    # Create test tensors and wrappers in the main process
    num_tensors = 3
    tensors = []
    test_data = []
    wrappers = []

    for i in range(num_tensors):
        # Create a tensor with known values
        tensor = torch.full(
            (2, 3),
            fill_value=float(i + 1),
            dtype=torch.float32,
            device=torch_device_type,
        )
        tensors.append(tensor)
        wrapper = CudaIPCWrapper(tensor)
        wrappers.append(wrapper)

        # Store expected checksum and shape
        expected_checksum = float(tensor.sum().cpu().item())
        expected_shape = list(tensor.shape)
        test_data.append((expected_checksum, expected_shape))

    # Serialize the wrappers
    encoder = get_customized_encoder(type=list[CudaIPCWrapper])
    encoded_data = encoder.encode(wrappers)

    # Create a queue for results
    result_queue = ctx.Queue()

    # Start worker process
    process = ctx.Process(
        target=_worker_process_deserialize_and_reconstruct,
        args=(encoded_data, result_queue),
    )
    process.start()

    # Wait for result with timeout
    process.join(timeout=10)

    # Check if process completed successfully
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("Worker process timed out")

    assert process.exitcode == 0, (
        f"Worker process failed with exit code {process.exitcode}"
    )

    # Get result from queue
    assert not result_queue.empty(), "No result received from worker process"
    status, checksums, shapes = result_queue.get()

    assert status == "success", f"Worker process encountered error: {checksums}"
    assert len(checksums) == num_tensors, "Number of checksums does not match"
    assert len(shapes) == num_tensors, "Number of shapes does not match"

    # Verify checksums and shapes match
    for i, (
        (expected_checksum, expected_shape),
        actual_checksum,
        actual_shape,
    ) in enumerate(zip(test_data, checksums, shapes, strict=False)):
        assert actual_shape == expected_shape, (
            f"Tensor {i}: shape mismatch. Expected {expected_shape}, got {actual_shape}"
        )
        assert abs(actual_checksum - expected_checksum) < 1e-5, (
            f"Tensor {i}: checksum mismatch. Expected {expected_checksum}, "
            f"got {actual_checksum}"
        )

    # Verify that the tensors are being modified in the worker process
    for i, (tensor, (expected_checksum, _)) in enumerate(
        zip(tensors, test_data, strict=False)
    ):
        # After adding 1 to each element, the new checksum should be:
        num_elements = tensor.numel()
        new_expected_checksum = expected_checksum + float(num_elements)
        actual_checksum = float(tensor.sum().cpu().item())
        assert abs(actual_checksum - new_expected_checksum) < 1e-5, (
            f"Tensor {i}: post-modification checksum mismatch. "
            f"Expected {new_expected_checksum}, got {actual_checksum}"
        )


def _worker_reconstruct_offset_tensor(encoded_data: bytes, result_queue: Queue):
    """Worker: decode a single CudaIPCWrapper and reconstruct its tensor,
    reporting the layout metadata and a checksum back to the parent."""
    try:
        CudaIPCWrapper = _get_cuda_ipc_wrapper()
        torch_dev.init()
        decoder = get_customized_decoder(type=CudaIPCWrapper)
        wrapper = decoder.decode(encoded_data)
        tensor = wrapper.to_tensor()
        result_queue.put(
            (
                "success",
                int(tensor.storage_offset()),
                list(tensor.shape),
                list(tensor.stride()),
                float(tensor.sum().cpu().item()),
            )
        )
    except Exception as e:
        result_queue.put(("error", str(e), None, None, None))


@pytest.mark.cuda
@pytest.mark.skipif(
    not (torch_dev.is_available() and torch_device_type == "cuda"),
    reason="requires available CUDA runtime",
)
def test_cudaipc_wrapper_nonzero_storage_offset():
    """CudaIPCWrapper must round-trip a slice/narrow view with
    ``storage_offset > 0`` bit-identically across processes.

    This is the property PR #3853 relies on: with MTP speculative decoding +
    CPU offload, per-layer KV are non-zero-``storage_offset`` slices of a
    unified pool, so ``_validate_dim0_padded_layout`` must accept them. The
    view here is both dim-0-padded (``stride[0] > prod(shape[1:])``) and
    offset-shifted, exactly that shape. ``CudaIPCWrapper`` encodes
    ``storage_offset`` and the receiver rebuilds the view via
    ``set_(storage, storage_offset, shape, stride)``; this verifies the
    reconstructed tensor reads from the correct (offset, strided) region.
    """
    CudaIPCWrapper = _get_cuda_ipc_wrapper()
    ctx = mp.get_context("spawn")

    # arange so each element's value equals its flat storage index -- the
    # checksum then pins down exactly which storage positions were read.
    base = torch.arange(64, dtype=torch.float32, device=torch_device_type)
    # dim-0-padded view: shape (3, 2, 4), per-block stride 12 > prod(shape[1:])=8
    # (4 elements of padding per block), shifted by storage_offset=8.
    view = base.as_strided((3, 2, 4), (12, 4, 1), storage_offset=8)
    assert view.storage_offset() == 8
    assert not view.is_contiguous()

    wrapper = CudaIPCWrapper(view)
    assert wrapper.storage_offset == view.storage_offset()
    assert wrapper.shape == tuple(view.shape)
    assert wrapper.stride == tuple(view.stride())

    encoder = get_customized_encoder(type=CudaIPCWrapper)
    encoded = encoder.encode(wrapper)

    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_worker_reconstruct_offset_tensor,
        args=(encoded, result_queue),
    )
    process.start()
    process.join(timeout=10)

    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("Worker process timed out")
    assert process.exitcode == 0, (
        f"Worker process failed with exit code {process.exitcode}"
    )
    assert not result_queue.empty(), "No result received from worker process"

    status, offset, shape, stride, checksum = result_queue.get()
    assert status == "success", f"Worker process encountered error: {offset}"
    assert offset == view.storage_offset()
    assert shape == list(view.shape)
    assert stride == list(view.stride())
    assert abs(checksum - float(view.sum().cpu().item())) < 1e-5


def test_block_allocation_record_serialization():
    """Test encoding and decoding of BlockAllocationRecord using msgspec."""
    original = BlockAllocationRecord(
        req_id="req-42",
        new_block_ids=[10, 20, 30],
        new_token_ids=[100, 200, 300, 400],
    )

    encoded = msgspec.msgpack.encode(original)
    decoded = msgspec.msgpack.decode(encoded, type=BlockAllocationRecord)

    assert decoded.req_id == original.req_id
    assert decoded.new_block_ids == original.new_block_ids
    assert decoded.new_token_ids == original.new_token_ids


def test_block_allocation_record_list_serialization():
    """Test encoding and decoding of a list of BlockAllocationRecord."""
    records = [
        BlockAllocationRecord(
            req_id="req-1",
            new_block_ids=[1, 2],
            new_token_ids=[10, 20, 30],
        ),
        BlockAllocationRecord(
            req_id="req-2",
            new_block_ids=[],
            new_token_ids=[40, 50],
        ),
    ]

    encoded = msgspec.msgpack.encode(records)
    decoded = msgspec.msgpack.decode(encoded, type=list[BlockAllocationRecord])

    assert len(decoded) == 2
    assert decoded[0].req_id == "req-1"
    assert decoded[0].new_block_ids == [1, 2]
    assert decoded[1].req_id == "req-2"
    assert decoded[1].new_block_ids == []
    assert decoded[1].new_token_ids == [40, 50]
