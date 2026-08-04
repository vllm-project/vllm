# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Generator
import multiprocessing as mp
import os
import time

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_response_class,
)
from lmcache.v1.multiprocess.server import run_cache_server

# Configuration constants
SERVER_HOST = "localhost"
SERVER_PORT = 5599
SERVER_URL = f"tcp://{SERVER_HOST}:{SERVER_PORT}"
CHUNK_SIZE = 256
CPU_BUFFER_SIZE = 5.0
DEFAULT_TIMEOUT = 20.0
pytestmark = pytest.mark.cuda


def _has_working_new_shared_cuda() -> bool:
    try:
        buf = torch.empty(1024, device=torch_device_type)
        shared = buf.untyped_storage()._share_cuda_()
        return shared is not None
    except Exception:
        return False


if not (torch_dev.is_available() and torch_device_type == "cuda"):
    pytest.skip(
        "requires available CUDA runtime",
        allow_module_level=True,
    )

# First Party
from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper  # noqa: E402

if not _has_working_new_shared_cuda():
    pytest.skip(
        ("new_shared_cuda is not available or not working on this system"),
        allow_module_level=True,
    )


def initialize_kv_cache(
    device: torch.device,
    num_pages: int = 1024,
    num_layers: int = 32,
    page_size: int = 16,
    num_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> list[torch.Tensor]:
    """
    Initialize KV cache tensors on GPU for testing.
    """
    torch.random.manual_seed(42)

    gpu_tensors = [
        torch.rand(
            (2, num_pages, page_size, num_heads, head_size),
            dtype=dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]

    return gpu_tensors


class ClientContext:
    """
    Client context that manages GPU KV cache tensors.
    """

    def __init__(
        self,
        device: torch.device,
        num_pages: int = 1024,
        num_layers: int = 32,
        page_size: int = 16,
        num_heads: int = 8,
        head_size: int = 128,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.num_pages = num_pages
        self.num_layers = num_layers
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype

        self.gpu_kv_caches = initialize_kv_cache(
            device, num_pages, num_layers, page_size, num_heads, head_size, dtype
        )

    def get_kv_cache(self) -> KVCache:
        """
        Wrap GPU tensors in CudaIPCWrapper for IPC communication.
        """
        return [CudaIPCWrapper(tensor) for tensor in self.gpu_kv_caches]

    def get_tensor_slice(
        self, layer: int, start_page: int, num_pages: int
    ) -> torch.Tensor:
        """
        Get a slice of the KV cache tensor for a specific layer.
        """
        return self.gpu_kv_caches[layer][:, start_page : start_page + num_pages]


def create_cache_key(index: int, model: str = "testmodel") -> IPCCacheServerKey:
    """
    Create a cache key for testing.
    """
    global CHUNK_SIZE
    token_ids = [index] * CHUNK_SIZE
    return IPCCacheServerKey.from_token_ids(
        model,
        1,
        0,
        token_ids,
        start=0,
        end=CHUNK_SIZE,
        request_id=f"test_request_{index}",
    )


BLOCKS_PER_KEY = 16


def lookup_all(
    client: MessageQueueClient,
    keys: list[IPCCacheServerKey],
    timeout: float = DEFAULT_TIMEOUT,
) -> int:
    """Lookup all keys individually and return total found count.

    Uses the two-phase lookup protocol: LOOKUP registers the job server-side,
    then QUERY_PREFETCH_STATUS is polled by request_id until the result is ready.
    """
    total = 0
    for key in keys:
        lookup_key = key.no_worker_id_version()
        # Phase 1: Submit lookup (server tracks by request_id, returns None)
        client.submit_request(
            RequestType.LOOKUP,
            [lookup_key, 1],
            get_response_class(RequestType.LOOKUP),
        ).result(timeout=timeout)
        # Phase 2: Poll by request_id until done
        while True:
            result = client.submit_request(
                RequestType.QUERY_PREFETCH_STATUS,
                [lookup_key.request_id],
                get_response_class(RequestType.QUERY_PREFETCH_STATUS),
            ).result(timeout=timeout)
            if result is not None:
                total += result
                break
    return total


def store_keys(
    client: MessageQueueClient,
    keys: list[IPCCacheServerKey],
    instance_id: int,
    gpu_block_ids: list[int],
    event: Any,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """Store keys one at a time using the single-key API."""
    for i, key in enumerate(keys):
        start = i * BLOCKS_PER_KEY
        end = start + BLOCKS_PER_KEY
        block_ids = gpu_block_ids[start:end]
        future = client.submit_request(
            RequestType.STORE,
            [key, instance_id, [block_ids], event.ipc_handle()],
            get_response_class(RequestType.STORE),
        )
        result = future.to_device_future().result(timeout=timeout)
        assert result is True, f"Store should succeed for key {i}"


def retrieve_keys(
    client: MessageQueueClient,
    keys: list[IPCCacheServerKey],
    instance_id: int,
    gpu_block_ids: list[int],
    event: Any,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[bool]:
    """Retrieve keys one at a time using the single-key API."""
    results = []
    for i, key in enumerate(keys):
        start = i * BLOCKS_PER_KEY
        end = start + BLOCKS_PER_KEY
        block_ids = gpu_block_ids[start:end]
        future = client.submit_request(
            RequestType.RETRIEVE,
            [key, instance_id, [block_ids], event.ipc_handle(), 0],
            get_response_class(RequestType.RETRIEVE),
        )
        result = future.to_device_future().result(timeout=timeout)
        results.append(result)
    return results


def server_process_runner(
    host: str, port: int, chunk_size: int, cpu_buffer_size: float
):
    """
    Entry point for the server process.
    """
    mp_config = MPServerConfig(host=host, port=port, chunk_size=chunk_size)
    storage_manager_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=int(cpu_buffer_size * 1024**3),
                use_lazy=True,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
    )


@pytest.fixture(scope="module")
def server_process() -> Generator[mp.Process, None, None]:
    """
    Fixture that starts the cache server in a separate process.
    The server runs for the entire test module.
    """
    # Start server process
    mp.set_start_method("spawn", force=True)
    process = mp.Process(
        target=server_process_runner,
        args=(SERVER_HOST, SERVER_PORT, CHUNK_SIZE, CPU_BUFFER_SIZE),
        daemon=True,
    )
    process.start()

    # Wait for server to initialize
    time.sleep(2)

    yield process

    # Cleanup: terminate the server process
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()


@pytest.fixture(scope="module")
def zmq_context() -> Generator[zmq.Context, None, None]:
    """
    Fixture that provides a ZMQ context for the test module.
    """
    context = zmq.Context.instance()
    yield context
    # Context cleanup is handled by ZMQ


@pytest.fixture(scope="function")
def client(
    server_process: mp.Process, zmq_context: zmq.Context
) -> Generator[MessageQueueClient, None, None]:
    """
    Fixture that provides a message queue client for each test function.
    """
    client = MessageQueueClient(server_url=SERVER_URL, context=zmq_context)
    yield client
    # Client cleanup
    client.close()


@pytest.fixture(scope="function")
def client_context() -> Generator[ClientContext, None, None]:
    """
    Fixture that provides a client context with initialized KV cache.
    """
    device = torch.device(torch_device_type)
    ctx = ClientContext(device=device)
    yield ctx

    # Cleanup GPU memory
    del ctx.gpu_kv_caches
    torch_dev.empty_cache()


@pytest.fixture(scope="function")
def registered_instance(
    client: MessageQueueClient, client_context: ClientContext
) -> Generator[int, None, None]:
    """
    Fixture that registers a KV cache instance and returns the instance ID.
    Automatically unregisters after the test.
    """
    instance_id = os.getpid()

    # Register KV cache. No engine group infos are sent, so the server
    # detects ``slots_per_block`` from the tensors and treats every group
    # as uncompressed (``compress_ratio == 1``).
    future = client.submit_request(
        RequestType.REGISTER_KV_CACHE,
        [
            instance_id,
            client_context.get_kv_cache(),
            "testmodel",
            1,
            EngineType.VLLM,
            {},
            [],
        ],
        get_response_class(RequestType.REGISTER_KV_CACHE),
    )
    result = future.result(timeout=DEFAULT_TIMEOUT)
    assert result is None, "Register should return None"

    yield instance_id

    # Unregister KV cache
    try:
        client.submit_request(
            RequestType.CLEAR, [], get_response_class(RequestType.CLEAR)
        ).result(timeout=DEFAULT_TIMEOUT)
        future = client.submit_request(
            RequestType.UNREGISTER_KV_CACHE,
            [instance_id],
            get_response_class(RequestType.UNREGISTER_KV_CACHE),
        )
        future.result(timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        print(f"Error during unregister: {e}")


# ============================================================================
# Test Functions
# ============================================================================


def test_server_running(server_process: mp.Process):
    """
    Test that the server process is running.
    """
    assert server_process.is_alive(), "Server process should be running"


def test_register_unregister_kv_cache(
    client: MessageQueueClient, client_context: ClientContext
):
    """
    Test registering and unregistering a KV cache.
    """
    instance_id = os.getpid()

    # Register. No engine group infos: geometry is detected from the
    # tensors (uncompressed).
    future = client.submit_request(
        RequestType.REGISTER_KV_CACHE,
        [
            instance_id,
            client_context.get_kv_cache(),
            "testmodel",
            1,
            EngineType.VLLM,
            {},
            [],
        ],
        get_response_class(RequestType.REGISTER_KV_CACHE),
    )
    result = future.result(timeout=DEFAULT_TIMEOUT)
    assert result is None

    # Unregister
    future = client.submit_request(
        RequestType.UNREGISTER_KV_CACHE,
        [instance_id],
        get_response_class(RequestType.UNREGISTER_KV_CACHE),
    )
    result = future.result(timeout=DEFAULT_TIMEOUT)
    assert result is None


def test_store_and_lookup(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """
    Test storing KV cache entries and looking them up.
    """
    num_keys = 10
    keys = [create_cache_key(i) for i in range(num_keys)]
    gpu_block_ids = list(range(0, 16 * num_keys))
    event = torch_dev.Event(interprocess=True)
    event.record()

    # Store
    store_keys(client, keys, registered_instance, gpu_block_ids, event)

    # Lookup - keys that exist
    lookup_result = lookup_all(client, keys)
    assert lookup_result == num_keys, "All stored keys should exist"

    # Lookup - keys that don't exist
    non_existent_keys = [create_cache_key(i + 1000) for i in range(5)]
    lookup_result2 = lookup_all(client, non_existent_keys)
    assert lookup_result2 == 0, "Non-existent keys should not be found"


def test_store_fails_closed_on_incomplete_block_ids(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """An under-length block-id list skips the whole store (fail-closed).

    Regression guard for the all-or-nothing store contract: a ``gpu_block_ids``
    list too short to fully cover a chunk (e.g. a caller/protocol bug) must skip
    the store entirely (returning ``False``) and commit nothing — the previous
    fail-open path raised internally but then ``finish_write``-committed the
    reservation anyway, turning the key into a retrievable garbage entry (lookup
    would find it).

    The committed-state assertion is on a *miss* (lookup == 0), which is robust
    to this harness's known store->lookup race (that race can only turn a true
    hit into a miss, never the reverse).
    """
    # One-chunk key (256 tokens == BLOCKS_PER_KEY blocks) but only half the
    # block IDs needed, so the chunk is not fully covered.
    key = create_cache_key(90001)
    event = torch_dev.Event(interprocess=True)
    event.record()

    result = (
        client.submit_request(
            RequestType.STORE,
            [
                key,
                registered_instance,
                [list(range(BLOCKS_PER_KEY // 2))],
                event.ipc_handle(),
            ],
            get_response_class(RequestType.STORE),
        )
        .to_device_future()
        .result(timeout=DEFAULT_TIMEOUT)
    )
    assert result is False, "Store should fail closed (skip) on a short list"
    assert lookup_all(client, [key]) == 0, "An uncovered chunk must not be committed"


def test_store_retrieve_verify(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """
    Test storing and retrieving KV cache entries, verifying correctness.
    """
    num_keys = 20
    keys = [create_cache_key(i) for i in range(num_keys)]
    event = torch_dev.Event(interprocess=True)
    event.record()

    # Store at the beginning of the cache
    store_block_ids = list(range(0, 16 * num_keys))
    store_keys(client, keys, registered_instance, store_block_ids, event)

    event = torch_dev.Event(interprocess=True)
    event.record()

    # Call look up to ensure the data is ready to be retrieved
    lookup_result = lookup_all(client, keys)
    assert lookup_result == num_keys

    # Retrieve to a different location in the cache
    # Use offset of 40 blocks (640 pages total needed: 320 + 320)
    retrieve_offset = 40 * 16
    retrieve_block_ids = list(range(retrieve_offset, retrieve_offset + 16 * num_keys))
    retrieve_result = retrieve_keys(
        client, keys, registered_instance, retrieve_block_ids, event
    )

    assert len(retrieve_result) == num_keys
    assert all(retrieve_result), "All keys should be retrieved successfully"

    # Verify correctness by comparing tensors
    for i in range(num_keys):
        for layer in range(client_context.num_layers):
            original_block = i * 16
            retrieved_block = retrieve_offset + i * 16

            original_tensor = client_context.gpu_kv_caches[layer][
                :, original_block : original_block + 16
            ]
            retrieved_tensor = client_context.gpu_kv_caches[layer][
                :, retrieved_block : retrieved_block + 16
            ]

            assert torch.allclose(original_tensor, retrieved_tensor, atol=1e-4), (
                f"Mismatch for key {i}, layer {layer}"
            )


def test_retrieve_partial_miss(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """
    Test retrieving when some keys exist and some don't.
    The retrieve should return ALL FALSE if any key is missing.
    """
    # Store first 30 keys (480 pages)
    num_stored = 30
    stored_keys = [create_cache_key(i) for i in range(num_stored)]
    store_block_ids = list(range(0, 16 * num_stored))
    event = torch_dev.Event(interprocess=True)
    event.record()

    store_keys(client, stored_keys, registered_instance, store_block_ids, event)

    # Lookup to ensure keys are stored
    lookup_result = lookup_all(client, stored_keys)
    assert lookup_result == num_stored

    # Try to retrieve 60 keys (only first 30 exist)
    # Total pages needed: 60 * 16 = 960 (< 1024)
    num_requested = 60
    all_keys = [create_cache_key(i) for i in range(num_requested)]
    # Start retrieve at offset 2 keys (32 pages)
    retrieve_offset_keys = 2
    retrieve_block_ids = list(
        range(retrieve_offset_keys * 16, (retrieve_offset_keys + num_requested) * 16)
    )

    event = torch_dev.Event(interprocess=True)
    event.record()

    retrieve_result = retrieve_keys(
        client, all_keys, registered_instance, retrieve_block_ids, event
    )

    assert len(retrieve_result) == num_requested
    # First 30 keys exist, remaining 30 don't
    assert all(retrieve_result[:num_stored]), "Stored keys should be retrieved"
    assert not any(retrieve_result[num_stored:]), (
        "Non-existent keys should fail to retrieve"
    )

    # Doing look up again to ensure data is ready
    lookup_result_2 = lookup_all(client, stored_keys)
    assert lookup_result_2 == num_stored

    # Try to retrieve the first 30 keys only (all exist)
    retrieve_block_ids_2 = list(range(0, 16 * num_stored))
    event = torch_dev.Event(interprocess=True)
    event.record()
    retrieve_result_2 = retrieve_keys(
        client, stored_keys, registered_instance, retrieve_block_ids_2, event
    )
    assert len(retrieve_result_2) == num_stored
    assert all(retrieve_result_2), "All stored keys should be retrieved successfully"


def test_multiple_retrieve_operations(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """
    Test multiple retrieve operations:
    Store 8,8,8,8 keys and then retrieve 8,8,8,8 keys in sequence.
    """
    num_batches = 4
    keys_per_batch = 8
    pages_per_key = 16

    # Initialize the values in GPU KV cache
    for layer in range(client_context.num_layers):
        layer_cache = client_context.gpu_kv_caches[layer]
        for i in range(num_batches):
            start_page = (i * keys_per_batch) * pages_per_key
            end_page = start_page + (keys_per_batch * pages_per_key)
            layer_cache[:, start_page:end_page] = (i + 1) / num_batches

    # Store in batches
    for batch_idx in range(num_batches):
        keys = [
            create_cache_key(batch_idx * keys_per_batch + i)
            for i in range(keys_per_batch)
        ]
        blocks = list(
            range(
                (batch_idx * keys_per_batch) * 16,
                (batch_idx * keys_per_batch + keys_per_batch) * 16,
            )
        )
        event = torch_dev.Event(interprocess=True)
        event.record()
        store_keys(client, keys, registered_instance, blocks, event)

    # Doing look up to ensure data is ready to be retrieved
    all_keys = [
        create_cache_key(batch_idx * keys_per_batch + i)
        for batch_idx in range(num_batches)
        for i in range(keys_per_batch)
    ]
    lookup_result = lookup_all(client, all_keys)
    assert lookup_result == num_batches * keys_per_batch, "All stored keys should exist"

    # Retrieve in batches
    retrieve_offset = 32  # Start retrieving at offset of 32 chunks
    event = torch_dev.Event(interprocess=True)
    event.record()
    for batch_idx in range(num_batches):
        keys = [
            create_cache_key(batch_idx * keys_per_batch + i)
            for i in range(keys_per_batch)
        ]
        blocks = list(
            range(
                (batch_idx * keys_per_batch + retrieve_offset) * pages_per_key,
                (batch_idx * keys_per_batch + retrieve_offset + keys_per_batch)
                * pages_per_key,
            )
        )

        retrieve_result = retrieve_keys(
            client, keys, registered_instance, blocks, event
        )
        assert len(retrieve_result) == keys_per_batch
        assert all(retrieve_result), "All keys should be retrieved successfully"

    # Verify correctness
    for layer in range(client_context.num_layers):
        layer_cache = client_context.gpu_kv_caches[layer]
        for batch_idx in range(num_batches):
            start_page = (retrieve_offset + batch_idx * keys_per_batch) * pages_per_key
            end_page = start_page + (keys_per_batch * pages_per_key)
            retrieved_tensor = layer_cache[:, start_page:end_page]
            expected_value = (batch_idx + 1) / num_batches
            assert torch.allclose(
                retrieved_tensor,
                torch.full_like(retrieved_tensor, expected_value),
            ), f"Mismatch in batch {batch_idx}, layer {layer}"


def test_multiple_store_operations(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """
    Test multiple store operations in sequence.
    """
    # Store batch 1
    keys1 = [create_cache_key(i) for i in range(30)]
    blocks1 = list(range(0, 16 * 30))
    event = torch_dev.Event(interprocess=True)
    event.record()
    store_keys(client, keys1, registered_instance, blocks1, event)

    # Store batch 2
    keys2 = [create_cache_key(i + 30) for i in range(20)]
    blocks2 = list(range(30 * 16, 50 * 16))

    # Test with the same event for 2 store requests
    store_keys(client, keys2, registered_instance, blocks2, event)

    # Verify all keys exist
    all_keys = keys1 + keys2
    lookup_result = lookup_all(client, all_keys)
    assert lookup_result == 50, "All stored keys from both batches should exist"


def test_get_chunk_size(
    client: MessageQueueClient,
):
    """
    Test retrieving the chunk size from the server.
    """
    chunk_size = client.submit_request(
        RequestType.GET_CHUNK_SIZE,
        [],
        get_response_class(RequestType.GET_CHUNK_SIZE),
    ).result(timeout=DEFAULT_TIMEOUT)

    assert chunk_size == CHUNK_SIZE, f"Chunk size should be {CHUNK_SIZE}"
