# SPDX-License-Identifier: Apache-2.0
# Standard
from multiprocessing.synchronize import Event as EventClass
from typing import Any, Callable
import multiprocessing as mp
import sys
import time

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.mq import (
    BlockingRequestHandler,
    MessageQueueClient,
    MessageQueueServer,
)
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)
from lmcache.v1.multiprocess.server import add_handler_helper

# Test helpers
from tests.v1.multiprocess import test_mq_handler_helpers

# ==============================================================================
# MessageQueueServer and MessageQueueClient Tests Infrastructure
# ==============================================================================


def create_cache_key(index: int, model: str = "testmodel") -> IPCCacheServerKey:
    """
    Create a cache key for testing.
    """
    chunk_size = 256
    token_ids = [index] * chunk_size
    return IPCCacheServerKey.from_token_ids(
        model,
        1,
        0,
        token_ids,
        start=0,
        end=chunk_size,
        request_id=f"test_request_{index}",
    )


def _server_process(
    server_url: str,
    ready_event: EventClass,
    shutdown_event: EventClass,
    request_handlers: dict[RequestType, Callable],
):
    """
    Server process that runs the MessageQueueServer.

    Args:
        server_url: URL to bind the server to
        ready_event: Event to signal when server is ready
        shutdown_event: Event to signal server shutdown
        request_handlers: Dict mapping RequestType to handler functions
    """
    # First Party
    from lmcache.v1.multiprocess.protocol import HandlerType

    context = zmq.Context.instance()
    server = MessageQueueServer(server_url, context)

    # Register all handlers
    blocking_types: list[RequestType] = []
    for request_type, handler in request_handlers.items():
        payload_classes = get_payload_classes(request_type)
        handler_type = get_handler_type(request_type)
        server.add_handler(request_type, payload_classes, handler_type, handler)
        if handler_type == HandlerType.BLOCKING:
            blocking_types.append(request_type)

    # Assign a normal pool for all blocking handlers in tests
    if blocking_types:
        server.add_normal_thread_pool(blocking_types, max_workers=4)

    server.start()

    # Signal that server is ready
    ready_event.set()

    # Wait for shutdown signal
    shutdown_event.wait()

    # Cleanup
    server.close()


def _run_client_test(
    server_url: str,
    ready_event: EventClass,
    request_type: RequestType,
    payloads: list[Any],
    expected_response: Any,
    num_requests: int = 1,
    client_id: int = 0,
) -> None:
    """
    Client process that sends requests and validates responses.

    Args:
        server_url: URL to connect to
        ready_event: Event to wait for server to be ready
        request_type: Type of request to send
        payloads: List of payloads for the request
        expected_response: Expected response from server
        num_requests: Number of requests to send
        client_id: ID of this client (for debugging)

    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Wait for server to be ready
    if not ready_event.wait(timeout=5):
        print(f"Client {client_id}: Server failed to start within timeout")
        sys.exit(1)

    # Small delay to ensure server is fully initialized
    time.sleep(0.1)

    context = zmq.Context.instance()
    client = MessageQueueClient(server_url, context)
    successful = True

    try:
        futures = []
        # Submit requests
        for _ in range(num_requests):
            future = client.submit_request(request_type, payloads)  # type: ignore
            futures.append(future)

        # Validate responses
        for i, future in enumerate(futures):
            response = future.result(timeout=5)
            if response != expected_response:
                print(
                    f"Client {client_id}, Request {i}: Expected "
                    f"{expected_response}, got {response}"
                )

                # Exit with error code
                client.close()
                sys.exit(1)

    except Exception as e:
        print(f"Client {client_id} test failed with exception: {e}")
        successful = False
    finally:
        client.close()
        if not successful:
            sys.exit(1)


class MessageQueueTestHelper:
    """
    Helper class to facilitate testing MessageQueueServer and MessageQueueClient.

    Supports testing with single or multiple concurrent clients, where each client
    can send multiple requests to the server.

    Usage:
        1. Create an instance with server URL
        2. Register handlers for different RequestTypes
        3. Call run_test() to execute the test with client requests

    Example:
        helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5556")
        helper.register_handler(RequestType.NOOP, noop_handler)
        helper.run_test(
            request_type=RequestType.NOOP,
            payloads=[],
            expected_response="NOOP_OK",
            num_requests=10,  # Each client sends 10 requests
            num_clients=3,    # Start 3 concurrent clients
        )
    """

    def __init__(self, server_url: str = "tcp://127.0.0.1:5556"):
        self.server_url = server_url
        self.handlers: dict[RequestType, Callable] = {}
        self.ctx = mp.get_context("spawn")

    def register_handler(
        self,
        request_type: RequestType,
        handler: Callable,
    ) -> "MessageQueueTestHelper":
        """
        Register a handler for a specific RequestType.

        Args:
            request_type: The type of request to handle
            handler: Handler function that matches the protocol signature

        Returns:
            self for method chaining
        """
        self.handlers[request_type] = handler
        return self

    def run_test(
        self,
        request_type: RequestType,
        payloads: list[Any],
        expected_response: Any,
        num_requests: int = 1,
        num_clients: int = 1,
        timeout: float = 10.0,
    ) -> None:
        """
        Run a test by starting server and client processes.

        Args:
            request_type: Type of request to send
            payloads: List of payloads for the request
            expected_response: Expected response from server
            num_requests: Number of requests each client should send
            num_clients: Number of client processes to start
            timeout: Maximum time to wait for test completion

        Raises:
            AssertionError: If test fails
        """
        ready_event = self.ctx.Event()
        shutdown_event = self.ctx.Event()

        # Start server process
        server_process = self.ctx.Process(
            target=_server_process,
            args=(self.server_url, ready_event, shutdown_event, self.handlers),
        )
        server_process.start()

        # Start multiple client processes
        client_processes = []
        for client_id in range(num_clients):
            client_process = self.ctx.Process(
                target=_run_client_test,
                args=(
                    self.server_url,
                    ready_event,
                    request_type,
                    payloads,
                    expected_response,
                    num_requests,
                    client_id,
                ),
            )
            client_process.start()
            client_processes.append(client_process)

        # Wait for all clients to complete
        failed_clients = []
        for client_id, client_process in enumerate(client_processes):
            client_process.join(timeout=timeout)

            # Check if client completed successfully
            if client_process.is_alive():
                client_process.terminate()
                client_process.join()
                failed_clients.append((client_id, "timeout"))
            elif client_process.exitcode != 0:
                failed_clients.append(
                    (client_id, f"exit code {client_process.exitcode}")
                )

        # Shutdown server
        shutdown_event.set()
        server_process.join(timeout=2)

        if server_process.is_alive():
            server_process.terminate()
            server_process.join()

        # Report any failures
        if failed_clients:
            failure_details = ", ".join(
                [f"Client {cid}: {reason}" for cid, reason in failed_clients]
            )
            pytest.fail(f"Some clients failed: {failure_details}")

        if server_process.exitcode != 0:
            pytest.fail(
                f"Server process failed with exit code {server_process.exitcode}"
            )


# ==============================================================================
# Tests for Different RequestTypes
# ==============================================================================


def test_mq_noop_request():
    """
    Test MessageQueue with NOOP request type.
    NOOP takes no payloads and returns a string response.
    """
    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5556")
    helper.register_handler(RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    # Run test with single request
    helper.run_test(
        request_type=RequestType.NOOP,
        payloads=[],
        expected_response="NOOP_OK",
        num_requests=1,
    )


def test_mq_noop_multiple_requests():
    """
    Test MessageQueue with multiple NOOP requests.
    Verifies that server can handle multiple sequential requests.
    """
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5557")
    helper.register_handler(RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    # Run test with multiple requests
    helper.run_test(
        request_type=RequestType.NOOP,
        payloads=[],
        expected_response="NOOP_OK",
        num_requests=10,
    )


def test_mq_noop_multiple_clients():
    """
    Test MessageQueue with multiple concurrent clients.
    Verifies that server can handle requests from multiple clients simultaneously.
    """
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5558")
    helper.register_handler(RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    # Run test with multiple clients, each sending multiple requests
    helper.run_test(
        request_type=RequestType.NOOP,
        payloads=[],
        expected_response="NOOP_OK",
        num_requests=5,
        num_clients=3,
    )


@pytest.mark.cuda
@pytest.mark.skipif(
    not (torch_dev.is_available() and torch_device_type == "cuda"),
    reason="requires available CUDA runtime",
)
def test_mq_register_kv_cache():
    """
    Test MessageQueue with REGISTER_KV_CACHE request type.
    REGISTER_KV_CACHE takes (gpu_id: int, kv_cache: KVCache) and returns None.
    """
    # First Party
    from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper

    # Create test KV cache (list of CudaIPCWrapper objects)
    kv_cache = []
    for _ in range(3):
        tensor = torch.randn(2, 4, device=torch_device_type)
        wrapper = CudaIPCWrapper(tensor)
        kv_cache.append(wrapper)

    gpu_id = 0

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5559")
    helper.register_handler(
        RequestType.REGISTER_KV_CACHE, test_mq_handler_helpers.register_kv_cache_handler
    )

    # Run test with REGISTER_KV_CACHE request
    helper.run_test(
        request_type=RequestType.REGISTER_KV_CACHE,
        payloads=[
            gpu_id,
            kv_cache,
            "testmodel",
            1,
            EngineType.VLLM,
            {"vllm_block_size": 16},
            [],
        ],
        expected_response=None,
        num_requests=1,
    )


def test_mq_unregister_kv_cache():
    """
    Test MessageQueue with UNREGISTER_KV_CACHE request type.
    UNREGISTER_KV_CACHE takes (gpu_id: int) and returns None.
    """
    gpu_id = 0

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5560")
    helper.register_handler(
        RequestType.UNREGISTER_KV_CACHE,
        test_mq_handler_helpers.unregister_kv_cache_handler,
    )

    # Run test with UNREGISTER_KV_CACHE request
    helper.run_test(
        request_type=RequestType.UNREGISTER_KV_CACHE,
        payloads=[gpu_id],
        expected_response=None,
        num_requests=1,
    )


def test_mq_unregister_kv_cache_multiple_clients():
    """
    Test MessageQueue with UNREGISTER_KV_CACHE from multiple clients.
    Verifies that multiple clients can unregister KV caches concurrently.
    """
    gpu_id = 0

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5561")
    helper.register_handler(
        RequestType.UNREGISTER_KV_CACHE,
        test_mq_handler_helpers.unregister_kv_cache_handler,
    )

    # Run test with multiple clients
    helper.run_test(
        request_type=RequestType.UNREGISTER_KV_CACHE,
        payloads=[gpu_id],
        expected_response=None,
        num_requests=3,
        num_clients=2,
    )


def test_mq_store():
    """
    Test MessageQueue with STORE request type.
    STORE takes (key: KeyType, gpu_id: int, gpu_block_ids: list[list[int]],
    event_ipc_handle: bytes) and returns (bytes, bool).
    """
    # Create test key
    key = create_cache_key(0)
    gpu_id = 0
    gpu_block_ids = [[0, 1, 2]]
    test_handle = b"\x00" * 64

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5562")
    helper.register_handler(RequestType.STORE, test_mq_handler_helpers.store_handler)

    # Run test with STORE request
    helper.run_test(
        request_type=RequestType.STORE,
        payloads=[key, gpu_id, gpu_block_ids, test_handle],
        expected_response=(b"\x01" * 64, True),
        num_requests=1,
    )


def test_mq_retrieve():
    """
    Test MessageQueue with RETRIEVE request type.
    RETRIEVE takes (key: KeyType, gpu_id: int, gpu_block_ids: list[list[int]],
    event_ipc_handle: bytes) and returns (bytes, bool).
    """
    # Create test key
    key = create_cache_key(0)
    gpu_id = 0
    gpu_block_ids = [[0, 1, 2]]
    test_handle = b"\x00" * 64

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5563")
    helper.register_handler(
        RequestType.RETRIEVE, test_mq_handler_helpers.retrieve_handler
    )

    # Run test with RETRIEVE request
    helper.run_test(
        request_type=RequestType.RETRIEVE,
        payloads=[key, gpu_id, gpu_block_ids, test_handle, 0],
        expected_response=(b"\x01" * 64, True),
        num_requests=1,
    )


def test_mq_lookup():
    """
    Test MessageQueue with LOOKUP request type.
    LOOKUP takes (key: KeyType, tp_size: int) and returns None.
    The job is tracked server-side by request_id; poll via QUERY_PREFETCH_STATUS.
    """
    # Create a single test key
    key = create_cache_key(0)

    # Expected response: None (LOOKUP no longer returns a job_id)
    expected_response = None

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5564")
    helper.register_handler(RequestType.LOOKUP, test_mq_handler_helpers.lookup_handler)

    # Run test with LOOKUP request
    helper.run_test(
        request_type=RequestType.LOOKUP,
        payloads=[key, 1],
        expected_response=expected_response,
        num_requests=1,
    )


def test_mq_lookup_with_different_key():
    """
    Test MessageQueue with LOOKUP request type with a different key.
    Tests that the handler correctly processes a single key.
    """
    # Create a different test key
    key = create_cache_key(42)

    # Expected response: None (LOOKUP no longer returns a job_id)
    expected_response = None

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5565")
    helper.register_handler(RequestType.LOOKUP, test_mq_handler_helpers.lookup_handler)

    # Run test with LOOKUP request
    helper.run_test(
        request_type=RequestType.LOOKUP,
        payloads=[key, 1],
        expected_response=expected_response,
        num_requests=1,
    )


def test_mq_report_block_allocation():
    """
    Test MessageQueue with REPORT_BLOCK_ALLOCATION request type.
    REPORT_BLOCK_ALLOCATION takes (instance_id, model_name, records)
    and returns None.
    """
    records = [
        BlockAllocationRecord(
            req_id="req-1",
            new_block_ids=[0, 1, 2],
            new_token_ids=[100, 200, 300],
        ),
        BlockAllocationRecord(
            req_id="req-2",
            new_block_ids=[3, 4],
            new_token_ids=[400, 500],
        ),
    ]

    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5566")
    helper.register_handler(
        RequestType.REPORT_BLOCK_ALLOCATION,
        test_mq_handler_helpers.report_block_allocations_handler,
    )

    helper.run_test(
        request_type=RequestType.REPORT_BLOCK_ALLOCATION,
        payloads=[42, "test-model", records],
        expected_response=None,
        num_requests=1,
    )


def test_mq_report_block_allocation_empty():
    """
    Test REPORT_BLOCK_ALLOCATION with an empty records list.
    """
    records: list[BlockAllocationRecord] = []

    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5567")
    helper.register_handler(
        RequestType.REPORT_BLOCK_ALLOCATION,
        test_mq_handler_helpers.report_block_allocations_handler,
    )

    helper.run_test(
        request_type=RequestType.REPORT_BLOCK_ALLOCATION,
        payloads=[0, "", records],
        expected_response=None,
        num_requests=1,
    )


# ==============================================================================
# Shared Polling Loop Lifecycle Tests
# ==============================================================================


def test_shared_loop_lifecycle():
    """
    Test that multiple clients share a single ClientPollingLoop and
    that the loop is torn down when all clients close.
    """
    # First Party
    from lmcache.v1.multiprocess.mq import ClientPollingLoop

    context = zmq.Context.instance()

    # No loop before any clients exist
    assert ClientPollingLoop._instance is None

    client_a = MessageQueueClient("tcp://127.0.0.1:16000", context)
    client_b = MessageQueueClient("tcp://127.0.0.1:16001", context)

    # Both share the same singleton
    loop = ClientPollingLoop._instance
    assert loop is not None
    assert loop._ref_count == 2
    assert len(loop._socket_to_client) == 2

    # Close one — loop persists
    client_a.close()
    assert ClientPollingLoop._instance is loop
    assert loop._ref_count == 1
    assert len(loop._socket_to_client) == 1

    # Close the last — loop is destroyed
    client_b.close()
    assert ClientPollingLoop._instance is None


def test_shared_loop_dispatch():
    """
    Test that the shared polling loop correctly dispatches responses
    to multiple clients connected to the same server.

    Server and clients run in the same process (different threads),
    so both clients share one ClientPollingLoop.
    """
    # First Party
    from lmcache.v1.multiprocess.mq import ClientPollingLoop

    server_url = "tcp://127.0.0.1:16020"
    context = zmq.Context.instance()

    # Start server in-process
    server = MessageQueueServer(server_url, context)
    add_handler_helper(server, RequestType.NOOP, test_mq_handler_helpers.noop_handler)
    server.start()

    try:
        # Create two clients sharing the same polling loop
        client_a = MessageQueueClient(server_url, context)
        client_b = MessageQueueClient(server_url, context)

        loop = ClientPollingLoop._instance
        assert loop is not None
        assert loop._ref_count == 2

        # Both clients submit requests concurrently
        futures_a = [client_a.submit_request(RequestType.NOOP, []) for _ in range(5)]
        futures_b = [client_b.submit_request(RequestType.NOOP, []) for _ in range(5)]

        # All futures should resolve with the correct response
        for future in futures_a:
            assert future.result(timeout=5) == "NOOP_OK"
        for future in futures_b:
            assert future.result(timeout=5) == "NOOP_OK"

        client_a.close()
        client_b.close()
        assert ClientPollingLoop._instance is None
    finally:
        server.close()


def test_shared_loop_recreate():
    """
    Test that closing all clients and creating new ones starts a fresh loop.
    """
    # First Party
    from lmcache.v1.multiprocess.mq import ClientPollingLoop

    context = zmq.Context.instance()

    client = MessageQueueClient("tcp://127.0.0.1:16010", context)
    first_loop = ClientPollingLoop._instance
    assert first_loop is not None
    client.close()
    assert ClientPollingLoop._instance is None

    # New client creates a brand-new loop
    client2 = MessageQueueClient("tcp://127.0.0.1:16011", context)
    second_loop = ClientPollingLoop._instance
    assert second_loop is not None
    assert second_loop is not first_loop
    client2.close()
    assert ClientPollingLoop._instance is None


# ==============================================================================
# Thread Pool Tests
# ==============================================================================


def test_add_normal_thread_pool():
    """
    Test that add_normal_thread_pool assigns handler executors.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15700", context)

    add_handler_helper(
        server, RequestType.LOOKUP, test_mq_handler_helpers.lookup_handler
    )
    add_handler_helper(server, RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    lookup_handler = server.handlers[RequestType.LOOKUP]
    assert isinstance(lookup_handler, BlockingRequestHandler)
    assert lookup_handler.executor is None

    server.add_normal_thread_pool([RequestType.LOOKUP], max_workers=4)

    assert lookup_handler.executor is not None
    assert len(server.extra_pools) == 1

    server.close()


def test_add_affinity_thread_pool():
    """
    Test that add_affinity_thread_pool assigns AffinityThreadPool executors.
    """
    # First Party
    from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool

    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15700", context)

    add_handler_helper(server, RequestType.STORE, test_mq_handler_helpers.store_handler)
    add_handler_helper(
        server, RequestType.RETRIEVE, test_mq_handler_helpers.retrieve_handler
    )

    store_handler = server.handlers[RequestType.STORE]
    retrieve_handler = server.handlers[RequestType.RETRIEVE]
    assert isinstance(store_handler, BlockingRequestHandler)
    assert isinstance(retrieve_handler, BlockingRequestHandler)
    assert store_handler.executor is None

    server.add_affinity_thread_pool(
        [RequestType.STORE, RequestType.RETRIEVE], max_workers=2
    )

    assert isinstance(store_handler.executor, AffinityThreadPool)
    assert store_handler.executor is retrieve_handler.executor
    assert len(server.extra_pools) == 1

    server.close()


def test_normal_pool_error_on_sync_handler():
    """
    Test that add_normal_thread_pool raises TypeError for SYNC handlers.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15701", context)

    add_handler_helper(server, RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    with pytest.raises(TypeError, match="not BlockingRequestHandler"):
        server.add_normal_thread_pool([RequestType.NOOP], max_workers=1)

    server.close()


def test_affinity_pool_error_on_sync_handler():
    """
    Test that add_affinity_thread_pool raises TypeError for SYNC handlers.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15701", context)

    add_handler_helper(server, RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    with pytest.raises(TypeError, match="not BlockingRequestHandler"):
        server.add_affinity_thread_pool([RequestType.NOOP], max_workers=1)

    server.close()


def test_pool_error_on_unregistered():
    """
    Test that pool methods raise ValueError for unregistered request types.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15702", context)

    with pytest.raises(ValueError, match="No handler registered"):
        server.add_normal_thread_pool([RequestType.STORE], max_workers=1)

    with pytest.raises(ValueError, match="No handler registered"):
        server.add_affinity_thread_pool([RequestType.STORE], max_workers=1)

    server.close()


def test_multiple_pools():
    """
    Test that normal and affinity pools can coexist.
    """
    # First Party
    from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool

    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15703", context)

    add_handler_helper(server, RequestType.STORE, test_mq_handler_helpers.store_handler)
    add_handler_helper(
        server, RequestType.RETRIEVE, test_mq_handler_helpers.retrieve_handler
    )
    add_handler_helper(
        server, RequestType.LOOKUP, test_mq_handler_helpers.lookup_handler
    )

    server.add_affinity_thread_pool(
        [RequestType.STORE, RequestType.RETRIEVE], max_workers=2
    )
    server.add_normal_thread_pool([RequestType.LOOKUP], max_workers=3)

    store_handler = server.handlers[RequestType.STORE]
    retrieve_handler = server.handlers[RequestType.RETRIEVE]
    lookup_handler = server.handlers[RequestType.LOOKUP]
    assert isinstance(store_handler, BlockingRequestHandler)
    assert isinstance(retrieve_handler, BlockingRequestHandler)
    assert isinstance(lookup_handler, BlockingRequestHandler)

    # STORE/RETRIEVE share affinity pool
    assert isinstance(store_handler.executor, AffinityThreadPool)
    assert store_handler.executor is retrieve_handler.executor
    # LOOKUP uses normal pool
    assert store_handler.executor is not lookup_handler.executor
    assert not isinstance(lookup_handler.executor, AffinityThreadPool)
    assert len(server.extra_pools) == 2

    server.close()


def test_start_fails_without_pool_assignment():
    """
    Test that start() raises RuntimeError if a blocking handler
    has no executor assigned.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15704", context)

    add_handler_helper(server, RequestType.STORE, test_mq_handler_helpers.store_handler)
    # Don't assign any pool

    with pytest.raises(RuntimeError, match="no thread pool assigned"):
        server.start()

    server.close()
