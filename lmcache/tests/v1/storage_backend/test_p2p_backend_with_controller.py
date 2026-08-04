# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import contextlib
import multiprocessing as mp
import threading
import time

# Third Party
import msgspec
import pytest
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.message import (
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.paged_cpu_gpu_memory_allocator import (
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.p2p_backend import P2PBackend
from lmcache.v1.transfer_channel.transfer_utils import P2PInitSideRetMsg
from tests.v1.utils import get_available_ports

logger = init_logger(__name__)


def wait_for_keys_in_cache(
    backend: LocalCPUBackend,
    keys: list,
    timeout: float = 5.0,
    poll_interval: float = 0.1,
) -> bool:
    """Wait for keys to appear in the backend's cache."""
    start_time = time.time()
    missing_keys = keys.copy()  # Initialize with all keys
    while time.time() - start_time < timeout:
        missing_keys = [key for key in keys if not backend.contains(key, pin=False)]
        if not missing_keys:
            return True
        time.sleep(poll_interval)
    logger.warning(
        f"Timeout waiting for keys: {[key.to_string() for key in missing_keys]}"
    )
    return False


def wait_for_backend_ready(
    backend: P2PBackend, timeout: float = 5.0, poll_interval: float = 0.1
) -> bool:
    """Wait for P2P backend to be ready for operations."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check if backend is properly initialized by checking its attributes
        if (
            hasattr(backend, "tp_rank")
            and hasattr(backend, "peer_init_url")
            and backend.peer_init_url
            and hasattr(backend, "peer_lookup_url")
            and backend.peer_lookup_url
        ):
            return True
        time.sleep(poll_interval)
    logger.warning("Timeout waiting for backend to be ready")
    return False


class MockLMCacheWorker:
    """Mock worker that simulates controller responses"""

    def __init__(self, controller_url: str):
        self.controller_url = controller_url
        self.messages: list = []
        self.context = None
        self.socket = None

    def _ensure_socket(self):
        """Ensure socket is created and connected"""
        if self.socket is None:
            self.context = zmq.asyncio.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(self.controller_url)
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)
            self.socket.setsockopt(zmq.SNDTIMEO, 5000)

    async def async_put_and_wait_msg(self, msg):
        """Send message to controller and wait for response"""
        self.messages.append(msg)

        if isinstance(msg, BatchedP2PLookupMsg):
            self._ensure_socket()
            try:
                await self.socket.send(msgspec.msgpack.encode(msg))
                response_bytes = await self.socket.recv()
                response = msgspec.msgpack.decode(
                    response_bytes, type=BatchedP2PLookupRetMsg
                )
                return response
            except zmq.Again as e:
                logger.error("Timeout waiting for controller response: %s", e)
                return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])
            except Exception as e:
                logger.error("Error communicating with controller: %s", e)
                return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])

        return None

    def close(self):
        """Close socket and context"""
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def run_mock_controller(
    controller_url: str,
    peer_mappings: dict,
    stop_event: "mp.synchronize.Event",
    ready_event: "mp.synchronize.Event",
):
    """Run a mock controller that handles P2P lookup requests"""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(controller_url)
    socket.setsockopt(zmq.RCVTIMEO, 1000)

    logger.info(f"Mock controller started at {controller_url}")
    ready_event.set()  # Signal that the controller is ready

    while not stop_event.is_set():
        try:
            msg_bytes = socket.recv()
            msg = msgspec.msgpack.decode(msg_bytes, type=BatchedP2PLookupMsg)

            # Simulate lookup logic
            response_key = (msg.instance_id, msg.worker_id, tuple(msg.hashes))
            if response_key in peer_mappings:
                response = peer_mappings[response_key]
            else:
                response = BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])

            socket.send(msgspec.msgpack.encode(response))
        except zmq.Again:
            continue
        except Exception as e:
            logger.error("Controller error: %s", e)
            if not stop_event.is_set():
                time.sleep(0.01)

    socket.close()
    context.term()
    logger.info("Mock controller stopped")


def create_test_config(
    p2p_host: str,
    p2p_init_ports: list[int],
    p2p_lookup_ports: list[int],
    transfer_channel: str = "mock_memory",
    extra_config: Optional[dict] = None,
):
    """Create test configuration for P2P backend"""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=1.0,
        p2p_host=p2p_host,
        p2p_init_ports=p2p_init_ports,
        p2p_lookup_ports=p2p_lookup_ports,
        transfer_channel=transfer_channel,
        lmcache_instance_id="test_instance",
    )
    if extra_config:
        config.extra_config = extra_config
    return config


def create_test_metadata(worker_id: int = 0):
    """Create test metadata"""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=2,
        local_world_size=2,
        worker_id=worker_id,
        local_worker_id=worker_id,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
    )


def create_test_key(key_id: str = "test_key") -> CacheEngineKey:
    """Create a test CacheEngineKey"""
    return CacheEngineKey(
        model_name="test_model",
        world_size=2,
        worker_id=0,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
        request_configs=None,
    )


@pytest.fixture
def async_loop():
    """Create an asyncio event loop running in a background thread"""
    loop = asyncio.new_event_loop()
    ready_event = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        ready_event.set()  # Signal that the loop is ready
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()

    # Wait for the thread to initialize the event loop
    if not ready_event.wait(timeout=5.0):
        raise RuntimeError("Event loop thread failed to initialize within 5 seconds")

    yield loop

    # Stop the loop first to prevent new tasks from being scheduled
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=3)

    # Force terminate if thread is still alive
    if thread.is_alive():
        logger.warning("Event loop thread did not stop gracefully")

    # Clean up remaining tasks without recursion
    if not loop.is_closed():
        try:
            # Get all pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                if not task.done():
                    # Cancel without waiting to avoid recursion
                    task.cancel()
        except Exception as e:
            logger.warning("Error cancelling tasks: %s", e)

        # Close the loop
        try:
            loop.close()
        except Exception as e:
            logger.warning("Error closing loop: %s", e)


@contextlib.contextmanager
def cpu_allocator():
    """Create and cleanup CPU memory allocator"""
    allocator = PagedCpuGpuMemoryAllocator()
    allocator.init_cpu_memory_allocator(
        size=1120 * 1024 * 1024,
        shapes=[torch.Size([28, 2, 256, 8, 128])],
        dtypes=[torch.bfloat16],
        fmt=MemoryFormat.KV_2LTD,
    )
    try:
        yield allocator
    finally:
        allocator.close()


@pytest.fixture
def local_cpu_backend():
    """Create local CPU backend with allocator"""

    def _create_backend(config):
        allocator = PagedCpuGpuMemoryAllocator()
        allocator.init_cpu_memory_allocator(
            size=1120 * 1024 * 1024,
            shapes=[torch.Size([28, 2, 256, 8, 128])],
            dtypes=[torch.bfloat16],
            fmt=MemoryFormat.KV_2LTD,
        )
        return LocalCPUBackend(config=config, memory_allocator=allocator)

    return _create_backend


@contextlib.contextmanager
def mock_controller_context(peer_mappings: dict):
    """Context manager for mock controller process"""
    controller_port = get_available_ports(1)[0]
    controller_url = f"tcp://localhost:{controller_port}"
    stop_event = mp.Event()
    ready_event = mp.Event()

    process = mp.Process(
        target=run_mock_controller,
        args=(controller_url, peer_mappings, stop_event, ready_event),
    )
    process.start()

    # Wait for the controller to signal it's ready
    if not ready_event.wait(timeout=5.0):
        raise RuntimeError("Mock controller failed to start within 5 seconds")

    try:
        yield controller_url
    finally:
        stop_event.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join()


@contextlib.contextmanager
def p2p_backend_context(config, metadata, async_loop, local_backend, mock_worker):
    """Context manager for P2P backend with automatic cleanup"""
    backend = P2PBackend(
        config=config,
        metadata=metadata,
        loop=async_loop,
        local_cpu_backend=local_backend,
        lmcache_worker=mock_worker,
    )
    try:
        yield backend
    finally:
        try:
            backend.close()
        except Exception as e:
            logger.warning("Error closing P2P backend: %s", e)
        # Give some time for cleanup
        time.sleep(0.1)


class TestP2PBackendWithController:
    """Test cases for P2P Backend with Controller"""

    def test_p2p_backend_initialization(self, async_loop, local_cpu_backend):
        """
        Test the basic initialization of P2P backend with mock_memory transfer channel.

        This test verifies that:
        - P2P backend can be properly initialized with valid configuration
        - All required attributes are set correctly (tp_rank, peer URLs)
        - The backend is ready for P2P communication
        """
        peer_init_port, peer_lookup_port, controller_port = get_available_ports(3)

        config = create_test_config(
            p2p_host="localhost",
            p2p_init_ports=[peer_init_port],
            p2p_lookup_ports=[peer_lookup_port],
        )
        metadata = create_test_metadata(worker_id=0)
        local_backend = local_cpu_backend(config)
        controller_url = f"tcp://localhost:{controller_port}"

        with (
            MockLMCacheWorker(controller_url) as mock_worker,
            p2p_backend_context(
                config, metadata, async_loop, local_backend, mock_worker
            ) as p2p_backend,
        ):
            assert p2p_backend is not None
            assert p2p_backend.tp_rank == 0
            assert p2p_backend.peer_init_url == f"localhost:{peer_init_port}"
            assert p2p_backend.peer_lookup_url == f"localhost:{peer_lookup_port}"

    @pytest.mark.skip(reason="Complex P2P communication test that may hang in CI/CD")
    def test_p2p_backend_normal_flow(self, async_loop):
        """
        Test the complete P2P data retrieval flow between two peers
        with controller coordination.

        This test verifies:
        - Two P2P backends can communicate through the controller
        - Data can be put into one peer and retrieved from another
        - Batched contains and get operations work correctly
        - The controller properly routes lookup requests
        """
        peer1_init_port, peer1_lookup_port, peer2_init_port, peer2_lookup_port = (
            get_available_ports(4)
        )

        config1 = create_test_config(
            p2p_host="localhost",
            p2p_init_ports=[peer1_init_port, peer2_init_port],
            p2p_lookup_ports=[peer1_lookup_port, peer2_lookup_port],
        )
        config2 = create_test_config(
            p2p_host="localhost",
            p2p_init_ports=[peer1_init_port, peer2_init_port],
            p2p_lookup_ports=[peer1_lookup_port, peer2_lookup_port],
        )

        metadata1 = create_test_metadata(worker_id=0)
        metadata2 = create_test_metadata(worker_id=1)

        test_keys = [create_test_key(f"key_{i}") for i in range(2)]
        test_hashes = tuple([key.chunk_hash for key in test_keys])
        peer2_init_url = f"localhost:{peer2_init_port}"

        peer_mappings = {
            ("test_instance", 0, test_hashes): BatchedP2PLookupRetMsg(
                layout_info=[("test_instance", "cpu", 2, peer2_init_url)]
            )
        }

        with cpu_allocator() as allocator1, cpu_allocator() as allocator2:
            local_backend1 = LocalCPUBackend(
                config=config1, memory_allocator=allocator1
            )
            local_backend2 = LocalCPUBackend(
                config=config2, memory_allocator=allocator2
            )

            with mock_controller_context(peer_mappings) as controller_url:
                with (
                    MockLMCacheWorker(controller_url) as mock_worker1,
                    MockLMCacheWorker(controller_url) as mock_worker2,
                ):
                    with (
                        p2p_backend_context(
                            config1, metadata1, async_loop, local_backend1, mock_worker1
                        ) as p2p_backend1,
                        p2p_backend_context(
                            config2, metadata2, async_loop, local_backend2, mock_worker2
                        ) as _p2p_backend2,
                    ):
                        # Wait for backends to be ready instead of using fixed sleep
                        assert wait_for_backend_ready(p2p_backend1), (
                            "P2P backend 1 failed to initialize"
                        )

                        # Prepare test data on Peer2
                        test_objs = []
                        for key in test_keys:
                            mem_obj = local_backend2.allocate(
                                torch.Size([28, 2, 256, 8, 128]),
                                torch.bfloat16,
                                MemoryFormat.KV_2LTD,
                            )
                            if mem_obj.tensor is not None:
                                mem_obj.tensor.fill_(float(hash(key.to_string()) % 100))
                            test_objs.append(mem_obj)

                        for key, obj in zip(test_keys, test_objs, strict=True):
                            local_backend2.submit_put_task(key, obj)

                        # Wait for keys to be stored in cache
                        assert wait_for_keys_in_cache(local_backend2, test_keys), (
                            "Keys not stored in cache"
                        )

                        # Test P2P get
                        async def test_get():
                            num_hits = await p2p_backend1.batched_async_contains(
                                lookup_id="test_lookup_1", keys=test_keys, pin=False
                            )
                            assert num_hits == 2, f"Expected 2 hits, got {num_hits}"

                            cum_chunk_lengths = [0, 256, 512]
                            retrieved_objs = (
                                await p2p_backend1.batched_get_non_blocking(
                                    lookup_id="test_lookup_1",
                                    keys=test_keys,
                                    transfer_spec={
                                        "cum_chunk_lengths": cum_chunk_lengths
                                    },
                                )
                            )

                            assert len(retrieved_objs) == 2, (
                                f"Expected 2 objects, got {len(retrieved_objs)}"
                            )
                            for retrieved_obj in retrieved_objs:
                                assert retrieved_obj.tensor is not None, (
                                    "Retrieved tensor should not be None"
                                )

                            logger.info("P2P normal flow test passed")

                        try:
                            asyncio.run_coroutine_threadsafe(
                                test_get(), async_loop
                            ).result(timeout=10)
                        except Exception as e:
                            logger.error("Test failed: %s", e)
                            raise

    def test_p2p_backend_no_hits(self, async_loop):
        """
        Test P2P backend behavior when the controller returns no matching data.

        This test verifies:
        - The backend correctly handles empty lookup results
        - No false positives are reported
        - The system doesn't crash on empty responses
        - Proper error handling for missing data
        """
        peer_init_port, peer_lookup_port = get_available_ports(2)

        config = create_test_config(
            p2p_host="localhost",
            p2p_init_ports=[peer_init_port],
            p2p_lookup_ports=[peer_lookup_port],
        )
        metadata = create_test_metadata(worker_id=0)
        peer_mappings = {}

        with cpu_allocator() as allocator:
            local_backend = LocalCPUBackend(config=config, memory_allocator=allocator)

            with mock_controller_context(peer_mappings) as controller_url:
                with MockLMCacheWorker(controller_url) as mock_worker:
                    with p2p_backend_context(
                        config, metadata, async_loop, local_backend, mock_worker
                    ) as p2p_backend:
                        # Wait for backend to be ready instead of using fixed sleep
                        assert wait_for_backend_ready(p2p_backend), (
                            "P2P backend failed to initialize"
                        )

                        test_keys = [create_test_key("key_1")]

                        async def test_no_hits():
                            num_hits = await p2p_backend.batched_async_contains(
                                lookup_id="test_lookup_no_hits",
                                keys=test_keys,
                                pin=False,
                            )
                            assert num_hits == 0, f"Expected 0 hits, got {num_hits}"

                        asyncio.run_coroutine_threadsafe(
                            test_no_hits(), async_loop
                        ).result(timeout=5)

    def test_ensure_peer_connection_race_condition(self, async_loop):
        peer_init_port, peer_lookup_port = get_available_ports(2)

        config = create_test_config(
            p2p_host="localhost",
            p2p_init_ports=[peer_init_port],
            p2p_lookup_ports=[peer_lookup_port],
        )
        metadata = create_test_metadata(worker_id=0)
        peer_mappings = {}

        with cpu_allocator() as allocator:
            local_backend = LocalCPUBackend(config=config, memory_allocator=allocator)

            with mock_controller_context(peer_mappings) as controller_url:
                with MockLMCacheWorker(controller_url) as mock_worker:
                    with patch(
                        "lmcache.v1.rpc_utils.get_zmq_socket_with_timeout"
                    ) as mock_get_socket:
                        mock_get_socket.return_value = MagicMock()

                        with p2p_backend_context(
                            config, metadata, async_loop, local_backend, mock_worker
                        ) as p2p_backend:

                            async def async_lazy_init_peer_connection(*args, **kwargs):
                                logger.info("async_lazy_init_peer_connection called")
                                await asyncio.sleep(0.5)
                                return P2PInitSideRetMsg(
                                    peer_lookup_url="127.0.0.1:19998"
                                )

                            p2p_backend.transfer_channel.async_lazy_init_peer_connection = AsyncMock(  # noqa: E501
                                side_effect=async_lazy_init_peer_connection
                            )

                            # by mock_get_socket, the port is not used in the test
                            target_peer_init_url = "127.0.0.1:19999"
                            future_1 = asyncio.run_coroutine_threadsafe(
                                p2p_backend._ensure_peer_connection(
                                    target_peer_init_url
                                ),
                                async_loop,
                            )
                            future_2 = asyncio.run_coroutine_threadsafe(
                                p2p_backend._ensure_peer_connection(
                                    target_peer_init_url
                                ),
                                async_loop,
                            )
                            future_1.result(timeout=5)
                            future_2.result(timeout=5)

                            assert (
                                p2p_backend.transfer_channel.async_lazy_init_peer_connection.call_count
                                == 1
                            )
