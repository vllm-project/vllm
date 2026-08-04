# SPDX-License-Identifier: Apache-2.0
"""
Tests for the FREE_LOOKUP_LOCKS protocol: enum registration, protocol definition,
message-queue round-trip, server handler, and client-side adapter API.
"""

# Standard
from unittest.mock import MagicMock, patch
import threading

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType

# Test helpers
from tests.v1.multiprocess import test_mq_handler_helpers
from tests.v1.multiprocess.test_mq import (
    MessageQueueTestHelper,
    create_cache_key,
)

# ============================================================================
# Protocol definition tests
# ============================================================================


def test_free_locks_in_request_type():
    """FREE_LOOKUP_LOCKS should be a member of RequestType."""
    assert hasattr(RequestType, "FREE_LOOKUP_LOCKS")
    assert isinstance(RequestType.FREE_LOOKUP_LOCKS, RequestType)


def test_free_locks_payload_classes():
    """FREE_LOOKUP_LOCKS payload should be [IPCCacheServerKey, int]."""
    payload_classes = get_payload_classes(RequestType.FREE_LOOKUP_LOCKS)
    assert len(payload_classes) == 2
    assert payload_classes[0] is IPCCacheServerKey
    assert payload_classes[1] is int


def test_free_locks_response_class():
    """FREE_LOOKUP_LOCKS should have no response (None)."""
    response_class = get_response_class(RequestType.FREE_LOOKUP_LOCKS)
    assert response_class is None


def test_free_locks_handler_type():
    """FREE_LOOKUP_LOCKS should use BLOCKING handler type."""
    handler_type = get_handler_type(RequestType.FREE_LOOKUP_LOCKS)
    assert handler_type == HandlerType.BLOCKING


# ============================================================================
# Message-queue round-trip test
# ============================================================================


def test_mq_free_locks():
    """
    Test MessageQueue with FREE_LOOKUP_LOCKS request type.
    FREE_LOOKUP_LOCKS takes (key: KeyType) and returns None.
    """
    key = create_cache_key(0)

    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5570")
    helper.register_handler(
        RequestType.FREE_LOOKUP_LOCKS, test_mq_handler_helpers.free_locks_handler
    )

    helper.run_test(
        request_type=RequestType.FREE_LOOKUP_LOCKS,
        payloads=[key, 1],
        expected_response=None,
        num_requests=1,
    )


# ============================================================================
# Server handler tests
# ============================================================================


def test_server_free_lookup_locks_calls_finish_read_prefetched():
    """LookupModule.free_lookup_locks should resolve hash keys and call
    finish_read_prefetched on the storage manager."""
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    ctx = MagicMock()
    ctx.token_hasher.chunk_size = 256
    ctx.token_hasher.compute_chunk_hashes.return_value = [b"hash0"]

    module = LookupModule(ctx)

    # Build a key
    key = create_cache_key(0).no_worker_id_version()

    sentinel_obj_keys = [MagicMock()]
    with patch(
        "lmcache.v1.multiprocess.modules.lookup.ipc_key_to_object_keys",
        return_value=[sentinel_obj_keys],
    ):
        module.free_lookup_locks(key, 1)

    module.context.storage_manager.finish_read_prefetched.assert_called_once_with(
        sentinel_obj_keys, extra_count=0
    )


def test_server_free_lookup_locks_no_matching_chunks():
    """LookupModule.free_lookup_locks with no chunks in range should be a no-op."""
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    ctx = MagicMock()
    ctx.token_hasher.chunk_size = 256
    ctx.token_hasher.compute_chunk_hashes.return_value = []

    module = LookupModule(ctx)

    # Key with start == end means no chunks to free
    key = IPCCacheServerKey(
        model_name="testmodel",
        world_size=1,
        worker_id=None,
        token_ids=tuple(range(256)),
        start=0,
        end=0,
        request_id="req-empty",
    )

    module.free_lookup_locks(key, 1)

    module.context.storage_manager.finish_read_prefetched.assert_not_called()


def test_server_handler_registered():
    """LookupModule should have a free_lookup_locks method."""
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    assert hasattr(LookupModule, "free_lookup_locks")
    assert callable(LookupModule.free_lookup_locks)


# ============================================================================
# Client adapter tests
# ============================================================================


def test_adapter_free_lookup_locks_sends_request():
    """LMCacheMPSchedulerAdapter.free_lookup_locks should send a FREE_LOOKUP_LOCKS
    request with the correct key payload."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
        ParallelStrategy,
    )

    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.model_name = "test_model"
    adapter.lmcache_tokens_per_chunk = 256
    adapter.blocks_in_chunk = 16
    adapter.parallel_strategy = ParallelStrategy(False, 1, 0, 1, 1, 1)
    adapter._health_events = {"tcp://test:0": threading.Event()}
    adapter._health_events["tcp://test:0"].set()
    adapter._server_urls = ["tcp://test:0"]
    adapter._mq_timeout = 30.0

    mock_client = MagicMock(spec=MessageQueueClient)
    mock_future = MagicMock()
    mock_client.submit_request.return_value = mock_future
    adapter.mq_clients = {"tcp://test:0": mock_client}
    adapter._pending_lookups = set()

    token_ids = list(range(512))
    adapter.free_lookup_locks(
        token_ids=token_ids,
        start=0,
        end=512,
        request_id="req-1",
    )

    mock_client.submit_request.assert_called_once()
    call_args = mock_client.submit_request.call_args
    req_type = call_args[0][0]
    payloads = call_args[0][1]
    assert req_type == RequestType.FREE_LOOKUP_LOCKS

    # Payload should be [key, tp_size]
    assert isinstance(payloads, list)
    assert len(payloads) == 2

    key = payloads[0]
    assert isinstance(key, IPCCacheServerKey)
    assert key.worker_id is None
    assert key.model_name == "test_model"
    assert key.request_id == "req-1"
    assert payloads[1] == 1  # tp_size


def test_adapter_free_lookup_locks_key_matches_lookup():
    """The key created by free_lookup_locks should match the key created by
    maybe_submit_lookup_request (no_worker_id_version, same start/end)."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
        ParallelStrategy,
    )

    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.model_name = "test_model"
    adapter.lmcache_tokens_per_chunk = 256
    adapter.blocks_in_chunk = 16
    adapter.parallel_strategy = ParallelStrategy(False, 1, 0, 1, 1, 1)
    adapter._server_urls = ["tcp://test:0"]
    adapter._health_events = {"tcp://test:0": threading.Event()}
    adapter._health_events["tcp://test:0"].set()
    adapter._mq_timeout = 30.0
    adapter._heartbeats: dict[str, object] = {}
    adapter._heartbeat_lock = threading.Lock()
    adapter._heartbeat_interval = 5.0

    mock_client = MagicMock(spec=MessageQueueClient)
    mock_future = MagicMock()
    mock_future.result.return_value = None  # LOOKUP returns None
    mock_client.submit_request.return_value = mock_future
    adapter.mq_clients = {"tcp://test:0": mock_client}
    adapter._pending_lookups = set()
    adapter._lookup_params = {}

    token_ids = list(range(512))

    # Submit lookup – patch heartbeat to avoid spawning a real thread
    with patch.object(adapter, "_ensure_heartbeat_started"):
        adapter.maybe_submit_lookup_request("req-1", token_ids)
    lookup_call = mock_client.submit_request.call_args
    lookup_payloads = lookup_call[0][1]
    lookup_key = lookup_payloads[0]

    mock_client.submit_request.reset_mock()

    # Submit free_lookup_locks with aligned end
    tokens_per_chunk = adapter.lmcache_tokens_per_chunk
    aligned_end = (len(token_ids) // tokens_per_chunk) * tokens_per_chunk
    adapter.free_lookup_locks(
        token_ids=token_ids,
        start=0,
        end=aligned_end,
        request_id="req-1",
    )
    free_call = mock_client.submit_request.call_args
    free_payloads = free_call[0][1]
    assert len(free_payloads) == 2
    free_key = free_payloads[0]
    assert free_payloads[1] == 1  # tp_size

    # Keys should be identical
    assert lookup_key.model_name == free_key.model_name
    assert lookup_key.world_size == free_key.world_size
    assert lookup_key.worker_id == free_key.worker_id
    assert lookup_key.worker_id is None
    assert lookup_key.start == free_key.start
    assert lookup_key.end == free_key.end
    assert lookup_key.request_id == free_key.request_id
    assert lookup_key.token_ids == free_key.token_ids
