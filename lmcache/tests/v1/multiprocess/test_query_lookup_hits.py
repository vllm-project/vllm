# SPDX-License-Identifier: Apache-2.0
"""
Tests for the QUERY_PREFETCH_LOOKUP_HITS protocol: enum registration,
protocol definition, message-queue round-trip, and server handler.
"""

# Standard
from unittest.mock import MagicMock
import time

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.storage_manager import PrefetchHandle
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule, _PrefetchJob
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType

# Test helpers
from tests.v1.multiprocess.test_mq import (
    MessageQueueTestHelper,
)

# ============================================================================
# Protocol definition tests
# ============================================================================


def test_query_prefetch_lookup_hits_in_request_type():
    """QUERY_PREFETCH_LOOKUP_HITS should be a member of RequestType."""
    assert hasattr(RequestType, "QUERY_PREFETCH_LOOKUP_HITS")
    assert isinstance(RequestType.QUERY_PREFETCH_LOOKUP_HITS, RequestType)


def test_query_prefetch_lookup_hits_payload_classes():
    """QUERY_PREFETCH_LOOKUP_HITS payload should be [str]."""
    payload_classes = get_payload_classes(RequestType.QUERY_PREFETCH_LOOKUP_HITS)
    assert len(payload_classes) == 1
    assert payload_classes[0] is str


def test_query_prefetch_lookup_hits_response_class():
    """QUERY_PREFETCH_LOOKUP_HITS response should be int | None."""
    response_class = get_response_class(RequestType.QUERY_PREFETCH_LOOKUP_HITS)
    assert response_class == int | None


def test_query_prefetch_lookup_hits_handler_type():
    """QUERY_PREFETCH_LOOKUP_HITS should use BLOCKING handler type."""
    handler_type = get_handler_type(RequestType.QUERY_PREFETCH_LOOKUP_HITS)
    assert handler_type == HandlerType.BLOCKING


# ============================================================================
# Message-queue round-trip test
# ============================================================================


def _query_lookup_hits_handler(request_id: str) -> int | None:
    """Dummy handler for QUERY_PREFETCH_LOOKUP_HITS requests."""
    assert isinstance(request_id, str)
    return 42


def test_mq_query_prefetch_lookup_hits():
    """Test MessageQueue with QUERY_PREFETCH_LOOKUP_HITS request type."""
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5575")
    helper.register_handler(
        RequestType.QUERY_PREFETCH_LOOKUP_HITS, _query_lookup_hits_handler
    )

    helper.run_test(
        request_type=RequestType.QUERY_PREFETCH_LOOKUP_HITS,
        payloads=["req-1"],
        expected_response=42,
        num_requests=1,
    )


def _query_lookup_hits_none_handler(request_id: str) -> int | None:
    """Dummy handler that returns None (lookup still in progress)."""
    assert isinstance(request_id, str)
    return None


def test_mq_query_prefetch_lookup_hits_none_response():
    """Test MessageQueue returns None when lookup is still in progress."""
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5576")
    helper.register_handler(
        RequestType.QUERY_PREFETCH_LOOKUP_HITS, _query_lookup_hits_none_handler
    )

    helper.run_test(
        request_type=RequestType.QUERY_PREFETCH_LOOKUP_HITS,
        payloads=["req-1"],
        expected_response=None,
        num_requests=1,
    )


# ============================================================================
# Server handler tests
# ============================================================================


def _make_module_with_job(
    world_size: int, storage_return: int | None, num_object_groups: int = 1
) -> tuple[LookupModule, str]:
    """Create a LookupModule with a mock context and a single prefetch job.

    Returns:
        (module, request_id)
    """
    ctx = MagicMock()
    ctx.token_hasher.chunk_size = 256
    module = LookupModule(ctx)

    handle = PrefetchHandle(
        prefetch_request_id=0,
        external_request_id="req-0",
        l1_found_indices=(),
        total_requested_keys=10,
        submit_time=time.monotonic(),
    )
    request_id = "req-1"
    job = _PrefetchJob(
        handle=handle,
        world_size=world_size,
        request_id=request_id,
        requested_tokens=0,
        num_object_groups=num_object_groups,
    )
    module._prefetch_jobs[request_id] = job
    # The storage layer returns the prefix-hit count; the module divides it by
    # world_size.
    ctx.storage_manager.query_prefetch_lookup_hits.return_value = storage_return

    return module, request_id


def test_server_lookup_hits_returns_count():
    """query_prefetch_lookup_hits returns chunk count when lookup is done."""
    module, request_id = _make_module_with_job(world_size=1, storage_return=5)

    result = module.query_prefetch_lookup_hits(request_id)

    assert result == 5
    module.context.storage_manager.query_prefetch_lookup_hits.assert_called_once()


def test_server_lookup_hits_divides_by_world_size():
    """Result should be divided by world_size for tensor parallelism."""
    module, request_id = _make_module_with_job(world_size=2, storage_return=10)

    result = module.query_prefetch_lookup_hits(request_id)

    assert result == 5  # 10 // 2


def test_server_lookup_hits_divides_by_world_size_times_num_groups():
    """Chunk-major layout packs world_size * num_object_groups keys per chunk."""
    module, request_id = _make_module_with_job(
        world_size=2, storage_return=12, num_object_groups=3
    )

    result = module.query_prefetch_lookup_hits(request_id)

    assert result == 2  # 12 // (2 * 3)


def test_server_lookup_hits_returns_none_when_in_progress():
    """Returns None when storage manager lookup is still in progress."""
    module, request_id = _make_module_with_job(world_size=1, storage_return=None)

    result = module.query_prefetch_lookup_hits(request_id)

    assert result is None


def test_server_lookup_hits_returns_zero_for_invalid_request():
    """Returns 0 for a request_id that doesn't exist (prevents infinite spin)."""
    ctx = MagicMock()
    ctx.token_hasher.chunk_size = 256
    module = LookupModule(ctx)

    result = module.query_prefetch_lookup_hits("nonexistent-req")

    assert result == 0
    ctx.storage_manager.query_prefetch_lookup_hits.assert_not_called()


def test_server_lookup_hits_returns_zero_after_prefetch_consumed():
    """Returns 0 after query_prefetch_status has consumed the job.

    This prevents the caller from spinning forever on a completed request.
    """
    module, request_id = _make_module_with_job(world_size=1, storage_return=5)

    # Simulate query_prefetch_status consuming the job
    module._prefetch_jobs.pop(request_id)

    result = module.query_prefetch_lookup_hits(request_id)

    assert result == 0


def test_server_lookup_hits_zero_count():
    """Returns 0 when no keys matched (not None)."""
    module, request_id = _make_module_with_job(world_size=1, storage_return=0)

    result = module.query_prefetch_lookup_hits(request_id)

    assert result == 0
    assert result is not None


def test_server_handler_registered():
    """LookupModule should have a query_prefetch_lookup_hits method."""
    assert hasattr(LookupModule, "query_prefetch_lookup_hits")
    assert callable(LookupModule.query_prefetch_lookup_hits)


# ============================================================================
# Chunk-major key layout
# ============================================================================


def _lookup_key(world_size: int) -> IPCCacheServerKey:
    """A lookup-side IPC key (worker_id None -> expand over all workers)."""
    return IPCCacheServerKey(
        model_name="m",
        world_size=world_size,
        worker_id=None,
        token_ids=(0,),
        start=0,
        end=0,
        request_id="r",
    )


def _captured_lookup_object_keys(
    world_size: int, num_groups: int, chunk_hashes: list[bytes]
) -> list[ObjectKey]:
    """Drive the public ``lookup()`` and return the object keys it submits.

    The engine context is mocked so ``lookup()`` runs end-to-end; the
    chunk-major key list it builds is recovered from the ``submit_prefetch_task``
    call rather than by reaching into a private helper.
    """
    ctx = MagicMock()
    ctx.chunk_size = 16
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = MagicMock()  # non-None layout
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1] * num_groups
    )
    ctx.token_hasher.compute_chunk_hashes.return_value = chunk_hashes

    module = LookupModule(ctx)
    module.lookup(_lookup_key(world_size=world_size), tp_size=1)

    ctx.storage_manager.submit_prefetch_task.assert_called_once()
    return ctx.storage_manager.submit_prefetch_task.call_args.args[0].keys


def test_lookup_lays_keys_out_chunk_then_group_then_rank():
    """lookup() submits keys laid out chunk -> object group -> kv_rank, so each
    chunk's keys are contiguous (the property that makes a leading-ones prefix
    equal to the full-attention model-wide hit)."""
    keys = _captured_lookup_object_keys(
        world_size=2, num_groups=2, chunk_hashes=[b"c0", b"c1"]
    )

    # 2 chunks * 2 groups * 2 ranks.
    assert len(keys) == 8
    # Each chunk's 4 keys are contiguous.
    assert [k.chunk_hash for k in keys[:4]] == [b"c0"] * 4
    assert [k.chunk_hash for k in keys[4:]] == [b"c1"] * 4
    # Within a chunk, group 0 (both ranks) precedes group 1 (both ranks).
    assert [k.object_group_id for k in keys[:4]] == [0, 0, 1, 1]
    assert [k.object_group_id for k in keys[4:]] == [0, 0, 1, 1]
    # The two ranks within one (chunk, group) cell are distinct.
    assert keys[0].kv_rank != keys[1].kv_rank


def test_lookup_single_group_matches_single_group_layout():
    """With one object group the submitted layout is byte-identical to the
    single-group layout (the object-group-separation-disabled / non-hybrid
    case)."""
    chunk_hashes = [b"c0", b"c1"]
    keys = _captured_lookup_object_keys(
        world_size=2, num_groups=1, chunk_hashes=chunk_hashes
    )

    expected = ipc_key_to_object_keys(_lookup_key(world_size=2), chunk_hashes, [0])[0]
    assert keys == expected
