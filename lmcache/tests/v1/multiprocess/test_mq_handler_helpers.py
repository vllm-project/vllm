# SPDX-License-Identifier: Apache-2.0
"""
Helper handler functions for MessageQueue tests.

These handlers are defined at module level to allow them to be pickled
and passed between processes during multiprocessing tests.
"""

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    KVCache,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocol import KeyType

# ==============================================================================
# NOOP Request Handlers
# ==============================================================================


def noop_handler() -> str:
    """
    Dummy handler for NOOP requests.
    Takes no arguments and returns a simple string response.
    """
    return "NOOP_OK"


# ==============================================================================
# REGISTER_KV_CACHE Request Handlers
# ==============================================================================


def register_kv_cache_handler(
    gpu_id: int,
    kv_cache: KVCache,
    model_name: str,
    world_size: int,
    engine_type: EngineType,
    layout_hints: LayoutHints,
    engine_group_infos: list[EngineGroupInfo],
) -> None:
    """
    Dummy handler for REGISTER_KV_CACHE requests.

    Args:
        gpu_id: GPU device ID
        kv_cache: List of CudaIPCWrapper objects representing KV cache
        model_name: Name of the model associated with this KV cache
        world_size: World size associated with this KV cache
        engine_type: Which serving engine produced the caches
        layout_hints: Engine-provided hints dict.
        engine_group_infos: Engine-neutral KV cache group metadata,
            msgspec-decoded from the request payload.

    Returns:
        None
    """
    # In a real implementation, this would register the KV cache
    # For testing, we just validate the inputs are received correctly
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert isinstance(kv_cache, list), (
        f"Expected kv_cache to be list, got {type(kv_cache)}"
    )
    assert isinstance(model_name, str), (
        f"Expected model_name to be str, got {type(model_name)}"
    )
    assert isinstance(world_size, int), (
        f"Expected world_size to be int, got {type(world_size)}"
    )
    assert isinstance(engine_type, EngineType), (
        f"Expected engine_type to be EngineType, got {type(engine_type)}"
    )
    assert isinstance(layout_hints, dict), (
        f"Expected layout_hints to be dict, got {type(layout_hints)}"
    )
    assert isinstance(engine_group_infos, list), (
        f"Expected engine_group_infos to be a list, got {type(engine_group_infos)}"
    )
    # No return value (returns None implicitly)


# ==============================================================================
# UNREGISTER_KV_CACHE Request Handlers
# ==============================================================================


def unregister_kv_cache_handler(gpu_id: int) -> None:
    """
    Dummy handler for UNREGISTER_KV_CACHE requests.

    Args:
        gpu_id: GPU device ID

    Returns:
        None
    """
    # In a real implementation, this would unregister the KV cache for the given GPU
    # For testing, we just validate the input is received correctly
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    # No return value (returns None implicitly)


# ==============================================================================
# STORE Request Handlers
# ==============================================================================


def store_handler(
    key: KeyType, gpu_id: int, gpu_block_ids: list[list[int]], ipc_handle: bytes
) -> tuple[bytes, bool]:
    """
    Dummy handler for STORE requests.

    Args:
        key: Cache key to store
        gpu_id: GPU device ID
        gpu_block_ids: GPU block IDs per KV cache group
        ipc_handle: CUDA event IPC handle

    Returns:
        tuple[bytes, bool]: (event handle, success flag)
    """
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert isinstance(gpu_block_ids, list), (
        f"Expected gpu_block_ids to be list, got {type(gpu_block_ids)}"
    )
    assert all(isinstance(block_ids, list) for block_ids in gpu_block_ids), (
        "Expected gpu_block_ids to be list[list[int]]"
    )
    assert isinstance(ipc_handle, bytes), (
        f"Expected ipc_handle to be bytes, got {type(ipc_handle)}"
    )
    return b"\x01" * 64, True


# ==============================================================================
# RETRIEVE Request Handlers
# ==============================================================================


def retrieve_handler(
    key: KeyType,
    gpu_id: int,
    gpu_block_ids: list[list[int]],
    event_handler: bytes,
    skip_first_n_tokens: int = 0,
) -> tuple[bytes, bool]:
    """
    Dummy handler for RETRIEVE requests.

    Args:
        key: Cache key to retrieve
        gpu_id: GPU device ID
        gpu_block_ids: GPU block IDs per KV cache group
        event_handler: CUDA event IPC handle
        skip_first_n_tokens: Number of tokens to skip at retrieve start

    Returns:
        tuple[bytes, bool]: (event handle, success flag)
    """
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert isinstance(gpu_block_ids, list), (
        f"Expected gpu_block_ids to be list, got {type(gpu_block_ids)}"
    )
    assert all(isinstance(block_ids, list) for block_ids in gpu_block_ids), (
        "Expected gpu_block_ids to be list[list[int]]"
    )
    assert isinstance(event_handler, bytes), (
        f"Expected event_handler to be bytes, got {type(event_handler)}"
    )
    assert isinstance(skip_first_n_tokens, int), (
        f"Expected skip_first_n_tokens to be int, got {type(skip_first_n_tokens)}"
    )
    return b"\x01" * 64, True


# ==============================================================================
# LOOKUP Request Handlers
# ==============================================================================


def lookup_handler(key: KeyType, tp_size: int) -> None:
    """
    Dummy handler for LOOKUP requests.

    Args:
        key: Cache key to look up (request_id embedded in the key)
        tp_size: Tensor-parallel size for MLA
            multi-reader locking

    Returns:
        None: LOOKUP registers the job server-side; poll via QUERY_PREFETCH_STATUS.
    """
    # In a real implementation, this would look up the key in the cache
    # For testing, we just validate the input
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(tp_size, int), f"Expected tp_size to be int, got {type(tp_size)}"


# ==============================================================================
# FREE_LOOKUP_LOCKS Request Handlers
# ==============================================================================


def free_locks_handler(key: KeyType, tp_size: int) -> None:
    """
    Dummy handler for FREE_LOOKUP_LOCKS requests.

    Args:
        key: Cache key whose read locks should be released
        tp_size: Tensor-parallel size for MLA
            multi-reader locking

    Returns:
        None
    """
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(tp_size, int), f"Expected tp_size to be int, got {type(tp_size)}"


# ==============================================================================
# REPORT_BLOCK_ALLOCATION Request Handlers
# ==============================================================================


def report_block_allocations_handler(
    instance_id: int,
    model_name: str,
    records: list[BlockAllocationRecord],
) -> None:
    """
    Dummy handler for REPORT_BLOCK_ALLOCATION requests.

    Args:
        instance_id: The scheduler instance ID.
        model_name: The model name from the adapter.
        records: List of BlockAllocationRecord with per-request
            block and token allocation deltas.

    Returns:
        None
    """
    assert isinstance(records, list), (
        f"Expected records to be list, got {type(records)}"
    )
    for rec in records:
        assert isinstance(rec, BlockAllocationRecord), (
            f"Expected BlockAllocationRecord, got {type(rec)}"
        )
        assert isinstance(rec.req_id, str)
        assert isinstance(rec.new_block_ids, list)
        assert isinstance(rec.new_token_ids, list)
