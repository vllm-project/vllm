# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import threading
from unittest.mock import MagicMock, patch

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_metrics import (  # noqa: E501
    MooncakeStoreConnectorStats,
)


def _make_store_sending_thread(
    store: MagicMock,
) -> mooncake_store_worker.KVCacheStoreSendingThread:
    token_database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), block_size=16
    )
    token_database.set_kv_caches_base_addr([0x1000])
    token_database.set_block_len([256])
    thread = mooncake_store_worker.KVCacheStoreSendingThread(
        store=store,
        token_database=token_database,
        block_size=16,
        tp_rank=0,
        put_step=1,
        kv_role="kv_producer",
        ready_event=threading.Event(),
    )
    thread.request_queue.task_done = MagicMock()
    return thread


def _make_store_recving_thread(
    store: MagicMock,
    *,
    disk_offload_buffer_budget_bytes: int | None = None,
) -> mooncake_store_worker.KVCacheStoreRecvingThread:
    token_database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), block_size=16
    )
    token_database.set_kv_caches_base_addr([0x1000])
    token_database.set_block_len([256])
    thread = mooncake_store_worker.KVCacheStoreRecvingThread(
        store=store,
        token_database=token_database,
        block_size=16,
        tp_rank=0,
        ready_event=threading.Event(),
        disk_offload_buffer_budget_bytes=disk_offload_buffer_budget_bytes,
    )
    thread.request_queue.task_done = MagicMock()
    return thread


def _make_load_req(
    req_id: str,
    block_hashes: list[bytes],
    *,
    token_len: int,
    vllm_cached_tokens: int = 0,
) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=token_len,
        block_ids=list(range(len(block_hashes))),
        block_hashes=block_hashes,
        load_spec=LoadSpec(
            vllm_cached_tokens=vllm_cached_tokens,
            kvpool_cached_tokens=token_len,
            can_load=True,
            token_len=token_len,
        ),
    )


def _make_store_req(req_id: str, block_hashes: list[bytes]) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=32,
        block_ids=[0, 1],
        block_hashes=block_hashes,
        can_save=True,
        original_block_size=16,
    )


_DISK_OFFLOAD_SINGLE_KEY_BYTES = (
    mooncake_store_worker._estimate_disk_offload_staging_bytes([256])
)
_DISK_OFFLOAD_USABLE_BUDGET_RATIO = 0.9
_DISK_OFFLOAD_BUDGET_FOR_THREE_KEYS = 4 * _DISK_OFFLOAD_SINGLE_KEY_BYTES
_DISK_OFFLOAD_BUDGET_FOR_SPLIT = math.ceil(
    2 * _DISK_OFFLOAD_SINGLE_KEY_BYTES / _DISK_OFFLOAD_USABLE_BUDGET_RATIO
)  # Allows two 256-byte chunks but not the third.
_DISK_OFFLOAD_BUDGET_TOO_SMALL = (
    _DISK_OFFLOAD_SINGLE_KEY_BYTES - 1
)  # Smaller than a single 256-byte chunk.


def test_store_sending_thread_skips_request_during_cpu_pressure():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [
        [-200, -200],
        [256, 256],
        [256, 256],
    ]
    thread = _make_store_sending_thread(store)

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a0", b"a1"]))

    assert thread._store_pressure_active is True
    assert "req-a" in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 1

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a2", b"a3"]))

    assert store.batch_put_from_multi_buffers.call_count == 1

    thread.add_stored_request("req-b")
    thread._handle_request(_make_store_req("req-b", [b"b0", b"b1"]))

    assert thread._store_pressure_active is False
    assert "req-a" not in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 2

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a4", b"a5"]))

    assert store.batch_put_from_multi_buffers.call_count == 3


def test_store_sending_thread_only_skips_on_no_available_handle():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [
        [-500, -500],
        [256, 256],
    ]
    thread = _make_store_sending_thread(store)

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a0", b"a1"]))

    assert thread._store_pressure_active is False
    assert "req-a" not in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 1

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a2", b"a3"]))

    assert store.batch_put_from_multi_buffers.call_count == 2


def test_store_sending_thread_records_mooncake_metrics():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    thread = _make_store_sending_thread(store)
    stats = MooncakeStoreConnectorStats()
    thread._record_operation_cb = stats.record_operation

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a0", b"a1"]))

    assert len(stats.data["save_exists"]) == 1
    assert stats.data["save_exists"][0]["num_keys"] == 2
    assert len(stats.data["save_put"]) == 1
    assert stats.data["save_put"][0]["num_bytes"] == 512
    assert stats.data["save_put"][0]["status"] == "ok"


def test_get_disk_offload_buffer_budget_bytes_uses_effective_offload_flag(
    monkeypatch,
):
    monkeypatch.delenv("MOONCAKE_ENABLE_OFFLOAD", raising=False)
    monkeypatch.setenv("MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES", "2mb")

    assert (
        mooncake_store_worker._get_disk_offload_buffer_budget_bytes(enable_offload=True)
        == 2 * 1024 * 1024
    )
    assert (
        mooncake_store_worker._get_disk_offload_buffer_budget_bytes(
            enable_offload=False
        )
        is None
    )


def test_estimate_disk_offload_staging_bytes_sums_multi_segment_sizes():
    assert (
        mooncake_store_worker._estimate_disk_offload_staging_bytes([256, 512]) == 12288
    )


def test_recv_thread_uses_single_batch_when_no_disk_offload_budget():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256, 256]
    thread = _make_store_recving_thread(store, disk_offload_buffer_budget_bytes=None)

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 1
    keys, addrs, sizes = store.batch_get_into_multi_buffers.call_args.args
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6132",
    ]
    assert sizes == [[256], [256], [256]]


def test_recv_thread_records_partial_failure_metrics():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, -10]
    thread = _make_store_recving_thread(store, disk_offload_buffer_budget_bytes=None)
    stats = MooncakeStoreConnectorStats()
    thread._record_operation_cb = stats.record_operation

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1"],
        token_len=32,
    )

    thread._handle_request(req)

    assert len(stats.data["load_get"]) == 1
    assert stats.data["load_get"][0]["num_keys"] == 2
    assert stats.data["load_get"][0]["num_bytes"] == 512
    assert stats.data["load_get"][0]["status"] == "partial_failure"
    assert stats.data["load_get"][0]["num_failed_keys"] == 1


def test_recv_thread_uses_ratio_scaled_budget_for_first_pass_split():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = [
        [256],
        [256],
    ]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=2 * _DISK_OFFLOAD_SINGLE_KEY_BYTES,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1"],
        token_len=32,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 2
    first_keys = store.batch_get_into_multi_buffers.call_args_list[0].args[0]
    second_keys = store.batch_get_into_multi_buffers.call_args_list[1].args[0]
    assert first_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6130",
    ]
    assert second_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6131",
    ]


def test_recv_thread_splits_disk_offload_loads_by_budget():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = [
        [256, 256],
        [256],
    ]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_SPLIT,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 2

    first_keys = store.batch_get_into_multi_buffers.call_args_list[0].args[0]
    second_keys = store.batch_get_into_multi_buffers.call_args_list[1].args[0]
    first_addrs = store.batch_get_into_multi_buffers.call_args_list[0].args[1]
    second_addrs = store.batch_get_into_multi_buffers.call_args_list[1].args[1]
    first_sizes = store.batch_get_into_multi_buffers.call_args_list[0].args[2]
    second_sizes = store.batch_get_into_multi_buffers.call_args_list[1].args[2]
    assert first_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6131",
    ]
    assert second_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6132",
    ]
    base_addr = thread.token_database.kv_caches_base_addr[0]
    block_len = thread.token_database.block_len[0]
    assert first_addrs == [[base_addr], [base_addr + block_len]]
    assert second_addrs == [[base_addr + 2 * block_len]]
    expected_size = block_len
    assert first_sizes == [[expected_size], [expected_size]]
    assert second_sizes == [[expected_size]]


def test_recv_thread_stops_after_first_failing_disk_offload_sub_batch():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [-10, -10]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_SPLIT,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 1


def test_recv_thread_uses_soft_key_cap_for_disk_offload_split():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = [
        [256, 256],
        [256],
    ]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_THREE_KEYS,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 2
    assert store.batch_get_into_multi_buffers.call_args_list[0].args[0] == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6131",
    ]
    assert store.batch_get_into_multi_buffers.call_args_list[1].args[0] == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6132",
    ]


def test_recv_thread_reports_unsplittable_key_larger_than_budget():
    store = MagicMock()
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_TOO_SMALL,
    )

    req = _make_load_req(
        "req-a",
        [b"a0"],
        token_len=16,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 0


# ---------------------------------------------------------------------------
# Helpers for register_kv_caches tests
# ---------------------------------------------------------------------------


def _auto_set_ready_event(*args, **kwargs):
    """Side effect for mocked thread constructors that auto-sets ready_event."""
    for arg in args:
        if isinstance(arg, threading.Event):
            arg.set()
    for val in kwargs.values():
        if isinstance(val, threading.Event):
            val.set()
    return MagicMock()


def _make_bare_worker(
    *,
    num_gpu_blocks: int = 10,
    block_size: int = 16,
    kv_role: str = "kv_both",
) -> mooncake_store_worker.MooncakeStoreWorker:
    """Construct a MooncakeStoreWorker via __new__, bypassing __init__.

    Sets only the attributes that register_kv_caches() reads so we can
    test the stride-based layout detection without a real
    MooncakeDistributedStore.
    """
    worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    worker.cache_config = MagicMock()
    worker.cache_config.num_gpu_blocks = num_gpu_blocks
    worker.store = MagicMock()
    worker.store.register_buffer.return_value = 0
    worker.use_mla = False
    worker.token_database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), block_size=block_size
    )
    worker.kv_role = kv_role
    worker.block_size = block_size
    worker.tp_rank = 0
    worker.put_step = 1
    worker.enable_kv_events = False
    worker.disk_offload_buffer_budget_bytes = None
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    worker._kv_connector_stats_lock = threading.Lock()
    worker.kv_connector_stats = MooncakeStoreConnectorStats()
    worker.tp_size = 1
    worker.num_kv_head = 1
    worker.pp_size = 1
    return worker


def test_mooncake_store_stats_aggregate_reduce():
    stats = MooncakeStoreConnectorStats()
    stats.record_operation("save_put", 0.01, 2, num_bytes=128)
    other = MooncakeStoreConnectorStats()
    other.record_operation(
        "save_put",
        0.03,
        1,
        num_bytes=64,
        status="error",
        num_failed_keys=1,
    )

    reduced = stats.aggregate(other).reduce()

    assert reduced["save_put_count"] == 2
    assert reduced["save_put_total_keys"] == 3
    assert reduced["save_put_total_bytes"] == 192
    assert reduced["save_put_failed_keys"] == 1
    assert reduced["save_put_error_count"] == 1


def test_worker_get_kv_connector_stats_resets_after_read():
    worker = _make_bare_worker()
    worker._record_kv_connector_operation(
        "save_put",
        0.01,
        2,
        num_bytes=128,
    )

    stats = worker.get_kv_connector_stats()

    assert isinstance(stats, MooncakeStoreConnectorStats)
    assert stats.data["save_put"][0]["num_bytes"] == 128
    assert worker.get_kv_connector_stats() is None


def test_lookup_records_mooncake_metrics():
    worker = _make_bare_worker()
    worker.store.batch_is_exist.return_value = [1, 1]

    result = worker.lookup(32, [b"a0", b"a1"])
    stats = worker.get_kv_connector_stats()

    assert result == 32
    assert isinstance(stats, MooncakeStoreConnectorStats)
    assert len(stats.data["lookup_exists"]) == 1
    assert stats.data["lookup_exists"][0]["num_keys"] == 2


# ---------------------------------------------------------------------------
# register_kv_caches tests
# ---------------------------------------------------------------------------


def test_register_kv_caches_blocks_first_single_segment():
    """Blocks-first layout (FlashInfer/MLA): one segment per layer."""
    num_blocks = 10
    page_size_elements = 64  # elements per block
    worker = _make_bare_worker(num_gpu_blocks=num_blocks)

    # Shape: (num_blocks, page_size_elements) — blocks outermost, no outer_dims
    tensor = torch.zeros(num_blocks, page_size_elements, dtype=torch.float16)

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        worker.register_kv_caches({"layer0": tensor})

    assert len(worker.kv_caches_base_addr) == 1
    assert worker.kv_caches_base_addr[0] == tensor.untyped_storage().data_ptr()

    expected_block_len = tensor.untyped_storage().nbytes() // num_blocks
    assert len(worker.block_len) == 1
    assert worker.block_len[0] == expected_block_len

    worker.store.register_buffer.assert_called_once_with(
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
    )


def test_register_kv_caches_kv_first_two_segments():
    """K/V-first layout (FlashAttn): two segments (K, V) per layer."""
    num_blocks = 10
    block_size_tokens = 16
    num_kv_heads = 4
    head_size = 8

    worker = _make_bare_worker(num_gpu_blocks=num_blocks)

    # Shape: (2, num_blocks, block_size, num_kv_heads, head_size) — K/V outermost
    tensor = torch.zeros(
        2,
        num_blocks,
        block_size_tokens,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
    )

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        worker.register_kv_caches({"layer0": tensor})

    # K/V-first: dim 0 has stride > page_size, so 2 segments
    assert len(worker.kv_caches_base_addr) == 2
    assert len(worker.block_len) == 2

    el = tensor.element_size()
    seg_stride = tensor.stride(0) * el  # stride of the K/V dim in bytes
    base = tensor.untyped_storage().data_ptr()
    assert worker.kv_caches_base_addr[0] == base
    assert worker.kv_caches_base_addr[1] == base + seg_stride
    assert worker.block_len[0] == seg_stride // num_blocks
    assert worker.block_len[1] == seg_stride // num_blocks


def test_register_kv_caches_cross_layer_single_segment():
    """Cross-layer tensor: single segment with block_len = page_size * num_layers."""
    num_blocks = 10
    num_layers = 4
    per_layer_page_elements = 64  # elements per layer per block

    worker = _make_bare_worker(num_gpu_blocks=num_blocks)

    # Cross-layer blocks-first tensor: all layers packed into a single
    # contiguous block.  Shape (num_blocks, num_layers * per_layer_page)
    # mimics the physical layout after stride reordering.
    total_page_elements = num_layers * per_layer_page_elements
    tensor = torch.zeros(num_blocks, total_page_elements, dtype=torch.float16)

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        # Use the cross-layer wrapper key, same as register_cross_layers_kv_caches
        worker.register_kv_caches({"__cross_layer__": tensor})

    assert len(worker.kv_caches_base_addr) == 1
    assert worker.kv_caches_base_addr[0] == tensor.untyped_storage().data_ptr()

    expected_block_len = tensor.untyped_storage().nbytes() // num_blocks
    # block_len should be per_layer_page_size * num_layers
    assert (
        expected_block_len
        == num_layers * per_layer_page_elements * tensor.element_size()
    )
    assert len(worker.block_len) == 1
    assert worker.block_len[0] == expected_block_len

    # Also verify via register_cross_layers_kv_caches wrapper
    worker2 = _make_bare_worker(num_gpu_blocks=num_blocks)
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        worker2.register_cross_layers_kv_caches(tensor)

    assert worker2.kv_caches_base_addr == worker.kv_caches_base_addr
    assert worker2.block_len == worker.block_len
