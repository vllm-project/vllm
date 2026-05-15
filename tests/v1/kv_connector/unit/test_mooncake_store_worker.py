# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from unittest.mock import MagicMock, patch

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
    ChunkedTokenDatabase,
    KeyMetadata,
    ReqMeta,
)


def _make_store_sending_thread(
    store: MagicMock,
) -> worker.KVCacheStoreSendingThread:
    token_database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), block_size=16
    )
    token_database.set_kv_caches_base_addr([0x1000])
    token_database.set_block_len([256])
    thread = worker.KVCacheStoreSendingThread(
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


def _make_store_req(req_id: str, block_hashes: list[bytes]) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=32,
        block_ids=[0, 1],
        block_hashes=block_hashes,
        can_save=True,
        original_block_size=16,
    )


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
) -> worker.MooncakeStoreWorker:
    """Construct a MooncakeStoreWorker via __new__, bypassing __init__.

    Sets only the attributes that register_kv_caches() reads so we can
    test the stride-based layout detection without a real
    MooncakeDistributedStore.
    """
    w = object.__new__(worker.MooncakeStoreWorker)
    w.cache_config = MagicMock()
    w.cache_config.num_gpu_blocks = num_gpu_blocks
    w.store = MagicMock()
    w.store.register_buffer.return_value = 0
    w.use_mla = False
    w.token_database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), block_size=block_size
    )
    w.kv_role = kv_role
    w.block_size = block_size
    w.tp_rank = 0
    w.put_step = 1
    w.enable_kv_events = False
    w.kv_send_thread = None
    w.kv_recv_thread = None
    return w


# ---------------------------------------------------------------------------
# register_kv_caches tests
# ---------------------------------------------------------------------------


def test_register_kv_caches_blocks_first_single_segment():
    """Blocks-first layout (FlashInfer/MLA): one segment per layer."""
    num_blocks = 10
    page_size_elements = 64  # elements per block
    w = _make_bare_worker(num_gpu_blocks=num_blocks)

    # Shape: (num_blocks, page_size_elements) — blocks outermost, no outer_dims
    tensor = torch.zeros(num_blocks, page_size_elements, dtype=torch.float16)

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        w.register_kv_caches({"layer0": tensor})

    assert len(w.kv_caches_base_addr) == 1
    assert w.kv_caches_base_addr[0] == tensor.untyped_storage().data_ptr()

    expected_block_len = tensor.untyped_storage().nbytes() // num_blocks
    assert len(w.block_len) == 1
    assert w.block_len[0] == expected_block_len

    w.store.register_buffer.assert_called_once_with(
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
    )


def test_register_kv_caches_kv_first_two_segments():
    """K/V-first layout (FlashAttn): two segments (K, V) per layer."""
    num_blocks = 10
    block_size_tokens = 16
    num_kv_heads = 4
    head_size = 8

    w = _make_bare_worker(num_gpu_blocks=num_blocks)

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
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        w.register_kv_caches({"layer0": tensor})

    # K/V-first: dim 0 has stride > page_size, so 2 segments
    assert len(w.kv_caches_base_addr) == 2
    assert len(w.block_len) == 2

    el = tensor.element_size()
    seg_stride = tensor.stride(0) * el  # stride of the K/V dim in bytes
    base = tensor.untyped_storage().data_ptr()
    assert w.kv_caches_base_addr[0] == base
    assert w.kv_caches_base_addr[1] == base + seg_stride
    assert w.block_len[0] == seg_stride // num_blocks
    assert w.block_len[1] == seg_stride // num_blocks


def test_register_kv_caches_cross_layer_single_segment():
    """Cross-layer tensor: single segment with block_len = page_size * num_layers."""
    num_blocks = 10
    num_layers = 4
    per_layer_page_elements = 64  # elements per layer per block

    w = _make_bare_worker(num_gpu_blocks=num_blocks)

    # Cross-layer blocks-first tensor: all layers packed into a single
    # contiguous block.  Shape (num_blocks, num_layers * per_layer_page)
    # mimics the physical layout after stride reordering.
    total_page_elements = num_layers * per_layer_page_elements
    tensor = torch.zeros(num_blocks, total_page_elements, dtype=torch.float16)

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        # Use the cross-layer wrapper key, same as register_cross_layers_kv_caches
        w.register_kv_caches({"__cross_layer__": tensor})

    assert len(w.kv_caches_base_addr) == 1
    assert w.kv_caches_base_addr[0] == tensor.untyped_storage().data_ptr()

    expected_block_len = tensor.untyped_storage().nbytes() // num_blocks
    # block_len should be per_layer_page_size * num_layers
    assert (
        expected_block_len
        == num_layers * per_layer_page_elements * tensor.element_size()
    )
    assert len(w.block_len) == 1
    assert w.block_len[0] == expected_block_len

    # Also verify via register_cross_layers_kv_caches wrapper
    w2 = _make_bare_worker(num_gpu_blocks=num_blocks)
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        w2.register_cross_layers_kv_caches(tensor)

    assert w2.kv_caches_base_addr == w.kv_caches_base_addr
    assert w2.block_len == w.block_len
