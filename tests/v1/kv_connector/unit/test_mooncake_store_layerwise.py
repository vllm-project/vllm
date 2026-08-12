# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MooncakeStoreConnector layerwise KV cache support."""

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as mooncake_store_worker,
)

# ============================================================================
# Mock helpers for _build_layer_tasks and sync tests
# ============================================================================

import threading
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerTransferTask,
    ReqMeta,
    LoadSpec,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
)


def _make_layerwise_bare_worker(
    *,
    num_layers: int = 2,
    num_groups: int = 1,
    block_size: int = 16,
    tp_rank: int = 0,
    put_step: int = 1,
) -> "mooncake_store_worker.MooncakeStoreWorker":  # noqa: F821
    """Construct a minimal MooncakeStoreWorker with layerwise attributes."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
        worker as mooncake_store_worker,
    )

    worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    worker._layerwise_enabled = True
    worker._num_layers = num_layers
    worker.tp_rank = tp_rank
    worker._group_tp_replication_factors = tuple([put_step] * num_groups)
    worker.block_size = block_size
    worker.num_blocks = 10
    worker.kv_send_thread = None
    worker.kv_recv_threads = []
    worker.store = MagicMock()
    worker._kv_connector_stats_lock = threading.Lock()
    worker.kv_connector_stats = MagicMock()

    # Layerwise task / event dictionaries
    worker._layer_save_tasks = {l: [] for l in range(num_layers)}
    worker._layer_load_tasks = {l: [] for l in range(num_layers)}
    worker._layer_save_finished_events = {
        l: threading.Event() for l in range(num_layers)
    }
    worker._layer_load_finished_events = {
        l: threading.Event() for l in range(num_layers)
    }
    worker._current_save_layer = 0
    worker._current_load_layer = 0
    worker._next_load_layer_to_submit = 0
    worker._num_prefetch_layers = 1

    # Create token databases
    worker.token_dbs = []
    for g_idx in range(num_groups):
        md = KeyMetadata("test-model", tp_rank, 0, 0, 0, group_id=g_idx)
        db = ChunkedTokenDatabase(md, block_size=block_size, hash_block_size=block_size)
        db.set_kv_caches_base_addr([0x1000 + g_idx * 0x1000])
        db.set_block_len([256])
        worker.token_dbs.append(db)

    # Coordinator
    specs = [
        FullAttentionSpec(block_size=block_size, num_kv_heads=8, head_size=64, dtype=None)
        for _ in range(num_groups)
    ]
    groups = [KVCacheGroupSpec([f"layer{g}"], spec) for g, spec in enumerate(specs)]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        groups,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    return worker


def _make_test_reqmeta(
    req_id: str = "test_req",
    token_len: int = 32,
    block_ids: tuple[list[int], ...] | None = None,
    can_save: bool = True,
    can_load: bool = False,
    block_hashes: list[BlockHash] | None = None,
    num_prompt_tokens: int = 32,
) -> ReqMeta:
    """Create a ReqMeta for testing _build_layer_tasks_from_requests."""
    if block_hashes is None:
        block_hashes = [BlockHash(f"h{i:016d}".encode()) for i in range(32)]
    if block_ids is None:
        num_blocks = (token_len + 15) // 16
        block_ids = (list(range(num_blocks)),)
    load_spec = None
    if can_load:
        load_spec = LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=token_len,
            can_load=True,
            token_len=token_len,
        )
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=token_len,
        block_ids=block_ids,
        block_hashes=block_hashes,
        can_save=can_save,
        load_spec=load_spec,
        num_prompt_tokens=num_prompt_tokens,
    )

class TestSubmitReadyLayerLoads:
    """Test _submit_ready_layer_loads round-robin distribution."""

    def test_round_robin_across_recv_threads(self):
        """Tasks are distributed round-robin, not broadcast to all threads."""
        worker = _make_layerwise_bare_worker(num_layers=2)

        # Build some load tasks
        for layer_id in range(2):
            task = LayerTransferTask(
                req_id=f"req_{layer_id}",
                group_id=0,
                layer_idx_in_group=layer_id,
                physical_layer_id=layer_id,
                key_list=[f"key_{layer_id}"],
                addr_list=[[0x2000]],
                size_list=[[256]],
                block_ids=[layer_id],
                is_save=False,
            )
            worker._layer_load_tasks[layer_id].append(task)

        # Create mock recv threads
        mock_threads = [MagicMock() for _ in range(3)]
        worker.kv_recv_threads = mock_threads
        worker._num_prefetch_layers = 2

        worker._submit_ready_layer_loads()

        # Each task should be sent to exactly one thread
        for mt in mock_threads:
            assert len(mt.add_request.call_args_list) <= 2, (
                f"Thread received {len(mt.add_request.call_args_list)} tasks, "
                f"expected ≤2 (round-robin)"
            )

        # Total calls across all threads = total tasks = 2
        total_calls = sum(len(mt.add_request.call_args_list) for mt in mock_threads)
        assert total_calls == 2, f"Expected 2 total calls, got {total_calls}"


class TestLayerwiseStateReset:
    """Test _reset_layer_state event and state management."""

    def test_reset_creates_new_events(self):
        """_reset_layer_state replaces Event objects."""
        worker = _make_layerwise_bare_worker(num_layers=3)
        old_save_events = {
            l: worker._layer_save_finished_events[l] for l in range(3)
        }
        old_load_events = {
            l: worker._layer_load_finished_events[l] for l in range(3)
        }

        worker._reset_layer_state()

        for l in range(3):
            assert worker._layer_save_finished_events[l] is not old_save_events[l]
            assert worker._layer_load_finished_events[l] is not old_load_events[l]

    def test_reset_syncs_events_to_threads(self):
        """Layerwsie: new events are synced to transfer threads after reset."""
        worker = _make_layerwise_bare_worker(num_layers=2)

        # Create layerwise-enabled mock threads
        worker.kv_send_thread = MagicMock()
        worker.kv_send_thread._layerwise_enabled = True
        worker.kv_recv_threads = [
            MagicMock(_layerwise_enabled=True),
            MagicMock(_layerwise_enabled=False),  # one non-layerwise
        ]

        worker._reset_layer_state()

        # Sending thread: should receive 2 set_layer_finished_event calls (is_save=True)
        assert worker.kv_send_thread.set_layer_finished_event.call_count == 2
        for call_args in worker.kv_send_thread.set_layer_finished_event.call_args_list:
            assert call_args.args[1] is True  # is_save=True

        # Only the layerwise-enabled recv thread should receive calls
        thread0 = worker.kv_recv_threads[0]
        assert thread0.set_layer_finished_event.call_count == 2
        for call_args in thread0.set_layer_finished_event.call_args_list:
            assert call_args.args[1] is False  # is_save=False

        # Non-layerwise thread should not receive calls
        thread1 = worker.kv_recv_threads[1]
        assert thread1.set_layer_finished_event.call_count == 0


# ============================================================================
# Layerwise Save/Load Flow Tests
# Tests for save_kv_layer, wait_for_layer_load, and _handle_request
# ============================================================================

import time
from unittest.mock import MagicMock


def _make_layerwise_send_thread(
    store: MagicMock,
    *,
    num_layers: int = 2,
    block_size: int = 16,
) -> "mooncake_store_worker.KVCacheStoreSendingThread":
    """Create a KVCacheStoreSendingThread with layerwise enabled for testing."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
        worker as mooncake_store_worker,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    db = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), block_size=block_size
    )
    db.set_kv_caches_base_addr([0x1000])
    db.set_block_len([256])
    spec = FullAttentionSpec(block_size=block_size, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    thread = mooncake_store_worker.KVCacheStoreSendingThread(
        store=store,
        coord=coord,
        token_databases=[db],
        block_size=block_size,
        tp_rank=0,
        group_put_steps=[1],
        kv_role="kv_producer",
        ready_event=threading.Event(),
    )
    # Enable layerwise mode.
    thread.enable_layerwise(num_layers)
    # Replace task_done to prevent blocking.
    thread.request_queue.task_done = MagicMock()
    return thread


def _make_layerwise_recv_thread(
    store: MagicMock,
    *,
    num_layers: int = 2,
    block_size: int = 16,
    token_databases: list | None = None,
) -> "mooncake_store_worker.KVCacheStoreRecvingThread":
    """Create a KVCacheStoreRecvingThread with layerwise enabled for testing."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
        worker as mooncake_store_worker,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    if token_databases is None:
        db = ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0), block_size=block_size
        )
        db.set_kv_caches_base_addr([0x1000])
        db.set_block_len([256])
        token_databases = [db]

    spec = FullAttentionSpec(block_size=block_size, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    thread = mooncake_store_worker.KVCacheStoreRecvingThread(
        store=store,
        coord=coord,
        token_databases=token_databases,
        block_size=block_size,
        tp_rank=0,
        ready_event=threading.Event(),
        disk_offload_buffer_budget_bytes=None,
    )
    # Enable layerwise mode.
    thread.enable_layerwise(num_layers)
    thread.request_queue.task_done = MagicMock()
    return thread


class TestSaveKvLayer:
    """Test save_kv_layer submits tasks and handles last-layer completion."""

    def test_save_kv_layer_submits_tasks_to_send_thread(self):
        """save_kv_layer must send tasks for the matching layer to kv_send_thread."""
        worker = _make_layerwise_bare_worker(num_layers=4)
        worker.kv_send_thread = MagicMock()
        worker.kv_send_thread._layerwise_enabled = True

        # Pre-populate save tasks for each layer.
        for layer_id in range(4):
            task = LayerTransferTask(
                req_id="test_req",
                group_id=0,
                layer_idx_in_group=layer_id,
                physical_layer_id=layer_id,
                key_list=[f"layer{layer_id}_key"],
                addr_list=[[0x2000 + layer_id * 0x1000]],
                size_list=[[256]],
                block_ids=[layer_id],
                is_save=True,
            )
            worker._layer_save_tasks[layer_id].append(task)

        # Simulate the per-layer forward calls to save_kv_layer.
        for layer_id in range(4):
            worker.save_kv_layer(f"model.layers.{layer_id}.self_attn", None, None)

        # Each layer should have called add_request (4 total).
        assert worker.kv_send_thread.add_request.call_count == 4
        # Verify the submitted task layer_ids.
        for i, call_args in enumerate(worker.kv_send_thread.add_request.call_args_list):
            task = call_args.args[0]
            assert isinstance(task, LayerTransferTask)
            assert task.physical_layer_id == i

    def test_save_kv_layer_last_layer_triggers_reset(self):
        """Last layer must call _wait_for_all_layer_saves and _reset_layer_state."""
        worker = _make_layerwise_bare_worker(num_layers=2)
        worker.kv_send_thread = MagicMock()
        worker.kv_send_thread._layerwise_enabled = True

        # Pre-populate save tasks.
        for layer_id in range(2):
            task = LayerTransferTask(
                req_id="test_req1", group_id=0,
                layer_idx_in_group=layer_id, physical_layer_id=layer_id,
                key_list=["k"], addr_list=[[0x2000]], size_list=[[256]],
                block_ids=[layer_id], is_save=True,
            )
            worker._layer_save_tasks[layer_id].append(task)
            # Pre-set the save event (simulating transfer-thread completion).
            worker._layer_save_finished_events[layer_id].set()

        old_save_events = {l: worker._layer_save_finished_events[l] for l in range(2)}
        old_load_events = {l: worker._layer_load_finished_events[l] for l in range(2)}

        # Save the last layer.
        worker.save_kv_layer("model.layers.1.self_attn", None, None)

        # The last layer must trigger a reset: events are replaced ...
        for l in range(2):
            assert worker._layer_save_finished_events[l] is not old_save_events[l]
            assert worker._layer_load_finished_events[l] is not old_load_events[l]
        # ... and task lists are cleared.
        for l in range(2):
            assert worker._layer_save_tasks[l] == []
            assert worker._layer_load_tasks[l] == []


class TestWaitForLayerLoad:
    """Test wait_for_layer_load submits prefetch and waits for completion."""

    def test_wait_for_layer_load_submits_prefetch(self):
        """wait_for_layer_load must trigger _submit_ready_layer_loads."""
        worker = _make_layerwise_bare_worker(num_layers=4)
        worker.kv_recv_threads = [MagicMock()]

        # Pre-populate load tasks so _submit_ready_layer_loads has work.
        for layer_id in range(4):
            task = LayerTransferTask(
                req_id="test_req", group_id=0,
                layer_idx_in_group=layer_id, physical_layer_id=layer_id,
                key_list=["k"], addr_list=[[0x2000]], size_list=[[256]],
                block_ids=[layer_id], is_save=False,
            )
            worker._layer_load_tasks[layer_id].append(task)

        # Ensure each layer's event starts unset.
        for layer_id in range(4):
            worker._layer_load_finished_events[layer_id].clear()

        # Set the event from another thread (simulating async transfer completion).
        def _set_events_after_delay():
            time.sleep(0.05)
            worker._layer_load_finished_events[0].set()

        import threading as _thr
        setter = _thr.Thread(target=_set_events_after_delay, daemon=True)
        setter.start()

        # wait_for_layer_load blocks until layer 0 is loaded.
        worker.wait_for_layer_load("model.layers.0.self_attn")

        assert worker._current_load_layer == 1

    def test_wait_for_layer_load_prefetch_advances_counter(self):
        """_submit_ready_layer_loads advances _next_load_layer_to_submit."""
        worker = _make_layerwise_bare_worker(num_layers=4)
        worker.kv_recv_threads = [MagicMock()]

        # Pre-populate load tasks.
        for layer_id in range(4):
            task = LayerTransferTask(
                req_id="test_req", group_id=0,
                layer_idx_in_group=layer_id, physical_layer_id=layer_id,
                key_list=["k"], addr_list=[[0x2000]], size_list=[[256]],
                block_ids=[layer_id], is_save=False,
            )
            worker._layer_load_tasks[layer_id].append(task)

        # Pre-set the events.
        for i in range(4):
            worker._layer_load_finished_events[i].set()

        assert worker._next_load_layer_to_submit == 0

        # First call submits prefetch_layers layers.
        worker.wait_for_layer_load("model.layers.0.self_attn")
        # prefetch_layers=1, so only layer 0 was submitted.
        assert worker._next_load_layer_to_submit == 1
        assert worker._current_load_layer == 1

        # Second call submits layer 1.
        worker.wait_for_layer_load("model.layers.1.self_attn")
        assert worker._next_load_layer_to_submit == 2
        assert worker._current_load_layer == 2


class TestLayerwiseSendHandleLayerTask:
    """Test KVCacheStoreSendingThread._handle_request layerwise path."""

    def test_send_handle_request_puts_to_store(self):
        """Layerwise save must call batch_put_from_multi_buffers for non-existent keys."""
        store = MagicMock()
        store.batch_is_exist.return_value = [False, False]  # both keys absent
        store.batch_put_from_multi_buffers.return_value = True

        thread = _make_layerwise_send_thread(store, num_layers=2)
        layer_id = 1  # last layer — set_finished_request is only called on the final layer
        event = thread._layer_save_finished_events[layer_id]

        task = LayerTransferTask(
            req_id="test_send",
            group_id=0,
            layer_idx_in_group=layer_id,
            physical_layer_id=layer_id,
            key_list=["save_key_1", "save_key_2"],
            addr_list=[[0x3000], [0x4000]],
            size_list=[[256], [256]],
            block_ids=[1, 2],
            is_save=True,
        )

        thread._handle_request(task)

        # batch_is_exist dedup check should be called.
        store.batch_is_exist.assert_called_once_with(["save_key_1", "save_key_2"])
        # Only absent keys should be put.
        store.batch_put_from_multi_buffers.assert_called_once_with(
            ["save_key_1", "save_key_2"],
            [[0x3000], [0x4000]],
            [[256], [256]],
        )
        # Event should be set.
        assert event.is_set()
        # Request should be marked finished.
        assert "test_send" in thread.finished_requests

    def test_send_handle_request_skips_existing_keys(self):
        """Keys that already exist in store should be skipped (dedup)."""
        store = MagicMock()
        # First key exists, second does not.
        store.batch_is_exist.return_value = [True, False]
        store.batch_put_from_multi_buffers.return_value = True

        thread = _make_layerwise_send_thread(store, num_layers=1)
        event = thread._layer_save_finished_events[0]

        task = LayerTransferTask(
            req_id="test_dedup",
            group_id=0,
            layer_idx_in_group=0,
            physical_layer_id=0,
            key_list=["existing_key", "new_key"],
            addr_list=[[0x3000], [0x4000]],
            size_list=[[256], [256]],
            block_ids=[1, 2],
            is_save=True,
        )

        thread._handle_request(task)

        # Only the absent key should be put.
        store.batch_put_from_multi_buffers.assert_called_once_with(
            ["new_key"],  # only new_key
            [[0x4000]],   # its address
            [[256]],      # its size
        )
        assert event.is_set()


class TestLayerwiseRecvHandleLayerTask:
    """Test KVCacheStoreRecvingThread._handle_request layerwise path."""

    def test_recv_handle_request_gets_from_store(self):
        """Layerwise load must call batch_get_into_multi_buffers."""
        store = MagicMock()
        store.batch_get_into_multi_buffers.return_value = [256, 256]  # all succeed

        thread = _make_layerwise_recv_thread(store, num_layers=2)
        layer_id = 0
        event = thread._layer_load_finished_events[layer_id]

        task = LayerTransferTask(
            req_id="test_recv",
            group_id=0,
            layer_idx_in_group=layer_id,
            physical_layer_id=layer_id,
            key_list=["load_key_1", "load_key_2"],
            addr_list=[[0x3000], [0x4000]],
            size_list=[[256], [256]],
            block_ids=[1, 2],
            is_save=False,
        )

        thread._handle_request(task)

        store.batch_get_into_multi_buffers.assert_called_once_with(
            ["load_key_1", "load_key_2"],
            [[0x3000], [0x4000]],
            [[256], [256]],
        )
        assert event.is_set()

    def test_recv_handle_request_load_failure_tracks_block_ids(self):
        """Failed blocks should be tracked via _add_load_error_block_ids."""
        store = MagicMock()
        # Second key fails to load (-5 indicates failure).
        store.batch_get_into_multi_buffers.return_value = [256, -5]

        thread = _make_layerwise_recv_thread(store, num_layers=2)
        layer_id = 0
        event = thread._layer_load_finished_events[layer_id]

        task = LayerTransferTask(
            req_id="test_fail_load",
            group_id=0,
            layer_idx_in_group=layer_id,
            physical_layer_id=layer_id,
            key_list=["ok_key", "fail_key"],
            addr_list=[[0x3000], [0x4000]],
            size_list=[[256], [256]],
            block_ids=[10, 20],  # second chunk fails
            is_save=False,
        )

        thread._handle_request(task)

        # The failed block_id (20) should be recorded.
        assert event.is_set()
        failed_blocks = thread.get_and_clear_block_ids_with_load_errors()
        assert 20 in failed_blocks


class TestSaveKvLayerIntegration:
    """Integration tests: worker.save_kv_layer -> send thread _handle_request."""

    def test_save_flow_integration(self):
        """Full integration: populated tasks -> save_kv_layer -> thread processes."""
        store = MagicMock()
        store.batch_is_exist.return_value = [False, False]
        store.batch_put_from_multi_buffers.return_value = True

        worker = _make_layerwise_bare_worker(num_layers=2)
        # Attach the real send thread to the worker.
        send_thread = _make_layerwise_send_thread(store, num_layers=2)
        worker.kv_send_thread = send_thread

        # Sync worker events to the send thread (as register_kv_caches() does).
        for layer_id in range(2):
            send_thread.set_layer_finished_event(
                layer_id, True, worker._layer_save_finished_events[layer_id]
            )

        # Build tasks via _build_layer_tasks_from_requests.
        req = _make_test_reqmeta(token_len=32)
        worker._build_layer_tasks_from_requests([req])

        # Verify tasks were built.
        assert len(worker._layer_save_tasks[0]) == 1
        assert len(worker._layer_save_tasks[1]) == 1

        # Simulate the forward pass, saving layer by layer.
        worker.save_kv_layer("model.layers.0.self_attn", None, None)
        # Layer 0's tasks are queued but not yet processed.

        # Manually drain the send thread's queue.
        while not send_thread.request_queue.empty():
            item = send_thread.request_queue.get_nowait()
            send_thread._handle_request(item)
            send_thread.request_queue.task_done()

        # Layer 0's event should be set (the send thread uses the worker's event).
        assert worker._layer_save_finished_events[0].is_set()
        store.batch_put_from_multi_buffers.assert_called()

        # Reset mocks, then handle layer 1.
        store.reset_mock()
        store.batch_is_exist.return_value = [False]
        store.batch_put_from_multi_buffers.return_value = True

        worker.save_kv_layer("model.layers.1.self_attn", None, None)

        while not send_thread.request_queue.empty():
            item = send_thread.request_queue.get_nowait()
            send_thread._handle_request(item)
            send_thread.request_queue.task_done()

        # Layer 1's event should be set.
        assert worker._layer_save_finished_events[1].is_set()


class TestLoadFlowIntegration:
    """Integration tests: worker.wait_for_layer_load -> recv thread _handle_request."""

    def test_load_flow_integration(self):
        """Full integration: populated tasks -> wait_for_layer_load -> thread processes."""
        store = MagicMock()
        store.batch_get_into_multi_buffers.return_value = [256, 256]

        worker = _make_layerwise_bare_worker(num_layers=2)
        recv_thread = _make_layerwise_recv_thread(store, num_layers=2)
        worker.kv_recv_threads = [recv_thread]

        # Build load tasks.
        req = _make_test_reqmeta(can_save=False, can_load=True, token_len=32)
        worker._build_layer_tasks_from_requests([req])

        assert len(worker._layer_load_tasks[0]) == 1

        # Start a background thread to drain the recv thread's queue.
        def _consume_recv_queue():
            while not recv_thread.request_queue.empty():
                item = recv_thread.request_queue.get_nowait()
                recv_thread._handle_request(item)
                recv_thread.request_queue.task_done()

        consumer = threading.Thread(target=_consume_recv_queue, daemon=True)
        consumer.start()

        # wait_for_layer_load blocks until the event is set, so the consumer
        # must be running before we call it.
        consumer.join()

        # Manually set the event (simulating consumer completion).
        worker._layer_load_finished_events[0].set()

        old_count = worker._current_load_layer
        worker.wait_for_layer_load("model.layers.0.self_attn")

        assert worker._current_load_layer == old_count + 1

    def test_load_failure_propagates_to_worker(self):
        """Load failures in recv thread should propagate to worker."""
        store = MagicMock()
        # The key fails to load.
        store.batch_get_into_multi_buffers.return_value = [-5]

        worker = _make_layerwise_bare_worker(num_layers=1, num_groups=1)
        recv_thread = _make_layerwise_recv_thread(store, num_layers=1)
        worker.kv_recv_threads = [recv_thread]

        task = LayerTransferTask(
            req_id="test_fail_prop",
            group_id=0,
            layer_idx_in_group=0,
            physical_layer_id=0,
            key_list=["fail_key"],
            addr_list=[[0x3000]],
            size_list=[[256]],
            block_ids=[42],
            is_save=False,
        )
        recv_thread._handle_request(task)

        # Verify the worker can observe the failure.
        failed_blocks = worker.get_block_ids_with_load_errors()
        assert 42 in failed_blocks


class TestSendingThreadSessionApi:
    """Test KVCacheStoreSendingThread session API methods."""

    def test_start_put_sessions_calls_batch_put_session_start(self):
        """start_put_sessions calls batch_put_session_start with correct args."""
        store = MagicMock()
        store.batch_put_session_start.return_value = [0, 0]  # both succeed
        thread = _make_layerwise_send_thread(store, num_layers=2)
        thread._use_session_api = True

        keys = ["key_a", "key_b"]
        object_size = 8192
        thread.start_put_sessions(keys, object_size)

        store.batch_put_session_start.assert_called_once()
        called_keys, called_sizes, _ = store.batch_put_session_start.call_args[0]
        assert called_keys == keys
        assert called_sizes == [object_size, object_size]

    def test_start_put_sessions_partial_failure_revokes(self):
        """Failed session starts are revoked."""
        store = MagicMock()
        store.batch_put_session_start.return_value = [0, -1]  # second fails
        thread = _make_layerwise_send_thread(store, num_layers=1)
        thread._use_session_api = True

        thread.start_put_sessions(["ok_key", "fail_key"], 4096)

        # Failed key should be revoked
        store.batch_put_session_revoke.assert_called_once_with(["fail_key"])

    def test_handle_request_dispatches_session_path(self):
        """use_key_major_ranges=True dispatches to session handler."""
        store = MagicMock()
        store.batch_put_from_multi_buffer_ranges.return_value = [256]
        thread = _make_layerwise_send_thread(store, num_layers=2)
        thread._use_session_api = True
        thread._active_put_keys = {"block_key_0", "block_key_1"}

        task = LayerTransferTask(
            req_id="test_session",
            group_id=0,
            layer_idx_in_group=0,
            physical_layer_id=0,
            key_list=["block_key_0", "block_key_1"],
            addr_list=[[0x3000], [0x4000]],
            size_list=[[256], [256]],
            dst_offset_list=[[0], [256]],
            block_ids=[1, 2],
            is_save=True,
            use_key_major_ranges=True,
        )

        thread._handle_request(task)

        store.batch_put_from_multi_buffer_ranges.assert_called_once()
        assert thread._layer_save_finished_events[0].is_set()

    def test_handle_request_session_revokes_failed_keys(self):
        """Failed range-put keys are revoked and excluded from next layers."""
        store = MagicMock()
        # key_1 succeeds, key_2 fails
        store.batch_put_from_multi_buffer_ranges.return_value = [256, -5]
        thread = _make_layerwise_send_thread(store, num_layers=2)
        thread._use_session_api = True
        thread._active_put_keys = {"key_1", "key_2"}

        task = LayerTransferTask(
            req_id="test_revoke",
            group_id=0,
            layer_idx_in_group=0,
            physical_layer_id=0,
            key_list=["key_1", "key_2"],
            addr_list=[[0x3000], [0x4000]],
            size_list=[[256], [256]],
            dst_offset_list=[[0], [256]],
            block_ids=[1, 2],
            is_save=True,
            use_key_major_ranges=True,
        )

        thread._handle_request(task)

        # key_2 should be revoked
        store.batch_put_session_revoke.assert_called_once_with(["key_2"])
        # key_2 removed from active set
        assert "key_2" not in thread._active_put_keys
        assert "key_1" in thread._active_put_keys

    def test_handle_request_session_last_layer_commits(self):
        """Last layer calls batch_put_session_end to commit."""
        store = MagicMock()
        store.batch_put_from_multi_buffer_ranges.return_value = [256]
        store.batch_put_session_end.return_value = [0]
        thread = _make_layerwise_send_thread(store, num_layers=2)
        thread._use_session_api = True
        thread._active_put_keys = {"k1"}

        task = LayerTransferTask(
            req_id="test_last",
            group_id=0,
            layer_idx_in_group=1,
            physical_layer_id=1,  # last layer (0-indexed, num_layers=2)
            key_list=["k1"],
            addr_list=[[0x3000]],
            size_list=[[256]],
            dst_offset_list=[[512]],
            block_ids=[10],
            is_save=True,
            use_key_major_ranges=True,
        )

        thread._handle_request(task)

        store.batch_put_session_end.assert_called_once_with(["k1"])
        assert thread._active_put_keys is None  # reset after last layer


class TestRecvingThreadSessionApi:
    """Test KVCacheStoreRecvingThread session API methods."""

    def test_start_get_sessions_calls_batch_get_session_start(self):
        """start_get_sessions calls batch_get_session_start."""
        store = MagicMock()
        store.batch_get_session_start.return_value = [0, 0]
        thread = _make_layerwise_recv_thread(store, num_layers=2)
        thread._use_session_api = True

        keys = ["load_key_a", "load_key_b"]
        thread.start_get_sessions(keys)

        store.batch_get_session_start.assert_called_once_with(keys)

    def test_end_get_sessions_calls_batch_get_session_end(self):
        """end_get_sessions calls batch_get_session_end."""
        store = MagicMock()
        thread = _make_layerwise_recv_thread(store, num_layers=1)
        thread._use_session_api = True

        keys = ["load_key_a", "load_key_b"]
        thread.end_get_sessions(keys)

        store.batch_get_session_end.assert_called_once_with(keys)

    def test_handle_request_dispatches_session_path(self):
        """use_key_major_ranges=True dispatches to session handler."""
        store = MagicMock()
        store.batch_get_into_multi_buffer_ranges.return_value = [256, 256]
        thread = _make_layerwise_recv_thread(store, num_layers=2)
        thread._use_session_api = True
        thread._active_load_indices = {0, 1}

        task = LayerTransferTask(
            req_id="test_load_session",
            group_id=0,
            layer_idx_in_group=0,
            physical_layer_id=0,
            key_list=["load_key_1", "load_key_2"],
            addr_list=[[0x3000], [0x4000]],
            size_list=[[256], [256]],
            dst_offset_list=[[0], [512]],
            block_ids=[10, 20],
            is_save=False,
            use_key_major_ranges=True,
        )

        thread._handle_request(task)

        store.batch_get_into_multi_buffer_ranges.assert_called_once()
        assert thread._layer_load_finished_events[0].is_set()

    def test_handle_request_session_tracks_failures(self):
        """Failed range-get keys are tracked and excluded from next layers."""
        store = MagicMock()
        store.batch_get_into_multi_buffer_ranges.return_value = [256, -5]
        thread = _make_layerwise_recv_thread(store, num_layers=2)
        thread._use_session_api = True
        thread._active_load_indices = {0, 1}

        task = LayerTransferTask(
            req_id="test_load_fail",
            group_id=0,
            layer_idx_in_group=0,
            physical_layer_id=0,
            key_list=["ok_key", "fail_key"],
            addr_list=[[0x3000], [0x4000]],
            size_list=[[256], [256]],
            dst_offset_list=[[0], [512]],
            block_ids=[10, 20],
            is_save=False,
            use_key_major_ranges=True,
        )

        thread._handle_request(task)

        # Failed blocks tracked
        failed = thread.get_and_clear_block_ids_with_load_errors()
        assert 20 in failed
        # Failed index removed from active set
        assert 1 not in thread._active_load_indices

    def test_handle_request_session_last_layer_finalizes(self):
        """Last layer finalizes request tracking."""
        store = MagicMock()
        store.batch_get_into_multi_buffer_ranges.return_value = [256]
        thread = _make_layerwise_recv_thread(store, num_layers=2)
        thread._use_session_api = True
        thread._active_load_indices = {0}

        task = LayerTransferTask(
            req_id="test_last_load",
            group_id=0,
            layer_idx_in_group=1,
            physical_layer_id=1,  # last layer
            key_list=["load_key"],
            addr_list=[[0x3000]],
            size_list=[[256]],
            dst_offset_list=[[0]],
            block_ids=[30],
            is_save=False,
            use_key_major_ranges=True,
        )

        thread._handle_request(task)

        assert "test_last_load" in thread.finished_requests
        assert thread._active_load_indices is None


class TestWorkerSessionApiIntegration:
    """Integration tests for MooncakeStoreWorker session API."""

    def test_start_layerwise_sessions_starts_put_and_get(self):
        """_start_layerwise_sessions starts both put and get sessions."""
        store = MagicMock()
        store.batch_put_session_start.return_value = [0]
        store.batch_get_session_start.return_value = [0]
        worker = _make_layerwise_bare_worker(num_layers=2)
        worker._use_session_api = True
        worker._page_size_bytes = 256
        worker._load_sessions_closed = True
        worker._load_session_lock = threading.Lock()
        worker._opened_load_keys = []

        # Setup send thread
        send_thread = _make_layerwise_send_thread(store, num_layers=2)
        send_thread._use_session_api = True
        worker.kv_send_thread = send_thread

        # Setup recv thread
        recv_thread = _make_layerwise_recv_thread(store, num_layers=2)
        recv_thread._use_session_api = True
        worker.kv_recv_threads = [recv_thread]

        req = _make_test_reqmeta(can_save=True, can_load=True, token_len=32)
        worker._start_layerwise_sessions([req])

        store.batch_put_session_start.assert_called()
        store.batch_get_session_start.assert_called()
        assert not worker._load_sessions_closed

    def test_close_load_sessions_once_idempotent(self):
        """_close_load_sessions_once only releases sessions once."""
        store = MagicMock()
        worker = _make_layerwise_bare_worker(num_layers=1)
        worker._use_session_api = True
        worker._load_session_lock = threading.Lock()
        worker._load_sessions_closed = False
        worker._opened_load_keys = ["key_a"]

        recv_thread = _make_layerwise_recv_thread(store, num_layers=1)
        recv_thread._use_session_api = True
        worker.kv_recv_threads = [recv_thread]

        # First call
        worker._close_load_sessions_once()
        assert worker._load_sessions_closed is True
        assert store.batch_get_session_end.call_count == 1

        # Second call — should be a no-op
        worker._close_load_sessions_once()
        assert store.batch_get_session_end.call_count == 1  # unchanged


class TestBuildLayerTasksSessionApi:
    """Test _build_layer_tasks_from_requests with session API enabled."""

    def test_session_api_save_task_uses_block_key(self):
        """Session API save tasks use block keys (no @layer:N suffix)."""
        store = MagicMock()
        store.batch_put_session_start.return_value = [0, 0]
        store.batch_get_session_start.return_value = [0]
        worker = _make_layerwise_bare_worker(num_layers=2)
        worker._use_session_api = True
        worker._page_size_bytes = 256
        worker._load_sessions_closed = False
        worker._load_session_lock = threading.Lock()
        worker._opened_load_keys = []

        send_thread = _make_layerwise_send_thread(store, num_layers=2)
        send_thread._use_session_api = True
        worker.kv_send_thread = send_thread

        recv_thread = _make_layerwise_recv_thread(store, num_layers=2)
        recv_thread._use_session_api = True
        worker.kv_recv_threads = [recv_thread]

        req = _make_test_reqmeta()
        worker._build_layer_tasks_from_requests([req])

        # Verify save tasks use block-level keys
        for layer_id in range(2):
            tasks = worker._layer_save_tasks[layer_id]
            assert len(tasks) > 0
            for task in tasks:
                assert task.use_key_major_ranges is True
                for key in task.key_list:
                    assert "@layer:" not in key
                if task.dst_offset_list:
                    assert len(task.dst_offset_list) == len(task.key_list)

    def test_session_api_load_task_has_offsets(self):
        """Session API load tasks include per-layer byte offsets."""
        store = MagicMock()
        store.batch_put_session_start.return_value = [0]
        store.batch_get_session_start.return_value = [0]
        worker = _make_layerwise_bare_worker(num_layers=1)
        worker._use_session_api = True
        worker._page_size_bytes = 256
        worker._load_sessions_closed = False
        worker._load_session_lock = threading.Lock()
        worker._opened_load_keys = []

        send_thread = _make_layerwise_send_thread(store, num_layers=1)
        send_thread._use_session_api = True
        worker.kv_send_thread = send_thread

        recv_thread = _make_layerwise_recv_thread(store, num_layers=1)
        recv_thread._use_session_api = True
        worker.kv_recv_threads = [recv_thread]

        req = _make_test_reqmeta(can_save=False, can_load=True, token_len=32)
        worker._build_layer_tasks_from_requests([req])

        for layer_id in range(1):
            tasks = worker._layer_load_tasks[layer_id]
            for task in tasks:
                assert task.use_key_major_ranges is True
                assert len(task.dst_offset_list) == len(task.key_list)
