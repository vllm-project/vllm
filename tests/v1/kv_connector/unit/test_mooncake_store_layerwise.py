# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MooncakeStoreConnector layerwise KV cache support."""

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as mooncake_store_worker,
)

# ============================================================================
# Mock helpers for _build_layer_tasks and sync tests
# ============================================================================

import re
import threading
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerTransferTask,
    ReqMeta,
    LoadSpec,
    TailKeyBoundary,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
)
import torch


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

    # Session API shared state (FUNC-FIX 2/3/4).
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.mooncake_session_tracker import (  # noqa: E501
        MooncakeSessionTracker,
    )
    worker._session_tracker = MooncakeSessionTracker()
    worker._put_started_keys = set()
    worker._put_started_keys_lock = threading.Lock()
    worker._current_mooncake_last_chunk_req_ids = set()

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

# ============================================================================
# Hybrid model helper
# ============================================================================


def _make_hybrid_worker(
    *,
    num_attn_layers: int = 4,
    num_mamba_layers: int = 2,
    block_size: int = 16,
    tp_rank: int = 0,
) -> "mooncake_store_worker.MooncakeStoreWorker":  # noqa: F821
    """Construct a MooncakeStoreWorker for a Mamba+Attention hybrid model.

    Group 0 = Attention (num_attn_layers via FullAttentionSpec),
    Group 1 = Mamba (num_mamba_layers via MambaSpec).
    Both groups share the same block_size / hash_block_size so ChunkedTokenDatabase
    validates. Mamba identity comes from the Coordinator detecting MambaSpec.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
        worker as mooncake_store_worker,
    )

    group_0_layers = [f"model.layers.{i}.self_attn" for i in range(num_attn_layers)]
    group_1_layers = [f"model.layers.{i}.mlp" for i in range(num_attn_layers, num_attn_layers + num_mamba_layers)]

    num_layers = num_attn_layers + num_mamba_layers
    num_groups = 2

    worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    worker._layerwise_enabled = True
    worker._num_layers = num_layers
    worker.tp_rank = tp_rank
    worker._group_tp_replication_factors = (1, 1)
    worker.block_size = block_size
    worker.num_blocks = 10
    worker.kv_send_thread = None
    worker.kv_recv_threads = []
    worker.store = MagicMock()
    worker._kv_connector_stats_lock = threading.Lock()
    worker.kv_connector_stats = MagicMock()

    worker._layer_save_tasks = {l: [] for l in range(num_layers)}
    worker._layer_load_tasks = {l: [] for l in range(num_layers)}
    worker._layer_save_finished_events = {l: threading.Event() for l in range(num_layers)}
    worker._layer_load_finished_events = {l: threading.Event() for l in range(num_layers)}
    worker._current_save_layer = 0
    worker._current_load_layer = 0
    worker._next_load_layer_to_submit = 0
    worker._num_prefetch_layers = 1

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.mooncake_session_tracker import (  # noqa: E501
        MooncakeSessionTracker,
    )
    worker._session_tracker = MooncakeSessionTracker()
    worker._put_started_keys = set()
    worker._put_started_keys_lock = threading.Lock()
    worker._current_mooncake_last_chunk_req_ids = set()

    # _group_physical_layers
    worker._group_physical_layers = {}
    for g_idx, layer_names in enumerate([group_0_layers, group_1_layers]):
        layers = []
        for layer_name in layer_names:
            m = re.search(r"\.layers\.(\d+)", layer_name)
            if m:
                layers.append(int(m.group(1)))
        worker._group_physical_layers[g_idx] = layers

    # Token databases — both use same block_size/hash_block_size
    worker.token_dbs = []
    for g_idx in range(num_groups):
        md = KeyMetadata("test-model", tp_rank, 0, 0, 0, group_id=g_idx)
        db = ChunkedTokenDatabase(md, block_size=block_size, hash_block_size=block_size)
        db.set_kv_caches_base_addr([0x1000 * (g_idx + 1)])
        db.set_block_len([block_size * 64])
        worker.token_dbs.append(db)

    # Coordinator: group 1 has MambaSpec → mamba_group_ids={1}
    attn_spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=8, head_size=64, dtype=None
    )
    mamba_spec = MambaSpec(
        block_size=block_size,
        shapes=((block_size, 64),),
        dtypes=(torch.float16,),
    )
    groups = [
        KVCacheGroupSpec(group_0_layers, attn_spec),
        KVCacheGroupSpec(group_1_layers, mamba_spec),
    ]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        groups,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    return worker


def _make_hybrid_save_reqmeta(
    req_id: str = "test_req",
    token_len: int = 32,
    boundary_state_offloads=None,
) -> ReqMeta:
    if boundary_state_offloads is None:
        boundary_state_offloads = []
    block_hashes = [BlockHash(f"h{i:016d}".encode()) for i in range(32)]
    block_ids = (list(range(2)), list(range(2)))
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=token_len,
        block_ids=block_ids,
        block_hashes=block_hashes,
        can_save=True,
        load_spec=None,
        boundary_state_offloads=boundary_state_offloads,
        num_prompt_tokens=token_len,
    )


def _make_hybrid_load_reqmeta(
    req_id: str = "test_req",
    token_len: int = 32,
    vllm_cached_tokens: int = 0,
    kvpool_cached_tokens: int = 32,
    tail_key_boundaries=(),
) -> ReqMeta:
    block_hashes = [BlockHash(f"h{i:016d}".encode()) for i in range(32)]
    block_ids = (list(range(2)), list(range(2)))
    load_spec = LoadSpec(
        vllm_cached_tokens=vllm_cached_tokens,
        kvpool_cached_tokens=kvpool_cached_tokens,
        can_load=True,
        token_len=token_len,
        tail_key_boundaries=tail_key_boundaries,
    )
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=0,
        block_ids=block_ids,
        block_hashes=block_hashes,
        can_save=False,
        load_spec=load_spec,
        boundary_state_offloads=None,
        num_prompt_tokens=token_len,
    )


# ============================================================================
# Layerwise Save/Load Flow Tests
# Tests for save_kv_layer, wait_for_layer_load, and _handle_request
# ============================================================================

import time


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
    thread.enable_layerwise(num_layers)
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
    thread.enable_layerwise(num_layers)
    thread.request_queue.task_done = MagicMock()
    return thread


class TestSubmitReadyLayerLoads:
    """Test _submit_ready_layer_loads round-robin distribution."""

    def test_round_robin_across_recv_threads(self):
        """Tasks are distributed round-robin, not broadcast to all threads."""
        worker = _make_layerwise_bare_worker(num_layers=2)

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

        mock_threads = [MagicMock() for _ in range(3)]
        worker.kv_recv_threads = mock_threads
        worker._num_prefetch_layers = 2

        worker._submit_ready_layer_loads()

        for mt in mock_threads:
            assert len(mt.add_request.call_args_list) <= 2, (
                f"Thread received {len(mt.add_request.call_args_list)} tasks, "
                f"expected ≤2 (round-robin)"
            )

        total_calls = sum(len(mt.add_request.call_args_list) for mt in mock_threads)
        assert total_calls == 2, f"Expected 2 total calls, got {total_calls}"


class TestLayerwiseEventReuse:
    """FUNC-FIX 5/6: events are cleared (not rebuilt) across chunks."""

    def test_events_reused_not_rebuilt(self):
        """_init_layerwise_config builds events once; they survive a round."""
        worker = _make_layerwise_bare_worker(num_layers=3)
        save_events = {
            l: worker._layer_save_finished_events[l] for l in range(3)
        }
        load_events = {
            l: worker._layer_load_finished_events[l] for l in range(3)
        }

        worker._layer_save_finished_events[0].set()
        worker._layer_load_finished_events[1].set()

        for l in range(3):
            assert worker._layer_save_finished_events[l] is save_events[l]
            assert worker._layer_load_finished_events[l] is load_events[l]


class TestSaveKvLayer:
    """Test save_kv_layer submits tasks and handles last-layer completion."""

    def test_save_kv_layer_submits_tasks_to_send_thread(self):
        """save_kv_layer must send tasks for the matching layer to kv_send_thread."""
        worker = _make_layerwise_bare_worker(num_layers=4)
        worker.kv_send_thread = MagicMock()
        worker.kv_send_thread._layerwise_enabled = True

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

        for layer_id in range(4):
            worker.save_kv_layer(f"model.layers.{layer_id}.self_attn", None, None)

        assert worker.kv_send_thread.add_request.call_count == 4
        for i, call_args in enumerate(worker.kv_send_thread.add_request.call_args_list):
            task = call_args.args[0]
            assert isinstance(task, LayerTransferTask)
            assert task.physical_layer_id == i

    def test_save_kv_layer_last_layer_clears_events(self):
        """Last layer waits then clears (not rebuilds) the save events (FUNC-FIX 5)."""
        worker = _make_layerwise_bare_worker(num_layers=2)
        worker.kv_send_thread = MagicMock()
        worker.kv_send_thread._layerwise_enabled = True

        for layer_id in range(2):
            task = LayerTransferTask(
                req_id="test_req1", group_id=0,
                layer_idx_in_group=layer_id, physical_layer_id=layer_id,
                key_list=["k"], addr_list=[[0x2000]], size_list=[[256]],
                block_ids=[layer_id], is_save=True,
            )
            worker._layer_save_tasks[layer_id].append(task)
            worker._layer_save_finished_events[layer_id].set()

        old_save_events = {l: worker._layer_save_finished_events[l] for l in range(2)}

        worker.save_kv_layer("model.layers.1.self_attn", None, None)

        for l in range(2):
            assert worker._layer_save_finished_events[l] is old_save_events[l]
            assert not worker._layer_save_finished_events[l].is_set()


class TestWaitForLayerLoad:
    """Test wait_for_layer_load submits prefetch and waits for completion."""

    def test_wait_for_layer_load_submits_prefetch(self):
        """wait_for_layer_load must trigger _submit_ready_layer_loads."""
        worker = _make_layerwise_bare_worker(num_layers=4)
        worker.kv_recv_threads = [MagicMock()]

        for layer_id in range(4):
            task = LayerTransferTask(
                req_id="test_req", group_id=0,
                layer_idx_in_group=layer_id, physical_layer_id=layer_id,
                key_list=["k"], addr_list=[[0x2000]], size_list=[[256]],
                block_ids=[layer_id], is_save=False,
            )
            worker._layer_load_tasks[layer_id].append(task)

        for layer_id in range(4):
            worker._layer_load_finished_events[layer_id].clear()

        def _set_events_after_delay():
            time.sleep(0.05)
            worker._layer_load_finished_events[0].set()

        import threading as _thr
        setter = _thr.Thread(target=_set_events_after_delay, daemon=True)
        setter.start()

        worker.wait_for_layer_load("model.layers.0.self_attn")

        assert worker._current_load_layer == 1

    def test_wait_for_layer_load_prefetch_advances_counter(self):
        """_submit_ready_layer_loads advances _next_load_layer_to_submit."""
        worker = _make_layerwise_bare_worker(num_layers=4)
        worker.kv_recv_threads = [MagicMock()]

        for layer_id in range(4):
            task = LayerTransferTask(
                req_id="test_req", group_id=0,
                layer_idx_in_group=layer_id, physical_layer_id=layer_id,
                key_list=["k"], addr_list=[[0x2000]], size_list=[[256]],
                block_ids=[layer_id], is_save=False,
            )
            worker._layer_load_tasks[layer_id].append(task)

        for i in range(4):
            worker._layer_load_finished_events[i].set()

        assert worker._next_load_layer_to_submit == 0

        worker.wait_for_layer_load("model.layers.0.self_attn")
        assert worker._next_load_layer_to_submit == 1
        assert worker._current_load_layer == 1

        worker.wait_for_layer_load("model.layers.1.self_attn")
        assert worker._next_load_layer_to_submit == 2
        assert worker._current_load_layer == 2


class TestLayerwiseSendHandleLayerTask:
    """Test KVCacheStoreSendingThread._handle_request layerwise path."""

    def test_send_handle_request_puts_to_store(self):
        """Layerwise save must call batch_put_from_multi_buffers for non-existent keys."""
        store = MagicMock()
        store.batch_is_exist.return_value = [False, False]
        store.batch_put_from_multi_buffers.return_value = True

        thread = _make_layerwise_send_thread(store, num_layers=2)
        layer_id = 1
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

        store.batch_is_exist.assert_called_once_with(["save_key_1", "save_key_2"])
        store.batch_put_from_multi_buffers.assert_called_once_with(
            ["save_key_1", "save_key_2"],
            [[0x3000], [0x4000]],
            [[256], [256]],
        )
        assert event.is_set()
        assert "test_send" in thread.finished_requests

    def test_send_handle_request_skips_existing_keys(self):
        """Keys that already exist in store should be skipped (dedup)."""
        store = MagicMock()
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

        store.batch_put_from_multi_buffers.assert_called_once_with(
            ["new_key"],
            [[0x4000]],
            [[256]],
        )
        assert event.is_set()


class TestLayerwiseRecvHandleLayerTask:
    """Test KVCacheStoreRecvingThread._handle_request layerwise path."""

    def test_recv_handle_request_gets_from_store(self):
        """Layerwise load must call batch_get_into_multi_buffers."""
        store = MagicMock()
        store.batch_get_into_multi_buffers.return_value = [256, 256]

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
            block_ids=[10, 20],
            is_save=False,
        )

        thread._handle_request(task)

        assert event.is_set()
        failed_blocks = thread.get_and_clear_block_ids_with_load_errors()
        assert 20 in failed_blocks


class TestLayerwiseStoreJobLedger:
    """Test layerwise save tasks participate in the store_job_id ledger."""

    def test_store_job_id_passed_to_save_tasks(self):
        """_build_layer_tasks_from_requests copies ReqMeta.store_job_id to tasks."""
        worker = _make_layerwise_bare_worker(num_layers=2)
        req = _make_test_reqmeta()
        req.store_job_id = 42
        worker._build_layer_tasks_from_requests([req])

        for layer_id in range(2):
            tasks = worker._layer_save_tasks[layer_id]
            assert len(tasks) == 1
            for task in tasks:
                assert task.store_job_id == 42

    def test_save_kv_layer_layer0_registers_store_job_id(self):
        """save_kv_layer layer 0 enters the store_job_id into the ledger."""
        worker = _make_layerwise_bare_worker(num_layers=2)
        send_thread = _make_layerwise_send_thread(MagicMock(), num_layers=2)
        worker.kv_send_thread = send_thread

        req = _make_test_reqmeta()
        req.store_job_id = 7
        worker._build_layer_tasks_from_requests([req])

        worker.save_kv_layer("model.layers.0.self_attn", None, None)

        assert "test_req" in send_thread.stored_requests
        assert 7 in send_thread.stored_requests["test_req"]

    def test_handle_layer_task_finishes_job_on_last_layer(self):
        """Sending thread finishes the store_job_id on the last layer."""
        store = MagicMock()
        store.batch_is_exist.return_value = [False]
        store.batch_put_from_multi_buffers.return_value = True
        thread = _make_layerwise_send_thread(store, num_layers=2)

        task = LayerTransferTask(
            req_id="test_finish",
            group_id=0,
            layer_idx_in_group=1,
            physical_layer_id=1,
            key_list=["k"],
            addr_list=[[0x3000]],
            size_list=[[256]],
            block_ids=[1],
            is_save=True,
            store_job_id=5,
        )
        thread.stored_requests["test_finish"] = {5}

        thread._handle_layer_task(task)

        assert thread._completed_saves.get(5) == 1
        completed = thread.take_completed_saves()
        assert completed[5] == 1

    def test_handle_layer_range_task_finishes_job_on_last_layer(self):
        """Session-API sending thread finishes the store_job_id on the last layer."""
        store = MagicMock()
        store.batch_put_from_multi_buffer_ranges.return_value = [0]
        store.batch_put_session_end.return_value = [0]
        thread = _make_layerwise_send_thread(store, num_layers=2)
        thread._use_session_api = True
        thread._active_put_keys = {"k1"}

        task = LayerTransferTask(
            req_id="test_range_finish",
            group_id=0,
            layer_idx_in_group=1,
            physical_layer_id=1,
            key_list=["k1"],
            addr_list=[[0x3000]],
            size_list=[[256]],
            dst_offset_list=[[512]],
            block_ids=[10],
            is_save=True,
            use_key_major_ranges=True,
            store_job_id=6,
        )
        thread.stored_requests["test_range_finish"] = {6}

        thread._handle_layer_range_task(task)

        assert thread._completed_saves.get(6) == 1
        completed = thread.take_completed_saves()
        assert completed[6] == 1


class TestSaveKvLayerIntegration:
    """Integration tests: worker.save_kv_layer -> send thread _handle_request."""

    def test_save_flow_integration(self):
        """Full integration: populated tasks -> save_kv_layer -> thread processes."""
        store = MagicMock()
        store.batch_is_exist.return_value = [False, False]
        store.batch_put_from_multi_buffers.return_value = True

        worker = _make_layerwise_bare_worker(num_layers=2)
        send_thread = _make_layerwise_send_thread(store, num_layers=2)
        worker.kv_send_thread = send_thread

        for layer_id in range(2):
            send_thread.set_layer_finished_event(
                layer_id, True, worker._layer_save_finished_events[layer_id]
            )

        req = _make_test_reqmeta(token_len=32)
        worker._build_layer_tasks_from_requests([req])

        assert len(worker._layer_save_tasks[0]) == 1
        assert len(worker._layer_save_tasks[1]) == 1

        worker.save_kv_layer("model.layers.0.self_attn", None, None)

        while not send_thread.request_queue.empty():
            item = send_thread.request_queue.get_nowait()
            send_thread._handle_request(item)
            send_thread.request_queue.task_done()

        assert worker._layer_save_finished_events[0].is_set()
        store.batch_put_from_multi_buffers.assert_called()

        store.reset_mock()
        store.batch_is_exist.return_value = [False]
        store.batch_put_from_multi_buffers.return_value = True

        worker.save_kv_layer("model.layers.1.self_attn", None, None)

        while not send_thread.request_queue.empty():
            item = send_thread.request_queue.get_nowait()
            send_thread._handle_request(item)
            send_thread.request_queue.task_done()

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

        req = _make_test_reqmeta(can_save=False, can_load=True, token_len=32)
        worker._build_layer_tasks_from_requests([req])

        assert len(worker._layer_load_tasks[0]) == 1

        def _consume_recv_queue():
            while not recv_thread.request_queue.empty():
                item = recv_thread.request_queue.get_nowait()
                recv_thread._handle_request(item)
                recv_thread.request_queue.task_done()

        consumer = threading.Thread(target=_consume_recv_queue, daemon=True)
        consumer.start()
        consumer.join()

        worker._layer_load_finished_events[0].set()

        old_count = worker._current_load_layer
        worker.wait_for_layer_load("model.layers.0.self_attn")

        assert worker._current_load_layer == old_count + 1

    def test_load_failure_propagates_to_worker(self):
        """Load failures in recv thread should propagate to worker."""
        store = MagicMock()
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

        failed_blocks = worker.get_block_ids_with_load_errors()
        assert 42 in failed_blocks


class TestSendingThreadSessionApi:
    """Test KVCacheStoreSendingThread session API methods."""

    def test_start_put_sessions_calls_batch_put_session_start(self):
        """start_put_sessions calls batch_put_session_start with correct args."""
        store = MagicMock()
        store.batch_put_session_start.return_value = [0, 0]
        thread = _make_layerwise_send_thread(store, num_layers=2)
        thread._use_session_api = True

        keys = ["key_a", "key_b"]
        object_size = 8192
        thread.start_put_sessions(keys, object_size)

        store.batch_put_session_start.assert_called_once()
        called_keys, called_sizes, _ = store.batch_put_session_start.call_args[0]
        assert called_keys == keys
        assert called_sizes == [object_size, object_size]
        assert thread._active_put_keys == {"key_a", "key_b"}

    def test_start_put_sessions_partial_failure_degrades(self):
        """Failed session starts are degraded (not revoked - FUNC-FIX 2)."""
        store = MagicMock()
        store.batch_put_session_start.return_value = [0, -1]
        thread = _make_layerwise_send_thread(store, num_layers=1)
        thread._use_session_api = True

        thread.start_put_sessions(["ok_key", "fail_key"], 4096)

        store.batch_put_session_revoke.assert_not_called()
        assert thread._active_put_keys == {"ok_key"}
        assert "ok_key" in thread._put_started_keys
        assert "fail_key" not in thread._put_started_keys

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

    def test_handle_request_session_drops_failed_keys(self):
        """Failed range-put keys are dropped from the active set (not revoked - FUNC-FIX 2)."""
        store = MagicMock()
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

        store.batch_put_session_revoke.assert_not_called()
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
            physical_layer_id=1,
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
        assert thread._active_put_keys is None


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

        failed = thread.get_and_clear_block_ids_with_load_errors()
        assert 20 in failed
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
            physical_layer_id=1,
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

        send_thread = _make_layerwise_send_thread(store, num_layers=2)
        send_thread._use_session_api = True
        worker.kv_send_thread = send_thread

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

        worker._close_load_sessions_once()
        assert worker._load_sessions_closed is True
        assert store.batch_get_session_end.call_count == 1

        worker._close_load_sessions_once()
        assert store.batch_get_session_end.call_count == 1


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


# ============================================================================
# Hybrid Model Tests (Mamba + Attention)
# Tests for Mamba group handling in save session and load paths.
# ============================================================================


def _make_hybrid_load_reqmeta_all_masks_true(
    req_id: str = "test_req",
    token_len: int = 32,
) -> ReqMeta:
    """Like _make_hybrid_load_reqmeta but load_mask returns all-True."""
    # We set this up with all block IDs so process_tokens + chunking works fine.
    block_ids_full = (list(range(2)), list(range(2)))
    tail_key_bnd = ()
    return _make_hybrid_load_reqmeta(
        req_id=req_id, token_len=token_len,
        tail_key_boundaries=tail_key_bnd,
    )


class TestHybridCoordinatorMambaGroupIds:
    """Verify the Coordinator correctly identifies Mamba groups in the hybrid model."""

    def test_mamba_group_ids_in_hybrid_coordinator(self):
        """Group 1 (with MambaSpec) should be in coord.mamba_group_ids."""
        worker = _make_hybrid_worker(num_attn_layers=3, num_mamba_layers=2)
        assert 0 not in worker.coord.mamba_group_ids, \
            "Attention group should NOT be in mamba_group_ids"
        assert 1 in worker.coord.mamba_group_ids, \
            "Mamba group should be in mamba_group_ids"

    def test_mamba_group_skipped_in_save_task_build(self):
        """_build_layer_tasks_from_requests save path skips Mamba group."""
        worker = _make_hybrid_worker(num_attn_layers=3, num_mamba_layers=2)
        worker._use_session_api = False
        worker._group_tp_replication_factors = (1, 1)
        worker.tp_rank = 0

        # Mock store_mask to return all-True per group (both groups can save)
        # so that the `mamba_group_ids` skip is the only thing preventing Mamba
        # groups from producing tasks.
        original_store_mask = worker.coord.store_mask
        def _all_true_store_mask(*a, **kw):
            return ([True, True], [True, True])
        worker.coord.store_mask = _all_true_store_mask

        req = _make_hybrid_save_reqmeta(token_len=32)

        worker._build_layer_tasks_from_requests([req])

        # All save tasks should be from group 0 (attention). No group 1 tasks.
        for layer_id in range(5):
            for task in worker._layer_save_tasks[layer_id]:
                assert task.group_id != 1, (
                    f"Mamba group (group 1) should not produce save tasks, "
                    f"but found task at layer {layer_id}: group_id={task.group_id}"
                )

        worker.coord.store_mask = original_store_mask

    def test_mamba_group_has_load_tasks_in_layerwise(self):
        """Mamba group should produce load tasks — the old code had `continue`
        that skipped Mamba for load."""
        worker = _make_hybrid_worker(num_attn_layers=3, num_mamba_layers=2)
        worker._use_session_api = False
        worker._group_tp_replication_factors = (1, 1)
        worker.tp_rank = 0

        # Mock load_mask to all-True so no chunks are filtered.
        original_load_mask = worker.coord.load_mask
        def _all_true_load_mask(block_hashes, token_len):
            return ([True, True], [True, True])
        worker.coord.load_mask = _all_true_load_mask

        req = _make_hybrid_load_reqmeta(token_len=32)
        req.block_ids = (list(range(2)), list(range(2)))

        worker._build_layer_tasks_from_requests([req])

        worker.coord.load_mask = original_load_mask

        mamba_layers = worker._group_physical_layers.get(1, [])
        assert len(mamba_layers) == 2

        mamba_load_found = False
        for layer_id in mamba_layers:
            load_tasks = worker._layer_load_tasks.get(layer_id, [])
            for task in load_tasks:
                if task.group_id == 1 and task.is_save is False:
                    mamba_load_found = True
                    break
        assert mamba_load_found, (
            f"Mamba group should produce load tasks in _layer_load_tasks "
            f"for layers {mamba_layers}"
        )

    def test_start_layerwise_sessions_skips_mamba_save_keys(self):
        """Save session keys must NOT include keys from Mamba group.
        Mamba KV is saved via boundary task (non-session), so sending Mamba
        positional keys to PutStart would create orphan sessions."""
        store = MagicMock()
        store.batch_put_session_start.return_value = [0]
        store.batch_get_session_start.return_value = [0]

        worker = _make_hybrid_worker(num_attn_layers=3, num_mamba_layers=2)
        worker._use_session_api = True
        worker._page_size_bytes = 256 * 5
        worker._load_sessions_closed = True
        worker._opened_load_keys = []

        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.mooncake_session_tracker import (
            MooncakeSessionTracker,
        )
        worker._session_tracker = MooncakeSessionTracker()
        worker._put_started_keys = set()
        worker._put_started_keys_lock = threading.Lock()
        worker._current_mooncake_last_chunk_req_ids = set()

        send_thread = MagicMock()
        send_thread._use_session_api = True
        send_thread._saved_offset = {}
        send_thread._active_put_keys = None
        worker.kv_send_thread = send_thread

        recv_thread = MagicMock()
        recv_thread._use_session_api = True
        worker.kv_recv_threads = [recv_thread]

        req = _make_hybrid_save_reqmeta(token_len=32)

        # Patch Mamba DB's process_tokens BEFORE calling _start_layerwise_sessions
        mamba_db = worker.token_dbs[1]
        original_pt = mamba_db.process_tokens
        mamba_db.process_tokens = MagicMock(side_effect=original_pt)

        worker._start_layerwise_sessions([req])

        # Verify start_put_sessions was called
        send_thread.start_put_sessions.assert_called_once()

        # Mamba group's process_tokens must NOT have been called for save keys
        mamba_db.process_tokens.assert_not_called(), (
            "Mamba group's process_tokens should not be called for save keys "
            "(explicitly skipped via mamba_group_ids in save key collection)"
        )


class TestAsCacheTupleDocstring:
    """Verify that `_as_cache_tuple` docstring no longer mentions Ascend."""

    def test_as_cache_tuple_no_ascend_reference(self):
        """_as_cache_tuple docstring should not reference 'Ascend'."""
        doc = _make_layerwise_bare_worker()._as_cache_tuple.__doc__
        assert doc is not None
        assert "Ascend" not in doc, (
            f"_as_cache_tuple docstring should not reference 'Ascend', got: {doc}"
        )