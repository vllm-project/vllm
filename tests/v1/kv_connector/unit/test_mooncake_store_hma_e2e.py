# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end save->lookup test for MooncakeStoreConnector on a hybrid
(SWA + Full) attention config, using a dict-backed mock store."""

import sys
import threading
import types
from unittest.mock import MagicMock, patch

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as mooncake_store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (  # noqa: E501
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowSpec,
)


class _DictStore:
    """In-memory MooncakeDistributedStore stand-in."""

    def __init__(self):
        self._data: dict[str, bytes] = {}

    def setup(self, *_args, **_kwargs):
        return 0

    def register_buffer(self, addr, length):
        return 0

    def batch_is_exist(self, keys):
        return [1 if k in self._data else 0 for k in keys]

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, *_args, **_kwargs):
        for k in keys:
            self._data[k] = b"x"
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, addrs, sizes, *_args, **_kwargs):
        return [0 if k in self._data else -1 for k in keys]


def _minimal_vllm_config(cache_block_size=16):
    cfg = MagicMock()
    cfg.cache_config.block_size = cache_block_size
    cfg.cache_config.num_gpu_blocks = 4
    cfg.cache_config.hash_block_size = None
    cfg.cache_config.enable_prefix_caching = True
    cfg.parallel_config.prefill_context_parallel_size = 1
    cfg.parallel_config.decode_context_parallel_size = 1
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.parallel_config.world_size = 1
    cfg.parallel_config.rank = 0
    cfg.parallel_config.data_parallel_rank_local = 0
    cfg.parallel_config.data_parallel_size_local = 1
    cfg.kv_transfer_config.kv_role = "kv_both"
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.kv_events_config = None
    cfg.model_config.model = "/tmp/m"
    cfg.model_config.use_mla = False
    cfg.model_config.get_num_layers.return_value = 2
    cfg.model_config.get_total_num_kv_heads.return_value = 8
    cfg.model_config.max_model_len = 4096
    cfg.scheduler_config.max_num_batched_tokens = 8192
    # Without this, MagicMock's truthy use_eagle() triggers the coordinator's
    # "use_eagle && nothing annotated → flag all groups" fallback.
    cfg.speculative_config = None
    return cfg


def _build_worker_with_dict_store(vllm_config, kv_cache_config, store):
    """Build a MooncakeStoreWorker patching all distributed dependencies."""
    fake_mooncake_store = types.ModuleType("mooncake.store")
    fake_mooncake_store.MooncakeDistributedStore = lambda: store  # type: ignore[attr-defined]
    fake_mooncake_store.ReplicateConfig = MagicMock  # type: ignore[attr-defined]
    with (
        patch.dict(sys.modules, {"mooncake.store": fake_mooncake_store}),
        patch.object(mooncake_store_worker, "MooncakeStoreConfig") as MCfg,
    ):
        sc = MCfg.load_from_env.return_value
        sc.metadata_server = ""
        sc.global_segment_size = 1 << 20
        sc.local_buffer_size = 1 << 20
        sc.protocol = "tcp"
        sc.device_name = ""
        sc.master_server_address = ""
        sc.enable_offload = False
        with (
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_pcp_group"
            ) as mock_pcp,
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_dcp_group"
            ) as mock_dcp,
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_ip",
                return_value="127.0.0.1",
            ),
        ):
            mock_pcp.return_value.world_size = 1
            mock_dcp.return_value.world_size = 1
            worker = mooncake_store_worker.MooncakeStoreWorker(
                vllm_config, kv_cache_config=kv_cache_config
            )
    return worker


def test_e2e_swa_plus_full_save_then_lookup_hits():
    """
    E2E: build a SWA+Full hybrid worker, save all blocks via the sending
    thread (synchronously), then verify lookup returns the full hit length.
    Also verify that evicting SWA's early blocks (outside its window) still
    allows a full hit because the window covers the tail.
    """
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=32,
    )
    cfg = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(size=8192, shared_by=["L0"]),
            KVCacheTensor(size=8192, shared_by=["L1"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["L0"], full),
            KVCacheGroupSpec(["L1"], swa),
        ],
    )
    vllm_config = _minimal_vllm_config(cache_block_size=16)
    store = _DictStore()

    worker = _build_worker_with_dict_store(vllm_config, cfg, store)
    worker.tp_size = 1
    worker.pp_size = 1
    worker.put_step = 1
    worker.num_kv_head = 8

    # Register kv_caches using mocked thread classes so register_kv_caches
    # doesn't try to start real background threads (which set ready_event).
    kv_caches = {
        "L0": torch.zeros(2, 4, 8, 8, 64),
        "L1": torch.zeros(2, 4, 8, 8, 64),
    }

    def _fake_thread_init(*args, **kwargs):
        """Mock thread that sets all threading.Event args so waits don't block."""
        for v in list(args) + list(kwargs.values()):
            if isinstance(v, threading.Event):
                v.set()
        m = MagicMock()
        m.start = lambda: None
        return m

    with (
        patch.object(
            mooncake_store_worker,
            "KVCacheStoreSendingThread",
            side_effect=_fake_thread_init,
        ),
        patch.object(
            mooncake_store_worker,
            "KVCacheStoreRecvingThread",
            side_effect=_fake_thread_init,
        ),
    ):
        worker.register_kv_caches(kv_caches)

    # Now build a real sending thread (no .start()) over the worker's token_dbs
    # and the dict-backed store, so _handle_request runs synchronously.
    ready = threading.Event()
    send_thread = KVCacheStoreSendingThread(
        store=store,
        token_databases=worker.token_dbs,
        block_size=worker.block_size,
        coord=worker.coord,
        tp_rank=worker.tp_rank,
        put_step=worker.put_step,
        kv_role=worker.kv_role,
        ready_event=ready,
        enable_kv_event=False,
    )

    hs = [BlockHash(bytes([i + 1]) * 4) for i in range(4)]
    save_req = ReqMeta(
        req_id="r0",
        token_len_chunk=64,
        block_ids=([0, 1, 2, 3], [0, 1, 2, 3]),
        block_hashes=hs,
        can_save=True,
    )
    send_thread.add_stored_request("r0")
    # Put the request in the queue so task_done() doesn't underflow.
    send_thread.request_queue.put(save_req)
    req = send_thread.request_queue.get()
    send_thread._handle_request(req)

    # Point worker.store at the dict store (the worker constructor captured
    # the MagicMock; replace with the real dict store for lookup).
    worker.store = store

    # Both groups stored all 4 blocks -> full hit.
    assert worker.lookup(token_len=64, block_hashes=hs) == 64

    # Evict SWA's first two blocks (outside its window of 32 tokens = 2 blocks).
    swa_keys_outside_window = [
        k
        for k in list(store._data.keys())
        if "@group:1" in k and (("@" + hs[0].hex()) in k or ("@" + hs[1].hex()) in k)
    ]
    for k in swa_keys_outside_window:
        del store._data[k]

    # SWA window=32 -> only last 2 blocks must be present in SWA group.
    # Full has all 4. Coordinator should still return 64.
    assert worker.lookup(token_len=64, block_hashes=hs) == 64


def test_recv_skips_swa_blocks_before_window():
    """Producer stored every block for both groups; consumer must only fetch
    SWA blocks within the sliding window, not the head."""
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    # sliding_window=32, block_size=16 → 2 contiguous blocks within window.
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=32,
    )
    groups = [
        KVCacheGroupSpec(["L0"], full),
        KVCacheGroupSpec(["L1"], swa),
    ]
    md0 = KeyMetadata("m", 0, 0, 0, 0, group_id=0)
    md1 = KeyMetadata("m", 0, 0, 0, 0, group_id=1)
    db_full = ChunkedTokenDatabase(md0, block_size=16, hash_block_size=16)
    db_swa = ChunkedTokenDatabase(md1, block_size=16, hash_block_size=16)
    db_full.set_kv_caches_base_addr([0])
    db_full.set_block_len([1024])
    db_swa.set_kv_caches_base_addr([1 << 20])
    db_swa.set_block_len([1024])

    requested_keys: list[str] = []

    class _CapturingStore:
        def batch_get_into_multi_buffers(self, keys, addrs, sizes):
            requested_keys.extend(keys)
            return [0] * len(keys)

    ready = threading.Event()
    coord = MooncakeStoreCoordinator(
        groups, scheduler_block_size=16, hash_block_size=16
    )
    recv = KVCacheStoreRecvingThread(
        store=_CapturingStore(),
        token_databases=[db_full, db_swa],
        block_size=16,
        tp_rank=0,
        ready_event=ready,
        coord=coord,
    )

    hs = [BlockHash(bytes([i + 1]) * 4) for i in range(4)]
    req = ReqMeta(
        req_id="r0",
        token_len_chunk=64,
        block_ids=([0, 1, 2, 3], [0, 1, 2, 3]),
        block_hashes=hs,
        load_spec=LoadSpec(
            vllm_cached_tokens=0, kvpool_cached_tokens=64, can_load=True, token_len=64
        ),
    )
    recv.request_queue.put(req)
    recv._handle_request(recv.request_queue.get())

    full_keys = [k for k in requested_keys if "@group:0" in k]
    swa_keys = [k for k in requested_keys if "@group:1" in k]
    # Full attention: load every block (4).
    assert len(full_keys) == 4
    # SWA: only the 2 tail-window blocks (hashes hs[2], hs[3]).
    assert len(swa_keys) == 2
    swa_hashes = {k.rsplit("@", 1)[-1] for k in swa_keys}
    assert swa_hashes == {hs[2].hex(), hs[3].hex()}


def test_chunked_token_database_hash_block_size_smaller_than_block_size():
    """DSv4-style: hash_block_size=4, group block_size=16 — process_tokens
    must merge every 4 fine hashes into one chunk hash via
    BlockHashListWithBlockSize."""
    md = KeyMetadata("m", 0, 0, 0, 0, group_id=3)
    db = ChunkedTokenDatabase(md, block_size=16, hash_block_size=4)
    db.set_kv_caches_base_addr([0])
    db.set_block_len([512])
    # 8 fine-grained hashes (32 tokens at hash_block_size=4) → 2 group chunks.
    fine_hashes = [BlockHash(bytes([i + 1]) * 4) for i in range(8)]
    out = list(db.process_tokens(token_len=32, block_hashes=fine_hashes))
    assert len(out) == 2
    assert out[0][0] == 0 and out[0][1] == 16
    assert out[1][0] == 16 and out[1][1] == 32
    # Each chunk's hash is the concatenation of 4 fine hashes.
    expected0 = b"".join(fine_hashes[0:4]).hex()
    expected1 = b"".join(fine_hashes[4:8]).hex()
    assert out[0][2].chunk_hash == expected0
    assert out[1][2].chunk_hash == expected1
