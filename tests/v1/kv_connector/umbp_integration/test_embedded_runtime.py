# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Embedded runtime contract tests.

CPU contract tests also cover hybrid tails on MORI when a GPU is available.
"""

import time
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.umbp_test_utils import MemoryRuntime
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    BlockTransferPlan,
    KVLayoutDescriptor,
    KVLayoutPlanner,
    KVRange,
    KVRegion,
    RankTopology,
    TransferJobStatus,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    EmbeddedRuntime,
    UMBPRuntimeConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import (
    UMBPStoreConnectorWorker,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)


def _runtime():
    return MemoryRuntime()


def _layout() -> KVLayoutDescriptor:
    return KVLayoutDescriptor(
        regions=(
            KVRegion(
                layer_name="layer0",
                group_id=0,
                block_stride=16,
                block_bytes=16,
                object_offset=0,
            ),
        ),
        topology=RankTopology(),
    )


def _plan(key: str, tensor: torch.Tensor, block_id: int = 0):
    return BlockTransferPlan(
        key=key,
        block_id=block_id,
        ranges=(
            KVRange(
                layer_name="layer0",
                group_id=0,
                block_id=block_id,
                base_address=tensor.data_ptr(),
                stride=tensor.numel(),
                length=tensor.numel(),
                object_offset=0,
            ),
        ),
    )


def test_embedded_store_is_invisible_until_publish():
    runtime = _runtime()
    layout = _layout()
    scheduler = runtime.create_scheduler_handle("test", RankTopology(), layout)
    worker = runtime.create_worker_handle("test", RankTopology(), layout)
    source = torch.arange(16, dtype=torch.uint8)
    plan = _plan("atomic-key", source)

    worker.register_buffers({"layer0": source})
    job = worker.store([plan])

    assert scheduler.lookup(["atomic-key"]) == [False]
    assert worker.wait(job).status is TransferJobStatus.COMPLETED

    worker.publish(job)

    assert scheduler.lookup(["atomic-key"]) == [True]
    worker.close()
    scheduler.close()


def test_embedded_store_load_roundtrip_restores_bytes():
    runtime = _runtime()
    layout = _layout()
    scheduler = runtime.create_scheduler_handle("test", RankTopology(), layout)
    worker = runtime.create_worker_handle("test", RankTopology(), layout)
    source = torch.arange(16, dtype=torch.uint8)
    destination = torch.zeros_like(source)
    plan = _plan("roundtrip-key", source, block_id=3)

    worker.register_buffers({"layer0": source})
    store_job = worker.store([plan])
    worker.publish(worker.wait(store_job))

    load_plan = _plan("roundtrip-key", destination, block_id=7)
    load_job = worker.load([load_plan])
    result = worker.wait(load_job)

    assert result.status is TransferJobStatus.COMPLETED
    assert torch.equal(source, destination)
    worker.close()
    scheduler.close()


def test_embedded_missing_object_is_a_load_failure():
    runtime = _runtime()
    layout = _layout()
    worker = runtime.create_worker_handle("test", RankTopology(), layout)
    destination = torch.zeros(16, dtype=torch.uint8)

    worker.register_buffers({"layer0": destination})
    result = worker.wait(worker.load([_plan("missing-key", destination, 9)]))

    assert result.status is TransferJobStatus.FAILED
    assert result.failed_block_ids == {9}
    worker.close()


def test_embedded_batch_isolates_failed_objects():
    runtime = _runtime()
    layout = _layout()
    scheduler = runtime.create_scheduler_handle("test", RankTopology(), layout)
    worker = runtime.create_worker_handle("test", RankTopology(), layout)
    source = torch.arange(16, dtype=torch.uint8)
    destination = torch.zeros_like(source)

    worker.register_buffers({"layer0": source})
    stored = _plan("present-key", source, block_id=1)
    store_job = worker.store([stored])
    worker.publish(worker.wait(store_job))

    present = _plan("present-key", destination, block_id=4)
    missing = _plan("missing-key", destination, block_id=5)
    result = worker.wait(worker.load([present, missing]))

    assert result.status is TransferJobStatus.FAILED
    assert result.completed_keys == {"present-key"}
    assert result.failed_block_ids == {5}
    assert scheduler.lookup(["present-key", "missing-key"]) == [True, False]
    assert torch.equal(source, destination)
    worker.close()
    scheduler.close()


def test_rank_local_keys_are_distinct_for_tp_pp_dcp():
    topology = RankTopology(
        tp_size=2,
        pp_size=2,
        dcp_size=2,
    )
    codec = BlockIdentityCodec(UMBPNamespace("rank-test"))
    keys = codec.keys_for_topology(b"hash", topology, [0])

    assert len(keys) == 4
    assert len(set(keys)) == 4
    assert all(":tp" in key and ":dcp" in key and ":pp" in key for key in keys)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_finished_hybrid_tail_roundtrip_restores_full_state(tmp_path, device):
    """A hash-only tail must survive request cleanup, match, and restore every byte."""
    runtime = MemoryRuntime()
    if device != "cpu":
        if not torch.accelerator.is_available():
            pytest.skip("requires a ROCm GPU")
        if not hasattr(pytest.importorskip("mori.cpp"), "UMBPClient"):
            pytest.skip("MORI was built without UMBP")
        torch.accelerator.set_device_index(0)
        runtime = EmbeddedRuntime.from_config(
            UMBPRuntimeConfig(
                "embedded",
                {"capacity_bytes": 16 << 20, "lookup_dir": str(tmp_path)},
            )
        )
    groups = [
        KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(
                block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
            ),
        ),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=16,
                shapes=((4,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        ),
    ]
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            KVCacheTensor(
                size=8 * group.kv_cache_spec.page_size_bytes,
                layers=group.layer_names,
                layer_stride=8 * group.kv_cache_spec.page_size_bytes,
                block_stride=group.kv_cache_spec.page_size_bytes,
            )
            for group in groups
        ],
    )
    extra = {"enable_partial_hash_hits": True, "load_async": True}
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        cache_config=SimpleNamespace(
            block_size=16, prefix_match_unit=4, enable_prefix_caching=True
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        kv_events_config=None,
        max_in_flight_tokens=64,
        num_prefill_lookahead_tokens=0,
    )
    topology = RankTopology()
    layout = KVLayoutPlanner.from_kv_cache_config(config)
    descriptor = layout.describe(topology)
    scheduler = UMBPStoreConnectorScheduler(
        vllm_config,
        config,
        runtime.create_scheduler_handle("tail", topology, descriptor),
        BlockIdentityCodec(UMBPNamespace("tail")),
    )
    worker = UMBPStoreConnectorWorker(
        runtime.create_worker_handle("tail", topology, descriptor), layout
    )
    pool = BlockPool(8, True, 4)
    scheduler.bind_gpu_block_pool(pool)
    caches = {
        group.layer_names[0]: torch.zeros(
            (8, group.kv_cache_spec.page_size_bytes), dtype=torch.uint8, device=device
        )
        for group in groups
    }
    expected = {}
    for index, (name, cache) in enumerate(caches.items(), 1):
        cache[index] = torch.arange(cache.shape[1], dtype=torch.uint8, device=device)
        expected[name] = cache[index].clone()
    worker.register_kv_caches(caches)
    request = SimpleNamespace(
        request_id="tail",
        num_tokens=13,
        num_computed_tokens=12,
        block_hashes=[b"a", b"b", b"c"],
    )
    source_ids = ([1], [2])
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: source_ids), 0
    )
    sources = [pool.blocks[1], pool.blocks[2]]
    pool.touch(sources)
    assert not scheduler.register_finished_partial_tail(
        request, source_ids, [(1, 2, 12)]
    )
    assert scheduler.request_finished(request, source_ids) == (False, None)
    pool.free_blocks(sources)
    output = SimpleNamespace(
        finished_req_ids={"tail"},
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={},
    )
    try:
        metadata = scheduler.build_connector_meta(output)
        assert len(metadata.store_plans) == 2
        assert [block.ref_cnt for block in sources] == [1, 1]
        worker.enqueue_stores(metadata)
        worker.wait_for_save()
        deadline = time.monotonic() + 10
        while scheduler.has_pending_push_work():
            assert time.monotonic() < deadline, "store completion was not reported"
            worker.get_finished(set())
            scheduler.update_connector_output(
                SimpleNamespace(
                    kv_connector_worker_meta=worker.build_connector_worker_meta()
                )
            )
            time.sleep(0.001)
        assert [block.ref_cnt for block in sources] == [0, 0]
        assert not scheduler.has_pending_push_work()
        for cache in caches.values():
            cache.zero_()
        request.request_id = "replay"
        assert scheduler.get_num_new_matched_tokens(request, 0) == (12, True)
        scheduler.update_state_after_alloc(
            request, SimpleNamespace(get_block_ids=lambda group_ids: ([3], [4])), 12
        )
        output.finished_req_ids = set()
        worker.start_load_kv(None, scheduler.build_connector_meta(output))
        # Asynchronous loads settle through get_finished, not the forward pass.
        deadline = time.monotonic() + 10
        while (finished := worker.get_finished(set())) == (None, None):
            assert time.monotonic() < deadline, "load completion was not reported"
            time.sleep(0.001)
        assert finished == (None, {"replay"})
        assert worker.get_failed_recving() == set()
        for index, (name, cache) in enumerate(caches.items(), 3):
            assert torch.equal(cache[index], expected[name])
        counters = worker.get_kv_connector_stats().reduce()
        page_bytes = sum(group.kv_cache_spec.page_size_bytes for group in groups)
        assert counters["load_num_bytes"] == page_bytes
        assert counters["store_num_bytes"] == page_bytes
    finally:
        worker.close()
        scheduler.close()
