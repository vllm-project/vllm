# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
import threading
import time

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    BlockTransferPlan,
    KVLayoutDescriptor,
    KVLayoutPlanner,
    KVRange,
    KVShardSlice,
    LoadSpec,
    LookupState,
    LookupStatus,
    RankTopology,
    RequestTracker,
    TPShardMapping,
    TransferJobState,
    TransferJobStatus,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
    UMBPStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.factory import (
    UMBPRuntimeConfig,
    UMBPRuntimeFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    UMBPRuntimeCapabilities,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import (
    UMBPStoreConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.stats import (
    UMBPStoreConnectorStats,
)
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)


def _kv_cache_config() -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[
            KVCacheTensor(
                size=8192,
                layers=["layer1", "layer2"],
                layer_stride=4096,
                block_stride=1024,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer1", "layer2"], spec)],
    )


def _hybrid_kv_cache_config() -> KVCacheConfig:
    full = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=16,
        shapes=((4,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full),
            KVCacheGroupSpec(["mamba"], mamba),
        ],
    )


def _vllm_config(extra: dict, **parallel_overrides) -> SimpleNamespace:
    parallel = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "world_size": 1,
    }
    parallel.update(parallel_overrides)
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra,
        ),
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(model="test-model", revision="r1"),
        parallel_config=SimpleNamespace(**parallel),
    )


def test_block_identity_codec_does_not_use_physical_block_id():
    codec = BlockIdentityCodec(UMBPNamespace("namespace"), 2, 3)

    assert codec.key(b"\x01\x02", group_id=4) == (
        "umbp:vllm:v1:namespace:tp2:pcp0:dcp0:pp3:g4:0102"
    )
    assert codec.key(b"\x01\x02", group_id=4) == codec.key(
        b"\x01\x02", group_id=4
    )


def test_layout_planner_orders_all_transfer_layers():
    planner = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())

    assert [region.layer_name for region in planner.regions] == ["layer1", "layer2"]
    assert planner.regions[0].object_offset == 0
    assert planner.regions[1].object_offset == planner.regions[0].block_bytes
    assert planner.object_size == 2 * planner.regions[0].block_bytes


def test_rank_topology_is_part_of_layout_identity():
    topology = RankTopology(
        tp_rank=1,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        pcp_rank=1,
        pcp_size=2,
        dcp_rank=0,
        dcp_size=1,
    )

    assert topology.local_namespace == (1, 1, 0, 0)
    assert topology.rank_count == 4
    assert len(topology.all_namespaces()) == 4


@pytest.mark.parametrize(
    ("producer", "consumer", "rank", "expected"),
    [
        (
            4,
            2,
            0,
            (
                KVShardSlice(0, 0, 0, 2),
                KVShardSlice(1, 0, 2, 2),
            ),
        ),
        (2, 4, 1, (KVShardSlice(0, 2, 0, 2),)),
    ],
)
def test_tp_shard_mapping_covers_consumer_heads(
    producer, consumer, rank, expected
):
    mapping = TPShardMapping.build(producer, consumer, rank, num_kv_heads=8)

    mapping.validate()
    assert mapping.slices == expected


def test_layout_descriptor_carries_topology_and_format():
    topology = RankTopology(tp_size=2)
    descriptor = KVLayoutPlanner.from_kv_cache_config(
        _kv_cache_config()
    ).describe(topology, "lbh_nc")

    assert isinstance(descriptor, KVLayoutDescriptor)
    assert descriptor.layout_format == "lbh_nc"
    assert descriptor.topology.tp_size == 2


def test_layout_planner_builds_scatter_gather_ranges():
    planner = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    plan = planner.plan_for_block(
        "key", block_id=3, base_addresses={"layer1": 1000, "layer2": 2000}
    )

    assert [item.base_address for item in plan.ranges] == [4072, 5072]
    assert [item.object_offset for item in plan.ranges] == [
        0,
        planner.regions[0].block_bytes,
    ]


def test_layout_planner_registers_real_tensor_addresses():
    planner = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    caches = {
        name: torch.empty_strided(
            (8, 2, 16, 8),
            (512, 256, 8, 1),
            dtype=torch.float16,
        )
        for name in ("layer1", "layer2")
    }

    planner.register_kv_caches(caches)
    plan = planner.plan_registered_block("key", block_id=2)

    assert plan.ranges[0].base_address == caches["layer1"].data_ptr() + 2048
    assert plan.ranges[1].base_address == caches["layer2"].data_ptr() + 2048


def test_layout_planner_expands_kernel_subblocks():
    planner = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    caches = {
        name: torch.empty((8, 2, 16, 8), dtype=torch.float16)
        for name in ("layer1", "layer2")
    }

    planner.register_kv_caches(caches)
    plan = planner.plan_registered_block("key", block_id=2)

    assert len(plan.ranges) == 4
    assert [item.base_address for item in plan.ranges[:2]] == [
        caches["layer1"].data_ptr() + 2048,
        caches["layer1"].data_ptr() + 2560,
    ]


def test_layout_planner_partial_range_spans_physical_subblocks():
    planner = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    caches = {
        name: torch.empty((8, 2, 16, 8), dtype=torch.float16)
        for name in ("layer1", "layer2")
    }
    planner.register_kv_caches(caches)

    plan = planner.plan_registered_block(
        "partial",
        block_id=2,
        token_start=6,
        token_end=12,
    )

    assert [item.length for item in plan.ranges] == [128, 256, 128, 256]
    assert [item.object_offset for item in plan.ranges] == [0, 128, 384, 512]
    assert plan.ranges[0].base_address == caches["layer1"].data_ptr() + 2432
    assert plan.ranges[1].base_address == caches["layer1"].data_ptr() + 2560


def test_transfer_job_isolates_failed_keys():
    plans = (
        BlockTransferPlan("key-a", 7),
        BlockTransferPlan("key-b", 8),
    )
    job = TransferJobState(plans)
    job.start()
    job.complete(["key-a"])
    job.fail(["key-b"], "timeout")

    assert job.status is TransferJobStatus.FAILED
    assert job.failed_block_ids == {8}
    assert job.completed_keys == {"key-a"}


def test_request_tracker_advances_only_complete_block_watermarks():
    tracker = RequestTracker("req", generation=3)

    assert tracker.mark_saved(31, 16) == 16
    assert tracker.mark_saved(15, 16) == 16
    tracker.reset()

    assert tracker.generation == 4
    assert tracker.saved_tokens == 0


def test_load_spec_reports_only_external_tokens():
    spec = LoadSpec(local_tokens=32, external_tokens=48)

    assert spec.num_tokens_to_load == 16


def test_lookup_state_transitions_to_error_and_cancelled():
    state = LookupState("req")

    state.fail("lookup timeout")
    assert state.status is LookupStatus.ERROR
    assert state.error == "lookup timeout"
    state.cancel()
    assert state.status is LookupStatus.CANCELLED
    assert state.matched_tokens == 0


def test_request_tracker_retries_failed_store_suffix():
    tracker = RequestTracker("req", generation=1)
    tracker.mark_saved(48, 16)
    tracker.record_store_failure(32)
    tracker.record_store_failure(16)

    assert tracker.saved_tokens == 48
    assert tracker.retry_from_tokens == 16
    tracker.clear_store_retry()
    assert tracker.retry_from_tokens is None


def test_worker_metadata_aggregates_per_key_and_block_failures():
    metadata = UMBPConnectorWorkerMetadata(
        completed_loads={"a"}, failed_block_ids={1}
    )
    metadata.aggregate(
        UMBPConnectorWorkerMetadata(
            completed_loads={"b"}, completed_stores={"c"}, failed_block_ids={2}
        )
    )

    assert metadata.completed_loads == {"a", "b"}
    assert metadata.completed_stores == {"c"}
    assert metadata.failed_block_ids == {1, 2}


@pytest.mark.parametrize("mode", ["embedded"])
def test_runtime_config_accepts_local_modes(mode):
    config = UMBPRuntimeConfig.from_vllm(_vllm_config({"mode": mode}))

    assert config.mode == mode


def test_runtime_config_requires_standalone_endpoint():
    with pytest.raises(ValueError, match="requires endpoint"):
        UMBPRuntimeConfig.from_vllm(_vllm_config({"mode": "standalone"}))


def test_runtime_config_requires_distributed_identity():
    with pytest.raises(ValueError, match="master_address"):
        UMBPRuntimeConfig.from_vllm(_vllm_config({"mode": "distributed"}))


def test_runtime_factory_builds_embedded_adapter():
    config = UMBPRuntimeConfig("embedded", {"backend": "memory"})
    runtime = UMBPRuntimeFactory.build(config)

    assert runtime.capabilities.lookup
    assert runtime.capabilities.load
    assert runtime.capabilities.store
    assert runtime.capabilities.publish


class _SchedulerHandle:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def lookup(self, keys):
        self.queries.append(list(keys))
        return self.hits

    def close(self):
        pass


def test_scheduler_tracks_consecutive_load_plan():
    config = _kv_cache_config()
    handle = _SchedulerHandle([True, False])
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": False}),
        config,
        handle,
        BlockIdentityCodec(UMBPNamespace("test")),
    )
    request = SimpleNamespace(
        request_id="req",
        num_tokens=32,
        block_hashes=[b"a", b"b"],
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (16, False)
    assert handle.queries[0] == [
        "umbp:vllm:v1:test:tp0:pcp0:dcp0:pp0:g0:61",
        "umbp:vllm:v1:test:tp0:pcp0:dcp0:pp0:g0:62",
    ]


def test_scheduler_async_lookup_defers_then_returns_hit():
    gate = threading.Event()

    class _GatedHandle(_SchedulerHandle):
        def lookup(self, keys):
            gate.wait(timeout=5)
            return [True] * len(keys)

    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {
                "mode": "embedded",
                "lookup_async": True,
                "load_async": False,
            }
        ),
        _kv_cache_config(),
        _GatedHandle([]),
        BlockIdentityCodec(UMBPNamespace("async")),
    )
    request = SimpleNamespace(
        request_id="async",
        num_tokens=32,
        block_hashes=[b"a", b"b"],
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    gate.set()
    result = (None, False)
    for _ in range(100):
        result = scheduler.get_num_new_matched_tokens(request, 0)
        if result != (None, False):
            break
        time.sleep(0.01)
    scheduler.close()

    assert result == (32, False)


def test_scheduler_cached_decode_uses_save_watermark_and_new_blocks():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {
                "mode": "embedded",
                "load_async": False,
                "save_decode_cache": True,
            }
        ),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("decode")),
    )
    request = SimpleNamespace(
        request_id="decode",
        req_id="decode",
        num_tokens=48,
        block_hashes=[b"a", b"b", b"c"],
        block_ids=([1, 2, 3],),
        num_computed_tokens=32,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2],)),
        0,
    )

    def output(num_computed_tokens, new_block_ids):
        return SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["decode"],
                new_block_ids=[new_block_ids],
                num_computed_tokens=[num_computed_tokens],
            ),
            num_scheduled_tokens={"decode": 1},
        )

    first = scheduler.build_connector_meta(output(32, ([3],)))
    assert [plan.block_id for plan in first.store_plans] == [1, 2]

    second = scheduler.build_connector_meta(output(48, ()))
    assert [plan.block_id for plan in second.store_plans] == [3]


def test_scheduler_lazy_offload_stores_only_when_request_finishes():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {
                "mode": "embedded",
                "load_async": False,
                "lazy_offload": True,
            }
        ),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("lazy")),
    )
    request = SimpleNamespace(
        request_id="lazy",
        req_id="lazy",
        num_tokens=32,
        num_computed_tokens=32,
        block_hashes=[b"a", b"b"],
        block_ids=([4, 5],),
        prompt_token_ids=list(range(32)),
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([4, 5],)),
        0,
    )
    active_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"lazy": 32},
    )

    assert scheduler.build_connector_meta(active_output).store_plans == []
    assert scheduler.request_finished(request, ([4, 5],)) == (False, None)
    finished_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={},
    )
    plans = scheduler.build_connector_meta(finished_output).store_plans

    assert [plan.block_id for plan in plans] == [4, 5]


def test_scheduler_emits_partial_tail_plan_for_next_step():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": False}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("partial")),
    )
    request = SimpleNamespace(
        request_id="partial",
        req_id="partial",
        num_tokens=12,
        block_hashes=[b"a", b"b", b"c"],
        block_ids=([7],),
        num_computed_tokens=0,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([7],)),
        0,
    )

    assert scheduler.register_finished_partial_tail(
        request, ([7],), [(0, 7, 12)]
    )
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={},
        )
    )

    assert len(metadata.partial_tail_plans) == 1
    tail = metadata.partial_tail_plans[0]
    assert (tail.start_token, tail.end_token) == (0, 12)
    assert tail.block_id == 7


def test_scheduler_stores_exact_hybrid_boundary_state():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": False}),
        _hybrid_kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("hybrid")),
    )
    request = SimpleNamespace(
        request_id="hybrid",
        req_id="hybrid",
        num_tokens=32,
        block_hashes=[b"a", b"b"],
        block_ids=([1, 2], [8, 9]),
        num_computed_tokens=16,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2], [8, 9])),
        0,
    )
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={},
            kv_connector_block_state=SimpleNamespace(
                boundary_state_offloads={
                    "hybrid": [
                        (1, 9, 16),
                        (1, NULL_BLOCK_ID, 16),
                    ]
                }
            ),
        )
    )

    assert len(metadata.store_plans) == 1
    plan = metadata.store_plans[0]
    assert plan.group_id == 1
    assert plan.block_id == 9
    assert plan.block_hash == b"a"


class _WorkerHandle:
    def register_buffers(self, kv_caches):
        self.kv_caches = kv_caches

    def load(self, plans):
        self.loaded_plans = list(plans)
        job = TransferJobState(tuple(plans))
        job.start()
        job.complete()
        return job

    def store(self, plans):
        self.stored_plans = list(plans)
        job = TransferJobState(tuple(plans))
        job.start()
        job.complete()
        return job

    def wait(self, job):
        return job

    def publish(self, job):
        self.published = job

    def close(self):
        pass


class _LayerRecordingWorkerHandle(_WorkerHandle):
    def __init__(self):
        self.load_calls = []
        self.wait_calls = []

    def load(self, plans):
        self.load_calls.append(list(plans))
        return super().load(plans)

    def wait(self, job):
        self.wait_calls.append(job)
        return super().wait(job)


class _CancellableWorkerHandle(_WorkerHandle):
    def __init__(self):
        self.cancelled = []

    def cancel(self, job):
        self.cancelled.append(job)
        job.cancel("preempted")
        return job


def test_worker_preemption_cancels_only_matching_request():
    handle = _CancellableWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    first = BlockTransferPlan("first", 1, request_id="first")
    second = BlockTransferPlan("second", 2, request_id="second")
    worker.enqueue_stores(
        UMBPConnectorMetadata(
            store_plans=[first, second],
            store_requests={"first": [first], "second": [second]},
        )
    )

    worker.handle_preemptions(
        UMBPConnectorMetadata(preempted_request_ids={"first"})
    )

    assert len(handle.cancelled) == 1
    assert handle.cancelled[0].plans == (first,)
    assert "second" in worker._store_jobs


def test_worker_waits_for_layers_independently():
    handle = _LayerRecordingWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    plan = BlockTransferPlan(
        key="layered",
        block_id=1,
        ranges=(
            KVRange("layer1", 0, 1, 1000, 16, 16, 0),
            KVRange("layer2", 0, 1, 2000, 16, 16, 16),
        ),
    )
    worker.start_load_kv(
        None,
        UMBPConnectorMetadata(
            load_plans=[plan],
            load_requests={"req": [plan]},
        ),
    )

    assert len(handle.load_calls) == 2
    worker.wait_for_layer_load("layer1")
    assert len(handle.wait_calls) == 1
    assert worker.get_finished({"req"}) == (None, None)
    worker.wait_for_layer_load("layer2")
    assert len(handle.wait_calls) == 2
    assert worker.get_finished({"req"}) == (None, {"req"})


def test_worker_falls_back_to_bulk_when_layerwise_is_unsupported():
    handle = _LayerRecordingWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle, layerwise_load=False)
    plan = BlockTransferPlan(
        key="bulk",
        block_id=1,
        ranges=(
            KVRange("layer1", 0, 1, 1000, 16, 16, 0),
            KVRange("layer2", 0, 1, 2000, 16, 16, 16),
        ),
    )

    worker.start_load_kv(
        None,
        UMBPConnectorMetadata(
            load_plans=[plan],
            load_requests={"req": [plan]},
        ),
    )
    worker.wait_for_layer_load("layer1")

    assert len(handle.load_calls) == 1
    assert handle.loaded_plans == [plan]
    assert worker.get_finished({"req"}) == (None, {"req"})


class _EmbeddedSchedulerHandle:
    def __init__(self, store):
        self.store = store

    def lookup(self, keys):
        return [key in self.store for key in keys]

    def close(self):
        pass


class _EmbeddedWorkerHandle(_WorkerHandle):
    def __init__(self, store):
        self.keys = store

    def load(self, plans):
        job = TransferJobState(tuple(plans))
        job.start()
        present = [plan.key for plan in plans if plan.key in self.keys]
        missing = [plan.key for plan in plans if plan.key not in self.keys]
        job.complete(present)
        if missing:
            job.fail(missing, "embedded key missing")
        return job

    def store(self, plans):
        job = TransferJobState(tuple(plans))
        job.start()
        self.keys.update(plan.key for plan in plans)
        job.complete()
        return job


class _EmbeddedRuntime:
    capabilities = UMBPRuntimeCapabilities()

    def __init__(self):
        self.store = set()
        self.scheduler_args = None
        self.worker_args = None

    def create_scheduler_handle(self, namespace, topology, layout):
        self.scheduler_args = (namespace, topology, layout)
        return _EmbeddedSchedulerHandle(self.store)

    def create_worker_handle(self, namespace, topology, layout):
        self.worker_args = (namespace, topology, layout)
        return _EmbeddedWorkerHandle(self.store)


def test_worker_reports_load_and_store_completion():
    worker = UMBPStoreConnectorWorker(_WorkerHandle())
    metadata = UMBPConnectorMetadata(
        load_plans=[BlockTransferPlan("load", 3)],
        store_plans=[BlockTransferPlan("store", 4)],
        load_requests={"req": [BlockTransferPlan("load", 3)]},
        store_requests={"req": [BlockTransferPlan("store", 4)]},
    )

    worker.start_load_kv(None, metadata)
    worker.wait_for_layer_load("layer0")
    worker.enqueue_stores(metadata)
    worker.wait_for_save()
    result = worker.build_connector_worker_meta()

    assert result.completed_loads == {"load"}
    assert result.completed_stores == {"store"}
    assert worker.get_finished({"req"}) == ({"req"}, {"req"})


def test_worker_emits_block_stored_event_after_store_completion():
    worker = UMBPStoreConnectorWorker(_WorkerHandle())
    plan = BlockTransferPlan(
        key="event-key",
        block_id=3,
        block_hash=b"event-hash",
        parent_block_hash=b"parent",
        token_ids=(1, 2, 3, 4),
        block_size=4,
        group_id=0,
    )
    metadata = UMBPConnectorMetadata(
        store_plans=[plan],
        store_requests={"req": [plan]},
    )

    worker.enqueue_stores(metadata)
    worker.wait_for_save()
    result = worker.build_connector_worker_meta()

    assert len(result.kv_events) == 1
    assert result.kv_events[0].block_hashes == [
        maybe_convert_block_hash(b"event-hash")
    ]
    assert result.kv_events[0].parent_block_hash == maybe_convert_block_hash(
        b"parent"
    )
    assert result.kv_events[0].token_ids == [1, 2, 3, 4]


def test_worker_emits_block_removed_for_runtime_eviction():
    handle = _WorkerHandle()
    handle.take_evicted_keys = lambda: [
        "umbp:vllm:v1:test:tp0:pcp0:dcp0:pp0:g2:65766963746564"
    ]
    worker = UMBPStoreConnectorWorker(handle)

    [event] = worker.get_kv_events()

    assert event.block_hashes == [maybe_convert_block_hash(b"evicted")]
    assert event.group_idx == 2
    assert event.medium == "CPU"


def test_umbp_stats_aggregate_and_reduce():
    first = UMBPStoreConnectorStats()
    first.record("load", submitted=2, completed=1, failed=1, num_bytes=64)
    second = UMBPStoreConnectorStats()
    second.record("load", completed=1, num_bytes=32)

    merged = first.aggregate(second)

    assert merged.reduce() == {
        "load_submitted": 2,
        "load_completed": 2,
        "load_failed": 1,
        "load_num_bytes": 96,
    }


def test_worker_preserves_scheduler_supplied_ranges():
    handle = _WorkerHandle()
    worker = UMBPStoreConnectorWorker(
        handle, KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    )
    worker.register_kv_caches(
        {
            name: torch.empty_strided(
                (8, 2, 16, 8),
                (512, 256, 8, 1),
                dtype=torch.float16,
            )
            for name in ("layer1", "layer2")
        }
    )
    supplied = BlockTransferPlan(
        key="custom",
        block_id=1,
        ranges=(
            KVRange(
                layer_name="custom",
                group_id=3,
                block_id=1,
                base_address=1234,
                stride=256,
                length=128,
                object_offset=0,
            ),
        ),
    )
    metadata = UMBPConnectorMetadata(
        load_plans=[supplied],
        load_requests={"req": [supplied]},
    )

    worker.start_load_kv(None, metadata)

    assert handle.loaded_plans == [supplied]


def test_embedded_connector_core_flow(monkeypatch):
    runtime = _EmbeddedRuntime()
    monkeypatch.setitem(
        UMBPRuntimeFactory._builders,
        "embedded",
        lambda config: runtime,
    )
    config = _kv_cache_config()
    vllm_config = _vllm_config({"mode": "embedded", "load_async": False})
    producer = SimpleNamespace(
        request_id="producer",
        req_id="producer",
        num_tokens=32,
        block_hashes=[b"a", b"b"],
        block_ids=([1, 2],),
        num_computed_tokens=0,
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[producer],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"producer": 32},
    )

    scheduler_connector = UMBPStoreConnector(
        vllm_config, KVConnectorRole.SCHEDULER, config
    )
    worker_connector = UMBPStoreConnector(
        vllm_config, KVConnectorRole.WORKER, config
    )
    caches = {
        name: torch.empty_strided(
            (8, 2, 16, 8),
            (512, 256, 8, 1),
            dtype=torch.float16,
        )
        for name in ("layer1", "layer2")
    }
    worker_connector.register_kv_caches(caches)

    assert scheduler_connector.get_num_new_matched_tokens(producer, 0) == (0, False)
    metadata = scheduler_connector.build_connector_meta(scheduler_output)
    worker_connector.bind_connector_metadata(metadata)
    worker_connector.wait_for_save()
    assert runtime.store

    consumer = SimpleNamespace(
        request_id="consumer",
        req_id="consumer",
        num_tokens=32,
        block_hashes=[b"a", b"b"],
    )
    assert scheduler_connector.get_num_new_matched_tokens(consumer, 0) == (
        32,
        False,
    )
    scheduler_connector.update_state_after_alloc(
        consumer,
        SimpleNamespace(get_block_ids=lambda group_ids: ([7, 8],)),
        32,
    )
    consumer_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[consumer],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"consumer": 32},
    )
    load_metadata = scheduler_connector.build_connector_meta(consumer_output)
    worker_connector.bind_connector_metadata(load_metadata)
    worker_connector.start_load_kv(None)
    worker_connector.wait_for_layer_load("layer0")

    assert worker_connector.get_block_ids_with_load_errors() == set()
    assert worker_connector.get_transfer_results({"consumer"}).finished_recving == {
        "consumer"
    }


def test_builtin_embedded_runtime_register_store_and_load():
    config = _kv_cache_config()
    vllm_config = _vllm_config(
        {
            "mode": "embedded",
            "backend": "memory",
            "load_async": False,
            "key_namespace": "builtin-embedded-test",
        }
    )
    source_caches = {
        name: torch.empty_strided(
            (8, 2, 16, 8),
            (512, 256, 8, 1),
            dtype=torch.float16,
        )
        for name in ("layer1", "layer2")
    }
    for index, cache in enumerate(source_caches.values()):
        cache.copy_(
            torch.arange(cache.numel(), dtype=torch.float16).reshape(cache.shape)
            + index
        )

    scheduler_connector = UMBPStoreConnector(
        vllm_config, KVConnectorRole.SCHEDULER, config
    )
    worker_connector = UMBPStoreConnector(
        vllm_config, KVConnectorRole.WORKER, config
    )
    worker_connector.register_kv_caches(source_caches)
    producer = SimpleNamespace(
        request_id="builtin-producer",
        req_id="builtin-producer",
        num_tokens=32,
        block_hashes=[b"builtin-a", b"builtin-b"],
        block_ids=([1, 2],),
        num_computed_tokens=0,
    )
    producer_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[producer],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"builtin-producer": 32},
    )
    store_metadata = scheduler_connector.build_connector_meta(producer_output)
    worker_connector.bind_connector_metadata(store_metadata)
    worker_connector.wait_for_save()

    consumer = SimpleNamespace(
        request_id="builtin-consumer",
        req_id="builtin-consumer",
        num_tokens=32,
        block_hashes=[b"builtin-a", b"builtin-b"],
    )
    assert scheduler_connector.get_num_new_matched_tokens(consumer, 0) == (
        32,
        False,
    )
    scheduler_connector.update_state_after_alloc(
        consumer,
        SimpleNamespace(get_block_ids=lambda group_ids: ([5, 6],)),
        32,
    )
    consumer_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[consumer],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"builtin-consumer": 32},
    )
    load_metadata = scheduler_connector.build_connector_meta(consumer_output)

    destination_caches = {
        name: torch.empty_strided(
            (8, 2, 16, 8),
            (512, 256, 8, 1),
            dtype=torch.float16,
        )
        for name in source_caches
    }
    for cache in destination_caches.values():
        cache.zero_()
    worker_connector.register_kv_caches(destination_caches)
    worker_connector.bind_connector_metadata(load_metadata)
    worker_connector.start_load_kv(None)
    worker_connector.wait_for_layer_load("layer1")
    assert worker_connector.get_finished({"builtin-consumer"}) == (None, None)
    worker_connector.wait_for_layer_load("layer2")
    assert worker_connector.get_finished({"builtin-consumer"}) == (
        None,
        {"builtin-consumer"},
    )
    load_errors = worker_connector.get_block_ids_with_load_errors()
    if load_errors:
        raise AssertionError(f"embedded load errors: {sorted(load_errors)}")

    for name in source_caches:
        if not torch.equal(source_caches[name][1], destination_caches[name][5]):
            raise AssertionError(
                f"{name} block 1 round-trip mismatch: "
                f"src={source_caches[name][1, 0, 0, 0].item()} "
                f"dst={destination_caches[name][5, 0, 0, 0].item()}"
            )
        if not torch.equal(source_caches[name][2], destination_caches[name][6]):
            raise AssertionError(
                f"{name} block 2 round-trip mismatch: "
                f"src={source_caches[name][2, 0, 0, 0].item()} "
                f"dst={destination_caches[name][6, 0, 0, 0].item()}"
            )


def test_builtin_embedded_tp2_pp2_dcp2_rank_store_completeness():
    config = _kv_cache_config()
    extra = {
        "mode": "embedded",
        "backend": "memory",
        "load_async": False,
        "key_namespace": "builtin-tp2-pp2-dcp2-test",
    }
    request = SimpleNamespace(
        request_id="ranked-producer",
        req_id="ranked-producer",
        num_tokens=32,
        block_hashes=[b"ranked-a", b"ranked-b"],
        block_ids=([1, 2],),
        num_computed_tokens=0,
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"ranked-producer": 32},
    )
    caches = {
        name: torch.empty_strided(
            (8, 2, 16, 8),
            (512, 256, 8, 1),
            dtype=torch.float16,
        )
        for name in ("layer1", "layer2")
    }

    for pp_rank in range(2):
        for dcp_rank in range(2):
            for tp_rank in range(2):
                vllm_config = _vllm_config(
                    extra,
                    tensor_parallel_rank=tp_rank,
                    tensor_parallel_size=2,
                    pipeline_parallel_rank=pp_rank,
                    pipeline_parallel_size=2,
                    decode_context_parallel_rank=dcp_rank,
                    decode_context_parallel_size=2,
                    world_size=8,
                )
                scheduler_connector = UMBPStoreConnector(
                    vllm_config, KVConnectorRole.SCHEDULER, config
                )
                worker_connector = UMBPStoreConnector(
                    vllm_config, KVConnectorRole.WORKER, config
                )
                worker_connector.register_kv_caches(caches)
                metadata = scheduler_connector.build_connector_meta(
                    scheduler_output
                )
                worker_connector.bind_connector_metadata(metadata)
                worker_connector.wait_for_save()

    vllm_config = _vllm_config(
        extra,
        tensor_parallel_rank=0,
        tensor_parallel_size=2,
        pipeline_parallel_rank=0,
        pipeline_parallel_size=2,
        decode_context_parallel_rank=0,
        decode_context_parallel_size=2,
        world_size=8,
    )
    scheduler_connector = UMBPStoreConnector(
        vllm_config, KVConnectorRole.SCHEDULER, config
    )
    consumer = SimpleNamespace(
        request_id="ranked-consumer",
        num_tokens=32,
        block_hashes=[b"ranked-a", b"ranked-b"],
    )

    assert scheduler_connector.get_num_new_matched_tokens(consumer, 0) == (
        32,
        False,
    )


def test_embedded_tp_dcp_pp_rank_local_store_flow(monkeypatch):
    runtime = _EmbeddedRuntime()
    monkeypatch.setitem(
        UMBPRuntimeFactory._builders,
        "embedded",
        lambda config: runtime,
    )
    config = _kv_cache_config()
    vllm_config = _vllm_config(
        {"mode": "embedded", "load_async": False},
        tensor_parallel_rank=1,
        tensor_parallel_size=2,
        pipeline_parallel_rank=1,
        pipeline_parallel_size=2,
        decode_context_parallel_rank=1,
        decode_context_parallel_size=2,
        world_size=8,
    )
    request = SimpleNamespace(
        request_id="ranked",
        req_id="ranked",
        num_tokens=32,
        block_hashes=[b"a", b"b"],
        block_ids=([3, 4],),
        num_computed_tokens=0,
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"ranked": 32},
    )

    scheduler_connector = UMBPStoreConnector(
        vllm_config, KVConnectorRole.SCHEDULER, config
    )
    worker_connector = UMBPStoreConnector(
        vllm_config, KVConnectorRole.WORKER, config
    )
    worker_connector.register_kv_caches(
        {
            name: torch.empty_strided(
                (8, 2, 16, 8),
                (512, 256, 8, 1),
                dtype=torch.float16,
            )
            for name in ("layer1", "layer2")
        }
    )

    metadata = scheduler_connector.build_connector_meta(scheduler_output)
    worker_connector.bind_connector_metadata(metadata)
    worker_connector.wait_for_save()

    namespace = UMBPNamespace.from_vllm_config(vllm_config, config).value
    assert runtime.store == {
        f"umbp:vllm:v1:{namespace}:tp1:pcp0:dcp1:pp1:g0:61",
        f"umbp:vllm:v1:{namespace}:tp1:pcp0:dcp1:pp1:g0:62",
    }
    assert runtime.scheduler_args[1].local_namespace == (1, 0, 1, 1)
    assert runtime.scheduler_args[1].rank_count == 8


def test_embedded_logical_hit_requires_all_tp_dcp_pp_objects(monkeypatch):
    runtime = _EmbeddedRuntime()
    monkeypatch.setitem(
        UMBPRuntimeFactory._builders,
        "embedded",
        lambda config: runtime,
    )
    config = _kv_cache_config()
    vllm_config = _vllm_config(
        {"mode": "embedded", "load_async": False},
        tensor_parallel_rank=0,
        tensor_parallel_size=2,
        pipeline_parallel_rank=0,
        pipeline_parallel_size=2,
        decode_context_parallel_rank=0,
        decode_context_parallel_size=2,
        world_size=8,
    )
    connector = UMBPStoreConnector(
        vllm_config, KVConnectorRole.SCHEDULER, config
    )
    scheduler = connector.connector_scheduler
    assert scheduler is not None
    request = SimpleNamespace(
        request_id="logical",
        num_tokens=32,
        block_hashes=[b"a", b"b"],
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    for block_hash in request.block_hashes:
        runtime.store.update(
            scheduler.codec.keys_for_topology(
                block_hash, scheduler.topology, (0,)
            )
        )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, False)
    runtime.store.remove(
        scheduler.codec.key_for_namespace(
            b"b", 0, (1, 0, 1, 1)
        )
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (16, False)
