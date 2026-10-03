# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.umbp_test_utils import install_memory_runtime
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
    UMBPStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    BlockLoadBatch,
    BlockTransferPlan,
    KVLayoutDescriptor,
    KVLayoutPlanner,
    KVRange,
    KVRegion,
    LoadSpec,
    RankTopology,
    RequestTracker,
    StoreEventResult,
    TransferJobState,
    TransferJobStatus,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    UMBPRuntimeCapabilities,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.factory import (
    UMBPRuntimeConfig,
    UMBPRuntimeFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
    _decode_lazy_block_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.stats import (
    UMBPStoreConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import (
    UMBPStoreConnectorWorker,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


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
        "decode_context_parallel_size": 1,
        "world_size": 1,
    }
    parallel.update(parallel_overrides)
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra,
        ),
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=True,
            prefix_match_unit=None,
        ),
        model_config=SimpleNamespace(
            model="test-model",
            revision="r1",
            max_model_len=128,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=64,
            num_lookahead_slots=0,
        ),
        speculative_config=None,
        kv_events_config=None,
        max_in_flight_tokens=64,
        parallel_config=SimpleNamespace(**parallel),
    )


def test_block_identity_codec_does_not_use_physical_block_id():
    codec = BlockIdentityCodec(UMBPNamespace("namespace"), 2, 3)

    assert codec.key(b"\x01\x02", group_id=4) == (
        "umbp:vllm:v1:namespace:tp2:pcp0:dcp0:pp3:g4:0102"
    )
    assert codec.key(b"\x01\x02", group_id=4) == codec.key(b"\x01\x02", group_id=4)


@pytest.mark.parametrize(
    "identity_field", ["namespace", "tp_rank", "pp_rank", "pcp_rank", "dcp_rank"]
)
def test_lazy_key_cache_preserves_full_object_identity(identity_field):
    codec = BlockIdentityCodec(UMBPNamespace("key-cache"))
    value = UMBPNamespace("other") if identity_field == "namespace" else 1
    changed = replace(codec, **{identity_field: value})
    for current, group, block_hash in (
        (codec, 0, b"a"),
        (changed, 0, b"a"),
        (codec, 1, b"a"),
        (codec, 0, b"b"),
        (codec, 0, b"a"),
    ):
        packed_hash = make_block_hash_with_group_id(block_hash, group)
        assert _decode_lazy_block_hash(current, packed_hash) == (
            group,
            block_hash,
            current.key(block_hash, group),
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


def test_rank_topology_maps_dcp_onto_tp_workers():
    topology = RankTopology(tp_rank=1, tp_size=4, dcp_rank=1, dcp_size=2)

    assert topology.rank_count == 4
    assert topology.all_namespaces() == (
        (0, 0, 0, 0),
        (1, 0, 1, 0),
        (2, 0, 0, 0),
        (3, 0, 1, 0),
    )


def test_layout_descriptor_carries_topology_and_format():
    topology = RankTopology(tp_size=2)
    descriptor = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config()).describe(
        topology, "lbh_nc"
    )

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


@pytest.mark.parametrize("full_token_bounds", [False, True])
def test_layout_planner_builds_compact_group_objects(full_token_bounds):
    planner = KVLayoutPlanner(
        (
            KVRegion("attention.0", 0, 256, 256, 0, 16),
            KVRegion("mamba.0", 1, 512, 512, 256, 32),
            KVRegion("attention.1", 0, 256, 256, 768, 16),
        )
    )
    addresses = {"attention.0": 1000, "mamba.0": 2000, "attention.1": 3000}

    attention = planner.plan_for_block(
        "attention",
        2,
        addresses,
        group_id=0,
        **({"token_start": 0, "token_end": 16} if full_token_bounds else {}),
    )
    mamba = planner.plan_for_block(
        "mamba",
        2,
        addresses,
        group_id=1,
        **({"token_start": 0, "token_end": 32} if full_token_bounds else {}),
    )

    assert [item.layer_name for item in attention.ranges] == [
        "attention.0",
        "attention.1",
    ]
    assert [item.object_offset for item in attention.ranges] == [0, 256]
    assert max(item.object_offset + item.length for item in attention.ranges) == 512
    assert [item.layer_name for item in mamba.ranges] == ["mamba.0"]
    assert [item.object_offset for item in mamba.ranges] == [0]
    assert max(item.object_offset + item.length for item in mamba.ranges) == 512


def test_layout_planner_rejects_unknown_group():
    planner = KVLayoutPlanner((KVRegion("attention", 0, 256, 256, 0, 16),))

    with pytest.raises(ValueError, match="does not contain cache group 1"):
        planner.plan_for_block("missing", 0, {"attention": 1000}, group_id=1)


def test_worker_materializes_only_the_requested_group():
    planner = KVLayoutPlanner(
        (
            KVRegion("attention", 0, 32, 32, 0, 16),
            KVRegion("mamba", 1, 32, 32, 32, 16),
        )
    )
    planner.register_kv_caches(
        {
            "attention": torch.empty((8, 16), dtype=torch.float16),
            "mamba": torch.empty((8, 16), dtype=torch.float16),
        }
    )
    worker = UMBPStoreConnectorWorker(_WorkerHandle(), planner)

    [materialized] = worker._materialize_plans(
        [BlockTransferPlan("mamba-key", 2, group_id=1)]
    )

    assert [item.layer_name for item in materialized.ranges] == ["mamba"]
    assert [item.object_offset for item in materialized.ranges] == [0]


@pytest.mark.parametrize("bounds", [(), (6, 12)])
def test_materialization_preserves_plan_metadata_after_reregistration(bounds):
    """Compact ranges must use current buffers without losing logical identity."""
    planner = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    worker = UMBPStoreConnectorWorker(_WorkerHandle(), planner)
    plan = BlockTransferPlan(
        "key",
        2,
        request_id="req",
        generation=7,
        group_id=0,
        block_hash=b"hash",
        parent_block_hash=b"parent",
        token_ids=(1, 2),
        block_size=16,
        logical_key="logical",
        token_offset=32,
        **(dict(zip(("token_start", "token_end"), bounds)) if bounds else {}),
    )
    previous = None
    for _ in range(2):
        caches = {
            name: torch.empty((8, 2, 16, 8), dtype=torch.float16)
            for name in ("layer1", "layer2")
        }
        planner.register_kv_caches(caches)
        [actual] = worker._materialize_plans([plan])
        expected = planner.plan_registered_block(
            plan.key,
            plan.block_id,
            group_id=0,
            token_start=plan.token_start,
            token_end=plan.token_end,
        )
        assert actual == replace(plan, ranges=expected.ranges)
        assert actual.ranges[0].base_address == (
            caches["layer1"].data_ptr() + 2048 + (384 if bounds else 0)
        )
        if previous is not None:
            assert actual.ranges != previous.ranges
        previous = actual


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
    tracker = RequestTracker(generation=3)

    assert tracker.mark_saved(31, 16) == 16
    assert tracker.mark_saved(15, 16) == 16
    tracker.reset()

    assert tracker.generation == 4
    assert tracker.saved_tokens == 0


def test_load_spec_reports_only_external_tokens():
    spec = LoadSpec(local_tokens=32, external_tokens=48)

    assert spec.num_tokens_to_load == 16


def test_request_tracker_retries_failed_store_suffix():
    tracker = RequestTracker(generation=1)
    tracker.mark_saved(48, 16)
    tracker.record_store_failure(32)
    tracker.record_store_failure(16)

    assert tracker.saved_tokens == 48
    assert tracker.retry_from_tokens == 16
    tracker.clear_store_retry()
    assert tracker.retry_from_tokens is None


def test_worker_metadata_aggregates_store_events_and_block_failures():
    metadata = UMBPConnectorWorkerMetadata(
        failed_block_ids={1},
        store_events={7: StoreEventResult(1, {("c", 1)})},
    )
    other = UMBPConnectorWorkerMetadata(
        failed_block_ids={2},
        store_events={7: StoreEventResult(1, {("c", 2)})},
    )
    metadata.aggregate(other)

    assert metadata.failed_block_ids == {1, 2}
    assert metadata.store_events == {7: StoreEventResult(2, {("c", 1), ("c", 2)})}
    assert other.store_events == {7: StoreEventResult(1, {("c", 2)})}


@pytest.mark.parametrize("mode", ["embedded"])
def test_runtime_config_accepts_local_modes(mode):
    config = UMBPRuntimeConfig.from_vllm(_vllm_config({"mode": mode}))

    assert config.mode == mode


@pytest.mark.parametrize("mode", ["standalone", "distributed"])
def test_runtime_adapter_owns_its_configuration(monkeypatch, mode):
    runtime = SimpleNamespace(capabilities=UMBPRuntimeCapabilities())

    def build(config):
        if not config.options.get("adapter_key"):
            raise ValueError("adapter_key required")
        assert config.rank_count == 4
        return runtime

    monkeypatch.setitem(UMBPRuntimeFactory._builders, mode, build)
    config = UMBPRuntimeConfig.from_vllm(_vllm_config({"mode": mode}))
    with pytest.raises(ValueError, match="adapter_key required"):
        UMBPRuntimeFactory.build(config)
    config = UMBPRuntimeConfig.from_vllm(
        _vllm_config({"mode": mode, "adapter_key": "pool"})
    ).resolve_for_rank_count(4)
    assert UMBPRuntimeFactory.build(config) is runtime


def test_runtime_config_validates_embedded_dram_options():
    config = UMBPRuntimeConfig.from_vllm(
        _vllm_config(
            {
                "mode": "embedded",
                "capacity_bytes": 1024,
                "dram_high_watermark": 0.9,
                "dram_low_watermark": 0.7,
                "dram_use_hugepages": True,
                "dram_hugepage_size": 2 * 1024**2,
                "dram_numa_node": -1,
                "dram_prefault": True,
            }
        )
    )

    assert config.options["dram_high_watermark"] == 0.9
    assert UMBPRuntimeFactory.build(config).options == config.options


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"dram_low_watermark": 0.9, "dram_high_watermark": 0.7}, "must not"),
        ({"dram_use_hugepages": 1}, "boolean"),
        ({"dram_numa_node": -2}, ">= -1"),
        ({"dram_hugepage_size": 0}, "positive integer"),
    ],
)
def test_runtime_config_rejects_invalid_embedded_dram_options(options, error):
    with pytest.raises(ValueError, match=error):
        UMBPRuntimeFactory.build(
            UMBPRuntimeConfig.from_vllm(_vllm_config({"mode": "embedded", **options}))
        )


def test_runtime_factory_builds_embedded_adapter():
    config = UMBPRuntimeConfig("embedded", {})
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
        if isinstance(self.hits, dict):
            return [self.hits.get(key, False) for key in keys]
        return self.hits or [False] * len(keys)

    def close(self):
        pass

    def clear(self):
        self.hits = []
        return True


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
        num_tokens=33,
        block_hashes=[b"a", b"b"],
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (16, False)
    assert handle.queries[0] == [
        "umbp:vllm:v1:test:tp0:pcp0:dcp0:pp0:g0:61",
        "umbp:vllm:v1:test:tp0:pcp0:dcp0:pp0:g0:62",
    ]


def test_scheduler_async_lookup_defers_then_returns_hit(monkeypatch):
    gate = threading.Event()
    key_calls = 0
    context_builds = 0
    original_keys = BlockIdentityCodec.keys_for_topology
    original_context = UMBPStoreConnectorScheduler._build_lookup_context

    def build_context(*args, **kwargs):
        nonlocal context_builds
        context_builds += 1
        return original_context(*args, **kwargs)

    def counted_keys(*args, **kwargs):
        nonlocal key_calls
        key_calls += 1
        return original_keys(*args, **kwargs)

    monkeypatch.setattr(BlockIdentityCodec, "keys_for_topology", counted_keys)
    monkeypatch.setattr(
        UMBPStoreConnectorScheduler, "_build_lookup_context", build_context
    )

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
    for _ in range(3):
        assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    assert key_calls == 1
    gate.set()
    result = (None, False)
    for _ in range(100):
        result = scheduler.get_num_new_matched_tokens(request, 0)
        if result != (None, False):
            break
        time.sleep(0.01)
    scheduler.close()

    # The final prompt token must still execute, leaving one reusable block.
    assert result == (16, False)
    assert key_calls == 1
    assert context_builds == 1


@pytest.mark.parametrize("change", ["local", "length", "hash", "request"])
def test_async_lookup_discards_results_for_changed_request(monkeypatch, change):
    """A completed old query must not be interpreted using a new prefix."""
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"lookup_async": True}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("changed-query")),
    )
    queries = []

    def submit(fn, keys):
        future: Future[list[bool]] = Future()
        queries.append((keys, future))
        return future

    monkeypatch.setattr(scheduler._lookup_executor, "submit", submit)
    request = SimpleNamespace(
        request_id="r", num_tokens=49, block_hashes=[b"a", b"b", b"c"]
    )
    try:
        assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
        queries[0][1].set_result([True] * len(queries[0][0]))
        local_tokens = 0
        if change == "local":
            local_tokens = 16
        elif change == "length":
            request.num_tokens = 65
            request.block_hashes.append(b"d")
        elif change == "hash":
            request.block_hashes[-1] = b"different"
        else:
            request = SimpleNamespace(**vars(request))
        assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (
            None,
            False,
        )
        assert len(queries) == 2
        queries[1][1].set_result([False] * len(queries[1][0]))
        assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (0, False)
    finally:
        scheduler.close()


@pytest.mark.parametrize("lookup_async", [False, True])
@pytest.mark.parametrize("failure", [TimeoutError("lookup timeout"), [True, False]])
def test_scheduler_lookup_failure_recomputes_and_allows_retry(lookup_async, failure):
    class _Handle(_SchedulerHandle):
        def lookup(self, keys):
            if isinstance(self.hits, Exception):
                raise self.hits
            return self.hits

    handle = _Handle(failure)
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"lookup_async": lookup_async, "load_async": False}),
        _kv_cache_config(),
        handle,
        BlockIdentityCodec(UMBPNamespace("lookup-retry")),
    )
    request = SimpleNamespace(
        request_id="retry", num_tokens=32, block_hashes=[b"a", b"b"]
    )
    try:
        for expected in ((0, False), (16, False)):
            deadline = time.monotonic() + 5
            result = scheduler.get_num_new_matched_tokens(request, 0)
            while result == (None, False) and time.monotonic() < deadline:
                time.sleep(0.01)
                result = scheduler.get_num_new_matched_tokens(request, 0)
            assert result == expected
            handle.hits = [True]
    finally:
        scheduler.close()


def test_scheduler_emits_async_load_without_scheduled_model_tokens():
    config = _kv_cache_config()
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": True}),
        config,
        _SchedulerHandle([True]),
        BlockIdentityCodec(UMBPNamespace("async-load")),
    )
    request = SimpleNamespace(
        request_id="async-load",
        num_tokens=32,
        block_hashes=[b"a", b"b"],
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (16, True)
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([7],)),
        16,
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

    assert [plan.block_id for plan in metadata.load_requests["async-load"]] == [7]
    assert scheduler._pending_loads == {}


def test_load_batch_metadata_roundtrip_preserves_destinations_and_identity():
    batch = BlockLoadBatch(["g1", "g0"], [7, 3], [1, 0], "r", 4)
    meta = UMBPConnectorMetadata(load_requests={"r": batch})
    restored = (
        MsgpackDecoder(UMBPConnectorMetadata)
        .decode(MsgpackEncoder().encode(meta))
        .load_requests["r"]
    )
    assert list(restored) == [
        BlockTransferPlan("g1", 7, group_id=1, request_id="r", generation=4),
        BlockTransferPlan("g0", 3, group_id=0, request_id="r", generation=4),
    ]
    assert restored[:1] == [restored[0]]
    with pytest.raises(ValueError, match="equal lengths"):
        BlockLoadBatch(["missing-destination"], [], [0], "r", 4)


def test_scheduler_reset_clears_pending_lookup_state(monkeypatch):
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lookup_async": True}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("reset")),
    )
    future: Future[list[bool]] = Future()
    monkeypatch.setattr(scheduler._lookup_executor, "submit", lambda *a: future)
    request = SimpleNamespace(request_id="req", num_tokens=32, block_hashes=[b"a"])
    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    scheduler._load_specs["req"] = LoadSpec(0, 16)

    assert scheduler.reset_store()
    assert future.cancelled()
    assert scheduler._pending_lookups == {}
    assert scheduler._load_specs == {}
    scheduler.close()


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


@pytest.mark.parametrize("failure", [None, "missing-rank", "exception", "short"])
@pytest.mark.parametrize("restored_tokens", [32, 64])
@pytest.mark.parametrize("null_block", [False, True])
def test_eager_restored_prefix_rechecks_residency_per_group_and_rank(
    failure, restored_tokens, null_block, monkeypatch
):
    """Only currently complete objects may skip store; fresh suffixes still store."""
    config = _kv_cache_config()
    config.kv_cache_groups.append(
        KVCacheGroupSpec(
            ["other"], replace(config.kv_cache_groups[0].kv_cache_spec, block_size=32)
        )
    )
    codec = BlockIdentityCodec(UMBPNamespace("eager-residency"))
    topology = RankTopology(tp_size=2)
    handle = _SchedulerHandle({})
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}, tensor_parallel_size=2, world_size=2),
        config,
        handle,
        codec,
        topology,
    )
    request = SimpleNamespace(
        request_id="req",
        req_id="req",
        num_tokens=65,
        block_hashes=[b"a", b"b", b"c", b"d"],
        block_ids=([1, 2, 3, 4], [NULL_BLOCK_ID if null_block else 5, 6]),
        num_computed_tokens=63,
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: request.block_ids), 0
    )
    tracker = scheduler._request_trackers["req"]
    # The group hit windows can be sparse; a scalar watermark cannot deduplicate.
    tracker.load_spec = LoadSpec(0, restored_tokens, ((None, b"b"), (b"b",)))
    keys = [
        key
        for group, block_size in scheduler.group_block_sizes.items()
        for end in range(block_size, restored_tokens + 1, block_size)
        if request.block_ids[group][end // block_size - 1] != NULL_BLOCK_ID
        for block_hash in (request.block_hashes[end // 16 - 1],)
        for key in codec.keys_for_topology(block_hash, topology, (group,))
    ]
    handle.hits = dict.fromkeys(keys, True)
    if failure == "missing-rank":
        handle.hits[keys[1]] = False
    elif failure == "short":
        handle.hits = [True]
    elif failure == "exception":

        def unavailable(keys):
            raise TimeoutError("lookup unavailable")

        monkeypatch.setattr(handle, "lookup", unavailable)
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"req": 1},
    )
    created = []

    def record_plan(**kwargs):
        plan = BlockTransferPlan(**kwargs)
        created.append(plan)
        return plan

    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler.BlockTransferPlan",
        record_plan,
    )
    try:
        meta = scheduler.build_connector_meta(output)
        expected = {(0, b"c"), (0, b"d"), (1, b"d")} if restored_tokens == 32 else set()
        if failure == "missing-rank":
            expected.add((0, b"a"))
        elif failure is not None:
            expected = {(0, h) for h in request.block_hashes} | {(1, b"b"), (1, b"d")}
        if null_block:
            expected.discard((1, b"b"))
        assert {(p.group_id, p.block_hash) for p in meta.store_plans} == expected
        assert meta.store_requests.get("req", []) == meta.store_plans
        # Resident blocks must not allocate plans that will only be discarded.
        assert len(created) == len(expected)
        if failure is None:
            # A later retry after native eviction must not reuse the earlier hit.
            handle.hits = {}
            tracker.record_store_failure(0)
            retried = scheduler.build_connector_meta(output)
            assert len(retried.store_plans) == 6 - int(null_block)
            assert handle.queries == [keys, keys]
    finally:
        scheduler.close()


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
    block = SimpleNamespace(
        block_id=4,
        block_hash=make_block_hash_with_group_id(b"a", 0),
        is_null=False,
        ref_cnt=0,
    )
    freed = []
    scheduler.bind_gpu_block_pool(
        SimpleNamespace(
            blocks=[None, None, None, None, block],
            free_block_queue=SimpleNamespace(
                iter_blocks_after=lambda cursor: iter((block,))
            ),
            touch=lambda blocks: None,
            free_blocks=lambda blocks: freed.extend(blocks),
        )
    )
    assert scheduler.request_finished(request, ([4, 5],)) == (False, None)
    finished_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={},
    )
    metadata = scheduler.build_connector_meta(finished_output)
    plans = metadata.store_plans

    assert [plan.block_id for plan in plans] == [4]
    assert metadata.store_event == 0
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                store_events={0: StoreEventResult(1)}
            )
        )
    )
    assert freed == [block]
    assert not scheduler.has_pending_push_work()


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("fail_first", [False, True])
@pytest.mark.parametrize("new_generation", [False, True])
def test_store_event_owns_refs_until_all_ranks_finish(lazy, fail_first, new_generation):
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {"mode": "embedded", "lazy_offload": lazy},
            tensor_parallel_size=2,
            world_size=2,
        ),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("lazy-failure")),
        RankTopology(tp_size=2),
    )
    block = SimpleNamespace(
        block_id=4,
        block_hash=make_block_hash_with_group_id(b"a", 0),
        is_null=False,
        ref_cnt=0,
    )
    freed = []
    scheduler.bind_gpu_block_pool(
        SimpleNamespace(
            blocks=[None, None, None, None, block],
            free_block_queue=SimpleNamespace(
                iter_blocks_after=lambda cursor: iter((block,))
            ),
            touch=lambda blocks: None,
            free_blocks=lambda blocks: freed.extend(blocks),
        )
    )
    request = SimpleNamespace(
        request_id="store-failure",
        req_id="store-failure",
        num_tokens=16,
        num_computed_tokens=0,
        block_hashes=[b"a"],
        block_ids=([4],),
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: ([4],)), 0
    )
    if lazy:
        scheduler.request_finished(request, ([4],))
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[] if lazy else [request],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={request.req_id: 16},
    )
    metadata = scheduler.build_connector_meta(output)
    token = (metadata.store_plans[0].key, metadata.store_plans[0].generation)
    tracker = scheduler._request_trackers.get(request.request_id)

    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                store_events={
                    metadata.store_event: StoreEventResult(
                        1, {token} if fail_first else set()
                    )
                }
            )
        )
    )
    assert freed == []
    assert scheduler.has_pending_push_work()
    assert not scheduler.reset_store()
    if new_generation and tracker is not None:
        tracker.reset()

    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                store_events={
                    metadata.store_event: StoreEventResult(
                        1, set() if fail_first else {token}
                    )
                }
            )
        )
    )
    assert freed == [block]
    assert not scheduler.has_pending_push_work()
    assert scheduler.reset_store()
    if tracker is not None:
        assert tracker.retry_from_tokens == (None if new_generation else 0)
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                store_events={metadata.store_event: StoreEventResult(1)}
            )
        )
    )
    assert freed == [block]


def test_store_events_hold_independent_refs_to_the_same_gpu_block():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("event-refs")),
    )
    pinned, freed = [], []
    scheduler.bind_gpu_block_pool(
        SimpleNamespace(
            blocks=list(range(8)),
            touch=lambda blocks: pinned.extend(blocks),
            free_blocks=lambda blocks: freed.extend(blocks),
        )
    )
    events = []
    for request_id in ("first", "second"):
        request = SimpleNamespace(
            request_id=request_id,
            req_id=request_id,
            num_tokens=16,
            num_computed_tokens=0,
            block_hashes=[b"a"],
            block_ids=([4],),
        )
        scheduler.update_state_after_alloc(
            request,
            SimpleNamespace(get_block_ids=lambda group_ids: ([4],)),
            0,
        )
        metadata = scheduler.build_connector_meta(
            SimpleNamespace(
                finished_req_ids=set(),
                preempted_req_ids=set(),
                scheduled_new_reqs=[request],
                scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
                num_scheduled_tokens={request_id: 16},
            )
        )
        events.append(metadata.store_event)
    assert pinned == [4, 4]
    assert events == [0, 1]

    for event_id in (1, 1, 0):
        scheduler.update_connector_output(
            SimpleNamespace(
                kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                    store_events={event_id: StoreEventResult(1)},
                ),
            )
        )
        assert len(freed) == (2 if event_id == 0 else 1)
        assert scheduler.has_pending_push_work() == (event_id != 0)


def test_lazy_store_resident_block_satisfies_free_queue_watermark():
    handle = _SchedulerHandle([])
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lazy_offload": True}),
        _kv_cache_config(),
        handle,
        BlockIdentityCodec(UMBPNamespace("lazy-reselect")),
    )
    block = SimpleNamespace(
        block_id=4,
        block_hash=make_block_hash_with_group_id(b"a", 0),
        is_null=False,
        ref_cnt=0,
    )

    scheduler.bind_gpu_block_pool(
        SimpleNamespace(
            blocks=[None, None, None, None, block],
            free_block_queue=SimpleNamespace(
                iter_blocks_after=lambda cursor: iter((block,))
            ),
            touch=lambda blocks: None,
            free_blocks=lambda blocks: None,
        )
    )
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={},
    )

    scheduler.request_finished(SimpleNamespace(request_id="first"), ([],))
    first = scheduler.build_connector_meta(output)
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                store_events={first.store_event: StoreEventResult(1)}
            )
        )
    )
    handle.hits = [True]
    scheduler.request_finished(SimpleNamespace(request_id="second"), ([],))
    second = scheduler.build_connector_meta(output)

    assert len(first.store_plans) == 1
    assert second.store_plans == []
    assert handle.queries == [
        [first.store_plans[0].key],
        [first.store_plans[0].key],
    ]


def test_lazy_store_rechecks_runtime_residency_after_publication_and_eviction():
    handle = _SchedulerHandle([])
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lazy_offload": True}),
        _kv_cache_config(),
        handle,
        BlockIdentityCodec(UMBPNamespace("lazy-resident")),
    )
    block = SimpleNamespace(
        block_id=4,
        block_hash=make_block_hash_with_group_id(b"a", 0),
        is_null=False,
        ref_cnt=0,
    )
    scheduler.bind_gpu_block_pool(
        SimpleNamespace(
            blocks=[None, None, None, None, block],
            free_block_queue=SimpleNamespace(
                iter_blocks_after=lambda cursor: iter((block,))
            ),
            touch=lambda blocks: None,
            free_blocks=lambda blocks: None,
        )
    )
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={},
    )

    first = scheduler.build_connector_meta(output)
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                store_events={first.store_event: StoreEventResult(1)}
            )
        )
    )
    handle.hits = [True]
    second = scheduler.build_connector_meta(output)
    handle.hits = [False]
    third = scheduler.build_connector_meta(output)

    assert len(first.store_plans) == 1
    assert second.store_plans == []
    assert [plan.key for plan in third.store_plans] == [first.store_plans[0].key]
    assert handle.queries == [
        [first.store_plans[0].key],
        [first.store_plans[0].key],
        [first.store_plans[0].key],
    ]


def test_lazy_target_matches_cache_group_geometry():
    assert (
        UMBPStoreConnectorScheduler._estimate_lazy_target_blocks(
            _hybrid_kv_cache_config(),
            max_num_batched_tokens=64,
            dcp_size=1,
        )
        == 12
    )


def test_layerwise_load_groups_consecutive_layers_into_stages():
    handle = _LayerRecordingWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle, layerwise_load_stages=2)
    plan = BlockTransferPlan(
        key="staged",
        block_id=1,
        ranges=tuple(
            KVRange(f"layer{i}", 0, 1, 1000 * (i + 1), 16, 16, 16 * i) for i in range(4)
        ),
    )
    worker.start_load_kv(
        None, UMBPConnectorMetadata(async_load=False, load_requests={"req": [plan]})
    )

    assert [
        sorted(item.layer_name for item in calls[0].ranges)
        for calls in handle.load_calls
    ] == [["layer0", "layer1"], ["layer2", "layer3"]]
    worker.wait_for_layer_load("layer0")
    worker.wait_for_layer_load("layer1")
    assert len(handle.wait_calls) == 1
    worker.wait_for_layer_load("layer2")
    worker.wait_for_layer_load("layer3")
    assert len(handle.wait_calls) == 2


def test_hybrid_model_loads_asynchronously_even_when_sync_is_requested():
    codec = BlockIdentityCodec(UMBPNamespace("hybrid-sync"))
    block_hashes = [b"a", b"b"]
    hits = {
        codec.key(block_hashes[0], 0): True,
        codec.key(block_hashes[1], 0): True,
        codec.key(block_hashes[1], 1): True,
    }
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": False}),
        _hybrid_kv_cache_config(),
        _SchedulerHandle(hits),
        codec,
    )
    request = SimpleNamespace(
        request_id="hybrid-sync", block_hashes=block_hashes, num_tokens=33
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, True)
    scheduler.close()


@pytest.mark.parametrize("lookup_async", [False, True])
def test_hybrid_lookup_uses_latest_mamba_checkpoint(lookup_async, monkeypatch):
    codec = BlockIdentityCodec(UMBPNamespace("hybrid-hit-window"))
    block_hashes = [b"a", b"b"]
    hits = {
        codec.key(block_hashes[0], 0): True,
        codec.key(block_hashes[1], 0): True,
        codec.key(block_hashes[1], 1): True,
    }
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {"mode": "embedded", "lazy_offload": True, "lookup_async": lookup_async}
        ),
        _hybrid_kv_cache_config(),
        _SchedulerHandle(hits),
        codec,
    )
    pool = scheduler._lookup_coordinator.block_pool
    for operation in ("get_new_blocks", "_insert_block_hash", "touch", "free_blocks"):
        monkeypatch.setattr(
            pool, operation, lambda *a, **kw: pytest.fail("lookup must be read-only")
        )
    request = SimpleNamespace(
        request_id="hybrid-lookup",
        block_hashes=block_hashes,
        num_tokens=33,
    )

    matched, is_async = scheduler.get_num_new_matched_tokens(request, 0)
    deadline = time.monotonic() + 5
    while matched is None and time.monotonic() < deadline:
        time.sleep(0.001)
        matched, is_async = scheduler.get_num_new_matched_tokens(request, 0)

    assert (matched, is_async) == (32, True)
    spec = scheduler._load_specs[request.request_id]
    assert spec.block_hashes_by_group == (
        (block_hashes[0], block_hashes[1]),
        (None, block_hashes[1]),
    )

    tracker = RequestTracker(generation=7)
    tracker.load_spec = spec
    plans = scheduler._load_plans_for_external_tokens(
        request,
        tracker,
        ([10, 11], [NULL_BLOCK_ID, 12]),
        matched,
    )
    assert [(plan.group_id, plan.block_id, plan.key) for plan in plans] == [
        (0, 10, codec.key(block_hashes[0], 0)),
        (0, 11, codec.key(block_hashes[1], 0)),
        (1, 12, codec.key(block_hashes[1], 1)),
    ]
    scheduler.close()


@pytest.mark.parametrize("failure_stage", ["lookup", "match"])
def test_hybrid_shadow_lookup_cleans_up_after_failure(monkeypatch, failure_stage):
    """A failed query cannot leak shadow hits or consume pool capacity."""
    handle = _SchedulerHandle({})
    codec = BlockIdentityCodec(UMBPNamespace("shadow-cleanup"))
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({}), _hybrid_kv_cache_config(), handle, codec
    )
    request = SimpleNamespace(request_id="r", num_tokens=33, block_hashes=[b"a", b"b"])
    handle.hits = {codec.key(h, g): True for h in request.block_hashes for g in (0, 1)}
    coordinator = scheduler._lookup_coordinator
    pool = coordinator.block_pool
    free_blocks = pool.get_num_free_blocks()

    def fail(*args, **kwargs):
        raise RuntimeError("injected shadow failure")

    try:
        with monkeypatch.context() as patch:
            if failure_stage == "lookup":
                lookup = pool.get_cached_block

                def fail_after_lookup(*args, **kwargs):
                    lookup(*args, **kwargs)
                    fail()

                patch.setattr(pool, "get_cached_block", fail_after_lookup)
            else:
                patch.setattr(coordinator, "find_longest_cache_hit", fail)
            with pytest.raises(RuntimeError, match="injected shadow failure"):
                scheduler.get_num_new_matched_tokens(request, 0)
        assert pool.get_num_free_blocks() == free_blocks
        assert len(pool.cached_block_hash_to_block) == 0
        assert not pool.cached_block_hashes_by_block
        assert pool.hits == []
        assert pool.hit_blocks == {}
        handle.hits = {}
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        handle.hits = {
            codec.key(h, g): True for h in request.block_hashes for g in (0, 1)
        }
        assert scheduler.get_num_new_matched_tokens(request, 0) == (32, True)
        assert pool.get_num_free_blocks() == free_blocks
        assert len(pool.cached_block_hash_to_block) == 0
    finally:
        scheduler.close()


def test_lazy_store_publishes_secondary_block_hashes():
    handle = _SchedulerHandle([])
    codec = BlockIdentityCodec(UMBPNamespace("lazy-secondary"))
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lazy_offload": True}),
        _kv_cache_config(),
        handle,
        codec,
    )
    primary = make_block_hash_with_group_id(b"a", 0)
    secondary = make_block_hash_with_group_id(b"b", 0)
    block = SimpleNamespace(
        block_id=4,
        block_hash=primary,
        is_null=False,
        ref_cnt=0,
    )
    scheduler.bind_gpu_block_pool(
        SimpleNamespace(
            blocks=[None, None, None, None, block],
            cached_block_hashes_by_block={block.block_id: {secondary}},
            free_block_queue=SimpleNamespace(
                iter_blocks_after=lambda cursor: iter((block,))
            ),
            touch=lambda blocks: None,
            free_blocks=lambda blocks: None,
        )
    )
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={},
    )

    metadata = scheduler.build_connector_meta(output)

    assert {(plan.block_id, plan.key) for plan in metadata.store_plans} == {
        (block.block_id, codec.key(b"a", 0)),
        (block.block_id, codec.key(b"b", 0)),
    }


@pytest.mark.parametrize("aliased_head", [False, True])
def test_lazy_store_closes_full_attention_prefix_from_at_risk_tail(aliased_head):
    handle = _SchedulerHandle([])
    codec = BlockIdentityCodec(UMBPNamespace("lazy-prefix-closure"))
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lazy_offload": True}),
        _kv_cache_config(),
        handle,
        codec,
    )
    head = SimpleNamespace(
        block_id=3,
        block_hash=make_block_hash_with_group_id(b"a", 0),
        is_null=False,
        ref_cnt=0,
    )
    tail = SimpleNamespace(
        block_id=4,
        block_hash=make_block_hash_with_group_id(b"b", 0),
        is_null=False,
        ref_cnt=0,
    )
    aliases = {}
    if aliased_head:
        aliases[head.block_id] = {head.block_hash}
        head.block_hash = make_block_hash_with_group_id(b"other", 0)
    scheduler.bind_gpu_block_pool(
        SimpleNamespace(
            blocks=[None, None, None, head, tail],
            cached_block_hashes_by_block=aliases,
            free_block_queue=SimpleNamespace(
                iter_blocks_after=lambda cursor: iter((tail,))
            ),
            touch=lambda blocks: None,
            free_blocks=lambda blocks: None,
        )
    )
    request = SimpleNamespace(
        request_id="lazy-prefix-closure",
        num_tokens=33,
        num_prompt_tokens=33,
        num_computed_tokens=33,
        block_hashes=[b"a", b"b"],
    )
    scheduler.request_finished(request, ([head.block_id, tail.block_id],))
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={},
    )

    metadata = scheduler.build_connector_meta(output)

    assert [(plan.block_id, plan.key) for plan in metadata.store_plans] == [
        (tail.block_id, codec.key(b"b", 0)),
        (head.block_id, codec.key(b"a", 0)),
    ]


@pytest.mark.parametrize("reverse", [False, True])
def test_lazy_prefix_closure_checks_each_chain_block_once_per_scan(reverse):
    """Keep prefix completeness without a quadratic walk or stale cross-step cache."""

    class CountingBlocks(list):
        reads = 0

        def __getitem__(self, index):
            self.reads += 1
            return super().__getitem__(index)

    count = 32
    hashes = [i.to_bytes(2, "big") for i in range(count)]
    nodes = [
        SimpleNamespace(
            block_id=i + 1,
            block_hash=make_block_hash_with_group_id(h, 0),
            is_null=False,
            ref_cnt=0,
        )
        for i, h in enumerate(hashes)
    ]
    blocks = CountingBlocks([None, *nodes])
    codec = BlockIdentityCodec(UMBPNamespace("linear-prefix"))
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {"mode": "embedded", "lazy_offload": True, "lazy_offload_max_blocks": count}
        ),
        _kv_cache_config(),
        _SchedulerHandle([]),
        codec,
    )
    scheduler.bind_gpu_block_pool(
        SimpleNamespace(
            blocks=blocks,
            cached_block_hashes_by_block={},
            free_block_queue=SimpleNamespace(
                iter_blocks_after=lambda cursor: (
                    reversed(nodes) if reverse else iter(nodes)
                )
            ),
            touch=lambda blocks: None,
            free_blocks=lambda blocks: None,
        )
    )
    request = SimpleNamespace(
        request_id="chain",
        num_tokens=count * 16 + 1,
        num_prompt_tokens=count * 16 + 1,
        num_computed_tokens=count * 16 + 1,
        block_hashes=hashes,
    )
    scheduler.request_finished(request, (list(range(1, count + 1)),))
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        num_scheduled_tokens={},
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
    )
    for iteration in range(2):
        if iteration:
            hashes[7] = b"recycled"
            blocks[8].block_hash = make_block_hash_with_group_id(hashes[7], 0)
        blocks.reads = 0
        metadata = scheduler.build_connector_meta(output)
        assert {plan.key for plan in metadata.store_plans} == {
            codec.key(h, 0) for h in hashes
        }
        assert blocks.reads <= 2 * count
        scheduler.update_connector_output(
            SimpleNamespace(
                kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                    store_events={metadata.store_event: StoreEventResult(1)}
                )
            )
        )
    scheduler.close()


def test_scheduler_resumed_request_replaces_stale_block_table():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {
                "mode": "embedded",
                "save_decode_cache": True,
            }
        ),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("resumed")),
    )
    request = SimpleNamespace(
        request_id="resumed",
        req_id="resumed",
        num_tokens=32,
        num_computed_tokens=0,
        block_hashes=[b"a", b"b"],
        block_ids=([7, 8],),
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2],)),
        0,
    )
    old_generation = scheduler._request_trackers["resumed"].generation
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["resumed"],
                new_block_ids=[([7, 8],)],
                num_computed_tokens=[31],
                resumed_req_ids={"resumed"},
            ),
            num_scheduled_tokens={"resumed": 1},
        )
    )

    tracker = scheduler._request_trackers["resumed"]
    assert tracker.block_ids == ([7, 8],)
    assert tracker.generation == old_generation + 1
    assert [plan.block_id for plan in metadata.store_plans] == [7, 8]


@pytest.mark.parametrize("boundary", [0, 3, 8, 11, 12, 16])
def test_scheduler_accepts_only_computed_hash_aligned_tail(boundary):
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": False, "hash_block_size": 4}),
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
        num_computed_tokens=12,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([7],)),
        0,
    )

    pool = BlockPool(8, True, 4)
    scheduler.bind_gpu_block_pool(pool)
    assert not scheduler.register_finished_partial_tail(
        request, ([7],), [(0, 7, boundary)]
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

    assert len(metadata.store_plans) == (1 if boundary == 12 else 0)
    if metadata.store_plans:
        tail = metadata.store_plans[0]
        assert (tail.token_start, tail.token_end) == (None, None)
        assert tail.block_id == 7
        assert tail.block_hash == b"c"


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

    def poll(self, job):
        if job.status in (
            TransferJobStatus.COMPLETED,
            TransferJobStatus.FAILED,
            TransferJobStatus.CANCELLED,
        ):
            return job
        return None

    def publish(self, job):
        self.published = job

    def close(self):
        pass


class _LayerRecordingWorkerHandle(_WorkerHandle):
    def __init__(self, fail_first_load=False):
        self.load_calls = []
        self.wait_calls = []
        self.fail_first_load = fail_first_load

    def load(self, plans):
        self.load_calls.append(list(plans))
        job = super().load(plans)
        if self.fail_first_load and len(self.load_calls) == 1:
            job.completed_keys.clear()
            job.fail([plan.key for plan in plans], "first layer failed")
        return job

    def wait(self, job):
        self.wait_calls.append(job)
        return super().wait(job)

    def poll(self, job):
        return None


class _DelayedLoadWorkerHandle(_WorkerHandle):
    def load(self, plans):
        self.load_job = TransferJobState(tuple(plans))
        self.load_job.start()
        return self.load_job


@pytest.mark.parametrize("fail_load", [False, True])
def test_worker_polls_async_load_without_forward_execution(fail_load):
    handle = _DelayedLoadWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle, layerwise_load=True)
    plan = BlockTransferPlan("async-load", 1, request_id="req")
    worker.start_load_kv(
        None,
        UMBPConnectorMetadata(
            async_load=True,
            load_requests={"req": [plan]},
        ),
    )

    assert worker.get_finished(set()) == (None, None)

    if fail_load:
        handle.load_job.fail([plan.key], "load failed")
    else:
        handle.load_job.complete()

    assert worker.get_finished(set()) == (None, {"req"})
    assert worker.get_failed_recving() == ({"req"} if fail_load else set())
    assert worker.get_failed_recving() == set()
    assert worker.get_block_ids_with_load_errors() == ({1} if fail_load else set())


class _WaitRecordingDelayedLoadHandle(_DelayedLoadWorkerHandle):
    def __init__(self):
        self.waited = []

    def wait(self, job):
        self.waited.append(job)
        job.complete()
        return job


@pytest.mark.parametrize("layerwise", [False, True])
def test_forward_does_not_wait_for_async_loads(layerwise):
    """Async loads fill blocks of requests the running batch does not contain."""
    handle = _WaitRecordingDelayedLoadHandle()
    worker = UMBPStoreConnectorWorker(handle, layerwise_load=layerwise)
    plan = BlockTransferPlan(
        "async-load",
        1,
        request_id="req",
        ranges=(KVRange("layer1", 0, 1, 1000, 16, 16, 0),),
    )
    worker.start_load_kv(
        None, UMBPConnectorMetadata(async_load=True, load_requests={"req": [plan]})
    )

    worker.wait_for_layer_load("layer1")
    worker.wait_for_layer_load("")

    assert handle.waited == []
    assert worker.get_finished(set()) == (None, None)
    # Blocks of a preempted batch may be reused, so their loads must settle.
    worker.handle_preemptions(UMBPConnectorMetadata(preempted_block_ids={1}))
    assert handle.waited == [handle.load_job]
    assert worker.get_finished(set()) == (None, {"req"})


class _CancellableWorkerHandle(_WorkerHandle):
    def __init__(self):
        self.cancelled = []

    def cancel(self, job):
        self.cancelled.append(job)
        job.cancel("preempted")
        return job


class _DelayedStoreWorkerHandle(_CancellableWorkerHandle):
    def __init__(self):
        super().__init__()
        self.jobs = []
        self.waited = []
        self.publications = []

    def store(self, plans):
        job = TransferJobState(tuple(plans))
        job.start()
        self.jobs.append(job)
        return job

    def wait(self, job):
        self.waited.append(job)
        job.complete()
        return job

    def publish(self, job):
        self.publications.append(job)


@pytest.mark.parametrize("fail_first", [False, True])
def test_async_stores_keep_independent_events_across_steps(fail_first):
    """Later stores of one request must not unpin its earlier in-flight KV."""
    handle = _DelayedStoreWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    for event_id in (1, 2):
        plan = BlockTransferPlan(str(event_id), event_id, request_id="req")
        worker.enqueue_stores(
            UMBPConnectorMetadata(store_event=event_id, store_plans=[plan])
        )
        worker.wait_for_save()
        assert worker.get_finished({"req"}) == (None, None)
        assert worker.build_connector_worker_meta().store_events == {}
    assert not handle.waited
    assert not handle.publications

    handle.jobs[1].complete()
    worker.get_finished(set())
    assert worker.build_connector_worker_meta().store_events == {2: StoreEventResult(1)}
    assert handle.publications == [handle.jobs[1]]
    if fail_first:
        handle.jobs[0].fail(["1"], "copy failed")
    else:
        handle.jobs[0].complete()
    worker.get_finished(set())
    assert worker.build_connector_worker_meta().store_events == {
        1: StoreEventResult(1, {("1", 0)} if fail_first else set())
    }
    assert len(handle.publications) == (1 if fail_first else 2)
    worker.get_finished(set())
    assert worker.build_connector_worker_meta().store_events == {}
    assert not handle.waited


def test_preemption_cancels_pending_stores_from_previous_steps():
    handle = _DelayedStoreWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    for event_id, request_id in enumerate(("first", "other", "first")):
        worker.enqueue_stores(
            UMBPConnectorMetadata(
                store_event=event_id,
                store_plans=[
                    BlockTransferPlan(str(event_id), event_id, request_id=request_id)
                ],
            )
        )
        worker.wait_for_save()
    worker.handle_preemptions(UMBPConnectorMetadata(preempted_request_ids={"first"}))
    worker.get_finished(set())
    assert handle.cancelled == [handle.jobs[0], handle.jobs[2]]
    assert worker.build_connector_worker_meta().store_events == {
        0: StoreEventResult(1, {("0", 0)}),
        2: StoreEventResult(1, {("2", 0)}),
    }
    assert not handle.publications
    assert not handle.waited
    handle.jobs[1].complete()
    worker.get_finished(set())
    assert worker.build_connector_worker_meta().store_events == {1: StoreEventResult(1)}


@pytest.mark.parametrize("action", ["preempt_blocks", "close"])
def test_async_store_drains_before_buffers_can_be_reused(action):
    handle = _DelayedStoreWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    worker.enqueue_stores(
        UMBPConnectorMetadata(store_event=4, store_plans=[BlockTransferPlan("kv", 1)])
    )
    worker.wait_for_save()
    assert not handle.waited
    if action == "close":
        worker.close()
    else:
        worker.handle_preemptions(UMBPConnectorMetadata(preempted_block_ids={1}))
    assert handle.waited == handle.jobs
    assert worker.build_connector_worker_meta().store_events == {4: StoreEventResult(1)}


def test_worker_preemption_cancels_only_matching_request():
    handle = _CancellableWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    first = BlockTransferPlan("first", 1, request_id="first")
    second = BlockTransferPlan("second", 2, request_id="second")
    worker.enqueue_stores(
        UMBPConnectorMetadata(
            store_event=21,
            store_plans=[first, second],
            store_requests={"first": [first], "second": [second]},
        )
    )

    worker.handle_preemptions(UMBPConnectorMetadata(preempted_request_ids={"first"}))

    assert len(handle.cancelled) == 1
    assert handle.cancelled[0].plans == (first,)
    assert worker.build_connector_worker_meta().store_events == {}
    worker.wait_for_save()
    assert handle.published.plans == (second,)
    assert worker.build_connector_worker_meta().store_events == {
        21: StoreEventResult(1, {("first", 0)})
    }


@pytest.mark.parametrize("poll_first", [False, True])
@pytest.mark.parametrize("fail_first", [False, True])
def test_worker_waits_for_layers_independently(poll_first, fail_first):
    handle = _LayerRecordingWorkerHandle(fail_first_load=fail_first)
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
            async_load=False,
            load_requests={"req": [plan]},
        ),
    )

    assert len(handle.load_calls) == 2
    if poll_first:
        worker.get_finished(set())
    worker.wait_for_layer_load("layer1")
    assert len(handle.wait_calls) == 1
    assert worker.get_finished({"req"}) == (None, None)
    assert worker.get_failed_recving() == set()
    worker.wait_for_layer_load("layer2")
    assert len(handle.wait_calls) == 2
    worker.wait_for_layer_load("")
    assert len(handle.wait_calls) == 2
    assert worker.get_finished({"req"}) == (None, None)
    assert worker.get_failed_recving() == ({"req"} if fail_first else set())
    assert worker.get_failed_recving() == set()
    assert worker.get_kv_connector_stats().reduce()["load_num_bytes"] == (
        16 if fail_first else 32
    )


@pytest.mark.parametrize("layerwise", [False, True])
def test_load_cancellation_preserves_other_requests(layerwise):
    handle = _CancellableWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle, layerwise_load=layerwise)
    plans = {
        name: [
            BlockTransferPlan(
                name,
                block_id,
                request_id=name,
                ranges=(KVRange("layer1", 0, block_id, 1000, 16, 16, 0),),
            )
        ]
        for name, block_id in (("first", 1), ("second", 2))
    }
    worker.start_load_kv(
        None,
        UMBPConnectorMetadata(
            async_load=not layerwise,
            load_requests=plans,
        ),
    )
    worker.handle_preemptions(UMBPConnectorMetadata(preempted_request_ids={"first"}))
    worker.wait_for_layer_load("")
    # Asynchronous loads settle through get_finished, not the forward pass.
    finished = worker.get_finished(set())

    assert len(handle.cancelled) == 1
    assert handle.cancelled[0].plans[0].request_id == "first"
    assert worker.get_kv_connector_stats().reduce()["load_completed"] == 1
    assert finished == (None, None if layerwise else {"second"})
    worker.wait_for_layer_load("")
    assert worker.get_kv_connector_stats() is None


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
            async_load=False,
            load_requests={"req": [plan]},
        ),
    )
    worker.wait_for_layer_load("layer1")

    assert len(handle.load_calls) == 1
    assert handle.loaded_plans == [plan]
    assert worker.get_finished({"req"}) == (None, None)


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


def test_worker_reports_load_completion_and_store_event_without_deferring_request():
    worker = UMBPStoreConnectorWorker(_WorkerHandle())
    metadata = UMBPConnectorMetadata(
        async_load=True,
        store_event=7,
        store_plans=[BlockTransferPlan("store", 4)],
        load_requests={"req": [BlockTransferPlan("load", 3)]},
        store_requests={"req": [BlockTransferPlan("store", 4)]},
    )

    worker.start_load_kv(None, metadata)
    worker.wait_for_layer_load("layer0")
    worker.enqueue_stores(metadata)
    worker.wait_for_save()
    result = worker.build_connector_worker_meta()

    assert result.store_events == {7: StoreEventResult(1)}
    assert worker.get_finished({"req"}) == (None, {"req"})


def test_worker_partial_store_failure_reports_unpublished_objects():
    class _PartialFailureHandle(_WorkerHandle):
        def store(self, plans):
            job = TransferJobState(tuple(plans))
            job.start()
            job.complete(["ok"])
            job.fail(["bad"], "store failed")
            return job

    worker = UMBPStoreConnectorWorker(_PartialFailureHandle())
    plans = [
        BlockTransferPlan("ok", 11, generation=3),
        BlockTransferPlan("bad", 12, generation=3),
    ]
    worker.enqueue_stores(UMBPConnectorMetadata(store_plans=plans, store_event=9))
    worker.wait_for_save()

    assert worker.get_finished(set()) == (None, None)
    assert worker.get_block_ids_with_load_errors() == set()
    metadata = worker.build_connector_worker_meta()
    assert metadata.store_events == {9: StoreEventResult(1, {("ok", 3), ("bad", 3)})}
    assert not hasattr(worker.runtime, "published")
    assert worker.get_kv_connector_stats().reduce()["store_failed"] == 1


@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("layerwise_store", [False, True])
def test_connector_finalizes_each_store_once_when_forward_hook_is_skipped(
    monkeypatch, forward, layerwise_store
):
    """No-forward steps must not strand scheduler events or resubmit stores."""
    runtime = _EmbeddedRuntime()
    runtime.capabilities = UMBPRuntimeCapabilities(layerwise_store=layerwise_store)
    monkeypatch.setitem(
        UMBPRuntimeFactory._builders, "embedded", lambda config: runtime
    )
    connector = UMBPStoreConnector(
        _vllm_config({"mode": "embedded"}), KVConnectorRole.WORKER, _kv_cache_config()
    )
    for event_id in (7, 8):
        plan = BlockTransferPlan(
            key=f"store-{event_id}",
            block_id=1,
            ranges=(KVRange("layer1", 0, 1, 1000, 16, 16, 0),),
        )
        connector.bind_connector_metadata(
            UMBPConnectorMetadata(store_plans=[plan], store_event=event_id)
        )
        if forward:
            connector.wait_for_save()
            connector.wait_for_save()
        # Both model runners collect completions before clearing metadata.
        before_clear = connector.build_connector_worker_meta().store_events
        connector.clear_connector_metadata()
        connector.clear_connector_metadata()
        assert not connector.has_connector_metadata()

        # A no-forward store completion must survive until the next step.
        connector.bind_connector_metadata(UMBPConnectorMetadata())
        connector.wait_for_save()
        next_step = connector.build_connector_worker_meta().store_events
        connector.clear_connector_metadata()
        expected = {event_id: StoreEventResult(1)}
        assert before_clear == (expected if forward else {})
        assert next_step == ({} if forward else expected)
        assert plan.key in runtime.store

    stats = connector.get_kv_connector_stats().reduce()
    assert stats["store_submitted"] == 2
    assert stats["store_completed"] == 2


@pytest.mark.parametrize("enable_events", [False, True])
def test_worker_emits_block_stored_event_after_store_completion(enable_events):
    worker = UMBPStoreConnectorWorker(
        _WorkerHandle(), enable_kv_cache_events=enable_events
    )
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
    events = worker.get_kv_events()
    result = worker.build_connector_worker_meta()

    assert result.store_events == {}
    if not enable_events:
        assert events == []
        return
    assert len(events) == 1
    assert events[0].block_hashes == [maybe_convert_block_hash(b"event-hash")]
    assert events[0].parent_block_hash == maybe_convert_block_hash(b"parent")
    assert events[0].token_ids == [1, 2, 3, 4]
    assert worker.get_kv_events() == []


@pytest.mark.parametrize("enable_events", [False, True])
def test_worker_emits_block_removed_for_runtime_eviction(enable_events):
    class _EvictingWorkerHandle(_WorkerHandle):
        evicted_keys = ["umbp:vllm:v1:test:tp0:pcp0:dcp0:pp0:g2:65766963746564"]

        def take_evicted_keys(self):
            keys, self.evicted_keys = self.evicted_keys, []
            return keys

    handle = _EvictingWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle, enable_kv_cache_events=enable_events)

    events = worker.get_kv_events()
    assert handle.evicted_keys == []
    if not enable_events:
        assert events == []
        return
    [event] = events

    assert event.block_hashes == [maybe_convert_block_hash(b"evicted")]
    assert event.group_idx == 2
    assert event.medium == "CPU"


def test_umbp_stats_aggregate_and_reduce():
    first = UMBPStoreConnectorStats()
    first.record("load", submitted=2, completed=1, failed=1, num_bytes=64)
    second = UMBPStoreConnectorStats(
        {"load": {"completed": 1, "num_bytes": 32, "unknown": 7}, "store": {}}
    )

    merged = first.aggregate(second)

    assert merged.reduce() == {
        "load_submitted": 2,
        "load_completed": 2,
        "load_failed": 1,
        "load_num_bytes": 96,
        "store_submitted": 0,
        "store_completed": 0,
        "store_failed": 0,
        "store_num_bytes": 0,
    }
    assert first.data["load"]["completed"] == 1
    assert second.data["load"] == {"completed": 1, "num_bytes": 32, "unknown": 7}


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
        load_requests={"req": [supplied]},
    )

    worker.start_load_kv(None, metadata)

    assert handle.loaded_plans == [supplied]


@pytest.mark.parametrize("enable_events", [False, True])
@pytest.mark.parametrize("layerwise_store", [False, True])
def test_embedded_connector_core_flow(monkeypatch, enable_events, layerwise_store):
    runtime = _EmbeddedRuntime()
    runtime.capabilities = UMBPRuntimeCapabilities(layerwise_store=layerwise_store)
    monkeypatch.setitem(
        UMBPRuntimeFactory._builders,
        "embedded",
        lambda config: runtime,
    )
    config = _kv_cache_config()
    vllm_config = _vllm_config({"mode": "embedded", "load_async": False})
    if enable_events:
        vllm_config.kv_events_config = SimpleNamespace(enable_kv_cache_events=True)
    producer = SimpleNamespace(
        request_id="producer",
        req_id="producer",
        num_tokens=32,
        prompt_token_ids=list(range(32)),
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
    worker_connector = UMBPStoreConnector(vllm_config, KVConnectorRole.WORKER, config)
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
    scheduler_connector.update_state_after_alloc(
        producer,
        SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2],)),
        0,
    )
    metadata = scheduler_connector.build_connector_meta(scheduler_output)
    assert metadata.store_plans_by_layer == (
        {name: metadata.store_plans for name in ("layer1", "layer2")}
        if layerwise_store
        else {}
    )
    assert [plan.token_ids for plan in metadata.store_plans] == (
        [tuple(range(16)), tuple(range(16, 32))] if enable_events else [(), ()]
    )
    assert [plan.parent_block_hash for plan in metadata.store_plans] == (
        [None, b"a"] if enable_events else [None, None]
    )
    worker_connector.bind_connector_metadata(metadata)
    worker_connector.wait_for_save()
    assert runtime.store
    events = worker_connector.get_kv_connector_kv_cache_events()
    if enable_events:
        assert events is not None
        assert len(events.get_all_events()) == 2
    else:
        assert events is None

    consumer = SimpleNamespace(
        request_id="consumer",
        req_id="consumer",
        num_tokens=33,
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
    assert not worker_connector.get_transfer_results({"consumer"}).finished_recving


def test_embedded_runtime_register_store_and_load(monkeypatch):
    install_memory_runtime(monkeypatch)
    config = _kv_cache_config()
    vllm_config = _vllm_config(
        {
            "mode": "embedded",
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
    worker_connector = UMBPStoreConnector(vllm_config, KVConnectorRole.WORKER, config)
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
    scheduler_connector.update_state_after_alloc(
        producer,
        SimpleNamespace(get_block_ids=lambda group_ids: producer.block_ids),
        0,
    )
    store_metadata = scheduler_connector.build_connector_meta(producer_output)
    worker_connector.bind_connector_metadata(store_metadata)
    worker_connector.wait_for_save()

    consumer = SimpleNamespace(
        request_id="builtin-consumer",
        req_id="builtin-consumer",
        num_tokens=33,
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
    assert worker_connector.get_finished({"builtin-consumer"}) == (None, None)
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


@pytest.mark.parametrize(("tp_size", "dcp_size"), [(2, 1), (2, 2), (4, 2), (8, 8)])
def test_embedded_tp_dcp_rank_store_completeness(monkeypatch, tp_size, dcp_size):
    """A prefix is reusable only after every actual TP worker publishes."""
    install_memory_runtime(monkeypatch)
    config = _kv_cache_config()
    extra = {
        "mode": "embedded",
        "load_async": False,
        "key_namespace": f"builtin-tp{tp_size}-dcp{dcp_size}-test",
    }
    parallel = {
        "tensor_parallel_size": tp_size,
        "decode_context_parallel_size": dcp_size,
        "world_size": tp_size,
    }
    token_count = 32 * dcp_size
    request = SimpleNamespace(
        request_id="ranked-producer",
        req_id="ranked-producer",
        num_tokens=token_count,
        block_hashes=[b"ranked-a", b"ranked-b"],
        block_ids=([1, 2],),
        num_computed_tokens=0,
    )
    consumer = SimpleNamespace(
        request_id="ranked-consumer",
        num_tokens=token_count + 1,
        block_hashes=request.block_hashes,
    )
    scheduler = UMBPStoreConnector(
        _vllm_config(extra, **parallel), KVConnectorRole.SCHEDULER, config
    )
    scheduler.reset_cache()
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: request.block_ids),
        0,
    )
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[request],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={request.req_id: token_count},
        )
    )
    caches = {
        name: torch.empty_strided((8, 2, 16, 8), (512, 256, 8, 1), dtype=torch.float16)
        for name in ("layer1", "layer2")
    }
    try:
        for tp_rank in range(tp_size):
            assert scheduler.get_num_new_matched_tokens(consumer, 0) == (0, False)
            worker = UMBPStoreConnector(
                _vllm_config(extra, tensor_parallel_rank=tp_rank, **parallel),
                KVConnectorRole.WORKER,
                config,
            )
            try:
                worker.register_kv_caches(caches)
                worker.bind_connector_metadata(metadata)
                worker.wait_for_save()
            finally:
                worker.shutdown()
        assert scheduler.get_num_new_matched_tokens(consumer, 0) == (
            token_count,
            False,
        )
    finally:
        scheduler.shutdown()


def test_embedded_tp_dcp_rank_local_store_flow(monkeypatch):
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
        decode_context_parallel_rank=1,
        decode_context_parallel_size=2,
        world_size=2,
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
    worker_connector = UMBPStoreConnector(vllm_config, KVConnectorRole.WORKER, config)
    scheduler_connector.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([3, 4],)),
        0,
    )
    scheduler_output.scheduled_new_reqs = [
        SimpleNamespace(
            req_id="ranked",
            num_computed_tokens=0,
            block_ids=([3, 4],),
        )
    ]
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
        f"umbp:vllm:v1:{namespace}:tp1:pcp0:dcp1:pp0:g0:61",
    }
    assert runtime.scheduler_args is not None
    assert runtime.scheduler_args[1].local_namespace == (1, 0, 1, 0)
    assert runtime.scheduler_args[1].rank_count == 2


def test_embedded_logical_hit_requires_all_tp_dcp_objects(monkeypatch):
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
        decode_context_parallel_rank=0,
        decode_context_parallel_size=2,
        world_size=2,
    )
    connector = UMBPStoreConnector(vllm_config, KVConnectorRole.SCHEDULER, config)
    scheduler = connector.connector_scheduler
    assert scheduler is not None
    request = SimpleNamespace(
        request_id="logical",
        num_tokens=65,
        block_hashes=[b"a", b"b"],
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    for block_hash in request.block_hashes:
        runtime.store.update(
            scheduler.codec.keys_for_topology(block_hash, scheduler.topology, (0,))
        )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (64, False)
    runtime.store.remove(scheduler.codec.key_for_namespace(b"b", 0, (1, 0, 1, 0)))
    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, False)


@pytest.mark.parametrize("fail_store", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 3])
def test_worker_localizes_scheduler_plan_key_to_its_tp_rank(
    fail_store, tp_rank, monkeypatch
):
    handle = _LayerStoreRecordingWorkerHandle()
    if fail_store:
        store = handle.store

        def fail(plans):
            job = store(plans)
            job.completed_keys.clear()
            job.fail([plan.key for plan in plans], "store failed")
            return job

        monkeypatch.setattr(handle, "store", fail)
    codec = BlockIdentityCodec(UMBPNamespace("rank-local"), tp_rank=tp_rank)
    layout = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    layout.register_kv_caches(
        {
            name: torch.empty_strided(
                (8, 2, 16, 8),
                (512, 256, 8, 1),
                dtype=torch.float16,
            )
            for name in ("layer1", "layer2")
        }
    )
    worker = UMBPStoreConnectorWorker(handle, layout, codec=codec)
    plan = BlockTransferPlan(
        key=BlockIdentityCodec(UMBPNamespace("rank-local")).key(b"hash", 0),
        block_id=1,
        group_id=0,
        block_hash=b"hash",
    )
    metadata = UMBPConnectorMetadata(store_plans=[plan], store_event=7)

    worker.enqueue_stores(metadata)
    worker.wait_for_save()

    assert handle.store_calls[0][0].key == codec.key(b"hash", 0)
    worker_meta = worker.build_connector_worker_meta()
    assert worker_meta.store_events == {
        7: StoreEventResult(1, {(plan.key, 0)} if fail_store else set())
    }


@pytest.mark.parametrize("tp_rank", [0, 3])
@pytest.mark.parametrize("fast_path", [False, True])
def test_batched_load_localizes_rank_without_changing_request_or_destination(
    tp_rank, fast_path
):
    """Fast and fallback loads target the same rank, generation and GPU blocks."""
    codec = BlockIdentityCodec(UMBPNamespace("rank-local"), tp_rank=tp_rank)
    scheduler_codec = replace(codec, tp_rank=0)
    calls = []

    def load(plans):
        calls.append(plans)
        job = TransferJobState(plans if fast_path else tuple(plans))
        job.start()
        job.complete([codec.key(b"a", 0)])
        job.fail([codec.key(b"b", 1)], "injected load failure")
        return job

    handle = SimpleNamespace(load=load)
    if fast_path:
        handle.load_blocks = load
    worker = UMBPStoreConnectorWorker(handle, codec=codec, layerwise_load=False)
    batch = BlockLoadBatch(
        [scheduler_codec.key(b"a", 0), scheduler_codec.key(b"b", 1)],
        [7, 9],
        [0, 1],
        "request",
        4,
    )
    worker.start_load_kv(None, UMBPConnectorMetadata(load_requests={"request": batch}))

    assert len(calls) == 1
    assert [
        (p.key, p.block_id, p.group_id, p.request_id, p.generation) for p in calls[0]
    ] == [
        (codec.key(b"a", 0), 7, 0, "request", 4),
        (codec.key(b"b", 1), 9, 1, "request", 4),
    ]
    job = worker._load_jobs["request"][None]
    assert job.failed_block_ids == {9}
    assert batch.keys == [scheduler_codec.key(b"a", 0), scheduler_codec.key(b"b", 1)]


@pytest.mark.parametrize("local_tokens", [0, 16])
def test_scheduler_partial_prefix_loads_full_page_at_absolute_index(local_tokens):
    config = _kv_cache_config()
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {
                "mode": "embedded",
                "load_async": False,
                "enable_partial_hash_hits": True,
                "hash_block_size": 4,
            }
        ),
        config,
        _SchedulerHandle(
            [False, False, True] + [False] * (4 if local_tokens == 0 else 0)
        ),
        BlockIdentityCodec(UMBPNamespace("partial-prefix")),
    )
    request = SimpleNamespace(
        request_id="partial-prefix",
        num_tokens=32,
        block_hashes=[b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h"],
    )

    assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (12, False)
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(
            get_block_ids=lambda group_ids: ([6, 7] if local_tokens else [7],)
        ),
        12,
    )
    plans = scheduler._pending_loads["partial-prefix"]

    assert len(plans) == 1
    assert plans[0].block_id == 7
    assert (plans[0].token_start, plans[0].token_end) == (None, None)


def test_hybrid_fine_hash_lookup_is_not_limited_by_gpu_page_count():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {
                "load_async": False,
                "enable_partial_hash_hits": True,
                "hash_block_size": 4,
            }
        ),
        _hybrid_kv_cache_config(),
        _SchedulerHandle([True] * 62),
        BlockIdentityCodec(UMBPNamespace("many-hashes")),
    )
    request = SimpleNamespace(
        request_id="many-hashes",
        num_tokens=128,
        block_hashes=[bytes([index]) for index in range(32)],
    )
    for _ in range(2):
        # Mamba-style groups always load asynchronously.
        assert scheduler.get_num_new_matched_tokens(request, 0) == (124, True)
    scheduler.close()


@pytest.mark.parametrize("fail_store", [False, True])
def test_finished_hybrid_tail_pins_full_pages_until_all_workers_finish(fail_store):
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {"mode": "embedded", "load_async": False, "hash_block_size": 4},
            world_size=2,
        ),
        _hybrid_kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("multi-partial")),
    )
    request = SimpleNamespace(
        request_id="multi-partial",
        req_id="multi-partial",
        num_tokens=32,
        block_hashes=[b"a", b"b", b"c"],
        block_ids=([7], [9]),
        num_computed_tokens=12,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([7], [9])),
        0,
    )

    pool = BlockPool(16, True, 4)
    scheduler.bind_gpu_block_pool(pool)
    blocks = [pool.blocks[7], pool.blocks[9]]
    pool.touch(blocks)
    assert not scheduler.register_finished_partial_tail(
        request,
        ([7], [9]),
        [(1, 9, 12)],
    )
    assert [block.ref_cnt for block in blocks] == [2, 2]
    assert scheduler.request_finished(request, ([7], [9])) == (False, None)
    pool.free_blocks(blocks)
    assert [block.ref_cnt for block in blocks] == [1, 1]
    assert scheduler.has_pending_push_work()
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids={request.request_id},
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={},
        )
    )

    assert len(metadata.store_plans) == 2
    assert {(plan.group_id, plan.block_id) for plan in metadata.store_plans} == {
        (0, 7),
        (1, 9),
    }
    assert all(
        plan.token_start is None and plan.token_end is None
        for plan in metadata.store_plans
    )
    assert [block.ref_cnt for block in blocks] == [1, 1]
    for rank in range(2):
        scheduler.update_connector_output(
            SimpleNamespace(
                kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                    store_events={
                        metadata.store_event: StoreEventResult(
                            1,
                            {(metadata.store_plans[0].key, 0)} if fail_store else set(),
                        ),
                    }
                )
            )
        )
        assert [block.ref_cnt for block in blocks] == ([1, 1] if rank == 0 else [0, 0])
    assert not scheduler.has_pending_push_work()


class _LayerStoreRecordingWorkerHandle(_LayerRecordingWorkerHandle):
    def __init__(self):
        super().__init__()
        self.store_calls = []

    def store(self, plans):
        self.store_calls.append(list(plans))
        return super().store(plans)

    def poll(self, job):
        return _WorkerHandle.poll(self, job)


def test_layer_store_cancellation_reports_once_without_publishing():
    handle = _CancellableWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle, layerwise_store=True)
    plan = BlockTransferPlan(
        "layer-cancel",
        1,
        request_id="req",
        ranges=(KVRange("layer1", 0, 1, 1000, 16, 16, 0),),
    )
    metadata = UMBPConnectorMetadata(
        store_event=22,
        store_plans=[plan],
        store_plans_by_layer={"layer1": [plan]},
    )
    worker.save_kv_layer(metadata, "layer1", None, None)
    worker.enqueue_stores(metadata)
    worker.handle_preemptions(UMBPConnectorMetadata(preempted_request_ids={"req"}))
    worker.wait_for_save()

    assert not hasattr(handle, "published")
    assert worker.build_connector_worker_meta().store_events == {
        22: StoreEventResult(1, {("layer-cancel", 0)})
    }
    worker.wait_for_save()
    assert worker.build_connector_worker_meta().store_events == {}


@pytest.mark.parametrize("empty_group", [False, True])
@pytest.mark.parametrize("layerwise", [False, True])
def test_worker_merges_grouped_and_flat_store_plans_once(empty_group, layerwise):
    """Boundary/partial stores must not suppress independent lazy stores."""
    handle = _LayerStoreRecordingWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle, layerwise_store=layerwise)
    plans = [] if empty_group else [BlockTransferPlan("grouped", 1)]
    flat = BlockTransferPlan("flat", 2)
    metadata = UMBPConnectorMetadata(
        store_event=20,
        store_plans=[*plans, flat, flat],
        store_requests={"req": plans},
    )
    worker.enqueue_stores(metadata)
    worker.wait_for_save()

    assert handle.store_calls == ([plans] if plans else []) + [[flat]]
    assert metadata.store_requests == {"req": plans}
    assert worker.build_connector_worker_meta().store_events == {
        20: StoreEventResult(1)
    }


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("layer_callbacks", [False, True])
def test_worker_submits_stores_once(grouped, layerwise, layer_callbacks):
    """Grouped and flat fallback plans must survive optional layer callbacks."""
    handle = _LayerStoreRecordingWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle, layerwise_store=layerwise)
    plan = BlockTransferPlan(
        key="layered-store",
        block_id=1,
        request_id="req",
        ranges=(
            KVRange("layer1", 0, 1, 1000, 16, 16, 0),
            KVRange("layer2", 0, 1, 2000, 16, 16, 16),
        ),
    )
    metadata = UMBPConnectorMetadata(
        store_event=19,
        store_plans=[plan],
        store_requests={"req": [plan]} if grouped else {},
        store_plans_by_layer={"layer1": [plan], "layer2": [plan]},
    )

    if layer_callbacks:
        worker.save_kv_layer(metadata, "layer1", None, None)
        worker.save_kv_layer(metadata, "layer2", None, None)
    worker.enqueue_stores(metadata)
    worker.wait_for_save()

    assert len(handle.store_calls) == (2 if layerwise and layer_callbacks else 1)
    assert [
        item.layer_name for call in handle.store_calls for item in call[0].ranges
    ] == ["layer1", "layer2"]
    assert metadata.store_plans == [plan]
    assert metadata.store_requests == ({"req": [plan]} if grouped else {})
    assert worker.get_finished({"req"}) == (None, None)
    completed = worker.build_connector_worker_meta()
    assert completed.store_events == {19: StoreEventResult(1)}
    assert worker.get_kv_connector_stats().reduce()["store_num_bytes"] == 32
