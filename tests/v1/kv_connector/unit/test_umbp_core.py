# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.umbp_test_utils import (
    _CancellableWorkerHandle,
    _DelayedLoadWorkerHandle,
    _DelayedStoreWorkerHandle,
    _EmbeddedRuntime,
    _kv_cache_config,
    _SchedulerHandle,
    _StoreRecordingWorkerHandle,
    _vllm_config,
    _WaitRecordingDelayedLoadHandle,
    _WorkerHandle,
    install_memory_runtime,
)
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
    EmbeddedRuntime,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.factory import (
    UMBPRuntimeConfig,
    UMBPRuntimeFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import (
    UMBPStoreConnectorWorker,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def test_block_identity_codec_does_not_use_physical_block_id():
    codec = BlockIdentityCodec(UMBPNamespace("namespace"), 2, 3)

    assert codec.key(b"\x01\x02", group_id=4) == (
        "umbp:vllm:v1:namespace:tp2:pcp0:dcp0:pp3:g4:0102"
    )
    assert codec.key(b"\x01\x02", group_id=4) == codec.key(b"\x01\x02", group_id=4)


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
    ("rank", "expected"),
    [(0, (0, 0, 0, 0)), (1, (1, 0, 0, 0)), (2, (0, 1, 0, 0)), (5, (1, 0, 0, 1))],
)
def test_rank_topology_derives_ranks_from_worker_rank(rank, expected):
    # Workers are laid out as [PP, PCP, TP], TP innermost.
    topology = RankTopology.from_vllm_config(
        _vllm_config(
            {},
            rank=rank,
            tensor_parallel_size=2,
            prefill_context_parallel_size=2,
            pipeline_parallel_size=2,
            world_size=8,
        )
    )

    assert topology.local_namespace == expected


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


def test_namespace_distinguishes_draft_models():
    config = _kv_cache_config()

    def namespace(speculative_config):
        vllm_config = _vllm_config({"mode": "embedded"})
        if speculative_config is not None:
            vllm_config.speculative_config = speculative_config
        return UMBPNamespace.from_vllm_config(vllm_config, config).value

    def draft(model, revision="main"):
        return SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(model=model, revision=revision),
        )

    plain = namespace(None)
    assert namespace(SimpleNamespace()) != plain
    assert namespace(draft("drafter-a")) != plain
    assert namespace(draft("drafter-a")) == namespace(draft("drafter-a"))
    assert namespace(draft("drafter-a")) != namespace(draft("drafter-b"))
    assert namespace(draft("drafter-a")) != namespace(draft("drafter-a", "v2"))


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


def test_layout_planner_builds_compact_group_objects():
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
    )
    mamba = planner.plan_for_block(
        "mamba",
        2,
        addresses,
        group_id=1,
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


def test_materialization_preserves_plan_metadata_after_reregistration():
    """Compact ranges must use current buffers without losing logical identity."""
    planner = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    worker = UMBPStoreConnectorWorker(_WorkerHandle(), planner)
    plan = BlockTransferPlan(
        "key",
        2,
        request_id="req",
        group_id=0,
        block_hash=b"hash",
        parent_block_hash=b"parent",
        token_ids=(1, 2),
        block_size=16,
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
        )
        assert actual == replace(plan, ranges=expected.ranges)
        assert actual.ranges[0].base_address == (caches["layer1"].data_ptr() + 2048)
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
    tracker = RequestTracker()

    assert tracker.mark_saved(31, 16) == 16
    assert tracker.mark_saved(15, 16) == 16
    tracker.reset()

    assert tracker.saved_tokens == 0


def test_load_spec_reports_only_external_tokens():
    spec = LoadSpec(local_tokens=32, external_tokens=48)

    assert spec.num_tokens_to_load == 16


def test_worker_metadata_aggregates_store_events():
    metadata = UMBPConnectorWorkerMetadata(store_events={7: StoreEventResult(1)})
    other = UMBPConnectorWorkerMetadata(store_events={7: StoreEventResult(1)})
    metadata.aggregate(other)

    assert metadata.store_events == {7: StoreEventResult(2)}
    assert other.store_events == {7: StoreEventResult(1)}


@pytest.mark.parametrize("mode", ["embedded"])
def test_runtime_config_accepts_local_modes(mode):
    config = UMBPRuntimeConfig.from_vllm(_vllm_config({"mode": mode}))

    assert config.mode == mode


@pytest.mark.parametrize("mode", ["embedded"])
def test_runtime_adapter_owns_its_configuration(monkeypatch, mode):
    runtime = SimpleNamespace()

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

    assert isinstance(runtime, EmbeddedRuntime)


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


@pytest.mark.parametrize("failure", [TimeoutError("lookup timeout"), [True, False]])
def test_scheduler_lookup_failure_recomputes_and_allows_retry(failure):
    class _Handle(_SchedulerHandle):
        def lookup(self, keys):
            if isinstance(self.hits, Exception):
                raise self.hits
            return self.hits

    handle = _Handle(failure)
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"load_async": False}),
        _kv_cache_config(),
        handle,
        BlockIdentityCodec(UMBPNamespace("lookup-retry")),
    )
    request = SimpleNamespace(
        request_id="retry", num_tokens=32, block_hashes=[b"a", b"b"]
    )
    try:
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        handle.hits = [True]
        assert scheduler.get_num_new_matched_tokens(request, 0) == (16, False)
    finally:
        scheduler.close()


@pytest.mark.parametrize(("use_eagle", "expected"), [(False, 32), (True, 16)])
def test_scheduler_single_group_lookup_drops_last_block_for_eagle(use_eagle, expected):
    vllm_config = _vllm_config({"mode": "embedded", "load_async": False})
    vllm_config.speculative_config = SimpleNamespace(
        use_eagle_block_drop=lambda: use_eagle
    )
    scheduler = UMBPStoreConnectorScheduler(
        vllm_config,
        _kv_cache_config(),
        _SchedulerHandle([True, True]),
        BlockIdentityCodec(UMBPNamespace("eagle")),
    )
    request = SimpleNamespace(
        request_id="req", num_tokens=33, block_hashes=[b"a", b"b"]
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (expected, False)


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
    batch = BlockLoadBatch(["g1", "g0"], [7, 3], [1, 0], "r")
    meta = UMBPConnectorMetadata(load_requests={"r": batch})
    restored = (
        MsgpackDecoder(UMBPConnectorMetadata)
        .decode(MsgpackEncoder().encode(meta))
        .load_requests["r"]
    )
    assert list(restored) == [
        BlockTransferPlan("g1", 7, group_id=1, request_id="r"),
        BlockTransferPlan("g0", 3, group_id=0, request_id="r"),
    ]
    assert restored[:1] == [restored[0]]
    with pytest.raises(ValueError, match="equal lengths"):
        BlockLoadBatch(["missing-destination"], [], [0], "r")


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
        num_prompt_tokens=32,
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


def test_scheduler_decode_store_skips_unverified_speculative_tokens():
    """Scheduled draft tokens reach past the last hashed block; stop there."""
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
        BlockIdentityCodec(UMBPNamespace("spec")),
    )
    request = SimpleNamespace(
        request_id="spec",
        req_id="spec",
        num_tokens=47,
        num_prompt_tokens=32,
        block_hashes=[b"a", b"b"],
        block_ids=([1, 2, 3],),
        num_computed_tokens=46,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2, 3],)),
        0,
    )
    # One verified token plus three draft tokens: 46 + 4 = 50 > 47 tokens.
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["spec"],
                new_block_ids=[()],
                num_computed_tokens=[46],
            ),
            num_scheduled_tokens={"spec": 4},
        )
    )
    assert [plan.block_id for plan in metadata.store_plans] == [1, 2]


def test_scheduler_skips_decode_blocks_without_save_decode_cache():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": False}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("no-decode")),
    )
    request = SimpleNamespace(
        request_id="r",
        req_id="r",
        num_tokens=48,
        num_prompt_tokens=32,
        block_hashes=[b"a", b"b", b"c"],
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2, 3],)), 0
    )
    scheduler._request_trackers["r"].saved_tokens = 32
    # A decode step under synchronous scheduling: num_tokens - 1 are computed.
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["r"], new_block_ids=[None], num_computed_tokens=[47]
            ),
            num_scheduled_tokens={"r": 1},
        )
    )

    assert metadata.store_plans == []


@pytest.mark.parametrize("kv_role", ["kv_both", "kv_consumer"])
def test_kv_consumer_never_stores(kv_role):
    vllm_config = _vllm_config({"mode": "embedded", "load_async": False})
    vllm_config.kv_transfer_config.kv_role = kv_role
    scheduler = UMBPStoreConnectorScheduler(
        vllm_config,
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("role")),
    )
    request = SimpleNamespace(
        request_id="r",
        req_id="r",
        num_tokens=32,
        block_hashes=[b"a", b"b"],
        block_ids=([3, 4],),
        num_computed_tokens=0,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([3, 4],)),
        0,
    )
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[
                SimpleNamespace(req_id="r", num_computed_tokens=0, block_ids=([3, 4],))
            ],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={"r": 32},
        )
    )
    stored = [plan.block_id for plan in metadata.store_plans]
    assert stored == ([] if kv_role == "kv_consumer" else [3, 4])
    assert bool(metadata.store_requests) == (kv_role != "kv_consumer")


@pytest.mark.parametrize("resident", [False, True])
def test_restored_prefix_skips_resident_blocks(resident):
    """A restored block that is still in the pool is not stored again."""
    codec = BlockIdentityCodec(UMBPNamespace("restored-residency"))
    handle = _SchedulerHandle({})
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}), _kv_cache_config(), handle, codec
    )
    request = SimpleNamespace(
        request_id="req",
        req_id="req",
        num_tokens=49,
        block_hashes=[b"a", b"b", b"c"],
        block_ids=([1, 2, 3],),
        num_computed_tokens=0,
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: request.block_ids), 0
    )
    scheduler._request_trackers["req"].load_spec = LoadSpec(0, 32)
    handle.hits = {codec.key(h, 0): resident for h in (b"a", b"b")}
    meta = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[request],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={"req": 48},
        )
    )

    stored = [b"c"] if resident else [b"a", b"b", b"c"]
    assert [plan.key for plan in meta.store_plans] == [codec.key(h, 0) for h in stored]
    scheduler.close()


def test_store_event_owns_refs_until_all_ranks_finish():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {"mode": "embedded"},
            tensor_parallel_size=2,
            world_size=2,
        ),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("store-refs")),
        RankTopology(tp_size=2),
    )
    block = SimpleNamespace(block_id=4)
    freed = []
    scheduler.bind_gpu_block_pool(
        SimpleNamespace(
            blocks=[None, None, None, None, block],
            touch=lambda blocks: None,
            free_blocks=lambda blocks: freed.extend(blocks),
        )
    )
    request = SimpleNamespace(
        request_id="store-refs",
        req_id="store-refs",
        num_tokens=16,
        num_computed_tokens=0,
        block_hashes=[b"a"],
        block_ids=([4],),
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: ([4],)), 0
    )
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[request],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={request.req_id: 16},
        )
    )
    one_rank = SimpleNamespace(
        kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
            store_events={metadata.store_event: StoreEventResult(1)}
        )
    )

    scheduler.update_connector_output(one_rank)
    assert freed == []
    assert scheduler.has_pending_push_work()
    assert not scheduler.reset_store()

    scheduler.update_connector_output(one_rank)
    assert freed == [block]
    assert not scheduler.has_pending_push_work()
    assert scheduler.reset_store()
    # A late report for a finished event releases nothing twice.
    scheduler.update_connector_output(one_rank)
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
        num_prompt_tokens=32,
        num_computed_tokens=0,
        block_hashes=[b"a", b"b"],
        block_ids=([7, 8],),
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2],)),
        0,
    )
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
    assert [plan.block_id for plan in metadata.store_plans] == [7, 8]


def test_scheduler_stores_from_cores_current_block_table():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("current-table")),
    )
    request = SimpleNamespace(
        request_id="r",
        req_id="r",
        num_tokens=64,
        num_prompt_tokens=64,
        num_computed_tokens=0,
        block_hashes=[b"a", b"b", b"c", b"d"],
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: ([3, 4],)), 0
    )
    # Core has since freed block 3, e.g. a sliding-window block out of the window.
    current = {"r": ([NULL_BLOCK_ID, 4],)}
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["r"],
                new_block_ids=[None],
                num_computed_tokens=[0],
                resumed_req_ids=set(),
            ),
            num_scheduled_tokens={"r": 32},
            kv_connector_block_state=SimpleNamespace(
                get_block_ids=current.get, boundary_state_offloads={}
            ),
        )
    )

    assert [plan.block_id for plan in metadata.store_plans] == [4]


def test_scheduler_readmission_after_failed_load_uses_new_blocks():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("readmitted")),
    )
    request = SimpleNamespace(
        request_id="readmitted",
        req_id="readmitted",
        num_tokens=64,
        num_prompt_tokens=64,
        num_computed_tokens=0,
        block_hashes=[b"a", b"b", b"c", b"d"],
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2],)), 0
    )
    # The load failed: core freed blocks 1 and 2 and recomputes from token 0.
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: ([5, 6],)), 0
    )
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["readmitted"],
                new_block_ids=[None],
                num_computed_tokens=[0],
                resumed_req_ids=set(),
            ),
            num_scheduled_tokens={"readmitted": 32},
        )
    )

    assert scheduler._request_trackers["readmitted"].block_ids == ([5, 6],)
    assert [plan.block_id for plan in metadata.store_plans] == [5, 6]


@pytest.mark.parametrize("fail_load", [False, True])
def test_worker_polls_async_load_without_forward_execution(fail_load):
    handle = _DelayedLoadWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
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
    assert worker.get_block_ids_with_load_errors() == ({1} if fail_load else set())


def test_forward_does_not_wait_for_async_loads():
    """Async loads fill blocks of requests the running batch does not contain."""
    handle = _WaitRecordingDelayedLoadHandle()
    worker = UMBPStoreConnectorWorker(handle)
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
    assert worker.build_connector_worker_meta().store_events == {1: StoreEventResult(1)}
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
        0: StoreEventResult(1),
        2: StoreEventResult(1),
    }
    assert not handle.publications
    assert not handle.waited
    handle.jobs[1].complete()
    worker.get_finished(set())
    assert worker.build_connector_worker_meta().store_events == {1: StoreEventResult(1)}


def test_async_store_drains_before_buffers_can_be_reused():
    handle = _DelayedStoreWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    worker.enqueue_stores(
        UMBPConnectorMetadata(store_event=4, store_plans=[BlockTransferPlan("kv", 1)])
    )
    worker.wait_for_save()
    assert not handle.waited
    worker.close()
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
        21: StoreEventResult(1)
    }


@pytest.mark.parametrize("async_load", [False, True])
def test_load_cancellation_preserves_other_requests(async_load):
    handle = _CancellableWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
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
            async_load=async_load,
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
    assert finished == (None, {"second"} if async_load else None)
    worker.wait_for_layer_load("")
    assert worker.get_kv_connector_stats() is None


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
        BlockTransferPlan("ok", 11),
        BlockTransferPlan("bad", 12),
    ]
    worker.enqueue_stores(UMBPConnectorMetadata(store_plans=plans, store_event=9))
    worker.wait_for_save()

    assert worker.get_finished(set()) == (None, None)
    assert worker.get_block_ids_with_load_errors() == set()
    metadata = worker.build_connector_worker_meta()
    assert metadata.store_events == {9: StoreEventResult(1)}
    assert not hasattr(worker.runtime, "published")
    assert worker.get_kv_connector_stats().reduce()["store_failed"] == 1


@pytest.mark.parametrize("forward", [False, True])
def test_connector_finalizes_each_store_once_when_forward_hook_is_skipped(
    monkeypatch, forward
):
    """No-forward steps must not strand scheduler events or resubmit stores."""
    runtime = _EmbeddedRuntime()
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
def test_embedded_connector_core_flow(monkeypatch, enable_events):
    runtime = _EmbeddedRuntime()
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
                _vllm_config(extra, rank=tp_rank, **parallel),
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
        rank=1,
        tensor_parallel_size=2,
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
        rank=0,
        tensor_parallel_size=2,
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
    handle = _StoreRecordingWorkerHandle()
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
    assert worker_meta.store_events == {7: StoreEventResult(1)}
    assert hasattr(handle, "published") != fail_store


@pytest.mark.parametrize("tp_rank", [0, 3])
@pytest.mark.parametrize("fast_path", [False, True])
def test_batched_load_localizes_rank_without_changing_request_or_destination(
    tp_rank, fast_path
):
    """Fast and fallback loads target the same rank and GPU blocks."""
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
    worker = UMBPStoreConnectorWorker(handle, codec=codec)
    batch = BlockLoadBatch(
        [scheduler_codec.key(b"a", 0), scheduler_codec.key(b"b", 1)],
        [7, 9],
        [0, 1],
        "request",
    )
    worker.start_load_kv(None, UMBPConnectorMetadata(load_requests={"request": batch}))

    assert len(calls) == 1
    assert [(p.key, p.block_id, p.group_id, p.request_id) for p in calls[0]] == [
        (codec.key(b"a", 0), 7, 0, "request"),
        (codec.key(b"b", 1), 9, 1, "request"),
    ]
    job = worker._load_jobs["request"][None]
    assert job.failed_block_ids == {9}
    assert batch.keys == [scheduler_codec.key(b"a", 0), scheduler_codec.key(b"b", 1)]


@pytest.mark.parametrize("empty_group", [False, True])
def test_worker_merges_grouped_and_flat_store_plans_once(empty_group):
    """Request-grouped stores must not suppress independent flat stores."""
    handle = _StoreRecordingWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
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
def test_worker_submits_stores_once(grouped):
    """Grouped and flat fallback plans are each stored once."""
    handle = _StoreRecordingWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
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
    )

    worker.enqueue_stores(metadata)
    worker.wait_for_save()

    assert len(handle.store_calls) == 1
    assert [
        item.layer_name for call in handle.store_calls for item in call[0].ranges
    ] == ["layer1", "layer2"]
    assert metadata.store_plans == [plan]
    assert metadata.store_requests == ({"req": [plan]} if grouped else {})
    assert worker.get_finished({"req"}) == (None, None)
    completed = worker.build_connector_worker_meta()
    assert completed.store_events == {19: StoreEventResult(1)}
    assert worker.get_kv_connector_stats().reduce()["store_num_bytes"] == 32
